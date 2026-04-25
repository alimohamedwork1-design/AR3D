import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import requests
import runpod
from plyfile import PlyData
from pygltflib import (
    Accessor,
    Asset,
    Buffer,
    BufferView,
    GLTF2,
    Mesh,
    Node,
    Primitive,
    Scene,
)
import struct
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import quote


def _supabase_credentials() -> tuple[str, str]:
    """
    Read at call time (not only import) so RunPod-injected env is visible.
    Service role is required for Storage PUT from the worker (not the anon key).
    """
    url = (
        os.environ.get("SUPABASE_URL")
        or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        or os.environ.get("EXPO_PUBLIC_SUPABASE_URL")
        or ""
    ).strip()
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_SERVICE_KEY")
        or os.environ.get("SERVICE_ROLE_KEY")
        or ""
    ).strip()
    return url, key

# COLMAP tuning
COLMAP_USE_GPU_DEFAULT = os.environ.get("COLMAP_USE_GPU", "1").strip()  # "1" or "0"
# Higher default helps sparse scenes; can override via env or pipeline input.
COLMAP_MAX_IMAGE_SIZE = os.environ.get("COLMAP_MAX_IMAGE_SIZE", "3200").strip()  # pixels
COLMAP_SINGLE_CAMERA = os.environ.get("COLMAP_SINGLE_CAMERA", "1").strip()  # "1" or "0"
COLMAP_MATCHER_DEFAULT = os.environ.get("COLMAP_MATCHER", "").strip().lower()  # exhaustive|sequential
QT_QPA_PLATFORM = os.environ.get("QT_QPA_PLATFORM", "offscreen").strip()  # offscreen|minimal


def _run(cmd: list[str], *, cwd: Optional[str] = None, env: Optional[dict[str, str]] = None) -> str:
    """
    Run a command and return combined stdout/stderr (trimmed).
    Raises CalledProcessError on non-zero exit.
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items() if v is not None})

    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=merged_env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return out.strip()


def _is_sigabrt(err: BaseException) -> bool:
    if not isinstance(err, subprocess.CalledProcessError):
        return False
    # On Linux, signals show up as negative return codes (e.g., -6 for SIGABRT).
    return err.returncode in (-6, 134)


def _boolish(v: object) -> bool:
    s = str(v or "").strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _want_mesh_export(job_input: dict) -> bool:
    # Explicit input list wins; otherwise env default.
    of = job_input.get("output_formats")
    if isinstance(of, list):
        vals = {str(x).strip().lower() for x in of}
        return bool(vals & {"mesh", "mesh_glb", "glb_mesh", "glb-mesh"})
    return _boolish(os.environ.get("EXPORT_MESH_DEFAULT", "0"))


def try_export_mesh_glb_from_ply(ply_path: Path, glb_path: Path) -> bool:
    """
    Best-effort surface reconstruction for users who want a real mesh (triangles),
    not a POINTS point-cloud GLB. Uses Open3D (if available).
    """
    try:
        import open3d as o3d  # type: ignore
    except Exception as e:
        print(f"[mesh] open3d not available: {e}", flush=True)
        return False

    try:
        import trimesh  # type: ignore
    except Exception as e:
        print(f"[mesh] trimesh not available: {e}", flush=True)
        return False

    try:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if pcd.is_empty():
            print("[mesh] empty point cloud; skipping", flush=True)
            return False

        # Downsample to keep Poisson reasonable (can be heavy).
        target_pts = int(os.environ.get("MESH_MAX_POINTS", "200000") or 200000)
        if target_pts > 0 and len(pcd.points) > target_pts:
            # Voxel size from bounding box diagonal.
            bbox = pcd.get_axis_aligned_bounding_box()
            diag = float((bbox.get_max_bound() - bbox.get_min_bound()).max())
            voxel = max(0.002, min(0.05, diag / 250.0))
            pcd = pcd.voxel_down_sample(voxel_size=voxel)
            print(f"[mesh] voxel_down_sample voxel={voxel:.4f} points={len(pcd.points)}", flush=True)

        # Estimate normals for reconstruction.
        radius = float(os.environ.get("MESH_NORMAL_RADIUS", "0.06") or 0.06)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        pcd.normalize_normals()

        # Poisson reconstruction.
        depth = int(os.environ.get("MESH_POISSON_DEPTH", "9") or 9)
        print(f"[mesh] poisson depth={depth} points={len(pcd.points)}", flush=True)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

        # Crop to original bounding box (helps remove floating junk).
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Density-based cleanup: drop lowest density vertices.
        if densities is not None:
            import numpy as np  # type: ignore

            dens = np.asarray(densities)
            if dens.size:
                q = float(os.environ.get("MESH_DENSITY_QUANTILE", "0.02") or 0.02)
                thr = float(np.quantile(dens, q))
                mask = dens < thr
                mesh.remove_vertices_by_mask(mask)
                print(f"[mesh] density cleanup q={q} thr={thr:.6f}", flush=True)

        # Optional simplification
        target_tris = int(os.environ.get("MESH_TARGET_TRIANGLES", "250000") or 250000)
        if target_tris > 0 and len(mesh.triangles) > target_tris:
            mesh = mesh.simplify_quadric_decimation(target_tris)
            print(f"[mesh] simplified triangles={len(mesh.triangles)}", flush=True)

        v = mesh.vertices
        f = mesh.triangles
        if len(v) < 100 or len(f) < 200:
            print(f"[mesh] too small after cleanup v={len(v)} f={len(f)}; skipping", flush=True)
            return False

        tm = trimesh.Trimesh(vertices=list(v), faces=list(f), process=False)
        tm.export(str(glb_path))
        ok = glb_path.is_file() and glb_path.stat().st_size > 1024
        print(f"[mesh] wrote glb ok={ok} size={glb_path.stat().st_size if glb_path.exists() else 0}", flush=True)
        return ok
    except Exception as e:
        print(f"[mesh] export failed: {e}", flush=True)
        return False

def ensure_colmap_sparse_zero_layout(dataset_root: Path) -> None:
    """
    GraphDeco gaussian-splatting expects COLMAP files under sparse/0/
    (cameras/images/points3D as .txt or .bin).

    `colmap image_undistorter` often writes the model directly under sparse/ instead of sparse/0/.
    """
    sparse = dataset_root / "sparse"
    if not sparse.is_dir():
        raise RuntimeError(f"colmap_missing_sparse: {sparse}")

    zero = sparse / "0"
    colmap_names = (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "cameras.txt",
        "images.txt",
        "points3D.txt",
    )

    def has_any_model_files(p: Path) -> bool:
        return any((p / n).is_file() for n in colmap_names)

    if has_any_model_files(zero):
        return

    # Undistorted COLMAP: files live in sparse/ (flat); move into sparse/0/
    if has_any_model_files(sparse):
        zero.mkdir(parents=True, exist_ok=True)
        for name in colmap_names:
            src = sparse / name
            if src.is_file():
                dst = zero / name
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))
        print(f"[colmap] moved COLMAP model into {zero}")
        return

    raise RuntimeError(f"colmap_sparse_layout_unrecognized: {sparse}")


def download_images(image_urls: Iterable[str], work_dir: Path) -> Tuple[Path, int]:
    images_dir = work_dir / "input" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.9,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    downloaded = 0
    total_urls = len(image_urls) if hasattr(image_urls, "__len__") else None
    for i, url in enumerate(image_urls):
        try:
            r = session.get(url, timeout=60)
            if r.status_code < 200 or r.status_code >= 300:
                raise RuntimeError(f"http_{r.status_code}")
            ext = url.split("?")[0].split(".")[-1].lower()
            if ext not in {"jpg", "jpeg", "png", "webp"}:
                ext = "jpg"
            filepath = images_dir / f"img_{i:04d}.{ext}"
            filepath.write_bytes(r.content)
            downloaded += 1
        except Exception as e:
            print(f"[download] failed url={url} err={e}")

    print(f"[download] downloaded {downloaded} / {total_urls if total_urls is not None else 'n/a'}")
    return images_dir, downloaded


def download_supabase_objects(
    *,
    bucket: str,
    object_paths: Iterable[str],
    work_dir: Path,
) -> Tuple[Path, int]:
    """
    Download images from Supabase Storage using service role.
    `object_paths` are paths within the bucket (e.g. "<uid>/<tourId>/scan_....jpg").
    Requires SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY to be set on the worker.
    """
    supabase_url, service_role_key = _supabase_credentials()
    missing = [n for n, v in [("SUPABASE_URL", supabase_url), ("SUPABASE_SERVICE_ROLE_KEY", service_role_key)] if not v]
    if missing:
        raise RuntimeError("missing_supabase_env: " + ", ".join(missing))

    images_dir = work_dir / "input" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.9,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    object_paths_list = [str(x) for x in object_paths]
    downloaded = 0
    for i, obj_path in enumerate(object_paths_list):
        p = str(obj_path).lstrip("/")
        try:
            enc_path = quote(p, safe="/")
            url = f"{supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{enc_path}"
            r = session.get(
                url,
                headers={
                    "Authorization": f"Bearer {service_role_key}",
                    "apikey": service_role_key,
                },
                timeout=60,
            )
            if r.status_code < 200 or r.status_code >= 300:
                raise RuntimeError(f"http_{r.status_code}")
            ext = p.split("?")[0].split(".")[-1].lower()
            if ext not in {"jpg", "jpeg", "png", "webp"}:
                ext = "jpg"
            filepath = images_dir / f"img_{i:04d}.{ext}"
            filepath.write_bytes(r.content)
            downloaded += 1
        except Exception as e:
            print(f"[download_supabase] failed path={p} err={e}", flush=True)

    print(f"[download_supabase] downloaded {downloaded} / {len(object_paths_list)}", flush=True)
    return images_dir, downloaded


def run_colmap(work_dir: Path, *, matcher: str = "exhaustive", max_image_size: Optional[int] = None) -> Path:
    # gaussian-splatting expects COLMAP outputs under the dataset root:
    #   <dataset>/images
    #   <dataset>/sparse/0
    dataset_dir = work_dir / "input"
    images_dir = dataset_dir / "images"
    sparse_root = dataset_dir / "sparse"
    sparse_root.mkdir(parents=True, exist_ok=True)
    # COLMAP mapper writes to <output_path>/<model_id>/..., we ensure at least "0" exists.
    db_path = dataset_dir / "colmap.db"

    matcher_norm = (matcher or "exhaustive").strip().lower()
    if matcher_norm not in {"exhaustive", "sequential"}:
        matcher_norm = "exhaustive"
    max_img = int(max_image_size) if isinstance(max_image_size, int) and max_image_size > 0 else int(COLMAP_MAX_IMAGE_SIZE)

    def attempt(use_gpu: bool) -> None:
        gpu_flag = "1" if use_gpu else "0"
        # Ensure headless execution; prevents Qt/X11 SIGABRT on serverless workers.
        colmap_env = {
            "QT_QPA_PLATFORM": QT_QPA_PLATFORM or "offscreen",
            "DISPLAY": "",
        }
        print(f"[colmap] feature_extractor use_gpu={gpu_flag} max_image_size={max_img} matcher={matcher_norm}")
        _run(
            [
                "colmap",
                "feature_extractor",
                "--database_path",
                str(db_path),
                "--image_path",
                str(images_dir),
                "--ImageReader.single_camera",
                COLMAP_SINGLE_CAMERA,
                "--SiftExtraction.use_gpu",
                gpu_flag,
                "--SiftExtraction.max_image_size",
                str(max_img),
            ],
            env=colmap_env,
        )

        # For photo bursts (unordered), exhaustive matching is much more robust than sequential.
        # Sequential matching is appropriate for video-like ordered frames.
        matcher_cmd = "exhaustive_matcher" if matcher_norm == "exhaustive" else "sequential_matcher"
        print(f"[colmap] {matcher_cmd} use_gpu={gpu_flag}")
        _run(
            [
                "colmap",
                matcher_cmd,
                "--database_path",
                str(db_path),
                "--SiftMatching.use_gpu",
                gpu_flag,
            ],
            env=colmap_env,
        )

        print("[colmap] mapper")
        _run(
            [
                "colmap",
                "mapper",
                "--database_path",
                str(db_path),
                "--image_path",
                str(images_dir),
                "--output_path",
                str(sparse_root),
            ],
            env=colmap_env,
        )

    def pick_sparse_model_dir() -> Path:
        # COLMAP mapper writes sparse/<model_id>/...
        candidates = [p for p in sparse_root.iterdir() if p.is_dir()]
        if not candidates:
            raise RuntimeError("colmap_sparse_empty")
        # Prefer model id "0" if present.
        zero = sparse_root / "0"
        if zero.exists() and zero.is_dir():
            return zero
        # Otherwise pick the largest directory (most files).
        return max(candidates, key=lambda p: sum(1 for _ in p.rglob("*") if _.is_file()))

    def undistort_to_gs_dataset(sparse_model_dir: Path) -> Path:
        """
        GraphDeco gaussian-splatting expects undistorted COLMAP cameras (PINHOLE / SIMPLE_PINHOLE).
        """
        undist_dir = work_dir / "input_undist"
        if undist_dir.exists():
            shutil.rmtree(undist_dir, ignore_errors=True)
        undist_dir.mkdir(parents=True, exist_ok=True)

        colmap_env = {
            "QT_QPA_PLATFORM": QT_QPA_PLATFORM or "offscreen",
            "DISPLAY": "",
        }
        print(f"[colmap] image_undistorter sparse_model={sparse_model_dir}")
        _run(
            [
                "colmap",
                "image_undistorter",
                "--image_path",
                str(images_dir),
                "--input_path",
                str(sparse_model_dir),
                "--output_path",
                str(undist_dir),
                "--output_type",
                "COLMAP",
            ],
            env=colmap_env,
        )
        ensure_colmap_sparse_zero_layout(undist_dir)
        return undist_dir

    # Try GPU first (if enabled), then CPU fallback for stability.
    want_gpu = COLMAP_USE_GPU_DEFAULT == "1"
    if want_gpu:
        try:
            attempt(True)
            print("[colmap] mapper done (gpu)")
        except subprocess.CalledProcessError as e:
            print(f"[colmap] gpu failed returncode={e.returncode}")
            tail = ((e.stderr or "")[-1500:]).strip()
            if tail:
                print(f"[colmap] gpu stderr tail:\n{tail}")
            # Fallback on SIGABRT / common GPU crashes.
            if _is_sigabrt(e) or "cuda" in (e.stderr or "").lower() or "out of memory" in (e.stderr or "").lower():
                print("[colmap] retrying on CPU…")
                # GPU crashes can corrupt the COLMAP DB; recreate it for CPU retry.
                try:
                    if db_path.exists():
                        db_path.unlink()
                        print("[colmap] deleted colmap.db before CPU retry")
                except Exception as de:
                    print(f"[colmap] could not delete colmap.db: {de}")
                attempt(False)
                print("[colmap] mapper done (cpu after gpu fail)")
            else:
                raise
    else:
        attempt(False)
        print("[colmap] mapper done (cpu)")

    sparse_model_dir = pick_sparse_model_dir()
    gs_dataset = undistort_to_gs_dataset(sparse_model_dir)
    print(f"[colmap] undistorted dataset ready at {gs_dataset}")
    return gs_dataset


def run_gaussian_splatting(gs_source: Path, iterations: int = 500) -> Path:
    # gs_source is e.g. /workspace/job_<id>/input_undist
    work_root = gs_source.parent
    output_dir = work_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    gs_path = Path("/workspace/gaussian-splatting/train.py")
    if not gs_path.exists():
        raise RuntimeError("gaussian-splatting not installed in image")

    # Official GraphDeco train.py does: args.save_iterations.append(args.iterations) — final iter is always saved.
    print(f"[gs] training iterations={iterations} source={gs_source}")
    p = subprocess.run(
        [
            "python",
            str(gs_path),
            "-s",
            str(gs_source),
            "--model_path",
            str(output_dir),
            "--iterations",
            str(iterations),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        cwd="/workspace/gaussian-splatting",
    )
    if p.returncode != 0:
        tail = ((p.stderr or "")[-2000:]).strip()
        raise RuntimeError(f"gaussian_splatting_failed: {tail or 'unknown_error'}")

    print("[gs] done")
    return output_dir


def upload_to_supabase(local_path: Path, remote_path: str) -> str:
    supabase_url, service_role = _supabase_credentials()
    missing = [n for n, v in [("SUPABASE_URL", supabase_url), ("SUPABASE_SERVICE_ROLE_KEY", service_role)] if not v]
    if missing:
        raise RuntimeError(
            "missing_supabase_env: "
            + ", ".join(missing)
            + ". Set these on RunPod: Serverless → your endpoint → Environment / Secrets "
            "(Dashboard → API: Project URL + service_role key; never use anon key here)."
        )

    # Storage upload endpoint expects PUT.
    url = f"{supabase_url}/storage/v1/object/{remote_path.lstrip('/')}"
    with local_path.open("rb") as f:
        resp = requests.put(
            url,
            headers={
                "Authorization": f"Bearer {service_role}",
                "Content-Type": "application/octet-stream",
                "x-upsert": "true",
            },
            data=f,
            timeout=120,
        )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"supabase_upload_failed: http={resp.status_code} body={resp.text[:400]}")

    # If bucket is public you can use public URL; otherwise this is just a stable path to sign later.
    return f"{supabase_url}/storage/v1/object/public/{remote_path.lstrip('/')}"


def _is_payload_too_large(err: BaseException) -> bool:
    msg = str(err)
    return "413" in msg and ("payload too large" in msg.lower() or "maximum allowed size" in msg.lower())


def _target_max_upload_bytes() -> int:
    """
    Supabase Storage object size limits vary by plan.
    Use a conservative cap to avoid repeated 413 errors.
    Override with SUPABASE_MAX_UPLOAD_MB on the worker if needed.
    """
    raw = str(os.environ.get("SUPABASE_MAX_UPLOAD_MB", "45")).strip()
    try:
        mb = float(raw)
    except Exception:
        mb = 45.0
    mb = max(5.0, min(200.0, mb))
    return int(mb * 1024 * 1024)


def downsample_until_under_limit(ply_path: Path, *, start_max_points: int, max_bytes: int) -> int:
    """
    Iteratively downsample PLY until it fits under `max_bytes` (best-effort).
    Returns resulting point count.
    """
    max_points = int(start_max_points)
    if max_points <= 0:
        return 0
    last_kept = 0
    for _ in range(6):
        kept = downsample_and_rewrite_ply_inplace(ply_path, max_points)
        last_kept = kept
        sz = ply_path.stat().st_size
        print(f"[ply] after downsample: points={kept} size={sz} bytes max_bytes={max_bytes}", flush=True)
        if sz <= max_bytes:
            return kept
        # Reduce aggressively for next round
        max_points = max(20000, int(max_points * 0.6))
    return last_kept

def _vertex_rgb_u8(v, i: int) -> tuple[int, int, int]:
    """Classic PLY rgb (uchar) or 3DGS SH DC → display RGB."""
    names = v.data.dtype.names or ()
    if all(k in names for k in ("red", "green", "blue")):
        return int(v["red"][i]) & 255, int(v["green"][i]) & 255, int(v["blue"][i]) & 255
    # GraphDeco 3DGS PLY: degree-0 SH (same convention as common viewers)
    if all(k in names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        sh_c0 = 0.28209479177387814
        r = float(v["f_dc_0"][i]) * sh_c0 + 0.5
        g = float(v["f_dc_1"][i]) * sh_c0 + 0.5
        b = float(v["f_dc_2"][i]) * sh_c0 + 0.5
        return (
            int(max(0.0, min(1.0, r)) * 255),
            int(max(0.0, min(1.0, g)) * 255),
            int(max(0.0, min(1.0, b)) * 255),
        )
    raise RuntimeError("ply_no_rgb: need red/green/blue or f_dc_0/1/2")


def ply_point_cloud_to_glb(ply_path: Path, glb_path: Path) -> None:
    """
    Convert a point-cloud PLY to a minimal GLB using glTF POINTS primitive.
    Supports classic uchar RGB or GraphDeco Gaussian Splatting (f_dc_0..2).
    """
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise RuntimeError("ply_missing_vertex")
    v = ply["vertex"]
    x = v["x"]
    y = v["y"]
    z = v["z"]
    n = len(x)
    if n <= 0:
        raise RuntimeError("ply_empty")

    names = v.data.dtype.names or ()
    has_classic_rgb = all(k in names for k in ("red", "green", "blue"))
    has_gs_dc = all(k in names for k in ("f_dc_0", "f_dc_1", "f_dc_2"))
    has_color = has_classic_rgb or has_gs_dc
    if not has_color:
        raise RuntimeError("ply_no_color_fields")

    # Pack binary buffer: positions (float32*3) then colors (uint8*4) if present.
    buf = bytearray()
    for i in range(n):
        buf += struct.pack("<fff", float(x[i]), float(y[i]), float(z[i]))
    pos_offset = 0
    pos_len = len(buf)

    col_offset = None
    if has_color:
        # Align to 4 bytes for bufferView
        while len(buf) % 4 != 0:
            buf += b"\x00"
        col_offset = len(buf)
        for i in range(n):
            r, g, b = _vertex_rgb_u8(v, i)
            buf += bytes((r, g, b, 255))

    # Compute bounds
    minx = float(min(x))
    miny = float(min(y))
    minz = float(min(z))
    maxx = float(max(x))
    maxy = float(max(y))
    maxz = float(max(z))

    gltf = GLTF2(asset=Asset(version="2.0"))
    gltf.buffers = [Buffer(byteLength=len(buf))]
    gltf.bufferViews = [
        BufferView(buffer=0, byteOffset=pos_offset, byteLength=pos_len, target=34962),  # ARRAY_BUFFER
    ]
    accessors = [
        Accessor(
            bufferView=0,
            byteOffset=0,
            componentType=5126,  # FLOAT
            count=n,
            type="VEC3",
            min=[minx, miny, minz],
            max=[maxx, maxy, maxz],
        )
    ]

    attributes = {"POSITION": 0}
    if has_color and col_offset is not None:
        gltf.bufferViews.append(
            BufferView(buffer=0, byteOffset=col_offset, byteLength=(n * 4), target=34962)
        )
        accessors.append(
            Accessor(
                bufferView=1,
                byteOffset=0,
                componentType=5121,  # UNSIGNED_BYTE
                normalized=True,
                count=n,
                type="VEC4",
            )
        )
        attributes["COLOR_0"] = 1

    gltf.accessors = accessors
    prim = Primitive(attributes=attributes, mode=0)  # POINTS
    gltf.meshes = [Mesh(primitives=[prim])]
    gltf.nodes = [Node(mesh=0)]
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0

    gltf.set_binary_blob(bytes(buf))
    gltf.save_binary(str(glb_path))


def downsample_and_rewrite_ply_inplace(ply_path: Path, max_points: int) -> int:
    """
    Reduce point count to avoid oversized uploads. Rewrites the PLY in binary format.
    Returns the resulting point count.
    """
    if max_points <= 0:
        return 0

    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise RuntimeError("ply_missing_vertex")
    v = ply["vertex"].data
    n = len(v)
    if n <= max_points:
        # Still rewrite as binary to reduce size if it was ascii.
        PlyData(ply.elements, text=False).write(str(ply_path))
        return n

    # Deterministic-ish stride sampling (faster than random, no numpy needed)
    step = max(1, n // max_points)
    sampled = v[::step][:max_points]
    new_ply = PlyData([ply["vertex"].__class__(sampled, "vertex")], text=False)
    new_ply.write(str(ply_path))
    return len(sampled)


def handler(job):
    t0 = time.time()
    job_input = job.get("input", {}) or {}
    image_urls = job_input.get("image_urls", []) or []
    supabase_bucket = str(job_input.get("supabase_bucket", "") or "").strip()
    supabase_image_paths = job_input.get("supabase_image_paths", None)
    tour_id = str(job_input.get("tour_id", "unknown")).strip()
    quality_profile = str(job_input.get("quality_profile", "") or "").strip().lower()
    iterations = int(job_input.get("iterations", 15000) or 15000)
    # Auto-max quality mode: bump defaults unless explicitly set.
    if quality_profile in {"auto_max", "auto-max", "max"} and ("iterations" not in job_input):
        iterations = int(os.environ.get("GS_ITERATIONS_DEFAULT", "20000") or 20000)

    has_supabase_paths = isinstance(supabase_image_paths, list) and len(supabase_image_paths) > 0 and bool(supabase_bucket)
    if not image_urls and not has_supabase_paths:
        return {"ok": False, "error": "no_image_urls_or_supabase_paths", "error_code": "INSUFFICIENT_IMAGES", "retryable": False}
    if image_urls and len(image_urls) < 10:
        return {"ok": False, "error": f"need_at_least_10_images_got_{len(image_urls)}", "error_code": "INSUFFICIENT_IMAGES", "retryable": False}
    if has_supabase_paths and len(supabase_image_paths) < 10:
        return {"ok": False, "error": f"need_at_least_10_images_got_{len(supabase_image_paths)}", "error_code": "INSUFFICIENT_IMAGES", "retryable": False}

    work_dir = Path(f"/workspace/job_{tour_id}")

    try:
        # 1) Download
        if has_supabase_paths:
            images_dir, downloaded = download_supabase_objects(
                bucket=supabase_bucket,
                object_paths=supabase_image_paths,
                work_dir=work_dir,
            )
        else:
            images_dir, downloaded = download_images(image_urls, work_dir)
        if downloaded < 10:
            return {"ok": False, "error": f"downloaded_only_{downloaded}"}

        # 2) COLMAP (gpu -> cpu fallback)
        print("[job] running COLMAP…")
        pipeline = job_input.get("pipeline", {}) if isinstance(job_input.get("pipeline", {}), dict) else {}
        colmap_cfg = pipeline.get("colmap", {}) if isinstance(pipeline.get("colmap", {}), dict) else {}
        # Default: exhaustive for photo_burst/panorama, sequential for video.
        capture_mode = str(job_input.get("capture_mode", "") or "").strip().lower()
        matcher = str(colmap_cfg.get("matcher", "") or COLMAP_MATCHER_DEFAULT or "").strip().lower()
        if not matcher:
            matcher = "sequential" if capture_mode == "video" else "exhaustive"
        # Auto-max quality: always exhaustive unless it's a video capture.
        if quality_profile in {"auto_max", "auto-max", "max"} and capture_mode != "video":
            matcher = "exhaustive"
        max_img = colmap_cfg.get("max_image_size", None)
        max_img_int = int(max_img) if isinstance(max_img, (int, float, str)) and str(max_img).strip().isdigit() else None
        # Auto-max quality: raise image size unless explicitly provided.
        if quality_profile in {"auto_max", "auto-max", "max"} and max_img_int is None:
            max_img_int = int(os.environ.get("COLMAP_MAX_IMAGE_SIZE", "3200") or 3200)
        gs_source = run_colmap(work_dir, matcher=matcher, max_image_size=max_img_int)

        # 3) Train
        print("[job] running Gaussian Splatting…")
        out_dir = run_gaussian_splatting(gs_source, iterations)

        # 4) Collect outputs
        ply = next(iter(out_dir.rglob("point_cloud.ply")), None)
        if not ply or not ply.exists():
            return {"ok": False, "error": "no_point_cloud"}

        ply_size_mb = ply.stat().st_size / (1024 * 1024)
        print(f"[ply] path={ply} size_mb={ply_size_mb:.3f}")
        ply_min_mb = float((job_input.get("ply_min_mb") or os.environ.get("PLY_MIN_MB") or "0").strip() or 0)
        if ply_min_mb > 0 and ply_size_mb < ply_min_mb:
            return {
                "ok": False,
                "error": f"ply_too_small_mb={ply_size_mb:.3f}_min={ply_min_mb}",
                "ply_path": str(ply),
                "colmap_hint": "too_few_points_or_bad_registration",
            }

        # Optional size reduction to satisfy Storage max object size (413 Payload too large).
        # Set via env or input; default keeps a reasonable cap.
        max_points = int(job_input.get("max_points", os.environ.get("PLY_MAX_POINTS", "800000")) or 800000)
        try:
            kept = downsample_and_rewrite_ply_inplace(ply, max_points)
            print(f"[ply] points={kept} max_points={max_points} size={ply.stat().st_size} bytes")
        except Exception as e:
            print(f"[ply] downsample skipped: {e}")

        # 5) Convert to GLB (point cloud) — required for Arqary viewer / API extractors
        glb = out_dir / "point_cloud.glb"
        try:
            ply_point_cloud_to_glb(ply, glb)
        except Exception as e:
            return {"ok": False, "error": f"glb_convert_failed: {e}"}
        if not glb.is_file() or glb.stat().st_size < 64:
            return {"ok": False, "error": "glb_write_failed_empty"}

        # 5b) Optional: create a real triangle mesh GLB for better “complete” visuals.
        mesh_glb = out_dir / "mesh.glb"
        mesh_ok = False
        if quality_profile in {"auto_max", "auto-max", "max"}:
            # Auto-max: generate mesh whenever possible.
            mesh_ok = try_export_mesh_glb_from_ply(ply, mesh_glb)
        elif _want_mesh_export(job_input):
            mesh_ok = try_export_mesh_glb_from_ply(ply, mesh_glb)

        # 6) Upload outputs
        bucket = os.environ.get("SUPABASE_SPLATS_BUCKET", "splats").strip() or "splats"
        ply_remote = f"{bucket}/{tour_id}/point_cloud.ply"
        glb_remote = f"{bucket}/{tour_id}/point_cloud.glb"
        mesh_remote = f"{bucket}/{tour_id}/mesh.glb"
        # Upload GLB first: viewer mainly depends on GLB; PLY can be skipped if too large.
        glb_url = upload_to_supabase(glb, glb_remote)
        print(f"[glb] uploaded glb_url={glb_url}", flush=True)

        ply_url = None
        max_bytes = _target_max_upload_bytes()
        # Ensure PLY is under the limit before upload (best-effort).
        try:
            if ply.stat().st_size > max_bytes:
                downsample_until_under_limit(ply, start_max_points=max_points, max_bytes=max_bytes)
        except Exception as e:
            print(f"[ply] pre-upload size check/downsample failed: {e}", flush=True)

        try:
            ply_url = upload_to_supabase(ply, ply_remote)
        except Exception as e:
            if _is_payload_too_large(e):
                print(f"[ply] upload too large; skipping ply upload err={e}", flush=True)
                ply_url = None
            else:
                raise

        mesh_url = None
        if mesh_ok and mesh_glb.exists():
            try:
                mesh_url = upload_to_supabase(mesh_glb, mesh_remote)
                print(f"[mesh] uploaded mesh_url={mesh_url}", flush=True)
            except Exception as e:
                print(f"[mesh] upload failed: {e}", flush=True)
                mesh_url = None

        return {
            "ok": True,
            "tour_id": tour_id,
            "images_processed": downloaded,
            # Top-level URLs: apps/api extractModelUrl / extractAuxUrls read these on runpod.output
            "glb_url": glb_url,
            "ply_url": ply_url,
            # Mesh GLB (triangles). Pollers look for keys like mesh_asset_url / mesh_url / glb_url.
            "mesh_asset_url": mesh_url,
            "output": {"ply_url": ply_url, "glb_url": glb_url, "mesh_asset_url": mesh_url},
            "total_seconds": round(time.time() - t0, 2),
        }
    except Exception as e:
        msg = str(e)
        err_lower = msg.lower()
        if "need_at_least_10_images" in err_lower or "downloaded_only_" in err_lower:
            code = "INSUFFICIENT_IMAGES"
            retryable = False
        elif "colmap" in err_lower:
            code = "COLMAP_FAILED"
            retryable = False
        elif "gaussian_splatting_failed" in err_lower:
            code = "GAUSSIAN_FAILED"
            retryable = False
        elif "missing_supabase_env" in err_lower or "supabase_upload_failed" in err_lower:
            code = "UPLOAD_FAILED"
            retryable = True
        elif "glb_convert_failed" in err_lower or "ply_" in err_lower:
            code = "EXPORT_FAILED"
            retryable = False
        else:
            code = "UNKNOWN"
            retryable = False
        return {
            "ok": False,
            "error": msg,
            "error_code": code,
            "retryable": retryable,
            "total_seconds": round(time.time() - t0, 2),
        }
    finally:
        print(f"[job] total_seconds={time.time() - t0:.2f} tour_id={tour_id}", flush=True)
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})


import os
import shutil
import subprocess
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
COLMAP_MAX_IMAGE_SIZE = os.environ.get("COLMAP_MAX_IMAGE_SIZE", "2000").strip()  # pixels
COLMAP_SINGLE_CAMERA = os.environ.get("COLMAP_SINGLE_CAMERA", "1").strip()  # "1" or "0"
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

    downloaded = 0
    total_urls = len(image_urls) if hasattr(image_urls, "__len__") else None
    for i, url in enumerate(image_urls):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
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


def run_colmap(work_dir: Path) -> Path:
    # gaussian-splatting expects COLMAP outputs under the dataset root:
    #   <dataset>/images
    #   <dataset>/sparse/0
    dataset_dir = work_dir / "input"
    images_dir = dataset_dir / "images"
    sparse_root = dataset_dir / "sparse"
    sparse_root.mkdir(parents=True, exist_ok=True)
    # COLMAP mapper writes to <output_path>/<model_id>/..., we ensure at least "0" exists.
    db_path = dataset_dir / "colmap.db"

    def attempt(use_gpu: bool) -> None:
        gpu_flag = "1" if use_gpu else "0"
        # Ensure headless execution; prevents Qt/X11 SIGABRT on serverless workers.
        colmap_env = {
            "QT_QPA_PLATFORM": QT_QPA_PLATFORM or "offscreen",
            "DISPLAY": "",
        }
        print(f"[colmap] feature_extractor use_gpu={gpu_flag} max_image_size={COLMAP_MAX_IMAGE_SIZE}")
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
                str(COLMAP_MAX_IMAGE_SIZE),
            ],
            env=colmap_env,
        )

        print(f"[colmap] sequential_matcher use_gpu={gpu_flag}")
        _run(
            [
                "colmap",
                "sequential_matcher",
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
    job_input = job.get("input", {}) or {}
    image_urls = job_input.get("image_urls", []) or []
    tour_id = str(job_input.get("tour_id", "unknown")).strip()
    iterations = int(job_input.get("iterations", 15000) or 15000)

    if not image_urls:
        return {"ok": False, "error": "no_image_urls"}
    if len(image_urls) < 10:
        return {"ok": False, "error": f"need_at_least_10_images_got_{len(image_urls)}"}

    work_dir = Path(f"/workspace/job_{tour_id}")

    try:
        # 1) Download
        images_dir, downloaded = download_images(image_urls, work_dir)
        if downloaded < 10:
            return {"ok": False, "error": f"downloaded_only_{downloaded}"}

        # 2) COLMAP (gpu -> cpu fallback)
        print("[job] running COLMAP…")
        gs_source = run_colmap(work_dir)

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
        max_points = int(job_input.get("max_points", os.environ.get("PLY_MAX_POINTS", "500000")) or 500000)
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

        # 6) Upload outputs
        bucket = os.environ.get("SUPABASE_SPLATS_BUCKET", "splats").strip() or "splats"
        ply_remote = f"{bucket}/{tour_id}/point_cloud.ply"
        glb_remote = f"{bucket}/{tour_id}/point_cloud.glb"
        ply_url = upload_to_supabase(ply, ply_remote)
        glb_url = upload_to_supabase(glb, glb_remote)
        print(f"[glb] uploaded glb_url={glb_url}", flush=True)

        return {
            "ok": True,
            "tour_id": tour_id,
            "images_processed": downloaded,
            # Top-level URLs: apps/api extractModelUrl / extractAuxUrls read these on runpod.output
            "glb_url": glb_url,
            "ply_url": ply_url,
            "output": {"ply_url": ply_url, "glb_url": glb_url},
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})


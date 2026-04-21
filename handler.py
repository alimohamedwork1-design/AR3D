import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Tuple

import requests
import runpod


SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").strip()
# Use service role for Storage uploads from server-side workers.
SUPABASE_SERVICE_ROLE_KEY = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()

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
    output_dir = work_dir / "output"
    output_dir.mkdir(exist_ok=True)

    gs_path = Path("/workspace/gaussian-splatting/train.py")
    if not gs_path.exists():
        raise RuntimeError("gaussian-splatting not installed in image")

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
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("missing_supabase_env: SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY")

    # Storage upload endpoint expects PUT.
    url = f"{SUPABASE_URL}/storage/v1/object/{remote_path.lstrip('/')}"
    with local_path.open("rb") as f:
        resp = requests.put(
            url,
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/octet-stream",
                "x-upsert": "true",
            },
            data=f,
            timeout=120,
        )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"supabase_upload_failed: http={resp.status_code} body={resp.text[:400]}")

    # If bucket is public you can use public URL; otherwise this is just a stable path to sign later.
    return f"{SUPABASE_URL}/storage/v1/object/public/{remote_path.lstrip('/')}"


def handler(job):
    job_input = job.get("input", {}) or {}
    image_urls = job_input.get("image_urls", []) or []
    tour_id = str(job_input.get("tour_id", "unknown")).strip()
    iterations = int(job_input.get("iterations", 500) or 500)

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

        # 5) Upload (PLY now; GLB requires a separate conversion/export step)
        bucket = os.environ.get("SUPABASE_SPLATS_BUCKET", "splats").strip() or "splats"
        remote = f"{bucket}/{tour_id}/point_cloud.ply"
        ply_url = upload_to_supabase(ply, remote)

        return {
            "ok": True,
            "tour_id": tour_id,
            "images_processed": downloaded,
            "output": {"ply_url": ply_url},
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})


import os, json, subprocess, requests, shutil, uuid
from pathlib import Path
import runpod

def download_images(image_urls, images_dir: Path):
    images_dir.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(image_urls):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        (images_dir / f"img_{i:04d}.jpg").write_bytes(r.content)

def upload_to_supabase(file_path: Path, object_path: str) -> str:
    supabase_url = os.environ["SUPABASE_URL"].rstrip("/")
    supabase_key = os.environ["SUPABASE_KEY"]

    headers = {
        "Authorization": f"Bearer {supabase_key}",
        "apikey": supabase_key,
        "Content-Type": "application/zip",
    }

    with file_path.open("rb") as f:
        resp = requests.post(
            f"{supabase_url}/storage/v1/object/{object_path}",
            headers=headers,
            data=f,
            timeout=300,
        )

    if resp.status_code >= 300:
        raise RuntimeError(f"Supabase upload failed: {resp.status_code} {resp.text}")

    return f"{supabase_url}/storage/v1/object/public/{object_path}"

def handler(job):
    inp = job.get("input", {})
    image_urls = inp.get("image_urls", [])
    tour_id = inp.get("tour_id", "unknown")
    iterations = str(inp.get("iterations", 3000))

    if not image_urls:
        return {"error": "No image_urls provided"}

    job_id = str(uuid.uuid4())[:8]
    work_dir = Path(f"/tmp/gs_{tour_id}_{job_id}")
    images_dir = work_dir / "images"
    out_dir = work_dir / "output"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) download images
    download_images(image_urls, images_dir)

    # 2) run training
    # NOTE: هذا هو الجزء الوحيد اللي قد يحتاج تعديل حسب تركيب image.
    # جرّب أولًا -s images_dir، ولو المشروع يتوقع structure أخرى غيّره.
    cmd = [
        "python", "train.py",
        "-s", str(images_dir),
        "--iterations", iterations,
        "--model_path", str(out_dir),
    ]
    result = subprocess.run(cmd, cwd="/workspace", capture_output=True, text=True)

    if result.returncode != 0:
        return {
            "error": "Training failed",
            "stderr": result.stderr[-4000:],
            "stdout": result.stdout[-4000:],
        }

    # 3) zip output directory
    zip_base = work_dir / f"{tour_id}_output_{job_id}"
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=out_dir))

    # 4) upload zip
    object_path = f"splats/{tour_id}/output_{job_id}.zip"
    public_url = upload_to_supabase(zip_path, object_path)

    return {"tour_id": tour_id, "result_zip_url": public_url, "job_id": job_id}

runpod.serverless.start({"handler": handler})

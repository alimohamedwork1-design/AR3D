import os, subprocess, requests, shutil, uuid
from pathlib import Path
import runpod

GS_DIR = Path("/workspace/gaussian-splatting")

def download_images(image_urls, images_dir: Path):
    images_dir.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(image_urls):
        r = requests.get(url, timeout=120)
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
            timeout=600,
        )

    if resp.status_code >= 300:
        raise RuntimeError(f"Supabase upload failed: {resp.status_code} {resp.text}")

    return f"{supabase_url}/storage/v1/object/public/{object_path}"

def run(cmd, cwd: Path):
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDERR:\n{p.stderr[-4000:]}\n\nSTDOUT:\n{p.stdout[-4000:]}"
        )
    return p.stdout

def handler(job):
    inp = job.get("input", {})
    image_urls = inp.get("image_urls", [])
    tour_id = inp.get("tour_id", "unknown")
    iterations = str(inp.get("iterations", 3000))

    if not image_urls:
        return {"error": "No image_urls provided"}

    job_id = str(uuid.uuid4())[:8]
    work_dir = Path(f"/tmp/gs_{tour_id}_{job_id}")
    images_dir = work_dir / "input_images"
    dataset_dir = work_dir / "dataset"
    out_dir = work_dir / "output"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) Download images
    download_images(image_urls, images_dir)

    # 2) Prepare dataset structure expected by gaussian-splatting: dataset/images
    (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    for f in images_dir.glob("*.jpg"):
        shutil.copy2(f, dataset_dir / "images" / f.name)

   # 3) Convert (COLMAP step) + تأكد إن colmap موجود
run(["bash", "-lc", "which colmap && colmap --version"], cwd=GS_DIR)

run(["python3", "convert.py", "-s", str(dataset_dir)], cwd=GS_DIR)

# Debug: اعرض أهم الملفات اللي اتعملت بعد convert
run(["bash", "-lc", f"ls -R {dataset_dir} | head -n 200"], cwd=GS_DIR)

# 4) Train
out_dir.mkdir(parents=True, exist_ok=True)
run(
    ["python3", "train.py", "-s", str(dataset_dir), "-m", str(out_dir), "--iterations", iterations],
    cwd=GS_DIR
)


    # 5) Zip output directory
    zip_base = work_dir / f"{tour_id}_output_{job_id}"
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=out_dir))

    # 6) Upload zip
    object_path = f"splats/{tour_id}/output_{job_id}.zip"
    public_url = upload_to_supabase(zip_path, object_path)

    return {"tour_id": tour_id, "result_zip_url": public_url, "job_id": job_id}

runpod.serverless.start({"handler": handler})

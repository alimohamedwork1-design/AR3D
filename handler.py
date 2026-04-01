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

def run(cmd, cwd

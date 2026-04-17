import runpod
import os
import requests
import subprocess
import shutil
from pathlib import Path

SUPABASE_URL = os.environ.get("https://ecypnsmehlznwbfongxt.supabase.co", "")
SUPABASE_KEY = os.environ.get("sb_publishable_g2lFyE11sRT9GvDeCIQHpw_dqiBT-GG", "")

def download_images(image_urls, work_dir):
    """Download images from URLs to local directory"""
    images_dir = work_dir / "input" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    for i, url in enumerate(image_urls):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                ext = url.split(".")[-1].lower()
                if ext not in ["jpg", "jpeg", "png"]:
                    ext = "jpg"
                filepath = images_dir / f"img_{i:04d}.{ext}"
                with open(filepath, "wb") as f:
                    f.write(r.content)
                downloaded += 1
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    
    print(f"Downloaded {downloaded}/{len(image_urls)} images")
    return images_dir, downloaded

def run_colmap(work_dir):
    """Run COLMAP for camera pose estimation"""
    db_path = work_dir / "colmap.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    images_dir = work_dir / "input" / "images"

    # Feature extraction
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1"
    ], check=True, capture_output=True)

    # Feature matching
    subprocess.run([
       "colmap", "sequential_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1"
    ], check=True, capture_output=True)

    # Reconstruction
    subprocess.run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir)
    ], check=True, capture_output=True)

    print("COLMAP done")
    return sparse_dir

def run_gaussian_splatting(work_dir, iterations=500):
    """Run Gaussian Splatting training"""
    output_dir = work_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    gs_path = Path("/workspace/gaussian-splatting/train.py")
    
    result = subprocess.run([
        "python", str(gs_path),
        "-s", str(work_dir / "input"),
        "--model_path", str(output_dir),
        "--iterations", str(iterations),
        "--quiet"
    ], capture_output=True, text=True, cwd="/workspace/gaussian-splatting")
    
    if result.returncode != 0:
        print("GS Error:", result.stderr[-2000:])
        raise Exception(f"Gaussian Splatting failed: {result.stderr[-500:]}")
    
    print("Gaussian Splatting done")
    return output_dir

def upload_to_supabase(file_path, tour_id):
    """Upload result file to Supabase storage"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise Exception("Supabase credentials not set")
    
    bucket = "splats"
    filename = f"{tour_id}/point_cloud.ply"
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{filename}"
    
    with open(file_path, "rb") as f:
        response = requests.post(
            upload_url,
            headers={
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/octet-stream"
            },
            data=f
        )
    
    if response.status_code not in [200, 201]:
        raise Exception(f"Upload failed: {response.text}")
    
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}"
    return public_url

def handler(job):
    """Main RunPod handler"""
    job_input = job.get("input", {})
    image_urls = job_input.get("image_urls", [])
    tour_id = job_input.get("tour_id", "unknown")
    iterations = job_input.get("iterations", 500)    
    print(f"Starting job: tour_id={tour_id}, images={len(image_urls)}")
    
    if not image_urls:
        return {"error": "No image_urls provided"}
    
    if len(image_urls) < 10:
        return {"error": f"Need at least 10 images, got {len(image_urls)}"}
    
    work_dir = Path(f"/workspace/job_{tour_id}")
    
    try:
        # 1. Download images
        images_dir, downloaded = download_images(image_urls, work_dir)
        if downloaded < 10:
            return {"error": f"Only {downloaded} images downloaded successfully"}
        
        # 2. Run COLMAP
        print("Running COLMAP...")
        run_colmap(work_dir)
        
        # 3. Run Gaussian Splatting
        print(f"Running Gaussian Splatting ({iterations} iterations)...")
        output_dir = run_gaussian_splatting(work_dir, iterations)
        
        # 4. Find output file
        ply_files = list(output_dir.rglob("point_cloud.ply"))
        if not ply_files:
            return {"error": "No output file generated"}
        
        output_file = ply_files[0]
        print(f"Output file: {output_file} ({output_file.stat().st_size} bytes)")
        
        # 5. Upload to Supabase
        print("Uploading to Supabase...")
        splat_url = upload_to_supabase(output_file, tour_id)
        
        return {
            "success": True,
            "splat_url": splat_url,
            "tour_id": tour_id,
            "images_processed": downloaded
        }
        
    except Exception as e:
        print(f"Job failed: {e}")
        return {"error": str(e)}
    
    finally:
        # Cleanup
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

# Start RunPod serverless
runpod.serverless.start({"handler": handler})

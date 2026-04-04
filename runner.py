import subprocess
import os

# فولدرات
job_dir = "/tmp/job_ad6f2302-0dd2-428b-bc6b-97cb1a51bd5d"
image_path = os.path.join(job_dir, "input/images")
db_path = os.path.join(job_dir, "colmap.db")

os.makedirs(image_path, exist_ok=True)

# تشغيل COLMAP
try:
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_path,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0"  # خليها 0 لو GPU مش جاهز
    ], check=True)
    print("COLMAP finished successfully!")
except subprocess.CalledProcessError as e:
    print("COLMAP crashed!", e)

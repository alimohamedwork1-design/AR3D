from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import subprocess
import os

app = FastAPI()

# create folders
os.makedirs("outputs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# serve viewer
app.mount("/", StaticFiles(directory="viewer", html=True), name="viewer")

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    paths = []
    for file in files:
        path = f"uploads/{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())
        paths.append(path)

    return {"files": paths}


@app.post("/run")
def run_model():
    subprocess.run(["python", "app/runner.py"])
    return {"status": "model generated"}

"""Integrity-checked source and pinned public dependencies before accepting jobs."""
import base64,hashlib,io,os
from pathlib import Path
import runpy,subprocess,sys,zipfile

data=base64.b64decode(os.environ['ARQARY_RECON_BUNDLE'],validate=True)
if hashlib.sha256(data).hexdigest()!=os.environ['ARQARY_RECON_SHA256']:
    raise RuntimeError('source_digest_mismatch')
root=Path('/tmp/arqary-gsplat-runtime');root.mkdir(mode=0o700,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as bundle:
    names={'gsplat_backend.py','gsplat_smoke.py','reconstruction_worker.py','LICENSE.gsplat',
        'handler.py','asset_transfer.py','private_scan.py','photogrammetry_fixture.py'}
    if set(bundle.namelist())!=names or len(bundle.namelist())!=len(names):
        raise RuntimeError('source_inventory_mismatch')
    for name in names:
        content=bundle.read(name)
        if len(content)>100_000:raise RuntimeError('source_size_limit')
        (root/name).write_bytes(content)
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-cache-dir',
    'gsplat==1.5.3','pycolmap==4.2.1','ninja==1.13.2','jaxtyping==0.3.9','scipy==1.18.1','pillow==13.0.0',
    'requests==2.34.2','plyfile==1.1.5','pygltflib==1.16.5'],check=True,timeout=180)
os.environ['MAX_JOBS']='4'
os.environ['OMP_NUM_THREADS']='4'
os.environ['GS_ITERATIONS_DEFAULT']='3000'
os.environ['COLMAP_PYTHON_ENGINE']='1'
sys.path.insert(0,str(root))
runpy.run_path(str(root/'reconstruction_worker.py'),run_name='__main__')

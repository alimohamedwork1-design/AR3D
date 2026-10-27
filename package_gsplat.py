"""Deterministic source-only reconstruction release, never model/data/credentials."""
import base64,hashlib,io,json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
IMAGE='runpod/flash@sha256:039efd6638ec6d6728246bb5130d8976c2f39be357c4ddd1e71866b238474848'


def package():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(['gsplat_backend.py','gsplat_smoke.py','reconstruction_worker.py','LICENSE.gsplat',
                            'handler.py','asset_transfer.py','private_scan.py','photogrammetry_fixture.py']):
            content=(ROOT/name).read_bytes()
            if name.endswith('.py'):compile(content,name,'exec')
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            bundle.writestr(info,content)
    data=buffer.getvalue()
    return {'image':IMAGE,'disk':25,'args':json.dumps({'entrypoint':['/opt/nvidia/nvidia_entrypoint.sh'],
        'cmd':['python','-u','-c',(ROOT/'bootstrap_gsplat.py').read_text()]}),
        'env':{'ARQARY_RECON_BUNDLE':base64.b64encode(data).decode(),
               'ARQARY_RECON_SHA256':hashlib.sha256(data).hexdigest()}}


if __name__=='__main__':print(json.dumps(package()))

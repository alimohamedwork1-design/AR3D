"""Fixed-source operator diagnostics and private reconstruction routing."""
import os
import shutil


def handler(job):
    body=job.get('input') if isinstance(job,dict) else None
    if body=={'action':'deployment_health'}:
        import torch,pycolmap,importlib.metadata
        return {'source_sha256':os.environ.get('ARQARY_RECON_SHA256'),
            'torch':torch.__version__,'cuda':torch.cuda.is_available(),'nvcc':shutil.which('nvcc'),
            'gsplat':importlib.metadata.version('gsplat'),'pycolmap':pycolmap.__version__,
            'engine_license':'Apache-2.0','private_jobs_enabled':os.environ.get('ARQARY_PRIVATE_PROTOCOL_ENABLED')=='true'}
    if body=={'action':'synthetic_smoke'}:
        from gsplat_smoke import run
        return run()
    if body=={'action':'photogrammetry_smoke'}:
        from gsplat_smoke import photogrammetry_run
        return photogrammetry_run()
    if body=={'action':'photogrammetry_quality'}:
        from gsplat_smoke import photogrammetry_run
        return photogrammetry_run(iterations=3000)
    if isinstance(body,dict) and body.get('protocol_version')=='arqary-private-scan-v1':
        from handler import handler as private_handler
        return private_handler(job)
    return {'failure_code':'private_reconstruction_not_yet_enabled'}


if __name__=='__main__':
    import runpod
    runpod.serverless.start({'handler':handler})

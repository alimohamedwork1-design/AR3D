"""Bounded COLMAP + gsplat reconstruction, without GraphDeco CUDA extensions.

SPDX-License-Identifier: Apache-2.0
Training recipe adapted from gsplat v1.5.3 examples/simple_trainer.py (Apache-2.0).
The model format remains standard Gaussian PLY; coordinates are SfM-relative.
"""
import argparse
import json
import math
from pathlib import Path
import time


def contained_image(root, name):
    root = Path(root).resolve()
    path = root / name
    if not isinstance(name, str) or Path(name).name != name or path.is_symlink() or not path.is_file() or path.resolve().parent != root:
        raise ValueError('unsafe_reconstruction_image')
    return path


def prepare_scene(work_dir, max_image_size=1600):
    import pycolmap
    root = Path(work_dir)
    images = root / 'input/images' if (root / 'input/images').is_dir() else root / 'images'
    names = sorted(p.name for p in images.iterdir() if p.is_file())
    if not 10 <= len(names) <= 500:
        raise ValueError('reconstruction_image_count')
    for name in names:
        contained_image(images, name)
    database = root / 'colmap.db'
    sparse = root / 'sparse'
    sparse.mkdir(exist_ok=True)
    extract = pycolmap.FeatureExtractionOptions()
    extract.num_threads = 4
    extract.max_image_size = min(2400, max(256, int(max_image_size or 1600)))
    match = pycolmap.FeatureMatchingOptions()
    match.num_threads = 4
    pycolmap.extract_features(database, images, image_names=names,
        extraction_options=extract, device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(database, matching_options=match, device=pycolmap.Device.cpu)
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 4
    models = pycolmap.incremental_mapping(database, images, sparse, options=options)
    if not models:
        raise RuntimeError('camera_registration_failed')
    best_key = max(models, key=lambda key: models[key].num_reg_images())
    if models[best_key].num_reg_images() < max(10, math.ceil(len(names) * 0.8)):
        raise RuntimeError('insufficient_registered_views')
    output = root / 'input_undist'
    pycolmap.undistort_images(output, sparse / str(best_key), images,
        undistort_options=pycolmap.UndistortCameraOptions({'max_image_size':1600}), num_threads=4)
    return output


def load_scene(source, max_side=800):
    import numpy as np
    import pycolmap
    from PIL import Image
    source = Path(source)
    sparse = source / 'sparse/0'
    if not sparse.is_dir():
        sparse = source / 'sparse'
    model = pycolmap.Reconstruction(sparse)
    points = np.array([p.xyz for p in model.points3D.values()], dtype=np.float32)
    colors = np.array([p.color for p in model.points3D.values()], dtype=np.float32) / 255
    if points.ndim != 2 or points.shape[1] != 3 or not 10 <= len(points) <= 500_000 or not np.isfinite(points).all():
        raise ValueError('invalid_sparse_geometry')
    views = []
    for _, image in sorted(model.images.items()):
        camera = model.cameras[image.camera_id]
        if camera.model_name != 'PINHOLE':
            raise ValueError('undistorted_pinhole_required')
        path = contained_image(source / 'images', image.name)
        with Image.open(path) as original:
            original.load()
            pixels = original.convert('RGB')
            pixels.thumbnail((max_side, max_side))
            width, height = pixels.size
            scale_x, scale_y = width / camera.width, height / camera.height
            fx, fy, cx, cy = camera.params
            intrinsic = np.array([[fx*scale_x,0,cx*scale_x],[0,fy*scale_y,cy*scale_y],[0,0,1]], dtype=np.float32)
            pose = np.eye(4, dtype=np.float32)
            pose[:3] = image.cam_from_world().matrix()
            views.append((np.asarray(pixels, dtype=np.float32) / 255, intrinsic, pose))
    if not 10 <= len(views) <= 500:
        raise ValueError('registered_views_required')
    return points, colors, views


def train_arrays(points, colors, views, output, *, iterations=3000, deadline_seconds=300, max_gaussians=200_000):
    import numpy as np
    import torch
    from gsplat import rasterization, export_splats
    from gsplat.strategy import DefaultStrategy
    from scipy.spatial import cKDTree
    if not torch.cuda.is_available():
        raise RuntimeError('cuda_required')
    if not 1 <= iterations <= 15000 or not 1 <= deadline_seconds <= 900:
        raise ValueError('invalid_training_budget')
    torch.set_num_threads(4)
    torch.manual_seed(20261027)
    rng = np.random.default_rng(20261027)
    if len(points) > 50_000:
        keep = rng.choice(len(points), 50_000, replace=False)
        points, colors = points[keep], colors[keep]
    distance = cKDTree(points).query(points, k=4)[0][:,1:]
    scales = np.log(np.maximum(1e-5, np.sqrt((distance**2).mean(axis=-1)))).astype(np.float32)
    scene_scale = max(float(np.linalg.norm(points - np.median(points, axis=0), axis=-1).mean()), 1e-3)
    parameters = torch.nn.ParameterDict({
        'means':torch.nn.Parameter(torch.tensor(points, device='cuda')),
        'scales':torch.nn.Parameter(torch.tensor(np.repeat(scales[:,None],3,axis=1), device='cuda')),
        'quats':torch.nn.Parameter(torch.nn.functional.normalize(torch.randn(len(points),4,device='cuda'),dim=-1)),
        'opacities':torch.nn.Parameter(torch.full((len(points),), math.log(0.1/0.9),device='cuda')),
        'sh0':torch.nn.Parameter(torch.tensor((colors-0.5)/0.28209479177387814,device='cuda')[:,None,:]),
        'shN':torch.nn.Parameter(torch.zeros(len(points),15,3,device='cuda')),
    })
    rates = {'means':1.6e-4*scene_scale,'scales':5e-3,'quats':1e-3,'opacities':5e-2,'sh0':2.5e-3,'shN':1.25e-4}
    optimizers = {key:torch.optim.Adam([value],lr=rates[key],eps=1e-15) for key,value in parameters.items()}
    strategy = DefaultStrategy(refine_start_iter=500,refine_stop_iter=min(15000,iterations),reset_every=3000)
    strategy.check_sanity(parameters,optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    started = time.monotonic()
    losses = []
    for step in range(iterations):
        if time.monotonic()-started > deadline_seconds:
            raise TimeoutError('training_deadline_exceeded')
        if len(parameters['means']) > max_gaussians:
            raise RuntimeError('gaussian_count_budget_exceeded')
        pixels,intrinsic,pose = views[step % len(views)]
        target = torch.tensor(pixels,device='cuda')[None]
        rendered,_,info = rasterization(means=parameters['means'],quats=parameters['quats'],
            scales=parameters['scales'].exp(),opacities=parameters['opacities'].sigmoid(),
            colors=torch.cat([parameters['sh0'],parameters['shN']],dim=1),
            viewmats=torch.tensor(pose,device='cuda')[None],Ks=torch.tensor(intrinsic,device='cuda')[None],
            width=pixels.shape[1],height=pixels.shape[0],sh_degree=min(step//1000,3),packed=False)
        strategy.step_pre_backward(parameters,optimizers,state,step,info)
        # Pure PyTorch SSIM avoids the separate GraphDeco/fused-ssim extension.
        a,b = rendered.permute(0,3,1,2),target.permute(0,3,1,2)
        avg = lambda x: torch.nn.functional.avg_pool2d(x,7,1,3)
        ma,mb = avg(a),avg(b)
        va,vb,cov = avg(a*a)-ma*ma,avg(b*b)-mb*mb,avg(a*b)-ma*mb
        ssim = ((2*ma*mb+0.01**2)*(2*cov+0.03**2)/((ma*ma+mb*mb+0.01**2)*(va+vb+0.03**2))).mean()
        loss = 0.8*torch.nn.functional.l1_loss(rendered,target)+0.2*(1-ssim)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite_training_loss')
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        strategy.step_post_backward(parameters,optimizers,state,step,info,packed=False)
        losses.append(float(loss.detach()))
    output = Path(output)
    path = output / 'point_cloud' / ('iteration_'+str(iterations)) / 'point_cloud.ply'
    path.parent.mkdir(parents=True,exist_ok=True)
    export_splats(**{key:value.detach() for key,value in parameters.items()},format='ply',save_to=str(path))
    stats = {'engine':'gsplat','version':'1.5.3','license':'Apache-2.0','iterations':iterations,
        'views':len(views),'gaussians':len(parameters['means']),'first_loss':losses[0],'last_loss':losses[-1],
        'seconds':round(time.monotonic()-started,3),'coordinate_space':'sfm_relative_not_metric'}
    (output/'training-stats.json').write_text(json.dumps(stats,allow_nan=False))
    return stats


def train(source, output, iterations=3000):
    points,colors,views = load_scene(source)
    return train_arrays(points,colors,views,output,iterations=iterations)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--iterations',type=int,default=3000)
    args=parser.parse_args()
    train(args.source,args.output,args.iterations)

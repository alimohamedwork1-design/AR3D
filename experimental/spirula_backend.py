"""Operator-only Spirula evaluation. Not wired into paid/customer dispatch.

Invokes the separately installed GPL-3.0 executable; no upstream binary/code is
bundled. Produces an unpublished local artifact for the existing PLY/SOG path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

RELEASE = 'v2026.10.16'
UPSTREAM_COMMIT = '65246c5579c127081bff6f4422bd12a542906f78'
BINARY_SHA256 = 'f29ef66ed881a3d5a5c9f2e1f1bb587c21617442eed16ee9d11830150f41ae03'
ARCHIVE_SHA256 = 'df5bb7acbfea63c16497c229b70c7a05ed13c22807554dee83cd6415aa411bf3'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checked_run(command, log, deadline):
    """One shared wall clock, including SfM; kill the whole process group."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('spirula_deadline_exceeded')
    with Path(log).open('wb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise TimeoutError('spirula_deadline_exceeded') from None
    if code:
        raise RuntimeError(f'spirula_process_failed:{code}; inspect {Path(log).name}')


def checked_images(source):
    source = Path(source).resolve(strict=True)
    if not source.is_dir():
        raise ValueError('image_directory_required')
    images = sorted(source.iterdir())
    if not 10 <= len(images) <= 200:
        raise ValueError('image_count_must_be_10_to_200')
    total = 0
    for image in images:
        if image.is_symlink() or not image.is_file() or image.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            raise ValueError('regular_local_images_required')
        total += image.stat().st_size
    if total > 256 * 1024 * 1024:
        raise ValueError('input_byte_budget_exceeded')
    return images


def train_command(binary, dataset, output, iterations, max_gaussians):
    if type(iterations) is not int or not 100 <= iterations <= 5000:
        raise ValueError('iterations_must_be_100_to_5000')
    if type(max_gaussians) is not int or not 1000 <= max_gaussians <= 200000:
        raise ValueError('gaussian_budget_must_be_1000_to_200000')
    return [str(binary), 'train', '3dgs', '--data', str(dataset),
            '--output-dir-prefix', str(output), '--output-dir-name', 'model',
            '--num-iterations', str(iterations), '--cap-max', str(max_gaussians),
            '--train-resolution-divisor', '1', '--sh-degree', '3',
            '--disable-viewer', '1', '--keep-viewer-alive', '0',
            '--steps-per-save', str(iterations), '--save-full-checkpoint', '0',
            '--eval-mode', 'interval', '--eval-interval', '4', '--save-eval-images', '1']


def run(binary, images, output, *, experimental=False, iterations=3000,
        max_gaussians=200000, timeout=600):
    if experimental is not True:
        raise PermissionError('explicit_experimental_opt_in_required')
    if type(timeout) is not int or not 1 <= timeout <= 900:
        raise ValueError('timeout_must_be_1_to_900_seconds')
    binary = Path(binary).resolve(strict=True)
    if not binary.is_file() or digest(binary) != BINARY_SHA256:
        raise ValueError('pinned_spirula_binary_required')
    files = checked_images(images)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('new_output_directory_required')
    # Validate all budgets before any filesystem mutation or GPU process.
    command = train_command(binary, output / 'dataset', output, iterations, max_gaussians)
    output.mkdir(parents=True)
    dataset = output / 'dataset'
    (dataset / 'images').mkdir(parents=True)
    manifest = {'schema': 'arqary.spirula.experiment.v1', 'engine': 'spirula',
                'release': RELEASE, 'upstream_commit': UPSTREAM_COMMIT,
                'binary_sha256': BINARY_SHA256, 'engine_license': 'GPL-3.0',
                'state': 'running', 'eligible_for_publication': False,
                'coordinate_space': 'reconstruction_relative_not_metric',
                'iterations': iterations, 'max_gaussians': max_gaussians,
                'timeout_seconds': timeout, 'input': [], 'stages': {}}
    receipt = output / 'experiment.json'
    started = time.monotonic()
    deadline = started + timeout
    try:
        from PIL import Image
        for index, file in enumerate(files):
            # Decode/re-encode before native parsing; fixed paths avoid CLI/control names.
            with Image.open(file) as original:
                if original.width * original.height > 24000000:
                    raise ValueError('image_pixel_budget_exceeded')
                image = original.convert('RGB')
                image.thumbnail((640, 640))
                name = f'frame_{index:04d}.png'
                image.save(dataset / 'images' / name)
            manifest['input'].append({'name': name, 'sha256': digest(file),
                                      'prepared_sha256': digest(dataset / 'images' / name)})
        stage = time.monotonic()
        checked_run([str(binary), 'sfm', 'auto', str(dataset), '-o', str(dataset),
                     '--quality', 'low'], output / 'sfm.log', deadline)
        manifest['stages']['sfm_seconds'] = round(time.monotonic() - stage, 3)
        import pycolmap
        model = pycolmap.Reconstruction(dataset / 'sparse/0')
        manifest['registered_views'] = model.num_reg_images()
        manifest['input_views'] = len(files)
        if model.num_reg_images() < max(10, (len(files) * 8 + 9) // 10):
            raise RuntimeError('insufficient_registered_views')
        if not 10 <= model.num_points3D() <= max_gaussians:
            raise RuntimeError('sparse_seed_budget_exceeded')
        stage = time.monotonic()
        checked_run(command, output / 'train.log', deadline)
        manifest['stages']['training_seconds'] = round(time.monotonic() - stage, 3)
        candidate = output / 'model' / f'step-{iterations:09d}.ckpt' / 'splat.ply'
        if not candidate.is_file() or candidate.is_symlink():
            raise RuntimeError('missing_gaussian_output')
        # The evaluation export is copied to one stable local delivery candidate.
        from plyfile import PlyData
        import numpy as np
        if candidate.stat().st_size > 128 * 1024 * 1024:
            raise ValueError('output_byte_budget_exceeded')
        ply = PlyData.read(candidate)
        vertex = ply['vertex'].data
        required = ['x', 'y', 'z', 'opacity', 'f_dc_0', 'f_dc_1', 'f_dc_2']
        required += [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
        if not 1 <= len(vertex) <= max_gaussians or not set(required).issubset(vertex.dtype.names):
            raise ValueError('unsupported_gaussian_output')
        # Reject bad rotations; normalize finite nonzero exported quaternions only.
        for name in vertex.dtype.names:
            if vertex[name].dtype.kind != 'f' or not np.isfinite(vertex[name]).all():
                raise ValueError('nonfinite_gaussian_output')
        rotations = np.stack([vertex[f'rot_{i}'] for i in range(4)], axis=1).astype(np.float64)
        norm = np.linalg.norm(rotations, axis=1)
        if not np.isfinite(norm).all() or (norm < 1e-8).any():
            raise ValueError('invalid_gaussian_rotation')
        for i in range(4):
            vertex[f'rot_{i}'] = rotations[:, i] / norm
        target = output / 'point_cloud.ply'
        ply.write(target)
        metrics = json.loads((output / 'model/metrics.json').read_text())
        if metrics.get('num_eval_images', 0) < 1 or not all(
                isinstance(metrics.get(name), (float, int)) and np.isfinite(metrics[name])
                for name in ['avg_psnr', 'avg_ssim']):
            raise RuntimeError('missing_finite_held_out_metrics')
        manifest['evaluation'] = {'held_out_views': metrics['num_eval_images'],
                                  'psnr_db': metrics['avg_psnr'], 'ssim': metrics['avg_ssim']}
        # Carry a captured viewpoint: fitting all bounds can look at unseen sides
        # of a partial capture. PlayCanvas's viewer rotates loaded splats 180° Z.
        transform = json.loads((output / 'model/scene_transform.json').read_text())
        train_from_world = np.array(transform['train_from_world']['matrix_4x4'])
        image = min(model.images.values(), key=lambda item: item.name)
        pose = np.array(image.cam_from_world().matrix())
        rotation, translation = pose[:, :3], pose[:, 3]
        center = -rotation.T @ translation
        forward = rotation.T @ np.array([0, 0, 1])
        def viewer_point(point):
            p = (train_from_world @ np.append(point, 1))[:3]
            return (p * np.array([-1, -1, 1])).tolist()
        camera = model.cameras[image.camera_id]
        fov = float(np.degrees(2 * np.arctan(camera.height / (2 * camera.focal_length_y))))
        manifest['preview_camera'] = {'initial': {'position': viewer_point(center),
                                                  'target': viewer_point(center + forward),
                                                  'fov': fov}}
        manifest.update(state='completed', gaussians=len(vertex),
                        output={'path': target.name, 'bytes': target.stat().st_size,
                                'sha256': digest(target), 'normalized_rotations': True})
    except Exception as error:
        manifest.update(state='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest['elapsed_seconds'] = round(time.monotonic() - started, 3)
        receipt.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--experimental', action='store_true')
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    print(json.dumps(run(args.binary, args.images, args.output, experimental=args.experimental,
                         iterations=args.iterations, timeout=args.timeout), indent=2))

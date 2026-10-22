"""Private Scanner v1 protocol. Standard library only; no GPU/network at import.

The platform owns authorization, atomic budget reservation and dispatch claims.
This worker verifies immutable bytes; it never establishes financial consent.
"""
import base64
import hashlib
import hmac
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

PROTOCOL = 'arqary-private-scan-v1'
UUID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}')
PART_BYTES = 8 * 1024 * 1024
MAX_BYTES = 128 * 1024 * 1024
OUTPUTS = {'point_cloud.splat', 'point_cloud.glb', 'point_cloud.ply', 'mesh.glb', 'mesh.ply'}

def _require(condition):
    if not condition:
        raise RuntimeError('private_scan_contract_invalid')

def _uuid(value):
    return isinstance(value, str) and UUID.fullmatch(value) is not None

def _url(value, origins):
    _require(isinstance(value, str) and len(value) <= 16384)
    parsed = urlsplit(value)
    _require(parsed.scheme == 'https' and parsed.hostname and not parsed.username and not parsed.password and not parsed.fragment)
    _require(parsed.netloc in origins)

def validate_private_job(data, *, enabled=False, input_origins=(), output_origins=()):
    _require(enabled and isinstance(data, dict) and data.get('protocol_version') == PROTOCOL)
    _require(set(data) == {'protocol_version', 'capture_mode', 'dispatch_id', 'owner_id', 'session_id', 'tour_id', 'image_manifest', 'image_urls', 'dataset_digest', 'output_targets'})
    _require(data.get('capture_mode') == 'gaussian-splat')
    _require(all(_uuid(data.get(key)) for key in ('dispatch_id', 'owner_id', 'session_id')))
    _require(data.get('tour_id') == data['dispatch_id'])
    _require(not any(key in data for key in ('dataset_url', 'supabase_bucket', 'supabase_image_paths', 'property_id', 'pipeline', 'quality_profile', 'iterations')))
    manifest, urls = data.get('image_manifest'), data.get('image_urls')
    _require(isinstance(manifest, list) and 10 <= len(manifest) <= 500 and isinstance(urls, list) and len(urls) == len(manifest))
    rows, seen = [], set()
    for image, url in zip(manifest, urls):
        _require(isinstance(image, dict) and _uuid(image.get('id')) and image['id'] not in seen)
        _require(set(image) == {'id', 'filename', 'content_type', 'byte_size', 'etag', 'part_checksums'})
        seen.add(image['id'])
        name, size, mime = image.get('filename'), image.get('byte_size'), image.get('content_type')
        _require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,119}', name) and '..' not in name)
        _require({'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp'}.get(name.rsplit('.', 1)[-1].lower()) == mime)
        _require(type(size) is int and 0 < size <= MAX_BYTES and isinstance(image.get('etag'), str) and image['etag'])
        parts = image.get('part_checksums')
        count = (size + PART_BYTES - 1) // PART_BYTES
        _require(isinstance(parts, dict) and set(parts) == {str(i + 1) for i in range(count)})
        checksums = [parts[str(i + 1)] for i in range(count)]
        _require(all(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9+/]{43}=', value) for value in checksums))
        _url(url, input_origins)
        rows.append([image['id'], name, mime, size, image['etag'], checksums])
    _require(sum(row[3] for row in rows) <= 2 * 1024 * 1024 * 1024)
    _require(rows == sorted(rows, key=lambda row: row[0]))
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    _require(isinstance(data.get('dataset_digest'), str) and hmac.compare_digest(digest, data['dataset_digest']))
    targets = data.get('output_targets')
    _require(isinstance(targets, dict) and 1 <= len(targets) <= 5 and set(targets) <= OUTPUTS)
    for name, target in targets.items():
        _require(isinstance(target, dict) and set(target) == {'url', 'reference'})
        _url(target.get('url'), output_origins)
        reference = target.get('reference', '')
        _require(isinstance(reference, str) and reference.startswith('r2:') and '/' in reference)
        asset, planned_name = reference[3:].split('/', 1)
        _require(_uuid(asset) and planned_name == name)
    return json.loads(json.dumps(data))  # Freeze the approved input before processing.

def verify_private_images(data, directory):
    root = Path(directory)
    _require(len(list(root.glob('img_*.*'))) == len(data['image_manifest']))
    for index, image in enumerate(data['image_manifest']):
        files = list(root.glob(f'img_{index:04d}.*'))
        _require(len(files) == 1 and files[0].is_file() and not files[0].is_symlink() and files[0].stat().st_size == image['byte_size'])
        with files[0].open('rb') as stream:
            for part in range(1, len(image['part_checksums']) + 1):
                block = stream.read(PART_BYTES)
                digest = base64.b64encode(hashlib.sha256(block).digest()).decode()
                _require(hmac.compare_digest(digest, image['part_checksums'][str(part)]))
            _require(not stream.read(1))

def measure_private_output(path, name, target):
    path = Path(path)
    _require(name in OUTPUTS and path.is_file() and not path.is_symlink())
    reference = target.get('reference', '')
    _require(isinstance(reference, str) and reference.startswith('r2:') and '/' in reference)
    asset, planned_name = reference[3:].split('/', 1)
    _require(_uuid(asset) and planned_name == name)
    size = path.stat().st_size
    _require(32 <= size <= MAX_BYTES)
    sha, consumed = hashlib.sha256(), 0
    with path.open('rb') as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            consumed += len(chunk)
            _require(consumed <= size)
            sha.update(chunk)
    _require(consumed == size)
    return {'name': name, 'asset_id': target['reference'][3:].split('/')[0], 'byte_size': size, 'sha256': sha.hexdigest()}

def private_result(data, artifacts):
    _require(isinstance(artifacts, dict) and artifacts and set(artifacts) <= set(data['output_targets']))
    records = []
    for name, artifact in artifacts.items():
        _require(isinstance(artifact, dict) and artifact.get('name') == name
                 and artifact.get('asset_id') == data['output_targets'][name]['reference'][3:].split('/')[0]
                 and type(artifact.get('byte_size')) is int and 32 <= artifact['byte_size'] <= MAX_BYTES
                 and isinstance(artifact.get('sha256'), str) and re.fullmatch(r'[a-f0-9]{64}', artifact['sha256']))
        records.append({key: artifact[key] for key in ('name', 'asset_id', 'byte_size', 'sha256')})
    return {'protocol_version': PROTOCOL, 'worker_algorithm': 'gaussian-splat', 'dispatch_id': data['dispatch_id'],
            'dataset_digest': data['dataset_digest'], 'artifacts': records}


def preserve_private_outputs(data, paths, transfer):
    """Only planned generated files; never enter legacy public preservation."""
    artifacts = {}
    for name, target in data['output_targets'].items():
        path = paths.get(name)
        if path is None or not Path(path).is_file():
            continue  # An optional exporter may not have produced this format.
        measured = measure_private_output(path, name, target)
        _require(transfer(path, target) == target['reference'])
        artifacts[name] = measured
    return private_result(data, artifacts)

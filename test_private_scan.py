import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from private_scan import validate_private_job, verify_private_images, measure_private_output, private_result, preserve_private_outputs, PROTOCOL

HOST = 'a' * 32 + '.r2.cloudflarestorage.com'
ID = '10000000-0000-4000-8000-000000000001'

def fixture():
    images = [{'id': f'30000000-0000-4000-8000-{i + 1:012}', 'filename': f'{i}.jpg', 'content_type': 'image/jpeg', 'byte_size': 64, 'etag': f'fixture-{i}', 'part_checksums': {'1': base64.b64encode(hashlib.sha256(bytes([i]) * 64).digest()).decode()}} for i in range(10)]
    rows = [[v['id'], v['filename'], v['content_type'], v['byte_size'], v['etag'], [v['part_checksums']['1']]] for v in images]
    return {'protocol_version': PROTOCOL, 'dispatch_id': ID, 'owner_id': ID, 'session_id': ID, 'tour_id': ID, 'capture_mode': 'gaussian-splat',
            'image_manifest': images, 'image_urls': ['https://' + HOST + '/image.jpg?signature=synthetic'] * 10,
            'dataset_digest': hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest(),
            'output_targets': {'point_cloud.splat': {'url': 'https://' + HOST + '/result', 'reference': f'r2:{ID}/point_cloud.splat'}}}

def validate(data, **options):
    return validate_private_job(data, enabled=True, input_origins=[HOST], output_origins=[HOST], **options)

class PrivateProtocolTests(unittest.TestCase):
    def test_preserves_only_available_planned_outputs(self):
        data = validate(fixture())
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            splat = Path(directory) / 'point_cloud.splat'; splat.write_bytes(bytes(64))
            unplanned = Path(directory) / 'point_cloud.ply'; unplanned.write_bytes(bytes(64))
            def transfer(path, target):
                calls.append(path.name)
                return target['reference']
            result = preserve_private_outputs(data, {'point_cloud.splat': splat, 'point_cloud.ply': unplanned}, transfer)
            self.assertEqual(calls, ['point_cloud.splat'])
            self.assertEqual(result['artifacts'][0]['name'], 'point_cloud.splat')
            with self.assertRaises(RuntimeError): preserve_private_outputs(data, {}, transfer)
            with self.assertRaises(RuntimeError): preserve_private_outputs(data, {'point_cloud.splat': splat}, lambda *_: 'wrong-reference')

    def test_default_disabled(self):
        with self.assertRaises(RuntimeError): validate_private_job(fixture())

    def test_freezes_input(self):
        original = fixture(); result = validate(original)
        original['image_manifest'][0]['filename'] = 'changed.jpg'
        self.assertEqual(result['image_manifest'][0]['filename'], '0.jpg')

    def test_identity_mode_and_dataset(self):
        for change in ({'owner_id': '../private'}, {'tour_id': '../escape'}, {'capture_mode': 'nerf'}, {'dataset_digest': '0' * 64}, {'dataset_url': 'https://public.invalid'}, {'property_id': ID}):
            with self.subTest(change=change), self.assertRaises(RuntimeError): validate({**fixture(), **change})

    def test_origins_and_plan(self):
        for target in ['https://evil.invalid/result', 'http://' + HOST + '/x', 'https://user:pass@' + HOST + '/x', 'https://' + HOST + '/x#private']:
            data = fixture(); data['output_targets']['point_cloud.splat']['url'] = target
            with self.assertRaises(RuntimeError): validate(data)
        data = fixture(); data['output_targets']['point_cloud.splat']['reference'] = f'r2:{ID}/mesh.glb'
        with self.assertRaises(RuntimeError): validate(data)

    def test_duplicate_or_corrupt_manifest(self):
        for mutate in [lambda x: x['image_manifest'].append(x['image_manifest'][0]), lambda x: x['image_manifest'][0].update(byte_size=True), lambda x: x['image_manifest'][0].update(filename='../file.jpg')]:
            data = fixture(); mutate(data)
            with self.assertRaises(RuntimeError): validate(data)

    def test_actual_original_integrity_and_result(self):
        data = validate(fixture())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for i in range(10): (path / f'img_{i:04d}.jpg').write_bytes(bytes([i]) * 64)
            verify_private_images(data, path)
            (path / 'img_0000.jpg').write_bytes(b'x' * 64)
            with self.assertRaises(RuntimeError): verify_private_images(data, path)
            output = path / 'point_cloud.splat'; output.write_bytes(bytes(64))
            artifact = measure_private_output(output, output.name, data['output_targets'][output.name])
            artifact['secret'] = 'must not escape'
            result = private_result(data, {output.name: artifact})
            self.assertNotIn('secret', json.dumps(result))
            forged = copy.deepcopy(artifact); forged['asset_id'] = '20000000-0000-4000-8000-000000000001'
            with self.assertRaises(RuntimeError): private_result(data, {output.name: forged})

if __name__ == '__main__': unittest.main()

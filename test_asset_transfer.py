import tempfile
import unittest
from pathlib import Path
from asset_transfer import download_file, upload_output, extract_dataset, reject_job_credentials
import zipfile
import stat


class Response:
    def __init__(self, chunks, length=None, status=200):
        self.chunks = chunks
        self.headers = {} if length is None else {'Content-Length': str(length)}
        self.status_code = status
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def iter_content(self, **_kwargs):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class Session:
    def __init__(self, response):
        self.response = response

    def get(self, _url, **kwargs):
        assert kwargs['stream'] is True
        assert kwargs['allow_redirects'] is False
        return self.response


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'frame.jpg'

    def transfer(self, response, limit=5):
        return download_file(Session(response), 'https://fixture.invalid/?signature=private', self.path, limit)

    def test_rejects_job_payload_credentials_before_work(self):
        for key in ("supabase_service_role_key", "supabase_service_key",
                    "service_role_key", "SUPABASE_SERVICE_ROLE_KEY"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(RuntimeError, "forbidden_job_credentials"):
                    reject_job_credentials({"tour_id": "test", key: ""})
        reject_job_credentials({"tour_id": "test", "output_targets": {"scene.sog": {"reference": "opaque"}}})

    def test_streamed_file_is_committed_only_when_complete(self):
        response = Response([b'abc', b'de'], 5)
        self.assertEqual(self.transfer(response), 5)
        self.assertEqual(self.path.read_bytes(), b'abcde')
        self.assertTrue(response.closed)

    def test_size_limit_without_content_length(self):
        with self.assertRaisesRegex(RuntimeError, 'size_limit'):
            self.transfer(Response([b'abc', b'def']))
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_rejects_oversize_before_reading(self):
        with self.assertRaisesRegex(RuntimeError, 'size_limit'):
            self.transfer(Response([AssertionError('body must not be read')], 6))

    def test_interruption_preserves_previous_complete_file(self):
        self.path.write_bytes(b'previous')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.transfer(Response([b'ab', RuntimeError('interrupted')]))
        self.assertEqual(self.path.read_bytes(), b'previous')
        self.assertFalse(self.path.with_name('frame.jpg.part').exists())

    def test_empty_and_truncated_responses_fail(self):
        for response in [Response([]), Response([b'ab'], 3)]:
            with self.assertRaises(RuntimeError):
                self.transfer(response)
        self.assertFalse(self.path.exists())

    def test_redirect_does_not_forward_authority(self):
        with self.assertRaisesRegex(RuntimeError, 'http_302'):
            self.transfer(Response([], status=302))

    def test_private_output_streams_to_scoped_target(self):
        self.path.write_bytes(b'abc')
        target={'url':'https://'+'a'*32+'.r2.cloudflarestorage.com/object?signature=secret', 'reference':'r2:10000000-0000-4000-8000-000000000001/model.splat'}
        class UploadSession:
            def put(inner, url, **kwargs):
                self.assertEqual(kwargs['data'].read(), b'abc')
                self.assertEqual(kwargs['headers']['If-None-Match'], '*')
                self.assertFalse(kwargs['allow_redirects'])
                return Response([], status=200)
        self.assertEqual(upload_output(UploadSession(), self.path, target),target['reference'])

    def test_private_output_refuses_arbitrary_upload_host(self):
        self.path.write_bytes(b'abc')
        with self.assertRaisesRegex(RuntimeError,'invalid_output_target'):
            upload_output(None,self.path,{'url':'https://attacker.invalid','reference':'r2:bad'})

    def test_private_output_does_not_log_signed_url_on_failure(self):
        self.path.write_bytes(b'abc')
        class Broken:
            def put(self,*args,**kwargs):
                raise RuntimeError('signed-secret-url')
        with self.assertRaisesRegex(RuntimeError,'^private_output_transfer_failed$'):
            upload_output(Broken(),self.path,{'url':'https://'+'a'*32+'.r2.cloudflarestorage.com/object','reference':'r2:10000000-0000-4000-8000-000000000001/model.splat'})

    def test_dataset_valid_subset(self):
        archive=self.path.parent/'dataset.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('images/a.jpg',b'abc')
            z.writestr('other/b.jpg',b'other')
        extract_dataset(archive,self.path.parent/'output',['images/a.jpg'])
        self.assertEqual((self.path.parent/'output/images/a.jpg').read_bytes(),b'abc')
        self.assertFalse((self.path.parent/'output/other').exists())

    def test_dataset_traversal_and_links_are_rejected(self):
        for name in ['../escape','/absolute','C:/escape','..\\escape']:
            archive=self.path.parent/'dataset.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr(name,b'bad')
            with self.assertRaisesRegex(RuntimeError,'unsafe_path'):extract_dataset(archive,self.path.parent/'output')
        entry=zipfile.ZipInfo('link');entry.external_attr=(stat.S_IFLNK|0o777)<<16
        with zipfile.ZipFile(archive,'w') as z:z.writestr(entry,b'../escape')
        with self.assertRaisesRegex(RuntimeError,'link_rejected'):extract_dataset(archive,self.path.parent/'output')


if __name__ == '__main__':
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from asset_transfer import download_file


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


if __name__ == '__main__':
    unittest.main()

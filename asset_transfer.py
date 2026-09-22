"""Bounded worker transfers. No cloud credentials, GPU imports or network at import."""
import time
from pathlib import Path


def download_file(session, url, destination: Path, max_bytes: int, headers=None):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + '.part')
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    written = 0
    try:
        # Signed storage URLs are direct. Never forward credentials across redirects.
        with session.get(url, headers=headers or {}, stream=True,
                         allow_redirects=False, timeout=(10, 60)) as response:
            if not 200 <= response.status_code < 300:
                raise RuntimeError('download_http_' + str(response.status_code))
            length = response.headers.get('Content-Length')
            if length is not None and (not length.isdigit() or int(length) > max_bytes):
                raise RuntimeError('download_size_limit')
            with temporary.open('wb') as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if time.monotonic() - started > 180:
                        raise RuntimeError('download_time_limit')
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        raise RuntimeError('download_size_limit')
                    output.write(chunk)
            if not written:
                raise RuntimeError('download_empty')
            if length is not None and written != int(length):
                raise RuntimeError('download_incomplete')
        temporary.replace(destination)
        return written
    finally:
        temporary.unlink(missing_ok=True)

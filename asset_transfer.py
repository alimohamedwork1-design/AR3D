"""Bounded worker transfers. No cloud credentials, GPU imports or network at import."""


def reject_job_credentials(job_input):
    """No privileged credentials in third-party job inputs, even empty placeholders."""
    if not isinstance(job_input, dict):
        raise RuntimeError("invalid_job_input")
    forbidden = (
        "supabase_service_role_key",
        "supabase_service_key",
        "service_role_key",
        "SUPABASE_SERVICE_ROLE_KEY",
    )
    if any(key in job_input for key in forbidden):
        raise RuntimeError("forbidden_job_credentials")


import time
import re
from urllib.parse import urlparse
from pathlib import Path
import stat
import zipfile
import shutil


def extract_dataset(archive, destination, members=None):
    """Reject traversal, links and zip bombs before materializing dataset bytes."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if len(entries) > 10000:
            raise RuntimeError('dataset_entry_limit')
        selected = set(members) if members is not None else None
        total = 0
        checked = []
        for entry in entries:
            if selected is not None and entry.filename not in selected:
                continue
            name = entry.filename.replace('\\', '/')
            target = (destination / name).resolve()
            if ':' in name or name.startswith('/') or '..' in name.split('/') or not target.is_relative_to(destination):
                raise RuntimeError('dataset_unsafe_path')
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise RuntimeError('dataset_link_rejected')
            total += entry.file_size
            if entry.file_size > 134217728 or total > 1073741824:
                raise RuntimeError('dataset_size_limit')
            checked.append((entry, target))
        for entry, target in checked:
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(entry) as incoming, target.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)


def upload_output(session, local_path, target):
    """Write a server-assigned immutable output, without database credentials."""
    url = str(target.get('url', ''))
    reference = str(target.get('reference', ''))
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not re.fullmatch(r'[a-f0-9]{32}\.r2\.cloudflarestorage\.com', parsed.hostname or ''):
        raise RuntimeError('invalid_output_target')
    if not re.fullmatch(r'r2:[a-f0-9-]{36}/[A-Za-z0-9._-]+', reference):
        raise RuntimeError('invalid_output_reference')
    size = Path(local_path).stat().st_size
    if not 0 < size <= 134217728:
        raise RuntimeError('output_size_limit')
    try:
        with Path(local_path).open('rb') as data:
            with session.put(url, data=data, headers={'Content-Type': 'application/octet-stream',
                             'Content-Length': str(size), 'If-None-Match': '*'},
                             allow_redirects=False, timeout=(10, 180)) as response:
                if response.status_code != 412 and not 200 <= response.status_code < 300:
                    raise RuntimeError('output_http_' + str(response.status_code))
    except Exception:
        raise RuntimeError('private_output_transfer_failed') from None
    return reference


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

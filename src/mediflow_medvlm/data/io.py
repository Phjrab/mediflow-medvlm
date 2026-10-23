"""Small bounded acquisition helpers. Original files are never edited."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ALLOWED_HOSTS = {'github.com', 'raw.githubusercontent.com', 'huggingface.co', 'cdn-lfs.huggingface.co', 'cdn-lfs-us-1.huggingface.co', 'cas-bridge.xethub.hf.co'}
GIT_SOURCES = {
    'medquad': 'https://github.com/abachaa/MedQuAD.git',
    'pubmedqa_labeled': 'https://github.com/pubmedqa/pubmedqa.git',
}
DATA_BYTES_CAP = 25 * 1024**3
GIT_SNAPSHOT_BYTES_CAP = 1024**3


def _tree_bytes(path):
    total=0
    for item in Path(path).rglob('*'):
        try:
            if item.is_file() and not item.is_symlink():total+=item.stat().st_size
        except FileNotFoundError:
            # Another bounded download worker may atomically rename its .part.
            continue
    return total


def _data_root(path):
    """Infer the existing raw/images storage contract without guessing HOME."""
    path = Path(path).resolve()
    for parent in (path, *path.parents):
        if parent.name in {'raw', 'images'}:
            return parent.parent
    configured = os.environ.get('MEDVLM_DATA_ROOT')
    if configured and path.is_relative_to(Path(configured).resolve()):
        return Path(configured).resolve()
    return None


class _OfficialRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.password:
            raise ValueError('Redirect outside official distribution allowlist')
        # Official object storage redirects contain ephemeral signatures. They
        # may be followed on approved hosts, but never written to a manifest.
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.jsonl.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
    os.replace(tmp, path)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def disk_guard(root: Path, expected_bytes=0):
    root = Path(root)
    if expected_bytes < 0:
        raise ValueError('expected_bytes cannot be negative')
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    if usage.free - expected_bytes < max(20 * 1024**3, usage.total * .15):
        raise RuntimeError('SKIPPED_BUDGET: insufficient disk reserve')
    data_root = _data_root(root)
    if data_root is not None:
        # Archives and extracted copies both count; models/checkpoints do not.
        used = _tree_bytes(data_root / 'raw') + _tree_bytes(data_root / 'images')
        if used + expected_bytes > DATA_BYTES_CAP:
            raise RuntimeError('SKIPPED_BUDGET: raw plus extracted data exceeds 25 GiB')


def download(url: str, target: Path, *, max_bytes=32 * 1024**2, expected_sha256=None):
    """Sequential, capped, resumable HTTP transfer, atomic installation, 3 retries."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != 'https' or parts.hostname not in ALLOWED_HOSTS or parts.username or parts.password:
        raise ValueError('URL is outside official source allowlist')
    if parts.query:
        raise ValueError('Store and use credential-free official URLs only')
    source_prefixes = {
        'github.com': ('/abachaa/MedQuAD/', '/pubmedqa/pubmedqa/'),
        'raw.githubusercontent.com': ('/abachaa/MedQuAD/', '/pubmedqa/pubmedqa/'),
        'huggingface.co': ('/api/datasets/sean0042/KorMedMCQA', '/datasets/sean0042/KorMedMCQA/resolve/',
                           '/api/datasets/BoKelvin/SLAKE', '/datasets/BoKelvin/SLAKE/resolve/'),
    }
    if not any(parts.path == prefix or parts.path.startswith(prefix if prefix.endswith('/') else prefix + '/') for prefix in source_prefixes.get(parts.hostname, ())):
        raise ValueError('URL dataset is outside official source allowlist')
    target = Path(target)
    if max_bytes <= 0:
        raise ValueError('max_bytes must be positive')
    if target.exists():
        if expected_sha256 and sha256(target) != expected_sha256:
            raise ValueError('Existing immutable raw file checksum mismatch')
        return target
    part = target.with_suffix(target.suffix + '.part')
    offset = part.stat().st_size if part.exists() else 0
    if offset > max_bytes:
        raise ValueError('Retained partial file exceeds acquisition budget')
    disk_guard(target.parent, max_bytes - offset)
    opener = urllib.request.build_opener(_OfficialRedirectHandler())
    for attempt in range(3):
        offset = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url, headers={'User-Agent': 'mediflow-research/1', **({'Range': f'bytes={offset}-'} if offset else {})})
        try:
            with opener.open(request, timeout=45) as response:
                append = offset and response.status == 206
                total = None
                if response.status == 206:
                    content_range = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', response.headers.get('Content-Range', ''))
                    if not content_range or int(content_range[1]) != offset or int(content_range[2]) < offset:
                        raise ValueError('Invalid resume Content-Range')
                    total = None if content_range[3] == '*' else int(content_range[3])
                    if total is not None and total > max_bytes:
                        raise ValueError('Resumed file exceeds acquisition budget')
                if not append:
                    offset = 0
                length = response.headers.get('Content-Length')
                if length and int(length) + offset > max_bytes:
                    raise ValueError('Declared file exceeds acquisition budget')
                with part.open('ab' if append else 'wb') as f:
                    count = offset
                    while block := response.read(1024 * 1024):
                        count += len(block)
                        if count > max_bytes:
                            raise ValueError('Downloaded file exceeds acquisition budget')
                        f.write(block)
                if length and count - offset != int(length):
                    raise OSError('Incomplete download Content-Length')
                if total is not None and count != total:
                    raise OSError('Incomplete resumed download')
                if response.status == 206 and count != int(content_range[2]) + 1:
                    raise ValueError('Resume byte count differs from Content-Range')
            if expected_sha256 and sha256(part) != expected_sha256:
                raise ValueError('Downloaded file checksum mismatch')
            os.replace(part, target)
            return target
        except urllib.error.HTTPError as e:
            # A connection may have ended after all bytes reached .part but
            # before atomic installation. A verified 416 total permits resume.
            if e.code == 416 and part.exists():
                complete = re.fullmatch(r'bytes \*/(\d+)', e.headers.get('Content-Range', ''))
                if complete and 0 < int(complete[1]) == part.stat().st_size <= max_bytes:
                    if expected_sha256 and sha256(part) != expected_sha256:
                        raise ValueError('Completed partial file checksum mismatch') from e
                    os.replace(part,target)
                    return target
            if e.code in {401, 403, 404, 416} or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError, OSError, http.client.IncompleteRead):
            if attempt == 2:
                raise
        time.sleep(2**attempt)
    raise RuntimeError('Transfer did not complete')


def git_snapshot(source, data_root):
    if source not in GIT_SOURCES:
        raise ValueError('Git source not allowlisted')
    destination = Path(data_root) / 'raw' / source / 'repository'
    if not destination.exists():
        disk_guard(destination.parent, GIT_SNAPSHOT_BYTES_CAP)
        tmp = destination.with_name('repository.part')
        if tmp.exists():
            raise RuntimeError('Interrupted clone retained; inspect repository.part before retrying')
        process = subprocess.Popen(['git', 'clone', '--depth', '1', GIT_SOURCES[source], str(tmp)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        began = time.monotonic()
        try:
            while process.poll() is None:
                if time.monotonic() - began > 600:
                    raise TimeoutError('Git snapshot exceeded 600 second timeout')
                if tmp.exists():
                    disk_guard(tmp)
                    if _tree_bytes(tmp) > GIT_SNAPSHOT_BYTES_CAP:
                        raise RuntimeError('SKIPPED_BUDGET: Git snapshot exceeds 1 GiB source cap')
                time.sleep(.5)
            stdout, stderr = process.communicate()
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, 'git clone', output=stdout, stderr=stderr)
            disk_guard(tmp)
            if _tree_bytes(tmp) > GIT_SNAPSHOT_BYTES_CAP:
                raise RuntimeError('SKIPPED_BUDGET: Git snapshot exceeds 1 GiB source cap')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
        os.replace(tmp, destination)
    origin = subprocess.check_output(['git', '-C', str(destination), 'remote', 'get-url', 'origin'], text=True).strip()
    if origin.removesuffix('.git') != GIT_SOURCES[source].removesuffix('.git'):
        raise ValueError('Existing raw snapshot origin differs from allowlisted source')
    revision = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Raw git snapshot changed; refusing to bless modified content')
    files = sorted(p for p in destination.rglob('*') if p.is_file() and '.git' not in p.relative_to(destination).parts)
    return destination, revision, [{'path': str(p.relative_to(destination)), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files]

"""Bounded acquisition contracts, without making network requests."""
import hashlib
from collections import namedtuple
from types import SimpleNamespace
import urllib.request

import pytest

from mediflow_medvlm.data import io

URL='https://huggingface.co/datasets/sean0042/KorMedMCQA/resolve/'+'a'*40+'/README.md'


class Response:
    def __init__(self,body,status=200,headers=None):
        self.body=body;self.status=status;self.headers=headers or {}
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self,_):body,self.body=self.body,b'';return body


def install_response(monkeypatch,factory):
    calls=[]
    def open_(request,timeout):
        calls.append((request,timeout));return factory()
    monkeypatch.setattr(io.urllib.request,'build_opener',lambda *_:SimpleNamespace(open=open_))
    monkeypatch.setattr(io.time,'sleep',lambda *_:None)
    monkeypatch.setattr(io,'disk_guard',lambda *_:None)
    return calls


def test_successful_range_resume_installed_atomically_with_checksum(tmp_path,monkeypatch):
    target=tmp_path/'file';part=tmp_path/'file.part';part.write_bytes(b'ab')
    calls=install_response(monkeypatch,lambda:Response(b'cd',206,{'Content-Length':'2','Content-Range':'bytes 2-3/4'}))
    io.download(URL,target,max_bytes=4,expected_sha256=hashlib.sha256(b'abcd').hexdigest())
    assert target.read_bytes()==b'abcd' and not part.exists()
    assert calls[0][0].get_header('Range')=='bytes=2-' and calls[0][1]==45


def test_invalid_resume_offset_never_commits(tmp_path,monkeypatch):
    target=tmp_path/'file';(tmp_path/'file.part').write_bytes(b'ab')
    install_response(monkeypatch,lambda:Response(b'cd',206,{'Content-Length':'2','Content-Range':'bytes 0-1/4'}))
    with pytest.raises(ValueError,match='Content-Range'):io.download(URL,target,max_bytes=4)
    assert not target.exists()


def test_incomplete_transfer_retries_three_times_and_preserves_partial(tmp_path,monkeypatch):
    target=tmp_path/'file'
    calls=install_response(monkeypatch,lambda:Response(b'ab',200,{'Content-Length':'4'}))
    with pytest.raises(OSError,match='Incomplete'):io.download(URL,target,max_bytes=4)
    assert len(calls)==3 and not target.exists()
    assert (tmp_path/'file.part').read_bytes()==b'ab'


def test_checksum_failure_never_replaces_original_or_installs_partial(tmp_path,monkeypatch):
    target=tmp_path/'file';target.write_bytes(b'original')
    with pytest.raises(ValueError,match='Existing immutable'):io.download(URL,target,expected_sha256='bad')
    assert target.read_bytes()==b'original'
    absent=tmp_path/'absent';install_response(monkeypatch,lambda:Response(b'bad',headers={'Content-Length':'3'}))
    with pytest.raises(ValueError,match='checksum mismatch'):io.download(URL,absent,max_bytes=3,expected_sha256='bad')
    assert not absent.exists() and (tmp_path/'absent.part').exists()


def test_unapproved_dataset_and_redirect_rejected(tmp_path):
    with pytest.raises(ValueError,match='source allowlist'):io.download('https://huggingface.co/datasets/other/medical/resolve/main/data',tmp_path/'file')
    request=urllib.request.Request(URL)
    with pytest.raises(ValueError,match='Redirect outside'):io._OfficialRedirectHandler().redirect_request(request,None,302,'redirect',{},'https://example.org/file')


@pytest.mark.parametrize('total,free,expected',[(100,21,2),(200,31,2)])
def test_reserve_requires_both_twenty_gib_and_fifteen_percent(tmp_path,monkeypatch,total,free,expected):
    gib=1024**3;usage=namedtuple('Usage','total used free')
    monkeypatch.setattr(io.shutil,'disk_usage',lambda _:usage(total*gib,(total-free)*gib,free*gib))
    with pytest.raises(RuntimeError,match='disk reserve'):io.disk_guard(tmp_path,expected*gib)


def test_combined_archive_and_extracted_cap(tmp_path,monkeypatch):
    (tmp_path/'raw/source').mkdir(parents=True);(tmp_path/'images/source').mkdir(parents=True)
    (tmp_path/'raw/source/archive').write_bytes(b'x'*75);(tmp_path/'images/source/image').write_bytes(b'x'*20)
    usage=namedtuple('Usage','total used free');monkeypatch.setattr(io.shutil,'disk_usage',lambda _:usage(100*1024**3,0,100*1024**3))
    monkeypatch.setattr(io,'DATA_BYTES_CAP',100)
    io.disk_guard(tmp_path/'images/source',5)
    with pytest.raises(RuntimeError,match='25 GiB'):io.disk_guard(tmp_path/'images/source',6)


def test_slake_source_can_use_shared_official_downloader(tmp_path,monkeypatch):
    install_response(monkeypatch,lambda:Response(b'ok',headers={'Content-Length':'2'}))
    target=tmp_path/'slake.json'
    io.download('https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/'+'a'*40+'/train.json',target,max_bytes=2)
    assert target.read_bytes()==b'ok'


def test_fully_received_partial_file_recovers_from_verified_range_416(tmp_path,monkeypatch):
    import urllib.error
    target=tmp_path/'file';(tmp_path/'file.part').write_bytes(b'abcd')
    def completed():raise urllib.error.HTTPError(URL,416,'range complete',{'Content-Range':'bytes */4'},None)
    calls=install_response(monkeypatch,completed)
    io.download(URL,target,max_bytes=4,expected_sha256=hashlib.sha256(b'abcd').hexdigest())
    assert target.read_bytes()==b'abcd' and len(calls)==1

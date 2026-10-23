import hashlib
import pytest

from mediflow_medvlm.data import vqarad


def test_vqarad_canonical_image_hash_is_consumable_by_renderer(tmp_path):
    import json
    from PIL import Image
    from mediflow_medvlm.training.rendering import image_paths
    raw = tmp_path / 'raw/vqa_rad'
    images = tmp_path / 'images/vqa_rad'
    manifests = tmp_path / 'manifests/sources'
    for directory in (raw, images, manifests):directory.mkdir(parents=True)
    asset = images / 'fixture.jpg'
    Image.new('RGB', (16, 16), (20, 60, 90)).save(asset)
    qa = raw / 'VQA_RAD Dataset Public.json'
    qa.write_text(json.dumps([{'qid': 1, 'phrase_type': 'test_freeform',
        'image_name': asset.name, 'question': 'Engineering image fixture?',
        'answer': 'yes', 'image_case_url': 'https://example.invalid/fixture',
        'qid_linked_id': 1}]))
    qa_hash = hashlib.sha256(qa.read_bytes()).hexdigest()
    (manifests / 'vqa_rad.json').write_text(json.dumps({
        'resolved_revision': 'fixture-vqarad-revision', 'retrieved_at': '2026-10-23T00:00:00Z',
        'files': [{'sha256': qa_hash}]}))
    vqarad.run(tmp_path)
    row = json.loads((tmp_path / 'processed/vqa_rad/test.jsonl').read_text())
    paths, metadata = image_paths(row)
    assert paths == [asset]
    assert metadata[0]['sha256'] == hashlib.sha256(asset.read_bytes()).hexdigest()
    assert row['provenance']['local_asset_sha256'] == qa_hash
    assert metadata[0]['sha256'] != qa_hash
    asset.write_bytes(asset.read_bytes() + b'modified')
    with pytest.raises(ValueError, match='image hash changed'):
        image_paths(row)


def entry(name="image.jpg"):
    return {"id": "fixture-file", "attributes": {"name": name, "size": 3, "current_version": 1,
            "extra": {"hashes": {"sha256": hashlib.sha256(b"img").hexdigest()}}},
            "links": {"download": "https://osf.io/download/fixture/"}}


def test_vqarad_asset_checks_reserve_before_network(tmp_path, monkeypatch):
    def denied(*args):
        raise RuntimeError("SKIPPED_BUDGET: test reserve")
    def unexpected(*args, **kwargs):
        pytest.fail("network must not begin")
    monkeypatch.setattr(vqarad, "disk_guard", denied)
    monkeypatch.setattr(vqarad.urllib.request, "urlopen", unexpected)
    with pytest.raises(RuntimeError, match="SKIPPED_BUDGET"):
        vqarad.fetch_asset(entry(), tmp_path / "raw")
    assert not (tmp_path / "raw/image.jpg").exists()


def test_vqarad_asset_rejects_partial_symlink(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.write_bytes(b"retain")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "image.jpg.part").symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe destination"):
        vqarad.fetch_asset(entry(), raw)
    assert outside.read_bytes() == b"retain"


def test_vqarad_batch_reserved_before_individual_downloads(tmp_path, monkeypatch):
    question = entry("VQA_RAD Dataset Public.json")
    folder = {"attributes": {"name": "images", "kind": "folder"}, "relationships": {"files": {"links": {"related": {"href": "https://api.osf.io/v2/images/"}}}}}
    question["attributes"]["kind"] = "file"
    responses = iter([{}, {"data": {"attributes": {"name": "CC0 1.0 Universal"}}},
                      {"data": [question, folder]}, {"data": [entry()], "links": {"next": None}}])
    monkeypatch.setattr(vqarad, "get_json", lambda url: next(responses))
    calls = []
    def denied(root, expected_bytes):
        calls.append((root, expected_bytes))
        raise RuntimeError("SKIPPED_BUDGET: batch")
    def unexpected(*args, **kwargs):
        pytest.fail("asset transfer must not begin")
    monkeypatch.setattr(vqarad, "disk_guard", denied)
    monkeypatch.setattr(vqarad, "fetch_asset", unexpected)
    with pytest.raises(RuntimeError, match="SKIPPED_BUDGET"):
        vqarad.run(tmp_path, fetch=True)
    assert len(calls) == 1
    assert calls[0][1] > 6  # QA + image bytes plus preserved source metadata.


@pytest.mark.parametrize('url',[
    'http://api.osf.io/v2/nodes/89kps/',
    'https://example.org/v2/nodes/89kps/',
    'https://api.osf.io/v2/nodes/other/',
    'https://user:secret@api.osf.io/v2/nodes/89kps/',
    'https://api.osf.io:8443/v2/nodes/89kps/',
])
def test_vqarad_metadata_requires_exact_https_source(url,monkeypatch):
    monkeypatch.setattr(vqarad,'_osf_open',lambda *a,**k:pytest.fail('network must not begin'))
    with pytest.raises(ValueError,match='HTTPS source'):vqarad.get_json(url)


def test_vqarad_redirect_rejects_other_host_node_bucket_and_downgrade():
    import urllib.request
    request=urllib.request.Request('https://osf.io/download/fixture/')
    handler=vqarad._OSFRedirectHandler()
    blocked=['https://example.org/file','http://files.osf.io/v1/resources/89kps/providers/osfstorage/file',
             'https://files.osf.io/v1/resources/other/providers/osfstorage/file',
             'https://'+vqarad.OSF_OBJECT_HOST+'/unapproved-bucket/file']
    for url in blocked:
        with pytest.raises(ValueError,match='redirect outside'):handler.redirect_request(request,None,302,'redirect',{},url)
    allowed='https://'+vqarad.OSF_OBJECT_HOST+vqarad.OSF_OBJECT_PREFIX+'object?temporary-signature=not-recorded'
    assert handler.redirect_request(request,None,302,'redirect',{},allowed).full_url==allowed
    with pytest.raises(ValueError,match='redirect outside'):
        vqarad._OSFRedirectHandler(metadata_only=True).redirect_request(request,None,302,'redirect',{},allowed)


class MetadataResponse:
    def __init__(self,body):self.body=body
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self,limit):return self.body[:limit]


def test_vqarad_metadata_has_bounded_bytes_and_transient_attempts(monkeypatch):
    calls=[]
    def fail(*args,**kwargs):calls.append(1);raise OSError('transient transport')
    monkeypatch.setattr(vqarad,'_osf_open',fail);monkeypatch.setattr(vqarad.time,'sleep',lambda *_:None)
    with pytest.raises(OSError):vqarad.get_json(vqarad.NODE)
    assert len(calls)==3
    monkeypatch.setattr(vqarad,'MAX_METADATA_BYTES',8)
    monkeypatch.setattr(vqarad,'_osf_open',lambda *a,**k:MetadataResponse(b'123456789'))
    with pytest.raises(ValueError,match='bounded response'):vqarad.get_json(vqarad.NODE)


def test_vqarad_access_denial_does_not_retry_metadata(monkeypatch):
    import urllib.error
    calls=[]
    def denied(*args,**kwargs):
        calls.append(1);raise urllib.error.HTTPError(vqarad.NODE,403,'denied',{},None)
    monkeypatch.setattr(vqarad,'_osf_open',denied)
    with pytest.raises(urllib.error.HTTPError):vqarad.get_json(vqarad.NODE)
    assert len(calls)==1


def test_vqarad_rejects_pagination_cycle_before_asset_transfer(tmp_path,monkeypatch):
    question=entry('VQA_RAD Dataset Public.json');question['attributes']['kind']='file'
    prefix='https://api.osf.io/v2/nodes/89kps/files/osfstorage/folder/'
    page=prefix+'?page[size]=100'
    folder={'attributes':{'name':'images','kind':'folder'},'relationships':{'files':{'links':{'related':{'href':prefix}}}}}
    responses=iter([{}, {'data':{'attributes':{'name':'CC0 1.0 Universal'}}}, {'data':[question,folder]}, {'data':[entry()],'links':{'next':page}}])
    monkeypatch.setattr(vqarad,'get_json',lambda _:next(responses))
    monkeypatch.setattr(vqarad,'fetch_asset',lambda *a,**k:pytest.fail('transfer must not begin'))
    with pytest.raises(ValueError,match='pagination'):vqarad.run(tmp_path,fetch=True)

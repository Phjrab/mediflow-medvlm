"""Private, immutable AI-Hub imports and conservative Korean QA normalization.

Delivery archives remain untouched. No archive path, document text, or parser
exception is printed; callers publish aggregate counts only. Source rights and
the 71762 pairing contract must be supplied from actual evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import unicodedata
from urllib.parse import urlsplit, urlunsplit
import zipfile

from .io import atomic_json, digest, disk_guard, now, sha256, write_jsonl, read_jsonl
from .schema import normalized, sample, validate_record, input_digest

DATASETS = {"71875", "71762"}
VERSION = "medvlm.aihub.v2.1"
EXCLUDED_PATH = re.compile(r"(?:^|[/_\-. ])(?:models?|weights?|checkpoints?|evaluation|eval|kormedmcqa|소스코드|모델)(?:$|[/_\- .])", re.I)
JSON_LIMIT = 64 * 1024**2


def _private(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('UNSAFE_IMPORT_SYMLINK')
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def _safe_member(name, *, streaming_root_marker=False):
    # Supplied 71875 archives use a single leading slash on member names.
    # In the streaming reader this is a logical ZIP-root marker, never a local
    # destination. Nested archives are installed only under their SHA basename.
    if streaming_root_marker and name.startswith('/') and not name.startswith('//'):
        name = name[1:]
    path = PurePosixPath(name)
    if not name or "\\" in name or "\x00" in name or path.is_absolute() or any(p in {"", ".", ".."} for p in name.rstrip("/").split("/")) or re.match(r"^[A-Za-z]:", name):
        raise ValueError("UNSAFE_ARCHIVE_PATH")
    return path


def _safe_json(raw):
    """Strip file BOM separately from leading key BOM, rejecting all collisions."""
    events = Counter()
    if isinstance(raw, bytes):
        events["file_bom"] = int(raw.startswith(b"\xef\xbb\xbf"))
        raw = raw.decode("utf-8-sig")
    elif raw.startswith("\ufeff"):
        raw = raw[1:]; events["file_bom"] += 1
    def pairs(items):
        result = {}
        for key, value in items:
            clean = key.lstrip("\ufeff")
            events["key_bom"] += int(clean != key)
            if clean in result:
                raise ValueError("JSON_KEY_NORMALIZATION_COLLISION")
            result[clean] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs), dict(events)


def credential_free_url(value):
    p = urlsplit(value)
    if p.scheme not in {"http", "https"} or not p.hostname:
        raise ValueError("INVALID_DELIVERY_URL")
    return urlunsplit((p.scheme, p.hostname, p.path, "", ""))


def merge_parts(parts, output_dir):
    """Reassemble explicit byte-split ZIP parts, requiring contiguous numbering.

    Standard multi-disk .z01 archives require a dedicated tool and are rejected
    rather than guessed. Complete ZIPs are never concatenated with more bytes.
    """
    parts = [Path(p) for p in parts]
    if len(parts) == 1 and parts[0].suffix.lower() == ".zip":
        return parts[0], None
    parsed = []
    for path in parts:
        match = re.fullmatch(r"(.+\.zip)(?:\.part|\.)(\d+|[a-z]{2})", path.name, re.I)
        if not match:
            raise ValueError("UNSUPPORTED_OR_MIXED_ARCHIVE_PART_NAMES")
        suffix = match[2].lower()
        ordinal = int(suffix) if suffix.isdigit() else (ord(suffix[0])-97)*26+ord(suffix[1])-97
        parsed.append((match[1], suffix.isdigit(), ordinal, path))
    if len({(x[0], x[1]) for x in parsed}) != 1:
        raise ValueError("MIXED_ARCHIVE_PART_FAMILIES")
    parsed.sort(key=lambda x: x[2]); numbers = [x[2] for x in parsed]
    if numbers[0] not in {0, 1} or numbers != list(range(numbers[0], numbers[-1]+1)):
        raise ValueError("MISSING_OR_DUPLICATE_ARCHIVE_PART")
    for _, _, _, p in parsed:
        # A valid standalone archive among parts indicates a wrong input set.
        if zipfile.is_zipfile(p):
            try:
                with zipfile.ZipFile(p) as z:
                    if z.testzip() is None:
                        raise ValueError("COMPLETE_ZIP_MUST_NOT_BE_MERGED")
            except (zipfile.BadZipFile, OSError, ValueError) as exc:
                if str(exc) == "COMPLETE_ZIP_MUST_NOT_BE_MERGED":
                    raise
    receipts = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p), "ordinal": n} for _, _, n, p in parsed]
    out = _private(output_dir) / (digest(receipts)+".zip")
    if not out.exists():
        disk_guard(out.parent, sum(x["bytes"] for x in receipts))
        tmp = out.with_suffix(".part")
        if tmp.is_symlink() or out.is_symlink(): raise ValueError("UNSAFE_IMPORT_SYMLINK")
        with tmp.open("wb") as target:
            for _, _, _, source in parsed:
                with source.open("rb") as stream: shutil.copyfileobj(stream, target)
        try:
            with zipfile.ZipFile(tmp) as z:
                if z.testzip() is not None: raise ValueError("ARCHIVE_CRC_FAILURE")
            os.replace(tmp, out)
        except Exception:
            # Retain originals and incomplete concatenation for diagnosis.
            raise ValueError("INCOMPLETE_OR_CORRUPT_SPLIT_ARCHIVE") from None
    return out, {"parts": receipts, "merged_sha256": sha256(out), "original_parts_preserved": True}


def import_aihub(inputs, data_root, dataset_id, *, version, delivery=None):
    """Verify all archive CRCs; unpack only nested archives, not the whole corpus."""
    if str(dataset_id) not in DATASETS or not version:
        raise ValueError("DATASET_ID_AND_VERSION_REQUIRED")
    root = _private(data_root); raw_root = _private(root/"raw"/f"aihub_{dataset_id}")
    paths = [Path(x).resolve() for x in inputs]
    if not paths or any(not p.exists() for p in paths): raise ValueError("INPUT_NOT_PROVIDED")
    if any(Path(x).is_symlink() for x in inputs): raise ValueError("UNSAFE_IMPORT_SYMLINK")
    original = []
    for p in paths:
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in files:
            if f.is_symlink(): raise ValueError("UNSAFE_IMPORT_SYMLINK")
            if f.is_file(): original.append({"path": str(f), "relative_path": str(f.relative_to(p)) if p.is_dir() else f.name, "bytes": f.stat().st_size, "sha256": sha256(f)})
    identity = {"dataset_id": str(dataset_id), "version": version, "files": original}
    import_id = digest(identity); manifest_path = root/"manifests"/"imports"/(import_id+".json")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for item in previous["members"]:
            if item.get("archive") and not Path(item["archive"]).is_file(): raise ValueError("IMPORTED_ARCHIVE_MISSING")
        return previous
    members, archives, merged = [], [], []
    # An explicitly supplied set of split fragments is one archive family.
    if all(p.is_file() for p in paths) and any(re.search(r"\.zip(?:\.part|\.)(?:\d+|[a-z]{2})$", p.name, re.I) for p in paths):
        archive, receipt = merge_parts(paths, raw_root/"merged")
        paths = [archive]; merged.append(receipt)
    def archive_members(path, logical, depth=0):
        if depth > 4: raise ValueError("NESTED_ARCHIVE_DEPTH_LIMIT")
        with zipfile.ZipFile(path) as z:
            seen = set()
            infos = z.infolist()
            expanded = sum(i.file_size for i in infos)
            # Even streaming reads reject unreasonable expansion beyond raw cap.
            if expanded > 25*1024**3: raise ValueError("ARCHIVE_EXPANSION_EXCEEDS_DATA_CAP")
            archives.append({"path": str(path), "logical_path": logical, "sha256": sha256(path), "bytes": path.stat().st_size, "expanded_bytes": expanded, "entries": len(infos)})
            for info in infos:
                safe_name = str(_safe_member(info.filename, streaming_root_marker=True))
                if safe_name in seen: raise ValueError("DUPLICATE_ARCHIVE_MEMBER")
                seen.add(safe_name)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode): raise ValueError("UNSAFE_ARCHIVE_SYMLINK")
                if info.flag_bits & 1: raise ValueError("ENCRYPTED_ARCHIVE_UNSUPPORTED")
                if info.is_dir(): continue
                h = hashlib.sha256(); size = 0
                with z.open(info) as stream:
                    while block := stream.read(1024**2): h.update(block); size += len(block)
                if size != info.file_size: raise ValueError("ARCHIVE_MEMBER_SIZE_MISMATCH")
                item = {"archive": str(path), "member": info.filename, "relative_path": logical+"!/"+safe_name, "streaming_root_marker_removed": info.filename.startswith('/'), "bytes": size, "sha256": h.hexdigest(), "original_split": infer_split(logical+"/"+safe_name), "excluded_path": bool(EXCLUDED_PATH.search(logical+"/"+safe_name))}
                members.append(item)
                if info.filename.lower().endswith(".zip") and not item["excluded_path"]:
                    nested = raw_root/"nested"/(item["sha256"]+".zip")
                    if not nested.exists():
                        disk_guard(nested.parent, info.file_size)
                        temporary = nested.with_suffix(".part")
                        if nested.is_symlink() or temporary.is_symlink(): raise ValueError("UNSAFE_IMPORT_SYMLINK")
                        with z.open(info) as source, temporary.open("wb") as target: shutil.copyfileobj(source, target)
                        if sha256(temporary) != item["sha256"]: raise ValueError("NESTED_ARCHIVE_HASH_MISMATCH")
                        os.replace(temporary, nested)
                    elif sha256(nested) != item["sha256"]: raise ValueError("NESTED_ARCHIVE_HASH_MISMATCH")
                    archive_members(nested, item["relative_path"], depth+1)
    for path in paths:
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        consumed=set()
        for f in files:
            if f in consumed:continue
            part_match=re.fullmatch(r'(.+\.zip)(?:\.part|\.)(\d+|[a-z]{2})',f.name,re.I)
            if path.is_dir() and part_match:
                family=[p for p in files if p.parent==f.parent and re.fullmatch(re.escape(part_match[1])+r'(?:\.part|\.)(\d+|[a-z]{2})',p.name,re.I)]
                consumed.update(family)
                complete=f.parent/part_match[1]
                if complete.is_file():
                    # Its normal .zip visit validates it; never concatenate an
                    # existing full ZIP with stale split fragments.
                    continue
                archive,receipt=merge_parts(family,raw_root/'merged');merged.append(receipt)
                archive_members(archive,str(complete.relative_to(path)))
                continue
            if f.suffix.lower() == ".zip": archive_members(f, str(f.relative_to(path)) if path.is_dir() else f.name)
            else:
                rel = str(f.relative_to(path)) if path.is_dir() else f.name
                members.append({"path": str(f), "relative_path": rel, "bytes": f.stat().st_size, "sha256": sha256(f), "original_split": infer_split(rel), "excluded_path": bool(EXCLUDED_PATH.search(rel))})
    delivery = dict(delivery or {"method": "user_provided_local"})
    for key in list(delivery):
        if "url" in key.lower(): delivery[key] = credential_free_url(delivery[key])
        if any(x in key.lower() for x in ("token", "secret", "password", "api_key")): raise ValueError("SECRET_IN_DELIVERY_METADATA")
    result = {**identity, "schema_version": VERSION, "import_id": import_id, "official_source": f"https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn={dataset_id}", "delivery": delivery, "received_at": now(), "status": "SUCCEEDED", "evidence_level": "ACTUAL_SOURCE_VERIFIED", "archive_integrity": "all_read_members_CRC_verified", "archives": archives, "members": members, "split_archives": merged, "originals_preserved": True, "raw_redistribution": False, "manifest_path": str(manifest_path)}
    atomic_json(manifest_path, result)
    return result


def infer_split(path):
    parts = re.split(r"[/!\\]", path.casefold())
    found = set()
    for part in parts:
        if re.search(r"(?:^|[_\- ])(?:training|train|tl|ts)(?:$|[_\- .])", part): found.add("train")
        if re.search(r"(?:^|[_\- ])(?:validation|valid|vl|vs)(?:$|[_\- .])", part): found.add("validation")
        if re.search(r"(?:^|[_\- ])test(?:$|[_\- .])", part): found.add("test")
    if len(found) > 1: raise ValueError("AMBIGUOUS_SOURCE_SPLIT")
    return next(iter(found), None)


def json_documents(manifest):
    """Yield source JSON and private asset receipt; never infer QA from corpus."""
    handles = {}
    try:
        for item in manifest["members"]:
            if item["excluded_path"] or not item["relative_path"].lower().endswith(".json"): continue
            if item["bytes"] > JSON_LIMIT: raise ValueError("JSON_DOCUMENT_SIZE_LIMIT")
            if "archive" in item:
                archive = item["archive"]
                if archive not in handles: handles[archive] = zipfile.ZipFile(archive)
                raw = handles[archive].read(item["member"])
            else: raw = Path(item["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["sha256"]: raise ValueError("IMPORTED_SOURCE_CHANGED")
            try:
                value, events = _safe_json(raw)
                yield value, {**item, "normalization_events": events}
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                yield None, {**item, "parse_error": "JSON_KEY_NORMALIZATION_COLLISION" if str(exc)=="JSON_KEY_NORMALIZATION_COLLISION" else "INVALID_JSON_OR_ENCODING"}
    finally:
        for stream in handles.values(): stream.close()


def parse_71875_mcq(question, answer):
    """Only complete line-anchored, consecutive option blocks are accepted."""
    pattern = re.compile(r"(?m)^[ \t]*(?:([1-5])[.)]|([①②③④⑤]))[ \t]*(\S.*)$")
    matches = list(pattern.finditer(question))
    if not 2 <= len(matches) <= 5: raise ValueError("MCQ_OPTION_BLOCK_AMBIGUOUS")
    numbers = [int(m[1]) if m[1] else "①②③④⑤".index(m[2])+1 for m in matches]
    if numbers != list(range(1,len(matches)+1)): raise ValueError("MCQ_NONCONSECUTIVE_OPTIONS")
    stem = question[:matches[0].start()].strip()
    if not stem: raise ValueError("MCQ_EMPTY_STEM")
    values = [question[m.start(3):matches[i+1].start() if i+1<len(matches) else len(question)].strip() for i,m in enumerate(matches)]
    if any(not v for v in values) or len(set(map(normalized,values)))!=len(values): raise ValueError("MCQ_EMPTY_OR_DUPLICATE_OPTION")
    answer_match = re.fullmatch(r"\s*(?:(?:정답|답)\s*[:：]?\s*)?(?:([1-5])(?:[.)]|번)?|([①②③④⑤]))\s*(?:[:：\-–]\s*)?(.*?)\s*", answer, re.S)
    if not answer_match: raise ValueError("MCQ_ANSWER_NUMBER_REQUIRED")
    number = int(answer_match[1]) if answer_match[1] else "①②③④⑤".index(answer_match[2])+1
    if not 1 <= number <= len(values): raise ValueError("MCQ_INVALID_ANSWER_NUMBER")
    text = answer_match[3].strip()
    if text and normalized(text) != normalized(values[number-1]): raise ValueError("MCQ_ANSWER_TEXT_MISMATCH")
    options = {chr(65+i): value for i,value in enumerate(values)}
    return stem, options, chr(64+number)


def records_71875(documents, *, revision, license_ref, fetched_at):
    if not revision or not license_ref: raise ValueError("SOURCE_RIGHTS_EVIDENCE_REQUIRED")
    records, excluded, counts, seen_ids = [], [], Counter(), Counter()
    for document, asset in documents:
        if document is None:
            excluded.append({"asset_sha256":asset["sha256"],"reason":asset["parse_error"]});continue
        # Containers are deliberately not guessed: original corpus != QA labels.
        values = document if isinstance(document,list) else [document]
        for index, raw in enumerate(values):
            if not isinstance(raw,dict) or not any(k in raw for k in ("qa_id","q_type")):
                counts["non_qa_documents"]+=1;continue
            counts["raw_qa"]+=1
            counts['raw_official_split_'+str(asset.get('original_split'))]+=1
            identity = f'{asset["relative_path"]}#{index}'
            reason = None
            if not all(isinstance(raw.get(k),str) and raw[k].strip() for k in ("question","answer")): reason="MISSING_OR_INVALID_QA_FIELDS"
            if isinstance(raw.get('qa_id'),bool) or not isinstance(raw.get('qa_id'),(str,int)) or not str(raw['qa_id']).strip():reason="MISSING_OR_INVALID_QA_ID"
            qtype = str(raw.get("q_type"))
            if qtype not in {"1","2","3"}: reason="UNKNOWN_Q_TYPE"
            domain=raw.get("domain")
            if domain is not None and not isinstance(domain,(str,int)):reason="INVALID_DOMAIN_TYPE"
            if reason:
                excluded.append({"asset_sha256":asset["sha256"],"row_index":index,"reason":reason});continue
            counts['q_type_'+qtype]+=1;seen_ids[raw['qa_id']]+=1
            options=option_id=None;question=raw['question'];answer=raw['answer']
            if qtype=='1':
                try: question,options,option_id=parse_71875_mcq(question,answer);answer=options[option_id]
                except ValueError as exc:
                    excluded.append({"asset_sha256":asset["sha256"],"row_index":index,"reason":str(exc)});continue
            split=asset.get('original_split')
            row=sample('aihub_71875',revision,identity,'medical_mcq' if qtype=='1' else 'medical_qa',question,answer,split or 'train',language='ko',original_split=split,options=options,option_id=option_id,license_ref=license_ref,fetched_at=fetched_at,raw_sha256=asset['sha256'],source_url='https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71875',annotation='official_source_includes_LLM_generation; per_record_expert_authorship_unknown',metadata={'source_qa_id':raw['qa_id'],'q_type':qtype,'domain':domain,'source_relative_path':asset['relative_path'],'source_includes_llm_generation':True,'per_record_generation_method':'unknown','official_split_available':bool(split)})
            row['provenance']['transformation_version']=VERSION
            row['review_status']='human_review_pending'
            row['grouping_quality']={'status':'proxy','level':'question_content','patient_group_verified':False}
            records.append(row)
    return records, {'counts':dict(counts),'exclusions':excluded,'excluded_counts':dict(Counter(x['reason'] for x in excluded)),'repeated_qa_ids':sum(n>1 for n in seen_ids.values()),'qa_id_global_uniqueness_assumed':False}


def _leak_norm(text):
    # No punctuation, numeric, unit, case, or negation deletion.
    return normalized(text)


def leakage_audit(candidates, protected, *, near_threshold=.90):
    """Protected official dev/test remain unchanged; near matches are candidates."""
    by_ordered=defaultdict(list);by_unordered=defaultdict(list);by_question=defaultdict(list)
    def signature(row):
        question=_leak_norm(row['input']['question']);options=row['input'].get('options') or {}
        values=[_leak_norm(x) for x in (options.values() if isinstance(options,dict) else options)]
        return question,values
    for row in protected:
        q,o=signature(row);by_ordered[digest([q,o])].append(row['sample_id']);by_unordered[digest([q,sorted(o)])].append(row['sample_id']);by_question[digest(q)].append(row['sample_id'])
    # Character 5-grams handle Korean spacing differences without changing source.
    def grams(text):return {text[i:i+5] for i in range(max(0,len(text)-4))}
    index=defaultdict(set);protected_grams={}
    for row in protected:
        q,_=signature(row);g=grams(q);protected_grams[row['sample_id']]=g
        for token in g:index[token].add(row['sample_id'])
    confirmed=[];near=[];remove=set()
    for row in candidates:
        q,o=signature(row);sid=row['sample_id'];reasons=[]
        for kind,key,table in [('exact_question_options',digest([q,o]),by_ordered),('option_order_permutation',digest([q,sorted(o)]),by_unordered),('exact_question_only',digest(q),by_question)]:
            if key in table:reasons.append({'kind':kind,'protected_sample_ids':table[key]})
        if reasons:
            confirmed.append({'sample_id':sid,'matches':reasons,'action':'exclude_candidate_preserve_heldout'});remove.add(sid);continue
        g=grams(q)
        if len(g)<15:continue
        overlaps=Counter(x for token in g for x in index.get(token,()))
        for other,n in overlaps.items():
            score=n/(len(g)+len(protected_grams[other])-n)
            if score>=near_threshold:
                near.append({'sample_id':sid,'protected_sample_id':other,'question_char5_jaccard':round(score,6),'action':'quarantine_candidate_pending_review','clinical_equivalence_verified':False});remove.add(sid)
    return [r for r in candidates if r['sample_id'] not in remove], {'protected_count':len(protected),'protected_splits':dict(Counter(r['split'] for r in protected)),'candidate_count':len(candidates),'confirmed_overlap_rows':len(confirmed),'confirmed_overlaps':confirmed,'near_candidate_rows':len({r['sample_id'] for r in near}),'near_candidates':near,'near_threshold':near_threshold,'near_method':'NFC_whitespace_preserved_punctuation_question_character_5gram_Jaccard','near_candidates_are_confirmed_duplicates':False,'near_candidates_quarantined':True,'translation_semantic_equivalence_verified':False,'protected_rows_modified':False}


def deduplicate_and_split(records, *, seed=20261023):
    """Preserve official train; divide official validation by question groups.

    A question-content proxy is explicitly not a patient/document identifier.
    Repeated questions with different choices remain in one split. Conflicting
    references for the same ordered input are quarantined, never majority-voted.
    """
    by_input=defaultdict(list);excluded=[];unique=[]
    for row in records:
        inp=row['input'];by_input[digest([normalized(inp['question']),inp.get('options')])].append(row)
    priority={'test':0,'validation':1,'train':2,None:3}
    for rows in by_input.values():
        if len({normalized(r['supervision']['answer']) for r in rows})>1:
            excluded.extend({'sample_id':r['sample_id'],'reason':'CONFLICTING_REFERENCE_SAME_INPUT'} for r in rows);continue
        rows.sort(key=lambda r:(priority.get(r['original_split'],3),r['sample_id']))
        unique.append(rows[0]);excluded.extend({'sample_id':r['sample_id'],'reason':'EXACT_INPUT_DUPLICATE'} for r in rows[1:])
    groups=defaultdict(list)
    for row in unique:groups[digest(normalized(row['input']['question']))].append(row)
    result=[]
    for key,rows in sorted(groups.items()):
        official={r['original_split'] for r in rows}
        if 'test' in official:split='test';retain='test'
        elif 'validation' in official:
            split='validation' if int(digest([seed,key])[:16],16)%2==0 else 'test';retain='validation'
        elif 'train' in official:split='train';retain='train'
        else:
            n=int(digest([seed,key])[:16],16)%100
            split='test' if n<5 else 'validation' if n<10 else 'train';retain=None
        for row in rows:
            if row['original_split']!=retain:
                excluded.append({'sample_id':row['sample_id'],'reason':'QUESTION_GROUP_OVERLAP_WITH_OFFICIAL_HELDOUT'});continue
            row['split']=split;row['group_ids']['duplicate_cluster']='aihub_71875:question:'+key
            row['metadata']['v2_split_policy']='official_train; official_validation_question_groups_hash_even_dev_odd_test; seed20261023'
            result.append(row)
    return result,excluded


def build_71875(import_manifest, data_root, *, protected_freeze, license_ref):
    """Produce an uncapped v2 candidate pool; selection/training is a later step."""
    root=Path(data_root);manifest=json.loads(Path(import_manifest).read_text())
    if manifest['dataset_id']!='71875':raise ValueError('WRONG_DATASET_IMPORT')
    if any(not Path(f['path']).is_file() or sha256(f['path'])!=f['sha256'] for f in manifest['files']):raise ValueError('DELIVERY_SOURCE_CHANGED')
    protected_manifest=json.loads(Path(protected_freeze).read_text());protected=[];protected_receipts={}
    for split in ('validation','test'):
        entry=protected_manifest['splits']['kormedmcqa/'+split];p=Path(entry.get('snapshot_path',entry['path']))
        if sha256(p)!=entry['sha256']:raise ValueError('PROTECTED_KORMED_FILE_CHANGED')
        rows=read_jsonl(p)
        if len(rows)!=entry['count']:raise ValueError('PROTECTED_KORMED_COUNT_CHANGED')
        protected+=rows;protected_receipts[split]={'path':str(p),'sha256':entry['sha256'],'count':len(rows)}
    if len(protected)!=4068:raise ValueError('FULL_OFFICIAL_KORMED_DEV_TEST_4068_REQUIRED')
    records,stats=records_71875(json_documents(manifest),revision=manifest['import_id'],license_ref=license_ref,fetched_at=manifest['received_at'])
    stats['parsed_official_split_counts']=dict(Counter(r['original_split'] for r in records))
    records,extra=deduplicate_and_split(records);stats['exclusions']+=extra
    records,leak=leakage_audit(records,protected)
    for row in records:
        if validate_record(row):raise ValueError('INVALID_CANONICAL_AI_HUB_RECORD')
    stats.update({'schema_version':VERSION,'source_id':'aihub_71875','status':'SUCCEEDED','evidence_level':'ACTUAL_SOURCE_NORMALIZED','import_id':manifest['import_id'],'import_manifest_sha256':sha256(import_manifest),'protected_freeze_sha256':sha256(protected_freeze),'protected_files':protected_receipts,'rights_evidence':license_ref,'seed':20261023,'test_is_new_blind_source_holdout':True,'original_official_test_supplied':False,'per_record_human_review':'unknown','clinical_review':'not_performed','near_semantic_translation_equivalence':'not_verified','excluded_counts':dict(Counter(x['reason'] for x in stats['exclusions'])),'leakage_audit':leak,'canonical_counts':dict(Counter(r['split'] for r in records)),'task_counts':dict(Counter(r['task_type'] for r in records)),'q_type_split_counts':dict(Counter(r['split']+'/'+r['metadata']['q_type'] for r in records)),'source_version':'1.1','patient_group_verified':False})
    pool_id=digest({'import_id':manifest['import_id'],'code_sha256':sha256(__file__),'protected_freeze':sha256(protected_freeze),'seed':20261023,'license_ref':license_ref})
    folder=_private(root/'prepared'/'aihub_71875'/pool_id)
    stats['pool_id']=pool_id;stats['canonical_files']={}
    code=Path(__file__).read_bytes();snapshot=_private(folder/'source_snapshot')/'aihub.py'
    if snapshot.exists() and snapshot.read_bytes()!=code:raise ValueError('IMMUTABLE_TRANSFORM_SNAPSHOT_CHANGED')
    if not snapshot.exists():snapshot.write_bytes(code)
    stats['transformation_code_sha256']=sha256(snapshot)
    for split in ('train','validation','test'):
        rows=sorted((r for r in records if r['split']==split),key=lambda r:r['sample_id']);path=folder/(split+'.jsonl')
        if path.exists():
            if read_jsonl(path)!=rows:raise ValueError('IMMUTABLE_PREPARED_POOL_CHANGED')
        else:write_jsonl(path,rows)
        stats['canonical_files'][split]={'path':str(path),'count':len(rows),'sha256':sha256(path)}
    atomic_json(folder/'audit.json',stats)
    return stats


def join_71762(questions, answers, *, contract):
    """Prepared explicit-ID contract; real 71762 pairing remains unverified.

    A declared one-to-many relation is allowed only with unique child IDs and
    evidence defining which question parent owns each answer. Cartesian joins
    and guessed disease/intent keys are never allowed.
    """
    forbidden={'disease','disease_name','intent','q_intent','intention','index','row','file_index'}
    required=('question_id','answer_parent_key','answer_id','cardinality','evidence_sha256')
    if any(not contract.get(k) for k in required) or any(contract[k] in forbidden for k in required[:3]):raise ValueError('UNRESOLVED_PAIRING: explicit evidenced identifier contract required')
    if contract['cardinality'] not in {'one_to_one','one_to_many','many_to_one'}:raise ValueError('UNRESOLVED_PAIRING: unsupported relationship')
    if contract['cardinality']=='many_to_one':
        foreign=contract.get('question_answer_key')
        if not foreign or foreign in forbidden:raise ValueError('UNRESOLVED_PAIRING: evidenced answer foreign key required')
        answers_by_id={};question_ids=set();result=[];unresolved=[]
        for answer in answers:
            key=answer.get(contract['answer_id'])
            if key is None or key in answers_by_id:raise ValueError('UNRESOLVED_PAIRING: duplicate or missing answer ID')
            answers_by_id[key]=answer
        for question in questions:
            key=question.get(contract['question_id']);answer_id=question.get(foreign)
            if key is None or key in question_ids:raise ValueError('UNRESOLVED_PAIRING: duplicate or missing question ID')
            question_ids.add(key)
            if answer_id not in answers_by_id:unresolved.append({'question_id_digest':digest(key),'reason':'UNRESOLVED_PAIRING'});continue
            result.append({'question_record':question,'answer_record':answers_by_id[answer_id],'parent_group':question.get(contract.get('augmentation_parent_key','')) or answer_id,'pairing_evidence_sha256':contract['evidence_sha256']})
        return result,unresolved
    qmap={};amap=defaultdict(list);seen_answers=set();unresolved=[]
    for q in questions:
        key=q.get(contract['question_id'])
        if key is None or key in qmap:raise ValueError('UNRESOLVED_PAIRING: duplicate or missing question ID')
        qmap[key]=q
    for a in answers:
        aid=a.get(contract['answer_id']);parent=a.get(contract['answer_parent_key'])
        if aid is None or aid in seen_answers:raise ValueError('UNRESOLVED_PAIRING: duplicate or missing answer child ID')
        seen_answers.add(aid)
        if parent not in qmap:unresolved.append({'answer_id_digest':digest(aid),'reason':'UNRESOLVED_PAIRING'});continue
        amap[parent].append(a)
    result=[]
    for key,q in qmap.items():
        children=amap[key]
        if not children:unresolved.append({'question_id_digest':digest(key),'reason':'UNRESOLVED_PAIRING'});continue
        if len(children)>1 and contract['cardinality']=='one_to_one':raise ValueError('UNRESOLVED_PAIRING: cardinality contradicts contract')
        for a in children:result.append({'question_record':q,'answer_record':a,'parent_group':q.get(contract.get('augmentation_parent_key','')) or key,'pairing_evidence_sha256':contract['evidence_sha256']})
    return result,unresolved


def balanced_select(records, quotas, *, field='q_type', seed=20261023):
    """Deterministic domain round-robin with indivisible known parent groups."""
    grouped=defaultdict(list)
    for row in records:
        group=row['group_ids'].get('parent') or row['group_ids'].get('duplicate_cluster') or row['group_ids'].get('document') or row['sample_id']
        grouped[group].append(row)
    bins=defaultdict(lambda:defaultdict(list));ineligible=[]
    for group,rows in grouped.items():
        strata={str(r.get('metadata',{}).get(field,r.get(field))) for r in rows}
        if len(strata)!=1:ineligible.extend(r['sample_id'] for r in rows);continue
        stratum=next(iter(strata));domain=str(rows[0].get('metadata',{}).get('domain','unspecified'))
        bins[stratum][domain].append((digest([seed,group]),rows))
    selected=[];counts=Counter()
    for stratum,limit in sorted(quotas.items()):
        domains=bins[str(stratum)]
        for domain in domains:domains[domain].sort(reverse=True,key=lambda x:x[0])
        while counts[str(stratum)]<limit and any(domains.values()):
            progressed=False
            for domain in sorted(domains):
                while domains[domain]:
                    _,rows=domains[domain].pop()
                    if counts[str(stratum)]+len(rows)>limit:continue
                    selected.extend(rows);counts[str(stratum)]+=len(rows);progressed=True;break
            if not progressed:break
    if dict(counts)!={str(k):v for k,v in quotas.items()}:raise ValueError('INSUFFICIENT_WHOLE_GROUPS_FOR_FIXED_QUOTAS')
    selected.sort(key=lambda r:digest([seed,r['sample_id']]))
    return selected,{'method':'whole_group_domain_round_robin_sha256_order','seed':seed,'quotas':quotas,'selected_counts':dict(counts),'mixed_stratum_group_excluded_ids':ineligible,'selected_sample_ids':[r['sample_id'] for r in selected],'domain_counts':dict(Counter(str(r['metadata'].get('domain','unspecified')) for r in selected))}


def freeze_corpus(data_root, name, splits, *, provenance):
    """Separate v2 content-addressed corpus; no v1 pointers/reports are touched."""
    if not re.fullmatch(r'[a-z0-9_\-]+',name):raise ValueError('INVALID_CORPUS_NAME')
    root=Path(data_root).resolve();seen=set();groups=defaultdict(set);entries={};input_splits=defaultdict(set)
    for split,records in splits.items():
        if split not in {'train','validation','test'}:raise ValueError('INVALID_SPLIT')
        for row in records:
            if row['split']!=split or validate_record(row):raise ValueError('CORPUS_RECORD_INVALID')
            if row['sample_id'] in seen:raise ValueError('DUPLICATE_CORPUS_SAMPLE_ID')
            seen.add(row['sample_id'])
            input_splits[input_digest(row)].add(split)
            for kind,value in row['group_ids'].items():
                if value:groups[(row['source_id'],kind,str(value))].add(split)
    if any(len(s)>1 for s in groups.values()):raise ValueError('CORPUS_GROUP_CROSS_SPLIT')
    if any(len(s)>1 for s in input_splits.values()):raise ValueError('CORPUS_EXACT_INPUT_CROSS_SPLIT')
    for split,records in splits.items():
        rows=sorted(records,key=lambda r:r['sample_id'])
        payload=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n' for r in rows).encode()
        h=hashlib.sha256(payload).hexdigest();path=_private(root/'frozen'/h)/(split+'.jsonl')
        if path.exists():
            if sha256(path)!=h:raise ValueError('IMMUTABLE_FROZEN_FILE_CHANGED')
        else:
            tmp=path.with_suffix('.tmp');tmp.write_bytes(payload);os.replace(tmp,path)
        entries[name+'/'+split]={'path':str(path),'snapshot_path':str(path),'sha256':h,'count':len(rows),'ordered_sample_ids':[r['sample_id'] for r in rows],'source_counts':dict(Counter(r['source_id'] for r in rows))}
    identity={'schema_version':'medvlm.freeze.v1','corpus_name':name,'seed':20261023,'splits':entries,'provenance':provenance,'transformation_code_sha256':sha256(__file__),'status':'SUCCEEDED','evidence_level':'ACTUAL_SOURCE_NORMALIZED'}
    identity['freeze_id']=digest(identity);path=root/'manifests'/('freeze-'+identity['freeze_id']+'.json')
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('IMMUTABLE_FREEZE_MANIFEST_CHANGED')
    if not path.exists():atomic_json(path,identity)
    return {**identity,'manifest_path':str(path),'manifest_sha256':sha256(path)}


def token_audit(records, configs, *, max_length=2048):
    """CPU native-template boundaries and actual target tokens, without weights."""
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer
    from ..training.rendering import render_messages, assistant_target
    models={};encodings={}
    for key,config in configs.items():
        files={}
        for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
            path=try_to_load_from_cache(config['model_id'],name,cache_dir=config.get('cache_dir'),revision=config['revision'])
            if isinstance(path,str):files[name]=sha256(path)
        tokenizer=AutoTokenizer.from_pretrained(config['model_id'],revision=config['revision'],cache_dir=config.get('cache_dir'),local_files_only=True,trust_remote_code=False)
        template=tokenizer.get_chat_template();identity=digest({'files':files,'chat_template':template,'enable_thinking':False})
        if identity not in encodings:
            measured={};counts=Counter()
            for row in records:
                messages=render_messages(row)
                # Match training.rendering.native_encode exactly. Transformers
                # 5 may return BatchEncoding from tokenize=True (len=field count).
                prompt_text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                full_text=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':assistant_target(row)}],tokenize=False,add_generation_prompt=False,enable_thinking=False)
                prompt=tokenizer(prompt_text,add_special_tokens=False)['input_ids']
                full=tokenizer(full_text,add_special_tokens=False)['input_ids']
                reason='ASSISTANT_BOUNDARY_MISMATCH' if full[:len(prompt)]!=prompt or len(full)<=len(prompt) else 'CONTEXT_OVERFLOW' if len(full)>max_length else None
                measured[row['sample_id']]={'sequence_tokens':len(full),'prompt_tokens':len(prompt),'supervised_tokens':len(full)-len(prompt),'eligible':reason is None,'reason':reason}
                counts[reason or 'ELIGIBLE']+=1
            encodings[identity]={'samples':measured,'counts':dict(counts)}
        models[key]={'model_id':config['model_id'],'revision':config['revision'],'native_tokenizer_files':files,'chat_template_sha256':hashlib.sha256(template.encode()).hexdigest(),'encoding_identity':identity}
    common=[row['sample_id'] for row in records if all(encodings[m['encoding_identity']]['samples'][row['sample_id']]['eligible'] for m in models.values())]
    return {'schema_version':VERSION,'evidence_level':'ACTUAL_CPU_NATIVE_TOKENIZATION','max_length':max_length,'models':models,'tokenizers_identical':len(encodings)==1,'encodings':encodings,'common_eligible_sample_ids':common,'common_eligible_count':len(common),'measured_count':len(records),'no_model_weights_loaded':True,'no_gpu_used':True}


def semantic_audit(candidates, protected, model_receipt, output_dir, *, threshold=.92, top_k=5):
    """Local CPU multilingual candidate search, never a clinical duplicate verdict.

    Whole question and original ordered choices are retained. Inputs exceeding
    the native 512-token context use overlapping chunks, each with query prefix;
    their unit mean-pooled vectors are averaged then normalized. Gold is absent.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    out=_private(output_dir);receipt=json.loads(Path(model_receipt).read_text());local=Path(receipt['local_path'])
    for name,item in receipt['files'].items():
        if sha256(local/name)!=item['sha256']:raise ValueError('EMBEDDING_NATIVE_FILE_CHANGED')
    records=sorted(protected,key=lambda r:r['sample_id'])+sorted(candidates,key=lambda r:r['sample_id'])
    if len({r['sample_id'] for r in records})!=len(records):raise ValueError('SEMANTIC_AUDIT_DUPLICATE_ID')
    def text_for(row):
        options=row['input'].get('options') or {}
        values=options.items() if isinstance(options,dict) else enumerate(options)
        return row['input']['question']+''.join('\n'+str(k)+'. '+str(v) for k,v in values)
    policy={'model_id':receipt['model_id'],'revision':receipt['revision'],'model_receipt_sha256':sha256(model_receipt),'threshold':threshold,'top_k':top_k,'input_policy':'question_plus_original_ordered_options_no_answer_no_context','prefix':'query: ','pooling':'masked_token_mean_then_l2; mean_of_l2_chunk_vectors_then_l2','max_tokens_per_chunk':512,'chunk_overlap_tokens':64,'device':'cpu','dtype':'float32','cpu_threads':4,'candidate_count':len(candidates),'protected_count':len(protected),'sample_ids':[r['sample_id'] for r in records],'text_hashes':[digest(text_for(r)) for r in records],'transform_version':'medvlm.semantic-candidates.v1'}
    policy_path=out/'policy.json'
    if policy_path.exists() and json.loads(policy_path.read_text())!=policy:raise ValueError('IMMUTABLE_SEMANTIC_POLICY_CHANGED')
    if not policy_path.exists():atomic_json(policy_path,policy)
    embeddings_path=out/'embeddings.npy';encoding_path=out/'encoding.json'
    if embeddings_path.exists() and encoding_path.exists():
        encoding=json.loads(encoding_path.read_text())
        if encoding['policy_sha256']!=sha256(policy_path) or encoding['embeddings_sha256']!=sha256(embeddings_path):raise ValueError('SEMANTIC_CACHE_CHANGED')
        vectors=np.load(embeddings_path,allow_pickle=False)
    else:
        torch.set_num_threads(4)
        tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False)
        model=AutoModel.from_pretrained(local,local_files_only=True,trust_remote_code=False,dtype=torch.float32).to('cpu').eval()
        # Verify the pinned XLM-R tokenizer's native special-token layout rather
        # than relying on a removed Transformers build_inputs helper.
        probe=tokenizer('query: structural fixture',add_special_tokens=False)['input_ids']
        native_probe=tokenizer('query: structural fixture',add_special_tokens=True)['input_ids']
        if native_probe!=[tokenizer.bos_token_id]+probe+[tokenizer.eos_token_id]:raise ValueError('UNSUPPORTED_EMBEDDING_SPECIAL_TOKEN_LAYOUT')
        prefix=tokenizer('query: ',add_special_tokens=False)['input_ids'];budget=512-len(prefix)-tokenizer.num_special_tokens_to_add(pair=False)
        chunks=[];row_chunks=Counter();lengths={}
        for i,row in enumerate(records):
            ids=tokenizer(text_for(row),add_special_tokens=False,truncation=False)['input_ids'];lengths[row['sample_id']]=len(ids)
            for start in range(0,max(1,len(ids)),budget-64):
                part=ids[start:start+budget];tokens=[tokenizer.bos_token_id]+prefix+part+[tokenizer.eos_token_id]
                chunks.append((i,tokens));row_chunks[i]+=1
                if start+budget>=len(ids):break
        chunks.sort(key=lambda x:len(x[1]));vectors=np.zeros((len(records),model.config.hidden_size),dtype=np.float32)
        with torch.inference_mode():
            for start in range(0,len(chunks),32):
                batch=chunks[start:start+32]
                encoded=tokenizer.pad({'input_ids':[x[1] for x in batch]},padding=True,return_tensors='pt')
                hidden=model(**encoded).last_hidden_state
                mask=encoded['attention_mask'].unsqueeze(-1).bool()
                pooled=hidden.masked_fill(~mask,0).sum(1)/mask.sum(1)
                pooled=F.normalize(pooled,p=2,dim=1).cpu().numpy()
                for (i,_),vector in zip(batch,pooled):vectors[i]+=vector
                if start%2048==0:atomic_json(out/'progress.json',{'status':'RUNNING','completed_chunks':min(start+len(batch),len(chunks)),'total_chunks':len(chunks),'device':'cpu'})
        vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
        temporary=embeddings_path.with_suffix('.tmp')
        with temporary.open('wb') as stream:np.save(stream,vectors,allow_pickle=False)
        os.replace(temporary,embeddings_path)
        encoding={'policy_sha256':sha256(policy_path),'embeddings_sha256':sha256(embeddings_path),'rows':len(records),'chunks':len(chunks),'multi_chunk_rows':sum(n>1 for n in row_chunks.values()),'all_input_tokens_covered':True,'truncated_rows':0,'source_token_lengths':lengths,'torch_version':torch.__version__}
        atomic_json(encoding_path,encoding)
    found=[];n=len(protected);protected_ids=[r['sample_id'] for r in records[:n]]
    for start in range(n,len(records),256):
        scores=vectors[start:start+256]@vectors[:n].T
        for offset,row_scores in enumerate(scores):
            # Stable ID order resolves equal score ties deterministically.
            best=np.argsort(-row_scores,kind='stable')[:top_k]
            for rank,index in enumerate(best,1):
                score=float(row_scores[index])
                if score>=threshold:found.append({'sample_id':records[start+offset]['sample_id'],'protected_sample_id':protected_ids[index],'cosine':score,'rank':rank,'action':'quarantine_candidate_pending_clinical_equivalence_review','confirmed_duplicate':False})
    report={'status':'SUCCEEDED','evidence_level':'ACTUAL_CPU_SEMANTIC_CANDIDATE_SEARCH','policy_path':str(policy_path),'policy_sha256':sha256(policy_path),'encoding_path':str(encoding_path),'encoding_sha256':sha256(encoding_path),'candidate_count':len(candidates),'protected_count':n,'threshold':threshold,'top_k':top_k,'candidate_pairs':found,'candidate_rows':len({x['sample_id'] for x in found}),'all_input_tokens_covered':True,'semantic_equivalence_verified':False,'translation_equivalence_verified':False,'raw_data_sent_to_external_service':False,'gpu_used':False,'limitations':['Multilingual similarity is candidate screening, not proof of medical or translated equivalence.','Threshold is fixed before search; false positive and missed paraphrase candidates remain possible.','Chunk means for long inputs can dilute localized differences; numbers, units, negation and original option order are retained.']}
    atomic_json(out/'audit.json',report);atomic_json(out/'progress.json',{'status':'SUCCEEDED','candidate_rows':report['candidate_rows'],'device':'cpu'})
    return report


def prepare_text_corpora_v2(data_root, *, pool_audit, token_audit_path, semantic_audit_path,
                           internal_audit_path, en_validation_token_audit, v1_freeze):
    """Freeze the explicitly approved KO4k / MIX8k, one-pass experiment corpus."""
    root=Path(data_root).resolve();pool=json.loads(Path(pool_audit).read_text())
    token=json.loads(Path(token_audit_path).read_text());semantic=json.loads(Path(semantic_audit_path).read_text());internal=json.loads(Path(internal_audit_path).read_text());en_token=json.loads(Path(en_validation_token_audit).read_text());old=json.loads(Path(v1_freeze).read_text())
    if semantic['status']!='SUCCEEDED' or semantic['protected_count']!=4068:raise ValueError('FULL_PROTECTED_SEMANTIC_AUDIT_REQUIRED')
    if token['max_length']!=2048 or en_token['max_length']!=2048:raise ValueError('FIXED_CORPUS_LENGTH_MISMATCH')
    def load_entry(entry):
        path=Path(entry.get('snapshot_path',entry['path']))
        if sha256(path)!=entry['sha256']:raise ValueError('SOURCE_CORPUS_CHANGED')
        rows=read_jsonl(path)
        if len(rows)!=entry['count']:raise ValueError('SOURCE_CORPUS_COUNT_CHANGED')
        return rows
    ko={s:load_entry(pool['canonical_files'][s]) for s in ('train','validation','test')}
    en_train=load_entry(old['splits']['text/train']);en_valid=load_entry(old['splits']['text/validation'])
    allowed=set(token['common_eligible_sample_ids']);allowed_valid=set(en_token['common_eligible_sample_ids'])
    blocked={x['sample_id'] for x in semantic['candidate_pairs']}|{x['sample_id'] for x in internal['confirmed_overlaps']}|{x['sample_id'] for x in internal['near_candidates']}
    all_candidates=ko['train']+ko['validation']+en_train
    def group(row):return row['group_ids'].get('parent') or row['group_ids'].get('duplicate_cluster') or row['group_ids'].get('document') or row['sample_id']
    blocked_groups={group(r) for r in all_candidates if r['sample_id'] in blocked}
    def eligible(rows,ids):return [r for r in rows if r['sample_id'] in ids and group(r) not in blocked_groups]
    ko_selected,ko_selection=balanced_select(eligible(ko['train'],allowed),{'1':1500,'2':1300,'3':1200})
    en_selected,en_selection=balanced_select(eligible(en_train,allowed),{'pubmedqa_labeled':450,'medquad':3550},field='source_id')
    ko_valid=eligible(ko['validation'],allowed)
    en_valid_available=eligible(en_valid,allowed_valid)
    # Preserve the 1:1 KO/EN validation count and all 50 official PubMedQA dev
    # rows when available, with indivisible MedQuAD document groups.
    en_valid_selected,en_valid_selection=balanced_select(en_valid_available,{'pubmedqa_labeled':50,'medquad':len(ko_valid)-50},field='source_id')
    selection={'seed':20261023,'max_length':2048,'ko':ko_selection,'en':en_selection,'en_validation':en_valid_selection,'semantic_quarantine_rows':len(blocked),'semantic_quarantine_groups':len(blocked_groups),'ko_validation_count':len(ko_valid),'common_validation_count':len(ko_valid)+len(en_valid_selected),'validation_policy':'identical_KO_EN_1_to_1_retention_selection_for_both_train_conditions; EN_validation_is_not_KO_training','full_aihub_heldout_preserved':True,'corpus_selection_used_test_performance':False,'train_lengths':{'ko':4000,'mix':8000},'one_pass_update_counts_at_batch16':{'ko':250,'mix':500},'pre_freeze_decision_history':[{'initial_ko_qtype_quotas':{'1':1334,'2':1333,'3':1333},'initial_common_validation_count':1912,'replacement_ko_qtype_quotas':{'1':1500,'2':1300,'3':1200},'replacement_common_validation_count':1516,'reason':'fixed_0.92_semantic_candidate_quarantine_left_only_1240_long_train_and_758_KO_dev; conservative_candidates_are_not_confirmed_duplicates','decision_basis':'source_contamination_screening_before_any_v2_training_or_test_score','threshold_changed':False}]}
    if len(ko_valid)!=758 or len(en_valid_selected)!=758:raise ValueError('APPROVED_COMMON_VALIDATION_758_PLUS_758_REQUIRED')
    selection_path=root/'manifests/text-corpus-selection-v2.json'
    if selection_path.exists() and json.loads(selection_path.read_text())!=selection:raise ValueError('IMMUTABLE_CORPUS_SELECTION_CHANGED')
    if not selection_path.exists():atomic_json(selection_path,selection)
    sources={'source_audit_sha256':sha256(pool_audit),'v1_freeze_sha256':sha256(v1_freeze),'token_audit_sha256':sha256(token_audit_path),'en_validation_token_audit_sha256':sha256(en_validation_token_audit),'semantic_audit_sha256':sha256(semantic_audit_path),'internal_split_audit_sha256':sha256(internal_audit_path),'selection_manifest_sha256':sha256(selection_path),'source_license':pool['rights_evidence'],'test_used_for_selection':False}
    result_path=root/'manifests/text-corpora-v2.json'
    if result_path.exists():
        previous=json.loads(result_path.read_text())
        # Completed freezes retain their original transform snapshot even when
        # unrelated import utilities are subsequently repaired. Verify receipts
        # and bytes instead of silently issuing a different successful freeze.
        manifest_files=[Path(previous['aihub_holdout_freeze'])]
        for corpus in previous['corpora'].values():
            if sha256(corpus['freeze_manifest'])!=corpus['freeze_manifest_sha256']:raise ValueError('COMPLETED_CORPUS_MANIFEST_CHANGED')
            manifest_files.append(Path(corpus['freeze_manifest']))
        for path in manifest_files:
            frozen=json.loads(path.read_text());identity=dict(frozen);found=identity.pop('freeze_id')
            if digest(identity)!=found or frozen['provenance']!=sources:raise ValueError('COMPLETED_CORPUS_PROVENANCE_CHANGED')
            for entry in frozen['splits'].values():
                if sha256(entry['snapshot_path'])!=entry['sha256']:raise ValueError('COMPLETED_CORPUS_BYTES_CHANGED')
        return previous
    holdout=freeze_corpus(root,'aihub_v2_holdout',{'validation':ko['validation'],'test':ko['test']},provenance=sources)
    result={'schema_version':'medvlm.text-corpora.v2','seed':20261023,'max_length':2048,'training_recipe':'one_pass','corpora':{},'aihub_test_file':holdout['splits']['aihub_v2_holdout/test']['snapshot_path'],'aihub_validation_file':holdout['splits']['aihub_v2_holdout/validation']['snapshot_path'],'aihub_holdout_freeze':holdout['manifest_path'],'source_audit_path':str(Path(pool_audit).resolve()),'token_audit_path':str(Path(token_audit_path).resolve()),'semantic_audit_path':str(Path(semantic_audit_path).resolve()),'internal_audit_path':str(Path(internal_audit_path).resolve()),'selection_manifest_path':str(selection_path),'source_import_path':str(root/'manifests/imports'/(pool['import_id']+'.json')),'status':'SUCCEEDED','evidence_level':'ACTUAL_SOURCE_FROZEN','training_completed':False}
    common_validation=ko_valid+en_valid_selected
    for name,train,valid in [('ko',ko_selected,common_validation),('mix',ko_selected+en_selected,common_validation)]:
        frozen=freeze_corpus(root,name+'_v2',{'train':train,'validation':valid,'test':ko['test']},provenance=sources)
        model_tokens={}
        for key,model in token['models'].items():
            measures=token['encodings'][model['encoding_identity']]['samples']
            model_tokens[key]={'supervised_tokens':sum(measures[r['sample_id']]['supervised_tokens'] for r in train),'sequence_tokens':sum(measures[r['sample_id']]['sequence_tokens'] for r in train)}
        result['corpora'][name]={'train_file':frozen['splits'][name+'_v2/train']['snapshot_path'],'validation_file':frozen['splits'][name+'_v2/validation']['snapshot_path'],'freeze_manifest':frozen['manifest_path'],'freeze_manifest_sha256':frozen['manifest_sha256'],'train_count':len(train),'validation_count':len(valid),'supervised_tokens':next(iter(model_tokens.values()))['supervised_tokens'] if token['tokenizers_identical'] else None,'token_counts_by_model':model_tokens,'source_counts':dict(Counter(r['source_id'] for r in train)),'task_counts':dict(Counter(r['task_type'] for r in train)),'language_counts':dict(Counter(r['language'] for r in train)),'optimizer_steps_at_batch16':(len(train)+15)//16,'selected_ko_identical_in_mix':True}
    holdout_ids={split:{qtype:[r['sample_id'] for r in ko[split] if r['metadata']['q_type']==qtype] for qtype in ('1','2','3')} for split in ('validation','test')}
    eval_selection={'seed':20261023,'selection_rule':'all_new_source_heldout_before_evaluation','source_official_split':'validation','internal_split_method':'question_content_group_hash_even_dev_odd_test','sample_ids_by_split_q_type':holdout_ids,'counts_by_split_q_type':{s:{q:len(ids) for q,ids in groups.items()} for s,groups in holdout_ids.items()},'holdout_freeze':holdout['manifest_path'],'test_used_for_selection':False,'protected_kormed_semantic_candidate_test_ids':[r['sample_id'] for r in ko['test'] if r['sample_id'] in blocked],'protected_kormed_semantic_candidates_are_confirmed_duplicates':False,'semantic_candidate_test_rows_preserved':True}
    eval_path=root/'manifests/aihub-heldout-v2.json';atomic_json(eval_path,eval_selection);result['aihub_evaluation_selection_path']=str(eval_path)
    path=root/'manifests/text-corpora-v2.json'
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError('IMMUTABLE_CORPUS_MANIFEST_CHANGED')
    if not path.exists():atomic_json(path,result)
    return result

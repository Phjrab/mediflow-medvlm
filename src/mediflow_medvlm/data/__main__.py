import argparse
import json
import os
import urllib.error
from pathlib import Path
from .pipeline import SOURCES, fetch, build, audit, freeze


def main(argv=None):
    parser=argparse.ArgumentParser(description='Acquire, normalize, audit and freeze actual public medical data')
    parser.add_argument('command',choices=['discover','fetch','build','audit','freeze','aihub-import','aihub-build','aihub-corpora'])
    parser.add_argument('--data-root',type=Path,default=os.environ.get('MEDVLM_DATA_ROOT'))
    parser.add_argument('--source',choices=[*SOURCES,'all'],default='all')
    parser.add_argument('--input',type=Path,action='append',help='User-provided AI-Hub ZIP, split part, or original folder; repeat for parts')
    parser.add_argument('--dataset-id',choices=['71875','71762'])
    parser.add_argument('--version',help='Verified AI-Hub delivery version')
    parser.add_argument('--delivery-receipt',type=Path,help='Local credential-free delivery receipt')
    parser.add_argument('--import-manifest',type=Path)
    parser.add_argument('--license-evidence')
    parser.add_argument('--protected-freeze',type=Path)
    parser.add_argument('--pool-audit',type=Path)
    parser.add_argument('--token-audit',type=Path)
    parser.add_argument('--semantic-audit',type=Path)
    parser.add_argument('--internal-audit',type=Path)
    parser.add_argument('--en-validation-token-audit',type=Path)
    args=parser.parse_args(argv)
    if args.command=='discover':print(json.dumps({'sources':SOURCES,'status':'PREPARED_NOT_RUN'}));return 0
    if not args.data_root:parser.error('--data-root or MEDVLM_DATA_ROOT required')
    if args.command=='aihub-import':
        if not args.input or not args.dataset_id or not args.version:parser.error('aihub-import requires --input, --dataset-id and --version')
        from .aihub import import_aihub
        delivery=json.loads(args.delivery_receipt.read_text()) if args.delivery_receipt else None
        if delivery and len(args.input)==1 and args.input[0].is_file():
            from .io import sha256
            if delivery.get('sha256')!=sha256(args.input[0]) or delivery.get('bytes')!=args.input[0].stat().st_size:raise ValueError('DELIVERY_RECEIPT_MISMATCH')
        result=import_aihub(args.input,args.data_root,args.dataset_id,version=args.version,delivery=delivery)
        print(json.dumps({k:result[k] for k in ('status','evidence_level','import_id','dataset_id','version','manifest_path')}|{'archive_count':len(result['archives']),'member_count':len(result['members'])},ensure_ascii=False));return 0
    if args.command=='aihub-build':
        if not args.import_manifest or not args.protected_freeze or not args.license_evidence:parser.error('aihub-build requires --import-manifest, --protected-freeze and --license-evidence')
        from .aihub import build_71875
        result=build_71875(args.import_manifest,args.data_root,protected_freeze=args.protected_freeze,license_ref=args.license_evidence)
        print(json.dumps({k:result[k] for k in ('status','evidence_level','pool_id','canonical_counts','excluded_counts','canonical_files')},ensure_ascii=False));return 0
    if args.command=='aihub-corpora':
        if not all((args.pool_audit,args.token_audit,args.semantic_audit,args.internal_audit,args.en_validation_token_audit,args.protected_freeze)):parser.error('aihub-corpora requires all audit paths and --protected-freeze')
        from .aihub import prepare_text_corpora_v2
        result=prepare_text_corpora_v2(args.data_root,pool_audit=args.pool_audit,token_audit_path=args.token_audit,semantic_audit_path=args.semantic_audit,internal_audit_path=args.internal_audit,en_validation_token_audit=args.en_validation_token_audit,v1_freeze=args.protected_freeze)
        print(json.dumps({'status':result['status'],'manifest_path':str(args.data_root/'manifests/text-corpora-v2.json'),'corpora':{k:{f:v[f] for f in ('train_count','validation_count','supervised_tokens','optimizer_steps_at_batch16')} for k,v in result['corpora'].items()}},ensure_ascii=False));return 0
    if args.command in {'audit','freeze'}:
        result=(audit if args.command=='audit' else freeze)(args.data_root)
        summary={k:result[k] for k in ('status','evidence_level','freeze_id','manifest_path','record_count','critical_error_count') if k in result}
        summary['counts']={key:item['count'] for key,item in result.get('splits',result.get('files',{})).items()}
        if args.command=='audit':summary['manifest_path']=str(args.data_root/'manifests'/'audit.json')
        print(json.dumps(summary,ensure_ascii=False));return 0 if result['status']=='SUCCEEDED' else 1
    outcomes={}
    for source in SOURCES if args.source=='all' else [args.source]:
        try:outcomes[source]=(fetch if args.command=='fetch' else build)(args.data_root,source)
        except Exception as exc:
            status='FAILED_REPRODUCIBLE'
            if isinstance(exc,urllib.error.HTTPError) and exc.code in {401,403}:status='BLOCKED_ACCESS'
            elif isinstance(exc,(urllib.error.URLError,TimeoutError)):status='NETWORK_UNREACHABLE'
            elif 'BLOCKED_LICENSE' in str(exc):status='BLOCKED_LICENSE'
            elif 'SKIPPED_BUDGET' in str(exc):status='SKIPPED_BUDGET'
            outcomes[source]={'status':status,'reason_type':type(exc).__name__,'reason':str(exc)}
    print(json.dumps({s:{**{k:v for k,v in r.items() if k in {'status','acquisition_status','evidence_level','resolved_revision','total_bytes','fetched_files_sha256','counts','canonical_counts','reason','reason_type'}},'manifest_path':str(args.data_root/'manifests'/('sources' if args.command=='fetch' else 'build')/f'{s}.json')} for s,r in outcomes.items()},ensure_ascii=False,indent=2))
    return int(any(r.get('status') not in {'SUCCEEDED','PREPARED_NOT_RUN'} for r in outcomes.values()))

if __name__=='__main__':raise SystemExit(main())

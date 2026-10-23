import argparse
import json
import os
import urllib.error
from pathlib import Path
from .pipeline import SOURCES, fetch, build, audit, freeze


def main(argv=None):
    parser=argparse.ArgumentParser(description='Acquire, normalize, audit and freeze actual public medical data')
    parser.add_argument('command',choices=['discover','fetch','build','audit','freeze'])
    parser.add_argument('--data-root',type=Path,default=os.environ.get('MEDVLM_DATA_ROOT'))
    parser.add_argument('--source',choices=[*SOURCES,'all'],default='all')
    args=parser.parse_args(argv)
    if args.command=='discover':print(json.dumps({'sources':SOURCES,'status':'PREPARED_NOT_RUN'}));return 0
    if not args.data_root:parser.error('--data-root or MEDVLM_DATA_ROOT required')
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

"""Verified-bundle CLI for the six historical union strategies.

No original paper ledgers or plans are read/written. Every attempt gets a new
output directory; failed admission writes diagnostics and no performance table.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys
import pandas as pd
from historical_backtest import FEATURE_FIELDS,run_all
P=Path(__file__).resolve().parent
CERTIFICATES=('membership','identities','sectors','price_units_and_actions','class_representatives')
PANELS=FEATURE_FIELDS+('dividends','splits','sectors','raw_per_signal')


def checked_file(root,record):
    path=(root/record['file']).resolve()
    if root.resolve() not in path.parents:raise ValueError('Bundle file outside bundle directory')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError('Bundle checksum mismatch: '+record['file'])
    return path


def load_bundle(path):
    root=path.parent;manifest=json.loads(path.read_text())
    if manifest.get('universe_version')!='historical_ndx_sp100_union_v3_available_data':
        raise ValueError('Wrong universe version')
    for kind in CERTIFICATES:
        proof=json.loads(checked_file(root,manifest['certificates'][kind]).read_text())
        if proof.get('verified') is not True or proof.get('kind')!=kind:
            raise ValueError('Unverified input certificate: '+kind)
        if proof['from']>manifest['start'] or proof['through']<manifest['end']:
            raise ValueError('Certificate does not cover requested run: '+kind)
        if not proof.get('sources'):raise ValueError('Certificate has no source evidence: '+kind)
        for source in proof['sources']:checked_file(root,source)
    data={}
    for key in PANELS:
        frame=pd.read_csv(checked_file(root,manifest['panels'][key]),index_col=0,parse_dates=True)
        if key in ('market_close','market_dividends'):
            if len(frame.columns)!=1:raise ValueError('Expected single market column')
            data[key]=frame.iloc[:,0]
        else:data[key]=frame
    data.update(issuers=manifest['issuers'],signal_basis=manifest['signal_basis'],execution_basis=manifest['execution_basis'])
    for key in ('cash_acquisitions','terminal_baskets','spin_distributions','true_stock_splits'):
        if key in manifest:data[key]=json.loads(checked_file(root,manifest[key]).read_text())
    return manifest,data


def run_attempt(bundle,output):
    output.mkdir(parents=True,exist_ok=False)
    status=dict(generated_at=datetime.now(timezone.utc).isoformat(),bundle=str(bundle),
                historical_backtest_complete=False,performance_publishable=False,original_accounts_modified=False)
    try:
        manifest,data=load_bundle(bundle)
        results=run_all(data,manifest['start'],manifest['end'])
        complete=all(r['status']=='complete' for r in results.values())
        status.update(status='complete' if complete else 'blocked',historical_backtest_complete=complete,
                      performance_publishable=complete,accounts={n:dict(status=r['status'],error=r.get('error'),blocked_session=r.get('blocked_session')) for n,r in results.items()})
        for r in results.values():
            r['performance_publishable']=complete
            if not complete:r['metrics']=None
        (output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    except (OSError,ValueError,KeyError) as exc:
        status.update(status='blocked',error=str(exc),required_input_certificates=list(CERTIFICATES))
    (output/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    return status


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,default=P/'HISTORICAL_BUNDLE.json')
    ap.add_argument('--output',type=Path,default=P/'backtest_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args=ap.parse_args();status=run_attempt(args.bundle,args.output)
    print(json.dumps(dict(**status,output=str(args.output)),indent=2))
    return 0 if status['status']=='complete' else 2

if __name__=='__main__':sys.exit(main())

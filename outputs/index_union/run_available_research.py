"""Run the explicitly qualified available-data study, leaving strict gates intact.

This entrypoint does not certify complete index history or contemporaneous GICS.
It refuses unresolved eligible identities, changed evidence and incomplete runs.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,argparse
import pandas as pd
from research_scope import sha,check_nested_sources,evidence,assumed_payment
from historical_backtest import FEATURE_FIELDS,run_account,validate_inputs
from dynamic_features import features
from union_execution import ACCOUNTS
P=Path(__file__).resolve().parent

def load(path,lag):
 m=json.loads(path.read_text());data={}
 for r in m['fields']:
  p=evidence(P,r,'file','sha256')
  v=pd.read_csv(p,index_col=0,parse_dates=True,low_memory=False)
  data[r['field']]=v.iloc[:,0] if r['field'].startswith('market_') else v
 data.update(issuers=m['issuers'],signal_basis='split_adjusted',execution_basis='raw_with_explicit_actions')
 permitted={r['symbol'] for r in m['identity_proofs']}
 excluded={r['symbol'] for r in m['scope_exclusions'] if r['reason']=='unresolved_identity'}
 for s in data['eligible']:
  if data['eligible'][s].any() and (s in excluded or (s not in permitted and not s.startswith('research_equity_'))):
   raise ValueError('Unresolved eligible identity: '+s)
 for r in m['identity_proofs']:evidence(P.parent,r)
 evidence_count=0
 for key,file in [('terminal_baskets','TERMINAL_BASKET_CANDIDATES.json'),('spin_distributions','SPIN_DISTRIBUTION_CANDIDATES.json'),('true_stock_splits','TRUE_STOCK_SPLITS.json')]:
  rows=json.loads((P/file).read_text());evidence_count+=check_nested_sources(rows)
  data[key]=assumed_payment(rows,data['close'].index,lag) if key=='terminal_baskets' else rows
 validate_inputs(data,m['start'],m['end'])
 return m,data,evidence_count

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--settlement-lag',type=int,default=3);ap.add_argument('--manifest',type=Path,default=P/'alias_resolved_inputs/MANIFEST.json');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 if args.settlement_lag<1:raise ValueError('Positive settlement convention required')
 args.output.mkdir(parents=True,exist_ok=False)
 status=dict(generated_at=datetime.now(timezone.utc).isoformat(),scope='available_data_research_under_declared_assumptions',strict_complete_history_certified=False,can_start_qualified_research=False,all_six_completed=False,original_accounts_modified=False,assumptions_file='AVAILABLE_RESEARCH_ASSUMPTIONS.json',accounts={})
 try:
  manifest,data,n=load(args.manifest,args.settlement_lag)
  status.update(manifest_sha256=sha(args.manifest),assumptions_sha256=sha(P/'AVAILABLE_RESEARCH_ASSUMPTIONS.json'),corporate_action_evidence_checked=n,start=manifest['start'],end=manifest['end'],settlement_lag=args.settlement_lag)
  calculated=features(**{k:data[k] for k in FEATURE_FIELDS})
  results={}
  for name in ACCOUNTS:
   r=run_account(name,data,manifest['start'],manifest['end'],calculated);results[name]=r
   status['accounts'][name]=dict(status=r['status'],sessions=len(r['daily']),error=r.get('error'),blocked_session=r.get('blocked_session'))
   (args.output/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
   # Always preserve results of a blocked run without advertising returns.
   (args.output/(name+'.json')).write_text(json.dumps(r,allow_nan=False)+'\n')
   print(name,json.dumps(status['accounts'][name]),flush=True)
  complete=all(r['status']=='complete' for r in results.values())
  status.update(all_six_completed=complete,can_start_qualified_research=complete,status='complete' if complete else 'blocked')
  if complete:
   summary={name:r['metrics'] for name,r in results.items()}
   (args.output/'METRICS.json').write_text(json.dumps(dict(scope=status['scope'],strict_complete_history_certified=False,accounts=summary),indent=2)+'\n')
   for name,r in results.items():
    pd.DataFrame(r['daily']).to_csv(args.output/(name+'_daily.csv'),index=False)
    pd.DataFrame(r['trades']).to_csv(args.output/(name+'_trades.csv'),index=False)
  (args.output/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
 except (ValueError,KeyError,OSError) as exc:
  status.update(status='blocked',error=str(exc));(args.output/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n');raise
 return 0 if status['all_six_completed'] else 2
if __name__=='__main__':raise SystemExit(main())

"""Locate execution blockers in actual candidate inputs; never publish returns.

This deliberately cannot grant admission. Ticker identity/sector coverage remains
provisional, so only missing-input locations and progress counts are exported.
"""
import json,hashlib,argparse
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
from historical_backtest import FEATURE_FIELDS,run_account
from dynamic_features import features
from union_execution import ACCOUNTS
P=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=P/'input_diagnostics/MANIFEST.json');ap.add_argument('--output',type=Path,default=P/'ACTUAL_EXECUTION_DIAGNOSTIC.json');args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text())
    data={}
    for r in manifest['fields']:
        path=P/r['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Input changed')
        frame=pd.read_csv(path,index_col=0,parse_dates=True)
        data[r['field']]=frame.iloc[:,0] if r['field'].startswith('market_') else frame
    data.update(issuers=manifest.get('issuers',{s:'PROVISIONAL_'+s for s in data['close']}),signal_basis='split_adjusted',execution_basis='raw_with_explicit_actions')
    data['terminal_baskets']=json.loads((P/'TERMINAL_BASKET_CANDIDATES.json').read_text())
    data['spin_distributions']=json.loads((P/'SPIN_DISTRIBUTION_CANDIDATES.json').read_text())
    data['true_stock_splits']=json.loads((P/'TRUE_STOCK_SPLITS.json').read_text())
    calculated=features(**{k:data[k] for k in FEATURE_FIELDS})
    result=dict(generated_at=datetime.now(timezone.utc).isoformat(),status='diagnostic_only',performance_publishable=False,admission=False,accounts={})
    for name in ACCOUNTS:
        r=run_account(name,data,manifest['start'],manifest['end'],calculated)
        v=dict(status=r['status'],completed_sessions=len(r['daily']),blocked_session=r.get('blocked_session'),error=r.get('error'),filled_order_count=len(r['orders']),closed_trade_count=len(r['trades']))
        v['nonstandard_held_share_actions']=[e for e in r['actions']
            if 'split' in e and abs(e['split']-round(e['split']))>1e-6
            and abs(1/e['split']-round(1/e['split']))>1e-6]
        v['terminal_baskets']=[e['event_id'] for e in r['actions'] if e.get('kind')=='terminal_basket_liquidation']
        if r['status']=='blocked':
            state=r['last_valid_state'];v['held_symbols']=list(state['positions'])
            v['last_valid_session']=r['daily'][-1]['date'] if r['daily'] else None
        result['accounts'][name]=v
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(name,json.dumps(v),flush=True)
    # Return/equity/trade-price arrays are intentionally not exported.
if __name__=='__main__':main()

"""Recompute selected decision inputs using only the price prefix at each date."""
import numpy as np
import pandas as pd
from explore import *
from decisions import feature_frames
from joint_explore import JOINT_WINDOWS

def main():
    env=prepare();px=env['px'];raw={}
    for k in ['close','high','low','volume','sector60']:
        raw[k]=pd.read_csv(O/'index_union/alias_resolved_inputs'/f'{k}.csv.gz',index_col=0,parse_dates=True)
    for entry in read(O/'four_factor_rebound/ETF_SOURCES.json')+read(O/'monthly_reconstruction_v2/ETF_MANIFEST.json'):
        z=pd.read_csv(entry['file'],index_col=0,parse_dates=True)
        for field,col in [('close','Close'),('high','High'),('low','Low'),('volume','Volume')]:raw[field][entry['symbol']]=z[col].reindex(px.index)
        raw['sector60'][entry['symbol']]=px.SPY.pct_change(60,fill_method=None)
    raw={k:v.reindex(index=px.index,columns=px.columns) for k,v in raw.items()}
    dates=sorted({m['start'] for a,b in JOINT_WINDOWS for m in env['meta'] if a<=m['start']<=b})
    dates=sorted(set(dates)|{m['order']['start'] for m in read(P/'ROUNDED_RULE.json')['months']}|{m['order']['start'] for r in read(P/'FROZEN_EXTENSIONS.json') for m in r['months']})
    saved=read(O/'monthly_decision_reconstruction/SNAPSHOTS.json');count=0;cells=0
    for d in dates:
        row=env['rows'][d];actual=feature_frames(px.loc[:d],{k:v.loc[:d] for k,v in raw.items()})
        old={v['symbol']:v['raw'] for v in saved if v['date']==d}
        for k in FEATURES:
            x=actual[k].iloc[-1][row['symbols']].to_numpy();y=np.array([old[s][k] if old[s][k] is not None else np.nan for s in row['symbols']])
            assert np.allclose(x,y,equal_nan=True,rtol=1e-12,atol=1e-12),(d,k)
            count+=1;cells+=len(x)
        print('PREFIX CHECK',d,'passed',flush=True)
    protected=read(P/'PROTECTED_SOURCES.json')
    for f,h in protected.items():assert sha(f)==h,f
    dump('TIMING_VALIDATION.json',dict(dates=dates,feature_date_checks=count,security_feature_cells=cells,
        protected_files_unchanged=len(protected),rule_weights_and_candidate_labels_retrofit=True,
        note='Feature prefix invariance proves no later prices entered these factor values, not point-in-time membership accuracy or independent out-of-sample selection.'))
if __name__=='__main__':main()

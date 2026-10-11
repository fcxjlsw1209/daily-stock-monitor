"""Timing, shared-rule contributions and non-optimizing weight sensitivity."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(P))
from infer_window import picks,replay,image_check
from decisions import CORE,SIDE,FEATURES,feature_frames,CN,display
from frontier import load_first_months
from study import load,read,sha

def dump(n,z): (P/n).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def normalized_group_weights(w):
    w=w.copy();nc=len(CORE)
    w[:nc]*=.5/w[:nc].sum();w[nc:]*=.5/abs(w[nc:]).sum()
    return w

def main():
    selected=read(P/'SELECTED_RULE.json');fit=selected['fit'];w=np.array(fit['weights']);family=fit['family']
    rows=[r for r in load_first_months(121) if r['date'] in {o['start'] for o in selected['orders']}]
    targets=[set(o['holdings']) for o in selected['orders']]
    def hits(v):return [len(set(picks(r,v,family))&t) for r,t in zip(rows,targets)]
    assert hits(w)==[3,3,3]
    rounding=[]
    for decimals in [3,2]:
        v=normalized_group_weights(np.round(w,decimals))
        rounding.append(dict(decimals=decimals,hits=hits(v),holdings=[picks(r,v,family) for r in rows]))
    rng=np.random.default_rng(20261009);random=[]
    for fraction in [.01,.05]:
        outcomes=[hits(normalized_group_weights(w*rng.uniform(1-fraction,1+fraction,len(w)))) for _ in range(500)]
        random.append(dict(relative_weight_perturbation=fraction,trials=500,
            all_three_months_preserved_fraction=float(np.mean([h==[3,3,3] for h in outcomes])),
            mean_names_matched_by_month=np.mean(outcomes,axis=0).tolist()))
    details=[]
    for row,target in zip(rows,targets):
        x=row['X'][family];score=np.einsum('ij,j->i',x,w,optimize=False)
        order=np.argsort(-score,kind='stable');rival=next(i for i in order if row['symbols'][i] not in target)
        entries=[]
        for i in order:
            if row['symbols'][i] not in target:continue
            con=(x[i]-x[rival])*w
            entries.append(dict(symbol=row['symbols'][i],nearest_unchosen=row['symbols'][rival],margin=float(score[i]-score[rival]),
                contributions={k:float(v) for k,v in zip(FEATURES,con)}))
        details.append(dict(date=row['date'],picked=[row['symbols'][i] for i in order[:3]],
            runners_up=[row['symbols'][i] for i in order[3:6]],entries=entries))
    px,_,sources=load();raw={}
    for k in ['close','high','low','volume','sector60']:
        f=O/'index_union/alias_resolved_inputs'/f'{k}.csv.gz';raw[k]=pd.read_csv(f,index_col=0,parse_dates=True);sources[str(f)]=sha(f)
    manifests=read(O/'four_factor_rebound/ETF_SOURCES.json')+read(O/'monthly_reconstruction_v2/ETF_MANIFEST.json')
    for entry in manifests:
        z=pd.read_csv(entry['file'],index_col=0,parse_dates=True)
        for field,col in [('close','Close'),('high','High'),('low','Low'),('volume','Volume')]:raw[field][entry['symbol']]=z[col].reindex(px.index)
        raw['sector60'][entry['symbol']]=px.SPY.pct_change(60,fill_method=None)
    raw={k:v.reindex(index=px.index,columns=px.columns) for k,v in raw.items()}
    saved=read(O/'monthly_decision_reconstruction/SNAPSHOTS.json');checks=0
    for row in rows:
        d=row['date'];f=feature_frames(px.loc[:d],{k:v.loc[:d] for k,v in raw.items()})
        old={v['symbol']:v['raw'] for v in saved if v['date']==d}
        for k in FEATURES:
            actual=f[k].iloc[-1][row['symbols']].to_numpy()
            expected=np.array([old[s][k] if old[s][k] is not None else np.nan for s in row['symbols']])
            assert np.allclose(actual,expected,equal_nan=True,rtol=1e-12,atol=1e-12),(d,k)
            checks+=1
    protected=read(O/'monthly_decision_reconstruction/SOURCE_HASHES.json')
    protected.update(read(P/'SOURCE_HASHES.json'));protected.update(sources)
    for f,h in protected.items():assert sha(f)==h,f
    result=dict(original_sources_unchanged=len(protected),prefix_feature_checks=checks,
        uniform_weights_match_all_three=True,rounding_checks=rounding,perturbation_checks=random,
        sensitivity_is_not_author_trade_probability=True,full_period_backtest_run=False,
        coefficients_refitted_for_next_month=False,minimum_selected_vs_unchosen_gap=fit['margin'])
    dump('RULE_STABILITY.json',result);dump('DECISION_CONTRIBUTIONS.json',details)
    dump('PROTECTED_SOURCES.json',protected)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    for d in details:
        print(d['date'],d['picked'],'runners',d['runners_up'])
        for e in d['entries']:
            print(e['symbol'],'vs',e['nearest_unchosen'],'margin',e['margin'],
                sorted(e['contributions'].items(),key=lambda v:-abs(v[1]))[:4])

if __name__=='__main__':main()

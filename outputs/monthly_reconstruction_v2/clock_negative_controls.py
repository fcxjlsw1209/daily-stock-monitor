"""Intentionally invalid future-information controls. NEVER candidate strategies.
A matching return here diagnoses sensitivity, not evidence about the author's code.
"""
import json
import numpy as np,pandas as pd
from research import P,O,load,raw_features,fixed_rules,Simulator,dump,metrics,fit_errors

def main():
 px,fs,pools=load();raw=raw_features(px,fs);obs=pd.read_csv(O/'monthly_curve_reverse/DIGITIZED_WEALTH.csv');cal=json.loads((O/'monthly_curve_reverse/CALIBRATION.json').read_text());results={};curves={}
 for pool in ['core17','theme17']:
  member=pools[pool];rank={k:v.where(member).rank(axis=1,pct=True) for k,v in raw.items()};rules=fixed_rules(rank,raw);sim=Simulator(px,fs,member)
  for name in ['m252_42','balanced','relative','short_blend']:
   for mode in ['future_score_only','future_score_and_pool']:
    level=1.;nav=pd.Series(index=sim.dates,dtype=float);orders=[]
    for n,i in enumerate(sim.events):
     j=sim.events[n+1] if n+1<len(sim.events) else px.index.get_loc(pd.Timestamp('2026-09-24'));signal=px.index[j];entry=px.index[i]
     mdate=signal if mode=='future_score_and_pool' else entry;valid=sim.valid.loc[mdate]&(raw['m252_42'].loc[signal]>0)&px.iloc[i].gt(0);sc=rules[name].loc[signal].where(valid).dropna().sort_values(ascending=False,kind='stable');chosen=sc.index[:3].tolist()
     if len(chosen)<2:chosen=[]
     frame=px.iloc[i:j+1][chosen]
     if chosen and (not frame.notna().all().all() or not frame.gt(0).all().all()):raise ValueError('unpriceable invalid-control path')
     path=(frame/frame.iloc[0]).mean(axis=1) if chosen else pd.Series(1.,index=px.index[i:j+1]);nav.loc[path.index]=level*path;level=float(nav.iloc[nav.index.get_loc(px.index[j])]);orders.append(dict(entry=str(entry.date()),ILLEGAL_future_signal=str(signal.date()),holdings=chosen))
    tag=f'{pool}__{name}__{mode}';results[tag]=dict(**metrics(nav),**fit_errors(nav,cal,obs),mode='INVALID_LOOKAHEAD_NEGATIVE_CONTROL',not_a_strategy=True);curves[tag]=nav;print(tag,round(results[tag]['cagr_252']*100,2),round(results[tag]['max_dd']*100,2),flush=True)
 dump(P/'INVALID_CLOCK_CONTROLS.json',results);pd.DataFrame(curves).to_csv(P/'INVALID_CLOCK_NAV.csv.gz');dump(P/'INVALID_CLOCK_POLICY.json',dict(description='Intentionally measures future month-end factors and optionally future pool before entering that same month. Forbidden for investment decisions. Used to assess timing sensitivity only; no claim author did this.',tests=len(results)))
if __name__=='__main__':main()

"""Fourth batch: decompose pullback-shape and mature path groups, with frozen ML."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import frontier as z
P=z.P/'shape'
KINDS=['route90','route100','close_low','close_neutral','close_mean3','z075','z175','z225','z_deep',
       'depth1','depth3','depth_atr','no_cmf','no_depth','deep_close_low','route90_close_low']
CONFIG=dict(declared_after_120_runs=True,declared_before_this_batch_results=True,variants=KINDS,cost_bps=[10,30],
 group_weights=[.25]*4,no_leverage=True,max_positions=3,data_certified=False,performance_publishable=False,
 keep='Prior shock pause2,stable gaps,original cached causal Ridge score,entry.75,stop2ATR,profit3ATR,cap10; no changed pool or observer.',
 definitions={
 'route90':'Path group90% matured path rank+10%vol rank vs70/30.',
 'route100':'Path group entirely matured path rank.',
 'close_low':'Replace location rank in shape with inverse location rank. Shape has4equal components.',
 'close_neutral':'Drop location component and equally weight other3 within shape group.',
 'close_mean3':'Replace single-day location rank with rank of trailing3day average location.',
 'z075':'Replace shape target3day standardized pullback -1.25 with-.75.',
 'z175':'Replace standardized pullback target with-1.75.',
 'z225':'Replace standardized pullback target with-2.25.',
 'z_deep':'Use rank(-z) instead of distance to-1.25; original z>-3.5 entry exclusion remains.',
 'depth1':'Distance from20day high target -1vol instead of-2vol.',
 'depth3':'Distance from20day high target -3vol.',
 'depth_atr':'Distance from20day high standardized by ATR/close, target -2; replace std-return normalization.',
 'no_cmf':'Drop CMF component and equally weight remaining3 shape components.',
 'no_depth':'Drop20day-high-depth component and equally weight remaining3 shape components.',
 'deep_close_low':'Predeclared combo z target-1.75 and inverse close location.',
 'route90_close_low':'Predeclared combo route90 and inverse close location.'},
 target_policy=z.CONFIG['target_policy'],development_selection=z.CONFIG['development_selection'],all_history_reused=True)

def shape_scores(d,f,base):
 a=d['adj'];c=d['close'];r=a.pct_change(fill_method=None);sigma=r.rolling(20).std();rank=lambda x:x.where(f['valid']).rank(axis=1,pct=True)
 location=((c-d['low'])/(d['high']-d['low'])).fillna(.5);std=a.pct_change(3,fill_method=None)/(sigma*np.sqrt(3));depth=a/a.rolling(20).max()-1
 cmf=(((2*c-d['high']-d['low'])/(d['high']-d['low'])).fillna(0)*d['volume']).rolling(21).sum()/d['volume'].rolling(21).sum()
 zr=rank(-abs(std+1.25));loc=rank(location);flow=rank(cmf);dep=rank(-abs(depth/sigma+2));shape=(zr+loc+flow+dep)/4
 np.testing.assert_allclose(shape.to_numpy(),f['groups'][3].to_numpy(),equal_nan=True,rtol=1e-12,atol=1e-12)
 vol=rank(sigma);path=(f['groups'][2]-.3*vol)/.7
 new_shapes={
 'close_low':(zr+rank(-location)+flow+dep)/4,
 'close_neutral':(zr+flow+dep)/3,
 'close_mean3':(zr+rank(location.rolling(3).mean())+flow+dep)/4,
 'z075':(rank(-abs(std+.75))+loc+flow+dep)/4,
 'z175':(rank(-abs(std+1.75))+loc+flow+dep)/4,
 'z225':(rank(-abs(std+2.25))+loc+flow+dep)/4,
 'z_deep':(rank(-std)+loc+flow+dep)/4,
 'depth1':(zr+loc+flow+rank(-abs(depth/sigma+1)))/4,
 'depth3':(zr+loc+flow+rank(-abs(depth/sigma+3)))/4,
 'depth_atr':(zr+loc+flow+rank(-abs(depth/(f['atr']/c)+2)))/4,
 'no_cmf':(zr+loc+dep)/3,'no_depth':(zr+loc+flow)/3,
 'deep_close_low':(rank(-abs(std+1.75))+rank(-location)+flow+dep)/4}
 scores={k:base+.25*(value-shape) for k,value in new_shapes.items()}
 scores['route90']=base+.05*(path-vol);scores['route100']=base+.075*(path-vol)
 scores['route90_close_low']=scores['close_low']+.05*(path-vol)
 return scores

def main():
 P.mkdir(exist_ok=True)
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite shape batch')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n');protected=z.protected_hashes()
 oldp=z.t.P;z.t.P=P
 try:d,f,base,gaps,_=z.t.prepare()
 finally:z.t.P=oldp
 x=z.extras(d,f);scores=shape_scores(d,f,base);summary={}
 np.savez_compressed(P/'SCORES.npz',**{k:v.to_numpy() for k,v in scores.items()})
 print('SHAPE_PREPARED',flush=True)
 for name in KINDS:
  for cost in CONFIG['cost_bps']:z.save(P,f'{name}_cost{cost}',z.run(z.BASE,cost,d,f,scores[name],gaps,x),summary)
 original=json.loads((z.P/'SUMMARY.json').read_text());eligible=[k for k in KINDS if summary[k+'_cost10']['development']['max_drawdown']>=-.35 and summary[k+'_cost30']['development']['cagr']>=original['shock_control_cost30']['development']['cagr']]
 selected=max(eligible,key=lambda k:summary[k+'_cost10']['development']['cagr']) if eligible else None
 hits=[k for k,v in summary.items() if v['last252']['return_']>=1]
 (P/'SELECTION.json').write_text(json.dumps(dict(development_selected=selected,eligible=eligible,in_sample_target_hits=hits,criterion=CONFIG['development_selection']),indent=2)+'\n')
 assert protected==z.protected_hashes()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,base_shape_reproduced=True,no_leverage=True),indent=2)+'\n')
 print('SHAPE_COMPLETE',len(summary),'selected',selected,'hits',hits,flush=True)

if __name__=='__main__':main()

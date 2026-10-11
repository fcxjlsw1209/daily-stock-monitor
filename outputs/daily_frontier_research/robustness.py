"""First target-hit validation; declared after discovery, never relabeled out-of-sample."""
import json
import frontier as z
P=z.P/'robustness'
CANDIDATES={'no_profit_cap30':dict(tp=None,cap=30),'trail3_cap30':dict(tp=None,cap=30,trail_atr=3.)}
NEIGHBORS={'cap25':dict(cap=25),'cap35':dict(cap=35),'stop15':dict(sl=1.5),'stop25':dict(sl=2.5),
 'fixedgap10':dict(gap=.1),'fixedgap15':dict(gap=.15),'fixedgap20':dict(gap=.2),'no_shock':dict(shock=False),'min_age3':dict(min_age=3)}
CONFIG=dict(declared_after_first_target_hit=True,first_hit='no_profit_cap30',do_not_reselect_from_neighbors=True,candidates=CANDIDATES,neighbors=NEIGHBORS,
 costs_stress=[50,100],cash_restarts=['2024-01-02','2025-10-06'],cost_bps=[10,30],
 target_is_historical_return_not_future_promise=True,no_observation_changes=True,data_certified=False,performance_publishable=False)
def main():
 P.mkdir(exist_ok=True)
 if (P/'CHECKS.json').exists():raise RuntimeError('Do not overwrite completed validation')
 if (P/'CONFIG.json').exists():assert json.loads((P/'CONFIG.json').read_text())==CONFIG
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n');protected=z.protected_hashes()
 oldp=z.t.P;z.t.P=P
 try:d,f,base,gaps,_=z.t.prepare()
 finally:z.t.P=oldp
 x=z.extras(d,f);summary=json.loads((P/'SUMMARY.json').read_text()) if (P/'SUMMARY.json').exists() else {}
 def perform(key,cfg,cost,start='2019-01-02'):
  if key in summary:return
  file=P/(key+'.json');out=json.loads(file.read_text()) if file.exists() else z.run(cfg,cost,d,f,base,gaps,x,start=start)
  assert out['status']=='complete';z.save(P,key,out,summary)
 for name,overlay in CANDIDATES.items():
  for cost in CONFIG['costs_stress']:perform(f'{name}_cost{cost}',{**z.BASE,**overlay},cost)
  for start,label in [('2024-01-02','fresh2024'),('2025-10-06','fresh252')]:
   for cost in [10,30]:perform(f'{label}_{name}_cost{cost}',{**z.BASE,**overlay},cost,start)
 for start,label in [('2024-01-02','fresh2024'),('2025-10-06','fresh252')]:
  for cost in [10,30]:perform(f'{label}_control_cost{cost}',z.BASE,cost,start)
 for name,overlay in NEIGHBORS.items():
  for cost in [10,30]:perform(f'{name}_cost{cost}',{**z.BASE,**CANDIDATES['no_profit_cap30'],**overlay},cost)
 # QQ Q comparison: all cash, adjusted open entry, 5/15bp buy fee only; no sell at end.
 benchmark={};a=d['adj']['QQQ'];c=d['close']['QQQ'];opening=d['opening']['QQQ'];adjopen=opening*a/c
 for start,label in [('2019-01-02','full'),('2024-01-02','fresh2024'),('2025-10-06','last252')]:
  for cost in [10,30]:
   nav=100000*a.loc[start:z.CONFIG['end']]/(adjopen.loc[start]*(1+cost/20000))
   rows=[dict(date=str(day.date()),equity=float(value),return_=0) for day,value in nav.items()]
   prev=100000
   for v in rows:v['return']=v['equity']/prev-1;prev=v['equity']
   benchmark[f'{label}_cost{cost}']=z.r.period_metrics(rows,start,z.CONFIG['end'])
 (P/'QQQ.json').write_text(json.dumps(benchmark,indent=2)+'\n')
 assert protected==z.protected_hashes()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,no_reselection=True,no_leverage=True),indent=2)+'\n')
 print('ROBUSTNESS_COMPLETE',len(summary),flush=True)
if __name__=='__main__':main()

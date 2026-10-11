from research import *
data,raw=load_inputs();models=candidates(data,raw);rows=[]
for name,(mask,score) in models.items():
 for rule in ['fixed3','ma5_max5']:
  for cost in [10,20]:
   d,t,o,u=simulate(data,mask,score,rule,cost)
   for split,(start,end) in {'all':('2019-01-01','2026-09-18'),'early':('2019-01-01','2021-12-31'),'middle':('2022-01-01','2023-12-31'),'late':('2024-01-01','2026-09-18')}.items():rows.append(dict(strategy=name,exit=rule,cost_bps=cost,split=split,**summarize(d,t,start,end)))
   pd.DataFrame(rows).to_csv(P/'exit_metrics.csv',index=False)
   print(name,rule,cost,rows[-4]['cagr'],flush=True)

"""Post-screen robustness only: neighbors4/6, retain5 as the declared candidate."""
import json,functools,hashlib
import research as r
P=r.P

def main():
 (P/'ROBUSTNESS_CONFIG.json').write_text(json.dumps(dict(reason='Check whether buffer5 improvement is a single-threshold accident; robustness introduced after initial screening',buffer_limits=[4,6],cost_bps=[10,30],candidate_remains=5,not_a_new_selection_round=True),indent=2)+'\n')
 m,d=r.r.prepare();d=r.r.apply(m,d,P/'pool');df,mf=r.r.features(d);dates=d['close'].index;ends=set(dates[:-1][dates[:-1].to_period('M')!=dates[1:].to_period('M')]);md=dict(d);md['opening']=d['close'];ff=dict(score=mf['base'],mask=mf['valid']&mf['positive'],atr=mf['atr'],ma5=mf['ma5']);ret=d['adj'].pct_change(fill_method=None);summary={}
 for lim in [4,6]:
  for cost in [10,30]:
   r.h.plan=r.monthly_plan('buffer5',ends,ret,df['event'],buffer_limit=lim);r.h.execute=functools.partial(r.net_execute,cost_bps=cost);ot=r.h.terminal_baskets.apply;os=r.h.spin_distributions.apply
   try:
    r.h.terminal_baskets.apply=functools.partial(ot,cost_bps=cost);r.h.spin_distributions.apply=functools.partial(os,cost_bps=cost)
    out=r.h.run_account('baseline',md,'2010-01-04',m['end'],{'baseline':ff})
   finally:r.h.terminal_baskets.apply=ot;r.h.spin_distributions.apply=os
   key=f'buffer{lim}_cost{cost}';(P/(key+'.json')).write_text(json.dumps(out,allow_nan=False)+'\n');summary[key]=r.metrics(out);print(key,json.dumps(summary[key]),flush=True)
   (P/'ROBUSTNESS.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()

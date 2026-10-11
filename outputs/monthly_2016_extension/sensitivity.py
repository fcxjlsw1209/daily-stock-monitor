"""No re-optimization: foreign-cap scenarios on frozen non-ML controls."""
from extend import *
import optimize as opt

def main():
 m,d,L,x,px0,op=load();px,raw,members,_,_,_=prices();f=read(P/'FROZEN.json');result={}
 for label,shell,ptr,china in [('shell_unscaled',1.,1.,True),('ptr_premium2',2.1,2.,True),('without_two_delisted',2.1,1.,False)]:
  pp,audit=early_pools(px,raw,pd.to_datetime(m['dates'][:53]).tolist(),shell,ptr,china);ss=[sorted(set(map(e.actual,pp[day]))|members['theme17'][1]) for day in m['dates'][:53]];em=dict(dates=m['dates'][:53],symbols=ss);ed,ex,el=features(px,raw,em,members);dm={k:np.concatenate([ed[k],d[k][53:]],axis=0) for k in d};xm={k:np.concatenate([ex[k],x[k][53:]],axis=0) for k in x};lm={k:dict(score=np.concatenate([el[k]['score'],L[k]['score'][53:]],axis=0),active=np.ones(122)) for k in el};mm=dict(m,symbols=ss+m['symbols'][53:]);out={}
  for name in ['anchor77','industry80','return_primary']:
   spec={**f['controls'],**f['selected']}[name]['spec'];z,orders,eq=opt.replay_model(label+'_'+name,spec,mm,dm,lm,xm,px,op,export=False);base=read(P/(name+'_ORDERS.json'));out[name]=dict(metrics=z['metrics'],early=z['slices']['early2016_2020'],changed_early_holdings=sum(a['symbols']!=b['symbols'] for a,b in zip(orders[:53],base[:53])))
  result[label]=dict(shell_multiplier=shell,ptr_multiplier=ptr,include_china=china,changed_early_pools=sum(set(ss[i])!=set(m['symbols'][i]) for i in range(53)),controls=out);print(label,{k:(round(v['metrics']['cagr'],4),v['changed_early_holdings']) for k,v in out.items()},flush=True)
 dump('POOL_SENSITIVITIES.json',result)

if __name__=='__main__':main()

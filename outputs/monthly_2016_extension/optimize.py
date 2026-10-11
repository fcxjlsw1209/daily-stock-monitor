"""Finite train/freeze/evaluate for the 2016 monthly extension."""
from extend import *

def specs():
 rows=[]
 for r in ref.single_specs():
  if r['spec']['addon'] not in ['none','anti_chase','volume_recovery','pressure_easing','shape_mix','ridge_10_12']:continue
  rows.append(dict(r,spec=dict(r['spec'],core_weight=.52,core252_blend=0.)))
 primary=read(O/'monthly_industry_refinement/FROZEN.json')['selected']['annual_primary']['spec']
 rows.append(dict(id=len(rows),family='previous82',spec=dict(primary,core_weight=.52,core252_blend=0.)))
 for seed in [ref.default(),primary]:
  for w,b in itertools.product([.52,.55,.60,.65],[0.,.25,.5]):
   rows.append(dict(id=len(rows),family='long_core',spec=dict(seed,core_weight=w,core252_blend=b)))
 seen=set();out=[]
 for row in rows:
  k=json.dumps(row['spec'],sort_keys=True)
  if k in seen:continue
  seen.add(k);out.append(dict(row,id=len(out)))
 return out

def score(s,d,L,x,n):
 ans=ref.score(s,d,L,x,n);c=d['core'][:n].copy()
 if s['core']!='original':c+=(.52*x['core_'+s['core']][:n]-c)*s['core_blend']
 newcore=(1-s['core252_blend'])*c/.52+s['core252_blend']*x['core252'][:n]
 return s['core_weight']*newcore+(ans-c)*(1-s['core_weight'])/.48

def decisions(s,m,d,L,x,n):
 sc=score(s,d,L,x,n);return rp.choose(sc,d['eligible'][:n],m['symbols'][:n],s['retain']),sc

def train():
 assert not (P/'FROZEN.json').exists();m,d,L,x,px,op=load();n=m['train_months'];b,dates,bounds,valid=rp.paths(px,m,n);rows=[]
 for row in specs():
  ch,sc=decisions(row['spec'],m,d,L,x,n);rr=rp.fast_returns(ch,b,valid);rob,met,seg=rp.robust(rr,bounds,m,n);rows.append(dict(row,training=met,robust=rob,segments=seg))
 base=rows[0];good=[z for z in rows if z['training']['sharpe']>=base['training']['sharpe']-.10 and z['training']['max_drawdown']>=base['training']['max_drawdown']-.05]
 selected={'return_primary':max(good,key=lambda z:(z['training']['cagr'],-z['id'])),'robust_control':max(good,key=lambda z:(z['robust'],-z['id']))}
 dump('TRAIN_TRIALS.json',rows)
 controls={'industry80':base,'anchor77':next(r for r in rows if r['family']=='old_anchor'),'prior82':next(r for r in rows if r['family']=='previous82')}
 f=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=selected,controls=controls,candidate_count=len(rows),admissible_count=len(good),training_return_end=m['ends'][n-1],code_hashes={str(f):sha(f) for f in [Path(__file__),P/'extend.py']},data_hashes={str(P/f):sha(P/f) for f in ['INPUTS.json','DATA.npz','FEATURES.npz','LAYERS.npz','PREDICTIONS.npz','PRICES.npz','POOLS.json','FORECAST_AUDIT.json']})
 dump('FROZEN.json',f);(P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
 for k,v in selected.items():print('FROZEN',k,v['spec'],v['training'],flush=True)
 print('Trials',len(rows),flush=True)

def cuts(eq,m):
 out={}
 for name,lo,hi in [('early2016_2020',m['dates'][0],'2020-12-31'),('original2021_2026','2020-12-31',m['ends'][-1]),('training2016_2023',m['dates'][0],'2023-11-30'),('later34','2023-11-30',m['ends'][-1])]:
  z=eq.loc[lo:hi];out[name]=e.engine.stats(z/z.iloc[0]*100000.)
 return out

def replay_model(name,spec,m,d,L,x,px,op,export=True):
 ch,sc=decisions(spec,m,d,L,x,len(m['dates']));orders=rp.make_orders(ch,sc,m);res,eq,tr,holds=e.engine.replay(px,op,orders,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]));b,dd,bounds,valid=rp.paths(px,m,len(ch));rr=rp.fast_returns(ch,b,valid)
 np.testing.assert_allclose(eq.to_numpy(),np.r_[1.,np.cumprod(1+rr)]*100000.,rtol=1e-11,atol=1e-5)
 result=dict(metrics=res,slices=cuts(eq,m),spec=spec)
 if export:
  dump(name+'_ORDERS.json',orders);dump(name+'_TRADES.json',tr);dump(name+'_HOLDINGS.json',holds);dump(name+'_DAILY.json',[dict(date=str(day.date()),equity=float(v)) for day,v in eq.items()])
 return result,orders,eq

def evaluate():
 f=read(P/'FROZEN.json');assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
 for file,h in {**f['code_hashes'],**f['data_hashes'],**read(P/'SOURCE_HASHES.json')}.items():assert sha(file)==h,file
 m,d,L,x,px,op=load();result={};prior=read(O/'monthly_industry_refinement/RESULTS.json')
 for name,row in {**f['controls'],**f['selected']}.items():
  z,orders,eq=replay_model(name,row['spec'],m,d,L,x,px,op);result[name]=z
  assert abs(z['slices']['training2016_2023']['cagr']-row['training']['cagr'])<1e-10
  if name in f['controls']:
   prev=prior[{'industry80':'industry_reference','anchor77':'original_anchor','prior82':'annual_primary'}[name]]
   for key in ['cagr','sharpe','max_drawdown']:assert abs(z['slices']['original2021_2026'][key]-prev['metrics'][key])<1e-10,(name,key)
   expected=read(O/'monthly_industry_refinement'/({'industry80':'industry_reference','anchor77':'original_anchor','prior82':'annual_primary'}[name]+'_ORDERS.json'))
   assert [o['symbols'] for o in orders[53:]]==[o['symbols'] for o in expected]
  for tag,mode,cost in [('next_open_zero','open',0.),('next_open_cost','open',.0005)]:
   mm,ee,_,_=e.engine.replay(px,op,orders,cost=cost,execution=mode,start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]));z[tag]=dict(metrics=mm,slices=cuts(ee,m))
  print(name,dict(full=z['metrics']['cagr'],sharpe=z['metrics']['sharpe'],early=z['slices']['early2016_2020']['cagr'],recent=z['slices']['original2021_2026']['cagr'],later=z['slices']['later34']['cagr']),flush=True)
 q=px.loc[m['dates'][0]:m['ends'][-1],'QQQ'];qe=q/q.iloc[0]*100000.;result['qqq']=dict(metrics=e.engine.stats(qe),slices=cuts(qe,m));dump('RESULTS.json',result)
 dump('VALIDATION.json',dict(exact69_baseline_holdings_and_metrics=True,independent_monthly_path_vs_daily_ledger=True,source_hashes=True,annual_forecast_no_future_labels=all(z['latest_label_end']<=z['cutoff'] for z in read(P/'FORECAST_AUDIT.json')),candidate_count=f['candidate_count'],author_labels_used=False,pristine_oos=False,protected_changes=[file for file,h in read(P/'PROTECTED_BEFORE.json').items() if sha(file)!=h]))
 assert not read(P/'VALIDATION.json')['protected_changes']

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['train','evaluate']);globals()[ap.parse_args().phase]()

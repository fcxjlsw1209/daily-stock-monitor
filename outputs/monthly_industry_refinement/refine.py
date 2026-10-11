"""Finite monthly guidance refinement. Training35 only selects parameters."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,argparse,itertools
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_guidance_tweaks'));import tweaks as t
rp,e=t.rp,t.e
read,sha=t.read,t.sha

def dump(n,x):
 (P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda z:z.item() if isinstance(z,np.generic) else z)+'\n')

def initialize():
 assert not (P/'PLAN.json').exists()
 dump('PLAN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),reference='Frozen industry candidate80.8349%CAGR. Use identical monthly Top32 archive/17ETF pool and gate. Anchor selected from reused history; no pristine OOS claim.',
 hypothesis='Refine risk denominator of skipped-two-month long momentum; distinguish upside/idiosyncratic volatility; multi-horizon stock-versus-own-sector relative momentum; tilt industry correction towards stability/tail resilience; small price-volume or ML layer; market-conditioned activation and monthend turnover buffer.',
 hard_rules='52%long-core category, numerator skips42sessions, at most25%ofcore changes definition. Other48% redistributed. No daily pullback gate or intramonth exits. Max3equal positions;2half;<2cash. Zero-cost same-close primary; independent next-open checks.',
 finite_search='Single-family bounded changes, then pairs of up to2training-selected architectures per family. No holdout-driven grid expansion. Core alternatives half-power/aligned252/downside63 blend10%or25%; support volatility replacements25%or50%; sector126/252blend; industry budgets10/15/20% and weights; global/weak/highvol/union state; addon2.5/5%; retention3/5.',
 selection='First35months only. Require trainSharpe>=industry-.10, maxDD>=industry-.05 and at least2of3trainingblocks not worse. Freeze highesttrainCAGR primary and highestrobustscore control. Also report prefix-only walkforward12->18,18->24,24->35. Later34 evaluated once after freeze.',
 protection='No change to any registered monthly/daily observer, frozen prior study or source input. No author holding targets in search. Annual prior Ridge forecasts are past-five-year mature-label only.',
 limitations='Archive universe/cap/sector proxies and assumedETF17; all history previously explored. Search not proof of future alpha. Event-calendar tweaks omitted absent exact reliable definitions.'))
 files={str(f):sha(f) for folder in ['monthly_industry_monitor','monthly_guidance_tweaks','monthly_regime_monitor','monthly_buffer5_monitor','daily_stock_monitor','pressure_shadow_monitor','resilience_shadow_monitor','accel_shadow_monitor','high52_shadow_monitor','accel40_shadow_monitor'] for f in (O/folder).rglob('*') if f.is_file() and f.suffix!='.pyc' and f.name!='run.lock'}
 dump('PROTECTED_BEFORE.json',files)

def prepare():
 m,d,layers,px0,op=t.load();px,raw,members,periods,sources,_=e.load_prices();rs=px.pct_change(fill_method=None);vol=rs.rolling(63).std();long=px.shift(42)/px.shift(294)-1
 down=np.sqrt(rs.clip(upper=0).pow(2).rolling(63).mean());beta=rs.rolling(126).cov(rs.SPY).div(rs.SPY.rolling(126).var(),axis=0).shift(1);residual=rs-beta.mul(rs.SPY,axis=0)
 feat={'core_half':long/np.sqrt(vol.replace(0,np.nan)),'core_aligned':long/rs.rolling(252).std().shift(42).replace(0,np.nan),'core_downside':long/down.replace(0,np.nan),'vol126':rs.rolling(126).std(),'upside126':np.sqrt(rs.clip(lower=0).pow(2).rolling(126).mean()),'residual126':residual.rolling(126).std(),'upside_balance':np.sqrt(rs.clip(lower=0).pow(2).rolling(126).mean())-np.sqrt(rs.clip(upper=0).pow(2).rolling(126).mean())}
 sectors=pd.read_csv(O/'index_union/alias_resolved_inputs/sectors.csv.gz',index_col=0,parse_dates=True,low_memory=False);mom126=px.shift(42)/px.shift(126)-1
 extra={k:np.zeros((69,49)) for k in list(feat)+['sector126']}
 for i,(day,ss) in enumerate(zip(m['dates'],m['symbols'])):
  for k,v in feat.items():extra[k][i]=t.ranks(v.loc[day,ss].replace([np.inf,-np.inf],np.nan))
  bm=t.strict.benchmark_map(day,ss,sectors,members['theme17'][1]);extra['sector126'][i]=t.ranks(t.strict.relative_row(mom126,day,ss,bm))
 sh=np.load(O/'monthly_guidance_tweaks/SHAPES.npz');extra['sector252']=sh['sector_relative'];extra['stability']=sh['sector_stability'];extra['tail']=sh['tail_resilience']
 np.savez_compressed(P/'FEATURES.npz',**extra)
 paths=[O/'monthly_guidance_tweaks'/n for n in ['INPUTS.json','DATA.npz','SHAPES.npz','PRICES.npz','PREDICTIONS.npz','tweaks.py']]+[Path(rp.__file__),Path(__file__)]
 sources.update({str(f):sha(f) for f in paths});dump('SOURCE_HASHES.json',sources);print('New skipped-momentum risk and sector/volatility features prepared',flush=True)

def load():
 m,d,layers,px,op=t.load();return m,d,layers,dict(np.load(P/'FEATURES.npz')),px,op

def default():return dict(core='original',core_blend=0.,vol='vol126',vol_blend=0.,horizon='252',industry_weights=[.5,.25,.25],budget=.15,mode='highvol',addon='none',addon_budget=0.,retain=3)

def score(spec,d,layers,x,n):
 core=d['core'][:n].copy();support=d['support'][:n].copy()
 if spec['core']!='original':core+=(.52*x['core_'+spec['core']][:n]-core)*spec['core_blend']
 support+=.219375*spec['vol_blend']*(x[spec['vol']][:n]-x['vol126'][:n])
 a,b,c=spec['industry_weights'];rel=x['sector252'] if spec['horizon']=='252' else x['sector126'] if spec['horizon']=='126' else .5*x['sector252']+.5*x['sector126']
 correction=a*rel[:n]+b*x['stability'][:n]+c*x['tail'][:n]
 mode=spec['mode'];active=d['high'][:n] if mode=='highvol' else d['weak'][:n] if mode=='weak' else d['high'][:n]|d['weak'][:n] if mode=='union' else np.ones(n,bool)
 budget=spec['budget']*active.astype(float);addon=np.zeros_like(core);ab=np.zeros(n)
 if spec['addon']!='none':
  layer=layers[spec['addon']];ab=spec['addon_budget']*active*layer['active'][:n];addon=ab[:,None]*layer['score'][:n]
 assert np.max(budget+ab)<=.25+1e-12
 return core+(1-(budget+ab)[:,None]/.48)*support+budget[:,None]*correction+addon

def single_specs():
 arr=[]
 def add(family,**kw):s=default();s.update(kw);arr.append(dict(id=len(arr),family=family,spec=s))
 add('control');add('old_anchor',budget=0.)
 for k,b in itertools.product(['half','aligned','downside'],[.10,.25]):add('core',core=k,core_blend=b)
 for k,b in itertools.product(['upside126','residual126','upside_balance'],[.25,.50]):add('volatility',vol=k,vol_blend=b)
 for h,w,b in itertools.product(['252','126','blend'],[[.5,.25,.25],[.25,.25,.5],[.25,.5,.25],[.75,0.,.25],[.5,0.,.5]],[.10,.15,.20]):add('industry',horizon=h,industry_weights=w,budget=b)
 for mode,b in itertools.product(['weak','union','global'],[.10,.15,.20]):add('regime',mode=mode,budget=b)
 for k,b in itertools.product(['anti_chase','volume_recovery','pressure_easing','shape_mix','ridge_10_0','ridge_10_12','weak_tree_2'],[.025,.05]):add('addon',addon=k,addon_budget=b)
 add('retention',retain=5)
 # Deduplicate exact specs, keeping original control first.
 seen=set();out=[]
 for row in arr:
  key=json.dumps(row['spec'],sort_keys=True)
  if key not in seen:row['id']=len(out);out.append(row);seen.add(key)
 return out

def measure(row,m,d,layers,x,blocks,valid,bounds):
 sc=score(row['spec'],d,layers,x,35);ch=rp.choose(sc,d['eligible'][:35],m['symbols'][:35],row['spec']['retain']);rr=rp.fast_returns(ch,blocks,valid);rob,met,seg=rp.robust(rr,bounds,m,35)
 return dict(row,robust=rob,training=met,segments=seg),rr

def allowed(row,base):
 a,b=row['training'],base['training'];return a['sharpe']>=b['sharpe']-.10 and a['max_drawdown']>=b['max_drawdown']-.05 and sum(x['total_return']>=y['total_return']-1e-12 for x,y in zip(row['segments'],base['segments']))>=2

def train():
 assert not (P/'FROZEN.json').exists();m,d,l,x,px,op=load();blocks,dates,bounds,valid=rp.paths(px,m,35);rows=[];returns=[]
 for row in single_specs():
  z,rr=measure(row,m,d,l,x,blocks,valid,bounds);rows.append(z);returns.append(rr)
 base=rows[0];seeds=[]
 for family in ['core','volatility','industry','regime','addon','retention']:
  subset=[r for r in rows if r['family']==family and allowed(r,base)]
  seeds.extend(sorted(subset,key=lambda r:r['training']['cagr'],reverse=True)[:2])
 seen={json.dumps(r['spec'],sort_keys=True) for r in rows};original=default()
 for a,b in itertools.combinations(seeds,2):
  if a['family']==b['family']:continue
  da={k:v for k,v in a['spec'].items() if v!=original[k]};db={k:v for k,v in b['spec'].items() if v!=original[k]}
  if any(da[k]!=db[k] for k in set(da)&set(db)):continue
  spec=dict(original,**da);spec.update(db)
  if spec['budget']+spec['addon_budget']>.25:continue
  key=json.dumps(spec,sort_keys=True)
  if key in seen:continue
  seen.add(key);row=dict(id=len(rows),family='pair',parents=[a['id'],b['id']],spec=spec);z,rr=measure(row,m,d,l,x,blocks,valid,bounds);rows.append(z);returns.append(rr)
 good=[r for r in rows if allowed(r,base)];key=lambda r:(r['training']['cagr'],r['training']['sharpe'],-r['id'])
 winners={'annual_primary':max(good,key=key),'robust_control':max(good,key=lambda r:(r['robust'],-r['id']))}
 # Prefix-only walkforward is restricted to predefined single-family choices:
 # pair seeds were picked using the entire35 and must NOT leak into early folds.
 fixed_count=len(single_specs());wf=[];stitched=[]
 for n,end in [(12,18),(18,24),(24,35)]:
  j=max(range(fixed_count),key=lambda i:rp.robust(returns[i],bounds,m,n)[0]);part=returns[j][bounds[n]:bounds[end]];stitched.extend(part)
  wf.append(dict(training_months=n,chosen=rows[j]['spec'],test_start=m['dates'][n],test_end=m['ends'][end-1],metrics=rp.stats_rr(part,m['dates'][n],m['ends'][end-1])))
 dump('TRAIN_TRIALS.json',rows);dump('TRAIN_WALKFORWARD.json',dict(folds=wf,stitched=rp.stats_rr(np.array(stitched),m['dates'][12],m['ends'][34]),pair_seeds_excluded=True))
 f=dict(created_utc=datetime.now(timezone.utc).isoformat(),candidate_count=len(rows),single_count=fixed_count,admissible_count=len(good),primary='annual_primary',selected=winners,base=base,old=rows[1],code_sha256=sha(__file__),feature_sha256=sha(P/'FEATURES.npz'),source_hashes=read(P/'SOURCE_HASHES.json'),author_labels_used=False)
 dump('FROZEN.json',f);(P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
 for k,r in winners.items():print('Frozen',k,r,flush=True)
 print('Total',len(rows),'admissible',len(good),'walkforward',rp.stats_rr(np.array(stitched),m['dates'][12],m['ends'][34]),flush=True)

def evaluate():
 f=read(P/'FROZEN.json');assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip();assert f['code_sha256']==sha(__file__);assert f['feature_sha256']==sha(P/'FEATURES.npz');assert all(sha(p)==h for p,h in f['source_hashes'].items())
 m,d,l,x,px,op=load();result={};models=dict(f['selected'],industry_reference=f['base'],original_anchor=f['old'])
 for name,row in models.items():
  sc=score(row['spec'],d,l,x,69);ch=rp.choose(sc,d['eligible'],m['symbols'],row['spec']['retain']);orders=rp.make_orders(ch,sc,m);met,eq=rp.replay(name,orders,px,op,m,export=False)
  assert abs(row['training']['cagr']-met['slices']['training35']['cagr'])<1e-10
  result[name]=dict(met,spec=row['spec']);dump(name+'_ORDERS.json',orders);dump(name+'_DAILY.json',[dict(date=str(day.date()),equity=float(v)) for day,v in eq.items()])
  if name in f['selected']:
   for key,mode,cost in [('next_open_zero','open',0.),('next_open_cost','open',.0005)]:result[name][key]=rp.replay(name,orders,px,op,m,cost,mode,export=False)[0]
   _,_,tr,holds=e.engine.replay(px,op,orders,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]));dump(name+'_TRADES.json',tr);dump(name+'_HOLDINGS.json',holds)
  print(name,'full',met['metrics']['cagr'],met['metrics']['sharpe'],'later',met['slices']['validation34']['cagr'],met['slices']['validation34']['sharpe'],flush=True)
 prior=read(O/'monthly_guidance_tweaks/RESULTS.json')
 for a,b in [('industry_reference','category_industry'),('original_anchor','anchor')]:
  for k in ['cagr','sharpe','max_drawdown']:assert abs(result[a]['metrics'][k]-prior[b]['metrics'][k])<1e-10
 result['author']=prior['author'];dump('RESULTS.json',result)
 changes=[p for p,h in read(P/'PROTECTED_BEFORE.json').items() if sha(p)!=h];assert not changes,changes
 dump('VALIDATION.json',dict(protected_changes=[],source_hashes_verified=True,frozen_hash=sha(P/'FROZEN.json'),independent_fixed_units_replay=True,reference_exact_parity=True,train_replay_parity=True,author_labels_used=False,pristine_oos=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','prepare','train','evaluate']);globals()[ap.parse_args().phase]()

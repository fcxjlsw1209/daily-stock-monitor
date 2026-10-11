"""Return-focused micro-adjustments of the existing monthly momentum anchor."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,itertools,sys
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_return_priority'))
import return_priority as rp
e,read,sha=rp.e,rp.read,rp.sha
sys.path.insert(0,str(O/'monthly_decision_audit'))
import audit as anchor
sys.path.insert(0,str(O/'monthly_pattern_recheck'))
import patterns
strict=rp.prior.old.strict


def dump(n,z):
 (P/n).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def initialize():
 assert not (P/'PLAN.json').exists()
 dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),objective='Improve monthly CAGR subject to training Sharpe/drawdown guards. Author holdings/order are diagnostics only.',
 anchor='Preserve exactly previous77.6221%CAGR candidate monthly reconstructedTop32+theme17 universe, positive252skip42 momentum and295-session warmup, normalization and52%core. Verify zero-tweak69month holdings parity before any search.',
 factors='Author-guided anti-chase/gaps, shakeout recovery/dry consolidation/failed rebounds, tail resilience, stock-vs-own-sector strength, sector-link stability, down-volume pressure easing, annual ordinary/recent Ridge and weak-market GBDT, monthend own-position profit penalties.',
 ML='Reuse cached102-feature weekly dataset. Prior5years,21session excess return clipped[-1,1], only label_end<=annual previousDecember cutoff. Ordinary/recent Ridge alpha1/10/100; GBDT depth2/3,100iterations,.05rate,minleaf100,L2=1,seed1729,early_stoppingFalse. Annual method retrains on then-past prices, no author labels.',
 small_budget='Core contribution .52 remains exactly unchanged. Correction budget0/2.5/5/7.5/10/15% taken proportionally from original48%support. Global/weak/highvol/low-score-conviction activation. Monthend retention3/5. Own-profit penalty0/2.5/5points, bounded tanh(log episode gain / monthlyvol) for own held names only.',
 grid='StageA single signals plus fixed multi-layer bundles,5positive budgets x4activation x2retention plus zero controls;StageB top8 distinct training-only architectures x3profit penalties. Finite grid, no subsequent holdout-directed expansion.',
 split='First35 holding months Jan2021-Nov2023 selection; later34 after freeze. Entire history was viewed in prior work: not untouched OOS. Original anchor itself selected by historical research.',
 primary='Highest trainingCAGR with Sharpe>=anchor-.10, drawdown>=anchor-.05, and at least2of3 training12/12/11month blocks outperform anchor. Baseline always eligible. Secondary highestSharpe and robust(Sharpe+.25loggrowth-.5DD). Tie lowerbudget then simpler configuration.',
 execution='Long-only, max3 equally weighted at monthend,2half,<2cash; fixed intramonth units, no intramonth stops. Primaryzero-cost same-close idealization; next-open/cost sensitivities. No live account, ledger, rules, automation or historical source modifications.',
 calendar='FOMC/election/holiday layers excluded this round: no verified event dataset or author specification. Do not guess dates or imply these daily hints necessarily belong to monthly strategy.'))
 files=read(O/'monthly_return_priority/PROTECTED_BEFORE.json');files.update({str(f):sha(f) for f in (O/'monthly_return_priority').iterdir() if f.is_file()})
 dump('PROTECTED_BEFORE.json',{f:sha(f) for f in files})

def ranks(values,eligible=None):
 s=pd.Series(values)
 if eligible is not None:s=s.where(eligible)
 return s.rank(pct=True).fillna(.5).to_numpy()-.5

def prepare():
 px,raw,members,periods,sources,_=e.load_prices();audit=read(O/'monthly_decision_audit/MONTHS.json')
 symbols=[sorted(e.actual(r['symbol']) for r in z['rows'] if r['in_pool']) for z in audit];dates=[z['date'] for z in audit];ends=[z['end'] for z in audit];N,M=69,49
 core=np.zeros((N,M));support=core.copy();good=np.zeros((N,M),bool);vol=core.copy();price=core.copy();missing={}
 for i,(z,ss) in enumerate(zip(audit,symbols)):
  rows={e.actual(x['symbol']):x for x in z['rows']}
  for j,t in enumerate(ss):
   q=rows[t];good[i,j]=q['eligible'];core[i,j]=q['contributions']['momentum']/100;support[i,j]=(q['score']-50)/100-core[i,j];vol[i,j]=q['raw']['vol63'] or 0.
   price[i,j]=px.at[z['date'],t]
 ch=rp.choose(core+support,good,symbols,3)
 assert [[e.display(symbols[i][j]) for j in ids] for i,ids in enumerate(ch)]==[z['ours'] for z in audit]
 feat=patterns.shape_only(px,raw);rs=px.pct_change(fill_method=None);sectorsfile=O/'index_union/alias_resolved_inputs/sectors.csv.gz';sectors=pd.read_csv(sectorsfile,index_col=0,parse_dates=True,low_memory=False)
 mom=px.shift(42)/px.shift(252)-1
 # Compare the stock with its own sector ETF. Unknown sector and ETFs are neutral.
 correlations={b:(rs.rolling(63).corr(rs[b]),rs.rolling(252).corr(rs[b])) for b in strict.SECTOR.values()}
 down=(raw['volume']*rs.clip(upper=0).abs());pressure=down.rolling(5).mean()/raw['volume'].rolling(5).mean()-down.rolling(21).mean()/raw['volume'].rolling(21).mean();feat['pressure_easing']=-pressure
 names=list(feat)+['sector_relative','sector_stability'];Q={k:np.zeros((N,M)) for k in names};missing={k:0 for k in names}
 for i,(date,ss) in enumerate(zip(dates,symbols)):
  bm=strict.benchmark_map(date,ss,sectors,members['theme17'][1]);tab=pd.DataFrame({k:v.loc[date,ss] for k,v in feat.items()})
  tab['sector_relative']=strict.relative_row(mom,date,ss,bm)
  tab['sector_stability']=[-abs(correlations[b][0].at[pd.Timestamp(date),t]-correlations[b][1].at[pd.Timestamp(date),t]) if b is not None else np.nan for t,b in bm.items()]
  for k in names:missing[k]+=int(tab[k].isna().sum());Q[k][i]=ranks(tab[k])
 weak=np.array([not z['state']['spy_above200'] for z in audit]);high=np.array([z['state']['high_vol'] for z in audit])
 layers={'anti_chase':(-Q['rebound21_atr']-Q['positive_gap5_vol'])/2,'volume_recovery':(Q['shakeout_recovery5']+Q['dry_consolidation']-Q['failed_rebounds5'])/3,'tail_resilience':Q['market_tail_residual252'],'pressure_easing':Q['pressure_easing'],'sector_relative':Q['sector_relative'],'sector_stability':Q['sector_stability']}
 layers['industry_mix']=.5*layers['sector_relative']+.25*layers['sector_stability']+.25*layers['tail_resilience']
 layers['shape_mix']=.3*layers['anti_chase']+.4*layers['volume_recovery']+.3*layers['pressure_easing']
 dump('INPUTS.json',dict(dates=dates,ends=ends,symbols=symbols,etfs=sorted(members['theme17'][1]),base='original auditedTop32/17ETF,exact69monthparity',split=35,missing_shape_values_neutral=missing))
 np.savez_compressed(P/'DATA.npz',core=core,support=support,eligible=good,vol=vol,price=price,weak=weak,high=high)
 np.savez_compressed(P/'SHAPES.npz',**layers)
 keep=sorted(set(t for ss in symbols for t in ss)|{'QQQ','SPY'});op=raw['opening']*px/raw['close']
 np.savez_compressed(P/'PRICES.npz',close=px[keep].to_numpy(),opening=op[keep].to_numpy(),dates=px.index.strftime('%Y-%m-%d').to_numpy(dtype='U10'),symbols=np.array(keep))
 for f in [O/'monthly_decision_audit/MONTHS.json',sectorsfile,Path(patterns.research.__file__),O/'monthly_causal_ridge/MARKET_TRAINING.npz',O/'monthly_causal_ridge/MARKET_TRAINING.json',O/'monthly_causal_ridge/FORECAST_AUDIT.json',O/'monthly_adaptive_core/INPUTS.json',Path(rp.__file__)]:sources[str(f)]=sha(f)
 dump('SOURCE_HASHES.json',sources);print('Prepared exact77.62% anchor; shape layers',list(layers),flush=True)

def mature(dates,ends,cutoff):
 start=str((pd.Timestamp(cutoff)-pd.DateOffset(years=5)).date())
 return [i for i,(date,end) in enumerate(zip(dates,ends)) if start<=date<=cutoff and end<=cutoff]

def forecast():
 from sklearn.ensemble import HistGradientBoostingRegressor
 import joblib
 meta=read(P/'INPUTS.json');D=dict(np.load(P/'DATA.npz'));root=O/'monthly_causal_ridge';fm=read(root/'MARKET_TRAINING.json');a=np.load(root/'MARKET_TRAINING.npz');spec=read(O/'monthly_adaptive_core/INPUTS.json')['packs']['top32_percentile'];audit=read(root/'FORECAST_AUDIT.json')
 targetX=np.empty_like(a['top32'])
 for i,ss in enumerate(meta['symbols']):
  source=spec['symbols'][i];assert set(ss)==set(source);targetX[i]=a['top32'][i,[source.index(t) for t in ss]]
 preds={f'ridge_{alpha:g}_{half}':np.zeros(D['core'].shape) for alpha,half in itertools.product([1.,10.,100.],[0,12])};preds.update({f'weak_tree_{depth}':np.zeros(D['core'].shape) for depth in [2,3]});checks=[];models=P/'models';models.mkdir(exist_ok=True)
 for year in range(2021,2027):
  ids=[i for i,end in enumerate(meta['ends']) if pd.Timestamp(end).year==year];cutoff=meta['dates'][min(ids)];ii=mature(fm['dates'],fm['ends'],cutoff);XX=[];Y=[]
  for i in ii:
   valid=a['eligible'][i];XX.append(a['X'][i,valid]);Y.extend(a['Y'][i,valid])
  X=np.vstack(XX);y=np.array(Y);latest=max(fm['ends'][i] for i in ii);assert latest<=cutoff
  for entry in [r for r in audit if r['year']==year]:
   assert entry['cutoff']==cutoff and entry['latest_label_end']<=cutoff
   cfg=entry['config'];key=f'ridge_{cfg["alpha"]:g}_{cfg["half_life"]}';p=np.einsum('ijk,k->ij',targetX[ids],np.array(entry['coef']),optimize=False)+entry['intercept']
   for j,i in enumerate(ids):preds[key][i]=ranks(p[j],D['eligible'][i])
  for depth in [2,3]:
   model=HistGradientBoostingRegressor(max_depth=depth,max_leaf_nodes=2**depth,max_iter=100,learning_rate=.05,min_samples_leaf=100,l2_regularization=1.,early_stopping=False,random_state=1729).fit(X,y)
   pp=model.predict(targetX[ids].reshape(-1,targetX.shape[-1])).reshape(len(ids),49)
   for j,i in enumerate(ids):preds[f'weak_tree_{depth}'][i]=ranks(pp[j],D['eligible'][i])
   path=models/f'{year}_depth{depth}.joblib';joblib.dump(model,path)
   checks.append(dict(kind='tree',year=year,depth=depth,cutoff=cutoff,latest_label_end=latest,examples=len(X),model=str(path),sha256=sha(path)))
  checks.extend([dict(kind='ridge',year=z['year'],cutoff=z['cutoff'],latest_label_end=z['latest_label_end'],config=z['config'],examples=z['rows']) for z in audit if z['year']==year])
  print('Causal forecasts',year,'labels through',latest,'examples',len(X),flush=True)
 np.savez_compressed(P/'PREDICTIONS.npz',**preds);dump('FORECAST_AUDIT.json',checks)
 dump('PREPARED_HASHES.json',{n:sha(P/n) for n in ['INPUTS.json','DATA.npz','SHAPES.npz','PRICES.npz','PREDICTIONS.npz','FORECAST_AUDIT.json']})

def load():
 m=read(P/'INPUTS.json');d=dict(np.load(P/'DATA.npz'));sh=dict(np.load(P/'SHAPES.npz'));pred=dict(np.load(P/'PREDICTIONS.npz'));a=np.load(P/'PRICES.npz');idx=pd.to_datetime(a['dates']);ss=a['symbols'];px=pd.DataFrame(a['close'],index=idx,columns=ss);op=pd.DataFrame(a['opening'],index=idx,columns=ss)
 layers={k:dict(score=v,active=np.ones(69)) for k,v in sh.items()}
 for k,v in pred.items():layers[k]=dict(score=v,active=d['weak'].astype(float) if k.startswith('weak_tree') else np.ones(69))
 for alpha in [1,10,100]:
  layers[f'ridge_mix_{alpha}']=dict(score=.7*pred[f'ridge_{alpha}_0']+.3*pred[f'ridge_{alpha}_12'],active=np.ones(69))
 # Component budgets not active in strong markets return to the anchor support.
 weak=d['weak'][:,None].astype(float);eff=.9+.1*weak
 combined=.35*pred['ridge_10_0']+.15*pred['ridge_10_12']+.1*weak*pred['weak_tree_2']+.2*sh['shape_mix']+.2*sh['industry_mix']
 layers['author_bundle']=dict(score=combined/eff,active=eff[:,0])
 return m,d,layers,px,op

def activate(spec,d,count,layer):
 mask=np.ones(count)
 if spec['mode']=='weak':mask=d['weak'][:count].astype(float)
 if spec['mode']=='highvol':mask=d['high'][:count].astype(float)
 if spec['mode']=='uncertain':
  base=np.where(d['eligible'][:count],d['core'][:count]+d['support'][:count],-np.inf);rank=np.sort(base,axis=1)[:,-3:];mask=(rank[:,-1]-rank[:,0]<.10).astype(float)
 return spec['budget']*mask*layer['active'][:count]

def decisions(spec,m,d,layers,count=69,details=False):
 layer=layers[spec['layer']];b=activate(spec,d,count,layer);sc=d['core'][:count]+(1-b[:,None]/.48)*d['support'][:count]+b[:,None]*layer['score'][:count]
 previous=[];entry={};chosen=[];events=[]
 for i,ss in enumerate(m['symbols'][:count]):
  penalty=np.zeros(len(ss))
  for j,t in enumerate(ss):
   if t in previous and t in entry:
    pnl=np.log(d['price'][i,j]/entry[t]);penalty[j]=spec['profit']*max(0.,np.tanh(pnl/max(d['vol'][i,j]*np.sqrt(21),.01)))
  sc[i]-=penalty;rank=sorted(np.flatnonzero(d['eligible'][i]),key=lambda j:(-sc[i,j],ss[j]));positions={ss[j]:k+1 for k,j in enumerate(rank)}
  kept=sorted([t for t in previous if positions.get(t,999)<=spec['retain']],key=lambda t:positions[t])[:3]
  new=kept+[ss[j] for j in rank if ss[j] not in kept][:3-len(kept)]
  if len(new)<2:new=[]
  new=sorted(new,key=lambda t:positions[t]);entry={t:entry[t] if t in entry and t in previous else float(d['price'][i,ss.index(t)]) for t in new}
  ids=[ss.index(t) for t in new];chosen.append(ids)
  if details:events.append(dict(date=m['dates'][i],budget=float(b[i]),previous=list(map(e.display,previous)),selected=list(map(e.display,new)),candidates=[dict(ticker=e.display(ss[j]),rank=k+1,core=float(d['core'][i,j]),base_support=float(d['support'][i,j]),correction=float(b[i]*layer['score'][i,j]),profit_penalty=float(penalty[j]),score=float(sc[i,j])) for k,j in enumerate(rank)]))
  previous=new
 return chosen,sc,events

def scan_specs(layers):
 ans=[]
 def add(**kw):ans.append(dict(id=len(ans),profit=0.,**kw))
 for retain in [3,5]:add(layer='anti_chase',budget=0.,mode='global',retain=retain)
 for name,budget,mode,retain in itertools.product(layers,[.025,.05,.075,.10,.15],['global','weak','highvol','uncertain'],[3,5]):add(layer=name,budget=budget,mode=mode,retain=retain)
 return ans

def metric(ch,blocks,valid,bounds,m):
 rr=rp.fast_returns(ch,blocks,valid);rob,full,segments=rp.robust(rr,bounds,m,35);return dict(training=full,segments=segments,robust=rob)

def permitted(row,base):
 a,b=row['training'],base['training'];wins=sum(x['total_return']>=y['total_return']-1e-12 for x,y in zip(row['segments'],base['segments']))
 return a['sharpe']>=b['sharpe']-.10 and a['max_drawdown']>=b['max_drawdown']-.05 and wins>=2

def key(row,kind='annual'):
 z=row['training'];s=row['spec'];primary=z['cagr'] if kind=='annual' else z['sharpe'] if kind=='sharpe' else row['robust']
 return primary,-s['budget'],-s['profit'],-s['retain'],-s['id']

def train():
 assert not (P/'FROZEN.json').exists()
 m,d,layers,px,op=load();blocks,_,bounds,valid=rp.paths(px,m,35);rows=[]
 for spec in scan_specs(layers):
  ch,_,_=decisions(spec,m,d,layers,35);row=dict(spec=spec,**metric(ch,blocks,valid,bounds,m));rows.append(row)
  if len(rows)%200==0:print('Training micro-rules',len(rows),flush=True)
 base=rows[0];assert base['spec']['budget']==0 and base['spec']['retain']==3
 seeds=[];seen=set()
 for row in sorted(rows,key=lambda z:key(z),reverse=True):
  if not permitted(row,base):continue
  k=(row['spec']['layer'],row['spec']['mode'],row['spec']['retain'])
  if k in seen:continue
  seen.add(k);seeds.append(row)
  if len(seeds)==8:break
 for seed in seeds:
  for penalty in [.025,.05]:
   spec=dict(seed['spec'],id=len(rows),profit=penalty);ch,_,_=decisions(spec,m,d,layers,35);rows.append(dict(spec=spec,**metric(ch,blocks,valid,bounds,m)))
 admissible=[z for z in rows if permitted(z,base)];leaders={name:max(admissible,key=lambda z:key(z,kind)) for name,kind in [('annual_primary','annual'),('sharpe_control','sharpe'),('robust_control','robust')]}
 # Category best by training only; diagnostics are all frozen before holdout.
 cats={}
 for cat in ['shape','industry','ml','profit']:
  subset=[z for z in admissible if z['spec']['budget']>0 and (z['spec']['profit']>0 if cat=='profit' else z['spec']['layer'].startswith(('ridge','weak_tree','author_bundle')) if cat=='ml' else z['spec']['layer'].startswith(('sector','industry')) if cat=='industry' else z['spec']['layer'] in ['anti_chase','volume_recovery','pressure_easing','tail_resilience','shape_mix'])]
  if subset:cats[cat]=max(subset,key=key)
 dump('TRAIN_TRIALS.json',rows);dump('STAGE_B_SEEDS.json',seeds)
 f=dict(created_at=datetime.now(timezone.utc).isoformat(),primary='annual_primary',leaders=leaders,categories=cats,base=base,candidate_count=len(rows),admissible_count=len(admissible),code_sha256=sha(__file__),input_hashes=read(P/'PREPARED_HASHES.json'),author_labels_used=False)
 dump('FROZEN.json',f);(P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
 for name,row in leaders.items():print('Frozen',name,row,flush=True)

def evaluate():
 frozen=read(P/'FROZEN.json');digest=sha(P/'FROZEN.json');assert digest==(P/'FROZEN.sha256').read_text().strip();assert frozen['code_sha256']==sha(__file__)
 assert all(sha(P/n)==h for n,h in frozen['input_hashes'].items());assert all(sha(p)==h for p,h in read(P/'SOURCE_HASHES.json').items())
 m,d,layers,px,op=load();periods=read(O/'monthly_supplied_ledger_audit/PERIODS.json');models=dict(frozen['leaders']);models.update({'category_'+k:v for k,v in frozen['categories'].items()});models['anchor']=frozen['base'];results={};cache={}
 for name,row in models.items():
  spec=row['spec'];ch,sc,events=decisions(spec,m,d,layers,details=True);orders=rp.make_orders(ch,sc,m);met,eq=rp.replay(name,orders,px,op,m,export=False)
  assert abs(met['slices']['training35']['cagr']-row['training']['cagr'])<1e-10
  met['rule']=spec;matches=[]
  for i,(p,ids) in enumerate(zip(periods,ch)):
   ours=[e.display(m['symbols'][i][j]) for j in ids];matches.append(dict(signal_date=p['start'],holding_end=p['end'],selected=ours,author=p['holdings'],set_exact=set(ours)==set(p['holdings']),order_exact=ours==p['holdings'],shared_slots=len(set(ours)&set(p['holdings']))))
  met['matching']={label:dict(months=len(ss),sets=sum(x['set_exact'] for x in ss),orders=sum(x['order_exact'] for x in ss),slots=sum(x['shared_slots'] for x in ss)) for label,ss in [('train35',matches[:35]),('later34',matches[35:]),('all69',matches)]}
  dump(name+'_DAILY.json',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()]);dump(name+'_MONTHS.json',matches);dump(name+'_DECISIONS.json',events)
  if name in frozen['leaders']:
   for tag,mode,cost in [('next_open_zero','open',0.),('next_open_cost','open',.0005)]:met[tag]=rp.replay(name+'_'+tag,orders,px,op,m,cost,mode,export=False)[0]
   _,_,tr,holds=e.engine.replay(px,op,orders,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]));dump(name+'_TRADES.json',tr);dump(name+'_HOLDINGS.json',holds)
  results[name]=met;print(name,'Full',met['metrics']['cagr'],met['metrics']['sharpe'],'Validation',met['slices']['validation34']['cagr'],met['slices']['validation34']['sharpe'],flush=True)
 for label in ['author','qqq']:
  source=read(O/'monthly_return_priority/RESULTS.json')[label];results[label]=source
 previous=read(O/'monthly_return_priority/RESULTS.json')['old_candidate_original']
 for k in ['cagr','sharpe','max_drawdown']:assert abs(results['anchor']['metrics'][k]-previous['metrics'][k])<1e-10
 dump('RESULTS.json',results);assert sha(P/'FROZEN.json')==digest;changes=[p for p,h in read(P/'PROTECTED_BEFORE.json').items() if sha(p)!=h];assert not changes,changes
 dump('VALIDATION.json',dict(source_hashes_verified=True,frozen_hash=digest,protected_changes=[],anchor_exact_parity=True,independent_fixed_units_replay=True,train_and_full_replay_parity=True,author_labels_used_for_selection=False,annual_forecast_labels_matured=True,pristine_oos=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','prepare','forecast','train','evaluate']);globals()[ap.parse_args().phase]()

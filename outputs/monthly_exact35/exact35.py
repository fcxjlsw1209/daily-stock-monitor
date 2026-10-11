"""Exact first35 author-ranking fit, followed by sealed later34 evaluation.

Capacity experiment: interpolation is not evidence of original rule recovery.
Only numerical price/volume/sector/state features are passed to estimators.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,itertools,sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.tree import DecisionTreeRegressor
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_open_hypotheses'))
import open_inverse as old
read,sha=old.read,old.sha
e=old.e
TRAIN=35
MARGIN=.005

def dump(n,z):
 (P/n).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def initialize():
 assert not (P/'PLAN.json').exists()
 dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
 request='Fit first35 monthly three-name holdings AND order exactly by any explainable method, then freeze and measure same method on later34.',
 labels='Supplied records immutable. All35 training targets available to fit. Later34 records not consumed by training or complexity selection; history already viewed in previous studies, so not pristine OOS.',
 acceptance='35/35 exact ordered triples for every admitted model, then one same-method34-month replay. No changes after seeing validation.',
 universe='Choose smallest uniform width32..40 that covers every training author target in cached cap ranking. This boundary envelope is a disclosed substitute for originalTop32; never one-month/ticker injection. Same width and assumed theme17 apply to all69 months.',
 features='Known-at-monthend long momentum(skip42sessions/two calendar months), volatility/risk, own-sector relative strength, price-volume patterns and observable market state. No ticker, calendar date, month id, future price or author label enters predictor features.',
 baseline='Fixed70% mean four long-momentum ranks +15%vol126 +10%own-sector skipped252 +5%CMF21. Diagnostic starting rule, not author-disclosed formula.',
 models=['Explainable ordinal regression tree for scores3/2/1/0; reduce leaves while maintaining all35 ordered matches','Long-momentum baseline plus residual tree; training score corrections are minimum-L2 projection satisfying author order with .005margin','Same baseline plus15-neighbor inverse-square-distance residual interpolation; historical feature analogues, no ticker/date lookup'],
 selection='For trees test depth1..32 then unbounded; among exactly matching trees also test cost-complexity pruning, choose least leaves then depth. No return/validation optimization. k15 fixed in advance.',
 limitation='Exact high-capacity fitting may memorize samples. Residual size/complexity explicitly reported; no claim that this is the author small-ML model or original32 pool.',
 execution='Zero costs, monthend final-close/same-close idealization, equal three positions, fixed shares within month. Next-open sensitivity also exported. No observer-account changes.'))
 protected=read(O/'monthly_open_hypotheses/PROTECTED_BEFORE.json')
 dump('PROTECTED_BEFORE.json',{f:sha(f) for f in protected})

def rank_archive():
 f1=O/'monthly_top35_extension/EARLIER_RANKS.json';f2=O/'us_listed_top32_yahoo/PUBLISHED_MONTHLY_RANKINGS.json'
 ranks={z['date']:z['ranks'] for z in read(f1)}
 for z in read(f2)['periods']:
  rows=[r for r in z['top32']+z['boundary33_40'] if not(r['symbol']=='LTM' and '2020-06-22'<=z['date']<'2024-07-25')]
  ranks[z['date']]=rows
 return ranks,{str(f):sha(f) for f in [f1,f2]}

def choose_width(ranks,training,etfs):
 checks=[]
 for n in range(32,41):
  miss=[dict(date=p['start'],ticker=t) for p in training for t in p['holdings'] if t not in {z['symbol'] for z in ranks[p['start']][:n]}|set(etfs)]
  checks.append(dict(width=n,missing=miss))
  if not miss:return n,checks
 raise ValueError('Training targets still outside uniformly expanded archive')

def prepare():
 px,raw,members,periods,sources,_=e.load_prices();ranks,rsources=rank_archive();sources.update(rsources)
 n,audit=choose_width(ranks,periods[:TRAIN],members['theme17'][1]);print('Training-only universe width',n,flush=True)
 symbols=[sorted({e.actual(z['symbol']) for z in ranks[p['start']][:n]}|members['theme17'][1]) for p in periods]
 for t in sorted({t for ss in symbols for t in ss}-set(px.columns)):
  f=O/'monthly_top35_extension/prices'/f'{t}.csv';md=read(f.with_suffix('.source.json'));assert sha(f)==md['sha256'];sources[str(f)]=sha(f)
  z=pd.read_csv(f,index_col=0,parse_dates=True);px[t]=z['Adj Close'].reindex(px.index)
  for key,col in [('close','Close'),('high','High'),('low','Low'),('volume','Volume'),('opening','Open')]:raw[key][t]=z[col].reindex(px.index)
  raw['sector60'][t]=px['XLF'].pct_change(60,fill_method=None) if t=='SMFG' else np.nan
 f,relative,groups=old.feature_bank(px,raw)
 sectorfile=O/'index_union/alias_resolved_inputs/sectors.csv.gz';sectors=pd.read_csv(sectorfile,index_col=0,parse_dates=True,low_memory=False);sources[str(sectorfile)]=sha(sectorfile)
 base_names=list(f)+list(relative);state_names=['market_trend','market_highvol','market_growth_relative','market_smallcap_relative']
 dates=[p['start'] for p in periods];styles=old.old.market_styles(px,dates)
 states=np.column_stack([styles.spy_above200.astype(float),styles.highvol.astype(float),(1+np.tanh(styles.growth_relative63/.08))/2,(1+np.tanh(styles.smallcap_relative63/.08))/2])-.5
 names=base_names+state_names;group_list=[groups[k] for k in base_names]+['market']*4
 X=np.zeros((69,max(map(len,symbols)),len(names)));good=np.zeros(X.shape[:2],bool);base=np.zeros(X.shape[:2]);order=np.full((TRAIN,3),-1,int)
 coverage=[]
 for i,(p,ss) in enumerate(zip(periods,symbols)):
  date=p['start'];bm=old.strict.benchmark_map(date,ss,sectors,members['theme17'][1]);tab=pd.DataFrame({k:v.loc[date,ss] for k,v in f.items()})
  for k,v in relative.items():tab[k]=old.strict.relative_row(v,date,ss,bm)
  valid=tab[[k for k in base_names if groups[k]=='core']].notna().all(axis=1)&px.loc[date,ss].gt(0)
  q=old.normalize(tab.where(valid,axis=0),'rank');X[i,:len(ss),:len(base_names)]=q[base_names];X[i,:len(ss),len(base_names):]=states[i];good[i,:len(ss)]=valid
  mom=q[['core_252_raw_v0','core_294_raw_v0','core_calendar12_v0','core_calendar14_v0']].mean(axis=1)
  base[i,:len(ss)]=.70*mom+.15*q.vol126+.10*q.own_sector_skip42_252+.05*q.cmf21
  if i<TRAIN:
   for j,t in enumerate(p['holdings']):
    ticker=e.actual(t)
    if ticker in ss and valid.loc[ticker]:order[i,j]=ss.index(ticker)
   coverage.append(dict(date=date,missing=[t for j,t in enumerate(p['holdings']) if order[i,j]<0]))
 assert (order>=0).all(),coverage
 keep=sorted({t for ss in symbols for t in ss}|{'QQQ','SPY'});opening=raw['opening']*px/raw['close']
 np.savez_compressed(P/'FEATURES.npz',X=X,eligible=good,base=base)
 np.savez_compressed(P/'TRAIN_TARGETS.npz',order=order)
 np.savez_compressed(P/'PRICES.npz',close=px[keep].to_numpy(),opening=opening[keep].to_numpy(),dates=px.index.strftime('%Y-%m-%d').to_numpy(dtype='U10'),symbols=np.array(keep))
 dump('INPUTS.json',dict(dates=dates,ends=[p['end'] for p in periods],symbols=symbols,features=names,groups=group_list,universe_width=n,etfs=sorted(members['theme17'][1]),stock_pool=[[z['symbol'] for z in ranks[p['start']][:n]] for p in periods],split=TRAIN))
 dump('TRAIN_LABELS.json',periods[:TRAIN]);dump('POOL_AUDIT.json',dict(training_only_width_selection=audit,training_coverage=coverage,width=n,warning='Uniform boundary-envelope proxy, not claimed originalTop32. Width chosen from training records; original rankings remain unchanged.'))
 sources[str(O/'monthly_open_hypotheses/open_inverse.py')]=sha(O/'monthly_open_hypotheses/open_inverse.py');sources[str(Path(__file__))]=sha(__file__)
 dump('SOURCE_HASHES.json',sources)
 dump('PREPARED_HASHES.json',{n:sha(P/n) for n in ['INPUTS.json','FEATURES.npz','TRAIN_TARGETS.npz','TRAIN_LABELS.json','PRICES.npz']})
 print('Prepared105/105 training labels; features',X.shape,flush=True)

def load():
 return read(P/'INPUTS.json'),dict(np.load(P/'FEATURES.npz')),np.load(P/'TRAIN_TARGETS.npz')['order']

def picks(score,base,eligible,symbols):
 ans=[]
 for i in range(len(score)):
  choice=sorted(np.flatnonzero(eligible[i]),key=lambda j:(-score[i,j],-base[i,j],symbols[i][j]))[:3]
  ans.append(choice if len(choice)>=2 else [])
 return ans

def compare_indices(ch,order):
 return dict(months=len(order),set_exact=sum(set(ch[i])==set(row) for i,row in enumerate(order)),order_exact=sum(ch[i]==list(row) for i,row in enumerate(order)),shared_slots=sum(len(set(ch[i])&set(row)) for i,row in enumerate(order)))

def project_scores(base,order,margin=MARGIN):
 n=len(base);pairs=[(int(order[0]),int(order[1])),(int(order[1]),int(order[2]))]+[(int(order[2]),j) for j in range(n) if j not in order]
 A=np.zeros((len(pairs),n))
 for k,(a,b) in enumerate(pairs):A[k,a]=1;A[k,b]=-1
 def objective(z):return float(.5*np.sum((z-base)**2)),z-base
 result=minimize(objective,base.copy(),jac=True,method='SLSQP',constraints=[dict(type='ineq',fun=lambda z:np.einsum('ij,j->i',A,z,optimize=False)-margin,jac=lambda z:A)],options=dict(maxiter=300,ftol=1e-12))
 assert result.success,result.message
 assert np.min(np.einsum('ij,j->i',A,result.x,optimize=False))>=margin-1e-8
 return result.x

def training_rows(X,good,base,order):
 XX=[];by=[];res=[];ordinal=[];where=[];adjustments=[]
 for i in range(TRAIN):
  js=np.flatnonzero(good[i]);target=[list(js).index(int(j)) for j in order[i]];b=base[i,js];z=project_scores(b,target)
  yy=np.zeros(len(js));yy[target]=[3,2,1]
  XX.append(X[i,js]);by.extend(b);res.extend(z-b);ordinal.extend(yy);where.extend((i,int(j)) for j in js)
  adjustments.append(dict(month=i,rms=float(np.sqrt(np.mean((z-b)**2))),max_abs=float(np.max(np.abs(z-b))),baseline_std=float(np.std(b))))
 return np.vstack(XX),np.array(by),np.array(res),np.array(ordinal),where,adjustments

def flatten(X,good,count=None):
 count=len(X) if count is None else count
 return np.vstack([X[i,np.flatnonzero(good[i])] for i in range(count)])

def unpack(values,good):
 out=np.zeros(good.shape);out[good]=values;return out

def serialize_tree(model):
 t=model.tree_
 return dict(children_left=t.children_left.tolist(),children_right=t.children_right.tolist(),feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=t.value[:,0,0].tolist(),samples=t.n_node_samples.tolist(),depth=model.get_depth(),leaves=model.get_n_leaves())

def tree_predict(tree,X,paths=False):
 result=[];nodes=[];traces=[]
 for row in np.asarray(X,dtype=np.float32):
  node=0;trace=[]
  while tree['children_left'][node]!=-1:
   k=tree['feature'][node];left=bool(row[k]<=tree['threshold'][node]);trace.append(dict(node=node,feature=k,value=float(row[k]),threshold=tree['threshold'][node],branch='le' if left else 'gt'));node=tree['children_left' if left else 'children_right'][node]
  result.append(tree['value'][node]);nodes.append(node);traces.append(trace)
 return (np.array(result),nodes,traces) if paths else np.array(result)

def fit_tree(XX,y,where,base,good,order,symbols,residual):
 trials=[];successful=[]
 def check(model,config):
  raw=model.predict(XX);scores=np.zeros_like(base[:TRAIN]);scores[good[:TRAIN]]=raw
  if residual:scores+=base[:TRAIN]
  mm=compare_indices(picks(scores,base[:TRAIN],good[:TRAIN],symbols[:TRAIN]),order)
  trial=dict(**config,leaves=model.get_n_leaves(),depth=model.get_depth(),matching=mm);trials.append(trial)
  if mm['order_exact']==TRAIN:successful.append((trial,model))
 for depth in list(range(1,33))+[None]:
  model=DecisionTreeRegressor(max_depth=depth,random_state=42).fit(XX,y);check(model,dict(max_depth=depth,ccp_alpha=0.))
 assert successful,'A tree could not interpolate all training rankings'
 # Prune the most parsimonious depth candidate; all selection uses training only.
 seed=min(successful,key=lambda t:(t[0]['leaves'],t[0]['depth']))[1]
 for alpha in seed.cost_complexity_pruning_path(XX,y).ccp_alphas:
  if alpha<=0:continue
  model=DecisionTreeRegressor(max_depth=seed.max_depth,random_state=42,ccp_alpha=float(alpha)).fit(XX,y);check(model,dict(max_depth=seed.max_depth,ccp_alpha=float(alpha)))
 config,model=min(successful,key=lambda t:(t[0]['leaves'],t[0]['depth'],-t[0]['ccp_alpha']))
 tree=serialize_tree(model);np.testing.assert_array_equal(tree_predict(tree,XX),model.predict(XX))
 _,leaf,_=tree_predict(tree,XX,True);support={}
 for k,node in enumerate(leaf):support.setdefault(str(node),[]).append(k)
 return dict(config=config,tree=tree,support=support),trials

def scale_for_knn(XX,groups):
 mean=XX.mean(axis=0);std=XX.std(axis=0);std=np.where(std<1e-8,1.,std)
 weights=np.array([np.sqrt((2. if g=='core' else 1.)/groups.count(g)) for g in groups])
 return dict(mean=mean.tolist(),std=std.tolist(),weights=weights.tolist())

def scale(X,pre):return (X-np.array(pre['mean']))/np.array(pre['std'])*np.array(pre['weights'])

def knn_predict(model,X,details=False):
 train=np.array(model['train_features']);y=np.array(model['residual']);Q=scale(X,model['preprocess']);D=cdist(Q,train,'sqeuclidean');k=min(model['neighbors'],len(train));out=[];extra=[]
 for row in D:
  nearest=np.argsort(row,kind='stable')[:k];dist=row[nearest];zero=dist<=1e-24
  if zero.any():weight=zero.astype(float)/zero.sum()
  else:weight=1/np.maximum(dist,1e-24);weight/=weight.sum()
  out.append(float(np.sum(y[nearest]*weight)));extra.append(dict(training_rows=nearest.tolist(),squared_distances=dist.tolist(),weights=weight.tolist()))
 return (np.array(out),extra) if details else np.array(out)

def train():
 assert not (P/'FROZEN.json').exists()
 for n,h in read(P/'PREPARED_HASHES.json').items():assert sha(P/n)==h
 meta,data,order=load();X,good,base=data['X'],data['eligible'],data['base']
 XX,by,res,ordinal,where,adjustments=training_rows(X,good,base,order);models={};trials={}
 for name,y,residual in [('ordinal_tree',ordinal,False),('momentum_residual_tree',res,True)]:
  model,trial=fit_tree(XX,y,where,base,good,order,meta['symbols'],residual);model.update(kind='tree',residual=residual)
  models[name]=model;trials[name]=trial;print(name,model['config'],flush=True)
 pre=scale_for_knn(XX,meta['groups']);knn=dict(kind='knn',neighbors=15,preprocess=pre,train_features=scale(XX,pre).tolist(),residual=res.tolist())
 pred=base[:TRAIN]+unpack(knn_predict(knn,XX),good[:TRAIN]);mm=compare_indices(picks(pred,base[:TRAIN],good[:TRAIN],meta['symbols'][:TRAIN]),order);assert mm['order_exact']==TRAIN,mm
 models['momentum_knn15']=knn;print('momentum_knn15',mm,flush=True)
 dump('TREE_TRIALS.json',trials);dump('TRAIN_SCORE_CORRECTIONS.json',adjustments)
 dump('TRAIN_EXAMPLES.json',[dict(row=k,signal_date=meta['dates'][i],ticker=e.display(meta['symbols'][i][j]),ordinal_target=float(ordinal[k]),base=float(by[k]),correction=float(res[k])) for k,(i,j) in enumerate(where)])
 frozen=dict(created_at=datetime.now(timezone.utc).isoformat(),code_sha256=sha(__file__),input_hashes=read(P/'PREPARED_HASHES.json'),train_months=TRAIN,feature_count=X.shape[-1],training_examples=len(XX),models=models,primary='momentum_residual_tree',choice_policy='all models frozen before validation; primary fixed by plan, no selection on holdout outcomes')
 dump('FROZEN.json',frozen)
 # Save the frozen digest before the separate evaluate process can see holdout labels.
 (P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
 print('Frozen',len(models),'models, all35 ordered months matched',flush=True)

def predict(model,data):
 X,good,base=data['X'],data['eligible'],data['base'];XX=flatten(X,good)
 if model['kind']=='tree':values=tree_predict(model['tree'],XX);residual=model['residual']
 else:values=knn_predict(model,XX);residual=True
 score=unpack(values,good)
 if residual:score+=base
 return score

def replay(px,opening,orders,execution='close'):
 met,eq,trades,holds=e.engine.replay(px,opening,orders,cost=0.,execution=execution,start=pd.Timestamp(orders[0]['date']),end=pd.Timestamp(orders[-1]['end']))
 if execution=='close':np.testing.assert_allclose(eq,e.previous.independent(px,orders),rtol=1e-11,atol=1e-6)
 slices={}
 for k,lo,hi in [('training35',orders[0]['date'],orders[34]['end']),('validation34',orders[35]['date'],orders[-1]['end'])]:
  sub=eq.loc[lo:hi];slices[k]=e.engine.stats(sub/sub.iloc[0]*100000.)
 return dict(metrics=met,slices=slices),eq,trades

def evaluate():
 frozen=read(P/'FROZEN.json');digest=sha(P/'FROZEN.json');assert digest==(P/'FROZEN.sha256').read_text().strip();assert frozen['code_sha256']==sha(__file__)
 for n,h in frozen['input_hashes'].items():assert sha(P/n)==h
 assert all(sha(f)==h for f,h in read(P/'SOURCE_HASHES.json').items())
 meta,data,order=load();good,base=data['eligible'],data['base'];prices=np.load(P/'PRICES.npz');index=pd.to_datetime(prices['dates']);symbols=prices['symbols'];px=pd.DataFrame(prices['close'],index=index,columns=symbols);op=pd.DataFrame(prices['opening'],index=index,columns=symbols)
 # First access to later34 labels in the execution workflow is here.
 periods=read(O/'monthly_supplied_ledger_audit/PERIODS.json');assert len(periods)==69
 authororders=[dict(date=p['start'],end=p['end'],signal_date=p['start'],symbols=list(map(e.actual,p['holdings'])),scores=[3.,2.,1.]) for p in periods]
 author,aeq,atrades=replay(px,op,authororders);results={'author':author};details={};expl={};training_examples=read(P/'TRAIN_EXAMPLES.json')
 for name,model in frozen['models'].items():
  score=predict(model,data);chosen=picks(score,base,good,meta['symbols']);mm=compare_indices(chosen[:TRAIN],order);assert mm['order_exact']==TRAIN,mm
  matched={};monthrows=[];orders=[]
  for i,(p,ss) in enumerate(zip(periods,meta['symbols'])):
   picked=[ss[j] for j in chosen[i]];author_names=list(map(e.actual,p['holdings']));allrank=sorted(np.flatnonzero(good[i]),key=lambda j:(-score[i,j],-base[i,j],ss[j]));ranks={ss[j]:k+1 for k,j in enumerate(allrank)}
   mr=float((px.loc[p['end'],picked]/px.loc[p['start'],picked]-1).mean()) if picked else 0.;ar=float((px.loc[p['end'],author_names]/px.loc[p['start'],author_names]-1).mean())
   monthrows.append(dict(signal_date=p['start'],holding_end=p['end'],split='training' if i<TRAIN else 'validation',author=p['holdings'],selected=list(map(e.display,picked)),set_exact=set(picked)==set(author_names),order_exact=picked==author_names,shared_slots=len(set(picked)&set(author_names)),model_monthly_return=mr,author_monthly_return=ar,return_difference=mr-ar,author_model_ranks={e.display(t):ranks.get(t) for t in author_names},scores=[dict(ticker=e.display(ss[j]),score=float(score[i,j]),base_score=float(base[i,j])) for j in allrank]))
   orders.append(dict(date=p['start'],end=p['end'],signal_date=p['start'],symbols=picked,scores=[float(score[i,j]) for j in chosen[i]]))
  for label,lo,hi in [('training',0,35),('validation',35,69),('all',0,69)]:
   rows=monthrows[lo:hi];matched[label]=dict(months=len(rows),set_exact=sum(x['set_exact'] for x in rows),order_exact=sum(x['order_exact'] for x in rows),shared_slots=sum(x['shared_slots'] for x in rows),possible_slots=3*len(rows))
  r,eq,trades=replay(px,op,orders);opened,_,_=replay(px,op,orders,'open')
  np.testing.assert_allclose(eq.loc[:periods[34]['end']],aeq.loc[:periods[34]['end']],rtol=1e-12,atol=1e-6)
  r.update(matching=matched,next_open=opened,complexity=dict(depth=model['tree']['depth'],leaves=model['tree']['leaves']) if model['kind']=='tree' else dict(stored_examples=frozen['training_examples'],neighbors=model['neighbors']))
  results[name]=r;dump(name+'_MONTHS.json',monthrows);dump(name+'_TRADES.json',trades);dump(name+'_DAILY.json',[dict(date=str(d.date()),equity=float(v)) for d,v in eq.items()])
  # Explain every selected stock and missed author stock with actual model paths or nearest examples.
  explanations=[]
  for i,(p,ss) in enumerate(zip(periods,meta['symbols'])):
   wanted=sorted(set([ss[j] for j in chosen[i]])|set(map(e.actual,p['holdings'])))
   for t in wanted:
    if t not in ss:explanations.append(dict(date=p['start'],ticker=e.display(t),unavailable=True));continue
    j=ss.index(t);x=data['X'][i,j:j+1]
    ex=dict(date=p['start'],split='training' if i<TRAIN else 'validation',ticker=e.display(t),selected=j in chosen[i],author=t in set(map(e.actual,p['holdings'])),base_score=float(base[i,j]),final_score=float(score[i,j]))
    if model['kind']=='tree':
     v,leaf,traces=tree_predict(model['tree'],x,True);ex.update(predicted_component=float(v[0]),leaf=int(leaf[0]),rules=[dict(feature=meta['features'][z['feature']],observed=z['value'],threshold=z['threshold'],branch=z['branch']) for z in traces[0]],training_support=[training_examples[k] for k in model['support'].get(str(leaf[0]),[])])
    else:
     v,nb=knn_predict(model,x,True);ex.update(predicted_correction=float(v[0]),nearest_examples=[dict(**training_examples[k],distance=float(np.sqrt(d)),weight=w) for k,d,w in zip(nb[0]['training_rows'],nb[0]['squared_distances'],nb[0]['weights'])])
    explanations.append(ex)
  dump(name+'_EXPLANATIONS.json',explanations)
  print(name,dict(matching=matched,train=r['slices']['training35'],validation=r['slices']['validation34'],complexity=r['complexity']),flush=True)
 qorders=[dict(date=periods[0]['start'],end=periods[-1]['end'],signal_date=periods[0]['start'],symbols=['QQQ'],scores=[1.])]
 qm,qe,_,_=e.engine.replay(px,op,qorders,cost=0.,execution='close',start=pd.Timestamp(periods[0]['start']),end=pd.Timestamp(periods[-1]['end']))
 results['qqq']=dict(metrics=qm,slices={k:e.engine.stats(qe.loc[lo:hi]/qe.loc[lo:hi].iloc[0]*100000.) for k,lo,hi in [('training35',periods[0]['start'],periods[34]['end']),('validation34',periods[35]['start'],periods[-1]['end'])]})
 dump('RESULTS.json',results)
 assert sha(P/'FROZEN.json')==digest
 changes=[f for f,h in read(P/'PROTECTED_BEFORE.json').items() if sha(f)!=h];assert not changes,changes
 dump('VALIDATION.json',dict(frozen_hash=digest,protected_changes=changes,all_models_exact35=True,training_equity_matches_author=True,independent_close_replay_verified=True,holdout_not_used_by_fit=True,validation_already_viewed_in_prior_research=True,complete_author_algorithm_recovery=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','prepare','train','evaluate']);globals()[ap.parse_args().phase]()

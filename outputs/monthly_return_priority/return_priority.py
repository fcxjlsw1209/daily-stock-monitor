"""Frozen, bounded monthly performance study. Author labels are diagnostics only."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,itertools,json,sys
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent;S=O/'monthly_exact35'
sys.path.insert(0,str(S))
import exact35 as prior
read,sha,e=prior.read,prior.sha,prior.e
TRAIN=35
CORES=['core_252_raw_v0','core_294_raw_v0','core_calendar12_v0','core_calendar14_v0','core_252_raw_v-1','core_294_raw_v-1','core_252_raw_v-0.5','core_294_raw_v-0.5']
SIDES={'growth':dict(vol=.4,accel=.4,industry=.15,flow=.05),'balanced':dict(vol=.2,accel=.3,industry=.35,flow=.15),'quality':dict(vol=.35,accel=.25,industry=.25,flow=.15),'flow':dict(vol=.2,accel=.2,industry=.3,flow=.3)}

def dump(n,z):
 (P/n).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def load():
 m=read(S/'INPUTS.json');d=dict(np.load(S/'FEATURES.npz'));a=np.load(S/'PRICES.npz');idx=pd.to_datetime(a['dates']);ss=a['symbols']
 return m,d,pd.DataFrame(a['close'],index=idx,columns=ss),pd.DataFrame(a['opening'],index=idx,columns=ss)

def initialize():
 assert not (P/'PLAN.json').exists()
 dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),objective='Prioritize return, Sharpe and drawdown; holdings/order agreement is an after-freeze diagnostic, never training objective.',
 split='First35 holding months Jan2021-Nov2023 for design selection; later34 Dec2023-Sep24,2026 evaluated after freeze. These dates already viewed in prior work; not pristine independent OOS.',
 universe='Unchanged cached uniformTop36+assumedtheme17 from exact35 experiment; pool width is not searched. Historical ranking remains a proxy.',
 family='8 long-momentum definitions all skip latest2months;4 supporting portfolios; core weights50/65/80%;3 market modes(static/risk-aware/cash-belowSPY200);2 core gates(all/upperhalf);2 monthly retention limits(top3/top5):1152 fixed configurations.',
 guidance='Long only, no leverage, at most3 equal positions,2 half,<2cash; fixed units intra-month. Risk-aware substitutes inverse downside volatility when SPYbelow200 or highvol, and reverses CMF in highvol. No daily entry pullback condition.',
 primary='Maximize robust training criterion .4*overall_dailySharpe+.4*minimum_blockSharpe+.2*mean_blockSharpe+.25*log(1+CAGR)-.5*abs(maxDD). Blocks first12,next12,last11 months. Tie lowerID.',
 controls='TrainingSharpe winner and trainingCAGR winner also frozen; compare old high-return candidate exact historical holdings and old formula remapped to same pool, unmodified exact35 model, author and QQQ.',
 inner_check='Expanding-prefix choice at12/18/24 months, followed by next6/6/11 months. Concatenate subsequent returns of chosen virtual rule; this is an inner diagnostic, not a second tuning objective. Final choice still all35 criterion.',
 execution='Primary zero-cost same-final-close monthly selection/rebalance idealization. Next-open and5bp/side cost controls. No author monthly return targets, curve-fitting, deep tree, date/ticker features or per-month exceptions.',
 output='Report all3 predeclared selectors regardless of holdout ordering; never switch primary after evaluation. No live observer, ledger, plan, fingerprint, or automation modifications.'))
 protected=read(S/'PROTECTED_BEFORE.json');protected.update({str(f):sha(f) for f in S.iterdir() if f.is_file()})
 dump('PROTECTED_BEFORE.json',{f:sha(f) for f in protected})
 files=[S/'INPUTS.json',S/'FEATURES.npz',S/'PRICES.npz',S/'exact35.py',O/'monthly_decision_audit/MONTHS.json',O/'monthly_supplied_ledger_audit/PERIODS.json',Path(e.engine.__file__)]
 dump('SOURCE_HASHES.json',{str(f):sha(f) for f in files})

def specs():
 out=[]
 for core,side,w,regime,gate,retain in itertools.product(CORES,SIDES,[.5,.65,.8],['static','risk_aware','cash_weak'],['all','upperhalf'],[3,5]):
  out.append(dict(id=len(out),core=core,side=side,core_weight=w,market=regime,gate=gate,retain=retain))
 assert len(out)==1152
 return out

def score(spec,X,names):
 g=lambda n:X[:,:,names.index(n)]
 core=g(spec['core']);v=-g('downside63') if spec['side']=='quality' else g('vol126');flow=g('cmf21');weak=g('market_trend')<0;high=g('market_highvol')>0
 if spec['market']=='risk_aware':
  v=np.where(weak|high,-g('downside63'),v);flow=np.where(high,-g('cmf63'),flow)
 a=SIDES[spec['side']];support=a['vol']*v+a['accel']*g('acceleration_skip42')+a['industry']*g('own_sector_skip42_252')+a['flow']*flow
 return spec['core_weight']*core+(1-spec['core_weight'])*support,core,weak

def choose(sc,good,symbols,retain=3):
 previous=[];ans=[]
 for i,ss in enumerate(symbols):
  rank=sorted(np.flatnonzero(good[i]),key=lambda j:(-sc[i,j],ss[j]));r={ss[j]:k+1 for k,j in enumerate(rank)}
  keep=[t for t in previous if r.get(t,10**6)<=retain]
  keep=sorted(keep,key=lambda t:r[t])[:3]
  selected=keep+[ss[j] for j in rank if ss[j] not in keep][:3-len(keep)]
  if len(selected)<2:selected=[]
  selected=sorted(selected,key=lambda t:r[t]);ans.append([ss.index(t) for t in selected]);previous=selected
 return ans

def decisions(spec,m,d,count=69):
 X=d['X'][:count];sc,core,weak=score(spec,X,m['features']);good=d['eligible'][:count].copy()
 if spec['gate']=='upperhalf':good&=core>=0
 if spec['market']=='cash_weak':good&=~weak
 return choose(sc,good,m['symbols'][:count],spec['retain']),sc

def paths(px,m,count):
 blocks=[];dates=[];bounds=[0];valid=[]
 for i in range(count):
  z=px.loc[m['dates'][i]:m['ends'][i],m['symbols'][i]];a=z.to_numpy()/z.iloc[0].to_numpy();blocks.append(a);valid.append(np.isfinite(a).all(axis=0)&(a>0).all(axis=0));dates.extend(z.index[1:]);bounds.append(len(dates))
 return blocks,pd.DatetimeIndex(dates),bounds,valid

def fast_returns(chosen,blocks,valid):
 returns=[]
 for ids,a,ok in zip(chosen,blocks,valid):
  if not ids:r=np.zeros(len(a)-1)
  else:
   assert ok[ids].all(),'Missing held historical price; must resolve data, not silently drop candidate'
   nav=a[:,ids].mean(axis=1);r=nav[1:]/nav[:-1]-1
  returns.extend(r)
 return np.array(returns)

def stats_rr(r,start,end):
 nav=np.r_[1.,np.cumprod(1+r)];years=(pd.Timestamp(end)-pd.Timestamp(start)).days/365.2425;sd=np.std(r,ddof=1)
 return dict(cagr=float(nav[-1]**(1/years)-1),sharpe=float(np.mean(r)/sd*np.sqrt(252)) if sd>1e-14 else 0.,max_drawdown=float(np.min(nav/np.maximum.accumulate(nav)-1)),total_return=float(nav[-1]-1))

def robust(r,bounds,m,n):
 full=stats_rr(r[:bounds[n]],m['dates'][0],m['ends'][n-1]);blocks=[]
 for lo in range(0,n,12):
  hi=min(lo+12,n);blocks.append(stats_rr(r[bounds[lo]:bounds[hi]],m['dates'][lo],m['ends'][hi-1]))
 sh=[q['sharpe'] for q in blocks]
 criterion=.4*full['sharpe']+.4*min(sh)+.2*np.mean(sh)+.25*np.log1p(full['cagr'])-.5*abs(full['max_drawdown'])
 return float(criterion),full,blocks

def train():
 assert not (P/'FROZEN.json').exists()
 for p,h in read(P/'SOURCE_HASHES.json').items():assert sha(p)==h
 m,d,px,op=load();blocks,dates,bounds,valid=paths(px,m,TRAIN);grid=specs();rr=[];trials=[]
 for spec in grid:
  ch,sc=decisions(spec,m,d,TRAIN);r=fast_returns(ch,blocks,valid);criterion,metric,segments=robust(r,bounds,m,TRAIN)
  rr.append(r);trials.append(dict(spec=spec,robust=criterion,training=metric,blocks=segments))
  if len(trials)%192==0:print('Evaluated training rules',len(trials),flush=True)
 rr=np.vstack(rr)
 selected={'balanced':max(trials,key=lambda z:(z['robust'],-z['spec']['id'])),'sharpe':max(trials,key=lambda z:(z['training']['sharpe'],-z['spec']['id'])),'return':max(trials,key=lambda z:(z['training']['cagr'],-z['spec']['id']))}
 wf=[];wr=[]
 for n,end in [(12,18),(18,24),(24,35)]:
  vals=[robust(r,bounds,m,n)[0] for r in rr];win=int(np.argmax(vals));piece=rr[win,bounds[n]:bounds[end]];wr.extend(piece)
  wf.append(dict(train_months=n,selected_id=win,test_start=m['dates'][n],test_end=m['ends'][end-1],metrics=stats_rr(piece,m['dates'][n],m['ends'][end-1])))
 dump('TRAIN_TRIALS.json',trials);np.savez_compressed(P/'TRAIN_RETURNS.npz',returns=rr,bounds=bounds,dates=np.array(dates.strftime('%Y-%m-%d')))
 dump('TRAIN_WALKFORWARD.json',dict(folds=wf,stitched=stats_rr(np.array(wr),m['dates'][12],m['ends'][34]),note='Prefix-only selection of virtual strategy followed by forward block; no author labels. Same historical data has prior research exposure.'))
 frozen=dict(created_at=datetime.now(timezone.utc).isoformat(),primary='balanced',selected=selected,code_sha256=sha(__file__),source_hashes=read(P/'SOURCE_HASHES.json'),candidate_count=len(trials),author_labels_used=False)
 dump('FROZEN.json',frozen);(P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
 for k,z in selected.items():print('Frozen',k,z,flush=True)

def make_orders(ch,sc,m):
 return [dict(date=m['dates'][i],end=m['ends'][i],signal_date=m['dates'][i],symbols=[m['symbols'][i][j] for j in ids],scores=[float(sc[i,j]) for j in ids]) for i,ids in enumerate(ch)]

def slices(eq,m):
 out={}
 for label,lo,hi in [('training35',m['dates'][0],m['ends'][34]),('validation34',m['dates'][35],m['ends'][-1])]:
  s=eq.loc[lo:hi];out[label]=e.engine.stats(s/s.iloc[0]*100000.)
 return out

def replay(name,orders,px,op,m,cost=0.,execution='close',export=True):
 met,eq,tr,hold=e.engine.replay(px,op,orders,cost=cost,execution=execution,start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
 if execution=='close' and cost==0:np.testing.assert_allclose(eq,e.previous.independent(px,orders),rtol=1e-11,atol=1e-6)
 if export:
  dump(name+'_DAILY.json',[dict(date=str(d.date()),equity=float(v)) for d,v in eq.items()]);dump(name+'_TRADES.json',tr);dump(name+'_HOLDINGS.json',hold)
 return dict(metrics=met,slices=slices(eq,m)),eq

def evaluate():
 frozen=read(P/'FROZEN.json');digest=sha(P/'FROZEN.json');assert digest==(P/'FROZEN.sha256').read_text().strip();assert frozen['code_sha256']==sha(__file__)
 assert all(sha(p)==h for p,h in frozen['source_hashes'].items())
 m,d,px,op=load();periods=read(O/'monthly_supplied_ledger_audit/PERIODS.json');models={};results={};allrows={}
 for name,z in frozen['selected'].items():
  ch,sc=decisions(z['spec'],m,d);models[name]=make_orders(ch,sc,m)
 # Fixed controls, chosen before this experiment's holdout evaluation.
 b=d['base'];ch=choose(b,d['eligible'],m['symbols']);models['fixed70_base']=make_orders(ch,b,m)
 g=lambda n:d['X'][:,:,m['features'].index(n)]
 oldsc=.52*g('core_294_raw_v-1')+.219375*g('vol126')+.185625*g('acceleration_skip42')-.075*(2*g('market_highvol'))*g('cmf63')
 good=d['eligible'].copy();mom=px.shift(42)/px.shift(252)-1;warm=px.notna().rolling(295).sum().eq(295)
 for i,ss in enumerate(m['symbols']):good[i,:len(ss)]&=(warm.loc[m['dates'][i],ss]&mom.loc[m['dates'][i],ss].gt(0)).to_numpy()
 models['old_formula_top36']=make_orders(choose(oldsc,good,m['symbols']),oldsc,m)
 audit=read(O/'monthly_decision_audit/MONTHS.json')
 for name,lists in [('old_candidate_original',[z['ours'] for z in audit]),('author',[p['holdings'] for p in periods])]:
  models[name]=[dict(date=p['start'],end=p['end'],signal_date=p['start'],symbols=list(map(e.actual,ss)),scores=[]) for p,ss in zip(periods,lists)]
 models['qqq']=[dict(date=m['dates'][0],end=m['ends'][-1],signal_date=m['dates'][0],symbols=['QQQ'],scores=[])]
 blocks,dates,bounds,valid=paths(px,m,69)
 for name,orders in models.items():
  result,eq=replay(name,orders,px,op,m);results[name]=result
  if name=='qqq':continue
  rows=[]
  for i,(o,p) in enumerate(zip(orders,periods)):
   a=list(map(e.actual,p['holdings']));ss=o['symbols'];r=float((px.loc[p['end'],ss]/px.loc[p['start'],ss]-1).mean()) if ss else 0.
   rows.append(dict(signal_date=p['start'],holding_end=p['end'],split='training' if i<35 else 'validation',selected=list(map(e.display,ss)),author=p['holdings'],set_exact=set(ss)==set(a),order_exact=ss==a,shared_slots=len(set(ss)&set(a)),monthly_return=r))
  result['matching']={k:dict(months=len(z),set_exact=sum(r['set_exact'] for r in z),order_exact=sum(r['order_exact'] for r in z),shared_slots=sum(r['shared_slots'] for r in z),possible_slots=len(z)*3) for k,z in [('training35',rows[:35]),('validation34',rows[35:]),('all69',rows)]}
  allrows[name]=rows;dump(name+'_MONTHS.json',rows)
  if name in frozen['selected']:
   ch,sc=decisions(frozen['selected'][name]['spec'],m,d);r=fast_returns(ch,blocks,valid)
   np.testing.assert_allclose(np.r_[1.,np.cumprod(1+r)]*100000,eq.to_numpy(),rtol=1e-11,atol=1e-5)
   tr=frozen['selected'][name]['training'];assert abs(tr['cagr']-result['slices']['training35']['cagr'])<1e-10
   for tag,mode,cost in [('next_open_zero','open',0.),('next_open_cost','open',.0005)]:result[tag]=replay(name+'_'+tag,orders,px,op,m,cost,mode,False)[0]
   print(name,'Full',result['metrics']['cagr'],result['metrics']['sharpe'],'Later34',result['slices']['validation34'],flush=True)
 old_expected=read(O/'monthly_ordered_inverse/RESULTS.json')['previous_candidate']['metrics']
 for k in ['cagr','sharpe','max_drawdown']:assert abs(results['old_candidate_original']['metrics'][k]-old_expected[k])<1e-10,(k,results['old_candidate_original']['metrics'][k],old_expected[k])
 dump('RESULTS.json',results)
 assert sha(P/'FROZEN.json')==digest
 changes=[p for p,h in read(P/'PROTECTED_BEFORE.json').items() if sha(p)!=h];assert not changes,changes
 dump('VALIDATION.json',dict(frozen_unchanged=True,author_labels_not_used_for_selection=True,independent_fixed_units_replay=True,vectorized_train_and_full_replay_parity=True,old_candidate_reproduced=True,protected_changes=changes,primary=frozen['primary'],pristine_oos=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','train','evaluate']);globals()[ap.parse_args().phase]()

"""Independent round135 outcome audit; actual market replay requires explicit CLI.

Only the previously frozen independent accounting helper is reused. No producer
study, feature, selector or replay engine is imported. Imports read source only.
"""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
STUDY='outputs/monthly_absolute_state135'
PRICE='outputs/monthly_2016_extension/PRICES.npz'
HELPER=ROOT/'work/monthly_ridge_support70_result_review.py'
HELPER_SHA='d0f053881cb94e49e7fff9eaaeaa9e260f0b3c66419f87819391c309cb7cb8ef'
if hashlib.sha256(HELPER.read_bytes()).hexdigest()!=HELPER_SHA:raise ValueError('Frozen independent account helper changed')
_spec=importlib.util.spec_from_file_location('absolute135_independent_accounting',HELPER)
h=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(h)
NAMES=('incumbent_local67','matched_memoryless_state','filtered_latent_state')
PATHS=(('close',0.,'same_close'),('open',0.,'next_open_zero'),('open',.0005,'next_open_cost'))
CACHE_FILES=('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json','STATE_AUDIT.json')
FROZEN_FILES=('PLAN.json','CACHE.json','PRETRAIN_ORDERS.json','TRAIN_TRIALS.json','WALKFORWARD.json','INDEPENDENT_PREPARE.json')
PREPARE_CHECKS=('original27_arrays_byte_identical','all_probabilities_and_core_ranks_independently_verified','all_three_score_matrices_and_ordered_choices_independently_verified','one_episode_and_prefix_isolation_verified','zero_and_missing_exact_fallback')

def require(ok,msg):
 if not ok:raise ValueError(msg)

def source_files():return [Path(__file__).resolve(),Path(__file__).with_name('test_monthly_absolute_state135_outcome_review.py').resolve(),HELPER]

class Reader:
 def __init__(self,root):self.root=Path(root).resolve();self.seen={}
 def path(self,path):
  p=Path(path);return p.resolve()if p.is_absolute()else(self.root/p).resolve()
 def sha(self,path):
  p=self.path(path);digest=hashlib.sha256()
  with p.open('rb')as f:
   for part in iter(lambda:f.read(1<<20),b''):digest.update(part)
  value=digest.hexdigest();require(str(p)not in self.seen or self.seen[str(p)]==value,'Artifact changed during review: '+str(p));self.seen[str(p)]=value;return value
 def json(self,path):
  self.sha(path)
  def pairs(items):
   out={}
   for key,value in items:require(key not in out,'Duplicate JSON key');out[key]=value
   return out
  def bad(value):raise ValueError('Nonfinite JSON token: '+value)
  return json.loads(self.path(path).read_text(),object_pairs_hook=pairs,parse_constant=bad)
 def npz(self,path):
  self.sha(path)
  with np.load(self.path(path),allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
 def verify(self,mapping,base=None):
  require(isinstance(mapping,dict)and bool(mapping),'Nonempty hash mapping');out={}
  for path,digest in mapping.items():
   if base is not None:
    require(Path(path).name==path and path not in ('','.','..'),'Manifest basename required');path=Path(base)/path
   p=str(self.path(path));require(isinstance(digest,str)and len(digest)==64 and all(c in '0123456789abcdef'for c in digest),'SHA256 required')
   require(p not in out or out[p]==digest,'Conflicting canonical pins');require(self.sha(p)==digest,'Hash mismatch: '+str(path));out[p]=digest
  return out

def daily_axis(values,label):
 a=pd.DatetimeIndex(values);require(len(a)>1 and not a.hasnans and a.tz is None and a.equals(a.normalize())and a.is_unique and a.is_monotonic_increasing,'Daily axis '+label);return a

def validate_orders(close,opening,orders,meta):
 require(isinstance(close,pd.DataFrame)and isinstance(opening,pd.DataFrame),'Price frames')
 dates=daily_axis(close.index,'prices');require(opening.index.equals(dates)and opening.columns.equals(close.columns)and close.columns.is_unique,'Same unique price axes')
 days=daily_axis(meta['dates'],'signals');ends=pd.DatetimeIndex(meta['ends'])
 require(len(days)==len(ends)==len(meta['symbols'])==122 and ends.isin(dates).all()and days.isin(dates).all()and (ends>days).all(),'122 complete signal periods')
 require(days[1:].equals(ends[:-1]),'Adjacent frozen holding periods')
 require(set(orders)==set(NAMES),'Three exact order arms')
 for name in NAMES:
  require(len(orders[name])==122,'122 orders per arm')
  for i,row in enumerate(orders[name]):
   require(set(row)=={'date','end','signal_date','symbols','scores'},'Exact order fields')
   require(row['date']==row['signal_date']==meta['dates'][i]and row['end']==meta['ends'][i],'Order date and period')
   names=row['symbols'];require(isinstance(names,list)and len(names)==len(set(names))and len(names)in(0,2,3),'Cash or two/three equal positions')
   require(all(isinstance(s,str)and s in meta['symbols'][i]and s in close.columns for s in names),'Original order security identifiers')
   require(len(row['scores'])==len(names)and all(type(v)in(int,float)and math.isfinite(v)for v in row['scores']),'Finite aligned scores')
   require(dates.get_loc(days[i])+1<len(dates),'Next price session exists')
 return dates[(dates>=days[0])&(dates<=ends[-1])]

def replay(close,opening,orders,dates,execution='close',cost=0.):
 require(execution in('close','open')and cost in(0.,.0005),'Registered execution scenario')
 return h.replay(close,opening,orders,dates,execution=execution,cost=cost)

def compare_stats(actual,expected,errors,path=''):
 if isinstance(expected,dict):
  require(isinstance(actual,dict)and set(actual)==set(expected),'Statistic fields '+path)
  for k in expected:compare_stats(actual[k],expected[k],errors,path+'/'+k)
 elif isinstance(expected,list):
  require(isinstance(actual,list)and len(actual)==len(expected),'Statistic length '+path)
  for i,(a,b)in enumerate(zip(actual,expected)):compare_stats(a,b,errors,path+'/'+str(i))
 elif expected is None or isinstance(expected,(str,bool,int)):
  require(type(actual)is type(expected)and actual==expected,'Statistic identity '+path)
 else:
  require(type(actual)in(int,float)and math.isfinite(actual)and math.isfinite(expected),'Finite statistic '+path)
  error=abs(actual-expected);errors.append((path,error));dollar=path.rsplit('/',1)[-1]in('final_equity','fees_paid','dollar_turnover')
  require(math.isclose(actual,expected,abs_tol=1e-5 if dollar else 1e-10,rel_tol=2e-11 if dollar else 0.),'Statistic tolerance '+path)

def normalize(rows):
 rows=json.loads(json.dumps(rows))
 def name(s):return {'research_equity_FB':'META','BRK.B':'BRK-B'}.get(s,s)
 for row in rows:
  if 'symbol'in row:row['symbol']=name(row['symbol'])
  if 'symbols'in row:row['symbols']=[name(s)for s in row['symbols']]
 return rows

def select(rows):
 require([r['name']for r in rows]==list(NAMES),'Three exact selection arms')
 fields=(('training','cagr'),('training','sharpe'),('training','max_drawdown'),('early','cagr'),('recent','cagr'))
 values=[[row.get(a,{}).get(b)for a,b in fields]for row in (rows[2],rows[0])]
 valid=all(type(v)in(int,float)and math.isfinite(v)for group in values for v in group)
 win=bool(valid and values[0][0]>values[1][0]+1e-12 and all(x>=y-t for x,y,t in zip(values[0],values[1],(1e-12,.05,.03,.025,.025))))
 return dict(primary=NAMES[2]if win else NAMES[0],admitted=win,guards=win,guards_against={NAMES[0]:win})

def rr_statistics(rr,start,end):
 # Same fixed zero-riskfree and partial-year convention as prior independent audits.
 rr=np.asarray(rr,float);require(rr.ndim==1 and len(rr)>1 and np.isfinite(rr).all()and (rr>-1).all(),'Valid return vector')
 years=(pd.Timestamp(end)-pd.Timestamp(start)).days/365.2425;require(years>0.,'Positive elapsed years')
 nav=np.r_[1.,np.cumprod(1+rr)];sd=np.std(rr,ddof=1)
 return dict(cagr=float(nav[-1]**(1/years)-1),sharpe=float(np.mean(rr)/sd*np.sqrt(252))if sd>1e-14 else 0.,max_drawdown=float(np.min(nav/np.maximum.accumulate(nav)-1)),total_return=float(nav[-1]-1))

def walkforward(dates,navs,meta):
 folds=[];stitched=[];control=[]
 for n,end in((36,60),(60,76),(76,88)):
  pick=select([dict(name=name,**h.measures(dates,navs[name],meta,n))for name in NAMES]);folds.append(dict(n=n,end=end,selection=pick))
  lo=dates.get_loc(pd.Timestamp(meta['dates'][n]));hi=dates.get_loc(pd.Timestamp(meta['ends'][end-1]))
  for out,name in((stitched,pick['primary']),(control,'incumbent_local67')):
   nav=navs[name];out.extend((nav[1:]/nav[:-1]-1.)[lo:hi])
 return dict(folds=folds,pristine_oos=False,metrics=rr_statistics(stitched,meta['dates'][36],meta['ends'][87]),incumbent=rr_statistics(control,meta['dates'][36],meta['ends'][87]))

def monthly(orders,close,meta):
 rows=[]
 for i,day in enumerate(meta['dates']):
  returns={};held={}
  for name in NAMES:
   symbols=orders[name][i]['symbols'];held[name]=symbols
   returns[name]=math.fsum(float(close.at[pd.Timestamp(meta['ends'][i]),s])/float(close.at[pd.Timestamp(day),s])-1. for s in symbols)/len(symbols)if symbols else 0.
  rows.append(dict(date=day,end=meta['ends'][i],split='early53'if i<53 else'recent35'if i<88 else'later34',holdings=held,returns=returns))
 return rows


def attribution(rows,metrics):
 comparisons={}
 for label,name,reference in (('filtered_vs_incumbent',NAMES[2],NAMES[0]),('memoryless_vs_incumbent',NAMES[1],NAMES[0]),('filtered_vs_memoryless',NAMES[2],NAMES[1])):
  changes=[];ordered=0
  for row in rows:
   ordered+=int(row['holdings'][reference]!=row['holdings'][name])
   if set(row['holdings'][reference])!=set(row['holdings'][name]):
    changes.append(dict(date=row['date'],split=row['split'],old=row['holdings'][reference],new=row['holdings'][name],guidance_events=[],log_contribution=math.log1p(row['returns'][name])-math.log1p(row['returns'][reference])))
  positive=sorted((x['log_contribution']for x in changes if x['log_contribution']>0.),reverse=True);metric=metrics[name]
  comparisons[label]=dict(changed_months=len(changes),ordered_changed_months=ordered,changes=changes,undo_best1_cagr=((1+metric['total_return'])*math.exp(-math.fsum(positive[:1])))**(1/metric['years'])-1,undo_best3_cagr=((1+metric['total_return'])*math.exp(-math.fsum(positive[:3])))**(1/metric['years'])-1)
 return dict(comparisons=comparisons)

def verify_state_reuse(state,original):
 require(set(original)=={'years','signals','new_model_fits'},'Original state audit schema')
 require(len(state['years'])==len(original['years'])and len(state['signals'])==len(original['signals']),'Reuse audit lengths')
 for current,old in zip(state['years'],original['years']):
  keep={k:v for k,v in current.items()if k not in('annualized_state_volatility','quiet_state_mask')}
  h.assert_exact_tree(keep,old)
 for current,old in zip(state['signals'],original['signals']):
  keep={k:v for k,v in current.items()if k not in('original_filtered_probability','original_memoryless_probability')}
  keep['filtered_probability']=current['original_filtered_probability'];keep['memoryless_probability']=current['original_memoryless_probability']
  h.assert_exact_tree(keep,old)
 require(state['reused_model_fits']==original['new_model_fits'],'Exact historical fit count reuse')


def model_audit_counts(state,feature,validation,meta):
 require(isinstance(state,dict)and set(state)=={'years','signals','new_model_fits','reused_model_fits'},'Exact state audit schema')
 years=state['years'];expected_years=sorted({pd.Timestamp(day).year for day in meta['dates']})
 require(isinstance(years,list)and [row.get('year')for row in years]==expected_years,'Complete annual state audit')
 by_year={};count=0
 for row in years:
  year=row['year'];require(type(year)is int and type(row.get('fit_attempted'))is bool,'Annual fit count type')
  require(row.get('cutoff')==str(year-1)+'-12-31'and row.get('lower_bound_exclusive')==str(year-6)+'-12-31','Annual training timing')
  require(row.get('status')in('available','unavailable'),'Annual model status');model=row.get('model')
  if model is None:require(not row['fit_attempted']and row['status']=='unavailable','Unfitted year status')
  else:
   require(isinstance(model,dict)and type(model.get('fit_attempted'))is bool and model['fit_attempted']==row['fit_attempted'],'Nested annual fit count')
   require(model.get('status')==row['status'],'Nested annual status')
  if row['status']=='available':
   require(row['fit_attempted'],'Available model requires fit')
   sd=model['scaler']['sd'];variances=model['parameters']['variances']
   require(type(sd)in(int,float)and math.isfinite(sd)and sd>0 and len(variances)==2 and all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in variances),'Finite positive reused scale and variances')
   require(variances[0]<variances[1],'Original states must be low-variance first')
   annual=[sd*math.sqrt(v)*math.sqrt(252)for v in variances]
   require(all(math.isfinite(v)for v in annual),'Finite annualized state volatility')
   compare_stats(row['annualized_state_volatility'],annual,[],'annualized_state_volatility')
   quiet=row['quiet_state_mask'];require(isinstance(quiet,list)and len(quiet)==2 and all(type(v)is bool for v in quiet)and quiet==[v<.10 for v in annual],'Strict absolute quiet threshold')
  else:require(row['annualized_state_volatility']is None and row['quiet_state_mask']is None,'Unavailable annual interpretation remains missing')
  count+=int(row['fit_attempted']);by_year[year]=row
 signals=state['signals'];require(isinstance(signals,list)and len(signals)==len(meta['dates']),'Complete signal state audit')
 keys={'signal_index','signal_date','year','available','reason','first_invalid_query_date','filtered_probability','memoryless_probability','original_filtered_probability','original_memoryless_probability'}
 for i,(row,day)in enumerate(zip(signals,meta['dates'])):
  require(set(row)==keys and type(row['signal_index'])is int and row['signal_index']==i and row['signal_date']==day,'State signal identity')
  year=pd.Timestamp(day).year;require(type(row['year'])is int and row['year']==year and type(row['available'])is bool,'State signal types')
  values=(row['filtered_probability'],row['memoryless_probability']);original=(row['original_filtered_probability'],row['original_memoryless_probability'])
  if row['available']:
   require(by_year[year]['status']=='available'and row['reason']is None,'Available signal requires available annual model')
   require(all(type(v)in(int,float)and math.isfinite(v)and 0.<=v<=1. for v in values+original),'State probability range')
   quiet=by_year[year]['quiet_state_mask']
   for actual,prior in zip(values,original):
    expected=1. if all(quiet)else 0. if not any(quiet)else prior if quiet[0]else 1.-prior
    h.assert_exact_tree(actual,expected)
  else:require(all(v is None for v in values+original),'Unavailable state exact missing probabilities')
 for label,obj in(('state',state),('feature',feature),('validation',validation)):
  require(type(obj.get('new_model_fits'))is int and obj['new_model_fits']==0,'No new model fits '+label)
  require(type(obj.get('reused_model_fits'))is int and obj['reused_model_fits']==count,'Reused model fit count parity '+label)
 require(feature.get('returns_or_nav_computed')is False and feature.get('original_arrays_unchanged')is True,'Feature preparation scope')
 return count


def check_outcomes(close,opening,orders,meta,read,original_state=None):
 dates=validate_orders(close,opening,orders,meta);results=read('RESULTS.json');require(set(results)==set(NAMES),'Three result arms')
 bounds=dict(early2016_2020=(meta['dates'][0],meta['ends'][52]),original2021_2026=(meta['dates'][53],meta['ends'][-1]),training2016_2023=(meta['dates'][0],meta['ends'][87]),later34=(meta['dates'][88],meta['ends'][-1]))
 errors=[];navs={};metrics={};details={};maxnav=0.
 for name in NAMES:
  require(set(results[name])=={x[2]for x in PATHS},'Three execution paths per arm')
  for execution,cost,key in PATHS:
   met,nav,trades,hold=replay(close,opening,orders[name],dates,execution,cost)
   prefix=name+'_'+key;compare_stats(results[name][key]['metrics'],met,errors,prefix+'/metrics')
   compare_stats(results[name][key]['slices'],{k:h.slice_stat(dates,nav,*span)for k,span in bounds.items()},errors,prefix+'/slices')
   daily=read(prefix+'_DAILY.json');require(all(set(x)=={'date','equity'}for x in daily)and [x['date']for x in daily]==[str(x.date())for x in dates],'Exact daily NAV dates')
   actual=np.asarray([x['equity']for x in daily],float);require(np.isfinite(actual).all()and (actual>0).all(),'Finite positive NAV')
   np.testing.assert_allclose(actual,nav,atol=1e-5,rtol=2e-11);maxnav=max(maxnav,float(np.max(np.abs(actual-nav))))
   actual_trades=read(prefix+'_TRADES.json');actual_hold=read(prefix+'_HOLDINGS.json')
   h.compare(normalize(actual_trades),normalize(trades),errors,prefix+'/trades');h.compare(normalize(actual_hold),normalize(hold),errors,prefix+'/holdings')
   allowed={r['date']if execution=='close'else str(close.index[close.index.get_loc(pd.Timestamp(r['date']))+1].date())for r in orders[name]}
   require(all(t['date']in allowed for t in actual_trades),'No intramonth trades')
   require([x['year']for x in met['annual_returns']]==sorted({int(x.year)for x in dates})and all(x['partial']is(x['year']in(2016,2026))for x in met['annual_returns']),'Annual coverage and partial flags')
   details[prefix]=dict(trades=len(trades),holding_periods=len(hold),fees_paid=met['fees_paid'])
   if execution=='close':navs[name]=nav;metrics[name]=met
  td=dates[dates<=pd.Timestamp(meta['ends'][87])];_,tn,_,_=replay(close.loc[:td[-1]],opening.loc[:td[-1]],orders[name][:88],td)
  np.testing.assert_allclose(tn,navs[name][:len(td)],atol=1e-5,rtol=2e-11)
 measured=[dict(name=name,**h.measures(dates,navs[name],meta))for name in NAMES]
 compare_stats(read('TRAIN_TRIALS.json'),measured,errors,'training');frozen=read('FROZEN.json');h.assert_exact_tree(frozen['selection'],select(measured))
 compare_stats(read('WALKFORWARD.json'),walkforward(dates,navs,meta),errors,'walkforward')
 months=monthly(orders,close,meta);compare_stats(read('MONTH_COMPARISON.json'),months,errors,'monthly');compare_stats(read('ATTRIBUTION.json'),attribution(months,metrics),errors,'attribution')
 validation=read('VALIDATION.json');require(validation['primary']==frozen['selection']['primary']and validation.get('pristine_oos')is False,'Validation selected result')
 state=read('STATE_AUDIT.json');fit_count=model_audit_counts(state,read('FEATURE_AUDIT.json'),validation,meta)
 if original_state is not None:verify_state_reuse(state,original_state)
 require(all(validation[k]is True for k in('original_arrays_unchanged','no_observer_writes','no_intramonth_trades'))and validation['independent_outcome_review_pending']is True,'Validation claims')
 return dict(passed=True,stage='evaluate',execution_paths_verified=9,daily_nav_trades_holdings_metrics_slices=True,first88_and_three_fold_selection_verified=True,annual_partial_flags_verified=True,monthly_difference_and_attribution_verified=True,first88_causal_replay_verified=True,selected_name=frozen['selection']['primary'],primary_cagr=metrics[frozen['selection']['primary']]['cagr'],primary_target_reached=metrics[frozen['selection']['primary']]['cagr']>=.7,max_compared_absolute_difference=max((x for _,x in errors),default=0.),max_nav_absolute_difference=maxnav,replay_details=details,producer_modules_imported=False,models_fit=0,protected_member_scan_repeated=False,producer_model_fits=0,reused_model_fits=fit_count,annual_model_fit_counts_verified=True,all_changed_months_reported=True)


def reuse_bindings(reader,inputs,pins):
 old=reader.root/'outputs/monthly_latent_state134'
 expected={'reused_plan':'PLAN.json','reused_cache':'CACHE.json','reused_state':'STATE_AUDIT.json','reused_data':'DATA.npz','reused_prepare':'INDEPENDENT_PREPARE.json','reused_completion':'COMPLETION.json','reused_outcome':'INDEPENDENT_OUTCOME.json'}
 for key,name in expected.items():require(inputs[key]==old/name,'Fixed reused artifact '+key)
 plan=reader.json(inputs['reused_plan']);require(plan.get('round')==134 and plan.get('status')=='REGISTERED'and plan.get('arms')==list(NAMES),'Frozen original plan')
 require(all(pins.get(path)==digest for path,digest in reader.verify(plan['hashes']).items()),'Original plan source closure')
 cache=reader.json(inputs['reused_cache']);require(set(cache)==set(CACHE_FILES),'Original cache schema')
 cached=reader.verify(cache,base=old);require(all(pins.get(path)==digest for path,digest in cached.items()),'Original cache closure')
 prep=reader.json(inputs['reused_prepare']);outcome=reader.json(inputs['reused_outcome']);completion=reader.json(inputs['reused_completion'])
 require(prep.get('passed')is True and prep.get('stage')=='prepare'and all(prep.get(k)is True for k in PREPARE_CHECKS),'Original independent prepare')
 require(outcome.get('passed')is True and outcome.get('stage')=='evaluate'and outcome.get('execution_paths_verified')==9,'Original independent outcome')
 require(prep.get('plan_sha256')==reader.sha(inputs['reused_plan'])and outcome.get('plan_sha256')==reader.sha(inputs['reused_plan']),'Original review PLAN identity')
 require(completion.get('round')==134 and completion.get('status')=='completed_independently_verified','Completed original round')
 for key,name in expected.items():
  if key=='reused_completion':continue
  ref=completion['artifacts'][name];require(reader.path(ref['path'])==inputs[key]and ref['sha256']==reader.sha(inputs[key]),'Completed original artifact '+key)
 for key in('reused_plan','reused_cache','reused_state','reused_data','reused_prepare'):
  require(outcome['artifact_hashes'].get(str(inputs[key]))==reader.sha(inputs[key]),'Original outcome binds reused artifact '+key)
 for key in('reused_plan','reused_cache','reused_state','reused_data'):
  require(prep['artifact_hashes'].get(str(inputs[key]))==reader.sha(inputs[key]),'Original prepare binds reused artifact '+key)
 return reader.json(inputs['reused_state'])

def bindings(reader,plan_sha):
 p=reader.root/STUDY;require(reader.sha(p/'PLAN.json')==plan_sha,'Exact registered PLAN')
 plan=reader.json(p/'PLAN.json');require(plan.get('round')==135 and plan.get('status')=='REGISTERED'and plan.get('arms')==list(NAMES)and plan.get('primary_challenger')==NAMES[2],'Registered three-arm protocol')
 require(reader.path(plan['output_directory'])==p,'Fixed research directory');pins=reader.verify(plan['hashes'])
 for key in('raw_source_hashes','independent_prepare_source_hashes'):
  mapping=reader.verify(plan[key]);require(all(pins.get(path)==digest for path,digest in mapping.items()),'Source closure in PLAN '+key)
 for file in source_files():require(pins.get(str(file))==reader.sha(file),'Outcome reviewer source in PLAN')
 require(plan.get('actual_feature_calculations_before_registration')==0 and plan.get('actual_returns_before_registration')==0 and plan.get('actual_model_fits_before_registration')==0,'Registered study provenance')
 keys={'proposal','meta','policy','original_orders','prices','transition','protected_manifest','reused_plan','reused_cache','reused_state','reused_data','reused_prepare','reused_completion','reused_outcome'}
 require(set(plan['inputs'])==keys,'Exact named inputs');inputs={}
 for key,ref in plan['inputs'].items():
  path=reader.path(ref['path']);require(pins.get(str(path))==ref['sha256']==reader.sha(path),'Named input binding');inputs[key]=path
 require(inputs['prices']==reader.root/PRICE and inputs['original_orders']==reader.root/'outputs/monthly_local_linear70/local_linear_ORDERS.json','Unchanged original price/order inputs')
 require(inputs['transition']==reader.root/'outputs/monthly_target70_research/TRANSITION.npz','Original transition input')
 reuse_bindings(reader,inputs,pins)
 cache=reader.json(p/'CACHE.json');require(set(cache)==set(CACHE_FILES),'Exact CACHE artifacts');cached=reader.verify(cache,base=p)
 prep=reader.json(p/'INDEPENDENT_PREPARE.json');require(prep.get('passed')is True and prep.get('stage')=='prepare'and all(prep.get(k)is True for k in PREPARE_CHECKS),'Independent prepare passed')
 require(prep.get('producer_modules_imported')is False and prep.get('returns_or_nav_computed')is False and type(prep.get('monthly_label_models_fit'))is int and prep['monthly_label_models_fit']==0,'Independent prepare scope')
 for key,name in(('plan_sha256','PLAN.json'),('cache_sha256','CACHE.json'),('orders_sha256','PRETRAIN_ORDERS.json')):require(prep[key]==reader.sha(p/name),'Prepare receipt binding '+key)
 require(reader.verify(prep['source_hashes'])==reader.verify(plan['independent_prepare_source_hashes']),'Prepare implementation closure')
 artifacts=reader.verify(prep['artifact_hashes']);required=dict(cached)
 required.update({str(p/name):reader.sha(p/name)for name in('PLAN.json','CACHE.json')})
 require(all(artifacts.get(path)==digest for path,digest in required.items()),'Complete prepare artifact coverage')
 allowed=dict(pins,**required);require(all(allowed.get(path)==digest for path,digest in artifacts.items()),'No unbound prepare artifacts')
 frozen=reader.json(p/'FROZEN.json');require(set(frozen['hashes'])==set(FROZEN_FILES),'Exact FROZEN input set');reader.verify(frozen['hashes'],base=p)
 require(frozen.get('pristine_oos')is False and frozen.get('all_diagnostic_orders_frozen')is True,'Frozen order/history scope')
 orders=reader.json(p/'PRETRAIN_ORDERS.json');h.assert_exact_tree(orders[NAMES[0]],reader.json(inputs['original_orders']))
 return p

def audit(root,plan_sha):
 require(len(plan_sha)==64 and all(c in '0123456789abcdef'for c in plan_sha),'SHA256 required')
 reader=Reader(root);p=bindings(reader,plan_sha);meta=reader.json(p/'META.json');orders=reader.json(p/'PRETRAIN_ORDERS.json');data=reader.npz(PRICE)
 require(meta==reader.json(reader.json(p/'PLAN.json')['inputs']['meta']['path']),'Original metadata unchanged')
 require(set(('dates','symbols','close','opening'))<=set(data),'Price payload arrays')
 index=pd.to_datetime(data['dates']);close=pd.DataFrame(data['close'],index=index,columns=data['symbols']);opening=pd.DataFrame(data['opening'],index=index,columns=data['symbols'])
 require(meta['dates'][0]=='2016-08-17'and meta['ends'][-1]=='2026-09-24','Fixed research horizon')
 original_state=reader.json(reader.json(p/'PLAN.json')['inputs']['reused_state']['path'])
 report=check_outcomes(close,opening,orders,meta,lambda name:reader.json(p/name),original_state=original_state)
 report.update(schema_version=1,baseline_local67_full_orders_unchanged=True,plan_sha256=plan_sha,cache_sha256=reader.sha(p/'CACHE.json'),orders_sha256=reader.sha(p/'PRETRAIN_ORDERS.json'),frozen_sha256=reader.sha(p/'FROZEN.json'),source_hashes={str(f):reader.sha(f)for f in source_files()},limitations=['Reused development history; not pristine out-of-sample validation.','Fixed historical membership, adjusted-price and execution-timing limitations are unchanged.','Protected inventory members are not rescanned by this outcome-only audit.'])
 for path,digest in list(reader.seen.items()):require(reader.sha(path)==digest,'Post-review artifact binding')
 report['artifact_hashes']=dict(reader.seen);return report

def main():
 cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--plan-sha',required=True);cli.add_argument('--evaluate-complete',action='store_true');cli.add_argument('--output',type=Path);args=cli.parse_args()
 require(args.evaluate_complete,'Actual audit only after completed evaluation')
 report=audit(ROOT,args.plan_sha);dest=args.output or ROOT/STUDY/'INDEPENDENT_OUTCOME.json'
 with dest.open('x')as out:json.dump(report,out,sort_keys=True,indent=2,allow_nan=False);out.write('\n')
 print(json.dumps(dict(passed=True,output=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),execution_paths=9)))
if __name__=='__main__':main()

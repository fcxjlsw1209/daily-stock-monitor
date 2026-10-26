"""Independent round142 outcome audit; actual market replay requires explicit CLI.

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
STUDY='outputs/monthly_formation142'
PRICE='outputs/monthly_2016_extension/PRICES.npz'
HELPER=ROOT/'work/monthly_ridge_support70_result_review.py'
HELPER_SHA='d0f053881cb94e49e7fff9eaaeaa9e260f0b3c66419f87819391c309cb7cb8ef'
if hashlib.sha256(HELPER.read_bytes()).hexdigest()!=HELPER_SHA:raise ValueError('Frozen independent account helper changed')
_spec=importlib.util.spec_from_file_location('formation142_independent_accounting',HELPER)
h=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(h)
NAMES=('incumbent_local67','formation_common_delete_mean','formation_common_delete_worst')
PATHS=(('close',0.,'same_close'),('open',0.,'next_open_zero'),('open',.0005,'next_open_cost'))
CACHE_FILES=('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json')
FROZEN_FILES=('PLAN.json','CACHE.json','PRETRAIN_ORDERS.json','TRAIN_TRIALS.json','WALKFORWARD.json','INDEPENDENT_PREPARE.json')
PREPARE_CHECKS=('original27_arrays_byte_identical','all_features_independently_reconstructed','all_three_score_matrices_and_ordered_choices_independently_verified','common_scenario_selector_and_no_double_episode_verified','zero_and_missing_exact_fallback','raw_prices_independently_verified','calendar_overlay_independently_verified','all_signal_prefixes_verified','formation_block_deletions_independently_verified')
PROPOSAL_SHA='13d6f8c27ff86a6a81fa54bcb53012db49163a95c98da2c7f8aeec80463f437b'

def require(ok,msg):
 if not ok:raise ValueError(msg)

def source_files():return [Path(__file__).resolve(),Path(__file__).with_name('test_monthly_formation142_outcome_review.py').resolve(),HELPER]

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
 values=[[row.get(a,{}).get(b)for a,b in fields]for row in rows]
 def guard(reference):
  candidate=values[2];baseline=values[reference]
  if not all(type(v)in(int,float)and math.isfinite(v)for group in(candidate,baseline)for v in group):return False
  cagr=candidate[0]>baseline[0]+1e-12
  return bool(cagr and all(x>=y-t for x,y,t in zip(candidate[1:],baseline[1:],(.05,.03,.025,.025))))
 checks={NAMES[i]:guard(i)for i in(0,1)};win=all(checks.values())
 return dict(primary=NAMES[2]if win else NAMES[0],admitted=win,guards=win,guards_against=checks)

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

def validate_months(months,meta):
 require(isinstance(months,list) and len(months)==len(meta['dates'])==len(meta['symbols'])==122,'Complete 122-month formation audit')
 keys={'signal_index','date','calm','horizon','ready','reason','blocks','original_finite_mask','selectors'}
 selector_keys={'arm','selected_indices','selected_keys','candidate_indices','candidate_keys','scenario_count','objective','fallback','reason'}
 for i,(row,day,pool) in enumerate(zip(months,meta['dates'],meta['symbols'])):
  require(isinstance(row,dict) and set(row)==keys,'Exact formation month fields')
  require(type(row['signal_index']) is int and row['signal_index']==i and row['date']==day,'Original month identity/order')
  require(type(row['calm']) is bool and type(row['ready']) is bool,'Formation states are booleans')
  horizon=252 if row['calm'] else 294
  require(type(row['horizon']) is int and row['horizon']==horizon,'Frozen state formation horizon')
  mask=row['original_finite_mask']
  require(isinstance(mask,list) and len(mask)==len(pool)==49 and all(type(v) is bool for v in mask),'Original finite-core mask')
  blocks=row['blocks'];require(isinstance(blocks,list),'Formation blocks list')
  if row['ready']:
   require(row['reason']=='ready' and len(blocks)==(horizon-42)//21,'Exact shared scenario count')
   previous=None
   for block in blocks:
    require(isinstance(block,dict) and set(block)=={'start_price_index','end_price_index','deleted_returns'},'Exact block fields')
    require(all(type(v) is int for v in block.values()) and block['start_price_index']>=0 and block['end_price_index']-block['start_price_index']==block['deleted_returns']==21,'21-return block endpoints')
    require(previous is None or block['start_price_index']==previous,'Contiguous oldest-first nonoverlapping blocks')
    previous=block['end_price_index']
   require(blocks[-1]['end_price_index']-blocks[0]['start_price_index']==horizon-42,'Complete formation deletion partition')
  else:
   require(row['reason'] in ('insufficient_formation_history','incomplete_originally_measurable_formation') and not blocks,'Unavailable formation has no scenarios')
  require(isinstance(row['selectors'],dict) and set(row['selectors'])==set(NAMES[1:]),'Both formation selectors audited')
  for name,arm in zip(NAMES[1:],('mean_scenario','worst_scenario')):
   item=row['selectors'][name]
   require(isinstance(item,dict) and set(item)==selector_keys and item['arm']==arm,'Exact selector fields and arm')
   require(type(item['scenario_count']) is int and item['scenario_count']==len(blocks),'Selector scenario count')
   for prefix in ('selected','candidate'):
    indices=item[prefix+'_indices'];names=item[prefix+'_keys']
    require(isinstance(indices,list) and len(indices)==len(set(indices)) and all(type(j) is int and 0<=j<49 for j in indices),'Unique selector indices')
    require(names==[pool[j]for j in indices],'Selector security keys follow original pool')
   require(len(item['candidate_indices'])<=11 and set(item['selected_indices'])<=set(item['candidate_indices']),'Original selection within at most top8 union original3')
   require(type(item['fallback']) is bool,'Selector fallback boolean')
   if item['fallback']:
    require(item['reason'] in ('original_fewer_than_three','missing_scenarios','unchanged_weighted_core','candidate_history_unavailable') and item['objective'] is None,'Exact available fallback reason')
    if item['reason']=='original_fewer_than_three':require(len(item['selected_indices']) in (0,2),'Two-or-cash fallback')
    if item['reason']=='missing_scenarios':require(not row['ready'],'Missing scenario fallback state')
    if item['reason']=='unchanged_weighted_core':require(row['ready'] and all(mask[j] for j in item['candidate_indices']),'Unchanged-core fallback needs valid scenarios')
    if item['reason']=='candidate_history_unavailable':require(row['ready'] and any(not mask[j] for j in item['candidate_indices']),'Candidate-history fallback must identify invalid candidate')
   else:
    require(row['ready'] and all(mask[j] for j in item['candidate_indices']) and item['reason']=='selected_by_'+arm and len(item['selected_indices'])==3 and type(item['objective']) in (int,float) and math.isfinite(item['objective']),'Active three-security finite objective')
  left,right=[row['selectors'][name] for name in NAMES[1:]]
  require(left['candidate_indices']==right['candidate_indices'] and left['candidate_keys']==right['candidate_keys'],'Both arms share the same ordered candidate set')
 return months


def monthly(orders,close,meta,months):
 validate_months(months,meta);rows=[]
 for i,day in enumerate(meta['dates']):
  returns={};held={}
  for name in NAMES:
   symbols=orders[name][i]['symbols'];held[name]=symbols
   returns[name]=math.fsum(float(close.at[pd.Timestamp(meta['ends'][i]),s])/float(close.at[pd.Timestamp(day),s])-1. for s in symbols)/len(symbols)if symbols else 0.
  for name in NAMES[1:]:
   item=months[i]['selectors'][name]
   require(item['selected_keys']==orders[name][i]['symbols'],'Audited selector matches actual ordered holdings')
   require(set(orders[NAMES[0]][i]['symbols'])<=set(item['candidate_keys']),'Candidate set retains original selected securities')
   if item['fallback']:h.assert_exact_tree(orders[name][i],orders[NAMES[0]][i])
   if len(orders[NAMES[0]][i]['symbols'])<3:require(item['fallback'] and item['reason']=='original_fewer_than_three','Original two-or-cash must bypass new objective')
  rows.append(dict(date=day,end=meta['ends'][i],split='early53'if i<53 else'recent35'if i<88 else'later34',holdings=held,returns=returns,formation_scenario=months[i]))
 return rows


def attribution(rows,metrics):
 comparisons={}
 for label,name,reference in (('worst_vs_incumbent',NAMES[2],NAMES[0]),('mean_vs_incumbent',NAMES[1],NAMES[0]),('worst_vs_mean',NAMES[2],NAMES[1])):
  changes=[];ordered=0
  for row in rows:
   ordered+=int(row['holdings'][reference]!=row['holdings'][name])
   if set(row['holdings'][reference])!=set(row['holdings'][name]):
    changes.append(dict(date=row['date'],split=row['split'],old=row['holdings'][reference],new=row['holdings'][name],guidance_events=[],formation_scenario=row['formation_scenario'],log_contribution=math.log1p(row['returns'][name])-math.log1p(row['returns'][reference])))
  positive=sorted((x['log_contribution']for x in changes if x['log_contribution']>0.),reverse=True);metric=metrics[name]
  comparisons[label]=dict(changed_months=len(changes),ordered_changed_months=ordered,changes=changes,undo_best1_cagr=((1+metric['total_return'])*math.exp(-math.fsum(positive[:1])))**(1/metric['years'])-1,undo_best3_cagr=((1+metric['total_return'])*math.exp(-math.fsum(positive[:3])))**(1/metric['years'])-1)
 return dict(comparisons=comparisons)

def feature_checks(feature,validation,meta):
 for label,obj in(('feature',feature),('validation',validation)):
  require(type(obj.get('new_model_fits'))is int and obj['new_model_fits']==0,'No model fits '+label)
 require(feature.get('returns_or_nav_computed')is False and feature.get('original_arrays_unchanged')is True and feature.get('raw_core_mask_and_rank_verified')is True,'Feature preparation scope')
 require(type(feature.get('original_months'))is int and feature['original_months']==122 and type(feature.get('original_width'))is int and feature['original_width']==49,'Original feature dimensions')
 require(type(feature.get('monthly_label_models_fit'))is int and feature['monthly_label_models_fit']==0,'No prediction-model fits')
 validate_months(feature['months'],meta)
 return 0


def verify_feature_matrices(arrays,feature,meta):
 months=validate_months(feature['months'],meta)
 specs={'formation_rank':((122,12,49),np.dtype('float64')),'formation_count':((122,),np.dtype('int64')),'formation_ready':((122,),np.dtype('bool')),'formation_valid':((122,49),np.dtype('bool'))}
 for key,(shape,dtype) in specs.items():
  require(key in arrays and arrays[key].shape==shape and arrays[key].dtype==dtype,'Formation array schema '+key)
 for i,row in enumerate(months):
  count=len(row['blocks'])
  require(arrays['formation_count'][i]==count and bool(arrays['formation_ready'][i])==row['ready'],'Scenario count/readiness matrix matches month')
  np.testing.assert_array_equal(arrays['formation_valid'][i],row['original_finite_mask'])
  ranks=arrays['formation_rank'][i]
  require(np.isnan(ranks[count:]).all(),'Unused scenario rows must be NaN')
  if row['ready']:
   active=ranks[:count];require(np.isfinite(active).all() and (active>=-.5).all() and (active<=.5).all(),'Finite bounded active scenario ranks')
   mask=arrays['formation_valid'][i]
   require((active[:,~mask]==0.).all(),'Original unavailable core remains neutral')
   if mask.any():require(np.allclose(active[:,mask].sum(axis=1),.5,rtol=0.,atol=1e-12),'Average-tie centered rank mass')
   else:require((active==0.).all(),'Empty core remains neutral')
 return 4


def check_outcomes(close,opening,orders,meta,read):
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
 feature=read('FEATURE_AUDIT.json');months=monthly(orders,close,meta,feature['months']);compare_stats(read('MONTH_COMPARISON.json'),months,errors,'monthly');compare_stats(read('ATTRIBUTION.json'),attribution(months,metrics),errors,'attribution')
 validation=read('VALIDATION.json');require(validation['primary']==frozen['selection']['primary']and validation.get('pristine_oos')is False,'Validation selected result')
 fit_count=feature_checks(feature,validation,meta)
 require(all(validation[k]is True for k in('original_arrays_unchanged','no_observer_writes','no_intramonth_trades'))and validation['independent_outcome_review_pending']is True,'Validation claims')
 return dict(passed=True,stage='evaluate',execution_paths_verified=9,total_monthly_orders_verified=366,formation_feature_arrays_verified=4,daily_nav_trades_holdings_metrics_slices=True,first88_and_three_fold_selection_verified=True,annual_partial_flags_verified=True,monthly_difference_and_attribution_verified=True,first88_causal_replay_verified=True,selected_name=frozen['selection']['primary'],primary_cagr=metrics[frozen['selection']['primary']]['cagr'],primary_target_reached=metrics[frozen['selection']['primary']]['cagr']>=.7,max_compared_absolute_difference=max((x for _,x in errors),default=0.),max_nav_absolute_difference=maxnav,replay_details=details,producer_modules_imported=False,models_fit=0,protected_member_scan_repeated=False,producer_model_fits=fit_count,zero_model_fit_count_verified=True,formation_months_verified=122,available_formation_months=sum(row['ready']for row in feature['months']),unavailable_formation_months=sum(not row['ready']for row in feature['months']),all_changed_months_reported=True)


def verify_diagnostic_start(reader,directory,plan_sha,audit_started):
 path=directory/'DIAGNOSE_STARTED.json'
 require(path.is_file(),'Diagnostic start marker required')
 marker=reader.json(path)
 require(isinstance(marker,dict)and set(marker)=={'plan_sha256','frozen_sha256','started_utc'},'Exact diagnostic start marker fields')
 require(marker['plan_sha256']==plan_sha and marker['frozen_sha256']==reader.sha(directory/'FROZEN.json'),'Diagnostic start marker hash binding')
 value=marker['started_utc']
 require(isinstance(value,str)and value.endswith(('+00:00','Z')),'Diagnostic start time must be UTC')
 try:started=datetime.fromisoformat(value[:-1]+'+00:00'if value.endswith('Z')else value)
 except ValueError:raise ValueError('Invalid diagnostic start UTC time')from None
 require(started.tzinfo is not None and started.utcoffset().total_seconds()==0 and started<audit_started,'Diagnostic start must precede audit in UTC')
 return marker


def bindings(reader,plan_sha,audit_started=None):
 if audit_started is None:audit_started=datetime.now(timezone.utc)
 p=reader.root/STUDY;require(reader.sha(p/'PLAN.json')==plan_sha,'Exact registered PLAN')
 plan=reader.json(p/'PLAN.json');require(plan.get('round')==142 and plan.get('status')=='REGISTERED'and plan.get('arms')==list(NAMES)and plan.get('primary_challenger')==NAMES[2],'Registered three-arm protocol')
 require(reader.path(plan['output_directory'])==p,'Fixed research directory');pins=reader.verify(plan['hashes'])
 for key in('raw_source_hashes','independent_prepare_source_hashes'):
  mapping=reader.verify(plan[key]);require(all(pins.get(path)==digest for path,digest in mapping.items()),'Source closure in PLAN '+key)
 for file in source_files():require(pins.get(str(file))==reader.sha(file),'Outcome reviewer source in PLAN')
 require(all(type(plan.get(k)) is int and plan[k]==0 for k in('actual_feature_calculations_before_registration','actual_returns_before_registration','actual_model_fits_before_registration')),'Registered study provenance')
 keys={'proposal','meta','policy','original_orders','prices','sessions','protected_manifest'}
 require(set(plan['inputs'])==keys,'Exact named inputs');inputs={}
 for key,ref in plan['inputs'].items():
  path=reader.path(ref['path']);require(pins.get(str(path))==ref['sha256']==reader.sha(path),'Named input binding');inputs[key]=path
 require(reader.sha(inputs['proposal'])==PROPOSAL_SHA,'Final prospective proposal binding')
 require(inputs['prices']==reader.root/PRICE and inputs['original_orders']==reader.root/'outputs/monthly_local_linear70/local_linear_ORDERS.json','Unchanged original price/order inputs')
 cache=reader.json(p/'CACHE.json');require(set(cache)==set(CACHE_FILES),'Exact CACHE artifacts');cached=reader.verify(cache,base=p)
 prep=reader.json(p/'INDEPENDENT_PREPARE.json');require(prep.get('passed')is True and prep.get('stage')=='prepare'and all(prep.get(k)is True for k in PREPARE_CHECKS),'Independent prepare passed')
 require(all(type(prep.get(k)) is int and prep[k]==v for k,v in [('orders_checked',366),('months_checked',122),('candidate_cells_checked',5978),('new_model_fits',0)]),'Independent prepare counts')
 require(prep.get('producer_modules_imported')is False and prep.get('returns_or_nav_computed')is False and type(prep.get('monthly_label_models_fit'))is int and prep['monthly_label_models_fit']==0,'Independent prepare scope')
 for key,name in(('plan_sha256','PLAN.json'),('cache_sha256','CACHE.json'),('orders_sha256','PRETRAIN_ORDERS.json')):require(prep[key]==reader.sha(p/name),'Prepare receipt binding '+key)
 require(reader.verify(prep['source_hashes'])==reader.verify(plan['independent_prepare_source_hashes']),'Prepare implementation closure')
 artifacts=reader.verify(prep['artifact_hashes']);required=dict(cached)
 required.update({str(p/name):reader.sha(p/name)for name in('PLAN.json','CACHE.json')})
 require(all(artifacts.get(path)==digest for path,digest in required.items()),'Complete prepare artifact coverage')
 allowed=dict(pins,**required);require(all(allowed.get(path)==digest for path,digest in artifacts.items()),'No unbound prepare artifacts')
 frozen=reader.json(p/'FROZEN.json');require(set(frozen['hashes'])==set(FROZEN_FILES),'Exact FROZEN input set');reader.verify(frozen['hashes'],base=p)
 require(frozen.get('pristine_oos')is False and frozen.get('all_diagnostic_orders_frozen')is True,'Frozen order/history scope')
 verify_diagnostic_start(reader,p,plan_sha,audit_started)
 orders=reader.json(p/'PRETRAIN_ORDERS.json');h.assert_exact_tree(orders[NAMES[0]],reader.json(inputs['original_orders']))
 return p

def audit(root,plan_sha):
 audit_started=datetime.now(timezone.utc)
 require(len(plan_sha)==64 and all(c in '0123456789abcdef'for c in plan_sha),'SHA256 required')
 reader=Reader(root);p=bindings(reader,plan_sha,audit_started);meta=reader.json(p/'META.json');orders=reader.json(p/'PRETRAIN_ORDERS.json');data=reader.npz(PRICE)
 require(meta==reader.json(reader.json(p/'PLAN.json')['inputs']['meta']['path']),'Original metadata unchanged')
 require(set(('dates','symbols','close','opening'))<=set(data),'Price payload arrays')
 index=pd.to_datetime(data['dates']);close=pd.DataFrame(data['close'],index=index,columns=data['symbols']);opening=pd.DataFrame(data['opening'],index=index,columns=data['symbols'])
 require(meta['dates'][0]=='2016-08-17'and meta['ends'][-1]=='2026-09-24','Fixed research horizon')
 report=check_outcomes(close,opening,orders,meta,lambda name:reader.json(p/name))
 report['formation_feature_arrays_verified']=verify_feature_matrices(reader.npz(p/'DATA.npz'),reader.json(p/'FEATURE_AUDIT.json'),meta)
 report.update(schema_version=1,round=142,proposal_sha256=PROPOSAL_SHA,diagnostic_start_marker_verified=True,diagnose_started_sha256=reader.sha(p/'DIAGNOSE_STARTED.json'),audit_started_utc=audit_started.isoformat(),baseline_local67_full_orders_unchanged=True,plan_sha256=plan_sha,cache_sha256=reader.sha(p/'CACHE.json'),orders_sha256=reader.sha(p/'PRETRAIN_ORDERS.json'),frozen_sha256=reader.sha(p/'FROZEN.json'),source_hashes={str(f):reader.sha(f)for f in source_files()},limitations=['Reused development history; not pristine out-of-sample validation.','Fixed historical membership, adjusted-price and execution-timing limitations are unchanged.','Protected inventory members are not rescanned by this outcome-only audit.','Both new arms replace the old final episode action; changes versus incumbent cannot be attributed solely to the worst shared-scenario objective.','Worst versus mean compares a common candidate set and common episode-selector replacement.','Original std63, eligibility, other score components and market state retain their declared recent data; only formation numerators exclude the latest42 sessions.','Novelty is limited to inspected sources; complete historical novelty is not certified.'])
 for path,digest in list(reader.seen.items()):require(reader.sha(path)==digest,'Post-review artifact binding')
 report['artifact_hashes']=dict(reader.seen);return report

def main():
 cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--plan-sha',required=True);cli.add_argument('--evaluate-complete',action='store_true');cli.add_argument('--output',type=Path);args=cli.parse_args()
 require(args.evaluate_complete,'Actual audit only after completed evaluation')
 report=audit(ROOT,args.plan_sha);dest=args.output or ROOT/STUDY/'INDEPENDENT_OUTCOME.json'
 with dest.open('x')as out:json.dump(report,out,sort_keys=True,indent=2,allow_nan=False);out.write('\n')
 print(json.dumps(dict(passed=True,output=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),execution_paths=9)))
if __name__=='__main__':main()

"""Independent nine-path accounting for the registered endpoint retest.

No producer model, runner, portfolio engine or selector is imported. The frozen
independent accounting implementation is reused; this phase fits zero models.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd
import monthly_paired_target143_outcome_review as base
import monthly_endpoint155_prepare_review as prep

ROOT=Path(__file__).resolve().parents[1]
STUDY='outputs/monthly_endpoint155'
PRICE='outputs/monthly_2016_extension/PRICES.npz'
NAMES=prep.NAMES
EXTRA=prep.EXTRA
CACHE_FILES=prep.CACHE_FILES
PREPARE_CHECKS=prep.CHECKS
OUTCOME_CHECKS=('daily_nav_trades_holdings_metrics_slices','first88_and_three_fold_selection_verified',
    'annual_partial_flags_verified','monthly_difference_and_attribution_verified',
    'first88_causal_replay_verified','model_fit_counts_verified','all_changed_months_reported',
    'diagnostic_start_marker_verified','baseline_local67_full_orders_unchanged')
FROZEN_FILES=base.FROZEN_FILES
PATHS=base.PATHS
h,require,Reader=base.h,base.require,base.Reader
verify_diagnostic_start=base.verify_diagnostic_start

def source_files(root=None):
    root=ROOT if root is None else Path(root).resolve()
    return sorted(set(prep.source_files(root)+[root/'work'/n for n in (
        'monthly_endpoint155_outcome_review.py','test_monthly_endpoint155_outcome_review.py',
        'monthly_paired_target143_outcome_review.py','test_monthly_paired_target143_outcome_review.py',
        'monthly_ridge_support70_result_review.py')]))
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

def select(rows,open_rows):
 fields=(('training','cagr'),('training','sharpe'),('training','max_drawdown'),('early','cagr'),('recent','cagr'))
 def checks(group,opening):
  require(isinstance(group,list)and len(group)==3 and [r.get('name')for r in group]==list(NAMES),'Three exact selection arms')
  values=[[row.get(a,{}).get(b)for a,b in fields]for row in group]
  def guard(reference):
   candidate=values[2];baseline=values[reference]
   if not all(type(v)in(int,float)and math.isfinite(v)for pair in(candidate,baseline)for v in pair):return False
   cagr=candidate[0]>=baseline[0]-1e-12 if opening and reference==0 else candidate[0]>baseline[0]+1e-12
   return bool(cagr and all(x>=y-t for x,y,t in zip(candidate[1:],baseline[1:],(.05,.03,.025,.025))))
  return {NAMES[i]:guard(i)for i in(0,1)}
 close_checks=checks(rows,False);open_checks=checks(open_rows,True);win=all(close_checks.values())and all(open_checks.values())
 return dict(primary=NAMES[2]if win else NAMES[0],admitted=win,guards=win,guards_against=close_checks,next_open_guards_against=open_checks,mechanism_confirmed=all(open_checks.values()))

def rr_statistics(rr,start,end):
 # Same fixed zero-riskfree and partial-year convention as prior independent audits.
 rr=np.asarray(rr,float);require(rr.ndim==1 and len(rr)>1 and np.isfinite(rr).all()and (rr>-1).all(),'Valid return vector')
 years=(pd.Timestamp(end)-pd.Timestamp(start)).days/365.2425;require(years>0.,'Positive elapsed years')
 nav=np.r_[1.,np.cumprod(1+rr)];sd=np.std(rr,ddof=1)
 return dict(cagr=float(nav[-1]**(1/years)-1),sharpe=float(np.mean(rr)/sd*np.sqrt(252))if sd>1e-14 else 0.,max_drawdown=float(np.min(nav/np.maximum.accumulate(nav)-1)),total_return=float(nav[-1]-1))

def walkforward(dates,navs,open_navs,meta):
 folds=[];stitched=[];control=[]
 for n,end in((36,60),(60,76),(76,88)):
  opened=[dict(name=name,**h.measures(dates,open_navs[name],meta,n))for name in NAMES]
  pick=select([dict(name=name,**h.measures(dates,navs[name],meta,n))for name in NAMES],opened)
  folds.append(dict(n=n,end=end,selection=pick,open_training=opened))
  lo=dates.get_loc(pd.Timestamp(meta['dates'][n]));hi=dates.get_loc(pd.Timestamp(meta['ends'][end-1]))
  for out,name in((stitched,pick['primary']),(control,'incumbent_local67')):
   nav=navs[name];out.extend((nav[1:]/nav[:-1]-1.)[lo:hi])
 return dict(folds=folds,pristine_oos=False,metrics=rr_statistics(stitched,meta['dates'][36],meta['ends'][87]),incumbent=rr_statistics(control,meta['dates'][36],meta['ends'][87]))


def validate_months(months,meta):
    require(type(months) is list and len(months)==len(meta['dates'])==len(meta['symbols'])==122,'All122 rank rows')
    keys={'signal_index','date','shared_query_population','available','initial_partial_fallback','unchanged_return_correction','unchanged_loss_correction'}
    for i,row in enumerate(months):
        require(set(row)==keys and type(row['signal_index']) is int and row['signal_index']==i and row['date']==meta['dates'][i],'Ordered rank audit')
        count=row['shared_query_population'];require(type(count) is int and 0<=count<=49,'Shared query count')
        require(all(type(row[k]) is bool for k in keys-{'signal_index','date','shared_query_population'}),'Exact rank booleans')
        require(row['available']==(count>0),'Available count equality')
        require(row['initial_partial_fallback']==(row['date']=='2016-08-17'),'Only initial partial fallback')
        if row['initial_partial_fallback']:require(count==0,'Initial partial unavailable')
        if count<=1:require(row['unchanged_return_correction'] and row['unchanged_loss_correction'],'At most one query means zero rank')
    return months

def monthly(orders,close,meta,months):
    validate_months(months,meta);rows=[]
    for i,day in enumerate(meta['dates']):
        returns,held={},{}
        for name in NAMES:
            names=orders[name][i]['symbols'];held[name]=names
            returns[name]=math.fsum(float(close.at[pd.Timestamp(meta['ends'][i]),s])/float(close.at[pd.Timestamp(day),s])-1. for s in names)/len(names) if names else 0.
        for name,key in zip(NAMES[1:],('unchanged_return_correction','unchanged_loss_correction')):
            if months[i][key]:h.assert_exact_tree(orders[name][i],orders[NAMES[0]][i])
        rows.append(dict(date=day,end=meta['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34',holdings=held,returns=returns,endpoint_target=months[i]))
    return rows
def attribution(rows,metrics):
 comparisons={}
 for label,name,reference in (('loss_vs_incumbent',NAMES[2],NAMES[0]),('return_vs_incumbent',NAMES[1],NAMES[0]),('loss_vs_return',NAMES[2],NAMES[1])):
  changes=[];ordered=0
  for row in rows:
   ordered+=int(row['holdings'][reference]!=row['holdings'][name])
   if set(row['holdings'][reference])!=set(row['holdings'][name]):
    changes.append(dict(date=row['date'],split=row['split'],old=row['holdings'][reference],new=row['holdings'][name],guidance_events=[],endpoint_target=row['endpoint_target'],log_contribution=math.log1p(row['returns'][name])-math.log1p(row['returns'][reference])))
  positive=sorted((x['log_contribution']for x in changes if x['log_contribution']>0.),reverse=True);metric=metrics[name]
  comparisons[label]=dict(changed_months=len(changes),ordered_changed_months=ordered,changes=changes,undo_best1_cagr=((1+metric['total_return'])*math.exp(-math.fsum(positive[:1])))**(1/metric['years'])-1,undo_best3_cagr=((1+metric['total_return'])*math.exp(-math.fsum(positive[:3])))**(1/metric['years'])-1)
 return dict(comparisons=comparisons)


def model_checks(model,meta):
    for key,value in dict(alpha=10.,half_life_months=12.,lookback_years=5,minimum_mature_months=18,local_model_fits=0,
                         training_only_standardization=True,calibrated_probabilities=False,target_clipping=False,target_centering=False).items():
        require(type(model.get(key)) is type(value) and model[key]==value,'Model contract '+key)
    annual=model.get('annual');years=sorted({pd.Timestamp(d).year for d in meta['dates']})
    require(type(annual) is list and [r.get('year') for r in annual]==years,'All annual model years')
    fitted=0
    for row in annual:
        year=row['year'];cutoff='%04d-12-31'%(year-1)
        require(type(year) is int and row['cutoff']==cutoff and row['lookback_exclusive_start']=='%04d-12-31'%(year-6),'Year cutoff and lower bound')
        require(row['status'] in ('fit','cold_start','no_queries'),'Model status')
        for k in ('training_rows','training_months','prediction_rows','model_fits'):
            require(type(row.get(k)) is int and row[k]>=0,'Model count '+k)
        ids=row['training_indices'];require(type(ids) is list and ids==sorted(set(ids)) and all(type(x) is int and 0<=x<8820 for x in ids) and len(ids)==row['training_rows'],'Unique full training row IDs')
        require(row['training_indices_sha256']==prep.digest(np.asarray(ids,dtype=int)),'Training indices hash')
        last=row['latest_label_end'];require(last is None or pd.Timestamp(last)<=pd.Timestamp(cutoff),'Maturity before cutoff')
        isfit=row['status']=='fit';require(row['model_fits']==(2 if isfit else 0),'Two models per fit year')
        require((row['training_months']<18)==(row['status']=='cold_start'),'Fixed18 month minimum')
        if isfit:
            require(row['prediction_rows']>0 and row['shared_rows_weights_scalers'] is True,'Shared fit cohort')
            require(type(row['weight_sum']) is float and math.isfinite(row['weight_sum']) and abs(row['weight_sum']-len(ids))<1e-9,'Normalized row weights')
            for key in ('weights_sha256','scaler_mean_sha256','scaler_scale_sha256'):
                require(type(row.get(key)) is str and len(row[key])==64,'Fitted hashes')
            require(set(row['target_sha256'])=={'return_target','loss_target'},'Both target hashes')
            fitted+=2
        elif row['status']=='no_queries':require(row['prediction_rows']==0,'No query status')
    require(type(model.get('new_pooled_model_fits')) is int and model['new_pooled_model_fits']==fitted,'Total producer model fits')
    return fitted

def verify_feature_matrices(arrays,feature,meta):
    validate_months(feature['months'],meta)
    require(all(arrays[k].shape==(122,49) and arrays[k].dtype==(np.dtype(bool) if k=='common_available' else np.dtype('float64')) for k in EXTRA),'Five exact array axes/types')
    require(arrays['eligible'].shape==(122,49) and arrays['eligible'].dtype==bool and not (arrays['common_available']&~arrays['eligible']).any(),'Original eligible query subset')
    expected=prep.ranked_arrays(arrays,(122,49))
    for key in EXTRA[2:]:np.testing.assert_array_equal(arrays[key],expected[key])
    h.assert_exact_tree(feature,prep.rank_audit(arrays,meta))
    return 5
def check_outcomes(close,opening,orders,meta,read):
 dates=validate_orders(close,opening,orders,meta);results=read('RESULTS.json');require(set(results)==set(NAMES),'Three result arms')
 bounds=dict(early2016_2020=(meta['dates'][0],meta['ends'][52]),original2021_2026=(meta['dates'][53],meta['ends'][-1]),training2016_2023=(meta['dates'][0],meta['ends'][87]),later34=(meta['dates'][88],meta['ends'][-1]))
 errors=[];open_navs={};navs={};metrics={};details={};maxnav=0.
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
   if key=='next_open_cost':open_navs[name]=nav
  td=dates[dates<=pd.Timestamp(meta['ends'][87])]
  for execution,cost,full in(('close',0.,navs[name]),('open',.0005,open_navs[name])):
   _,tn,_,_=replay(close.loc[:td[-1]],opening.loc[:td[-1]],orders[name][:88],td,execution,cost)
   np.testing.assert_allclose(tn,full[:len(td)],atol=1e-5,rtol=2e-11)
 measured=[dict(name=name,**h.measures(dates,navs[name],meta))for name in NAMES]
 opened=[dict(name=name,**h.measures(dates,open_navs[name],meta))for name in NAMES]
 compare_stats(read('TRAIN_TRIALS.json'),measured,errors,'training')
 compare_stats(read('TRAIN_OPEN_TRIALS.json'),opened,errors,'open_training')
 frozen=read('FROZEN.json');h.assert_exact_tree(frozen['selection'],select(measured,opened))
 compare_stats(read('WALKFORWARD.json'),walkforward(dates,navs,open_navs,meta),errors,'walkforward')
 feature=read('RANK_AUDIT.json');months=monthly(orders,close,meta,feature['months'])
 compare_stats(read('MONTH_COMPARISON.json'),months,errors,'monthly')
 compare_stats(read('ATTRIBUTION.json'),attribution(months,metrics),errors,'attribution')
 validation=read('VALIDATION.json')
 require(validation['primary']==frozen['selection']['primary']and validation.get('pristine_oos')is False,'Validation selected result')
 h.assert_exact_tree(validation['selection'],frozen['selection'])
 fit_count=model_checks(read('MODEL_AUDIT.json'),meta)
 require(all(validation[k]is True for k in('original_arrays_unchanged','no_observer_writes','no_intramonth_trades'))and validation['independent_outcome_review_pending']is True,'Validation claims')
 return dict(passed=True,stage='evaluate',execution_paths_verified=9,total_monthly_orders_verified=366,endpoint_forecast_arrays_verified=5,daily_nav_trades_holdings_metrics_slices=True,first88_and_three_fold_selection_verified=True,annual_partial_flags_verified=True,monthly_difference_and_attribution_verified=True,first88_causal_replay_verified=True,selected_name=frozen['selection']['primary'],primary_cagr=metrics[frozen['selection']['primary']]['cagr'],primary_target_reached=metrics[frozen['selection']['primary']]['cagr']>=.7,max_compared_absolute_difference=max((x for _,x in errors),default=0.),max_nav_absolute_difference=maxnav,replay_details=details,producer_modules_imported=False,models_fit=0,protected_member_scan_repeated=False,producer_pooled_model_fits=fit_count,model_fit_counts_verified=True,endpoint_months_verified=122,available_endpoint_months=sum(row['available']for row in feature['months']),unavailable_endpoint_months=sum(not row['available']for row in feature['months']),all_changed_months_reported=True)



def bindings(reader,plan_sha,audit_started=None):
    started=audit_started or datetime.now(timezone.utc)
    p,plan,pins,_,_=prep.validate_plan(reader,reader.root/STUDY/'PLAN.json',plan_sha,stage='evaluate')
    sources={str(f):reader.sha(f) for f in source_files(reader.root)}
    require(reader.verify(plan['independent_outcome_source_hashes'])==sources and all(pins.get(k)==v for k,v in sources.items()),'Independent outcome source closure')
    cache=reader.json(p/'CACHE.json');cached=reader.verify(cache,p)
    prepare_receipt=reader.json(p/'INDEPENDENT_PREPARE.json')
    require(prepare_receipt.get('passed') is True and prepare_receipt.get('stage')=='prepare' and all(prepare_receipt.get(k) is True for k in PREPARE_CHECKS),'Independent prepare passed')
    for key,value in (('orders_checked',366),('months_checked',122),('candidate_cells_checked',5978),('training_rows_checked',8820),('forecast_arrays_verified',5),('prefixes_checked',123)):
        require(type(prepare_receipt.get(key)) is int and prepare_receipt[key]==value,'Prepare count '+key)
    require(prepare_receipt.get('producer_modules_imported') is False and prepare_receipt.get('returns_or_nav_computed') is False,'Prepare scope')
    require(type(prepare_receipt.get('new_model_fits')) is int and prepare_receipt['new_model_fits']==reader.json(p/'MODEL_AUDIT.json')['new_pooled_model_fits'],'Prepare/model fit count')
    for key,file in (('plan_sha256','PLAN.json'),('cache_sha256','CACHE.json'),('orders_sha256','PRETRAIN_ORDERS.json')):
        require(prepare_receipt[key]==reader.sha(p/file),'Prepare artifact binding '+key)
    require(reader.verify(prepare_receipt['source_hashes'])==reader.verify(plan['independent_prepare_source_hashes']),'Prepare source map')
    artifacts=reader.verify(prepare_receipt['artifact_hashes']);required=dict(cached,**{str(p/file):reader.sha(p/file) for file in ('PLAN.json','CACHE.json')})
    require(all(artifacts.get(k)==v for k,v in required.items()),'Complete prepared artifact coverage')
    allowed=dict(pins,**required);require(all(allowed.get(k)==v for k,v in artifacts.items()),'No unbound prepare artifacts')
    frozen=reader.json(p/'FROZEN.json');require(set(frozen['hashes'])==set(FROZEN_FILES),'Exact frozen phase files');reader.verify(frozen['hashes'],p)
    require(frozen.get('pristine_oos') is False and frozen.get('all_diagnostic_orders_frozen') is True,'Frozen diagnostics scope')
    verify_diagnostic_start(reader,p,plan_sha,started)
    orders=reader.json(p/'PRETRAIN_ORDERS.json');h.assert_exact_tree(orders[NAMES[0]],reader.json(plan['inputs']['original_orders']['path']))
    for name in NAMES[1:]:h.assert_exact_tree(orders[name][0],orders[NAMES[0]][0])
    prep.verify_arrays(reader.npz(prep.LOCAL+'/DATA.npz'),reader.npz(p/'DATA.npz'))
    return p

def audit(root,plan_sha):
    started=datetime.now(timezone.utc);reader=Reader(root);p=bindings(reader,plan_sha,started)
    plan,meta,orders=(reader.json(p/f) for f in ('PLAN.json','META.json','PRETRAIN_ORDERS.json'))
    require(meta==reader.json(plan['inputs']['meta']['path']) and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Original evaluation horizon/meta')
    data=reader.npz(plan['inputs']['prices']['path']);require(set(data)=={'dates','symbols','close','opening'},'Price arrays')
    axis=pd.to_datetime(data['dates']);close=pd.DataFrame(data['close'],index=axis,columns=data['symbols']);opening=pd.DataFrame(data['opening'],index=axis,columns=close.columns)
    report=check_outcomes(close,opening,orders,meta,lambda file:reader.json(p/file))
    report['endpoint_forecast_arrays_verified']=verify_feature_matrices(reader.npz(p/'DATA.npz'),reader.json(p/'RANK_AUDIT.json'),meta)
    validation=reader.json(p/'VALIDATION.json')
    require(validation.get('model_audit_sha256')==reader.sha(p/'MODEL_AUDIT.json') and validation.get('new_model_fits')==report['producer_pooled_model_fits'],'Validation model binding')
    require(validation.get('initial_partial_query')=='2016-08-17_incumbent_fallback' and validation.get('final_partial_evaluation')=='2026-09-24_not_a_training_label_endpoint','Disclosed partial periods')
    require(all(validation.get(k) is True for k in ('inherited_common_good_restrictions','linear_probability_not_calibrated','historical_novelty_not_certified')),'Research limitations preserved')
    report.update(schema_version=1,round=155,proposal_sha256=plan['inputs']['proposal']['sha256'],
        diagnostic_start_marker_verified=True,diagnose_started_sha256=reader.sha(p/'DIAGNOSE_STARTED.json'),audit_started_utc=started.isoformat(),
        baseline_local67_full_orders_unchanged=True,plan_sha256=plan_sha,cache_sha256=reader.sha(p/'CACHE.json'),orders_sha256=reader.sha(p/'PRETRAIN_ORDERS.json'),frozen_sha256=reader.sha(p/'FROZEN.json'),
        source_hashes={str(f):reader.sha(f) for f in source_files(root)},
        limitations=['Historical repetition is not excluded; reused history is not pristine out of sample.','Inherited common-good training restrictions remain.','Linear binary forecasts are rankings, not calibrated probabilities.','Final 2026-09-24 replay is partial and not a training-label endpoint.','Protected files are independently scanned at closure.'])
    for path,digest in list(reader.seen.items()):require(reader.sha(path)==digest,'Post-review artifact binding')
    report['artifact_hashes']=dict(reader.seen)
    return report

def synthetic_receipt(result,root=None):
    root=ROOT if root is None else Path(root).resolve();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    implementation=[root/'work'/('monthly_endpoint155_'+n+'.py') for n in ('model','runner','lifecycle')]
    inputs=[root/'work/monthly_endpoint155_proposal.json',root/prep.DESIGN]
    return dict(round=155,stage='preregistration_outcome_review',passed=result.wasSuccessful(),synthetic_only=True,
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        real_155_label_fit_or_return_calculations=0,proposal_sha256=sha(inputs[0]),
        source_hashes={str(p):sha(p) for p in source_files(root)},implementation_hashes={str(p):sha(p) for p in implementation},input_hashes={str(p):sha(p) for p in inputs},
        arms=list(NAMES),required_checks=list(OUTCOME_CHECKS),runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__))

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--plan-sha',required=True);cli.add_argument('--evaluate-complete',action='store_true');cli.add_argument('--output',type=Path);args=cli.parse_args()
    require(args.evaluate_complete,'Explicit completed evaluation required')
    report=audit(ROOT,args.plan_sha);out=args.output or ROOT/STUDY/'INDEPENDENT_OUTCOME.json'
    with out.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,output=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),execution_paths=9)))

if __name__=='__main__':main()

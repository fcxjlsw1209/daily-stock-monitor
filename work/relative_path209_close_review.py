"""Closed209 saved-evidence review; no model/strategy/raw-price imports or replay.

Only bound saved labels, forecasts, receipts, orders and exported NAV are read.
Pure frozen208 lineage/fee helpers and191 wealth-attribution helpers are reused.
Actual review is gated by prospective pins and an explicit root completion SHA.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_relative_path209'
PRIOR=ROOT/'outputs/monthly_volume_stability208'
CONTROL=ROOT/'outputs/monthly_relative_severity191'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
ARMS=('incumbent_local67','relative_path_shortfall_ridge','relative_terminal_shortfall_ridge')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
FEATURES=('candidate_prediction','control_prediction','candidate_rank','control_rank','common_available')
LABELS=('dates','ends','symbols','candidate_target','control_target','base_good','good')
TARGETS=('candidate_target','control_target')
PATH_COUNTS=('source_path_rows','source_price_observations','target_session_observations','missing_path_rows')
PREPARED_FILES={'FEATURES.npz','TRANSFORMS_DELTAS.npz','LABELS.npz','LABEL_AUDIT.json','MODEL_AUDIT.json','RANK_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','IDENTITY_AUDIT.json','CALENDAR.json','INDEPENDENT_PREPARE.json','CONTROL_PARITY.json'}
RESULT_FILES={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
CONTEXT='work/monthly_relative_path209_context.json'
CONTEXT_SHA='0029cb2f60b8afc8585d34789f7b18b827a7de392f04036c42823dcfeacdbe25'
SOURCE='work/relative_path209_SOURCE_REVIEW.json'
SOURCE_SHA='976f4b210176e3a4173bddd28e17004a951b0c5c9f0735929df180d0d6da177e'
TESTS='work/monthly_relative_path209_TESTS.json'
TESTS_SHA='844fd93778e2a7c26762566892061bf6a041441bb5b61d5af9a9e4a615531833'
CODE_REVIEW='work/relative_path209_CODE_REVIEW.json'
CODE_REVIEW_SHA='657fe73ed329b0f6a53caf788fcea4f73496b003e44fa0edb7219e9dc3f575c7'
CONTROL_PLAN_SHA='573e04e327544782a091ea3d9651809d4a6e5c2e55dd61adf33ea8618f99f97d'
CONTROL_PREPARED_SHA='3a2ca3c727d76ef708c1527a5b0d677e4988ff746575865dfe24266ea14bb069'
HELPERS={'work/volume_stability208_close_review.py':'33015a5e800b36cb15fd9ed2438da99bf9106593600ed8538aa2c584daa0c4ff','work/monthly_relative_severity191_report.py':'5e75e9cec58ef40a509496fbdfd7bac24e64f1600ef0641d51f7693c2e31bd88'}
CODE_FILES={f'work/{prefix}monthly_relative_path209_{kind}.py' for prefix in ('','test_') for kind in ('host','model','independent')}
FROZEN_COMPONENTS={
 'work/monthly_relative_path209_host.py':'a2eccc737dca4506ddd65c343d94e1b09ab368da49aa7c6f9f53c485c9f7ab36',
 'work/test_monthly_relative_path209_host.py':'80bb4e7a70141b7a5dc60e3790f0bf44d89a49e4ab845cd28d4a243a8008fe7a',
 'work/monthly_relative_path209_model.py':'f6b3ec8e72a12494eaaa5cdb0ed2a3d6c6240f00468f06578a6be5adfab7e696',
 'work/test_monthly_relative_path209_model.py':'b120886b5392c70e6310043bf30798ea1bf77e89cd913396a236600228c96a81',
 'work/monthly_relative_path209_independent.py':'43913c61dc1f649645c12f1de4c8c97014df83c326c2edd2492984f8101c2c82',
 'work/test_monthly_relative_path209_independent.py':'ceefc5000febda8bb55adc5fb0f29276c859afb61911900b2345b9d9a9a0dfc2'}
MODEL_FIT_ACCOUNTING='Main full preparation counts its actual paired annual Ridge fits once. Independent augmented-least-squares fits and five truncated-price prefix fits are diagnostic repeats reported separately, never additional strategy models. Existing Local67 fits are reused unchanged. Each main pair contains one new path-average-shortfall fit and one refitted terminal-shortfall control model required to reproduce191candidate exactly on the identical cohort.'
BUILD_FLAGS=('passed','audit_checked','independent_exact_rational_labels','endpoint_calendar_checked','shared_maturity_rows_weights_scalers_verified','annual_augmented_least_squares_verified','target_statistics_checked','severity_rounded_once_checked','full_official_path_checked','path_mean_rounded_once_checked','separate_target_bounds_checked','terminal_shortfall_checked','shared_query_mask_and_centered_ranks_verified','no_query_future_outcomes','initial_partial_fallback_checked')
DELTA_FLAGS=('passed','common_query_population_checked','negative_prediction_ranks_checked','average_ties_checked','initial_delta_zero','common_available_only','no_new_eligibility_filter','missing_exact_zero')


def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def integer(x,label):require(type(x) is int and x>=0,label);return x

def helpers():
    answer=[]
    for i,(name,pin) in enumerate(HELPERS.items()):
        path=ROOT/name;require(sha(path)==pin,'Frozen reporting helper changed')
        spec=importlib.util.spec_from_file_location('relative_path209_saved_helper_'+str(i),path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);answer.append(m)
    h,p=answer;h.OUT=OUT;h.PRIOR=PRIOR;h.ARMS=ARMS
    return h,p


def exact_array(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    require(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),label+' dtype/shape/bytes')


def zero_fee_evidence(result,trades,holdings):
    require(result['metrics']['fees_paid']==0.,'Zero-cost full-path fees')
    require(all(type(row['cost']) in (int,float) and row['cost']==0. for row in trades),'Zero-cost fill fees')
    require(all(type(row['fee']) in (int,float) and row['fee']==0. for row in holdings),'Zero-cost period fees')
    return dict(fees_paid=0.,fill_rows=len(trades),holding_periods=len(holdings),all_saved_fees_exact_zero=True)


def selection(metrics,orders):
    h,_=helpers();changed=lambda ref:[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ref][i]['symbols'])]
    first,control=changed(ARMS[0]),changed(ARMS[2])
    gates={}
    for label,ref,gain in (('incumbent',ARMS[0],.005),('relative_terminal',ARMS[2],0.)):
        gates['close_vs_'+label]=h.guard(metrics[ARMS[1]]['same_close_zero'],metrics[ref]['same_close_zero'],gain)
        gates['open_vs_'+label]=h.guard(metrics[ARMS[1]]['next_open_cost'],metrics[ref]['next_open_cost'],0.)
    for key,ids in (('changed_support',first),('changed_support_vs_control',control)):
        gates[key]=len(ids)>=3 and any(i<53 for i in ids) and any(i>=53 for i in ids)
    ok=all(gates.values())
    return dict(primary=ARMS[1] if ok else ARMS[0],admitted=ok,guards=gates,changed_months=first,changed_months_vs_control=control,first88_only=True,later34_used=False)


def label_history(labels,audit,sessions):
    require(set(labels)==set(LABELS),'Seven saved label arrays')
    d,e=np.asarray(labels['dates']),np.asarray(labels['ends']);n=len(d)
    require(d.dtype==e.dtype==np.dtype('datetime64[D]') and e.shape==(n,) and np.all(e>d),'Label date axes')
    require(all(np.asarray(v).shape==(n,) for v in labels.values()),'Label shapes')
    base,good=labels['base_good'],labels['good']
    require(base.dtype==good.dtype==np.dtype(bool) and not np.any(good&~base),'Shared cohort flags')
    ys=[labels[k] for k in TARGETS]
    require(all(y.dtype==np.float64 and not np.isinf(y).any() and np.all((y[np.isfinite(y)]>=0)&(y[np.isfinite(y)]<=1)) for y in ys),'Separate continuous target bounds')
    complete=np.isfinite(ys[0]);require(np.array_equal(complete,np.isfinite(ys[1])) and not np.any(complete&~base),'Common full-path target support')
    require(audit['targets']==list(TARGETS) and audit['rows']==n and audit['base_rows']==int(base.sum()) and audit['good_rows']==int(good.sum()),'Label audit coverage')
    for key in ('exact_binary_float_rational','rounded_targets_once','complete_positive_finite_paths','target_bounds_independent','inherited_common_good_retained','no_endpoint_shortening'):require(audit[key] is True,'Label proof '+key)
    require(audit['clipped'] is False and audit['centered'] is False and audit['calendar_completeness_certified_by_this_function'] is False,'Label transformation/scope')
    require(audit['relative_target_definitions']==dict(candidate='mean_d max(0,1-R_d)',control='max(0,1-R_end)') and audit['maturity']=='next official calendar month-end close','Label definitions')
    axis=np.asarray(sessions,dtype='datetime64[D]');require(len(axis)>0 and np.all(axis[1:]>axis[:-1]),'Official calendar')
    months={x.astype('datetime64[M]'):x for x in axis};checks=audit['path_checks'];require(len(checks)==n,'Every source row checked')
    counts=dict(source_path_rows=int(complete.sum()),source_price_observations=0,target_session_observations=0,missing_path_rows=int(base.sum()-complete.sum()))
    expected_good=complete.copy();records=[]
    for day in np.unique(d):
        ids=np.flatnonzero(d==day);end=e[ids[0]]
        require(np.all(e[ids]==end) and months[day.astype('datetime64[M]')]==day and months[day.astype('datetime64[M]')+1]==end,'Official next-natural-month endpoints')
        position=np.searchsorted(axis,end);require(position+1<len(axis) and axis[position+1].astype('datetime64[M]')>end.astype('datetime64[M]'),'Verified following session')
        window=axis[(axis>=day)&(axis<=end)];L=len(window)-1;count=int(complete[ids].sum())
        if count<2:expected_good[ids]=False
        records.append(dict(signal_date=str(day),label_end=str(end),base_rows=int(base[ids].sum()),endpoint_rows=count,good_rows=count if count>=2 else 0,status='complete' if count>=2 else 'fewer_than_two_shared_rows',complete_path_rows=count,source_dates=window.astype(str).tolist(),target_dates=window[1:].astype(str).tolist(),target_session_count=L))
        for i in ids:
            row=checks[i]
            require(row['row_index']==int(i) and row['signal_date']==str(day) and row['label_end']==str(end) and row['symbol']==str(labels['symbols'][i]),'Per-row path identity')
            require(row['base_good']==bool(base[i]) and row['path_complete']==bool(complete[i]) and row['target_session_count']==L,'Per-row path support')
            if complete[i]:
                require(row['status']=='complete' and all(row[k]==[] for k in ('missing_official_sessions','invalid_stock_dates','invalid_spy_dates')),'Complete path evidence')
                counts['source_price_observations']+=2*(L+1);counts['target_session_observations']+=L
            else:require(row['status']!='complete','Incomplete status')
    require(np.array_equal(good,expected_good) and audit['months']==records,'Common min-two cohort/month evidence')
    require(all(type(audit[k]) is int and audit[k]==v for k,v in counts.items()),'Path observation counters')
    return counts


def model_history(labels,values,audit,meta):
    d,e=labels['dates'],labels['ends'];good=labels['good'];qd=np.array(meta['dates'],dtype='datetime64[D]');mask=values['common_available']
    require(audit['alpha']==10. and audit['half_life_months']==12. and audit['lookback_years']==5 and audit['minimum_mature_months']==18,'Fixed annual Ridge')
    for key in ('training_only_standardization',):require(audit[key] is True,'Training standardization')
    require(audit['calibrated_probabilities'] is False and audit['target_clipping'] is False and audit['target_centering'] is False and audit['local_model_fits']==0,'Model meaning')
    require(audit['shared_training_mask_sha256']==digest(good),'Training mask hash')
    annual=audit['annual'];years=sorted(set(int(str(x)[:4]) for x in qd));require([row['year'] for row in annual]==years,'Annual query years')
    fits=0
    for row in annual:
        year=row['year'];cut=np.datetime64(str(year-1)+'-12-31');low=np.datetime64(str(year-6)+'-12-31')
        ids=np.flatnonzero(good&(d>low)&(e<=cut));months=len(np.unique(d[ids].astype('datetime64[M]')))
        require(row['cutoff']==str(cut) and row['lookback_exclusive_start']==str(low) and row['training_indices']==ids.tolist(),'Annual causal row selection')
        require(row['training_rows']==len(ids) and row['training_months']==months and row['latest_label_end']==(str(e[ids].max()) if len(ids) else None),'Annual maturity/counts')
        require(row['training_indices_sha256']==digest(ids) and row['training_dates_sha256']==digest(d[ids]) and row['training_symbols_sha256']==digest(labels['symbols'][ids]),'Annual identity hashes')
        for key,func in (('target_positive_counts',lambda y:int(np.count_nonzero(y>0))),('target_sums',lambda y:float(y.sum())),('target_means',lambda y:float(y.mean()) if len(y) else None),('target_maxima',lambda y:float(y.max()) if len(y) else None)):
            require(row[key]=={k:func(labels[k][ids]) for k in TARGETS},'Annual target statistics '+key)
        status=row['status'];expected=2 if status=='fit' else 0
        require(status in ('fit','cold_start','no_queries') and row['model_fits']==expected,'Actual paired fit count')
        require((status=='cold_start')==(months<18),'Minimum18-month policy')
        observed=int(mask[np.array([int(str(x)[:4])==year for x in qd])].sum())
        if status=='fit':
            require(row['prediction_rows']==observed and observed>0 and row['shared_rows_weights_scalers'] is True and math.isfinite(row['weight_sum']) and abs(row['weight_sum']-len(ids))<=1e-10,'Paired query/weight mass')
            require(row['target_sha256']=={k:digest(labels[k][ids]) for k in TARGETS},'Annual target hashes')
            require(all(0<=row['weighted_target_means'][k]<=1 for k in TARGETS),'Weighted targets independently bounded')
        else:require(observed==0,'Unavailable annual forecasts')
        fits+=expected
    require(audit['new_pooled_model_fits']==fits,'Main fit count')
    return fits


def array_evidence(values,transforms,rank_audit,delta_check):
    require(set(values)==set(FEATURES),'Exact five forecasts')
    mask=values['common_available'];require(mask.dtype==bool and mask.ndim==2,'Common forecast grid')
    n,width=mask.shape;changed={}
    require(not n or not mask[0].any(),'Initial partial forecast fallback')
    require(set(transforms)=={prefix+arm for prefix in ('transform_','delta_') for arm in ARMS[1:]},'Exact transforms')
    for arm,short in zip(ARMS[1:],('candidate','control')):
        pred,rank=values[short+'_prediction'],values[short+'_rank'];require(pred.dtype==rank.dtype==np.float64 and pred.shape==rank.shape==mask.shape,'Forecast shape/type')
        require(not np.isinf(pred).any() and np.array_equal(np.isfinite(pred),mask),'Common finite prediction support')
        expected=np.zeros(mask.shape)
        for i in range(n):
            domain=-pred[i,mask[i]];m=len(domain)
            if m>1:expected[i,mask[i]]=[(sum(domain<x)+(sum(domain==x)-1)/2)/(m-1)-.5 for x in domain]
        require(np.array_equal(rank,expected),'Exact negative average-tie ranks')
        exact_array(transforms['transform_'+arm],rank,'Saved positive rank transform')
        delta=.01*rank
        if n:delta[0]=0.
        exact_array(transforms['delta_'+arm],delta,'Fixed .01 rank delta');require(np.all(abs(delta)<=.005),'Delta bound')
        changed[short]=int(np.count_nonzero(delta))
    require(rank_audit['common_counts']==mask.sum(axis=1).astype(int).tolist() and rank_audit['max_absolute_rank']==.5 and rank_audit['query_evaluation_ends_not_used_for_labels'] is True and rank_audit['non_monthend_fallback_indices']==([0] if n else []),'Ranking receipt')
    for key in DELTA_FLAGS:require(delta_check[key] is True,'Independent delta proof '+key)
    require(delta_check['dose']==.01 and delta_check['max_absolute_correction']==.005 and delta_check['changed_cells']==changed,'Independent delta counts')
    return dict(events=n,width=width,available_cells=int(mask.sum()),unavailable_cells=int((~mask).sum()),changed_cells=changed)


def fit_receipts(validation,prepared,closure,model,counts,meta):
    for key in BUILD_FLAGS:require(validation['model'][key] is True,'Independent model flag '+key)
    check=validation['model'];fits=model['new_pooled_model_fits']
    require(check['producer_modules_imported'] is False and check['computes_portfolio_returns'] is False and check['max_forecast_error']<=1e-12,'Independent model scope/tolerance')
    require(check['new_pooled_model_fits']==check['independent_model_fits']==fits and check['local_model_fits']==0,'Independent model count')
    require(check['events']==122 and check['symbols_per_event']==49 and check['label_arrays_checked']==7 and check['forecast_arrays_checked']==5 and check['annual_records_checked']==11,'Full build coverage')
    require(validation['prefixes']==123 and validation['feature_prefixes']==[0,1,53,88,122] and validation['prefix_prices_truncated'] is True and validation['prefix_training_inputs_truncated'] is True,'Physical prefixes')
    for key in ('original_orders_exact','original_arrays_unchanged','original_training_arrays_unchanged','source_cohort_exact'):require(validation[key] is True,'Original/source invariant '+key)
    prefixes=validation['prefix_model_fit_counts'];require([x['prefix'] for x in prefixes]==[0,1,53,88,122],'Five prefix fit records')
    for row in prefixes:
        n=row['prefix'];a=integer(row['producer_model_fits'],'Prefix fits');b=integer(row['independent_model_fits'],'Independent prefix fits')
        require(a==b and a%2==0,'Paired prefix fits')
        cutoff=None if n==0 else str(int(meta['dates'][n-1][:4])-1)+'-12-31'
        require(row['training_label_end_cutoff']==cutoff and integer(row['training_rows'],'Prefix rows')>=0 and integer(row['training_months'],'Prefix months')>=0,'Physical prefix cutoff')
        if n==0:require(a==row['training_rows']==row['training_months']==0,'Empty prefix')
    for record in (validation,prepared,closure):
        require(record['model_fit_accounting']==MODEL_FIT_ACCOUNTING,'Fit accounting definition')
        require(record['original_array_count']==27,'Original27')
        for key in ('new_model_fits','new_predictive_model_fits','new_pooled_model_fits','independent_diagnostic_model_fits'):require(type(record[key]) is int and record[key]==fits,'Main/independent model accounting '+key)
        require(record['prefix_model_fit_counts']==prefixes and record['prefix_diagnostic_model_fits']==sum(x['producer_model_fits'] for x in prefixes) and record['prefix_independent_diagnostic_model_fits']==sum(x['independent_model_fits'] for x in prefixes),'Separate repeated fit accounting')
        require(all(type(record[k]) is int and record[k]==v for k,v in counts.items()),'Saved path counts')
    require(all(check[k]==v for k,v in counts.items()),'Independent path counts')
    require(prepared['new_returns_computed'] is False,'No preparation returns')
    for key in ('passed','scores_and_choices_checked','one_original_episode_checked','initial_partial_fallback_checked'):require(validation['selection'][key] is True,'Independent selection proof')
    require(validation['selection']['ordered_choices_checked']==366,'Three-by122 selections')
    return dict(main_paired_annual_fits=fits,independent_diagnostic_fits=fits,prefix_diagnostic_fits=sum(x['producer_model_fits'] for x in prefixes),prefix_independent_diagnostic_fits=sum(x['independent_model_fits'] for x in prefixes),prefixes=prefixes)


def control_evidence(e,labels,values,orders,check):
    require(check['passed'] is True and check['control_orders_exact'] is True and check['arrays_dtype_shape_bytes_exact'] is True and check['incidence191_arrays_loaded'] is False and check['prior191_returns_read'] is False,'191 control evidence')
    require(check['prior_plan_sha256']==CONTROL_PLAN_SHA and check['prior_prepared_completion_sha256']==CONTROL_PREPARED_SHA,'191 reference pins')
    e.verify(CONTROL/'PLAN.json',CONTROL_PLAN_SHA)
    receipt=e.manifest(CONTROL/'prepared',CONTROL_PREPARED_SHA,CONTROL_PLAN_SHA,PREPARED_FILES)
    mappings=[('LABELS.npz',labels,{'dates':'dates','ends':'ends','symbols':'symbols','control_target':'candidate_target','base_good':'base_good','good':'good'}),('FEATURES.npz',values,{'control_prediction':'candidate_prediction','control_rank':'candidate_rank','common_available':'common_available'})]
    require(check['label_keys_checked']==list(mappings[0][2]) and check['forecast_keys_checked']==list(mappings[1][2]),'Complete control comparison keys')
    for file,current,keys in mappings:
        with np.load(CONTROL/'prepared'/file,allow_pickle=False) as old:
            for key,prior in keys.items():exact_array(current[key],old[prior],'191 candidate →209 control '+key)
    previous=e.load(CONTROL/'prepared/ORDERS.json',receipt['artifacts_sha256']['ORDERS.json'])
    require(orders[ARMS[2]]==previous['relative_severity_ridge'],'191 candidate orders exact')
    return dict(passed=True,reference='191 candidate',label_arrays_exact=6,forecast_arrays_exact=3,orders_exact=True,incidence191_arrays_loaded=False,prior191_returns_read=False)


def npz(e,path):
    e.verify(path)
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def final_pins():
    valid=lambda s:isinstance(s,str) and len(s)==64 and all(x in '0123456789abcdef' for x in s)
    require(all(valid(x) for x in (CONTEXT_SHA,SOURCE_SHA,TESTS_SHA,CODE_REVIEW_SHA,*FROZEN_COMPONENTS.values())) and set(FROZEN_COMPONENTS)==CODE_FILES,'Final source/test/review pins required before actual reads')


def review(completion_sha,check_only=False):
    final_pins();require(isinstance(completion_sha,str) and len(completion_sha)==64,'Explicit root completion SHA')
    require(not (OUT/'RESEARCH_REVIEW.json').exists() and not (OUT/'RESEARCH_REVIEW.md').exists(),'Write-once research review')
    h,p=helpers();e=h.Evidence();closure=e.load(OUT/'COMPLETION.json',completion_sha)
    require(closure['status']=='COMPLETED_INDEPENDENTLY_VERIFIED','Close must precede review')
    plan_sha=closure['plan_sha256'];plan=e.load(OUT/'PLAN.json',plan_sha);context=e.load(CONTEXT,CONTEXT_SHA)
    require(plan['schema']=='monthly_relative_path209_v1' and plan['contract']['arms']==list(ARMS) and plan['contract']['context']==context and plan['contract']['prospective_context_sha256']==CONTEXT_SHA and plan['contract']['dose']==.01 and plan['contract']['max_absolute_delta']==.005,'Exact209 contract')
    require(plan['contract']['candidate']==ARMS[1] and plan['contract']['control_nonpromotable']==ARMS[2],'Fixed candidate/control')
    require(plan['model_fit_accounting']==MODEL_FIT_ACCOUNTING,'Prospective model fit accounting')
    require(len(plan['raw_source_hashes'])==64,'Original raw64 pins')
    for mapping in (plan['input_sha256'],plan['raw_source_hashes']):
        for name,pin in mapping.items():
            path=h.input_path(name,pin);require(path!=CENTRAL and plan['input_sha256'].get(str(path))==pin,'Immutable complete source pins');e.verify(path,pin)
    for key in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration','actual_new_labels_before_registration','actual_new_returns_before_registration'):require(type(plan[key]) is int and plan[key]==0,'Prospective registration boundary')
    before=e.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256']);require(len(before['completed'])==206 and sum(x['trials'] for x in before['completed'])==1135,'Predecessor counters')
    tests=e.load(TESTS,TESTS_SHA);code=e.load(CODE_REVIEW,CODE_REVIEW_SHA);source=e.load(SOURCE,SOURCE_SHA)
    require(all(plan['input_sha256'][str(ROOT/k)]==v for k,v in ((TESTS,TESTS_SHA),(CODE_REVIEW,CODE_REVIEW_SHA),(SOURCE,SOURCE_SHA))) and plan['numerical_code_review']['sha256']==CODE_REVIEW_SHA and plan['source_review_binding']['sha256']==SOURCE_SHA,'Formal proof bindings')
    require(tests['passed'] is True and tests['synthetic_only'] is True and integer(tests['tests'],'Formal tests')==134 and code['passed'] is True and code['unresolved_blockers']==[],'Formal synthetic/code evidence')
    require(code['context_sha256']==source['context_sha256']==CONTEXT_SHA and code['code_sha256']==FROZEN_COMPONENTS,'Exact reviewed six-file closure')
    require(tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Pinned runtime')
    for file,pin in FROZEN_COMPONENTS.items():require(tests['code_sha256'][str(ROOT/file)]==plan['input_sha256'][str(ROOT/file)]==pin,'Tested numerical source');e.verify(file,pin)
    for mapping in (code['input_sha256'],tests['code_sha256'],source['source_sha256']):
        for name,pin in mapping.items():require(h.input_path(name,pin)!=CENTRAL,'No mutable review input');e.verify(name,pin)
    for key in ('actual_features_computed','actual_model_fits_computed','actual_portfolio_returns_computed','production_files_modified','strategy_registered_by_reviewer'):require(code[key] is False,'Code review scope')
    require(source['passed'] is True and source['source_only'] is True and source['raw_source_hashes']==plan['raw_source_hashes'],'Source-only review')
    for key in ('actual_features_computed','actual_labels_computed','actual_model_fits_computed','actual_orders_computed','actual_returns_computed'):require(source[key] is False,'Source-only boundary')
    rc=e.manifest(OUT/'results',closure['result_completion_sha256'],plan_sha,RESULT_FILES);require(rc['paths']==9 and rc['slices']==45,'9paths45slices')
    frozen=e.load(OUT/'selected/FROZEN.json',rc['frozen_sha256']);prepared=e.manifest(OUT/'prepared',frozen['prepared_sha256'],plan_sha,PREPARED_FILES)
    require(frozen['plan_sha256']==plan_sha and frozen['orders_sha256']==prepared['artifacts_sha256']['ORDERS.json'],'Selected frozen orders')
    training=e.load(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    saved={name:e.load(OUT/'prepared'/name) for name in PREPARED_FILES if name.endswith('.json')}
    meta,orders=saved['META.json'],saved['ORDERS.json'];validation=saved['INDEPENDENT_PREPARE.json']
    values=npz(e,OUT/'prepared/FEATURES.npz');labels=npz(e,OUT/'prepared/LABELS.npz');transforms=npz(e,OUT/'prepared/TRANSFORMS_DELTAS.npz')
    require(values['common_available'].shape==(122,49) and len(meta['dates'])==122,'Full209 query domain')
    counts=label_history(labels,saved['LABEL_AUDIT.json'],saved['CALENDAR.json']['sessions'])
    fits=model_history(labels,values,saved['MODEL_AUDIT.json'],meta)
    integration=array_evidence(values,transforms,saved['RANK_AUDIT.json'],validation['delta'])
    model_counts=fit_receipts(validation,prepared,closure,saved['MODEL_AUDIT.json'],counts,meta)
    check=validation['model'];require(check['training_rows']==len(labels['dates']) and check['good_training_rows']==int(labels['good'].sum()) and check['available_cells']==integration['available_cells'] and check['unavailable_cells']==integration['unavailable_cells'],'Independent all-row coverage')
    coverage=source['training_path_coverage'];require(coverage['same_good_as191'] is True and coverage['candidate191_good_rows']==int(labels['good'].sum()) and coverage['final_good_mask_sha256']==digest(labels['good']),'Admitted same cohort')
    require(plan['source_review_binding']['training_path_coverage']=={k:v for k,v in coverage.items() if k!='records'},'Registered compact path coverage')
    parity=saved['SOURCE_PARITY.json'];require(parity['source_cohort_exact'] is True and parity['source_review_binding']==plan['source_review_binding'] and parity['raw_source_hashes']==plan['raw_source_hashes'],'Source parity')
    require(saved['CONTROL_PARITY.json']==validation['control_parity'],'Repeated control receipt')
    control=control_evidence(e,labels,values,orders,saved['CONTROL_PARITY.json'])
    for file in ('META.json','local_linear_ORDERS.json'):
        path=ROOT/'outputs/monthly_local_linear70'/file;old=e.load(path,plan['input_sha256'][str(path)])
        require((meta if file=='META.json' else orders[ARMS[0]])==old,'Original identity/orders')
    results=e.load(OUT/'results/RESULTS.json');outcome=e.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    require(results['relative_path_shortfall_target'] is True and results['same_support_both_arms'] is True and results['observer_promotion'] is False and results['strict_pit'] is False and results['reused_sample'] is True,'Result interpretation')
    require(set(orders)==set(training)==set(results['arms'])==set(outcome['paths'])==set(ARMS),'Exact three arms')
    require(results['selection']==frozen['selection']==selection(training,orders),'First88 selection exact')
    require(outcome['passed'] is True and outcome['independent_accounts'] is True and outcome['independent_of_production_model_and_engine'] is True and outcome['checked_paths']==9 and outcome['checked_slices']==45 and outcome['terminal_liquidation'] is False,'Independent outcome evidence')
    e.verify('work/ptr169_results_independent.py',outcome['accounting_source_sha256'])
    summaries={};slices=groups=0;zero_fees={}
    for arm in ARMS:
        require(len(orders[arm])==122 and set(results['arms'][arm])==set(outcome['paths'][arm])==set(PATHS),'Full three-path coverage')
        require([o['date'] for o in orders[arm]]==meta['dates'] and [o['end'] for o in orders[arm]]==meta['ends'],'Saved order calendar')
        summaries[arm]={}
        for path in PATHS:
            actual,ind=results['arms'][arm][path],outcome['paths'][arm][path]
            require(ind['passed'] is True and ind['monthly_executions']==122 and set(actual['slices'])==set(ind['slices'])==set(BOUNDS) and actual['metrics']==actual['slices']['full122'],'Complete recorded outcome slices')
            if path.endswith('_zero'):
                zero_fees[arm+'/'+path]=zero_fee_evidence(actual,e.load(OUT/'results'/(arm+'_'+path+'_TRADES.json')),e.load(OUT/'results'/(arm+'_'+path+'_HOLDINGS.json')))
            for part in BOUNDS:h.metric_evidence(ind['slices'][part],actual['slices'][part],arm+'/'+path+'/'+part);slices+=1
            summaries[arm][path]={part:{k:actual['slices'][part][k] for k in h.METRICS} for part in BOUNDS}
        require(set(training[arm])=={'same_close_zero','next_open_cost'},'First88 execution paths')
        for path in training[arm]:
            require(set(training[arm][path])=={'training','early','recent'},'First88 eras')
            for short,full in (('training','first88'),('early','early53'),('recent','recent35')):h.metric_evidence(outcome['paths'][arm][path]['slices'][full],training[arm][path][short],'First88/'+arm+'/'+path+'/'+short);groups+=1
    require(slices==45 and groups==18,'Independent45/18 accounting')
    predecessor=plan['predecessor_binding'];require(predecessor['directory']==PRIOR.name,'Predecessor208')
    prior_closed=e.load(PRIOR/'COMPLETION.json',predecessor['input_sha256'][str(PRIOR/'COMPLETION.json')]);prior_result=e.load(PRIOR/'results/COMPLETION.json',prior_closed['result_completion_sha256'])
    require(prior_closed['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior_result['status']=='COMPLETE' and set(prior_result['artifacts_sha256'])==h.RESULT_FILES and prior_result['plan_sha256']==prior_closed['plan_sha256'],'Closed predecessor exact38 results')
    prior_results=e.load(PRIOR/'results/RESULTS.json',prior_result['artifacts_sha256']['RESULTS.json'])
    incumbent=h.incumbent_comparison(e,prior_result,results,prior_results)
    incumbent=json.loads(json.dumps(incumbent).replace('previous207','previous208').replace('current208','current209'))
    changed={key:h.changes(orders,a,b) for key,a,b in (('candidate_vs_incumbent',ARMS[0],ARMS[1]),('control_vs_incumbent',ARMS[0],ARMS[2]),('candidate_vs_control',ARMS[2],ARMS[1]))}
    attribution={};monthly={}
    for path in PATHS:
        factors={}
        for arm in ARMS:
            daily=e.load(OUT/'results'/(arm+'_'+path+'_DAILY.json'));factors[arm]=p.monthly_factors(daily,meta['dates'],meta['ends'])
        years=(datetime.fromisoformat(meta['ends'][-1])-datetime.fromisoformat(meta['dates'][0])).days/365.2425
        attribution[path]={}
        for key,a,b in (('candidate_vs_incumbent',ARMS[1],ARMS[0]),('control_vs_incumbent',ARMS[2],ARMS[0]),('candidate_vs_control',ARMS[1],ARMS[2])):
            ref=results['arms'][b][path]['metrics']['cagr'];record=p.relative_contributions(factors[a],factors[b],ref,years)
            p.near(record['cagr_difference'],results['arms'][a][path]['metrics']['cagr']-ref,'Saved NAV CAGR attribution');attribution[path][key]=record
    protected=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
    require(len(e.load(protected,plan['input_sha256'][str(protected)]))==closure['protected_files']==21444 and closure['observers_modified'] is False and closure['new_trials']==3,'Protected closure')
    central=e.load(CENTRAL);require(central['current']==OUT.name and central['current_status']=='completed_independently_verified' and central['registered_queue']==[] and len(central['completed'])==207 and sum(x['trials'] for x in central['completed'])==1138,'Immediate closed209 counters')
    chosen=frozen['selection'];last=central['completed'][-1];best=closure['best_training_selected_candidate']
    require(last['directory']==OUT.name and last['completion']['sha256']==completion_sha and last['primary']==chosen['primary'] and last['new_admitted']==int(chosen['admitted']) and last['trials']==3 and central['best_training_selected_candidate']==best,'Central closure linkage')
    oldbest=before['best_training_selected_candidate'];selected=results['arms'][chosen['primary']]['same_close_zero']['metrics'];update=chosen['admitted'] and selected['cagr']>oldbest['cagr']
    if update:require(best['directory']==OUT.name and best['spec']==dict(arm=chosen['primary']) and all(best[k]==selected[k] for k in ('cagr','sharpe','max_drawdown')) and best['next_open_5bp_cagr']==results['arms'][chosen['primary']]['next_open_cost']['metrics']['cagr'] and best['selected_on_first88'] is True and best['observer_promotion'] is False,'Recorded best update')
    else:require(best==oldbest,'Unqualified historical best change')
    latest=central['latest_training_selected_candidate'];require(latest['directory']==OUT.name and latest['arm']==chosen['primary'] and latest['selection']==chosen and latest['full_period_cagr']==selected['cagr'] and latest['full_period_sharpe']==selected['sharpe'] and latest['selected_on_first88'] is True,'Central latest selected result')
    require(closure['target_reached']==(best['cagr']>=.7),'Target flag')
    for name,pin in HELPERS.items():e.verify(name,pin)
    e.verify(__file__)
    e.verify('work/test_relative_path209_close_review.py')
    for name,pin in list(e.pins.items()):e.verify(name,pin)
    report=dict(schema='monthly_relative_path209_research_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,result='ADMITTED' if chosen['admitted'] else 'NOT_ADOPTED',selection=chosen,failed_training_guards=[k for k,v in chosen['guards'].items() if not v],summary=summaries,holding_changes=changed,attribution=attribution,source_refs=e.pins,phase_sha256=dict(plan=plan_sha,prepared=frozen['prepared_sha256'],selected=rc['frozen_sha256'],results=closure['result_completion_sha256'],closure=completion_sha),source_path_counts=counts,zero_fee_evidence=zero_fees,model_fit_accounting=model_counts,control191=control,score_integration=integration,incumbent208_export_diagnostic=incumbent,historical_best_unchanged=best==oldbest,best_training_selected_candidate=best,target_reached=closure['target_reached'],validation=dict(synthetic_tests=tests['tests'],reader_synthetic_tests=22,independent_query_cells=5978,independent_paths=9,independent_slices=45,first88_metric_groups=18,model_prefixes=[0,1,53,88,122],order_prefixes=123,original_arrays=27,protected_files=21444,protection_scope='Hash-bound manifest and successful registration/closure receipts; no repeated protected archive scan.',zero_cost_path_fees_exact_zero=True,completed_entries=207,trials=1138),limitations=context['limitations'],observers_modified=False,central_modified=False,new_labels_computed=False,new_model_fits_computed=False,new_orders_computed=False,new_portfolio_returns_computed=False,production_modules_imported=False,raw_price_arrays_loaded=False,source_bytes_hash_verified_only=True,saved_NAV_attribution_only=True)
    if not check_only:
        with (OUT/'RESEARCH_REVIEW.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        with (OUT/'RESEARCH_REVIEW.md').open('x') as f:f.write(markdown(report))
    return report


def markdown(report):
    _,p=helpers();lines=['# 第209轮：月度相对跑输的深度与持续时间','',('候选通过首88期筛选。' if report['selection']['admitted'] else '候选未通过首88期筛选，保留Local67。'),'','候选标签为下个自然月每日相对SPY财富短缺的均值，对照为同月终点短缺；年度Ridge输出仅用于排名。不是ES、最大回撤或月内退出策略。','']
    for path,title in (('same_close_zero','同收盘零成本'),('next_open_cost','次开盘、每边5基点')):
        lines+=['## '+title,'','|区间|Local67|路径短缺|终点短缺|','|---|---:|---:|---:|']
        for period in BOUNDS:lines.append('|'+period+'|'+'|'.join(p.format_metrics(report['summary'][a][path][period]) for a in ARMS)+'|')
        lines.append('')
    lines+=['未通过门槛：'+('、'.join(report['failed_training_guards']) or '无'),'','持仓集合变化（全期／前88／后34）：']
    for name,item in report['holding_changes'].items():lines.append(f"- {name}: {item['counts']['full122']} / {item['counts']['first88']} / {item['counts']['later34']}")
    lines+=['','已核对13份prepared、38份results、9条路径／45个分段／18组训练指标、5组物理模型前缀及123组订单前缀。191候选复现为终点控制，208基线指标和导出保持一致；个别次开盘成本period fee仅允许不超过1e−10的舍入差，总费仍精确一致。','',f"主调用配对年度拟合：{report['model_fit_accounting']['main_paired_annual_fits']}；独立及前缀拟合单列，不新增策略试验。",'','后34期是重复使用的诊断样本。源版本、动态股票池代理和同收盘执行假设仍有局限。未修改观察账户。','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completion-sha',required=True);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    result=review(args.completion_sha,args.check_only)
    print(json.dumps(dict(passed=True,result=result['result'],output=str(OUT/'RESEARCH_REVIEW.json'),sha256=None if args.check_only else sha(OUT/'RESEARCH_REVIEW.json')),ensure_ascii=False))

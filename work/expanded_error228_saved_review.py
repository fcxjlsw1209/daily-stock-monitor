"""Read terminal228 saved evidence; no model imports, array decoding or replay.
Prospectively fixed before228 actual outputs: ONLY cost-path period HOLDINGS.fee
may differ <=1e-10 from226. Cumulative fees and all other incumbent fields exact.
Production-vs-independent nine-statistic accounting retains its fixed bound.
Run only after root authorization of explicit terminal COMPLETION/PLAN/review SHAs.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_expanded_error228'
PRIOR=ROOT/'outputs/monthly_joint_error226'
EXPANDED=ROOT/'outputs/monthly_ineligible_source227'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CONTEXT=ROOT/'work/monthly_expanded_error228_context.json'
CONTEXT_SHA='5831d860f78d87cf47bee1dac8267f180615a2add1c64b01b787ed4bc5dcf747'
SOURCE_REVIEW=ROOT/'work/expanded_error228_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA='d5981dbb6797894f46e54fc92e5b8702426b1b2c8a993ee655d2b818636b65db'
TESTS=ROOT/'work/monthly_expanded_error228_TESTS.json'
TESTS_SHA='7872225522e2e8062844a48eeab61d47114135c99145b63433ec1a010f740f6f'
CODE_REVIEW=ROOT/'work/expanded_error228_CODE_REVIEW.json'
LOCAL=ROOT/'outputs/monthly_local_linear70'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
ARMS=('incumbent_local67','joint_prediction_error_variance','diagonal_prediction_error_variance')
PATHS={'same_close_zero':('close',0.),'next_open_zero':('open',0.),'next_open_cost':('open',.0005)}
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'RISKS_EFFECTS.npz','SCORES.npz','MODEL_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
MODEL_FLAGS=('passed', 'independent_scalar_rebuild', 'exact_identity_join_checked', 'cached_local_linear_reused', 'annual_prior_december_cutoffs_checked', 'five_year_mature_window_checked', 'four_target_common_cohort_checked', 'historical_original_universe_checked', 'fixed_top_two_checked', 'original_boundary_domain_checked', 'common_complete_history_checked', 'minimum_18_months_checked', 'original_targets_unchanged', 'column_centering_checked', 'joint_gram_identity_checked', 'gram_positive_semidefinite_checked', 'matched_diagonal_checked', 'negative_risk_effect_ranks_checked', 'constant_risk_zero_effect_checked', 'paired_fallback_checked', 'no_future_or_outcome_inputs', 'audit_checked', 'estimates_counted')
SELECTION_FLAGS=('passed', 'independent_original_pre_checked', 'original_support_checked', 'original_budget_checked', 'fixed_top_two_checked', 'replacement_not_additive_checked', 'one_episode_per_arm', 'paired_fallback_checked', 'exact_scores_and_choices_checked')
COUNT_FIELDS=('query_months', 'initial_fallback_queries', 'domain_fallback_queries', 'history_fallback_queries', 'supported_queries', 'boundary_candidates', 'attempted_history_months', 'complete_history_months', 'incomplete_history_months', 'joined_label_rows', 'invalid_label_rows', 'missing_security_rows', 'unsupported_source_rows', 'skipped_initial_months', 'error_observations', 'security_mean_estimates', 'security_variance_estimates', 'joint_variance_estimates', 'diagonal_variance_estimates', 'parameter_estimates', 'cached_forecast_rows')
ARRAY_KEYS=('candidate_risk', 'control_risk', 'candidate_effect', 'control_effect', 'ready')
FEATURE_PREFIXES=(0,1,53,88,122)
NUMERICAL_FILES=('monthly_expanded_error228_host.py','test_monthly_expanded_error228_host.py','monthly_expanded_error228_adapter.py','test_monthly_expanded_error228_adapter.py','monthly_expanded_error228_independent.py','test_monthly_expanded_error228_independent.py')
PERIOD_FEE_ATOL=1e-10
ACCOUNTING_RESIDUALS=[]
# Formal test count must match the explicit CLI and immutable registered receipt.
# Final PLAN and code-review digests must be provided explicitly at authorized execution.


COVERAGE=ROOT/'work/after225_prediction_error_covariance_coverage.json'
COVERAGE_SHA='bfc526e34c4a1473340aa14792c555ded106c3d323a74f37e161fc468dbfa26a'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'

def need(ok,message):
    if not ok:raise ValueError(message)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def path(value):
    p=Path(value);return (p if p.is_absolute() else ROOT/p).resolve()


def read_json(value):
    def nonfinite(x):raise ValueError('Nonfinite JSON constant: '+x)
    p=path(value);need(p.is_relative_to(ROOT),'Only local JSON evidence may be read')
    return json.loads(p.read_text(),parse_constant=nonfinite)


class Evidence:
    def __init__(self):self.pins={}
    def pin(self,value,digest=None):
        p=path(value)
        need(p!=CENTRAL.resolve(),'Mutable central must not enter persistent input closure')
        need('CloudStorage' not in p.parts and (p.is_relative_to(ROOT) or RUNTIME.get(str(p))==digest),'Unapproved external input')
        value=sha(p);need(digest is None or value==digest,'Changed evidence: '+str(p))
        key=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        need(key not in self.pins or self.pins[key]==value,'Evidence changed within review: '+key)
        self.pins[key]=value;return value
    def load(self,value,digest=None):
        self.pin(value,digest);return read_json(value)
    def phase(self,folder,digest,plan_sha,names):
        c=self.load(folder/'COMPLETION.json',digest)
        need(c.get('status')=='COMPLETE' and c.get('plan_sha256')==plan_sha and set(c.get('artifacts_sha256',{}))==names,'Incomplete exact phase manifest: '+str(folder))
        for filename,h in c['artifacts_sha256'].items():
            need(Path(filename).name==filename,'Phase artifact path escape');self.pin(folder/filename,h)
        return c


def registered_digest(plan,value):
    p=path(value);matches=[digest for name,digest in plan['input_sha256'].items() if path(name)==p]
    need(matches and len(set(matches))==1,'Missing/conflicting registered dependency: '+str(p))
    return matches[0]


def numerical_close(actual,expected,label):
    # Exactly the already frozen independent accounting tolerance; never used
    # for comparing the incumbent with its prior run or for zero-cost fees.
    need(finite(actual) and finite(expected) and abs(actual-expected)<=1e-7+2e-11*abs(expected),'Independent recorded accounting differs: '+label)
    if actual!=expected:
        ACCOUNTING_RESIDUALS.append(dict(label=label,production=actual,independent=expected,
            signed_difference=actual-expected,absolute_difference=abs(actual-expected),absolute_tolerance=1e-7,relative_tolerance=2e-11))


def metric_evidence(independent,actual,label):
    need(set(independent)==STAT_KEYS and STAT_KEYS<=set(actual),'All nine independent statistics required: '+label)
    for key,wanted in independent.items():
        value=actual[key]
        if key in ('start','end','trading_sessions') or wanted is None:
            need(value==wanted,'Independent exact statistic differs: '+label+'/'+key)
        else:numerical_close(value,wanted,label+'/'+key)


def guard(candidate,reference,gain):
    return (candidate['training']['cagr']>=reference['training']['cagr']+gain-1e-12 and
        candidate['training']['sharpe']>=reference['training']['sharpe']-.05 and
        candidate['training']['max_drawdown']>=reference['training']['max_drawdown']-.03 and
        all(candidate[era]['cagr']>=reference[era]['cagr']-.025 for era in ('early','recent')))


def changed(orders,reference,candidate):
    rows=[]
    for i,(old,new) in enumerate(zip(orders[reference],orders[candidate])):
        if set(old['symbols'])!=set(new['symbols']):
            rows.append(dict(index=i,date=old['date'],end=old['end'],reference=list(old['symbols']),candidate=list(new['symbols']),
                removed=sorted(set(old['symbols'])-set(new['symbols'])),added=sorted(set(new['symbols'])-set(old['symbols']))))
    return dict(counts={name:sum(lo<=r['index']<hi for r in rows) for name,(lo,hi) in BOUNDS.items()},months=rows)


def selection_from_saved(training,changes):
    ids=[x['index'] for x in changes[ARMS[1]+'_vs_'+ARMS[0]]['months'] if x['index']<88]
    ids2=[x['index'] for x in changes[ARMS[1]+'_vs_'+ARMS[2]]['months'] if x['index']<88]
    def support(rows):return len(rows)>=3 and any(i<53 for i in rows) and any(i>=53 for i in rows)
    # Explicit names prevent positional reuse from changing the candidate.
    b,c,m=training[ARMS[0]],training[ARMS[1]],training[ARMS[2]]
    checks=dict(close_vs_incumbent=guard(c['same_close_zero'],b['same_close_zero'],.005),
        open_vs_incumbent=guard(c['next_open_cost'],b['next_open_cost'],0.),
        close_vs_diagonal_error_control=guard(c['same_close_zero'],m['same_close_zero'],0.),
        open_vs_diagonal_error_control=guard(c['next_open_cost'],m['next_open_cost'],0.),
        changed_support=support(ids),changed_support_vs_control=support(ids2))
    admitted=all(checks.values())
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=checks,
        changed_months=ids,changed_months_vs_control=ids2,first88_only=True,later34_used=False)


def model_coverage(audit,prepared,completion,closure,meta,training_meta,calendar,orders):
    """Recount saved identities/support only; no residual/covariance/rank rebuild."""
    td=training_meta['training'];triples=[(d,s,e) for d,e,ss in zip(td['dates'],td['ends'],td['symbols']) for s in ss]
    need(len(triples)==len(set(triples))==8820,'Original169 training identities')
    source_ids=[(d,s,e) for d,e,ss in zip(meta['dates'],meta['ends'],meta['symbols']) for s in ss]
    axis=calendar['sessions'];need(axis==sorted(set(axis)),'Official sessions ordered')
    monthends={d[:7]:d for d in axis};events=audit['event_history'];counts=audit['counts']
    need(audit['method']=='joint_asof_error_boundary_variance' and audit['arms']==list(ARMS) and audit['query_months']==122 and audit['symbols_per_event']==49,'Saved method/axes')
    need(audit['min_months']==18 and audit['lookback_calendar_years']==5 and audit['ceiling']==5 and audit['gap']==.02,'Registered common-history/domain parameters')
    need(audit['historical_errors']=='cached raw_local_linear minus unchanged target_log_excess','Error definition')
    for k in ('source_forecasts_refitted','labels_recomputed','local_available_filtered'):need(audit[k] is False,'Audit source scope '+k)
    for k in ('historical_use_explicit','common_complete_history'):need(audit[k] is True,'Shared history '+k)
    need(set(counts)==set(COUNT_FIELDS) and len(events)==122,'Complete counts/events')
    total={k:0 for k in COUNT_FIELDS};by_year={};supported=[];unique_errors=set();repeated_errors=0;statuses=Counter()
    for i,event in enumerate(events):
        d,e,names=meta['dates'][i],meta['ends'][i],meta['symbols'][i];year=int(d[:4]);cut=f'{year-1}-12-31';low=f'{year-6}-12-31'
        need(event['query_index']==i and event['date']==d and event['end']==e and event['symbols']==names and event['year']==year and event['cutoff']==cut and event['lookback_exclusive_start']==low,'Query identity/annual cutoff')
        order,top,js,b=map(event.get,('eligible_order','ordered_top2','J','B'))
        need(len(set(order))==len(order) and all(type(j) is int and 0<=j<49 for j in order),'Saved eligible order')
        need(top==order[:2] and b==top+js and len(set(b))==len(b) and all(j in order[2:5] for j in js),'Fixed top2/boundary domain')
        need(js==[j for j in order[2:5] if j in js],'Boundary must preserve original order; no sixth tie expansion')
        status=event['status'];need(status in ('initial_partial','insufficient_candidates','insufficient_history','ready'),'Event status');statuses[status]+=1
        c={k:0 for k in COUNT_FIELDS};c['query_months']=1;c['boundary_candidates']=len(js)
        cached=event['counts']['cached_forecast_rows'];need(type(cached) is int and 0<=cached<=49,'Saved current cache availability')
        c['cached_forecast_rows']=cached # hash-bound proof; no source mask is decoded here
        if status=='initial_partial':need(i==0 and d=='2016-08-17','Initial exception');c['initial_fallback_queries']=1
        elif status=='insufficient_candidates':need(len(order)<3 or len(js)<2,'Narrow-domain fallback');c['domain_fallback_queries']=1
        else:need(d!='2016-08-17' and len(order)>=3 and len(js)>=2,'History evaluation domain')
        evaluated=status in ('insufficient_history','ready')
        wanted=[(dd,ee) for dd,ee in zip(meta['dates'],meta['ends']) if dd>low and ee<=cut and dd!='2016-08-17'] if evaluated else []
        need([(h['date'],h['end']) for h in event['history']]==wanted,'Every potential mature common month audited once')
        c['skipped_initial_months']=int(evaluated and meta['dates'][0]>low and meta['ends'][0]<=cut)
        hdates=[];hends=[]
        for h in event['history']:
            hd,he=h['date'],h['end'];need(monthends.get(hd[:7])==hd and monthends.get(he[:7])==he and hd>low and he<=cut,'Historical natural month maturity')
            need([r['symbol'] for r in h['rows']]==[names[j] for j in b],'All B identities in one shared month')
            complete=all(r['reason']=='valid' for r in h['rows']);need(h['complete'] is complete,'Shared complete rows')
            c['attempted_history_months']+=1;c['complete_history_months' if complete else 'incomplete_history_months']+=1
            if complete:hdates.append(hd);hends.append(he)
            for row in h['rows']:
                symbol=row['symbol'];reason=row['reason'];sid,tid=row['source_row_id'],row['training_row_id']
                need(reason in ('missing_security','historical_use_false','source_unavailable','common_good_false','nonfinite_training','valid'),'Historical reason')
                if reason=='missing_security':
                    need(sid is None and tid is None and symbol not in meta['symbols'][meta['dates'].index(hd)],'Missing exact identity');c['missing_security_rows']+=1
                else:
                    need(type(sid) is int and 0<=sid<len(source_ids) and source_ids[sid]==(hd,symbol,he),'Exact cached source identity')
                    if reason in ('historical_use_false','source_unavailable'):
                        need(tid is None,'Unsupported source must not join target');c['unsupported_source_rows']+=1
                    else:
                        need(type(tid) is int and 0<=tid<len(triples) and triples[tid]==(hd,symbol,he),'Exact mature training triple');c['joined_label_rows']+=1
                        if reason!='valid':c['invalid_label_rows']+=1
                if reason=='valid':
                    need(all(finite(row[k]) for k in ('prediction','target','error')),'Finite saved errors')
                    if complete:unique_errors.add(sid);repeated_errors+=1
                else:need(all(row[k] is None for k in ('prediction','target','error')),'Unavailable rows are not filled')
        need(event['H']==hdates and event['latest_label_end']==max(hends,default=None),'Shared H and latest maturity')
        c['error_observations']=len(hdates)*len(b)
        if status=='insufficient_history':need(len(hdates)<18,'History fallback threshold');c['history_fallback_queries']=1
        if status=='ready':
            need(len(hdates)>=18,'Shared minimum18');supported.append(i);c['supported_queries']=1
            c['security_mean_estimates']=c['security_variance_estimates']=len(b)
            c['joint_variance_estimates']=c['diagonal_variance_estimates']=len(js);c['parameter_estimates']=2*len(b)+2*len(js)
            stats=event['security_statistics'];trios=event['trios']
            need([r['index'] for r in stats]==b and [r['symbol'] for r in stats]==[names[j] for j in b],'Same marginal statistics')
            for row in stats:
                need(finite(row['error_mean']) and all(finite(row[k]) and row[k]>=0 for k in ('error_sum_squares','sample_variance')),'Finite nonnegative saved marginal variance')
            need([t['third_index'] for t in trios]==js,'Every J alternative')
            for j,t in zip(js,trios):
                need(t['indices']==top+[j] and t['symbols']==[names[q] for q in top+[j]] and t['symbol']==names[j],'Fixed EW3 trio')
                need(all(finite(t[k]) and t[k]>=0 for k in ('candidate_risk','control_risk')),'Saved matched risk nonnegative')
                need(all(finite(t[k]) and -.5<=t[k]<=.5 for k in ('candidate_effect','control_effect')),'Saved bounded replacement effects')
            for arm in ARMS[1:]:need(set(names[j] for j in top)<=set(orders[arm][i]['symbols']) and len(orders[arm][i]['symbols'])==3 and set(orders[arm][i]['symbols'])<=set(names[j] for j in b),'Fixed top-two membership in saved selection')
        else:
            need(event['security_statistics']==[] and event['trios']==[],'No estimates on fallback')
            for arm in ARMS[1:]:need(orders[arm][i]==orders[ARMS[0]][i],'Whole-query exact incumbent fallback')
        need(event['counts']==c,'Recounted event counts')
        for k in COUNT_FIELDS:total[k]+=c[k]
        yr=by_year.setdefault(year,dict(year=year,cutoff=cut,status_counts={},supported_queries=0,parameters=0,common_history_months=[]))
        yr['status_counts'][status]=yr['status_counts'].get(status,0)+1;yr['supported_queries']+=int(status=='ready');yr['parameters']+=c['parameter_estimates'];yr['common_history_months'].append(len(hdates))
    need(total==counts and repeated_errors==counts['error_observations'],'21 top counters recounted')
    for item in (audit,prepared['model']):
        need(item['source_model_refits']==item['new_ridge_fits']==item['new_tree_fits']==0,'No predictive source refits')
        need(item['new_statistical_estimation_queries']==counts['supported_queries'] and item['parameter_estimates']==counts['parameter_estimates'],'Honest descriptive statistics')
    model=prepared['model'];need(len(MODEL_FLAGS)==23 and all(model[k] is True for k in MODEL_FLAGS),'23 independent model flags')
    need(model['counts']==counts and model['query_rows']==122 and model['numerical_tolerance']==0. and model['gram_relative_tolerance']==1e-12 and model['producer_modules_imported'] is False and model['computes_returns'] is False,'Independent scope/exact canonical proof')
    gram=model['gram_diagnostics'];need([r['query_index'] for r in gram]==supported,'Every supported Gram diagnostic')
    for row in gram:need(all(finite(row[k]) for k in ('identity_max_error','minimum_eigenvalue','tolerance')) and row['tolerance']>=1e-12 and 0<=row['identity_max_error']<=row['tolerance'] and row['minimum_eigenvalue']>=-row['tolerance'],'Saved PSD and Gram identity proof')
    selection=prepared['selection'];need(all(selection[k] is True for k in SELECTION_FLAGS) and selection['ordered_choices_checked']==366 and selection['numerical_tolerance']==0. and selection['original_source_parity_tolerance']==1e-12 and selection['producer_modules_imported'] is False and selection['computes_returns'] is False,'Independent single-episode exact selection')
    need(prepared['prefixes']==123 and prepared['model_prefixes']==list(FEATURE_PREFIXES),'123 decisions and five physical histories')
    for k in ('model_prefix_sources_truncated','model_prefix_metadata_truncated','model_prefix_original_arrays_truncated','model_prefix_future_labels_removed','source_history_starts_at_original_zero','original_orders_exact','original_arrays_unchanged','original_source_arrays_unchanged','original_core_metadata_and_support_preserved','single_original_episode','no_new_score_budget','historical_expansion_contract_checked','error_prediction_source_local_linear'):need(prepared[k] is True,'Preparation invariant '+k)
    for item in (prepared,completion,closure):need(item['original_array_count']==27 and item['model_counts']==counts,'27 originals and main counters')
    need(completion['new_returns_computed'] is False and completion['source_model_refits']==0,'Prepare did not replay/refit')
    prefix=prepared['model_prefix_evidence'];need([r['prefix'] for r in prefix]==list(FEATURE_PREFIXES),'Physical prefix receipts')
    for p in prefix:
        n=p['prefix'];check=p['model'];s=p['selection'];c={k:sum(e['counts'][k] for e in events[:n]) for k in COUNT_FIELDS}
        need(p['query_rows']==n and p['counts']==check['counts']==c and p['source_history_starts_at_original_zero'] is True,'Physical prefix support counters')
        need(all(check[k] is True for k in MODEL_FLAGS) and check['query_rows']==n and check['numerical_tolerance']==0. and check['producer_modules_imported'] is False and check['computes_returns'] is False,'Prefix full independent model proof')
        need(check['gram_diagnostics']==[g for g in gram if g['query_index']<n] and check['gram_relative_tolerance']==1e-12,'Physical prefix Gram subset')
        need(check['parameter_estimates']==c['parameter_estimates'] and check['new_statistical_estimation_queries']==c['supported_queries'] and check['source_model_refits']==check['new_ridge_fits']==check['new_tree_fits']==0,'Diagnostic prefix estimate count')
        need(all(s[k] is True for k in SELECTION_FLAGS) and s['ordered_choices_checked']==3*n and s['numerical_tolerance']==0.,'Prefix selection proof')
        if n:
            cut=f"{int(meta['dates'][n-1][:4])-1}-12-31"
            need(p['query_date']==meta['dates'][n-1] and p['cutoff']==cut and (p['latest_label_end'] is None or p['latest_label_end']<=cut),'Physical prefix mature cutoff')
        else:need(p['training_rows']==0 and all(p[k] is None for k in ('query_date','cutoff','latest_label_end')),'Empty physical prefix')
    return dict(counts=counts,status_counts=dict(statuses),by_year=list(by_year.values()),supported_query_indices=supported,
        distinct_used_source_rows=len(unique_errors),repeated_used_error_observations=repeated_errors,
        main_statistical_estimation_queries=counts['supported_queries'],main_parameter_estimates=counts['parameter_estimates'],
        independent_diagnostic_queries=model['new_statistical_estimation_queries'],prefix_diagnostic_queries=sum(p['counts']['supported_queries'] for p in prefix),
        prefix_diagnostic_parameter_estimates=sum(p['counts']['parameter_estimates'] for p in prefix),source_prefixes=list(FEATURE_PREFIXES),
        predictive_source_refits=0,statistics_recomputed=False,npz_arrays_decoded=False,
        limitation='Saved error/mean/risk numbers are bound to the independent preparation receipt, not recomputed here. Cached availability/U and original targets are covered by the saved source proof, not freshly decoded.')


def compare_incumbent(evidence,prior_completion,results,prior_results):
    report={}
    for name in PATHS:
        old=prior_results['arms'][ARMS[0]][name];new=results['arms'][ARMS[0]][name]
        need(old['metrics']==new['metrics'] and old['slices']==new['slices'],'Every incumbent metric/slice, including cumulative fees and turnover, must be exact')
        exports={}
        for kind in ('TRADES','DAILY','DAILY_POSITIONS','HOLDINGS'):
            filename=ARMS[0]+'_'+name+'_'+kind+'.json'
            old_rows=evidence.load(PRIOR/'results'/filename,prior_completion['artifacts_sha256'][filename]);new_rows=evidence.load(OUT/'results'/filename)
            if kind!='HOLDINGS':need(old_rows==new_rows,'Incumbent exact '+name+'/'+kind)
            exports[kind]=(old_rows,new_rows)
        oh,nh=exports['HOLDINGS'];need(len(oh)==len(nh)==122,'Incumbent full holdings')
        residuals=[]
        for a,b in zip(oh,nh):
            need(set(a)==set(b) and {k:v for k,v in a.items() if k!='fee'}=={k:v for k,v in b.items() if k!='fee'},'Incumbent nonfee holdings exact')
            need(finite(a['fee']) and finite(b['fee']),'Finite fees')
            if a['fee']!=b['fee']:
                difference=b['fee']-a['fee'];need(name=='next_open_cost' and abs(difference)<=PERIOD_FEE_ATOL,'Only cost period HOLDINGS.fee may differ <=1e-10')
                residuals.append(dict(signal_date=b['signal_date'],execution_date=b['execution_date'],previous226=a['fee'],current228=b['fee'],difference=difference))
        for rows,metric in ((oh,old['metrics']),(nh,new['metrics'])):
            total=0.
            for row in rows:total+=row['fee']
            need(total==metric['fees_paid'],'Saved chronological fees exactly equal cumulative fee metric')
        report[name]=dict(all_metrics_and_slices_exact=True,total_fees_exact=True,all_fills_exact=True,daily_equity_exact=True,daily_positions_exact=True,holding_nonfee_fields_exact=True,period_fee_differences=residuals,max_absolute_period_fee_difference=max([abs(x['difference']) for x in residuals],default=0.))
    return dict(paths=report,period_fee_tolerance=PERIOD_FEE_ATOL,tolerance_scope='ONLY individual next_open_cost HOLDINGS.fee <=1e-10; all metrics/slices including cumulative fees, turnover, trades, DAILY, positions and holding nonfee fields EXACT; zero-cost fee fields EXACT0.',prospective_policy='Inherited frozen226 comparison scope: no accumulated-fee exception and no result-driven widening.')


ADAPTER_FLAGS=('passed','independent_adapter_rebuild','exact_training_identity','exact_query_identity',
 'source227_metadata_exact','original_train_fields_preserved','old_forecasts_byte_exact','old_labels_byte_exact',
 'original_common_good_preserved','only_ineligible_labels_added','expanded_masks_checked','historical_use_or_expansion',
 'finite_used_values_checked','outputs_detached','input_bytes_unchanged_by_checker','dtype_shape_bytes_checked',
 'audit_checked','no_numerical_recomputation')
ADAPTER_COUNTS=('training_rows','query_rows','original_forecast_rows','expanded_forecast_rows','added_forecast_rows',
 'original_label_rows','added_label_rows','extended_label_rows','original_history_rows','expanded_history_rows')
HISTORY_SEMANTICS='originalU OR newlyavailable227; frozen226 historical_original_universe_checked verifies supplied mask, not originalU'


def expanded_source_evidence(evidence,plan,binding,sr,parity,before,meta,training_meta):
    """Check saved expanded-source and adapter receipts, never array contents."""
    for k in ('passed','source_only','metadata_only','source_bytes_reverified','metadata_axes_verified','calendar_clock_verified'):
        need(sr[k] is True,'Expanded source metadata scope '+k)
    for k in ('source_npz_decoded','training_npz_decoded','npz_decoded','source_models_imported','actual_support_computed',
              'actual_joint_history_support_computed','actual_features_computed','actual_risks_computed','actual_labels_computed',
              'actual_forecasts_computed','actual_returns_computed','actual_orders_computed','observers_modified','central_modified',
              'frozen_files_modified','strict_pit'):
        need(sr[k] is False,'Expanded source-only limitation '+k)
    need(sr['actual_model_fits']==0 and sr['raw_source_hashes']==plan['raw_source_hashes'],'No source metadata fits or raw-map change')
    for filename,digest in sr['input_sha256'].items():
        need(registered_digest(plan,filename)==digest,'Expanded source closure pin not registered');evidence.pin(filename,digest)
    need(binding['directory']==EXPANDED.name and binding['risk_history_only'] is True and binding['current_trading_universe_unchanged'] is True
        and binding['original_models_not_retrained'] is True and binding['source_model_refits']==0 and binding['new_labels'] is False
        and binding['new_raw_features'] is False and binding['strict_pit'] is False,'Risk-only frozen source binding')
    source_spec=sr['expanded227']
    source_plan=evidence.load(EXPANDED/'PLAN.json',source_spec['plan_sha256'])
    closed=evidence.load(EXPANDED/'COMPLETION.json',source_spec['completion_sha256'])
    prepared=evidence.load(EXPANDED/'prepared/COMPLETION.json',source_spec['prepared_sha256'])
    old_review=evidence.load(EXPANDED/'RESEARCH_REVIEW.json',source_spec['saved_review_sha256'])
    need(source_spec['source_only'] is True and source_spec['strategy_trials']==0,'Source227 is not a strategy trial')
    need(closed['status']=='SOURCE_COMPLETED_INDEPENDENTLY_VERIFIED' and closed['source_only'] is True and closed['strategy_trials']==0
        and closed['new_portfolio_returns'] is False and closed['observers_modified'] is False and closed['protected_files']==21444,'Expanded source terminal scope')
    need(closed['plan_sha256']==prepared['plan_sha256']==source_spec['plan_sha256'] and closed['prepared_sha256']==source_spec['prepared_sha256'],'Source227 receipt chain')
    need(prepared['status']=='COMPLETE' and prepared['source_only'] is True and prepared['portfolio_returns'] is False
        and prepared['strategy_trials']==0 and set(prepared['artifacts_sha256'])=={'FORECASTS.npz','LABELS.npz','META.json','SOURCE_AUDIT.json','INDEPENDENT.json'}
        and prepared['artifacts_sha256']==source_spec['prepared_artifacts_sha256'],'Expanded source exact five artifacts')
    for filename,digest in prepared['artifacts_sha256'].items():
        need(registered_digest(plan,EXPANDED/'prepared'/filename)==digest,'Expanded source artifact not registered')
        evidence.pin(EXPANDED/'prepared'/filename,digest)
    sm=evidence.load(EXPANDED/'prepared/META.json')
    need(sm==training_meta and sm['monthly']==meta,'227/169/current identities identical')
    need(binding['source_plan_sha256']==source_spec['plan_sha256'] and binding['source_completion_sha256']==source_spec['completion_sha256']
        and binding['source_prepared_sha256']==source_spec['prepared_sha256']
        and binding['source_forecasts_sha256']==prepared['artifacts_sha256']['FORECASTS.npz']
        and binding['source_labels_sha256']==prepared['artifacts_sha256']['LABELS.npz'],'228 pinned source payload identity')
    previous=[x for x in before.get('source_experiments',[]) if x['directory']==EXPANDED.name]
    need(len(previous)==1 and previous[0]['strategy_trials']==0 and previous[0]['completion']['sha256']==source_spec['completion_sha256'],'227 separate source predecessor recorded')
    need(old_review['passed'] is True and old_review['source_only'] is True and old_review['strategy_trials']==0
        and old_review['original_strategy_unchanged'] is True and old_review['portfolio_returns_computed'] is False
        and old_review['source_npz_decoded'] is False and old_review['strict_PIT'] is False,'Bound source227 saved review scope')
    need(old_review['completion_sha256']==source_spec['completion_sha256'] and old_review['plan_sha256']==source_spec['plan_sha256']
        and old_review['prepared_sha256']==source_spec['prepared_sha256'],'Source227 saved review lineage')
    fc=sr['saved_source_counts']['forecasts'];lc=sr['saved_source_counts']['labels']
    need(fc==closed['forecasts']['counts']==old_review['forecasts'] and lc==closed['labels']['counts']==old_review['labels']['counts'],'Bound saved source227 count agreement')
    guarantees=sr['original_source_guarantees']
    for k in ('original_forecasts_exact_copy','original_forecast_masks_preserved','original_targets_exact_copy',
              'original_common_good_subset_preserved','original_clipped_centering_cohort_preserved','source227_fit_work_already_performed'):
        need(guarantees[k] is True,'Original source227 preservation '+k)
    for k in ('added_labels_used_in_original_models','query_features_recomputed','query_feature_reranking','new_source_model_refits_required_in228'):
        need(guarantees[k] is False,'Original model scope '+k)
    for k in ('original_ridge_training_unchanged','current_trading_eligibility_unchanged','historical_expansion_contract_checked'):
        need(parity[k] is True,'Expanded source integration scope '+k)
    need(parity['historical_mask_semantics']==HISTORY_SEMANTICS,'Historical mask explicitly expanded, not original U claim')
    adapter=parity['risk_history_adapter'];check=parity['independent_adapter']
    need(adapter['method']=='expanded_same_query_error_history_adapter' and adapter['train_output_role']=='risk_history_only_never_Ridge_training'
        and adapter['historical_use_rule']=='original_use OR newly_available227','Detached risk-only adapter method')
    for k in ('original_identity_exact','old_forecasts_exact','old_labels_exact','original_common_good_preserved',
              'added_labels_only_ineligible','historical_use_or_expansion','outputs_detached'):
        need(adapter[k] is True,'Adapter invariant '+k)
    for k in ('source_forecasts_refitted','labels_recomputed','numerical_risk_estimates_computed','features_recomputed','portfolio_returns_computed'):
        need(adapter[k] is False,'Adapter calculation scope '+k)
    need(all(check.get(k) is True for k in ADAPTER_FLAGS),'18 independent adapter flags')
    need(check['numerical_tolerance']==0. and check['new_model_fits']==0,'Exact adapter/no new models')
    for k in ('producer_modules_imported','computes_returns','computes_predictions','computes_labels','computes_risk_estimates'):
        need(check[k] is False,'Independent adapter scope '+k)
    need(check['mutation_scope']=='checker inputs unchanged; producer mutation requires host before/after snapshots','Honest input-mutation proof scope')
    ac=adapter['counts'];need(set(ac)==set(ADAPTER_COUNTS) and ac==check['counts'] and all(type(v) is int and v>=0 for v in ac.values()),'Exact ten adapter counters')
    want=dict(training_rows=lc['training_rows'],query_rows=fc['query_rows'],original_forecast_rows=fc['original_available_rows'],
        expanded_forecast_rows=fc['available_rows'],added_forecast_rows=fc['added_available_rows'],
        original_label_rows=lc['original_good_rows'],added_label_rows=lc['added_rows'],extended_label_rows=lc['extended_good_rows'])
    need(all(ac[k]==v for k,v in want.items()),'Risk adapter counts agree with source227 coverage')
    need(ac['original_history_rows']<=ac['original_forecast_rows'] and ac['expanded_history_rows']==ac['original_history_rows']+ac['added_forecast_rows']
        and ac['expanded_history_rows']<=ac['expanded_forecast_rows'],'Only newly available history rows OR into original U')
    need(sr['interfaces']['risk_prediction_key']=='raw_local_linear' and sr['interfaces']['risk_target_key']=='target_log_excess'
        and set(sr['interfaces']['risk_only_training_overrides'])=={'common_good','target_simple_excess','target_log_excess','target_vol_scaled','target_loss_utility'},'Fixed risk-only target fields')
    dedup=evidence.load(COVERAGE,COVERAGE_SHA)
    need(registered_digest(plan,COVERAGE)==COVERAGE_SHA and dedup['coverage_scope']['exhaustive'] is False
        and dedup['coverage_scope']['new_economic_family'] is False,'Inherited bounded coverage, no new-family claim')
    for filename,digest in dedup['source_sha256'].items():evidence.pin(filename,digest)
    evidence.pin(sr['implementation']['path'],sr['implementation']['sha256'])
    return dict(source227_completion_sha256=source_spec['completion_sha256'],source227_saved_review_sha256=source_spec['saved_review_sha256'],
        source227_forecast_counts=fc,source227_label_counts=lc,adapter_counts=ac,adapter_flags_checked=list(ADAPTER_FLAGS),
        history_mask_semantics=HISTORY_SEMANTICS,added_labels_enter_original_Ridge=False,source_refits_in228=0,
        no_raw_source_or_label_values_rebuilt=True,statistics_label='joint covariance of transformed, clipped, month-centered relative target prediction errors; not actual portfolio return variance')


def main(expected,test_count,expected_plan_sha,expected_code_review_sha):
    need(type(test_count) is int and test_count==129,'Exact129 formal synthetic tests required')
    for value in (expected,expected_plan_sha,expected_code_review_sha):
        need(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Explicit terminal/PLAN/code-review SHA required')
    ACCOUNTING_RESIDUALS.clear()
    need(not (OUT/'RESEARCH_REVIEW.json').exists(),'Saved review is write-once')
    evidence=Evidence();closure=evidence.load(OUT/'COMPLETION.json',expected)
    need(closure['plan_sha256']==expected_plan_sha,'Explicit registered PLAN differs')
    plan=evidence.load(OUT/'PLAN.json',expected_plan_sha);plan_sha=expected_plan_sha
    need(closure.get('status')=='COMPLETED_INDEPENDENTLY_VERIFIED' and closure['plan_sha256']==plan_sha and closure['protected_files']==21444 and closure['observers_modified'] is False,'Terminal closure required')
    need(isinstance(plan.get('input_sha256'),dict) and plan['input_sha256'],'Registered input closure missing')
    for filename,digest in plan['input_sha256'].items():evidence.pin(filename,digest)
    context=evidence.load(CONTEXT,CONTEXT_SHA)
    need(registered_digest(plan,CONTEXT)==CONTEXT_SHA==plan['contract']['prospective_context_sha256'],'Context not registered exactly')
    need(plan['contract']['context']==context and context['arms']==list(ARMS) and plan['contract']['replace_original_episode_effects'] is True and plan['contract']['no_new_score_budget'] is True and plan['contract']['candidate']==ARMS[1] and plan['contract']['control_nonpromotable']==ARMS[2],'Contract/source theory differs')
    need(plan['protected_files']==21444 and len(evidence.load(PROTECTED,registered_digest(plan,PROTECTED)))==21444,'Registered protection manifest')
    before=evidence.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(before['completed'])==224 and sum(r['trials'] for r in before['completed'])==1189 and not before['registered_queue'],'Pre-registration state differs')
    prior_binding=plan['predecessor_binding'];need(prior_binding['directory']==PRIOR.name,'Wrong predecessor')
    for filename,digest in prior_binding['input_sha256'].items():evidence.pin(filename,digest)
    prior_plan=evidence.load(PRIOR/'PLAN.json',registered_digest(plan,PRIOR/'PLAN.json'))
    prior_plan_sha=sha(PRIOR/'PLAN.json')
    prior_root=evidence.load(PRIOR/'COMPLETION.json',registered_digest(plan,PRIOR/'COMPLETION.json'))
    need(before['completed'][-1]['directory']==PRIOR.name and before['completed'][-1]['completion']['sha256']==sha(PRIOR/'COMPLETION.json'),'Prior central identity')
    prior_done=evidence.load(PRIOR/'results/COMPLETION.json',registered_digest(plan,PRIOR/'results/COMPLETION.json'))
    need(prior_root['plan_sha256']==prior_done['plan_sha256']==prior_plan_sha and prior_root['result_completion_sha256']==sha(PRIOR/'results/COMPLETION.json') and prior_done['status']=='COMPLETE','Prior outcome lineage')
    prior_results=evidence.load(PRIOR/'results/RESULTS.json',prior_done['artifacts_sha256']['RESULTS.json'])
    need(registered_digest(plan,TESTS)==TESTS_SHA,'Pinned129 test receipt differs')
    tests=evidence.load(TESTS,TESTS_SHA)
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==test_count and tests['runtime']==plan['runtime'],'Formal tests/runtime differ')
    for filename,digest in tests['code_sha256'].items():
        need(registered_digest(plan,filename)==digest,'Tested source not registered');evidence.pin(filename,digest)
    need(registered_digest(plan,CODE_REVIEW)==expected_code_review_sha,'Explicit final code review differs')
    review=evidence.load(CODE_REVIEW,expected_code_review_sha)
    wanted_codes={str((ROOT/'work'/name).relative_to(ROOT)):registered_digest(plan,ROOT/'work'/name) for name in NUMERICAL_FILES}
    need(review['passed'] is True and review['unresolved_blockers']==[] and review['context_sha256']==CONTEXT_SHA and review['code_sha256']==wanted_codes,'Full six-file code review differs')
    for key in ('actual_features_computed','actual_model_fits_computed','actual_portfolio_returns_computed','production_files_modified','strategy_registered_by_reviewer'):
        need(review.get(key) is False,'Code-review scope differs: '+key)
    for filename,digest in review['input_sha256'].items():evidence.pin(filename,digest)
    need(plan['numerical_code_review']['sha256']==sha(CODE_REVIEW) and path(plan['numerical_code_review']['path'])==CODE_REVIEW,'Registered numerical review binding')
    frozen=evidence.load(OUT/'selected/FROZEN.json');training=evidence.load(OUT/'selected/TRAIN_RESULTS.json')
    need(frozen['plan_sha256']==plan_sha and frozen['training_sha256']==sha(OUT/'selected/TRAIN_RESULTS.json'),'Frozen first88 source')
    pc=evidence.phase(OUT/'prepared',frozen['prepared_sha256'],plan_sha,PREP)
    rc=evidence.phase(OUT/'results',closure['result_completion_sha256'],plan_sha,RESULT)
    need(rc['paths']==9 and rc['slices']==45 and rc['frozen_sha256']==sha(OUT/'selected/FROZEN.json'),'Outcome/frozen lineage')
    need(frozen['orders_sha256']==sha(OUT/'prepared/ORDERS.json'),'Frozen prepared orders')
    meta=evidence.load(OUT/'prepared/META.json');orders=evidence.load(OUT/'prepared/ORDERS.json')
    need(meta==evidence.load(LOCAL/'META.json',registered_digest(plan,LOCAL/'META.json')),'Original monthly metadata differs')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and all(len(set(s))==49 for s in meta['symbols']),'Original122x49 axes')
    need(meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Original period/calendar')
    need(meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30','Frozen split identities')
    need(set(orders)==set(ARMS) and all(len(v)==122 for v in orders.values()),'Three complete order streams')
    need(orders[ARMS[0]]==evidence.load(LOCAL/'local_linear_ORDERS.json',registered_digest(plan,LOCAL/'local_linear_ORDERS.json')),'Incumbent orders differ')
    for rows in orders.values():
        for i,row in enumerate(rows):
            need(row['date']==row['signal_date']==meta['dates'][i] and row['end']==meta['ends'][i],'Order timestamps')
            need(len(row['symbols']) in (0,2,3) and len(set(row['symbols']))==len(row['symbols']) and set(row['symbols'])<=set(meta['symbols'][i]),'Order targets outside original universe')
    prepared=evidence.load(OUT/'prepared/INDEPENDENT_PREPARE.json');audit=evidence.load(OUT/'prepared/MODEL_AUDIT.json')
    training_path=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control/MONTHLY_META.json'
    training_meta=evidence.load(training_path,registered_digest(plan,training_path))
    calendar=evidence.load(CALENDAR,registered_digest(plan,CALENDAR))
    coverage=model_coverage(audit,prepared,pc,closure,meta,training_meta,calendar,orders)
    need(pc['model_fit_accounting']==closure['model_fit_accounting']==plan['contract']['model_fit_accounting'],'Statistical accounting description')
    for name,keys in (('RISKS_EFFECTS.npz',ARRAY_KEYS),('SCORES.npz',ARMS)):
        with zipfile.ZipFile(OUT/'prepared'/name) as archive:
            need(set(archive.namelist())=={k+'.npy' for k in keys} and len(archive.namelist())==len(keys),'NPZ member names without decoding: '+name)
    source=evidence.load(OUT/'prepared/SOURCE_PARITY.json');binding=plan['model_source_binding']
    need(source['raw_source_hashes']==plan['raw_source_hashes'] and len(plan['raw_source_hashes'])==64 and source['model_source_binding']==binding,'Original169/206 source lineage')
    for k in ('original_training_cache_exact','original_identity_exact','cached206_source_vectors_exact','cached206_control_constant_exact','source_availability_exact','historical_expansion_contract_checked','no_source_model_refits','no_fitted_residual_error_inputs'):need(source[k] is True,'Source parity '+k)
    need(source['fresh_training_array_comparison'] is False and source['new_labels'] is False and source['new_raw_features'] is False and source['error_prediction_source']=='raw_local_linear' and source['target_source']=='source227_original_center_extended_log_excess','Saved source proof scope')
    parity_path=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control/PARITY.json'
    evidence.pin(parity_path,source['saved_training_parity_sha256'])
    sr=evidence.load(SOURCE_REVIEW,SOURCE_REVIEW_SHA)
    need(sr['passed'] is True and sr['context_sha256']==CONTEXT_SHA and sr['evidence_sha256']==context['evidence_sha256'] and registered_digest(plan,SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Registered final source review')
    for mapping in (binding['input_sha256'],sr['source_sha256'],sr['evidence_sha256']):
        for filename,digest in mapping.items():evidence.pin(filename,digest)
    expanded_evidence=expanded_source_evidence(evidence,plan,binding,sr,source,before,meta,training_meta)
    results=evidence.load(OUT/'results/RESULTS.json');independent=evidence.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(results['selection']==frozen['selection'] and results['strict_pit'] is False and results['reused_sample'] is True and results['observer_promotion'] is False and results['joint_error_portfolio_variance'] is True and results['cached_source_models_refitted'] is False and results['same_support_both_arms'] is True,'Outcome interpretation or selection differs')
    need(independent['passed'] is True and independent['independent_accounts'] is True and independent['independent_of_production_model_and_engine'] is True and independent['checked_paths']==9 and independent['checked_slices']==45 and independent['terminal_liquidation'] is False,'Independent execution scope')
    need(independent['accounting_source_sha256']==registered_digest(plan,ROOT/'work/ptr169_results_independent.py'),'Independent accounting source')
    need(set(results['arms'])==set(independent['paths'])==set(training)==set(ARMS),'Outcome/training arms')
    metrics={};fee_evidence={};training_checks=0;zero_streams=0
    for arm in ARMS:
        need(set(results['arms'][arm])==set(independent['paths'][arm])==set(PATHS),'Three execution paths required')
        need(set(training[arm])=={'same_close_zero','next_open_cost'},'Training uses undeclared execution path')
        metrics[arm]={};fee_evidence[arm]={}
        for name,(execution,cost) in PATHS.items():
            result=results['arms'][arm][name];m=result['metrics'];proof=independent['paths'][arm][name]
            need(m['execution']==execution and m['cost_per_side']==cost and m['start']==meta['dates'][0] and m['end']==meta['ends'][-1],'Execution identity/price timing')
            need(proof['passed'] is True and proof['monthly_executions']==122 and set(proof['slices'])==set(result['slices'])==set(BOUNDS),'Complete independent slices')
            need(result['accounting']['exported_fill_ledger_matches'] is True and result['accounting']['terminal_liquidation'] is False,'Daily ledger verification absent')
            metric_evidence(proof['slices']['full122'],m,arm+'/'+name+'/metrics')
            metrics[arm][name]={}
            for label,(lo,hi) in BOUNDS.items():
                value=result['slices'][label]
                need(value['start']==meta['dates'][lo] and value['end']==meta['ends'][hi-1],'Saved split boundary')
                metric_evidence(proof['slices'][label],value,arm+'/'+name+'/'+label)
                metrics[arm][name][label]={k:value[k] for k in ('start','end','cagr','sharpe','max_drawdown')}
            for key in ('fees_paid','dollar_turnover','trade_rows'):numerical_close(m[key],proof[key],arm+'/'+name+'/'+key)
            if name in training[arm]:
                need(set(training[arm][name])=={'training','early','recent'},'Three training periods required')
                for subset,label in (('training','first88'),('early','early53'),('recent','recent35')):
                    need(set(training[arm][name][subset])==STAT_KEYS,'Training metric set')
                    metric_evidence(proof['slices'][label],training[arm][name][subset],arm+'/'+name+'/'+subset);training_checks+=1
            trades=evidence.load(OUT/'results'/f'{arm}_{name}_TRADES.json')
            holdings=evidence.load(OUT/'results'/f'{arm}_{name}_HOLDINGS.json')
            need(len(holdings)==122 and m['trade_rows']==len(trades),'Fee stream coverage')
            need(all(finite(x['cost']) and x['cost']>=0 for x in trades) and all(finite(x['fee']) and x['fee']>=0 for x in holdings),'Invalid saved fees')
            if cost==0.:
                need(m['fees_paid']==0. and proof['fees_paid']==0. and all(x['cost']==0. for x in trades) and all(x['fee']==0. for x in holdings),'Nonzero fee in a zero-cost stream')
                zero_streams+=1
            fee_evidence[arm][name]=dict(execution=execution,cost_per_side=cost,fees_paid=m['fees_paid'],trade_rows=len(trades),holding_rows=len(holdings),all_zero_fee_fields_checked=cost==0.)
    need(training_checks==18 and zero_streams==6,'Incomplete training/zero-cost checks')
    incumbent=compare_incumbent(evidence,prior_done,results,prior_results)
    changes={a+'_vs_'+b:changed(orders,b,a) for a,b in ((ARMS[1],ARMS[0]),(ARMS[2],ARMS[0]),(ARMS[1],ARMS[2]))}
    selection=selection_from_saved(training,changes);need(selection==frozen['selection'],'Frozen first88 guards differ')
    expected_best=before['best_training_selected_candidate'];pick=selection['primary'];picked=results['arms'][pick]
    if selection['admitted'] and picked['same_close_zero']['metrics']['cagr']>expected_best['cagr']:
        m=picked['same_close_zero']['metrics'];expected_best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=picked['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=plan['contract']['limitation'])
    need(closure['best_training_selected_candidate']==expected_best and closure['target_reached']==(expected_best['cagr']>=.7),'Best/target update differs from frozen admission')
    central_bytes=CENTRAL.read_bytes();state=json.loads(central_bytes)
    need(len(state['completed'])==225 and sum(x['trials'] for x in state['completed'])==1192 and state['registered_queue']==[],'Central postclosure counters')
    need(state['completed'][-1]['directory']==OUT.name and state['completed'][-1]['completion']['sha256']==expected and path(state['completed'][-1]['completion']['path'])==OUT/'COMPLETION.json','Central closure reference')
    need(state['best_training_selected_candidate']==expected_best and state['latest_training_selected_candidate']['selection']==selection,'Central selected/best mismatch')
    evidence.pin(Path(__file__))
    report=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),reviewer_role='Independent228 risk-input adapter checker author and frozen226 numerical oracle author; this saved-evidence reader imports no model, oracle or execution engine.',scope='Saved JSON/hash/identity/count/selection/accounting only. NPZ directory member names inspected without decoding arrays. Only saved common-history counters, identities, causal cutoffs, and selection/accounting receipts are rechecked; no numerical residuals, covariance, ranks, orders or returns recomputed.',completion_sha256=expected,plan_sha256=plan_sha,prepared_completion_sha256=sha(OUT/'prepared/COMPLETION.json'),result_completion_sha256=sha(OUT/'results/COMPLETION.json'),input_sha256=evidence.pins,
        formal_synthetic_tests=test_count,reviewed_numerical_files=6,model_flags_checked=list(MODEL_FLAGS),selection_flags_checked=list(SELECTION_FLAGS),model_coverage=coverage,selection=selection,changed_holdings=changes,metrics=metrics,fee_evidence=fee_evidence,incumbent_versus226=incumbent,expanded_source_evidence=expanded_evidence,independent_recorded_metric_residuals=ACCOUNTING_RESIDUALS,
        independent_metric_tolerance=dict(absolute=1e-7,relative=2e-11,scope='Only production-vs-independent accounting nine-statistic checks use this established bound. Across rounds all incumbent metrics including cumulative fees are EXACT; only cost-path individual HOLDINGS.fee <=1e-10. Zero-cost fees exact0.'),
        independent_paths=9,independent_slices=45,independent_training_groups=18,zero_cost_streams=6,order_prefixes=123,source_prefixes=list(FEATURE_PREFIXES),original_array_count=27,protected_files_verified_by_registered_host_and_closer=21444,protected_archive_reread_in_this_review=False,limitations=context['limitations'],strict_pit=False,reused_sample=True,observers_modified=False,actual_new_features_computed=False,actual_new_parameters_computed=False,actual_new_orders_computed=False,actual_portfolio_returns_computed=False,source_npz_decoded=False,best=expected_best,target_reached=closure['target_reached'],completed_entries=225,trials=1192,central_snapshot_sha256=hashlib.sha256(central_bytes).hexdigest())
    need(CENTRAL.read_bytes()==central_bytes and sha(OUT/'COMPLETION.json')==expected and sha(OUT/'PLAN.json')==expected_plan_sha,'Central or immutable closure changed during review')
    with (OUT/'RESEARCH_REVIEW.json').open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,review_sha256=sha(OUT/'RESEARCH_REVIEW.json'),selection=selection,model_counts=coverage['counts'],same_close={a:metrics[a]['same_close_zero']['full122'] for a in ARMS}),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completion-sha256',required=True);parser.add_argument('--tests',type=int,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--code-review-sha256',required=True)
    args=parser.parse_args();main(args.completion_sha256,args.tests,args.plan_sha256,args.code_review_sha256)

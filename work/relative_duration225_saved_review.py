"""Read terminal225 saved evidence; no model imports, array decoding or replay.
Prospectively fixed before225 actual outputs: ONLY cost-path period HOLDINGS.fee
may differ <=1e-10 from224. Cumulative fees and all other incumbent fields exact.
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
OUT=ROOT/'outputs/monthly_relative_duration225'
PRIOR=ROOT/'outputs/monthly_increment_calibration224'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CONTEXT=ROOT/'work/monthly_relative_duration225_context.json'
CONTEXT_SHA='0392086957d2cd7a843811de6971cf0544788cdb484aa2fdcd7f74b3b195d5d8'
SOURCE_REVIEW=ROOT/'work/relative_duration225_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA='d77b6d94fbbea3b4308686f5850288967024908219c8a09ca22868f31973d57b'
TESTS=ROOT/'work/monthly_relative_duration225_TESTS.json'
CODE_REVIEW=ROOT/'work/relative_duration225_CODE_REVIEW.json'
LOCAL=ROOT/'outputs/monthly_local_linear70'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
ARMS=('incumbent_local67','relative_lag_run_ridge','relative_lag_incidence_ridge')
PATHS={'same_close_zero':('close',0.),'next_open_zero':('open',0.),'next_open_cost':('open',.0005)}
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FEATURES.npz','TRANSFORMS_DELTAS.npz','LABELS.npz','LABEL_AUDIT.json','MODEL_AUDIT.json','RANK_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','IDENTITY_AUDIT.json','CALENDAR.json','INDEPENDENT_PREPARE.json','COHORT_PARITY.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
PATH_COUNT_FIELDS=('source_path_rows', 'source_price_observations', 'target_session_observations', 'missing_path_rows', 'lag_session_observations', 'longest_run_session_observations')
MODEL_FLAGS=('passed', 'audit_checked', 'independent_exact_rational_labels', 'endpoint_calendar_checked', 'shared_maturity_rows_weights_scalers_verified', 'annual_augmented_least_squares_verified', 'target_statistics_checked', 'count_ratios_rounded_once_checked', 'full_official_path_checked', 'longest_lag_run_checked', 'lag_incidence_checked', 'equality_breaks_run_checked', 'signal_excluded_end_included_checked', 'candidate_bounded_by_control_checked', 'shared_query_mask_and_centered_ranks_verified', 'no_query_future_outcomes', 'initial_partial_fallback_checked')
DELTA_FLAGS=('passed','common_query_population_checked','negative_prediction_ranks_checked','average_ties_checked','initial_delta_zero','common_available_only','no_new_eligibility_filter','missing_exact_zero')
VALUE_KEYS=('candidate_prediction','control_prediction','candidate_rank','control_rank','common_available')
LABEL_KEYS=('dates','ends','symbols','candidate_target','control_target','base_good','good')
FEATURE_PREFIXES=(0,1,53,88,122)
NUMERICAL_FILES=('monthly_relative_duration225_host.py', 'test_monthly_relative_duration225_host.py', 'monthly_relative_duration225_model.py', 'test_monthly_relative_duration225_model.py', 'monthly_relative_duration225_independent.py', 'test_monthly_relative_duration225_independent.py')
PERIOD_FEE_ATOL=1e-10
ACCOUNTING_RESIDUALS=[]
# Formal test count must match the explicit CLI and immutable registered receipt.
# Final PLAN and code-review digests must be provided explicitly at authorized execution.


COVERAGE=ROOT/'work/after224_relative_duration_coverage.json'
COVERAGE_SHA='50c6caa66552e954ea6b2b44dc62cc45df08a5d7c914d2d381ef286a2f7f6fcf'
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
        close_vs_relative_incidence=guard(c['same_close_zero'],m['same_close_zero'],0.),
        open_vs_relative_incidence=guard(c['next_open_cost'],m['next_open_cost'],0.),
        changed_support=support(ids),changed_support_vs_control=support(ids2))
    admitted=all(checks.values())
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=checks,
        changed_months=ids,changed_months_vs_control=ids2,first88_only=True,later34_used=False)


def model_coverage(labels, audit, ranks, prepared, completion, closure, meta, training_meta, calendar):
    """Recount saved row/run/cohort evidence. Never form labels or refit Ridge."""
    td=training_meta['training']
    identities=[(d,e,s) for d,e,ss in zip(td['dates'],td['ends'],td['symbols']) for s in ss]
    need(len(identities)==len(set(identities))==8820,'Original155 training identities')
    axis=calendar['sessions'];need(axis==sorted(set(axis)) and len(axis)>0,'Ordered official sessions')
    monthends={d[:7]:d for d in axis};positions={d:i for i,d in enumerate(axis)}
    def nextmonth(month):
        y,m=map(int,month.split('-'));return f'{y+int(m==12):04d}-{m%12+1:02d}'
    rows=labels['path_checks'];need(len(rows)==labels['rows']==8820,'Full source path records')
    need(labels['targets']==list(LABEL_KEYS[3:5]) and labels['maturity']=='next official calendar month-end close','Targets and maturity')
    for k in ('exact_binary_float_rational','rounded_targets_once','complete_positive_finite_paths','candidate_bounded_by_control','lag_equality_breaks_run','signal_session_excluded','target_end_included','inherited_common_good_retained','no_endpoint_shortening'):
        need(labels[k] is True,'Saved label proof '+k)
    for k in ('clipped','centered','calendar_completeness_certified_by_this_function'):need(labels[k] is False,'Label scope '+k)
    need(labels['relative_target_definitions']==dict(candidate='max_consecutive_d I[R_d<1]/L',control='sum_d I[R_d<1]/L'),'Duration definitions')
    counts={k:0 for k in PATH_COUNT_FIELDS};base_count=0;complete_ids=[];perdate={};states=Counter()
    for i,(row,identity) in enumerate(zip(rows,identities)):
        d,e,s=identity
        need((row['signal_date'],row['label_end'],row['symbol'])==(d,e,s) and row['row_index']==i,'Per-row original identity')
        need(monthends.get(d[:7])==d and monthends.get(nextmonth(d[:7]))==e and d in positions and e in positions,'Official natural month endpoints')
        left,right=positions[d],positions[e]
        need(right>left and right+1<len(axis) and axis[right+1][:7]>e[:7],'Next month complete witness')
        length=right-left;need(row['target_session_count']==length,'All official target sessions required')
        need(type(row['base_good']) is bool and type(row['path_complete']) is bool,'Boolean saved source support')
        base_count+=int(row['base_good']);states[row['status']]+=1
        current=perdate.setdefault(d,dict(ids=[],complete=[],base=0,end=e,length=length,source_dates=axis[left:right+1]))
        current['ids'].append(i);current['base']+=int(row['base_good'])
        if row['path_complete']:
            need(row['base_good'] and row['status']=='complete' and all(row[k]==[] for k in ('missing_official_sessions','invalid_stock_dates','invalid_spy_dates')),'Complete path admission')
            run,lag=row['longest_lag_run'],row['lag_session_count']
            need(type(run) is int and type(lag) is int and 0<=run<=lag<=length and (run==0)==(lag==0),'Saved integer duration bounds')
            counts['source_path_rows']+=1;counts['source_price_observations']+=2*(length+1)
            counts['target_session_observations']+=length;counts['lag_session_observations']+=lag;counts['longest_run_session_observations']+=run
            complete_ids.append(i);current['complete'].append(i)
        else:
            need(row['status']!='complete' and row['longest_lag_run'] is None and row['lag_session_count'] is None,'Unavailable durations are not zero labels')
            counts['missing_path_rows']+=int(row['base_good'])
    good_ids=[];monthrecords=[]
    for d,record in perdate.items():
        complete=len(record['complete']);good=complete if complete>=2 else 0
        if good:good_ids.extend(record['complete'])
        monthrecords.append(dict(signal_date=d,label_end=record['end'],base_rows=record['base'],endpoint_rows=complete,good_rows=good,status='complete' if good else 'fewer_than_two_shared_rows',complete_path_rows=complete,source_dates=record['source_dates'],target_dates=record['source_dates'][1:],target_session_count=record['length']))
    need(labels['months']==monthrecords and labels['base_rows']==base_count and labels['good_rows']==len(good_ids),'Common minimum-two monthly cohort')
    need(all(type(labels[k]) is int and labels[k]==v for k,v in counts.items()),'Six path counters')
    check=prepared['model'];need(len(MODEL_FLAGS)==17 and all(check[k] is True for k in MODEL_FLAGS),'17 independent model flags')
    need(check['producer_modules_imported'] is False and check['computes_portfolio_returns'] is False and finite(check['max_forecast_error']) and check['max_forecast_error']<=1e-12,'Independent fit scope')
    need(check['events']==122 and check['symbols_per_event']==49 and check['training_rows']==8820 and check['good_training_rows']==len(good_ids),'Full independent coverage')
    need(check['label_arrays_checked']==7 and check['forecast_arrays_checked']==5 and check['annual_records_checked']==11,'All labels and forecasts audited')
    common=ranks['common_counts'];need(len(common)==122 and all(type(v) is int and 0<=v<=49 for v in common) and common[0]==0,'Paired query rank counts and initial fallback')
    need(ranks['population']=='signal-time eligible with both finite forecasts' and ranks['mapping']=='average_rank0/(n-1)-0.5; both use negative prediction; n<=1 => 0','Original209 rank domain/mapping')
    need(ranks['missing_correction']==0 and ranks['max_absolute_rank']==.5 and ranks['non_monthend_fallback_indices']==[0] and ranks['query_evaluation_ends_not_used_for_labels'] is True and ranks['no_holdings_or_returns_computed'] is True,'Rank and maturity scope')
    need(check['available_cells']==sum(common) and check['unavailable_cells']==5978-sum(common),'Available query coverage')
    need(audit['alpha']==10. and audit['half_life_months']==12. and audit['lookback_years']==5 and audit['minimum_mature_months']==18 and audit['local_model_fits']==0,'Fixed paired annual Ridge parameters')
    need(audit['training_only_standardization'] is True and audit['calibrated_probabilities'] is False and audit['target_clipping'] is False and audit['target_centering'] is False,'Ridge meaning')
    years=sorted({int(d[:4]) for d in meta['dates']});need([row['year'] for row in audit['annual']]==years,'Every query year once')
    fits=0;by_year=[]
    for row in audit['annual']:
        y=row['year'];low,cut=f'{y-6}-12-31',f'{y-1}-12-31'
        ids=[i for i in good_ids if identities[i][0]>low and identities[i][1]<=cut]
        nm=len({identities[i][0][:7] for i in ids});observed=sum(v for d,v in zip(meta['dates'],common) if int(d[:4])==y)
        need(row['cutoff']==cut and row['lookback_exclusive_start']==low and row['training_indices']==ids,'Exact causal training rows')
        need(row['training_rows']==len(ids) and row['training_months']==nm and row['latest_label_end']==max((identities[i][1] for i in ids),default=None),'Annual maturity/support')
        status=row['status'];need(status in ('fit','cold_start','no_queries') and (status=='cold_start')==(nm<18),'Minimum18 mature months')
        nf=2 if status=='fit' else 0;need(row['model_fits']==nf,'One candidate plus one matched-control fit')
        if nf:
            need(row['prediction_rows']==observed and observed>0 and row['shared_rows_weights_scalers'] is True and abs(row['weight_sum']-len(ids))<=1e-10,'Paired query/weight mass')
            need(all(finite(row['weighted_target_means'][k]) and 0<=row['weighted_target_means'][k]<=1 for k in LABEL_KEYS[3:5]),'Bounded training target means')
        else:need(observed==0,'No annual predictions without fit')
        for k in LABEL_KEYS[3:5]:
            need(type(row['target_positive_counts'][k]) is int and 0<=row['target_positive_counts'][k]<=len(ids),'Target positive count')
            need(finite(row['target_sums'][k]) and 0<=row['target_sums'][k]<=len(ids),'Saved target sum bounds')
            if ids:need(all(finite(row[f][k]) and 0<=row[f][k]<=1 for f in ('target_means','target_maxima')),'Saved target summary bounds')
            else:need(row['target_means'][k] is None and row['target_maxima'][k] is None,'Empty target summaries')
        fits+=nf;by_year.append(dict(year=y,cutoff=cut,status=status,training_rows=len(ids),training_months=nm,prediction_rows=observed,model_fits=nf))
    need(audit['new_pooled_model_fits']==check['new_pooled_model_fits']==check['independent_model_fits']==fits and check['local_model_fits']==0,'Honest main and verification fit counts')
    need(prepared['prefixes']==123 and prepared['feature_prefixes']==list(FEATURE_PREFIXES) and prepared['prefix_prices_truncated'] is True and prepared['prefix_training_inputs_truncated'] is True,'Five physical plus123 order prefixes')
    for key in ('original_orders_exact','original_arrays_unchanged','original_training_arrays_unchanged','source_cohort_exact'):need(prepared[key] is True,'Original/source preservation '+key)
    prefixes=prepared['prefix_model_fit_counts'];need([p['prefix'] for p in prefixes]==list(FEATURE_PREFIXES),'Five explicit prefix fit records')
    for p in prefixes:
        n=p['prefix'];a,b=p['producer_model_fits'],p['independent_model_fits']
        need(type(a) is type(b) is int and a>=0 and a%2==0 and a==b,'Paired diagnostic fits')
        cutoff=None if n==0 else f"{int(meta['dates'][n-1][:4])-1}-12-31"
        need(p['training_label_end_cutoff']==cutoff and type(p['training_rows']) is int and type(p['training_months']) is int and p['training_rows']>=0 and p['training_months']>=0,'Physical prefix cutoff')
        if n==0:need(a==p['training_rows']==p['training_months']==0,'Empty physical prefix')
    for rec in (prepared,completion,closure):
        need(rec['original_array_count']==27,'Original27 arrays')
        for k in ('new_model_fits','new_predictive_model_fits','new_pooled_model_fits','independent_diagnostic_model_fits'):need(type(rec[k]) is int and rec[k]==fits,'Recorded main/independent fits '+k)
        need(rec['prefix_model_fit_counts']==prefixes and rec['prefix_diagnostic_model_fits']==sum(p['producer_model_fits'] for p in prefixes) and rec['prefix_independent_diagnostic_model_fits']==sum(p['independent_model_fits'] for p in prefixes),'Diagnostic prefix totals')
        need(all(rec[k]==v for k,v in counts.items()),'Six lineage counters')
    need(all(check[k]==v for k,v in counts.items()),'Independent source counters')
    need(completion['new_returns_computed'] is False,'No preparation portfolio returns')
    for k in DELTA_FLAGS:need(prepared['delta'][k] is True,'Independent delta flag '+k)
    need(prepared['delta']['dose']==.01 and prepared['delta']['max_absolute_correction']==.005,'Fixed small correction')
    need(set(prepared['delta']['changed_cells'])=={'candidate','control'} and all(type(v) is int and 0<=v<=5978 for v in prepared['delta']['changed_cells'].values()),'Delta count axes')
    for k in ('passed','scores_and_choices_checked','one_original_episode_checked','initial_partial_fallback_checked'):need(prepared['selection'][k] is True,'Independent selection '+k)
    need(prepared['selection']['ordered_choices_checked']==366,'Three by122 ordered selections')
    return dict(path_counts=counts,base_rows=base_count,good_training_rows=len(good_ids),path_status_counts=dict(states),available_query_cells=sum(common),by_year=by_year,main_paired_annual_fits=fits,independent_diagnostic_fits=fits,prefixes=prefixes,source_prefixes=list(FEATURE_PREFIXES),source_prices_or_labels_rebuilt=False,forecast_arrays_decoded=False,new_fits_computed=False)


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
                residuals.append(dict(signal_date=b['signal_date'],execution_date=b['execution_date'],previous224=a['fee'],current225=b['fee'],difference=difference))
        for rows,metric in ((oh,old['metrics']),(nh,new['metrics'])):
            total=0.
            for row in rows:total+=row['fee']
            need(total==metric['fees_paid'],'Saved chronological fees exactly equal cumulative fee metric')
        report[name]=dict(all_metrics_and_slices_exact=True,total_fees_exact=True,all_fills_exact=True,daily_equity_exact=True,daily_positions_exact=True,holding_nonfee_fields_exact=True,period_fee_differences=residuals,max_absolute_period_fee_difference=max([abs(x['difference']) for x in residuals],default=0.))
    return dict(paths=report,period_fee_tolerance=PERIOD_FEE_ATOL,tolerance_scope='ONLY individual next_open_cost HOLDINGS.fee <=1e-10; all metrics/slices including cumulative fees, turnover, trades, DAILY, positions and holding nonfee fields EXACT; zero-cost fee fields EXACT0.',prospective_policy='Root explicitly confirmed before any actual225 result that the older accumulated-fee exception is NOT inherited. No result-driven widening.')


def main(expected,test_count,expected_plan_sha,expected_code_review_sha):
    need(type(test_count) is int and test_count>0,'Explicit positive formal synthetic test count required')
    for value in (expected,expected_plan_sha,expected_code_review_sha):
        need(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Explicit terminal/PLAN/code-review SHA required')
    ACCOUNTING_RESIDUALS.clear()
    need(not (OUT/'RESEARCH_REVIEW.json').exists() and not (OUT/'RESEARCH_REVIEW.md').exists(),'Saved review is write-once')
    evidence=Evidence();closure=evidence.load(OUT/'COMPLETION.json',expected)
    need(closure['plan_sha256']==expected_plan_sha,'Explicit registered PLAN differs')
    plan=evidence.load(OUT/'PLAN.json',expected_plan_sha);plan_sha=expected_plan_sha
    need(closure.get('status')=='COMPLETED_INDEPENDENTLY_VERIFIED' and closure['plan_sha256']==plan_sha and closure['protected_files']==21444 and closure['observers_modified'] is False,'Terminal closure required')
    need(isinstance(plan.get('input_sha256'),dict) and plan['input_sha256'],'Registered input closure missing')
    for filename,digest in plan['input_sha256'].items():evidence.pin(filename,digest)
    context=evidence.load(CONTEXT,CONTEXT_SHA)
    need(registered_digest(plan,CONTEXT)==CONTEXT_SHA==plan['contract']['prospective_context_sha256'],'Context not registered exactly')
    need(plan['contract']['context']==context and context['arms']==list(ARMS) and plan['contract']['dose']==.01 and plan['contract']['max_absolute_delta']==.005 and plan['contract']['candidate']==ARMS[1] and plan['contract']['control_nonpromotable']==ARMS[2],'Contract/source theory differs')
    need(plan['protected_files']==21444 and len(evidence.load(PROTECTED,registered_digest(plan,PROTECTED)))==21444,'Registered protection manifest')
    before=evidence.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(before['completed'])==222 and sum(r['trials'] for r in before['completed'])==1183 and not before['registered_queue'],'Pre-registration state differs')
    prior_binding=plan['predecessor_binding'];need(prior_binding['directory']==PRIOR.name,'Wrong predecessor')
    for filename,digest in prior_binding['input_sha256'].items():evidence.pin(filename,digest)
    prior_plan=evidence.load(PRIOR/'PLAN.json',registered_digest(plan,PRIOR/'PLAN.json'))
    prior_plan_sha=sha(PRIOR/'PLAN.json')
    prior_root=evidence.load(PRIOR/'COMPLETION.json',registered_digest(plan,PRIOR/'COMPLETION.json'))
    need(before['completed'][-1]['directory']==PRIOR.name and before['completed'][-1]['completion']['sha256']==sha(PRIOR/'COMPLETION.json'),'Prior central identity')
    prior_done=evidence.load(PRIOR/'results/COMPLETION.json',registered_digest(plan,PRIOR/'results/COMPLETION.json'))
    need(prior_root['plan_sha256']==prior_done['plan_sha256']==prior_plan_sha and prior_root['result_completion_sha256']==sha(PRIOR/'results/COMPLETION.json') and prior_done['status']=='COMPLETE','Prior outcome lineage')
    prior_results=evidence.load(PRIOR/'results/RESULTS.json',prior_done['artifacts_sha256']['RESULTS.json'])
    tests=evidence.load(TESTS,registered_digest(plan,TESTS))
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
    la=evidence.load(OUT/'prepared/LABEL_AUDIT.json');ra=evidence.load(OUT/'prepared/RANK_AUDIT.json')
    training_path=plan['endpoint_source_binding']['inputs']['training_meta']['path']
    training_meta=evidence.load(training_path,registered_digest(plan,training_path))
    calendar=evidence.load(OUT/'prepared/CALENDAR.json')
    need(calendar==evidence.load(CALENDAR,registered_digest(plan,CALENDAR)),'Original official calendar retained')
    coverage=model_coverage(la,audit,ra,prepared,pc,closure,meta,training_meta,calendar)
    need(pc['model_fit_accounting']==closure['model_fit_accounting']==prepared['model_fit_accounting']==plan['model_fit_accounting']==plan['contract']['model_fit_accounting'],'Paired new-target model accounting')
    for name,keys in (('FEATURES.npz',VALUE_KEYS),('LABELS.npz',LABEL_KEYS),('TRANSFORMS_DELTAS.npz',tuple(p+a for p in ('transform_','delta_') for a in ARMS[1:]))):
        with zipfile.ZipFile(OUT/'prepared'/name) as archive:
            need(set(archive.namelist())=={k+'.npy' for k in keys} and len(archive.namelist())==len(keys),'Exact saved NPZ member names without array decoding: '+name)
    cp=evidence.load(OUT/'prepared/COHORT_PARITY.json');need(cp==prepared['cohort_parity'],'Cohort certificate binding')
    need(cp['passed'] is True and cp['arrays_dtype_shape_bytes_exact'] is True and cp['label_keys_checked']==['dates','ends','symbols','base_good','good'],'Source209 identities/masks exact')
    ref=plan['predecessor_binding']['cohort_reference']
    need(cp['prior_plan_sha256']==ref['prior_plan_sha256'] and cp['prior_prepared_completion_sha256']==ref['prior_prepared_completion_sha256'],'Prior209 prepared lineage')
    for k in ('old_targets_loaded','old_forecasts_loaded','old_orders_loaded','old_returns_read'):need(cp[k] is False,'No false old-target/prediction parity')
    source=evidence.load(OUT/'prepared/SOURCE_PARITY.json');binding=plan['source_review_binding']
    need(source['raw_source_hashes']==plan['raw_source_hashes'] and len(plan['raw_source_hashes'])==64 and source['source_review_binding']==binding and source['endpoint_source_binding']==plan['endpoint_source_binding'],'Original155 source lineage')
    for k in ('independent_raw_prices_verified','training_identity_verified','official_calendar_independently_verified','source_cohort_exact'):need(source[k] is True,'Source parity '+k)
    sr=evidence.load(SOURCE_REVIEW,SOURCE_REVIEW_SHA)
    need(path(binding['path'])==SOURCE_REVIEW and binding['sha256']==SOURCE_REVIEW_SHA and sr['passed'] is True and sr['context_sha256']==CONTEXT_SHA and sr['evidence_sha256']==context['evidence_sha256'],'Registered final source review')
    for mapping in (binding['input_sha256'],sr['source_sha256'],sr['evidence_sha256']):
        for filename,digest in mapping.items():evidence.pin(filename,digest)
    for k in ('source_only','source_admission_metadata_only','metadata_axes_verified','calendar_clock_verified','raw_source_hashes_verified','raw_source_map_matches_original','endpoint155_wiring_verified','prior209_complete_path_certificate_verified','prior209_prepared_manifest_verified','annual_cutoff_calendar_verified','canonical_identity_scope_reviewed','both225_targets_new'):
        need(sr[k] is True,'Metadata source scope '+k)
    for k in ('old_numeric_label_parity_claimed','old_control_prediction_parity_claimed','old_control_order_parity_claimed','source_code_imported','npz_decoded','actual_source_coverage_recomputed','actual_new_support_computed','actual_features_computed','actual_labels_computed','actual_model_fits_computed','actual_orders_computed','actual_returns_computed','strategy_registered','observers_modified','central_modified','strict_point_in_time'):
        need(sr[k] is False,'Source computation/admission scope '+k)
    need(sr['training_path_coverage']==binding['training_path_coverage'] and sr['training_path_coverage']['admitted_good_rows']==coverage['good_training_rows'] and sr['post_registration_required_cohort_parity']==cp['label_keys_checked'],'Inherited complete-path certificate')
    dedup=evidence.load(COVERAGE,COVERAGE_SHA)
    need(dedup['family_novelty'] is False and dedup['context_sha256']==CONTEXT_SHA and sr['coverage_sha256']==COVERAGE_SHA and registered_digest(plan,COVERAGE)==COVERAGE_SHA,'Coverage pin and bounded novelty')
    for k in ('parameter_search','actual_new_labels','actual_new_features','actual_new_support_computed','actual_model_fits_computed','actual_orders_computed','actual_returns_computed','npz_decoded','strategy_registered','strict_point_in_time','observers_modified'):need(dedup[k] is False,'Coverage scope '+k)
    for item in dedup['reviewed_definitions']:evidence.pin(item['path'],item['sha256'])
    evidence.pin(sr['implementation']['path'],sr['implementation']['sha256'])
    results=evidence.load(OUT/'results/RESULTS.json');independent=evidence.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(results['selection']==frozen['selection'] and results['strict_pit'] is False and results['reused_sample'] is True and results['observer_promotion'] is False and results['relative_lag_duration_target'] is True and results['same_support_both_arms'] is True,'Outcome interpretation or selection differs')
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
    need(len(state['completed'])==223 and sum(x['trials'] for x in state['completed'])==1186 and state['registered_queue']==[],'Central postclosure counters')
    need(state['completed'][-1]['directory']==OUT.name and state['completed'][-1]['completion']['sha256']==expected and path(state['completed'][-1]['completion']['path'])==OUT/'COMPLETION.json','Central closure reference')
    need(state['best_training_selected_candidate']==expected_best and state['latest_training_selected_candidate']['selection']==selection,'Central selected/best mismatch')
    evidence.pin(Path(__file__))
    report=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),reviewer_role='Independent numerical oracle author; this separate saved-evidence reader imports no model, oracle or execution engine.',scope='Saved JSON/hash/identity/count/selection/accounting only. NPZ directory member names inspected without decoding arrays. Only saved integer path/support counters and causal row identities are rechecked; no numerical labels, forecasts, coefficients, ranks, orders or returns recomputed.',completion_sha256=expected,plan_sha256=plan_sha,prepared_completion_sha256=sha(OUT/'prepared/COMPLETION.json'),result_completion_sha256=sha(OUT/'results/COMPLETION.json'),input_sha256=evidence.pins,
        formal_synthetic_tests=test_count,reviewed_numerical_files=6,model_flags_checked=list(MODEL_FLAGS),delta_flags_checked=list(DELTA_FLAGS),model_coverage=coverage,selection=selection,changed_holdings=changes,metrics=metrics,fee_evidence=fee_evidence,incumbent_versus224=incumbent,independent_recorded_metric_residuals=ACCOUNTING_RESIDUALS,
        independent_metric_tolerance=dict(absolute=1e-7,relative=2e-11,scope='Only production-vs-independent accounting nine-statistic checks use this established bound. Across rounds all incumbent metrics including cumulative fees are EXACT; only cost-path individual HOLDINGS.fee <=1e-10. Zero-cost fees exact0.'),
        independent_paths=9,independent_slices=45,independent_training_groups=18,zero_cost_streams=6,order_prefixes=123,source_prefixes=list(FEATURE_PREFIXES),original_array_count=27,protected_files_verified_by_registered_host_and_closer=21444,protected_archive_reread_in_this_review=False,limitations=context['limitations'],strict_pit=False,reused_sample=True,observers_modified=False,actual_new_features_computed=False,actual_new_parameters_computed=False,actual_new_orders_computed=False,actual_portfolio_returns_computed=False,source_npz_decoded=False,best=expected_best,target_reached=closure['target_reached'],completed_entries=223,trials=1186,central_snapshot_sha256=hashlib.sha256(central_bytes).hexdigest())
    lines=['# 第225轮：相对落后的连续时长','', '候选通过前88期筛选。' if selection['admitted'] else '候选未通过前88期筛选，保留原Local67。','',
        '候选预测下个完整自然月中，相对SPY累计落后的最长连续交易日占比；对照预测同一月总落后交易日占比。两者使用相同完整价格路径，平手打断连续段，每年仅用此前五年且上一年末已成熟的原始plain10样本做配对Ridge。固定0.01的负预测排名微调加入原Local67，最多正负0.005；初始部分月份不变。原长期动量核心、股票池、月末等权与月内固定持仓保留。','',
        '|执行口径|原Local67：年化 / Sharpe / 回撤|最长连续落后|总落后天数对照|','|---|---:|---:|---:|']
    def fmt(value):return f"{value['cagr']:.2%} / {value['sharpe']:.3f} / {value['max_drawdown']:.2%}"
    for name,label in (('same_close_zero','同收盘零成本'),('next_open_zero','次开盘零成本'),('next_open_cost','次开盘每边0.05%')):lines.append('|'+label+'|'+'|'.join(fmt(metrics[a][name]['full122']) for a in ARMS)+'|')
    lines+=['','|同收盘零成本区间|原Local67|最长连续落后|总落后天数对照|','|---|---:|---:|---:|']
    for label in BOUNDS:lines.append('|'+label+'|'+'|'.join(fmt(metrics[a]['same_close_zero'][label]) for a in ARMS)+'|')
    lines+=['','|持仓集合变化比较|全期|前88|前53|后35训练期|后34诊断期|','|---|---:|---:|---:|---:|---:|']
    for label,change in changes.items():lines.append('|'+label+'|'+'|'.join(str(change['counts'][k]) for k in BOUNDS)+'|')
    c=coverage['path_counts']
    lines+=['',f"本次完整路径{c['source_path_rows']}行，正式训练可用{coverage['good_training_rows']}行；主流程新做{coverage['main_paired_annual_fits']}个配对年度Ridge拟合，独立与前缀重算单列。",'',
        f'已核{test_count}项正式合成测试、6份数值代码、17模型标志、6路径计数、5个物理数据前缀及123订单前缀、9路径45区间与18训练核算。六条零成本流逐笔/逐月费用严格为零。','',
        'Local67所有收益指标、累计费用、区间、周转额、成交、每日净值与仓位均须与224精确一致。仅收费路径逐期HOLDINGS.fee允许既有1e-10误差，并记录全部差值；累计费用仍精确。','',
        '前88期决定是否采用；后34期和全期仅作已反复使用样本的诊断。时长目标不考虑落后幅度；标签在0至1且连续天数不超过总天数，Ridge预测不保证这些约束。仅继承209的原始身份及完整路径准入，不宣称复现旧标签或预测。数据含回溯复权与原股票池假设，不是严格PIT或新鲜样本外。本报告只核验已存证据，不重算模型或收益，不更改观察账户。','']
    need(CENTRAL.read_bytes()==central_bytes,'Central changed during review')
    with (OUT/'RESEARCH_REVIEW.json').open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (OUT/'RESEARCH_REVIEW.md').open('x') as stream:stream.write('\n'.join(lines))
    print(json.dumps(dict(passed=True,review_sha256=sha(OUT/'RESEARCH_REVIEW.json'),markdown_sha256=sha(OUT/'RESEARCH_REVIEW.md'),selection=selection,path_counts=coverage['path_counts'],same_close={a:metrics[a]['same_close_zero']['full122'] for a in ARMS}),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completion-sha256',required=True);parser.add_argument('--tests',type=int,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--code-review-sha256',required=True)
    args=parser.parse_args();main(args.completion_sha256,args.tests,args.plan_sha256,args.code_review_sha256)

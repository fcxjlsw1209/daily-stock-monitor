"""Read terminal222 saved evidence; no array decoding or replay.
Fee tolerances inherited from frozen220 before actual222 results: individual
next_open_cost HOLDINGS.fee <=1e-10; only metrics.fees_paid/full122.fees_paid
use1e-7+2e-11*abs(reference). All other incumbent values and zero fees exact.
Run only after root authorization of explicit terminal COMPLETION SHA.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_idio_risk222'
PRIOR=ROOT/'outputs/monthly_tail_moment221'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CONTEXT=ROOT/'work/monthly_idio_risk222_context.json'
CONTEXT_SHA='68bc4e9362a9c4489d2a7725961ff84f262e5ae844ddb80ff46c8122879b65a9'
SOURCE_REVIEW=ROOT/'work/idio_risk222_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA='13c45f4009f5b580364f0584c3ad6427d323184fbd2f230d2854dc43d29a53f2'
TESTS=ROOT/'work/monthly_idio_risk222_TESTS.json'
CODE_REVIEW=ROOT/'work/idio_risk222_CODE_REVIEW.json'
LOCAL=ROOT/'outputs/monthly_local_linear70'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
ARMS=('incumbent_local67','idiosyncratic_core','total_volatility_control')
PATHS={'same_close_zero':('close',0.),'next_open_zero':('open',0.),'next_open_cost':('open',.0005)}
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FEATURES.npz','TRANSFORMS_DELTAS.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
ESTIMATES=('formation_price_observations','risk_price_observations','valid_price_observations','simple_returns','descriptive_beta_fits','normalized_momentum_scores','ranked_observations')
COUNT_FIELDS=('checked_cells','source_available_cells','available_cells','unavailable_cells',*ESTIMATES)
FEATURE_FLAGS=('passed','independent_scalar_rebuild','skip42_cumulative_momentum_checked','current63_risk_window_checked','full_source_windows_checked','stock_market_centering_checked','exact_ols_projection_checked','residual_not_recentered_checked','denominator62_checked','zero_risk_common_exclusion_checked','self_market_zero_residual_checked','common_rank_domain_checked','original_pct_ranks_checked','official_session_windows_checked','monthend_queries_checked','exact_common_availability','audit_checked','no_future_or_outcome_inputs','descriptive_fit_counts_checked','no_predictive_fits_checked')
VALUE_KEYS=('available','candidate','control','momentum','beta','idiosyncratic_risk','total_risk','raw_candidate','raw_control')
DELTA_FLAGS=('passed', 'half_core_formula_checked', 'original_pct_ranks_preserved', 'first_partial_unchanged', 'calm_rows_unchanged', 'missing_preserves_core', 'core_weights_checked', 'no_second_rank', 'independent_delta_rebuild')
NUMERICAL_FILES=('monthly_idio_risk222_host.py', 'test_monthly_idio_risk222_host.py', 'monthly_idio_risk222_kernel.py', 'test_monthly_idio_risk222_kernel.py', 'monthly_idio_risk222_independent.py', 'test_monthly_idio_risk222_independent.py')
PERIOD_FEE_ATOL=1e-10
ACCOUNTING_RESIDUALS=[]
EXPECTED_TESTS=88
# Final PLAN and code-review digests must be provided explicitly at authorized execution.


COVERAGE=ROOT/'work/after221_idio_risk_coverage.json'
COVERAGE_SHA='bb6f530a6b3d8d650dd419c96dba86b832a720a1d6523cd543e8f923eb1f9f82'
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
        close_vs_total_control=guard(c['same_close_zero'],m['same_close_zero'],0.),
        open_vs_total_control=guard(c['next_open_cost'],m['next_open_cost'],0.),
        changed_support=support(ids),changed_support_vs_control=support(ids2))
    admitted=all(checks.values())
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=checks,
        changed_months=ids,changed_months_vs_control=ids2,first88_only=True,later34_used=False)


def feature_coverage(feature,prepared,completion,closure,meta,calendar):
    need(feature.get('events')==122 and feature.get('symbols_per_event')==49 and feature.get('checked_cells')==5978,'Full feature cell audit')
    need(feature['price_count']==253 and feature['skip_sessions']==42 and feature['risk_price_count']==64 and feature['risk_returns']==63 and feature['market_symbol']=='SPY' and feature['sample_ddof']==1 and feature['residual_recentered'] is False,'Fixed formation/current-risk contract')
    need(feature['price_observation_counting']=='consumed observations: 253 formation + 64 stock risk + 64 market risk; overlaps repeated','Observation counting contract')
    for key in ('common_availability','ranking_performed'):need(feature[key] is True,'Feature shared rank contract')
    for key in ('ranking_eligibility_filtered','trading_eligibility_modified','portfolio_returns_calculated'):need(feature[key] is False,'Unexpected feature behavior')
    history=feature['event_history'];need(len(history)==122,'Full event audit')
    reasons=Counter();security={};source_counts=[];available=[];fit_counts=[];different_counts=[]
    byera={name:0 for name in BOUNDS};reasonsera={name:Counter() for name in BOUNDS}
    sessions=calendar['sessions'];pos={day:i for i,day in enumerate(sessions)}
    source_fields=('momentum','stock_mean','market_mean','stock_centered_sum_squares','market_centered_sum_squares','cross_product_sum','total_risk')
    fit_fields=('beta','residual_sum_squares','idiosyncratic_risk')
    target_fields=('raw_candidate','raw_control','candidate','control')
    endpoints=('start_price','formation_end_price','stock_risk_start_price','stock_risk_end_price','market_risk_start_price','market_risk_end_price')
    for i,event in enumerate(history):
        need(event['date']==meta['dates'][i] and event['symbols']==meta['symbols'][i] and len(event['cells'])==49,'Event identity')
        t=pos[event['date']];formation=sessions[t-294:t-41];risk=sessions[t-63:t+1]
        need(event['session_position']==t and len(formation)==253 and len(risk)==64,'Official window lengths')
        need(event['price_start']==event['formation_price_start']==formation[0] and event['price_end']==event['formation_price_end']==formation[-1],'Official formation clock')
        need(event['risk_price_start']==risk[0] and event['first_risk_return_date']==risk[1] and event['risk_price_end']==risk[-1]==event['date'],'Official current63 stock/market risk clock')
        need(type(event['missing_formation_sessions']) is list and type(event['missing_risk_sessions']) is list,'Missing-session audit arrays')
        need(set(event['missing_formation_sessions'])<=set(formation) and set(event['missing_risk_sessions'])<=set(risk),'Missing-session audit clock')
        s=0;a=0;fits=0;different=0
        for j,cell in enumerate(event['cells']):
            symbol=meta['symbols'][i][j]
            need(cell['date']==event['date'] and cell['symbol']==symbol and type(cell['source_available']) is bool and type(cell['available']) is bool,'Cell identity/status')
            need(type(cell['descriptive_beta_fit']) is int and cell['descriptive_beta_fit'] in (0,1),'Descriptive fit count type')
            reason=cell['reason'];need(type(reason) is str and reason,'Source reason');reasons[reason]+=1
            item=security.setdefault(symbol,dict(source_available_cells=0,available_cells=0,descriptive_beta_fits=0,reasons=Counter()));item['reasons'][reason]+=1
            for name,(lo,hi) in BOUNDS.items():
                if lo<=i<hi:reasonsera[name][reason]+=1
            if cell['source_available']:
                s+=1;item['source_available_cells']+=1
                need(not event['missing_formation_sessions'] and not event['missing_risk_sessions'],'Valid source cannot miss an official source session')
                need(all(finite(cell[k]) for k in source_fields),'Source descriptors')
                need(all(finite(cell[k]) and cell[k]>0 for k in endpoints),'Source endpoints')
                need(all(cell[k]>=0 for k in ('stock_centered_sum_squares','market_centered_sum_squares','total_risk')),'Nonnegative saved total/market risks')
                if cell['market_centered_sum_squares']==0.:
                    need(cell['available'] is False and reason=='zero_market_variance' and cell['descriptive_beta_fit']==0,'Zero-market common support')
                    need(all(cell[k] is None for k in fit_fields+target_fields),'Zero-market fit and targets must be absent')
                else:
                    fits+=1;item['descriptive_beta_fits']+=1
                    need(cell['descriptive_beta_fit']==1 and all(finite(cell[k]) for k in fit_fields),'One current-window descriptive projection required')
                    need(cell['residual_sum_squares']>=0 and cell['idiosyncratic_risk']>=0,'Nonnegative saved residual risk')
                    if cell['total_risk']==0.:
                        need(cell['stock_centered_sum_squares']==0. and cell['available'] is False and reason=='zero_total_risk','Zero-total common support')
                        need(all(cell[k] is None for k in target_fields),'Zero-total targets must be absent')
                    elif cell['idiosyncratic_risk']==0.:
                        need(cell['residual_sum_squares']==0. and cell['available'] is False and reason=='zero_idiosyncratic_risk','Zero-residual common support')
                        need(all(cell[k] is None for k in target_fields),'Zero-residual targets must be absent')
                    else:
                        a+=1;item['available_cells']+=1
                        need(cell['available'] is True and reason=='available' and cell['stock_centered_sum_squares']>0 and cell['residual_sum_squares']>0,'Positive-risk support')
                        need(all(finite(cell[k]) for k in target_fields),'Paired targets')
                        need(all(-.5<=cell[k]<=.5 for k in ('candidate','control')),'Rank bounds')
                        different+=int(cell['candidate']!=cell['control'])
                    if symbol=='SPY':
                        need(cell['available'] is False and cell['beta']==1. and cell['residual_sum_squares']==0. and cell['idiosyncratic_risk']==0.,'SPY self projection must be excluded from both ranks')
            else:
                need(cell['available'] is False and cell['descriptive_beta_fit']==0,'Invalid source cannot fit or rank')
                need(all(cell[k] is None for k in source_fields+fit_fields+target_fields+endpoints),'Invalid source values/endpoints')
        need(event['source_available']==[c['source_available'] for c in event['cells']] and event['available']==[c['available'] for c in event['cells']],'Event masks')
        need(type(event['source_available_count']) is int and event['source_available_count']==s and type(event['available_count']) is int and event['available_count']==a and type(event['descriptive_beta_fits']) is int and event['descriptive_beta_fits']==fits,'Event counts')
        need(0<=a<=fits<=s<=49,'Common support/fit relationship')
        source_counts.append(s);available.append(a);fit_counts.append(fits);different_counts.append(different)
        for name,(lo,hi) in BOUNDS.items():
            if lo<=i<hi:byera[name]+=a
    s=sum(source_counts);a=sum(available);f=sum(fit_counts)
    counts=dict(checked_cells=5978,source_available_cells=s,available_cells=a,unavailable_cells=5978-a,formation_price_observations=253*s,risk_price_observations=128*s,valid_price_observations=381*s,simple_returns=126*s,descriptive_beta_fits=f,normalized_momentum_scores=2*a,ranked_observations=2*a)
    need(len(counts)==11 and set(counts)==set(COUNT_FIELDS),'Eleven counters')
    for record in (feature,prepared['feature']):
        need(record['events']==122 and record['symbols_per_event']==49,'Independent axes')
        need(record['source_available_by_event']==source_counts and record['available_by_event']==available and record['descriptive_beta_fits_by_event']==fit_counts,'Independent masks and descriptive-fit series')
        for key,value in counts.items():need(type(record.get(key)) is int and record[key]==value,'Count lineage: '+key)
    need(feature['reason_counts']==dict(reasons),'Reason counts')
    for record in (prepared,completion,closure):
        for key in ESTIMATES:need(type(record.get(key)) is int and record[key]==counts[key],'Estimate lineage: '+key)
    for record in (feature,prepared['feature'],prepared,completion,closure):
        for key in ('new_model_fits','predictive_model_fits'):need(type(record.get(key)) is int and record[key]==0,'Unexpected predictive fit')
    for item in security.values():item['reasons']=dict(sorted(item['reasons'].items()))
    return dict(**counts,source_available_by_event=source_counts,available_by_event=available,descriptive_beta_fits_by_event=fit_counts,source_supported_months=sum(x>0 for x in source_counts),feature_supported_months=sum(x>0 for x in available),support_by_era=byera,reasons=dict(sorted(reasons.items())),reasons_by_era={k:dict(sorted(v.items())) for k,v in reasonsera.items()},by_security=dict(sorted(security.items())),saved_candidate_control_different_rank_cells=sum(different_counts),saved_difference_rank_cells_by_era={k:sum(different_counts[lo:hi]) for k,(lo,hi) in BOUNDS.items()},estimate_counts={k:counts[k] for k in ESTIMATES},feature_value_keys=list(VALUE_KEYS),actual_features_recomputed=False)


def compare_incumbent(evidence,prior_completion,results,prior_results):
    report={}
    for name in PATHS:
        old=prior_results['arms'][ARMS[0]][name];new=results['arms'][ARMS[0]][name]
        accumulated_fee_residuals=[]
        def exact_metrics_with_declared_fee_exception(previous,current,location,allow_fee):
            need(set(previous)==set(current),'Incumbent metric schema differs: '+location)
            for key in previous:
                old_value=previous[key];new_value=current[key]
                if old_value==new_value:continue
                need(name=='next_open_cost' and allow_fee and key=='fees_paid','Nonfee incumbent metric differs: '+location+'/'+key)
                need(finite(old_value) and finite(new_value),'Nonfinite accumulated incumbent fee')
                delta=new_value-old_value;bound=1e-7+2e-11*abs(old_value)
                need(abs(delta)<=bound,'Accumulated fee exceeds frozen219 accounting tolerance')
                accumulated_fee_residuals.append(dict(field=location+'/'+key,previous221=old_value,current222=new_value,difference=delta,absolute_difference=abs(delta),allowed_absolute_difference=bound))
        exact_metrics_with_declared_fee_exception(old['metrics'],new['metrics'],'metrics',True)
        need(set(old['slices'])==set(new['slices'])==set(BOUNDS),'Incumbent slice schema differs')
        for label in BOUNDS:
            exact_metrics_with_declared_fee_exception(old['slices'][label],new['slices'][label],'slices/'+label,label=='full122')
        exported={}
        for kind in ('TRADES','DAILY','DAILY_POSITIONS','HOLDINGS'):
            filename=ARMS[0]+'_'+name+'_'+kind+'.json'
            previous=evidence.load(PRIOR/'results'/filename,prior_completion['artifacts_sha256'][filename])
            current=evidence.load(OUT/'results'/filename)
            if kind!='HOLDINGS':need(current==previous,'Incumbent export must be EXACT: '+name+'/'+kind)
            exported[kind]=(previous,current)
        prior_holds,current_holds=exported['HOLDINGS'];need(len(prior_holds)==len(current_holds)==122,'Incumbent periods')
        residuals=[]
        for old_row,new_row in zip(prior_holds,current_holds):
            need(set(old_row)==set(new_row) and {k:v for k,v in old_row.items() if k!='fee'}=={k:v for k,v in new_row.items() if k!='fee'},'Incumbent nonfee HOLDINGS field differs')
            need(finite(old_row['fee']) and finite(new_row['fee']),'Nonfinite incumbent fee')
            if old_row['fee']!=new_row['fee']:
                delta=new_row['fee']-old_row['fee']
                need(name=='next_open_cost' and abs(delta)<=PERIOD_FEE_ATOL,'Only existing period-fee1e-10 allowance is permitted')
                residuals.append(dict(signal_date=new_row['signal_date'],execution_date=new_row['execution_date'],previous221=old_row['fee'],current222=new_row['fee'],difference=delta))
        totals={}
        for label,holds,metrics in (('previous221',prior_holds,old['metrics']),('current222',current_holds,new['metrics'])):
            total=0.
            for row in holds:total+=row['fee']
            need(total==metrics['fees_paid'],'Saved chronological fees do not exactly sum to metric')
            totals[label]=total
        # Each saved chronological sum must equal its own metric exactly; the only
        # permitted cross-round difference is the metrics.fees_paid exception above.
        report[name]=dict(all_metrics_and_slices_exact=not accumulated_fee_residuals,all_nonfee_metrics_and_slices_exact=True,accumulated_fee_differences=accumulated_fee_residuals,all_fills_exact=True,daily_equity_exact=True,daily_positions_exact=True,
            holding_nonfee_fields_exact=True,total_fees_exact=totals['previous221']==totals['current222'],fee_totals=totals,period_fee_differences=residuals,
            max_absolute_period_fee_difference=max([abs(x['difference']) for x in residuals],default=0.))
    return dict(paths=report,period_fee_tolerance=PERIOD_FEE_ATOL,
        tolerance_scope='Only individual next_open_cost HOLDINGS.fee may differ by at most1e-10. Only next_open_cost metrics.fees_paid and slices.full122.fees_paid use the existing219 accumulated-accounting tolerance1e-7+2e-11*abs(previous). All other metrics/slices including turnover, fills, equity, positions and all nonfee holding fields are exact.',accumulated_fee_tolerance_source='work/risk_consensus219_saved_review.py:34-35,124',prospective_inheritance='Frozen221 compare_incumbent policy (already inherited from220) adopted before actual222 features/orders/results; no result-driven tolerance change.',
        explanation='Frozen engine iterates a set when summing turnover; per-period fee summation tails may vary between processes. No new replay or relaxed total/metric comparison is used.')


def main(expected,test_count,expected_plan_sha,expected_code_review_sha):
    need(type(test_count) is int and test_count==EXPECTED_TESTS,'Registered88 formal synthetic count required')
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
    need(plan['contract']['context']==context and context['arms']==list(ARMS) and context['integration']['blend']==.5,'Contract/source theory differs')
    need(plan['protected_files']==21444 and len(evidence.load(PROTECTED,registered_digest(plan,PROTECTED)))==21444,'Registered protection manifest')
    before=evidence.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(before['completed'])==219 and sum(r['trials'] for r in before['completed'])==1174 and not before['registered_queue'],'Pre-registration state differs')
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
    prepared=evidence.load(OUT/'prepared/INDEPENDENT_PREPARE.json');feature=evidence.load(OUT/'prepared/FEATURE_AUDIT.json')
    for key in ('original_orders_exact','original_arrays_unchanged','single_original_episode','original_core_metadata_and_support_preserved','source_support_exact','source_admission_metadata_only','feature_prefix_sources_truncated','feature_prefix_metadata_truncated','feature_prefix_independently_verified'):
        need(prepared.get(key) is True,'Independent prepare invariant missing: '+key)
    need(prepared['prefixes']==123 and prepared['feature_prefixes']==[0,1,53,88,122] and prepared['original_array_count']==pc['original_array_count']==closure['original_array_count']==27,'Prefix/original-array coverage')
    need(pc['new_returns_computed'] is False,'Preparation must not calculate returns')
    for key in FEATURE_FLAGS:need(prepared['feature'].get(key) is True,'Independent feature flag missing: '+key)
    for key in DELTA_FLAGS:need(prepared['delta'].get(key) is True,'Independent delta flag missing: '+key)
    need(prepared['feature']['computes_returns'] is False and prepared['feature']['producer_modules_imported'] is False and prepared['feature']['numerical_tolerance']==0.,'Independent numeric scope differs')
    need(prepared['delta']['blend']==.5 and prepared['delta']['max_absolute_delta']==.26 and set(prepared['delta']['changed_cells'])==set(ARMS[1:]),'Independent fixed score budget differs')
    for record in (prepared,pc,closure):
        need(type(record['new_predictive_model_fits']) is int and record['new_predictive_model_fits']==0 and record['model_fit_accounting']==plan['model_fit_accounting'],'Predictive fit/accounting scope differs')
    calendar=evidence.load(CALENDAR,registered_digest(plan,CALENDAR))
    coverage=feature_coverage(feature,prepared,pc,closure,meta,calendar)
    source=evidence.load(OUT/'prepared/SOURCE_PARITY.json');binding=plan['price_source_binding']
    need(source['raw_source_hashes']==plan['raw_source_hashes'] and len(plan['raw_source_hashes'])==64 and source['price_source_binding']==binding,'Source lineage')
    sr=evidence.load(SOURCE_REVIEW,SOURCE_REVIEW_SHA)
    need(path(binding['path'])==SOURCE_REVIEW and binding['sha256']==SOURCE_REVIEW_SHA and sr['passed'] is True and sr['context_sha256']==CONTEXT_SHA,'Registered source review')
    for key in ('source_evidence_sha256','context_evidence_sha256'):
        for filename,digest in binding[key].items():evidence.pin(filename,digest)
    for key in ('source_admission_metadata_only','metadata_axes_verified','calendar_clock_verified','raw_source_hashes_verified','raw_source_map_matches_original','integration_source_verified','source_window_matches199','current63_market_risk_window_checked'):need(sr[key] is True,'Source metadata flag')
    for key in ('actual_new_features_computed','actual_new_orders_computed','actual_new_returns_computed','npz_decoded','actual_source_coverage_recomputed'):need(sr[key] is False,'Source computation boundary')
    dedup=evidence.load(COVERAGE,COVERAGE_SHA)
    need(dedup['family_novelty'] is False and dedup['new_features_computed']==dedup['new_returns_computed']==dedup['new_orders_computed']==0,'Limited coverage')
    need(sr['coverage_sha256']==COVERAGE_SHA and registered_digest(plan,COVERAGE)==COVERAGE_SHA,'Coverage pin')
    for old_reader in ('work/risk_consensus219_saved_review.py','work/roa_stability220_saved_review.py','work/tail_moment221_saved_review.py'):evidence.pin(old_reader)
    results=evidence.load(OUT/'results/RESULTS.json');independent=evidence.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(results['selection']==frozen['selection'] and results['strict_pit'] is False and results['reused_sample'] is True and results['observer_promotion'] is False and results['idio_risk'] is True and results['same_support_both_arms'] is True,'Outcome interpretation or selection differs')
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
    need(len(state['completed'])==220 and sum(x['trials'] for x in state['completed'])==1177 and state['registered_queue']==[],'Central postclosure counters')
    need(state['completed'][-1]['directory']==OUT.name and state['completed'][-1]['completion']['sha256']==expected and path(state['completed'][-1]['completion']['path'])==OUT/'COMPLETION.json','Central closure reference')
    need(state['best_training_selected_candidate']==expected_best and state['latest_training_selected_candidate']['selection']==selection,'Central selected/best mismatch')
    evidence.pin(Path(__file__))
    report=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),reviewer_role='Source-metadata and saved-evidence reviewer; not producer or host author. Independent numerical rebuild is the separate pinned oracle.',scope='Saved JSON/hash/source coverage/selection/accounting only; no arrays decoded or new features/orders/returns calculated.',completion_sha256=expected,plan_sha256=plan_sha,prepared_completion_sha256=sha(OUT/'prepared/COMPLETION.json'),result_completion_sha256=sha(OUT/'results/COMPLETION.json'),input_sha256=evidence.pins,
        formal_synthetic_tests=test_count,reviewed_numerical_files=6,feature_flags_checked=list(FEATURE_FLAGS),delta_flags_checked=list(DELTA_FLAGS),source_coverage=coverage,selection=selection,changed_holdings=changes,metrics=metrics,fee_evidence=fee_evidence,incumbent_versus221=incumbent,independent_recorded_metric_residuals=ACCOUNTING_RESIDUALS,
        independent_metric_tolerance=dict(absolute=1e-7,relative=2e-11,scope='Independent nine-statistic comparisons use pinned accounting bound. Across rounds only next_open_cost metrics.fees_paid/full122.fees_paid use it; period HOLDINGS.fee1e-10 was inherited from frozen220/221 before222 results. All residuals listed; zero-cost fees and other incumbent values exact.'),
        independent_paths=9,independent_slices=45,independent_training_groups=18,zero_cost_streams=6,order_prefixes=123,source_prefixes=[0,1,53,88,122],original_array_count=27,protected_files_verified_by_registered_host_and_closer=21444,limitations=context['limitations'],strict_pit=False,reused_sample=True,observers_modified=False,actual_new_features_computed=False,actual_new_orders_computed=False,actual_portfolio_returns_computed=False,source_npz_decoded=False,best=expected_best,target_reached=closure['target_reached'],completed_entries=220,trials=1177,central_snapshot_sha256=hashlib.sha256(central_bytes).hexdigest())
    lines=['# 第222轮：特质风险与总风险归一化','', '候选通过前88期筛选。' if selection['admitted'] else '候选未通过前88期筛选，保留原Local67。','',
        '长期累计动量M仍使用t−294至t−42的253个价格。风险使用截至信号日的63个简单收益：对股票与SPY收益中心化并做单因子投影，候选M除以sqrt(RSS/62)，对照M除以sqrt(stockSS/62)。不剥离M中的市场收益。仅非平静正式月末替换一半原长期动量核心，初始部分月份和calm行不变；原股票池、辅助分、机器学习、单次episode和月内固定持仓保留。','',
        '|执行口径|原Local67：年化 / Sharpe / 回撤|特质风险候选|总风险对照|','|---|---:|---:|---:|']
    def fmt(value):return f"{value['cagr']:.2%} / {value['sharpe']:.3f} / {value['max_drawdown']:.2%}"
    for name,label in (('same_close_zero','同收盘零成本'),('next_open_zero','次开盘零成本'),('next_open_cost','次开盘每边0.05%')):lines.append('|'+label+'|'+'|'.join(fmt(metrics[a][name]['full122']) for a in ARMS)+'|')
    lines+=['','|零成本同收盘区间|原Local67|特质风险候选|总风险对照|','|---|---:|---:|---:|']
    for label in BOUNDS:lines.append('|'+label+'|'+'|'.join(fmt(metrics[a]['same_close_zero'][label]) for a in ARMS)+'|')
    lines+=['','|持仓集合变化比较|全期|前88|前53|后35训练期|后34诊断期|','|---|---:|---:|---:|---:|---:|']
    for label,change in changes.items():lines.append('|'+label+'|'+'|'.join(str(change['counts'][k]) for k in BOUNDS)+'|')
    lines+=['',f"5978格中完整来源{coverage['source_available_cells']}格，正总风险和残差风险共同可用{coverage['available_cells']}格，不可用{coverage['unavailable_cells']}格。",'','|来源状态|证券月份数|','|---|---:|']
    for reason,count in coverage['reasons'].items():lines.append('|'+reason.replace('|','/')+'|'+str(count)+'|')
    lines+=['',f'核对{test_count}项正式合成测试、6份数值源码审查、20特征标志、9delta标志与11描述计数；原27数组、5组物理前缀、123订单前缀、9路径45区间和18训练核算完整。六条零成本流总费用、逐笔及逐月费用严格为零。','',
        'Local67三路径收益指标、周转额、成交、每日净值及仓位与221精确一致。仅收费路径逐月HOLDINGS.fee沿用220既有1e-10，metrics.fees_paid和full122.fees_paid沿用219既有累计容差；全部残差列于JSON。这是在222实际结果出现前继承的规范，未按结果放宽。','',
        '本轮是已有特质风险家族内的分母应用窄对照；描述性OLS投影次数明确计数，不当作零估计；沿用历史代理股票池及回溯数据，strict PIT未认证。后34期和全期已反复研究，不是独立前向样本。报告只复核已保存证据，没有重算特征、订单或收益，没有修改观察账户。','']
    need(CENTRAL.read_bytes()==central_bytes,'Central changed during review')
    with (OUT/'RESEARCH_REVIEW.json').open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (OUT/'RESEARCH_REVIEW.md').open('x') as stream:stream.write('\n'.join(lines))
    print(json.dumps(dict(passed=True,review_sha256=sha(OUT/'RESEARCH_REVIEW.json'),markdown_sha256=sha(OUT/'RESEARCH_REVIEW.md'),selection=selection,source_available_cells=coverage['source_available_cells'],available_cells=coverage['available_cells'],same_close={a:metrics[a]['same_close_zero']['full122'] for a in ARMS}),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completion-sha256',required=True);parser.add_argument('--tests',type=int,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--code-review-sha256',required=True)
    args=parser.parse_args();main(args.completion_sha256,args.tests,args.plan_sha256,args.code_review_sha256)

"""Read terminal224 saved evidence; no model imports, array decoding or replay.
Prospectively fixed before224 actual outputs: ONLY cost-path period HOLDINGS.fee
may differ <=1e-10 from223. Cumulative fees and all other incumbent fields exact.
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
OUT=ROOT/'outputs/monthly_increment_calibration224'
PRIOR=ROOT/'outputs/monthly_risk_reliability223'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CONTEXT=ROOT/'work/monthly_increment_calibration224_context.json'
CONTEXT_SHA='0381d106c1f355f752987e7cb1c4a1a07a8ee1b05e1091dff66080a6a0f17244'
SOURCE_REVIEW=ROOT/'work/increment_calibration224_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA='71f891512ac151d7b8fb9534d6fec89fac1c4a48eb79f7d99e2f436e4eba0a1f'
TESTS=ROOT/'work/monthly_increment_calibration224_TESTS.json'
CODE_REVIEW=ROOT/'work/increment_calibration224_CODE_REVIEW.json'
LOCAL=ROOT/'outputs/monthly_local_linear70'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
ARMS=('incumbent_local67','own_increment_calibration','pooled_increment_control')
PATHS={'same_close_zero':('close',0.),'next_open_zero':('open',0.),'next_open_cost':('open',.0005)}
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FORECASTS.npz','RANKS_DELTAS.npz','MODEL_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
COUNT_FIELDS=('annual_cohorts', 'pooled_fits', 'cold_start_years', 'zero_denominator_years', 'clipped_pooled_fits', 'own_history_groups', 'own_fits', 'cold_start_groups', 'zero_denominator_groups', 'clipped_own_fits', 'parameter_estimates', 'joined_label_rows', 'invalid_label_rows', 'small_month_rows', 'skipped_initial_rows', 'calibration_months', 'calibration_rows', 'own_history_rows', 'cached_forecast_rows')
MODEL_FLAGS=('passed', 'independent_scalar_rebuild', 'exact_identity_join_checked', 'cached_forecasts_reused', 'local_and_pooled_source_checked', 'annual_prior_december_cutoffs_checked', 'five_year_mature_window_checked', 'four_target_common_cohort_checked', 'historical_original_universe_checked', 'two_rows_per_month_common_cohort_checked', 'pooled_equal_month_weights_checked', 'own_security_normal_equations_checked', 'bounded_alpha_checked', 'pooled_not_ready_copy_checked', 'own_pooled_fallback_checked', 'endpoint_and_equal_prediction_copy_checked', 'no_local_history_filter', 'availability_unchanged', 'original_targets_unchanged', 'no_future_or_outcome_inputs', 'source_forecasts_not_refitted', 'audit_checked', 'parameter_estimates_counted')
SLOT_FLAGS=('passed', 'full_original_universe', 'includes_eligible_etfs', 'no_local_history_filter', 'replacement_not_additive_local_slot', 'paired_whole_row_readiness', 'initial_partial_unchanged', 'exact_rank_rebuild')
VALUE_KEYS=('raw_candidate','raw_control','available')
NUMERICAL_FILES=('monthly_increment_calibration224_host.py', 'test_monthly_increment_calibration224_host.py', 'monthly_increment_calibration224_model.py', 'test_monthly_increment_calibration224_model.py', 'monthly_increment_calibration224_independent.py', 'test_monthly_increment_calibration224_independent.py')
PERIOD_FEE_ATOL=1e-10
ACCOUNTING_RESIDUALS=[]
# Formal test count must match the explicit CLI and immutable registered receipt.
# Final PLAN and code-review digests must be provided explicitly at authorized execution.


COVERAGE=ROOT/'work/after223_increment_calibration_coverage.json'
COVERAGE_SHA='b38e92f53b04690c67aef5ab0b5e143727c61931cd6837702deef22785b67618'
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
        close_vs_pooled_increment_control=guard(c['same_close_zero'],m['same_close_zero'],0.),
        open_vs_pooled_increment_control=guard(c['next_open_cost'],m['next_open_cost'],0.),
        changed_support=support(ids),changed_support_vs_control=support(ids2))
    admitted=all(checks.values())
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=checks,
        changed_months=ids,changed_months_vs_control=ids2,first88_only=True,later34_used=False)


def model_coverage(audit, prepared, completion, closure, meta, training_meta):
    """Check saved identities, maturity, counts and branches, not refit slopes."""
    need(audit['method']=='annual_own_increment_convex_calibration' and audit['arms']==list(ARMS),'Fixed model identity')
    need(audit['query_months']==122 and audit['query_rows']==5978 and audit['symbols_per_event']==49,'Full forecast axes')
    need(audit['min_months']==18 and audit['min_rows_per_month']==2 and audit['lookback_calendar_years']==5,'Fixed maturity and cohort thresholds')
    for key in ('source_forecasts_refitted','labels_recomputed','local_available_filtered'):need(audit[key] is False,'Unexpected model behavior '+key)
    for key in ('historical_use_explicit','current_availability_unchanged','pool_uses_all_historical_ids','monthly_equal_pool_weights'):need(audit[key] is True,'Missing fixed model behavior '+key)
    td=training_meta['training']; train_keys=[(d,s,e) for d,e,ss in zip(td['dates'],td['ends'],td['symbols']) for s in ss]
    need(len(train_keys)==len(set(train_keys))==8820 and training_meta['monthly']==meta,'Original169 identity metadata')
    dates,ends,symbols=meta['dates'],meta['ends'],meta['symbols']; date_rows={d:i for i,d in enumerate(dates)}
    years=sorted({int(d[:4]) for d in dates});need([r['year'] for r in audit['annual']]==years,'Every query year exactly once')
    counts={k:0 for k in COUNT_FIELDS};status_counts=Counter();by_year=[];by_security={};seen_queries=set()
    def coefficient_record(record, months):
        n,d,a=record['numerator'],record['denominator'],record['alpha']
        need(finite(n) and finite(d) and d>=0. and type(record['clipped']) is bool,'Finite saved normal equations')
        expected='cold_start' if months<18 else 'zero_denominator' if d==0. else 'calibrated'
        need(record['status']==expected,'Declared coefficient support')
        if expected!='calibrated':need(a is None and record['clipped'] is False,'Unsupported coefficient cannot be estimated')
        else:
            need(finite(a) and 0.<=a<=1. and record['clipped']==(n<0. or n>d),'Convex coefficient boundary/clipping')
            if n<=0.:need(a==0.,'Lower endpoint copy')
            elif n>=d:need(a==1.,'Upper endpoint copy')
            else:need(0.<a<1.,'Interior coefficient')
        return expected
    for annual in audit['annual']:
        y=annual['year'];start,cutoff=f'{y-6}-12-31',f'{y-1}-12-31'
        need(annual['cutoff']==cutoff and annual['lookback_exclusive_start']==start,'Prior December five-year clock')
        months=annual['months'];need([m['date'] for m in months]==sorted(set(m['date'] for m in months)),'Chronological unique source months')
        joined=invalid=small=0;history=[];retained=[];seen_source=set()
        for month in months:
            day,end=month['date'],month['end'];need(day in date_rows and day!='2016-08-17' and day>start and end<=cutoff and end==ends[date_rows[day]],'Mature formal source month')
            rows=month['rows'];size=len(rows)
            for k in ('joined_rows','invalid_rows','valid_rows'):need(type(month[k]) is int and month[k]>=0,'Saved cohort counts')
            need(month['joined_rows']==month['invalid_rows']+size and month['valid_rows']==size and type(month['retained']) is bool and month['retained']==(size>=2),'Common two-row monthly cohort')
            source_ids=[item['source_row_id'] for item in rows]
            need(source_ids==sorted(set(source_ids)) and not(seen_source&set(source_ids)),'No duplicate historical source rows');seen_source.update(source_ids)
            for item in rows:
                q,t=item['source_row_id'],item['training_row_id']
                need(type(q) is int and 0<=q<5978 and type(t) is int and 0<=t<len(train_keys),'Source/training row bounds')
                triple=(dates[q//49],symbols[q//49][q%49],ends[q//49])
                need(triple==(day,item['symbol'],end)==train_keys[t] and item['date']==day and item['end']==end,'Exact source/label/security triple')
                need(all(finite(item[k]) for k in ('prediction','pooled_prediction','target')),'Saved valid input values')
                if size>=2:
                    need(all(finite(item[k]) for k in ('disagreement','residual_target','product','square')) and item['square']>=0.,'Retained saved product fields')
                else:need(all(item[k] is None for k in ('disagreement','residual_target','product','square')),'Small month must not form products')
            if size>=2:
                for k in ('numerator_sum','denominator_sum','numerator_mean','denominator_mean'):need(finite(month[k]),'Finite saved monthly statistics')
                need(month['denominator_sum']>=0. and month['denominator_mean']>=0.,'Nonnegative saved denominator')
                history.extend(rows);retained.append(month)
            else:
                need(all(month[k] is None for k in ('numerator_sum','denominator_sum','numerator_mean','denominator_mean')),'Small month statistics absent');small+=size
            joined+=month['joined_rows'];invalid+=month['invalid_rows']
        need(annual['joined_label_rows']==joined and annual['invalid_label_rows']==invalid and annual['small_month_rows']==small,'Annual row counters')
        need(annual['calibration_months']==len(retained) and annual['calibration_rows']==len(history) and annual['latest_label_end']==max((x['end'] for x in history),default=None),'Retained annual cohort clock/count')
        status=coefficient_record(annual,len(retained));poolready=status=='calibrated';counts['annual_cohorts']+=1
        counts[{'calibrated':'pooled_fits','cold_start':'cold_start_years','zero_denominator':'zero_denominator_years'}[status]]+=1
        counts['clipped_pooled_fits']+=int(annual['clipped'])
        groups=annual['groups'];need([g['symbol'] for g in groups]==sorted(set(g['symbol'] for g in groups)) and annual['queried_security_count']==len(groups),'Unique sorted current-ID groups')
        ownfit=cold=zero=clipped=predictions=ownrows=0
        for group in groups:
            symbol=group['symbol'];qs=group['query_row_ids'];need(qs==sorted(set(qs)) and len(qs)>0,'Unique nonempty annual query IDs')
            for q in qs:need(type(q) is int and 0<=q<5978 and q not in seen_queries and int(dates[q//49][:4])==y and symbols[q//49][q%49]==symbol,'Exact current year/security source IDs')
            seen_queries.update(qs);predictions+=len(qs)
            need(group['query_dates']==[dates[q//49] for q in qs],'Current query date identity')
            own=[h for h in history if h['symbol']==symbol];ownrows+=len(own)
            need(group['history_source_row_ids']==[h['source_row_id'] for h in own] and group['own_rows']==group['own_months']==len(own)==len({h['date'][:7] for h in own}),'Own subset of retained shared cohort')
            need(group['latest_label_end']==max((h['end'] for h in own),default=None),'Own maturity clock')
            need(finite(group['numerator']) and finite(group['denominator']) and group['denominator']>=0.,'Saved own normal equations')
            if poolready:
                gs=coefficient_record(group,len(own));ready=gs=='calibrated'
                need(group['applied_alpha']==(group['alpha'] if ready else annual['alpha']) and group['fallback']==('none' if ready else 'pooled'),'Exact pooled fallback coefficient')
            else:
                gs='pooled_not_ready';need(group['status']==gs and group['alpha'] is None and group['applied_alpha']==1. and group['clipped'] is False and group['fallback']=='original','Pool not ready exact original copy')
            ownfit+=int(gs=='calibrated');zero+=int(gs=='zero_denominator');cold+=int(gs in ('cold_start','pooled_not_ready'));clipped+=int(group['clipped']);status_counts[gs]+=1
            sec=by_security.setdefault(symbol,dict(groups=0,fits=0,cold_groups=0,zero_denominator_groups=0,history_rows=0,query_rows=0))
            sec['groups']+=1;sec['fits']+=int(gs=='calibrated');sec['cold_groups']+=int(gs in ('cold_start','pooled_not_ready'));sec['zero_denominator_groups']+=int(gs=='zero_denominator');sec['history_rows']+=len(own);sec['query_rows']+=len(qs)
        need(predictions==annual['prediction_rows'] and annual['own_fits']==ownfit and annual['cold_start_groups']==cold and annual['zero_denominator_groups']==zero and annual['clipped_own_fits']==clipped,'Annual group accounting')
        need(annual['parameter_estimates']==int(poolready)+ownfit,'One fit per supported pooled/own coefficient')
        need(type(annual['skipped_initial_rows']) is int and 0<=annual['skipped_initial_rows']<=49,'Initial partial row accounting')
        counts['own_history_groups']+=len(groups);counts['own_history_rows']+=ownrows;counts['cached_forecast_rows']+=predictions
        for key in ('own_fits','cold_start_groups','zero_denominator_groups','clipped_own_fits','parameter_estimates','joined_label_rows','invalid_label_rows','small_month_rows','skipped_initial_rows','calibration_months','calibration_rows'):counts[key]+=annual[key]
        by_year.append(dict(year=y,cutoff=cutoff,status=status,calibration_months=len(retained),calibration_rows=len(history),pooled_alpha=annual['alpha'],own_groups=len(groups),own_fits=ownfit,cold_groups=cold,zero_denominator_groups=zero,parameter_estimates=annual['parameter_estimates']))
    need(len(COUNT_FIELDS)==19 and set(counts)==set(COUNT_FIELDS),'Exactly19 model counters')
    for record in (audit['counts'],prepared['model']['counts'],prepared['model_counts'],completion['model_counts'],closure['model_counts']):need(record==counts and all(type(v) is int and v>=0 for v in record.values()),'Full model-count lineage')
    need(counts['parameter_estimates']==counts['pooled_fits']+counts['own_fits'],'Fitted calibration parameter identity')
    need(counts['own_history_groups']==counts['own_fits']+counts['cold_start_groups']+counts['zero_denominator_groups'],'Group support identity')
    need(counts['annual_cohorts']==counts['pooled_fits']+counts['cold_start_years']+counts['zero_denominator_years'],'Annual support identity')
    need(counts['joined_label_rows']==counts['invalid_label_rows']+counts['small_month_rows']+counts['calibration_rows'],'Joined cohort accounting identity')
    for record in (audit,prepared['model']):
        need(record['new_model_fits']==record['predictive_model_fits']==counts['parameter_estimates'] and record['new_ridge_fits']==record['new_tree_fits']==0,'Calibration is fitted but no source Ridge/tree refits')
    need(prepared['model']['annual_cohorts']==11 and prepared['model']['query_rows']==5978,'Independent full axes')
    return dict(counts=counts,by_year=by_year,by_security=dict(sorted(by_security.items())),own_status_counts=dict(status_counts),cached_prediction_cells=len(seen_queries),unavailable_prediction_cells=5978-len(seen_queries),forecast_value_keys=list(VALUE_KEYS),normal_equations_or_coefficients_recomputed=False,new_forecasts_recomputed=False,model_fits_repeated=False)


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
                residuals.append(dict(signal_date=b['signal_date'],execution_date=b['execution_date'],previous223=a['fee'],current224=b['fee'],difference=difference))
        for rows,metric in ((oh,old['metrics']),(nh,new['metrics'])):
            total=0.
            for row in rows:total+=row['fee']
            need(total==metric['fees_paid'],'Saved chronological fees exactly equal cumulative fee metric')
        report[name]=dict(all_metrics_and_slices_exact=True,total_fees_exact=True,all_fills_exact=True,daily_equity_exact=True,daily_positions_exact=True,holding_nonfee_fields_exact=True,period_fee_differences=residuals,max_absolute_period_fee_difference=max([abs(x['difference']) for x in residuals],default=0.))
    return dict(paths=report,period_fee_tolerance=PERIOD_FEE_ATOL,tolerance_scope='ONLY individual next_open_cost HOLDINGS.fee <=1e-10; all metrics/slices including cumulative fees, turnover, trades, DAILY, positions and holding nonfee fields EXACT; zero-cost fee fields EXACT0.',prospective_policy='Root explicitly confirmed before any actual224 result that the older accumulated-fee exception is NOT inherited. No result-driven widening.')


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
    need(plan['contract']['context']==context and context['arms']==list(ARMS) and context['formation']['parameters']=={'min_months':18,'min_rows_per_month':2,'lookback_calendar_years':5},'Contract/source theory differs')
    need(plan['protected_files']==21444 and len(evidence.load(PROTECTED,registered_digest(plan,PROTECTED)))==21444,'Registered protection manifest')
    before=evidence.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(before['completed'])==221 and sum(r['trials'] for r in before['completed'])==1180 and not before['registered_queue'],'Pre-registration state differs')
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
    for key in ('original_orders_exact','original_arrays_unchanged','single_original_episode','original_core_metadata_and_support_preserved','model_prefix_sources_truncated','model_prefix_metadata_truncated','model_prefix_original_arrays_truncated','model_prefix_future_labels_removed','source_history_starts_at_original_zero','no_new_score_budget','historical_original_universe_exact','error_prediction_source_local_linear'):
        need(prepared.get(key) is True,'Independent prepare invariant '+key)
    need(prepared['prefixes']==123 and prepared['model_prefixes']==list(range(123)) and prepared['original_array_count']==pc['original_array_count']==closure['original_array_count']==27,'All123 physical and order prefixes/original27')
    need(pc['new_returns_computed'] is False and pc['source_model_refits']==0,'Preparation source-fit/return boundary')
    for key in MODEL_FLAGS:need(prepared['model'].get(key) is True,'Independent model flag '+key)
    for key in SLOT_FLAGS:need(prepared['slot'].get(key) is True,'Independent slot flag '+key)
    need(len(MODEL_FLAGS)==23 and len(SLOT_FLAGS)==8 and prepared['model']['computes_returns'] is False and prepared['model']['producer_modules_imported'] is False and prepared['model']['atol']==0.,'Independent model scope')
    ready=prepared['slot']['ready_rows'];changed_cells=prepared['slot']['changed_cells']
    need(set(ready)==set(changed_cells)==set(ARMS[1:]) and len(set(ready.values()))==1 and all(type(v) is int and 0<=v<=122 for v in ready.values()) and all(type(v) is int and 0<=v<=5978 for v in changed_cells.values()),'Paired original-U slot readiness')
    need(pc['model_fit_accounting']==closure['model_fit_accounting']==plan['contract']['model_fit_accounting'],'Fit accounting contract')
    training_meta=evidence.load(ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control/MONTHLY_META.json',registered_digest(plan,ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control/MONTHLY_META.json'))
    coverage=model_coverage(audit,prepared,pc,closure,meta,training_meta)
    prefixes=prepared['model_prefix_evidence'];need([p['prefix'] for p in prefixes]==list(range(123)),'All123 explicit physical-prefix records')
    for p in prefixes:
        n=p['prefix'];need(p['query_rows']==n*49 and type(p['training_rows']) is int and p['training_rows']>=0 and p['source_history_starts_at_original_zero'] is True,'Physical-prefix axes')
        need(set(p['counts'])==set(COUNT_FIELDS) and all(type(v) is int and v>=0 for v in p['counts'].values()),'Physical-prefix19count schema')
        if n==0:need(p['training_rows']==0 and p['query_date'] is None and p['cutoff'] is None and p['latest_label_end'] is None and all(v==0 for v in p['counts'].values()),'Empty prefix')
        else:
            d=meta['dates'][n-1];cutoff=f'{int(d[:4])-1}-12-31'
            need(p['query_date']==d and p['cutoff']==cutoff and (p['latest_label_end'] is None or p['latest_label_end']<=cutoff),'Prefix removes future labels')
    need(prefixes[-1]['counts']==audit['counts'],'Full physical prefix model counter identity')
    with zipfile.ZipFile(OUT/'prepared/FORECASTS.npz') as archive:
        need(set(archive.namelist())=={k+'.npy' for k in VALUE_KEYS} and len(archive.namelist())==3,'Exact three saved FORECASTS members; no arrays decoded')
    source=evidence.load(OUT/'prepared/SOURCE_PARITY.json');binding=plan['model_source_binding']
    need(source['raw_source_hashes']==plan['raw_source_hashes'] and len(plan['raw_source_hashes'])==64 and source['model_source_binding']==binding,'Cached-source lineage')
    for key in ('original_training_cache_exact','original_identity_exact','cached206_source_vectors_exact','cached206_control_constant_exact','source_availability_exact','historical_original_universe_exact','no_source_model_refits','no_fitted_residual_error_inputs'):need(source[key] is True,'Cached source parity '+key)
    for key in ('fresh_training_array_comparison','new_labels','new_raw_features'):need(source[key] is False,'Source reuse scope '+key)
    need(source['target_source']=='original169_target_log_excess_without_recentring' and source['error_prediction_source']=='raw_local_linear','Original target/source endpoint')
    sr=evidence.load(SOURCE_REVIEW,SOURCE_REVIEW_SHA)
    need(path(binding['path'])==SOURCE_REVIEW and binding['sha256']==SOURCE_REVIEW_SHA and sr['passed'] is True and sr['context_sha256']==CONTEXT_SHA and sr['evidence_sha256']==context['evidence_sha256'],'Registered final source review')
    for filename,digest in binding['input_sha256'].items():evidence.pin(filename,digest)
    for key in ('cached_forecasts_are_prior_annual_asof_replays','not_realtime_archival_vintages','numeric_calibration_coverage_not_yet_computed','numeric_own_security_support_not_yet_computed','source_bytes_reverified'):need(sr[key] is True,'Source metadata clock/scope '+key)
    for key in ('uses_fitted_residuals_as_historical_errors','source_npz_decoded','training_npz_decoded','observers_modified','pooled_history_current_identity_filter','local_available_filter'):need(sr[key] is False,'Source computation/support boundary '+key)
    for key in ('actual_new_losses','actual_new_coefficients','actual_new_parameters','actual_new_calibration_coefficients','actual_new_orders','actual_new_returns'):need(type(sr[key]) is int and sr[key]==0,'Source pre-registration computation '+key)
    need(sr['own_history_mask']=='originalU AND206available' and sr['matched_formal_months']==113 and len(sr['source_annual'])==11 and sr['current_and_historical_prediction_key']=='raw_local_linear' and sr['disagreement_prediction_key']=='raw_source_log','Exact source identity and vintage')
    dedup=evidence.load(COVERAGE,COVERAGE_SHA)
    need(dedup['family_novelty'] is False and dedup['actual_new_features']==dedup['actual_new_returns']==dedup['actual_new_orders']==dedup['actual_new_model_fits']==0,'Limited coverage/no unseen novelty claim')
    need(sr['coverage_sha256']==COVERAGE_SHA and registered_digest(plan,COVERAGE)==COVERAGE_SHA,'Coverage pin')
    evidence.pin('work/risk_reliability223_saved_review.py','914fbc728013d73499a1404e639ff504bb5319f17f04b366607bf640fa51f7d1')
    results=evidence.load(OUT/'results/RESULTS.json');independent=evidence.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(results['selection']==frozen['selection'] and results['strict_pit'] is False and results['reused_sample'] is True and results['observer_promotion'] is False and results['annual_own_increment_calibration'] is True and results['cached_source_models_refitted'] is False and results['same_support_both_arms'] is True,'Outcome interpretation or selection differs')
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
    need(len(state['completed'])==222 and sum(x['trials'] for x in state['completed'])==1183 and state['registered_queue']==[],'Central postclosure counters')
    need(state['completed'][-1]['directory']==OUT.name and state['completed'][-1]['completion']['sha256']==expected and path(state['completed'][-1]['completion']['path'])==OUT/'COMPLETION.json','Central closure reference')
    need(state['best_training_selected_candidate']==expected_best and state['latest_training_selected_candidate']['selection']==selection,'Central selected/best mismatch')
    evidence.pin(Path(__file__))
    report=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),reviewer_role='Source and saved-evidence reviewer; not model/host/oracle author. Registered scalar oracle owns independent numerical rebuild.',scope='Saved JSON/hash/identity/count/selection/accounting only. NPZ directory member names inspected without decoding arrays. No forecasts, normal equations, coefficients, ranks, orders or returns recomputed.',completion_sha256=expected,plan_sha256=plan_sha,prepared_completion_sha256=sha(OUT/'prepared/COMPLETION.json'),result_completion_sha256=sha(OUT/'results/COMPLETION.json'),input_sha256=evidence.pins,
        formal_synthetic_tests=test_count,reviewed_numerical_files=6,model_flags_checked=list(MODEL_FLAGS),slot_flags_checked=list(SLOT_FLAGS),model_coverage=coverage,selection=selection,changed_holdings=changes,metrics=metrics,fee_evidence=fee_evidence,incumbent_versus223=incumbent,independent_recorded_metric_residuals=ACCOUNTING_RESIDUALS,
        independent_metric_tolerance=dict(absolute=1e-7,relative=2e-11,scope='Only production-vs-independent accounting nine-statistic checks use this established bound. Across rounds all incumbent metrics including cumulative fees are EXACT; only cost-path individual HOLDINGS.fee <=1e-10. Zero-cost fees exact0.'),
        independent_paths=9,independent_slices=45,independent_training_groups=18,zero_cost_streams=6,order_prefixes=123,source_prefixes=list(range(123)),original_array_count=27,protected_files_verified_by_registered_host_and_closer=21444,protected_archive_reread_in_this_review=False,limitations=context['limitations'],strict_pit=False,reused_sample=True,observers_modified=False,actual_new_features_computed=False,actual_new_parameters_computed=False,actual_new_orders_computed=False,actual_portfolio_returns_computed=False,source_npz_decoded=False,best=expected_best,target_reached=closure['target_reached'],completed_entries=222,trials=1183,central_snapshot_sha256=hashlib.sha256(central_bytes).hexdigest())
    lines=['# 第224轮：局部预测增量的年度校准','', '候选通过前88期筛选。' if selection['admitted'] else '候选未通过前88期筛选，保留原Local67。','',
        '每年使用上年12月31日前五年已成熟的原始预测和标签，局部预测为p、通用预测为b。候选按同一证券过去的d=p-b与y-b关系估计0到1之间的保留比例；对照按相同历史月份估计共享比例。两臂均至少18个月，每个历史月共同至少2个有效标的；池级按月份等权，不按当前名单筛历史。个股不足时原样使用共享比例，共享不足则两臂都保留原预测。原四分之一机器学习微调槽、skip42核心、原股票池、月末EW3及月内固定持仓不变。','',
        '|执行口径|原Local67：年化 / Sharpe / 回撤|个股增量校准|共享增量对照|','|---|---:|---:|---:|']
    def fmt(value):return f"{value['cagr']:.2%} / {value['sharpe']:.3f} / {value['max_drawdown']:.2%}"
    for name,label in (('same_close_zero','同收盘零成本'),('next_open_zero','次开盘零成本'),('next_open_cost','次开盘每边0.05%')):lines.append('|'+label+'|'+'|'.join(fmt(metrics[a][name]['full122']) for a in ARMS)+'|')
    lines+=['','|同收盘零成本区间|原Local67|个股增量校准|共享增量对照|','|---|---:|---:|---:|']
    for label in BOUNDS:lines.append('|'+label+'|'+'|'.join(fmt(metrics[a]['same_close_zero'][label]) for a in ARMS)+'|')
    lines+=['','|持仓集合变化比较|全期|前88|前53|后35训练期|后34诊断期|','|---|---:|---:|---:|---:|---:|']
    for label,change in changes.items():lines.append('|'+label+'|'+'|'.join(str(change['counts'][k]) for k in BOUNDS)+'|')
    c=coverage['counts']
    lines+=['',f"11年度中共享校准{c['pooled_fits']}次；{c['own_history_groups']}个年度证券组中个股校准{c['own_fits']}次，冷启动{c['cold_start_groups']}组，零分母{c['zero_denominator_groups']}组。合计{c['parameter_estimates']}个拟合参数，无新Ridge/tree或源模型重拟合。",'',
        f'已核{test_count}项正式合成测试、6份数值代码、23模型标志、8评分槽标志、19计数、123物理数据前缀及订单前缀、9路径45区间与18训练核算。六条零成本流逐笔/逐月费用严格为零。','',
        'Local67所有收益指标、累计费用、区间、周转额、成交、每日净值与仓位均与223精确一致。仅收费路径逐期HOLDINGS.fee允许既有1e-10误差，全部残差记录在JSON；本轮未保留旧累计费用容差，这在实际结果出现前已明确。','',
        '年度预测是使用回溯版本数据的因果回放，不是历史实时存档。原标签不重新计算或按个股重新中心化。后34期与全期已反复研究，不是独立样本外；早期2016/17/18最多0/4/16个成熟月，必然冷启动。旧honest_local/kernel_residual缺失数值定义仍UNKNOWN，不作完整新颖性或严格PIT声明。本报告不重算模型或收益、不更改观察账户。','']
    need(CENTRAL.read_bytes()==central_bytes,'Central changed during review')
    with (OUT/'RESEARCH_REVIEW.json').open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (OUT/'RESEARCH_REVIEW.md').open('x') as stream:stream.write('\n'.join(lines))
    print(json.dumps(dict(passed=True,review_sha256=sha(OUT/'RESEARCH_REVIEW.json'),markdown_sha256=sha(OUT/'RESEARCH_REVIEW.md'),selection=selection,model_counts=coverage['counts'],same_close={a:metrics[a]['same_close_zero']['full122'] for a in ARMS}),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completion-sha256',required=True);parser.add_argument('--tests',type=int,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--code-review-sha256',required=True)
    args=parser.parse_args();main(args.completion_sha256,args.tests,args.plan_sha256,args.code_review_sha256)

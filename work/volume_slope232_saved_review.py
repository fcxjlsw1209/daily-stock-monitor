"""Read terminal232 saved evidence; no model imports, array decoding or replay.
Prospectively fixed before232 actual outputs: ONLY cost-path period HOLDINGS.fee
may differ <=1e-10 from231. Cumulative fees and all other incumbent fields exact.
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
OUT=ROOT/'outputs/monthly_volume_slope232'
PRIOR=ROOT/'outputs/monthly_market_drift231'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CONTEXT=ROOT/'work/monthly_volume_slope232_context.json'
CONTEXT_SHA='d541a8533a6694dd6a74d19a23dda4d97a5a104d7ac85e9beeb733e6df9df6d8'
SOURCE_REVIEW=ROOT/'work/volume_slope232_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA='868858c5af2988e9863d63e0bd32f1a5952f31a3f42cc6fa87cbeedb33e96734'
TESTS=ROOT/'work/monthly_volume_slope232_TESTS.json'
CODE_REVIEW=ROOT/'work/volume_slope232_CODE_REVIEW.json'
LOCAL=ROOT/'outputs/monthly_local_linear70'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
ARMS=('incumbent_local67','volume_rank_slope','unweighted_slope_control')
PATHS={'same_close_zero':('close',0.),'next_open_zero':('open',0.),'next_open_cost':('open',.0005)}
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FEATURES.npz','SCORES.npz','MODEL_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
FEATURE_PREFIXES=(0,1,53,88,122)
NUMERICAL_FILES=('monthly_volume_slope232_host.py','test_monthly_volume_slope232_host.py','monthly_volume_slope232_kernel.py','test_monthly_volume_slope232_kernel.py','monthly_volume_slope232_independent.py','test_monthly_volume_slope232_independent.py')
PERIOD_FEE_ATOL=1e-10
ACCOUNTING_RESIDUALS=[]
# Formal test count must match the explicit CLI and immutable registered receipt.
# Explicit PLAN and terminal digests are mandatory. Optional code-review digest defaults to the registered immutable pin.


CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'

MODEL_FLAGS=('passed', 'independent_scalar_rebuild', 'official_session_windows_checked', 'fixed_253_prices_and_volumes_checked', 'skip42_checked', 'positive_finite_common_sources_checked', 'eligible_only_consumption_checked', 'same_window_control_checked', 'within_window_average_volume_ranks_checked', 'bounded_positive_weights_checked', 'log_price_relative_path_checked', 'centered_wls_checked', 'identical_ols_control_checked', 'constant_volume_exact_copy_checked', 'constant_price_zero_checked', 'common_availability_checked', 'initial_partial_unchanged', 'audit_checked', 'descriptive_counts_checked', 'no_predictive_fits_checked', 'no_future_inputs', 'no_source_replacements')
SELECTION_FLAGS=('passed', 'independent_original_pre_checked', 'original_support_checked', 'original_budget_checked', 'symmetric_common_eligible_ranks_checked', 'additive_pre_correction_checked', 'original_effects_unchanged', 'one_episode_per_arm', 'initial_partial_unchanged', 'exact_scores_and_choices_checked')
COUNT_FIELDS=('checked_cells', 'eligible_cells', 'attempted_cells', 'available_cells', 'unavailable_cells', 'initial_partial_cells', 'ineligible_cells', 'source_price_observations', 'source_volume_observations', 'volume_rank_observations', 'descriptive_time_slope_estimates', 'unweighted_slope_solves', 'weighted_slope_solves', 'constant_volume_copies', 'constant_price_shortcuts', 'constant_price_cells', 'constant_volume_cells', 'cross_sectional_rank_observations')
HOST_COUNT_FIELDS=('events',)+COUNT_FIELDS
ARRAY_KEYS=('candidate','control','available')
CLARIFICATIONS=ROOT/'work/volume_slope232_CLARIFICATIONS.json'
CLARIFICATIONS_SHA='5e0b50bb216fb73531b15d43d6e77a0ee029e56f892dc71894216810ddb91c1d'
COVERAGE=ROOT/'work/volume_slope232_COVERAGE.json'
COVERAGE_SHA='58fa25bde4c040140c10fee0f1931d5e5d5c8760b95e2e15a2da87314adca46a'
FORMAL_TESTS=87

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
        open_vs_incumbent=guard(c['next_open_zero'],b['next_open_zero'],0.),
        close_vs_unweighted_control=guard(c['same_close_zero'],m['same_close_zero'],0.),
        open_vs_unweighted_control=guard(c['next_open_zero'],m['next_open_zero'],0.),
        changed_support=support(ids),changed_support_vs_control=support(ids2))
    admitted=all(checks.values())
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=checks,
        changed_months=ids,changed_months_vs_control=ids2,first88_only=True,later34_used=False)



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
                residuals.append(dict(signal_date=b['signal_date'],execution_date=b['execution_date'],previous231=a['fee'],current232=b['fee'],difference=difference))
        for rows,metric in ((oh,old['metrics']),(nh,new['metrics'])):
            total=0.
            for row in rows:total+=row['fee']
            need(total==metric['fees_paid'],'Saved chronological fees exactly equal cumulative fee metric')
        report[name]=dict(all_metrics_and_slices_exact=True,total_fees_exact=True,all_fills_exact=True,daily_equity_exact=True,daily_positions_exact=True,holding_nonfee_fields_exact=True,period_fee_differences=residuals,max_absolute_period_fee_difference=max([abs(x['difference']) for x in residuals],default=0.))
    return dict(paths=report,period_fee_tolerance=PERIOD_FEE_ATOL,tolerance_scope='ONLY individual next_open_cost HOLDINGS.fee <=1e-10; all metrics/slices including cumulative fees, turnover, trades, DAILY, positions and holding nonfee fields EXACT; zero-cost fee fields EXACT0.',prospective_policy='Inherited frozen231 comparison scope: no accumulated-fee exception and no result-driven widening.')


def model_check(check,raw_counts,n):
    need(len(MODEL_FLAGS)==22 and all(check.get(k) is True for k in MODEL_FLAGS),'All22 independent feature flags')
    need(check['counts']==raw_counts and check['query_rows']==n and check['numerical_tolerance']==0.,'Exact independent feature counts/tolerance')
    need(check['producer_modules_imported'] is False and check['computes_returns'] is False,'Independent feature scope')
    for k in ('new_model_fits','predictive_model_fits'):need(type(check[k]) is int and check[k]==0,'No predictive fit '+k)


def selection_check(check,n):
    need(len(SELECTION_FLAGS)==10 and all(check.get(k) is True for k in SELECTION_FLAGS),'All10 original-episode selection flags')
    need(check['ordered_choices_checked']==3*n and check['numerical_tolerance']==0. and check['original_source_parity_tolerance']==1e-12,'Exact independent selection proof')
    need(check['producer_modules_imported'] is False and check['computes_returns'] is False,'Independent selection scope')
    need(finite(check['max_absolute_delta']) and 0.<=check['max_absolute_delta']<=.005,'Fixed .01 rank budget')


def host_counts(raw,n):
    answer=dict(raw,events=n);answer['cross_sectional_rank_observations']=2*raw['available_cells']
    need(set(answer)==set(HOST_COUNT_FIELDS),'Exactly19 host counters')
    return answer


def model_coverage(audit,prepared,completion,closure,meta,calendar,orders):
    """Recount saved flags/lengths and clocks. Do not rebuild slope or ranks."""
    need(len(COUNT_FIELDS)==18 and len(HOST_COUNT_FIELDS)==19,'Count schema')
    expected=dict(schema='volume_slope232_features_v1',events=122,symbols_per_event=49,
        window_price_count=253,window_volume_count=253,span_sessions=252,skip_sessions=42,
        source_start_offset=-294,source_end_offset=-42,feature_domain='original_eligible')
    need(all(audit[k]==v for k,v in expected.items()),'Fixed volume-slope formation')
    for key in ('common_availability','within_volume_ranking_performed'):need(audit[key] is True,'Feature assertion '+key)
    for key in ('cross_sectional_ranking_performed','trading_eligibility_modified','portfolio_returns_calculated'):need(audit[key] is False,'Feature scope '+key)
    for key in ('new_model_fits','predictive_model_fits'):need(type(audit[key]) is int and audit[key]==0,'Zero predictive fits')
    axis=calendar['sessions'];need(axis==sorted(set(axis)),'Unique official calendar')
    positions={d:i for i,d in enumerate(axis)};monthends={d[:7]:d for d in axis}
    events=audit['event_history'];need(len(events)==122,'122 query records')
    reasons=Counter();per_event=[];support=[];by_year={};constant={arm:[] for arm in ARMS[1:]}
    for i,event in enumerate(events):
        d=meta['dates'][i];names=meta['symbols'][i];t=positions[d]
        need(event['date']==d and event['symbols']==names and event['session_position']==t,'Saved event identity')
        need(d=='2016-08-17' if i==0 else monthends[d[:7]]==d,'Formal date / initialpartial')
        window=axis[t-294:t-41] if t>=294 else []
        need(event['source_dates']==window and event['source_start']==(window[0] if window else None) and event['source_end']==(window[-1] if window else None),'Fixed253 official source dates')
        need(not window or len(window)==253,'Full formation length')
        need(set(event['missing_price_sessions'])<=set(window) and set(event['missing_volume_sessions'])<=set(window),'Missing sessions belong to formation')
        cells=event['cells'];need(len(cells)==49,'All candidate cells retained')
        counts=dict.fromkeys(COUNT_FIELDS,0);counts['checked_cells']=49
        for j,cell in enumerate(cells):
            need(cell['symbol']==names[j] and cell['column']==j and cell['date']==d,'Exact cell identity')
            E,A=cell['eligible'],cell['available'];need(type(E) is bool and type(A) is bool,'Boolean source domain')
            reason=cell['reason'];reasons[reason]+=1;counts['eligible_cells']+=int(E)
            if i==0:
                need(reason=='initial_partial' and not A,'Initial feature skipped');counts['initial_partial_cells']+=1
            elif not E:
                need(reason=='ineligible' and not A,'Original ineligible cells untouched');counts['ineligible_cells']+=1
            else:counts['attempted_cells']+=1
            if A:
                need(E and i>0 and reason=='available' and len(window)==253,'Accepted eligible full formation')
                need(not event['missing_price_sessions'] and not event['missing_volume_sessions'],'Complete shared calendar before acceptance')
                need(all(finite(cell[k]) for k in ('candidate','control')),'Saved finite accepted scalar outputs')
                need(all(isinstance(cell[k],list) and len(cell[k])==253 for k in ('prices','volumes','volume_ranks','weights','log_prices_relative')),'Saved accepted observation lengths only')
                cp,cv=cell['constant_price'],cell['constant_volume'];need(type(cp) is bool and type(cv) is bool,'Constant branch flags')
                counts['available_cells']+=1;counts['constant_price_cells']+=int(cp);counts['constant_volume_cells']+=int(cv)
                counts['constant_price_shortcuts']+=int(cp);counts['unweighted_slope_solves']+=int(not cp)
                counts['weighted_slope_solves']+=int(not cp and not cv);counts['constant_volume_copies']+=int(not cp and cv)
                if cp:
                    need(cell['candidate']==cell['control']==0. and cell['control_branch']==cell['candidate_branch']=='constant_price_zero','Exact constant-price shortcut')
                    need(cell['control_stats'] is None and cell['candidate_stats'] is None,'No synthetic solve on constant price')
                else:
                    need(cell['control_branch']=='unweighted_slope' and cell['candidate_branch']==('constant_volume_copy' if cv else 'weighted_slope'),'Recorded descriptive branches')
                    need(cell['control_stats']['slope']==cell['control'] and cell['candidate_stats']['slope']==cell['candidate'],'Recorded slope references')
                    if cv:need(cell['candidate']==cell['control'] and cell['candidate_stats']==cell['control_stats'],'Exact copy for constant volume')
            else:
                need(cell['candidate'] is None and cell['control'] is None,'No numeric imputation for unavailable cells')
                need(all(cell[k]==[] for k in ('prices','volumes','volume_ranks','weights','log_prices_relative')),'Unavailable source arrays remain empty')
        A=counts['available_cells'];counts['unavailable_cells']=49-A
        for k in ('source_price_observations','source_volume_observations','volume_rank_observations'):counts[k]=253*A
        counts['descriptive_time_slope_estimates']=2*A
        # Feature producer leaves cross-sectional ranking to host; zero here is intentional.
        need(counts['cross_sectional_rank_observations']==0,'Kernel never cross-section ranks')
        per_event.append(counts);support.append(dict(index=i,date=d,eligible=counts['eligible_cells'],available=A,missing_eligible=counts['attempted_cells']-A))
        year=by_year.setdefault(d[:4],dict(year=int(d[:4]),months=0,available_cells=0,eligible_cells=0))
        year['months']+=1;year['available_cells']+=A;year['eligible_cells']+=counts['eligible_cells']
        eligible_symbols={c['symbol'] for c in cells if c['eligible']}
        for arm,key in zip(ARMS[1:],('candidate','control')):
            need(set(orders[arm][i]['symbols'])<=eligible_symbols,'Orders retain original eligibility')
            scalars=[c[key] for c in cells if c['available']]
            if len(scalars)<2 or len(set(scalars))==1:
                constant[arm].append(i);need(orders[arm][i]==orders[ARMS[0]][i],'All-zero auxiliary must retain original episode result')
    raw={k:sum(x[k] for x in per_event) for k in COUNT_FIELDS}
    need(all(type(audit[k]) is int and audit[k]>=0 and audit[k]==raw[k] for k in COUNT_FIELDS),'All18 raw descriptive counters')
    need(audit['reason_counts']==dict(reasons),'Saved source reason counts')
    model_check(prepared['model'],raw,122);selection_check(prepared['selection'],122)
    hc=host_counts(raw,122)
    for record in (prepared,completion,closure):need(record['original_array_count']==27 and record['model_counts']==hc,'Original27 and19 host counts')
    need(prepared['prefixes']==123 and prepared['model_prefixes']==list(FEATURE_PREFIXES),'123decision/5source prefixes')
    prefixes=prepared['physical_prefix_evidence'];need([x['prefix'] for x in prefixes]==list(FEATURE_PREFIXES),'Exact physical prefixes')
    for item in prefixes:
        n=item['prefix'];c={k:sum(x[k] for x in per_event[:n]) for k in COUNT_FIELDS}
        model_check(item['feature'],c,n);selection_check(item['selection'],n)
        need(item['counts']==host_counts(c,n) and set(item['source_rows'])=={'adj','volume'},'Physical prefix counts and fields')
        need(all(type(v) is int and (v>0 if n else v==0) for v in item['source_rows'].values()),'Truncated source row metadata')
    for key in ('original_orders_exact','original_arrays_unchanged','original_core_preserved','single_original_episode','selection_uses_zero_cost_paths_only'):need(prepared[key] is True,'Preparation invariant '+key)
    need(prepared['additional_max_score_budget']==.005 and completion['new_returns_computed'] is False,'Additive pre-score budget/no preparation replay')
    for key in ('new_model_fits','predictive_model_fits'):need(type(completion[key]) is int and completion[key]==0,'No fitted preparation model')
    return dict(counts=hc,raw_feature_counts=raw,source_reason_counts=dict(reasons),source_status_by_query=support,
        by_year=list(by_year.values()),supported_months=sum(x['available']>0 for x in support),
        support_by_period={b:dict(months=sum(x['available']>0 for x in support[lo:hi]),cells=sum(x['available'] for x in support[lo:hi])) for b,(lo,hi) in BOUNDS.items()},
        all_zero_auxiliary_query_indices=constant,independent_descriptive_slope_outputs=raw['descriptive_time_slope_estimates'],
        physical_prefix_descriptive_slope_outputs=sum(x['counts']['descriptive_time_slope_estimates'] for x in prefixes),
        raw_values_or_ranks_recomputed=False,npz_arrays_decoded=False,volume_units_certified=False,
        limitation='Counts/identity/branch/clock proofs reread from saved audit. No slope or volume rank rebuilt; numerical correctness belongs to independently verified registered preparation.')


def price_source_evidence(evidence,plan,context,sr,parity,meta,calendar,model_audit,prepared):
    need(registered_digest(plan,SOURCE_REVIEW)==SOURCE_REVIEW_SHA and sr['context_sha256']==CONTEXT_SHA and sr['coverage_sha256']==COVERAGE_SHA,'Source prospective binding')
    for filename,digest in ((CLARIFICATIONS,CLARIFICATIONS_SHA),(COVERAGE,COVERAGE_SHA)):
        evidence.load(filename,digest);need(registered_digest(plan,filename)==digest,'Definition evidence binding')
    for key in ('passed','source_only','metadata_only','conditional_proxy_admission','source_bytes_reverified','raw_source_map_matches_original','metadata_axes_verified','calendar_clock_verified','volume_source_semantics_reviewed','single_episode_budget_source_reviewed'):need(sr[key] is True,'Source metadata proof '+key)
    for key in ('volume_units_certified','strict_pit','source_npz_decoded','npz_decoded','price_rows_read','volume_rows_read','source_models_imported','source_loader_executed','actual_support_computed','actual_features_computed','actual_returns_computed','actual_orders_computed','observers_modified','central_modified','frozen_files_modified'):need(sr[key] is False,'Source scope '+key)
    need(sr['actual_model_fits']==0 and sr['raw_source_hashes']==plan['raw_source_hashes'] and len(sr['raw_source_hashes'])==64,'64 raw source hashes')
    need(sr['evidence_sha256']==context['evidence_sha256'],'Final context evidence')
    binding=plan['price_source_binding']
    need(path(binding['path'])==SOURCE_REVIEW and binding['sha256']==SOURCE_REVIEW_SHA and binding['conditional_proxy_admission'] is True and binding['volume_units_certified'] is False and binding['strict_pit'] is False,'Conditional proxy source binding')
    for field in ('input_sha256','source_sha256','evidence_sha256','direct_feature_source_sha256','reviewed_code_sha256'):
        for filename,digest in sr[field].items():evidence.pin(filename,digest)
    for filename,digest in binding['input_sha256'].items():evidence.pin(filename,digest)
    need(parity['raw_source_hashes']==plan['raw_source_hashes'] and parity['price_source_binding']==binding,'Prepared original source provenance')
    for key in ('original_arrays_unchanged','original_orders_exact'):need(parity[key] is True,'Original frozen source assertion')
    need(parity['volume_units_certified'] is False and parity['strict_pit'] is False and parity['calendar_sha256']==registered_digest(plan,CALENDAR),'Volume limitation/calendar unchanged')
    fps=parity['original_array_fingerprints'];need(len(fps)==27,'27 cached array byte fingerprints')
    for key,value in fps.items():
        need(set(value)=={'dtype','shape','sha256'} and value['shape'][0]==122 and type(value['sha256']) is str and len(value['sha256'])==64,'Original byte fingerprint metadata '+key)
    for key in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact'):need(parity['execution_prices'][key] is True,'Original execution parity '+key)
    return dict(raw_source_count=64,original_arrays=27,metadata_review_sha256=SOURCE_REVIEW_SHA,
        source_clock='253 official sessions t-294 through t-42 inclusive for adjusted price and stored Volume; current eligibility unchanged.',
        conditional_proxy_admission=True,volume_units_certified=False,strict_pit=False,volume_semantics=sr['volume_semantics'],
        metadata_review_computed_support=False,registered_preparation_support_verified=True,source_arrays_decoded=False)



def main(expected,test_count,expected_plan_sha,expected_code_review_sha=None):
    need(type(test_count) is int and test_count==FORMAL_TESTS,'Exactly87 formal synthetic tests required')
    for value in (expected,expected_plan_sha)+((expected_code_review_sha,) if expected_code_review_sha is not None else ()):
        need(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Explicit terminal/PLAN/code-review SHA required')
    ACCOUNTING_RESIDUALS.clear()
    need(not (OUT/'RESEARCH_REVIEW.json').exists(),'Saved review is write-once')
    evidence=Evidence();closure=evidence.load(OUT/'COMPLETION.json',expected)
    need(closure['plan_sha256']==expected_plan_sha,'Explicit registered PLAN differs')
    plan=evidence.load(OUT/'PLAN.json',expected_plan_sha);plan_sha=expected_plan_sha
    need(closure.get('new_trials')==3 and closure.get('original_array_count')==27,'Exact three new trials and27 original arrays')
    need(closure.get('status')=='COMPLETED_INDEPENDENTLY_VERIFIED' and closure['plan_sha256']==plan_sha and closure['protected_files']==21444 and closure['observers_modified'] is False,'Terminal closure required')
    need(isinstance(plan.get('input_sha256'),dict) and plan['input_sha256'],'Registered input closure missing')
    for filename,digest in plan['input_sha256'].items():evidence.pin(filename,digest)
    context=evidence.load(CONTEXT,CONTEXT_SHA)
    need(registered_digest(plan,CONTEXT)==CONTEXT_SHA==plan['contract']['prospective_context_sha256'],'Context not registered exactly')
    need(plan['contract']['context']==context and context['arms']==list(ARMS) and plan['contract']['dose']==.01 and plan['contract']['max_absolute_delta']==.005 and plan['contract']['candidate']==ARMS[1] and plan['contract']['control_nonpromotable']==ARMS[2],'Contract/source theory differs')
    need(plan['contract']['selection_paths']==['same_close_zero','next_open_zero'] and plan['contract']['next_open_cost_diagnostic_only'] is True,'No fee-based selection guard in232')
    need(plan['contract']['original_pool_preserved'] is True and plan['contract']['stored_volume_proxy'] is True and plan['contract']['volume_units_certified'] is False and plan['contract']['strict_pit'] is False,'Fixed232 architecture')
    for key in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration','actual_new_parameters_before_registration','actual_new_orders_before_registration','actual_new_returns_before_registration'):need(plan[key]==0,'No preregistration actual calculation')
    need(plan['protected_files']==21444 and len(evidence.load(PROTECTED,registered_digest(plan,PROTECTED)))==21444,'Registered protection manifest')
    before=evidence.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(before['completed'])==227 and sum(r['trials'] for r in before['completed'])==1198 and not before['registered_queue'],'Pre-registration state differs')
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
    registered_review=registered_digest(plan,CODE_REVIEW)
    if expected_code_review_sha is None:expected_code_review_sha=registered_review
    need(registered_review==expected_code_review_sha,'Explicit final code review differs')
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
    calendar=evidence.load(CALENDAR,registered_digest(plan,CALENDAR))
    source=evidence.load(OUT/'prepared/SOURCE_PARITY.json')
    coverage=model_coverage(audit,prepared,pc,closure,meta,calendar,orders)
    need(pc['model_fit_accounting']==closure['model_fit_accounting']==plan['contract']['model_fit_accounting'],'Descriptive accounting description')
    for name,keys in (('FEATURES.npz',ARRAY_KEYS),('SCORES.npz',ARMS)):
        with zipfile.ZipFile(OUT/'prepared'/name) as archive:
            need(set(archive.namelist())=={k+'.npy' for k in keys} and len(archive.namelist())==len(keys),'NPZ member names without decoding: '+name)
    sr=evidence.load(SOURCE_REVIEW,SOURCE_REVIEW_SHA)
    source_evidence=price_source_evidence(evidence,plan,context,sr,source,meta,calendar,audit,prepared)
    results=evidence.load(OUT/'results/RESULTS.json');independent=evidence.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(results['selection']==frozen['selection'] and results['strict_pit'] is False and results['reused_sample'] is True and results['observer_promotion'] is False,'Outcome interpretation or selection differs')
    for key in ('volume_rank_time_slope','stored_volume_proxy','original_episode_preserved','zero_cost_selection','next_open_cost_diagnostic_only','same_support_both_arms'):need(results[key] is True,'Outcome mechanism differs '+key)
    need(results['source_volume_units_certified'] is False,'No volume unit certification added')
    need(independent['passed'] is True and independent['independent_accounts'] is True and independent['independent_of_production_model_and_engine'] is True and independent['checked_paths']==9 and independent['checked_slices']==45 and independent['terminal_liquidation'] is False,'Independent execution scope')
    need(independent['accounting_source_sha256']==registered_digest(plan,ROOT/'work/ptr169_results_independent.py'),'Independent accounting source')
    need(set(results['arms'])==set(independent['paths'])==set(training)==set(ARMS),'Outcome/training arms')
    metrics={};fee_evidence={};training_checks=0;zero_streams=0
    for arm in ARMS:
        need(set(results['arms'][arm])==set(independent['paths'][arm])==set(PATHS),'Three execution paths required')
        need(set(training[arm])=={'same_close_zero','next_open_zero'},'Training must use only the two zero-cost execution paths')
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
    need(results['arms'][ARMS[0]]['same_close_zero']['metrics']['cagr']==0.6766617372041686,'Exact frozen Local67 full CAGR differs')
    comparisons={arm:{name:{period:dict(
        cagr_difference=metrics[arm][name][period]['cagr']-metrics[ARMS[0]][name][period]['cagr'],
        sharpe_difference=metrics[arm][name][period]['sharpe']-metrics[ARMS[0]][name][period]['sharpe'],
        max_drawdown_difference=metrics[arm][name][period]['max_drawdown']-metrics[ARMS[0]][name][period]['max_drawdown'],
        cagr_relation=('equal' if metrics[arm][name][period]['cagr']==metrics[ARMS[0]][name][period]['cagr'] else
            'higher' if metrics[arm][name][period]['cagr']>metrics[ARMS[0]][name][period]['cagr'] else 'lower'))
        for period in ('full122','first88','later34')} for name in PATHS} for arm in ARMS[1:]}
    changes={a+'_vs_'+b:changed(orders,b,a) for a,b in ((ARMS[1],ARMS[0]),(ARMS[2],ARMS[0]),(ARMS[1],ARMS[2]))}
    selection=selection_from_saved(training,changes);need(selection==frozen['selection'],'Frozen first88 guards differ')
    expected_best=before['best_training_selected_candidate'];pick=selection['primary'];picked=results['arms'][pick]
    if selection['admitted'] and picked['same_close_zero']['metrics']['cagr']>expected_best['cagr']:
        m=picked['same_close_zero']['metrics'];expected_best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=picked['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=plan['contract']['limitation'])
    need(closure['best_training_selected_candidate']==expected_best and closure['target_reached']==(expected_best['cagr']>=.7),'Best/target update differs from frozen admission')
    central_bytes=CENTRAL.read_bytes();state=json.loads(central_bytes)
    need(len(state['completed'])==228 and sum(x['trials'] for x in state['completed'])==1201 and state['registered_queue']==[],'Central postclosure counters')
    need(state['completed'][-1]['directory']==OUT.name and state['completed'][-1]['completion']['sha256']==expected and path(state['completed'][-1]['completion']['path'])==OUT/'COMPLETION.json','Central closure reference')
    need(state['best_training_selected_candidate']==expected_best and state['latest_training_selected_candidate']['selection']==selection,'Central selected/best mismatch')
    evidence.pin(Path(__file__))
    report=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),reviewer_role='Source232 metadata reviewer; this saved-evidence reader imports no model, oracle or execution engine.',scope='Saved JSON/hash/identity/count/selection/accounting only. NPZ directory member names inspected without decoding arrays. Only saved source-window/branch/count identities and selection/accounting receipts are rechecked; no raw slope, volume ranks, orders or returns recomputed.',completion_sha256=expected,plan_sha256=plan_sha,prepared_completion_sha256=sha(OUT/'prepared/COMPLETION.json'),result_completion_sha256=sha(OUT/'results/COMPLETION.json'),input_sha256=evidence.pins,
        formal_synthetic_tests=test_count,reviewed_numerical_files=6,model_flags_checked=list(MODEL_FLAGS),selection_flags_checked=list(SELECTION_FLAGS),model_coverage=coverage,selection=selection,changed_holdings=changes,metrics=metrics,candidate_control_vs_incumbent=comparisons,fee_evidence=fee_evidence,incumbent_versus231=incumbent,price_source_evidence=source_evidence,independent_recorded_metric_residuals=ACCOUNTING_RESIDUALS,
        independent_metric_tolerance=dict(absolute=1e-7,relative=2e-11,scope='Only production-vs-independent accounting nine-statistic checks use this established bound. Across rounds all incumbent metrics including cumulative fees are EXACT; only cost-path individual HOLDINGS.fee <=1e-10. Zero-cost fees exact0.'),
        independent_paths=9,independent_slices=45,independent_training_groups=18,zero_cost_streams=6,order_prefixes=123,source_prefixes=list(FEATURE_PREFIXES),original_array_count=27,protected_files_verified_by_registered_host_and_closer=21444,protected_archive_reread_in_this_review=False,limitations=context['limitations'],strict_pit=False,conditional_proxy_admission=True,volume_units_certified=False,reused_sample=True,observers_modified=False,actual_new_features_computed=False,actual_new_parameters_computed=False,actual_new_orders_computed=False,actual_portfolio_returns_computed=False,source_npz_decoded=False,best=expected_best,target_reached=closure['target_reached'],completed_entries=228,trials=1201,central_snapshot_sha256=hashlib.sha256(central_bytes).hexdigest())
    need(CENTRAL.read_bytes()==central_bytes and sha(OUT/'COMPLETION.json')==expected and sha(OUT/'PLAN.json')==expected_plan_sha,'Central or immutable closure changed during review')
    with (OUT/'RESEARCH_REVIEW.json').open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,review_sha256=sha(OUT/'RESEARCH_REVIEW.json'),selection=selection,model_counts=coverage['counts'],same_close={a:metrics[a]['same_close_zero']['full122'] for a in ARMS}),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completion-sha256',required=True);parser.add_argument('--tests',type=int,default=FORMAL_TESTS)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--code-review-sha256')
    args=parser.parse_args();main(args.completion_sha256,args.tests,args.plan_sha256,args.code_review_sha256)

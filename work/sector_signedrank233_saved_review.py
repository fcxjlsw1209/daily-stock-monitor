"""Read terminal233 saved evidence; no producer, oracle, engine or array decoding.
Only saved hashes, metadata, scalar audit records and accounting certificates are
reviewed. No raw feature, z, rank, order stream or portfolio return is rebuilt.
The inherited incumbent-comparison policy permits ONLY cost-path period
HOLDINGS.fee differences <=1e-10; all other incumbent fields remain exact.
Run only after root authorizes terminal COMPLETION and PLAN digests.
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
OUT=ROOT/'outputs/monthly_sector_signedrank233'
PRIOR=ROOT/'outputs/monthly_volume_slope232'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CONTEXT=ROOT/'work/monthly_sector_signedrank233_context.json'
CONTEXT_SHA='565c11f6a0343f9c321c57816a0becb68ee236f403a9a802098d26fa1d0d87ab'
SOURCE_REVIEW=ROOT/'work/sector_signedrank233_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA='85257cd82248ba86dee631cedac9726ecd19a632f53de0df39a3c068e6ea6c03'
TESTS=ROOT/'work/monthly_sector_signedrank233_TESTS.json'
CODE_REVIEW=ROOT/'work/sector_signedrank233_CODE_REVIEW.json'
LOCAL=ROOT/'outputs/monthly_local_linear70'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
ARMS=('incumbent_local67','sector_signed_magnitude','sector_sign_control')
PATHS={'same_close_zero':('close',0.),'next_open_zero':('open',0.),'next_open_cost':('open',.0005)}
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FEATURES.npz','TRANSFORMS_DELTAS.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
FEATURE_PREFIXES=(0,1,53,88,122)
NUMERICAL_FILES=('monthly_sector_signedrank233_host.py','test_monthly_sector_signedrank233_host.py','monthly_sector_signedrank233_kernel.py','test_monthly_sector_signedrank233_kernel.py','monthly_sector_signedrank233_independent.py','test_monthly_sector_signedrank233_independent.py')
PERIOD_FEE_ATOL=1e-10
ACCOUNTING_RESIDUALS=[]
# Formal test count must match the explicit CLI and immutable registered receipt.
# Explicit PLAN and terminal digests are mandatory. Optional code-review digest defaults to the registered immutable pin.


CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'

COUNT_FIELDS=('events', 'symbols_per_event', 'checked_cells', 'available_cells', 'unavailable_cells', 'source_price_observations', 'relative_sign_observations', 'magnitude_rank_observations', 'magnitude_pair_comparisons', 'signed_rank_estimates', 'sign_balance_estimates', 'constant_magnitude_copies', 'predictive_model_fits')
FEATURE_FLAGS=('passed', 'independent_scalar_rebuild', 'own_sector_mapping_checked', 'exact_rational_relative_ratios_checked', 'exact_rational_relative_signs_checked', 'twelve_nonoverlapping_blocks_checked', 'four_neutral_pseudoblocks_checked', 'full_253_price_support_checked', 'nonboolean_positive_real_sources_checked', 'official_session_windows_checked', 'monthend_queries_checked', 'exact_common_availability', 'zero_blocks_included_checked', 'doubled_average_magnitude_ranks_checked', 'doubled_rank_sum_checked', 'fixed_denominators_checked', 'constant_magnitude_exact_copy_checked', 'control_signed_balance_checked', 'no_cross_sectional_ranking', 'audit_checked', 'descriptive_counts_checked', 'no_predictive_fits_checked', 'no_future_or_outcome_inputs')
DELTA_FLAGS=('passed', 'full_candidate_universe', 'common_available_only', 'common_availability', 'initial_delta_zero', 'no_eligibility_or_model_filter', 'no_cross_sectional_ranking', 'fixed_linear_mapping_checked', 'direct_additive_pre_score')
SELECTION_FLAGS=('passed', 'independent_original_pre_checked', 'original_support_checked', 'original_budget_checked', 'fixed_raw_delta_checked', 'additive_pre_correction_checked', 'original_effects_unchanged', 'one_episode_per_arm', 'initial_partial_unchanged', 'exact_scores_and_choices_checked')
MODEL_FLAGS=FEATURE_FLAGS
HOST_COUNT_FIELDS=COUNT_FIELDS
ARRAY_KEYS=('candidate','control','available')
CONTROL192=ROOT/'outputs/monthly_sector_recurrence192'
MAPPING_SHA='ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
COVERAGE=ROOT/'work/sector_signedrank233_COVERAGE.json'
COVERAGE_SHA='73662f9fa137176ad9a3e1a0b3869a6b802559e560e70c8cfb40b6715ae2e3cf'
FORMAL_TESTS=94


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
        close_vs_sign_control=guard(c['same_close_zero'],m['same_close_zero'],0.),
        open_vs_sign_control=guard(c['next_open_zero'],m['next_open_zero'],0.),
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
                residuals.append(dict(signal_date=b['signal_date'],execution_date=b['execution_date'],previous232=a['fee'],current233=b['fee'],difference=difference))
        for rows,metric in ((oh,old['metrics']),(nh,new['metrics'])):
            total=0.
            for row in rows:total+=row['fee']
            need(total==metric['fees_paid'],'Saved chronological fees exactly equal cumulative fee metric')
        report[name]=dict(all_metrics_and_slices_exact=True,total_fees_exact=True,all_fills_exact=True,daily_equity_exact=True,daily_positions_exact=True,holding_nonfee_fields_exact=True,period_fee_differences=residuals,max_absolute_period_fee_difference=max([abs(x['difference']) for x in residuals],default=0.))
    return dict(paths=report,period_fee_tolerance=PERIOD_FEE_ATOL,tolerance_scope='ONLY individual next_open_cost HOLDINGS.fee <=1e-10; all metrics/slices including cumulative fees, turnover, trades, DAILY, positions and holding nonfee fields EXACT; zero-cost fee fields EXACT0.',prospective_policy='Inherited frozen232 comparison scope: no accumulated-fee exception and no result-driven widening.')


def model_check(check,counts,n):
    need(len(FEATURE_FLAGS)==23 and all(check.get(k) is True for k in FEATURE_FLAGS),'All23 independent feature flags')
    need(check.get('counts')==counts and all(type(check.get(k)) is int and check[k]==v for k,v in counts.items()),'Exact13 independent feature counters')
    need(counts['events']==n and check.get('numerical_tolerance')==0.,'Feature coverage/tolerance')
    need(check.get('producer_modules_imported') is False and check.get('computes_returns') is False and check.get('portfolio_returns_calculated') is False,'Independent feature scope')
    need(check.get('mapping_source_sha256')==MAPPING_SHA and counts['predictive_model_fits']==0,'Frozen mapping/no predictive fitting')


def selection_check(check,n):
    need(len(SELECTION_FLAGS)==10 and all(check.get(k) is True for k in SELECTION_FLAGS),'All10 original-episode selection flags')
    need(check['ordered_choices_checked']==3*n and check['numerical_tolerance']==0. and check['original_source_parity_tolerance']==1e-12,'Exact independent selection proof')
    need(check['producer_modules_imported'] is False and check['computes_returns'] is False and check['full_scores_supplied'] is True,'Independent full-score selection scope')
    need(check['maximum_absolute_delta']==.01875,'Fixed .025 raw budget')


def delta_check(check,events):
    need(len(DELTA_FLAGS)==9 and all(check.get(k) is True for k in DELTA_FLAGS),'All9 independent delta flags')
    need(check.get('checked_events')==len(events) and check.get('dose')==.025 and check.get('max_absolute_delta')==.01875 and check.get('numerical_tolerance')==0.,'Fixed direct delta coverage')
    need(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Delta checker scope')
    changed={arm:sum(cell['available'] and cell[key]!=0. for i,event in enumerate(events) if i>0 for cell in event['cells']) for arm,key in zip(ARMS[1:],('candidate','control'))}
    need(check.get('changed_cells')==changed,'Saved nonzero raw contribution counts differ')


def aggregate_counts(per_event,n):
    result={k:sum(row[k] for row in per_event[:n]) for k in COUNT_FIELDS if k not in ('events','symbols_per_event')}
    result.update(events=n,symbols_per_event=49)
    return result


def model_coverage(audit,prepared,completion,closure,meta,calendar,orders):
    """Recount saved identities and support only, without rebuilding z or ranks."""
    need(len(COUNT_FIELDS)==13,'Count schema')
    expected=dict(schema='monthly_sector_signedrank233_features_v1',events=122,symbols_per_event=49,
        price_count=253,skip_sessions=42,return_count=252,block_count=12,block_sessions=21,
        neutral_pseudo_blocks=4,candidate_denominator=208,control_denominator=16,expected_twice_rank_sum=156)
    need(all(audit.get(k)==v for k,v in expected.items()),'Fixed signed-rank formation')
    for key in ('common_availability','within_cell_ranking_performed'):need(audit.get(key) is True,'Feature assertion '+key)
    for key in ('cross_sectional_ranking_performed','trading_eligibility_modified','portfolio_returns_calculated'):need(audit.get(key) is False,'Feature scope '+key)
    axis=calendar['sessions'];need(axis==sorted(set(axis)),'Unique official calendar')
    positions={d:i for i,d in enumerate(axis)};monthends={d[:7]:d for d in axis}
    events=audit['event_history'];need(len(events)==122,'122 query records')
    reasons=Counter();per_event=[];support=[];by_year={};neutral={arm:[] for arm in ARMS[1:]}
    for i,event in enumerate(events):
        d=meta['dates'][i];names=meta['symbols'][i];t=positions[d]
        need(event['date']==d and event['symbols']==names and event['session_position']==t,'Saved event identity')
        need(d=='2016-08-17' if i==0 else monthends[d[:7]]==d,'Formal date / initialpartial')
        window=axis[t-294:t-41] if t>=294 else []
        need(event['price_start']==(window[0] if window else None) and event['price_end']==(window[-1] if window else None),'Fixed253 official source endpoints')
        need(not window or len(window)==253,'Full formation length')
        windows=(dict(formation=dict(price_start=window[0],price_end=window[-1],return_count=252),blocks=dict(first_start_date=window[0],last_end_date=window[-1],block_sessions=21,block_count=12)) if window else None)
        need(event['windows']==windows and set(event['missing_official_sessions'])<=set(window),'Saved formation/missing sessions')
        need(set(event['benchmarks'])==set(names),'All event mapping identities')
        cells=event['cells'];need(len(cells)==49,'All original candidate cells retained')
        counts=dict.fromkeys(COUNT_FIELDS,0);counts.update(events=1,symbols_per_event=49,checked_cells=49)
        for j,cell in enumerate(cells):
            need(cell['symbol']==names[j] and cell['date']==d and cell['benchmark']==event['benchmarks'][names[j]] and cell['windows']==windows,'Exact cell identity/mapping/window')
            A=cell['available'];need(type(A) is bool,'Boolean source support')
            reason=cell['reason'];reasons[reason]+=1
            if names[j] in meta['etfs']:need(not A and reason=='etf_neutral' and cell['benchmark'] is None,'ETF neutrality')
            if A:
                need(reason=='available' and len(window)==253 and not event['missing_official_sessions'],'Accepted complete source support')
                need(isinstance(cell['benchmark'],str) and names[j] not in meta['etfs'],'Accepted industry mapping')
                need(all(finite(cell[k]) and -.75<=cell[k]<=.75 for k in ('candidate','control')),'Saved bounded raw outputs')
                need(cell['endpoint_indices']==list(range(0,253,21)) and cell['block_start_indices']==list(range(0,252,21)) and cell['block_end_indices']==list(range(21,253,21)),'Twelve fixed blocks')
                for ik,dk in (('endpoint_indices','endpoint_dates'),('block_start_indices','block_start_dates'),('block_end_indices','block_end_dates')):
                    need(cell[dk]==[window[q] for q in cell[ik]],'Saved endpoint calendar '+dk)
                signs=cell['relative_signs'];ranks=cell['twice_magnitude_ranks']
                need(len(signs)==len(ranks)==len(cell['exact_relative_z'])==12,'Saved block observation lengths')
                need(all(type(x) is int and x in (-1,0,1) for x in signs),'Exact signed flags')
                need(all(type(x) is int and 2<=x<=24 for x in ranks) and sum(ranks)==cell['twice_rank_sum']==156,'Saved rank cardinality/sum only')
                need(all(type(z) is str and z for z in cell['exact_relative_z']),'Saved exact rational string witnesses')
                need(cell['relative_positive_count']==signs.count(1) and cell['relative_negative_count']==signs.count(-1) and cell['relative_tie_count']==signs.count(0) and cell['relative_signed_count']==sum(signs),'Saved sign counters')
                need(type(cell['candidate_numerator']) is int and -156<=cell['candidate_numerator']<=156,'Recorded integer numerator bound')
                need(type(cell['constant_magnitude_copy']) is bool,'Constant branch flag')
                if cell['constant_magnitude_copy']:
                    need(ranks==[13]*12 and cell['candidate']==cell['control'],'Exact recorded constant-magnitude copy')
                need(cell['relative_sign_observations']==cell['magnitude_rank_observations']==12 and cell['magnitude_pair_comparisons']==66,'Accepted scalar descriptive counts')
                counts['available_cells']+=1;counts['constant_magnitude_copies']+=int(cell['constant_magnitude_copy'])
            else:
                need(cell['candidate'] is None and cell['control'] is None,'Unavailable cells remain unfilled')
                need(all(cell[k]==[] for k in ('endpoint_indices','block_start_indices','block_end_indices','endpoint_dates','block_start_dates','block_end_dates','relative_signs','exact_relative_z','twice_magnitude_ranks')),'Unavailable observation witnesses empty')
                need(cell['constant_magnitude_copy'] is False and all(cell[k]==0 for k in ('relative_sign_observations','magnitude_rank_observations','magnitude_pair_comparisons','candidate_numerator','twice_rank_sum')),'Unavailable descriptor counts zero')
        A=counts['available_cells'];counts['unavailable_cells']=49-A
        for key,factor in (('source_price_observations',506),('relative_sign_observations',12),('magnitude_rank_observations',12),('magnitude_pair_comparisons',66),('signed_rank_estimates',1),('sign_balance_estimates',1)):
            counts[key]=factor*A
        per_event.append(counts);support.append(dict(index=i,date=d,available=A,unavailable=49-A))
        year=by_year.setdefault(d[:4],dict(year=int(d[:4]),months=0,available_cells=0))
        year['months']+=1;year['available_cells']+=A
        for arm,key in zip(ARMS[1:],('candidate','control')):
            scalars=[c[key] for c in cells if c['available']]
            if i==0 or all(x==0. for x in scalars):
                neutral[arm].append(i);need(orders[arm][i]==orders[ARMS[0]][i],'Zero auxiliary/initialpartial retains original order')
    counts=aggregate_counts(per_event,122)
    need(all(type(audit.get(k)) is int and audit[k]==v for k,v in counts.items()),'All13 raw descriptor counters')
    need(audit['reason_counts']==dict(reasons),'Saved source reason counts')
    model_check(prepared['feature'],counts,122);delta_check(prepared['delta'],events);selection_check(prepared['selection'],122)
    for record in (prepared,completion,closure):need(record['original_array_count']==27 and record['model_counts']==counts,'Original27 and13 model counts')
    need(prepared['prefixes']==123 and prepared['feature_prefixes']==list(FEATURE_PREFIXES),'123order/5source prefixes')
    prefixes=prepared['physical_prefix_evidence'];need([x['prefix'] for x in prefixes]==list(FEATURE_PREFIXES),'Exact physical prefixes')
    for item in prefixes:
        n=item['prefix'];c=aggregate_counts(per_event,n)
        model_check(item['feature'],c,n);delta_check(item['delta'],events[:n]);selection_check(item['selection'],n)
        need(item['counts']==c and set(item['source_rows'])=={'adj','sectors'},'Physical prefix count/source fields')
        need(all(type(v) is int and (v>0 if n else v==0) for v in item['source_rows'].values()),'Physical source row metadata')
    for key in ('original_orders_exact','original_arrays_unchanged','original_core_preserved','single_original_episode','feature_prefix_sources_truncated','selection_uses_zero_cost_paths_only'):need(prepared[key] is True,'Preparation invariant '+key)
    need(prepared['additional_max_score_budget']==.01875 and completion['new_returns_computed'] is False,'Fixed additive budget/no preparation returns')
    for record in (prepared,completion):
        for key in ('new_model_fits','new_predictive_model_fits','predictive_model_fits'):need(type(record[key]) is int and record[key]==0,'No fitted model '+key)
        need(all(type(record[k]) is int and record[k]==v for k,v in counts.items()),'Flat descriptor receipt')
    return dict(counts=counts,source_reason_counts=dict(reasons),source_status_by_query=support,
        by_year=list(by_year.values()),supported_months=sum(x['available']>0 for x in support),
        support_by_period={b:dict(months=sum(x['available']>0 for x in support[lo:hi]),cells=sum(x['available'] for x in support[lo:hi])) for b,(lo,hi) in BOUNDS.items()},
        all_zero_or_initial_auxiliary_query_indices=neutral,raw_values_or_ranks_recomputed=False,npz_arrays_decoded=False,
        limitation='Saved support, identities, branches, clocks and counts checked. Exact rational gains and magnitude ranks were not rebuilt; mathematical correctness is covered by the registered independent preparation.')


def control192_evidence(evidence,plan,parity,prepared,meta,audit,orders):
    binding=plan['price_source_binding']['control192']
    need(binding['old_feature_key']=='candidate' and binding['old_order_arm']=='sector_relative_recurrence','Correct192 target reused')
    for filename,digest in binding['input_sha256'].items():evidence.pin(filename,digest)
    previous=evidence.phase(CONTROL192/'prepared',binding['prepared_completion_sha256'],binding['plan_sha256'],PREP)
    old_meta=evidence.load(CONTROL192/'prepared/META.json');old_orders=evidence.load(CONTROL192/'prepared/ORDERS.json');old_audit=evidence.load(CONTROL192/'prepared/FEATURE_AUDIT.json')
    need(meta==old_meta and orders[ARMS[2]]==old_orders['sector_relative_recurrence'],'Saved192 metadata and complete control orders exact')
    proof=parity['control192'];need(proof==prepared['control192'],'Control certificates agree')
    for key in ('passed','metadata_exact','available_exact','control_values_exact','control_orders_exact'):need(proof.get(key) is True,'Exact192 control proof '+key)
    need(proof['numerical_tolerance']==0. and proof['prepared_completion_sha256']==binding['prepared_completion_sha256'] and proof['checked_cells']==122*49 and proof['available_cells']==audit['available_cells'],'Exact192 control certificate coverage')
    need(proof['new_order_arm']==ARMS[2] and proof['old_order_arm']=='sector_relative_recurrence' and proof['old_feature_key']=='candidate','Control identity certificate')
    need(len(old_audit['event_history'])==122,'Saved192 full feature audit')
    checked=0
    for current,old in zip(audit['event_history'],old_audit['event_history']):
        for key in ('date','symbols','session_position','price_start','price_end','windows','missing_official_sessions','benchmarks'):
            need(current[key]==old[key],'Saved192 event parity '+key)
        need(len(current['cells'])==len(old['cells'])==49,'Saved192 cell identity count')
        for new,prior in zip(current['cells'],old['cells']):
            for key in ('symbol','benchmark','date','available','endpoint_dates','block_start_dates','block_end_dates','relative_signs'):
                need(new[key]==prior[key],'Saved192 cell parity '+key)
            need(new['control']==prior['candidate'],'Saved233 control equals192 relative candidate')
            checked+=1
    need(checked==122*49,'Full saved192 cell coverage')
    return dict(passed=True,saved_audit_cells_checked=checked,saved_metadata_exact=True,saved_order_stream_exact=True,
        saved_control_scalars_and_support_exact=True,registered_exact_binary64_array_certificate=proof,
        old_prepared_completion_sha256=binding['prepared_completion_sha256'],npz_decoded=False,new_features_or_orders_computed=False)


def price_source_evidence(evidence,plan,context,sr,parity,meta,calendar,model_audit,prepared):
    need(registered_digest(plan,SOURCE_REVIEW)==SOURCE_REVIEW_SHA and sr['context_sha256']==CONTEXT_SHA,'Source prospective binding')
    evidence.load(COVERAGE,COVERAGE_SHA);need(registered_digest(plan,COVERAGE)==COVERAGE_SHA,'Bounded coverage binding')
    true_flags=('passed','source_only','metadata_only','source_admission_metadata_only','metadata_axes_verified','calendar_clock_verified','raw_source_hashes_verified','raw_source_map_matches_original','sector_source_binding_verified','prior192_control_artifacts_bound','integration_source_verified')
    for key in true_flags:need(sr.get(key) is True,'Source metadata proof '+key)
    false_flags=('strict_pit','source_npz_decoded','npz_decoded','price_rows_read','source_loader_executed','source_models_imported','actual_support_computed','actual_source_coverage_recomputed','actual_new_features_computed','actual_features_computed','actual_orders_computed','actual_new_orders_computed','actual_returns_computed','observers_modified','central_modified','frozen_files_modified','source_publication_clock_certified','issuer_identity_certified')
    for key in false_flags:need(sr.get(key) is False,'Source review scope '+key)
    need(sr['actual_model_fits']==sr['actual_new_estimates']==0 and sr['raw_source_hashes']==plan['raw_source_hashes'] and len(sr['raw_source_hashes'])==64,'64 raw source hashes/no calculation')
    need(sr['evidence_sha256']==context['evidence_sha256'],'Context evidence pins')
    binding=plan['price_source_binding']
    need(path(binding['path'])==SOURCE_REVIEW and binding['sha256']==SOURCE_REVIEW_SHA and binding['coverage_sha256']==COVERAGE_SHA and binding['strict_pit'] is False,'Source admission binding')
    for field in ('input_sha256','source_sha256','evidence_sha256','direct_feature_source_sha256'):
        for filename,digest in sr[field].items():evidence.pin(filename,digest)
    for filename,digest in binding['input_sha256'].items():evidence.pin(filename,digest)
    need(sr['sector_source_binding']==plan['sector_source_binding']==binding['sector_source_binding']==parity['sector_source_binding'],'Own-industry source binding')
    need(parity['raw_source_hashes']==plan['raw_source_hashes'] and parity['price_source_binding']==binding,'Prepared original source provenance')
    for key in ('original_arrays_unchanged','original_orders_exact'):need(parity[key] is True,'Original source invariant '+key)
    need(parity['strict_pit'] is False and parity['calendar_sha256']==registered_digest(plan,CALENDAR),'Calendar/causality qualification')
    fps=parity['original_array_fingerprints'];need(len(fps)==27,'27 original byte fingerprints')
    for key,value in fps.items():
        need(set(value)=={'dtype','shape','sha256'} and value['shape'][0]==122 and type(value['sha256']) is str and len(value['sha256'])==64,'Array fingerprint metadata '+key)
    for key in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact'):need(parity['execution_prices'][key] is True,'Original execution parity '+key)
    return dict(raw_source_count=64,original_arrays=27,metadata_review_sha256=SOURCE_REVIEW_SHA,
        source_clock='253 official-session adjusted stock/own-industry prices t-294..t-42 inclusive; query-date industry mapping fixed through the formation window.',
        source_admission_metadata_only=True,strict_pit=False,source_publication_clock_certified=False,
        metadata_review_computed_support=False,registered_preparation_support_verified=True,source_arrays_decoded=False,
        mapping_policy=sr['mapping_policy'],price_policy=sr['price_policy'])


def main(expected,test_count,expected_plan_sha,expected_code_review_sha=None):
    need(type(test_count) is int and test_count==FORMAL_TESTS,'Exactly94 formal synthetic tests required')
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
    need(plan['contract']['context']==context and context['arms']==list(ARMS) and plan['contract']['dose']==.025 and plan['contract']['max_absolute_delta']==.01875 and plan['contract']['candidate']==ARMS[1] and plan['contract']['control_nonpromotable']==ARMS[2],'Contract/source theory differs')
    need(plan['contract']['selection_paths']==['same_close_zero','next_open_zero'] and plan['contract']['next_open_cost_diagnostic_only'] is True,'No fee-based selection guard in233')
    need(plan['contract']['original_pool_preserved'] is True and plan['contract']['control192_exact_feature_availability_metadata_orders_required'] is True and plan['contract']['strict_pit'] is False,'Fixed233 architecture')
    for key in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration','actual_new_parameters_before_registration','actual_new_orders_before_registration','actual_new_returns_before_registration'):need(plan[key]==0,'No preregistration actual calculation')
    need(plan['protected_files']==21444 and len(evidence.load(PROTECTED,registered_digest(plan,PROTECTED)))==21444,'Registered protection manifest')
    before=evidence.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(before['completed'])==228 and sum(r['trials'] for r in before['completed'])==1201 and not before['registered_queue'],'Pre-registration state differs')
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
    prepared=evidence.load(OUT/'prepared/INDEPENDENT_PREPARE.json');audit=evidence.load(OUT/'prepared/FEATURE_AUDIT.json')
    calendar=evidence.load(CALENDAR,registered_digest(plan,CALENDAR))
    source=evidence.load(OUT/'prepared/SOURCE_PARITY.json')
    coverage=model_coverage(audit,prepared,pc,closure,meta,calendar,orders)
    need(pc['model_fit_accounting']==closure['model_fit_accounting']==plan['contract']['model_fit_accounting'],'Descriptive accounting description')
    control_proof=control192_evidence(evidence,plan,source,prepared,meta,audit,orders)
    for name,keys in (('FEATURES.npz',ARRAY_KEYS),('TRANSFORMS_DELTAS.npz',tuple(prefix+arm for prefix in ('transform_','delta_') for arm in ARMS[1:]))):
        with zipfile.ZipFile(OUT/'prepared'/name) as archive:
            need(set(archive.namelist())=={k+'.npy' for k in keys} and len(archive.namelist())==len(keys),'NPZ member names without decoding: '+name)
    sr=evidence.load(SOURCE_REVIEW,SOURCE_REVIEW_SHA)
    source_evidence=price_source_evidence(evidence,plan,context,sr,source,meta,calendar,audit,prepared)
    results=evidence.load(OUT/'results/RESULTS.json');independent=evidence.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(results['selection']==frozen['selection'] and results['strict_pit'] is False and results['reused_sample'] is True and results['observer_promotion'] is False,'Outcome interpretation or selection differs')
    for key in ('sector_signed_magnitude_rank','control192_orders_exact','original_episode_preserved','zero_cost_selection','next_open_cost_diagnostic_only','same_support_both_arms'):need(results[key] is True,'Outcome mechanism differs '+key)
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
    need(len(state['completed'])==229 and sum(x['trials'] for x in state['completed'])==1204 and state['registered_queue']==[],'Central postclosure counters')
    need(state['completed'][-1]['directory']==OUT.name and state['completed'][-1]['completion']['sha256']==expected and path(state['completed'][-1]['completion']['path'])==OUT/'COMPLETION.json','Central closure reference')
    need(state['best_training_selected_candidate']==expected_best and state['latest_training_selected_candidate']['selection']==selection,'Central selected/best mismatch')
    evidence.pin(Path(__file__))
    report=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),reviewer_role='Round233 kernel author performing a saved-evidence audit; mathematical independence is supplied by the separately authored registered oracle. This reader imports no producer, oracle or engine.',scope='Saved JSON/hash/identity/count/selection/accounting only. NPZ directory member names inspected without decoding arrays. Only saved source-window/branch/count identities,192control parity and selection/accounting receipts are rechecked; no raw z, magnitude ranks, orders or returns recomputed.',completion_sha256=expected,plan_sha256=plan_sha,prepared_completion_sha256=sha(OUT/'prepared/COMPLETION.json'),result_completion_sha256=sha(OUT/'results/COMPLETION.json'),input_sha256=evidence.pins,
        formal_synthetic_tests=test_count,reviewed_numerical_files=6,model_flags_checked=list(MODEL_FLAGS),delta_flags_checked=list(DELTA_FLAGS),control192_evidence=control_proof,selection_flags_checked=list(SELECTION_FLAGS),model_coverage=coverage,selection=selection,changed_holdings=changes,metrics=metrics,candidate_control_vs_incumbent=comparisons,fee_evidence=fee_evidence,incumbent_versus232=incumbent,price_source_evidence=source_evidence,independent_recorded_metric_residuals=ACCOUNTING_RESIDUALS,
        independent_metric_tolerance=dict(absolute=1e-7,relative=2e-11,scope='Only production-vs-independent accounting nine-statistic checks use this established bound. Across rounds all incumbent metrics including cumulative fees are EXACT; only cost-path individual HOLDINGS.fee <=1e-10. Zero-cost fees exact0.'),
        independent_paths=9,independent_slices=45,independent_training_groups=18,zero_cost_streams=6,order_prefixes=123,source_prefixes=list(FEATURE_PREFIXES),original_array_count=27,protected_files_verified_by_registered_host_and_closer=21444,protected_archive_reread_in_this_review=False,limitations=context['limitations'],strict_pit=False,source_admission_metadata_only=True,reused_sample=True,same_close_zero_idealized=True,observers_modified=False,actual_new_features_computed=False,actual_new_parameters_computed=False,actual_new_orders_computed=False,actual_portfolio_returns_computed=False,source_npz_decoded=False,best=expected_best,target_reached=closure['target_reached'],completed_entries=229,trials=1204,central_snapshot_sha256=hashlib.sha256(central_bytes).hexdigest())
    need(CENTRAL.read_bytes()==central_bytes and sha(OUT/'COMPLETION.json')==expected and sha(OUT/'PLAN.json')==expected_plan_sha,'Central or immutable closure changed during review')
    with (OUT/'RESEARCH_REVIEW.json').open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,review_sha256=sha(OUT/'RESEARCH_REVIEW.json'),selection=selection,model_counts=coverage['counts'],same_close={a:metrics[a]['same_close_zero']['full122'] for a in ARMS}),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completion-sha256',required=True);parser.add_argument('--tests',type=int,default=FORMAL_TESTS)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--code-review-sha256')
    args=parser.parse_args();main(args.completion_sha256,args.tests,args.plan_sha256,args.code_review_sha256)

"""Write-once independent reader for closed round202.

Reads saved evidence/results only; never imports a strategy, loads raw prices,
rebuilds signals/orders, replays returns, changes thresholds or writes central.
Run review only after root confirms closure with its explicit completion SHA.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_breadth_pair202'
PRIOR=ROOT/'outputs/monthly_median_risk201'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
ARMS=('incumbent_local67','breadth_paired_relative','relative_mean_control')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
METRICS=('start','end','cagr','sharpe','max_drawdown')
COUNTS=('valid_price_observations','excess_block_observations','breadth_block_observations','down_sign_comparisons','covariance_estimates','raw_score_observations','ranked_score_observations')
BASKET=('XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY')
EXPECTED_TESTS=57
PREPARED_FILES={'FEATURES.npz','TRANSFORMS_DELTAS.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{
    a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
CONTEXT='work/monthly_breadth_pair202_context.json'
CONTEXT_SHA='0e7eb021c65a22b843c285f83525ce9ff0bc5e908e2a5a0a4cf0b1fd66952eaf'
TESTS='work/monthly_breadth_pair202_TESTS.json'
TESTS_SHA='08f7afa29cb63d87a7c0301bbe5287f088ff4ee065d9ba117e11dd0c9bd81d7b'
CODE_REVIEW='work/breadth202_CODE_REVIEW.json'
CODE_REVIEW_SHA='7001fe11be651fd01a60063da1716a0c7827cee38e5d178e06643d168be0bde5'
SOURCE='work/breadth202_SOURCE_REVIEW.json'
SOURCE_SHA='7ff43af545a29b76b8becd1f0d65b8b560baf723d484ce52d80921285b24dd5d'
FEATURE_FLAGS=('passed', 'independent_scalar_rebuild', 'own_sector_mapping_checked', 'excluded_own_benchmark_checked', 'twelve_nonoverlapping_blocks_checked', 'log_excess_returns_checked', 'strict_down_breadth_checked', 'centered_covariance_checked', 'matched_relative_mean_checked', 'exact_constant_branches_checked', 'full253_price_support_checked', 'official_session_windows_checked', 'monthend_queries_checked', 'exact_common_availability', 'audit_checked', 'no_future_or_outcome_inputs', 'centered_high_tie_ranks_checked')
DELTA_FLAGS=('passed','full_candidate_universe','initial_delta_zero','common_available_only','no_eligibility_or_model_filter','centered_cross_sectional_ranking','direct_additive_pre_score','fixed_linear_mapping_checked')
PERIOD_FEE_ATOL=1e-10
ENGINE='outputs/monthly_factor_search_v2/validate.py'
ENGINE_SHA='b5c6bbbfa5f4db0821f6b0dacb33a43c974cceb8a150578803eef73132cd30e4'


def require(ok,message):
    if not ok: raise ValueError(message)


def local_path(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    require(p.is_relative_to(ROOT),'Evidence must be local')
    return p


class Evidence:
    def __init__(self): self.pins={}
    def verify(self,value,expected=None):
        p=local_path(value);h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        digest=h.hexdigest();key=str(p.relative_to(ROOT))
        require(expected is None or digest==expected,'Changed evidence: '+key)
        require(key not in self.pins or self.pins[key]==digest,'Evidence changed during review: '+key)
        self.pins[key]=digest
        return digest
    def load(self,value,expected=None):
        self.verify(value,expected)
        return json.loads(local_path(value).read_text())
    def manifest(self,folder,expected,plan_sha,names):
        r=self.load(folder/'COMPLETION.json',expected)
        require(r['status']=='COMPLETE' and r['plan_sha256']==plan_sha,'Incomplete phase: '+str(folder))
        require(set(r['artifacts_sha256'])==names,'Phase artifact set differs')
        for p,h in r['artifacts_sha256'].items():
            require(Path(p).name==p,'Nonlocal phase artifact')
            self.verify(folder/p,h)
        return r


def metric_evidence(actual,expected,label):
    """Compare already recorded independent stats under frozen accounting tolerance."""
    for k,v in actual.items():
        require(k in expected,label+'/'+k+' missing')
        other=expected[k]
        if isinstance(v,str) or v is None or k=='trading_sessions':
            require(v==other,label+'/'+k)
        else:
            require(type(v) in (int,float) and type(other) in (int,float) and math.isfinite(v)
                and math.isfinite(other) and abs(v-other)<=1e-7+2e-11*abs(other),label+'/'+k)


def guard(candidate,reference,gain):
    return (candidate['training']['cagr']>=reference['training']['cagr']+gain-1e-12
        and candidate['training']['sharpe']>=reference['training']['sharpe']-.05
        and candidate['training']['max_drawdown']>=reference['training']['max_drawdown']-.03
        and all(candidate[p]['cagr']>=reference[p]['cagr']-.025 for p in ('early','recent')))


def selection_from_saved(metrics,orders):
    changed=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])]
    control=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[2]][i]['symbols'])]
    guards=dict(close_vs_incumbent=guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[0]]['same_close_zero'],.005),
        open_vs_incumbent=guard(metrics[ARMS[1]]['next_open_cost'],metrics[ARMS[0]]['next_open_cost'],0.),
        close_vs_relative_mean=guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[2]]['same_close_zero'],0.),
        open_vs_relative_mean=guard(metrics[ARMS[1]]['next_open_cost'],metrics[ARMS[2]]['next_open_cost'],0.),
        changed_support=len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed),
        changed_support_vs_control=len(control)>=3 and any(i<53 for i in control) and any(i>=53 for i in control))
    admitted=all(guards.values())
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=guards,
        changed_months=changed,changed_months_vs_control=control,first88_only=True,later34_used=False)


def changes(orders,reference,candidate):
    rows=[]
    for i,(old,new) in enumerate(zip(orders[reference],orders[candidate])):
        require(old['date']==new['date'] and old['end']==new['end'],'Order axes differ')
        if set(old['symbols'])!=set(new['symbols']):
            rows.append(dict(index=i,date=old['date'],end=old['end'],reference=old['symbols'],candidate=new['symbols'],
                removed=sorted(set(old['symbols'])-set(new['symbols'])),added=sorted(set(new['symbols'])-set(old['symbols']))))
    return dict(counts={p:sum(lo<=r['index']<hi for r in rows) for p,(lo,hi) in BOUNDS.items()},months=rows)


def compare_holdings(old,new,path):
    require(len(old)==len(new)==122,'Incumbent holding period count differs')
    residuals=[]
    for a,b in zip(old,new):
        require(set(a)==set(b) and {k:v for k,v in a.items() if k!='fee'}==
            {k:v for k,v in b.items() if k!='fee'},'Nonfee incumbent holding field changed')
        require(type(a['fee']) in (int,float) and type(b['fee']) in (int,float)
            and math.isfinite(a['fee']) and math.isfinite(b['fee']),'Nonfinite period fee')
        if a['fee']!=b['fee']:
            difference=b['fee']-a['fee']
            require(path=='next_open_cost' and abs(difference)<=PERIOD_FEE_ATOL,'Period fee exceeds strict1e-10 bound')
            residuals.append(dict(signal_date=a['signal_date'],execution_date=a['execution_date'],
                previous201=a['fee'],current202=b['fee'],signed_difference=difference,
                absolute_difference=abs(difference),absolute_tolerance=PERIOD_FEE_ATOL))
    return residuals


def incumbent_comparison(evidence,prior_manifest,results,prior_results):
    evidence.verify(ENGINE,ENGINE_SHA)
    code=(ROOT/ENGINE).read_text()
    snippets=['symbols=set(old)|set(chosen)','def turnover(post):return sum(abs((post/n if s in chosen and n else 0)-old.get(s,0)) for s in symbols)',
        'post=(lo+hi)/2;fee=cost*turnover(post)','for s in sorted(symbols):','totalcost+=fee;trades+=tt','fees_paid=totalcost']
    require(all(x in code for x in snippets),'Frozen fee implementation differs')
    report={}
    for path in PATHS:
        current,prior=results['arms'][ARMS[0]][path],prior_results['arms'][ARMS[0]][path]
        require(current['metrics']==prior['metrics'] and current['slices']==prior['slices'],
            'Incumbent metrics/slices must be EXACT including total fees: '+path)
        exported={}
        for kind in ('TRADES','DAILY','DAILY_POSITIONS','HOLDINGS'):
            name=ARMS[0]+'_'+path+'_'+kind+'.json'
            a=evidence.load(PRIOR/'results'/name,prior_manifest['artifacts_sha256'][name])
            b=evidence.load(OUT/'results'/name)
            if kind!='HOLDINGS':require(a==b,'Incumbent exact '+kind+' differs: '+path)
            exported[kind]=(a,b)
        residuals=compare_holdings(*exported['HOLDINGS'],path)
        total_sums={}
        for label,holds,result in (('previous201',exported['HOLDINGS'][0],prior),('current202',exported['HOLDINGS'][1],current)):
            running=0.
            for row in holds:running+=row['fee']
            require(running==result['metrics']['fees_paid'],'Chronological period fees do not exactly match recorded total')
            total_sums[label]=running
        require(total_sums['previous201']==total_sums['current202'],'Incumbent total fees must be exact')
        report[path]=dict(metrics_and_all_slices_exact=True,fee_total_exact=True,fee_totals=total_sums,
            individual_fills_exact=True,fill_rows=len(exported['TRADES'][0]),fill_fee_values_exact=True,
            daily_equity_exact=True,daily_equity_rows=len(exported['DAILY'][0]),daily_positions_exact=True,
            holding_nonfee_fields_exact=True,period_fee_differences=residuals,
            max_absolute_period_fee_difference=max([x['absolute_difference'] for x in residuals],default=0.))
    return dict(paths=report,engine_sha256=ENGINE_SHA,
        tolerance_scope='Only individual next_open_cost HOLDINGS.fee differences may be<=1e-10; all metrics/slices including total fees, fills, daily equity, positions and nonfee holding fields are exact.',
        explanation='Pinned engine sums turnover over set(old)|set(chosen), then accumulates rebalance fees chronologically. Tiny per-period fee-only differences are consistent with floating summation order; process hash seeds are not reconstructed. No wider tolerance or portfolio recomputation is used.')


def saved_arrays(evidence,feature,validation,plan):
    """Inspect saved arrays only; no factor, rank or delta is rebuilt."""
    original=ROOT/'outputs/monthly_local_linear70/DATA.npz'
    evidence.verify(original,plan['input_sha256'][str(original)])
    with np.load(original,allow_pickle=False) as data:
        require(len(data.files)==27,'Original27 stored arrays differ')
        calm=data['state_calm10'].copy()
    require(calm.shape==(122,) and calm.dtype==bool,'Original calm array differs')
    with np.load(OUT/'prepared/FEATURES.npz',allow_pickle=False) as data:v={k:data[k] for k in data.files}
    with np.load(OUT/'prepared/TRANSFORMS_DELTAS.npz',allow_pickle=False) as data:z={k:data[k] for k in data.files}
    require(set(v)=={'candidate','control','available','candidate_rank','control_rank'} and
        set(z)=={'transform_'+a for a in ARMS[1:]}|{'delta_'+a for a in ARMS[1:]},'Saved numerical schema differs')
    mask=v['available'];require(mask.shape==(122,49) and mask.dtype==bool,'Saved common availability differs')
    for key in ('candidate','control'):
        raw,rank=v[key],v[key+'_rank']
        require(raw.shape==rank.shape==mask.shape and np.issubdtype(raw.dtype,np.floating) and
            np.issubdtype(rank.dtype,np.floating) and not np.isinf(raw).any() and
            np.array_equal(np.isfinite(raw),mask),'Saved raw common support differs')
        require(np.isfinite(rank).all() and np.all(np.abs(rank)<=.5) and np.all(rank[~mask]==0.),'Saved symmetric rank bounds/missing neutrality')
    active=mask.copy();active[0]=False
    changed={};ranges={};state_counts={}
    for arm,key in zip(ARMS[1:],('candidate','control')):
        target,delta=z['transform_'+arm],z['delta_'+arm]
        require(target.shape==delta.shape==mask.shape and np.issubdtype(target.dtype,np.floating) and
            np.issubdtype(delta.dtype,np.floating),'Saved auxiliary shape/type')
        require(np.array_equal(target,v[key+'_rank']),'Saved transform changed symmetric rank')
        require(np.isfinite(delta).all() and np.max(np.abs(delta))<=.005 and
            np.all(delta[~active]==0.),'Initial/missing auxiliary must be zero within fixed dose')
        require(np.array_equal(delta!=0.,active&(target!=0.)),'Available nonzero rank was filtered or changed to zero')
        changed[arm]=int(np.count_nonzero(delta))
        ranges[arm]=dict(minimum=float(delta.min()),maximum=float(delta.max()),max_absolute=float(np.abs(delta).max()))
        state_counts[arm]=dict(calm_changed=int(np.count_nonzero(delta[calm])),noncalm_changed=int(np.count_nonzero(delta[~calm])))
    require(changed==validation['delta']['changed_cells'],'Changed delta count differs')
    for i,row in enumerate(feature['event_history']):
        require(row['available']==mask[i].tolist(),'Saved audit common mask differs')
        for j,cell in enumerate(row['cells']):
            for key in ('candidate','control'):
                actual=cell[key]
                require((actual is None and not mask[i,j]) or
                    (type(actual) in (int,float) and math.isfinite(actual) and mask[i,j] and actual==v[key][i,j]),'Saved raw/audit mismatch')
                require(cell[key+'_rank']==v[key+'_rank'][i,j],'Saved rank/audit mismatch')
    return dict(original27_archive_hash_unchanged=True,saved_features_checked=True,
        calm_rows=int(np.count_nonzero(calm)),noncalm_rows=int(np.count_nonzero(~calm)),
        raw_available_cells=int(mask.sum()),active_noninitial_cells=int(active.sum()),
        active_calm_noninitial_cells=int(active[calm].sum()),changed_cells=changed,changed_cells_by_state=state_counts,
        delta_extrema=ranges,firstpartial_and_missing_nochange=True,calm_not_suppressed=True,no_postrank=True,
        formula_verification='Hash-bound independent preparation proves canonical paired covariance, symmetric ranks and exact .01 delta. Reader only inspects saved values/masks/counts; it does not rebuild scores, ranks or deltas.')


def finite_numbers(values,length,label):
    require(isinstance(values,list) and len(values)==length and all(type(v) in (int,float) and
        math.isfinite(v) for v in values),label+' must contain finite numeric observations')


def inspect_cell(cell):
    """Validate stored evidence without recomputing block gains/breadth/covariance."""
    require(type(cell['available']) is bool,'Cell availability type')
    descriptors=('stock_log_gains','benchmark_log_gains','excess_log_returns','down_counts','breadth')
    scalars=('mean_excess','mean_breadth','covariance')
    if not cell['available']:
        require(all(cell[k]==0 for k in COUNTS) and all(cell[k]==[] for k in descriptors) and
            all(cell[k] is None for k in scalars+('candidate','control','constant_branch')) and
            cell['candidate_rank']==cell['control_rank']==0.,'Unavailable cell has computed descriptors')
        return 'unavailable'
    m=cell['breadth_count'];require(type(m) is int and m in (8,9),'Consumed breadth dimension')
    require(cell['reason']=='available','Available reason differs')
    for key in descriptors:finite_numbers(cell[key],12,key)
    require(all(type(v) is int and 0<=v<=m for v in cell['down_counts']) and
        all(0.<=v<=1. for v in cell['breadth']),'Recorded down count/breadth bounds')
    require(all(type(cell[k]) in (int,float) and math.isfinite(cell[k]) for k in scalars+('candidate','control')),'Recorded scalar nonfinite')
    require(0.<=cell['mean_breadth']<=1. and cell['control']==cell['mean_excess'],'Recorded scalar identities')
    require(cell['endpoint_indices']==list(range(0,253,21)) and
        cell['block_start_indices']==list(range(0,252,21)) and cell['block_end_indices']==list(range(21,253,21)),'Recorded21session blocks')
    w,e=cell['breadth'],cell['excess_log_returns']
    branch='all_breadth_equal' if all(x==w[0] for x in w) else 'all_excess_equal' if all(x==e[0] for x in e) else None
    require(cell['constant_branch']==branch,'Stored exactconstant branch differs')
    if branch:require(cell['covariance']==0. and cell['candidate']==cell['control'],'Constant input must copy control')
    expected=dict(valid_price_observations=253*(2+m),excess_block_observations=12,
        breadth_block_observations=12,down_sign_comparisons=12*m,covariance_estimates=1,
        raw_score_observations=2,ranked_score_observations=2)
    require(all(type(cell[k]) is int and cell[k]==v for k,v in expected.items()),'Accepted percell consumption counts')
    return branch or 'nonconstant'


def feature_history(feature,meta,sessions,source):
    require(feature['events']==122 and feature['symbols_per_event']==49 and feature['checked_cells']==5978,'Feature axes')
    require(feature['new_model_fits']==feature['predictive_model_fits']==0 and feature['portfolio_returns_calculated'] is False,'Feature scope')
    require(feature['price_count']==253 and feature['skip_sessions']==42 and feature['block_count']==12 and
        feature['block_sessions']==21 and feature['breadth_etfs']==list(BASKET),'Feature source-clock metadata')
    require(feature['common_availability'] is True and feature['ranking_performed'] is True and
        feature['ranking_eligibility_filtered'] is False and feature['trading_eligibility_modified'] is False,'Rank domain changed')
    require(len(feature['event_history'])==len(source['source_window_coverage']['by_query'])==122,'Query counts')
    counts={k:0 for k in COUNTS};reasons={};classes={};available_by=[];eight_by=[];nine_by=[]
    for i,(row,src) in enumerate(zip(feature['event_history'],source['source_window_coverage']['by_query'])):
        day=meta['dates'][i];t=sessions.index(day);endpoints=[sessions[t-294+k] for k in range(0,253,21)]
        require(row['date']==src['query']==day and row['symbols']==meta['symbols'][i] and len(row['symbols'])==49 and
            row['session_position']==t and t>=294,'Query identity')
        require(row['price_start']==src['price_start']==sessions[t-294] and
            row['price_end']==src['price_end']==sessions[t-42] and row['endpoint_dates']==src['endpoint_dates']==endpoints,'Exact253/skip42 clocks')
        require((i==0 and day=='2016-08-17') or sessions[t+1][:7]!=day[:7],'Noninitial query not monthend')
        require(row['missing_official_sessions']==src['missing_official_sessions']==[] and len(row['cells'])==49 and
            row['benchmarks']==src['benchmarks'],'Calendar or frozen own-sector mapping')
        av=eight=nine=0
        for symbol,cell,original in zip(row['symbols'],row['cells'],src['cells']):
            require(cell['symbol']==original['symbol']==symbol and cell['date']==day and
                cell['benchmark']==original['benchmark'] and cell['breadth_etfs']==original['breadth_etfs'] and
                cell['available']==original['available'],'Cell identity, basket or common source support changed')
            classification=inspect_cell(cell);classes[classification]=classes.get(classification,0)+1
            reasons[cell['reason']]=reasons.get(cell['reason'],0)+1
            if cell['available']:
                require(cell['endpoint_dates']==endpoints and cell['block_start_dates']==endpoints[:-1] and
                    cell['block_end_dates']==endpoints[1:],'Cell endpoint clocks')
                av+=1;eight+=int(cell['breadth_count']==8);nine+=int(cell['breadth_count']==9)
            for k in COUNTS:counts[k]+=cell[k]
        require(row['available']==[c['available'] for c in row['cells']] and row['available_count']==av and
            row['eight_sector_cells']==eight and row['nine_sector_cells']==nine,'Perquery counts')
        available_by.append(av);eight_by.append(eight);nine_by.append(nine)
    total=sum(available_by);coverage=source['source_window_coverage']
    require(total==coverage['common_available_cells']==3705 and sum(eight_by)==coverage['available_cells_by_basket_size']['8']==3407 and
        sum(nine_by)==coverage['available_cells_by_basket_size']['9']==298,'Frozen complete source coverage changed')
    require(counts['valid_price_observations']==coverage['accepted_source_price_observations']==9449044,'Price consumption differs')
    require(feature['available_by_event']==available_by and feature['eight_sector_cells_by_event']==eight_by and
        feature['nine_sector_cells_by_event']==nine_by and feature['available_cells']==total and
        feature['unavailable_cells']==5978-total and feature['reason_counts']==reasons,'Feature coverage totals differ')
    require(all(type(feature[k]) is int and feature[k]==v for k,v in counts.items()),'Saved descriptive totals differ')
    return counts,dict(common_available_cells=total,unavailable_cells=5978-total,eight_sector_cells=sum(eight_by),
        nine_sector_cells=sum(nine_by),constant_classes=classes,reason_counts=reasons,available_by_event=available_by,
        eight_sector_cells_by_event=eight_by,nine_sector_cells_by_event=nine_by)


def review(completion_sha):
    require(EXPECTED_TESTS is not None and FEATURE_FLAGS and DELTA_FLAGS and not TESTS_SHA.startswith('PENDING') and not CODE_REVIEW_SHA.startswith('PENDING'),'Final code/test pins must be frozen before review')
    require(not (OUT/'RESEARCH_REVIEW.json').exists(),'Final review is write-once')
    e=Evidence();closure=e.load(OUT/'COMPLETION.json',completion_sha)
    require(closure['status']=='COMPLETED_INDEPENDENTLY_VERIFIED','Round202 must already be closed')
    plan_sha=closure['plan_sha256'];plan=e.load(OUT/'PLAN.json',plan_sha)
    require(plan['schema']=='monthly_breadth_pair202_v1' and plan['contract']['arms']==list(ARMS)
        and plan['contract']['max_absolute_delta']==.005 and plan['contract']['dose']==.01 and plan['contract']['max_absolute_rank']==.5,'Different202 contract')
    context=e.load(CONTEXT,CONTEXT_SHA)
    require(plan['contract']['context']==context and plan['contract']['prospective_context_sha256']==CONTEXT_SHA,'Context differs')
    before=e.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    require(len(before['completed'])==199 and sum(x['trials'] for x in before['completed'])==1114,'Pre-registration counters differ')
    for key in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration',
        'actual_breadth_before_registration','actual_covariance_before_registration','actual_ranks_before_registration','actual_new_returns_before_registration'):
        require(type(plan[key]) is int and plan[key]==0,'Pre-registration actual computation')
    rc=e.manifest(OUT/'results',closure['result_completion_sha256'],plan_sha,RESULT_FILES)
    require(rc['paths']==9 and rc['slices']==45,'9paths45slices required')
    frozen=e.load(OUT/'selected/FROZEN.json',rc['frozen_sha256'])
    require(frozen['plan_sha256']==plan_sha,'Selection plan differs')
    prepared=e.manifest(OUT/'prepared',frozen['prepared_sha256'],plan_sha,PREPARED_FILES)
    training=e.load(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    orders=e.load(OUT/'prepared/ORDERS.json',frozen['orders_sha256'])
    results=e.load(OUT/'results/RESULTS.json');meta=e.load(OUT/'prepared/META.json')
    feature=e.load(OUT/'prepared/FEATURE_AUDIT.json');validation=e.load(OUT/'prepared/INDEPENDENT_PREPARE.json')
    outcome=e.load(OUT/'results/INDEPENDENT_OUTCOME.json');parity=e.load(OUT/'prepared/SOURCE_PARITY.json')
    require(parity['raw_source_hashes']==plan['raw_source_hashes'] and parity['price_source_binding']==plan['price_source_binding'] and parity['sector_source_binding']==plan['sector_source_binding'] and parity['source_support_exact'] is True,'Raw source binding differs')
    require(results['breadth_pair'] is True and results['same_support_both_arms'] is True
        and results['observer_promotion'] is False and results['strict_pit'] is False,'Result interpretation differs')
    code=e.load(CODE_REVIEW,CODE_REVIEW_SHA);source=e.load(SOURCE,SOURCE_SHA);tests=e.load(TESTS,TESTS_SHA)
    require(plan['numerical_code_review']['sha256']==CODE_REVIEW_SHA and plan['price_source_binding']['sha256']==SOURCE_SHA
        and plan['input_sha256'][str(ROOT/TESTS)]==TESTS_SHA,'Pre-registration evidence binding differs')
    require(code['passed'] is True and code['unresolved_blockers']==[] and source['passed'] is True
        and source['context_sha256']==CONTEXT_SHA and tests['passed'] is True and tests['synthetic_only'] is True
        and tests['tests']==EXPECTED_TESTS,'Prior reviews/tests incomplete')
    require(tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Runtime differs')
    for mapping in (code['code_sha256'],code['input_sha256'],tests['code_sha256']):
        for path,h in mapping.items():e.verify(path,h)
    for key in FEATURE_FLAGS: require(validation['feature'].get(key) is True,'Missing feature proof: '+key)
    for key in DELTA_FLAGS: require(validation['delta'].get(key) is True,'Missing delta proof: '+key)
    require(validation['feature']['computes_returns'] is False and validation['delta']['max_absolute_delta']==.005
        and validation['delta']['dose']==.01 and validation['delta']['checked_events']==122,'Numerical scope changed')
    require(validation['prefixes']==123 and validation['feature_prefixes']==[0,1,53,88,122]
        and validation['feature_prefix_sources_truncated'] is True,'Causal prefix evidence incomplete')
    for key in ('source_support_exact','original_orders_exact','original_arrays_unchanged',
        'original_core_metadata_and_support_preserved','single_original_episode'):
        require(validation[key] is True,'Missing original invariant: '+key)
    require(validation['original_array_count']==prepared['original_array_count']==closure['original_array_count']==27,'Original27 accounting differs')
    for record in (validation,prepared,closure):
        for key in ('new_model_fits','predictive_model_fits','new_predictive_model_fits'):
            require(type(record[key]) is int and record[key]==0,'New model fits reported')
    require(prepared['new_returns_computed'] is False,'Preparation computed portfolio returns')
    calendar=e.load(CALENDAR,plan['input_sha256'][str(CALENDAR)])
    counts,coverage=feature_history(feature,meta,calendar['sessions'],source)
    for record in (validation['feature'],validation,prepared,closure):
        require(all(type(record[k]) is int and record[k]==v for k,v in counts.items()),'Preparation/closure counters differ')
    integration=saved_arrays(e,feature,validation,plan)
    require(set(orders)==set(training)==set(results['arms'])==set(outcome['paths'])==set(ARMS),'Three arms required')
    require(len(meta['dates'])==len(meta['ends'])==122 and all(len(x)==122 for x in orders.values()),'122 monthly orders required')
    require(frozen['selection']==results['selection']==selection_from_saved(training,orders),'Exact first88 guards differ')
    selection=frozen['selection']
    require(outcome['passed'] is True and outcome['checked_paths']==9 and outcome['checked_slices']==45
        and outcome['independent_accounts'] is True and outcome['independent_of_production_model_and_engine'] is True
        and outcome['terminal_liquidation'] is False,'Independent outcome incomplete')
    e.verify('work/ptr169_results_independent.py',outcome['accounting_source_sha256'])
    require(closure['protected_files']==21444 and closure['observers_modified'] is False and closure['new_trials']==3,'Protection/trial closure differs')
    protected_path=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
    protected=e.load(protected_path,plan['input_sha256'][str(protected_path)])
    require(len(protected)==21444,'Protected manifest incomplete')
    predecessor=plan['predecessor_binding'];require(predecessor['directory']==PRIOR.name,'Different predecessor')
    old_closed=e.load(PRIOR/'COMPLETION.json',predecessor['input_sha256'][str(PRIOR/'COMPLETION.json')])
    old_manifest=e.load(PRIOR/'results/COMPLETION.json',old_closed['result_completion_sha256'])
    old_results=e.load(PRIOR/'results/RESULTS.json',old_manifest['artifacts_sha256']['RESULTS.json'])
    require(old_closed['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and old_manifest['status']=='COMPLETE','Prior closure incomplete')
    incumbent=incumbent_comparison(e,old_manifest,results,old_results)
    summaries={};checked_slices=0;checked_training=0
    for arm in ARMS:
        require(set(results['arms'][arm])==set(outcome['paths'][arm])==set(PATHS),'Missing execution path')
        summaries[arm]={}
        for path in PATHS:
            result,independent=results['arms'][arm][path],outcome['paths'][arm][path]
            require(independent['passed'] is True and independent['monthly_executions']==122
                and set(result['slices'])==set(independent['slices'])==set(BOUNDS),'Missing independent slice')
            require(result['metrics']==result['slices']['full122'],'Full metrics/slice mismatch')
            for period in BOUNDS:
                metric_evidence(independent['slices'][period],result['slices'][period],arm+'/'+path+'/'+period)
                checked_slices+=1
            summaries[arm][path]={period:{k:result['slices'][period][k] for k in METRICS} for period in BOUNDS}
        require(set(training[arm])=={'same_close_zero','next_open_cost'},'Selection paths differ')
        for path in training[arm]:
            require(set(training[arm][path])=={'training','early','recent'},'Selection eras differ')
            for short,full in (('training','first88'),('early','early53'),('recent','recent35')):
                metric_evidence(outcome['paths'][arm][path]['slices'][full],training[arm][path][short],'training/'+arm+'/'+path+'/'+short)
                checked_training+=1
    require(checked_slices==45 and checked_training==18,'Independent metrics coverage incomplete')
    changed=dict(candidate_vs_incumbent=changes(orders,ARMS[0],ARMS[1]),control_vs_incumbent=changes(orders,ARMS[0],ARMS[2]),
        candidate_vs_control=changes(orders,ARMS[2],ARMS[1]))
    require([x['index'] for x in changed['candidate_vs_incumbent']['months'] if x['index']<88]==selection['changed_months']
        and [x['index'] for x in changed['candidate_vs_control']['months'] if x['index']<88]==selection['changed_months_vs_control'],'First88 changed-set counts differ')
    central=e.load(CENTRAL)
    require(central['current']==OUT.name and central['current_status']=='completed_independently_verified'
        and central['registered_queue']==[] and len(central['completed'])==200
        and sum(r['trials'] for r in central['completed'])==1117,'Immediate closed202 central state differs')
    latest=central['completed'][-1];best=closure['best_training_selected_candidate'];oldbest=before['best_training_selected_candidate']
    require(latest['directory']==OUT.name and latest['completion']['sha256']==completion_sha
        and local_path(latest['completion']['path'])==OUT/'COMPLETION.json' and latest['primary']==selection['primary']
        and latest['new_admitted']==int(selection['admitted']) and latest['trials']==3
        and central['best_training_selected_candidate']==best,'Central receipt or selection differs')
    pick=selection['primary'];selected=results['arms'][pick]['same_close_zero']['metrics']
    should_update=selection['admitted'] and selected['cagr']>oldbest['cagr']
    if should_update:
        require(best['directory']==OUT.name and best['spec']==dict(arm=pick)
            and all(best[k]==selected[k] for k in ('cagr','sharpe','max_drawdown'))
            and best['next_open_5bp_cagr']==results['arms'][pick]['next_open_cost']['metrics']['cagr']
            and best['selected_on_first88'] is True and best['observer_promotion'] is False,'Best update does not match recorded selected result')
    else:require(best==oldbest,'Unqualified candidate changed historical best')
    latest_tracking=central['latest_training_selected_candidate']
    require(latest_tracking['directory']==OUT.name and latest_tracking['arm']==pick and latest_tracking['selection']==selection
        and latest_tracking['full_period_cagr']==selected['cagr'] and latest_tracking['full_period_sharpe']==selected['sharpe']
        and latest_tracking['selected_on_first88'] is True,'Latest central tracking differs')
    require(closure['target_reached']==(best['cagr']>=.7),'Target-reached label differs')
    e.verify(__file__)
    for p,h in list(e.pins.items()):e.verify(p,h)
    report=dict(schema='monthly_breadth_pair202_research_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        source_refs=e.pins,phase_sha256=dict(plan=plan_sha,prepared=frozen['prepared_sha256'],selected=rc['frozen_sha256'],results=closure['result_completion_sha256'],closure=completion_sha),
        result='ADMITTED' if selection['admitted'] else 'NOT_ADOPTED',selection=selection,
        failed_training_guards=[k for k,v in selection['guards'].items() if not v],
        best_training_selected_candidate=best,historical_best_unchanged=best==oldbest,best_update_matches_recorded_results=True,
        exact_incumbent_metrics_equal_predecessor=True,incumbent_export_diagnostic=incumbent,
        summary=summaries,holding_changes=changed,paired_feature_counts=counts,feature_coverage=coverage,rank_auxiliary_integration=integration,
        validation=dict(synthetic_tests=EXPECTED_TESTS,independent_cells=5978,independent_execution_paths=9,independent_period_slices=45,
            training_metric_groups=18,order_prefixes=123,source_prefixes=[0,1,53,88,122],original_arrays=27,
            protected_files=21444,protection_scope='Hash-bound manifest and completed registration/closure protection receipts; reader does not repeat the entire protected archive scan.',
            central_completed_entries=200,central_trials=1117,new_model_fits=0,new_predictive_model_fits=0),
        strict_pit=False,pristine_oos=False,reused_historical_sample=True,limitations=plan['contract']['limitation'],
        observers_modified=False,central_modified=False,new_factors_computed_in_this_review=False,
        new_orders_computed_in_this_review=False,new_portfolio_returns_computed_in_this_review=False,
        actual_production_modules_imported=False,saved_numerical_arrays_inspected=True,target_reached=closure['target_reached'])
    path=OUT/'RESEARCH_REVIEW.json'
    with path.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        result=report['result'],historical_best_unchanged=report['historical_best_unchanged'],target_reached=report['target_reached'],
        failed_guards=report['failed_training_guards'],holding_change_counts={k:v['counts'] for k,v in changed.items()},
        incumbent_period_fee_residuals={p:len(v['period_fee_differences']) for p,v in incumbent['paths'].items()},summary=summaries),allow_nan=False))


def self_test():
    # Fabricated summary/holding records only; no actual artifact reads.
    checks=0
    def reject(fn):
        nonlocal checks
        try:fn()
        except ValueError:checks+=1
        else:raise ValueError('Synthetic rejection expected')
    require(len(PREPARED_FILES)==7 and len(RESULT_FILES)==38,'Manifest cardinality');checks+=1
    a=[dict(signal_date=str(i),execution_date=str(i),fee=1.,symbol='A') for i in range(122)]
    b=copy.deepcopy(a);b[3]['fee']+=5e-11
    require(len(compare_holdings(a,b,'next_open_cost'))==1,'Fee residual disclosure');checks+=1
    reject(lambda:compare_holdings(a,b,'same_close_zero'))
    b[3]['fee']+=2e-10;reject(lambda:compare_holdings(a,b,'next_open_cost'))
    b=copy.deepcopy(a);b[1]['symbol']='B';reject(lambda:compare_holdings(a,b,'next_open_cost'))
    metric=dict(training=dict(cagr=.5,sharpe=1.5,max_drawdown=-.2),early=dict(cagr=.5),recent=dict(cagr=.5))
    metrics={arm:{path:copy.deepcopy(metric) for path in ('same_close_zero','next_open_cost')} for arm in ARMS}
    orders={arm:[dict(symbols=['A','B','C']) for _ in range(122)] for arm in ARMS}
    require(not selection_from_saved(metrics,orders)['admitted'],'No false admission');checks+=1
    for path in metrics[ARMS[1]]:metrics[ARMS[1]][path]['training']['cagr']=.51
    for i in (1,2,53):orders[ARMS[1]][i]['symbols']=['A','B','D']
    selected=selection_from_saved(metrics,orders);require(selected['admitted'],'Expected synthetic admission');checks+=1
    for i in range(88,122):orders[ARMS[1]][i]['symbols']=['POISON']
    require(selection_from_saved(metrics,orders)==selected,'Later34 affects selection');checks+=1
    orders[ARMS[2]]=copy.deepcopy(orders[ARMS[1]])
    require(not selection_from_saved(metrics,orders)['guards']['changed_support_vs_control'],'Matched control support guard');checks+=1
    cell=dict(available=True,reason='available',breadth_count=8,stock_log_gains=[.1]*12,
        benchmark_log_gains=[.0]*12,excess_log_returns=[.1]*12,down_counts=[0,8]*6,breadth=[0.,1.]*6,
        mean_excess=.1,mean_breadth=.5,covariance=0.,candidate=.1,control=.1,constant_branch='all_excess_equal',
        endpoint_indices=list(range(0,253,21)),block_start_indices=list(range(0,252,21)),
        block_end_indices=list(range(21,253,21)),candidate_rank=.5,control_rank=.5,
        valid_price_observations=2530,excess_block_observations=12,breadth_block_observations=12,
        down_sign_comparisons=96,covariance_estimates=1,raw_score_observations=2,ranked_score_observations=2)
    require(inspect_cell(cell)=='all_excess_equal','Exactconstant saved evidence');checks+=1
    bad=copy.deepcopy(cell);bad['candidate']=.1000000000000001;reject(lambda:inspect_cell(bad))
    bad=copy.deepcopy(cell);bad['breadth_count']=9;reject(lambda:inspect_cell(bad))
    bad=copy.deepcopy(cell);bad['stock_log_gains'].pop();reject(lambda:inspect_cell(bad))
    missing=dict(available=False,candidate=None,control=None,constant_branch=None,candidate_rank=0.,control_rank=0.,
        mean_excess=None,mean_breadth=None,covariance=None,stock_log_gains=[],benchmark_log_gains=[],
        excess_log_returns=[],down_counts=[],breadth=[],**{k:0 for k in COUNTS})
    require(inspect_cell(missing)=='unavailable','Missing source descriptors');checks+=1
    bad=copy.deepcopy(missing);bad['covariance_estimates']=1;reject(lambda:inspect_cell(bad))
    print(json.dumps(dict(passed=True,synthetic_checks=checks,actual_artifacts_read=False,new_features=False,new_orders=False,new_returns=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--completion-sha256');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        require(args.completion_sha256 is None,'Synthetic only self-test');self_test()
    else:
        require(bool(args.completion_sha256),'Explicit closed completion SHA required');review(args.completion_sha256)

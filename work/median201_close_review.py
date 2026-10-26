"""Write-once independent reader for closed round201.

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
OUT=ROOT/'outputs/monthly_median_risk201'
PRIOR=ROOT/'outputs/monthly_cmf_state200'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
ARMS=('incumbent_local67','median_mad_core','mean_mad_control')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
METRICS=('start','end','cagr','sharpe','max_drawdown')
COUNTS=('valid_price_observations','simple_return_observations','dispersion_estimates','ratio_observations','ranked_score_observations')
PREPARED_FILES={'FEATURES.npz','TRANSFORMS_DELTAS.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{
    a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
CONTEXT='work/monthly_median_risk201_context.json'
CONTEXT_SHA='4e937ed4474675b64069b410ef790697474ea2bfe03585e0f8f97d30afec9701'
TESTS='work/monthly_median_risk201_TESTS.json'
TESTS_SHA='a485c45ad23337412063184c144b8602f0a463b368196a3047ecc27d965e2da2'
CODE_REVIEW='work/median201_CODE_REVIEW.json'
CODE_REVIEW_SHA='8f76ae1dcf3fdff38671672f95516b93da9373cb632f8776adeee2b220032149'
SOURCE='work/median201_SOURCE_REVIEW.json'
SOURCE_SHA='4ca67a75465bb05f8b255272a185c436047c89fefe2193d939ec5ee9ae1e810d'
FEATURE_FLAGS=('passed','independent_scalar_rebuild','original_numerator_checked','median_center_checked','median_mean_deviations_checked','common_rank_domain_checked','original_pct_ranks_checked','full295_price_support_checked','official_session_windows_checked','monthend_queries_checked','exact_common_availability','audit_checked','no_future_or_outcome_inputs')
DELTA_FLAGS=('passed','half_core_formula_checked','original_pct_ranks_preserved','first_partial_unchanged','calm_rows_unchanged','missing_preserves_core','core_weights_checked','no_second_rank','independent_delta_rebuild')
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
        close_vs_mean=guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[2]]['same_close_zero'],0.),
        open_vs_mean=guard(metrics[ARMS[1]]['next_open_cost'],metrics[ARMS[2]]['next_open_cost'],0.),
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
                previous200=a['fee'],current201=b['fee'],signed_difference=difference,
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
        for label,holds,result in (('previous200',exported['HOLDINGS'][0],prior),('current201',exported['HOLDINGS'][1],current)):
            running=0.
            for row in holds:running+=row['fee']
            require(running==result['metrics']['fees_paid'],'Chronological period fees do not exactly match recorded total')
            total_sums[label]=running
        require(total_sums['previous200']==total_sums['current201'],'Incumbent total fees must be exact')
        report[path]=dict(metrics_and_all_slices_exact=True,fee_total_exact=True,fee_totals=total_sums,
            individual_fills_exact=True,fill_rows=len(exported['TRADES'][0]),fill_fee_values_exact=True,
            daily_equity_exact=True,daily_equity_rows=len(exported['DAILY'][0]),daily_positions_exact=True,
            holding_nonfee_fields_exact=True,period_fee_differences=residuals,
            max_absolute_period_fee_difference=max([x['absolute_difference'] for x in residuals],default=0.))
    return dict(paths=report,engine_sha256=ENGINE_SHA,
        tolerance_scope='Only individual next_open_cost HOLDINGS.fee differences may be<=1e-10; all metrics/slices including total fees, fills, daily equity, positions and nonfee holding fields are exact.',
        explanation='Pinned engine sums turnover over set(old)|set(chosen), then accumulates rebalance fees chronologically. Tiny per-period fee-only differences are consistent with floating summation order; process hash seeds are not reconstructed. No wider tolerance or portfolio recomputation is used.')


def saved_arrays(evidence,feature,validation,plan):
    """Inspect saved values/masks only; do not recalculate any factor or delta."""
    original=ROOT/'outputs/monthly_local_linear70/DATA.npz'
    evidence.verify(original,plan['input_sha256'][str(original)])
    with np.load(original,allow_pickle=False) as data:
        require(len(data.files)==27,'Original27 stored arrays differ')
        calm=data['state_calm10'].copy()
    require(calm.shape==(122,) and calm.dtype==bool,'Original calm array differs')
    with np.load(OUT/'prepared/FEATURES.npz',allow_pickle=False) as data:
        v={k:data[k] for k in data.files}
    with np.load(OUT/'prepared/TRANSFORMS_DELTAS.npz',allow_pickle=False) as data:
        z={k:data[k] for k in data.files}
    require(set(v)=={'candidate','control','raw_candidate','raw_control','available'} and
        set(z)=={'transform_'+a for a in ARMS[1:]}|{'delta_'+a for a in ARMS[1:]},'Saved numerical schema differs')
    mask=v['available']
    require(mask.shape==(122,49) and mask.dtype==bool,'Saved common availability differs')
    for key,value in v.items():
        if key=='available':continue
        require(value.shape==mask.shape and np.issubdtype(value.dtype,np.floating) and
            not np.isinf(value).any() and np.array_equal(np.isfinite(value),mask),'Saved feature support/type differs: '+key)
    for key in ('candidate','control'):
        require(np.all(np.abs(v[key][mask])<=.5),'Saved rank bounds differ')
    active=mask.copy();active[calm]=False;active[0]=False
    changed={};ranges={};neutral={}
    for arm,key in zip(ARMS[1:],('candidate','control')):
        target,delta=z['transform_'+arm],z['delta_'+arm]
        require(target.shape==mask.shape and np.issubdtype(target.dtype,np.floating) and
            not np.isinf(target).any() and np.array_equal(np.isfinite(target),active),'Saved noncalm target support differs')
        require(np.array_equal(target[active],v[key][active]),'Saved target changed original rank')
        require(delta.shape==mask.shape and np.issubdtype(delta.dtype,np.floating) and np.isfinite(delta).all()
            and np.max(np.abs(delta))<=.26,'Saved half-core delta invalid')
        require(np.all(delta[~active]==0.),'Initial/calm/missing correction must be zero')
        changed[arm]=int(np.count_nonzero(delta))
        ranges[arm]=dict(minimum=float(delta.min()),maximum=float(delta.max()),max_absolute=float(np.abs(delta).max()))
        neutral[arm]=int(np.count_nonzero(target[active]==0.))
    require(changed==validation['delta']['changed_cells'],'Changed delta count differs')
    require(validation['delta']['target_available_cells']=={a:int(active.sum()) for a in ARMS[1:]},'Target support receipt differs')
    for i,row in enumerate(feature['event_history']):
        require(row['available']==mask[i].tolist(),'Saved audit common mask differs')
        for j,cell in enumerate(row['cells']):
            for key in ('raw_candidate','raw_control','candidate','control'):
                actual=cell[key]
                require((actual is None and not mask[i,j]) or
                    (type(actual) in (int,float) and math.isfinite(actual) and mask[i,j] and actual==v[key][i,j]),
                    'Saved feature/audit mismatch: '+key)
    return dict(original27_archive_hash_unchanged=True,saved_features_checked=True,
        noncalm_rows=int(np.count_nonzero(~calm)),calm_rows=int(np.count_nonzero(calm)),
        raw_available_cells=int(mask.sum()),active_noncalm_noninitial_cells=int(active.sum()),
        changed_cells=changed,delta_extrema=ranges,active_finite_zero_targets=neutral,
        firstpartial_calm_and_missing_nochange=True,no_postrank=True,
        formula_verification='Exact allcell median/mean/rank/delta proof accepted from hash-bound independent preparation and code reviews. Reader only checks stored arrays, masks, counts and artifact equality; no ratios, ranks or deltas are rebuilt.')


def finite_numbers(values,length,label):
    require(isinstance(values,list) and len(values)==length and all(type(v) in (int,float) and
        math.isfinite(v) for v in values),label+' must contain finite numeric observations')


def inspect_cell(cell):
    """Validate recorded descriptors without recalculating medians or means."""
    require(type(cell['source_available']) is bool and type(cell['available']) is bool,'Cell availability types')
    pricekeys=('start_price','formation_end_price','risk_start_price','end_price')
    numerickeys=('numerator','median_return','median_deviation','mean_deviation')
    if not cell['source_available']:
        require(not cell['available'] and cell['dispersion_estimates']==0 and
            cell['simple_returns']==cell['absolute_deviations']==[] and
            all(cell[k] is None for k in pricekeys+numerickeys+('raw_candidate','raw_control','candidate','control')),
            'Unavailable source has computed descriptors')
        return 'missing_source'
    require(all(type(cell[k]) in (int,float) and math.isfinite(cell[k]) and cell[k]>0 for k in pricekeys),
        'Recorded price endpoints invalid')
    require(all(type(cell[k]) in (int,float) and math.isfinite(cell[k]) for k in numerickeys),
        'Recorded descriptive statistic invalid')
    finite_numbers(cell['simple_returns'],63,'Stored simple returns')
    finite_numbers(cell['absolute_deviations'],63,'Stored absolute deviations')
    require(all(v>=0 for v in cell['absolute_deviations']) and cell['median_deviation']>=0 and
        cell['mean_deviation']>=0 and cell['dispersion_estimates']==2,'Stored dispersion evidence invalid')
    if cell['available']:
        require(cell['reason']=='available' and cell['median_deviation']>0 and cell['mean_deviation']>0 and
            all(type(cell[k]) in (int,float) and math.isfinite(cell[k]) for k in
                ('raw_candidate','raw_control','candidate','control')),'Common positive denominator support differs')
        return 'available'
    require(cell['reason']=='zero_dispersion' and (cell['median_deviation']==0 or cell['mean_deviation']==0)
        and all(cell[k] is None for k in ('raw_candidate','raw_control','candidate','control')),
        'Common zero-denominator fallback differs')
    return 'zero_dispersion'


def feature_history(feature,meta,sessions,source):
    require(feature['events']==122 and feature['symbols_per_event']==49 and feature['checked_cells']==5978,'Feature axis differs')
    require(feature['new_model_fits']==feature['predictive_model_fits']==0 and feature['portfolio_returns_calculated'] is False,'Feature scope differs')
    require(feature['price_count']==295 and feature['skip_sessions']==42 and feature['current_return_count']==63,
        'Feature source-clock metadata differs')
    require(feature['common_availability'] is True and feature['ranking_performed'] is True and
        feature['ranking_eligibility_filtered'] is False and feature['trading_eligibility_modified'] is False,
        'Original rank domain changed')
    require(len(feature['event_history'])==122,'Feature query count differs')
    reasons={};source_counts=[];available_counts=[];zero_median=zero_mean=0;classes={}
    for i,row in enumerate(feature['event_history']):
        day=meta['dates'][i];t=sessions.index(day)
        require(row['date']==day and row['symbols']==meta['symbols'][i] and len(row['symbols'])==49
            and row['session_position']==t and t>=294,'Saved query identity differs')
        clocks=dict(price_start=sessions[t-294],price_end=day,numerator_start=sessions[t-294],
            numerator_end=sessions[t-42],risk_price_start=sessions[t-63],risk_price_end=day)
        require(all(row[k]==v for k,v in clocks.items()),'Saved295/63 source clocks differ')
        require((i==0 and day=='2016-08-17') or sessions[t+1][:7]!=day[:7],'Noninitial query not monthend')
        require(row['missing_official_sessions']==[] and len(row['cells'])==49,'Saved source calendar/cell width differs')
        sc=av=0
        for symbol,cell in zip(row['symbols'],row['cells']):
            require(cell['symbol']==symbol and cell['date']==day,'Saved cell identity differs')
            status=inspect_cell(cell);classes[status]=classes.get(status,0)+1
            reasons[cell['reason']]=reasons.get(cell['reason'],0)+1
            sc+=int(cell['source_available']);av+=int(cell['available'])
            if cell['source_available']:
                zero_median+=int(cell['median_deviation']==0);zero_mean+=int(cell['mean_deviation']==0)
        require(row['source_available']==[c['source_available'] for c in row['cells']] and
            row['available']==[c['available'] for c in row['cells']] and
            row['source_available_count']==sc and row['available_count']==av,'Per-event source/ratio counts differ')
        source_counts.append(sc);available_counts.append(av)
    source_total,available=sum(source_counts),sum(available_counts)
    require(source_counts==[r['complete_positive_price_cells'] for r in source['source_window_coverage']['by_query']]
        and source_total==source['source_window_coverage']['complete_positive_price_cells'],'Source coverage changed')
    require(feature['source_available_by_event']==source_counts and feature['available_by_event']==available_counts
        and feature['source_available_cells']==source_total and feature['available_cells']==available
        and feature['unavailable_cells']==5978-available and feature['reason_counts']==reasons,'Feature coverage totals differ')
    counts=dict(valid_price_observations=295*source_total,simple_return_observations=63*source_total,
        dispersion_estimates=2*source_total,ratio_observations=2*available,ranked_score_observations=2*available)
    require(all(type(feature[k]) is int and feature[k]==value for k,value in counts.items()),'Descriptive accounting differs')
    detail=dict(source_complete_cells=source_total,source_missing_cells=5978-source_total,common_ratio_cells=available,
        common_unavailable_cells=5978-available,zero_median_denominator_cells=zero_median,
        zero_mean_denominator_cells=zero_mean,cell_classes=classes,reason_counts=reasons,
        source_available_by_event=source_counts,available_by_event=available_counts,
        median_denominator_estimates=source_total,mean_denominator_estimates=source_total)
    return counts,detail


def review(completion_sha):
    require(not (OUT/'RESEARCH_REVIEW.json').exists(),'Final review is write-once')
    e=Evidence();closure=e.load(OUT/'COMPLETION.json',completion_sha)
    require(closure['status']=='COMPLETED_INDEPENDENTLY_VERIFIED','Round201 must already be closed')
    plan_sha=closure['plan_sha256'];plan=e.load(OUT/'PLAN.json',plan_sha)
    require(plan['schema']=='monthly_median_risk201_v1' and plan['contract']['arms']==list(ARMS)
        and plan['contract']['max_absolute_delta']==.26 and plan['contract']['max_absolute_rank']==.5,'Different201 contract')
    context=e.load(CONTEXT,CONTEXT_SHA)
    require(plan['contract']['context']==context and plan['contract']['prospective_context_sha256']==CONTEXT_SHA,'Context differs')
    before=e.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    require(len(before['completed'])==198 and sum(x['trials'] for x in before['completed'])==1111,'Pre-registration counters differ')
    for key in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration',
        'actual_dispersion_before_registration','actual_ratio_ranks_before_registration','actual_new_returns_before_registration'):
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
    require(parity['raw_source_hashes']==plan['raw_source_hashes'] and parity['price_source_binding']==plan['price_source_binding'],'Raw source binding differs')
    require(results['median_risk'] is True and results['same_support_both_arms'] is True
        and results['observer_promotion'] is False and results['strict_pit'] is False,'Result interpretation differs')
    code=e.load(CODE_REVIEW,CODE_REVIEW_SHA);source=e.load(SOURCE,SOURCE_SHA);tests=e.load(TESTS,TESTS_SHA)
    require(plan['numerical_code_review']['sha256']==CODE_REVIEW_SHA and plan['price_source_binding']['sha256']==SOURCE_SHA
        and plan['input_sha256'][str(ROOT/TESTS)]==TESTS_SHA,'Pre-registration evidence binding differs')
    require(code['passed'] is True and code['unresolved_blockers']==[] and source['passed'] is True
        and source['context_sha256']==CONTEXT_SHA and tests['passed'] is True and tests['synthetic_only'] is True
        and tests['tests']==60,'Prior reviews/tests incomplete')
    require(tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Runtime differs')
    for mapping in (code['code_sha256'],code['input_sha256'],tests['code_sha256']):
        for path,h in mapping.items():e.verify(path,h)
    for key in FEATURE_FLAGS: require(validation['feature'].get(key) is True,'Missing feature proof: '+key)
    for key in DELTA_FLAGS: require(validation['delta'].get(key) is True,'Missing delta proof: '+key)
    require(validation['feature']['computes_returns'] is False and validation['delta']['max_absolute_delta']==.26
        and validation['delta']['blend']==.5 and validation['delta']['checked_events']==122,'Numerical scope changed')
    require(validation['prefixes']==123 and validation['feature_prefixes']==[0,1,53,88,122]
        and validation['feature_prefix_sources_truncated'] is True,'Causal prefix evidence incomplete')
    for key in ('original_orders_exact','original_arrays_unchanged',
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
        and central['registered_queue']==[] and len(central['completed'])==199
        and sum(r['trials'] for r in central['completed'])==1114,'Immediate closed201 central state differs')
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
    report=dict(schema='monthly_median_risk201_research_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        source_refs=e.pins,phase_sha256=dict(plan=plan_sha,prepared=frozen['prepared_sha256'],selected=rc['frozen_sha256'],results=closure['result_completion_sha256'],closure=completion_sha),
        result='ADMITTED' if selection['admitted'] else 'NOT_ADOPTED',selection=selection,
        failed_training_guards=[k for k,v in selection['guards'].items() if not v],
        best_training_selected_candidate=best,historical_best_unchanged=best==oldbest,best_update_matches_recorded_results=True,
        exact_incumbent_metrics_equal_predecessor=True,incumbent_export_diagnostic=incumbent,
        summary=summaries,holding_changes=changed,dispersion_feature_counts=counts,feature_coverage=coverage,halfcore_integration=integration,
        validation=dict(synthetic_tests=60,independent_cells=5978,independent_execution_paths=9,independent_period_slices=45,
            training_metric_groups=18,order_prefixes=123,source_prefixes=[0,1,53,88,122],original_arrays=27,
            protected_files=21444,protection_scope='Hash-bound manifest and completed registration/closure protection receipts; reader does not repeat the entire protected archive scan.',
            central_completed_entries=199,central_trials=1114,new_model_fits=0,new_predictive_model_fits=0),
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
    cell=dict(source_available=True,available=True,reason='available',start_price=10.,formation_end_price=12.,
        risk_start_price=11.,end_price=12.,numerator=.2,median_return=.01,median_deviation=.02,
        mean_deviation=.03,simple_returns=[.01]*63,absolute_deviations=[.02]*63,
        raw_candidate=10.,raw_control=6.,candidate=.5,control=.5,dispersion_estimates=2)
    require(inspect_cell(cell)=='available','Saved positive-denominator evidence');checks+=1
    missing=copy.deepcopy(cell);missing.update(available=False,reason='zero_dispersion',median_deviation=0.,
        raw_candidate=None,raw_control=None,candidate=None,control=None)
    require(inspect_cell(missing)=='zero_dispersion','Median0/positiveMean shared fallback');checks+=1
    bad=copy.deepcopy(missing);bad['available']=True;reject(lambda:inspect_cell(bad))
    bad=copy.deepcopy(cell);bad['simple_returns'].pop();reject(lambda:inspect_cell(bad))
    print(json.dumps(dict(passed=True,synthetic_checks=checks,actual_artifacts_read=False,new_features=False,new_orders=False,new_returns=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--completion-sha256');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        require(args.completion_sha256 is None,'Synthetic only self-test');self_test()
    else:
        require(bool(args.completion_sha256),'Explicit closed completion SHA required');review(args.completion_sha256)

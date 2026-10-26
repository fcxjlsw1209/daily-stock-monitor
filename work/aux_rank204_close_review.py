"""Write-once saved-evidence reader for closed round204.

The author also implemented the production kernel; this reader is not an
independent mathematical audit of that implementation. Numerical independence
comes from the separately authored scalar oracle and its hash-bound receipts.

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
OUT=ROOT/'outputs/monthly_aux_rank204'
PRIOR=ROOT/'outputs/monthly_volume_pair203'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
ARMS=('incumbent_local67','aux_rank_full','aux_affine_control')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
METRICS=('start','end','cagr','sharpe','max_drawdown')
COUNTS=('rank_rows','constant_rows','two_level_rows','active_forecast_cells')
VALUE_KEYS=('support','weight','candidate','control','candidate_delta','control_delta')
EXPECTED_TESTS=60
PREPARED_FILES={'FEATURES.npz','TRANSFORMS_DELTAS.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{
    a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
CONTEXT='work/monthly_aux_rank204_context.json'
CONTEXT_SHA='602df5d3ec9ae914f147fee5c66167f6c554c4396c4a11d5dd1c24f78e0f8233'
TESTS='work/monthly_aux_rank204_TESTS.json'
TESTS_SHA='bf4557ccc4fd1432cf692e06934385d8796193f0e538a9203e3c9b17ec2f42a0'
CODE_REVIEW='work/aux_rank204_CODE_REVIEW.json'
CODE_REVIEW_SHA='cd3c89c3d4486a1d498d9b2328bebb68e1ae2bbf4803b375f247f7d9f1c40867'
COVERAGE='work/after203_aux_rank_coverage.json'
COVERAGE_SHA='a8f2388530dcfb0834ea0e3d9e98cce1717c6b6021b7e8ec44cf8c7cb7259b96'
FINAL_CODE_SHA={
    'work/monthly_aux_rank204_host.py':'d4713fe0b1d273b250bbde10947dbc836ef3d277135802811278cfb9fd2a6647',
    'work/test_monthly_aux_rank204_host.py':'1e4942b79433e5feb7708c23647036442b1689fbb45238c60a938e00f1942db7',
    'work/monthly_aux_rank204_kernel.py':'c44dd6b24ce060513a92890148675d7e091a5e0d91061da9d64637ecd160ac65',
    'work/test_monthly_aux_rank204_kernel.py':'1b233d87ec34cb83a30a1ede793ad6b20c552a9fe5b3249651eab382ee52d562',
    'work/monthly_aux_rank204_independent.py':'e9e878267ea4e0e94bfff760813c7df1deb4e017daa6d0a59e80751e5ea13d61',
    'work/test_monthly_aux_rank204_independent.py':'c81c22969f14dd0a4b7bd375c7a726c2b068be74b4f190534ffeef26d1e25244'}
FEATURE_FLAGS=('passed','independent_scalar_rebuild','full_auxiliary_support_checked','core_weights_checked','full_candidate_universe','average_tie_ranks_checked','affine_span_control_checked','two_level_equivalence_checked','constant_rows_unchanged','first_partial_unchanged','effective_support_weight_checked','no_eligibility_rank_filter','no_new_availability_filter','no_future_or_outcome_inputs','audit_checked')
DELTA_FLAGS=('passed','independent_delta_rebuild','effective_support_weight_checked','full_auxiliary_replacement_checked','first_partial_unchanged','constant_rows_unchanged','core_weights_checked','no_eligibility_delta_filter','source_values_checked')
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
        close_vs_affine_control=guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[2]]['same_close_zero'],0.),
        open_vs_affine_control=guard(metrics[ARMS[1]]['next_open_cost'],metrics[ARMS[2]]['next_open_cost'],0.),
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
                previous203=a['fee'],current204=b['fee'],signed_difference=difference,
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
        for label,holds,result in (('previous203',exported['HOLDINGS'][0],prior),('current204',exported['HOLDINGS'][1],current)):
            running=0.
            for row in holds:running+=row['fee']
            require(running==result['metrics']['fees_paid'],'Chronological period fees do not exactly match recorded total')
            total_sums[label]=running
        require(total_sums['previous203']==total_sums['current204'],'Incumbent total fees must be exact')
        report[path]=dict(metrics_and_all_slices_exact=True,fee_total_exact=True,fee_totals=total_sums,
            individual_fills_exact=True,fill_rows=len(exported['TRADES'][0]),fill_fee_values_exact=True,
            daily_equity_exact=True,daily_equity_rows=len(exported['DAILY'][0]),daily_positions_exact=True,
            holding_nonfee_fields_exact=True,period_fee_differences=residuals,
            max_absolute_period_fee_difference=max([x['absolute_difference'] for x in residuals],default=0.))
    return dict(paths=report,engine_sha256=ENGINE_SHA,
        tolerance_scope='Only individual next_open_cost HOLDINGS.fee differences may be<=1e-10; all metrics/slices including total fees, fills, daily equity, positions and nonfee holding fields are exact.',
        explanation='Pinned engine sums turnover over set(old)|set(chosen), then accumulates rebalance fees chronologically. Tiny per-period fee-only differences are consistent with floating summation order; process hash seeds are not reconstructed. No wider tolerance or portfolio recomputation is used.')


def inspect_saved(values,audit):
    """Inspect saved descriptors only; never rebuild support, ranks, W or deltas."""
    require(set(values)==set(VALUE_KEYS),'Six saved cache transformation fields required')
    support=values['support'];require(support.ndim==2 and support.shape[1]>0,'Saved support matrix axes')
    n,width=support.shape
    require(all(v.shape==(n,width) and v.dtype==np.dtype('float64') and np.isfinite(v).all()
        for v in values.values()),'Saved cache values must be aligned finite binary64')
    require(np.all(np.abs(support)<=.5+1e-12),'Stored support outside original validation bound')
    require(np.all(values['weight']>0.) and np.all(values['weight']<=.5),'Stored effective support budget')
    for key in ('candidate','control','candidate_delta','control_delta'):
        require(np.all(np.abs(values[key])<=.5),'Stored rank/delta bound')
    require(audit['events']==audit['rank_rows']==n and audit['symbols_per_event']==width and
        audit['checked_cells']==n*width and len(audit['event_history'])==n,'Audit axis/cardinality differs')
    require(audit['new_model_fits']==audit['predictive_model_fits']==0 and
        audit['ranking_eligibility_filtered'] is False and audit['new_availability_filter'] is False and
        audit['portfolio_returns_calculated'] is False,'Cached transform scope differs')
    constant=two=active=0
    changed={arm:0 for arm in ARMS[1:]};maxima={arm:0. for arm in ARMS[1:]}
    for i,row in enumerate(audit['event_history']):
        require(type(row['query_index']) is int and row['query_index']==i,'Recorded query index')
        levels=int(len(np.unique(support[i])))
        require(type(row['distinct_levels']) is int and row['distinct_levels']==levels and
            type(row['constant']) is bool and row['constant']==(levels==1) and
            type(row['two_levels']) is bool and row['two_levels']==(levels==2),'Saved degeneracy class')
        require(type(row['active_forecast_count']) is int and 0<=row['active_forecast_count']<=width,'Recorded funding mask count')
        active+=row['active_forecast_count'];constant+=int(row['constant']);two+=int(row['two_levels'])
        for key in ('support','candidate','control'):
            require(row[key+'_min']==float(values[key][i].min()) and
                row[key+'_max']==float(values[key][i].max()),'Recorded row extrema differ')
        q,c=values['candidate'][i],values['control'][i]
        require(row['max_absolute_candidate_control_difference']==float(np.abs(q-c).max()),'Recorded candidate/control difference')
        if row['constant']:
            require(np.all(q==0.) and np.all(c==0.),'Constant target rows must be zero')
        else:
            require(np.array_equal(c[support[i]==support[i].min()],q[support[i]==support[i].min()]) and
                np.array_equal(c[support[i]==support[i].max()],q[support[i]==support[i].max()]),'Affine endpoints differ')
        if row['two_levels']:
            require(np.array_equal(q,c) and np.array_equal(values['candidate_delta'][i],values['control_delta'][i]),'Exact two-level identity')
        for arm,key in zip(ARMS[1:],('candidate','control')):
            delta=values[key+'_delta'][i]
            if not i or row['constant']:require(np.all(delta==0.),'First/constant whole-row no-op differs')
            maximum=float(np.abs(delta).max())
            require(row['max_absolute_'+key+'_delta']==maximum,'Recorded delta maximum differs')
            maxima[arm]=max(maxima[arm],maximum);changed[arm]+=int(np.count_nonzero(delta))
    counts=dict(rank_rows=n,constant_rows=constant,two_level_rows=two,active_forecast_cells=active)
    require(all(type(audit[k]) is int and audit[k]==v for k,v in counts.items()),'Recorded cache transformation totals differ')
    return counts,dict(changed_cells=changed,per_arm_max_absolute_delta=maxima,
        observed_max_absolute_delta=max(maxima.values()),all_six_saved_arrays_finite=True,
        firstpartial_and_constant_nochange=True,two_level_equivalence_exact=True,full_candidate_domain=True,
        original_auxiliary_support_not_recomputed=True,original_effective_weight_not_recomputed=True,
        ranks_affine_and_deltas_not_recomputed=True)


def saved_arrays(evidence,feature,validation,plan):
    original=ROOT/'outputs/monthly_local_linear70/DATA.npz'
    evidence.verify(original,plan['input_sha256'][str(original)])
    with np.load(original,allow_pickle=False) as data:
        require(len(data.files)==27,'Original27 stored arrays differ')
        calm=data['state_calm10'].copy()
    require(calm.shape==(122,) and calm.dtype==bool,'Original calm array differs')
    with np.load(OUT/'prepared/FEATURES.npz',allow_pickle=False) as data:values={k:data[k] for k in data.files}
    with np.load(OUT/'prepared/TRANSFORMS_DELTAS.npz',allow_pickle=False) as data:saved={k:data[k] for k in data.files}
    require(set(saved)=={'transform_'+a for a in ARMS[1:]}|{'delta_'+a for a in ARMS[1:]},'Saved transform/export schema')
    require(values['support'].shape==(122,49),'Actual cached transformation axes')
    counts,integration=inspect_saved(values,feature)
    state_counts={}
    for arm,key in zip(ARMS[1:],('candidate','control')):
        require(np.array_equal(saved['transform_'+arm],values[key]) and
            np.array_equal(saved['delta_'+arm],values[key+'_delta']),'Cached feature/export exact parity')
        delta=values[key+'_delta']
        state_counts[arm]=dict(calm_changed=int(np.count_nonzero(delta[calm])),noncalm_changed=int(np.count_nonzero(delta[~calm])))
    for key in ('changed_cells','per_arm_max_absolute_delta','observed_max_absolute_delta'):
        require(integration[key]==validation['delta'][key],'Independent saved delta summary differs: '+key)
    integration.update(original27_archive_hash_unchanged=True,calm_rows=int(np.count_nonzero(calm)),
        noncalm_rows=int(np.count_nonzero(~calm)),changed_cells_by_state=state_counts,
        transformation_scope='Full existing effective PRE-episode auxiliary contribution; original episode still uses original support.',
        formula_verification='Hash-bound preparation from the separately authored scalar oracle verifies S, W, full-domain average-tie ranks, endpoint-matched affine values and deltas exactly. This kernel-author reader inspects saved evidence only.')
    return counts,integration


def cache_binding(evidence,plan,parity,meta):
    binding=plan['cache_source_binding']
    require(binding['directory']=='monthly_local_linear70' and binding['events']==122 and
        binding['symbols_per_event']==49 and binding['original_array_count']==27 and
        binding['fresh_raw_price_reconstruction'] is False and binding['new_source_features'] is False,'Original cache source scope')
    require(parity['cache_source_binding']==binding and parity['raw_source_hashes']==plan['raw_source_hashes'] and
        parity['original_loader_registered_checks'] is True and parity['original_cache_exact'] is True and
        parity['fresh_raw_price_reconstruction'] is False,'Original cache parity proof differs')
    for path,h in binding['input_sha256'].items():
        require(plan['input_sha256'].get(str(local_path(path)))==h,'Cache source absent from persistent PLAN pins')
        evidence.verify(path,h)
    original_meta=ROOT/'outputs/monthly_local_linear70/META.json'
    require(meta==evidence.load(original_meta,binding['input_sha256'][str(original_meta)]),'Original monthly identities differ')
    require(len(meta['dates'])==122 and all(len(x)==49 for x in meta['symbols']) and
        meta['dates'][0]=='2016-08-17' and meta['dates'][-1]=='2026-08-31' and meta['ends'][-1]=='2026-09-24','Original period/width differs')
    return dict(original_cache_and_identity_hashes_verified=True,original_arrays=27,events=122,width=49,
        fresh_raw_price_reconstruction=False,new_source_features=False,all5978_cached_cells_verified_by_separate_oracle=True)


def review(completion_sha):
    require(EXPECTED_TESTS is not None and FEATURE_FLAGS and DELTA_FLAGS and not TESTS_SHA.startswith('PENDING') and not CODE_REVIEW_SHA.startswith('PENDING') and all(not h.startswith('PENDING') for h in FINAL_CODE_SHA.values()),'Final code/test pins must be frozen before review')
    require(not (OUT/'RESEARCH_REVIEW.json').exists(),'Final review is write-once')
    e=Evidence();closure=e.load(OUT/'COMPLETION.json',completion_sha)
    require(closure['status']=='COMPLETED_INDEPENDENTLY_VERIFIED','Round204 must already be closed')
    plan_sha=closure['plan_sha256'];plan=e.load(OUT/'PLAN.json',plan_sha)
    require(plan['schema']=='monthly_aux_rank204_v1' and plan['contract']['arms']==list(ARMS)
        and plan['contract']['max_absolute_delta']==.5 and plan['contract']['coverage_sha256']==COVERAGE_SHA,'Different204 contract')
    context=e.load(CONTEXT,CONTEXT_SHA)
    require(plan['contract']['context']==context and plan['contract']['prospective_context_sha256']==CONTEXT_SHA,'Context differs')
    require(len(plan['raw_source_hashes'])==64,'Original raw64 source closure differs')
    for mapping in (plan['input_sha256'],plan['raw_source_hashes'],plan['calendar_source_hashes']):
        for path,h in mapping.items():
            require(local_path(path)!=CENTRAL,'Mutable central cannot be a persistent PLAN pin')
            require(plan['input_sha256'].get(str(local_path(path)))==h,'Source absent from persistent PLAN pins')
            e.verify(path,h)
    before=e.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    require(len(before['completed'])==201 and sum(x['trials'] for x in before['completed'])==1120,'Pre-registration counters differ')
    for key in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration',
        'actual_cached_transforms_before_registration','actual_ranks_before_registration','actual_new_returns_before_registration'):
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
    coverage=cache_binding(e,plan,parity,meta)
    require(results['full_pre_episode_auxiliary_rank'] is True and results['original_support_retained_for_episode'] is True
        and results['observer_promotion'] is False and results['strict_pit'] is False,'Result interpretation differs')
    code=e.load(CODE_REVIEW,CODE_REVIEW_SHA);e.load(COVERAGE,COVERAGE_SHA);tests=e.load(TESTS,TESTS_SHA)
    require(plan['numerical_code_review']['sha256']==CODE_REVIEW_SHA
        and plan['input_sha256'][str(ROOT/TESTS)]==TESTS_SHA,'Pre-registration evidence binding differs')
    require(code['passed'] is True and code['unresolved_blockers']==[] and code['context_sha256']==CONTEXT_SHA and tests['passed'] is True and tests['synthetic_only'] is True
        and tests['tests']==EXPECTED_TESTS,'Prior reviews/tests incomplete')
    require(tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Runtime differs')
    require(code['code_sha256']==FINAL_CODE_SHA,'Final six numerical source hashes differ')
    for path,h in FINAL_CODE_SHA.items():
        require(tests['code_sha256'][str(ROOT/path)]==h,'Test receipt final source mismatch')
    for mapping in (code['code_sha256'],code['input_sha256'],tests['code_sha256']):
        for path,h in mapping.items():e.verify(path,h)
    for key in FEATURE_FLAGS: require(validation['feature'].get(key) is True,'Missing feature proof: '+key)
    for key in DELTA_FLAGS: require(validation['delta'].get(key) is True,'Missing delta proof: '+key)
    require(validation['feature']['events']==122 and validation['feature']['symbols_per_event']==49
        and validation['feature']['checked_cells']==5978,'Independent cached transformation coverage differs')
    require(validation['feature']['computes_returns'] is False and validation['feature']['numerical_tolerance']==0.
        and validation['feature']['support_bound_tolerance']==1e-12 and validation['delta']['max_absolute_delta']==.5,'Numerical scope changed')
    require(validation['prefixes']==123 and validation['feature_prefixes']==list(range(123))
        and validation['feature_prefix_sources_truncated'] is True,'Causal prefix evidence incomplete')
    for key in ('original_cache_exact','original_orders_exact','original_arrays_unchanged',
        'original_core_metadata_and_support_preserved','single_original_episode'):
        require(validation[key] is True,'Missing original invariant: '+key)
    require(validation['original_array_count']==prepared['original_array_count']==closure['original_array_count']==27,'Original27 accounting differs')
    for record in (validation,prepared,closure):
        for key in ('new_model_fits','predictive_model_fits','new_predictive_model_fits'):
            require(type(record[key]) is int and record[key]==0,'New model fits reported')
    require(prepared['new_returns_computed'] is False,'Preparation computed portfolio returns')
    e.load(CALENDAR,plan['input_sha256'][str(CALENDAR)])
    counts,integration=saved_arrays(e,feature,validation,plan)
    for record in (validation['feature'],validation,prepared,closure):
        require(all(type(record[k]) is int and record[k]==v for k,v in counts.items()),'Preparation/closure counters differ')
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
        and central['registered_queue']==[] and len(central['completed'])==202
        and sum(r['trials'] for r in central['completed'])==1123,'Immediate closed204 central state differs')
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
    report=dict(schema='monthly_aux_rank204_research_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        reviewer_was_production_kernel_author=True,independent_of_production_kernel=False,
        review_scope='Saved evidence integrity, recorded cached-transform counts/bounds/identities, exact first88 selection guards, existing result consistency and unchanged incumbent accounting; no independent factor recomputation by this reader.',
        independent_numerical_evidence_source='Separately authored work/monthly_aux_rank204_independent.py and hash-bound INDEPENDENT_PREPARE.json/INDEPENDENT_OUTCOME.json',
        source_refs=e.pins,phase_sha256=dict(plan=plan_sha,prepared=frozen['prepared_sha256'],selected=rc['frozen_sha256'],results=closure['result_completion_sha256'],closure=completion_sha),
        result='ADMITTED' if selection['admitted'] else 'NOT_ADOPTED',selection=selection,
        failed_training_guards=[k for k,v in selection['guards'].items() if not v],
        best_training_selected_candidate=best,historical_best_unchanged=best==oldbest,best_update_matches_recorded_results=True,
        exact_incumbent_metrics_equal_predecessor=True,incumbent_export_diagnostic=incumbent,
        summary=summaries,holding_changes=changed,cached_transform_counts=counts,cache_source_coverage=coverage,rank_auxiliary_integration=integration,
        validation=dict(synthetic_tests=EXPECTED_TESTS,independent_cells=5978,independent_execution_paths=9,independent_period_slices=45,
            training_metric_groups=18,order_prefixes=123,physical_cache_prefixes=list(range(123)),original_arrays=27,
            protected_files=21444,protection_scope='Hash-bound manifest and completed registration/closure protection receipts; reader does not repeat the entire protected archive scan.',
            central_completed_entries=202,central_trials=1123,new_model_fits=0,new_predictive_model_fits=0),
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
    # Fabricated recorded arrays. No production functions or source values used.
    values=dict(support=np.array([[-.25,0.,.25],[-.25,0.,.25],[.125]*3]),weight=np.full((3,3),.5),
        candidate=np.array([[-.5,0.,.5],[-.5,0.,.5],[0.]*3]),control=np.array([[-.5,0.,.5],[-.5,0.,.5],[0.]*3]),
        candidate_delta=np.array([[0.]*3,[-.125,0.,.125],[0.]*3]),control_delta=np.array([[0.]*3,[-.125,0.,.125],[0.]*3]))
    history=[]
    for i in range(3):
        row=dict(query_index=i,distinct_levels=1 if i==2 else 3,constant=i==2,two_levels=False,active_forecast_count=0,
            max_absolute_candidate_delta=0. if i!=1 else .125,max_absolute_control_delta=0. if i!=1 else .125,
            max_absolute_candidate_control_difference=0.)
        for key in ('support','candidate','control'):
            row[key+'_min']=float(values[key][i].min());row[key+'_max']=float(values[key][i].max())
        history.append(row)
    audit=dict(events=3,symbols_per_event=3,checked_cells=9,rank_rows=3,constant_rows=1,two_level_rows=0,
        active_forecast_cells=0,new_model_fits=0,predictive_model_fits=0,ranking_eligibility_filtered=False,
        new_availability_filter=False,portfolio_returns_calculated=False,event_history=history)
    counts,integration=inspect_saved(values,audit)
    require(counts['constant_rows']==1 and integration['changed_cells']=={a:2 for a in ARMS[1:]},'Saved fixture summary');checks+=1
    bad=copy.deepcopy(values);bad['candidate_delta'][0,0]=.1;reject(lambda:inspect_saved(bad,audit))
    bad=copy.deepcopy(values);bad['control_delta'][2,0]=.1;reject(lambda:inspect_saved(bad,audit))
    bad=copy.deepcopy(values);bad['weight'][0,0]=0.;reject(lambda:inspect_saved(bad,audit))
    bad=copy.deepcopy(values);bad['support'][0,0]=np.nan;reject(lambda:inspect_saved(bad,audit))
    bad=copy.deepcopy(values);bad['candidate'][0,0]=-.5001;reject(lambda:inspect_saved(bad,audit))
    bad=copy.deepcopy(audit);bad['event_history'][0]['distinct_levels']=2;reject(lambda:inspect_saved(values,bad))
    bad=copy.deepcopy(audit);bad['active_forecast_cells']=1;reject(lambda:inspect_saved(values,bad))
    bad=copy.deepcopy(audit);bad['new_availability_filter']=True;reject(lambda:inspect_saved(values,bad))
    print(json.dumps(dict(passed=True,synthetic_checks=checks,actual_artifacts_read=False,new_features=False,new_orders=False,new_returns=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--completion-sha256');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        require(args.completion_sha256 is None,'Synthetic only self-test');self_test()
    else:
        require(bool(args.completion_sha256),'Explicit closed completion SHA required');review(args.completion_sha256)

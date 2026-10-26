"""Round201: frozen Local67 with noncalm half-core median versus mean deviation risk.

All actual features and portfolio returns require a prospectively bound PLAN.
Synthetic tests fit no predictive model; historical deviations are descriptive features, not portfolio returns.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import io
import math
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import ptr169_rebuild_evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_median_risk201'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_cmf_state200'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67', 'median_mad_core', 'mean_mad_control')
RAW = {NAMES[1]: 'candidate', NAMES[2]: 'control'}
TRAIN, BLEND, MAX_RANK, MAX_DELTA = 88, .5, .5, .26
FEATURE_PREFIXES = (0, 1, 53, 88, 122)
ESTIMATE_FIELDS = ('valid_price_observations', 'simple_return_observations', 'dispersion_estimates', 'ratio_observations', 'ranked_score_observations')
MODEL_FIT_ACCOUNTING = 'Zero new predictive model fits; current63 simple returns and two median-centered dispersion estimates per complete295-price cell, including zero-dispersion failures. Two ratios/ranks per commonly available cell. Main pass only.'
CONTEXT = ROOT / 'work/monthly_median_risk201_context.json'
CONTEXT_SHA = '4e937ed4474675b64069b410ef790697474ea2bfe03585e0f8f97d30afec9701'
SOURCE_REVIEW = ROOT / 'work/median201_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA = '4ca67a75465bb05f8b255272a185c436047c89fefe2193d939ec5ee9ae1e810d'
PRIOR_EXPECTED = {
 'PLAN.json': 'e0183ab68ac726d43e5886730179542e995e78241d97a1543c4a06dbfcf7bae0',
 'COMPLETION.json': '0e33b52c7d437d7778071343544aadf1d0b5b59b2987f02c46b4b201bd244c85',
 'results/COMPLETION.json': '6e857dc69a5d8bebf8b489e4f8eddc6d2d8723324da265845d0855733f147b9e',
 'RESEARCH_REVIEW.json': 'ac5e7652c9c83e442ff9a5d905780e78f44def46f3029092398efe5903841527',
}
FEATURE_FLAGS = ('passed','independent_scalar_rebuild','original_numerator_checked','median_center_checked','median_mean_deviations_checked','common_rank_domain_checked','original_pct_ranks_checked','full295_price_support_checked','official_session_windows_checked','monthend_queries_checked','exact_common_availability','audit_checked','no_future_or_outcome_inputs')
TEST_RECEIPT = ROOT / 'work/monthly_median_risk201_TESTS.json'
PREPARED_FILES = {'FEATURES.npz', 'TRANSFORMS_DELTAS.npz', 'FEATURE_AUDIT.json',
                  'ORDERS.json', 'META.json', 'SOURCE_PARITY.json', 'INDEPENDENT_PREPARE.json'}
sha, read, require, dump = ev.sha, ev.read, ev.require, ev.dump


def now():
    return datetime.now(timezone.utc).isoformat()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def local():
    return module('monthly_median_risk201_original', LOCAL / 'study.py')


def contract():
    require(sha(CONTEXT)==CONTEXT_SHA, 'Changed prospective context')
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,arms=list(NAMES),candidate=NAMES[1],control_nonpromotable=NAMES[2],period=['2016-08-17','2026-09-24'],training_months=88,later_months=34,blend=BLEND,max_absolute_rank=MAX_RANK,max_absolute_delta=MAX_DELTA,
        source_window=context['formation'],scoring='Noncalm half-core replacement delta=.5*(cw*T-C), original percentile rank of median versus mean absolute deviation normalized long momentum. Calm and firstpartial targets NaN; no second rank. Frozen adapter retains original support and one episode.',feature_prefix_source='Physically truncate adjusted close through prefix signal date, including empty prefix0.',verification=dict(independent_all_cells=True,feature_prefixes=list(FEATURE_PREFIXES),all123_order_prefixes=True),limitation=context['limitations'],historical_best_tracking='First88 selects; full-period best is descriptive among previously training-selected research candidates, not pristine OOS or observer promotion.',new_model_fits=0,new_predictive_model_fits=0,predictive_model_fits=0,model_fit_accounting=MODEL_FIT_ACCOUNTING,new_arms=2,registered_trials=3,execution_paths=9,no_observer_writes=True,original_pool_preserved=True,no_parameter_search=True)


def code_files():
    return [ROOT / 'work' / n for n in ('monthly_median_risk201_host.py',
        'test_monthly_median_risk201_host.py', 'monthly_median_risk201_kernel.py',
        'test_monthly_median_risk201_kernel.py', 'monthly_median_risk201_independent.py',
        'test_monthly_median_risk201_independent.py', 'monthly_median_risk201_context.json', 'monthly_nested_trend199_kernel.py', 'ptr169_rebuild_evaluate.py',
        'ptr169_results_independent.py', 'price_distribution171_independent.py', 'monthly_sector_recurrence192_independent.py')] + [ROOT/'outputs/monthly_core_timing70_research/core_adapter.py', ROOT/'outputs/monthly_core_timing70_research/test_core_adapter.py', ROOT/'outputs/monthly_path70_research/incumbent_adapter.py', ROOT/'outputs/monthly_daily_episodes/episodes.py']


def verify(pins):
    require(isinstance(pins, dict) and pins, 'Nonempty source pins required')
    for name, digest in pins.items():
        path = Path(name)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require('CloudStorage' not in path.parts and (path.is_relative_to(ROOT) or
                str(path).startswith('/Library/Developer/CommandLineTools/')), 'Unapproved source path')
        require(path.is_file() and sha(path) == digest, 'Changed or missing source: ' + str(path))


def bind(pins, path, digest=None):
    key, actual = str(path), sha(path)
    require(digest is None or actual == digest, 'Source differs from declared digest: ' + key)
    require(key not in pins or pins[key] == actual, 'Conflicting inherited source digest: ' + key)
    pins[key] = actual


def predecessor_binding(state):
    pins = {}
    for name, digest in PRIOR_EXPECTED.items(): bind(pins, PRIOR / name, digest)
    reference = state['completed'][-1]['completion']
    require(Path(reference['path']).resolve() == (PRIOR/'COMPLETION.json').resolve()
            and reference['sha256'] == PRIOR_EXPECTED['COMPLETION.json'], 'Predecessor central receipt')
    closed = read(PRIOR/'COMPLETION.json')
    require(closed.get('status') == 'COMPLETED_INDEPENDENTLY_VERIFIED' and
            closed.get('plan_sha256') == PRIOR_EXPECTED['PLAN.json'] and closed.get('protected_files') == 21444 and
            closed.get('observers_modified') is False, 'Predecessor closure incomplete')
    bind(pins, PRIOR/'results/COMPLETION.json', closed['result_completion_sha256'])
    result = read(PRIOR/'results/COMPLETION.json')
    require(result['status']=='COMPLETE' and result['plan_sha256']==PRIOR_EXPECTED['PLAN.json'] and
            result['paths']==9 and result['slices']==45, 'Predecessor results incomplete')
    for name, digest in read(CONTEXT)['evidence_sha256'].items(): bind(pins, ROOT/name, digest)
    return dict(directory=PRIOR.name, input_sha256=pins)


def price_source_binding(raw):
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Changed source review')
    review=read(SOURCE_REVIEW)
    require(review.get('passed') is True,'Adjusted-close source admission failed')
    require(review.get('context_sha256')==CONTEXT_SHA,'Source concept changed')
    require(review.get('raw_source_hashes')==raw,'Adjusted-close raw source lineage changed')
    for key in ('actual_features_computed','actual_orders_computed','actual_returns_computed'):
        require(review.get(key) is False or type(review.get(key)) is int and review[key]==0,'Source audit computed strategies: '+key)
    for key in ('actual_signal_returns_computed','actual_dispersion_computed','actual_ranks_computed','actual_deltas_computed'):
        require(review.get(key) is False,'Source review crossed feature boundary: '+key)
    pins={str(SOURCE_REVIEW):SOURCE_REVIEW_SHA}
    for name,digest in review['source_sha256'].items():
        bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    verify(pins)
    return dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,source_column='Original stored adjusted close;295 complete prices ending query, skip42 long numerator and current63 median/mean absolute deviations; no new adjustment or imputation',strict_pit=False)


def price_inputs(source,meta):
    require(len(meta['etfs'])==17 and len(set(meta['etfs']))==17,'Original17 ETF identities required')
    require(isinstance(source.get('adj'),pd.DataFrame),'Original stored adjusted close required')
    return {'adj':source['adj']}


def estimate_counts(check):
    keys=('source_available_by_event','available_by_event')
    series={k:check.get(k) for k in keys}
    require(all(isinstance(v,list) and len(v)==122 and all(type(n) is int and 0<=n<=49 for n in v) for v in series.values()),'Invalid event support/counts')
    require(all(a<=s for s,a in zip(series[keys[0]],series[keys[1]])),'Available exceeds complete source support')
    source,total=sum(series[keys[0]]),sum(series[keys[1]])
    require(type(check.get('source_available_cells')) is int and check['source_available_cells']==source,'Source coverage differs')
    require(type(check.get('available_cells')) is int and check['available_cells']==total,'Common coverage differs')
    expected=dict(valid_price_observations=295*source,simple_return_observations=63*source,
        dispersion_estimates=2*source,ratio_observations=2*total,ranked_score_observations=2*total)
    counts={key:check.get(key) for key in ESTIMATE_FIELDS}
    require(all(type(value) is int and value==expected[key] for key,value in counts.items()),'Median-risk accounting differs')
    return counts


def feature_prefix_panel(panel, meta, n):
    """Physically remove future source rows, with full original identities; do not inspect their values."""
    require(type(n) is int and 0 <= n <= len(meta['dates']), 'Invalid source prefix')
    require(set(panel) == {'adj'} and all(isinstance(v, pd.DataFrame) for v in panel.values()),
            'Original stored adjusted-close frame required')
    if not n:
        return {key: value.iloc[:0].copy() for key, value in panel.items()}
    day = pd.Timestamp(meta['dates'][n-1])
    return {key: value.loc[:day].copy() for key, value in panel.items()}

def state_check(state):
    require(len(state['completed']) == 198 and sum(r['trials'] for r in state['completed']) == 1111,
            'Expected198 completed entries /1111 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round200 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_MEDIAN_RISK201_' + tag + '_' + expected + '.json')
    if not backup.exists():
        with backup.open('xb') as stream:
            stream.write(before)
    require(sha(backup) == expected, 'Central backup mismatch')
    with tempfile.NamedTemporaryFile(dir=CENTRAL.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(after, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
        stream.flush(); os.fsync(stream.fileno())
    try:
        require(CENTRAL.read_bytes() == before, 'Concurrent central update before commit')
        os.replace(temporary, CENTRAL)
    finally:
        temporary.unlink(missing_ok=True)


def tests():
    names = ('test_monthly_median_risk201_host.py', 'test_monthly_median_risk201_kernel.py',
             'test_monthly_median_risk201_independent.py')
    modules = [module('test_median201_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    require(result.wasSuccessful(), 'Synthetic validation failed')
    receipt = TEST_RECEIPT
    require(not receipt.exists(), 'Test receipt is write-once')
    dump(receipt, dict(passed=True, synthetic_only=True, tests=result.testsRun,
        runtime=ev.runtime(), code_sha256={str(p): sha(p) for p in code_files()}))


def code_review_binding():
    path = ROOT / 'work/median201_CODE_REVIEW.json'
    review = read(path)
    expected = {str(p.relative_to(ROOT)): sha(p) for p in code_files()[:6]}
    require(review.get('passed') is True and review.get('unresolved_blockers') == [], 'Independent numerical code review incomplete')
    require(review.get('code_sha256') == expected, 'Numerical files changed after independent review')
    for key in ('actual_features_computed', 'actual_portfolio_returns_computed', 'production_files_modified', 'strategy_registered_by_reviewer'):
        require(review.get(key) is False, 'Independent review scope differs: '+key)
    pins = {str(path): sha(path)}
    for name, digest in review.get('input_sha256', {}).items():
        bind(pins, Path(name) if Path(name).is_absolute() else ROOT/name, digest)
    verify(pins)
    return dict(path=str(path), sha256=sha(path), input_sha256=pins)


def register():
    require(not OUT.exists(), 'Fresh experiment directory required')
    before = CENTRAL.read_bytes(); state = json.loads(before); state_check(state)
    require(not state['registered_queue'], 'Existing registered experiment')
    prior_binding = predecessor_binding(state)
    validation = TEST_RECEIPT; t = read(validation)
    require(t['passed'] is True and t['synthetic_only'] is True and t['runtime'] == ev.runtime(), 'Synthetic tests required')
    require(t['code_sha256'] == {str(p): sha(p) for p in code_files()}, 'Code changed since tests')
    pins = dict(read(PRIOR / 'PLAN.json')['input_sha256'])
    raw = read(LOCAL / 'PLAN.json')['raw_source_hashes']
    for name, digest in prior_binding['input_sha256'].items(): bind(pins, Path(name), digest)
    source_binding = price_source_binding(raw)
    numerical_review = code_review_binding()
    for name,digest in source_binding['input_sha256'].items(): bind(pins,Path(name),digest)
    for name,digest in numerical_review['input_sha256'].items(): bind(pins,Path(name),digest)
    calendar_sources = read(CALENDAR)['provenance']['source_hashes']
    for p in code_files() + [PRIOR / 'PLAN.json', PRIOR / 'COMPLETION.json', validation,
                            CALENDAR, PROTECTED, LOCAL / 'PLAN.json']:
        bind(pins, p)
    for mapping in (raw, calendar_sources):
        for name, digest in mapping.items(): bind(pins, Path(name), digest)
    for name in ('DATA.npz', 'META.json', 'local_linear_ORDERS.json', 'INCUMBENT_POLICY.json'):
        bind(pins, LOCAL / name)
    for name, digest in ev.ENGINE_HASHES.items(): bind(pins, ev.ENGINE_DIR / name, digest)
    verify(pins); protected = read(PROTECTED); verify(protected)
    require(len(protected) == 21444, 'Incomplete protected archive manifest')
    OUT.mkdir()
    with (OUT / 'STATE_BEFORE.json').open('xb') as stream: stream.write(before)
    plan = dict(schema='monthly_median_risk201_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, raw_source_hashes=raw, predecessor_binding=prior_binding,
        price_source_binding=source_binding,numerical_code_review=numerical_review,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_feature_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_dispersion_before_registration=0, actual_ratio_ranks_before_registration=0,
        actual_new_returns_before_registration=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_median_risk_features',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare fixed median/mean absolute-deviation noncalm half-core replacements; select only on first88 before later34 diagnostics.')
    commit_state(before, state, 'register')
    print(json.dumps(dict(status='REGISTERED', plan_sha256=sha(OUT / 'PLAN.json'))), flush=True)


def gate(expected):
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['contract'] == contract() and p['runtime'] == ev.runtime(), 'Frozen contract/runtime changed')
    verify(p['input_sha256'])
    require(p['price_source_binding'] == price_source_binding(p['raw_source_hashes']), 'Registered adjusted-close source binding changed')
    require(p['numerical_code_review']==code_review_binding(),'Registered independent code review changed')
    require(sha(OUT / 'STATE_BEFORE.json') == p['prior_central_sha256'], 'Changed pre-registration state')
    state = read(CENTRAL)
    require(state['registered_queue'] == [dict(directory=OUT.name, plan_sha256=expected)], 'Central registration mismatch')
    state_check(state)
    require(p.get('predecessor_binding') == predecessor_binding(state), 'Registered predecessor bindings changed')
    return p


def scale_and_delta(values, arrays, n):
    require(set(values)=={'candidate','control','raw_candidate','raw_control','available'},'Feature keys changed')
    available=np.asarray(values['available']);shape=available.shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0] and available.dtype==np.dtype(bool),'Invalid feature prefix/mask')
    finite=available[:n]; validated={}
    for key in ('candidate','control'):
        raw=np.asarray(values['raw_'+key]);ranked=np.asarray(values[key])
        require(raw.shape==ranked.shape==shape,'Ratio feature shape')
        x=np.asarray(raw[:n],float);r=np.asarray(ranked[:n],float)
        require(not np.isinf(x).any() and np.array_equal(np.isfinite(x),finite),'Common ratio availability')
        require(not np.isinf(r).any() and np.array_equal(np.isfinite(r),finite),'Common target availability')
        expected=np.full(r.shape,np.nan)
        for i in range(n):
            ids=np.flatnonzero(finite[i]);m=len(ids)
            for j in ids:
                less=np.count_nonzero(x[i,ids]<x[i,j]);equal=np.count_nonzero(x[i,ids]==x[i,j])
                expected[i,j]=(less+(equal+1)/2)/m-.5
        np.testing.assert_array_equal(r,expected)
        require((np.abs(r[finite])<=MAX_RANK).all(),'Original percentile bounds')
        validated[key]=r
    base=np.asarray(arrays['baseline']);core=np.asarray(arrays['core'])
    require(base.ndim==2 and base.shape[1]==shape[1] and base.shape[0]>=n and core.shape==base.shape,'Original core shape')
    c=np.asarray(core[:n],float)
    weights=np.asarray(arrays['core_weight'])
    if weights.shape==(len(base),):weights=weights[:,None]
    require(weights.shape in ((len(base),1),base.shape),'Original core weight shape')
    cw=np.broadcast_to(np.asarray(weights[:n],float),c.shape)
    calm=np.asarray(arrays['state_calm10'])
    require(calm.shape==(len(base),) and all(isinstance(x,(bool,np.bool_)) for x in calm[:n]),'Original calm state shape/type')
    calm=np.asarray(calm[:n],bool)
    expected_cw=np.broadcast_to(np.where(calm,.50,.52)[:,None],c.shape)
    np.testing.assert_array_equal(cw,expected_cw)
    require(np.isfinite(c).all() and (np.abs(c)<=cw*MAX_RANK).all(),'Original weighted core bounds')
    targets,deltas={},{}
    for arm,key in RAW.items():
        target=validated[key].copy();target[calm]=np.nan
        if n:target[0]=np.nan
        delta=np.zeros(target.shape,float);mask=np.isfinite(target)
        delta[mask]=BLEND*(cw[mask]*target[mask]-c[mask])
        require(np.max(np.abs(delta),initial=0.)<=MAX_DELTA,'Half-core budget exceeded')
        targets[arm],deltas[arm]=target,delta
    return targets,deltas


def core_adapter():
    return module('median201_frozen_core_adapter',ROOT/'outputs/monthly_core_timing70_research/core_adapter.py')


def decisions(study, arrays, meta, values, n):
    policy = read(LOCAL / 'INCUMBENT_POLICY.json')
    choices, scores = study.choices(study.configs()[4], arrays, meta, n, policy=policy)
    orders = {NAMES[0]: study.rp.make_orders(choices, scores, meta)}
    targets,deltas=scale_and_delta(values,arrays,n)
    adapter=core_adapter()
    for arm in NAMES[1:]:
        pre=study.modified_pre_score(study.configs()[4],arrays,n)
        full_target=np.full(arrays['baseline'].shape,np.nan)
        full_target[:n]=targets[arm]
        chosen,score=adapter.score_and_choose('incumbent66',BLEND,full_target,
            arrays,meta,n,pre_score=pre,effects=arrays['effects'],episode_policy=policy,
            boundary_selector=study.ep.select_boundary)
        orders[arm]=study.rp.make_orders(chosen,score,meta)
    return orders,targets,deltas


def require_feature_audit(check, values):
    for key in FEATURE_FLAGS:
        require(check.get(key) is True, 'Independent feature evidence missing '+key)
    expected = dict(events=122, symbols_per_event=49, checked_cells=122*49,
                    available_cells=int(values['available'].sum()),
                    unavailable_cells=int((~values['available']).sum()))
    require(values['available'].shape == (122, 49), 'Independent feature shape')
    for key, value in expected.items():
        require(type(check.get(key)) is int and check[key] == value,
                'Independent feature coverage: ' + key)
    require(check.get('computes_returns') is False, 'Feature audit must not compute portfolio returns')
    require(type(check.get('predictive_model_fits')) is int and check['predictive_model_fits'] == 0,
            'No predictive model fits allowed')
    estimate_counts(check)


def require_delta_audit(check, deltas):
    for key in ('passed','half_core_formula_checked','original_pct_ranks_preserved','first_partial_unchanged','calm_rows_unchanged','missing_preserves_core','core_weights_checked','no_second_rank','independent_delta_rebuild'):
        require(check.get(key) is True,'Independent delta evidence missing '+key)
    require(check.get('blend')==BLEND and check.get('max_absolute_delta')==MAX_DELTA,'Independent half-core bound differs')
    require(check.get('changed_cells')=={k:int(np.count_nonzero(v)) for k,v in deltas.items()},'Independent delta coverage')


def require_outcome_audit(check):
    for key in ('passed', 'independent_accounts', 'independent_of_production_model_and_engine'):
        require(check.get(key) is True, 'Independent outcome audit missing ' + key)
    require(check.get('checked_paths') == 9 and check.get('checked_slices') == 45 and
            check.get('terminal_liquidation') is False, 'Independent outcome coverage')
    require(check.get('accounting_source_sha256') == sha(ROOT / 'work/ptr169_results_independent.py'), 'Accounting source changed')
    require(set(check.get('paths', {})) == set(NAMES), 'Missing independent arms')
    for paths in check['paths'].values():
        require(set(paths) == {p[0] for p in ev.PATHS}, 'Missing independent paths')
        for path in paths.values():
            require(path.get('passed') is True and path.get('monthly_executions') == 122 and
                    set(path.get('slices', {})) == {'full122', 'first88', 'early53', 'recent35', 'later34'}, 'Incomplete path audit')


def prepare(expected):
    plan = gate(expected); require(not (OUT / 'prepared').exists(), 'Preparation is write-once')
    import monthly_median_risk201_kernel as kernel
    import monthly_median_risk201_independent as independent
    study = local(); meta, arrays, px, op = study.load()
    require(len(meta['dates']) == 122 and arrays['baseline'].shape == (122, 49), 'Original122x49 required')
    require(len(arrays) == 27, 'Original27 arrays required')
    require(all(len(s) == 49 for s in meta['symbols']), 'Original monthly candidate widths changed')
    before = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}
    source, info = study.arch.old.panel_load(meta['etfs'])
    raw_hashes = study.raw_source_hashes(info)
    require(raw_hashes == plan['raw_source_hashes'], 'Original raw source lineage changed')
    verify(raw_hashes)
    parity = study.verify_panel_parity(source, px, op)
    panel = price_inputs(source, meta)
    calendar = read(CALENDAR)['sessions']
    values, audit = kernel.build_features(panel, meta, calendar)
    check = independent.verify_features(panel, meta, calendar, values, audit)
    require_feature_audit(check, values)
    require(check['predictive_model_fits'] == audit['predictive_model_fits'] == 0,
            'Independent model accounting differs')
    counts = estimate_counts(check)
    require(counts == estimate_counts(audit), 'Independent median-risk accounting differs')
    orders, ranks, deltas = decisions(study, arrays, meta, values, 122)
    require(orders[NAMES[0]] == read(LOCAL / 'local_linear_ORDERS.json'), 'Original full order parity failed')
    rank_check = independent.verify_deltas(values, arrays, ranks, deltas, blend=BLEND)
    require_delta_audit(rank_check, deltas)
    for n in FEATURE_PREFIXES:
        prefix_panel = feature_prefix_panel(panel, meta, n)
        small, small_audit = kernel.build_features(prefix_panel, meta, calendar, limit=n)
        require(small_audit['event_history'] == audit['event_history'][:n], 'Truncated adjusted-close source prefix audit differs')
        for key in values: np.testing.assert_array_equal(small[key], values[key][:n])
    for n in range(123):
        oo, rr, dd = decisions(study, arrays, meta, values, n)
        require(all(oo[k] == orders[k][:n] for k in NAMES), 'Prefix order parity failed')
        for k in RAW:
            np.testing.assert_array_equal(rr[k], ranks[k][:n]); np.testing.assert_array_equal(dd[k], deltas[k][:n])
    require(before == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}, 'Original arrays modified')
    gate(expected)
    dest = OUT / 'prepared'; dest.mkdir()
    np.savez_compressed(dest / 'FEATURES.npz', **values)
    np.savez_compressed(dest / 'TRANSFORMS_DELTAS.npz', **{'transform_' + k: v for k, v in ranks.items()}, **{'delta_' + k: v for k, v in deltas.items()})
    dump(dest / 'FEATURE_AUDIT.json', audit); dump(dest / 'ORDERS.json', orders); dump(dest / 'META.json', meta)
    dump(dest / 'SOURCE_PARITY.json', dict(raw_source_hashes=raw_hashes, price_source_binding=plan['price_source_binding'], feature_panel='original stored adjusted close with common295-price support, skip42 long numerator and current63 median-centered deviations', execution_prices=parity))
    dump(dest / 'INDEPENDENT_PREPARE.json', dict(feature=check, delta=rank_check, prefixes=123, feature_prefixes=list(FEATURE_PREFIXES),
        feature_prefix_sources_truncated=True, original_core_metadata_and_support_preserved=True, single_original_episode=True, original_orders_exact=True, original_arrays_unchanged=True, original_array_count=27, new_model_fits=0,
        predictive_model_fits=0, new_predictive_model_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING, **counts))
    dump(dest / 'COMPLETION.json', dict(status='COMPLETE', plan_sha256=expected, created_utc=now(),
        artifacts_sha256={p.name: sha(p) for p in dest.iterdir()}, new_returns_computed=False, new_model_fits=0,
        predictive_model_fits=0, new_predictive_model_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING, original_array_count=len(arrays), **counts))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED', completion_sha256=sha(dest / 'COMPLETION.json'))), flush=True)


def verify_prepared_evidence(c, values, deltas, audit):
    require(set(c['artifacts_sha256']) == PREPARED_FILES, 'Prepared artifact set incomplete')
    require_feature_audit(audit['feature'], values)
    require_delta_audit(audit['delta'], deltas)
    require(audit.get('prefixes') == 123 and audit.get('feature_prefixes') == list(FEATURE_PREFIXES) and
            audit.get('feature_prefix_sources_truncated') is True and audit.get('original_orders_exact') is True and
            audit.get('original_core_metadata_and_support_preserved') is True and audit.get('single_original_episode') is True and audit.get('original_arrays_unchanged') is True and audit.get('original_array_count') == 27 and audit.get('new_model_fits') == 0 and
            audit.get('predictive_model_fits') == 0 and audit.get('model_fit_accounting') == MODEL_FIT_ACCOUNTING,
            'Prepared invariants not verified')
    for record in (audit, c):
        require(type(record.get('new_predictive_model_fits')) is int and record['new_predictive_model_fits'] == 0,
                'New predictive model fits not verified')
        expected_counts = estimate_counts(audit['feature'])
        require(all(type(record.get(key)) is int and record[key] == value
                    for key, value in expected_counts.items()), 'Descriptive estimate receipt accounting differs')
    require(c.get('new_returns_computed') is False and c.get('new_model_fits') == 0 and
            c.get('predictive_model_fits') == 0 and c.get('model_fit_accounting') == MODEL_FIT_ACCOUNTING and
            c.get('original_array_count') == 27, 'Prepared model/array accounting changed')


def prepared(expected):
    gate(expected); dest = OUT / 'prepared'; c = read(dest / 'COMPLETION.json')
    require(c['status'] == 'COMPLETE' and c['plan_sha256'] == expected, 'Preparation incomplete')
    require(set(c['artifacts_sha256']) == PREPARED_FILES, 'Prepared artifact set incomplete')
    for name, digest in c['artifacts_sha256'].items(): require(sha(dest / name) == digest, 'Changed prepared artifact ' + name)
    with np.load(dest / 'FEATURES.npz') as data: values = {k: data[k] for k in data.files}
    with np.load(dest / 'TRANSFORMS_DELTAS.npz') as data: deltas = {k: data['delta_' + k] for k in RAW}
    verify_prepared_evidence(c, values, deltas, read(dest / 'INDEPENDENT_PREPARE.json'))
    return read(dest / 'META.json'), read(dest / 'ORDERS.json')


def train_metrics(engine, eq, meta):
    answer = {}
    for name, lo, hi in (('training', 0, 87), ('early', 0, 52), ('recent', 53, 87)):
        part = eq.loc[meta['dates'][lo]:meta['ends'][hi]]
        answer[name] = engine.stats(part / part.iloc[0] * 100000.)
    return answer


def guard(candidate, reference, gain):
    return (candidate['training']['cagr'] >= reference['training']['cagr'] + gain - 1e-12 and
        candidate['training']['sharpe'] >= reference['training']['sharpe'] - .05 and
        candidate['training']['max_drawdown'] >= reference['training']['max_drawdown'] - .03 and
        all(candidate[e]['cagr'] >= reference[e]['cagr'] - .025 for e in ('early', 'recent')))


def select_first88(metrics, orders):
    changed = [i for i in range(TRAIN) if set(orders[NAMES[1]][i]['symbols']) != set(orders[NAMES[0]][i]['symbols'])]
    guards = dict(close_vs_incumbent=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[0]]['same_close_zero'], .005),
        open_vs_incumbent=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[0]]['next_open_cost'], 0.),
        close_vs_mean=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_mean=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
        changed_support=len(changed) >= 3 and any(i < 53 for i in changed) and any(i >= 53 for i in changed))
    changed_control = [i for i in range(TRAIN) if set(orders[NAMES[1]][i]['symbols']) != set(orders[NAMES[2]][i]['symbols'])]
    guards['changed_support_vs_control'] = len(changed_control)>=3 and any(i<53 for i in changed_control) and any(i>=53 for i in changed_control)
    return dict(primary=NAMES[1] if all(guards.values()) else NAMES[0], admitted=all(guards.values()),
        guards=guards, changed_months=changed, changed_months_vs_control=changed_control, first88_only=True, later34_used=False)


def select(expected):
    meta, orders = prepared(expected); require(not (OUT / 'selected').exists(), 'Selection is write-once')
    px, op = ev.prices(); engine = ev.engine(); metrics = {}
    for arm in NAMES:
        metrics[arm] = {}
        for path, mode, cost in (ev.PATHS[0], ev.PATHS[2]):
            _, eq, _, _ = engine.replay(px, op, orders[arm][:TRAIN], cost=cost, execution=mode,
                start=pd.Timestamp(meta['dates'][0]), end=pd.Timestamp(meta['ends'][87]))
            require(str(eq.index[-1].date()) == meta['ends'][87], 'Selection escaped first88 cutoff')
            metrics[arm][path] = train_metrics(engine, eq, meta)
    result = select_first88(metrics, orders); gate(expected)
    dest = OUT / 'selected'; dest.mkdir(); dump(dest / 'TRAIN_RESULTS.json', metrics)
    dump(dest / 'FROZEN.json', dict(created_utc=now(), plan_sha256=expected,
        prepared_sha256=sha(OUT / 'prepared/COMPLETION.json'), orders_sha256=sha(OUT / 'prepared/ORDERS.json'),
        training_sha256=sha(dest / 'TRAIN_RESULTS.json'), selection=result))
    print(json.dumps(result), flush=True)


def frozen_selection(expected, orders):
    frozen = read(OUT / 'selected/FROZEN.json')
    require(frozen['plan_sha256'] == expected and frozen['orders_sha256'] == sha(OUT / 'prepared/ORDERS.json') and
        frozen['training_sha256'] == sha(OUT / 'selected/TRAIN_RESULTS.json') and
        frozen['prepared_sha256'] == sha(OUT / 'prepared/COMPLETION.json'), 'Frozen selection dependencies changed')
    require(select_first88(read(OUT / 'selected/TRAIN_RESULTS.json'), orders) == frozen['selection'], 'Selection mismatch')
    return frozen


def verify_training_against_audit(training, audit):
    import monthly_median_risk201_independent as independent
    helper = independent.accounting()
    require(set(training) == set(NAMES), 'Training arms differ')
    for arm in NAMES:
        require(set(training[arm]) == {'same_close_zero', 'next_open_cost'}, 'Training paths differ')
        for path, slices in training[arm].items():
            require(set(slices) == {'training', 'early', 'recent'}, 'Training slices differ')
            for name, full_name in (('training', 'first88'), ('early', 'early53'), ('recent', 'recent35')):
                helper.check_metrics(audit['paths'][arm][path]['slices'][full_name], slices[name],
                                     'Frozen training/' + arm + '/' + path + '/' + name)


def diagnose(expected):
    meta, orders = prepared(expected); frozen = frozen_selection(expected, orders)
    require(not (OUT / 'results').exists(), 'Diagnostics are write-once')
    px, op = ev.prices(); engine = ev.engine(); dest = OUT / 'results'; dest.mkdir(); results, exports = {}, {}
    for arm in NAMES:
        results[arm], exports[arm] = {}, {}
        for path, mode, cost in ev.PATHS:
            result, eq, trades, holds, positions = ev.run_path(engine, px, op, orders[arm], meta, mode, cost)
            result['slices'] = dict(full122=result['metrics'], **ev.split_stats(engine, eq, meta))
            results[arm][path] = result
            data = dict(DAILY=[dict(date=str(d.date()), equity=float(v)) for d, v in eq.items()],
                TRADES=trades, HOLDINGS=holds, DAILY_POSITIONS=positions)
            exports[arm][path] = data
            for kind, value in data.items(): dump(dest / (arm + '_' + path + '_' + kind + '.json'), value)
    import monthly_median_risk201_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, median_risk=True, same_support_both_arms=True, observer_promotion=False))
    gate(expected)
    dump(dest / 'COMPLETION.json', dict(status='COMPLETE', created_utc=now(), plan_sha256=expected,
        frozen_sha256=sha(OUT / 'selected/FROZEN.json'), paths=9, slices=45,
        artifacts_sha256={p.name: sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='COMPLETE', primary=frozen['selection']['primary'],
        full_cagr={n: results[n]['same_close_zero']['metrics']['cagr'] for n in NAMES},
        completion_sha256=sha(dest / 'COMPLETION.json'))), flush=True)


def close(expected):
    gate(expected); completion = read(OUT / 'results/COMPLETION.json')
    require(completion['status'] == 'COMPLETE' and completion['plan_sha256'] == expected and
            completion['paths'] == 9 and completion['slices'] == 45, 'Incomplete diagnostics')
    for name, digest in completion['artifacts_sha256'].items(): require(sha(OUT / 'results' / name) == digest, 'Changed result ' + name)
    require_outcome_audit(read(OUT / 'results/INDEPENDENT_OUTCOME.json'))
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), read(OUT / 'results/INDEPENDENT_OUTCOME.json'))
    _, orders = prepared(expected); frozen = frozen_selection(expected, orders)
    require(completion['frozen_sha256'] == sha(OUT / 'selected/FROZEN.json'), 'Frozen selection receipt changed')
    results = read(OUT / 'results/RESULTS.json')
    require(results['selection'] == frozen['selection'], 'Closure selection mismatch')
    before = CENTRAL.read_bytes(); state = json.loads(before); state_check(state)
    protected = read(PROTECTED); verify(protected); require(len(protected) == 21444, 'Incomplete protection')
    require(not (OUT / 'COMPLETION.json').exists(), 'Closure is write-once')
    best = state['best_training_selected_candidate']; pick = results['selection']['primary']; arm = results['arms'][pick]
    metrics = arm['same_close_zero']['metrics']
    if results['selection']['admitted'] and metrics['cagr'] > best['cagr']:
        best = dict(directory=OUT.name, spec=dict(arm=pick), cagr=metrics['cagr'], sharpe=metrics['sharpe'],
            max_drawdown=metrics['max_drawdown'], next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],
            selected_on_first88=True, observer_promotion=False, limitations=contract()['limitation'])
    receipt = dict(status='COMPLETED_INDEPENDENTLY_VERIFIED', created_utc=now(), plan_sha256=expected,
        result_completion_sha256=sha(OUT / 'results/COMPLETION.json'), new_trials=3, new_model_fits=0,
        predictive_model_fits=0, new_predictive_model_fits=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        original_array_count=27, protected_files=len(protected), best_training_selected_candidate=best,
        target_reached=best['cagr'] >= .7, observers_modified=False,
        **{key: read(OUT / 'prepared/COMPLETION.json')[key] for key in ESTIMATE_FIELDS})
    dump(OUT / 'COMPLETION.json', receipt)
    state['completed'].append(dict(directory=OUT.name, trials=3, completed_utc=now(), primary=pick,
        primary_cagr=metrics['cagr'], primary_sharpe=metrics['sharpe'], new_admitted=int(results['selection']['admitted']),
        independent_execution_paths=9, protected_files=len(protected), target_reached=receipt['target_reached'],
        completion=dict(path=str(OUT / 'COMPLETION.json'), sha256=sha(OUT / 'COMPLETION.json'))))
    state.update(best_training_selected_candidate=best, latest_training_selected_candidate=dict(directory=OUT.name,
        arm=pick, selection=results['selection'], full_period_cagr=metrics['cagr'], full_period_sharpe=metrics['sharpe'], selected_on_first88=True),
        best_tracking_note=contract()['historical_best_tracking'], registered_queue=[], current_status='completed_independently_verified',
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Review median versus mean deviation core normalization and source limitations before another distinct hypothesis; preserve frozen observers.')
    require(len(state['completed']) == 199 and sum(r['trials'] for r in state['completed']) == 1114, 'Wrong final counters')
    commit_state(before, state, 'close'); print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['test', 'register', 'prepare', 'select', 'diagnose', 'close'])
    parser.add_argument('--plan-sha256'); args = parser.parse_args()
    if args.phase == 'test': tests()
    elif args.phase == 'register': register()
    else:
        require(bool(args.plan_sha256), 'An explicit registered PLAN SHA is required')
        globals()[args.phase](args.plan_sha256)

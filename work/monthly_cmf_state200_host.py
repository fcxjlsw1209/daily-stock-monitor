"""Round200: original CMF contribution with a soft empirical market-state gate.

All actual features and portfolio returns require a prospectively bound PLAN.
Synthetic tests fit no predictive model. Original core, support and episode inputs stay fixed.
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
OUT = ROOT / 'outputs/monthly_cmf_state200'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_nested_trend199'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
QUALITY = ROOT / 'outputs/monthly_quality_research/FEATURES.npz'
STABILITY = ROOT / 'outputs/monthly_stability_research/FEATURES.npz'
NAMES = ('incumbent_local67', 'cmf_empirical_state', 'cmf_half_sign_control')
RAW = {NAMES[1]: 'gate_candidate', NAMES[2]: 'gate_control'}
TRAIN, MAX_RANK, MAX_DELTA = 88, .5, .0375
PANEL_KEYS = {'adj', 'close', 'high', 'low', 'volume'}
VALUE_KEYS = {'cmf_rank','state','gate_candidate','gate_control','coefficient','calm','sigma','median','sigma_window'}
FEATURE_PREFIXES = (0, 1, 53, 88, 122)
ESTIMATE_FIELDS = ('finite_cmf_cells','neutral_cmf_cells','spy_price_observations','volatility_rank_observations')
MODEL_FIT_ACCOUNTING = 'Zero new predictive model fits; canonical SPY sample volatility, empirical state gate and original CMF ranks only. Main preparation pass excludes prefix and independent repeats.'
CONTEXT = ROOT / 'work/monthly_cmf_state200_context.json'
CONTEXT_SHA = 'bccab811afc6d2a1fa27d26e205b30b9d0b75df8ca83b69e03734177217d80a0'
SOURCE_REVIEW = ROOT / 'work/cmf200_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA = '9742ef91bba7617a95fc738911d167e6abb82b94d95a73f59455253bca815ee7'
PRIOR_EXPECTED = {
 'PLAN.json': '21e2b06854c8929aa8e2199a716ee20e6b45e21fd7a430d73d858004967f17ef',
 'COMPLETION.json': '4523b22685f58b092287f0de4aae0694316c41e61846193c770b3ddb692bbf96',
 'results/COMPLETION.json': '011a5b0c1f2ba6586afcc229f841de7222538024a1ba7e570bde14d0d0e27089',
 'RESEARCH_REVIEW.json': 'a435e878626330f63a654b59a6aacbbc0f07819133d8068be2eb8935d4784cce',
}
FEATURE_FLAGS = ('passed','independent_scalar_rebuild','canonical_sigma_checked','empirical_gate_checked','median_tie_neutral_checked','original_cmf_rank_checked','regime_coefficient_checked','strict315_official_spy_window_checked','monthend_queries_checked','audit_checked','no_future_or_outcome_inputs')
DELTA_FLAGS = ('passed','original_calm_and_core_weights_checked','cmf_state_delta_formula_checked','original_state_preserved','first_partial_unchanged','missing_cmf_neutral','no_new_rank','bounded_correction_checked','independent_delta_rebuild','effective_support_scale_checked')
TEST_RECEIPT = ROOT / 'work/monthly_cmf_state200_TESTS.json'
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
    return module('monthly_cmf_state200_original', LOCAL / 'study.py')


def contract():
    require(sha(CONTEXT)==CONTEXT_SHA, 'Changed prospective context')
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,arms=list(NAMES),candidate=NAMES[1],control_nonpromotable=NAMES[2],period=['2016-08-17','2026-09-24'],training_months=88,later_months=34,max_absolute_rank=MAX_RANK,max_absolute_delta=MAX_DELTA,
        scoring=context['integration'],feature_prefix_source='Physically truncate every adjusted/OHLCV frame through prefix signal date, including empty prefix0.',verification=dict(independent_all_cells=True,feature_prefixes=list(FEATURE_PREFIXES),all123_order_prefixes=True),limitation=context['limitations'],historical_best_tracking='First88 selects; full-period best is descriptive among previously training-selected research candidates, not pristine OOS or observer promotion.',new_model_fits=0,new_predictive_model_fits=0,predictive_model_fits=0,model_fit_accounting=MODEL_FIT_ACCOUNTING,new_arms=2,registered_trials=3,execution_paths=9,no_observer_writes=True,original_pool_preserved=True,no_parameter_search=True)


def code_files():
    return [ROOT / 'work' / n for n in ('monthly_cmf_state200_host.py',
        'test_monthly_cmf_state200_host.py', 'monthly_cmf_state200_kernel.py',
        'test_monthly_cmf_state200_kernel.py', 'monthly_cmf_state200_independent.py',
        'test_monthly_cmf_state200_independent.py', 'monthly_cmf_state200_context.json', 'monthly_nested_trend199_kernel.py', 'ptr169_rebuild_evaluate.py',
        'ptr169_results_independent.py', 'price_distribution171_independent.py', 'monthly_sector_recurrence192_independent.py')] + [ROOT/'outputs/monthly_path70_research/incumbent_adapter.py', ROOT/'outputs/monthly_daily_episodes/episodes.py']


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
    require(review.get('passed') is True,'CMF/state source admission failed')
    require(review.get('context_sha256')==CONTEXT_SHA,'Source concept changed')
    require(review.get('raw_source_hashes')==raw,'Original OHLCV source lineage changed')
    for key in ('actual_features_computed','actual_orders_computed','actual_returns_computed'):
        require(review.get(key) is False or type(review.get(key)) is int and review[key]==0,'Source audit computed strategies: '+key)
    pins={str(SOURCE_REVIEW):SOURCE_REVIEW_SHA}
    for name,digest in review['source_sha256'].items():
        bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    verify(pins)
    return dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,source_column='Original adjusted SPY and stored close/high/low/volume; no new adjustment or imputation',strict_pit=False)


def price_inputs(source,meta):
    require(len(meta['etfs'])==17 and len(set(meta['etfs']))==17,'Original17 ETF identities required')
    require(all(isinstance(source.get(key),pd.DataFrame) for key in PANEL_KEYS),'Original adjusted/OHLCV frames required')
    require('SPY' in source['adj'],'Original adjusted SPY required')
    return {key:source[key] for key in sorted(PANEL_KEYS)}


def estimate_counts(check):
    finite,neutral=check.get('finite_cmf_cells'),check.get('neutral_cmf_cells')
    require(type(finite) is int and type(neutral) is int and finite>=0 and neutral>=0 and finite+neutral==122*49,'CMF source count mismatch')
    expected=dict(finite_cmf_cells=finite,neutral_cmf_cells=neutral,spy_price_observations=122*315,volatility_rank_observations=122*252)
    counts={key:check.get(key) for key in ESTIMATE_FIELDS}
    require(all(type(value) is int and value==expected[key] for key,value in counts.items()),'CMF/state descriptive accounting differs')
    return counts


def feature_prefix_panel(panel, meta, n):
    """Physically remove future source rows, with full original identities; do not inspect their values."""
    require(type(n) is int and 0 <= n <= len(meta['dates']), 'Invalid source prefix')
    require(set(panel) == PANEL_KEYS and all(isinstance(v, pd.DataFrame) for v in panel.values()),
            'Original adjusted/OHLCV frames required')
    if not n:
        return {key: value.iloc[:0].copy() for key, value in panel.items()}
    day = pd.Timestamp(meta['dates'][n-1])
    return {key: value.loc[:day].copy() for key, value in panel.items()}

def state_check(state):
    require(len(state['completed']) == 197 and sum(r['trials'] for r in state['completed']) == 1108,
            'Expected197 completed entries /1108 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round199 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_CMF_STATE200_' + tag + '_' + expected + '.json')
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
    names = ('test_monthly_cmf_state200_host.py', 'test_monthly_cmf_state200_kernel.py',
             'test_monthly_cmf_state200_independent.py')
    modules = [module('test_cmf200_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
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
    path = ROOT / 'work/cmf200_CODE_REVIEW.json'
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
                            CALENDAR, PROTECTED, LOCAL / 'PLAN.json', QUALITY, STABILITY]:
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
    plan = dict(schema='monthly_cmf_state200_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, raw_source_hashes=raw, predecessor_binding=prior_binding,
        price_source_binding=source_binding,numerical_code_review=numerical_review,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_feature_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_cmf_state_gates_before_registration=0,
        actual_new_returns_before_registration=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_cmf_state_features',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare fixed empirical CMF state softening and half-sign control; select only on first88 before later34 diagnostics.')
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
    """Validate only the requested numeric prefix, then form the fixed pre-score correction."""
    require(set(values)==VALUE_KEYS,'Feature keys changed')
    q=np.asarray(values['cmf_rank']);shape=q.shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Invalid feature prefix')
    q=np.asarray(q[:n],float)
    require(np.isfinite(q).all() and (np.abs(q)<=MAX_RANK).all(),'Original finite CMF ranks required')
    vectors={}
    for key in VALUE_KEYS-{'cmf_rank','sigma_window'}:
        value=np.asarray(values[key]);require(value.shape==(shape[0],),'Invalid state vector shape: '+key)
        if key=='calm':
            require(all(isinstance(x,(bool,np.bool_)) for x in value[:n]),'Boolean calm states required')
            vectors[key]=np.asarray(value[:n],bool)
        else:
            vectors[key]=np.asarray(value[:n],float)
            require(np.isfinite(vectors[key]).all(),'Nonfinite state input: '+key)
    window=np.asarray(values['sigma_window']);require(window.shape==(shape[0],252),'Exact252 sigma slots required')
    window=np.asarray(window[:n],float)
    require(np.isfinite(window).all() and (window>=0).all(),'Finite nonnegative market risk required')
    np.testing.assert_array_equal(vectors['sigma'],window[:,-1])
    median=np.median(window,axis=1)
    np.testing.assert_array_equal(vectors['median'],median)
    state=np.where(vectors['sigma']>median,1.,-1.)
    np.testing.assert_array_equal(vectors['state'],state)
    calm=vectors['sigma']*np.sqrt(252)<.10
    np.testing.assert_array_equal(vectors['calm'],calm)
    gate=np.array([0. if sig==med else (2*int(np.count_nonzero(row<sig))+int(np.count_nonzero(row==sig)))/252-1
        for row,sig,med in zip(window,vectors['sigma'],median)],float)
    np.testing.assert_array_equal(vectors['gate_candidate'],gate)
    np.testing.assert_array_equal(vectors['gate_control'],.5*state)
    coefficient=np.where(calm,.075,.075*(1-np.where(state>0,.10,0.)/.48))
    np.testing.assert_array_equal(vectors['coefficient'],coefficient)
    base=np.asarray(arrays['baseline']);core=np.asarray(arrays['core'])
    require(base.ndim==2 and core.shape==base.shape and base.shape[0]>=n and base.shape[1]==shape[1],'Original array shape')
    require(np.isfinite(np.asarray(base[:n],float)).all() and np.isfinite(np.asarray(core[:n],float)).all(),'Original finite core/baseline required')
    old_calm=np.asarray(arrays['state_calm10']);require(old_calm.shape==(len(base),),'Original calm shape')
    require(all(isinstance(x,(bool,np.bool_)) for x in old_calm[:n]),'Original calm type')
    np.testing.assert_array_equal(np.asarray(old_calm[:n],bool),calm)
    weights=np.asarray(arrays['core_weight'])
    if weights.shape==(len(base),):weights=weights[:,None]
    require(weights.shape in ((len(base),1),base.shape),'Original core weight shape')
    cw=np.broadcast_to(np.asarray(weights[:n],float),(n,shape[1]))
    np.testing.assert_array_equal(cw,np.broadcast_to(np.where(calm,.50,.52)[:,None],cw.shape))
    scale=support_scale(arrays,n)
    transforms,deltas={},{}
    for arm,key in RAW.items():
        g=vectors[key]
        require((np.abs(g)<=1.).all() and (g*state>=0.).all(),'Gate must preserve original sign or neutral')
        delta=(coefficient[:,None]*q*(g-state)[:,None])*scale
        if n:delta[0]=0.
        require(np.isfinite(delta).all() and np.max(np.abs(delta),initial=0.)<=MAX_DELTA,'CMF correction exceeds nominal bound')
        transforms[arm],deltas[arm]=g.copy(),delta
    return transforms,deltas


def support_scale(arrays,n):
    """Exact existing pre-score auxiliary budgets; no new state or availability filter."""
    shape=np.asarray(arrays['baseline']).shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Invalid support prefix')
    states={}
    for key in ('state_calm10','state_weak_narrow'):
        value=np.asarray(arrays[key]);require(value.shape==(shape[0],),'State vector shape: '+key)
        require(all(isinstance(x,(bool,np.bool_)) for x in value[:n]),'Boolean original state required: '+key)
        states[key]=np.asarray(value[:n],bool)
    masks={}
    for key in ('extension_available','forecast_available','eligible'):
        value=np.asarray(arrays[key]);require(value.shape==shape,'Original mask shape: '+key)
        require(all(isinstance(x,(bool,np.bool_)) for x in value[:n].flat),'Boolean original mask required: '+key)
        masks[key]=np.asarray(value[:n],bool)
    rank=np.asarray(arrays['rank_log_excess']);require(rank.shape==shape,'Original scope rank shape')
    rank=np.asarray(rank[:n],float)
    calm,weak=states['state_calm10'],states['state_weak_narrow']
    active=(calm & ~weak)[:,None]&masks['extension_available']&masks['forecast_available']&masks['eligible']&np.isfinite(rank)
    weights=np.asarray(arrays['core_weight'])
    if weights.shape==(shape[0],):weights=weights[:,None]
    require(weights.shape in ((shape[0],1),shape),'Original core weight shape')
    cw=np.broadcast_to(np.asarray(weights[:n],float),(n,shape[1]))
    np.testing.assert_array_equal(cw,np.broadcast_to(np.where(calm,.5,.52)[:,None],cw.shape))
    result=1-(.025*calm[:,None]+.05*weak[:,None]+.01*active)/(1-cw)
    require(np.isfinite(result).all() and (result>=.85).all() and (result<=1).all(),'Original support scale range')
    return result


def original_state_parity(values,highvol,flow):
    """Cache flow tolerance only covers its subtraction-derived isolation, never a decision."""
    highvol=np.asarray(highvol);flow=np.asarray(flow)
    require(highvol.dtype.kind=='b' and highvol.shape==values['state'].shape,'Original high-vol state shape/type')
    require(flow.shape==values['cmf_rank'].shape and np.isfinite(flow).all(),'Original isolated flow shape/finite')
    np.testing.assert_array_equal(highvol,values['state']>0)
    np.testing.assert_allclose(flow,values['cmf_rank']*values['state'][:,None],rtol=0.,atol=1e-12)
    return dict(highvol_exact=True,flow_isolation_atol=1e-12,flow_isolation_rtol=0.,
        tolerance_scope='Subtraction-derived original flow cache only; no rank, state, gate or order tolerance')


def decisions(study, arrays, meta, values, n):
    policy = read(LOCAL / 'INCUMBENT_POLICY.json')
    choices, scores = study.choices(study.configs()[4], arrays, meta, n, policy=policy)
    orders = {NAMES[0]: study.rp.make_orders(choices, scores, meta)}
    transforms,deltas=scale_and_delta(values,arrays,n)
    for arm in NAMES[1:]:
        pre=study.modified_pre_score(study.configs()[4],arrays,n)
        require(np.asarray(pre).shape==np.asarray(arrays['baseline']).shape,'Original full pre-score shape required')
        pre=np.asarray(pre).copy();prefix=pre[:n].copy();changed=deltas[arm]!=0.
        prefix[changed]+=deltas[arm][changed];pre[:n]=prefix
        chosen,score=study.adapter.score_and_choose('incumbent66',0.,np.full(arrays['baseline'].shape,np.nan),
            arrays,meta,n,pre_score=pre,effects=arrays['effects'],episode_policy=policy,
            boundary_selector=study.ep.select_boundary)
        orders[arm]=study.rp.make_orders(chosen,score,meta)
    return orders,transforms,deltas


def require_feature_audit(check, values):
    for key in FEATURE_FLAGS:
        require(check.get(key) is True, 'Independent feature evidence missing '+key)
    expected = dict(events=122, symbols_per_event=49, checked_cells=122*49)
    require(set(values)==VALUE_KEYS and values['cmf_rank'].shape == (122, 49), 'Independent feature shape/schema')
    for key, value in expected.items():
        require(type(check.get(key)) is int and check[key] == value,
                'Independent feature coverage: ' + key)
    require(check.get('computes_returns') is False, 'Feature audit must not compute portfolio returns')
    require(type(check.get('predictive_model_fits')) is int and check['predictive_model_fits'] == 0,
            'No predictive model fits allowed')
    require(type(check.get('new_model_fits')) is int and check['new_model_fits']==0,'No new model fits allowed')
    estimate_counts(check)


def require_delta_audit(check, deltas):
    for key in DELTA_FLAGS:
        require(check.get(key) is True,'Independent delta evidence missing '+key)
    require(check.get('max_absolute_delta')==MAX_DELTA,'Independent CMF bound differs')
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
    import monthly_cmf_state200_kernel as kernel
    import monthly_cmf_state200_independent as independent
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
    with np.load(QUALITY) as cached: highvol=cached['state_highvol']
    with np.load(STABILITY) as cached: flow=cached['flow']
    original_parity=original_state_parity(values,highvol,flow)
    check = independent.verify_features(panel, meta, calendar, values, audit)
    require_feature_audit(check, values)
    require(check['predictive_model_fits'] == audit['predictive_model_fits'] == 0,
            'Independent model accounting differs')
    counts = estimate_counts(check)
    require(counts == estimate_counts(audit), 'Independent CMF and state accounting differs')
    orders, ranks, deltas = decisions(study, arrays, meta, values, 122)
    require(orders[NAMES[0]] == read(LOCAL / 'local_linear_ORDERS.json'), 'Original full order parity failed')
    rank_check = independent.verify_deltas(values, arrays, deltas)
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
    np.savez_compressed(dest / 'TRANSFORMS_DELTAS.npz', support_scale=support_scale(arrays,122),
        **{'transform_' + k: v for k, v in ranks.items()}, **{'delta_' + k: v for k, v in deltas.items()})
    dump(dest / 'FEATURE_AUDIT.json', audit); dump(dest / 'ORDERS.json', orders); dump(dest / 'META.json', meta)
    dump(dest / 'SOURCE_PARITY.json', dict(raw_source_hashes=raw_hashes, price_source_binding=plan['price_source_binding'], feature_panel='original stored OHLCV CMF63 and canonical adjusted SPY63 risk, trailing252 risk-window empirical gate', original_regime_and_flow=original_parity, execution_prices=parity))
    dump(dest / 'INDEPENDENT_PREPARE.json', dict(feature=check, delta=rank_check, prefixes=123, feature_prefixes=list(FEATURE_PREFIXES),
        feature_prefix_sources_truncated=True, original_regime_and_flow_checked=True, original_core_metadata_and_support_preserved=True, single_original_episode=True, original_orders_exact=True, original_arrays_unchanged=True, original_array_count=27, new_model_fits=0,
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
            audit.get('feature_prefix_sources_truncated') is True and audit.get('original_orders_exact') is True and audit.get('original_regime_and_flow_checked') is True and
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
        close_vs_half_sign=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_half_sign=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
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
    import monthly_cmf_state200_independent as independent
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
    import monthly_cmf_state200_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, cmf_state_softening=True, same_support_both_arms=True, observer_promotion=False))
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
        next_step='Review empirical CMF state softening versus fixed half-sign control and source limitations; preserve frozen observers.')
    require(len(state['completed']) == 198 and sum(r['trials'] for r in state['completed']) == 1111, 'Wrong final counters')
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

"""Round204: full pre-episode Local67 auxiliary rank versus affine calibration.

All actual features and portfolio returns require a prospectively bound PLAN.
Synthetic tests fit no predictive model; only existing cached support is transformed.
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
OUT = ROOT / 'outputs/monthly_aux_rank204'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_volume_pair203'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67', 'aux_rank_full', 'aux_affine_control')
RAW = {NAMES[1]: 'candidate', NAMES[2]: 'control'}
TRAIN, MAX_DELTA = 88, .5
FEATURE_PREFIXES = tuple(range(123))
VALUE_KEYS = ('support','weight','candidate','control','candidate_delta','control_delta')
ESTIMATE_FIELDS = ('rank_rows','constant_rows','two_level_rows','active_forecast_cells')
MODEL_FIT_ACCOUNTING = 'Zero new predictive fits. Cached support remapping only; main rank rows and constant/two-level/active forecast counts exclude prefix and independent repeats. No fresh raw-price reconstruction claimed.'
CONTEXT = ROOT / 'work/monthly_aux_rank204_context.json'
CONTEXT_SHA = '602df5d3ec9ae914f147fee5c66167f6c554c4396c4a11d5dd1c24f78e0f8233'
COVERAGE = ROOT / 'work/after203_aux_rank_coverage.json'
COVERAGE_SHA = 'a8f2388530dcfb0834ea0e3d9e98cce1717c6b6021b7e8ec44cf8c7cb7259b96'
PRIOR_EXPECTED = {
 'PLAN.json': '9edd7acc84995b8f37038de3f4a978fd1f7c6ad57d49e26f79cc2278b1987392',
 'COMPLETION.json': 'f19420fe5a77e72efba838b74ee5c40a4316c186984a213aaa19b71bee91f77d',
 'results/COMPLETION.json': 'c8d859d125069a17acb02d4141484022b81b4d14647028a949f3606d20bac352',
 'RESEARCH_REVIEW.json': 'b03806cda7124b95f46d1984d372f8b88dad439c0513a7f03c0a811336f6a9c1',
}
FEATURE_FLAGS = ('passed','independent_scalar_rebuild','full_auxiliary_support_checked','core_weights_checked','full_candidate_universe','average_tie_ranks_checked','affine_span_control_checked','two_level_equivalence_checked','constant_rows_unchanged','first_partial_unchanged','effective_support_weight_checked','no_eligibility_rank_filter','no_new_availability_filter','no_future_or_outcome_inputs','audit_checked')
DELTA_FLAGS = ('passed','independent_delta_rebuild','effective_support_weight_checked','full_auxiliary_replacement_checked','first_partial_unchanged','constant_rows_unchanged','core_weights_checked','no_eligibility_delta_filter','source_values_checked')
TEST_RECEIPT = ROOT / 'work/monthly_aux_rank204_TESTS.json'
PREPARED_FILES = {'FEATURES.npz', 'TRANSFORMS_DELTAS.npz', 'FEATURE_AUDIT.json',
                  'ORDERS.json', 'META.json', 'SOURCE_PARITY.json', 'INDEPENDENT_PREPARE.json'}
RESULT_FILES = {arm+'_'+path[0]+'_'+kind+'.json' for arm in NAMES for path in ev.PATHS
                for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')} | {'RESULTS.json','INDEPENDENT_OUTCOME.json'}
sha, read, require, dump = ev.sha, ev.read, ev.require, ev.dump


def now():
    return datetime.now(timezone.utc).isoformat()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def local():
    return module('monthly_aux_rank204_original', LOCAL / 'study.py')


def contract():
    require(sha(CONTEXT)==CONTEXT_SHA, 'Changed prospective context')
    require(sha(COVERAGE)==COVERAGE_SHA, 'Changed coverage evidence')
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,
        arms=list(NAMES),candidate=NAMES[1],control_nonpromotable=NAMES[2],
        period=['2016-08-17','2026-09-24'],training_months=88,later_months=34,max_absolute_delta=MAX_DELTA,
        scoring='W*(target-S) changes only effective pre-episode support. Original Local67 core and model corrections retained; first and constant rows delta0; one episode with original support; no rerank.',
        feature_prefix_source='Physically truncate every original cached array for all123 prefixes, including empty/full. No new raw-price feature.',
        verification=dict(independent_all_cells=True,feature_prefixes=list(FEATURE_PREFIXES),all123_order_prefixes=True),
        limitation=context['limitations'],historical_best_tracking='First88 selects; full-period best is descriptive among previously training-selected research candidates, not pristine OOS or observer promotion.',
        new_model_fits=0,new_predictive_model_fits=0,predictive_model_fits=0,model_fit_accounting=MODEL_FIT_ACCOUNTING,
        new_arms=2,registered_trials=3,execution_paths=9,no_observer_writes=True,original_pool_preserved=True,no_parameter_search=True)

def code_files():
    return [ROOT/'work'/n for n in ('monthly_aux_rank204_host.py','test_monthly_aux_rank204_host.py',
        'monthly_aux_rank204_kernel.py','test_monthly_aux_rank204_kernel.py',
        'monthly_aux_rank204_independent.py','test_monthly_aux_rank204_independent.py',
        'monthly_aux_rank204_context.json','after203_aux_rank_coverage.json','ptr169_rebuild_evaluate.py',
        'ptr169_results_independent.py','price_distribution171_independent.py','monthly_sector_recurrence192_independent.py')] + [
        ROOT/'outputs/monthly_path70_research/incumbent_adapter.py',ROOT/'outputs/monthly_daily_episodes/episodes.py']

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


def cache_source_binding():
    """Verify registered cache closure by metadata/digests, without loading arrays."""
    pins={}
    for name in ('PLAN.json','DESIGN.json','GRID.json','DESIGN.sha256','INPUT_HASHES.json',
                 'CACHE_HASHES.json','META.json','DATA.npz','INCUMBENT_POLICY.json','local_linear_ORDERS.json','study.py'):
        bind(pins,LOCAL/name)
    plan=read(LOCAL/'PLAN.json')
    for group in ('hashes','source_hashes','raw_source_hashes'):
        for name,digest in plan[group].items():
            bind(pins,Path(name) if Path(name).is_absolute() else LOCAL/name,digest)
    for name,digest in read(LOCAL/'INPUT_HASHES.json').items():
        bind(pins,Path(name) if Path(name).is_absolute() else LOCAL/name,digest)
    for name,digest in read(LOCAL/'CACHE_HASHES.json').items():
        bind(pins,LOCAL/name,digest)
    meta=read(LOCAL/'META.json')
    require(len(meta['dates'])==122 and len(meta['symbols'])==122 and all(len(x)==49 for x in meta['symbols']),
            'Original monthly identities required')
    require(meta['dates'][0]=='2016-08-17' and meta['dates'][-1]=='2026-08-31' and meta['ends'][-1]=='2026-09-24',
            'Original period changed')
    require(plan['calendar']==read(CALENDAR)['provenance'],'Original calendar binding differs')
    verify(pins)
    return dict(directory=LOCAL.name,input_sha256=pins,events=122,symbols_per_event=49,original_array_count=27,
        cache_loader='original study.load: registered plus original source/cache/identity checks',
        fresh_raw_price_reconstruction=False,new_source_features=False)


def cache_prefix(arrays,n):
    require(isinstance(arrays,dict) and 'baseline' in arrays,'Original cache mapping required')
    shape=np.asarray(arrays['baseline']).shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Cache prefix shape')
    result={}
    for key,value in arrays.items():
        raw=np.asarray(value)
        require(raw.ndim>=1 and raw.shape[0]==shape[0],'Original cache event axis differs: '+key)
        result[key]=raw[:n].copy()
    return result


def estimate_counts(a):
    n=a.get('events');width=a.get('symbols_per_event')
    require(type(n) is int and n>=0 and type(width) is int and width>0,'Counter axes')
    require(type(a.get('checked_cells')) is int and a['checked_cells']==n*width,'Counter cells')
    counts={key:a.get(key) for key in ESTIMATE_FIELDS}
    require(all(type(v) is int and v>=0 for v in counts.values()),'Integer transformation counters required')
    require(counts['rank_rows']==n and counts['constant_rows']+counts['two_level_rows']<=n
        and counts['active_forecast_cells']<=n*width,'Transformation count bounds')
    return counts


def state_check(state):
    require(len(state['completed']) == 201 and sum(r['trials'] for r in state['completed']) == 1120,
            'Expected201 completed entries /1120 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round203 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_AUX_RANK204_' + tag + '_' + expected + '.json')
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
    names = ('test_monthly_aux_rank204_host.py', 'test_monthly_aux_rank204_kernel.py',
             'test_monthly_aux_rank204_independent.py')
    modules = [module('test_aux_rank204_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
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
    path = ROOT / 'work/aux_rank204_CODE_REVIEW.json'
    review = read(path)
    expected = {str(p.relative_to(ROOT)): sha(p) for p in code_files()[:6]}
    require(review.get('passed') is True and review.get('unresolved_blockers') == [], 'Independent numerical code review incomplete')
    require(review.get('code_sha256') == expected, 'Numerical files changed after independent review')
    for key in ('actual_features_computed', 'actual_portfolio_returns_computed', 'production_files_modified', 'strategy_registered_by_reviewer'):
        require(review.get(key) is False, 'Independent review scope differs: '+key)
    pins = {str(path): sha(path)}
    for name, digest in review.get('input_sha256', {}).items():
        require((Path(name) if Path(name).is_absolute() else ROOT/name).resolve()!=CENTRAL.resolve(), 'Mutable central cannot be a persistent review input')
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
    source_binding = cache_source_binding()
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
    require(all((Path(name) if Path(name).is_absolute() else ROOT/name).resolve()!=CENTRAL.resolve()
                for name in pins), 'Mutable central cannot enter registered input pins')
    verify(pins); protected = read(PROTECTED); verify(protected)
    require(len(protected) == 21444, 'Incomplete protected archive manifest')
    OUT.mkdir()
    with (OUT / 'STATE_BEFORE.json').open('xb') as stream: stream.write(before)
    plan = dict(schema='monthly_aux_rank204_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, raw_source_hashes=raw, predecessor_binding=prior_binding,
        cache_source_binding=source_binding,numerical_code_review=numerical_review,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_feature_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_cached_transforms_before_registration=0, actual_ranks_before_registration=0,
        actual_new_returns_before_registration=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_auxiliary_cache_transforms',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare full auxiliary rank and affine transformations; select first88 before later34 diagnostics.')
    commit_state(before, state, 'register')
    print(json.dumps(dict(status='REGISTERED', plan_sha256=sha(OUT / 'PLAN.json'))), flush=True)


def gate(expected):
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['contract'] == contract() and p['runtime'] == ev.runtime(), 'Frozen contract/runtime changed')
    verify(p['input_sha256'])
    require(p['cache_source_binding'] == cache_source_binding(), 'Registered original cache binding changed')
    require(p['numerical_code_review']==code_review_binding(),'Registered independent code review changed')
    require(sha(OUT / 'STATE_BEFORE.json') == p['prior_central_sha256'], 'Changed pre-registration state')
    state = read(CENTRAL)
    require(state['registered_queue'] == [dict(directory=OUT.name, plan_sha256=expected)], 'Central registration mismatch')
    state_check(state)
    require(p.get('predecessor_binding') == predecessor_binding(state), 'Registered predecessor bindings changed')
    return p


def scale_and_delta(values,n):
    require(set(values)==set(VALUE_KEYS),'Feature keys changed')
    shape=np.asarray(values['support']).shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Feature prefix shape')
    v={}
    for key,value in values.items():
        raw=np.asarray(value);require(raw.shape==shape,'Feature axes differ')
        require(raw.dtype==np.dtype('float64'),'Feature storage must be float64')
        v[key]=raw[:n]
        require(np.isfinite(v[key]).all(),'Feature values must be finite')
    require((np.abs(v['support'])<=.5+1e-12).all() and (v['weight']>0).all(),'Invalid original support/weight')
    transforms,deltas={},{}
    constant=np.all(v['support']==v['support'][:,:1],axis=1)
    for arm,key in RAW.items():
        t=v[key];d=v[key+'_delta']
        require((np.abs(t)<=.5+1e-12).all() and (np.abs(d)<=MAX_DELTA).all(),'Target/delta bounds')
        expected=v['weight']*(t-v['support']);expected[constant]=0.
        if n:expected[0]=0.
        np.testing.assert_array_equal(d,expected)
        transforms[arm],deltas[arm]=t.copy(),d.copy()
    return transforms,deltas

def decisions(study, arrays, meta, values, n):
    policy = read(LOCAL / 'INCUMBENT_POLICY.json')
    choices, scores = study.choices(study.configs()[4], arrays, meta, n, policy=policy)
    orders = {NAMES[0]: study.rp.make_orders(choices, scores, meta)}
    ranks, deltas = scale_and_delta(values, n)
    for arm in NAMES[1:]:
        pre = study.modified_pre_score(study.configs()[4], arrays, n)
        view = pre[:n].copy(); changed = deltas[arm] != 0.
        view[changed] += deltas[arm][changed]; pre[:n] = view
        chosen, score = study.adapter.score_and_choose('incumbent66', 0., np.full(arrays['baseline'].shape, np.nan),
            arrays, meta, n, pre_score=pre, effects=arrays['effects'], episode_policy=policy,
            boundary_selector=study.ep.select_boundary)
        orders[arm] = study.rp.make_orders(chosen, score, meta)
    return orders, ranks, deltas


def require_feature_audit(check,values):
    for key in FEATURE_FLAGS:require(check.get(key) is True,'Independent feature evidence missing '+key)
    require(set(values)==set(VALUE_KEYS) and all(v.shape==(122,49) for v in values.values()),'Independent feature shape')
    for key,value in dict(events=122,symbols_per_event=49,checked_cells=5978).items():
        require(type(check.get(key)) is int and check[key]==value,'Independent feature coverage '+key)
    require(check.get('computes_returns') is False,'Feature audit must not compute returns')
    for key in ('new_model_fits','predictive_model_fits'):
        require(type(check.get(key)) is int and check[key]==0,'No new fits allowed')
    counts=estimate_counts(check)
    levels=[len(np.unique(row)) for row in values['support']]
    require(counts['constant_rows']==sum(x==1 for x in levels) and counts['two_level_rows']==sum(x==2 for x in levels),
        'Independent degeneracy accounting differs')

def require_delta_audit(check,deltas):
    for key in DELTA_FLAGS:require(check.get(key) is True,'Independent delta evidence missing '+key)
    maxima={k:float(np.max(np.abs(v),initial=0.)) for k,v in deltas.items()}
    require(set(deltas)==set(NAMES[1:]) and check.get('max_absolute_delta')==MAX_DELTA,'Independent delta bound/arms')
    require(max(maxima.values())<=MAX_DELTA and check.get('observed_max_absolute_delta')==max(maxima.values())
        and check.get('per_arm_max_absolute_delta')==maxima,'Independent observed delta bounds')
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
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation is write-once')
    import monthly_aux_rank204_kernel as kernel
    import monthly_aux_rank204_independent as independent
    study=local();meta,arrays,px,op=study.load()
    require(meta==read(LOCAL/'META.json'),'Original identities changed')
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original27 arrays/122x49 required')
    before={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in arrays.items()}
    require(plan['cache_source_binding']==cache_source_binding(),'Original cache source changed before prepare')
    values,audit=kernel.build_features(arrays)
    check=independent.verify_features(arrays,values,audit)
    require_feature_audit(check,values)
    counts=estimate_counts(check)
    require(counts==estimate_counts(audit),'Independent transformation accounting differs')
    orders,transforms,deltas=decisions(study,arrays,meta,values,122)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original full order parity failed')
    delta_check=independent.verify_deltas(values,arrays,deltas)
    require_delta_audit(delta_check,deltas)
    for n in FEATURE_PREFIXES:
        prefix_arrays=cache_prefix(arrays,n)
        small,small_audit=kernel.build_features(prefix_arrays)
        require(small_audit['event_history']==audit['event_history'][:n],'Physically truncated cache audit differs')
        for key in values:np.testing.assert_array_equal(small[key],values[key][:n])
        prefix_check=independent.verify_features(prefix_arrays,small,small_audit)
        require(prefix_check.get('passed') is True,'Independent cache prefix failed')
        oo,tt,dd=decisions(study,arrays,meta,values,n)
        require(all(oo[k]==orders[k][:n] for k in NAMES),'Prefix order parity failed')
        for key in RAW:
            np.testing.assert_array_equal(tt[key],transforms[key][:n])
            np.testing.assert_array_equal(dd[key],deltas[key][:n])
        require_delta_audit(independent.verify_deltas(small,prefix_arrays,dd),dd)
    require(before=={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in arrays.items()},'Original arrays modified')
    gate(expected)
    dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FEATURES.npz',**values)
    np.savez_compressed(dest/'TRANSFORMS_DELTAS.npz',**{'transform_'+k:v for k,v in transforms.items()},**{'delta_'+k:v for k,v in deltas.items()})
    dump(dest/'FEATURE_AUDIT.json',audit);dump(dest/'ORDERS.json',orders);dump(dest/'META.json',meta)
    dump(dest/'SOURCE_PARITY.json',dict(cache_source_binding=plan['cache_source_binding'],
        raw_source_hashes=plan['raw_source_hashes'],original_loader_registered_checks=True,
        fresh_raw_price_reconstruction=False,original_cache_exact=True))
    dump(dest/'INDEPENDENT_PREPARE.json',dict(feature=check,delta=delta_check,prefixes=123,feature_prefixes=list(FEATURE_PREFIXES),
        feature_prefix_sources_truncated=True,original_cache_exact=True,original_core_metadata_and_support_preserved=True,
        single_original_episode=True,original_orders_exact=True,original_arrays_unchanged=True,original_array_count=27,
        new_model_fits=0,predictive_model_fits=0,new_predictive_model_fits=0,model_fit_accounting=MODEL_FIT_ACCOUNTING,**counts))
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',plan_sha256=expected,created_utc=now(),
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,new_model_fits=0,
        predictive_model_fits=0,new_predictive_model_fits=0,model_fit_accounting=MODEL_FIT_ACCOUNTING,original_array_count=27,**counts))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)

def verify_prepared_evidence(c, values, deltas, audit):
    require(set(c['artifacts_sha256']) == PREPARED_FILES, 'Prepared artifact set incomplete')
    require_feature_audit(audit['feature'], values)
    require_delta_audit(audit['delta'], deltas)
    require(audit.get('prefixes') == 123 and audit.get('feature_prefixes') == list(FEATURE_PREFIXES) and
            audit.get('feature_prefix_sources_truncated') is True and audit.get('original_cache_exact') is True and audit.get('original_orders_exact') is True and
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
        close_vs_affine_control=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_affine_control=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
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
    import monthly_aux_rank204_independent as independent
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
    import monthly_aux_rank204_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, full_pre_episode_auxiliary_rank=True, original_support_retained_for_episode=True, observer_promotion=False))
    gate(expected)
    dump(dest / 'COMPLETION.json', dict(status='COMPLETE', created_utc=now(), plan_sha256=expected,
        frozen_sha256=sha(OUT / 'selected/FROZEN.json'), paths=9, slices=45,
        artifacts_sha256={p.name: sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='COMPLETE', primary=frozen['selection']['primary'],
        full_cagr={n: results[n]['same_close_zero']['metrics']['cagr'] for n in NAMES},
        completion_sha256=sha(dest / 'COMPLETION.json'))), flush=True)


def require_result_manifest(manifest):
    require(isinstance(manifest,dict) and set(manifest)==RESULT_FILES,
            'Exact38 result artifacts required; completion receipt is excluded')


def close(expected):
    gate(expected); completion = read(OUT / 'results/COMPLETION.json')
    require(completion['status'] == 'COMPLETE' and completion['plan_sha256'] == expected and
            completion['paths'] == 9 and completion['slices'] == 45, 'Incomplete diagnostics')
    require_result_manifest(completion['artifacts_sha256'])
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
        next_step='Review full pre-episode auxiliary rank versus affine calibration; preserve frozen observers.')
    require(len(state['completed']) == 202 and sum(r['trials'] for r in state['completed']) == 1123, 'Wrong final counters')
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




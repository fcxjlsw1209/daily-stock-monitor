"""Prospective annual matured risk-forecast reliability, fixed small auxiliary contrast."""
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
OUT = ROOT / 'outputs/monthly_risk_reliability223'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_idio_risk222'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67', 'risk_forecast_reliability', 'previous_month_reliability_control')
RAW = {NAMES[1]: 'candidate', NAMES[2]: 'control'}
TRAIN, BUDGET, MAX_RANK, MAX_DELTA = 88, .01, .5, .01000000000001
FEATURE_PREFIXES = (0, 1, 53, 88, 122)
COUNT_FIELDS = ('checked_cells','available_cells','unavailable_cells','year_security_groups','supported_groups','cold_start_groups','historical_month_pairs','source_available_pairs','available_pairs','valid_price_observations','simple_return_observations','second_moment_estimates','error_evaluations','calibration_estimates','ranked_observations')
ESTIMATE_FIELDS = COUNT_FIELDS[3:]
MODEL_FIT_ACCOUNTING = 'Two annual calibration means per supported unique year-security group, counted as new_model_fits and predictive_model_fits; zero new Ridge/tree fits. Source observations and estimates counted once per unique group, repeating overlapping consumption across distinct pairs; prefix and independent repeats excluded.'


CONTEXT = ROOT / 'work/monthly_risk_reliability223_context.json'
CONTEXT_SHA = '38c0dd8b6771f43e2b58ec46b985f2f3e5fb6a12cf209a5ba179768746860a70'
SOURCE_REVIEW = ROOT / 'work/risk_reliability223_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA = '6a5e230795072753ec1a44fc0ff0ebd35499828bc150b5d927cdecf38be1c3c1'
COVERAGE = ROOT / 'work/after222_risk_forecast_coverage.json'
COVERAGE_SHA = 'f5a494616cbe54aa2d71184c4b2c4e33edc00e5a4e8ddc37340306d2d6561213'
PRIOR_EXPECTED = {'PLAN.json': 'e36f207eb4524e27eb5e0ef257266056a0d11b77c0311e18a867d6b441dc834e', 'COMPLETION.json': '663e3e3802d6aba16341e1a267aa688ac527288a350044c1c272365ed9aa391c', 'results/COMPLETION.json': '8a6418a7a19207db8375c742d8f788f1b90311b1f841bbd6a2a3c36ad8ccd6a0', 'RESEARCH_REVIEW.json': 'bc56e39fc0ef3253c8977f7057278a281646d2887f3c8f47acde1c05d0b5fe22'}

FEATURE_FLAGS = ('passed', 'independent_scalar_rebuild', 'annual_prior_december_cutoffs_checked', 'five_year_sixty_calendar_months_checked', 'official_month_windows_checked', 'current63_forecast_checked', 'same_realized_outcome_checked', 'preceding_month_control_checked', 'noncentered_second_moments_checked', 'symmetric_relative_squared_errors_checked', 'common_pair_support_checked', 'same_column_history_checked', 'no_historical_membership_filter', 'eighteen_month_support_checked', 'annual_group_reuse_checked', 'common_rank_domain_checked', 'original_pct_ranks_checked', 'exact_common_availability', 'audit_checked', 'calibration_estimates_counted', 'no_ridge_or_tree_refits', 'no_current_or_future_outcome_inputs', 'monthend_queries_checked')
TEST_RECEIPT = ROOT / 'work/monthly_risk_reliability223_TESTS.json'
PREPARED_FILES = {'FEATURES.npz', 'TRANSFORMS_DELTAS.npz', 'FEATURE_AUDIT.json',
                  'ORDERS.json', 'META.json', 'SOURCE_PARITY.json', 'INDEPENDENT_PREPARE.json'}
RESULT_FILES = {arm+'_'+path[0]+'_'+kind+'.json' for arm in NAMES for path in ev.PATHS
                for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')} | {'RESULTS.json','INDEPENDENT_OUTCOME.json'}
EXTERNAL_RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
sha, read, require, dump = ev.sha, ev.read, ev.require, ev.dump


def now():
    return datetime.now(timezone.utc).isoformat()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def local():
    return module('monthly_risk_reliability223_original', LOCAL / 'study.py')


def contract():
    require(sha(CONTEXT)==CONTEXT_SHA,'Changed prospective context')
    require(sha(COVERAGE)==COVERAGE_SHA,'Changed prospective coverage')
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,arms=list(NAMES),candidate=NAMES[1],control_nonpromotable=NAMES[2],period=['2016-08-17','2026-09-24'],training_months=88,later_months=34,budget=BUDGET,max_absolute_rank=MAX_RANK,max_absolute_delta=MAX_DELTA,
        source_window=context['formation'],scoring='Funded auxiliary delta=.01*(rank-originalsupport), original common available percentile ranks, all states. Initialpartial missing. Original core and single episode unchanged.',feature_prefix_source='Physically truncate original adjusted prices and query metadata at prefix date; history cutoff remains priorDec31.',verification=dict(independent_all_cells=True,feature_prefixes=list(FEATURE_PREFIXES),all123_order_prefixes=True),limitation=context['limitations'],historical_best_tracking='First88 selects; full-period best is descriptive among previously training-selected research candidates, not pristine OOS or observer promotion.',model_fit_accounting=MODEL_FIT_ACCOUNTING,new_ridge_fits=0,new_tree_fits=0,new_arms=2,registered_trials=3,execution_paths=9,no_observer_writes=True,original_pool_preserved=True,no_parameter_search=True)


def code_files():
    return [ROOT / 'work' / n for n in ('monthly_risk_reliability223_host.py',
        'test_monthly_risk_reliability223_host.py', 'monthly_risk_reliability223_kernel.py',
        'test_monthly_risk_reliability223_kernel.py', 'monthly_risk_reliability223_independent.py',
        'test_monthly_risk_reliability223_independent.py', 'monthly_risk_reliability223_context.json', 'after222_risk_forecast_coverage.json', 'monthly_nested_trend199_kernel.py', 'monthly_risk_consensus219_kernel.py', 'monthly_tail_moment221_kernel.py', 'ptr169_rebuild_evaluate.py',
        'ptr169_results_independent.py', 'price_distribution171_independent.py', 'monthly_sector_recurrence192_independent.py')] + [ROOT/'outputs/monthly_core_timing70_research/core_adapter.py', ROOT/'outputs/monthly_core_timing70_research/test_core_adapter.py', ROOT/'outputs/monthly_path70_research/incumbent_adapter.py', ROOT/'outputs/monthly_daily_episodes/episodes.py']


def verify(pins):
    require(isinstance(pins,dict) and pins,'Nonempty source pins required')
    for name,digest in pins.items():
        path=Path(name);path=(path if path.is_absolute() else ROOT/path).resolve()
        require('CloudStorage' not in path.parts,'Cloud input forbidden')
        require(path.is_relative_to(ROOT) or EXTERNAL_RUNTIME.get(str(path))==digest,
                'Unapproved external dependency: '+str(path))
        require(path!=CENTRAL.resolve(),'Mutable central cannot be a persistent source input')
        require(path.is_file() and sha(path)==digest,'Changed or missing source: '+str(path))

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
    review=read(SOURCE_REVIEW);context=read(CONTEXT)
    require(review.get('passed') is True and review.get('context_sha256')==CONTEXT_SHA,'Source admission/context differs')
    require(review.get('raw_source_hashes')==raw and len(raw)==64,'Original64 raw-source lineage changed')
    require(review.get('npz_decoded') is False and review.get('actual_source_coverage_recomputed') is False,'Source review must remain metadata/hash only')
    for key in ('actual_new_features_computed','actual_new_orders_computed','actual_new_returns_computed'):
        require(review.get(key) is False,'Source audit crossed feature boundary: '+key)
    for key in ('source_admission_metadata_only','metadata_axes_verified','calendar_clock_verified','raw_source_hashes_verified','raw_source_map_matches_original','integration_source_verified','annual_cutoff_calendar_verified','canonical_identity_scope_reviewed'):
        require(review.get(key) is True,'Source admission evidence missing '+key)
    pins={str(SOURCE_REVIEW):SOURCE_REVIEW_SHA}
    for label,document,key in (('source inputs',review,'input_sha256'),('source evidence',review,'evidence_sha256'),('context evidence',context,'evidence_sha256')):
        mapping=document.get(key)
        require(isinstance(mapping,dict) and mapping,'Nonempty '+label+' required')
        for name,digest in mapping.items():bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    verify(pins)
    return dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,
        source_evidence_sha256=review['evidence_sha256'],context_evidence_sha256=context['evidence_sha256'],
        source_column='Original adjusted-close canonical columns with inherited vendor/alias/leading-gap splice; annual previous5years monthly matured forecast errors, no new filling or alias joins',
        source_admission_metadata_only=True,strict_pit=False)


def require_source_support(audit):
    groups=audit['group_history'];require(len(groups)==audit['year_security_groups'],'Group count')
    lookup={g['group_id']:g for g in groups};require(len(lookup)==len(groups),'Unique group identities')
    for g in groups:
        y=g['year'];require(g['cutoff']==str(y-1)+'-12-31' and g['history_start_month']==str(y-5)+'-01' and g['history_end_month']==str(y-1)+'-12','Annual clock')
        require(len(g['pairs'])==60,'Exactly60 prospective historical months')
        require(g['available_pairs']==sum(p['available'] for p in g['pairs']) and g['source_available_pairs']==sum(p['source_available'] for p in g['pairs']),'History support counts')
    events=audit['event_history'];require(len(events)==audit['events'],'Event count')
    require(audit['available_by_event']==[e['available_count'] for e in events],'Available event series')
    for e in events:
        require(e['symbols']==[c['symbol'] for c in e['cells']] and e['group_ids']==[c['group_id'] for c in e['cells']],'Event identities')
        require(all(g in lookup for g in e['group_ids']),'Missing group')
        require(all(lookup[k]['year']==int(e['date'][:4]) and lookup[k]['symbol']==symbol for k,symbol in zip(e['group_ids'],e['symbols'])),'Annual canonical-column join')
    return True


def price_inputs(source,meta):
    require(len(meta['etfs'])==17 and len(set(meta['etfs']))==17,'Original17ETF identities')
    require(isinstance(source.get('adj'),pd.DataFrame),'Original adjusted closes required')
    return {'adj':source['adj']}


def estimate_counts(check):
    counts={key:check.get(key) for key in COUNT_FIELDS}
    require(all(type(v) is int and v>=0 for v in counts.values()),'Nonnegative integer counters')
    c=counts;g=c['year_security_groups'];support=c['supported_groups'];source=c['source_available_pairs'];valid=c['available_pairs'];available=c['available_cells']
    require(c['checked_cells']==available+c['unavailable_cells'],'Query counts')
    require(g==support+c['cold_start_groups'] and c['historical_month_pairs']==60*g,'Unique annual groups')
    require(0<=valid<=source<=60*g and 18*support<=valid,'Paired historical support')
    require(c['second_moment_estimates']==3*source and c['error_evaluations']==2*valid and c['calibration_estimates']==2*support and c['ranked_observations']==2*available,'Estimation counts')
    require(c['valid_price_observations']==c['simple_return_observations']+3*source,'Consumed prices/returns')
    require(check.get('new_model_fits')==check.get('predictive_model_fits')==c['calibration_estimates'] and check.get('new_ridge_fits')==check.get('new_tree_fits')==0,'Calibration fit accounting')
    series=check.get('available_by_event');require(isinstance(series,list) and len(series)==122 and all(type(x)is int and 0<=x<=49 for x in series) and sum(series)==available,'Monthly query support')
    return {key:c[key] for key in ESTIMATE_FIELDS}


def feature_prefix_panel(panel, meta, n):
    """Physically remove future source rows, with full original identities; do not inspect their values."""
    require(type(n) is int and 0 <= n <= len(meta['dates']), 'Invalid source prefix')
    require(set(panel) == {'adj'} and all(isinstance(v, pd.DataFrame) for v in panel.values()),
            'Original stored adjusted-close frame required')
    if not n:
        return {key: value.iloc[:0].copy() for key, value in panel.items()}
    day = pd.Timestamp(meta['dates'][n-1])
    return {key: value.loc[:day].copy() for key, value in panel.items()}

def feature_prefix_meta(meta,n):
    require(type(n) is int and 0<=n<=len(meta['dates']),'Invalid metadata prefix')
    result=dict(meta)
    for key in ('dates','ends','symbols'):
        if key in meta:
            require(len(meta[key])==len(meta['dates']),'Metadata axis differs: '+key)
            result[key]=meta[key][:n]
    return result


def state_check(state):
    require(len(state['completed']) == 220 and sum(r['trials'] for r in state['completed']) == 1177,
            'Expected220 completed entries /1177 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round222 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_RISK_RELIABILITY223_' + tag + '_' + expected + '.json')
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
    names = ('test_monthly_risk_reliability223_host.py', 'test_monthly_risk_reliability223_kernel.py',
             'test_monthly_risk_reliability223_independent.py')
    modules = [module('test_risk_reliability223_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
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
    path = ROOT / 'work/risk_reliability223_CODE_REVIEW.json'
    review = read(path)
    expected = {str(p.relative_to(ROOT)): sha(p) for p in code_files()[:6]}
    require(review.get('passed') is True and review.get('unresolved_blockers') == [], 'Independent numerical code review incomplete')
    require(review.get('context_sha256')==CONTEXT_SHA,'Independent review context differs')
    require(review.get('code_sha256') == expected, 'Numerical files changed after independent review')
    for key in ('actual_features_computed', 'actual_model_fits_computed', 'actual_portfolio_returns_computed', 'production_files_modified', 'strategy_registered_by_reviewer'):
        require(review.get(key) is False, 'Independent review scope differs: '+key)
    require(isinstance(review.get('input_sha256'),dict) and review['input_sha256'],'Independent review source pins required')
    pins = {str(path): sha(path)}
    for name, digest in review['input_sha256'].items():
        require((Path(name) if Path(name).is_absolute() else ROOT/name).resolve()!=CENTRAL.resolve(),'Mutable central cannot be a persistent review input')
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
    require(all((Path(name) if Path(name).is_absolute() else ROOT/name).resolve()!=CENTRAL.resolve()
                for name in pins),'Mutable central cannot enter registered input pins')
    verify(pins); protected = read(PROTECTED); verify(protected)
    require(len(protected) == 21444, 'Incomplete protected archive manifest')
    OUT.mkdir()
    with (OUT / 'STATE_BEFORE.json').open('xb') as stream: stream.write(before)
    plan = dict(schema='monthly_risk_reliability223_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, raw_source_hashes=raw, predecessor_binding=prior_binding,
        price_source_binding=source_binding,numerical_code_review=numerical_review,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_feature_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_simple_return_estimates_before_registration=0, actual_calibration_estimates_before_registration=0,
        actual_historical_errors_before_registration=0, actual_ratio_ranks_before_registration=0,
        actual_new_returns_before_registration=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_annual_risk_reliability',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare annual matured same-security risk forecast errors; select first88 before later34 diagnostics.')
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


def scale_and_delta(values,arrays,n):
    require(set(values)=={'available','candidate','control','raw_candidate','raw_control'},'Feature keys')
    mask=np.asarray(values['available']);shape=mask.shape
    require(len(shape)==2 and shape[1]>0 and mask.dtype==np.dtype(bool) and type(n)is int and 0<=n<=shape[0],'Feature prefix')
    valid=mask[:n];ranks={}
    for rank,raw in (('candidate','raw_candidate'),('control','raw_control')):
        v=np.asarray(values[raw]);q=np.asarray(values[rank]);require(v.shape==q.shape==shape,'Feature axes')
        v=np.asarray(v[:n],float);q=np.asarray(q[:n],float)
        require(not np.isinf(v).any() and not np.isinf(q).any() and np.array_equal(np.isfinite(v),valid) and np.array_equal(np.isfinite(q),valid),'Common support')
        require(((v[valid]>=-1.)&(v[valid]<=0.)).all(),'Negative squared symmetric error bounds')
        expected=np.full(valid.shape,np.nan)
        for i in range(n):
            ids=np.flatnonzero(valid[i]);m=len(ids)
            for j in ids:
                row=v[i,ids];expected[i,j]=(np.count_nonzero(row<v[i,j])+(np.count_nonzero(row==v[i,j])+1)/2)/m-.5
        np.testing.assert_array_equal(q,expected);ranks[rank]=q
    baseline=np.asarray(arrays['baseline']);core=np.asarray(arrays['core']);require(baseline.ndim==2 and baseline.shape[1]==shape[1] and baseline.shape[0]>=n and core.shape==baseline.shape,'Original axes')
    weights=np.asarray(arrays['core_weight']);weights=weights[:,None] if weights.shape==(len(baseline),) else weights
    require(weights.shape in ((len(baseline),1),baseline.shape),'Core weights shape')
    cw=np.broadcast_to(np.asarray(weights[:n],float),(n,shape[1]));calm=np.asarray(arrays['state_calm10'])
    require(calm.shape==(len(baseline),) and calm.dtype.kind=='b','Calm exact boolean')
    np.testing.assert_array_equal(cw,np.broadcast_to(np.where(calm[:n],.5,.52)[:,None],cw.shape))
    b=np.asarray(baseline[:n],float);c=np.asarray(core[:n],float);require(np.isfinite(b).all() and np.isfinite(c).all(),'Finite original scores')
    support=(b-c)/(1-cw);require((np.abs(support)<=.5+1e-12).all(),'Original support bounds')
    targets,deltas={},{}
    for arm,key in RAW.items():
        target=ranks[key].copy()
        if n:target[0]=np.nan
        delta=np.zeros(target.shape);good=np.isfinite(target);delta[good]=BUDGET*(target[good]-support[good])
        require(np.max(np.abs(delta),initial=0.)<=MAX_DELTA,'Funded auxiliary budget')
        targets[arm]=target;deltas[arm]=delta
    return targets,deltas


def path_adapter():
    return module('risk_reliability223_frozen_path_adapter',ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')


def decisions(study,arrays,meta,values,n):
    policy=read(LOCAL/'INCUMBENT_POLICY.json')
    choices,scores=study.choices(study.configs()[4],arrays,meta,n,policy=policy)
    orders={NAMES[0]:study.rp.make_orders(choices,scores,meta)}
    targets,deltas=scale_and_delta(values,arrays,n);adapter=path_adapter()
    # Canonicalize only consumed weights; frozen adapter otherwise converts future suffixes.
    source=dict(arrays);weights=np.asarray(arrays['core_weight']);weights=weights[:,None] if weights.shape==(len(arrays['baseline']),) else weights
    canonical=np.full(arrays['baseline'].shape,.5);canonical[:n]=weights[:n];source['core_weight']=canonical
    for arm in NAMES[1:]:
        full=np.full(arrays['baseline'].shape,np.nan);full[:n]=targets[arm]
        pre=study.modified_pre_score(study.configs()[4],arrays,n)
        chosen,score=adapter.score_and_choose('incumbent66',BUDGET,full,source,meta,n,pre_score=pre,effects=arrays['effects'],episode_policy=policy,boundary_selector=study.ep.select_boundary)
        orders[arm]=study.rp.make_orders(chosen,score,meta)
    return orders,targets,deltas


def require_feature_audit(check,values):
    for key in FEATURE_FLAGS:require(check.get(key) is True,'Independent feature evidence '+key)
    require(values['available'].shape==(122,49),'Full122x49 feature coverage')
    expected=dict(events=122,symbols_per_event=49,checked_cells=5978,available_cells=int(values['available'].sum()),unavailable_cells=int((~values['available']).sum()))
    for k,v in expected.items():require(type(check.get(k))is int and check[k]==v,'Coverage '+k)
    require(check.get('computes_returns') is False and check.get('producer_modules_imported') is False and check.get('numerical_tolerance')==0.,'Independent scope')
    estimate_counts(check)


def require_delta_audit(check,deltas):
    for key in ('passed','funded_aux_formula_checked','original_pct_ranks_preserved','first_partial_unchanged','all_states_preserved','missing_preserves_score','core_weights_checked','no_second_rank','independent_delta_rebuild'):
        require(check.get(key) is True,'Independent delta evidence '+key)
    require(check.get('budget')==BUDGET and check.get('max_absolute_delta')==MAX_DELTA,'Independent funded bound')
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
    import monthly_risk_reliability223_kernel as kernel
    import monthly_risk_reliability223_independent as independent
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
    require(check['predictive_model_fits'] == audit['predictive_model_fits'] == check['calibration_estimates'],
            'Independent model accounting differs')
    counts = estimate_counts(check)
    require(counts == estimate_counts(audit), 'Independent calibration accounting differs')
    require_source_support(audit)
    orders, ranks, deltas = decisions(study, arrays, meta, values, 122)
    require(orders[NAMES[0]] == read(LOCAL / 'local_linear_ORDERS.json'), 'Original full order parity failed')
    rank_check = independent.verify_deltas(values, arrays, ranks, deltas, budget=BUDGET)
    require_delta_audit(rank_check, deltas)
    for n in FEATURE_PREFIXES:
        prefix_panel = feature_prefix_panel(panel, meta, n)
        prefix_meta=feature_prefix_meta(meta,n)
        small, small_audit = kernel.build_features(prefix_panel, prefix_meta, calendar)
        prefix_check=independent.verify_features(prefix_panel,prefix_meta,calendar,small,small_audit)
        require(prefix_check.get('passed') is True,'Physical prefix independent verification failed')
        require(small_audit['event_history'] == audit['event_history'][:n], 'Truncated adjusted-close source prefix audit differs')
        full_groups={g['group_id']:g for g in audit['group_history']}
        require(all(g['group_id'] in full_groups and g==full_groups[g['group_id']] for g in small_audit['group_history']), 'Annual group prefix differs')
        for key in values:
            if n:np.testing.assert_array_equal(small[key], values[key][:n])
            else:require(small[key].shape==(0,0),'Zero physical prefix must have no future query width')
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
    dump(dest / 'SOURCE_PARITY.json', dict(raw_source_hashes=raw_hashes, price_source_binding=plan['price_source_binding'], feature_panel='original canonical adjusted-close columns, annual priorDec31 complete monthly forecast reliability; source transformations unchanged', execution_prices=parity))
    dump(dest / 'INDEPENDENT_PREPARE.json', dict(feature=check, delta=rank_check, prefixes=123, feature_prefixes=list(FEATURE_PREFIXES),
        feature_prefix_sources_truncated=True, feature_prefix_metadata_truncated=True, feature_prefix_independently_verified=True, source_admission_metadata_only=True, source_support_exact=True, original_core_metadata_and_support_preserved=True, single_original_episode=True, original_orders_exact=True, original_arrays_unchanged=True, original_array_count=27, new_model_fits=counts['calibration_estimates'],
        predictive_model_fits=counts['calibration_estimates'], new_ridge_fits=0,new_tree_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING, **counts))
    dump(dest / 'COMPLETION.json', dict(status='COMPLETE', plan_sha256=expected, created_utc=now(),
        artifacts_sha256={p.name: sha(p) for p in dest.iterdir()}, new_returns_computed=False, new_model_fits=counts['calibration_estimates'],
        predictive_model_fits=counts['calibration_estimates'], new_ridge_fits=0,new_tree_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING, original_array_count=len(arrays), **counts))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED', completion_sha256=sha(dest / 'COMPLETION.json'))), flush=True)


def verify_prepared_evidence(c, values, deltas, audit):
    require(set(c['artifacts_sha256']) == PREPARED_FILES, 'Prepared artifact set incomplete')
    require_feature_audit(audit['feature'], values)
    require_delta_audit(audit['delta'], deltas)
    require(audit.get('prefixes') == 123 and audit.get('feature_prefixes') == list(FEATURE_PREFIXES) and
            audit.get('feature_prefix_sources_truncated') is True and audit.get('feature_prefix_metadata_truncated') is True and audit.get('feature_prefix_independently_verified') is True and audit.get('source_admission_metadata_only') is True and audit.get('source_support_exact') is True and audit.get('original_orders_exact') is True and
            audit.get('original_core_metadata_and_support_preserved') is True and audit.get('single_original_episode') is True and audit.get('original_arrays_unchanged') is True and audit.get('original_array_count') == 27 and audit.get('new_model_fits') == audit.get('calibration_estimates') and
            audit.get('predictive_model_fits') == audit.get('calibration_estimates') and audit.get('model_fit_accounting') == MODEL_FIT_ACCOUNTING,
            'Prepared invariants not verified')
    for record in (audit, c):
        require(type(record.get('new_model_fits')) is int and record['new_model_fits'] == record.get('predictive_model_fits') == record.get('calibration_estimates') and record.get('new_ridge_fits') == record.get('new_tree_fits') == 0,
                'Annual calibration fit counts not verified')
        expected_counts = estimate_counts(audit['feature'])
        require(all(type(record.get(key)) is int and record[key] == value
                    for key, value in expected_counts.items()), 'Annual calibration receipt accounting differs')
    require(c.get('new_returns_computed') is False and c.get('new_model_fits') == c.get('calibration_estimates') and
            c.get('predictive_model_fits') == c.get('calibration_estimates') and c.get('new_ridge_fits')==c.get('new_tree_fits')==0 and c.get('model_fit_accounting') == MODEL_FIT_ACCOUNTING and
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
        close_vs_previous_month_control=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_previous_month_control=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
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
    import monthly_risk_reliability223_independent as independent
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
    import monthly_risk_reliability223_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, risk_reliability=True, same_support_both_arms=True, observer_promotion=False))
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
        result_completion_sha256=sha(OUT / 'results/COMPLETION.json'), new_trials=3, new_model_fits=read(OUT/'prepared/COMPLETION.json')['calibration_estimates'],
        predictive_model_fits=read(OUT/'prepared/COMPLETION.json')['calibration_estimates'], new_ridge_fits=0,new_tree_fits=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
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
        next_step='Review annual risk prediction reliability contrast and preserved source limits; frozen observers unchanged.')
    require(len(state['completed']) == 221 and sum(r['trials'] for r in state['completed']) == 1180, 'Wrong final counters')
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


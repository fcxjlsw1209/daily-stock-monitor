"""Round225: paired annual Ridge uninterrupted relative-lag duration and total lag incidence.

All actual features and portfolio returns require a prospectively bound PLAN.
Synthetic tests may fit synthetic Ridge inputs only; main, independent and prefix fits are accounted separately.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import ptr169_rebuild_evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_relative_duration225'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_increment_calibration224'
COHORT_REFERENCE=ROOT/'outputs/monthly_relative_path209'
COHORT_PLAN_SHA='31ec884b14585af9b50b571d19b5afa7e954b7e2d89e4281efee1086212f863b'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67', 'relative_lag_run_ridge', 'relative_lag_incidence_ridge')
RAW = {NAMES[1]: 'candidate', NAMES[2]: 'control'}
TRAIN, DOSE = 88, .01
FEATURE_PREFIXES = (0, 1, 53, 88, 122)
EXTRA = ('candidate_prediction', 'control_prediction', 'candidate_rank', 'control_rank', 'common_available')
LABEL_KEYS = ('dates', 'ends', 'symbols', 'candidate_target', 'control_target', 'base_good', 'good')
ENDPOINT_PLAN = ROOT / 'outputs/monthly_endpoint155/PLAN.json'
ENDPOINT_PLAN_SHA = 'f10890b99c5b27d176eaa9467152358a2c3bd5b845c0bb4c9aa6865512f29201'
NUMERIC_PATH = ROOT / 'outputs/monthly_architecture_restart/residual_model.py'
NUMERIC_SHA = 'ea294b9a0a1e9493799f1014dcec464effa93dce70f44fb5de53a604cd832886'
PATH_COUNT_FIELDS=('source_path_rows','source_price_observations','target_session_observations','missing_path_rows','lag_session_observations','longest_run_session_observations')
MODEL_FIT_ACCOUNTING = 'Count mainpairedannualRidgefits honestly,onecandidateandonecontrolfit eachsupportedyear. Bothnewtargets; no refittedexistingtargetcontrol claimed. Independent/prefixfits separatelydiagnostic,notadditionalstrategytrials. No225actualnewlabels/parameters/forecasts/orders/returns beforePLAN.'
CONTEXT = ROOT / 'work/monthly_relative_duration225_context.json'
CONTEXT_SHA = '0392086957d2cd7a843811de6971cf0544788cdb484aa2fdcd7f74b3b195d5d8'
SOURCE_REVIEW=ROOT/'work/relative_duration225_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA='d77b6d94fbbea3b4308686f5850288967024908219c8a09ca22868f31973d57b'
PRIOR_EXPECTED = {'PLAN.json': 'd5bc22740047570c70cab82cb49ca8277ccf6fe12a739e492aa6e89e82a3c3d3', 'COMPLETION.json': 'c4575b9ce052224f8a2f7bc0367e5dddcb55a6a875ec5b74c6655c15b014af9a', 'results/COMPLETION.json': 'da58bb9d2b0c3634532f50052620eb4e38deeba14f10d35b8bba6d27ea7abbc8', 'RESEARCH_REVIEW.json': 'f9d06aff72a1feb24cd54c2d40c9a7058545bc26ced84cebc349ad6129faeb32'}
EXTERNAL_RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
RESULT_FILES={arm+'_'+path[0]+'_'+kind+'.json' for arm in NAMES for path in ev.PATHS
    for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}

BUILD_FLAGS = ('passed', 'audit_checked', 'independent_exact_rational_labels', 'endpoint_calendar_checked', 'shared_maturity_rows_weights_scalers_verified', 'annual_augmented_least_squares_verified', 'target_statistics_checked', 'count_ratios_rounded_once_checked', 'full_official_path_checked', 'longest_lag_run_checked', 'lag_incidence_checked', 'equality_breaks_run_checked', 'signal_excluded_end_included_checked', 'candidate_bounded_by_control_checked', 'shared_query_mask_and_centered_ranks_verified', 'no_query_future_outcomes', 'initial_partial_fallback_checked')
TEST_RECEIPT = ROOT / 'work/monthly_relative_duration225_TESTS.json'
PREPARED_FILES = {'FEATURES.npz', 'TRANSFORMS_DELTAS.npz', 'LABELS.npz', 'LABEL_AUDIT.json',
    'MODEL_AUDIT.json', 'RANK_AUDIT.json', 'ORDERS.json', 'META.json', 'SOURCE_PARITY.json',
    'IDENTITY_AUDIT.json', 'CALENDAR.json', 'INDEPENDENT_PREPARE.json'}
PRIOR_PREPARED_SHA = '7d862dd35658dc73a8f556fc4f6357c2855fbf530665510bf5b78a75f8864b7e'
PRIOR_PREPARED_FILES = frozenset(PREPARED_FILES|{'CONTROL_PARITY.json'})
PREPARED_FILES = PREPARED_FILES | {'COHORT_PARITY.json'}
COHORT_LABEL_KEYS = ('dates','ends','symbols','base_good','good')
sha, read, require, dump = ev.sha, ev.read, ev.require, ev.dump


def now():
    return datetime.now(timezone.utc).isoformat()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def local():
    return module('monthly_relative_duration225_original', LOCAL / 'study.py')


def code_review_binding():
    path = ROOT / 'work/relative_duration225_CODE_REVIEW.json'
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

def require_result_manifest(manifest):
    require(isinstance(manifest,dict) and set(manifest)==RESULT_FILES,
            'Exact38 result artifacts required; completion receipt is excluded')

def contract():
    require(sha(CONTEXT) == CONTEXT_SHA, 'Changed prospective context')
    context = read(CONTEXT)
    return dict(context=context, prospective_context_sha256=CONTEXT_SHA,
        arms=list(NAMES), candidate=NAMES[1], control_nonpromotable=NAMES[2],
        period=['2016-08-17', '2026-09-24'], training_months=88, later_months=34,
        dose=DOSE, max_absolute_delta=.005, targets=['candidate_target','control_target'],
        measurement=dict(candidate_target='longest_consecutive_cumulative_relative_lag_days/L', control_target='total_cumulative_relative_lag_days/L',
            exact_rational_binary_float_comparison=True, integer_count_ratios_rounded_once=True, complete_official_path=True,
            candidate_label_bounded_by_control=True, forecasts_unclipped_and_unordered=True, equality_breaks_run=True, natural_month_ends=True,
            inherited_common_good=True, minimum_rows_per_training_month=2,
            predictors='original_plain10', annual_prior_december31=True, lookback_years=5,
            exclusive_signal_lower_bound=True, minimum_mature_months=18, alpha=10., half_life_months=12.,
            shared_rows_weights_scalers=True, weighted_training_scaler=True, probability_calibration=False,
            rank='negative_prediction_average_rank0/(n-1)-0.5; n<=1 or constant =>0'),
        verification=dict(independent_all_cells=True, feature_prefixes=list(FEATURE_PREFIXES),
            feature_prefix_prices_truncated=True,feature_prefix_training_inputs_truncated=True,all123_order_prefixes=True, exact209_identity_and_masks=True),
        limitation=context['limitations'],
        historical_best_tracking='First88 selects this fixed relative-duration candidate. Historical best is descriptive full-period performance among previously training-selected candidates, not independent validation or observer promotion.',
        model_fit_accounting=MODEL_FIT_ACCOUNTING, new_model_fits='dynamic_main_paired_annual_fits',
        new_arms=2, registered_trials=3, execution_paths=9,
        no_observer_writes=True, original_pool_preserved=True, no_parameter_search=True)


def code_files():
    return [ROOT / 'work' / n for n in ('monthly_relative_duration225_host.py',
        'test_monthly_relative_duration225_host.py', 'monthly_relative_duration225_model.py',
        'test_monthly_relative_duration225_model.py', 'monthly_relative_duration225_independent.py',
        'test_monthly_relative_duration225_independent.py', 'monthly_relative_duration225_context.json', 'after224_relative_duration_coverage.json','ptr169_rebuild_evaluate.py',
        'ptr169_results_independent.py', 'price_distribution171_independent.py')]


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
    cohort = previous_cohort_binding()
    for path,digest in cohort['input_sha256'].items(): bind(pins,Path(path),digest)
    return dict(directory=PRIOR.name, input_sha256=pins, cohort_reference=cohort)


def endpoint_source_binding():
    """Metadata-only full frozen155 closure, including source/plain10/solver identities."""
    require(sha(ENDPOINT_PLAN) == ENDPOINT_PLAN_SHA, 'Frozen155 PLAN changed')
    old = read(ENDPOINT_PLAN)
    require(old['runtime'] == dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__) and
            old['round'] == 155 and old['status'] == 'REGISTERED', 'Frozen155 contract/runtime')
    pins = dict(old['hashes'])
    for section in ('raw_source_hashes', 'independent_prepare_source_hashes', 'independent_outcome_source_hashes'):
        for name, digest in old[section].items():
            require(pins.get(name) == digest, 'Incomplete155 source closure '+name)
    expected = dict(meta=LOCAL/'META.json', policy=LOCAL/'INCUMBENT_POLICY.json',
        original_orders=LOCAL/'local_linear_ORDERS.json', prices=ROOT/'outputs/monthly_2016_extension/PRICES.npz',
        train_inputs=ROOT/'outputs/monthly_target_definition70/TRAIN_INPUTS.npz',
        training_meta=ROOT/'outputs/monthly_earnings_factor_research/META.json', calendar=CALENDAR)
    for key, path in expected.items():
        ref = old['inputs'][key]
        require(Path(ref['path']).resolve() == path.resolve() and pins.get(str(path)) == ref['sha256'], 'Frozen155 input wiring '+key)
    require(pins.get(str(NUMERIC_PATH)) == NUMERIC_SHA, 'Frozen Ridge solver missing')
    require(old['raw_source_hashes'] == read(LOCAL/'PLAN.json')['raw_source_hashes'] and len(old['raw_source_hashes']) == 64, 'Original raw64 closure')
    require(old['calendar'] == read(LOCAL/'PLAN.json')['calendar'], 'Original calendar lineage')
    return dict(plan=dict(path=str(ENDPOINT_PLAN),sha256=ENDPOINT_PLAN_SHA), input_sha256=pins,
        inputs={key:old['inputs'][key] for key in expected}, calendar=old['calendar'],
        numeric_solver=dict(path=str(NUMERIC_PATH),sha256=NUMERIC_SHA))


def previous_cohort_binding():
    """Hash-only complete209 preparation binding; no numerical arrays decoded."""
    dest=COHORT_REFERENCE/'prepared';path=dest/'COMPLETION.json';pins={}
    bind(pins,COHORT_REFERENCE/'PLAN.json',COHORT_PLAN_SHA)
    bind(pins,path,PRIOR_PREPARED_SHA);c=read(path)
    require(c.get('status')=='COMPLETE' and c.get('plan_sha256')==COHORT_PLAN_SHA and
            c.get('new_returns_computed') is False and c.get('original_array_count')==27,
            'Prior209 preparation contract')
    require(set(c.get('artifacts_sha256',{}))==PRIOR_PREPARED_FILES,'Prior209 preparation artifact set')
    for name,digest in c['artifacts_sha256'].items():bind(pins,dest/name,digest)
    return dict(prior_plan_sha256=COHORT_PLAN_SHA,prior_prepared_completion_sha256=PRIOR_PREPARED_SHA,
                input_sha256=pins)


def cohort_parity(built):
    reference=previous_cohort_binding();dest=COHORT_REFERENCE/'prepared'
    # Lazy NPZ access: identities and eligibility only; neither old target/prediction/order.
    with np.load(dest/'LABELS.npz',allow_pickle=False) as old:
        for key in COHORT_LABEL_KEYS:
            previous=old[key];current=np.asarray(built['labels'][key])
            require(current.dtype==previous.dtype and current.shape==previous.shape and
                    current.tobytes()==previous.tobytes(),'Prior209 identity/common-cohort parity '+key)
    return dict(passed=True,prior_plan_sha256=reference['prior_plan_sha256'],
        prior_prepared_completion_sha256=reference['prior_prepared_completion_sha256'],
        label_keys_checked=list(COHORT_LABEL_KEYS),arrays_dtype_shape_bytes_exact=True,
        old_targets_loaded=False,old_forecasts_loaded=False,old_orders_loaded=False,old_returns_read=False)


def require_cohort_parity(check):
    for key in ('passed','arrays_dtype_shape_bytes_exact'):
        require(check.get(key) is True,'Missing exact source cohort parity '+key)
    require(check.get('prior_plan_sha256')==COHORT_PLAN_SHA and
            check.get('prior_prepared_completion_sha256')==PRIOR_PREPARED_SHA,'Cohort reference changed')
    require(check.get('label_keys_checked')==list(COHORT_LABEL_KEYS),'Incomplete cohort identity/mask coverage')
    for key in ('old_targets_loaded','old_forecasts_loaded','old_orders_loaded','old_returns_read'):
        require(check.get(key) is False,'Cohort parity scope expanded '+key)


def source_review_binding(raw):
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Changed225 source review')
    review=read(SOURCE_REVIEW)
    require(review.get('passed') is True and review.get('source_only') is True and
            review.get('context_sha256')==CONTEXT_SHA,'225 source admission/context differs')
    require(review.get('raw_source_hashes')==raw and len(raw)==64,'Original raw64 source lineage differs')
    flags=('source_admission_metadata_only','metadata_axes_verified','calendar_clock_verified',
           'raw_source_hashes_verified','raw_source_map_matches_original','endpoint155_wiring_verified',
           'prior209_complete_path_certificate_verified','prior209_prepared_manifest_verified',
           'annual_cutoff_calendar_verified','canonical_identity_scope_reviewed','both225_targets_new')
    for key in flags: require(review.get(key) is True,'Missing225 source proof '+key)
    for key in ('source_code_imported','npz_decoded','actual_source_coverage_recomputed',
                'actual_new_support_computed','actual_features_computed','actual_labels_computed',
                'actual_model_fits_computed','actual_orders_computed','actual_returns_computed',
                'strategy_registered','observers_modified','central_modified','strict_point_in_time',
                'old_numeric_label_parity_claimed','old_control_prediction_parity_claimed','old_control_order_parity_claimed'):
        require(review.get(key) is False,'225 source review crossed metadata boundary: '+key)
    require(review.get('post_registration_required_cohort_parity')==list(COHORT_LABEL_KEYS),
            '225 source cohort identity/mask scope differs')
    require(review.get('calendar_sha256')==sha(CALENDAR),'Source calendar changed')
    coverage=review.get('training_path_coverage',{})
    require(coverage.get('prior209_certificate') is True and coverage.get('fresh225_support') is False,
            '225 source coverage must remain prior209 certificate')
    require(coverage.get('training_rows')==8820 and coverage.get('training_months')==180 and
            coverage.get('admitted_good_rows')==coverage.get('base_rows')==coverage.get('path_complete_base_rows') and
            type(coverage.get('admitted_good_rows')) is int and coverage['admitted_good_rows']>0,
            'Source common cohort coverage differs')
    require(coverage.get('prepared_completion_sha256')==PRIOR_PREPARED_SHA and
            coverage.get('labels_archive_sha256')==sha(COHORT_REFERENCE/'prepared/LABELS.npz'),
            'Prior source cohort certificate changed')
    require(isinstance(coverage.get('final_good_mask_sha256'),str) and
            len(coverage['final_good_mask_sha256'])==64,'Source good mask digest required')
    require(isinstance(review.get('source_sha256'),dict) and review['source_sha256'],'Immutable source pins required')
    pins={str(SOURCE_REVIEW):SOURCE_REVIEW_SHA}
    for name,digest in review['source_sha256'].items():bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    bind(pins,ROOT/'work/after224_relative_duration_coverage.json',review['coverage_sha256'])
    bind(pins,ROOT/review['implementation']['path'],review['implementation']['sha256'])
    verify(pins)
    return dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,
                training_path_coverage=coverage,strict_pit=False)


def require_source_cohort(built,coverage):
    good=np.asarray(built['labels']['good'])
    require(good.dtype==np.dtype(bool) and good.ndim==1 and len(good)==coverage['training_rows'],
            'Admitted training shape/type changed')
    require(int(good.sum())==coverage['admitted_good_rows'] and
            hashlib.sha256(np.ascontiguousarray(good).tobytes()).hexdigest()==coverage['final_good_mask_sha256'],
            'Source-admitted complete-path training cohort changed')
    return True


def original_binding(study, plan):
    dependencies = list(study.implementations()) + list(study.source_files())
    dependencies += [LOCAL/name for name in ('PLAN.json','CACHE_HASHES.json','INPUT_HASHES.json','INCUMBENT_POLICY.json','local_linear_ORDERS.json')]
    dependencies += [LOCAL/name for name in read(LOCAL/'CACHE_HASHES.json')]
    for path in dependencies:
        require(plan['input_sha256'].get(str(path.resolve())) == sha(path), 'Unbound original dependency '+str(path))
    study.registered(); study.verify_source()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data: return {k:data[k].copy() for k in data.files}


def source_inputs(study, independent, plan, meta, arrays, px, op):
    """Original155 Backend.build axes, plus independent frozen source verification."""
    wiring = plan['endpoint_source_binding']['inputs']
    target = study.target
    require(Path(wiring['train_inputs']['path']) == target.P/'TRAIN_INPUTS.npz' and
            Path(wiring['training_meta']['path']) == target.old.P/'META.json', 'Original plain10 input wiring')
    cached = load_npz(Path(wiring['train_inputs']['path']))
    original_training = load_npz(target.old.P/'DATA.npz')
    training = read(Path(wiring['training_meta']['path'])); architecture = read(study.arch.P/'META.json')
    identity = study.validate_identity(training, cached, {k:original_training[k] for k in target.CACHED_KEYS}, architecture)
    target.validate_axes(meta, training, cached, arrays['eligible'])
    require(cached['train_plain'].shape == (8820,10) and cached['event_plain'].shape == (5978,10), 'Original plain10 axes')
    np.testing.assert_array_equal(cached['eligible'], arrays['eligible'])
    oracle = independent.frozen_source_oracle()
    require(identity == oracle.identity_oracle(meta,training,cached,original_training,architecture), 'Independent training identity')
    calendar_record = read(CALENDAR)
    provenance = study.calendar_provenance()
    require(provenance == plan['endpoint_source_binding']['calendar'] == calendar_record['provenance'], 'Calendar source changed')
    sessions = study.calendar_labels._CAL.market_calendar(start='2008-01-01',end='2035-12-31').sessions
    require(sessions.tz is None and sessions.is_unique and sessions.is_monotonic_increasing and
            calendar_record == dict(start='2008-01-01',end='2035-12-31',provenance=provenance,
                sessions=[str(d.date()) for d in sessions]), 'Original official session axis')
    reader = oracle.old.Reader(ROOT)
    independent_sessions = oracle.old.official_sessions(reader,'2035-12-31')
    require(sessions.equals(independent_sessions), 'Independent official calendar')
    source, info = study.arch.old.panel_load(meta['etfs'])
    raw_hashes = study.raw_source_hashes(info)
    require(raw_hashes == plan['raw_source_hashes'], 'Original raw64 identity'); verify(raw_hashes)
    parity = study.verify_panel_parity(source,px,op)
    loader_path = reader.path(oracle.old.LOADER)
    require(sha(loader_path) == oracle.old.LOADER_SHA, 'Independent raw loader SHA')
    loader = module('relative_duration225_independent_raw_loader',loader_path)
    raw_panel, raw_info = loader.load_panel(meta['etfs'])
    consumed = {str(reader.path(k)):v for k,v in raw_info['source_hashes'].items()}
    consumed[str(loader_path)] = oracle.old.LOADER_SHA
    require(consumed == raw_hashes, 'Independent raw source closure')
    prices = load_npz(Path(wiring['prices']['path']))
    require(set(prices)=={'dates','symbols','close','opening'}, 'Original price schema')
    require(pd.DatetimeIndex(prices['dates']).equals(px.index) and list(prices['symbols'])==list(px.columns), 'Original price axes')
    np.testing.assert_array_equal(prices['close'],px.to_numpy());np.testing.assert_array_equal(prices['opening'],op.to_numpy())
    require(oracle.old.price_parity(raw_panel,prices)==parity, 'Independent raw/execution price parity')
    return cached, training['training'], sessions, identity, calendar_record, dict(raw_source_hashes=raw_hashes,
        execution_prices=parity, independent_raw_prices_verified=True, training_identity_verified=True,
        official_calendar_independently_verified=True, endpoint_source_binding=plan['endpoint_source_binding'])


def fit_count(audit):
    total = audit.get('new_pooled_model_fits')
    require(type(total) is int and total >= 0 and total % 2 == 0, 'Invalid paired model count')
    annual = audit.get('annual'); require(isinstance(annual,list), 'Annual fit audit required')
    years=[]; counted=0
    for row in annual:
        require(type(row.get('year')) is int and row['year'] not in years, 'Duplicate or invalid fit year')
        years.append(row['year']); expected = 2 if row['status']=='fit' else 0
        require(row['status'] in ('fit','cold_start','no_queries') and type(row.get('model_fits')) is int and row['model_fits']==expected, 'Annual model count')
        if expected:
            require(row['training_months']>=18 and row['prediction_rows']>0 and row['shared_rows_weights_scalers'] is True, 'Invalid annual fit cohort')
        counted+=expected
    require(counted==total and audit.get('local_model_fits')==0, 'Paired count differs from actual annual fits')
    return total


def prefix_inputs(px,sessions,trainmeta,meta,cached,eligible,n):
    require(type(n) is int and 0<=n<=len(meta['dates']),'Invalid model prefix')
    short=dict(meta)
    for key in ('dates','ends','symbols'):short[key]=meta[key][:n]
    close=px.iloc[:0].copy() if n==0 else px.loc[:pd.Timestamp(meta['dates'][n-1])].copy()
    cutoff=None if n==0 else str(pd.Timestamp(meta['dates'][n-1]).year-1)+'-12-31'
    keep=np.zeros(len(trainmeta['dates']),bool) if cutoff is None else np.asarray(trainmeta['ends'],dtype='datetime64[D]')<=np.datetime64(cutoff,'D')
    widths=np.asarray([len(s) for s in trainmeta['symbols']],int);rows=np.repeat(keep,widths)
    training=dict(trainmeta)
    for key in ('dates','ends','symbols'):training[key]=[v for v,use in zip(trainmeta[key],keep) if use]
    width=eligible.shape[1]
    return (close,sessions,training,short,cached['train_plain'][rows],cached['event_plain'][:n*width],
        cached['common_good'][rows],cached['train_eligible'][rows],eligible[:n])



def state_check(state):
    require(len(state['completed']) == 222 and sum(r['trials'] for r in state['completed']) == 1183,
            'Expected222 completed rounds /1183 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round224 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_RELATIVE_DURATION225_' + tag + '_' + expected + '.json')
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
    names = ('test_monthly_relative_duration225_host.py', 'test_monthly_relative_duration225_model.py',
             'test_monthly_relative_duration225_independent.py')
    modules = [module('test_relativeduration225_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    require(result.wasSuccessful(), 'Synthetic validation failed')
    receipt = TEST_RECEIPT
    require(not receipt.exists(), 'Test receipt is write-once')
    dump(receipt, dict(passed=True, synthetic_only=True, tests=result.testsRun,
        runtime=ev.runtime(), code_sha256={str(p): sha(p) for p in code_files()}))


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
    source_binding = endpoint_source_binding()
    source_review=source_review_binding(raw);numerical_review=code_review_binding()
    for binding in (source_review,numerical_review):
        for name,digest in binding['input_sha256'].items():bind(pins,Path(name),digest)
    for name, digest in source_binding['input_sha256'].items(): bind(pins, Path(name), digest)
    bind(pins, ENDPOINT_PLAN, ENDPOINT_PLAN_SHA)
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
    plan = dict(schema='monthly_relative_duration225_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, raw_source_hashes=raw, predecessor_binding=prior_binding,
        endpoint_source_binding=source_binding,source_review_binding=source_review,numerical_code_review=numerical_review,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_feature_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_new_labels_before_registration=0,
        actual_new_returns_before_registration=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_relative_duration_labels_and_fits',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare matched uninterrupted versus total relative-lag duration annual Ridge; select only on first88 before later34 diagnostics.')
    commit_state(before, state, 'register')
    print(json.dumps(dict(status='REGISTERED', plan_sha256=sha(OUT / 'PLAN.json'))), flush=True)


def gate(expected):
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['contract'] == contract() and p['runtime'] == ev.runtime(), 'Frozen contract/runtime changed')
    verify(p['input_sha256'])
    require(p['endpoint_source_binding'] == endpoint_source_binding(), 'Registered155 source binding changed')
    require(p['source_review_binding']==source_review_binding(p['raw_source_hashes']),'Registered225 source review changed')
    require(p['numerical_code_review']==code_review_binding(),'Registered numerical review changed')
    require(sha(OUT / 'STATE_BEFORE.json') == p['prior_central_sha256'], 'Changed pre-registration state')
    state = read(CENTRAL)
    require(state['registered_queue'] == [dict(directory=OUT.name, plan_sha256=expected)], 'Central registration mismatch')
    state_check(state)
    require(p.get('predecessor_binding') == predecessor_binding(state), 'Registered predecessor bindings changed')
    return p


def scale_and_delta(values, n, eligible=None):
    require(set(values)==set(EXTRA), 'Exactly five model arrays required')
    mask=np.asarray(values['common_available']);shape=mask.shape
    require(len(shape)==2 and shape[1]>0 and mask.dtype==bool and type(n) is int and 0<=n<=shape[0], 'Invalid model shape/prefix')
    use=mask[:n]
    if eligible is not None:
        e=np.asarray(eligible);require(e.shape==shape and e.dtype==bool and not (use & ~e[:n]).any(),'Forecast outside original eligibility')
    ranks,deltas={},{}
    for arm,key in RAW.items():
        raw=np.asarray(values[key+'_prediction']);cached=np.asarray(values[key+'_rank'])
        require(raw.shape==cached.shape==shape,'Invalid prediction/rank shape')
        x=np.asarray(raw[:n],float); saved=np.asarray(cached[:n],float)
        require(not np.isinf(x).any() and np.array_equal(np.isfinite(x),use),'Both predictions finite exactly on common mask')
        expected=np.zeros((n,shape[1]))
        for i in range(n):
            count=int(use[i].sum())
            if count>1:
                signed=-x[i,use[i]]
                expected[i,use[i]]=(pd.Series(signed).rank(method='average').to_numpy()-1.)/(count-1.)-.5
        require(np.isfinite(saved).all() and np.array_equal(saved,expected),'Saved negative-prediction rank differs')
        if n:require(not use[0].any() and not expected[0].any(),'Initial partial must have unavailable forecasts')
        delta=DOSE*expected
        require(np.max(np.abs(delta),initial=0.)<=.005,'Fixed score budget exceeded')
        ranks[arm],deltas[arm]=expected,delta
    return ranks,deltas


def decisions(study, arrays, meta, values, n, return_details=False):
    policy=read(LOCAL/'INCUMBENT_POLICY.json')
    original, original_scores=study.choices(study.configs()[4],arrays,meta,n,policy=policy)
    choices={NAMES[0]:original};scores={NAMES[0]:original_scores}
    ranks,deltas=scale_and_delta(values,n,arrays['eligible'])
    for arm in NAMES[1:]:
        pre=study.modified_pre_score(study.configs()[4],arrays,n)
        view=pre[:n].copy();changed=deltas[arm]!=0.
        view[changed]+=deltas[arm][changed];pre[:n]=view
        chosen,score=study.adapter.score_and_choose('incumbent66',0.,np.full(arrays['baseline'].shape,np.nan),
            arrays,meta,n,pre_score=pre,effects=arrays['effects'],episode_policy=policy,boundary_selector=study.ep.select_boundary)
        for i in np.flatnonzero(~changed.any(axis=1)):
            require(chosen[i]==original[i] and score[i].tobytes()==original_scores[i].tobytes(),'Exact neutral incumbent fallback')
        choices[arm],scores[arm]=chosen,score
    orders={arm:study.rp.make_orders(choices[arm],scores[arm],meta) for arm in NAMES}
    return (orders,ranks,deltas,scores,choices) if return_details else (orders,ranks,deltas)


def require_build_audit(check, built, shape=(122,49), require_fits=True):
    for key in BUILD_FLAGS:require(check.get(key) is True,'Independent build evidence missing '+key)
    require(set(built)=={'arrays','labels','label_audit','model_audit','rank_audit'},'Exact model output schema')
    require(set(built['labels'])==set(LABEL_KEYS),'Seven label arrays required')
    values=built['arrays'];scale_and_delta(values,shape[0])
    require(values['common_available'].shape==shape,'Independent model shape')
    expected=dict(events=shape[0],symbols_per_event=shape[1],training_rows=len(built['labels']['dates']),
        good_training_rows=int(built['labels']['good'].sum()),available_cells=int(values['common_available'].sum()))
    for key,value in expected.items():require(type(check.get(key)) is int and check[key]==value,'Independent build coverage '+key)
    fits=fit_count(built['model_audit'])
    require(not require_fits or fits>0,'Main preparation must disclose actual nonzero paired model fits')
    for key in ('new_pooled_model_fits','independent_model_fits'):
        require(type(check.get(key)) is int and check[key]==fits,'Independent fit count '+key)
    require(check.get('computes_portfolio_returns') is False and check.get('producer_modules_imported') is False,'Build audit must remain independent and compute no portfolio returns')
    for key in PATH_COUNT_FIELDS:
        require(type(check.get(key)) is int and check[key]>=0 and check[key]==built['label_audit'].get(key),'Independent path count differs: '+key)
    return fits


def require_delta_audit(check,deltas):
    for key in ('passed','initial_delta_zero','common_available_only','common_query_population_checked','negative_prediction_ranks_checked','average_ties_checked','no_new_eligibility_filter','missing_exact_zero'):
        require(check.get(key) is True,'Independent delta evidence missing '+key)
    require(check.get('dose')==DOSE and check.get('max_absolute_correction')==.005,'Independent delta budget')
    require(check.get('changed_cells')=={RAW[k]:int(np.count_nonzero(v)) for k,v in deltas.items()},'Independent delta coverage')


def require_selection_audit(check):
    for key in ('passed','scores_and_choices_checked','one_original_episode_checked','initial_partial_fallback_checked'):
        require(check.get(key) is True,'Independent selection evidence missing '+key)
    require(check.get('ordered_choices_checked')==366,'Independent selection coverage')


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
    import monthly_relative_duration225_model as model
    import monthly_relative_duration225_independent as independent
    study=local();original_binding(study,plan);meta,arrays,px,op=study.load()
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27 and
            all(len(s)==49 for s in meta['symbols']),'Original27 arrays/122x49 required')
    before={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in arrays.items()}
    cached,trainmeta,sessions,identity,calendar,parity=source_inputs(study,independent,plan,meta,arrays,px,op)
    cached_before={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in cached.items()}
    args=(px,sessions,trainmeta,meta,cached['train_plain'],cached['event_plain'],cached['common_good'],cached['train_eligible'],arrays['eligible'])
    built=model.build(*args)
    source_cohort_exact=require_source_cohort(built,plan['source_review_binding']['training_path_coverage'])
    parity['source_cohort_exact']=source_cohort_exact
    parity['source_review_binding']=plan['source_review_binding']
    np.testing.assert_array_equal(built['labels']['ends'],np.asarray(cached['ends'],dtype='datetime64[D]'))
    require(not (built['labels']['good'] & ~cached['common_good']).any(),'Training cohort expanded')
    check=independent.verify_build(*args,built)
    fits=require_build_audit(check,built)
    values=built['arrays'];orders,ranks,deltas,scores,choices=decisions(study,arrays,meta,values,122,True)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original full order parity')
    delta_check=independent.verify_deltas(values,{RAW[k]:v for k,v in deltas.items()},dose=DOSE);require_delta_audit(delta_check,deltas)
    selection_check=independent.verify_selection(arrays,meta,{RAW[k]:v for k,v in deltas.items()},scores,choices);require_selection_audit(selection_check)
    cohort_check=cohort_parity(built);require_cohort_parity(cohort_check)
    prefix_counts=[]
    for n in FEATURE_PREFIXES:
        small_args=prefix_inputs(px,sessions,trainmeta,meta,cached,arrays['eligible'],n)
        small=model.build(*small_args);small_check=independent.verify_build(*small_args,small)
        small_fits=require_build_audit(small_check,small,shape=(n,49),require_fits=False)
        for key in EXTRA:np.testing.assert_array_equal(small['arrays'][key],values[key][:n])
        prefix_counts.append(dict(prefix=n,producer_model_fits=small_fits,independent_model_fits=small_check['independent_model_fits'],
            training_rows=len(small_args[4]),training_months=len(small_args[2]['dates']),
            training_label_end_cutoff=None if n==0 else str(pd.Timestamp(meta['dates'][n-1]).year-1)+'-12-31'))
    for n in range(123):
        oo,rr,dd,ss,cc=decisions(study,arrays,meta,values,n,True)
        require(all(oo[k]==orders[k][:n] and cc[k]==choices[k][:n] and ss[k].tobytes()==scores[k][:n].tobytes() for k in NAMES),'All-prefix score/order parity')
        for k in RAW:np.testing.assert_array_equal(rr[k],ranks[k][:n]);np.testing.assert_array_equal(dd[k],deltas[k][:n])
    require(before=={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in arrays.items()},'Original27 arrays modified')
    require(cached_before=={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in cached.items()},'Original training arrays modified')
    accounting=dict(new_model_fits=fits,new_predictive_model_fits=fits,new_pooled_model_fits=fits,
        independent_diagnostic_model_fits=check['independent_model_fits'],
        prefix_diagnostic_model_fits=sum(x['producer_model_fits'] for x in prefix_counts),
        prefix_independent_diagnostic_model_fits=sum(x['independent_model_fits'] for x in prefix_counts),
        prefix_model_fit_counts=prefix_counts,model_fit_accounting=MODEL_FIT_ACCOUNTING,
        **{key:built['label_audit'][key] for key in PATH_COUNT_FIELDS})
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FEATURES.npz',**values)
    np.savez_compressed(dest/'LABELS.npz',**built['labels'])
    np.savez_compressed(dest/'TRANSFORMS_DELTAS.npz',**{'transform_'+k:v for k,v in ranks.items()},**{'delta_'+k:v for k,v in deltas.items()})
    for name,value in [('LABEL_AUDIT.json',built['label_audit']),('MODEL_AUDIT.json',built['model_audit']),('RANK_AUDIT.json',built['rank_audit']),
            ('ORDERS.json',orders),('META.json',meta),('SOURCE_PARITY.json',parity),('IDENTITY_AUDIT.json',identity),('CALENDAR.json',calendar),('COHORT_PARITY.json',cohort_check)]:dump(dest/name,value)
    dump(dest/'INDEPENDENT_PREPARE.json',dict(model=check,delta=delta_check,selection=selection_check,cohort_parity=cohort_check,source_cohort_exact=source_cohort_exact,prefixes=123,
        feature_prefixes=list(FEATURE_PREFIXES),prefix_prices_truncated=True,prefix_training_inputs_truncated=True,original_orders_exact=True,
        original_arrays_unchanged=True,original_training_arrays_unchanged=True,original_array_count=27,**accounting))
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',plan_sha256=expected,created_utc=now(),
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,original_array_count=27,**accounting))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest/'COMPLETION.json'),new_model_fits=fits)),flush=True)


def verify_prepared_evidence(c,built,deltas,audit):
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Prepared artifact set incomplete')
    require_cohort_parity(audit['cohort_parity'])
    fits=require_build_audit(audit['model'],built);require_delta_audit(audit['delta'],deltas);require_selection_audit(audit['selection'])
    require(audit.get('source_cohort_exact') is True and audit.get('prefixes')==123 and audit.get('feature_prefixes')==list(FEATURE_PREFIXES) and audit.get('prefix_prices_truncated') is True and audit.get('prefix_training_inputs_truncated') is True
        and audit.get('original_orders_exact') is True and audit.get('original_arrays_unchanged') is True
        and audit.get('original_training_arrays_unchanged') is True and audit.get('original_array_count')==27,'Prepared invariants missing')
    prefix=audit.get('prefix_model_fit_counts');require(isinstance(prefix,list) and [x['prefix'] for x in prefix]==list(FEATURE_PREFIXES),'Prefix model accounting missing')
    for row in prefix:
        a,b=row['producer_model_fits'],row['independent_model_fits']
        require(type(a) is type(b) is int and a>=0 and a%2==0 and a==b,'Diagnostic prefix fits differ')
    for record in (audit,c):
        require(all(type(record.get(key)) is int and record[key]==built['label_audit'][key] for key in PATH_COUNT_FIELDS),'Saved path observation counts differ')
        for key in ('new_model_fits','new_predictive_model_fits','new_pooled_model_fits','independent_diagnostic_model_fits'):
            require(type(record.get(key)) is int and record[key]==fits,'Model fit receipt differs '+key)
        require(record.get('prefix_model_fit_counts')==prefix and record.get('prefix_diagnostic_model_fits')==sum(x['producer_model_fits'] for x in prefix)
            and record.get('prefix_independent_diagnostic_model_fits')==sum(x['independent_model_fits'] for x in prefix)
            and record.get('model_fit_accounting')==MODEL_FIT_ACCOUNTING,'Diagnostic fit accounting differs')
    require(c.get('new_returns_computed') is False and c.get('original_array_count')==27,'Prepared return/array scope')


def prepared(expected):
    gate(expected);dest=OUT/'prepared';c=read(dest/'COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and set(c['artifacts_sha256'])==PREPARED_FILES,'Preparation incomplete')
    for name,digest in c['artifacts_sha256'].items():require(sha(dest/name)==digest,'Changed prepared artifact '+name)
    built=dict(arrays=load_npz(dest/'FEATURES.npz'),labels=load_npz(dest/'LABELS.npz'),
        label_audit=read(dest/'LABEL_AUDIT.json'),model_audit=read(dest/'MODEL_AUDIT.json'),rank_audit=read(dest/'RANK_AUDIT.json'))
    transformed=load_npz(dest/'TRANSFORMS_DELTAS.npz');ranks,deltas=scale_and_delta(built['arrays'],122)
    require(set(transformed)=={'transform_'+k for k in RAW}|{'delta_'+k for k in RAW},'Prepared transform schema')
    for key in RAW:
        np.testing.assert_array_equal(transformed['transform_'+key],ranks[key]);np.testing.assert_array_equal(transformed['delta_'+key],deltas[key])
    audit=read(dest/'INDEPENDENT_PREPARE.json');cohort=read(dest/'COHORT_PARITY.json');orders=read(dest/'ORDERS.json')
    require(cohort==audit.get('cohort_parity')==cohort_parity(built),'Prepared source cohort parity receipt changed')
    verify_prepared_evidence(c,built,deltas,audit)
    return read(dest/'META.json'),orders


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
        close_vs_relative_incidence=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_relative_incidence=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
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
    import monthly_relative_duration225_independent as independent
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
    import monthly_relative_duration225_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, relative_lag_duration_target=True, same_support_both_arms=True, observer_promotion=False))
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
        result_completion_sha256=sha(OUT / 'results/COMPLETION.json'), new_trials=3, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        original_array_count=27, protected_files=len(protected), best_training_selected_candidate=best,
        target_reached=best['cagr'] >= .7, observers_modified=False,
        **{key: read(OUT / 'prepared/COMPLETION.json')[key] for key in ('new_model_fits','new_predictive_model_fits','new_pooled_model_fits','independent_diagnostic_model_fits','prefix_diagnostic_model_fits','prefix_independent_diagnostic_model_fits','prefix_model_fit_counts')+PATH_COUNT_FIELDS})
    dump(OUT / 'COMPLETION.json', receipt)
    state['completed'].append(dict(directory=OUT.name, trials=3, completed_utc=now(), primary=pick,
        primary_cagr=metrics['cagr'], primary_sharpe=metrics['sharpe'], new_admitted=int(results['selection']['admitted']),
        independent_execution_paths=9, protected_files=len(protected), target_reached=receipt['target_reached'],
        completion=dict(path=str(OUT / 'COMPLETION.json'), sha256=sha(OUT / 'COMPLETION.json'))))
    state.update(best_training_selected_candidate=best, latest_training_selected_candidate=dict(directory=OUT.name,
        arm=pick, selection=results['selection'], full_period_cagr=metrics['cagr'], full_period_sharpe=metrics['sharpe'], selected_on_first88=True),
        best_tracking_note=contract()['historical_best_tracking'], registered_queue=[], current_status='completed_independently_verified',
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Review uninterrupted relative-lag duration versus total lag incidence and matched Ridge evidence; preserve frozen observers.')
    require(len(state['completed']) == 223 and sum(r['trials'] for r in state['completed']) == 1186, 'Wrong final counters')
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

"""Round220: frozen Local67 with a bounded three-year own-report ROA stability correction.

All actual features and portfolio returns require a prospectively bound PLAN.
Only declared-assumption report versions may supply the small original pre-score correction.
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
import monthly_roa_stability220_scoring as scoring

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_roa_stability220'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_risk_consensus219'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
SOURCE_CALENDAR = ROOT / 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
SOURCE_CALENDAR_SHA = 'c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67', 'roa_stability_penalty', 'three_year_roa_mean_control')
RAW = {NAMES[1]: 'candidate', NAMES[2]: 'control'}
TRAIN, DOSE, MAX_DELTA = 88, .01, .01
FEATURE_PREFIXES = (0, 1, 53, 88, 122)
ESTIMATE_FIELDS = ('annual_ratio_evaluations','mean_estimates','sample_dispersion_estimates','bounded_scores')
MODEL_FIT_ACCOUNTING = 'Zero new predictive fits or learned parameters. Each common available cell contributes three annual own-report ROA ratios, one three-year mean, one sample dispersion and two bounded scores. Prefix/oracle repeats excluded.'
CONTEXT = ROOT / 'work/monthly_roa_stability220_context.json'
CONTEXT_SHA = '673aa44502d61490fef26a16a426cefa94469cdae9f059948bb471458e7370e5'
SOURCE_REVIEW = ROOT / 'work/roa_stability220_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA = '8dfbfd87bb91d703d5712024ee79edb8a695645e12da098274b93e15c9793a19'
COVERAGE = ROOT / 'work/after219_roa_stability_coverage.json'
COVERAGE_SHA = '98fa98c4a1328ef131f794d05094e325377b6c8a4caf0c4a4f2cc8682258f564'
PRIOR_EXPECTED = {
 'PLAN.json':'8c745473754e4ffb54b1a0ab4c1a22c04a10f576b4963528c7d5df1803c291e8',
 'COMPLETION.json':'9c70ef8763dfe6dbcb31d5bee932259212cf538d4f2264d45e8b6cfb17b0ccb6',
 'results/COMPLETION.json':'6ca8b9f5bfd6bf797b70bd1ba7eaaafe78f2f000edff9ce68c412d9f6e7627e6',
 'RESEARCH_REVIEW.json':'41fb671c7d42596c2fa16ef75736659789dcff82ac3f4674a71f7faf4eeca776',
} 

FEATURE_FLAGS = ('passed','independent_scalar_rebuild','full_catalog_latest_fiscal_anchor_checked','no_older_fallback_checked','exact_identity_checked','three_consecutive_original_reports_checked','edge_version_clocks_checked','original_fields_only_checked','own_report_assets_checked','latest550_and_historical_age_checked','rational_roa_mean_variance_checked','sample_ddof1_checked','binary64_sqrt_contract_checked','bounded_transform_checked','exact_common_availability','no_cross_sectional_ranking','no_predictive_fits_checked','audit_checked','no_future_or_outcome_inputs')
DELTA_FLAGS = ('passed','independent_delta_rebuild','fixed_point01_formula_checked','eligible_only_checked','zero_or_missing_zero_delta_checked','exact_targets_checked','no_second_rank','correction_bound_checked','penalty_monotonicity_checked')
SOURCE_FLAGS = ('source_only','source_admission_metadata_only','raw_source_hashes_verified','raw_source_map_matches_original','financial_package_hashes_verified','complete_supplied_catalog_preserved','original_single_report_admissions_preserved','conditional_chain_policy_reviewed','nine_field_source_bindings_reviewed','source_clock_assumptions_explicit','integration_source_verified','metadata_axes_verified')
SOURCE_DIR = ROOT / 'outputs/monthly_roa_stability_source220_v1'
SOURCE_MANIFEST_SHA = '328b19f5e30406dcec2094b0302f83d97a7c303fae5fa9ca0f8500ee861dffac'
SOURCE_DATASET_SHA = 'f69dca48d6b5aba713dfa94c5c9435b5eedb90ed6503752cdd077b4b1afc5ad6'
SOURCE_FILES = {'POLICY.json','DATASET.json','REVIEW_EVIDENCE.json','CLOCK_REVIEW.json','VALIDATION.json'}
TEST_RECEIPT = ROOT / 'work/monthly_roa_stability220_TESTS.json'
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
    return module('monthly_roa_stability220_original', LOCAL / 'study.py')


def contract():
    require(sha(CONTEXT)==CONTEXT_SHA, 'Changed prospective context')
    require(sha(COVERAGE)==COVERAGE_SHA, 'Changed prospective coverage')
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,arms=list(NAMES),candidate=NAMES[1],
        control_nonpromotable=NAMES[2],period=['2016-08-17','2026-09-24'],training_months=88,later_months=34,
        dose=DOSE,max_absolute_delta=MAX_DELTA,
        scoring='Eligible finite nonzero h enters original pre-score as .01*h; all-NaN rank input and zero blend retain original core/support; one original episode. No cross-sectional re-ranking.',
        feature_prefix_source='Physically truncate catalog, report, edge and chain versions to the last prefix signal; metadata axes also truncated; empty prefix0.',
        verification=dict(independent_all_cells=True,feature_prefixes=list(FEATURE_PREFIXES),all123_order_prefixes=True),
        limitation=context['limitations'],strict_pit=False,
        historical_best_tracking='First88 selects; full-period best is descriptive among previously training-selected research candidates, not pristine OOS or observer promotion.',
        new_model_fits=0,new_predictive_model_fits=0,predictive_model_fits=0,model_fit_accounting=MODEL_FIT_ACCOUNTING,
        new_arms=2,registered_trials=3,execution_paths=9,no_observer_writes=True,original_pool_preserved=True,no_parameter_search=True)


def code_files():
    return [ROOT/'work'/name for name in ('monthly_roa_stability220_host.py',
        'test_monthly_roa_stability220_host.py','monthly_roa_stability220_kernel.py',
        'test_monthly_roa_stability220_kernel.py','monthly_roa_stability220_independent.py',
        'test_monthly_roa_stability220_independent.py','monthly_roa_stability220_scoring.py',
        'test_monthly_roa_stability220_scoring.py','roa_stability_preflight_math.py',
        'test_roa_stability_preflight_math.py','monthly_roa_stability220_context.json',
        'after219_roa_stability_coverage.json','ptr169_rebuild_evaluate.py','ptr169_results_independent.py',
        'price_distribution171_independent.py','monthly_sector_recurrence192_independent.py')]+[
        ROOT/'outputs/monthly_core_timing70_research/core_adapter.py',
        ROOT/'outputs/monthly_core_timing70_research/test_core_adapter.py',
        ROOT/'outputs/monthly_path70_research/incumbent_adapter.py',
        ROOT/'outputs/monthly_daily_episodes/episodes.py']


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


def financial_source_binding(raw):
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Changed source review')
    review=read(SOURCE_REVIEW);context=read(CONTEXT)
    require(review.get('passed') is True and review.get('context_sha256')==CONTEXT_SHA,'Source admission/context differs')
    require(review.get('raw_source_hashes')==raw and len(raw)==64,'Original64 execution-source lineage changed')
    require(SOURCE_FLAGS and not any(key.startswith('PENDING') for key in SOURCE_FLAGS),'Source receipt contract not finalized')
    for key in SOURCE_FLAGS:require(review.get(key) is True,'Source review evidence missing '+key)
    for key in ('actual_new_features_computed','actual_new_orders_computed','actual_new_returns_computed'):
        require(review.get(key) is False,'Source audit crossed strategy boundary: '+key)
    pins={str(SOURCE_REVIEW):SOURCE_REVIEW_SHA}
    for label,document,key in (('source inputs',review,'input_sha256'),('source evidence',review,'evidence_sha256'),('context evidence',context,'evidence_sha256')):
        mapping=document.get(key)
        require(isinstance(mapping,dict) and mapping,'Nonempty '+label+' required')
        for name,digest in mapping.items():bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    manifest=read(SOURCE_DIR/'MANIFEST.json')
    require(sha(SOURCE_DIR/'MANIFEST.json')==SOURCE_MANIFEST_SHA,'Changed financial source manifest')
    require(manifest.get('source_only') is True and manifest.get('write_once') is True,'Source-only immutable package required')
    require(set(manifest.get('files',{}))==SOURCE_FILES,'Exact five source package files required')
    require(manifest['files']['DATASET.json']==SOURCE_DATASET_SHA,'Changed normalized financial dataset')
    bind(pins,SOURCE_DIR/'MANIFEST.json',SOURCE_MANIFEST_SHA)
    for name,digest in manifest['files'].items():bind(pins,SOURCE_DIR/name,digest)
    require(isinstance(manifest.get('input_sha256'),dict) and manifest['input_sha256'],'Source package closure required')
    for name,digest in manifest['input_sha256'].items():bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    clock=read(SOURCE_DIR/'CLOCK_REVIEW.json')
    require(clock.get('calendar_ref')==str(SOURCE_CALENDAR.relative_to(ROOT)) and clock.get('calendar_sha256')==SOURCE_CALENDAR_SHA,'Original financial availability calendar differs')
    require(clock.get('no_outside_calendar_clock_invented') is True and type(clock.get('actual_signal_queries')) is int and clock['actual_signal_queries']==0,'Source clocks crossed monthly selection boundary')
    bind(pins,SOURCE_CALENDAR,SOURCE_CALENDAR_SHA)
    validation=read(SOURCE_DIR/'VALIDATION.json');policy=read(SOURCE_DIR/'POLICY.json')
    require(validation.get('passed') is True and validation.get('source_only') is True,'Financial source validation incomplete')
    for key in ('raw_digest_bindings','own_report_three_field_bindings','original_field_admissions_preserved','exact_identity_no_ticker_join','reported_NI_and_assets_checked_independently','admitted_edge_exact_scaled_source_equality','event_versions_not_deduplicated_by_raw_sha','complete_supplied_catalog_preserved','no_first_public_PIT_certification'):
        require(validation.get('checks',{}).get(key) is True,'Financial source check missing '+key)
    for key in ('ROA_calculations','means','standard_deviations','factors','ranks','monthly_support_matches','orders','returns','network_requests','source_admission_updates_outside_new_directory','observer_changes'):
        require(type(validation['checks'].get(key)) is int and validation['checks'][key]==0,'Source package performed strategy work: '+key)
    require(policy.get('no_strategy_registration') is True,'Source package cannot register a strategy')
    for key in ('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'):
        require(policy.get(key) is False,'Source limitations changed: '+key)
    verify(pins)
    return dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,
        source_evidence_sha256=review['evidence_sha256'],context_evidence_sha256=context['evidence_sha256'],
        manifest_sha256=SOURCE_MANIFEST_SHA,dataset_sha256=SOURCE_DATASET_SHA,
        source_only=True,monthly_support_computed=False,strict_pit=False)


def load_source_dataset(binding):
    require(binding.get('dataset_sha256')==SOURCE_DATASET_SHA and binding.get('manifest_sha256')==SOURCE_MANIFEST_SHA,'Source package binding differs')
    require(sha(SOURCE_DIR/'DATASET.json')==SOURCE_DATASET_SHA,'Changed source dataset')
    dataset=read(SOURCE_DIR/'DATASET.json');manifest=read(SOURCE_DIR/'MANIFEST.json')
    require(dataset.get('schema_version')==1 and dataset.get('method')=='as_reported_original_NI_assets_three_annual_reports_v1','Normalized source schema changed')
    require(dataset.get('source_pins')==manifest['input_sha256'],'Dataset/manifest source closure differs')
    require(dataset.get('conditional_research_only') is True,'Conditional source scope missing')
    for key in ('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'):
        require(dataset.get(key) is False,'Dataset limitations changed: '+key)
    return dataset


def require_source_support(audit,meta,values):
    events=audit.get('event_history');n=len(meta['dates'])
    require(isinstance(events,list) and len(events)==n,'Financial source history length')
    total=0
    for i,event in enumerate(events):
        require(event.get('date')==meta['dates'][i] and event.get('symbols')==meta['symbols'][i],'Financial query identity differs')
        cells=event.get('cells');require(isinstance(cells,list) and len(cells)==len(meta['symbols'][i]),'Financial cell width differs')
        count=0
        for j,cell in enumerate(cells):
            require(cell.get('date')==meta['dates'][i] and cell.get('symbol')==meta['symbols'][i][j],'Financial cell identity differs')
            require(type(cell.get('available')) is bool and cell['available']==bool(values['available'][i,j]),'Financial availability differs')
            require(isinstance(cell.get('reason'),str) and cell['reason'],'Financial reason missing')
            if cell['available']:
                require(cell['symbol'] not in meta['etfs'],'ETF financial observation forbidden')
                require(len(cell.get('report_version_ids',[]))==3 and len(set(cell['report_version_ids']))==3 and len(cell.get('edge_version_ids',[]))==2 and len(set(cell['edge_version_ids']))==2,'Exact three reports/two edges required')
                require(isinstance(cell.get('latest_event_id'),str) and isinstance(cell.get('chain_version_id'),str),'Selected source identity missing')
                measurement=cell.get('measurement',{})
                require(measurement.get('candidate')==float(values['candidate'][i,j]) and measurement.get('control')==float(values['control'][i,j]),'Stored measurement/feature mismatch')
                count+=1
            else:
                require('measurement' not in cell,'Unavailable cell cannot carry a measurement')
        require(type(event.get('available_count')) is int and event['available_count']==count,'Event support count differs')
        total+=count
    require(audit.get('available_cells')==total,'Financial support total differs')
    return True


def feature_prefix_sources(kernel,dataset,meta,n):
    require(type(n) is int and 0<=n<=len(meta['dates']),'Invalid financial source prefix')
    return kernel.truncate_sources(dataset,meta['dates'][n-1] if n else None)


def estimate_counts(check):
    n=check.get('events');width=check.get('symbols_per_event');a=check.get('available_cells')
    require(type(n) is int and n>=0 and type(width) is int and width>=0 and type(a) is int and 0<=a<=n*width,'Invalid financial feature coverage')
    require(type(check.get('checked_cells')) is int and check['checked_cells']==n*width and type(check.get('unavailable_cells')) is int and check['unavailable_cells']==n*width-a,'Financial missing coverage differs')
    expected=dict(annual_ratio_evaluations=3*a,mean_estimates=a,sample_dispersion_estimates=a,bounded_scores=2*a)
    require(all(type(check.get(k)) is int and check[k]==v for k,v in expected.items()),'Financial estimate accounting differs')
    return expected


def feature_prefix_meta(meta,n):
    require(type(n) is int and 0<=n<=len(meta['dates']),'Invalid metadata prefix')
    result=dict(meta)
    for key in ('dates','ends','symbols'):
        if key in meta:
            require(len(meta[key])==len(meta['dates']),'Metadata axis differs: '+key)
            result[key]=meta[key][:n]
    return result


def state_check(state):
    require(len(state['completed']) == 217 and sum(r['trials'] for r in state['completed']) == 1168,
            'Expected217 completed entries /1168 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round219 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_ROA_STABILITY220_' + tag + '_' + expected + '.json')
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
    names = ('test_monthly_roa_stability220_host.py', 'test_monthly_roa_stability220_kernel.py',
             'test_monthly_roa_stability220_independent.py','test_monthly_roa_stability220_scoring.py',
             'test_roa_stability_preflight_math.py')
    modules = [module('test_roa_stability220_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
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
    path = ROOT / 'work/roa_stability220_CODE_REVIEW.json'
    review = read(path)
    expected = {str(p.relative_to(ROOT)): sha(p) for p in code_files()[:10]}
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
    prospective=contract()
    before = CENTRAL.read_bytes(); state = json.loads(before); state_check(state)
    require(not state['registered_queue'], 'Existing registered experiment')
    prior_binding = predecessor_binding(state)
    validation = TEST_RECEIPT; t = read(validation)
    require(t['passed'] is True and t['synthetic_only'] is True and t['runtime'] == ev.runtime(), 'Synthetic tests required')
    require(t['code_sha256'] == {str(p): sha(p) for p in code_files()}, 'Code changed since tests')
    pins = dict(read(PRIOR / 'PLAN.json')['input_sha256'])
    raw = read(LOCAL / 'PLAN.json')['raw_source_hashes']
    for name, digest in prior_binding['input_sha256'].items(): bind(pins, Path(name), digest)
    source_binding = financial_source_binding(raw)
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
    plan = dict(schema='monthly_roa_stability220_v1', created_utc=now(), runtime=ev.runtime(),
        contract=prospective, input_sha256=pins, raw_source_hashes=raw, predecessor_binding=prior_binding,
        financial_source_binding=source_binding,numerical_code_review=numerical_review,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_feature_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_annual_ratio_evaluations_before_registration=0, actual_sample_dispersion_estimates_before_registration=0,
        actual_bounded_scores_before_registration=0, actual_monthly_source_selections_before_registration=0,
        actual_new_returns_before_registration=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_roa_stability_features',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare as-reported three-year ROA mean-minus-dispersion versus mean corrections; select first88 before later34 diagnostics.')
    commit_state(before, state, 'register')
    print(json.dumps(dict(status='REGISTERED', plan_sha256=sha(OUT / 'PLAN.json'))), flush=True)


def gate(expected):
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['contract'] == contract() and p['runtime'] == ev.runtime(), 'Frozen contract/runtime changed')
    verify(p['input_sha256'])
    require(p['financial_source_binding'] == financial_source_binding(p['raw_source_hashes']), 'Registered financial source binding changed')
    require(p['numerical_code_review']==code_review_binding(),'Registered independent code review changed')
    require(sha(OUT / 'STATE_BEFORE.json') == p['prior_central_sha256'], 'Changed pre-registration state')
    state = read(CENTRAL)
    require(state['registered_queue'] == [dict(directory=OUT.name, plan_sha256=expected)], 'Central registration mismatch')
    state_check(state)
    require(p.get('predecessor_binding') == predecessor_binding(state), 'Registered predecessor bindings changed')
    return p


def scale_and_delta(values,arrays,n):
    return scoring.scale_and_delta(values,arrays,n)


def core_adapter():
    return module('roa_stability220_frozen_core_adapter',ROOT/'outputs/monthly_core_timing70_research/core_adapter.py')


def decisions(study,arrays,meta,values,n):
    return scoring.decisions(study,arrays,meta,values,n,
        policy=read(LOCAL/'INCUMBENT_POLICY.json'),adapter=core_adapter())


def require_feature_audit(check,values):
    require(FEATURE_FLAGS and not any(k.startswith('PENDING') for k in FEATURE_FLAGS),'Independent feature contract not finalized')
    for key in FEATURE_FLAGS:require(check.get(key) is True,'Independent feature evidence missing '+key)
    require(set(values)=={'available','candidate','control'},'Feature receipt schema differs')
    mask=np.asarray(values['available'])
    require(mask.shape==(122,49) and mask.dtype==np.dtype(bool),'Independent full feature axes required')
    expected=dict(events=122,symbols_per_event=49,checked_cells=5978,available_cells=int(mask.sum()),unavailable_cells=int((~mask).sum()))
    for key,value in expected.items():require(type(check.get(key)) is int and check[key]==value,'Independent coverage differs: '+key)
    require(check.get('computes_returns') is False and check.get('producer_modules_imported') is False,'Independent feature scope differs')
    for key in ('predictive_model_fits','new_model_fits'):
        require(type(check.get(key)) is int and check[key]==0,'No predictive fits allowed')
    require(type(check.get('numerical_tolerance')) in (int,float) and check['numerical_tolerance']==0.,'Canonical feature comparison must be exact')
    estimate_counts(check)


def require_delta_audit(check,deltas):
    require(DELTA_FLAGS and not any(k.startswith('PENDING') for k in DELTA_FLAGS),'Independent delta contract not finalized')
    for key in DELTA_FLAGS:require(check.get(key) is True,'Independent delta evidence missing '+key)
    require(check.get('dose')==DOSE and check.get('max_absolute_delta')==MAX_DELTA,'Independent fixed budget differs')
    require(check.get('changed_cells')=={k:int(np.count_nonzero(v)) for k,v in deltas.items()},'Independent delta coverage differs')


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
    import monthly_roa_stability220_kernel as kernel
    import monthly_roa_stability220_independent as independent
    study=local();meta,arrays,px,op=study.load()
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49),'Original122x49 required')
    require(len(arrays)==27 and all(len(row)==49 for row in meta['symbols']),'Original27 arrays/candidate axes required')
    before={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in arrays.items()}
    source,info=study.arch.old.panel_load(meta['etfs']);raw_hashes=study.raw_source_hashes(info)
    require(raw_hashes==plan['raw_source_hashes'],'Original execution-price source lineage changed');verify(raw_hashes)
    parity=study.verify_panel_parity(source,px,op)
    dataset=load_source_dataset(plan['financial_source_binding']);calendar=read(SOURCE_CALENDAR)['sessions']
    values,audit=kernel.build_features(dataset,meta,calendar)
    check=independent.verify_features(dataset,meta,calendar,values,audit);require_feature_audit(check,values)
    require(check['new_model_fits']==audit['new_model_fits']==check['predictive_model_fits']==audit['predictive_model_fits']==0,'Model accounting differs')
    counts=estimate_counts(check);require(counts==estimate_counts(audit),'Financial estimate accounting differs')
    require_source_support(audit,meta,values)
    orders,targets,deltas=decisions(study,arrays,meta,values,122)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original full order parity failed')
    delta_check=independent.verify_deltas(values,arrays,targets,deltas);require_delta_audit(delta_check,deltas)
    for n in FEATURE_PREFIXES:
        small_source=feature_prefix_sources(kernel,dataset,meta,n);small_meta=feature_prefix_meta(meta,n)
        small,small_audit=kernel.build_features(small_source,small_meta,calendar)
        prefix_check=independent.verify_features(small_source,small_meta,calendar,small,small_audit)
        require(prefix_check.get('passed') is True,'Physical source prefix independently failed')
        require(small_audit['event_history']==audit['event_history'][:n],'Physical financial source prefix audit differs')
        for key in values:
            if n:np.testing.assert_array_equal(small[key],values[key][:n])
            else:require(small[key].shape==(0,0),'Zero physical prefix must have no future query width')
    for n in range(123):
        oo,tt,dd=decisions(study,arrays,meta,values,n)
        require(all(oo[k]==orders[k][:n] for k in NAMES),'Prefix order parity failed')
        for k in RAW:np.testing.assert_array_equal(tt[k],targets[k][:n]);np.testing.assert_array_equal(dd[k],deltas[k][:n])
    require(before=={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in arrays.items()},'Original arrays modified')
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FEATURES.npz',**values)
    np.savez_compressed(dest/'TRANSFORMS_DELTAS.npz',**{'transform_'+k:v for k,v in targets.items()},**{'delta_'+k:v for k,v in deltas.items()})
    dump(dest/'FEATURE_AUDIT.json',audit);dump(dest/'ORDERS.json',orders);dump(dest/'META.json',meta)
    dump(dest/'SOURCE_PARITY.json',dict(raw_source_hashes=raw_hashes,financial_source_binding=plan['financial_source_binding'],
        financial_source_manifest_sha256=SOURCE_MANIFEST_SHA,financial_source_dataset_sha256=SOURCE_DATASET_SHA,
        financial_features_from_versioned_original_reports=True,source_selection_independently_checked=True,
        no_fresh_raw_pit_claim=True,execution_prices=parity))
    common=dict(new_model_fits=0,predictive_model_fits=0,new_predictive_model_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING,original_array_count=27,**counts)
    dump(dest/'INDEPENDENT_PREPARE.json',dict(common,feature=check,delta=delta_check,prefixes=123,
        feature_prefixes=list(FEATURE_PREFIXES),feature_prefix_sources_truncated=True,
        feature_prefix_metadata_truncated=True,feature_prefix_independently_verified=True,
        financial_source_selection_exact=True,original_core_metadata_and_support_preserved=True,
        single_original_episode=True,original_orders_exact=True,original_arrays_unchanged=True))
    dump(dest/'COMPLETION.json',dict(common,status='COMPLETE',plan_sha256=expected,created_utc=now(),
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def verify_prepared_evidence(c,values,deltas,audit):
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Prepared artifact set incomplete')
    require_feature_audit(audit['feature'],values);require_delta_audit(audit['delta'],deltas)
    require(audit.get('prefixes')==123 and audit.get('feature_prefixes')==list(FEATURE_PREFIXES),'Prepared prefix coverage differs')
    for key in ('feature_prefix_sources_truncated','feature_prefix_metadata_truncated','feature_prefix_independently_verified',
            'financial_source_selection_exact','original_orders_exact','original_core_metadata_and_support_preserved',
            'single_original_episode','original_arrays_unchanged'):
        require(audit.get(key) is True,'Prepared invariant not verified: '+key)
    counts=estimate_counts(audit['feature'])
    for record in (audit,c):
        require(record.get('original_array_count')==27 and record.get('model_fit_accounting')==MODEL_FIT_ACCOUNTING,'Prepared original arrays/accounting differs')
        for key in ('new_model_fits','predictive_model_fits','new_predictive_model_fits'):
            require(type(record.get(key)) is int and record[key]==0,'Prepared predictive fits not verified')
        require(all(type(record.get(k)) is int and record[k]==v for k,v in counts.items()),'Prepared estimate counts differ')
    require(c.get('new_returns_computed') is False,'Prepare computed portfolio returns')


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
        close_vs_mean_control=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_mean_control=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
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
    import monthly_roa_stability220_independent as independent
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
    import monthly_roa_stability220_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, roa_stability=True, same_support_both_arms=True, observer_promotion=False))
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
        next_step='Review as-reported financial stability and source-clock limitations; preserve frozen observers.')
    require(len(state['completed']) == 218 and sum(r['trials'] for r in state['completed']) == 1171, 'Wrong final counters')
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


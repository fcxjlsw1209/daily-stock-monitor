"""Round233 exact own-sector signed magnitude-rank auxiliary experiment.
Actual features, orders and returns require a prospectively pinned PLAN.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, io, json, os, tempfile, unittest
import numpy as np
import pandas as pd
import ptr169_rebuild_evaluate as ev
from monthly_sector_signedrank233_independent import FEATURE_FLAGS, DELTA_FLAGS, SELECTION_FLAGS, COUNT_FIELDS
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_sector_signedrank233'
LOCAL=ROOT/'outputs/monthly_local_linear70'
PRIOR=ROOT/'outputs/monthly_volume_slope232'
CONTROL192=ROOT/'outputs/monthly_sector_recurrence192'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES=('incumbent_local67','sector_signed_magnitude','sector_sign_control')
RAW={NAMES[1]:'candidate',NAMES[2]:'control'}
TRAIN,DOSE,MAX_RAW,MAX_DELTA=88,.025,.75,.01875
FEATURE_PREFIXES=(0,1,53,88,122)
MODEL_PREFIXES=FEATURE_PREFIXES
SECTOR_ETFS=('XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY')
MAPPING_PATH=ROOT/'outputs/monthly_strict_relative/strict_study.py'
MAPPING_SHA='ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
SECTORS_PATH=ROOT/'outputs/index_union/alias_resolved_inputs/sectors.csv.gz'
ETF_MANIFESTS=tuple(ROOT/p for p in ('outputs/four_factor_rebound/ETF_SOURCES.json','outputs/monthly_reconstruction_v2/ETF_MANIFEST.json','outputs/monthly_reconstruction_v2/foreign/ETF_MANIFEST.json'))
MODEL_FIT_ACCOUNTING='Zero new predictive or descriptive model fits. Each available cell uses506 stored prices,12 exact relative signs,12 doubled-average magnitude ranks,66 unordered magnitude comparisons and two fixed bounded summaries. Prefix and independent rebuild work is separately recorded. No fitted parameters.'
ESTIMATE_FIELDS=COUNT_FIELDS
CONTEXT=ROOT/'work/monthly_sector_signedrank233_context.json'
CONTEXT_SHA='565c11f6a0343f9c321c57816a0becb68ee236f403a9a802098d26fa1d0d87ab'
COVERAGE=ROOT/'work/sector_signedrank233_COVERAGE.json'
COVERAGE_SHA='73662f9fa137176ad9a3e1a0b3869a6b802559e560e70c8cfb40b6715ae2e3cf'
SOURCE_REVIEW=ROOT/'work/sector_signedrank233_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA='85257cd82248ba86dee631cedac9726ecd19a632f53de0df39a3c068e6ea6c03'
TEST_RECEIPT=ROOT/'work/monthly_sector_signedrank233_TESTS.json'
EXTERNAL_RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PRIOR_EXPECTED={'PLAN.json':'268c385aef1845ecdf6b244c7a3fe99c5a3f7fff22b73d315d54a8bdf6034256','COMPLETION.json':'8f85e2fbba3151af26f279d02c02552dfaa34d2c86ad1c24161d8fc523e8dd81','results/COMPLETION.json':'6bf89efde2adabbe9b333d5692bc61bc03c7cefbe9c4a8f4f46e29be37779ccd','RESEARCH_REVIEW.json':'ba29648ec9e6cacda1cb5b6631c1cd1d2b196c9af5a6977a0444e332d1630494'}
PREPARED_FILES={'FEATURES.npz','TRANSFORMS_DELTAS.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={arm+'_'+path[0]+'_'+kind+'.json' for arm in NAMES for path in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}
sha,read,require,dump=ev.sha,ev.read,ev.require,ev.dump


def now():
    return datetime.now(timezone.utc).isoformat()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def local():
    return module('monthly_sector_signedrank233_original', LOCAL / 'study.py')


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


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_SECTOR_SIGNEDRANK233_' + tag + '_' + expected + '.json')
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


def code_review_binding():
    path = ROOT / 'work/sector_signedrank233_CODE_REVIEW.json'
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


def gate(expected):
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['contract'] == contract() and p['runtime'] == ev.runtime(), 'Frozen contract/runtime changed')
    verify(p['input_sha256'])
    require(p['sector_source_binding']==sector_source_binding(p['raw_source_hashes']),'Registered sector source changed')
    require(p['price_source_binding'] == price_source_binding(p['raw_source_hashes']), 'Registered model source binding changed')
    require(p['numerical_code_review']==code_review_binding(),'Registered independent code review changed')
    require(sha(OUT / 'STATE_BEFORE.json') == p['prior_central_sha256'], 'Changed pre-registration state')
    state = read(CENTRAL)
    require(state['registered_queue'] == [dict(directory=OUT.name, plan_sha256=expected)], 'Central registration mismatch')
    state_check(state)
    require(p.get('predecessor_binding') == predecessor_binding(state), 'Registered predecessor bindings changed')
    return p


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
        open_vs_incumbent=guard(metrics[NAMES[1]]['next_open_zero'], metrics[NAMES[0]]['next_open_zero'], 0.),
        close_vs_sign_control=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_sign_control=guard(metrics[NAMES[1]]['next_open_zero'], metrics[NAMES[2]]['next_open_zero'], 0.),
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
        for path, mode, cost in (ev.PATHS[0], ev.PATHS[1]):
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
    import monthly_sector_signedrank233_independent as independent
    helper = independent.accounting()
    require(set(training) == set(NAMES), 'Training arms differ')
    for arm in NAMES:
        require(set(training[arm]) == {'same_close_zero', 'next_open_zero'}, 'Training paths differ')
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
    import monthly_sector_signedrank233_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, sector_signed_magnitude_rank=True, control192_orders_exact=True, original_episode_preserved=True, zero_cost_selection=True, next_open_cost_diagnostic_only=True, same_support_both_arms=True, observer_promotion=False))
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


def tests():
    require(not TEST_RECEIPT.exists(),'Formal synthetic test receipt is write-once')
    names=('test_monthly_sector_signedrank233_host.py','test_monthly_sector_signedrank233_kernel.py',
        'test_monthly_sector_signedrank233_independent.py')
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module('signedrank233test'+str(i),ROOT/'work'/name))
                             for i,name in enumerate(names)])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True);require(result.wasSuccessful(),'Synthetic tests failed')
    dump(TEST_RECEIPT,dict(passed=True,tests=result.testsRun,synthetic_only=True,runtime=ev.runtime(),
        code_sha256={str(p):sha(p) for p in code_files()}))


def slice_arrays(arrays,n):
    shape=np.asarray(arrays['baseline']).shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Invalid query prefix')
    out={key:value[:n].copy() if np.asarray(value).ndim and len(value)==shape[0] else value.copy()
         for key,value in arrays.items()}
    for key in ('eligible','forecast_available','model_available','extension_available','state_weak_narrow','state_calm10'):
        if key in out:
            require(all(isinstance(v,(bool,np.bool_)) for v in np.asarray(out[key]).flat),'Exact boolean values required: '+key)
            out[key]=np.asarray(out[key],dtype=bool)
    return out


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
        result_completion_sha256=sha(OUT / 'results/COMPLETION.json'), new_trials=3, model_counts=read(OUT/'prepared/COMPLETION.json')['model_counts'],
        model_fit_accounting=contract()['model_fit_accounting'],
        original_array_count=27, protected_files=len(protected), best_training_selected_candidate=best,
        target_reached=best['cagr'] >= .7, observers_modified=False)
    dump(OUT / 'COMPLETION.json', receipt)
    state['completed'].append(dict(directory=OUT.name, trials=3, completed_utc=now(), primary=pick,
        primary_cagr=metrics['cagr'], primary_sharpe=metrics['sharpe'], new_admitted=int(results['selection']['admitted']),
        independent_execution_paths=9, protected_files=len(protected), target_reached=receipt['target_reached'],
        completion=dict(path=str(OUT / 'COMPLETION.json'), sha256=sha(OUT / 'COMPLETION.json'))))
    state.update(best_training_selected_candidate=best, latest_training_selected_candidate=dict(directory=OUT.name,
        arm=pick, selection=results['selection'], full_period_cagr=metrics['cagr'], full_period_sharpe=metrics['sharpe'], selected_on_first88=True),
        best_tracking_note=contract()['historical_best_tracking'], registered_queue=[], current_status='completed_independently_verified',
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Review within-history own-sector magnitude ranking against exact192 sign-balance control; keep frozen observers unchanged.')
    require(len(state['completed']) == 229 and sum(r['trials'] for r in state['completed']) == 1204, 'Wrong final counters')
    commit_state(before, state, 'close'); print(json.dumps(receipt), flush=True)


def register():
    require(not OUT.exists(),'Fresh experiment directory required')
    before=CENTRAL.read_bytes();state=json.loads(before);state_check(state)
    require(not state['registered_queue'],'Existing registered experiment')
    previous=predecessor_binding(state);validation=read(TEST_RECEIPT)
    require(validation['passed'] is True and validation['synthetic_only'] is True and validation['runtime']==ev.runtime(),'Synthetic tests required')
    require(validation['code_sha256']=={str(p):sha(p) for p in code_files()},'Code changed since tests')
    pins=dict(read(PRIOR/'PLAN.json')['input_sha256']);raw=read(LOCAL/'PLAN.json')['raw_source_hashes']
    source=price_source_binding(raw);review=code_review_binding()
    for binding in (previous,source,review):
        for name,digest in binding['input_sha256'].items():bind(pins,Path(name),digest)
    for path in code_files()+[TEST_RECEIPT,PROTECTED,CALENDAR,LOCAL/'PLAN.json']:bind(pins,path)
    for name in ('DATA.npz','META.json','INPUT_HASHES.json','CACHE_HASHES.json','local_linear_ORDERS.json','INCUMBENT_POLICY.json'):
        bind(pins,LOCAL/name)
    for mapping in (raw,read(CALENDAR)['provenance']['source_hashes']):
        for name,digest in mapping.items():bind(pins,Path(name),digest)
    for name,digest in ev.ENGINE_HASHES.items():bind(pins,ev.ENGINE_DIR/name,digest)
    verify(pins);protected=read(PROTECTED);verify(protected)
    require(len(protected)==21444,'Incomplete protected archive manifest')
    OUT.mkdir()
    with (OUT/'STATE_BEFORE.json').open('xb') as stream:stream.write(before)
    plan=dict(schema='monthly_sector_signedrank233_v1',created_utc=now(),runtime=ev.runtime(),
        contract=contract(),input_sha256=pins,raw_source_hashes=raw,predecessor_binding=previous,
        price_source_binding=source,sector_source_binding=sector_source_binding(raw),numerical_code_review=review,
        prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=len(protected),
        actual_new_feature_evaluations_before_registration=0,actual_new_model_fits_before_registration=0,
        actual_new_parameters_before_registration=0,actual_new_orders_before_registration=0,
        actual_new_returns_before_registration=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_sector_signed_magnitude',
        registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare fixed own-sector signed magnitude-rank and exact192 sign-balance control; freeze first88 zero-cost selection before later34 diagnostics.')
    commit_state(before,state,'register')
    print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)


def array_snapshot(arrays):
    return {key:(value.dtype.str,value.shape,value.tobytes()) for key,value in arrays.items()}


def array_fingerprints(arrays):
    return {key:dict(dtype=value.dtype.str,shape=list(value.shape),
        sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key,value in arrays.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'))
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    else:
        require(args.plan_sha256,'Explicit immutable PLAN SHA required')
        globals()[args.phase](args.plan_sha256)


def sector_source_binding(raw):
    """Bind metadata only; never load price values before registration."""
    require(sha(MAPPING_PATH) == MAPPING_SHA, 'Original industry mapping source changed')
    required = set(SECTOR_ETFS)
    entries = {}
    manifests = {}
    require(raw.get(str(SECTORS_PATH)) == sha(SECTORS_PATH), 'Sector panel missing from original raw pins')
    for manifest in ETF_MANIFESTS:
        digest = sha(manifest)
        require(raw.get(str(manifest)) == digest, 'ETF manifest missing from original raw pins')
        manifests[str(manifest)] = digest
        for entry in read(manifest):
            symbol = entry['symbol']
            if symbol not in required:
                continue
            require(symbol not in entries, 'Ambiguous sector ETF source ' + symbol)
            path = Path(entry['file']).resolve()
            require(path.is_relative_to(ROOT) and 'CloudStorage' not in path.parts, 'Nonlocal sector ETF source')
            require(raw.get(str(path)) == entry['sha256'] == sha(path), 'Sector ETF raw pin mismatch ' + symbol)
            entries[symbol] = dict(path=str(path), sha256=entry['sha256'], manifest=str(manifest))
    require(set(entries) == required, 'Missing source for required industry ETF')
    return dict(mapping_source=dict(path=str(MAPPING_PATH), sha256=MAPPING_SHA),
        sectors_source=dict(path=str(SECTORS_PATH), sha256=raw[str(SECTORS_PATH)]),
        manifest_sha256=manifests, own_sector_etfs=list(SECTOR_ETFS), sources=entries,
        price_column='Adj Close', loader='original study.arch.old.panel_load')


def sector_inputs(study, source, meta):
    strict = study.ex.t.strict
    require(Path(strict.__file__).resolve() == MAPPING_PATH.resolve() and sha(MAPPING_PATH) == MAPPING_SHA,
            'Loaded industry mapping is not the pinned original')
    require(len(meta['etfs']) == 17 and len(set(meta['etfs'])) == 17, 'Original17 ETF identities required')
    require(isinstance(source.get('adj'), pd.DataFrame) and isinstance(source.get('sectors'), pd.DataFrame),
            'Original adjusted prices and sector classifications required')
    require(set(SECTOR_ETFS) <= set(source['adj'].columns), 'Original sector ETF injection incomplete')
    return {'adj': source['adj'], 'sectors': source['sectors']}, strict.benchmark_map


def feature_prefix_panel(panel, meta, n):
    """Physically remove future source rows, including classifications; do not inspect their values."""
    require(type(n) is int and 0 <= n <= len(meta['dates']), 'Invalid source prefix')
    require(set(panel) == {'adj', 'sectors'} and all(isinstance(v, pd.DataFrame) for v in panel.values()),
            'Original adjusted prices and sector classifications required')
    if not n:
        return {key: value.iloc[:0].copy() for key, value in panel.items()}
    day = pd.Timestamp(meta['dates'][n-1])
    return {key: value.loc[:day].copy() for key, value in panel.items()}



def contract():
    for path,digest in ((CONTEXT,CONTEXT_SHA),(COVERAGE,COVERAGE_SHA)):
        require(sha(path)==digest,'Prospective definition changed: '+str(path))
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,
        arms=list(NAMES),candidate=NAMES[1],control_nonpromotable=NAMES[2],
        period=['2016-08-17','2026-09-24'],training_months=88,later_months=34,
        dose=DOSE,max_absolute_raw=MAX_RAW,max_absolute_delta=MAX_DELTA,
        source_window=dict(price_count=253,earliest_offset=294,latest_offset=42,return_count=252,
            block_sessions=21,blocks=12,candidate_denominator=208,control_denominator=16,neutral_pseudo_blocks=4),
        feature_prefixes=list(FEATURE_PREFIXES),physical_feature_prefixes=True,all123_order_prefixes=True,
        selection_paths=['same_close_zero','next_open_zero'],next_open_cost_diagnostic_only=True,
        control192_exact_feature_availability_metadata_orders_required=True,
        model_fit_accounting=MODEL_FIT_ACCOUNTING,limitation=context['limitations'],
        historical_best_tracking='First88 zero-cost paths select; later34/full repeated research diagnostic, not pristine OOS or observer promotion.',
        new_arms=2,registered_trials=3,execution_paths=9,new_model_fits=0,predictive_model_fits=0,
        no_observer_writes=True,original_pool_preserved=True,no_parameter_search=True,strict_pit=False)


def code_files():
    names=('monthly_sector_signedrank233_host.py','test_monthly_sector_signedrank233_host.py',
        'monthly_sector_signedrank233_kernel.py','test_monthly_sector_signedrank233_kernel.py',
        'monthly_sector_signedrank233_independent.py','test_monthly_sector_signedrank233_independent.py',
        'monthly_sector_signedrank233_context.json','sector_signedrank233_COVERAGE.json',
        'sector_signedrank233_SOURCE_REVIEW.json','sector_signedrank233_source_metadata.py',
        'ptr169_rebuild_evaluate.py','ptr169_results_independent.py')
    direct=[ROOT/'work'/name for name in names]
    import monthly_sector_signedrank233_independent as independent
    return direct+sorted(set(independent.dependency_files())-set(direct))


def state_check(state):
    require(len(state['completed'])==228 and sum(row['trials'] for row in state['completed'])==1201,
            'Expected228 completed/1201trials')
    require(state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Incumbent changed')
    require(state['completed'][-1]['directory']==PRIOR.name,'Round232 must be predecessor')


def control192_binding():
    review=read(SOURCE_REVIEW)['control192'];pins={}
    for name,key in (('PLAN.json','plan_sha256'),('COMPLETION.json','completion_sha256'),
                     ('prepared/COMPLETION.json','prepared_completion_sha256')):
        bind(pins,CONTROL192/name,review[key])
    completion=read(CONTROL192/'prepared/COMPLETION.json')
    require(completion['status']=='COMPLETE' and completion['plan_sha256']==review['plan_sha256'],
            'Frozen192 preparation incomplete')
    require(completion['artifacts_sha256']==review['artifact_sha256'] and
            set(completion['artifacts_sha256'])==PREPARED_FILES,'Frozen192 seven artifacts differ')
    for name,digest in completion['artifacts_sha256'].items():bind(pins,CONTROL192/'prepared'/name,digest)
    verify(pins)
    return dict(input_sha256=pins,plan_sha256=review['plan_sha256'],
        prepared_completion_sha256=review['prepared_completion_sha256'],
        old_feature_key='candidate',old_order_arm='sector_relative_recurrence')


def price_source_binding(raw):
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Source metadata review changed')
    review=read(SOURCE_REVIEW)
    for key in ('passed','source_only','metadata_only','source_admission_metadata_only',
                'metadata_axes_verified','calendar_clock_verified','raw_source_hashes_verified',
                'raw_source_map_matches_original','sector_source_binding_verified',
                'prior192_control_artifacts_bound','integration_source_verified'):
        require(review.get(key) is True,'Missing source review '+key)
    for key in ('strict_pit','source_npz_decoded','actual_support_computed','actual_new_features_computed',
                'actual_orders_computed','actual_returns_computed','observers_modified','central_modified'):
        require(review.get(key) is False,'Source scope differs '+key)
    require(review.get('context_sha256')==CONTEXT_SHA and
            review.get('evidence_sha256')==read(CONTEXT)['evidence_sha256'],'Source prospective binding differs')
    require(review.get('raw_source_hashes')==raw and len(raw)==64,'Original64 raw map differs')
    sector=sector_source_binding(raw)
    require(review.get('sector_source_binding')==sector,'Original192 industry source binding differs')
    pins={str(SOURCE_REVIEW):SOURCE_REVIEW_SHA,str(COVERAGE):COVERAGE_SHA}
    for name,digest in review['input_sha256'].items():
        bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    old=control192_binding()
    for name,digest in old['input_sha256'].items():bind(pins,Path(name),digest)
    verify(pins)
    return dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,
        coverage_sha256=COVERAGE_SHA,sector_source_binding=sector,control192=old,strict_pit=False)


def scale_and_delta(values,n):
    require(set(values)=={'candidate','control','available'},'Feature keys changed')
    available=np.asarray(values['available']);shape=available.shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Invalid feature prefix')
    require(available.dtype==np.dtype(bool),'Boolean mask required')
    transforms,deltas={},{}
    for arm,key in RAW.items():
        raw=np.asarray(values[key]);require(raw.shape==shape,'Invalid raw feature shape')
        x=np.asarray(raw[:n],float)
        require(not np.isinf(x).any() and np.array_equal(np.isfinite(x),available[:n]),'Common feature mask invalid')
        require((np.abs(x[available[:n]])<=MAX_RAW).all(),'Raw signed-rank bound exceeded')
        transformed=np.zeros((n,shape[1]),float);transformed[available[:n]]=x[available[:n]]
        delta=DOSE*transformed
        if n:delta[0]=0.
        require(np.max(np.abs(delta),initial=0.)<=DOSE*MAX_RAW,'Score budget exceeded')
        transforms[arm],deltas[arm]=transformed,delta
    return transforms,deltas


def decisions(study,arrays,meta,values,n):
    # Keep the frozen192 arithmetic and full original array identity unchanged.
    transforms,deltas=scale_and_delta(values,n)
    for key in ('eligible','forecast_available','model_available','extension_available','state_weak_narrow','state_calm10'):
        if key in arrays:
            require(all(isinstance(v,(bool,np.bool_)) for v in np.asarray(arrays[key])[:n].flat),
                    'Exact boolean values required: '+key)
    policy=read(LOCAL/'INCUMBENT_POLICY.json')
    choices,score=study.choices(study.configs()[4],arrays,meta,n,policy=policy)
    scores={NAMES[0]:np.asarray(score)[:n].copy()}
    orders={NAMES[0]:study.rp.make_orders(choices,score,meta)}
    for arm in NAMES[1:]:
        pre=study.modified_pre_score(study.configs()[4],arrays,n)
        view=pre[:n].copy();changed=deltas[arm]!=0.
        view[changed]+=deltas[arm][changed];pre[:n]=view
        chosen,score=study.adapter.score_and_choose('incumbent66',0.,np.full(arrays['baseline'].shape,np.nan),
            arrays,meta,n,pre_score=pre,effects=arrays['effects'],episode_policy=policy,
            boundary_selector=study.ep.select_boundary)
        orders[arm]=study.rp.make_orders(chosen,score,meta);scores[arm]=np.asarray(score)[:n].copy()
    return orders,transforms,deltas,scores


def estimate_counts(check):
    counts={key:check.get(key) for key in COUNT_FIELDS}
    require(all(type(v) is int and v>=0 for v in counts.values()),'Exact nonnegative integer descriptive counts required')
    n,w,A=counts['events'],counts['symbols_per_event'],counts['available_cells']
    require(n<=122 and w>0 and counts['checked_cells']==n*w and
        A<=n*w and counts['unavailable_cells']==n*w-A,'Feature cell accounting differs')
    factors={'source_price_observations':506,'relative_sign_observations':12,
        'magnitude_rank_observations':12,'magnitude_pair_comparisons':66,
        'signed_rank_estimates':1,'sign_balance_estimates':1}
    require(all(counts[k]==v*A for k,v in factors.items()) and
        counts['constant_magnitude_copies']<=A and counts['predictive_model_fits']==0,'Descriptive accounting differs')
    return counts


model_counts=estimate_counts


def require_feature_audit(check,values,rows=None):
    for key in FEATURE_FLAGS:require(check.get(key) is True,'Independent feature evidence missing '+key)
    mask=np.asarray(values['available']);n=len(mask) if rows is None else rows
    require(mask.dtype==bool and mask.ndim==2 and mask.shape[0]==n,'Feature mask axes differ')
    counts=estimate_counts(check)
    require(counts['events']==n and counts['symbols_per_event']==mask.shape[1] and
        counts['available_cells']==int(mask.sum()),'Independent feature coverage differs')
    require(check.get('counts')==counts and check.get('numerical_tolerance')==0. and
        check.get('producer_modules_imported') is False and check.get('computes_returns') is False and
        check.get('mapping_source_sha256')==MAPPING_SHA,'Independent feature scope differs')
    for key in ('candidate','control'):
        raw=np.asarray(values[key]);require(raw.dtype==np.float64 and raw.shape==mask.shape and
            not np.isinf(raw).any() and np.array_equal(np.isfinite(raw),mask) and
            (np.abs(raw[mask])<=MAX_RAW).all(),'Exact common raw support differs')


def require_delta_audit(check,deltas):
    for key in DELTA_FLAGS:require(check.get(key) is True,'Independent delta missing '+key)
    require(check.get('dose')==DOSE and check.get('max_absolute_delta')==MAX_DELTA and
        check.get('numerical_tolerance')==0. and check.get('producer_modules_imported') is False and
        check.get('computes_returns') is False,'Independent delta scope differs')
    require(set(deltas)==set(RAW) and check.get('changed_cells')==
            {k:int(np.count_nonzero(v)) for k,v in deltas.items()},'Independent delta coverage differs')
    require(check.get('checked_events')==len(next(iter(deltas.values()))),'Delta event coverage differs')


def require_selection_audit(check,rows):
    for key in SELECTION_FLAGS:require(check.get(key) is True,'Independent selection missing '+key)
    require(check.get('producer_modules_imported') is False and check.get('computes_returns') is False and
        check.get('full_scores_supplied') is True and check.get('numerical_tolerance')==0. and
        check.get('original_source_parity_tolerance')==1e-12 and
        check.get('maximum_absolute_delta')==MAX_DELTA and check.get('ordered_choices_checked')==3*rows,
        'Independent full-score selection coverage/scope differs')


def control192_parity(meta,values,orders):
    binding=control192_binding();folder=CONTROL192/'prepared'
    require(meta==read(folder/'META.json'),'Exact192 control metadata differs')
    with np.load(folder/'FEATURES.npz',allow_pickle=False) as f:
        require(set(f.files)=={'candidate','control','available'},'Frozen192 feature keys differ')
        oldmask=f['available'];old=f['candidate']
    mask=np.asarray(values['available']);new=np.asarray(values['control'])
    require(mask.dtype==oldmask.dtype==bool and mask.shape==oldmask.shape and
            np.array_equal(mask,oldmask),'Exact192 common availability differs')
    require(new.dtype==old.dtype==np.float64 and new.shape==old.shape and
            np.array_equal(np.isnan(new),np.isnan(old)) and not np.isinf(new).any() and
            new[mask].tobytes()==old[mask].tobytes(),'Exact192 control raw values differ')
    require(orders[NAMES[2]]==read(folder/'ORDERS.json')['sector_relative_recurrence'],
            'Exact192 control orders differ')
    return dict(passed=True,metadata_exact=True,available_exact=True,control_values_exact=True,
        control_orders_exact=True,numerical_tolerance=0.,new_order_arm=NAMES[2],
        old_order_arm='sector_relative_recurrence',old_feature_key='candidate',
        prepared_completion_sha256=binding['prepared_completion_sha256'],
        checked_cells=int(mask.size),available_cells=int(mask.sum()))


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation is write-once')
    import monthly_sector_signedrank233_kernel as kernel
    import monthly_sector_signedrank233_independent as independent
    study=local();meta,arrays,px,op=study.load()
    require(meta==read(LOCAL/'META.json') and len(meta['dates'])==122 and
            arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original122x49/27arrays required')
    before=array_snapshot(arrays)
    source,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'],'Raw source lineage changed');verify(raw)
    execution=study.verify_panel_parity(source,px,op)
    panel,mapper=sector_inputs(study,source,meta);calendar=read(CALENDAR)['sessions']
    values,audit=kernel.build_features(panel,meta,calendar,mapper)
    check=independent.verify_features(panel,meta,calendar,mapper,values,audit)
    require_feature_audit(check,values,122);require(estimate_counts(check)==estimate_counts(audit),'Feature counts differ')
    orders,transforms,deltas,scores=decisions(study,arrays,meta,values,122)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original full order parity failed')
    delta_check=independent.verify_deltas(values,deltas,dose=DOSE);require_delta_audit(delta_check,deltas)
    selection=independent.verify_selection(arrays,meta,values,orders,scores=scores,n=122)
    require_selection_audit(selection,122);control=control192_parity(meta,values,orders)
    prefixes=[]
    for n in FEATURE_PREFIXES:
        pp=feature_prefix_panel(panel,meta,n)
        vv,aa=kernel.build_features(pp,meta,calendar,mapper,limit=n)
        ck=independent.verify_features(pp,meta,calendar,mapper,vv,aa,limit=n)
        require_feature_audit(ck,vv,n);require(estimate_counts(ck)==estimate_counts(aa),'Prefix feature counts differ')
        for key in values:np.testing.assert_array_equal(vv[key],values[key][:n])
        require(aa['event_history']==audit['event_history'][:n],'Physical source prefix audit differs')
        oo,tt,dd,ss=decisions(study,arrays,meta,vv,n)
        cs=independent.verify_selection(arrays,meta,vv,oo,scores=ss,n=n);require_selection_audit(cs,n)
        dc=independent.verify_deltas(vv,dd,dose=DOSE);require_delta_audit(dc,dd)
        require(all(oo[k]==orders[k][:n] for k in NAMES),'Physical order prefix differs')
        for key in NAMES:np.testing.assert_array_equal(ss[key],scores[key][:n])
        prefixes.append(dict(prefix=n,source_rows={k:len(v) for k,v in pp.items()},
            feature=ck,delta=dc,selection=cs,counts=estimate_counts(aa)))
    for n in range(123):
        oo,tt,dd,ss=decisions(study,arrays,meta,values,n)
        require(all(oo[k]==orders[k][:n] for k in NAMES),'Decision prefix differs')
        for key in RAW:
            np.testing.assert_array_equal(tt[key],transforms[key][:n]);np.testing.assert_array_equal(dd[key],deltas[key][:n])
        for key in NAMES:np.testing.assert_array_equal(ss[key],scores[key][:n])
    require(before==array_snapshot(arrays),'Original27arrays modified')
    parity=dict(raw_source_hashes=raw,price_source_binding=plan['price_source_binding'],
        sector_source_binding=plan['sector_source_binding'],execution_prices=execution,
        calendar_sha256=sha(CALENDAR),original_array_fingerprints=array_fingerprints(arrays),
        original_arrays_unchanged=True,original_orders_exact=True,control192=control,strict_pit=False)
    counts=estimate_counts(audit)
    evidence=dict(feature=check,delta=delta_check,selection=selection,prefixes=123,
        feature_prefixes=list(FEATURE_PREFIXES),feature_prefix_sources_truncated=True,physical_prefix_evidence=prefixes,
        original_orders_exact=True,original_arrays_unchanged=True,original_array_count=27,
        original_core_preserved=True,single_original_episode=True,additional_max_score_budget=MAX_DELTA,
        control192=control,model_counts=counts,selection_uses_zero_cost_paths_only=True,
        new_model_fits=0,new_predictive_model_fits=0,model_fit_accounting=MODEL_FIT_ACCOUNTING,**counts)
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FEATURES.npz',**values)
    np.savez_compressed(dest/'TRANSFORMS_DELTAS.npz',**{'transform_'+k:v for k,v in transforms.items()},**{'delta_'+k:v for k,v in deltas.items()})
    dump(dest/'FEATURE_AUDIT.json',audit);dump(dest/'ORDERS.json',orders);dump(dest/'META.json',meta)
    dump(dest/'SOURCE_PARITY.json',parity);dump(dest/'INDEPENDENT_PREPARE.json',evidence)
    completion=dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},original_array_count=27,model_counts=counts,
        new_returns_computed=False,new_model_fits=0,new_predictive_model_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING,**counts)
    verify_prepared_evidence(completion,values,deltas,evidence,meta=meta,parity=parity,feature_audit=audit)
    dump(dest/'COMPLETION.json',completion)
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest/'COMPLETION.json'),counts=counts)),flush=True)


def verify_prepared_evidence(c,values,deltas,evidence,*,meta,parity,feature_audit):
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Exact7 prepared artifacts required')
    require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and
        all(len(s)==49 and len(set(s))==49 for s in meta['symbols']),'Full122x49 identities required')
    require_feature_audit(evidence['feature'],values,122);require_delta_audit(evidence['delta'],deltas)
    require_selection_audit(evidence['selection'],122)
    counts=estimate_counts(feature_audit)
    require(counts==estimate_counts(evidence['feature'])==evidence['model_counts']==c['model_counts'],
            'Prepared descriptive counts differ')
    require([e['date'] for e in feature_audit['event_history']]==meta['dates'] and
            [e['symbols'] for e in feature_audit['event_history']]==meta['symbols'],'Feature identities differ')
    expected=dict(price_count=253,skip_sessions=42,return_count=252,block_count=12,block_sessions=21,
        neutral_pseudo_blocks=4,candidate_denominator=208,control_denominator=16,expected_twice_rank_sum=156)
    require(all(feature_audit.get(k)==v for k,v in expected.items()),'Formation definition differs')
    for key in ('common_availability','within_cell_ranking_performed'):require(feature_audit.get(key) is True,'Feature invariant missing '+key)
    for key in ('portfolio_returns_calculated','cross_sectional_ranking_performed','trading_eligibility_modified'):
        require(feature_audit.get(key) is False,'Feature scope differs '+key)
    for key in ('original_orders_exact','original_arrays_unchanged','original_core_preserved','single_original_episode',
                'feature_prefix_sources_truncated','selection_uses_zero_cost_paths_only'):
        require(evidence.get(key) is True,'Prepared invariant missing '+key)
    require(evidence.get('additional_max_score_budget')==MAX_DELTA and evidence.get('prefixes')==123 and
        evidence.get('feature_prefixes')==list(FEATURE_PREFIXES) and
        [x['prefix'] for x in evidence.get('physical_prefix_evidence',[])]==list(FEATURE_PREFIXES),'Prefix/dose evidence differs')
    for p in evidence['physical_prefix_evidence']:
        n=p['prefix'];small={k:v[:n] for k,v in values.items()};ds={k:v[:n] for k,v in deltas.items()}
        require_feature_audit(p['feature'],small,n);require_delta_audit(p['delta'],ds);require_selection_audit(p['selection'],n)
        require(p['counts']==estimate_counts(p['feature']) and set(p['source_rows'])=={'adj','sectors'},'Physical counts differ')
        require(all(type(v) is int and (v>0 if n else v==0) for v in p['source_rows'].values()),'Physical source row counts differ')
    for record in (c,evidence):
        require(record.get('original_array_count')==27 and record.get('model_fit_accounting')==MODEL_FIT_ACCOUNTING,
                'Original array/accounting differs')
        require(all(type(record.get(k)) is int and record[k]==0 for k in ('new_model_fits','new_predictive_model_fits','predictive_model_fits')),'Model fit counts differ')
        require(all(record.get(k)==v and type(record.get(k)) is int for k,v in counts.items()),'Flat count receipt differs')
    require(c.get('new_returns_computed') is False,'Preparation must not compute returns')
    require(parity.get('calendar_sha256')==sha(CALENDAR) and parity.get('original_arrays_unchanged') is True and
        parity.get('original_orders_exact') is True and parity.get('strict_pit') is False,'Source parity scope differs')
    require(len(parity['original_array_fingerprints'])==27,'Original27 byte fingerprints required')
    control=parity['control192'];require(control==evidence['control192'],'Control evidence differs')
    for key in ('passed','metadata_exact','available_exact','control_values_exact','control_orders_exact'):
        require(control.get(key) is True,'Exact192 control parity missing '+key)
    require(control.get('numerical_tolerance')==0. and control.get('checked_cells')==122*49 and
            control.get('available_cells')==counts['available_cells'],'Control coverage differs')
    for key in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact'):
        require(parity['execution_prices'].get(key) is True,'Execution parity missing '+key)


def prepared(expected):
    plan=gate(expected);dest=OUT/'prepared';c=read(dest/'COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected,'Preparation incomplete')
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Exact7 prepared artifacts required')
    for name,digest in c['artifacts_sha256'].items():require(sha(dest/name)==digest,'Changed prepared '+name)
    with np.load(dest/'FEATURES.npz',allow_pickle=False) as f:values={k:f[k] for k in f.files}
    with np.load(dest/'TRANSFORMS_DELTAS.npz',allow_pickle=False) as f:
        require(set(f.files)=={'transform_'+k for k in RAW}|{'delta_'+k for k in RAW},'Transform artifact keys differ')
        transforms={k:f['transform_'+k] for k in RAW};deltas={k:f['delta_'+k] for k in RAW}
    expected_transforms,expected_deltas=scale_and_delta(values,122)
    for k in RAW:
        np.testing.assert_array_equal(transforms[k],expected_transforms[k]);np.testing.assert_array_equal(deltas[k],expected_deltas[k])
    meta=read(dest/'META.json');orders=read(dest/'ORDERS.json');parity=read(dest/'SOURCE_PARITY.json')
    require(meta==read(LOCAL/'META.json') and orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original metadata/orders changed')
    require(parity['raw_source_hashes']==plan['raw_source_hashes'] and parity['price_source_binding']==plan['price_source_binding'] and
            parity['sector_source_binding']==plan['sector_source_binding'],'Prepared source bindings changed')
    require(parity['control192']==control192_parity(meta,values,orders),'Control parity receipt changed')
    verify_prepared_evidence(c,values,deltas,read(dest/'INDEPENDENT_PREPARE.json'),meta=meta,
        parity=parity,feature_audit=read(dest/'FEATURE_AUDIT.json'))
    return meta,orders


if __name__=='__main__':main()

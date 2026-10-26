"""Round238 fixed own-sector same-window relative qualification.
All real source features/orders/returns require an immutable registered PLAN.
Original pre-score uses original E; candidate E enters only the final episode.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, io, json, os, tempfile, unittest
import numpy as np
import pandas as pd
import ptr169_rebuild_evaluate as ev
from monthly_ownsector_gate238_independent import FEATURE_FLAGS, SELECTION_FLAGS, COUNT_FIELDS
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_ownsector_gate238'
LOCAL=ROOT/'outputs/monthly_local_linear70'
PRIOR=ROOT/'outputs/monthly_allstate_gate237'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES=('incumbent_local67','ownsector_relative_gate')
TRAIN=88
FEATURE_PREFIXES=(0,1,53,88,122)
VALUE_KEYS=('original_eligible','candidate_eligible','required','momentum','benchmark_momentum','threshold')
MODEL_FIT_ACCOUNTING='Zero new predictive fits, estimated parameters or score weights. Two security and two assigned-benchmark endpoint-role uses per required cell; SPY only assigned ETF uses. Original models/ranks/pre and one episode per arm retained; physical-prefix and independent work counted separately.'
ESTIMATE_FIELDS=COUNT_FIELDS
CONTEXT=ROOT/'work/monthly_ownsector_gate238_context.json'
CONTEXT_SHA='c6d4db0f570f41bc44a906589fd8a37b437ab5c5cf1a9db6f4111d0b83e18f69'
COVERAGE=ROOT/'work/after237_ownsector_gate_coverage.json'
COVERAGE_SHA='be0f6bc8bd9f111ed91c7a1106d61c734d09bd3b57df39ccfec9a50ddb19330d'
TEST_RECEIPT=ROOT/'work/monthly_ownsector_gate238_TESTS.json'
EXTERNAL_RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PREPARED_FILES={'FEATURES.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={arm+'_'+path[0]+'_'+kind+'.json' for arm in NAMES for path in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}
sha,read,require,dump=ev.sha,ev.read,ev.require,ev.dump
PRIOR_EXPECTED={'PLAN.json': '0ab103b7ec4b85c0814447bc47ff375be40d7005251c05836fbb82b6b0e68617', 'COMPLETION.json': 'e9da4c39a379f629998db8e162cd3bdb6cb1025f321d765f4917ffa887688a25', 'results/COMPLETION.json': '005a11ee7df81f73c54085c5a0f2f448de4acb252a4c053d16e7147563853f48', 'RESEARCH_REVIEW.json': '08f45c125ea4f197766a18167d3c632b931825fe5526eb8a500749d6ec09bf76'}
SOURCE_OUT=ROOT/'outputs/monthly_ownsector238_repaired_source_audit'
SOURCE_REPORT=SOURCE_OUT/'REPORT.json'
SOURCE_ACCEPTANCE=ROOT/'work/ownsector238_vox_acquisition_acceptance.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def local():
    return module('monthly_ownsector_gate238_original', LOCAL / 'study.py')


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
            result['paths']==6 and result['slices']==30, 'Predecessor results incomplete')
    for name, digest in read(CONTEXT)['evidence_sha256'].items(): bind(pins, ROOT/name, digest)
    return dict(directory=PRIOR.name, input_sha256=pins)


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_OWNSECTOR_GATE238_' + tag + '_' + expected + '.json')
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
    require(not TEST_RECEIPT.exists(), 'Formal synthetic test receipt is write-once')
    names = ('test_monthly_ownsector_gate238_host.py', 'test_monthly_ownsector_gate238_kernel.py',
             'test_monthly_ownsector_gate238_independent.py')
    modules = [module('test_ownsector_gate238_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
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
    path = ROOT / 'work/ownsector_gate238_CODE_REVIEW.json'
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


def meta_prefix(meta,n):
    require(isinstance(meta,dict) and isinstance(meta.get('symbols'),list),'Metadata required')
    total=len(meta['symbols'])
    require(type(n) is int and 0<=n<=total,'Metadata prefix bound')
    result=dict(meta)
    for key in ('dates','ends','symbols'):
        if key in meta:
            require(isinstance(meta[key],list) and len(meta[key])==total,'Metadata event axis '+key)
            result[key]=[list(x) if isinstance(x,list) else x for x in meta[key][:n]]
    require(isinstance(meta.get('etfs'),list),'Explicit ETF metadata required')
    result['etfs']=list(meta['etfs'])
    return result


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
    import monthly_ownsector_gate238_independent as independent
    helper = independent.accounting()
    require(set(training) == set(NAMES), 'Training arms differ')
    for arm in NAMES:
        require(set(training[arm]) == {'same_close_zero', 'next_open_zero'}, 'Training paths differ')
        for path, slices in training[arm].items():
            require(set(slices) == {'training', 'early', 'recent'}, 'Training slices differ')
            for name, full_name in (('training', 'first88'), ('early', 'early53'), ('recent', 'recent35')):
                helper.check_metrics(audit['paths'][arm][path]['slices'][full_name], slices[name],
                                     'Frozen training/' + arm + '/' + path + '/' + name)



def contract():
    require(sha(CONTEXT)==CONTEXT_SHA and sha(COVERAGE)==COVERAGE_SHA,'Changed prospective definition')
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,
        arms=list(NAMES),candidate=NAMES[1],control=NAMES[0],period=['2016-08-17','2026-09-24'],
        training_months=88,later_months=34,registered_trials=2,new_arms=1,execution_paths=6,
        source_window=dict(start_offset=252,end_offset=42,return_count=210,security_endpoints_per_cell=2,benchmark_endpoints_per_cell=2),
        original_pre_unchanged=True,original_eligibility_for_models=True,eligibility_injected_only_at_episode=True,
        original_episode_preserved=True,final_scores_may_change=True,additional_score_budget=0.,
        feature_prefixes=list(FEATURE_PREFIXES),all123_physical_cache_meta_order_prefixes=True,
        selection_paths=['same_close_zero','next_open_zero'],next_open_cost_diagnostic_only=True,
        no_parameter_search=True,new_model_fits=0,predictive_model_fits=0,new_predictive_model_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING,limitation=context['limitations'],strict_pit=False,
        historical_best_tracking='First88 zero-cost paths select; later34/full are repeatedly researched diagnostics, not pristine OOS or observer promotion.',
        original_pool_preserved=True,no_observer_writes=True)


def code_files():
    names=('monthly_ownsector_gate238_host.py','test_monthly_ownsector_gate238_host.py',
        'monthly_ownsector_gate238_kernel.py','test_monthly_ownsector_gate238_kernel.py',
        'monthly_ownsector_gate238_independent.py','test_monthly_ownsector_gate238_independent.py',
        'monthly_ownsector_gate238_context.json','after237_ownsector_gate_coverage.json',
        'ptr169_rebuild_evaluate.py','ptr169_results_independent.py','ownsector238_repair_sources.py')
    direct=[ROOT/'work'/name for name in names]
    import monthly_ownsector_gate238_independent as independent
    return direct+sorted(set(independent.dependency_files())-set(direct))


def state_check(state):
    require(len(state['completed'])==233 and sum(r['trials'] for r in state['completed'])==1213,
            'Expected233 completed entries/1213 trials')
    require(state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Incumbent changed')
    require(state['completed'][-1]['directory']==PRIOR.name,'Round237 must be predecessor')


def source_admission_binding(raw):
    context=read(CONTEXT)
    expected=context['evidence_sha256'];pins={}
    for path in (SOURCE_OUT/'SOURCE_PLAN.json',SOURCE_REPORT,SOURCE_OUT/'COMPLETION.json',
                 ROOT/'work/ownsector238_repair_RESULT_REVIEW.json',SOURCE_ACCEPTANCE,
                 ROOT/'work/ownsector238_repair_source_policy.json'):
        bind(pins,path,expected[str(path.relative_to(ROOT))])
    p=read(SOURCE_OUT/'SOURCE_PLAN.json');r=read(SOURCE_REPORT);c=read(SOURCE_OUT/'COMPLETION.json')
    require(c.get('status')=='COMPLETED_REPAIRED_SOURCE_AUDIT_INDEPENDENTLY_VERIFIED' and c.get('admitted') is True,
            'Repaired source completion not admitted')
    require(c.get('plan_sha256')==sha(SOURCE_OUT/'SOURCE_PLAN.json') and c.get('report_sha256')==sha(SOURCE_REPORT)
            and r.get('plan_sha256')==c['plan_sha256'],'Repaired source closure linkage')
    require(r.get('status')=='ADMITTED' and r.get('admitted') is True and r.get('gaps')==[]
            and c.get('counts')==r.get('counts'),'Repaired source report admission')
    counts=r['counts']
    require(counts.get('required_cells')==counts.get('admitted_cells')==4670 and
            counts.get('gap_cells')==counts.get('market_gap_events')==counts.get('missing_endpoints')==counts.get('invalid_endpoints')==0,
            'Repaired source coverage incomplete')
    require(c.get('source_only') is True and c.get('strategy_trials')==0 and c.get('actual_strategy_math_computed') is False
            and c.get('central_modified') is False and c.get('observers_modified') is False and c.get('protected_files')==21444,
            'Repaired source scope changed')
    review=read(ROOT/'work/ownsector238_repair_RESULT_REVIEW.json')
    require(review.get('passed') is True and review.get('unresolved_blockers')==[], 'Source result review failed')
    require(p.get('raw_source_hashes')==raw and r.get('input_sha256')==p.get('input_sha256'),'Repaired source raw dependency chain')
    for name,digest in p['input_sha256'].items():bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    ac=read(SOURCE_ACCEPTANCE)
    require(ac.get('acquired') is True and ac.get('metadata_identity_review_passed') is True and ac.get('symbol')=='VOX',
            'VOX acquisition identity')
    for name,digest in ac['input_sha256'].items():bind(pins,ROOT/name,digest)
    bind(pins,ROOT/ac['prices_raw_path'],ac['prices_raw_sha256'])
    verify(pins)
    return dict(input_sha256=pins,source_plan_sha256=sha(SOURCE_OUT/'SOURCE_PLAN.json'),
                source_report_sha256=sha(SOURCE_REPORT),source_completion_sha256=sha(SOURCE_OUT/'COMPLETION.json'),
                vox_prices_path=ac['prices_raw_path'],vox_prices_sha256=ac['prices_raw_sha256'],
                required_cells=4670,source_admitted=True,strict_pit=False,raw_original_map_preserved=True)


def price_source_binding(raw):
    """Digest/metadata only; never decode cached arrays or load price values here."""
    require(sha(CONTEXT)==CONTEXT_SHA and sha(COVERAGE)==COVERAGE_SHA,'Source context/coverage changed')
    coverage=read(COVERAGE)
    for key in ('actual_features_computed','actual_eligibility_computed','actual_orders_computed',
                'actual_returns_computed','npz_decoded','observer_files_modified'):
        require(coverage.get(key) is False,'Coverage scope differs '+key)
    require(coverage.get('source_only') is True,'Source-only coverage required')
    plan=read(LOCAL/'PLAN.json');require(raw==plan['raw_source_hashes'] and len(raw)==64,'Original64 source map differs')
    pins={}
    for name in ('PLAN.json','DESIGN.json','GRID.json','DESIGN.sha256','INPUT_HASHES.json',
                 'CACHE_HASHES.json','META.json','DATA.npz','INCUMBENT_POLICY.json','local_linear_ORDERS.json','study.py'):
        bind(pins,LOCAL/name)
    for path in (CONTEXT,COVERAGE,CALENDAR):bind(pins,path)
    for group in ('hashes','source_hashes','raw_source_hashes'):
        for name,digest in plan[group].items():bind(pins,Path(name) if Path(name).is_absolute() else LOCAL/name,digest)
    for name,digest in read(LOCAL/'INPUT_HASHES.json').items():
        bind(pins,Path(name) if Path(name).is_absolute() else LOCAL/name,digest)
    for name,digest in read(LOCAL/'CACHE_HASHES.json').items():bind(pins,LOCAL/name,digest)
    for mapping in (coverage['inspected_source_sha256'],read(CONTEXT)['evidence_sha256']):
        for name,digest in mapping.items():bind(pins,ROOT/name,digest)
    cal=read(CALENDAR);require(plan['calendar']==cal['provenance'],'Original calendar binding differs')
    for name,digest in cal['provenance']['source_hashes'].items():bind(pins,Path(name),digest)
    meta=read(LOCAL/'META.json')
    require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and
        all(len(ss)==49 and len(set(ss))==49 for ss in meta['symbols']),'Original122x49 identities')
    require(meta['dates'][0]=='2016-08-17' and meta['dates'][-1]=='2026-08-31' and meta['ends'][-1]=='2026-09-24',
        'Original period differs')
    require(len(meta['etfs'])==len(set(meta['etfs']))==17,'Original17 ETF identities')
    require(all(day in cal['sessions'] for day in meta['dates']),'Official signal calendar differs')
    repaired=source_admission_binding(raw)
    for name,digest in repaired['input_sha256'].items():bind(pins,Path(name),digest)
    verify(pins)
    return dict(input_sha256=pins,source_admission=repaired,raw_source_hashes=raw,events=122,symbols_per_event=49,original_array_count=27,
        calendar_sha256=sha(CALENDAR),context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,
        source_only=True,metadata_only=True,source_npz_decoded=False,actual_support_computed=False,
        actual_returns_computed=False,strict_pit=False,source_loader='Original panel_load plus verify_panel_parity; separately pinned VOX adjClose parser',
        qualification_scope='required regular-row endpoint evidence only; no reconstructed full211 admission')


def register():
    require(not OUT.exists(),'Fresh experiment directory required')
    before=CENTRAL.read_bytes();state=json.loads(before);state_check(state)
    require(not state['registered_queue'],'Existing registered experiment')
    previous=predecessor_binding(state);validation=read(TEST_RECEIPT)
    require(validation['passed'] is True and validation['synthetic_only'] is True and validation['runtime']==ev.runtime(),
        'Synthetic tests required')
    require(validation['code_sha256']=={str(p):sha(p) for p in code_files()},'Code changed since tests')
    pins=dict(read(PRIOR/'PLAN.json')['input_sha256']);raw=read(LOCAL/'PLAN.json')['raw_source_hashes']
    source=price_source_binding(raw);review=code_review_binding()
    for binding in (previous,source,review):
        for name,digest in binding['input_sha256'].items():bind(pins,Path(name),digest)
    for path in code_files()+[TEST_RECEIPT,PROTECTED,CALENDAR,LOCAL/'PLAN.json']:bind(pins,path)
    for name,digest in ev.ENGINE_HASHES.items():bind(pins,ev.ENGINE_DIR/name,digest)
    verify(pins);protected=read(PROTECTED);verify(protected);require(len(protected)==21444,'Incomplete protection')
    OUT.mkdir()
    with (OUT/'STATE_BEFORE.json').open('xb') as f:f.write(before)
    plan=dict(schema='monthly_ownsector_gate238_v1',created_utc=now(),runtime=ev.runtime(),contract=contract(),
        input_sha256=pins,raw_source_hashes=raw,predecessor_binding=previous,price_source_binding=source,
        numerical_code_review=review,prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,
        actual_new_feature_evaluations_before_registration=0,actual_new_model_fits_before_registration=0,
        actual_new_parameters_before_registration=0,actual_new_orders_before_registration=0,actual_new_returns_before_registration=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_new_eligibility',
        registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare fixed own-sector relative eligibility; freeze first88 zero-cost selection before diagnostics.')
    commit_state(before,state,'register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)


def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit PLAN SHA required');p=read(OUT/'PLAN.json')
    require(p['contract']==contract() and p['runtime']==ev.runtime(),'Frozen contract/runtime changed')
    verify(p['input_sha256'])
    require(p['price_source_binding']==price_source_binding(p['raw_source_hashes']),'Source binding changed')
    require(p['numerical_code_review']==code_review_binding(),'Code review changed')
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Changed pre-registration state')
    state=read(CENTRAL);state_check(state)
    require(state['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Central registration mismatch')
    require(p['predecessor_binding']==predecessor_binding(state),'Predecessor changed')
    return p


def feature_prefix_panel(panel,meta,n):
    require(set(panel)=={'adj','vox'} and all(isinstance(f,pd.DataFrame) for f in panel.values()),'Original and VOX adjusted panels required')
    require(type(n) is int and 0<=n<=len(meta['dates']),'Invalid source prefix')
    return {k:f.loc[:pd.Timestamp(meta['dates'][n-1])].copy() if n else f.iloc[:0].copy() for k,f in panel.items()}


def source_report_prefix(report,n):
    require(type(n) is int and 0<=n<=len(report['events']),'Invalid report prefix')
    events=report['events'][:n];records=[r for r in report['records'] if r['query_index']<n]
    require(report.get('admitted') is True and report.get('gaps')==[],'Prefix requires admitted source')
    counts=dict(report['counts']);w=counts['symbols_per_event']
    regular=sum(e['active'] for e in events);original=sum(len(e['original_eligible_indices']) for e in events)
    points=[p for e in events for p in e['market_endpoints']]+[p for r in records for p in r['endpoints']]
    counts.update(events=n,checked_cells=n*w,initial_partial_events=n-regular,regular_events=regular,
        original_eligible_cells=original,required_cells=len(records),stock_required_cells=sum(not r['is_etf'] for r in records),
        etf_required_cells=sum(r['is_etf'] for r in records),mapping_available_cells=len(records),mapping_missing_cells=0,
        security_benchmark_endpoint_checks=4*len(records),market_endpoint_checks=2*regular,
        valid_endpoints=sum(p['status']=='valid' for p in points),missing_endpoints=0,invalid_endpoints=0,
        admitted_cells=len(records),gap_cells=0,market_gap_events=0)
    return dict(schema=report['schema'],status='ADMITTED',admitted=True,events=events,records=records,gaps=[],counts=counts,
        repair_counts={k:sum(r['mapping_repair']==k for r in records) for k in report['repair_counts']},
        source_only=True,momentum_computed=False,eligibility_changed=False,rankings_computed=False,orders_computed=False,
        returns_computed=False,model_fits=0,strict_pit=False)


def feature_prefix_calendar(calendar,meta,n):
    require(type(n) is int and 0<=n<=len(meta['dates']),'Invalid calendar prefix')
    if not n:return []
    day=meta['dates'][n-1]
    require(day in calendar,'Query missing from official calendar')
    loc=calendar.index(day)
    require(loc+1<len(calendar),'Next official session witness required')
    return list(calendar[:loc+2])


def array_snapshot(arrays):
    return {k:(v.dtype.str,v.shape,v.tobytes()) for k,v in arrays.items()}


def array_fingerprints(arrays):
    return {k:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}


def value_prefix(values,n):
    require(set(values)==set(VALUE_KEYS),'Qualification value keys differ')
    return {k:np.asarray(v)[:n].copy() for k,v in values.items()}


def validate_values(values,arrays,n):
    require(set(values)==set(VALUE_KEYS),'Qualification value keys differ')
    shape=np.asarray(arrays['baseline']).shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Invalid decision prefix')
    old,new,required=[np.asarray(values[k]) for k in VALUE_KEYS[:3]]
    require(old.ndim==2 and old.shape[1]==shape[1] and old.shape[0]>=n and new.shape==required.shape==old.shape,
        'Qualification axes differ')
    for name,x in zip(VALUE_KEYS[:3],(old,new,required)):require(x.dtype==bool,'Exact bool '+name)
    require(np.array_equal(old[:n],np.asarray(arrays['eligible'])[:n]),'Original eligibility changed')
    require(not (new[:n]&~old[:n]).any() and not (required[:n]&~old[:n]).any(),'New qualification cannot add names')
    require(np.array_equal(new[:n][~required[:n]],old[:n][~required[:n]]),'Bypass eligibility changed')
    for key in ('momentum','benchmark_momentum','threshold'):
        x=np.asarray(values[key]);expected=old.shape
        require(x.dtype==np.float64 and x.shape==expected,'Numeric qualification shape/dtype '+key)
        require(not np.isinf(x[:n]).any(),'Nonfinite qualification values')
    for key in VALUE_KEYS[3:]:
        require(np.array_equal(np.isfinite(values[key][:n]),required[:n]),'Required numeric evidence differs '+key)
    return {NAMES[0]:old[:n].copy(),NAMES[1]:new[:n].copy()}


def decisions(study,arrays,meta,values,n):
    eligible=validate_values(values,arrays,n)
    for key in ('eligible','forecast_available','model_available','extension_available','state_weak_narrow','state_calm10'):
        if key in arrays:require(all(isinstance(v,(bool,np.bool_)) for v in np.asarray(arrays[key])[:n].flat),'Exact boolean '+key)
    before=array_snapshot(arrays);policy=read(LOCAL/'INCUMBENT_POLICY.json')
    pre=study.modified_pre_score(study.configs()[4],arrays,n)
    require(pre.shape==arrays['baseline'].shape and np.isfinite(pre[:n]).all(),'Original pre axes/numerics')
    pre_bytes=pre.tobytes();orders,scores,pres={},{},{}
    for arm in NAMES:
        calls=[]
        def boundary(base,old_good,symbols,effects,original,episode_policy):
            require(not calls,'Episode invoked more than once')
            require(np.array_equal(old_good,eligible[NAMES[0]]),'Original E changed before boundary')
            require(np.array_equal(base,pre[:n]),'Pre-score changed before boundary')
            calls.append(True)
            return study.ep.select_boundary(base,eligible[arm].copy(),symbols,effects,original,episode_policy)
        chosen,score=study.adapter.score_and_choose('incumbent66',0.,np.full(arrays['baseline'].shape,np.nan),
            arrays,meta,n,pre_score=pre.copy(),effects=arrays['effects'],episode_policy=policy,boundary_selector=boundary)
        require(len(calls)==1,'Exactly one original episode per arm required')
        require(pre.tobytes()==pre_bytes,'Original pre mutated')
        orders[arm]=study.rp.make_orders(chosen,score,meta);scores[arm]=np.asarray(score)[:n].copy();pres[arm]=pre[:n].copy()
    require(before==array_snapshot(arrays),'Original arrays changed')
    return orders,eligible,scores,pres


def estimate_counts(check):
    c={k:check.get(k) for k in COUNT_FIELDS}
    require(all(type(v) is int and v>=0 for v in c.values()),'Exact nonnegative integer counters required')
    n,w=c['events'],c['symbols_per_event'];require(n<=122 and w>0 and c['checked_cells']==n*w,'Counter axes')
    require(c['active_events']<=n and c['weak_events']<=n and c['required_cells']<=c['original_eligible_cells']<=n*w,
        'Required-event/cell bounds')
    require(c['required_cells']<=c['active_events']*w and c['rejected_cells']<=c['required_cells'] and
        c['candidate_eligible_cells']==c['original_eligible_cells']-c['rejected_cells'],'Eligibility counts')
    require(c['source_price_observations']==2*c['required_cells'] and c['benchmark_price_observations']==2*c['required_cells'] and c['spy_price_observations']==2*c['etf_required_cells'] and c['stock_required_cells']+c['etf_required_cells']==c['required_cells'] and
        c['new_model_fits']==c['predictive_model_fits']==0,'Endpoint/fit accounting')
    return c


model_counts=estimate_counts


def require_feature_audit(check,values,rows=None):
    for key in FEATURE_FLAGS:require(check.get(key) is True,'Independent feature missing '+key)
    old=np.asarray(values['original_eligible']);n=len(old) if rows is None else rows
    require(old.dtype==bool and old.ndim==2 and old.shape[0]==n,'Original eligibility axes')
    counts=estimate_counts(check)
    require(counts['events']==n and counts['symbols_per_event']==old.shape[1] and
        counts['original_eligible_cells']==int(old.sum()) and counts['required_cells']==int(values['required'].sum()) and
        counts['candidate_eligible_cells']==int(values['candidate_eligible'].sum()),'Independent coverage counts')
    require(check.get('counts')==counts and check.get('numerical_tolerance')==0. and
        check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Feature scope')
    validate_values(values,{'baseline':np.empty(old.shape),'eligible':old},n)


def require_selection_audit(check,rows):
    for key in SELECTION_FLAGS:require(check.get(key) is True,'Independent selection missing '+key)
    require(check.get('producer_modules_imported') is False and check.get('computes_returns') is False and
        check.get('full_scores_supplied') is True and check.get('full_pre_scores_supplied') is True and
        check.get('numerical_tolerance')==0. and check.get('original_source_parity_tolerance')==1e-12 and
        check.get('ordered_choices_checked')==2*rows,'Independent pre/full-score scope')


def require_outcome_audit(check):
    for key in ('passed','independent_accounts','independent_of_production_model_and_engine'):
        require(check.get(key) is True,'Independent outcome missing '+key)
    require(check.get('checked_paths')==6 and check.get('checked_slices')==30 and
        check.get('terminal_liquidation') is False,'Independent outcome coverage')
    require(check.get('accounting_source_sha256')==sha(ROOT/'work/ptr169_results_independent.py'),'Accounting source changed')
    require(set(check.get('paths',{}))==set(NAMES),'Independent arms')
    for paths in check['paths'].values():
        require(set(paths)=={p[0] for p in ev.PATHS},'Independent paths')
        for p in paths.values():require(p.get('passed') is True and p.get('monthly_executions')==122 and
            set(p.get('slices',{}))=={'full122','first88','early53','recent35','later34'},'Path coverage')


def select_first88(metrics,orders):
    require(set(metrics)==set(NAMES) and set(orders)==set(NAMES),'Two registered arms required')
    changed=[i for i in range(TRAIN) if set(orders[NAMES[1]][i]['symbols'])!=set(orders[NAMES[0]][i]['symbols'])]
    guards=dict(close_vs_incumbent=guard(metrics[NAMES[1]]['same_close_zero'],metrics[NAMES[0]]['same_close_zero'],.005),
        open_vs_incumbent=guard(metrics[NAMES[1]]['next_open_zero'],metrics[NAMES[0]]['next_open_zero'],0.),
        changed_support=len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed))
    return dict(primary=NAMES[1] if all(guards.values()) else NAMES[0],admitted=all(guards.values()),guards=guards,
        changed_months=changed,first88_only=True,later34_used=False)


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation is write-once')
    import monthly_ownsector_gate238_kernel as kernel
    import monthly_ownsector_gate238_independent as independent
    study=local();meta,arrays,px,op=study.load()
    require(meta==read(LOCAL/'META.json') and len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and
        len(arrays)==27,'Original122x49/27arrays required')
    before=array_snapshot(arrays);source,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'],'Raw source map differs');verify(raw)
    execution=study.verify_panel_parity(source,px,op);calendar=read(CALENDAR)['sessions']
    import ownsector238_repair_sources as repair
    admission=plan['price_source_binding']['source_admission']
    vox=repair.parse_vox(read(ROOT/admission['vox_prices_path']),calendar)
    panel={'adj':source['adj'],'vox':vox};source_report=read(SOURCE_REPORT)
    values,audit=kernel.build_features(panel,meta,calendar,arrays,source_report)
    check=independent.verify_features(panel,meta,calendar,arrays,source_report,values,audit)
    require_feature_audit(check,values,122);require(estimate_counts(check)==estimate_counts(audit),'Main counts differ')
    orders,elig,scores,pres=decisions(study,arrays,meta,values,122)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original full orders differ')
    selection=independent.verify_selection(arrays,meta,values,orders,scores=scores,pre_scores=pres,n=122)
    require_selection_audit(selection,122);prefixes=[]
    for n in FEATURE_PREFIXES:
        pp=feature_prefix_panel(panel,meta,n);aa=cache_prefix(arrays,n);mm=meta_prefix(meta,n)
        cc=feature_prefix_calendar(calendar,meta,n)
        rr=source_report_prefix(source_report,n)
        vv,fa=kernel.build_features(pp,mm,cc,aa,rr)
        fc=independent.verify_features(pp,mm,cc,aa,rr,vv,fa);require_feature_audit(fc,vv,n)
        require(estimate_counts(fc)==estimate_counts(fa),'Physical feature counts differ')
        for key in VALUE_KEYS:np.testing.assert_array_equal(vv[key],values[key][:n])
        require(fa['event_history']==audit['event_history'][:n] and fa['cell_history']==[c for c in audit['cell_history'] if c['query_index']<n],'Physical feature audit differs')
        oo,ee,ss,pr=decisions(study,aa,mm,vv,n)
        sc=independent.verify_selection(aa,mm,vv,oo,scores=ss,pre_scores=pr,n=n);require_selection_audit(sc,n)
        require(all(oo[k]==orders[k][:n] for k in NAMES),'Physical source orders differ')
        for k in NAMES:
            np.testing.assert_array_equal(ss[k],scores[k][:n]);np.testing.assert_array_equal(pr[k],pres[k][:n])
        prefixes.append(dict(prefix=n,source_rows={k:len(f) for k,f in pp.items()},source_last_session={k:str(f.index[-1].date()) if len(f) else None for k,f in pp.items()},report_records=len(rr['records']),
            calendar_rows=len(cc),calendar_last_session=cc[-1] if n else None,feature=fc,selection=sc,counts=estimate_counts(fa)))
    for n in range(123):
        aa=cache_prefix(arrays,n);mm=meta_prefix(meta,n);vv=value_prefix(values,n)
        oo,ee,ss,pr=decisions(study,aa,mm,vv,n)
        require(all(oo[k]==orders[k][:n] for k in NAMES),'All123 order prefix differs')
        for k in NAMES:
            np.testing.assert_array_equal(ss[k],scores[k][:n]);np.testing.assert_array_equal(pr[k],pres[k][:n])
    require(before==array_snapshot(arrays),'Original27 arrays modified')
    parity=dict(raw_source_hashes=raw,price_source_binding=plan['price_source_binding'],execution_prices=execution,
        calendar_sha256=sha(CALENDAR),original_array_fingerprints=array_fingerprints(arrays),
        original_arrays_unchanged=True,original_orders_exact=True,original_pre_preserved=True,
        original_E_for_pre=True,candidate_E_only_at_episode=True,strict_pit=False)
    counts=estimate_counts(audit)
    evidence=dict(feature=check,selection=selection,prefixes=123,physical_cache_meta_order_prefixes=True,
        feature_prefixes=list(FEATURE_PREFIXES),feature_prefix_sources_truncated=True,calendar_prefixes_truncated=True,physical_prefix_evidence=prefixes,
        original_orders_exact=True,original_arrays_unchanged=True,original_array_count=27,original_pre_preserved=True,
        original_core_preserved=True,original_E_for_pre=True,candidate_E_only_at_episode=True,single_original_episode=True,
        no_added_score_budget=True,model_counts=counts,selection_uses_zero_cost_paths_only=True,
        new_predictive_model_fits=0,model_fit_accounting=MODEL_FIT_ACCOUNTING,**counts)
    gate(expected);dest=OUT/'prepared';dest.mkdir();np.savez_compressed(dest/'FEATURES.npz',**values)
    for name,value in (('FEATURE_AUDIT.json',audit),('ORDERS.json',orders),('META.json',meta),
                       ('SOURCE_PARITY.json',parity),('INDEPENDENT_PREPARE.json',evidence)):dump(dest/name,value)
    completion=dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},original_array_count=27,model_counts=counts,
        new_returns_computed=False,new_predictive_model_fits=0,model_fit_accounting=MODEL_FIT_ACCOUNTING,**counts)
    verify_prepared_evidence(completion,values,evidence,meta=meta,parity=parity,feature_audit=audit)
    dump(dest/'COMPLETION.json',completion)
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest/'COMPLETION.json'),counts=counts)),flush=True)


def verify_prepared_evidence(c,values,evidence,*,meta,parity,feature_audit):
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Exact6 prepared artifacts required')
    require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and
        all(len(ss)==49 and len(set(ss))==49 for ss in meta['symbols']),'Full122x49 identities')
    require_feature_audit(evidence['feature'],values,122);require_selection_audit(evidence['selection'],122)
    counts=estimate_counts(feature_audit)
    require(counts==estimate_counts(evidence['feature'])==evidence['model_counts']==c['model_counts'],'Prepared counts differ')
    require([e['date'] for e in feature_audit['event_history']]==meta['dates'] and
        [e['symbols'] for e in feature_audit['event_history']]==meta['symbols'],'Feature identities differ')
    for key,value in dict(start_offset=252,end_offset=42,return_count=210).items():
        require(feature_audit.get(key)==value,'Source formation differs '+key)
    for key in ('rankings_recomputed','source_missing_filled','portfolio_returns_calculated'):
        require(feature_audit.get(key) is False,'Feature scope differs '+key)
    for key in ('original_orders_exact','original_arrays_unchanged','original_pre_preserved','original_core_preserved',
                'original_E_for_pre','candidate_E_only_at_episode','single_original_episode','no_added_score_budget',
                'feature_prefix_sources_truncated','calendar_prefixes_truncated','physical_cache_meta_order_prefixes','selection_uses_zero_cost_paths_only'):
        require(evidence.get(key) is True,'Prepared invariant '+key)
    require(evidence.get('prefixes')==123 and evidence.get('feature_prefixes')==list(FEATURE_PREFIXES) and
        [p['prefix'] for p in evidence.get('physical_prefix_evidence',[])]==list(FEATURE_PREFIXES),'Prefix evidence')
    for p in evidence['physical_prefix_evidence']:
        n=p['prefix'];vv=value_prefix(values,n);require_feature_audit(p['feature'],vv,n);require_selection_audit(p['selection'],n)
        require(p['counts']==estimate_counts(p['feature']) and set(p['source_rows'])=={'adj','vox'},'Physical counts differ')
        for key in ('adj','vox'):
            require(type(p['source_rows'][key]) is int and (p['source_rows'][key]>0 if n else p['source_rows'][key]==0),'Physical source rows differ')
        require(p.get('report_records')==p['counts']['required_cells'],'Physical report records differ')
        cc=feature_prefix_calendar(read(CALENDAR)['sessions'],meta,n)
        require(p.get('calendar_rows')==len(cc) and p.get('calendar_last_session')==(cc[-1] if n else None),
            'Physical calendar cutoff differs')
        lasts=p.get('source_last_session',{})
        require(set(lasts)=={'adj','vox'} and lasts['adj']==(meta['dates'][n-1] if n else None),'Physical original cutoff differs')
        require((isinstance(lasts['vox'],str) and lasts['vox']<=meta['dates'][n-1]) if n else lasts['vox'] is None,'Physical VOX cutoff differs')
    for record in (c,evidence):
        require(record.get('original_array_count')==27 and record.get('model_fit_accounting')==MODEL_FIT_ACCOUNTING,'Array/accounting')
        require(all(type(record.get(k)) is int and record[k]==0 for k in ('new_model_fits','new_predictive_model_fits','predictive_model_fits')),
            'Fit counts')
        require(all(type(record.get(k)) is int and record[k]==v for k,v in counts.items()),'Flat counts differ')
    require(c.get('new_returns_computed') is False,'Preparation cannot compute returns')
    for key in ('original_arrays_unchanged','original_orders_exact','original_pre_preserved','original_E_for_pre','candidate_E_only_at_episode'):
        require(parity.get(key) is True,'Source parity '+key)
    require(parity.get('calendar_sha256')==sha(CALENDAR) and parity.get('strict_pit') is False and
        len(parity.get('original_array_fingerprints',{}))==27,'Original27 fingerprints/calendar')
    for key in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact'):
        require(parity['execution_prices'].get(key) is True,'Execution parity '+key)


def prepared(expected):
    plan=gate(expected);dest=OUT/'prepared';c=read(dest/'COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected,'Preparation incomplete')
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Exact6 prepared artifacts required')
    for name,digest in c['artifacts_sha256'].items():require(sha(dest/name)==digest,'Changed prepared '+name)
    with np.load(dest/'FEATURES.npz',allow_pickle=False) as f:values={k:f[k] for k in f.files}
    meta=read(dest/'META.json');orders=read(dest/'ORDERS.json');parity=read(dest/'SOURCE_PARITY.json')
    require(set(orders)==set(NAMES) and all(len(v)==122 for v in orders.values()),'Prepared order axes')
    require(meta==read(LOCAL/'META.json') and orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original metadata/orders differ')
    require(parity['raw_source_hashes']==plan['raw_source_hashes'] and parity['price_source_binding']==plan['price_source_binding'],
        'Prepared source binding differs')
    verify_prepared_evidence(c,values,read(dest/'INDEPENDENT_PREPARE.json'),meta=meta,parity=parity,feature_audit=read(dest/'FEATURE_AUDIT.json'))
    return meta,orders

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
    import monthly_ownsector_gate238_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, own_sector_relative_qualification=True,zero_cost_selection=True,next_open_cost_diagnostic_only=True,original_pre_unchanged=True,candidate_eligibility_only_at_episode=True, observer_promotion=False))
    gate(expected)
    dump(dest / 'COMPLETION.json', dict(status='COMPLETE', created_utc=now(), plan_sha256=expected,
        frozen_sha256=sha(OUT / 'selected/FROZEN.json'), paths=6, slices=30,
        artifacts_sha256={p.name: sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='COMPLETE', primary=frozen['selection']['primary'],
        full_cagr={n: results[n]['same_close_zero']['metrics']['cagr'] for n in NAMES},
        completion_sha256=sha(dest / 'COMPLETION.json'))), flush=True)


def require_result_manifest(manifest):
    require(isinstance(manifest,dict) and set(manifest)==RESULT_FILES,
            'Exact26 result artifacts required; completion receipt is excluded')


def close(expected):
    gate(expected); completion = read(OUT / 'results/COMPLETION.json')
    require(completion['status'] == 'COMPLETE' and completion['plan_sha256'] == expected and
            completion['paths'] == 6 and completion['slices'] == 30, 'Incomplete diagnostics')
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
        result_completion_sha256=sha(OUT / 'results/COMPLETION.json'), new_trials=2, new_model_fits=0,
        predictive_model_fits=0, new_predictive_model_fits=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        original_array_count=27, protected_files=len(protected), best_training_selected_candidate=best,
        target_reached=best['cagr'] >= .7, observers_modified=False)
    receipt.update({key:read(OUT/'prepared/COMPLETION.json')[key] for key in ESTIMATE_FIELDS})
    dump(OUT / 'COMPLETION.json', receipt)
    state['completed'].append(dict(directory=OUT.name, trials=2, completed_utc=now(), primary=pick,
        primary_cagr=metrics['cagr'], primary_sharpe=metrics['sharpe'], new_admitted=int(results['selection']['admitted']),
        independent_execution_paths=6, protected_files=len(protected), target_reached=receipt['target_reached'],
        completion=dict(path=str(OUT / 'COMPLETION.json'), sha256=sha(OUT / 'COMPLETION.json'))))
    state.update(best_training_selected_candidate=best, latest_training_selected_candidate=dict(directory=OUT.name,
        arm=pick, selection=results['selection'], full_period_cagr=metrics['cagr'], full_period_sharpe=metrics['sharpe'], selected_on_first88=True),
        best_tracking_note=contract()['historical_best_tracking'], registered_queue=[], current_status='completed_independently_verified',
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Review fixed own-sector relative qualification and exposure changes; preserve frozen observers.')
    require(len(state['completed']) == 234 and sum(r['trials'] for r in state['completed']) == 1215, 'Wrong final counters')
    commit_state(before, state, 'close'); print(json.dumps(receipt), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'))
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    else:
        require(bool(args.plan_sha256),'Explicit immutable PLAN SHA required');globals()[args.phase](args.plan_sha256)

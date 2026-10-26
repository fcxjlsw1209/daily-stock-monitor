"""Independent round199 static code and bound-evidence review.

Reads Python source, JSON evidence and file digests only. Imports no strategy,
loads no market arrays, runs no formal tests, computes no actual gains/ranks,
orders or portfolio returns, and never registers the strategy.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/nested199_CODE_REVIEW.json'
SIX = (
    'work/monthly_nested_trend199_host.py',
    'work/test_monthly_nested_trend199_host.py',
    'work/monthly_nested_trend199_kernel.py',
    'work/test_monthly_nested_trend199_kernel.py',
    'work/monthly_nested_trend199_independent.py',
    'work/test_monthly_nested_trend199_independent.py',
)
EXPECTED = (
    'e2e08f78b719ad856432f6796a0d7b97aaf71d5cb55801890ed59c78226211fd',
    '25c84597002018166fc85cfc1363f4b0428d7c625b71d758f76b408b02e37679',
    '861963f7710b5821aa53410f69f7978cb1d6d76eed3c593248a2f9cefa52fb90',
    '989099829ccdace1c0b85ef09ec86bc5bbaff1f6b3d8b31546612559fdd12135',
    '5e8f5c622130465715d564f9ee101550850b2cdbc15c5cc7cf4ee84a1a93afe7',
    'da20f7c618773eb808eb59f10c47cee88027609ac70ec530514ce1e352c1d829',
)
CONTEXT = 'work/monthly_nested_trend199_context.json'
CONTEXT_SHA = '2a0b157600acf69ea9325fd3e46bc0a41d3337ff852e34677ee0d58d8b407233'
SOURCE = 'work/nested199_SOURCE_REVIEW.json'
SOURCE_SHA = 'd12cdcfc032404e297f1d95a2753785175db9892b71c0e8b49f241470bfad911'
TESTS = 'work/monthly_nested_trend199_TESTS.json'
TESTS_SHA = '48ee357cbe00b4df1f42f97514e0594dc1e621829a1434b32617dd148b5aa0bc'
PRIOR = 'outputs/monthly_event_risk_payoff198'
PRIOR_HASHES = {
    'PLAN.json': '4ee7b38ed136bcc74230ffd609b842ee82483878d07893549ba77b3e22b25426',
    'COMPLETION.json': 'a4e03ee8178dcaaf0f13829e0774e44ec72e7ea9c71fbf65f3e196e26f2bb351',
    'results/COMPLETION.json': '7ada48bcb926698b2f3a812c97e3281bbb9b9e582c78fb03217ea1ea7c8f0756',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    require(not OUT.exists(), 'Code-review receipt is write-once')
    require(not (ROOT / 'outputs/monthly_nested_trend199').exists(), 'Review must precede registration')
    pins = {}
    def bind(name, expected=None):
        path = Path(name)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require(path.is_relative_to(ROOT), 'Nonlocal review evidence')
        digest, key = sha(path), str(path.relative_to(ROOT))
        require(expected is None or digest == expected, 'Changed review evidence: ' + key)
        require(key not in pins or pins[key] == digest, 'Conflicting review evidence: ' + key)
        pins[key] = digest
        return digest

    code = {name: bind(name, digest) for name, digest in zip(SIX, EXPECTED)}
    bind(CONTEXT, CONTEXT_SHA); bind(SOURCE, SOURCE_SHA); bind(TESTS, TESTS_SHA); bind(__file__)
    context, source, tests = read(CONTEXT), read(SOURCE), read(TESTS)
    require(context['round'] == 199 and context['registered'] is False, 'Prospective round differs')
    require(context['arms'] == ['incumbent_local67', 'nested_min_core', 'nested_mean_control'], 'Arms differ')
    require(context['candidate'] == 'nested_min_core' and context['control_nonpromotable'] == 'nested_mean_control', 'Promotion contract differs')
    require(context['formation']['horizons'] == [126,189,252] and context['formation']['end_offset'] == -42 and
            context['formation']['start_offsets'] == [-168,-231,-294], 'Actual interval contract differs')
    require(context['integration']['blend'] == .5 and context['formation']['new_predictive_model_fits'] == 0,
            'Half-core or fit contract differs')
    require(source['passed'] is True and source['context_sha256'] == CONTEXT_SHA, 'Source review differs')
    for key in ('actual_features_computed','actual_orders_computed','actual_returns_computed',
                'actual_horizon_gains_computed','actual_horizon_ranks_computed'):
        require(source[key] is False, 'Source-only boundary crossed: ' + key)
    require(len(source['raw_source_hashes']) == 64 and len(source['source_sha256']) == 270, 'Source binding count differs')
    for mapping in (source['source_sha256'], source['raw_source_hashes'], context['evidence_sha256']):
        for name, digest in mapping.items(): bind(name, digest)
    require(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests'] == 62,
            'Expected completed62 synthetic tests')
    require(tests['runtime'] == dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'), 'Registered test runtime differs')
    for name, digest in tests['code_sha256'].items(): bind(name, digest)
    for name in SIX:
        require(tests['code_sha256'][str(ROOT/name)] == code[name], 'Tested code differs: ' + name)
    for name, digest in PRIOR_HASHES.items(): bind(PRIOR+'/'+name, digest)
    prior = read(PRIOR+'/PLAN.json')
    dependencies = (
        'outputs/monthly_local_linear70/study.py', 'outputs/monthly_local_linear70/PLAN.json',
        'outputs/monthly_local_linear70/INCUMBENT_POLICY.json',
        'outputs/monthly_core_timing70_research/core_adapter.py',
        'outputs/monthly_core_timing70_research/test_core_adapter.py',
        'outputs/monthly_path70_research/incumbent_adapter.py',
        'outputs/monthly_daily_episodes/episodes.py', 'outputs/monthly_quality_research/explore.py',
        'outputs/monthly_return_priority/return_priority.py',
        'work/ptr169_rebuild_evaluate.py', 'work/ptr169_results_independent.py',
        'work/price_distribution171_independent.py', 'work/monthly_sector_transition189_independent.py',
        'work/monthly_sector_recurrence192_independent.py',
        'outputs/monthly_paired_target143/CALENDAR.json',
        'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json',
        'run_local.sh', 'local_run.py', 'local.sandbox.sb',
    )
    for name in dependencies: bind(name, prior['input_sha256'].get(str(ROOT/name)))
    require(read('outputs/monthly_local_linear70/PLAN.json')['raw_source_hashes'] == source['raw_source_hashes'],
            'Original Local67 source map differs')
    closed, result = read(PRIOR+'/COMPLETION.json'), read(PRIOR+'/results/COMPLETION.json')
    require(closed['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED' and closed['protected_files'] == 21444
            and closed['observers_modified'] is False and closed['plan_sha256'] == PRIOR_HASHES['PLAN.json']
            and closed['result_completion_sha256'] == PRIOR_HASHES['results/COMPLETION.json'], 'Predecessor closure differs')
    require(result['status'] == 'COMPLETE' and result['paths'] == 9 and result['slices'] == 45, 'Predecessor results incomplete')
    require(len(read(dependencies[15])) == 21444, 'Protection manifest count differs')

    text = {name:(ROOT/name).read_text() for name in SIX}
    trees = {name:ast.parse(value) for name,value in text.items()}
    host = text[SIX[0]]
    functions = {node.name:ast.get_source_segment(host,node) for node in trees[SIX[0]].body if isinstance(node,ast.FunctionDef)}
    require(functions['prepare'].index('gate(expected)') < functions['prepare'].index('study.load()'), 'Actual compute gate absent')
    require(functions['register'].index('code_review_binding()') < functions['register'].index('OUT.mkdir()'), 'Review gate absent')
    require('if n: target[0]=np.nan' in functions['scale_and_delta'], 'First partial must use missing target')
    require("adapter.score_and_choose('incumbent66',BLEND,full_target" in functions['decisions'], 'Frozen half-core call differs')
    require("orders[arm]=study.rp.make_orders(chosen,score,meta)" in functions['decisions'], 'Episode choices must directly form orders')
    require('for n in range(123)' in functions['prepare'] and 'len(arrays) == 27' in functions['prepare'], 'Original/prefix checks absent')
    require("read(LOCAL / 'local_linear_ORDERS.json')" in functions['prepare'], 'Incumbent order parity absent')
    require('orders[arm][:TRAIN]' in functions['select'] and "meta['ends'][87]" in functions['select'], 'Training cutoff differs')
    require(functions['diagnose'].index('frozen_selection(expected, orders)') < functions['diagnose'].index('ev.run_path('), 'Later diagnostics lack frozen selection')
    require('monthly_nested_trend199_kernel' not in text[SIX[4]], 'Independent oracle imports producer')
    methods = {name:sum(isinstance(node,ast.FunctionDef) and node.name.startswith('test_') for node in ast.walk(trees[name])) for name in SIX[1::2]}
    require(list(methods.values()) == [30,15,17] and sum(methods.values()) == tests['tests'], 'Current test method count differs')

    checks = [
        dict(area='253-price common domain and actual nested intervals',passed=True,evidence=[SIX[2]+':measure/build_features',SIX[4]+':measure_oracle/independent_features',SOURCE],
             finding='Every cell requires all253 original stored prices on official sessions t-294..t-42 to be numeric finite positive. Intermediate missing prices reject the whole cell for all three horizons and both arms. End P252 is shared; starts P126/P63/P0 give126/189/252 actual simple-return intervals. No calendar compression, imputation, log-return replacement or eligibility/ETF filtering occurs before ranking. Finite-input gain overflow fails explicitly; finite underflow to gain-1 remains valid.'),
        dict(area='Original average percentile convention and independent reconstruction',passed=True,evidence=[SIX[2]+':percentile_rank',SIX[4]+':original_pct_ranks',SIX[0]+':scale_and_delta',*SIX[3::2]],
             finding='Producer uses pandas average ascending rank(pct=True)-.5. Independent scalar less/equal counts and host validation use (less+(equal+1)/2)/m-.5 on exactly the same complete-price population for each horizon. Singleton rank is+.5; all-tie offset is+.5/m; empty domains remain NaN. No m-1 denominator or recentering is used. Gains, ranks, masks, aggregates and complete cell audits compare exactly.'),
        dict(area='Direct min versus accurately summed mean',passed=True,evidence=[SIX[2]+':aggregate',SIX[4]+':aggregate_ranks',SIX[0]+':scale_and_delta',SIX[3],SIX[5]],
             finding='Candidate is minimum of the three original ranks; control is math.fsum(ranks)/3. Exact equality copies the common rank to both arms, preserving the identical binary64 value without near-equality snapping. No cross-sectional rerank, tanh, clipping, sign reversal or standardization follows aggregation. Missing targets remain NaN.'),
        dict(area='Half-core formula, first partial and finite zero',passed=True,evidence=[SIX[0]+':scale_and_delta/decisions',SIX[4]+':original_prefix/independent_deltas',SIX[1]],
             finding='Stored original C and cw are reused, with exact cw=.50 for calm and.52 otherwise. For finite target T only, delta=.5*(cw*T-C), bounded by.25/.26; new core is half old core plus half cw*T. Original core bounds and state/weight consistency are validated. First row target becomes NaN after raw validation, so its delta is exact zero. Other missing cells preserve original core/pre-score; finite0 remains a genuine neutral rank and produces-.5*C.'),
        dict(area='Original support and exactly one episode',passed=True,evidence=[SIX[0]+':decisions',SIX[1]+':RealAdapterTests',*dependencies[:9]],
             finding='Both arms call the frozen core_adapter with incumbent66, blend.5, full-shape NaN-suffixed targets, Local67 config4 modified_pre_score, original effects/policy/selector. Adapter applies the same nonzero delta to copied baseline/core/pre, then injects original baseline/core/cw into the single original episode. Original support is not recomputed from modified core. Host consumes returned choices/scores directly. Real-adapter synthetics capture original support and deliberately non-global choices to detect post-episode reranking; pinned adapter tests also cover cancellation-sensitive support and nonzero effects.'),
        dict(area='Original pool, arrays and causal prefixes',passed=True,evidence=[SIX[0]+':prepare/feature_prefix_panel',SIX[1],SIX[3],SIX[5]],
             finding='Actual preparation demands122x49 original domain and27 arrays, compares all arrays byte/dtype/shape unchanged, and requires exact frozen Local67 orders. It physically truncates source for prefixes0/1/53/88/122, with empty0, then compares all nine feature arrays and event histories. All123 order/target/delta prefixes must match. Synthetic tests poison future prices aftert-42, unused symbols, feature/core/weight/state suffixes, verify physical prefixes, and test symbol permutations. Full-calendar next-session lookup only verifies month-end metadata.'),
        dict(area='Descriptive counts and independent evidence validation',passed=True,evidence=[SIX[2]+':build_features',SIX[4]+':verify_features/verify_deltas',SIX[0]+':estimate_counts/verify_prepared_evidence'],
             finding='Source and available event counts are identical for all horizons and arms. Main-pass totals are253 prices,3 simple gains,3 ranks and2 aggregate scores per available cell. Complete independent scalar reconstruction checks all cell dates, gains, masks, ranks, exact-equality flags and counts, with strict audit value types. Prefix/oracle repetitions do not inflate the reported estimates; predictive model fits remain0.'),
        dict(area='Fixed first88 gates and control nonpromotion',passed=True,evidence=[SIX[0]+':guard/select_first88/select/frozen_selection',SIX[1]+':SelectionTests'],
             finding='Selection uses first88 orders through ends87 only. Candidate needs same-close CAGR+0.005 versus incumbent and cost-path noninferiority, with Sharpe.05/MDD.03/early53 and recent35 CAGR.025 guards; same guard structure checks both paths against mean control. At least3 changed holding sets, including each era, are required against each reference. Same-set reordering does not count. Mean never promotes, and later34 values cannot change the hash-bound frozen selection.'),
        dict(area='Independent outcome and original portfolio accounting',passed=True,evidence=[SIX[4]+':audit_outcomes',*dependencies[9:14],SIX[0]+':verify_training_against_audit'],
             finding='Pinned192-to189-to171 helpers propagate the three199 arms to independent scalar accounting. Required evidence covers9 paths,45 fixed slices,122 monthly executions, fill/fee/cash/units/NAV reconciliation, equal weights and no terminal liquidation. First88/early53/recent35 frozen training metrics are compared with independently verified full-path slices. Order generation retains selected three/two/cash membership; intramonth adjusted units and original execution engine remain unchanged.'),
        dict(area='Immutable source, runtime and phase governance',passed=True,evidence=[SIX[0]+':register/gate/prepared/close',CONTEXT,SOURCE,TESTS],
             finding='Before PLAN creation the host requires exact prospective context, source review,62-test runtime/source binding and independent six-file review. Original64 raw sources and270 source-review pins are preserved. All actual phases require explicit PLAN SHA and central registration. Prepared/selection/results phases are write-once, protected21444-file verification runs at registration/closure, central updates retain backups and compare bytes twice. Expected closure advances196/1105 to197/1108 only after independent evidence.'),
        dict(area='Synthetic evidence and independent reviewer scope',passed=True,evidence=[TESTS,*SIX[1::2]],
             finding='Reviewed all six final files and verified the formal62-test receipt against exact current source bytes:30 host,15 kernel,17 independent. Coverage includes hand-computed endpoints, simple/log distinction, common-price mask, overflow, original singleton/tie ranks, exact/near equality, direct min/mean, corrupt audit evidence, finite0/NaN, real adapter support and order wiring, prefixes/future poison, first88 isolation and lifecycle gates. Reviewer runs no production imports, actual feature/order/return computation, formal tests or registration.'),
    ]
    receipt = dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        unresolved_blockers=[],code_sha256=code,input_sha256=pins,review_code_sha256=sha(__file__),
        independent_reviewer='nested199_code_audit',
        review_method='Complete independent static review of final six sources/tests and frozen adapter call chain, followed by exact metadata/dependency digest validation; no production imports or market-array reads.',
        context_sha256=CONTEXT_SHA,source_review_sha256=SOURCE_SHA,formal_test_receipt_sha256=TESTS_SHA,
        actual_features_computed=False,actual_horizon_gains_computed=False,actual_horizon_ranks_computed=False,
        actual_aggregates_computed=False,actual_orders_computed=False,actual_portfolio_returns_computed=False,
        production_files_modified=False,strategy_registered_by_reviewer=False,production_modules_imported=False,
        formal_tests_rerun_by_reviewer=False,synthetic_test_receipt_verified=True,synthetic_tests=62,
        synthetic_test_methods=methods,review_checks=checks,limitations=[
            'This pre-registration review does not compute or certify actual199 feature availability, identification, holdings, returns or admission.',
            'Nested horizons share their endpoint and substantially overlap; the minimum does not supply independent trend observations or guaranteed stability.',
            'The noncalm half-core changes half of an existing volatility-adjusted core to raw-horizon ranks. Matched mean isolates aggregation between two otherwise identical replacements; it does not preserve the incumbent risk adjustment.',
            'Stored adjusted prices are retrospectively retrieved vendor data with inherited alias and extension conventions. Strict point-in-time membership/vintage and uniform corporate-action consistency remain uncertified.',
            'History has repeated research exposure and the original pool is proxyTop32+theme17; later34 is diagnostic rather than pristine out-of-sample.',
            'Missing historical horizon/concordance contracts leave exhaustive equivalence UNKNOWN; this receipt makes no global novelty claim.',
            'The host must verify all21444 protected file bytes during registration and closure. This review verifies the bound manifest rather than repeating that archive scan.',
        ])
    for name,digest in pins.items(): require(sha(ROOT/name) == digest,'Changed during review: '+name)
    with OUT.open('x') as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),review_code_sha256=sha(__file__),
        code_files=len(code),input_pins=len(pins),synthetic_tests_verified=62,unresolved_blockers=[])))


if __name__ == '__main__':
    main()

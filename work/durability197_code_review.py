"""Independent source/contract review receipt for round197.

Reads source text, JSON evidence and file digests only. No production imports,
market-array loading, recovery calculations, orders, replay or formal tests.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/durability197_CODE_REVIEW.json'
SIX = (
    'work/monthly_recovery_durability197_host.py',
    'work/test_monthly_recovery_durability197_host.py',
    'work/monthly_recovery_durability197_kernel.py',
    'work/test_monthly_recovery_durability197_kernel.py',
    'work/monthly_recovery_durability197_independent.py',
    'work/test_monthly_recovery_durability197_independent.py',
)
EXPECTED = (
    'af6e0a331e3f5eaf9a259d180c5b0e49db1d5ce72ec5fab1abbca7d9e54c93f4',
    '1dbf4492978f4abebdd97a95018222dc739aeb8939b9e30752ae6e487af36860',
    '27f84542f896862b93590478603db08598e514d17c89fd9797972654b12f03d7',
    '40655582df81149131638a8c76920982c5e2bbad144d06155a8da8bc955a58b1',
    'a5ee47f9d3629dd16d320e6802bd019fce69d84a7d9144afbce82f4ab154c631',
    '10d8ae0920c98239d1789d0c53fce5b257099ba91b7cd1a79a1e4a4be9c12d49',
)
CONTEXT = 'work/monthly_recovery_durability197_context.json'
CONTEXT_SHA = '89d00c63a1281700e1b55c490011d04c7ced178b29c9bac1f3d430f4389d1923'
SOURCE = 'work/durability197_SOURCE_REVIEW.json'
SOURCE_SHA = '2cc71fdf86d63bbb68c63149d078cbd6d686423b0f234e7f3c2e8b08fb6156cb'
TESTS = 'work/monthly_recovery_durability197_TESTS.json'
PRIOR = 'outputs/monthly_pairwise_majority196'
PRIOR_HASHES = {
    'PLAN.json': '650f8b4ba00c5c8ecc0db16b27cdb611f42bbf0d8337d9ef91e4c2ddc2af6893',
    'COMPLETION.json': '99132f0b7be990b95b8dbaef175e19818ddf45b69f683a26f5a9f5fd6294ac8d',
    'results/COMPLETION.json': 'c7aca0f43f69d864a16714cd0a13a117ccfb66caafc04f034e38ba61d78ed316',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    require(not OUT.exists(), 'Code-review receipt is write-once')
    pins = {}
    def bind(name, expected=None):
        path = Path(name)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require(path.is_relative_to(ROOT), 'Nonlocal review evidence')
        digest, key = sha(path), str(path.relative_to(ROOT))
        require(expected is None or expected == digest, 'Changed review evidence: ' + key)
        require(key not in pins or pins[key] == digest, 'Conflicting evidence pin')
        pins[key] = digest
        return digest

    code = {name: bind(name, digest) for name, digest in zip(SIX, EXPECTED)}
    bind(CONTEXT, CONTEXT_SHA); bind(SOURCE, SOURCE_SHA); bind(TESTS); bind(__file__)
    context, source, tests = read(CONTEXT), read(SOURCE), read(TESTS)
    require(context['arms'] == ['incumbent_local67', 'recovery_durability', 'recovery_frequency_control'], 'Arms differ')
    formation = context['formation']
    require((formation['earliest_offset'], formation['latest_offset'], formation['price_count'],
             formation['anchor_last'], formation['recovery_horizon'], formation['durability_horizon'])
            == (294, 42, 253, 210, 21, 21), 'Formation differs')
    require(context['integration']['dose'] == .01 and context['integration']['max_absolute_delta'] == .005,
            'Fixed correction differs')
    require(source['passed'] is True and source['context_sha256'] == CONTEXT_SHA, 'Source review differs')
    for key in ('actual_features_computed', 'actual_orders_computed', 'actual_returns_computed',
                'actual_negative_anchors_computed', 'actual_recovery_comparisons_computed',
                'actual_durability_comparisons_computed', 'actual_ranks_computed'):
        require(source[key] is False, 'Source-only review computed ' + key)
    require(len(source['raw_source_hashes']) == 64, 'Original raw64 map incomplete')
    for name, digest in source['source_sha256'].items(): bind(name, digest)
    for name, digest in source['raw_source_hashes'].items(): bind(name, digest)
    require(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests'] == 56,
            'Expected completed56 synthetic tests')
    require(tests['runtime'] == dict(python='3.9.6', numpy='2.0.2', pandas='2.3.3'), 'Test runtime differs')
    for name, digest in tests['code_sha256'].items(): bind(name, digest)
    for name in SIX:
        require(tests['code_sha256'][str(ROOT / name)] == code[name], 'Tested code differs: ' + name)
    for name, digest in context['evidence_sha256'].items(): bind(name, digest)
    for name, digest in PRIOR_HASHES.items(): bind(PRIOR + '/' + name, digest)
    prior = read(PRIOR + '/PLAN.json')
    dependency_files = (
        'outputs/monthly_local_linear70/study.py',
        'outputs/monthly_local_linear70/PLAN.json',
        'outputs/monthly_local_linear70/INCUMBENT_POLICY.json',
        'outputs/monthly_core_timing70_research/core_adapter.py',
        'outputs/monthly_path70_research/incumbent_adapter.py',
        'work/ptr169_rebuild_evaluate.py', 'work/ptr169_results_independent.py',
        'work/price_distribution171_independent.py',
        'work/monthly_sector_transition189_independent.py',
        'work/monthly_sector_recurrence192_independent.py',
        'outputs/monthly_paired_target143/CALENDAR.json',
        'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json',
    )
    for name in dependency_files:
        bind(name, prior['input_sha256'].get(str(ROOT/name)))
    require(read('outputs/monthly_local_linear70/PLAN.json')['raw_source_hashes'] == source['raw_source_hashes'],
            'Original Local67 source map differs')
    closed, result = read(PRIOR+'/COMPLETION.json'), read(PRIOR+'/results/COMPLETION.json')
    require(closed['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED' and closed['protected_files'] == 21444
            and closed['observers_modified'] is False and closed['plan_sha256'] == PRIOR_HASHES['PLAN.json']
            and closed['result_completion_sha256'] == PRIOR_HASHES['results/COMPLETION.json'], 'Predecessor closure differs')
    require(result['status'] == 'COMPLETE' and result['paths'] == 9 and result['slices'] == 45, 'Predecessor results incomplete')
    require(len(read(dependency_files[-1])) == 21444, 'Protection manifest count differs')

    text = {name: (ROOT/name).read_text() for name in SIX}
    trees = {name: ast.parse(value) for name, value in text.items()}
    host = text[SIX[0]]
    functions = {node.name: ast.get_source_segment(host, node) for node in trees[SIX[0]].body
                 if isinstance(node, ast.FunctionDef)}
    require(functions['prepare'].index('gate(expected)') < functions['prepare'].index('study.load()'), 'Missing pre-compute gate')
    require(functions['register'].index('code_review_binding()') < functions['register'].index('OUT.mkdir()'), 'Missing registration review gate')
    require('orders[arm][:TRAIN]' in functions['select'] and "meta['ends'][87]" in functions['select'], 'Training cutoff differs')
    require(functions['diagnose'].index('frozen_selection(expected, orders)') < functions['diagnose'].index('ev.run_path('), 'Diagnostics precede frozen selection')
    require('if n:delta[0]=0.' in functions['scale_and_delta'] and 'for n in range(123)' in functions['prepare'], 'Initial/prefix rules differ')
    require('len(arrays) == 27' in functions['prepare'] and "read(LOCAL / 'local_linear_ORDERS.json')" in functions['prepare'], 'Original invariants absent')
    require('monthly_recovery_durability197_kernel' not in text[SIX[4]], 'Independent oracle imports producer')
    methods = {name: sum(isinstance(n, ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(trees[name]))
               for name in SIX[1::2]}
    require(sum(methods.values()) == 56, 'Formal test count/source mismatch')

    checks = [
        dict(area='First recovery and fixed post-recovery durability', passed=True,
             evidence=[SIX[2]+':measure', SIX[4]+':measure_oracle', SIX[3], SIX[5]],
             finding='Negative anchors are exactly s1..210 with P[s]<P[s-1]. Both implementations select the first crossing at k1..21 with P[s+k]>=P[s-1]. Durability inspects precisely recovery+1..recovery+21, excluding the recovery close, and accepts equality; any strict breach fails. The last possible maturity is252, ending t-42. Production minimum and independent scalar survival loop reconstruct the full identical anchor paths; raw Q=durable/negative and C=recovered/negative use the same denominator.'),
        dict(area='Causal source window and matched availability', passed=True,
             evidence=[SIX[2]+':source_prices', SIX[2]+':build_features', SIX[4]+':independent_features', SOURCE],
             finding='Exactly253 official-session prices from t-294 through t-42 are all required positive and finite, including values outside any realized recovery path. Original event candidates including ETFs are treated equally without a new eligibility filter. No negative anchor means common feature-unavailability/NaN despite valid source; positive anchor count with no recovery is common valid zero. A singleton remains raw-available, unlike196, but its rank/correction is zero. No smoothing or minimum-anchor cutoff is added.'),
        dict(area='Ranks, fixed budget and first partial row', passed=True,
             evidence=[SIX[2]+':centered_rank', SIX[4]+':centered_ranks', SIX[4]+':verify_deltas', SIX[0]+':scale_and_delta'],
             finding='Higher values preferred, average tied zero-based rank/(n-1)-.5 over the common feature-available event population. Singleton/all-tied populations rank0. Host and independent enforce0<=Q<=C<=1 and reconstruct saved ranks exactly. Missing ranks/corrections0; direct .01 multiplication has maximum.005. Only the initial-row correction is suppressed; initial raw features and source evidence remain intact.'),
        dict(area='Descriptive count accounting', passed=True,
             evidence=[SIX[2]+':build_features', SIX[4]+':verify_features', SIX[0]+':estimate_counts'],
             finding='Source and feature availability are separate per-event lists. Price observations equal253 times source-valid cells. Each returned event has available<=negative<=210*available and0<=durable<=recovered<=negative; all three anchor totals equal their per-event sums. The independent audit compares all source/cell/path fields, not merely totals. Main-pass counts exclude repeated independent/prefix verification; predictive fits remain zero.'),
        dict(area='Original Local67 and single episode integration', passed=True,
             evidence=[SIX[0]+':decisions', SIX[0]+':prepare', *dependency_files[:5]],
             finding='Incumbent uses original config4 choices and must match all frozen orders. New arms start from complete config4 modified_pre_score, add only nonzero deltas, then call zero-blend original adapter exactly once. Original effects, policy, core and auxiliary support are supplied to the existing boundary selector without post-episode global reranking. All27 original arrays are byte/shape/dtype checked unchanged.'),
        dict(area='Future isolation and prefix validation', passed=True,
             evidence=[SIX[0]+':feature_prefix_panel', SIX[0]+':prepare', SIX[3], SIX[5]],
             finding='Numeric source reads are confined to each requested formation window and every anchor fully matures before its end. Host physically truncates adjusted close through each signal date for prefixes0/1/53/88/122, empty at0, and checks full event audit/all five arrays exactly. All123 decision/rank/delta prefixes are checked. Synthetic future poison, unrelated symbols, truncated source and permutations cover independence. Next-session lookup verifies calendar monthend, not a future market value.'),
        dict(area='First88 selection and control identification', passed=True,
             evidence=[SIX[0]+':guard', SIX[0]+':select_first88', SIX[0]+':select', SIX[0]+':frozen_selection'],
             finding='Only first88 order prefixes through ends[87] enter selection; early53/recent35 are subdivisions. Candidate requires +.005 same-close CAGR over incumbent, next-open5bp noninferiority, Sharpe tolerance.05/MDD.03/eraCAGR.025, both-path noninferiority versus recovery-frequency control, and at least3 changed holding sets with both eras against each reference. Control cannot promote; later34 runs only after hash-bound FROZEN and cannot change selection.'),
        dict(area='Independent outcome accounting', passed=True,
             evidence=[SIX[4]+':audit_outcomes', *dependency_files[5:10]],
             finding='The pinned192→189→171 adapter propagates the three197 arm names to independent scalar accounting. It checks all9 paths and45 slices,122 monthly executions, fixed dates, fill prices, fee/cash/units/NAV/self-financing/equal-weight reconciliation, annual metrics and no terminal liquidation. Frozen training metrics are rechecked against independently verified first88/early53/recent35 exports.'),
        dict(area='Source, runtime, phase and closure governance', passed=True,
             evidence=[SIX[0]+':register', SIX[0]+':gate', SIX[0]+':prepared', SIX[0]+':close', TESTS, SOURCE],
             finding='Exact formal synthetic-test source/runtime receipt and independent six-file review are required before PLAN creation. Original64 raw map, current248-pin source review, predecessor196 closure, inherited sources and official calendar remain bound. All actual phases require explicit PLAN SHA/central registration, verify sources/prepared manifests, and refuse replacement. Registration and closure scan21444 protected files. Central commits retain backups and compare original bytes twice;194/1099 becomes195/1102 only on closure.'),
        dict(area='Synthetic evidence and reviewer scope', passed=True,
             evidence=[TESTS, *SIX[1::2]],
             finding='Exact current six-file hashes match the recorded completed synthetic suite, including same recovery/different durability, first versus later crossing, lag1/21 and maturity252 edges, fixed21-close post-exposure, equality/strict breach, no-anchor versus true zero, singleton rank0, all253 validity, futures, corrupted path/raw/rank/count evidence and selection isolation. Reviewer inspected sources and receipt; did not rerun tests or compute actual features/orders/returns.'),
    ]
    receipt = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        unresolved_blockers=[], code_sha256=code, input_sha256=pins, review_code_sha256=sha(__file__),
        independent_reviewer='observer_notification_audit', review_method='Complete six-source/test static review plus pinned metadata and dependency digest validation; no production imports.',
        context_sha256=CONTEXT_SHA, source_review_sha256=SOURCE_SHA,
        actual_features_computed=False, actual_negative_anchors_computed=False,
        actual_recovery_comparisons_computed=False, actual_durability_comparisons_computed=False,
        actual_ranks_computed=False, actual_orders_computed=False, actual_portfolio_returns_computed=False,
        production_files_modified=False, strategy_registered_by_reviewer=False, production_modules_imported=False,
        formal_tests_rerun_by_reviewer=False, synthetic_test_receipt_verified=True,
        synthetic_tests=56, synthetic_test_methods=methods, review_checks=checks,
        limitations=[
            'Source-only coverage is already recorded; this review does not compute or certify actual197 feature availability, identification, orders, returns or admission.',
            'Stored adjusted close uses retrospective vendor adjustment, alias and early-extension conventions; strict original-public vintage and uniform corporate-action consistency remain uncertified.',
            'Exact comparisons start from stored binary64 prices and cannot reverse prior vendor rounding. Common positive scaling preserves order mathematically but does not certify within-window source splices.',
            'Anchors and post-recovery periods overlap and are not independent samples. Matched frequency control does not match loss severity, volatility, beta or every path attribute;Q<=C implies no performance ordering.',
            'Repeated research sample, estimatedTop32+theme17 and idealized same-close assumptions remain. Later34 is diagnostic, not pristine out-of-sample.',
            'The host enforces all21444 protected bytes at registration/closure. This receipt validates the bound manifest rather than repeating the complete archive scan.',
        ])
    for name, digest in pins.items(): require(sha(ROOT/name) == digest, 'Changed during review: '+name)
    with OUT.open('x') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True, path=str(OUT), sha256=sha(OUT),
        review_code_sha256=sha(__file__), code_files=len(code), input_pins=len(pins),
        synthetic_tests_verified=56, unresolved_blockers=[])))


if __name__ == '__main__':
    main()

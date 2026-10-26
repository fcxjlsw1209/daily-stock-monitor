"""Publish the independent static/numerical-contract review for round195.

Only source text, JSON metadata and SHA-256 bytes are read.  This script imports
no production model, source loader, feature builder, selector or replay engine.
It does not rerun the formal tests or any strategy stage.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/volume195_CODE_REVIEW.json'
SIX = (
    'work/monthly_volume_cluster195_host.py',
    'work/test_monthly_volume_cluster195_host.py',
    'work/monthly_volume_cluster195_kernel.py',
    'work/test_monthly_volume_cluster195_kernel.py',
    'work/monthly_volume_cluster195_independent.py',
    'work/test_monthly_volume_cluster195_independent.py',
)
EXPECTED = (
    '53d110d4f4cacd4454b82e3bf61558298ca86e0c57c42e5696c947d36bdd8262',
    '003b2bd39f68b641523e2826d0d28dd8298dfe49eea71fc3b0eb159fab2a81c3',
    '47ee029a10cf19c7a2e340429551b2dae80e8f9cb00a6bbad37aa18708e8d7ed',
    '9cdf617aa604eb8e37d22378083d0a443d19bc98372f1a039f020366c8a8ec32',
    'f42f75e475aff7a7c205162d16339ce953d624c322f2ec3f6a9ee322ced868d4',
    'df42141ab3bb8ac99cd5686e2d12ea51247cebe09d4be29e82e96620a0714c79',
)
CONTEXT = 'work/monthly_volume_cluster195_context.json'
CONTEXT_SHA = '1a6e06499d35667ae435887f3d2d2e56fce4ea4194531032b2ab4969b046b0a0'
SOURCE_REVIEW = 'work/volume195_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA = 'ea5d8c96d8c340cf01d6a6db4ec424a1379c03a6545446039821d2e17828fe40'
TESTS = 'work/monthly_volume_cluster195_TESTS.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(relative):
    return json.loads((ROOT / relative).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    require(not OUT.exists(), 'Independent code-review receipt is write-once')
    pins = {}
    def bind(path, expected=None):
        path = Path(path)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require(path.is_relative_to(ROOT), 'Nonlocal review dependency')
        digest = sha(path)
        require(expected is None or digest == expected, 'Dependency changed: ' + str(path))
        key = str(path.relative_to(ROOT))
        require(key not in pins or pins[key] == digest, 'Conflicting dependency')
        pins[key] = digest
        return digest

    code = {}
    for name, digest in zip(SIX, EXPECTED):
        code[name] = bind(name, digest)
    bind(CONTEXT, CONTEXT_SHA)
    bind(SOURCE_REVIEW, SOURCE_REVIEW_SHA)
    bind(TESTS)
    bind(Path(__file__))
    context, source, tests = read(CONTEXT), read(SOURCE_REVIEW), read(TESTS)
    require(context['arms'] == ['incumbent_local67', 'volume_cluster', 'volume_frequency_control'], 'Arms differ')
    require(context['formation']['volume_count'] == 273 and context['formation']['earliest_offset'] == 314
            and context['formation']['latest_offset'] == 42, 'Window differs')
    require(context['integration']['dose'] == .01 and context['integration']['max_absolute_delta'] == .005,
            'Dose differs')
    require(source['passed'] is True and source['context_sha256'] == CONTEXT_SHA and
            source['actual_features_computed'] is False and source['actual_orders_computed'] is False and
            source['actual_returns_computed'] is False, 'Source-only receipt scope differs')
    for path, digest in source['source_sha256'].items():
        bind(path, digest)
    require(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests'] == 56,
            'Expected completed56 synthetic tests')
    require(tests['runtime'] == dict(python='3.9.6', numpy='2.0.2', pandas='2.3.3'), 'Test runtime differs')
    for path, digest in tests['code_sha256'].items():
        bind(path, digest)
    for name in SIX:
        require(tests['code_sha256'][str(ROOT/name)] == code[name], 'Tested numerical file differs')

    texts = {name: (ROOT/name).read_text() for name in SIX}
    trees = {name: ast.parse(text) for name, text in texts.items()}
    host = texts[SIX[0]]
    functions = {node.name: ast.get_source_segment(host, node) for node in trees[SIX[0]].body
                 if isinstance(node, ast.FunctionDef)}
    require(functions['prepare'].index('gate(expected)') < functions['prepare'].index('study.load()'),
            'Preparation must verify registration before loading actual arrays')
    require(functions['register'].index('code_review_binding()') < functions['register'].index('OUT.mkdir()'),
            'Registration must require review before publishing')
    require("orders[arm][:TRAIN]" in functions['select'] and "meta['ends'][87]" in functions['select'],
            'Selection cutoff differs')
    require(functions['diagnose'].index('frozen_selection(expected, orders)') <
            functions['diagnose'].index('ev.run_path('), 'Later diagnostics require frozen selection')
    require("if n:delta[0]=0." in functions['scale_and_delta'] and
            'for n in range(123)' in functions['prepare'], 'Initialization/prefix checks differ')
    require("read(LOCAL / 'local_linear_ORDERS.json')" in functions['prepare'] and
            "len(arrays) == 27" in functions['prepare'], 'Original source invariants missing')
    require('monthly_volume_cluster195_kernel' not in texts[SIX[4]], 'Independent oracle imports producer')
    test_methods = {name: sum(isinstance(n, ast.FunctionDef) and n.name.startswith('test_')
                             for n in ast.walk(trees[name])) for name in SIX[1::2]}
    require(sum(test_methods.values()) == tests['tests'], 'Test method/receipt count mismatch')

    # Bind and inspect inherited execution interfaces, without importing them.
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
        'outputs/monthly_trade_wc194/PLAN.json',
        'outputs/monthly_trade_wc194/COMPLETION.json',
        'outputs/monthly_trade_wc194/results/COMPLETION.json',
        'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json',
    )
    prior = read('outputs/monthly_trade_wc194/PLAN.json')
    for name in dependency_files:
        bind(name, prior['input_sha256'].get(str(ROOT/name)))
    require(read('outputs/monthly_local_linear70/PLAN.json')['raw_source_hashes'] == source['raw_source_hashes'],
            'Original64 source map differs')
    closed = read('outputs/monthly_trade_wc194/COMPLETION.json')
    require(closed['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED' and
            closed['protected_files'] == 21444 and closed['observers_modified'] is False,
            'Predecessor closure metadata differs')
    require(len(read('work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json')) == 21444,
            'Protected manifest is incomplete')

    checks = [
        dict(area='Formula and causal support', passed=True,
             evidence=[SIX[2]+':measure', SIX[2]+':build_features', SIX[4]+':measure_oracle'],
             finding='Exactly273 official sessions t-314..t-42;210 event positions t-251..t-42. Each comparison excludes its own observation and uses the32nd sorted prior63 value. Equality is false. Q=(209*N11-Nleft*Nright)/209^2; F=N1/210. Production integer arithmetic agrees with independent Fraction rounding.'),
        dict(area='Common availability and ranking', passed=True,
             evidence=[SIX[2]+':centered_rank', SIX[4]+':arrays', SIX[0]+':scale_and_delta'],
             finding='All273 positive finite numeric volumes required for both arms. Missing numeric/symbol/session support gives NaN raw and zero rank/delta; constant positive volume remains valid. Average-tie zero-based percentile divided by n-1 minus.5; n<=1 and all ties zero. Domain is all common-available original49 candidates, including ETFs and without a new trading-eligibility/model filter.'),
        dict(area='Score integration and initial row', passed=True,
             evidence=[SIX[0]+':decisions', SIX[0]+':prepare', dependency_files[0], dependency_files[3]],
             finding='Fixed .01 times saved verified rank, bound.005. Initial partial-row delta is zero while its feature availability is retained. Only nonzero increments are added to exact Local67 config4 full pre-score. Zero-blend original adapter receives unchanged effects/policy/support and runs the original episode once for each arm. Original27 arrays are compared byte-for-byte and original full orders must match.'),
        dict(area='No future values and prefix checks', passed=True,
             evidence=[SIX[0]+':feature_prefix_panel', SIX[0]+':prepare', SIX[3], SIX[5]],
             finding='Kernel numerically inspects only requested historical windows. Host physically truncates volume for0/1/53/88/122 feature prefixes, including an empty time axis at0; requires exact arrays and event audit. It verifies all123 order/rank/delta prefixes. Synthetic future poison and future-security tests exercise unused suffix isolation. Official monthend verification may inspect next calendar date but no future market value.'),
        dict(area='Selection and control identification', passed=True,
             evidence=[SIX[0]+':select_first88', SIX[0]+':select', SIX[0]+':frozen_selection'],
             finding='Only first88 orders through ends[87] produce training metrics. Candidate requires same-close CAGR+.005 versus incumbent, next-open5bp noninferiority, both-path risk/era guards, noninferiority versus frequency control and at least3 changed holding sets including each era against both references. Control cannot promote. Later34 is computed only after FROZEN and cannot alter selection.'),
        dict(area='Independent accounting and frozen training validation', passed=True,
             evidence=[SIX[4]+':audit_outcomes', dependency_files[9], dependency_files[8], dependency_files[7], dependency_files[6]],
             finding='Hash-pinned192→189→171 independent adapter explicitly propagates new arm names to scalar account auditing. All3arms ×3execution paths ×5slices are required, with122monthly executions, fixed interval axes, independent fill/cash/units/NAV/fees/turnover/annual metrics and no terminal liquidation. Selected training metrics are rechecked against the matching independent full-export slices.'),
        dict(area='Registration, integrity and write-once phases', passed=True,
             evidence=[SIX[0]+':register', SIX[0]+':gate', SIX[0]+':prepared', SIX[0]+':close'],
             finding='Formal test/runtime and exact six-file independent review precede registration. Original64 source map and230-pin source review, predecessor194 closure, inherited dependency map, official calendar and21444-member protection manifest are bound. Every actual phase requires explicit PLAN SHA/current queue/source/runtime checks. Prepared/selected/results/completion refuse replacement, verify their artifact hashes, and central commits compare before-bytes twice with preserved backup. Counters192/1093 advance to193/1096 only at closure.'),
        dict(area='Tests and review scope', passed=True,
             evidence=[TESTS, *SIX[1::2]],
             finding='All56 recorded synthetic tests bind the exact current sources. Reviewed cases include hand-counted covariance, prior-window/equality edges, power-of-two scaling and time-varying-unit counterexample, missing/zero/nonfinite input, degenerate populations/ties, future poisoning, wrong masks/ranks/audit counters, score integration, source/review digest failure and later34 selection isolation. This reviewer verified the receipt and source, without rerunning formal tests or actual data.'),
    ]
    result = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True, unresolved_blockers=[], code_sha256=code, input_sha256=pins,
        review_code_sha256=sha(Path(__file__)), independent_reviewer='observer_notification_audit',
        review_method='Read complete six sources/tests, trace frozen original selector and independent accounting interfaces, verify pinned metadata/test receipt, static structural checks only.',
        actual_features_computed=False, actual_financial_ratios_computed=False,
        actual_orders_computed=False, actual_portfolio_returns_computed=False,
        actual_volume_event_indicators_computed=False, actual_rolling_medians_computed=False,
        production_files_modified=False, strategy_registered_by_reviewer=False,
        production_modules_imported=False, synthetic_test_receipt_verified=True,
        formal_tests_rerun_by_reviewer=False, synthetic_tests=56, synthetic_test_methods=test_methods,
        source_review_sha256=SOURCE_REVIEW_SHA, context_sha256=CONTEXT_SHA,
        review_checks=checks,
        limitations=[
            'This code review does not establish actual strategy availability, identification, returns or admission; only root may perform the registered lifecycle.',
            'The source-only review measured273-session coverage before strategy registration; no median/event/raw feature/rank was computed in that review.',
            'Original stored volume mixes normalized signal-volume and raw overrides. Cross-provider/action consistency and strict historical vintages remain uncertified; no new masks or adjustment rules are introduced.',
            'Source panels contain retrospective Top32/theme17 research proxies and repeatedly used historical samples. Neither novelty nor pristine OOS performance is certified.',
            'All21444 protected bytes are enforced by host registration/closure; this code review verifies the bound manifest rather than repeating the full archive scan.',
        ])
    for name, digest in pins.items():
        require(sha(ROOT/name) == digest, 'Dependency changed during review: '+name)
    with OUT.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True, path=str(OUT), sha256=sha(OUT),
        review_code_sha256=sha(Path(__file__)), code_files=len(code), input_pins=len(pins),
        synthetic_tests_verified=56, unresolved_blockers=[])))


if __name__ == '__main__':
    main()

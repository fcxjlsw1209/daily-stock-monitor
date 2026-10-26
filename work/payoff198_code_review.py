"""Independent source/contract review receipt for round198.

Reads source text, JSON evidence and file digests only. No production imports,
market-array loading, event/risk/payoff calculations, orders, replay or formal tests.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/payoff198_CODE_REVIEW.json'
SIX = (
    'work/monthly_event_risk_payoff198_host.py',
    'work/test_monthly_event_risk_payoff198_host.py',
    'work/monthly_event_risk_payoff198_kernel.py',
    'work/test_monthly_event_risk_payoff198_kernel.py',
    'work/monthly_event_risk_payoff198_independent.py',
    'work/test_monthly_event_risk_payoff198_independent.py',
)
EXPECTED = (
    '3db4f82ac7c7de494c1923871fcba3bf6115eeb3c8b3a24d47f832ab3bf367f9',
    '5a9553a5defbf2c3212e0c458917588014895bece0deae9fd96c3eb39249fa29',
    'd48097863305ba70410c10bae060263a5bcc1852efcd880ec068d4ffd1a19e29',
    'ca2728ed45fd4ed5d5c56ed107bbe6cb7e5271cf6c6c1142389c77bd22c53feb',
    '2b33b8ce09ff719fee5b4e6a591cd9c5353da7411d3c22b4ef956dd6b04d6b1e',
    'c04b1e51399219473c607ac72b51faf641bba10981588e7efbca01b73cd724e9',
)
CONTEXT = 'work/monthly_event_risk_payoff198_context.json'
CONTEXT_SHA = 'd31fcf18ed0c00aae487186c5dd0b1352594c5249cc51f69e87d7c70bcb5f01a'
SOURCE = 'work/payoff198_SOURCE_REVIEW.json'
SOURCE_SHA = '40848c92606cff55942492cafaefb0a7367a47435a5e6e3ce23cba4e8d78c84b'
TESTS = 'work/monthly_event_risk_payoff198_TESTS.json'
TESTS_SHA = '0d8e96f76135dd536a18f7bebd404092da051c90c5e990b7823b4d71f876c543'
PRIOR = 'outputs/monthly_recovery_durability197'
PRIOR_HASHES = {
    'PLAN.json': 'b4fe6309cc251bbd061e5728956f160d71f86dd6c13e6fa114af37add31f2ffc',
    'COMPLETION.json': 'b2b07861208cc9bd7a5a632d4c679efdec4f85eb00512fc971845891bf61deb5',
    'results/COMPLETION.json': '4e6e34526b8f1b6099274491d82197a1ce4559e927451c0d077028ef8cd68113',
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
    bind(CONTEXT, CONTEXT_SHA); bind(SOURCE, SOURCE_SHA); bind(TESTS, TESTS_SHA); bind(__file__)
    context, source, tests = read(CONTEXT), read(SOURCE), read(TESTS)
    require(context['arms'] == ['incumbent_local67', 'event_risk_payoff', 'inverse_risk_mass_control'], 'Arms differ')
    formation = context['formation']
    require((formation['price_count'], formation['skip_sessions'], formation['anchor_first'],
             formation['anchor_last'], formation['risk_returns'], formation['payoff_horizon'],
             formation['cooldown'], formation['prior_count']) == (253, 42, 64, 231, 63, 21, 21, 20), 'Formation differs')
    require(context['integration']['dose'] == .01 and context['integration']['max_absolute_delta'] == .005,
            'Fixed correction differs')
    require(source['passed'] is True and source['context_sha256'] == CONTEXT_SHA, 'Source review differs')
    for key in ('actual_features_computed', 'actual_orders_computed', 'actual_returns_computed',
                'actual_negative_anchors_computed', 'actual_selected_anchors_computed',
                'actual_pre_event_volatility_computed', 'actual_payoffs_computed', 'actual_ranks_computed'):
        require(source[key] is False, 'Source-only review computed ' + key)
    require(len(source['raw_source_hashes']) == 64, 'Original raw64 map incomplete')
    for name, digest in source['source_sha256'].items(): bind(name, digest)
    for name, digest in source['raw_source_hashes'].items(): bind(name, digest)
    require(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests'] == 59,
            'Expected completed59 synthetic tests')
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
    require('monthly_event_risk_payoff198_kernel' not in text[SIX[4]], 'Independent oracle imports producer')
    methods = {name: sum(isinstance(n, ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(trees[name]))
               for name in SIX[1::2]}
    require(sum(methods.values()) == 59, 'Formal test count/source mismatch')

    checks = [
        dict(area='Causal negative anchors, cooldown and pre-event risk', passed=True,
             evidence=[SIX[2]+':measure', SIX[4]+':measure_oracle', SIX[3], SIX[5]],
             finding='Both implementations first select strict negative anchors s64..231 greedily, with next s>=previous+21 and at most8 anchors. Selection uses no risk validity or payoff. Each risk uses exactly63 simple returns j=s-63..s-1 from64 prices s-64..s-1, excluding the loss day. Payoff is log(P[s+21])-log(P[s]); latest maturity252 is formation end t-42. No current drawdown filter or sign-based label exclusion exists.'),
        dict(area='Risk-payoff pairing and matched aggregate control', passed=True,
             evidence=[SIX[2]+':aggregate_payoffs', SIX[4]+':measure_oracle', SIX[3], SIX[5]],
             finding='Q=fsum(y/sigma)/((n+20)*sqrt21); C=(fsum(1/sigma)/n)*fsum(y)/((n+20)*sqrt21). Control matches the aggregate inverse-risk mass over exactly the same selected events and zero-prior count, not inverse mean sigma. At n1, exact equal computed risks or exact equal computed payoffs, C copies Q bit-for-bit. No near-equality threshold, clipping or extra smoothing is used. Both raw scores can be negative, unbounded finite, and Q can exceed or fall below C.'),
        dict(area='Causal source window and matched availability', passed=True,
             evidence=[SIX[2]+':source_prices', SIX[2]+':build_features', SIX[4]+':independent_features', SOURCE],
             finding='Exactly253 official-session prices from t-294 through t-42 must all be numeric finite positive, including prices unused by the selected events. Original49 candidates including ETFs are treated equally before eligibility. No selected anchor or any selected zero sigma makes both raw values unavailable; zero sigma does not remove the event. All selected risks and matured labels are evaluated before publishing common availability. A single selected event is allowed and a single available security has rank0. A genuinely zero payoff remains a valid zero when risk is positive.'),
        dict(area='Independent numerical checks and explicit failures', passed=True,
             evidence=[SIX[2]+':finite_sum', SIX[2]+':aggregate_payoffs', SIX[4]+':measure_oracle', SIX[4]+':verify_features'],
             finding='Production uses numpy sample std(ddof1); the oracle independently computes63-return two-pass variance using math.fsum and denominator62. Both use accurate finite sums and fail on nonfinite arithmetic from otherwise valid prices, including overflow after an earlier zero-risk event. Zero risk is not floored; arbitrarily small positive risk is allowed subject to finite arithmetic. Scalar audit/raw tolerance is rtol2e-12/atol1e-14, with sigma atol0. Counts, masks, identity branch labels and cross-sectional ranks reconstructed independently from raw values must match exactly, so float tolerance cannot silently admit a changed rank.'),
        dict(area='Ranks, fixed budget and first partial row', passed=True,
             evidence=[SIX[2]+':centered_rank', SIX[4]+':centered_ranks', SIX[4]+':verify_deltas', SIX[0]+':scale_and_delta'],
             finding='Higher values preferred, average tied zero-based rank/(m-1)-.5 over the common feature-available population. Singleton/all-tied populations rank0. Host accepts any finite raw magnitudes and reconstructs saved ranks exactly without197 nonnegative/Q<=C restrictions. Missing ranks/corrections0; direct .01 multiplication has maximum.005. Only the initial-row correction is suppressed; initial raw features and source evidence remain intact. No second tanh, clipping, eligibility rank or predictive model is introduced.'),
        dict(area='Descriptive count accounting', passed=True,
             evidence=[SIX[2]+':build_features', SIX[4]+':verify_features', SIX[0]+':estimate_counts'],
             finding='Four per-event lists separately preserve source cells, available cells, selected anchors and valid event payoffs. Selected counts include zero-risk cells; valid payoffs count only events in common feature-available cells. Five totals are exact:253*source cells; selected anchors;63*selected pre-event returns;selected matured payoffs;valid-event sum. Host checks0<=available<=source<=49,selected<=8*source,available<=valid<=8*available and valid<=selected for each event. Complete independent cell audits validate the underlying values, not totals alone. Repeated prefix/oracle work is excluded from main-pass counts; predictive fits remain0.'),
        dict(area='Original Local67 and single episode integration', passed=True,
             evidence=[SIX[0]+':decisions', SIX[0]+':prepare', *dependency_files[:5]],
             finding='Incumbent uses original config4 choices and must match all frozen orders. New arms start from complete config4 modified_pre_score, add only nonzero deltas, then call zero-blend original adapter exactly once. Original effects, policy, core and auxiliary support are supplied to the existing boundary selector without post-episode global reranking. All27 original arrays are byte/shape/dtype checked unchanged.'),
        dict(area='Future isolation and prefix validation', passed=True,
             evidence=[SIX[0]+':feature_prefix_panel', SIX[0]+':prepare', SIX[3], SIX[5]],
             finding='Numeric source reads are confined to each requested formation window and every selected label matures by its end. Host physically truncates adjusted close through each signal date for prefixes0/1/53/88/122, empty at0, and checks event audit/all five arrays exactly. All123 decision/rank/delta prefixes are checked. Synthetic future poison, unused symbols, truncated source and permutations cover independence. Next-session lookup verifies calendar monthend, not a future market value.'),
        dict(area='First88 selection and control identification', passed=True,
             evidence=[SIX[0]+':guard', SIX[0]+':select_first88', SIX[0]+':select', SIX[0]+':frozen_selection'],
             finding='Only first88 order prefixes through ends[87] enter selection; early53/recent35 are subdivisions. Candidate requires +.005 same-close CAGR over incumbent, next-open5bp noninferiority, Sharpe tolerance.05/MDD.03/eraCAGR.025, both-path noninferiority versus inverse-risk-mass control, and at least3 changed holding sets with both eras against each reference. Control cannot promote; later34 runs only after hash-bound FROZEN and cannot change selection.'),
        dict(area='Independent outcome accounting', passed=True,
             evidence=[SIX[4]+':audit_outcomes', *dependency_files[5:10]],
             finding='The pinned192→189→171 adapter propagates the three198 arm names to independent scalar accounting. It checks all9 paths and45 slices,122 monthly executions, fixed dates, fill prices, fee/cash/units/NAV/self-financing/equal-weight reconciliation, annual metrics and no terminal liquidation. Frozen training metrics are rechecked against independently verified first88/early53/recent35 exports.'),
        dict(area='Source, runtime, phase and closure governance', passed=True,
             evidence=[SIX[0]+':register', SIX[0]+':gate', SIX[0]+':prepared', SIX[0]+':close', TESTS, SOURCE],
             finding='Exact formal synthetic-test source/runtime receipt and independent six-file review are required before PLAN creation. Original64 raw map, current257-pin source review, predecessor197 closure, inherited sources and official calendar remain bound. Source review must explicitly deny actual anchor, volatility and payoff computation. All actual phases require explicit PLAN SHA/central registration, verify sources/prepared manifests, and refuse replacement. Registration and closure scan21444 protected files. Central commits retain backups and compare original bytes twice;195/1102 becomes196/1105 only on closure.'),
        dict(area='Synthetic evidence and reviewer scope', passed=True,
             evidence=[TESTS, *SIX[1::2]],
             finding='Exact current six-file hashes match recorded59 synthetic tests, including paired-versus-swapped inverse-risk weights, covariance identity, exact versus near equality, s64/s231 and21-day cooldown edges, zero risk with subsequent events, negative and zero payoffs, arithmetic overflow, source validity, singleton/ties, physical prefixes/future poison, corrupted complete audit/raw/rank/count evidence and first88 isolation. Independent tests perform full producer crosschecks on both fixture families. Reviewer inspected all six files and receipt; did not rerun tests or compute actual features/orders/returns.'),
    ]
    receipt = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        unresolved_blockers=[], code_sha256=code, input_sha256=pins, review_code_sha256=sha(__file__),
        independent_reviewer='observer_notification_audit', review_method='Complete six-source/test static review plus pinned metadata and dependency digest validation; no production imports.',
        context_sha256=CONTEXT_SHA, source_review_sha256=SOURCE_SHA,
        actual_features_computed=False, actual_negative_anchors_computed=False,
        actual_selected_anchors_computed=False, actual_pre_event_volatility_computed=False,
        actual_payoffs_computed=False, actual_signal_returns_computed=False,
        actual_ranks_computed=False, actual_orders_computed=False, actual_portfolio_returns_computed=False,
        production_files_modified=False, strategy_registered_by_reviewer=False, production_modules_imported=False,
        formal_tests_rerun_by_reviewer=False, synthetic_test_receipt_verified=True,
        synthetic_tests=59, synthetic_test_methods=methods, review_checks=checks,
        limitations=[
            'Source-only coverage is already recorded; this review does not compute or certify actual198 feature availability, identification, orders, returns or admission.',
            'Stored adjusted close uses retrospective vendor adjustment, alias and early-extension conventions; strict original-public vintage and uniform corporate-action consistency remain uncertified.',
            'Strict anchors start from stored binary64 prices and cannot reverse prior vendor rounding. Positive common scaling preserves simple returns and log differences mathematically but not every floating operation or within-window source splice.',
            'Very small positive risk can amplify raw values; no floor is added. Arithmetic nonfinite or independently reconstructed rank/identity mismatch will fail closed rather than be silently clipped or relaxed.',
            'Cooldown reduces direct payoff overlap but risk windows and monthly formation windows remain dependent. At most8 selected anchors and20 zero-prior events imply strong shrinkage and sparse evidence. Matched inverse-risk mass does not match beta, loss severity or every confounder, and covariance is descriptive rather than causal.',
            'Repeated research sample, estimatedTop32+theme17 and idealized same-close assumptions remain. Later34 is diagnostic, not pristine out-of-sample.',
            'The host enforces all21444 protected bytes at registration/closure. This receipt validates the bound manifest rather than repeating the complete archive scan.',
        ])
    for name, digest in pins.items(): require(sha(ROOT/name) == digest, 'Changed during review: '+name)
    with OUT.open('x') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True, path=str(OUT), sha256=sha(OUT),
        review_code_sha256=sha(__file__), code_files=len(code), input_pins=len(pins),
        synthetic_tests_verified=59, unresolved_blockers=[])))


if __name__ == '__main__':
    main()

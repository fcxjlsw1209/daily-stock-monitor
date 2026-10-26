"""Independent source/contract review receipt for round196.

Reads source text, JSON evidence and file digests only. No production imports,
market-array loading, feature/pair calculations, orders, replay or formal tests.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/majority196_CODE_REVIEW.json'
SIX = (
    'work/monthly_pairwise_majority196_host.py',
    'work/test_monthly_pairwise_majority196_host.py',
    'work/monthly_pairwise_majority196_kernel.py',
    'work/test_monthly_pairwise_majority196_kernel.py',
    'work/monthly_pairwise_majority196_independent.py',
    'work/test_monthly_pairwise_majority196_independent.py',
)
EXPECTED = (
    '57615dd66df1fba75dd53607e8452fc6bcaee62bee657811ab2f656205b38c64',
    '4544921f606aaa837466b63d898793195a52b2f21adbda6382f5dced19947867',
    'c72ca0d5cc7a7ec8a423f8048f92a3f013aa3a9b4e8392edb848d992743e2338',
    'd0a029b83dbc5371d98fcf3b08ae9fafdd45d594961d3983ced110b2fde51e5e',
    '10b017e3c4f288ba5e1d4da7e276e9b80bc76d5fe16858bdb27027e7766b8884',
    'c6ce4a535ea54067e885c0fe6b9c115252bcbb4e767d9841b0f5a1bb303a2707',
)
CONTEXT = 'work/monthly_pairwise_majority196_context.json'
CONTEXT_SHA = '9d95fc3b4fba608a2751aa0ecaeab0eacaee1d5b9bdfdad6f51024b7af628d3b'
SOURCE = 'work/majority196_SOURCE_REVIEW.json'
SOURCE_SHA = '5e1af9637afba453a181597a1de684dee841444825fe453ef13c2a99a84e4c32'
TESTS = 'work/monthly_pairwise_majority196_TESTS.json'
PRIOR = 'outputs/monthly_volume_cluster195'
PRIOR_HASHES = {
    'PLAN.json': '170af7dd7ac97eb2aaa61e239a1937dd574f159d80040d5ab5d0be9f388a9374',
    'COMPLETION.json': 'a5192918944fd2b6d92ab4d88805b2223de188e46e953de6cd2dd1e8fe0e502c',
    'results/COMPLETION.json': 'fbb68f08a4e65616f73276b6d5f042e66ef443d5873a6bc0d871f3b0efe6a5dc',
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
    require(context['arms'] == ['incumbent_local67', 'tournament_majority', 'tournament_balance_control'], 'Arms differ')
    formation = context['formation']
    require((formation['earliest_offset'], formation['latest_offset'], formation['price_count'],
             formation['blocks'], formation['block_sessions']) == (294, 42, 253, 12, 21), 'Formation differs')
    require(context['integration']['dose'] == .01 and context['integration']['max_absolute_delta'] == .005,
            'Fixed correction differs')
    require(source['passed'] is True and source['context_sha256'] == CONTEXT_SHA, 'Source review differs')
    for key in ('actual_features_computed', 'actual_orders_computed', 'actual_returns_computed',
                'actual_block_returns_computed', 'actual_pair_comparisons_computed', 'actual_ranks_computed'):
        require(source[key] is False, 'Source-only review computed ' + key)
    require(len(source['raw_source_hashes']) == 64, 'Original raw64 map incomplete')
    for name, digest in source['source_sha256'].items(): bind(name, digest)
    for name, digest in source['raw_source_hashes'].items(): bind(name, digest)
    require(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests'] == 57,
            'Expected completed57 synthetic tests')
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
    require('monthly_pairwise_majority196_kernel' not in text[SIX[4]], 'Independent oracle imports producer')
    methods = {name: sum(isinstance(n, ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(trees[name]))
               for name in SIX[1::2]}
    require(sum(methods.values()) == 57, 'Formal test count/source mismatch')

    checks = [
        dict(area='Opponent-specific majority and matched pooled control', passed=True,
             evidence=[SIX[2]+':tournament', SIX[4]+':cross_sign', SIX[4]+':tournament_oracle'],
             finding='Production Fraction.from_float cross-products compare each pair on each of12 blocks; the oracle independently multiplies integer-ratio numerators and denominators over ordered pairs. Both reconstruct identical block signs, antisymmetric margins, diagonal zero, Q=sum sign(M)/(n-1), F=sum M/[12(n-1)]. No rounded gross-ratio division or floating-product overflow. Each unordered pair is counted once in descriptive accounting.'),
        dict(area='Causal source window and common comparison population', passed=True,
             evidence=[SIX[2]+':source_prices', SIX[2]+':build_features', SIX[4]+':independent_features', SOURCE],
             finding='Exactly253 official-session prices from t-294 through t-42 are required, including non-endpoint observations;13 endpoints form12 disjoint21-session blocks. Original event identities include ETFs without a new eligibility filter. Invalid entire-source members are removed for both arms, never pair-specific deletions. If source n<2, both feature masks are false and raw NaN while source availability remains separately recorded.'),
        dict(area='Ranks, fixed budget and first partial row', passed=True,
             evidence=[SIX[2]+':centered_rank', SIX[4]+':centered_ranks', SIX[4]+':verify_deltas', SIX[0]+':scale_and_delta'],
             finding='Higher values preferred, average tied zero-based rank/(n-1)-.5 over the common source-valid event population. Exact ties use stored values; all-tied populations rank0. Missing ranks and corrections0. Host independently reconstructs saved ranks, then direct .01 multiplication, maximum .005, with only initial-row correction suppressed. Initial raw features and source evidence are retained.'),
        dict(area='Original Local67 and single episode integration', passed=True,
             evidence=[SIX[0]+':decisions', SIX[0]+':prepare', *dependency_files[:5]],
             finding='Incumbent uses original config4 choices and must match all frozen orders. New arms start from complete config4 modified_pre_score, add only nonzero deltas, then call zero-blend original adapter exactly once. Original effects, policy, core and auxiliary support are supplied to the existing boundary selector without post-episode global reranking. All27 original arrays are byte/shape/dtype checked unchanged.'),
        dict(area='Future isolation and prefix validation', passed=True,
             evidence=[SIX[0]+':feature_prefix_panel', SIX[0]+':prepare', SIX[3], SIX[5]],
             finding='Numeric source reads are confined to each requested formation window. Host physically truncates adjusted-close source through each signal date for prefixes0/1/53/88/122, with empty time axis at0, and checks full event audit and all five arrays exactly. All123 decision/rank/delta prefixes are checked. Synthetic future poison, unrelated future symbols, truncated source and column permutations cover causal independence. Next-session lookup verifies calendar monthend, not a future market value.'),
        dict(area='First88 selection and control identification', passed=True,
             evidence=[SIX[0]+':guard', SIX[0]+':select_first88', SIX[0]+':select', SIX[0]+':frozen_selection'],
             finding='Only first88 order prefixes through ends[87] enter selection; early53/recent35 are subdivisions. Candidate requires +.005 same-close CAGR over incumbent, next-open5bp noninferiority, Sharpe tolerance.05/MDD.03/eraCAGR.025, both-path noninferiority versus balance control, and at least3 changed holding sets with both eras against each reference. Control cannot promote; later34 runs only after hash-bound FROZEN and cannot change selection.'),
        dict(area='Independent outcome accounting', passed=True,
             evidence=[SIX[4]+':audit_outcomes', *dependency_files[5:10]],
             finding='The pinned192→189→171 adapter propagates the three196 arm names to independent scalar accounting. It checks all9 paths and45 slices,122 monthly executions, fixed dates, fill prices, fee/cash/units/NAV/self-financing/equal-weight reconciliation, annual metrics and no terminal liquidation. Frozen training metrics are rechecked against independently verified first88/early53/recent35 exports.'),
        dict(area='Source, runtime, phase and closure governance', passed=True,
             evidence=[SIX[0]+':register', SIX[0]+':gate', SIX[0]+':prepared', SIX[0]+':close', TESTS, SOURCE],
             finding='Exact57-test source/runtime receipt and independent six-file review are required before PLAN creation. Original64 raw map, current238-pin source review, predecessor195 closure, inherited source pins and official calendar remain bound. All actual phases require explicit PLAN SHA and central registration, verify sources and prepared manifests, and refuse replacement. Registration and closure scan21444 protected files. Central commits retain backups and compare original bytes twice;193/1096 becomes194/1099 only on closure.'),
        dict(area='Synthetic evidence and reviewer scope', passed=True,
             evidence=[TESTS, *SIX[1::2]],
             finding='Exact current six-file hashes match57 completed synthetic tests, including equal pooled-balance/different-majority counterexamples, cycles, time permutation, overflow/underflow/near ties, exact ties, all253 support, singleton source-vs-feature masks, futures, corrupted margin/raw/rank/counts, fixed integration and selection isolation. Reviewer inspected code and receipt; did not rerun tests or compute actual features/orders/returns.'),
    ]
    receipt = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        unresolved_blockers=[], code_sha256=code, input_sha256=pins, review_code_sha256=sha(__file__),
        independent_reviewer='observer_notification_audit', review_method='Complete six-source/test static review plus pinned metadata and dependency digest validation; no production imports.',
        context_sha256=CONTEXT_SHA, source_review_sha256=SOURCE_SHA,
        actual_features_computed=False, actual_block_returns_computed=False, actual_pair_comparisons_computed=False,
        actual_ranks_computed=False, actual_orders_computed=False, actual_portfolio_returns_computed=False,
        production_files_modified=False, strategy_registered_by_reviewer=False, production_modules_imported=False,
        formal_tests_rerun_by_reviewer=False, synthetic_test_receipt_verified=True,
        synthetic_tests=57, synthetic_test_methods=methods, review_checks=checks,
        limitations=[
            'Source-only coverage is already recorded; this review does not compute or certify actual196 strategy availability, identification, orders, returns or admission.',
            'Stored adjusted close uses retrospective vendor adjustment, alias and early-extension conventions; strict original-public vintage and uniform corporate-action consistency remain uncertified.',
            'Exact arithmetic starts from stored binary64 endpoints and cannot reverse prior vendor rounding. Common whole-window positive scaling cancels pair ratios, but that fact does not certify within-window source splices.',
            'Both aggregations are invariant to block ordering; majority cycles and dependent shared opponents preclude a chronological-persistence or independent-sample interpretation.',
            'Repeated research sample, estimatedTop32+theme17 and idealized same-close assumptions remain. Later34 is diagnostic, not pristine out-of-sample.',
            'The host enforces all21444 protected bytes at registration/closure. This receipt validates the bound manifest rather than repeating the complete archive scan.',
        ])
    for name, digest in pins.items(): require(sha(ROOT/name) == digest, 'Changed during review: '+name)
    with OUT.open('x') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True, path=str(OUT), sha256=sha(OUT),
        review_code_sha256=sha(__file__), code_files=len(code), input_pins=len(pins),
        synthetic_tests_verified=57, unresolved_blockers=[])))


if __name__ == '__main__':
    main()

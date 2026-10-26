"""Independent round201 static code review and exact evidence/hash gate.

No producer import, market-array decoding, factor, order or return calculation.
The write-once receipt records review of the final six sources, their formal
synthetic test evidence, and the frozen mathematical/integration dependencies.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/median201_CODE_REVIEW.json'
SIX = (
    'work/monthly_median_risk201_host.py', 'work/test_monthly_median_risk201_host.py',
    'work/monthly_median_risk201_kernel.py', 'work/test_monthly_median_risk201_kernel.py',
    'work/monthly_median_risk201_independent.py', 'work/test_monthly_median_risk201_independent.py',
)
EXPECTED = (
    '1af7b4bafd1cfe6316dc24b5892ba78a9237b26a8375df93dcda9a310f75349c',
    '02e47551a83c6d0e79b4ece9dc1f602ce4ada4ea7c470807722098036b98bff1',
    'a539ad18d639fb6b40cd9f9e4869093627b0533a1d7dab5c4cf98fd04aacc087',
    'b6a91856fe3e82e8f7f462a2b93854e18e256a52f789a5b3433c77cc181ff1ff',
    '39da098392548e6e5ab76de10cbfdc96d8dc831fb2ff0dabdb8437209b5e7ab3',
    '464d4da9a90e81b96783b2101d7075ae57582231889f591208d735ebc4d42c55',
)
CONTEXT = 'work/monthly_median_risk201_context.json'
CONTEXT_SHA = '4e937ed4474675b64069b410ef790697474ea2bfe03585e0f8f97d30afec9701'
SOURCE = 'work/median201_SOURCE_REVIEW.json'
SOURCE_SHA = '4ca67a75465bb05f8b255272a185c436047c89fefe2193d939ec5ee9ae1e810d'
COVERAGE = 'work/after200_median_risk_coverage.json'
COVERAGE_SHA = 'f888757b1cd4a97636bea330632215d0bcc6a45eaa63cfb9c9a14eb62b268e2e'
TESTS = 'work/monthly_median_risk201_TESTS.json'
TESTS_SHA = 'a485c45ad23337412063184c144b8602f0a463b368196a3047ecc27d965e2da2'
PRIOR = 'outputs/monthly_cmf_state200/PLAN.json'
PRIOR_SHA = 'e0183ab68ac726d43e5886730179542e995e78241d97a1543c4a06dbfcf7bae0'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    require(not OUT.exists(), 'Code review is write-once')
    require(not (ROOT/'outputs/monthly_median_risk201').exists(), 'Review must precede registration')
    pins = {}
    def bind(name, expected=None):
        path = Path(name)
        path = (path if path.is_absolute() else ROOT/path).resolve()
        require(path.is_relative_to(ROOT), 'Nonlocal review input')
        key, digest = str(path.relative_to(ROOT)), sha(path)
        require(expected is None or digest == expected, 'Changed evidence: '+key)
        require(key not in pins or pins[key] == digest, 'Conflicting evidence: '+key)
        pins[key] = digest
        return digest
    code = {p: bind(p,h) for p,h in zip(SIX,EXPECTED)}
    for p,h in ((CONTEXT,CONTEXT_SHA),(SOURCE,SOURCE_SHA),(COVERAGE,COVERAGE_SHA),
                (TESTS,TESTS_SHA),(PRIOR,PRIOR_SHA)):
        bind(p,h)
    bind(__file__)
    context, source, tests, prior = read(CONTEXT), read(SOURCE), read(TESTS), read(PRIOR)
    require(context['round'] == 201 and context['arms'] ==
            ['incumbent_local67','median_mad_core','mean_mad_control'] and
            context['candidate'] == 'median_mad_core' and
            context['control_nonpromotable'] == 'mean_mad_control', 'Prospective arms differ')
    for key in ('pre_registration_actual_new_features','pre_registration_actual_orders','pre_registration_actual_returns'):
        require(type(context[key]) is int and context[key] == 0, 'Prospective scope differs: '+key)
    require(context['formation']['price_count'] == 295 and context['formation']['return_count'] == 63 and
            context['integration']['blend'] == .5 and context['integration']['max_absolute_delta'] == .26,
            'Window or half-core contract differs')
    require(source['passed'] is True and source['unresolved_blockers'] == [] and source['source_only'] is True and
            source['context_sha256'] == CONTEXT_SHA and source['coverage_sha256'] == COVERAGE_SHA and
            source['strict_pit'] is False, 'Source-only review differs')
    for key in ('actual_features_computed','actual_signal_returns_computed','actual_dispersion_computed',
                'actual_ranks_computed','actual_deltas_computed','actual_orders_computed',
                'actual_returns_computed','actual_portfolio_returns_computed','production_files_modified',
                'strategy_registered_by_reviewer'):
        require(source[key] is False, 'Source-only boundary crossed: '+key)
    require(len(source['raw_source_hashes']) == 64 and len(source['source_sha256']) == 314,
            'Source closure count differs')
    cover = source['source_window_coverage']
    require(cover['price_count'] == 295 and cover['complete_positive_price_cells'] == 5973 and
            cover['incomplete_price_cells'] == 5 and cover['field_counts']['observations'] == 1763510,
            'Original price coverage differs')
    require(source['raw_source_parity']['exact_source_map_equal'] is True, 'Source map parity absent')
    for mapping in (source['source_sha256'],source['raw_source_hashes'],context['evidence_sha256']):
        for p,h in mapping.items():
            bind(p,h)
    require(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests'] == 60,
            'Formal60 synthetic tests required')
    require(tests['runtime'] == dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'), 'Pinned runtime differs')
    for p,h in tests['code_sha256'].items():
        bind(p,h)
    for p in SIX:
        require(tests['code_sha256'][str(ROOT/p)] == code[p], 'Final source differs from tested bytes')
    dependencies = (
        'outputs/monthly_core_timing70_research/core_adapter.py',
        'outputs/monthly_core_timing70_research/test_core_adapter.py',
        'outputs/monthly_path70_research/incumbent_adapter.py',
        'outputs/monthly_local_linear70/study.py','outputs/monthly_local_linear70/PLAN.json',
        'outputs/monthly_local_linear70/DATA.npz','outputs/monthly_local_linear70/META.json',
        'outputs/monthly_local_linear70/INCUMBENT_POLICY.json','outputs/monthly_local_linear70/local_linear_ORDERS.json',
        'outputs/monthly_daily_episodes/episodes.py','outputs/monthly_model_scope70/study.py',
        'outputs/monthly_target_definition70/study.py','outputs/monthly_state_rank_research/state_rank.py',
        'outputs/monthly_quality_research/explore.py','outputs/monthly_stability_monitor/strategy.py',
        'outputs/monthly_regime_monitor/strategy.py','outputs/monthly_paired_target143/CALENDAR.json',
        'work/monthly_nested_trend199_kernel.py','work/ptr169_rebuild_evaluate.py',
        'work/ptr169_results_independent.py','work/price_distribution171_independent.py',
        'work/monthly_sector_transition189_independent.py','work/monthly_sector_recurrence192_independent.py',
        'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json',
        'run_local.sh','local_run.py','local.sandbox.sb',
    )
    for p in dependencies:
        expected = prior['input_sha256'].get(str(ROOT/p)) or context['evidence_sha256'].get(p)
        require(expected is not None, 'Missing frozen critical dependency: '+p)
        bind(p,expected)
    require(len(read(dependencies[23])) == 21444, 'Protected manifest size differs')
    require(read(dependencies[4])['raw_source_hashes'] == source['raw_source_hashes'], 'Original raw64 differs')
    texts = {p:(ROOT/p).read_text() for p in SIX}
    trees = {p:ast.parse(s) for p,s in texts.items()}
    host = texts[SIX[0]]
    funcs = {node.name:ast.get_source_segment(host,node) for node in trees[SIX[0]].body
             if isinstance(node,ast.FunctionDef)}
    require(funcs['prepare'].index('gate(expected)') < funcs['prepare'].index('study.load()'), 'Missing compute gate')
    require(funcs['register'].index('code_review_binding()') < funcs['register'].index('OUT.mkdir()'), 'Missing review gate')
    require('target=validated[key].copy();target[calm]=np.nan' in funcs['scale_and_delta'] and
            'if n:target[0]=np.nan' in funcs['scale_and_delta'] and
            'BLEND*(cw[mask]*target[mask]-c[mask])' in funcs['scale_and_delta'], 'Half-core state mask differs')
    require("adapter.score_and_choose('incumbent66',BLEND,full_target" in funcs['decisions'] and
            'pre_score=pre,effects=arrays[\'effects\']' in funcs['decisions'] and
            'orders[arm]=study.rp.make_orders(chosen,score,meta)' in funcs['decisions'], 'Frozen core adapter wiring differs')
    require('for n in range(123)' in funcs['prepare'] and 'len(arrays) == 27' in funcs['prepare'] and
            "read(LOCAL / 'local_linear_ORDERS.json')" in funcs['prepare'], 'Original/prefix checks absent')
    require('orders[arm][:TRAIN]' in funcs['select'] and "meta['ends'][87]" in funcs['select'], 'First88 cutoff differs')
    require(funcs['diagnose'].index('frozen_selection(expected, orders)') < funcs['diagnose'].index('ev.run_path('),
            'Later outcomes precede frozen selection')
    require('monthly_median_risk201_kernel' not in texts[SIX[4]], 'Oracle imports producer')
    methods = {p:sum(isinstance(node,ast.FunctionDef) and node.name.startswith('test_')
                    for node in ast.walk(trees[p])) for p in SIX[1::2]}
    require(list(methods.values()) == [31,13,16], 'Final synthetic test method counts differ')
    checks = [
        dict(area='Source and official session clocks', passed=True, evidence=[CONTEXT,SOURCE,SIX[2],SIX[4]],
             finding='Both arms use all295 original positive finite nonboolean numeric adjusted prices t-294..t. Original raw64 mapping, official calendar and all49 identities are preserved. Numerator P[252]/P[0]-1 ends t-42; current risk uses P[232:295]/P[231:294]-1, exactly63 returns and64 prices t-63..t. Risk does not skip42. Missing source stays unavailable, no fill or compressed calendar. Source review computed coverage only, not MAD or feature returns.'),
        dict(area='Shared median center and exact descriptive arithmetic', passed=True, evidence=[SIX[2]+':measure',SIX[4]+':measure_oracle',SIX[3],SIX[5]],
             finding='Both ratios share sorted63returns[31]. Deviations are abs(r-m); candidate denominator is sorted63deviations[31], control math.fsum(deviations)/63. Both require strictly positive finite denominators; median0 with positive mean makes both unavailable. Zero numerator is valid; negative numerator remains signed. No epsilon floor, Gaussian constant, trimming or fitted model. Overflow from valid finite positive inputs raises, never silently reduces support. Independent handworked dyadic fixture verifies median0, Dmed.5, Dmean48.5/63 and exact ratios.'),
        dict(area='Original percentile and identical rank population', passed=True, evidence=[SIX[2]+':percentile_rank/build_features',SIX[4]+':original_pct_ranks/feature_arrays',SIX[0]+':scale_and_delta'],
             finding='Each ratio ranks separately on the same common available original49 pool before trading eligibility, including ETFs. Scalar independent percentile is (less+(equal+1)/2)/m-.5; denominator m, singleton+.5 and all-tie+.5/m retained. Missing remainsNaN, finite0 is valid, no second ranking. Host reconstructs exact original ranks before integration.'),
        dict(area='Only noncalm half-core replacement', passed=True, evidence=[SIX[0]+':scale_and_delta',SIX[4]+':independent_deltas',SIX[1]+':ScoreTests'],
             finding='Original cw is exactly .50 calm or .52 noncalm. Raw ratios/ranks also exist for initial/calm rows, but integration sets their targetsNaN. Every finite noncalm target gets .5*(cw*T-C), bounded.26; missing gives zero delta. Initial row0 exactneutral. Prefix0 and short prefixes read no future numeric values including object calm/core/weights/feature suffixes; three legal cw shapes agree.'),
        dict(area='Frozen core adapter, original support and one episode', passed=True, evidence=[dependencies[0],dependencies[2],dependencies[3],dependencies[9],SIX[0]+':decisions',SIX[1]+':RealAdapterTests'],
             finding='Uses original Local67config4 complete pre-score and pinned core_adapter. Same delta enters copied baseline/core/pre only at nonzero cells. Original baseline/core/cw are restored in episode callback, preserving support exactly. Original effects and policy reach one original boundary episode per candidate arm; original array metadata stays unchanged. Returned chosen indices directly form orders without global postrank. Synthetic callback deliberately chooses a different third name than globaltop3, detecting accidental reranking.'),
        dict(area='Independent validation and honest counts', passed=True, evidence=[SIX[0]+':prepare/estimate_counts/verify_prepared_evidence',SIX[4]+':verify_features/verify_deltas',*SIX[1::2]],
             finding='Independent scalar source lookup/medians/fsum/ratios/pair-count ranks and complete audit require exact equality, without numeric tolerance. Preparation must check5978 cells,123order prefixes and physical sourceprefixes0/1/53/88/122; original27 dtype/shape/bytes and incumbent122orders must remain exact. Two descriptive dispersion estimates and63signal returns count for every complete source cell including zero dispersion; ratios/ranks count only common available cells. New predictive fits0; descriptive signal returns are distinguished from portfolio returns.'),
        dict(area='First88 selection, frozen diagnosis and accounting', passed=True, evidence=[SIX[0]+':select_first88/select/frozen_selection/diagnose/close',SIX[4]+':audit_outcomes',*dependencies[18:23]],
             finding='Only median candidate promotable. First88 close CAGR gain.005 versus incumbent, costedopen noninferiority, both paths noninferior to matchedmean, and exact Sharpe/MDD/early53/recent35 guards. At least3changed holding sets with both eras versus each reference. Later34 cannot select. Pinned independent192->189->171->169 accounting checks9paths45slices and18trainingmetric sets, original EW3/EW2/cash fixedunits and no terminal liquidation.'),
        dict(area='Prospective governance and final source evidence', passed=True, evidence=[CONTEXT,SOURCE,TESTS,SIX[0]+':register/gate/close'],
             finding='Final60synthetic receipt binds exact31host+13kernel+16oracle sources and pinned runtime. Registration requires this exact six-file independent review, source receipt, predecessor200 closure, unchanged21444manifest bytes and empty queue. All actual phases require explicit PLAN hash. Write-once phase evidence and central CAS are retained; closure advances198/1111 to199/1114 only after independent outcomes. Review imports no production modules and computes no actual scores/orders/returns.'),
    ]
    result = dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        unresolved_blockers=[],code_sha256=code,input_sha256=pins,review_code_sha256=sha(__file__),
        independent_reviewer='nested199_code_audit',
        review_method='Static mathematical, source-clock, integration, lifecycle and test review of final six files; exact metadata and dependency hash verification only.',
        context_sha256=CONTEXT_SHA,source_review_sha256=SOURCE_SHA,coverage_sha256=COVERAGE_SHA,
        formal_test_receipt_sha256=TESTS_SHA,synthetic_tests=60,synthetic_test_methods=methods,
        synthetic_test_receipt_verified=True,formal_tests_rerun_by_reviewer=False,
        actual_features_computed=False,actual_signal_returns_computed=False,actual_dispersion_computed=False,
        actual_ranks_computed=False,actual_deltas_computed=False,actual_orders_computed=False,
        actual_portfolio_returns_computed=False,production_files_modified=False,
        strategy_registered_by_reviewer=False,production_modules_imported=False,
        resolved_findings=[dict(issue='Oracle scale-invariance test compared changing raw-price audit fields as mathematical invariants.',
            resolution='Author excludes the four price fields from invariance comparison and separately asserts their exact scale change; final16oracle tests and formal60suite pass.', production_change_required=False)],
        review_checks=checks,limitations=[
            'This pre-registration code review is not actual feature preparation, performance evidence or strategy adoption.',
            'Complete source prices do not prove positive MAD denominators; common exactzero fallback is mandatory and its actual count remains unknown until registered preparation.',
            'Symmetric robust dispersion cannot identify benign upside versus harmful downside; the mean control isolates aggregation around the same median, not every difference from incumbent standard deviation.',
            'Source strictPIT remains false: retrospective vendor/action conventions and proxyTop32+theme17 membership persist; later34 is reused diagnostic history.',
            'Historical robust-risk families already exist; missing earlier definitions prevent an exhaustive novelty claim.',
            'Reviewer verifies protected21444manifest binding; registration and closure must check the actual protected bytes.',
        ])
    for p,h in pins.items():
        require(sha(ROOT/p) == h, 'Evidence changed during review: '+p)
    with OUT.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),review_code_sha256=sha(__file__),
                         code_files=6,input_pins=len(pins),synthetic_tests_verified=60,unresolved_blockers=[])))


if __name__ == '__main__':
    main()

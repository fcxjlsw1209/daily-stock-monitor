"""Write-once metadata coverage audit. No market data or production imports."""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'work/after198_nested_trend_novelty.json'
PINS = {
 'outputs/monthly_path70_research/features.py': '90659640abb4c981ab91f562139bf1fce09308844125c79203b0940e8e8275cc',
 'outputs/monthly_path70_research/study.py': '5096eb2bebd43878f3322c7807e1a8813d7739cbbe29b632517587f78d091af3',
 'outputs/monthly_core_timing70_research/features.py': 'b37c06ab73bd1f82f823cab7a3453307a2894e46e7637f394067cfd79bcbb518',
 'outputs/monthly_core_timing70_research/report.py': '7bfec759ff89898ff4f2484e305bd1338147bc91afd333b2575d448b0e8c56e4',
 'outputs/monthly_core_expansion/study.py': '82b535d2afb0e1f615ae2f0ce00a84987fbaec1150921ef779fdd9d0aca199f9',
 'outputs/monthly_stability_research/stability.py': '2d22b1bcef8925c8f98d6778142968f2ee3e2f0847c46caf2aea8c6de2b72ad4',
 'outputs/monthly_sector_global70/PLAN.json': 'f96234274a075fa84533e04b23d3612dca230281df513d440322c516ed8eb06a',
 'outputs/monthly_trend_evidence_research/PLAN.json': '070d57dceb4a90e96daeefde04865503eaccb6666d5d297c99eb2134e7ef4c66',
 'work/after191_backbone_coverage.json': 'a02c137a54f4c9962508358f2e2896efdb2f34e4cadbeb3e8c9ef44fd6c2856e',
 'work/rank161_proposal.json': '1b08cf674b6b29d87d63c2cf060ab27f2f0660078ca7cf19b29e63e249f54b17',
 'work/monthly_sector_recurrence192_context.json': 'd31f34fee66694e99e3e8924d2c1eb8649f3da714f127af4ef959144c140e03c',
}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def ranks(values):
    n = len(values)
    return [(sum(v < x for v in values) + F(sum(v == x for v in values) + 1, 2)) / n - F(1, 2)
            for x in values]

def synthetic_examples():
    # Exact abstract log increments, deliberately invented; no actual prices.
    quarter = {'A': [F(3,10), F(-1,10), F(1,5), F(1,5)],
               'B': [F(1,20)] * 4}
    daily = {s: [x / 63 for x in q for _ in range(63)] for s, q in quarter.items()}
    nested = {s: [sum(z[-h:]) for h in (126,189,252)] for s,z in daily.items()}
    segments = {s: [sum(z[42+k*70:42+(k+1)*70]) for k in range(3)] for s,z in daily.items()}
    assert nested['A'] == [F(2,5), F(3,10), F(3,5)]
    assert nested['B'] == [F(1,10), F(3,20), F(1,5)]
    assert min(segments['A']) == F(1,45)
    assert min(segments['B']) == F(1,18)
    assert all(a > b for a,b in zip(nested['A'], nested['B']))
    assert min(segments['A']) < min(segments['B'])
    horizon_ranks = [ranks([nested['A'][i], nested['B'][i]]) for i in range(3)]
    assert horizon_ranks == [[F(1,2), F(0)]] * 3
    segment_rank = ranks([min(segments['A']), min(segments['B'])])
    assert segment_rank == [F(0), F(1,2)]
    # Both transformations use exactly the same invented four-member pool.
    cumulative = [[F(4,10),F(3,10),F(2,10),F(1,10)],
                  [F(4,10),F(3,10),F(2,10),F(1,10)],
                  [F(2,10),F(3,10),F(4,10),F(1,10)]]
    cross = [ranks(x) for x in cumulative]
    minima = [min(x) for x in zip(*cross)]
    means = [sum(x)/3 for x in zip(*cross)]
    assert minima[0] < minima[1] and means[0] > means[1]
    assert ranks([F(1), F(1), F(3)]) == [F(0), F(0), F(1,2)]
    return {
      'data_kind': 'invented exact-rational log increments and endpoint gains only; not market observations',
      'passed': True,
      'counterexample_to_segment_min_equivalence': {
        'quarter_log_increments_63_sessions_each': quarter,
        'nested_cumulative_log_gains_126_189_252': nested,
        'legacy_last210_three70_segment_log_gains': segments,
        'nested_rank_per_horizon_for_A_B': horizon_ranks,
        'legacy_min_segment_rank_for_A_B': segment_rank,
        'conclusion': 'A outranks B at every nested horizon, while legacy weakest-segment ranks B above A. The operators are not equivalent even on the same invented 253-price axis.'},
      'candidate_control_identification': {
        'synthetic_names': ['A','B','C','D'], 'cumulative_log_gains_by_horizon': cumulative,
        'same_pool_horizon_ranks': cross, 'candidate_min_rank': minima,
        'control_mean_rank': means,
        'conclusion': 'Candidate ranks B above A, control ranks A above B; the matched aggregates can differ before any strategy integration.'},
      'tie_check': 'Average percentile ranks minus 0.5: [1,1,3] -> [0,0,1/2]. No extra centering of all-tie rows.'}


def main():
    if OUTPUT.exists():
        raise FileExistsError('Write-once audit already exists')
    for relative, expected in PINS.items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Evidence checksum mismatch: ' + relative)
    absent = [
      'outputs/monthly_horizon70_research/study.py',
      'outputs/monthly_horizon70_research/PLAN.json',
      'outputs/monthly_rank_concordance70/PLAN.json',
      'work/monthly_rank_concordance70_result_review.json',
      'work/monthly_leadership_core70_result_review.json',
    ]
    absence = {p: not (ROOT/p).exists() for p in absent}
    if not all(absence.values()):
        raise ValueError('A previously unavailable item is now present; review before updating coverage')
    result = {
      'schema_version': 1,
      'created_utc': datetime.now(timezone.utc).isoformat(),
      'status': 'BOUNDED_COVERAGE_AUDIT_NOT_STRATEGY_REGISTRATION',
      'scope': 'Local source, contracts and prior metadata only; no 198 outputs, live observers, financial facts or actual numerical strategy data read.',
      'conclusion': {
        'new_family': False,
        'exact_algorithm_not_found_in_inspected_readable_sources': True,
        'historical_exhaustiveness_certified': False,
        'classification': 'Within-family nested-horizon consistency contrast; not a claim of globally novel information or an untried entire mechanism.',
        'guidance_fit': 'Long-horizon price momentum, all endpoints end at t-42, existing monthly EW3 and fixed intramonth holdings remain intended. Integration and original-state parity must be declared separately before any data evaluation.',
        'why_not_generic_alternatives': 'Weakest-segment trends, relative recurrence, state-dependent core mixtures, core replacements and risk-aware third-seat corrections have readable precedents.'},
      'prospective_feature_definition_only': {
        'lengths_in_official_return_sessions': [126,189,252],
        'calendar_month_description': 'Approximate 6/9/12 trading months, not calendar-month anchors.',
        'common_end': 'u=t-42',
        'price_offsets_inclusive': [[-168,-42],[-231,-42],[-294,-42]],
        'cumulative_gain': 'G_h(i)=P_i[u]/P_i[u-h]-1. For positive finite endpoints, log(P_i[u])-log(P_i[u-h]) gives the same within-horizon rank. Floating-point tie/equality convention must be fixed before registration.',
        'ranking_pool': 'One identical available same-date original candidate pool for all three horizons and both aggregates; do not use three separately filtered pools. This is not a new trading eligibility filter.',
        'horizon_rank': 'r_h(i)=average ascending rank of G_h(i) within the common pool / common_pool_size - 0.5.',
        'candidate': 'T_min(i)=min(r_126(i),r_189(i),r_252(i)).',
        'matched_control': 'T_mean(i)=(r_126(i)+r_189(i)+r_252(i))/3.',
        'hypothesis': 'Requiring strong relative cumulative momentum across nested long horizons may distinguish leaders whose single long-horizon score is concentrated in a particular part of the path.',
        'fixed_no_search': 'One fixed set 126/189/252 and one min-versus-mean contrast; no horizon, sign, threshold, dose or market-state grid.',
        'not_yet_defined': ['Original Local67 state-branch score binding and numerical core subtraction', 'Replacement fraction or auxiliary dose', 'Whether an additional rank transform follows min/mean', 'Precise source completeness, empty/singleton/common-pool and missing-row fallback contract', 'Initial partial-month treatment and final selection contract'],
        'registration_gate': 'Read and verify actual frozen Local67 calm/noncalm formulas and support invariance first. This file does not authorize actual new feature, score, order or return computation.'},
      'coverage_evidence': [
        {'path':'outputs/monthly_path70_research/features.py','lines':[3,6,43,61],
         'covered':'Three disjoint 70-return segments in 210 returns ending t-42; minimum segment log gain, mean minus population std, downside and centered-MAD variants.',
         'difference':'Min is applied to raw disjoint segment gains before cross-sectional rank. Proposed min is applied after ranking cumulative nested horizons sharing one endpoint. Longer nested windows contain the shorter windows; they are not independent segments.'},
        {'path':'outputs/monthly_path70_research/study.py','lines':[55,62],
         'covered':'26 configurations, original65/incumbent66, hand support replacement 2.5/5/10%, original50/52% core retained, pre-episode placement.'},
        {'path':'outputs/monthly_stability_research/stability.py','lines':[31,38,48,71,99,105],
         'covered':'Raw/risk-adjusted original252/294 shift endpoints, current63/126 or aligned252 volatility, 50/65/80% core menus, a 50:50 two-window v63 rank blend, path and regime auxiliary variants.',
         'difference':'The explicit two-window arithmetic blend does not implement min of three 126/189/252 cumulative horizon ranks.'},
        {'path':'outputs/monthly_core_timing70_research/features.py','lines':[97,129,152,174],
         'covered':'Calendar12/14-month anchors and complete5/21-day means of original raw/risk factors, current state chooses raw family before smoothing.',
         'difference':'Smoothing shifts the observation date for a fixed factor family; the proposal changes lookback length at one fixed endpoint.'},
        {'path':'outputs/monthly_core_timing70_research/report.py','lines':[1,21],
         'covered':'Reporter documents half/all core rank replacement at original50/52% weight; this makes half-core placement itself a known design, not novelty evidence.'},
        {'path':'outputs/monthly_core_expansion/study.py','lines':[40,49],
         'covered':'Half/full core replacement using ordinary/HAC/residual Sharpe or historical rank residence, plus paired models; global three-slot competition.'},
        {'path':'work/rank161_proposal.json','lines':[5,13],
         'covered':'Six calendar-observation rank history, mean-minus-dispersion and directed slope; same-security usable six-observation support.',
         'difference':'Historical rank observations use different signal dates and historical pools; proposed horizons share current pool and endpoint.'},
        {'path':'outputs/monthly_sector_global70/PLAN.json','lines':[7,23],
         'covered':'Six42-day own-sector relative versus absolute top-quartile occupancy blocks; fixed10% original support replacement and global all-three selection.'},
        {'path':'work/monthly_sector_recurrence192_context.json','lines':None,
         'covered':'Twelve nonoverlapping21-day relative versus absolute signed recurrence counts; known persistence family, not the proposed cumulative multiscale rank conjunction.'},
        {'path':'outputs/monthly_trend_evidence_research/PLAN.json','lines':[7,12],
         'covered':'Third-seat Ridge challenger advantage minus annual-block jackknife sensitivity; bounded forecast magnitude must exceed score gap; top2 locked.'},
        {'path':'work/after191_backbone_coverage.json','lines':None,
         'covered':'Existing bounded audit documents broad state-conditioned long raw/risk/sector architecture coverage and missing historical originals. Used as a map, not a substitute for the executable snippets above.'},
      ],
      'unknown_coverage': {
        'missing_paths_confirmed': absence,
        'horizon70_remaining_readable_file': 'outputs/monthly_horizon70_research/labels.py',
        'horizon70_boundary': 'The retained labels module does not recover the missing complete strategy contract. Exact prior coverage of the proposed horizon combination is UNKNOWN.',
        'rank_concordance_recorded_missing_review_sha256': 'e4b6eaa68a03d4bd306eac3ea512ca4e813cc97fb5124fd1c779f0baf7cd0083',
        'rank_concordance_boundary': 'An experiment name and central completion entry do not establish its exact aggregation formula. No claim that the proposed min operator was never tested.',
        'scope_limit': 'Targeted local evidence audit, not exhaustive reconstruction of every historical archive or conversation. Missing originals are not evidence of novelty.'},
      'falsifiable_contrast': 'After a separately frozen state-preserving integration, compare min versus mean on identical common support and identical placement. Require prospectively defined first88 return/risk and holding-set identification gates against incumbent and matched control; later34 only after freeze. Identical holdings, failure to beat matched mean, or failure of gates rejects the proposed added consistency benefit. No dose escalation is implied.',
      'synthetic_mathematics': synthetic_examples(),
      'source_sha256': PINS,
      'review_generator_sha256': sha(Path(__file__)),
      'actual_prices_loaded': False,
      'actual_features_computed': False,
      'actual_orders_computed': False,
      'actual_returns_computed': False,
      'round198_results_read': False,
      'production_files_modified': False,
      'strategy_registered': False,
      'formal_context_written': False,
      'observer_files_accessed_or_modified': False,
    }
    encoded=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False,
                       default=lambda x: str(x) if isinstance(x,F) else x)+'\n'
    with OUTPUT.open('x') as f:
        f.write(encoded)
    print(json.dumps({'path':str(OUTPUT),'sha256':sha(OUTPUT),'source_pins_verified':len(PINS),
                      'synthetic_assertions':'passed','actual_data_computed':False},ensure_ascii=False))

if __name__ == '__main__':
    main()

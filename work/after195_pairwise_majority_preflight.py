"""Write-once design note and synthetic voting proof, never market evaluation."""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/after195_pairwise_majority_preflight.json'
PINS = {
    'AGENTS.md': 'bd902d7ad0f7ef7ef0847bb0e12571ed1cb46490ff0ab203c9828423af4b7538',
    'LOCAL_SETUP.md': '07a0b260d1aa56dfe10492a5fff58d016b4da0d67c0a830c8a470ea47b830b00',
    'work/rank161_proposal.json': '1b08cf674b6b29d87d63c2cf060ab27f2f0660078ca7cf19b29e63e249f54b17',
    'work/monthly_sector_recurrence192_context.json': 'd31f34fee66694e99e3e8924d2c1eb8649f3da714f127af4ef959144c140e03c',
    'work/after191_backbone_coverage.json': 'a02c137a54f4c9962508358f2e2896efdb2f34e4cadbeb3e8c9ef44fd6c2856e',
    'work/monthly_next_mechanism_source_20261015.json': '1540b177a9cdae4a062c695bd50c7a9162a7bc605c8625afb37c3adc389e8a5b',
    'work/monthly_after118_novelty.json': '4ee232d2fb428d49e507df7e134ff8e9bc8b15baec3f04a2d7aa87a5e25a1435',
    'outputs/monthly_path70_research/features.py': '90659640abb4c981ab91f562139bf1fce09308844125c79203b0940e8e8275cc',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_votes(patterns):
    """Integer/Fraction arithmetic only on the explicitly invented three-name table."""
    names = ('A', 'B', 'C')
    assert sum(count for _, count in patterns) == 12
    pair_wins, margins = {}, {}
    for a in names:
        pair_wins[a], margins[a] = {}, {}
        for b in names:
            if a == b:
                continue
            wins = sum(count for order, count in patterns if order.index(a) < order.index(b))
            pair_wins[a][b] = wins
            margins[a][b] = 2*wins-12
    f = {a: Fraction(sum(margins[a].values()), 24) for a in names}
    q = {a: Fraction(sum((v > 0)-(v < 0) for v in margins[a].values()), 2) for a in names}
    return dict(patterns=[dict(descending_gross_order=list(order), blocks=count) for order, count in patterns],
                pair_wins=pair_wins, pair_signed_margins=margins,
                control_F={a: str(f[a]) for a in names}, candidate_Q={a: str(q[a]) for a in names}), f, q


def build_note():
    for name, expected in PINS.items():
        if sha(ROOT/name) != expected:
            raise ValueError('Changed evidence source: '+name)
    example, f, q = synthetic_votes([('ABC', 4), ('ACB', 1), ('BAC', 2), ('BCA', 3), ('CAB', 2)])
    assert f == {'A': Fraction(1, 6), 'B': Fraction(1, 6), 'C': Fraction(-1, 3)}
    assert q == {'A': Fraction(1), 'B': Fraction(0), 'C': Fraction(-1)}
    assert f['A'] == f['B'] and q['A'] != q['B']
    example.update(gross_return_assignment={'first': '1.03', 'second': '1.02', 'third': '1.01'},
        exact_equal_control_different_candidate=True,
        implication='Q cannot be a single-valued function of F, and therefore cannot be a monotone re-expression of F.',
        illustrative_centered_average_zero_based_rank=dict(
            control_F={'A': '.25', 'B': '.25', 'C': '-.5'},
            candidate_Q={'A': '.5', 'B': '0', 'C': '-.5'}))
    cycle, cf, cq = synthetic_votes([('ABC', 4), ('BCA', 4), ('CAB', 4)])
    assert all(value == 0 for value in cf.values()) and all(value == 0 for value in cq.values())
    cycle['majority_cycle'] = ['A beats B in8/12 blocks', 'B beats C in8/12 blocks', 'C beats A in8/12 blocks']
    unknowns = []
    for name in ('monthly_pair_preference70', 'monthly_leader_area70',
                 'monthly_record_growth70', 'monthly_sector_occupancy70'):
        path = ROOT/'outputs'/name
        unknowns.append(dict(directory=str(path.relative_to(ROOT)),
            local_directory_exists=path.exists(), coverage='UNKNOWN',
            reason='Exact original implementation not inspected or unavailable in this bounded source review; a title or preserved summary cannot establish absence of pair-specific majority aggregation.'))
    return dict(
        schema_version=1, status='UNREGISTERED_DESIGN_ONLY',
        created_utc=datetime.now(timezone.utc).isoformat(),
        name='pairwise_monthly_majority_vs_average_win_balance',
        registered=False, candidate_selected=False,
        scope='Prospective mathematical/source-coverage preflight only; no authorization to register or calculate real features, orders or returns.',
        novelty_scope='Potentially distinct aggregation within the already explored persistence/rank family, not a new information family or certified never-tested mechanism.',
        formation=dict(
            query='Original archived monthly signal t and its then-known original49 candidate identities, including ETFs; original trading eligibility is unchanged.',
            price_start='t-294', price_end='t-42', price_count=253,
            return_count=252, blocks=12, block_sessions=21,
            endpoints='t-294+21*k for k=0..12',
            common_domain='C is the original event candidate set with all253 adjusted prices finite strictly positive on every required official session. Both arms use exactly C before, and independently of, trading eligibility. No pair-specific deletion or older-window substitution.',
            missing='A name outside C receives no new correction. If fewer than2 names are in C, both additions are neutral. Original eligibility and fallback rules are not changed.',
            point_in_time_limit='The current known event pool supplies retrospective comparisons at t; this is not a reconstruction of the universe traded at each old block date.'),
        definition=dict(
            gross='G_i,k = P_i,end(k)/P_i,start(k)',
            pair_sign='s_i,j,k = sign(G_i,k-G_j,k), exact equal gross returns give0; comparisons can use products of exact binary-float rational endpoints without floating ratio overflow.',
            pair_margin='M_i,j = sum_k=1..12 s_i,j,k',
            candidate='Q_i = sum_(j in C,j!=i) sign(M_i,j) / (|C|-1)',
            matched_control='F_i = sum_(j in C,j!=i) M_i,j / (12*(|C|-1))',
            candidate_interpretation='Net fraction of distinct opponents beaten by a majority of the12 blocks; a6-6 pair is neutral.',
            control_interpretation='Mean signed win balance across opponent-block observations; equivalent to an affine normalization of average per-block cross-sectional midrank on fixed C.',
            support_and_transform='Use identical C and the same prospective rank/score-budget mapping for both arms. Centered average-tie ranks may match score scale; the example proves rank differences can survive that mapping.',
            integration_not_registered='Keep dominant skip42 core, complete Local67 model corrections, original support, original episode once, monthly EW3 and fixed intramonth units. Exact numeric dose and any first-row convention remain for a future separately frozen contract; none is authorized by this note.'),
        synthetic_equal_borda_proof=example, synthetic_cycle_proof=cycle,
        statistical_interpretation=dict(
            common_market_moves='For any positive common block multiplier m_k, replacing every G_i,k by m_k*G_i,k leaves all pair signs and both statistics unchanged. Common market rise/fall alone does not determine these scores.',
            measured_difference='Distribution of repeated relative wins across named opponents, rather than their pooled total. In the example A narrowly beats both rivals; B decisively beats C but narrowly loses to A.',
            not_chronological='Arbitrarily permuting the12 blocks leaves Q and F unchanged. This is majority recurrence against the same opponents, not consecutiveness, event clustering, migration slope or sensitivity to temporal order.',
            noncausal='Different market betas, sectors, styles and known pool composition can produce persistent relative wins. No issuer-quality, investor-flow, causal-leadership or future-return claim follows.',
            magnitude_loss='A tiny relative win and a large relative win each count1; the candidate further treats7-5 and12-0 pair victories identically. Endpoint wealth may be lower despite a majority of wins.',
            cycles='Pairwise majority need not be transitive; a Condorcet cycle can make all Q equal despite meaningful pair relations.',
            dependence='The12 contemporaneous blocks and all pair observations are cross-sectionally and serially dependent, especially pairs sharing a name. Do not treat12*N*(N-1) comparisons as independent evidence or claim significance from raw counts.',
            threshold_sensitivity='A single block can change a pair from6-6 to7-5; exact ties and calendar alignment require prospective fixed handling, not outcome-based smoothing.'),
        inspected_overlap=[
            dict(path='work/monthly_sector_recurrence192_context.json', locator='lines8-31',
                definition='Twelve21-session exact relative signs against one mapped industry ETF, versus absolute signs; no opponent-specific stock tournament.',
                distinction='Same formation/block family, different comparison target and aggregation order.'),
            dict(path='work/rank161_proposal.json', locator='lines7-11',
                definition='Six calendar-month observations of252skip42 cumulative momentum rank; mean-minus-dispersion and chronological slope.',
                distinction='Rank level/migration, not per-opponent majority of disjoint block gross returns.'),
            dict(path='work/monthly_next_mechanism_source_20261015.json', locator='lines195-198',
                definition='Preserved evidence of all-pairs price/time Kendall tau-a for monthly_rank_concordance70.',
                original_path='outputs/monthly_rank_concordance70/DESIGN.json',
                original_sha256_preserved_not_reverified='c80aa0ab6f21f08d25e89e3e74e17543ad5f63ae371b065cb6b1d994a02ad27a',
                distinction='Single-security chronological price order, not pair-specific cross-security monthly comparison.'),
            dict(path='work/monthly_next_mechanism_source_20261015.json', locator='lines231-234',
                definition='Preserved six-block top-quartile own-sector-relative occupancy versus absolute occupancy.',
                original_path='outputs/monthly_sector_occupancy70/DESIGN.json',
                original_sha256_preserved_not_reverified='b76db31a8b11aee36016d840245ec1a3c1dab32f581108350128bb931feb1751',
                distinction='Rank-threshold residence overlaps the persistence interpretation; missing original prevents exhaustive operator comparison.'),
            dict(path='work/monthly_after118_novelty.json', locator='lines31-40',
                definition='Prior monthly_session_type_research trains mature future-month pairwise payoff differences, with an annual prior-five-year Ridge clock.',
                distinction='Outcome-supervised pairwise prediction; proposed note is descriptive historical majority aggregation with no fit.'),
            dict(path='outputs/monthly_path70_research/features.py', locator='build_features',
                definition='Three70-return block minimum/dispersion/downside and centered-MAD momentum.',
                distinction='Within-name magnitude/path quality; a different window alone would not supply novelty.'),
            dict(path='work/after191_backbone_coverage.json', locator='confirmed_coverage/coverage_boundary',
                definition='Broad long-core, state, risk-adjustment and auxiliary mixing already covered.',
                distinction='No repeated backbone or window/weight grid is proposed here.')],
        unknown_original_coverage=unknowns,
        local_source_feasibility=dict(
            fields='Existing frozen adjusted-close panel, complete official session calendar and original event identities; no new financial or volume input needed.',
            actual_price_panel_opened=False,
            assurance='Feasible in principle from the already used192 inputs; actual per-event coverage has not been measured for this definition.',
            limitations='Inherited estimatedTop32+theme17, retrospective vendor adjustments and pool/classification assumptions remain. This note makes no certified point-in-time or pristine-out-of-sample claim.'),
        evidence_sha256=PINS,
        generator=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(Path(__file__)),
                       wrapper_command='./run_local.sh work/after195_pairwise_majority_preflight.py'),
        activity=dict(synthetic_tables_verified=2, actual_new_feature_calculations=0,
                      actual_new_orders=0, actual_new_returns=0, predictive_model_fits=0,
                      strategy_trials_registered=0, central_modified=False, observers_modified=False,
                      network_used=False))


if __name__ == '__main__':
    if OUT.exists():
        raise FileExistsError('Design note is write-once: '+str(OUT))
    note = build_note()
    with OUT.open('x', encoding='utf-8') as handle:
        json.dump(note, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=note['status'], path=str(OUT), sha256=sha(OUT),
                          synthetic_tables_verified=2, actual_features=0, actual_returns=0)))

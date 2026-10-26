"""Write-once, source-only recovery-retention design with invented-path proof.

This script cannot load price arrays, registered strategies or realized returns.
Its only numerical inputs are the two literal synthetic paths below.
"""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/after196_recovery_retention_preflight.json'
PINS = {
    'AGENTS.md': 'bd902d7ad0f7ef7ef0847bb0e12571ed1cb46490ff0ab203c9828423af4b7538',
    'LOCAL_SETUP.md': '07a0b260d1aa56dfe10492a5fff58d016b4da0d67c0a830c8a470ea47b830b00',
    'work/recovery165_proposal.json': 'ad58156a8d0d450c63d08d3d22f83cea60eb55619a7919998b19eb86436c5a4f',
    'work/recovery165_novelty_review.json': '29b26e28fdd8b453bf8b544beda6f89b300dd1bbd26d31d2121047fb8f3e9d79',
    'work/shock_absorption173_host.py': 'd6a9aa631ec63d905c23e2db24433ebabe0ebce9f21583b6bb800d69e0e5488b',
    'work/market_absorption174_host.py': 'f9714fac8421f456adb449c0747b2724691ebe0cfd6ae528bad345923b2b7503',
    'work/monthly_upside_retention181_context.json': '9804e2ddabd179093b7af64ca40ceab3bf1d4cbe6911bc4b97c19e428a31bfbe',
    'outputs/monthly_multilayer_reconstruction/research.py': '4abec311bfb77eee072479106dbb9f7b375d03d5f287eaf0d7dc9b46df26a798',
    'outputs/monthly_trend_evidence_research/trend_features.py': '94fdadc21b31288771d480cecb110418e6222803892a9041df758de645178364',
    'work/monthly_next_mechanism_source_20261015.json': '1540b177a9cdae4a062c695bd50c7a9162a7bc605c8625afb37c3adc389e8a5b',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toy_measure(prices):
    assert len(prices) == 253 and all(type(x) is int and x > 0 for x in prices)
    anchors = [s for s in range(1, 211) if prices[s] < prices[s - 1]]
    rows = []
    for s in anchors:
        threshold = prices[s - 1]
        first = next((k for k in range(1, 22) if prices[s + k] >= threshold), None)
        stop = s + first + 21 if first is not None else None
        assert stop is None or stop <= 252
        post = prices[s + first + 1:stop + 1] if first is not None else []
        assert first is None or len(post) == 21
        rows.append(dict(anchor=s, pre_loss_price=threshold, first_recovery_sessions=first,
            recovered=first is not None, post_recovery_start=s + first + 1 if first is not None else None,
            post_recovery_end=stop, post_recovery_min=min(post) if post else None,
            durable=bool(post) and min(post) >= threshold))
    return dict(anchors=anchors, events=rows, final_price=prices[-1],
        candidate=str(Fraction(sum(r['durable'] for r in rows), len(rows))) if rows else None,
        control=str(Fraction(sum(r['recovered'] for r in rows), len(rows))) if rows else None)


def build_note():
    for name, expected in PINS.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Changed evidence source: ' + name)
    survivor = [100, 90, 100, 105, 101] + [105] * 248
    relapse = [100, 90, 100, 105, 99] + [105] * 248
    a, b = toy_measure(survivor), toy_measure(relapse)
    assert a['anchors'] == b['anchors'] == [1, 4]
    assert [r['first_recovery_sessions'] for r in a['events']] == [1, 1]
    assert [r['first_recovery_sessions'] for r in b['events']] == [1, 1]
    assert a['final_price'] == b['final_price'] == 105
    assert a['control'] == b['control'] == '1'
    assert (a['candidate'], b['candidate']) == ('1', '1/2')
    assert (a['events'][0]['post_recovery_min'], b['events'][0]['post_recovery_min']) == (101, 99)
    assert a['events'][0]['post_recovery_end'] - a['events'][0]['post_recovery_start'] + 1 == 21
    unknowns = []
    for name in ('outputs/monthly_record_growth70/DESIGN.json',
                 'outputs/monthly_sector_down70/DESIGN.json',
                 'outputs/monthly_sector_residual70/DESIGN.json'):
        unknowns.append(dict(path=name, local_file_exists=(ROOT / name).is_file(), coverage='UNKNOWN',
            reason='Exact historical definition was not recovered or certified in this bounded review. Prior source notes record missing originals; a name, missing file or preserved summary does not prove novelty.'))
    return dict(
        schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        proposed_round=197, name='post_recovery_durability_vs_first_recovery_frequency',
        status='UNREGISTERED_DESIGN_ONLY', registered=False,
        strategy_admitted=False, source_admitted=False, observer_promotion=False,
        scope='Source hashes and two invented numerical paths only; prospective design, not permission to run real factors or a substitute for a registered PLAN.',
        independent_of_round196_outcomes=True, round196_result_files_read=[],
        hypothesis='Among unchanged long-momentum candidates, recovery from a historical one-day loss that then holds its reclaimed pre-loss price for21 sessions may distinguish durable from fleeting rebounds beyond recovery incidence alone.',
        novelty_scope='An incremental ordered-path mechanism within the existing recovery/persistence family, not a new information family, exhaustive novelty certificate, causal effect, or promised return improvement.',
        formation=dict(
            prices='P[0..252] is the original stored adjusted-close series on253 exact official sessions t-294..t-42.',
            source_support='Every253 value must be numeric, finite and strictly positive. No fill, window shortening, alternate source, or pair-specific availability. All original49 monthly candidates, including ETFs, are treated alike before trading eligibility.',
            negative_anchors='E={s in1..210: P[s]<P[s-1]}, with exact stored-price comparisons and no epsilon threshold.',
            first_recovery='T_s=min{k in1..21: P[s+k]>=P[s-1]}; absent if no crossing. The event day s itself is excluded.',
            post_recovery='For a recovered event inspect exactly21 closes P[s+T_s+1]..P[s+T_s+21], not the remainder of a common calendar endpoint.',
            maturity_proof='s<=210,T_s<=21 => s+T_s+21<=252. Both initial recovery and the entire fixed post-recovery horizon end at or before t-42.',
            candidate='Q=(1/|E|)*sum_s I[T_s exists AND min(P[s+T_s+1:s+T_s+22])>=P[s-1]].',
            control_nonpromotable='C=(1/|E|)*sum_s I[T_s exists]. Exactly the same negative anchors, denominator and full253-price support.',
            zeros_and_missing='With |E|=0 both unavailable. With |E|>0 but no recovery, both genuine zero. If recovery occurs but later breaks, only Q loses that event. No extra minimum count or smoothing prior.',
            ranges='0<=Q<=C<=1. A strict breach (<pre-loss price) fails durability; exact equality remains held.',
            exposure='The second observation phase is21 sessions for every recovered anchor, so early recoveries do not mechanically receive longer relapse exposure.',
            dependencies='Anchors and maturity windows overlap; event counts are descriptive and not independent sample sizes.'),
        integration=dict(
            arms=['incumbent_local67', 'recovery_durability', 'recovery_frequency_control'],
            candidate_only_promotable='recovery_durability',
            ranks='Separately higher-is-better centered average-tie rank0/(n-1)-0.5 within common available original49 domain; missing and n<=1 rank0. No eligibility filter for this feature rank.',
            dose=.01, max_absolute_delta=.005, initial_partial_delta_zero=True,
            score_operator='Add .01*rank to the unchanged complete Local67 config4 pre-score, then exactly one original episode using unchanged support. No replacement of core or frozen model corrections.',
            portfolio='Preserve original27 arrays, long-momentum majority and skip42, pool/eligibility, monthly EW3/two halves/cash rules, fixed intramonth units and original execution/cost accounting.',
            parameter_search=False, new_model_fits=0),
        proposed_selection=dict(
            status='Subject to separate next-round context/code/source admission and PLAN registration; not executed here.',
            select_only_first88=True, later34_diagnostic_only=True,
            against_incumbent='Same-close CAGR gain>=.005 and next-open5bp/side noninferiority; Sharpe>=reference-.05, maximum drawdown>=reference-.03, early53/recent35 CAGR>=reference-.025.',
            against_matched_control='Noninferior same-close and next-open5bp/side with the same Sharpe/drawdown/era guards.',
            coverage='At least3 changed first88 holding sets, with>=1 early and>=1 recent, against each reference, as in the existing fixed selection standard.',
            verification='Independent shared-support/ordered-path/rank checks; physically truncated source prefixes; original27 and order parity;9 execution paths45 slices; protected archive checks.',
            falsification='If the fixed candidate fails the first88 baseline/control and costed-execution guards, this reused sample does not support adding post-recovery durability under this design. Do not reverse its sign or retune horizons/dose after outcomes.'),
        synthetic_proof=dict(
            inputs='Each path is253 literal positive integers. First5 values are listed; all remaining248 equal105.',
            survivor_first5=survivor[:5], relapse_first5=relapse[:5],
            survivor=a, relapse=b,
            same_negative_anchors=True, same_first_passages=True, same_final_price=True,
            same_control_different_candidate=True,
            implication='The candidate cannot be recovered from first-passage times or endpoint price alone. The construction proves only a measurement distinction, not an economic benefit.'),
        prior_overlap=[
            dict(path='work/recovery165_proposal.json', locator='formula/history, lines12-17',
                tested='Joint stock/sector down days: difference in first recovery times within21 sessions, versus initial relative loss.',
                distinction='First passage stops at the first crossing; it does not observe21 subsequent closes for a relapse. The proposed own-price comparison is not a retune of its censor duration.'),
            dict(path='work/shock_absorption173_host.py', locator='contract.measurement, line58',
                tested='Volume-grouped negative-day anchors with cooldown;21-session terminal log return.',
                distinction='Endpoint payoff need not identify a recovered level that was later breached.'),
            dict(path='work/market_absorption174_host.py', locator='contract.measurement, line62',
                tested='Same historical recovery endpoint labels residualized by pre-anchor SPY beta.',
                distinction='Market residualization retains the terminal-payoff target, not a second ordered post-recovery path.'),
            dict(path='work/monthly_upside_retention181_context.json', locator='candidate/control/time_policy, lines7-11',
                tested='Positive-day original-gain retention after21 sessions versus binary full endpoint retention.',
                distinction='Its context explicitly says endpoint retention only. This proposal is negative anchors, a first crossing, then fixed-horizon no-rebreach survival; it is not merely switching the sign of the old anchor.'),
            dict(path='outputs/monthly_multilayer_reconstruction/research.py', locator='shape_features, lines54-65',
                tested='shakeout recovery from current daily range/location; failed=(high>previousclose and close<previousclose)*scaled upper move, summed5days.',
                distinction='An intraday failed rebound is not a subsequent21-session failure after first reclaiming a historical pre-loss close.'),
            dict(path='outputs/monthly_trend_evidence_research/trend_features.py', locator='_path_recovery, lines43-57',
                tested='Mean age below a nondecreasing running-window high.',
                distinction='A running maximum boundary differs from the fixed pre-loss level of every event and the second-phase durability observation.')],
        archive_unknown=unknowns,
        limitations=[
            'Complete historical novelty is UNKNOWN. Readable source differences do not establish that every missing older implementation lacks this mechanism.',
            'The control matches event set, first recovery definition, source support, denominator and score budget. It does not match volatility, initial loss severity, beta, or every price-path characteristic.',
            'A quiet or low-volatility path may be rewarded even without stronger expected continuation. Price-only evidence does not identify buying, support orders, liquidity or causal resilience.',
            'Both features can share the same cross-sectional ranks or holdings despite distinct raw values. Q<=C is structural and says nothing about comparative investment performance.',
            'Overlapping anchors and shared market regimes create dependence.21-session phases are fixed research assumptions, not estimated independent hazards or recovered author settings.',
            'Stored adjusted prices retain vendor revision/action limitations. The proxyTop32+theme17 pool and reused development history are not certified pristine PIT/OOS evidence.',
            'No current pullback entry rule, intramonth stop, model refit, observer change or adaptation to round196 performance is proposed.'
        ],
        source_sha256=PINS, script_sha256=sha(Path(__file__).resolve()),
        actual_features_computed=0, actual_orders_computed=0, actual_returns_computed=0,
        actual_price_arrays_opened=False, synthetic_paths_evaluated=2,
        network_requests=0, cloud_paths_read=0, frozen_files_changed=0,
        execution_command='./run_local.sh work/after196_recovery_retention_preflight.py')


if __name__ == '__main__':
    note = build_note()
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(note, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=note['status'], output=str(OUT), sha256=sha(OUT),
        evidence_files=len(PINS), synthetic_paths=2, actual_features=0, actual_orders=0, actual_returns=0)))

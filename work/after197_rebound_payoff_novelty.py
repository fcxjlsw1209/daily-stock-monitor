"""Source-only rebound-payoff preflight and exact synthetic algebra; no market loader."""
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'work/after197_rebound_payoff_novelty.json'
PINS = {
    'outputs/four_factor_rebound/factors.py': '18a3aa400aec342d2a9eb3161bfea8b3f8b2c575fea32a16122f7d87e29ba6ff',
    'outputs/monthly_daily_episodes/episodes.py': '8ae91e55e0d662cb6dee9cba6d1422aa3713b779976de69b4560194d311ae16a',
    'outputs/monthly_daily_episodes/PLAN.json': '992f33b96a13b1fa2b7fa33aea382ce88bc3d5690a535f79f119b8e53bcfc194',
    'outputs/monthly_company_context/context_research.py': '271fdb223f66a0d890d015073cb2b49664f94eaab9a86ca11742dad29e835f7c',
    'outputs/monthly_target_definition70/labels.py': '54662ea980b03e6088afa63318321da1316fd03cde1a72edffb33d4d2a26e618',
    'outputs/monthly_target_definition70/PLAN.json': 'ab9fac077e12c15cde259dd648dec18c5c6133195b38479240693050c1c47e39',
    'work/shock_absorption173_host.py': 'd6a9aa631ec63d905c23e2db24433ebabe0ebce9f21583b6bb800d69e0e5488b',
    'work/market_absorption174_host.py': 'f9714fac8421f456adb449c0747b2724691ebe0cfd6ae528bad345923b2b7503',
    'work/monthly_transition175_host.py': '062cba7ee33ec239e646421b250ddd4fe6cccaf0795f1a3811acd0ed0c0ff636',
    'work/monthly_sector_transition189_context.json': '93b13abbb2954c915e2283b19f849db97ad6c98709eb955d7472a30c8101dd48',
    'work/recovery165_proposal.json': 'ad58156a8d0d450c63d08d3d22f83cea60eb55619a7919998b19eb86436c5a4f',
    'work/monthly_upside_retention181_context.json': '9804e2ddabd179093b7af64ca40ceab3bf1d4cbe6911bc4b97c19e428a31bfbe',
    'work/monthly_recovery_durability197_context.json': '89d00c63a1281700e1b55c490011d04c7ced178b29c9bac1f3d430f4389d1923',
    'work/recovery165_novelty_review.json': '29b26e28fdd8b453bf8b544beda6f89b300dd1bbd26d31d2121047fb8f3e9d79',
    'work/monthly_next_mechanism_source_20261015.json': '1540b177a9cdae4a062c695bd50c7a9162a7bc605c8625afb37c3adc389e8a5b',
    'outputs/monthly_state_rank_research/state_rank.py': '53866e5ffcb06146f73bb409747df4bb6f5b7864c1bfe0edebb639d551aa4c6d',
    'outputs/monthly_stability_monitor/inputs.py': 'dd97fb4a9f7b2102d2c7f497825526a484628fa86ee9891312bfb5abad7c8ded',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toy(y, sigma):
    """Exact algebra on invented event summaries, NOT financial-source values."""
    n = len(y)
    assert n and n == len(sigma) and all(x > 0 for x in sigma)
    inverse = [1 / x for x in sigma]
    mean_y, mean_inverse = sum(y) / n, sum(inverse) / n
    candidate = sum(a * b for a, b in zip(y, inverse)) / (n + 20)
    control = mean_inverse * sum(y) / (n + 20)
    covariance = sum((a - mean_y) * (b - mean_inverse) for a, b in zip(y, inverse)) / n
    assert candidate - control == F(n, n + 20) * covariance
    return dict(n=n, sum_y=str(sum(y)), mean_sigma=str(sum(sigma) / n),
                mean_inverse_sigma=str(mean_inverse), candidate_times_sqrt21=str(candidate),
                control_times_sqrt21=str(control), covariance=str(covariance),
                difference_times_sqrt21=str(candidate - control))


def main():
    assert not DESTINATION.exists(), 'Write-once preflight already exists'
    for relative, expected in PINS.items():
        assert digest(ROOT / relative) == expected, relative
    first = toy([F(1, 100), F(3, 100)], [F(1, 100), F(3, 100)])
    switched = toy([F(1, 100), F(3, 100)], [F(3, 100), F(1, 100)])
    for key in ('n', 'sum_y', 'mean_sigma', 'mean_inverse_sigma', 'control_times_sqrt21'):
        assert first[key] == switched[key]
    assert first['candidate_times_sqrt21'] != switched['candidate_times_sqrt21']
    constant_risk = toy([F(-1, 100), F(3, 100)], [F(2, 100), F(2, 100)])
    constant_payoff = toy([F(2, 100), F(2, 100)], [F(1, 100), F(3, 100)])
    assert constant_risk['difference_times_sqrt21'] == '0'
    assert constant_payoff['difference_times_sqrt21'] == '0'
    absent = ['outputs/monthly_payoff_preference70', 'outputs/monthly_sector_down70',
              'outputs/monthly_sector_residual70', 'outputs/monthly_downrisk_horizon70']
    unknown = {p: {'present': (ROOT / p).exists(), 'exact_equivalence': 'UNKNOWN'} for p in absent}
    coverage = [
        dict(path='outputs/four_factor_rebound/factors.py', locator='matured_average/build, lines4-28',
             covered='Own event realized next-open-to-open payoff after6-session maturation, rolling756 sessions,20-observation prior at pooled cross-security mean; separate MFE+MAE route and volatility features.',
             difference='Already actual-payoff shrinkage, not win rate. Does not divide each own negative-event payoff by strictly pre-event63 volatility, then compare with the same inverse-risk mass applied to pooled payoff.'),
        dict(path='outputs/monthly_daily_episodes/episodes.py', locator='label_path/estimate, lines86-99 and141-180; PLAN episode/effect',
             covered='Own nonoverlapping A/B pullback episodes;5/10/21-session next-open excess-SPY or excess minus.5 MAE;24-month decay/context matching; subtract SE and mass/(mass+6) shrink, then divide by CURRENT QUERY vol63*sqrt(h). Current signal/eligibility and rank3..5 boundary required.',
             difference='Risk-adjusted shrunk rebound memory itself is covered. Event-specific pre-loss risk normalization before summation differs from a single query-risk denominator after estimation; proposed comparison has no current pullback trigger.'),
        dict(path='outputs/monthly_target_definition70/labels.py', locator='build_targets lines105-174; PLAN targets/model',
             covered='Monthly supervised excess target divided by its own signal-time63-simple-return sigma*sqrt21, clipped5 and monthly centered before annual Ridge.',
             difference='Per-observation risk normalization is also covered. Exact same-security negative-anchor mean with a matched inverse-risk-mass control is not this pooled monthly regression or clipped target.'),
        dict(path='outputs/monthly_company_context/context_research.py', locator='analogue, lines99-114; initialize',
             covered='Same-company monthly excess payoff, age/context weights, shrunk toward current global Ridge forecast.',
             difference='Same-security conditional magnitude estimates not new; this proposal conditions on negative daily anchors and isolates event-risk/payoff pairing, not new similarity bandwidth.'),
        dict(path='work/shock_absorption173_host.py', locator='contract lines56-65',
             covered='Greedy21-session negative-day events; y=21-session log payoff; activity H/L contrast sum(yH)/(nH+20)-sum(yL)/(nL+20); ordinary-negative control sum(y)/(n+20).',
             difference='Primary close overlap. Preserve21 horizon/cooldown and20 zero prior, but use no activity split and compare individual pre-event risk units with matched aggregate inverse-risk mass. Raw-payoff shrinkage alone cannot be proposed as new.'),
        dict(path='work/market_absorption174_host.py', locator='contract lines60-69',
             covered='Same173 event groups; subtract pre-event252-return SPY beta times future SPY payoff, vs raw contrast.',
             difference='Beta removal is not risk-unit scaling; proposal makes no market-neutral or causal alpha claim.'),
        dict(path='work/monthly_transition175_host.py', locator='contract measurement/design_choice',
             covered='Same-security60 blocks; prior block sign conditional next-block mean, shrunk with20 observations at unconditional mean.',
             difference='Conditional payoff mechanism covered, but neither daily-loss event sampling nor paired event risk weighting.'),
        dict(path='work/monthly_sector_transition189_context.json', locator='measurement/known_overlap',
             covered='Own-sector relative210-session state conditions mature21-session relative payoff;60 labels and pooled prior.',
             difference='State-conditioned relative mean differs from pre-loss risk-weight/payoff covariance; no novelty-family claim.'),
        dict(path='work/recovery165_proposal.json', locator='candidate/control definitions',
             covered='Stock/sector joint-down first-passage recovery race versus initial relative loss.',
             difference='Crossing time, not mature payoff magnitude.'),
        dict(path='work/monthly_upside_retention181_context.json', locator='fixed feature definition',
             covered='Positive original log gains retained after21 days, capped at original gain, versus weighted endpoint recovery indicator.',
             difference='Clipped original-gain retention is not negative-anchor realized payoff in ex-ante risk units.'),
        dict(path='work/monthly_recovery_durability197_context.json', locator='formation only; no outputs/results read',
             covered='Negative-day first recovery followed by21-session no-rebreach proportion versus recovery incidence.',
             difference='Current registered design is counts, not magnitude. This preflight did not read197 actual factors, orders or outcomes.'),
    ]
    report = dict(
        schema='after197_rebound_payoff_novelty_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        status='UNREGISTERED_DESIGN_ONLY', registration_authorized=False,
        conclusion='ONE_BOUNDED_REFINEMENT_NO_EQUIVALENT_FOUND_IN_READABLE_SUBSET',
        novelty='Not a new factor family: realized rebound payoff, shrinkage and volatility normalization each have prior implementations. The limited distinction is pairing own negative-event payoff with its own pre-loss risk, tested against identical aggregate inverse-risk weight. Missing originals make exhaustive equivalence UNKNOWN.',
        attribution='Executable local guidance reconstructions are evidence of prior tests, not direct proof of exact original-author formulas or an author-endorsed parameter set.',
        candidate=dict(
            name='same_security_event_risk_normalized_rebound_payoff',
            prices='253 original stored adjusted closes P[0..252], exact official sessions t-294..t-42. Require all numeric finite positive; no filling/window shortening.',
            anchors='Scan s64..231 in order. Strict negative P[s]<P[s-1]; greedily retain first then next negative s>=last+21. Anchor selection uses no labels or risk validity. Every selected anchor belongs to both arms.',
            risk='sigma_s=sample std(ddof1) of63 SIMPLE returns P[j]/P[j-1]-1, j=s-63..s-1. Uses64 prices P[s-64..s-1], excludes the loss day and all payoff observations.',
            payoff='y_s=log(P[s+21])-log(P[s]); all payoffs mature by P[252]=t-42. Historical descriptive labels, not newly computed portfolio returns.',
            candidate='Q=sum_s(y_s/sigma_s)/((n+20)*sqrt(21))',
            matched_control='C=mean_s(1/sigma_s)*sum_s(y_s)/((n+20)*sqrt(21))',
            identity='Q-C = n/(n+20)/sqrt(21) * population_Cov(y_s,1/sigma_s). Both exact mathematical identities Q=C when all risks equal or all payoffs equal.',
            support='n=0 makes both unavailable. A selected required sigma nonfinite or<=0 makes whole cell unavailable, never remove individual events. No min-n threshold, no positive-payoff-only filter, no volume or sector filter. Valid-source arithmetic overflow/nonfinite must fail explicitly rather than silently fill/clip. Numerical implementation requires separate registration review.',
            ranking='Separately rank higher Q/C over same available original49 universe before eligibility, centered average-tie rank0/(m-1)-.5, missing/m<=1 zero; fixed+.01 rank, first partial delta0.',
            integration='Complete unchanged Local67 config4 pre-score plus one delta, then exactly one unchanged episode/support. Long skip42 core, original27 arrays/models/eligibility/pool, EW3/twohalf/cash and fixed monthly units retained.',
            parameter_origins={'253_prices_skip42':'same existing skipped long formation and197 source envelope',
                '63_simple_return_risk':'existing original vol63 source convention; strictly shifted before event',
                's64_start':'minimum causal64-price support for63 pre-event returns',
                's231_end':'last anchor with21-session label matured by252',
                '21_horizon_and_cooldown':'fixed173 historical payoff/cooldown precedent, not outcome-selected',
                '20_zero_prior':'fixed173 zero-payoff pseudo-event count; retained without tuning',
                'dose_01':'existing small auxiliary rank budget, not a searched amplitude'}),
        discarded_before_calculation=dict(formula='sum(y)/((n+20)*sqrt21*mean(sigma))',
            reason='Jensen mismatch: mean(1/sigma) generally exceeds1/mean(sigma), so a comparison with Q includes deterministic risk-dispersion scaling even at constant payoff.',
            revised_before_registration=True, actual_features=0, trials=0),
        toy_proof=dict(input_type='Invented exact Fraction event summaries only, not prices or source financial values',
            common_scale='Reported Q and C omit common1/sqrt21, which does not affect algebraic comparisons.',
            same_sum_payoff_n_mean_risk_and_inverse_risk=True, pairing1=first, pairing2=switched,
            constant_risk=constant_risk, constant_payoff=constant_payoff,
            assertion='Changing only the payoff-risk pairing changes Q while leaving matched C unchanged. Difference exactly equals shrunken population covariance.'),
        prior_coverage=coverage, incomplete_originals=unknown,
        source_feasibility=dict(loader='outputs/monthly_state_rank_research/state_rank.py::panel_load -> outputs/monthly_stability_monitor/inputs.py::load_panel',
            required='Only original stored adjusted closes plus official calendar and original monthly meta. No new price/sector/event source or network collection.',
            inspected='Source code only; loader not imported/called, CSV/NPZ not decoded.',
            actual_support='UNKNOWN until separately registered prepare; original253 support does not guarantee positive required event sigma or any negative anchors.'),
        limitations=[
            'Inverse-volatility weighting can amplify low-risk regimes and small denominators. The control matches total inverse-risk mass, not stock beta, loss severity or every confounder.',
            'A positive covariance-style difference is descriptive. No causal resilience or future-profit conclusion follows from its algebra.',
            '21-session cooldown reduces direct label overlap but does not create independent events. In253 prices at most8 anchors are selected; n/(n+20)<=2/7, so shrinkage is deliberately strong and evidence sparse.',
            'Shrunk event averages can remain negative and identify persistent drifting losses. No success-rate interpretation or guaranteed separation of rebound from decline.',
            'No extra minimum event count, current drawdown gate, sign reversal, risk clipping or horizon search after outcomes.',
            'Reused development history and proxyTop32+theme17 pool/vendor adjusted prices retain declared non-pristine-PIT/OOS limitations.',
            'Passing a bounded source review is not factor/data admission or a strategy PLAN. First88 selection, matched-control and next-open-cost/era guards require separate prospective registration.'
        ],
        inputs_sha256=PINS, generator_sha256=digest(Path(__file__)),
        execution_command='./run_local.sh work/after197_rebound_payoff_novelty.py',
        source_files_verified=len(PINS), toy_cases=4,
        actual_new_feature_evaluations=0, actual_new_orders=0, actual_new_portfolio_returns=0,
        new_model_fits=0, new_registered_trials=0, round197_results_read=False,
        observer_modified=False, central_modified=False)
    with DESTINATION.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=report['status'], source_files_verified=len(PINS), toy_cases=4,
                         output=str(DESTINATION), output_sha256=digest(DESTINATION))))


if __name__ == '__main__':
    main()

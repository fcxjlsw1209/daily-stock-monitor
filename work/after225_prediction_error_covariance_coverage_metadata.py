"""Metadata-only bounded coverage receipt; no new strategy computations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path('/Users/shiweili/code/monthly-stock-research')
OUTPUT = ROOT / 'work/after225_prediction_error_covariance_coverage.json'
SOURCES = [
    'AGENTS.md', 'LOCAL_SETUP.md',
    'work/monthly_forecast_blend211_context.json',
    'work/monthly_forecast_blend211_model.py',
    'outputs/monthly_forecast_blend211/PLAN.json',
    'work/monthly_identity_transfer206_model.py',
    'work/monthly_identity_transfer206_host.py',
    'outputs/monthly_target_definition70/labels.py',
    'outputs/monthly_target_definition70/study.py',
    'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control/PARITY.json',
    'outputs/monthly_cost_portfolio_research/trio_model.py',
    'outputs/monthly_cost_portfolio_research/study.py',
    'outputs/monthly_trend_evidence_research/PLAN.json',
    'outputs/monthly_trend_evidence_research/study.py',
    'outputs/monthly_trend_evidence_research/uncertainty.py',
    'work/after211_monotone_path_coverage.json',
    'work/after210_forecast_blend_coverage.json',
    'work/after208_path_shortfall_coverage.json',
    'outputs/monthly_honest_local70/test_study.py',
    'outputs/monthly_local_linear70/study.py',
    'outputs/monthly_opportunity_alignment/study.py',
    'outputs/monthly_budget_calibration70/model.py',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run through the project wrapper at its registered local root')
    if OUTPUT.exists():
        raise FileExistsError('Never overwrite a saved coverage receipt')
    pins = {p: digest(ROOT / p) for p in SOURCES}
    old = json.loads((ROOT / 'work/monthly_forecast_blend211_context.json').read_text())
    expected = old['evidence_sha256']
    checked = {p: pins[p] for p in SOURCES if p in expected}
    for p, actual in checked.items():
        if actual != expected[p]:
            raise ValueError('Inherited source digest mismatch: ' + p)
    parity = json.loads((ROOT / 'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control/PARITY.json').read_text())
    target_check = parity['checks']['monthly']['target_log_excess']
    if parity.get('passed') is not True or target_check.get('max_abs_error') != 0.0:
        raise ValueError('Original169 target parity receipt is not exact')
    missing = [
        'outputs/monthly_listwise_top3_70/study.py',
        'outputs/monthly_listwise_top3_70/model.py',
        'outputs/monthly_trio_shortfall131/study.py',
        'outputs/monthly_trio_shortfall131/model.py',
        'outputs/monthly_honest_local70/study.py',
        'outputs/monthly_honest_local70/model.py',
        'outputs/monthly_kernel_residual70/study.py',
        'outputs/monthly_kernel_residual70/model.py',
    ]
    unknown = []
    for p in missing:
        present = (ROOT / p).is_file()
        unknown.append(dict(path=p, exists=present, definition_status='UNKNOWN',
            reason='Exact executable definition was not recovered in the bounded source search; existence alone would not resolve equivalence.'))
    payload = dict(
        schema='after225_prediction_error_covariance_coverage_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        status='BOUNDED_METADATA_COVERAGE_ONLY_NOT_EXPERIMENT_REGISTRATION',
        implementation_sha256=digest(Path(__file__)),
        source_sha256=pins,
        inherited211_source_pins_verified=checked,
        coverage_scope=dict(
            method='Read existing executable definitions, registered PLANs and prior coverage notes; hash their bytes under the offline local wrapper.',
            exhaustive=False,
            exact_prior_cross_security_asof_error_covariance='NOT_FOUND_IN_BOUNDED_READABLE_SOURCES',
            new_economic_family=False,
            older_missing_definitions='UNKNOWN',
            outcomes225_read=False,
            outcomes225_used_for_design=False,
            strategy_parameters_locked=False,
            future_registration_required=True),
        covered_definitions=[
            dict(name='211 annual convex forecast blend',
                evidence=['work/monthly_forecast_blend211_context.json', 'work/monthly_forecast_blend211_model.py', 'outputs/monthly_forecast_blend211/PLAN.json'],
                exact_definition='Same security own-constant and peer forecasts: d=own-peer; yearly alpha=clip(sum_month mean_rows((y-peer)*d)/sum_month mean_rows(d*d),0,1), fixed-half fallback. Previous December31 maturity cutoff, lower-exclusive five-year signal window, at least18 accepted months, at least2 rows per month; month-equal aggregate weights.',
                covariance_scope='The two prediction errors for the same security share a target. This is model-source mixing, not a cross-security joint EW3 error-risk calculation.',
                historical_source_keys={'own':'raw_control', 'peer':'raw_candidate', 'pooled':'raw_source_log'},
                distinction='Do not relabel this prior experiment as having already tested cross-security prediction-error covariance.'),
            dict(name='Existing joint trio utility',
                evidence=['outputs/monthly_cost_portfolio_research/trio_model.py', 'outputs/monthly_cost_portfolio_research/study.py'],
                exact_definition='All56 triples of the baseline top8 eligible assets; current126 simple daily price returns from127 closes. Joint features are mean pair centered return correlation, mean pair downside cosine and dispersion of individual sample volatilities. Annual Ridge learns next-calendar-month log(arithmetic EW3 gross return) relative to the original top3, with matched additive-feature control.',
                prior_budgets=[0.025,0.05,0.10],
                covariance_scope='Observed price-return co-movement and a portfolio-return target, not covariance of historical dated prediction errors.',
                distinction='A new prediction-error-risk comparison must not claim that portfolio-level scoring or diversification itself is novel.'),
            dict(name='Leave-one-signal-year-out Ridge evidence',
                evidence=['outputs/monthly_trend_evidence_research/uncertainty.py', 'outputs/monthly_trend_evidence_research/PLAN.json'],
                exact_definition='For the same current event compare pairwise prediction differences across deleted-year refits; SE=sqrt((B-1)/B*sum((fold_difference-mean_fold_difference)^2)). Annual previousDecember31 cutoff, five-year mature history, 24month half-life, alpha10, minimum36 full months,4 year blocks,18 months per deleted-year fold. Each fold refits its own scaler and weights.',
                covariance_scope='Joint prediction movement under training-parameter perturbation, not realized historical errors from each prediction original annual vintage.',
                prior_boundary_design='Original top2 locked; ranks4/5 within0.02 of third challenge with budgets0.025/0.05 and k0/1/2 stability penalties.',
                distinction='Prediction covariance is already acknowledged here, but its sampling object differs from cross-security matured forecast-error covariance.'),
            dict(name='Prior deferred direction',
                evidence=['work/after211_monotone_path_coverage.json'],
                exact_definition='The prior coverage receipt explicitly deferred prediction-error covariance portfolio controls; it also states generic type ranks, raw-correlation diversification and annual loss mixing were already covered.',
                distinction='A deferred note is not evidence an exact experiment was implemented, nor proof of exhaustive novelty.')],
        target_and_error_semantics=dict(
            target_key='target_log_excess',
            target_source='outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control/TRAIN_INPUTS.npz',
            target_source_access='NOT_OPENED_OR_DECODED; semantics and prior exact parity checked from source code and saved metadata only',
            original_target_definition='First clip asset next-calendar-month log gross return minus SPY log gross return to[-1,1]; then subtract the original common-cohort monthly mean of these clipped values. Do not center on a selected trio, covariance support subset or a new universe.',
            target_after_demeaning_not_necessarily_in_clip_interval=True,
            prior169_target_parity=target_check,
            prediction_source='outputs/monthly_identity_transfer206/prepared/FORECASTS.npz',
            prediction_key='raw_local_linear',
            prediction_source_access='NOT_OPENED_OR_DECODED; archive key mapping read from frozen206 host/model',
            error='e(security,signal)=original annual-asof raw_local_linear prediction minus original cached target_log_excess',
            not_the_error_source='206 MODEL_AUDIT own_residuals are residuals from its then-current fitted pooled model; they must not substitute for dated as-of prediction errors.',
            maturity='A historical target may enter only after its label_end is on or before the declared calibration cutoff. Join exact(signal_date,security_id,label_end); do not use a current model to backfill historical forecasts.',
            annual_prediction_clock='206 source forecasts are causal annual replays with priorDecember31 cutoff and a lower-exclusive precedingfive-calendar-year signal window; this is not proof of historical live archival availability.',
            economic_object='Relative transformed-target prediction-error risk. It is not actual arithmetic EW3 cash-return volatility, absolute market risk, a calibrated confidence interval or a guarantee of lower drawdown.',
            caveats=[
                'Clipping and monthly cross-sectional demeaning remove information and can change common-mode error covariance.',
                'The average of log-target errors is not the return of a fixed-unit three-asset portfolio.',
                'Exact cached identities and original alias assumptions are inherited; no new permanent-issuer continuity proof is established.',
                'Cached as-of forecasts begin in2016; early calibration support must be audited rather than filled using later models.',
                'Retrospective vendor/proxy universe histories and repeated development samples remain; strict point-in-time and pristine out-of-sample claims are not supported.']),
        conceptual_contrast_pending_registration=dict(
            hypothesis='Among near-equivalent long-trend opportunities, avoid combinations whose past dated relative prediction errors move together, while leaving the dominant skip42 momentum and monthly fixed-unit EW3 framework intact.',
            possible_matched_comparison='On one identical joint-complete historical error matrix, compare risk including cross-security covariance with its diagonal-only counterpart; preserve all marginal information and support so the contrast isolates cross terms.',
            not_a_formula_registration=True,
            parameters_not_fixed_here=['candidate_set','boundary_score_gap','lookback','minimum_common_months','variance_estimator','risk_transform','score_budget','integration_slot','selection_gates'],
            requirements_to_predeclare=['Exact historical joint support and maturity clocks','Treatment of invalid or insufficient common history','No pairwise deletion yielding inconsistent covariance samples','Causal prefix and poisoned-future tests','Identical source support and budgets for candidate and control','No expansion of total score budget or intramonth trading'],
            falsification='Adequate support but identical relative risk ordering means the comparison has little selection identification; changed holdings that fail fixed matched-control and baseline evaluation do not support promotion. Missing support must be reported separately from an economic failure.'),
        incomplete_legacy_definitions=unknown,
        bounded_filename_search='rg --files outputs work | rg (listwise_top3_70|trio_shortfall131|honest_local70|kernel_residual70|forecast.*cov|error.*cov); only honest_local70/test_study.py and recent coverage files were found for these legacy names.',
        scope_flags=dict(npz_decoded=False, strategy_code_imported=False, actual_new_statistics=False,
            actual_new_features=False, actual_new_labels=False, actual_new_models=False,
            actual_new_orders=False, actual_new_returns=False, actual_new_monthly_support=False,
            strategy_registered=False, frozen_files_modified=False, observer_files_modified=False,
            strict_point_in_time_certified=False))
    with OUTPUT.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(OUTPUT), sha256=digest(OUTPUT),
        implementation_sha256=payload['implementation_sha256'], source_pins=len(pins),
        inherited_pins_verified=len(checked), actual_calculations=False),ensure_ascii=False))


if __name__ == '__main__':
    main()

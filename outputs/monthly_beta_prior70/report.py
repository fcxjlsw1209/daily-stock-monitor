"""Render the frozen matched joint-beta estimator and support-budget study."""
from pathlib import Path
import json

P=Path(__file__).resolve().parent
CANDIDATES=tuple(mode+'_'+suffix for mode in ('ols','sector_prior','shrunk') for suffix in ('025','050'))


def main():
    read=lambda name:json.loads((P/name).read_text())
    results,frozen,validation,attribution=[read(name) for name in
        ('RESULTS.json','FROZEN.json','VALIDATION.json','ATTRIBUTION.json')]
    lines=[
        'Long residual leadership: matched OLS, own-sector prior and noise-dependent beta shrinkage.',
        'Eight fixed configurations: original65_cached, local67_cached, and ols/sector_prior/shrunk at support budgets0.025 and0.05. All candidates start from original65. Historical estimated32-stock+assumedtheme17 pools and eligibility remain unchanged. local67 is a cached benchmark only; there is no local model or episode overlay.',
        'Each signal uses379 positive finite price observations from t-420 through t-42 for the stock, SPY and the signal-date own-sector ETF. Their378 simple returns contain126 warmup returns and252 residual-formation days. For every residual day, the joint market/sector fit uses exactly126 returns ending one day earlier. The current residual return is not included in its coefficient estimate.',
        'The two-factor fit centers both factors and the stock return and includes an intercept. Its Gram determinant must exceed S00*S11*1e-8 with positive diagonals; no single-factor fallback is allowed. The fitted intercept is not subtracted when forming residuals, so alpha/drift remains in all three paths.',
        'OLS beta is inverse(S)*cross. The fixed sector prior is beta0=(0,1), giving residual y-sector. OLS noise variance is centered residual SSE/(126-3). With fixed prior covariance0.25*I, lambda=4*noise_variance and shrunk beta solves(S+lambda*I)*beta=cross+lambda*beta0. Exactly zero noise gives OLS. There is no window, prior-strength or state search.',
        'For each path the raw factor is sum of252 residuals divided by their sample standard deviation (ddof1). Every daily joint fit must be valid and all three path deviations must be finite and greater than1e-12. Availability is common to all three arms. Missing, invalid, collinear or undefined paths make every raw/rank unavailable; no fill or interpolation is used.',
        'Known original signal-date sector mapping applies throughout the formation window. ETFs and unknown mappings remain unavailable. Each arm uses average percentile minus0.5 over finite members of the full current49 pool, before original eligibility. Missing ranks stay NaN. This is a historical sector proxy and is not claimed identical to earlier residual studies that mapped sectors daily.',
        'On available cells, score=core+(1-budget/(1-cw))*(baseline-core)+budget*rank. The exact cached50%/52% long-momentum core is unchanged. The new factor replaces the stated amount of original support rather than adding budget. A valid zero factor still reduces the old support contribution; only unavailable cells retain baseline bytes exactly. The original eligible global selector chooses all three seats, without boundary locks or post-correction reranking.',
        'Residual momentum normalized by idiosyncratic variation was already tested. This comparison specifically tests a fixed-prior joint-coefficient estimator against matched OLS and the prior-only path. It uses Gaussian/ridge posterior-mean algebra, but does not estimate calibrated posterior uncertainty or a cross-sectional empirical-Bayes prior. It does not replicate Vasicek or Ledoit-Wolf.',
        'Initialize pins code, inputs and protected files before factor computation. Prepare verifies only the two old controls, with no new candidate holding selection. ACTUAL_FEATURE_REVIEW must pass with current seven-artifact hashes before train. META/PRICES are exact source copies; original64 raw bindings remain separate from calendar, mapping and implementation dependencies.',
        'First88 gates are recorded against original65 and local67 separately. Primary defaults to local67. A candidate must pass its five gates and improve training CAGR by more than1e-12. Highest eligible training CAGR wins; differences within1e-12 tie, then higher Sharpe and lower id decide. All eight diagnostics freeze before later34; walk-forward36/60/76 uses the same rule.',
        'Name | Full CAGR | Sharpe | MaxDD | First88 CAGR | Later34 CAGR | Nextopen0 CAGR | Nextopen5bp/side CAGR']
    for name,row in results.items():
        metric=row['metrics'];slices=row['slices']
        lines.append(f"{name} | {metric['cagr']:.2%} | {metric['sharpe']:.3f} | {metric['max_drawdown']:.2%} | {slices['training2016_2023']['cagr']:.2%} | {slices['later34']['cagr']:.2%} | {row['next_open_zero']['metrics']['cagr']:.2%} | {row['next_open_cost']['metrics']['cagr']:.2%}")
    lines.append('')
    for name in CANDIDATES:
        row=frozen['selected'][name]
        lines.append(f"{name}: first88 original65 gate={row['original65_admitted']}; local67 gate={row['local67_admitted']}.")
    lines += [f"Primary: {frozen['selected']['primary']['spec']}; target70 reached: {validation['primary_target70_reached']}.",
        '', 'Eighteen descriptive comparisons at matched budgets; undo-best diagnostics are not selection rules:']
    for name,row in attribution.items():
        lines.append(f"{name}: holding-set changes={row['changed_months']}, ordered changes={row['ordered_changed_months']}; entering/leaving seats={row['entering_seats']}/{row['leaving_seats']}; split changes={row['split_counts']}; undo-best1/3 CAGR={row['undo_best1_diagnostic_cagr']:.2%}/{row['undo_best3_diagnostic_cagr']:.2%}.")
    lines += ['',
        'Long-only EW3/2half/<2cash, monthly fixed adjusted units, no intramonth exit or forced final sale. Same-final-close signal/fill is an idealization. Eight configurations plus primary and QQQ each have same-order next-open0 and5bp/side replays:30 execution paths. Missing held future prices fail rather than changing selection.',
        'Repeatedly researched history is not pristine out-of-sample. Estimated market-cap pools, assumedETF17, historical industry proxies and vendor revisions remain limitations. A fixed prior can bias coefficients and erase real leadership. No observer changes or automatic promotion; results do not guarantee future returns.',
        f"Protected files checked by evaluate: {validation['protected_files']}; changes={validation['protected_changes']}."]
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

"""Render the frozen calendar-mean endpoint experiment without fitting or selection."""
from pathlib import Path
import json

P=Path(__file__).resolve().parent


def main():
    read=lambda name:json.loads((P/name).read_text())
    results,frozen,validation,attribution=[read(name) for name in
        ('RESULTS.json','FROZEN.json','VALIDATION.json','ATTRIBUTION.json')]
    lines=[
        'Long-trend core using complete-calendar-month average price endpoints.',
        'Four preregistered configurations: original65_cached, local67_cached, calendar_mean_half and calendar_mean_full. Historical Top32 plus fixed theme17,122 periods. No model fit, new labels, episode overlay, pool change or parameter search. The two new configurations start from exact original65 baseline bytes; local67 is a comparison only.',
        'For an official XNYS month-end signal in monthM, the numerator is the arithmetic mean of adjusted closes over every official session ofM-2. The calm denominator is the corresponding completeM-12 mean; otherwiseM-14. Calm raw is this price ratio minus1; noncalm divides the ratio-minus1 by the original current63 simple-return sample standard deviation, ddof1, without annualization. This is a ratio of monthly means, not a mean of daily ratios.',
        'The initial2016-08-17 partial-month signal and any non-month-end signal keep the original core. Required endpoint months must contain every official session with positive finite prices. Missing sessions/prices or an invalid noncalm risk denominator make the signal unavailable, without changing original eligibility. Calm does not require this denominator. The latest two calendar months are excluded from endpoint prices, while the original current risk denominator remains.',
        'Raw values are average-percentile-ranked across available members of all49 securities before eligibility. Missing-rank diagnostics equal oldcore/coreweight, but new_core starts as an exact oldcore copy; only available cells receive coreweight*rank. Constant raw ties retain the original percentile convention. No fill, shortened calendar month, last-visible-date anchor or manual zero-rank substitution.',
        'Half/full scores equal cached original65 plus0.5/1.0 times(new_core-oldcore). Only available nonzero deltas are written, preserving missing/non-month-end/zero-delta score bytes. The total long-momentum core weight remains50% in calm and52% otherwise. All original65 industry, volatility, path, acceleration and CMF support contributions stay unchanged; the original global eligible selector determines every seat.',
        'This experiment changes endpoint aggregation and formation timing, and can change cross-sectional ranks. It does not isolate smoothing alone relative to the original trading-day endpoints. Comparing half with full isolates the registered replacement intensity; comparing either with local67 also changes the auxiliary-model architecture.',
        'Prepare computes factors and validates only the two old controls. Candidate selection requires the passed independent actual review bound to DATA, META, PRICES, FEATURE_AUDIT, INPUT_HASHES, CACHE_HASHES and PLAN. META and PRICES are source-file byte copies. Original64 raw bindings are checked separately from additional calendar code, closure evidence, manifest and dependency bindings.',
        'First88 gates are reported against original65 and local67 separately. The primary defaults to local67. Candidates must pass its five gates and exceed its training CAGR by more than1e-12. Candidate CAGR differences within1e-12 tie, then higher training Sharpe and lower id decide. All four diagnostics freeze before later34. Walk-forward36/60/76 uses the same gates and selection rule.',
        'Name | Full CAGR | Sharpe | MaxDD | First88 CAGR | Later34 CAGR | Nextopen0 CAGR | Nextopen5bp/side CAGR']
    for name,row in results.items():
        metric=row['metrics'];slices=row['slices']
        lines.append(f"{name} | {metric['cagr']:.2%} | {metric['sharpe']:.3f} | {metric['max_drawdown']:.2%} | {slices['training2016_2023']['cagr']:.2%} | {slices['later34']['cagr']:.2%} | {row['next_open_zero']['metrics']['cagr']:.2%} | {row['next_open_cost']['metrics']['cagr']:.2%}")
    lines.append('')
    for name in ('calendar_mean_half','calendar_mean_full'):
        row=frozen['selected'][name]
        lines.append(f"{name}: first88 original65 gate={row['original65_admitted']}; local67 gate={row['local67_admitted']}.")
    lines += [f"Primary: {frozen['selected']['primary']['spec']}; target70 reached: {validation['primary_target70_reached']}.",
        '', 'Descriptive attribution, not another selection rule:']
    for name,row in attribution.items():
        lines.append(f"{name}: holding-set changes={row['changed_months']}, ordered changes={row['ordered_changed_months']}; entering/leaving seats={row['entering_seats']}/{row['leaving_seats']}; split changes={row['split_counts']}; undo-best1/3 CAGR={row['undo_best1_diagnostic_cagr']:.2%}/{row['undo_best3_diagnostic_cagr']:.2%}.")
    lines += ['',
        'Long-only EW3/2half/<2cash with fixed adjusted units within each month. Same-final-close signal/fill is an idealization. Identical orders are replayed at next open with0 and5bp per side, with no forced final sale. Missing held future prices fail validation instead of retrospectively deleting selections.',
        'Repeatedly researched history is not pristine out-of-sample. Proxy rankings, assumedETF17, historical sector mappings and vendor corporate-action adjustments remain. Research gates do not guarantee future returns or establish exact author replication. No observer changes or automatic promotion.',
        f"Protected files checked by evaluate: {validation['protected_files']}; changes={validation['protected_changes']}."]
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

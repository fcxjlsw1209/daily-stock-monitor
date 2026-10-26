"""Fixed cross-sectional maps of the original calm/noncalm long core.

The raw definitions, market-state flag and caller's current pool are retained.
Only normalization changes; these bounded outputs MUST NOT be percentile-ranked
again. The caller preserves its eligibility, 50/52 percent core weight, support
and selector, and retains the original core where an output is NaN.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


FEATURE_NAMES = ("z_tanh", "robust_tanh", "signed_square")
SKIP = 42
VOL_WINDOW = 63
MAD_SCALE = 1.4826
TANH_SCALE = 1.5
_ORIGINAL = Path(__file__).resolve().parents[1] / "monthly_stability_monitor"
SOURCE_PATHS = (_ORIGINAL / "base_strategy.py", _ORIGINAL / "strategy.py")


def _validate(adj, meta, calm_flags):
    if not isinstance(adj, pd.DataFrame) or not isinstance(adj.index, pd.DatetimeIndex):
        raise TypeError("adj must be a DataFrame indexed by exchange-session dates")
    if (adj.index.hasnans or adj.index.tz is not None
            or not adj.index.equals(adj.index.normalize())
            or not adj.index.is_unique or not adj.index.is_monotonic_increasing
            or not adj.columns.is_unique):
        raise ValueError("Price axes must be unique; dates naive, midnight and increasing")
    if not isinstance(meta, dict) or any(key not in meta for key in ("dates", "symbols")):
        raise ValueError("meta must supply dates and same-date symbol pools")
    dates = [pd.Timestamp(day) for day in meta["dates"]]
    if any(pd.isna(day) or day.tz is not None or day != day.normalize() for day in dates):
        raise ValueError("Signal dates must be nonmissing timezone-naive midnight dates")
    pools = [list(symbols) for symbols in meta["symbols"]]
    if len(dates) != len(pools) or dates != sorted(set(dates)):
        raise ValueError("Signal dates must be unique, increasing and aligned with pools")
    width = len(pools[0]) if pools else 0
    if any(len(pool) != width for pool in pools):
        raise ValueError("All monthly symbol pools must have the same width")
    if any(len(set(pool)) != len(pool)
           or any(not isinstance(symbol, str) or not symbol for symbol in pool) for pool in pools):
        raise ValueError("Each pool must contain unique nonempty symbol strings")
    calm = np.asarray(calm_flags)
    if calm.shape != (len(dates),) or calm.dtype.kind != "b":
        raise ValueError("calm_flags must be an exact boolean vector aligned with signal dates")
    if any(day not in adj.index for day in dates):
        raise ValueError("Every signal must be an exact supplied price-index session")
    try:
        values = adj.to_numpy(dtype=float, copy=True, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise TypeError("Prices must be numeric or missing") from exc
    values[~np.isfinite(values) | (values <= 0.)] = np.nan
    return pd.DataFrame(values, index=adj.index.copy(), columns=adj.columns.copy()), dates, pools, calm.copy()


def _raw_features(prices, dates, pools, calm):
    """Use the original pandas expressions, adding complete positive windows."""
    returns = prices.pct_change(fill_method=None)
    sigma = returns.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std(ddof=1)
    sigma = sigma.where(np.isfinite(sigma) & sigma.gt(0.))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        momentum = prices.shift(SKIP) / prices.shift(252) - 1.
        risk = (prices.shift(SKIP) / prices.shift(294) - 1.) / sigma
    width = len(pools[0]) if pools else 0
    raw = np.full((len(dates), width), np.nan)
    records = []
    for i, (day, symbols) in enumerate(zip(dates, pools)):
        t = prices.index.get_loc(day)
        lookback = 252 if calm[i] else 294
        start, end = t - lookback, t - SKIP
        present = np.asarray([symbol in prices.columns for symbol in symbols], dtype=bool)
        record = dict(signal_date=str(day.date()), calm=bool(calm[i]),
            raw_definition="raw252" if calm[i] else "risk294_recent63",
            formation_start=str(prices.index[start].date()) if start >= 0 else None,
            formation_end=str(prices.index[end].date()) if end >= 0 else None,
            required_formation_prices=lookback - SKIP + 1,
            volatility_first_price=(str(prices.index[t - VOL_WINDOW].date())
                if not calm[i] and t >= VOL_WINDOW else None),
            volatility_first_return=(str(prices.index[t - VOL_WINDOW + 1].date())
                if not calm[i] and t >= VOL_WINDOW else None),
            volatility_last_return=str(day.date()) if not calm[i] else None,
            required_volatility_returns=0 if calm[i] else VOL_WINDOW,
            unknown_symbol_cells=int((~present).sum()),
            insufficient_history_cells=int(present.sum()) if start < 0 else 0,
            invalid_formation_cells=0, invalid_recent_price_cells=0,
            invalid_volatility_cells=0, nonfinite_raw_cells=0)
        if start >= 0:
            formation = prices.iloc[start:end + 1].reindex(columns=symbols)
            formation_good = formation.notna().all(axis=0).to_numpy()
            good = formation_good.copy()
            record["invalid_formation_cells"] = int((present & ~formation_good).sum())
            if not calm[i]:
                recent = prices.iloc[t - VOL_WINDOW:t + 1].reindex(columns=symbols)
                recent_good = recent.notna().all(axis=0).to_numpy()
                vol_good = sigma.loc[day].reindex(symbols).notna().to_numpy()
                record["invalid_recent_price_cells"] = int((present & ~recent_good).sum())
                record["invalid_volatility_cells"] = int((present & recent_good & ~vol_good).sum())
                good &= recent_good & vol_good
            frame = momentum if calm[i] else risk
            values = frame.loc[day].reindex(symbols).to_numpy(dtype=float)
            record["nonfinite_raw_cells"] = int((good & ~np.isfinite(values)).sum())
            good &= np.isfinite(values)
            raw[i, good] = values[good]
        records.append(record)
    return raw, records


def _transform(values):
    """Normalize finite current-pool values only; missing never becomes neutral."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Expected one cross-sectional raw vector")
    out = {name: np.full(values.shape, np.nan) for name in FEATURE_NAMES}
    finite = np.isfinite(values)
    n = int(finite.sum())
    audit = dict(finite_raw_cells=n, insufficient_cross_section=n < 2,
                 constant_cross_section=False, robust_std_fallback=False)
    if n < 2:
        return out, audit
    x = values[finite]
    # A common positive rescaling cancels from both z formulas and prevents
    # overflow when finite raw magnitudes approach the floating-point limit.
    magnitude = float(np.max(np.abs(x)))
    scaled = x / magnitude if magnitude > 0. else x.copy()
    constant = bool((x == x[0]).all())
    audit["constant_cross_section"] = constant
    std = 0. if constant else float(scaled.std(ddof=0))
    median = float(np.median(scaled))
    mad = float(np.median(np.abs(scaled - median)))
    if std > 0.:
        out["z_tanh"][finite] = .5 * np.tanh((scaled - scaled.mean()) / (TANH_SCALE * std))
    else:
        out["z_tanh"][finite] = 0.
    robust_scale = MAD_SCALE * mad
    if robust_scale == 0.:
        robust_scale = std
        audit["robust_std_fallback"] = True
    if robust_scale > 0.:
        out["robust_tanh"][finite] = .5 * np.tanh((scaled - median) / (TANH_SCALE * robust_scale))
    else:
        out["robust_tanh"][finite] = 0.
    q = pd.Series(x).rank(method="average", pct=True).to_numpy() - .5
    out["signed_square"][finite] = 2. * q * np.abs(q)
    return out, audit


def build_features(adj, meta, calm_flags):
    """Return ({z_tanh, robust_tanh, signed_square}, audit), each [months,pool].

    calm raw: P[t-42]/P[t-252]-1, with211 positive finite formation prices.
    noncalm raw: (P[t-42]/P[t-294]-1)/STD(last63 simple returns through t,ddof1),
    with253 positive finite formation prices and64 positive recent prices.
    The noncalm denominator deliberately includes the recent42 observations.

    z_tanh = .5*tanh((x-mean(x))/(1.5*populationSTD(x))).
    robust_tanh = .5*tanh((x-median(x))/(1.5*1.4826*medianAbsDeviation(x)));
    zero MAD falls back to populationSTD, still centered on the median.
    Zero cross-sectional dispersion gives0 for both z transforms.
    signed_square = 2*q*abs(q), q=average percentile rank(x)-.5. Exact average
    ranks are retained even for constant pools, yielding1/(2*n*n), not0.

    All finite names in the current supplied pool enter every map, regardless
    of trading eligibility. Fewer than2 finite raw names gives allNaN. Missing
    raw remainsNaN. No imputation, new gate, market-state inference or reranking
    occurs. The caller alone handles trading and missing-core fallback.
    """
    prices, dates, pools, calm = _validate(adj, meta, calm_flags)
    raw, records = _raw_features(prices, dates, pools, calm)
    features = {name: np.full(raw.shape, np.nan) for name in FEATURE_NAMES}
    for i, record in enumerate(records):
        mapped, details = _transform(raw[i])
        for name in FEATURE_NAMES:
            features[name][i] = mapped[name]
        record.update(details)
        record["finite_output_cells"] = int(np.isfinite(mapped["z_tanh"]).sum())
    source = Path(__file__).resolve()
    paths = (source,) + SOURCE_PATHS
    audit = dict(feature_names=list(FEATURE_NAMES), signals=len(dates),
        symbols_per_signal=raw.shape[1],
        raw_formulas=dict(calm="P[t-42]/P[t-252]-1",
            noncalm="(P[t-42]/P[t-294]-1)/STD(simple_returns[t-62:t],ddof1)"),
        transforms=dict(z_tanh=".5*tanh((x-mean)/(1.5*populationSTD))",
            robust_tanh=".5*tanh((x-median)/(1.5*1.4826*medianAbsDeviation)); MAD0 uses populationSTD",
            signed_square="2*q*abs(q); q=average_pct_rank(x)-.5"),
        raw_std_ddof=1, cross_section_std_ddof=0, skip_sessions=SKIP,
        tanh_scale=TANH_SCALE, mad_normalization=MAD_SCALE,
        cross_section="All finite values in the current supplied pool, before eligibility; no historical or future pooled normalization",
        numeric_policy="Common positive rescaling before z moments avoids overflow and cancels algebraically; no fitted scales, epsilon denominator or clipping raw values",
        missing_policy="Incomplete/nonpositive/nonfinite required prices, zero risk STD or nonfinite raw => NaN; fewer than2 finite raw names => allNaN; caller keeps its old core",
        constant_policy="z_tanh and robust_tanh give0; signed_square retains exact average pct rank and gives1/(2*n*n)",
        clock="Supplied session observations only; original caller-provided calm flag; noncalm volatility includes current63 returns",
        interpretation="Only the original raw-core cross-sectional mapping changes; no formation-window alignment, annualization or momentum-definition change",
        rank_after_transform=False, trading_eligibility_modified=False,
        auxiliary_or_market_state_modified=False,
        finite_feature_cells={name: int(np.isfinite(value).sum()) for name, value in features.items()},
        missing_feature_cells={name: int(np.isnan(value).sum()) for name, value in features.items()},
        constant_months=sum(record["constant_cross_section"] for record in records),
        robust_fallback_months=sum(record["robust_std_fallback"] for record in records),
        insufficient_cross_section_months=sum(record["insufficient_cross_section"] for record in records),
        records=records,
        source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        original_formula_references=[str(path) for path in SOURCE_PATHS],
        dependency_versions={"numpy": np.__version__, "pandas": pd.__version__})
    return features, audit

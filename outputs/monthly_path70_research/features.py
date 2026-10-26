"""Four fixed skip-42 path hypotheses; no ranking, selection or return fitting.

At session t the formation prices are t-252 through t-42, inclusive. Their
210 daily returns form three disjoint 70-return segments. The latest 42
sessions are not inspected, even for availability. Caller-owned eligibility
and same-date cross-sectional ranking remain independent of this module.
"""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


LOOKBACK = 252
SKIP = 42
SEGMENT_RETURNS = 70
FORMATION_RETURNS = 210
FEATURE_NAMES = (
    "weakest_segment",
    "balanced_growth",
    "worst_segment_downside",
    "centered_mad_momentum",
)


def _validate(adj):
    if not isinstance(adj, pd.DataFrame):
        raise TypeError("adj must be a pandas DataFrame")
    if not isinstance(adj.index, pd.DatetimeIndex):
        raise TypeError("Price index must be a DatetimeIndex of exchange sessions")
    if adj.index.hasnans or not adj.index.is_unique or not adj.index.is_monotonic_increasing:
        raise ValueError("Price sessions must be nonmissing, unique and increasing")
    if not adj.columns.is_unique:
        raise ValueError("Security columns must be unique")
    try:
        return adj.to_numpy(dtype=float, copy=True, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise TypeError("Prices must be numeric or missing") from exc


def build_features(adj):
    """Return raw feature DataFrames and an audit, preserving index and columns.

    Write g_j = sum of each segment's 70 daily log returns, G = sum(g_j),
    and D_j = sqrt(mean(min(daily log return, 0)**2)) over ALL 70 days.
    The four fixed scores, with larger values ranked higher by the caller, are:

    * min(g_j);
    * mean(g_j) - population_std(g_j), with fixed coefficient one;
    * G / max(D_j);
    * (P[t-42]/P[t-252] - 1) / mean(abs(r - mean(r))), where r comprises
      the 210 simple daily returns from exactly the same formation period.

    All 211 formation prices must be finite and strictly positive. Missing
    prices, insufficient history, zero risk denominators and nonfinite
    computations stay NaN; there is no filling, epsilon denominator or gate on
    the sign of G. Data after t-42 cannot affect values OR availability at t.
    The caller must preserve its existing eligibility and make a missing
    feature an exact no-change correction, not rank missing values as zero.
    """
    prices = _validate(adj)
    rows, columns = prices.shape
    values = {name: np.full(prices.shape, np.nan) for name in FEATURE_NAMES}
    counts = Counter(insufficient_history_cells=min(LOOKBACK, rows) * columns,
                     invalid_formation_price_cells=0, complete_formation_cells=0,
                     zero_downside_denominator_cells=0, zero_mad_denominator_cells=0)
    nonfinite = Counter({name: 0 for name in FEATURE_NAMES})

    for t in range(LOOKBACK, rows):
        formation = prices[t - LOOKBACK:t - SKIP + 1]
        complete = (np.isfinite(formation) & (formation > 0.)).all(axis=0)
        counts["invalid_formation_price_cells"] += int((~complete).sum())
        counts["complete_formation_cells"] += int(complete.sum())
        if not complete.any():
            continue
        positions = np.flatnonzero(complete)
        p = formation[:, complete]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            log_returns = np.diff(np.log(p), axis=0)
            segments = log_returns.reshape(3, SEGMENT_RETURNS, len(positions))
            growth = segments.sum(axis=1)
            total_growth = growth.sum(axis=0)
            downside = np.sqrt(np.mean(np.minimum(segments, 0.) ** 2, axis=1))
            worst_downside = downside.max(axis=0)
            simple_returns = p[1:] / p[:-1] - 1.
            centered = simple_returns - simple_returns.mean(axis=0)
            mad = np.abs(centered).mean(axis=0)
            total_simple = p[-1] / p[0] - 1.
            raw = {
                "weakest_segment": growth.min(axis=0),
                "balanced_growth": growth.mean(axis=0) - growth.std(axis=0, ddof=0),
                "worst_segment_downside": np.divide(
                    total_growth, worst_downside,
                    out=np.full(len(positions), np.nan), where=worst_downside > 0.),
                "centered_mad_momentum": np.divide(
                    total_simple, mad,
                    out=np.full(len(positions), np.nan), where=mad > 0.),
            }
        counts["zero_downside_denominator_cells"] += int((worst_downside == 0.).sum())
        counts["zero_mad_denominator_cells"] += int((mad == 0.).sum())
        for name, score in raw.items():
            finite = np.isfinite(score)
            nonfinite[name] += int((~finite).sum())
            values[name][t, positions[finite]] = score[finite]

    frames = {name: pd.DataFrame(value, index=adj.index.copy(), columns=adj.columns.copy())
              for name, value in values.items()}
    source = Path(__file__).resolve()
    audit = {
        "feature_names": list(FEATURE_NAMES), "sessions": rows, "symbols": columns,
        "price_basis": "raw supplied adjusted close; no price filling or resampling",
        "first_price_offset": -LOOKBACK, "last_price_offset": -SKIP,
        "required_positive_finite_formation_prices": FORMATION_RETURNS + 1,
        "first_computable_row_position": LOOKBACK,
        "segment_price_offsets_inclusive": [[-252, -182], [-182, -112], [-112, -42]],
        "segment_return_counts": [70, 70, 70], "shared_boundary_returns": 0,
        "skipped_recent_sessions_used_for_values_or_availability": False,
        "formulas": {
            "weakest_segment": "min(g1,g2,g3); gj=sum(70 segment daily log returns)",
            "balanced_growth": "mean(g1,g2,g3)-population_std(g1,g2,g3); coefficient=1",
            "worst_segment_downside": "sum(gj)/max(Dj); Dj=sqrt(mean(min(logreturn,0)^2)) over all 70 days",
            "centered_mad_momentum": "(P[t-42]/P[t-252]-1)/mean(abs(r-mean(r))); 210 simple daily returns",
        },
        "zero_denominator_policy": "NaN; no epsilon, infinity or zero-score substitution",
        "nonpositive_growth_policy": "Keep any finite score; no additional sign gate",
        "missing_policy": "NaN; caller applies exact no-change correction",
        "trading_eligibility_modified": False, "cross_sectional_ranking_applied": False,
        "ranking_contract": "Caller ranks finite values within the same month's supplied 49-symbol row: rank(pct=True)-0.5",
        "coverage": dict(counts), "nonfinite_formula_cells": dict(nonfinite),
        "finite_feature_cells": {name: int(np.isfinite(value).sum()) for name, value in values.items()},
        "dependency_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "source_hashes": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    return frames, audit

"""Causal monthly three-holding utility forecasts, isolated research only.

Enumerate all triples of a signal-time baseline's first eight eligible names.
Features and the resulting candidate set never depend on forward price validity.
The two regression arms share the joint-feature availability mask. Neither
author choices nor identifiers enter a model, and no portfolio is traded here.
"""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

_HELPER = Path(__file__).resolve().parents[1] / "monthly_opportunity_alignment/portfolio_model.py"
_SPEC = importlib.util.spec_from_file_location("_monthly_trio_ridge_numeric", _HELPER)
_NUM = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_NUM)

TOP_N = 8
RISK_RETURNS = 126
ASSET_FEATURES = 10
JOINT_FEATURES = ["mean_pair_correlation126", "mean_pair_downside_cosine126",
                  "individual_volatility_dispersion126"]
ALPHA = 10.0
HALF_LIFE_MONTHS = 24.0
MIN_MONTHS = 18


def _bool(values, n, name):
    result = np.asarray(values)
    if result.shape != (n,) or result.dtype.kind != "b":
        raise ValueError("%s must be a length-N boolean array" % name)
    return result


def _day(value, name):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tz is not None or result != result.normalize():
        raise ValueError("%s must be a valid timezone-naive session date" % name)
    return result


def _panel(adj):
    if not isinstance(adj, pd.DataFrame) or not isinstance(adj.index, pd.DatetimeIndex):
        raise ValueError("adj must be a DataFrame with a DatetimeIndex")
    if (adj.index.tz is not None or not adj.index.equals(adj.index.normalize())
            or not adj.index.is_unique or not adj.index.is_monotonic_increasing
            or not adj.columns.is_unique):
        raise ValueError("adj must have unique sorted naive session dates and unique securities")


def _risk_features(adj, date, symbols, indices):
    """Each row is a triple; no filling of missing historical observations."""
    out = np.full((len(indices), len(JOINT_FEATURES)), np.nan)
    if not len(indices):
        return out
    required = sorted(set(int(i) for row in indices for i in row))
    if date not in adj.index or any(symbols[i] not in adj.columns for i in required):
        return out
    past = adj.loc[:date, [symbols[i] for i in required]].tail(RISK_RETURNS + 1)
    if len(past) != RISK_RETURNS + 1:
        return out
    try:
        prices = past.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return out
    valid = np.isfinite(prices).all(axis=0) & (prices > 0).all(axis=0)
    returns = np.full((RISK_RETURNS, len(required)), np.nan)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        returns[:, valid] = prices[1:, valid] / prices[:-1, valid] - 1.0
    valid &= np.isfinite(returns).all(axis=0)
    lookup = {original: local for local, original in enumerate(required)}
    for k, row in enumerate(indices):
        local = np.array([lookup[int(i)] for i in row])
        if not valid[local].all():
            continue
        r = returns[:, local]
        centered = r - r.mean(axis=0)
        var = np.mean(centered * centered, axis=0)
        # No fabricated correlation for zero-variance historical assets.
        if (var <= 0).any() or not np.isfinite(var).all():
            continue
        negative = np.minimum(r, 0.0)
        down_rms = np.sqrt(np.mean(negative * negative, axis=0))
        correlations, downside = [], []
        for i, j in itertools.combinations(range(3), 2):
            correlations.append(np.mean(centered[:, i] * centered[:, j]) / np.sqrt(var[i] * var[j]))
            denom = down_rms[i] * down_rms[j]
            # With no negative observations for either member, there was no
            # shared downside magnitude; this well-defined special case is 0.
            downside.append(float(np.mean(negative[:, i] * negative[:, j]) / denom) if denom > 0 else 0.0)
        vol = r.std(axis=0, ddof=1)
        out[k] = [np.clip(np.mean(correlations), -1.0, 1.0),
                  np.clip(np.mean(downside), 0.0, 1.0), float(vol.std(ddof=0))]
    return out


def _month_labels(adj, date, end, symbols, eligible, indices):
    y = np.full(len(indices), np.nan)
    audit = dict(status="not_requested", full_eligible_count=int(eligible.sum()), issues=[])
    if end is None:
        return y, audit
    end = _day(end, "end")
    if end <= date or end.to_period("M") != date.to_period("M") + 1:
        raise ValueError("label end must be in the next calendar month")
    if not len(indices):
        audit["status"] = "fewer_than_three_candidates"
        return y, audit
    selected = list(np.asarray(symbols)[eligible])
    columns = selected + (["SPY"] if "SPY" not in selected else [])
    missing = [str(s) for s in columns if s not in adj.columns]
    missing_dates = [str(d.date()) for d in (date, end) if d not in adj.index]
    if missing:
        audit["issues"].append(dict(kind="missing_price_columns", symbols=missing))
    if missing_dates:
        audit["issues"].append(dict(kind="missing_exact_dates", dates=missing_dates))
    if not audit["issues"]:
        path = adj.loc[date:end, columns]
        try:
            prices = path.to_numpy(dtype=float)
        except (TypeError, ValueError):
            audit["issues"].append(dict(kind="nonnumeric_prices"))
        else:
            valid = np.isfinite(prices) & (prices > 0)
            if not valid.all():
                bad = np.flatnonzero(~valid.all(axis=0))
                audit["issues"].append(dict(kind="incomplete_positive_price_path",
                    first_invalid_session_by_symbol={str(columns[i]):
                        str(path.index[np.flatnonzero(~valid[:, i])[0]].date()) for i in bad}))
            else:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    payoffs = prices[-1] / prices[0] - 1.0
                if not np.isfinite(payoffs).all():
                    audit["issues"].append(dict(kind="nonfinite_payoffs"))
                else:
                    lookup = dict(zip(columns, payoffs))
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        portfolio = np.array([np.mean([lookup[symbols[i]] for i in row]) for row in indices])
                        logs = np.log1p(portfolio)
                    if not np.isfinite(logs).all():
                        audit["issues"].append(dict(kind="unsupported_portfolio_log_return"))
                    else:
                        y[:] = logs - logs[0]
                        y[0] = 0.0
                        audit.update(status="complete", baseline_simple_return=float(portfolio[0]),
                                     baseline_log_return=float(logs[0]))
    if audit["issues"]:
        audit["status"] = "whole_month_excluded"
    return y, audit


def build_month(adj, date, end, symbols, eligible, base_scores, plain_features):
    """Build a single month without mixing forward labels into candidate choice.

    Fixed first-eight eligible baseline names, using ascending security symbol
    for exact score ties (the existing baseline's rule), produce all C(n,3)
    triples. Each triple is in baseline-rank
    order, with original top3 always row0. Fewer than 3 names returns no triples;
    the caller retains its unmodified two-name/cash behavior.

    X_additive[C,10] is mean plain-feature difference versus row0. X_joint[C,13]
    additionally includes three 126-return historical risk-summary differences.
    Exactly127 positive finite closes through signal are needed for risk inputs.
    ``event_good`` is joint-feature availability, NOT a trading-universe mask.
    Every enumerated triple remains in ``indices`` regardless of availability.
    Both model arms must use the returned common event_good, and train with
    event_good & label_good. End=None builds event features without any labels.
    """
    _panel(adj)
    date = _day(date, "date")
    symbols = np.asarray(symbols)
    if symbols.ndim != 1 or len(set(symbols.tolist())) != len(symbols):
        raise ValueError("symbols must be a unique one-dimensional sequence")
    n = len(symbols)
    eligible = _bool(eligible, n, "eligible")
    scores = np.asarray(base_scores, float)
    plain = np.asarray(plain_features, float)
    if scores.shape != (n,) or plain.shape != (n, ASSET_FEATURES):
        raise ValueError("base_scores and 10-dimensional plain_features must align with symbols")
    if not np.isfinite(scores[eligible]).all():
        raise ValueError("eligible baseline scores must be finite; never silently drop candidates")
    ids = np.flatnonzero(eligible)
    ranked = np.array(sorted(ids.tolist(), key=lambda j: (-scores[j], str(symbols[j])))[:TOP_N], dtype=int)
    indices = np.array(list(itertools.combinations(ranked.tolist(), 3)), dtype=int).reshape(-1, 3)
    additive = np.full((len(indices), ASSET_FEATURES), np.nan)
    joint = np.full((len(indices), ASSET_FEATURES + len(JOINT_FEATURES)), np.nan)
    if len(indices):
        raw_risk = _risk_features(adj, date, symbols, indices)
        # Nonfinite supplied features mean unavailable, not candidate removal.
        # Preserve NaN instead of emitting inf-inf arithmetic warnings.
        clean_plain = np.where(np.isfinite(plain), plain, np.nan)
        with np.errstate(over="ignore", invalid="ignore"):
            raw_additive = np.mean(clean_plain[indices], axis=1)
            additive = raw_additive - raw_additive[0]
            joint = np.column_stack([additive, raw_risk - raw_risk[0]])
    event_good = np.isfinite(joint).all(axis=1)
    y, label_audit = _month_labels(adj, date, end, symbols, eligible, indices)
    label_good = np.isfinite(y)
    audit = dict(signal_date=str(date.date()), label_end=str(_day(end, "end").date()) if end is not None else None,
        eligible_count=int(eligible.sum()), candidate_count=len(ranked), combinations=len(indices),
        top_n=TOP_N, risk_returns=RISK_RETURNS, required_positive_closes=RISK_RETURNS + 1,
        finite_joint_combinations=int(event_good.sum()),
        joint_feature_names=JOINT_FEATURES, centered_on_original_top3=True,
        forecast_masks_do_not_change_candidates=True,
        tie_rule="descending score then ascending symbol; original top3 is first triple and wins exact selection ties",
        label=label_audit)
    return dict(indices=indices, symbols=[[str(symbols[i]) for i in row] for row in indices],
        baseline_indices=ranked[:3].copy() if len(ranked) >= 3 else np.empty(0, dtype=int),
        candidate_indices=ranked, base_score=np.mean(scores[indices], axis=1),
        X_additive=additive, X_joint=joint, event_good=event_good, y=y, label_good=label_good,
        train_good=event_good & label_good, audit=audit)


def annual_forecasts(dates, ends, X, y, train_good, event_dates, event_X, event_good):
    """Return dict(pred, available, audit); one call per matched model arm.

    Inputs are flattened historical/event combinations. Pass the SAME common
    joint-feature train_good and event_good masks to additive and joint calls.
    Only labels completed by the signal year's previous Dec31 are trained, with
    signal dates in the prior5years. No-intercept weighted-RMS Ridge guarantees
    zero forecast for original-top3 features (all0). No output bounding here.
    """
    dates, ends = _NUM._days(dates, "dates"), _NUM._days(ends, "ends")
    event_dates = _NUM._days(event_dates, "event_dates")
    n, q = len(dates), len(event_dates)
    X, event_X, y = np.asarray(X, float), np.asarray(event_X, float), np.asarray(y, float)
    if X.ndim != 2 or X.shape[0] != n or X.shape[1] not in [10, 13]:
        raise ValueError("X must be an N by10 or13 feature matrix")
    if event_X.shape != (q, X.shape[1]) or y.shape != (n,) or ends.shape != (n,):
        raise ValueError("feature, target or date dimensions differ")
    good = _bool(train_good, n, "train_good")
    event_good = _bool(event_good, q, "event_good")
    horizon = ends.astype("datetime64[M]").astype(int) - dates.astype("datetime64[M]").astype(int)
    if (ends <= dates).any() or (horizon != 1).any():
        raise ValueError("labels must end in the next calendar month")
    for month in np.unique(dates.astype("datetime64[M]")):
        rows = dates.astype("datetime64[M]") == month
        if len(np.unique(dates[rows])) != 1 or len(np.unique(ends[rows])) != 1:
            raise ValueError("each month must have one signal and label end")
    if not np.isfinite(X[good]).all() or not np.isfinite(y[good]).all():
        raise ValueError("train_good must use common finite joint features and full-month label availability")
    valid_events = event_good & np.isfinite(event_X).all(axis=1)
    prediction, available = np.zeros(q), np.zeros(q, dtype=bool)
    years = event_dates.astype("datetime64[Y]").astype(int) + 1970
    logs = []
    for year in np.unique(years):
        cutoff = np.datetime64("%04d-12-31" % (year - 1), "D")
        start = _NUM._years_before(cutoff)
        train = good & (ends <= cutoff) & (dates > start)
        events = valid_events & (years == year)
        count = _NUM._month_count(dates[train])
        record = dict(year=int(year), cutoff=str(cutoff), lookback_exclusive_start=str(start),
            training_months=count, training_combinations=int(train.sum()),
            latest_label_end=str(ends[train].max()) if train.any() else None,
            first_training_signal=str(dates[train].min()) if train.any() else None,
            forecast_combinations=int(events.sum()), status="cold_start" if count < MIN_MONTHS else "fit")
        if count >= MIN_MONTHS and events.any():
            weights = _NUM.month_balanced_weights(dates[train], cutoff, HALF_LIFE_MONTHS)
            model = _NUM._fit_rms_ridge(X[train], y[train], weights)
            prediction[events] = _NUM._predict(model, event_X[events])
            available[events] = True
            record["training_feature_rms"] = model[0].tolist()
            record["weight_sum"] = float(weights.sum())
        logs.append(record)
    return dict(pred=prediction, available=available, audit=dict(
        model="annual zero-intercept RMS-scaled Ridge on baseline-centered trio features",
        target="log1p(mean three next-calendar-month simple returns) minus original top3 log1p(mean returns)",
        alpha=ALPHA, half_life_months=HALF_LIFE_MONTHS, min_months=MIN_MONTHS, lookback_years=5,
        intercept=False, training_only_scale=True, feature_count=int(X.shape[1]),
        weighting="equal total per signal month before 24month decay; normalize sum to training rows",
        common_joint_masks_required=True, baseline_zero_feature_prediction_exact_zero=True,
        target_availability_never_filters_candidates=True, author_or_ticker_features=False,
        numeric_dependencies=[str(_HELPER), str(_NUM._PREVIOUS)], annual=logs))

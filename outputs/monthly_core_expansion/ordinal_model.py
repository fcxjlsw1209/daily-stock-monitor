"""Matched monthly return-magnitude and ordinal Ridge learning targets.

Labels use the universe known on each signal date. A missing eligible payoff
invalidates its whole month for BOTH targets; no reranking a future-valid subset.
Outputs are research forecasts only; trading eligibility is supplied separately.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_FILE = Path(__file__).resolve().parents[1] / "monthly_architecture_restart/residual_model.py"
_SPEC = importlib.util.spec_from_file_location("_monthly_ordinal_numeric", _FILE)
_NUM = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_NUM)

ALPHA = 10.0
HALF_LIFE_MONTHS = 24.0
MIN_MONTHS = 18
FEATURES = 10
ARMS = ("return", "ordinal")


def _bool(values, n, name):
    result = np.asarray(values)
    if result.shape != (n,) or result.dtype.kind != "b":
        raise ValueError("%s must be a length-N boolean array" % name)
    return result


def _timeline(dates, ends):
    dates, ends = _NUM._days(dates, "dates"), _NUM._days(ends, "ends")
    if ends.shape != dates.shape:
        raise ValueError("dates and ends shapes differ")
    horizon = ends.astype("datetime64[M]").astype(int) - dates.astype("datetime64[M]").astype(int)
    if (ends <= dates).any() or (horizon != 1).any():
        raise ValueError("labels must end in the next calendar month")
    months = dates.astype("datetime64[M]")
    for month in np.unique(months):
        rows = months == month
        if len(np.unique(dates[rows])) != 1 or len(np.unique(ends[rows])) != 1:
            raise ValueError("each calendar month must have one signal date and one label end")
    return dates, ends, months


def _ordinal(values):
    """Centered midpoint ranks; exact ties receive equal scores, all ties zero."""
    if len(values) <= 1 or np.all(values == values[0]):
        return np.zeros(len(values))
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    midrank = np.cumsum(counts) - (counts - 1.0) / 2.0
    return 2.0 * (midrank[inverse] - .5) / len(values) - 1.0


def build_targets(adj, dates, ends, symbols, eligible_rows):
    """Return ({'return': N, 'ordinal': N}, JSON-serializable audit).

    ``adj`` is an adjusted-close DataFrame with unique, timezone-naive midnight
    session dates and unique security-ID columns, including SPY. Other inputs
    are flattened. Endpoints are looked up exactly: no asof, filling, or zero
    prices. Every existing exchange-session row from signal through label end
    must have positive finite prices for all eligible securities AND SPY.
    Nonpositive/nonfinite prices and nonfinite payoffs are unsupported.
    The ordinal target uses UNCLIPPED simple returns. The magnitude target is
    clip(stock return - SPY return, -1, 1), then same-month mean-centered.
    Noneligible and whole-month-invalid rows contain NaN in both arms.
    """
    dates, ends, months = _timeline(dates, ends)
    n = len(dates)
    symbols = np.asarray(symbols)
    if symbols.shape != (n,):
        raise ValueError("symbols must be a flattened length-N array")
    eligible = _bool(eligible_rows, n, "eligible_rows")
    if not isinstance(adj, pd.DataFrame) or not isinstance(adj.index, pd.DatetimeIndex):
        raise ValueError("adj must be a DataFrame with a DatetimeIndex")
    if adj.index.tz is not None or not adj.index.equals(adj.index.normalize()):
        raise ValueError("adj index must contain timezone-naive midnight session dates")
    if not adj.index.is_unique or not adj.columns.is_unique:
        raise ValueError("adj index and columns must be unique")
    output = {arm: np.full(n, np.nan) for arm in ARMS}
    audit_rows = []
    for month in np.unique(months):
        rows = np.flatnonzero((months == month) & eligible)
        all_rows = np.flatnonzero(months == month)
        date, end = dates[all_rows[0]], ends[all_rows[0]]
        record = dict(month=str(month), signal_date=str(date), label_end=str(end),
                      eligible_rows=int(len(rows)), status="no_eligible_rows", issues=[])
        if not len(rows):
            audit_rows.append(record)
            continue
        ids = list(symbols[rows])
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate eligible security ID within a month")
        columns = ids + (["SPY"] if "SPY" not in ids else [])
        absent = [str(s) for s in columns if s not in adj.columns]
        absent_dates = [str(d) for d in (date, end) if pd.Timestamp(d) not in adj.index]
        if absent:
            record["issues"].append(dict(kind="missing_price_column", symbols=absent))
        if absent_dates:
            record["issues"].append(dict(kind="missing_exact_price_date", dates=absent_dates))
        if not record["issues"]:
            try:
                prices = adj.loc[[pd.Timestamp(date), pd.Timestamp(end)], columns].to_numpy(dtype=float)
                path = adj.loc[(adj.index >= pd.Timestamp(date)) &
                               (adj.index <= pd.Timestamp(end)), columns]
                path_values = path.to_numpy(dtype=float)
            except (ValueError, TypeError):
                record["issues"].append(dict(kind="nonnumeric_price"))
            else:
                valid_cells = np.isfinite(path_values) & (path_values > 0)
                valid = valid_cells.all(axis=0)
                if not valid.all():
                    record["issues"].append(dict(kind="nonpositive_or_nonfinite_price",
                        symbols=[str(columns[i]) for i in np.flatnonzero(~valid)],
                        first_invalid_session_by_symbol={str(columns[i]):
                            str(path.index[np.flatnonzero(~valid_cells[:, i])[0]].date())
                            for i in np.flatnonzero(~valid)}))
                else:
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        payoffs = prices[1] / prices[0] - 1.0
                    if not np.isfinite(payoffs).all():
                        record["issues"].append(dict(kind="nonfinite_simple_payoff"))
                    else:
                        r = payoffs[:len(ids)]
                        spy = payoffs[columns.index("SPY")]
                        excess = np.clip(r - spy, -1.0, 1.0)
                        output["return"][rows] = excess - excess.mean()
                        output["ordinal"][rows] = _ordinal(r)
                        record.update(status="complete", unclipped_payoff_min=float(r.min()),
                            unclipped_payoff_max=float(r.max()),
                            magnitude_clipped_rows=int(np.sum(np.abs(r - spy) > 1.0)),
                            distinct_unclipped_payoffs=int(len(np.unique(r))))
        if record["issues"]:
            record["status"] = "whole_month_excluded"
        audit_rows.append(record)
    audit = dict(policy="complete signal-time eligible universe or exclude entire training month",
        price_path="positive finite prices on every existing exchange-session row from signal through label end, including SPY",
        magnitude_target="clip(simple_return - SPY_simple_return, -1, 1), then month-center",
        ordinal_target="2*(unclipped_simple_return_midrank - .5)/N - 1",
        eligible_rows=int(eligible.sum()), matched_valid_rows=int(np.isfinite(output["return"]).sum()),
        complete_months=sum(r["status"] == "complete" for r in audit_rows),
        dropped_months=sum(r["status"] == "whole_month_excluded" for r in audit_rows),
        months=audit_rows, trading_eligibility_not_modified=True)
    return output, audit


def annual_forecasts(dates, ends, X, targets, train_good, event_dates, event_X, event_eligible):
    """Return ({'return': Q, 'ordinal': Q, 'available': bool Q}, audit).

    Fixed alpha10, 24-month half-life, prior-five-year window and prior-Dec31
    matured labels. Both arms use identical complete training months, features,
    scaler fitting and weights. Cold starts/invalid events return zero and
    available=False. A fitted neutral zero remains available=True. No correction
    bounding or event cross-sectional normalization is performed here.
    """
    dates, ends, months = _timeline(dates, ends)
    event_dates = _NUM._days(event_dates, "event_dates")
    n, q = len(dates), len(event_dates)
    X, event_X = np.asarray(X, float), np.asarray(event_X, float)
    if X.shape != (n, FEATURES) or event_X.shape != (q, FEATURES):
        raise ValueError("X and event_X must have their respective row counts and 10 features")
    good = _bool(train_good, n, "train_good")
    event_good = _bool(event_eligible, q, "event_eligible")
    if not isinstance(targets, dict) or any(k not in targets for k in ARMS):
        raise ValueError("targets must contain return and ordinal arrays")
    y = {k: np.asarray(targets[k], float) for k in ARMS}
    if any(v.shape != (n,) for v in y.values()):
        raise ValueError("target row counts differ")
    if (np.abs(y["ordinal"][np.isfinite(y["ordinal"])]) > 1.0 + 1e-12).any():
        raise ValueError("ordinal target must be within [-1,1]")
    if (np.abs(y["return"][np.isfinite(y["return"])]) > 2.0 + 1e-12).any():
        raise ValueError("centered clipped return target must be within [-2,2]")
    finite_targets = np.isfinite(y["return"]) & np.isfinite(y["ordinal"])
    expected = good | np.isfinite(y["return"]) | np.isfinite(y["ordinal"])
    usable = np.zeros(n, dtype=bool)
    excluded = []
    for month in np.unique(months):
        rows = (months == month) & expected
        if rows.any() and (good[rows] & finite_targets[rows] & np.isfinite(X[rows]).all(axis=1)).all():
            usable[rows] = True
        elif rows.any():
            excluded.append(str(month))
    valid_events = event_good & np.isfinite(event_X).all(axis=1)
    years = event_dates.astype("datetime64[Y]").astype(int) + 1970
    output = {arm: np.zeros(q) for arm in ARMS}
    output["available"] = np.zeros(q, dtype=bool)
    logs = []
    for year in np.unique(years):
        cutoff = np.datetime64("%04d-12-31" % (year - 1), "D")
        start = _NUM._years_before(cutoff)
        train = usable & (ends <= cutoff) & (dates > start)
        events = valid_events & (years == year)
        count = _NUM._month_count(dates[train])
        row = dict(year=int(year), cutoff=str(cutoff), lookback_exclusive_start=str(start),
            training_rows=int(train.sum()), training_months=int(count),
            latest_label_end=str(ends[train].max()) if train.any() else None,
            first_signal_date=str(dates[train].min()) if train.any() else None,
            prediction_rows=int(events.sum()),
            status="cold_start" if count < MIN_MONTHS else "fit" if events.any() else "no_valid_events")
        if count >= MIN_MONTHS and events.any():
            weights = _NUM.month_balanced_weights(dates[train], cutoff, HALF_LIFE_MONTHS)
            for arm in ARMS:
                model = _NUM._fit_ridge(X[train], y[arm][train], weights, ALPHA)
                output[arm][events] = _NUM._predict(model, event_X[events])
                if arm == "return":
                    row["training_feature_mean"] = model[0].tolist()
                    row["training_feature_scale"] = model[1].tolist()
            output["available"][events] = True
            row["weight_sum"] = float(weights.sum())
        logs.append(row)
    audit = dict(alpha=ALPHA, half_life_months=HALF_LIFE_MONTHS, min_months=MIN_MONTHS,
        features=FEATURES, matched_training_rows=True, prior_dec31_mature_labels_only=True,
        whole_month_exclusions_on_input_validation=excluded, annual=logs,
        unsupported_or_ineligible_event_rows=int((~valid_events).sum()),
        output_is_native_target_prediction=True, author_or_ticker_targets_used=False)
    return output, audit

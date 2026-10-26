"""Fixed annual monthly-return models with causal stock/ETF partial pooling.

No hyperparameter search, ticker effects, author labels, or trading mutations.
State interactions, next-calendar-month labels and feature availability belong
in the caller. Prediction is in native return units; this module does not rank
or magnify predictions.
"""
import importlib.util
from pathlib import Path

import numpy as np

_PREVIOUS = Path(__file__).resolve().parents[1] / "monthly_architecture_restart" / "residual_model.py"
_spec = importlib.util.spec_from_file_location("monthly_session_proven_numerics", _PREVIOUS)
_numeric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_numeric)
_days = _numeric._days
_years_before = _numeric._years_before
_month_count = _numeric._month_count
_require_finite = _numeric._require_finite
_fit_ridge = _numeric._fit_ridge
_predict = _numeric._predict
month_balanced_weights = _numeric.month_balanced_weights


def _solve(design, target, weights, penalties):
    for name, values in (("design", design), ("target", target),
                         ("weights", weights), ("penalties", penalties)):
        _require_finite(name, values)
    if (weights < 0).any() or weights.sum() <= 0 or (penalties < 0).any():
        raise ValueError("invalid Ridge weights or penalties")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        lhs = np.einsum("ni,n,nj->ij", design, weights, design, optimize=False)
        lhs += np.diag(penalties)
        rhs = np.einsum("ni,n,n->i", design, weights, target, optimize=False)
    _require_finite("normal-equation lhs", lhs)
    _require_finite("normal-equation rhs", rhs)
    try:
        coefficient = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    _require_finite("coefficients", coefficient)
    return coefficient


def _standardize(X, weights):
    _require_finite("features", X)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        mean = np.average(X, axis=0, weights=weights)
        scale = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=weights))
        scale = np.where(scale > 1e-12, scale, 1.)
        z = (X - mean) / scale
    _require_finite("standardized features", z)
    return mean, scale, z


def _fit_partial(X, y, asset_type, weights, alpha, deviation_strength):
    """Stock=-1, ETF=+1; common slopes plus strongly penalized deviations.

    Interactions share each base feature's training scale. They are centered,
    but not separately variance-normalized, so the advertised penalty applies
    on the same feature units. There is one common unpenalized intercept and
    no stock/ETF intercept or ticker label.
    """
    mean, scale, z = _standardize(X, weights)
    interaction = z * (2. * asset_type[:, None] - 1.)
    interaction_mean = np.average(interaction, axis=0, weights=weights)
    design = np.column_stack((z, interaction - interaction_mean))
    intercept = float(np.average(y, weights=weights))
    p = X.shape[1]
    penalties = alpha * np.r_[np.ones(p), np.full(p, deviation_strength)]
    coefficient = _solve(design, y - intercept, weights, penalties)
    return dict(mean=mean, scale=scale, interaction_mean=interaction_mean,
                coefficient=coefficient, intercept=intercept)


def _predict_partial(model, X, asset_type):
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        z = (X - model["mean"]) / model["scale"]
        interaction = z * (2. * asset_type[:, None] - 1.) - model["interaction_mean"]
        design = np.column_stack((z, interaction))
        forecast = np.einsum("ni,i->n", design, model["coefficient"], optimize=False)
        forecast += model["intercept"]
    _require_finite("partial forecasts", forecast)
    return forecast


def _pairwise_examples(dates, X, y, cutoff, half_life_months=24., boundary_scores=None,
                       boundary_top3=None):
    """All unordered within-month pairs, including stock-versus-ETF pairs.

    Each calendar month has equal total weight before age decay. Total weight
    is the original observation count, avoiding artificial regularization
    weakening as the number of pairs grows. A provided *historical* baseline
    score gives top-three-boundary pairs fixed 2x relative weight, followed by
    renormalization inside each month. Prefer boundary_top3 flags computed from
    the complete signal-time eligible set BEFORE any future-label validity
    filter: these flags are carried through, never reranked. boundary_scores
    is a legacy fallback that ranks the supplied training subset. The future
    target never sets relative pair weights.
    """
    delta_x, delta_y, row_weights, pair_dates = [], [], [], []
    months = dates.astype("datetime64[M]")
    for month in np.unique(months):
        rows = np.flatnonzero(months == month)
        if len(rows) < 2:
            continue
        ii, jj = np.triu_indices(len(rows), 1)
        left, right = rows[ii], rows[jj]
        weight = np.ones(len(ii))
        if boundary_top3 is not None:
            top = np.asarray(boundary_top3, dtype=bool)[rows]
            weight[np.logical_xor(top[ii], top[jj])] = 2.
        elif boundary_scores is not None:
            # Legacy fallback ranks the available subset; main study uses flags.
            order = np.argsort(-boundary_scores[rows], kind="stable")
            top = np.zeros(len(rows), dtype=bool)
            top[order[:min(3, len(rows))]] = True
            weight[np.logical_xor(top[ii], top[jj])] = 2.
        age = int(np.datetime64(cutoff, "M").astype(int) - month.astype(int))
        age_weight = np.exp2(-age / half_life_months) if half_life_months else 1.
        weight = weight / weight.sum() * age_weight
        delta_x.append(X[left] - X[right])
        delta_y.append(y[left] - y[right])
        row_weights.append(weight)
        pair_dates.extend([month] * len(ii))
    if not delta_x:
        return (np.empty((0, X.shape[1])), np.empty(0), np.empty(0),
                np.empty(0, dtype="datetime64[M]"))
    weights = np.concatenate(row_weights)
    weights *= len(dates) / weights.sum()
    return (np.vstack(delta_x), np.concatenate(delta_y), weights,
            np.asarray(pair_dates, dtype="datetime64[M]"))


def _fit_pairwise(dates, X, y, weights, cutoff, alpha, half_life_months,
                  boundary_scores=None, boundary_top3=None):
    mean, scale, z = _standardize(X, weights)
    pair_x, pair_y, pair_w, pair_dates = _pairwise_examples(
        dates, z, y, cutoff, half_life_months, boundary_scores, boundary_top3)
    if not len(pair_y):
        return None, dict(pair_rows=0, pair_months=0)
    coefficient = _solve(pair_x, pair_y, pair_w, np.full(X.shape[1], alpha))
    # Pair differences identify slopes only; use zero common intercept. The
    # event-month cross section may be centered by the caller if desired.
    model = (mean, scale, coefficient, 0.)
    audit = dict(pair_rows=int(len(pair_y)), pair_months=int(len(np.unique(pair_dates))),
                 total_pair_weight=float(pair_w.sum()), cross_asset_type_pairs_included=True)
    return model, audit


def _type(values, n, name):
    raw = np.asarray(values)
    if raw.shape != (n,) or not np.isin(raw, [0, 1]).all():
        raise ValueError("%s must contain only 0=stock or 1=ETF" % name)
    return raw.astype(int)


def annual_forecasts(dates, ends, X, y, good, asset_type, event_dates, event_X,
                     event_good, event_asset_type, alpha=10., half_life_months=24.,
                     min_months=18, deviation_strengths=(10., 100.), boundary_scores=None,
                     boundary_top3=None):
    """Return pooled/partial_10/partial_100/independent/pairwise arrays and audit.

    ``available`` maps each model key to a boolean array; a fitted numerical
    zero is available, whereas invalid events and cold starts are unavailable.

    One model per signal year, prior Dec 31 cutoff, five-year trailing signal
    window and completed monthly targets only. Independent-type forecasts use
    their own training-only scaler and require min_months for that type; no
    silent pooling fallback. Insufficient models and invalid events return 0.
    """
    dates, ends = _days(dates, "dates"), _days(ends, "ends")
    event_dates = _days(event_dates, "event_dates")
    X, event_X = np.asarray(X, float), np.asarray(event_X, float)
    y, good, event_good = np.asarray(y, float), np.asarray(good, bool), np.asarray(event_good, bool)
    n, q = len(dates), len(event_dates)
    if X.ndim != 2 or X.shape[0] != n or not X.shape[1]:
        raise ValueError("X must be a nonempty-feature training matrix")
    if event_X.shape != (q, X.shape[1]):
        raise ValueError("event_X feature shape differs")
    if ends.shape != (n,) or y.shape != (n,) or good.shape != (n,) or event_good.shape != (q,):
        raise ValueError("date, target or eligibility length differs")
    asset_type = _type(asset_type, n, "asset_type")
    event_asset_type = _type(event_asset_type, q, "event_asset_type")
    if (ends <= dates).any():
        raise ValueError("monthly label end must be later than signal date")
    month_horizon = ends.astype("datetime64[M]").astype(int) - dates.astype("datetime64[M]").astype(int)
    if (month_horizon != 1).any():
        raise ValueError("labels must end in the next calendar month")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not np.isfinite(half_life_months) or half_life_months < 0:
        raise ValueError("half_life_months must be finite and nonnegative")
    if not isinstance(min_months, (int, np.integer)) or min_months < 1:
        raise ValueError("min_months must be a positive integer")
    strengths = tuple(float(v) for v in deviation_strengths)
    if not strengths or any(not np.isfinite(v) or v < 1 for v in strengths) or len(set(strengths)) != len(strengths):
        raise ValueError("deviation strengths must be unique finite values >=1")
    partial_keys = ["partial_%g" % v for v in strengths]
    if boundary_scores is not None:
        boundary_scores = np.asarray(boundary_scores, float)
        if boundary_scores.shape != (n,):
            raise ValueError("boundary_scores must match training observations")
    if boundary_top3 is not None:
        boundary_top3 = np.asarray(boundary_top3)
        if boundary_top3.shape != (n,) or boundary_top3.dtype != np.dtype(bool):
            raise ValueError("boundary_top3 must be a boolean flag for every training observation")
    usable = good & np.isfinite(y) & np.isfinite(X).all(axis=1)
    if boundary_scores is not None and boundary_top3 is None:
        usable &= np.isfinite(boundary_scores)
    valid_events = event_good & np.isfinite(event_X).all(axis=1)
    output = {k: np.zeros(q) for k in ["pooled", "independent", "pairwise"] + partial_keys}
    available = {k: np.zeros(q, dtype=bool) for k in output}
    years = event_dates.astype("datetime64[Y]").astype(int) + 1970
    logs = []
    for year in np.unique(years):
        cutoff = np.datetime64("%04d-12-31" % (year - 1), "D")
        train = usable & (ends <= cutoff) & (dates > _years_before(cutoff))
        events = valid_events & (years == year)
        months = _month_count(dates[train])
        row = dict(year=int(year), cutoff=str(cutoff),
                   lookback_exclusive_start=str(_years_before(cutoff)),
                   training_rows=int(train.sum()), training_months=months,
                   latest_label_end=str(ends[train].max()) if train.any() else None,
                   prediction_rows=int(events.sum()), status="fit" if months >= min_months else "cold_start",
                   independent={}, partial={}, pairwise={"status": "cold_start"})
        if months >= min_months and events.any():
            tx, ty, td, tt = X[train], y[train], dates[train], asset_type[train]
            weights = month_balanced_weights(td, cutoff, half_life_months)
            output["pooled"][events] = _predict(_fit_ridge(tx, ty, weights, alpha), event_X[events])
            available["pooled"][events] = True
            for key, strength in zip(partial_keys, strengths):
                model = _fit_partial(tx, ty, tt, weights, alpha, strength)
                output[key][events] = _predict_partial(model, event_X[events], event_asset_type[events])
                available[key][events] = True
                row["partial"][key] = dict(deviation_penalty_relative=strength,
                    deviation_coefficient_l2=float(np.linalg.norm(model["coefficient"][X.shape[1]:])))
            scores = boundary_scores[train] if boundary_scores is not None else None
            flags = boundary_top3[train] if boundary_top3 is not None else None
            pair_model, pair_audit = _fit_pairwise(td, tx, ty, weights, cutoff, alpha,
                                                  half_life_months, scores, flags)
            pair_audit["status"] = "fit" if pair_audit["pair_months"] >= min_months else "cold_start"
            row["pairwise"] = pair_audit
            if pair_audit["status"] == "fit":
                output["pairwise"][events] = _predict(pair_model, event_X[events])
                available["pairwise"][events] = True
        for kind, label in ((0, "stock"), (1, "etf")):
            subset = train & (asset_type == kind)
            target = events & (event_asset_type == kind)
            count = _month_count(dates[subset])
            row["independent"][label] = dict(training_rows=int(subset.sum()), training_months=count,
                prediction_rows=int(target.sum()), status="fit" if count >= min_months else "cold_start",
                latest_label_end=str(ends[subset].max()) if subset.any() else None)
            if count >= min_months and target.any():
                weights = month_balanced_weights(dates[subset], cutoff, half_life_months)
                model = _fit_ridge(X[subset], y[subset], weights, alpha)
                output["independent"][target] = _predict(model, event_X[target])
                available["independent"][target] = True
        logs.append(row)
    output["available"] = available
    output["audit"] = dict(method="annual_calendar_month_stock_etf_models", alpha=float(alpha),
        half_life_months=float(half_life_months), min_months=int(min_months), lookback_years=5,
        effect_code="stock=-1,ETF=+1; common intercept; common slopes plus type deviations",
        deviation_strengths=list(strengths), scale_fit="past_training_rows_only",
        pairwise="all_unordered_within_month_pairs_including_cross_type;zero_intercept",
        pair_weighting="equal_month_then_decay;sum_weights_equals_original_observation_count",
        boundary_weighting=("fixed_signal_time_top3_flags_crossing_2x_no_rerank" if boundary_top3 is not None else
            "available_training_subset_score_top3_legacy_fallback_2x" if boundary_scores is not None else "none"),
        monthly_targets="caller_supplied_actual_next_calendar_month_return",
        feature_contract="caller_includes_state_interactions;no_ticker_or_author_features",
        cold_start="zero_no_hidden_fallback", linear_algebra="einsum_optimize_false_with_finite_guards",
        annual=logs)
    return output

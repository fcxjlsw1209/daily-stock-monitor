"""Fixed calm-regime transitions and a skip-42 log-acceleration alternative.

Research-only factor preparation: no returns, labels, fitting, optimization,
trading or persistence. The original simple-acceleration hard switch is an
exact control. Both complete branches inherit the frozen adapter's frames,
ranks, weights, state and FOREIGN/sector mapping; only their mixture changes.
"""
from pathlib import Path
import importlib.util
import itertools

import numpy as np
import pandas as pd


P = Path(__file__).resolve().parent
ADAPTER_PATH = P.parent / "monthly_opportunity_alignment" / "baseline_adapter.py"
_spec = importlib.util.spec_from_file_location("target70_transition_baseline", ADAPTER_PATH)
BASELINE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BASELINE)
STRATEGY = BASELINE._STRATEGY
MODES = ("hard", "narrow", "wide")
ACCELERATIONS = ("simple", "log")
BANDS = {"hard": (.10, .10), "narrow": (.09, .11), "wide": (.08, .12)}


def configs():
    """Six fixed rows; hard/simple is the unchanged original65 control."""
    return [dict(id=i, family="transition", mode=mode, acceleration=acceleration)
            for i, (mode, acceleration) in enumerate(itertools.product(MODES, ACCELERATIONS))]


def calm_weight(annualized_volatility, low, high):
    """Calm share, linear inside a fixed band; zero width is the old < rule.

    Missing volatility takes the noncalm branch, matching the frozen test.
    This is a pure fixed transformation, never a fitted regime probability.
    """
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError("Transition endpoints must be finite and increasing")
    vol = np.asarray(annualized_volatility, dtype=float)
    if low == high:
        return (vol < low).astype(float)
    share = np.clip((high - vol) / (high - low), 0., 1.)
    return np.where(np.isnan(vol), 0., share)


def _blend(calm, noncalm, share):
    """Convex mixture with exact endpoint identities, including the control."""
    blended = share * calm + (1. - share) * noncalm
    return np.where(share == 1., calm, np.where(share == 0., noncalm, blended))


def log_acceleration(prices):
    """Two adjacent 84-session log returns, last endpoint exactly t-42.

    Only the three positive finite endpoint prices are needed. Invalid log
    inputs become NaN; prepare restores the original acceleration contribution
    for each unavailable candidate without changing its trading eligibility.
    """
    positive = prices.where(np.isfinite(prices) & prices.gt(0.))
    value = (np.log(positive.shift(42) / positive.shift(126))
             - np.log(positive.shift(126) / positive.shift(210)))
    return value.replace([np.inf, -np.inf], np.nan)


def prepare(panel, events, etfs):
    """Return ``(arrays, audit)`` in supplied event/candidate order.

    Every array is [events,49]. score/core/cw/eligible describe the unchanged
    control. Each of the six ``mode_acceleration`` scores has corresponding
    ``_core`` and ``_cw`` metadata. Branch and mixture arrays make preservation
    of the >=50% core and convexity directly auditable. Missing log factors use
    the original simple rank for that candidate, never an eligibility filter.
    """
    etfs = set(etfs)
    if len(etfs) != 17:
        raise ValueError("Expected 17 distinct ETFs")
    events = [(pd.Timestamp(day), list(symbols)) for day, symbols in events]
    px = panel["adj"]
    if not px.index.is_unique or not px.index.is_monotonic_increasing:
        raise ValueError("Price sessions must be unique and increasing")
    if not px.columns.is_unique:
        raise ValueError("Price columns must be unique")
    for day, symbols in events:
        if len(symbols) != 49 or len(set(symbols)) != 49 or not etfs.issubset(symbols):
            raise ValueError("Each event needs 32 stocks and the 17 distinct ETFs")
        if day not in px.index or not set(symbols).issubset(px.columns):
            raise ValueError("Event date or candidate missing from price panel")
    shape = (len(events), 49)
    float_keys = ["score", "core", "cw", "calm_score", "noncalm_simple", "noncalm_log",
                  "calm_core", "noncalm_core", "annualized_volatility",
                  "acceleration_simple_rank", "acceleration_log_rank"]
    float_keys += ["calm_weight_" + mode for mode in MODES]
    float_keys += [spec["mode"] + "_" + spec["acceleration"] + suffix
                   for spec in configs() for suffix in ("", "_core", "_cw")]
    out = {key: np.empty(shape, dtype=float) for key in float_keys}
    out.update({key: np.empty(shape, dtype=bool) for key in ("eligible", "log_acceleration_valid")})
    audit = dict(
        family="transition", events=len(events), candidate_count=49, grid_size=6,
        bands={mode: list(band) for mode, band in BANDS.items()},
        control="hard_simple: exact frozen adapter score/core/cw/eligible",
        regime_measure="SPY pct_change(fill_method=None), rolling63 std(ddof=1)*sqrt252",
        transition="calm_share=clip((high-vol)/(high-low),0,1); hard uses vol<0.10",
        minimum_core_weight=.5, calm_core_weight=.5, noncalm_core_weight=.52,
        original_acceleration="P[t-42]/P[t-126]-P[t-126]/P[t-210]",
        alternative_acceleration="log(P[t-42]/P[t-126])-log(P[t-126]/P[t-210])",
        acceleration_endpoints=[-210, -126, -42], acceleration_segment_sessions=84,
        log_missing_fallback="original simple acceleration centered rank for that candidate",
        ranks="pandas average percentile among finite values in supplied49, missing neutral0",
        sectors="exact inherited signal-date mapping, including frozen FOREIGN overrides",
        future_labels_used=False, trading_eligibility_modified=False,
        source_paths=[str(ADAPTER_PATH)] + [str(path) for path in BASELINE.SOURCE_PATHS],
        log_valid_count=0, log_fallback_count=0, eligible_log_fallback_count=0,
    )
    if not events:
        return out, audit
    vals, eligible, state, corr, tail, mom, efficiency, calm = STRATEGY.factors(panel)
    vol = px.pct_change(fill_method=None).SPY.rolling(63).std() * np.sqrt(252)
    log_values = log_acceleration(px)
    sectors = panel["sectors"]
    for i, (day, symbols) in enumerate(events):
        table = pd.DataFrame({key: value.loc[day, symbols] for key, value in vals.items()})
        q = table.rank(pct=True).fillna(.5) - .5
        q["regime_cmf"] *= state.at[day]
        # The exact adapter performs its relative-factor ranking before
        # candidate eligibility, and FOREIGN takes precedence over sectors.
        relatives = {126: {}, 252: {}}
        stability = {}
        for symbol in symbols:
            benchmark = (None if symbol in etfs else STRATEGY.FOREIGN.get(
                symbol, STRATEGY.SECTOR.get(sectors.at[day, symbol])
                if symbol in sectors else None))
            for window in relatives:
                if benchmark is None:
                    relatives[window][symbol] = np.nan
                    continue
                value, sector_value = mom[window].at[day, symbol], mom[window].at[day, benchmark]
                relatives[window][symbol] = ((1 + value) / (1 + sector_value) - 1
                    if np.isfinite(value) and np.isfinite(sector_value) and sector_value > -1
                    else np.nan)
            stability[symbol] = (-abs(corr[benchmark][0].at[day, symbol]
                                      - corr[benchmark][1].at[day, symbol])
                                 if benchmark is not None else np.nan)
        extra = pd.DataFrame({
            "sector126": pd.Series(relatives[126]),
            "sector252": pd.Series(relatives[252]),
            "stability": pd.Series(stability),
            "tail": tail.loc[day, symbols],
            "raw_momentum": mom[252].loc[day, symbols],
            "path": efficiency.loc[day, symbols],
        })
        ranks = extra.replace([np.inf, -np.inf], np.nan).rank(pct=True).fillna(.5) - .5
        calm_parts = dict(core=.5 * ranks.raw_momentum, volatility=.2 * q.volatility,
            sector=.0875 * (ranks.sector126 + ranks.sector252),
            path=.05 * ranks.path, cmf=.075 * q.regime_cmf)
        budget = .10 if state.at[day] > 0 else 0.
        industry = budget * (.25 * ranks.sector126 + .5 * ranks.stability + .25 * ranks["tail"])
        log_row = log_values.loc[day, symbols]
        log_valid = np.isfinite(log_row)
        q_log = q.copy()
        q_log["acceleration"] = (log_row.rank(pct=True).fillna(.5) - .5).where(
            log_valid, q.acceleration)
        out["acceleration_simple_rank"][i] = q.acceleration.reindex(symbols).to_numpy()
        out["acceleration_log_rank"][i] = q_log.acceleration.reindex(symbols).to_numpy()
        out["log_acceleration_valid"][i] = log_valid.to_numpy(bool)
        out["calm_score"][i] = sum(calm_parts.values()).reindex(symbols).to_numpy()
        out["calm_core"][i] = calm_parts["core"].reindex(symbols).to_numpy()
        for acceleration, branch_q in (("simple", q), ("log", q_log)):
            support = sum(branch_q[key] * value for key, value in STRATEGY.WEIGHTS.items()
                          if key != "momentum")
            parts = dict(core=.52 * q.momentum, support=(1 - budget / .48) * support,
                         industry=industry)
            out["noncalm_" + acceleration][i] = sum(parts.values()).reindex(symbols).to_numpy()
        out["noncalm_core"][i] = parts["core"].reindex(symbols).to_numpy()
        out["eligible"][i] = eligible.loc[day, symbols].to_numpy(bool)
        out["annualized_volatility"][i] = vol.at[day]
        for mode in MODES:
            share = calm_weight(vol.at[day], *BANDS[mode])
            # Assert the zero-width transformation retains the inherited rule.
            if mode == "hard" and bool(share) != bool(calm.at[day]):
                raise ValueError("Frozen calm rule and hard transition disagree")
            out["calm_weight_" + mode][i] = share
            core = _blend(out["calm_core"][i], out["noncalm_core"][i], share)
            cw = _blend(.5, .52, share)
            for acceleration in ACCELERATIONS:
                key = mode + "_" + acceleration
                out[key][i] = _blend(out["calm_score"][i], out["noncalm_" + acceleration][i], share)
                out[key + "_core"][i] = core
                out[key + "_cw"][i] = cw
    for key, control_key in (("score", "hard_simple"), ("core", "hard_simple_core"),
                              ("cw", "hard_simple_cw")):
        out[key][:] = out[control_key]
    for spec in configs():
        key = spec["mode"] + "_" + spec["acceleration"]
        if not np.isfinite(out[key]).all() or (out[key + "_cw"] < .5).any():
            raise ValueError("Nonfinite score or core below50percent: " + key)
    audit["log_valid_count"] = int(out["log_acceleration_valid"].sum())
    audit["log_fallback_count"] = int((~out["log_acceleration_valid"]).sum())
    audit["eligible_log_fallback_count"] = int((out["eligible"] & ~out["log_acceleration_valid"]).sum())
    return out, audit


def score(spec, arrays, n=None):
    """Read a registered score array; optionally restrict the event prefix."""
    if spec["mode"] not in MODES or spec["acceleration"] not in ACCELERATIONS:
        raise ValueError("Unregistered transition configuration")
    return arrays[spec["mode"] + "_" + spec["acceleration"]][:n]

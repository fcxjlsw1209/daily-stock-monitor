"""Exact frozen 65.34% monthly scores for caller-supplied historical pools.

Only signal-time prices and sectors enter eligibility or scores. The caller
supplies 32 issuer representatives and 17 ETFs; future label availability must
never be used to shorten that list. No observer, membership or source is edited.
"""
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd


_MONITOR = Path(__file__).resolve().parents[1] / "monthly_stability_monitor"
SOURCE_PATHS = (_MONITOR / "base_strategy.py", _MONITOR / "strategy.py")


def _load_strategy():
    """Resolve the frozen sibling import without changing another module's alias."""
    base = ModuleType("monthly_alignment_frozen_base")
    base.__file__ = str(SOURCE_PATHS[0])
    exec(compile(SOURCE_PATHS[0].read_text(), base.__file__, "exec"), base.__dict__)
    strategy = ModuleType("monthly_alignment_frozen_strategy")
    strategy.__file__ = str(SOURCE_PATHS[1])
    previous = sys.modules.get("base_strategy")
    sys.modules["base_strategy"] = base
    try:
        exec(compile(SOURCE_PATHS[1].read_text(), strategy.__file__, "exec"), strategy.__dict__)
    finally:
        if previous is None:
            del sys.modules["base_strategy"]
        else:
            sys.modules["base_strategy"] = previous
    return strategy


_STRATEGY = _load_strategy()


def prepare(panel, events, etfs):
    """Return score, eligible, weighted core and core weight, each [events,49].

    Factor frames are calculated once within this call and are not cached
    globally. Cross-sectional ranks include every supplied candidate before
    eligibility filtering, exactly as in the registered strategy. Event symbol
    order is preserved, including for early training pools absent from member.
    """
    etfs = set(etfs)
    if len(etfs) != 17:
        raise ValueError("Expected 17 distinct ETFs")
    events = [(pd.Timestamp(day), list(symbols)) for day, symbols in events]
    px = panel["adj"]
    if not px.index.is_unique or not px.index.is_monotonic_increasing:
        raise ValueError("Price sessions must be unique and increasing")
    for day, symbols in events:
        if len(symbols) != 49 or len(set(symbols)) != 49 or not etfs.issubset(symbols):
            raise ValueError("Each event needs 32 stocks and the 17 distinct ETFs")
        if day not in px.index or not set(symbols).issubset(px.columns):
            raise ValueError("Event date or candidate missing from price panel")
    shape = (len(events), 49)
    out = {key: np.empty(shape, dtype=bool if key == "eligible" else float)
           for key in ("score", "eligible", "core", "cw")}
    if not events:
        return out
    vals, eligible, state, corr, tail, mom, efficiency, calm = _STRATEGY.factors(panel)
    sectors = panel["sectors"]
    for i, (day, symbols) in enumerate(events):
        table = pd.DataFrame({key: value.loc[day, symbols] for key, value in vals.items()})
        q = table.rank(pct=True).fillna(.5) - .5
        q["regime_cmf"] *= state.at[day]
        relatives = {126: {}, 252: {}}
        stability = {}
        for symbol in symbols:
            benchmark = (None if symbol in etfs else _STRATEGY.FOREIGN.get(
                symbol, _STRATEGY.SECTOR.get(sectors.at[day, symbol])
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
        if calm.at[day]:
            parts = dict(core=.5 * ranks.raw_momentum, volatility=.2 * q.volatility,
                sector=.0875 * (ranks.sector126 + ranks.sector252),
                path=.05 * ranks.path, cmf=.075 * q.regime_cmf)
            weight = .5
        else:
            support = sum(q[key] * value for key, value in _STRATEGY.WEIGHTS.items()
                          if key != "momentum")
            budget = .10 if state.at[day] > 0 else 0.
            parts = dict(core=.52 * q.momentum, support=(1 - budget / .48) * support,
                industry=budget * (.25 * ranks.sector126 + .5 * ranks.stability + .25 * ranks["tail"]))
            weight = .52
        out["score"][i] = sum(parts.values()).reindex(symbols).to_numpy()
        out["core"][i] = parts["core"].reindex(symbols).to_numpy()
        out["cw"][i] = weight
        out["eligible"][i] = eligible.loc[day, symbols].to_numpy(dtype=bool)
    for key in ("score", "core", "cw"):
        if not np.isfinite(out[key]).all():
            raise ValueError("Nonfinite frozen baseline " + key)
    return out

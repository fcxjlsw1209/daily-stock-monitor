"""Observable SPY drawdown/recovery descriptors, without phase labels or trades.

Every event uses exactly the trailing 252 adjusted closes, including its own
signal close. A peak leaving that finite window changes the reference peak; this
is a documented causal rolling-window definition, not an all-history drawdown.
No eventual trough, future recovery, security ranks or eligibility enters here.
"""
from collections import Counter
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


WINDOW = 252
DEPTH_SCALE = .20
AGE_SCALE = 126.
STATE_NAMES = ("depth", "peak_age", "trough_depth", "recovered", "trough_age",
               "pressure", "repair", "underwater")


def _spy(panel: Mapping[str, pd.DataFrame]) -> pd.Series:
    if "adj" not in panel:
        raise KeyError("Missing required panel: adj")
    frame = panel["adj"]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("adj must be a pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Price index must be a DatetimeIndex of exchange sessions")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Price sessions must be unique and increasing")
    if not frame.columns.is_unique:
        raise ValueError("Security columns must be unique")
    if "SPY" not in frame:
        raise KeyError("SPY adjusted closes are required; no substitute benchmark")
    return frame["SPY"]


def build_states(panel: Mapping[str, pd.DataFrame],
                 events: Sequence[Tuple[object, Sequence[str]]]) -> Tuple[dict, dict]:
    """Return ``(states, audit)`` with one scalar per event and boolean valid.

    ``events`` follows the existing (signal_date, candidate_symbols) interface;
    candidate identities/counts do not affect the shared SPY state. Event dates
    must be exact sessions. Missing/nonpositive/nonfinite SPY observations or an
    incomplete 252-price history return NaN descriptors with ``valid=False``.

    The peak is the LAST occurrence of the window maximum. The trough is the
    LAST occurrence of the minimum observed from that peak through the signal.
    Ages count trading sessions. Let C=current, P=peak, L=observed trough:
      depth = 1-C/P; trough_depth = 1-L/P
      recovered = clip((C-L)/(P-L),0,1), or 0 if P=L
      pressure = clip(depth/.20,0,1)
      repair = clip(trough_depth/.20,0,1)*recovered*(1-clip(trough_age/126,0,1))
      underwater = clip(peak_age/126,0,1)
    These are continuous inputs, not mutually exclusive phases or trading gates.
    The signal close is observable only once it has completed.
    """
    spy = _spy(panel)
    days = [pd.Timestamp(day) for day, _ in events]
    states = {key: np.full(len(days), np.nan) for key in STATE_NAMES}
    states["valid"] = np.zeros(len(days), dtype=bool)
    reasons = Counter()
    observations = []
    for i, day in enumerate(days):
        if day not in spy.index:
            raise ValueError("Event date is not an exact panel session: " + str(day))
        t = spy.index.get_loc(day)
        start = t - WINDOW + 1
        row = {"date": str(day.date()), "valid": False}
        if start < 0:
            reasons["insufficient_252_price_history"] += 1
            row["reason"] = "insufficient_252_price_history"
            observations.append(row)
            continue
        window = spy.iloc[start:t + 1].to_numpy(dtype=float, copy=True)
        if not (np.isfinite(window) & (window > 0.)).all():
            reasons["missing_nonpositive_or_nonfinite_price"] += 1
            row["reason"] = "missing_nonpositive_or_nonfinite_price"
            observations.append(row)
            continue
        peak = float(window.max())
        peak_index = int(np.flatnonzero(window == peak)[-1])
        after_peak = window[peak_index:]
        trough = float(after_peak.min())
        trough_index = peak_index + int(np.flatnonzero(after_peak == trough)[-1])
        current = float(window[-1])
        depth = float(np.clip(1. - current / peak, 0., 1.))
        trough_depth = float(np.clip(1. - trough / peak, 0., 1.))
        recovered = float(np.clip((current - trough) / (peak - trough), 0., 1.)) if peak > trough else 0.
        peak_age, trough_age = WINDOW - 1 - peak_index, WINDOW - 1 - trough_index
        values = dict(depth=depth, peak_age=peak_age, trough_depth=trough_depth,
            recovered=recovered, trough_age=trough_age,
            pressure=float(np.clip(depth / DEPTH_SCALE, 0., 1.)),
            repair=float(np.clip(trough_depth / DEPTH_SCALE, 0., 1.) * recovered
                         * (1. - np.clip(trough_age / AGE_SCALE, 0., 1.))),
            underwater=float(np.clip(peak_age / AGE_SCALE, 0., 1.)))
        for key, value in values.items():
            states[key][i] = value
        states["valid"][i] = True
        row.update(valid=True, window_start=str(spy.index[start].date()),
                   peak_date=str(spy.index[start + peak_index].date()),
                   observed_trough_date=str(spy.index[start + trough_index].date()))
        observations.append(row)
    audit = {
        "events": len(days), "benchmark": "SPY", "price_basis": "adjusted close",
        "window_prices": WINDOW, "offsets_inclusive": [-(WINDOW - 1), 0],
        "depth_scale": DEPTH_SCALE, "age_scale_sessions": AGE_SCALE,
        "valid_events": int(states["valid"].sum()),
        "invalid_events": int((~states["valid"]).sum()), "invalid_reasons": dict(reasons),
        "peak_ties": "last exact maximum in trailing window",
        "trough_ties": "last exact minimum observed from that peak to signal",
        "rolling_peak_policy": "Reference peak may change when old prices leave the 252-price window; no all-history peak or future trough",
        "missing_policy": "NaN states and valid=False; no fill, substitute benchmark or eligibility change",
        "trading_eligibility_modified": False, "cross_sectional_ranking_applied": False,
        "discrete_phase_gates_applied": False, "observations": observations,
    }
    return states, audit

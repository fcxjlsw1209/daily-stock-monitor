"""Reconstruct execution share units from explicitly split-adjusted OHLCV.

No stitching providers. Requires complete split history through the source's
adjustment anchor, and a per-row dividend field. This converts units; it does
not certify identity, corporate-action completeness or disputed provider bars.
"""
import numpy as np
import pandas as pd


def split_adjusted_to_raw(frame):
    required=['open','high','low','close','volume','dividend','split']
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('Unique ordered price sessions required')
    if any(k not in frame for k in required):
        raise ValueError('Prices and explicit action coverage required')
    splits=pd.to_numeric(frame['split'],errors='coerce')
    dividend=pd.to_numeric(frame['dividend'],errors='coerce')
    if (~np.isfinite(splits)).any() or (splits<0).any() or (~np.isfinite(dividend)).any():
        raise ValueError('Unknown corporate-action data')
    splits=splits.mask(splits.eq(0),1.)
    # Events on day t have already changed that day's price units. Only events
    # strictly after t belong to the inverse adjustment multiplier.
    multiplier=splits.iloc[::-1].cumprod().iloc[::-1]/splits
    if (~np.isfinite(multiplier)).any() or (multiplier<=0).any():
        raise ValueError('Invalid split unit multiplier')
    out=frame.copy()
    for column in ['open','high','low','close','dividend']:
        out[column]=pd.to_numeric(frame[column],errors='coerce')*multiplier
    out['volume']=pd.to_numeric(frame['volume'],errors='coerce')/multiplier
    out['split']=splits
    out['raw_per_signal']=multiplier
    return out


def raw_to_split_adjusted(frame):
    """Convert documented raw OHLCV to a single split-only indicator basis.

    Dividend-adjusted close is kept untouched for total-return momentum. Cash
    dividends remain raw in the returned execution frame. Never derive share
    changes from adj_close/close: that ratio also includes cash distributions.
    Returns (signal_frame, execution_frame).
    """
    required=['open','high','low','close','volume','dividend','split','adj_close']
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('Unique ordered price sessions required')
    if any(k not in frame for k in required):
        raise ValueError('Raw prices and explicit actions required')
    splits=pd.to_numeric(frame['split'],errors='coerce')
    dividend=pd.to_numeric(frame['dividend'],errors='coerce')
    if (~np.isfinite(splits)).any() or (splits<=0).any() or (~np.isfinite(dividend)).any():
        raise ValueError('Raw source requires positive explicit split factors, including 1 for no action')
    multiplier=splits.iloc[::-1].cumprod().iloc[::-1]/splits
    if (~np.isfinite(multiplier)).any() or (multiplier<=0).any():
        raise ValueError('Invalid split multiplier')
    signal=frame.copy();raw=frame.copy()
    for column in ['open','high','low','close']:
        signal[column]=pd.to_numeric(frame[column],errors='coerce')/multiplier
    signal['volume']=pd.to_numeric(frame['volume'],errors='coerce')*multiplier
    # Signal dividend is on the same share basis; execution keeps raw amount.
    signal['dividend']=dividend/multiplier
    raw['raw_per_signal']=multiplier
    return signal,raw

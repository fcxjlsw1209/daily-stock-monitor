"""User-approved available-data entry filter; never a holding/valuation filter."""
import numpy as np

VERSION = 'historical_ndx_sp100_union_v3_available_data'


def entry_eligibility(membership, adj, opening, high, low, close, volume):
    """Use only this session's observed bars; preserve all historical price rows.

    Columns must already be validated permanent security IDs. This filter does
    not certify identity, adjustments or historical membership. Factor warmup
    remains the frozen strategy's responsibility. Never inspect next-day data.
    """
    if membership.isna().any().any() or not all(t == bool for t in membership.dtypes):
        raise ValueError('Complete boolean membership required')
    if not membership.index.is_unique or not membership.columns.is_unique:
        raise ValueError('Unique sessions and security IDs required')
    frames = [adj, opening, high, low, close, volume]
    for frame in frames:
        if not membership.index.equals(frame.index) or not membership.columns.equals(frame.columns):
            raise ValueError('Exact input alignment required')
    observed = membership.copy()
    for frame in frames:
        observed &= np.isfinite(frame) & (frame > 0)
    observed &= (high >= low) & (close >= low) & (close <= high)
    observed &= (opening >= low) & (opening <= high)
    return observed

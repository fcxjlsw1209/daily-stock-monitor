"""Pure prospective guidance features. Use on real data only after registration.

No file, market data, model, portfolio, or order access. None means unavailable;
a true zero is a measured no-growth comparison. This module does not admit data.
"""
from decimal import Decimal, localcontext
import math
import re
import monthly_forward_guidance_contract_v1 as contract


def _normalized_decimal(value):
    """Read contract-normalized units, which can exceed raw input's 40 digits."""
    if not isinstance(value, str) or re.fullmatch(r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?', value) is None:
        raise ValueError('Normalized amounts require finite unsigned decimal strings')
    # normalize_amount uses 100-digit arithmetic; allow its fixed-point output
    # (including fractional leading zeros) without accepting unbounded input.
    if len(value.replace('.', '')) > 200:
        raise ValueError('Normalized amount exceeds supported size')
    return Decimal(value)


def bounded_growth(numerator, base):
    """Map two positive source-normalized decimal strings to (-.5,.5)."""
    g, r = _normalized_decimal(numerator), _normalized_decimal(base)
    if g <= 0 or r <= 0:
        raise ValueError('Positive finite same-currency revenue values required')
    with localcontext() as ctx:
        ctx.prec = max(80, len(g.as_tuple().digits) + 20, len(r.as_tuple().digits) + 20)
        value = Decimal('0.5') * (g-r)/(g+r)
    number = float(value)
    if not math.isfinite(number) or abs(number) > .5:
        raise ValueError('Invalid bounded-growth conversion')
    return number


def event_signals(event):
    """Compute a matched pair, or explicit missing, after source-schema checks."""
    assessed = contract.assess_event(event)
    if not assessed['comparison_eligible']:
        return {'forward': None, 'realized': None,
                'status': 'incomplete_source_comparison',
                'reasons': assessed['source_reasons'] + assessed['comparison_reasons']}
    f = assessed['normalized_forecast']; a = assessed['normalized_actuals']
    return {'forward': bounded_growth(f['midpoint'], a['target_prior_year']['midpoint']),
            'realized': bounded_growth(a['current_quarter']['midpoint'], a['current_prior_year']['midpoint']),
            'status': 'matched_source_pair', 'reasons': []}


def features_for_rows(events, rows, reviewed_event_ids):
    """Preserve all source-month rows and require caller-bound review IDs.

    `reviewed_event_ids` must come from an independently verified admission
    manifest. Membership here never proves the review itself. Row availability
    must come from the separately bound latest-observed-source time join.
    """
    ids = [e.get('event_id') for e in events]
    if any(not isinstance(x,str) or not x for x in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Unique source-event IDs required')
    if not isinstance(reviewed_event_ids, (set, frozenset)) or not reviewed_event_ids <= set(ids):
        raise ValueError('Review IDs must be an explicit subset of source events')
    by_id = dict(zip(ids, events)); out = []; keys = set()
    for row in rows:
        key = (row['signal_index'], row['pool_index'])
        if key in keys: raise ValueError('Duplicate original source-month row')
        keys.add(key)
        result = {k: row[k] for k in ('signal_index','pool_index','signal_date','symbol')}
        result.update(event_id=row.get('event_id'), forward=None, realized=None,
                      status='source_not_available')
        if type(row.get('comparison_available')) is not bool:
            raise ValueError('Explicit boolean source availability required')
        if type(row.get('is_etf')) is not bool:
            raise ValueError('Explicit boolean fund classification required')
        if row['is_etf']:
            result['status']='etf_not_applicable'
        elif row['comparison_available']:
            eid = row.get('event_id')
            if eid not in reviewed_event_ids:
                result['status']='independent_review_missing'
            else:
                event=by_id[eid]
                if event['symbol']!=row['symbol']:
                    raise ValueError('Source and original-pool identity disagree')
                result.update(event_signals(event))
        out.append(result)
    return out

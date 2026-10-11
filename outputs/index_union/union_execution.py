"""Apply frozen engines to permanent IDs and point-in-time representatives.

Pure in-memory adapter: no paper ledger, monitoring plan or fingerprint writes.
Caller must pass separately certified prices, identity, industry and membership.
"""
from pathlib import Path
import copy,importlib.util
P=Path(__file__).resolve().parent.parent
ACCOUNTS={'baseline':'daily_stock_monitor','pressure':'pressure_shadow_monitor','resilience':'resilience_shadow_monitor',
          'accel':'accel_shadow_monitor','high52':'high52_shadow_monitor','accel40':'accel40_shadow_monitor'}

def engine(name):
    spec=importlib.util.spec_from_file_location('union_engine_'+name,P/ACCOUNTS[name]/'engine.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,representatives,issuers):
    """Representatives/sectors must be known at prior signal close, not fill day."""
    if name not in ACCOUNTS:raise ValueError('Unknown frozen strategy')
    selected=set(representatives)
    if len(selected)!=len(representatives):raise ValueError('Duplicate representative')
    needed=selected|set(state['positions'])
    if any(s not in issuers or s not in sectors or not sectors[s] for s in needed):
        raise ValueError('Missing identity or contemporaneous sector')
    if len({issuers[s] for s in selected})!=len(selected):raise ValueError('Multiple classes of one issuer')
    held=set(state['positions']);held_issuers={issuers[s] for s in held}
    if len(held_issuers)!=len(held):raise ValueError('Duplicate issuer in holdings')
    adjusted=mask.copy()
    for s in adjusted.columns:
        if s not in selected or (s not in held and issuers[s] in held_issuers):adjusted[s]=False
    config=dict(exit='guarded10' if name in ['baseline','pressure','resilience'] else 'fixed20',
                allocation=.1 if name in ['baseline','pressure','resilience'] else .2,max_positions=5,sector_cap=2)
    return engine(name).plan(copy.deepcopy(state),day,dates,close,ma5,atr,adjusted,score,sectors,config)


def execute(name, state, decision, bars, dates, cost_bps=10, sector=None):
    """Cancel only new entries with unavailable open; never drop held positions.

    Cancellation uses the fill-session open alone, not that day's close or any
    later gap. No replacement picks. Missing close for a valid filled entry is
    an unresolved valuation error, not permission to erase its trade.
    Returns the original engine's five outputs plus the cancelled entry log.
    """
    import math
    def has_open(symbol):
        try:
            value = float(bars[symbol]['open'])
            return math.isfinite(value) and value > 0
        except (KeyError, TypeError, ValueError):
            return False
    adjusted = copy.deepcopy(decision)
    cancelled = []
    kept = []
    for buy in adjusted['buys']:
        symbol = buy['symbol']
        if symbol in state['positions']:
            raise ValueError('Entry duplicates held security')
        if not has_open(symbol):
            cancelled.append(dict(session=decision['session'], symbol=symbol,
                                  reason='missing_execution_open', budget=buy['budget']))
        else:
            kept.append(buy)
    adjusted['buys'] = kept
    result = engine(name).execute(state, adjusted, bars, dates, cost_bps=cost_bps, sector=sector)
    return (*result, cancelled)

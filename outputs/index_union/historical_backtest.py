"""In-memory six-strategy historical executor; no writes to paper accounts.

The caller supplies effective-dated, security-ID keyed, certified panels. Missing
new-entry bars are excluded/cancelled; missing held valuation stops that account
with a diagnostic and preserves its last valid state. Synthetic tests do not
constitute historical data certification.
"""
import copy
import math
import numpy as np
import pandas as pd
from dynamic_features import features
import cash_acquisitions
import terminal_baskets
import spin_distributions
from union_execution import ACCOUNTS, engine, plan, execute

FEATURE_FIELDS = ('adj','close','opening','high','low','volume','sector60',
                  'market_close','market_dividends','eligible')


def validate_inputs(data, start, end):
    close = data['close']
    dates = close.index
    if (not isinstance(dates, pd.DatetimeIndex) or not dates.is_unique or
            not dates.is_monotonic_increasing or not close.columns.is_unique):
        raise ValueError('Unique ordered DatetimeIndex and permanent security IDs required')
    if dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError('Session labels must be timezone-naive normalized exchange dates')
    for key in FEATURE_FIELDS + ('dividends','splits','sectors','raw_per_signal'):
        value = data[key]
        if not dates.equals(value.index):
            raise ValueError('Session alignment: '+key)
        if isinstance(value, pd.DataFrame) and not close.columns.equals(value.columns):
            raise ValueError('Security alignment: '+key)
    if set(data['issuers']) != set(close.columns) or any(not v for v in data['issuers'].values()):
        raise ValueError('Complete permanent issuer mapping required')
    if data.get('signal_basis') != 'split_adjusted':
        raise ValueError('Signals require consistently split-adjusted OHLCV')
    if data.get('execution_basis') != 'raw_with_explicit_actions':
        raise ValueError('Execution must use raw prices and explicit split/dividend actions')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start not in dates or end not in dates or start > end or dates.get_loc(start) == 0:
        raise ValueError('Valid run bounds and prior signal session required')
    return dates[(dates >= start) & (dates <= end)]


def metrics(state, daily, trades):
    equity = np.array([state['initial']] + [r['equity'] for r in daily], dtype=float)
    peak = np.maximum.accumulate(equity)
    realized = sum(t['net_pnl'] for t in trades)
    total = equity[-1] - state['initial']
    elapsed = ((pd.Timestamp(daily[-1]['date']) - pd.Timestamp(daily[0]['date'])).days
               if len(daily) > 1 else 0)
    return dict(initial=state['initial'], equity=float(equity[-1]), net_return=float(total/state['initial']),
                realized_pnl=float(realized), unrealized_pnl=float(total-realized),
                closed_trades=len(trades), net_win_rate=(sum(t['net_pnl']>0 for t in trades)/len(trades) if trades else None),
                maximum_drawdown=float((equity/peak-1).min()),
                cagr=(float((equity[-1]/state['initial'])**(365.25/elapsed)-1) if elapsed>=365 else None),
                complete_sessions=len(daily))


def run_account(name, data, start, end, calculated=None):
    run_dates = validate_inputs(data, start, end)
    dates, close = data['close'].index, data['close']
    calculated = calculated or features(**{k:data[k] for k in FEATURE_FIELDS})
    f = calculated[name]
    state = engine(name).initial_state()
    daily, trades, orders, actions, cancellations, decisions = [], [], [], [], [], []
    for day in run_dates:
        prev = dates[dates.get_loc(day)-1]
        try:
            reps = list(data['eligible'].columns[data['eligible'].loc[prev]])
            # Missing signal bars can remove a representative; never use fill-day close.
            usable = f['score'].loc[prev].notna()
            reps = [s for s in reps if usable[s]]
            needed = set(reps) | set(state['positions'])
            sectors = {s:data['sectors'].at[prev,s] for s in needed}
            if any(pd.isna(v) or not isinstance(v,str) or not v for v in sectors.values()):
                raise ValueError('Missing contemporaneous sector')
            # Engine positions use actual historical share units; indicators use
            # one consistent split-adjusted basis. Convert only the planning copy.
            planning_state=copy.deepcopy(state)
            for symbol,position in planning_state['positions'].items():
                multiplier=data['raw_per_signal'].at[prev,symbol]
                if not np.isfinite(multiplier) or multiplier<=0:
                    raise ValueError('Unknown signal/ledger price units: '+symbol)
                position['entry_price']/=multiplier;position['atr']/=multiplier
            decision = plan(name,planning_state,day,dates,close,f['ma5'],f['atr'],f['mask'],f['score'],
                            sectors,reps,data['issuers'])
            for buy in decision['buys']:
                multiplier=data['raw_per_signal'].at[prev,buy['symbol']]
                if not np.isfinite(multiplier) or multiplier<=0:
                    raise ValueError('Unknown entry price units: '+buy['symbol'])
                buy['atr']*=multiplier;buy['reference_close']*=multiplier
            previous_equity=state['equity']
            component_ids={c['security_id'] for e in data.get('terminal_baskets',[])
                           if e['execution_session']==str(day.date()) for c in e['components']
                           if 'documented_first_trade' not in c}
            component_open={s:data['opening'].at[day,s]*data['raw_per_signal'].at[day,s]
                            for s in component_ids if s in close}
            staged,decision,basket_trades,basket_actions,basket_cancellations = terminal_baskets.apply(
                state,decision,data.get('terminal_baskets',[]),day,dates,component_open)
            staged,decision,action_trades,cash_actions,action_cancellations = cash_acquisitions.apply(
                staged,decision,data.get('cash_acquisitions',[]),day,dates)
            action_trades=basket_trades+action_trades
            cash_actions=basket_actions+cash_actions
            action_cancellations=basket_cancellations+action_cancellations
            symbols = set(staged['positions']) | {r['symbol'] for r in decision['buys']}
            bars = {s:dict(open=data['opening'].at[day,s]*data['raw_per_signal'].at[day,s],close=close.at[day,s]*data['raw_per_signal'].at[day,s],
                           dividends=data['dividends'].at[day,s],split=data['splits'].at[day,s]) for s in symbols}
            staged,decision,bars,spin_actions=spin_distributions.apply(staged,decision,bars,data.get('spin_distributions',[]),day)
            cash_actions+=spin_actions
            # Missing/invalid actions must not silently become zero/no-split in an engine.
            for s in symbols:
                opening = bars[s]['open']
                if s not in staged['positions'] and (not np.isfinite(opening) or opening <= 0):
                    continue
                split, dividend = bars[s]['split'], bars[s]['dividends']
                if not np.isfinite(split) or split <= 0 or not np.isfinite(dividend):
                    raise ValueError('Missing corporate-action coverage: '+s)
                standard = abs(split-round(split))<1e-7 or abs(1/split-round(1/split))<1e-7
                if not standard:
                    evidence=[e for e in data.get('true_stock_splits',[]) if e['security_id']==s and e['session']==str(day.date()) and abs(e['ratio']-split)<1e-7 and e.get('source_sha256') and e.get('verified')]
                    if not evidence:raise ValueError('Unclassified nonstandard share action: '+s+' '+str(day.date()))
            after, row, closed, fills, events, cancelled = execute(name,staged,decision,bars,dates,sector=sectors)
            after,row=cash_acquisitions.finish_valuation(after,row,previous_equity)
            trades.extend(action_trades);actions.extend(cash_actions);cancellations.extend(action_cancellations)
            state = after
            daily.append(row);trades.extend(closed);orders.extend(fills);actions.extend(dict(event,session=str(day.date())) for event in events)
            cancellations.extend(cancelled);decisions.append(decision)
        except (ValueError,KeyError,AssertionError) as exc:
            return dict(strategy=name,status='blocked',blocked_session=str(day.date()),
                        error=str(exc),last_valid_state=copy.deepcopy(state),daily=daily,trades=trades,
                        orders=orders,actions=actions,cancellations=cancellations,decisions=decisions,
                        metrics=None,performance_publishable=False)
    return dict(strategy=name,status='complete',state=state,daily=daily,trades=trades,orders=orders,
                actions=actions,cancellations=cancellations,decisions=decisions,
                metrics=metrics(state,daily,trades),performance_publishable=False)


def run_all(data, start, end):
    validate_inputs(data,start,end)
    calculated = features(**{k:data[k] for k in FEATURE_FIELDS})
    return {name:run_account(name,data,start,end,calculated) for name in ACCOUNTS}

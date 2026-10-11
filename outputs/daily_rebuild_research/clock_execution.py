"""Local copy of audited historical execution, with explicit signal clock.
Next-open path must reproduce original engine. Same-close mode deliberately uses
finalized OHLCV before filling at that close: a non-executable timing diagnostic.
Corporate actions and actual historical share units retained in both modes.
"""
import copy
import numpy as np
import pandas as pd
from historical_backtest import validate_inputs,metrics
from union_execution import engine,execute as original_execute
import cash_acquisitions,terminal_baskets,spin_distributions

def run_account(data, f, plan, start, end, cost_bps=10, clock="next_open"):
    if clock not in ["next_open", "same_close_diagnostic"]: raise ValueError(clock)
    name="baseline"
    data=dict(data)
    if clock=="same_close_diagnostic": data["opening"]=data["close"]
    def execute(name,state,decision,bars,dates,sector=None):
        return original_execute(name,state,decision,bars,dates,cost_bps=cost_bps,sector=None)
    run_dates = validate_inputs(data, start, end)
    dates, close = data['close'].index, data['close']
    state = engine(name).initial_state()
    daily, trades, orders, actions, cancellations, decisions = [], [], [], [], [], []
    for day in run_dates:
        prev = dates[dates.get_loc(day)-1]
        signal = day if clock=='same_close_diagnostic' else prev
        try:
            reps = list(data['eligible'].columns[data['eligible'].loc[signal]])
            # Missing signal bars can remove a representative; never use fill-day close.
            usable = f['score'].loc[signal].notna()
            reps = [s for s in reps if usable[s]]
            needed = set(reps) | set(state['positions'])
            sectors = {s:data['sectors'].at[signal,s] for s in needed}
            if any(pd.isna(v) or not isinstance(v,str) or not v for v in sectors.values()):
                raise ValueError('Missing contemporaneous sector')
            # Engine positions use actual historical share units; indicators use
            # one consistent split-adjusted basis. Convert only the planning copy.
            planning_state=copy.deepcopy(state)
            for symbol,position in planning_state['positions'].items():
                multiplier=data['raw_per_signal'].at[signal,symbol]
                if not np.isfinite(multiplier) or multiplier<=0:
                    raise ValueError('Unknown signal/ledger price units: '+symbol)
                # Before today's action ledger update, old holdings retain yesterday's shares.
                today_split=data['splits'].at[day,symbol] if clock=='same_close_diagnostic' else 1.
                position['entry_price']/=multiplier*today_split;position['atr']/=multiplier*today_split
            decision = plan(name,planning_state,day,dates,close,f['ma5'],f['atr'],f['mask'],f['score'],
                            sectors,reps,data['issuers'])
            for buy in decision['buys']:
                multiplier=data['raw_per_signal'].at[signal,buy['symbol']]
                if not np.isfinite(multiplier) or multiplier<=0:
                    raise ValueError('Unknown entry price units: '+buy['symbol'])
                # Shared engine divides new-entry ATR by today's split. Compensate only
                # when this diagnostic already observes post-split same-day indicators.
                today_split=data['splits'].at[day,buy['symbol']] if clock=='same_close_diagnostic' else 1.
                buy['atr']*=multiplier*today_split;buy['reference_close']*=multiplier
            previous_equity=state['equity']
            component_ids={c['security_id'] for e in data.get('terminal_baskets',[])
                           if e['execution_session']==str(day.date()) for c in e['components']
                           if 'documented_first_trade' not in c}
            component_open={s:data['opening'].at[day,s]*data['raw_per_signal'].at[day,s]
                            for s in component_ids if s in close}
            staged,decision,basket_trades,basket_actions,basket_cancellations = terminal_baskets.apply(
                state,decision,data.get('terminal_baskets',[]),day,dates,component_open,cost_bps=cost_bps)
            staged,decision,action_trades,cash_actions,action_cancellations = cash_acquisitions.apply(
                staged,decision,data.get('cash_acquisitions',[]),day,dates)
            action_trades=basket_trades+action_trades
            cash_actions=basket_actions+cash_actions
            action_cancellations=basket_cancellations+action_cancellations
            symbols = set(staged['positions']) | {r['symbol'] for r in decision['buys']}
            bars = {s:dict(open=data['opening'].at[day,s]*data['raw_per_signal'].at[day,s],close=close.at[day,s]*data['raw_per_signal'].at[day,s],
                           dividends=data['dividends'].at[day,s],split=data['splits'].at[day,s]) for s in symbols}
            staged,decision,bars,spin_actions=spin_distributions.apply(staged,decision,bars,data.get('spin_distributions',[]),day,cost_bps=cost_bps)
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
    assert all(row['cash']>=-1e-6 and row['exposure']<=1+1e-8 and row['positions']<=3 for row in daily)
    assert all((x['signal_date']<x['session']) if clock=='next_open' else (x['signal_date']==x['session']) for x in decisions)
    return dict(clock=clock,strategy=name,status='complete',state=state,daily=daily,trades=trades,orders=orders,
                actions=actions,cancellations=cancellations,decisions=decisions,
                metrics=metrics(state,daily,trades),performance_publishable=False)


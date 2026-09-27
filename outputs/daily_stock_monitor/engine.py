"""Deterministic shared research and forward-paper ledger engine."""
import copy
import numpy as np
import pandas as pd


def initial_state(capital=100000.):
    return dict(initial=float(capital),cash=float(capital),equity=float(capital),positions={},fees=0.,dividends=0.,last_date=None)


def valid_price(x):
    if not np.isfinite(x) or x<=0:raise ValueError('Invalid/missing price')
    return float(x)


def plan(state,day,dates,close,ma5,atr,mask,score,sector,config):
    day=pd.Timestamp(day);i=dates.get_loc(day);prev=dates[i-1]
    if i<1:raise ValueError('Prior signal session required')
    exits=[];remaining=set(state['positions'])
    for symbol,p in state['positions'].items():
        age=i-dates.get_loc(pd.Timestamp(p['entry_date']));price=valid_price(close.at[prev,symbol]);reason=None
        if config['exit']!='fixed5' and age>=1 and price>ma5.at[prev,symbol]:reason='close_above_sma5'
        if config['exit']=='guarded10' and age>=1 and price<p['entry_price']-2*p['atr']:reason='close_below_2atr'
        cap=10 if config['exit']=='guarded10' else 5
        if age>=cap:reason=reason or 'maximum_sessions'
        if reason:exits.append(dict(symbol=symbol,reason=reason));remaining.remove(symbol)
    sold={v['symbol'] for v in exits};buys=[]
    candidates=score.loc[prev].where(mask.loc[prev]).dropna().sort_values(ascending=False,kind='stable')
    for symbol,value in candidates.items():
        if len(remaining)>=config.get('max_positions',5):break
        if symbol in remaining or symbol in sold:continue
        if sum(sector[s]==sector[symbol] for s in remaining)>=config.get('sector_cap',2):continue
        a=valid_price(atr.at[prev,symbol]);budget=state['equity']*config.get('allocation',.10)
        buys.append(dict(symbol=symbol,budget=float(budget),score=float(value),atr=a,reference_close=float(close.at[prev,symbol])))
        remaining.add(symbol)
    return dict(session=str(day.date()),signal_date=str(prev.date()),sells=exits,buys=buys,
      equity_basis=float(state['equity']),candidate_count=int(mask.loc[prev].sum()))


def execute(state,decision,bars,dates,cost_bps=10,sector=None):
    """bars use current-session price units; splits only for unadjusted paper tape."""
    state=copy.deepcopy(state);day=decision['session'];half=cost_bps/20000;orders=[];trades=[];actions=[]
    needed=set(state['positions'])|{r['symbol'] for r in decision['buys']}
    for s in needed:
        bar=bars[s]
        for field in ['open','close']:valid_price(bar[field])
        if not np.isfinite(bar.get('dividends',0)):raise ValueError('Missing dividend')
    for s,p in state['positions'].items():
        split=float(bars[s].get('split',0))
        if split and split!=1:
            if split<=0 or not np.isfinite(split):raise ValueError('Invalid split')
            p['qty']*=split;p['entry_price']/=split;p['atr']/=split
            actions.append(dict(symbol=s,split=split))
        income=p['qty']*float(bars[s].get('dividends',0));state['cash']+=income;p['dividends']+=income;state['dividends']+=income
    for sell in decision['sells']:
        s=sell['symbol'];p=state['positions'].pop(s);price=valid_price(bars[s]['open']);gross=p['qty']*price;fee=gross*half
        state['cash']+=gross-fee;state['fees']+=fee;pnl=gross-fee+p['dividends']-p['cost']
        trades.append(dict(symbol=s,entry_date=p['entry_date'],exit_date=day,entry_price=p['entry_price'],exit_price=price,
          quantity=p['qty'],entry_outlay=p['cost'],dividends=p['dividends'],entry_fee=p['entry_fee'],exit_fee=fee,
          net_pnl=pnl,net_return=pnl/p['cost'],sessions=dates.get_loc(pd.Timestamp(day))-dates.get_loc(pd.Timestamp(p['entry_date'])),reason=sell['reason']))
        orders.append(dict(date=day,symbol=s,side='sell',price=price,quantity=p['qty'],fee=fee))
    for buy in decision['buys']:
        s=buy['symbol'];price=valid_price(bars[s]['open']);budget=min(state['cash'],buy['budget'])
        if budget<1e-8:continue
        if s in state['positions']:raise ValueError('Duplicate position')
        qty=budget/(1+half)/price;fee=qty*price*half;state['cash']-=budget;state['fees']+=fee
        split=float(bars[s].get('split',0)) or 1.
        state['positions'][s]=dict(entry_date=day,entry_price=price,qty=qty,cost=budget,entry_fee=fee,dividends=0.,atr=buy['atr']/split)
        orders.append(dict(date=day,symbol=s,side='buy',price=price,quantity=qty,fee=fee))
    invested=sum(p['qty']*valid_price(bars[s]['close']) for s,p in state['positions'].items())
    previous=state['equity'];state['equity']=state['cash']+invested;state['last_date']=day
    if state['cash']<-1e-7 or len(state['positions'])>5:raise AssertionError('Capacity/cash constraint')
    if sector:
        for s in state['positions']:assert sum(sector[t]==sector[s] for t in state['positions'])<=2
    daily=dict(date=day,equity=state['equity'],cash=state['cash'],invested=invested,exposure=invested/state['equity'],
      positions=len(state['positions']),return_=state['equity']/previous-1,fees_cumulative=state['fees'],dividends_cumulative=state['dividends'])
    daily['return']=daily.pop('return_')
    return state,daily,trades,orders,actions

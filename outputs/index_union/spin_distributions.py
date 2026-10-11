"""Receive and liquidate documented child shares, never invent parent shares.

Research protocol: sell distributed fractions at first regular-way open, with
normal sale cost. Signal reference prices retain the provider's documented
adjustment basis, while economic quantity/cost remain in actual parent units.
"""
import copy,math
import pandas as pd

def apply(state,decision,bars,events,day,cost_bps=10):
 state=copy.deepcopy(state);decision=copy.deepcopy(decision);bars=copy.deepcopy(bars);actions=[]
 done=state.setdefault('processed_spin_distributions',[]);label=str(pd.Timestamp(day).date())
 for e in events:
  if e['execution_session']!=label:continue
  s=e['security_id'];key=e['event_id']
  if s not in bars:continue
  if not e.get('terms_verified') or not e.get('source_sha256'):raise ValueError('Unverified spin distribution: '+key)
  known=pd.Timestamp(e['known_at']);cutoff=pd.Timestamp(label+'T09:30',tz='America/New_York')
  if known.tzinfo is None or known>=cutoff:raise ValueError('Future spin terms')
  adjustment=float(e['provider_adjustment']);actual=float(bars[s]['split'])
  if not math.isfinite(adjustment) or adjustment<=0 or abs(actual-adjustment)>1e-7:raise ValueError('Spin adjustment mismatch: '+key)
  bars[s]['split']=1.
  # New entries get no distribution, but ATR came from pre-ex-date units.
  for b in decision['buys']:
   if b['symbol']==s:b['atr']/=adjustment;b['reference_close']/=adjustment
  if key in done:continue
  if s in state['positions']:
   quote=e['child_open'];price=float(quote['price']);ratio=float(e['ratio'])
   if quote['session']!=label or quote['kind']!='regular_way_open' or not quote.get('source_sha256') or not math.isfinite(price) or price<=0 or ratio<=0:raise ValueError('Missing verified child open: '+key)
   p=state['positions'][s];qty=p['qty']*ratio;gross=qty*price;fee=gross*cost_bps/20000;net=gross-fee
   state['cash']+=net;state['fees']+=fee;p['dividends']+=net
   state.setdefault('spin_proceeds',0.);state['spin_proceeds']+=net
   p['entry_price']/=adjustment;p['atr']/=adjustment
   actions.append(dict(event_id=key,session=label,symbol=s,kind='spin_distribution_liquidation',child=e['child_security_id'],child_quantity=qty,child_open=price,gross=gross,fee=fee,parent_quantity=p['qty']))
  done.append(key)
 return state,decision,bars,actions

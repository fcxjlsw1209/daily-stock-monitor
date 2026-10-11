"""Cash-only acquisition receivables; never invent a payment date or quote.

Stock/CVR/spin-off consideration is deliberately unsupported by this adapter.
Input events must already have verified permanent identity and raw share units.
Receivables count in NAV but cannot finance trades until documented payment.
"""
import copy
import math
import pandas as pd


def apply(state, decision, events, day, dates):
    state, decision = copy.deepcopy(state), copy.deepcopy(decision)
    day = str(pd.Timestamp(day).date())
    claims = state.setdefault('cash_receivables', [])
    processed = state.setdefault('processed_cash_actions', [])
    trades, actions, cancellations = [], [], []
    for event in events:
        if event['effective_session'] != day:
            continue
        key, security = event['event_id'], event['security_id']
        if key in processed:
            continue
        if event.get('kind') != 'cash_acquisition' or not event.get('verified'):
            raise ValueError('Unverified or unsupported cash action: '+key)
        if event.get('price_basis') != 'raw_with_explicit_actions' or not event.get('source_sha256'):
            raise ValueError('Cash action needs verified raw ledger units and source')
        price = float(event['cash_per_share'])
        if not math.isfinite(price) or price <= 0:
            raise ValueError('Invalid acquisition consideration')
        payment = event.get('payment_session')
        if payment is not None and (payment < day or pd.Timestamp(payment) not in dates):
            raise ValueError('Invalid documented payment session')
        for buy in decision['buys']:
            if buy['symbol'] == security:
                cancellations.append(dict(session=day,symbol=security,reason='cash_acquisition_effective',budget=buy['budget']))
        decision['buys'] = [b for b in decision['buys'] if b['symbol'] != security]
        decision['sells'] = [s for s in decision['sells'] if s['symbol'] != security]
        if security in state['positions']:
            p = state['positions'].pop(security)
            amount = p['qty']*price
            claims.append(dict(event_id=key,security_id=security,amount=amount,payment_session=payment))
            pnl = amount + p['dividends'] - p['cost']
            trades.append(dict(symbol=security,entry_date=p['entry_date'],exit_date=day,
                entry_price=p['entry_price'],exit_price=price,quantity=p['qty'],entry_outlay=p['cost'],
                dividends=p['dividends'],entry_fee=p['entry_fee'],exit_fee=0.,net_pnl=pnl,
                net_return=pnl/p['cost'],sessions=dates.get_loc(pd.Timestamp(day))-dates.get_loc(pd.Timestamp(p['entry_date'])),
                reason='cash_acquisition_receivable',payment_session=payment))
            actions.append(dict(event_id=key,symbol=security,kind='cash_receivable_recognized',amount=amount))
        processed.append(key)
    remaining=[]
    for claim in claims:
        if claim['payment_session'] is not None and claim['payment_session'] <= day:
            state['cash'] += claim['amount']
            actions.append(dict(event_id=claim['event_id'],symbol=claim['security_id'],kind='cash_receivable_paid',amount=claim['amount']))
        else:
            remaining.append(claim)
    state['cash_receivables']=remaining
    return state, decision, trades, actions, cancellations


def finish_valuation(state, daily, previous_equity):
    receivable = sum(c['amount'] for c in state.get('cash_receivables',[]))
    state['equity'] += receivable
    daily.update(equity=state['equity'],cash_receivables=receivable,
                 exposure=daily['invested']/state['equity'],
                 **{'return':state['equity']/previous_equity-1})
    return state, daily

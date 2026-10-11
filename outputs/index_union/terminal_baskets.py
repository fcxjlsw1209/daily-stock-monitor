"""Research-only first-tradable-open liquidation of terminal merger baskets.

Explicit protocol assumption: a terminated security ends its strategy holding;
received listed components are sold at the documented first tradable open, with
normal sale costs. Cash consideration is a receivable until known payment. This
never treats a CVR as zero, or converts a valuation estimate into a market quote.
Existing positions in the acquiring company remain independent and unchanged.
"""
import copy,math
import pandas as pd


def apply(state,decision,events,day,dates,opening,cost_bps=10):
    state=copy.deepcopy(state);decision=copy.deepcopy(decision)
    label=str(pd.Timestamp(day).date());trades=[];actions=[];cancellations=[]
    done=state.setdefault('processed_terminal_baskets',[])
    for e in events:
        if e['execution_session']!=label or e['event_id'] in done:continue
        s=e['security_id'];key=e['event_id']
        if not e.get('terms_verified') or not e.get('source_sha256'):
            raise ValueError('Unverified terminal basket: '+key)
        if e.get('policy')!='liquidate_first_tradable_open' or e.get('price_basis')!='raw_with_explicit_actions':
            raise ValueError('Undeclared terminal-basket assumption')
        known=pd.Timestamp(e['known_at'])
        cutoff=pd.Timestamp(label+'T09:30:00',tz='America/New_York')
        if known.tzinfo is None or known>=cutoff or e['effective_session']>label:
            raise ValueError('Terminal terms not known before execution')
        for buy in decision['buys']:
            if buy['symbol']==s:cancellations.append(dict(session=label,symbol=s,reason='terminal_basket_effective',budget=buy['budget']))
        decision['buys']=[b for b in decision['buys'] if b['symbol']!=s]
        decision['sells']=[x for x in decision['sells'] if x['symbol']!=s]
        if s in state['positions']:
            p=state['positions'][s];cash=float(e.get('cash_per_share',0))
            if not math.isfinite(cash) or cash<0:raise ValueError('Invalid cash consideration')
            pieces=[];gross=0.
            for component in e['components']:
                ratio=float(component['ratio'])
                if not math.isfinite(ratio) or ratio<=0:raise ValueError('Invalid component ratio')
                quote=component.get('documented_first_trade')
                if quote:
                    if quote.get('session')!=label or quote.get('kind')!='first_trade' or not quote.get('source_sha256'):
                        raise ValueError('Wrong-date/unverified first trade')
                    price=float(quote['price'])
                else:price=float(opening.get(component['security_id'],float('nan')))
                if not math.isfinite(price) or price<=0:raise ValueError('Missing component execution price: '+component['security_id'])
                value=p['qty']*ratio*price;gross+=value
                pieces.append(dict(security_id=component['security_id'],quantity=p['qty']*ratio,price=price,gross=value,quote_kind='first_trade' if quote else 'daily_open'))
            fee=gross*cost_bps/20000;cash_amount=p['qty']*cash
            payment=e.get('payment_session')
            if payment is not None and (payment<label or pd.Timestamp(payment) not in dates):raise ValueError('Invalid payment date')
            state['positions'].pop(s);state['cash']+=gross-fee;state['fees']+=fee
            if cash_amount:
                state.setdefault('cash_receivables',[]).append(dict(event_id=key,security_id=s,amount=cash_amount,payment_session=payment))
            proceeds=gross+cash_amount;pnl=proceeds-fee+p['dividends']-p['cost']
            trades.append(dict(symbol=s,entry_date=p['entry_date'],exit_date=label,entry_price=p['entry_price'],exit_price=proceeds/p['qty'],quantity=p['qty'],entry_outlay=p['cost'],dividends=p['dividends'],entry_fee=p['entry_fee'],exit_fee=fee,net_pnl=pnl,net_return=pnl/p['cost'],sessions=dates.get_loc(pd.Timestamp(label))-dates.get_loc(pd.Timestamp(p['entry_date'])),reason='terminal_basket_first_trade_liquidation',cash_receivable=cash_amount,payment_session=payment,components=pieces))
            actions.append(dict(event_id=key,symbol=s,kind='terminal_basket_liquidation',components=pieces,cash_receivable=cash_amount,fee=fee))
        done.append(key)
    return state,decision,trades,actions,cancellations

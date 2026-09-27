import json
import pandas as pd

def reconcile(m,db,cfg,frames,manifest,now):
    state=m.latest_state(db);s=m.schedule(now);cutoff=pd.Timestamp(manifest['as_of'])
    days=s.index[(s.index>=cfg['paper_start'])&(s.index<=cutoff)]
    if state['last_date']:days=days[days>pd.Timestamp(state['last_date'])]
    count=0
    for day in days:
        ds=str(day.date());record=db.execute('SELECT created_utc,payload FROM plans WHERE session=?',(ds,)).fetchone()
        if record:
            decision=json.loads(record[1]);status='recorded_plan_executed'
            if pd.Timestamp(record[0])>=s.at[day,'open']:
                authorization=db.execute("SELECT value FROM meta WHERE key=?",('authorized_retrospective_'+ds,)).fetchone()
                if not authorization or decision.get('origin')!='user_authorized_retrospective':raise RuntimeError('Late plan without explicit user authorization')
                status='user_authorized_retrospective_executed'
        else:
            decision=dict(session=ds,buys=[],sells=[]);status='missed_plan_no_retroactive_trades'
        unit_adjustments=[]
        for buy in decision['buys']:
            if 'reference_close' in buy and 'signal_date' in decision:
                original_close=m.session_bar(frames[buy['symbol']],decision['signal_date'])['close']
                scale=buy['reference_close']/original_close
                if scale<=0:raise RuntimeError('Invalid plan price basis')
                buy['atr']/=scale
                if abs(scale-1)>1e-8:unit_adjustments.append(dict(symbol=buy['symbol'],plan_atr_unit_scale=scale))
        needed=set(state['positions'])|{r['symbol'] for r in decision['buys']}
        bars={symbol:m.session_bar(frames[symbol],day) for symbol in needed}
        new,daily,trades,orders,actions=m.execute(state,decision,bars,s.index,cfg['cost_bps'],cfg['sectors'])
        actions.extend(unit_adjustments)
        with db:
            db.execute('INSERT INTO sessions VALUES (?,?,?,?,?)',(ds,m.dumps(daily),m.dumps(new),status,manifest['snapshot_id']))
            db.executemany('INSERT INTO trades(data) VALUES (?)',[(m.dumps(t),) for t in trades])
            db.executemany('INSERT INTO orders(data) VALUES (?)',[(m.dumps(o),) for o in orders])
            db.executemany('INSERT INTO corporate_actions(session,data) VALUES (?,?)',[(ds,m.dumps(a)) for a in actions])
        state=new;count+=1
    return count


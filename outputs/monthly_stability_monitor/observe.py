"""Independent prospective monthly paper ledger. Only immutable pre-open plans trade."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,sqlite3,copy,fcntl
import numpy as np
import pandas as pd
import exchange_calendars as xc
from strategy import selection
from inputs import load_panel,normalized_prices,sha
P=Path(__file__).resolve().parent;ROOT=P.parent.parent;CAL=xc.get_calendar('XNYS')

def encoded(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))
def connect(path):
    db=sqlite3.connect(path)
    db.executescript('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);CREATE TABLE IF NOT EXISTS plans(session TEXT PRIMARY KEY,created_utc TEXT NOT NULL,payload TEXT NOT NULL);CREATE TABLE IF NOT EXISTS daily(session TEXT PRIMARY KEY,payload TEXT NOT NULL);CREATE TABLE IF NOT EXISTS fills(id TEXT PRIMARY KEY,payload TEXT NOT NULL);CREATE TABLE IF NOT EXISTS closed(id TEXT PRIMARY KEY,payload TEXT NOT NULL);')
    return db

def same(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,int)) and not isinstance(a,bool):return isinstance(b,(float,int)) and abs(a-b)<=1e-7+abs(a)*1e-10
    return a==b

def immutable(db,table,key,payload):
    assert table in ['daily','fills','closed'];col='session' if table=='daily' else 'id'
    old=db.execute(f'SELECT payload FROM {table} WHERE {col}=?',(key,)).fetchone()
    if old:
        if not same(json.loads(old[0]),payload):raise ValueError(f'Historical {table} revised:{key}; preserved existing rows')
        return False
    db.execute(f'INSERT INTO {table} VALUES (?,?)',(key,encoded(payload)));return True

def schedule(now):
    today=pd.Timestamp(now).tz_convert('America/New_York').tz_localize(None).normalize()
    sessions=CAL.sessions_in_range(today-pd.Timedelta(days=40),today+pd.Timedelta(days=70))
    done=[d for d in sessions if CAL.session_close(d)+pd.Timedelta(minutes=15)<=pd.Timestamp(now)]
    nextsign=[d for d in sessions if CAL.next_session(d).month!=d.month and CAL.session_open(CAL.next_session(d))-pd.Timedelta(minutes=1)>pd.Timestamp(now)]
    return done[-1],nextsign[0],CAL.next_session(nextsign[0])

def save_plan(db,decision,now,source,registered):
    day=decision['session'];old=db.execute('SELECT payload FROM plans WHERE session=?',(day,)).fetchone()
    if old:return json.loads(old[0])
    signal=pd.Timestamp(decision['signal_date']);fill=pd.Timestamp(day)
    if signal<pd.Timestamp(registered) or CAL.next_session(signal)!=fill or signal.month==fill.month:raise ValueError('Expected monthend signal followed by next-session open')
    if pd.Timestamp(now)<CAL.session_close(signal)+pd.Timedelta(minutes=15):raise ValueError('Signal bar not complete')
    if pd.Timestamp(now)>=CAL.session_open(fill)-pd.Timedelta(minutes=1):raise ValueError('Missed plan deadline; no retrospective entry')
    plan=dict(decision=copy.deepcopy(decision),created_utc=now.isoformat(),origin='prospective_monthend_next_open',source=copy.deepcopy(source))
    db.execute('INSERT INTO plans VALUES (?,?,?)',(day,now.isoformat(),encoded(plan)));return plan

def initial(capital=100000.):return dict(initial=capital,cash=capital,equity=capital,positions={},fees=0.,realized_pnl=0.,last_date=None)

def day_step(state,day,close,opening,decision,cost=.0005):
    state=copy.deepcopy(state);pos=state['positions'];orders=[];trades=[];key=str(day.date())
    if decision is not None:
        weights=decision['target_weights'];n=len(weights)
        if n not in [0,2,3] or (n and any(abs(w-1/n)>1e-12 for w in weights.values())):raise ValueError('Invalid equal target weights')
        symbols=sorted(set(pos)|set(weights))
        if any(not np.isfinite(opening[s]) or opening[s]<=0 for s in symbols):raise ValueError('Missing execution open; no partial substituted basket')
        old={s:p['qty']*float(opening[s]) for s,p in pos.items()};nav=state['cash']+sum(old.values());lo,hi=0.,nav
        def fee(net):return cost*sum(abs(weights.get(s,0)*net-old.get(s,0)) for s in symbols)
        for _ in range(60):
            mid=(lo+hi)/2
            if mid+fee(mid)>nav:hi=mid
            else:lo=mid
        net=(lo+hi)/2
        deltas={s:weights.get(s,0)*net-old.get(s,0) for s in symbols};tolerance=max(1e-8,nav*1e-12)
        for s in symbols:
            delta=deltas[s]
            if delta>=-tolerance:continue
            p=pos[s];price=float(opening[s]);qty=min(p['qty'],-delta/price);fraction=min(1.,qty/p['qty']);basis=p['cost']*fraction;f=qty*price*cost;pnl=qty*price-f-basis
            trades.append(dict(date=key,symbol=s,entry_date=p['entry_date'],quantity=qty,net_pnl=pnl,net_return=pnl/basis,partial=fraction<1-1e-10))
            state['cash']+=qty*price-f;state['fees']+=f;state['realized_pnl']+=pnl
            orders.append(dict(date=key,symbol=s,side='sell',quantity=qty,price=price,fee=f))
            if fraction>=1-1e-10:del pos[s]
            else:p['qty']-=qty;p['cost']-=basis
        for s in symbols:
            delta=deltas[s]
            if delta<=tolerance:continue
            price=float(opening[s]);qty=delta/price;f=delta*cost;outlay=delta+f
            if outlay>state['cash']+nav*1e-10:raise ValueError('Overspending target')
            if s in pos:pos[s]['qty']+=qty;pos[s]['cost']+=outlay
            else:pos[s]=dict(qty=qty,cost=outlay,entry_date=key)
            state['cash']-=outlay;state['fees']+=f;orders.append(dict(date=key,symbol=s,side='buy',quantity=qty,price=price,fee=f))
        if abs(state['cash'])<1e-7:state['cash']=0.
    if any(not np.isfinite(close[s]) or close[s]<=0 for s in pos):raise ValueError('Missing held close; no stale mark')
    state['equity']=state['cash']+sum(p['qty']*float(close[s]) for s,p in pos.items());state['last_date']=key
    unrealized=sum(p['qty']*float(close[s])-p['cost'] for s,p in pos.items())
    assert state['cash']>=-1e-6 and len(pos)<=3
    assert abs(state['equity']-state['initial']-state['realized_pnl']-unrealized)<1e-7*state['equity']
    return state,orders,trades

def initialize():
    config=json.loads((P/'config.json').read_text());freeze=json.loads((P/'FREEZE.json').read_text())
    for file,digest in freeze.items():
        if sha(ROOT/file)!=digest:raise ValueError('Frozen code/config changed:'+file)
    db=connect(P/'paper_ledger.sqlite');old=db.execute("SELECT value FROM meta WHERE key='strategy_id'").fetchone()
    if old and old[0]!=config['strategy_id']:raise ValueError('Ledger strategy mismatch')
    if old is None:
        with db:db.execute('INSERT INTO meta VALUES (?,?)',('strategy_id',config['strategy_id']))
    return config,db

def snapshot(panel,info,day):
    # Immutable input artifact only when a new plan is saved.
    dest=P/'snapshots';dest.mkdir(exist_ok=True)
    digest=hashlib.sha256(encoded(info['source_hashes']).encode()).hexdigest();file=dest/(str(day.date())+'_'+digest[:16]+'.npz')
    if not file.exists():
        cols=panel['adj'].columns;idx=panel['adj'].loc[:day].index
        arrays={k:panel[k].reindex(index=idx,columns=cols).to_numpy() for k in ['adj','open','close','high','low','volume']}
        arrays['member']=panel['member'].reindex(index=idx,columns=cols,fill_value=False).fillna(False).to_numpy(bool)
        arrays['sectors']=panel['sectors'].reindex(index=idx,columns=cols).fillna('').to_numpy(dtype=str)
        arrays['dates']=idx.strftime('%Y-%m-%d').to_numpy(dtype=str);arrays['symbols']=cols.to_numpy(dtype=str)
        np.savez_compressed(file,**arrays)
        file.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    return dict(file=str(file),sha256=sha(file),data_as_of=str(day.date()),source_hashes=info['source_hashes'])

def run(now=None):
    now=now or datetime.now(timezone.utc);config,db=initialize();latest,signal,execution=schedule(now)
    result=dict(strategy_id=config['strategy_id'],registered_utc=config['registered_utc'],generated_utc=now.isoformat(),status='REGISTERED',data_as_of=None,next_rebalance=str(execution.date()),required_signal_date=str(signal.date()),state=initial(config['initial_capital']),statistics=None,plan=None,orders=[],missed_rebalances=[],data_certified=False,units=config['units'])
    try:
        prior=db.execute("SELECT value FROM meta WHERE key='last_result'").fetchone()
        if prior:
            old=json.loads(prior[0]);result.update(state=old['state'],statistics=old['statistics'],valuation_as_of=old.get('valuation_as_of'))
        panel,info=load_panel(config['etfs']);result.update(data_as_of=info['data_as_of'],input_coverage={k:v for k,v in info.items() if k!='source_hashes'})
        if pd.Timestamp(info['data_as_of'])<latest:
            result.update(status='WAITING_CURRENT_PANEL',reason=f'Need current estimatedTop32+theme17inputs through{latest.date()}; cache ends{info["data_as_of"]}. No new plan/fill.')
            return result
        if latest<pd.Timestamp(config['start']):
            result.update(status='REGISTERED_WAITING_START',reason='Observation starts on '+config['start'])
            return result
        dates=panel['adj'].loc[config['start']:latest].index
        expected=CAL.sessions_in_range(config['start'],latest)
        if not dates.equals(expected):raise ValueError('Observation sessions missing from panel')
        plans={day:json.loads(payload) for day,payload in db.execute('SELECT session,payload FROM plans')}
        required=set(s for plan in plans.values() for s in plan['decision']['target_weights'])
        for p in plans.values():
            src=p['source']
            if not Path(src['file']).exists() or sha(src['file'])!=src['sha256']:raise ValueError('Saved plan snapshot changed')
        a=db.execute("SELECT value FROM meta WHERE key='anchors'").fetchone();anchors=json.loads(a[0]) if a else {}
        close,opening,anchors=normalized_prices(panel,anchors,config['start'],required)
        state=initial(config['initial_capital']);daily=[];fills=[];closed=[]
        for day in dates:
            decision=plans.get(str(day.date()),{}).get('decision');state,tt,cc=day_step(state,day,close.loc[day],opening.loc[day],decision,config['round_trip_cost_bps']/20000)
            daily.append(dict(date=str(day.date()),equity=state['equity'],cash=state['cash'],positions=copy.deepcopy(state['positions']),realized_pnl=state['realized_pnl'],fees=state['fees']))
            fills+=tt;closed+=cc
        equity=np.r_[config['initial_capital'],[r['equity'] for r in daily]];dd=1-equity/np.maximum.accumulate(equity)
        result.update(status='OBSERVING',state=state,valuation_as_of=str(latest.date()),statistics=dict(equity=state['equity'],cumulative_return=state['equity']/config['initial_capital']-1,closed_fragments=len(closed),win_rate=sum(t['net_pnl']>0 for t in closed)/len(closed) if closed else None,realized_pnl=state['realized_pnl'],unrealized_pnl=state['equity']-state['initial']-state['realized_pnl'],max_close_drawdown=float(dd.max()),open_positions=len(state['positions'])))
        result['missed_rebalances']=[str(d.date()) for d in dates if CAL.previous_session(d).month!=d.month and CAL.previous_session(d)>=pd.Timestamp(config['start']) and str(d.date()) not in plans]
        decision=None;source=None
        if latest==signal and signal>=pd.Timestamp(config['start']):
            existing=plans.get(str(execution.date()))
            if existing:result['plan']=existing
            elif signal not in panel['member'].index:
                result.update(status='WAITING_CURRENT_POOL',reason='Missing exact current monthend Top32 archive; no forwarded membership')
            else:
                chosen=selection(panel,signal,config['etfs']);decision=dict(session=str(execution.date()),signal_date=str(signal.date()),**chosen)
                source=snapshot(panel,info,signal)
        with db:
            for row in daily:immutable(db,'daily',row['date'],row)
            for i,fill in enumerate(fills):
                if immutable(db,'fills',str(i),fill):result['orders'].append(fill)
            for i,row in enumerate(closed):immutable(db,'closed',str(i),row)
            if decision:result['plan']=save_plan(db,decision,now,source,config['start'])
            db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('anchors',encoded(anchors)))
            db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('last_result',encoded(result)))
        return result
    finally:db.close()

def publish(result):
    (P/'STATUS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    state=result.get('state',{});lines=['月策略观察：长期动量＋低波动状态修正（65.34%研究候选）',f'状态：{result["status"]}；行情截至：{result.get("data_as_of")}',f'月末信号日：{result.get("required_signal_date")}；下一开盘调仓：{result.get("next_rebalance")}',f'独立模拟净值：{state.get("equity",100000):,.2f}美元；持仓数：{len(state.get("positions",{}))}',result.get('reason',''),'当期动态股票池、17ETF、价格均须完整；不借用旧静态池，不补造历史收益。','模拟数量是复权总回报单位，股息再投资近似，不是券商实际股数。','统计：'+encoded(result.get('statistics')),'计划：'+encoded(result.get('plan'))]
    (P/'STATUS.txt').write_text('\n'.join(lines)+'\n')

def main():
    with (P/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:result=run()
        except Exception as e:
            result=dict(status='ERROR_NO_NEW_PLAN',reason=str(e),generated_utc=datetime.now(timezone.utc).isoformat(),plan=None);publish(result);print(encoded(result));raise SystemExit(1)
        publish(result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'dynamic_exit_research'))
from backtest_dynamic import load,features

MODES=['A_open_5d','B_confirm_5d','C_confirm_day']

def load_minutes():
    h=pd.read_csv(P/'qqq_5m.csv.gz',index_col=0)
    h.index=pd.to_datetime(h.index,utc=True).tz_convert('America/New_York')
    assert not h.index.duplicated().any()
    return h.sort_index()

def confirmation(g):
    """State machine consumes completed candles sequentially, never a future close."""
    g=g.between_time('09:30','15:55').sort_index()
    opening=g.between_time('09:30','09:40')
    out={'status':'no_breakout','range_high':np.nan,'breakout_bar':'','retest_bar':'','confirmation_bar':'','entry_timestamp':'','entry_price':np.nan}
    if len(opening)!=3 or list(opening.index.strftime('%H:%M'))!=['09:30','09:35','09:40']:
        out['status']='missing_opening_range';return out
    threshold=float(opening.High.max());out['range_high']=threshold;state='breakout';retest_high=None
    seq=g.between_time('09:45','15:30');expected=opening.index[-1]+pd.Timedelta(minutes=5)
    for ts,row in seq.iterrows():
        if ts!=expected:out['status']='missing_signal_bar';return out
        expected=ts+pd.Timedelta(minutes=5)
        if not np.isfinite(row[['Open','High','Low','Close']].to_numpy(dtype=float)).all():out['status']='invalid_signal_bar';return out
        if state=='breakout':
            if row.Close>threshold:state='retest';out['breakout_bar']=str(ts);out['status']='no_retest'
            continue
        if state=='retest':
            if row.Low<=threshold:
                out['retest_bar']=str(ts)
                if row.Close<threshold:out['status']='failed_first_retest';return out
                retest_high=float(row.High);state='confirm';out['status']='no_rebound_confirmation'
            continue
        if state=='confirm':
            if row.Close<threshold:out['status']='lost_level_after_retest';return out
            if row.Close>retest_high:
                entry=ts+pd.Timedelta(minutes=5);out['confirmation_bar']=str(ts)
                if entry not in g.index:out['status']='missing_entry_bar';return out
                price=float(g.at[entry,'Open'])
                if not np.isfinite(price) or price<=0:out['status']='invalid_entry_open';return out
                out.update(status='confirmed',entry_timestamp=str(entry),entry_price=price)
                return out
    return out

def simulate(h,candidates,events,mode,cost_bps,start,end,initial=100000.):
    cash=initial;pos=None;prev_eq=initial;half=cost_bps/20000;daily=[];trades=[];orders=[];fee_total=0.;div_total=0.
    for i,day in enumerate(h.index):
        if day<pd.Timestamp(start) or day>pd.Timestamp(end):continue
        ds=str(day.date());r=h.loc[day];sold=False;processed_i=i
        if pos:
            income=pos['qty']*r.Dividends;cash+=income;pos['dividends']+=income;div_total+=income
        def sell(price,when):
            nonlocal cash,pos,sold,fee_total
            gross=pos['qty']*price;fee=gross*half;cash+=gross-fee;fee_total+=fee
            pnl=gross-fee+pos['dividends']-pos['outlay']
            trades.append({'entry_date':pos['date'],'entry_timestamp':pos['timestamp'],'exit_date':ds,'exit_time':when,'entry_price':pos['entry_price'],
              'exit_price':price,'qty':pos['qty'],'entry_outlay':pos['outlay'],'entry_fee':pos['fee'],'exit_fee':fee,'dividends':pos['dividends'],
              'net_pnl':pnl,'net_return':pnl/pos['outlay'],'sessions':i-pos['index']})
            orders.append({'date':ds,'time':when,'side':'sell','price':price,'qty':pos['qty'],'fee':fee});pos=None;sold=True
        if pos and mode!='C_confirm_day' and i-pos['index']>=5:sell(float(r.Open),'open')
        if pos is None and not sold and bool(candidates.get(day,False)):
            ev=events[ds]
            if mode=='A_open_5d' or ev['status']=='confirmed':
                price=float(r.Open) if mode=='A_open_5d' else ev['entry_price']
                stamp=ds+' 09:30:00 America/New_York' if mode=='A_open_5d' else ev['entry_timestamp']
                outlay=cash;notional=outlay/(1+half);qty=notional/price;fee=notional*half;cash-=outlay;fee_total+=fee
                pos={'date':ds,'timestamp':stamp,'index':i,'entry_price':price,'qty':qty,'outlay':outlay,'fee':fee,'dividends':0.}
                orders.append({'date':ds,'time':stamp,'side':'buy','price':price,'qty':qty,'fee':fee})
        if pos and mode=='C_confirm_day':sell(float(r.Close),'close')
        invested=pos['qty']*r.Close if pos else 0.;eq=cash+invested
        assert cash>=-1e-7
        daily.append({'date':ds,'equity':eq,'cash':cash,'invested':invested,'return':eq/prev_eq-1,'positions_at_close':int(pos is not None),
          'fees_cumulative':fee_total,'dividends_cumulative':div_total});prev_eq=eq
    openrec=[]
    if pos:openrec=[{'entry_date':pos['date'],'qty':pos['qty'],'value':invested,'marked_pnl':invested+pos['dividends']-pos['outlay'],'sessions':processed_i-pos['index']}]
    cols=['entry_date','entry_timestamp','exit_date','exit_time','entry_price','exit_price','qty','entry_outlay','entry_fee','exit_fee','dividends','net_pnl','net_return','sessions']
    t=pd.DataFrame(trades,columns=cols)
    assert abs(eq-initial-t.net_pnl.sum()-sum(x['marked_pnl'] for x in openrec))<1e-5
    return pd.DataFrame(daily),t,pd.DataFrame(orders,columns=['date','time','side','price','qty','fee']),openrec

def main():
    raw=load_minutes();data=load();h=data['QQQ'];f=features(h,data['SPY'])
    groups={str(day):g for day,g in raw.groupby(raw.index.date)}
    expected=set(pd.date_range('2000-01-01 09:30','2000-01-01 15:55',freq='5min').strftime('%H:%M'))
    complete=[s for s,g in groups.items() if set(g.index.strftime('%H:%M'))==expected]
    start=max(min(complete),str(h.index.min().date()));end=min(max(complete),'2026-09-18')
    candidate=f.legacy.shift(1,fill_value=False)&h.index.to_series().dt.strftime('%Y-%m-%d').isin(complete)
    events={day:confirmation(g) for day,g in groups.items() if day>=start and day<=end}
    diagnostics=[]
    for day in h.index[(h.index>=start)&(h.index<=end)]:
        ds=str(day.date());ev=events.get(ds,{'status':'missing_day'})
        diagnostics.append({'date':ds,'daily_candidate':bool(candidate.loc[day]),'prior_signal_date':str(h.index[h.index.get_loc(day)-1].date()),'daily_open':h.at[day,'Open'],**ev})
    diag=pd.DataFrame(diagnostics);diag.to_csv(P/'candidate_funnel.csv',index=False)
    results=[];ds=[];ts=[];os=[];us=[]
    for mode in MODES:
        for cost in [5,10,20]:
            d,t,o,u=simulate(h,candidate,events,mode,cost,start,end)
            wealth=d.equity/100000;peak=np.maximum.accumulate(np.r_[1.,wealth])[1:]
            results.append({'strategy':mode,'cost_bps':cost,'sessions':len(d),'entries':int((o.side=='buy').sum()),'closed_trades':len(t),
              'wins':int((t.net_return>0).sum()),'win_rate':float((t.net_return>0).mean()) if len(t) else None,
              'cumulative_return':float(wealth.iloc[-1]-1),'max_drawdown':float((wealth/peak-1).min()),
              'worst_trade':float(t.net_return.min()) if len(t) else None,'terminal_open_positions':len(u)})
            for frame in [d,t,o]:frame['strategy']=mode;frame['cost_bps']=cost
            ds.append(d);ts.append(t);os.append(o);us.extend([dict(x,strategy=mode,cost_bps=cost) for x in u])
    pd.DataFrame(results).to_csv(P/'summary.csv',index=False);pd.concat(ds).to_csv(P/'daily_equity.csv',index=False)
    pd.concat(ts).to_csv(P/'trades.csv',index=False);pd.concat(os).to_csv(P/'orders.csv',index=False)
    pd.DataFrame(us).to_csv(P/'open_positions.csv',index=False)
    pairs=[]
    for r in diag[diag.daily_candidate&(diag.status=='confirmed')].itertuples():
        i=h.index.get_loc(pd.Timestamp(r.date));j=i+5
        if j>=len(h):continue
        divs=h.Dividends.iloc[i+1:j+1].sum();sell=h.Open.iloc[j]
        for cost in [5,10,20]:
            fee=cost/20000
            baseline=(sell*(1-fee)+divs)/(r.daily_open*(1+fee))-1
            confirmed=(sell*(1-fee)+divs)/(r.entry_price*(1+fee))-1
            pairs.append({'entry_date':r.date,'exit_date':str(h.index[j].date()),'cost_bps':cost,'opening_entry':r.daily_open,'confirmed_entry':r.entry_price,
              'entry_premium':r.entry_price/r.daily_open-1,'baseline_return':baseline,'confirmed_return':confirmed,'return_difference':confirmed-baseline})
    pd.DataFrame(pairs,columns=['entry_date','exit_date','cost_bps','opening_entry','confirmed_entry','entry_premium','baseline_return','confirmed_return','return_difference']).to_csv(P/'paired_diagnostic.csv',index=False)
    audit={'start':start,'end':end,'sessions':len(diag),'complete_minute_sessions':len(complete),'minute_rows':len(raw),'daily_candidate_days':int(diag.daily_candidate.sum()),
      'candidate_status_counts':diag[diag.daily_candidate].status.value_counts().to_dict(),'all_day_status_counts':diag.status.value_counts().to_dict(),
      'candidate_missing_minute_days':[str(x.date()) for x in f.index[(f.index>=start)&(f.index<=end)&f.legacy.shift(1,fill_value=False)&~f.index.to_series().dt.strftime('%Y-%m-%d').isin(complete)]],
      'annualized':False,'limitations':['Recent sample only','Chosen after previous exploratory research','Bar-open/auction prices are proxies, costs assumed','No stops/targets or added confirmation factors','Portfolio paths differ as missed entries free capacity','Paired diagnostic may overlap; not a portfolio']}
    (P/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2));print(pd.DataFrame(results).to_string(index=False));print(diag[diag.daily_candidate].to_string(index=False))
if __name__=='__main__':main()

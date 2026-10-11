"""Independent daily share-ledger replay, cost accounting, and execution timing controls."""
from search import *

def load_open(px,raw):
    opening=pd.read_csv(O/'index_union/alias_resolved_inputs/opening.csv.gz',index_col=0,parse_dates=True)
    for f in [O/'four_factor_rebound/ETF_SOURCES.json',O/'monthly_reconstruction_v2/ETF_MANIFEST.json',O/'monthly_reconstruction_v2/foreign/ETF_MANIFEST.json']:
        for e in read(f):
            z=pd.read_csv(e['file'],index_col=0,parse_dates=True);opening[e['symbol']]=z.Open.reindex(px.index)
    return opening.reindex(index=px.index,columns=px.columns)*px/raw['close']

def score_of(spec,X,extra=None):
    c,a,b,wc,wa,wb=spec['recipe'];core=X[:,:,c]*wc;side=np.zeros(core.shape)
    if a>=0:side+=X[:,:,a]*wa
    if b>=0:side+=X[:,:,b]*wb
    sc=core+side
    if 'correction' in spec:
        g=spec['gamma'];sc+=g*(extra-side/(1-wc) if wc<1-1e-10 else extra-core)
    return sc

def cases():
    meta=read(P/'INPUTS.json');choices=[]
    for criterion in ['log_later DESC','log_dev DESC','exact DESC,overlap DESC']:
        found=[]
        for key,m in meta.items():
            for suffix in ['','__refined','__corrections']:
                f=P/(key+suffix+'.sqlite')
                if not f.exists():continue
                con=sqlite3.connect(f);con.row_factory=sqlite3.Row
                row=con.execute('SELECT * FROM trials WHERE valid=1 ORDER BY '+criterion+' LIMIT 1').fetchone()
                if row is None:continue
                r=dict(row);r['key']=key;r['source']=str(f);r['criterion']=criterion
                if suffix=='__corrections':
                    plan=read(P/(key+'_CORRECTIONS_PLAN.json'));r['recipe']=plan['seeds'][r['seed']]['recipe'];r['correction']=plan['extra_names'][r['extra']]
                else:r['recipe']=[r[t] for t in ['c','s1','s2','wc','w1','w2']]
                found.append(r);con.close()
        order=(lambda r:(r['exact'],r['overlap'])) if criterion.startswith('exact') else (lambda r:r['log_dev'] if criterion.startswith('log_dev') else r['log_later'])
        best=max(found,key=order);best['name']={'log_later DESC':'return_leader','log_dev DESC':'development_leader','exact DESC,overlap DESC':'holdings_leader'}[criterion];choices.append(best)
    # Keep the strongest unmodified compact formula as a simpler comparison.
    r=max([r for key in meta for suffix in ['','__refined'] for r in one_case(key,suffix)],key=lambda r:r['log_later']);r['name']='simple_leader';choices.append(r)
    dump('SELECTED_CASES.json',choices);return choices

def one_case(key,suffix):
    f=P/(key+suffix+'.sqlite');con=sqlite3.connect(f);con.row_factory=sqlite3.Row
    r=dict(con.execute('SELECT * FROM trials WHERE valid=1 ORDER BY log_later DESC LIMIT 1').fetchone());con.close()
    return [{**r,'key':key,'source':str(f),'recipe':[r[t] for t in ['c','s1','s2','wc','w1','w2']]}]

def normalize(frame,norm):
    if norm=='percentile':return frame.rank(pct=True).fillna(.5)-.5
    lo,hi=frame.quantile(.1),frame.quantile(.9)
    return ((frame-lo)/(hi-lo).replace(0,np.nan)).clip(0,1).fillna(.5)-.5

def orders_from_pack(spec):
    m=read(P/'INPUTS.json')[spec['key']];d=np.load(m['file']);X=d['X'];extra=None
    if 'correction' in spec:
        # Correction library intentionally covers only the supplied-ledger period.
        extra=np.zeros(X.shape[:2]);arr=np.load(P/(spec['key']+'__correction_features.npz'))['X'];extra[d['label_mask']]=arr[:,:,spec['extra']]
    scores=score_of(spec,X,extra);orders=[]
    for i,(date,end) in enumerate(zip(m['dates'],m['ends'])):
        if date<'2020-12-31':continue
        ids=np.flatnonzero(d['eligible'][i]);ids=sorted(ids,key=lambda j:(-scores[i,j],j))[:3]
        if len(ids)<2:ids=[]
        ss=[m['symbols'][i][j] for j in ids];target=[m['symbols'][i][j] for j in np.flatnonzero(d['labels'][i])]
        orders.append(dict(date=date,end=end,symbols=ss,target_in_proxy=target,signal_date=date,scores=[float(scores[i,j]) for j in ids]))
    return orders

def previous_close_orders(spec,px,raw,members,allfeat):
    # Recompute every factor and membership using t-1, not the current-date pack.
    m=read(P/'INPUTS.json')[spec['key']];member=members[m['pool']][0];base=orders_from_pack(spec)
    warm=px.notna().rolling(295,min_periods=295).sum().eq(295)&px.gt(0);states=market_states(px)*2-1
    if spec.get('correction','').startswith('ridge'):
        raise ValueError('Cached ridge forecasts only exist at monthends; t-1 control requires retraining/prediction at t-1.')
    for row in base:
        d=px.index[px.index.get_loc(pd.Timestamp(row['date']))-1];ss=sorted(px.columns[member.loc[d]])
        tab=pd.DataFrame({k:allfeat[k].loc[d,ss] for k in m['feature_names']});q=normalize(tab,m['norm'])
        extra=None
        if 'correction' in spec:
            state,side=spec['correction'].split('__',1);extra=q[side].to_numpy()[None,:]*states.at[d,state]
        scores=score_of(spec,q.to_numpy()[None,:,:],extra)[0]
        good=[i for i,s in enumerate(ss) if warm.at[d,s] and allfeat['mom252_skip42'].at[d,s]>0]
        ids=sorted(good,key=lambda i:(-scores[i],ss[i]))[:3]
        if len(ids)<2:ids=[]
        row.update(symbols=[ss[i] for i in ids],scores=[float(scores[i]) for i in ids],signal_date=str(d.date()))
    return base

def trade(units,cash,prices,chosen,cost,date):
    old={s:q*float(prices[s]) for s,q in units.items()};nav=cash+sum(old.values());symbols=set(old)|set(chosen)
    assert nav>0 and all(np.isfinite(prices[s]) and prices[s]>0 for s in symbols)
    n=len(chosen)
    def turnover(post):return sum(abs((post/n if s in chosen and n else 0)-old.get(s,0)) for s in symbols)
    lo,hi=0.,nav
    for _ in range(60):
        mid=(lo+hi)/2
        if mid+cost*turnover(mid)>nav:hi=mid
        else:lo=mid
    post=(lo+hi)/2;fee=cost*turnover(post)
    new={s:post/n/float(prices[s]) for s in chosen} if n else {};newcash=0. if n else post
    orders=[]
    for s in sorted(symbols):
        delta=new.get(s,0)-units.get(s,0)
        if abs(delta)>1e-9:orders.append(dict(date=str(date.date()),symbol=display(s),side='buy' if delta>0 else 'sell',adjusted_units=abs(delta),adjusted_price=float(prices[s]),dollars=abs(delta)*float(prices[s]),cost=abs(delta)*float(prices[s])*cost))
    assert abs(post+fee-nav)<1e-7*nav
    return new,newcash,fee,orders

def stats(equity,initial=100000.):
    assert equity.notna().all() and (equity>0).all()
    rr=equity.pct_change(fill_method=None).dropna();years=(equity.index[-1]-equity.index[0]).days/365.2425
    wealth=equity.to_numpy()/initial;dd=wealth/np.maximum.accumulate(np.r_[1.,wealth])[1:]-1
    return dict(start=str(equity.index[0].date()),end=str(equity.index[-1].date()),years=years,final_equity=float(equity.iloc[-1]),cagr=float((equity.iloc[-1]/initial)**(1/years)-1),sharpe=float(rr.mean()/rr.std(ddof=1)*np.sqrt(252)) if rr.std(ddof=1)>0 else None,max_drawdown=float(dd.min()),trading_sessions=len(equity),total_return=float(equity.iloc[-1]/initial-1))

def replay(px,opening,orders,cost=.0005,execution='close',start=LABEL_START,end=END):
    dates=px.loc[start:end].index;plan={};units={};cash=100000.;records=[];trades=[];periods=[];totalcost=0.
    for o in orders:
        signal=pd.Timestamp(o['date']);d=signal if execution=='close' else px.index[px.index.get_loc(signal)+1]
        if d<=end:plan[d]=o
    for d in dates:
        if d in plan:
            o=plan[d];prices=px.loc[d] if execution=='close' else opening.loc[d]
            units,cash,fee,tt=trade(units,cash,prices,o['symbols'],cost,d);totalcost+=fee;trades+=tt
            periods.append(dict(signal_date=o['signal_date'],execution_date=str(d.date()),symbols=[display(s) for s in o['symbols']],scores=o['scores'],fee=fee))
        held=px.loc[d,list(units)]
        if len(held):assert np.isfinite(held).all() and held.gt(0).all(),(d,'Missing held close')
        nav=cash+sum(q*px.at[d,s] for s,q in units.items());records.append(nav)
    eq=pd.Series(records,index=dates,name='equity');result=stats(eq);result.update(cost_per_side=cost,execution=execution,fees_paid=totalcost,trade_rows=len(trades),dollar_turnover=sum(t['dollars'] for t in trades))
    annual=[]
    for year in sorted(set(dates.year)):
        yearvals=eq[eq.index.year==year]
        prior=eq[eq.index<yearvals.index[0]]
        if len(yearvals)<2:continue
        initial=float(prior.iloc[-1]) if len(prior) else 100000.
        annual.append(dict(year=int(year),return_=float(yearvals.iloc[-1]/initial-1),partial=year==2026))
    result['annual_returns']=annual
    return result,eq,trades,periods

def main():
    specs=cases();px,raw,members,periods,sources=load_inputs();opening=load_open(px,raw);core,side=features(px,raw);feat={**core,**side};out=[]
    for spec in specs:
        name=spec['name'];orders=orders_from_pack(spec);m=read(P/'INPUTS.json')[spec['key']]
        formula={m['feature_names'][j]:w for j,w in zip(spec['recipe'][:3],spec['recipe'][3:]) if j>=0 and w}
        entry=dict(name=name,spec=spec,base_formula=formula,runs={})
        for mode,cost in [('close',0.),('close',.0005),('open',.0005)]:
            label=mode+('_zero' if cost==0 else '_cost');res,eq,tt,pp=replay(px,opening,orders,cost,mode)
            if label=='close_zero':assert abs(np.log(res['final_equity']/100000.)-spec['log_later'])<1e-10
            entry['runs'][label]=res
            dump(name+'_'+label+'_DAILY.json',[dict(date=str(d.date()),equity=float(v)) for d,v in eq.items()]);dump(name+'_'+label+'_TRADES.json',tt);dump(name+'_'+label+'_HOLDINGS.json',pp)
        if not spec.get('correction','').startswith('ridge'):
            oo=previous_close_orders(spec,px,raw,members,feat);res,eq,tt,pp=replay(px,opening,oo,.0005,'close');entry['runs']['prior_close_signal_cost']=res
            dump(name+'_prior_close_signal_HOLDINGS.json',pp);dump(name+'_prior_close_signal_DAILY.json',[dict(date=str(d.date()),equity=float(v)) for d,v in eq.items()])
        else:entry['prior_close_signal_note']='Not evaluated: monthend-only cached ML forecast cannot be relabeled as t-1.'
        # Development-selected later performance is not pristine OOS: this history has been inspected.
        eq=pd.Series({pd.Timestamp(r['date']):r['equity'] for r in read(P/(name+'_close_cost_DAILY.json'))})
        for s,e,l in [('2020-12-31','2024-12-31','development'),('2024-12-31','2026-09-24','later_check')]:
            z=eq.loc[s:e];entry[l]=stats(z/z.iloc[0]*100000.)
        out.append(entry);print(name,json.dumps({k:{kk:vv for kk,vv in v.items() if kk in ['cagr','sharpe','max_drawdown']} for k,v in entry['runs'].items()}),flush=True)
    benchmark=[]
    for s in ['QQQ','SPY']:
        eq=px.loc[LABEL_START:END,s]/px.at[LABEL_START,s]*100000./1.0005;benchmark.append(dict(symbol=s,**stats(eq),entry_cost=.0005,exit_cost=0))
    dump('VALIDATED_RESULTS.json',dict(cases=out,benchmarks=benchmark,reference='Only user-supplied69ledger periods; original image excluded',note='All strategies and factor combinations researched retrospectively; unknown ETF list and cap snapshot vintages remain proxy-universe limitations. Same-close signal/fill is an execution assumption, controlled with next open and prior close signals.'))
    assert all(sha(f)==h for f,h in sources.items())

if __name__=='__main__':main()

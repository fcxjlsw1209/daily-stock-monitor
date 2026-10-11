"""Fixed-rule, zero-cost Top32/Top35 monthly comparisons; no optimization."""
from pathlib import Path
import json, sys, hashlib
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
O=P.parent
sys.path.insert(0,str(O/'monthly_factor_search_v2'))
from data_inputs import load_inputs, read, sha
import validate as engine
sys.path.insert(0,str(O/'us_listed_top32_yahoo'))
from collect_monthly_rankings import listing_status

WEIGHTS={'mom252_risk_recent63':.5,'vol126':.35,'neg_downside63':.15}

def dump(name,x):
    (P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def actual(s):return {'META':'research_equity_FB','BRK-B':'BRK.B'}.get(s,s)
def display(s):return {'research_equity_FB':'META','BRK.B':'BRK-B'}.get(s,s)

def formula(px):
    r=px.pct_change(fill_method=None)
    mom=px.shift(42)/px.shift(252)-1
    return {'mom252_skip42':mom,'mom252_risk_recent63':mom/r.rolling(63).std().replace(0,np.nan),
            'vol126':r.rolling(126).std(),'neg_downside63':-r.clip(upper=0).pow(2).rolling(63).mean().pow(.5)}

def pick(px,feat,date,symbols):
    ss=sorted(set(map(actual,symbols)))
    assert all(s in px for s in ss),(date,'Missing input columns',set(ss)-set(px.columns))
    tab=pd.DataFrame({n:feat[n].loc[date,ss] for n in WEIGHTS})
    q=engine.normalize(tab,'magnitude')
    scores=q.mul(pd.Series(WEIGHTS)).sum(axis=1)
    history=px.loc[:date,ss].tail(295)
    valid=(history.notna().sum().eq(295)&history.gt(0).all()&feat['mom252_skip42'].loc[date,ss].gt(0))
    chosen=[s for s in scores.sort_values(ascending=False,kind='stable').index if valid[s]][:3]
    if len(chosen)<2:chosen=[]
    return chosen,[float(scores[s]) for s in chosen],dict(eligible=int(valid.sum()),ineligible=[display(s) for s in ss if not valid[s]])

def corrected_published(event,n):
    rows=[r.copy() for r in event['top32']+event['boundary33_40'] if not(r['symbol']=='LTM' and '2020-06-22'<=event['date']<'2024-07-25')]
    assert len({r['symbol'] for r in rows})==len(rows)
    for i,r in enumerate(rows):r['rank']=i+1
    return rows[:n]

def snapshot_cap(z,asof,prices=None):
    """Only backward observations; price drift uses dates no later than asof."""
    prior=z.loc[:asof]
    if prior.empty:return None
    observed=prior.index[-1];cap=float(prior.iloc[-1]);ratio=1.;mode='latest_observed_cap'
    if prices is not None:
        a=prices.loc[:observed].dropna();b=prices.loc[:asof].dropna()
        if len(a) and len(b) and a.iloc[-1]>0 and b.iloc[-1]>0:
            ratio=float(b.iloc[-1]/a.iloc[-1]);mode='prior_cap_times_split_consistent_close_ratio'
    return dict(observation_date=str(observed.date()),asof=str(asof.date()),age_days=int((asof-observed).days),
                observed_cap=cap,price_ratio=ratio,market_cap_usd=cap*ratio,method=mode)

def historical_ranks(px,raw,periods,sources):
    U=O/'us_listed_top32_yahoo';collection=P/'CAP_COLLECTION.json'
    sources[str(collection)]=sha(collection)
    evidencefile=U/'LISTING_EVIDENCE.json';sources[str(evidencefile)]=sha(evidencefile);evidence=read(evidencefile)
    records=read(collection)['records'];caps={};meta={}
    # Deduplicate share classes/aliases by vendor issuer URL before ranking.
    unique={}
    for r in records:
        if r['url'] in unique:
            assert r['symbol'] in {'GOOG','GOOGL','BRK-B','BRK.B','BRK-A'},('Unexpected duplicated issuer',r)
            if r['symbol'] not in {'GOOGL','BRK-B'}:continue
        unique[r['url']]=r
    for r in unique.values():
        assert sha(r['data_file'])==r['sha256']
        sources[r['data_file']]=r['sha256']
        assert sha(r['raw_file'])==r['raw_sha256']
        sources[r['raw_file']]=r['raw_sha256']
        z=pd.read_csv(r['data_file'],index_col=0,parse_dates=True).market_cap_usd
        assert z.index.is_unique and z.index.is_monotonic_increasing and z.gt(0).all()
        caps[r['symbol']]=z;meta[r['symbol']]=r
    bhp=pd.read_csv(P/'prices/BHP.csv',index_col=0,parse_dates=True)
    # Yahoo Close is split adjusted. Restore the historical quotation unit for
    # the known ordinary-share count; reverse future unit adjustments only.
    factors=bhp['Stock Splits'].replace(0,1.)
    after=factors.iloc[::-1].cumprod().iloc[::-1]/factors
    bhp_quote=bhp.Close*after
    rows=[];audit=[];active_union=set();missing_prices=[]
    for period in periods:
        date=pd.Timestamp(period['start']);asof=px.index[px.index.get_loc(date)-1]
        entries=[]
        for s,z in caps.items():
            active,conflict=listing_status(s,str(asof.date()),evidence)
            if not active or (s=='LTM' and '2020-06-22'<=str(asof.date())<'2024-07-25'):continue
            close=raw['close'][actual(s)] if actual(s) in raw['close'] else None
            v=snapshot_cap(z,asof,close)
            if v is None:continue
            v.update(symbol=s,issuer_url=meta[s]['url'],listing_conflict=conflict)
            if s=='SHEL' and asof<pd.Timestamp('2022-01-31'):
                # Vendor archive reports only one old share class. Conservative
                # whole-issuer upper bound is tested below the cutoff, not ranked
                # using an invented exact total.
                v['method']='old_Shell_single_class_cap_times_2.1_upper_bound'
                v['market_cap_usd']*=2.1;v['bounded_not_exact']=True
            if s=='BHP' and asof>=pd.Timestamp('2022-02-01'):
                # Issuer Jan31 announcement: 5,062,323,190 ordinary shares;
                # two ordinary shares per NYSE ADS. Last known count, no future
                # June/September annual filing backfilled into earlier months.
                count=5062323190
                v.update(market_cap_usd=float(bhp_quote.loc[:asof].dropna().iloc[-1])*count/2,
                         method='issuer_2022_01_31_known_shares_times_unadjusted_ADS_close_divided_by_two',
                         shares_known_date='2022-01-31',ordinary_shares=count,ordinary_per_ADS=2)
            entries.append(v)
        entries.sort(key=lambda r:(-r['market_cap_usd'],r['symbol']))
        for rank,r in enumerate(entries,1):r['rank']=rank
        assert len(entries)>=40
        for r in entries[:40]:
            if actual(r['symbol']) not in px or px[actual(r['symbol'])].loc[:asof].dropna().empty:
                missing_prices.append(dict(date=period['start'],**r))
        shell=next((r for r in entries if r['symbol']=='SHEL'),None)
        if shell and shell.get('bounded_not_exact'):
            assert shell['rank']>35,('Shell upper bound overlaps Top35',period['start'],shell)
        assert all(r['age_days']<=40 or 'known_shares' in r['method'] for r in entries[:35]),('Stale Top35',period['start'])
        active_union|={r['symbol'] for r in entries[:35]}
        rows.append(dict(date=period['start'],cap_asof=str(asof.date()),source_kind='reconstructed_prior_observation_archive',ranks=entries[:50]))
        audit.append(dict(date=period['start'],active_cached_issuers=len(entries),cutoff32=entries[31]['market_cap_usd'],cutoff35=entries[34]['market_cap_usd'],
                          shell_rank=shell['rank'] if shell else None,bhp_rank=next((r['rank'] for r in entries if r['symbol']=='BHP'),None),
                          no_price_roll_in_top35=[r['symbol'] for r in entries[:35] if r['method']=='latest_observed_cap']))
    dump('EARLIER_RANKS.json',rows);dump('EARLIER_RANK_AUDIT.json',dict(dates=len(rows),union=sorted(active_union),dates_audit=audit,top40_price_gaps=missing_prices))
    assert not missing_prices,('Missing likely-large-company historical prices; see EARLIER_RANK_AUDIT.json')
    return rows

def independent(px,orders):
    nav=100000.;parts=[]
    for i,o in enumerate(orders):
        z=px.loc[o['date']:o['end'],o['symbols']]
        assert np.isfinite(z.to_numpy()).all() and z.gt(0).all().all()
        e=z.div(z.iloc[0]).mean(axis=1)*nav if len(o['symbols']) else pd.Series(nav,index=z.index)
        parts.append(e if i==0 else e.iloc[1:]);nav=float(e.iloc[-1])
    return pd.concat(parts)

def main():
    px,raw,members,periods,sources=load_inputs()
    for folder in [O/'monthly_published44_reconstruction/prices',P/'prices']:
        for f in folder.glob('*.csv'):
            md=read(f.with_suffix('.source.json'));assert sha(f)==md['sha256'];sources[str(f)]=md['sha256']
            s=f.stem;z=pd.read_csv(f,index_col=0,parse_dates=True)
            if s in px and px[s].notna().any():raise ValueError('Refuse overwrite existing price series '+s)
            px[s]=z['Adj Close'].reindex(px.index)
            raw['close'][s]=z.Close.reindex(px.index)
    publishedfile=O/'us_listed_top32_yahoo/PUBLISHED_MONTHLY_RANKINGS.json';sources[str(publishedfile)]=sha(publishedfile)
    published=read(publishedfile)['periods'];recent_dates={e['date'] for e in published}
    earlier=[p for p in periods if p['start'] not in recent_dates]
    assert len(earlier)==25 and len(published)==44
    early=historical_ranks(px,raw,earlier,sources)
    pools={}
    for n in [32,35]:
        pools[n]={e['date']:[r['symbol'] for r in corrected_published(e,n)] for e in published}
        pools[n].update({e['date']:[r['symbol'] for r in e['ranks'][:n]] for e in early})
        assert len(pools[n])==69 and all(len(set(v))==n for v in pools[n].values())
    allsymbols={actual(s) for pool in pools.values() for ss in pool.values() for s in ss}|members['core17'][1]
    allsymbols|={actual(s) for p in periods for s in p['holdings']}|{'QQQ'}
    assert all(s in px for s in allsymbols)
    px=px[sorted(allsymbols)];feat=formula(px);results={};equities={};details=[];allorders={}
    def evaluate(name,ps,n=None):
        orders=[];match=[];covered=[]
        for p in ps:
            target=set(p['holdings']);stocks=pools[n][p['start']] if n else []
            if n:
                ss,sc,gates=pick(px,feat,p['start'],stocks+list(members['core17'][1]))
                coverage=len(target&(set(stocks)|members['core17'][1]));covered.append(coverage)
            elif name.startswith('author'):ss=list(map(actual,p['holdings']));sc=[];gates={};coverage=3
            else:ss=['QQQ'];sc=[];gates={};coverage=None
            orders.append(dict(date=p['start'],end=p['end'],signal_date=p['start'],symbols=ss,scores=sc))
            hit=len(set(map(display,ss))&target);match.append(hit)
            details.append(dict(model=name,date=p['start'],end=p['end'],selected=list(map(display,ss)),author=p['holdings'],shared=hit,
                                author_pool_coverage=coverage,scores=sc,pool=stocks,**gates))
        if name.startswith('QQQ'):orders=[dict(orders[0],end=ps[-1]['end'])]
        res,eq,trades,holds=engine.replay(px,px,orders,cost=0.,execution='close',start=pd.Timestamp(ps[0]['start']),end=pd.Timestamp(ps[-1]['end']))
        np.testing.assert_allclose(eq,independent(px,orders),rtol=1e-11,atol=1e-6)
        # The unchanged 44-period control begins at Jan31, not at the start
        # of 2023. Preserve returns while correcting that first-year label.
        for annual in res['annual_returns']:
            if annual['year']==pd.Timestamp(ps[0]['start']).year:
                annual['partial']=True
        res.update(exact_months=sum(x==3 for x in match),shared_slots=sum(match),months=len(ps),target_slots=3*len(ps),author_fully_in_pool_months=sum(c==3 for c in covered) if n else None)
        results[name]=res;equities[name]=eq;allorders[name]=orders
        dump(name+'_HOLDINGS.json',holds);dump(name+'_TRADES.json',trades)
        print(name,'cagr',res['cagr'],'sharpe',res['sharpe'],'exact',res['exact_months'],flush=True)
    recent=[p for p in periods if p['start'] in recent_dates]
    for name,ps,n in [('top32_published44',recent,32),('top35_published44',recent,35),('author44',recent,None),
                      ('top32_extended69',periods,32),('top35_extended69',periods,35),('author69',periods,None),('QQQ69',periods,None)]:evaluate(name,ps,n)
    control=read(O/'monthly_published44_reconstruction/DAILY_EQUITY.json')
    for key,old in [('top32_published44','simple_leader__published'),('author44','author__supplied_ledger')]:
        prior=pd.Series({pd.Timestamp(r['date']):r[old] for r in control})
        np.testing.assert_allclose(equities[key],prior,rtol=1e-12,atol=1e-6)
    author=pd.Series({pd.Timestamp(r['date']):r['nav'] for r in read(O/'monthly_supplied_ledger_audit/DAILY_REPLAY.json')})
    np.testing.assert_allclose(equities['author69'],author,rtol=1e-12,atol=1e-6)
    # The recent portion of extended accounts is the same strategy, rebase only.
    for n in [32,35]:
        z=equities['top%d_extended69'%n].loc['2023-01-31':]
        np.testing.assert_allclose(z/z.iloc[0]*100000,equities['top%d_published44'%n],rtol=1e-12,atol=1e-6)
    dump('RESULTS.json',dict(runs=results,cost_per_side=0.,strategy=WEIGHTS,earlier_pool_is_reconstructed=True,independent_out_of_sample=False))
    dump('MONTHLY_COMPARISON.json',details)
    dump('POOLS.json',{str(n):pool for n,pool in pools.items()})
    for suffix in ['44','69']:
        subset={k:v for k,v in equities.items() if k.endswith(suffix)}
        df=pd.DataFrame(subset);dump('DAILY_EQUITY_'+suffix+'.json',[dict(date=str(d.date()),**{k:float(v) for k,v in row.items()}) for d,row in df.iterrows()])
    dump('VALIDATION.json',{'independent_daily_share_ledger_matches':True,'previous_Top32_44_daily_curve_identical':True,'author_69_daily_curve_identical':True,'extended_recent_curve_identical_after_rebasing':True})
    for name in ['PLAN.json','run.py','EVIDENCE.json','merge_cap_cache.py']:
        sources[str(P/name)]=sha(P/name)
    assert all(sha(f)==h for f,h in sources.items())
    dump('SOURCE_HASHES.json',sources)

if __name__=='__main__':main()

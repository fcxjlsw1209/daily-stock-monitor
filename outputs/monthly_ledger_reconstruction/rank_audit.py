"""Read-only audit of cap-ranking boundaries; reproduces cached pool method."""
from reconstruct import *

def cap_panel(px,raw,sources):
    files=[O/'daily_rebuild_research/price_scaled_caps.csv.gz', O/'index_union/alias_resolved_inputs/eligible.csv.gz',O/'index_union/alias_resolved_inputs/MANIFEST.json']
    caps=pd.read_csv(files[0],index_col=0,parse_dates=True).reindex(px.index)
    eligible=pd.read_csv(files[1],index_col=0,parse_dates=True).reindex(px.index).fillna(False).astype(bool)
    issuers=read(files[2])['issuers']
    for f in files:sources[str(f)]=sha(f)
    for s in ['TSM','BABA','NVO','SAP','TM']:
        f=O/'monthly_reconstruction_v2/foreign'/f'{s}_cap.csv';sources[str(f)]=sha(f)
        z=pd.read_csv(f,parse_dates=['date']);z['available']=z.date+pd.Timedelta(days=1)
        join=pd.merge_asof(pd.DataFrame({'session':px.index}),z,left_on='session',right_on='available',direction='backward')
        close=raw['close'][s];known=pd.DataFrame({'price_date':px.index,'price':close}).dropna().reset_index(drop=True)
        den=pd.merge_asof(join.dropna().sort_values('date'),known,left_on='date',right_on='price_date',direction='backward').set_index('session').reindex(px.index)
        good=(pd.Series(px.index,index=px.index)-den.date).dt.days.le(45)&(den.date-den.price_date).dt.days.le(7)
        caps[s]=(den.market_cap_usd*close/den.price).where(good)
        eligible[s]=px[s].notna()
    return caps,eligible,issuers

def ordered(d,caps,eligible,issuers):
    names=[s for s in eligible if s in caps]
    values=caps.loc[d,names].where(eligible.loc[d,names]).dropna().sort_values(ascending=False,kind='stable')
    seen=set();out=[]
    for s in values.index:
        company=issuers.get(s,'FOREIGN_'+s)
        if company not in seen:
            out.append(s);seen.add(company)
    return out

def main():
    px,raw,members,periods,sources=load_inputs();caps,eligible,issuers=cap_panel(px,raw,sources)
    results=[];matches=[]
    for event in periods:
        d=pd.Timestamp(event['start']);ranked=ordered(d,caps,eligible,issuers)
        old=set(members['core17'][0].columns[members['core17'][0].loc[d]])-members['core17'][1]
        matches.append(old==set(ranked[:32]))
        for name in event['holdings']:
            s=actual(name)
            if s in members['core17'][1]:continue
            ix=ranked.index(s)+1 if s in ranked else None
            results.append(dict(date=event['start'],symbol=name,rank=ix,cap=float(caps.at[d,s]) if s in caps and pd.notna(caps.at[d,s]) else None,
                                rank32_symbol=display(ranked[31]),rank32_cap=float(caps.at[d,ranked[31]])))
    assert all(matches),'Cached cap rank method differs'
    dump('CAP_RANK_DIAGNOSTICS.json',results)
    dump('CAP_SOURCE_HASHES.json',sources)
    for r in results:
        if r['rank'] is None or r['rank']>32:print(r)
if __name__=='__main__':main()

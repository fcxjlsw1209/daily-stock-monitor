"""Disclosure-lagged SEC share-count market-cap proxy; never certifies exact ranks."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from prepare import P,ETFS
D=P.parent/'dynamic_top100'

def cap_series(rows,close,raw_per_signal,splits):
    """Use observations only after filing day; transport shares across known splits."""
    out=pd.Series(np.nan,index=close.index);used=pd.Series('',index=close.index,dtype=object)
    rows=rows.copy();rows['end']=pd.to_datetime(rows['end']);rows['filed']=pd.to_datetime(rows['filed'])
    rows=rows[(rows.end<=rows.filed)&rows.val.gt(0)].drop_duplicates(['end','filed','val'])
    # Same reported date / filing with multiple values is an unresolved class/unit conflict.
    conflicts=rows.groupby(['end','filed']).val.nunique();bad=set(conflicts[conflicts>1].index)
    rows=rows[[ (r.end,r.filed) not in bad for r in rows.itertuples() ]]
    rows=rows.sort_values(['end','filed']);split=splits.fillna(1).replace(0,1)
    cumulative=split.cumprod()
    # Ascending reported date ensures later observations supersede older ones,
    # even when an old period is re-filed later.
    for r in rows.itertuples():
        basis=r.filed if getattr(r,"concept","")=="CommonStockSharesOutstanding" else r.end
        base=cumulative.loc[:basis]
        if base.empty:continue
        known=(close.index>r.filed)&(close.index>=r.end)&(close.index<=r.end+pd.Timedelta(days=550))
        days=close.index[known]
        out.loc[days]=r.val*(cumulative.loc[days]/base.iloc[-1])*close.loc[days]*raw_per_signal.loc[days]
        used.loc[days]=str(r.filed.date())
    return out,used

def apply_pool(m,d,out):
    out.mkdir(parents=True,exist_ok=True)
    caps=pd.DataFrame(np.nan,index=d['close'].index,columns=d['close'].columns);sources=[];missing=[]
    # Raw companyfacts CSVs only: avoid silently blessing prior unverified cap scenarios.
    files=list(D.glob('*/*_shares.csv'));byname={}
    for f in files:byname.setdefault(f.stem[:-7],[]).append(f)
    aliases={r['security_id']:r['symbols'] for r in m['alias_decisions']}
    for s in caps:
        if s in ETFS or not d['eligible'][s].any():continue
        issuer=d['issuers'][s];cik=issuer.removeprefix('SEC_CIK_') if issuer.startswith('SEC_CIK_') else ''
        
        if not cik and s in aliases:
            for proof in m['identity_proofs']:
                if proof['symbol'] in aliases[s]:
                    import re
                    match=re.search(r'(\d{10})',proof['source_file'])
                    if match:cik=match.group(1);break
        keys=[s,cik]+aliases.get(s,[]);fs=sorted(set(f for key in keys for f in byname.get(key,[])))
        pieces=[]
        for f in fs:
            try:x=pd.read_csv(f)
            except (pd.errors.EmptyDataError,pd.errors.ParserError):continue
            if not {'concept','val','end','filed'}.issubset(x):continue
            x=x[x.concept.isin(['EntityCommonStockSharesOutstanding','CommonStockSharesOutstanding'])]
            if x.empty:continue
            pieces.append(x);sources.append(dict(symbol=s,file=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        classfile=out/'CLASS_SHARE_EVIDENCE.json'
        if classfile.exists():
            classrows=[r for r in json.loads(classfile.read_text())['rows'] if r['symbol']==s]
            if classrows:
                pieces=[pd.DataFrame(classrows)[['end','filed','val','concept','accn']]]
                sources.extend(dict(symbol=s,file=r['file'],sha256=r['sha256'],source=r['source']) for r in classrows)
        if not pieces:missing.append(s);continue
        x=pd.concat(pieces).drop_duplicates()
        # Prefer aggregate GAAP totals if this issuer has them, avoiding summing classes twice.
        x['_priority']=x.concept.eq('CommonStockSharesOutstanding').astype(int)
        best=x.groupby(['end','filed'])['_priority'].transform('max')
        x=x[x['_priority'].eq(best)]
        for fix in json.loads((D/'pilot_identity/share_corrections.json').read_text()):
            if fix['cik']!=cik:continue
            match=x.end.eq(fix['reported_as_of'])&x.filed.eq(fix['filed'])&x.val.eq(fix['observed_shares'])
            x.loc[match,'val']=fix['corrected_shares']
            if 'corrected_reported_as_of' in fix:x.loc[match,'end']=fix['corrected_reported_as_of']
        # Unresolved unit outliers are excluded, never divided by a guessed multiplier.
        x=x[x.val.le(100e9)]
        cap,dates=cap_series(x,d['close'][s],d['raw_per_signal'][s],d['splits'][s]);caps[s]=cap
        print('cap',s,int(cap.notna().sum()),flush=True)
    mask=d['eligible'].copy();stockcols=[s for s in mask if s not in ETFS]
    # One eligible representative per issuer, ranked using lagged-disclosure cap estimates.
    eligiblecaps=caps.where(mask);selected=pd.DataFrame(False,index=mask.index,columns=mask.columns)
    coverage=[]
    for day in mask.index:
        ranked=eligiblecaps.loc[day,stockcols].dropna().sort_values(ascending=False,kind='stable');seen=set();chosen=[]
        for s in ranked.index:
            issuer=d['issuers'][s]
            if issuer in seen:continue
            seen.add(issuer);chosen.append(s)
            if len(chosen)==32:break
        if len(chosen)==32:selected.loc[day,chosen]=True
        coverage.append(dict(date=str(day.date()),eligible_stocks=int(mask.loc[day,stockcols].sum()),cap_covered=int(ranked.size),selected=len(chosen)))
    selected[ETFS]=mask[ETFS];d['eligible']=selected
    caps.to_csv(out/'cap_estimates.csv.gz');selected.to_csv(out/'membership.csv.gz');pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    audit=dict(scope='estimated_top32_among_historical_union_members_with_available_SEC_share_counts_plus17ETFs',exact_requested_top32_certified=False,missing_share_symbols=missing,sources=sources,assumptions=['Filed date strictly before signal date; no future filings','Latest reported-date unambiguous DEI shares, maximum550calendar days old','Split transport from observation date to signal day; raw close reconstructed from source split basis','Original-filing share corrections applied; unresolved values above100billion shares excluded; aggregateGAAP preferred when available','DEI economic share classes not universally certified; ADR/class and missing-cap exclusions may alter rank','Insufficient32known caps: no stock candidates that day; ETFs remain eligible','Inherited index-union identity, membership, sector and available-price assumptions remain'],original_accounts_modified=False)
    (out/'POOL_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    return d

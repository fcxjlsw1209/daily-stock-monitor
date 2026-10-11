"""Finite, fully logged monthly factor enumeration; historical research only."""
from pathlib import Path
import sys,json,hashlib,sqlite3,time,argparse,itertools
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
from data_inputs import load_inputs,sha,read,actual,display,market_states
START=pd.Timestamp('2016-08-17');END=pd.Timestamp('2026-09-24');LABEL_START=pd.Timestamp('2020-12-31')

def dump(name,obj):
    (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def trend(px,window):
    logp=np.log(px.where(px>0));t=pd.Series(np.arange(len(px)),index=px.index)
    sy=logp.rolling(window).sum()
    cross=logp.mul(t,axis=0).rolling(window).sum()-sy.mul(t-(window-1)/2,axis=0)
    xx=window*(window**2-1)/12
    yy=logp.pow(2).rolling(window).sum()-sy.pow(2)/window
    r2=(cross.pow(2)/(xx*yy.where(yy>1e-12))).clip(0,1)
    return (cross/xx*252*r2).shift(42)

def features(px,raw):
    r=px.pct_change(fill_method=None);v={n:r.rolling(n).std() for n in [21,63,126,252]}
    core={};sides={}
    for n in [126,189,252,294]:
        m=px.shift(42)/px.shift(n)-1;av=r.rolling(n-42).std().shift(42)
        core[f'mom{n}_skip42']=m
        core[f'logmom{n}_skip42']=np.log1p(m)
        core[f'mom{n}_risk_recent63']=m/v[63].replace(0,np.nan)
        core[f'mom{n}_risk_aligned']=m/av.replace(0,np.nan)
        core[f'mom{n}_risk_half']=m/np.sqrt(av.replace(0,np.nan))
        core[f'mean_sharpe{n}_skip42']=r.rolling(n-42).mean().shift(42)/av.replace(0,np.nan)
        core[f'trend_quality{n}_skip42']=trend(px,n-42)
    c,h,l,vol=(raw[k] for k in ['close','high','low','volume'])
    clv=((2*c-h-l)/(h-l).replace(0,np.nan)).where(h!=l,0)
    market=r.SPY
    beta=r.rolling(63).cov(market).div(market.rolling(63).var(),axis=0)
    idio=(v[63].pow(2)-beta.pow(2).mul(market.rolling(63).var(),axis=0)).clip(lower=0).pow(.5)
    basic={**{f'vol{n}':v[n] for n in [21,63,126]},
           'downside63':r.clip(upper=0).pow(2).rolling(63).mean().pow(.5),
           'vol_ratio21_126':v[21]/v[126].replace(0,np.nan),
           'beta63':beta,'idio63':idio,
           'cmf21':(clv*vol).rolling(21).sum()/vol.rolling(21).sum(),
           'cmf63':(clv*vol).rolling(63).sum()/vol.rolling(63).sum(),
           'volume_ratio21_126':vol.rolling(21).mean()/vol.rolling(126).mean(),
           'dollar_volume_ratio21_126':(vol*c).rolling(21).mean()/(vol*c).rolling(126).mean(),
           'signed_volume63':(np.sign(r)*vol).rolling(63).sum()/vol.rolling(63).sum(),
           'close_location21':clv.rolling(21).mean(),
           'ma200_distance':px/px.rolling(200).mean()-1,
           'high63_distance':px/px.rolling(63).max()-1,
           'high252_distance':px/px.rolling(252).max()-1,
           'industry_relative60':(1+px.pct_change(60,fill_method=None))/(1+raw['sector60'])-1,
           'market_relative63':px.pct_change(63,fill_method=None).sub(px.SPY.pct_change(63,fill_method=None),axis=0),
           'acceleration_skip42':px.shift(42)/px.shift(126)-px.shift(126)/px.shift(210),
           'path_efficiency63':r.rolling(63).sum()/r.abs().rolling(63).sum(),
           'positive_day_share126_skip42':r.gt(0).rolling(126).mean().shift(42),
           'jump_concentration63':r.pow(2).rolling(63).max()/r.pow(2).rolling(63).sum(),
           'skew63':r.rolling(63).skew()}
    for name,z in basic.items():
        sides[name]=z.replace([np.inf,-np.inf],np.nan)
        sides['neg_'+name]=-sides[name]
    return {k:z.replace([np.inf,-np.inf],np.nan) for k,z in core.items()},sides

def events(px):
    ends=px.loc[START:END].index.to_series().groupby(px.loc[START:END].index.to_period('M')).last().tolist()
    return sorted(set([START]+[d for d in ends if d<END]))

def prepare():
    px,raw,members,periods,sources=load_inputs();core,sides=features(px,raw)
    ds=events(px);ends=ds[1:]+[END];warm=px.notna().rolling(295,min_periods=295).sum().eq(295)&px.gt(0)
    label={r['start']:set(actual(s) for s in r['holdings']) for r in periods}
    gross=px.loc[ends].to_numpy()/px.loc[ds].to_numpy()
    # A future price availability check is used only to reject an already selected candidate.
    complete=np.stack([px.loc[a:b].notna().all().to_numpy() & px.loc[a:b].gt(0).all().to_numpy() for a,b in zip(ds,ends)])
    packed={};metadata={};allfeat=dict(core,**sides);names=list(allfeat)
    for pool,(member,etfs) in members.items():
        maxn=max(int(member.loc[d].sum()) for d in ds)
        assert maxn==49
        eligible=np.zeros((len(ds),maxn),bool);ratios=np.ones((len(ds),maxn));valid=np.ones_like(eligible);labels=np.zeros_like(eligible)
        rawA=np.full((len(ds),maxn,len(names)),np.nan);symbol_rows=[]
        for k,d in enumerate(ds):
            symbols=sorted(px.columns[member.loc[d]]);symbol_rows.append(symbols)
            ids=px.columns.get_indexer(symbols);n=len(ids)
            eligible[k,:n]=(warm.loc[d,symbols]&core['mom252_skip42'].loc[d,symbols].gt(0)).to_numpy()
            ratios[k,:n]=gross[k,ids];valid[k,:n]=complete[k,ids]
            labels[k,:n]=[s in label.get(str(d.date()),set()) for s in symbols]
            rawA[k,:n]=np.column_stack([allfeat[f].loc[d,symbols].to_numpy() for f in names])
        for norm in ['percentile','magnitude']:
            arr=np.full_like(rawA,0.)
            for k,ss in enumerate(symbol_rows):
                tab=pd.DataFrame(rawA[k,:len(ss)])
                if norm=='percentile':q=tab.rank(pct=True).fillna(.5)-.5
                else:
                    lo,hi=tab.quantile(.1),tab.quantile(.9)
                    q=((tab-lo)/(hi-lo).replace(0,np.nan)).clip(0,1).fillna(.5)-.5
                arr[k,:len(ss)]=q.to_numpy()
            # Deduplicate mathematically identical core factors within this normalization.
            cn=[];si=[];seen={};duplicates={}
            for j in range(len(core)):
                key=hashlib.sha256(arr[:,:,j].tobytes()).hexdigest()
                if key in seen:duplicates[names[j]]=names[seen[key]]
                else:seen[key]=j;cn.append(j)
            for j in range(len(core),len(names)):si.append(j)
            key=pool+'__'+norm
            fname=P/(key+'.npz')
            np.savez_compressed(fname,X=arr,eligible=eligible,ratios=ratios,valid=valid,labels=labels,
                                label_mask=np.array([str(d.date()) in label for d in ds]),core_ids=cn,side_ids=si)
            metadata[key]=dict(file=str(fname),sha256=sha(fname),pool=pool,norm=norm,core_ids=cn,side_ids=si,duplicates=duplicates,
                               symbols=symbol_rows,feature_names=names,dates=[str(d.date()) for d in ds],ends=[str(d.date()) for d in ends],
                               primary_label_months=sum(str(d.date()) in label for d in ds),unavailable_label_count=sum(len(label.get(str(d.date()),set())-set(ss)) for d,ss in zip(ds,symbol_rows)))
            print('PACKED',key,len(cn),len(si),len(ds),flush=True)
    sources={f:h for f,h in sources.items() if '/monthly_curve_reverse/' not in f}
    sources[str(P/'data_inputs.py')]=sha(P/'data_inputs.py')
    dump('INPUTS.json',metadata);dump('SOURCE_HASHES.json',sources)
    return metadata

# Each tuple: core feature, optional two side features and explicit nonnegative weights.
def recipes(meta):
    for c in meta['core_ids']:
        yield(c,-1,-1,1.,0.,0.)
        for s in meta['side_ids']:
            for a in [.1,.2,.3,.4,.5]:yield(c,s,-1,1-a,a,0.)
        for a,b in itertools.combinations(meta['side_ids'],2):
            # Skip direct cancellation of exactly opposite instances of one factor.
            na,nb=meta['feature_names'][a],meta['feature_names'][b]
            if na=='neg_'+nb or nb=='neg_'+na:continue
            for wc,wa,wb in [(.8,.1,.1),(.6,.2,.2),(.5,.25,.25),(.7,.2,.1),(.6,.3,.1)]:
                yield(c,a,b,wc,wa,wb)
                if wa!=wb:yield(c,b,a,wc,wa,wb)

def screen(data,rec,dates,ends):
    X=data['X'];spec=np.asarray(rec);f=spec[:,:3].astype(int);w=spec[:,3:]
    # Vectorize over independent configurations, retaining stable lexical tie order.
    score=np.moveaxis(X[:,:,f[:,0]],2,0)*w[:,0,None,None]
    for j in [1,2]:score+=np.moveaxis(X[:,:,np.maximum(f[:,j],0)],2,0)*w[:,j,None,None]
    elig=data['eligible'];score=np.where(elig[None,:,:],score,-np.inf)
    picks=np.argsort(-score,axis=2,kind='stable')[:,:,:3]
    allowed=np.take_along_axis(np.broadcast_to(elig,score.shape),picks,axis=2)
    count=allowed.sum(axis=2);active=count>=2
    gains=np.take_along_axis(np.broadcast_to(data['ratios'],score.shape),picks,axis=2)
    pathok=np.take_along_axis(np.broadcast_to(data['valid'],score.shape),picks,axis=2)
    valid=((pathok|~allowed).all(axis=2)|~active).all(axis=1)
    period=np.where(active,np.where(allowed,gains,0).sum(axis=2)/np.maximum(count,1),1.)
    valid &=np.isfinite(period).all(axis=1)&(period>0).all(axis=1)
    logs=np.log(np.where(np.isfinite(period)&(period>0),period,1.))
    hits=np.take_along_axis(np.broadcast_to(data['labels'],score.shape),picks,axis=2)&allowed
    match=hits.sum(axis=2)*active
    lm=data['label_mask'];exact=(match[:,lm]==3).sum(axis=1);overlap=match[:,lm].sum(axis=1)
    dev=(np.array(dates)>='2020-12-31')&(np.array(ends)<='2024-12-31');later=np.array(dates)>='2020-12-31'
    cumul=np.cumsum(logs,axis=1);maxima=np.maximum.accumulate(np.column_stack([np.zeros(len(rec)),cumul]),axis=1)[:,1:]
    monthlydd=np.expm1((cumul-maxima).min(axis=1))
    return dict(valid=valid,exact=exact,overlap=overlap,log_full=logs.sum(axis=1),log_dev=logs[:,dev].sum(axis=1),log_later=logs[:,later].sum(axis=1),month_dd=monthlydd,picks=picks,period=period)

def run(key,limit=None):
    meta=read(P/'INPUTS.json')[key];assert sha(meta['file'])==meta['sha256'];data=np.load(meta['file'])
    con=sqlite3.connect(P/(key+'.sqlite'))
    con.execute('CREATE TABLE IF NOT EXISTS trials (id INTEGER PRIMARY KEY,c INTEGER,s1 INTEGER,s2 INTEGER,wc REAL,w1 REAL,w2 REAL,valid INTEGER,exact INTEGER,overlap INTEGER,log_full REAL,log_dev REAL,log_later REAL,month_dd REAL)')
    start=con.execute('SELECT COALESCE(MAX(id),-1)+1 FROM trials').fetchone()[0]
    total=0;batch=[];t=time.monotonic();pending=[]
    for i,r in enumerate(recipes(meta)):
        if i<start:continue
        if limit is not None and i>=limit:break
        batch.append(r);pending.append(i)
        if len(batch)==128:
            z=screen(data,batch,meta['dates'],meta['ends'])
            records=[(j,*rec,int(z['valid'][k]),int(z['exact'][k]),int(z['overlap'][k]),float(z['log_full'][k]),float(z['log_dev'][k]),float(z['log_later'][k]),float(z['month_dd'][k])) for k,(j,rec) in enumerate(zip(pending,batch))]
            con.executemany('INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',records);con.commit();total+=len(batch)
            batch=[];pending=[]
            if total%4096==0:
                best=con.execute('SELECT MAX(log_full),MAX(exact) FROM trials WHERE valid=1').fetchone()
                print('PROGRESS',key,'rows',i+1,'new_elapsed',round(time.monotonic()-t,1),'best_log',best[0],'best_matches',best[1],flush=True)
    if batch:
        z=screen(data,batch,meta['dates'],meta['ends'])
        records=[(j,*rec,int(z['valid'][k]),int(z['exact'][k]),int(z['overlap'][k]),float(z['log_full'][k]),float(z['log_dev'][k]),float(z['log_later'][k]),float(z['month_dd'][k])) for k,(j,rec) in enumerate(zip(pending,batch))]
        con.executemany('INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',records);con.commit();total+=len(batch)
    print('COMPLETE',key,con.execute('SELECT COUNT(*),MAX(exact),MAX(log_full) FROM trials WHERE valid=1').fetchone(),'elapsed',round(time.monotonic()-t,1),flush=True)
    con.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--key');parser.add_argument('--limit',type=int);a=parser.parse_args()
    if a.prepare:prepare()
    if a.key:run(a.key,a.limit)

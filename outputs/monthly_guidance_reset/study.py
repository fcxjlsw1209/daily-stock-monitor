"""Bounded monthly guidance study, isolated from every observation account."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent; O=P.parent
sys.path.insert(0,str(O/'monthly_factor_search_v2'))
from data_inputs import load_inputs, read, sha, market_states
from search import features
import validate as engine
sys.path.insert(0,str(O/'monthly_multilayer_reconstruction'))
from research import fit_weighted_ridge, recency_weights

def dump(name,x):
    (P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def actual(s):return {'META':'research_equity_FB','BRK-B':'BRK.B'}.get(s,s)
def display(s):return {'research_equity_FB':'META','BRK.B':'BRK-B'}.get(s,s)

def load_prices():
    px,raw,members,periods,sources=load_inputs()
    opening=engine.load_open(px,raw)*raw['close']/px
    poolsfile=O/'monthly_top35_extension/POOLS.json';sources[str(poolsfile)]=sha(poolsfile)
    pools=read(poolsfile)
    for folder in [O/'monthly_published44_reconstruction/prices',O/'monthly_top35_extension/prices']:
        for f in folder.glob('*.csv'):
            md=read(f.with_suffix('.source.json'));assert sha(f)==md['sha256'];sources[str(f)]=sha(f)
            s=f.stem;z=pd.read_csv(f,index_col=0,parse_dates=True)
            assert s not in px or not px[s].notna().any()
            px[s]=z['Adj Close'].reindex(px.index)
            for k,c in [('close','Close'),('high','High'),('low','Low'),('volume','Volume')]:raw[k][s]=z[c].reindex(px.index)
            opening[s]=z.Open.reindex(px.index)
    keep=set(members['core17'][0].columns[members['core17'][0].any()])
    keep|={actual(s) for pp in pools.values() for ss in pp.values() for s in ss}
    keep|=members['core17'][1]|members['theme17'][1]
    keep|={'SPY','QQQ','XLE','XLK','XLV','XLF','XLI','XLY','XLU','XLP','XLB','XLRE','XLC'}
    keep=sorted(keep);px=px[keep];raw={k:v.reindex(index=px.index,columns=keep) for k,v in raw.items()}
    for s,b in {'SHEL':'XLE','SKHY':'XLK','BHP':'XLB','HSBC':'XLF','SMFG':'XLF','NVS':'XLV','SNDK':'XLK'}.items():
        raw['sector60'][s]=px[b].pct_change(60,fill_method=None)
    raw['opening']=opening.reindex(index=px.index,columns=keep)
    return px,raw,members,periods,sources,pools

def calendar_momentum(px,months):
    # Calendar monthends, never rounding forward to a future trading session.
    end_dates=px.index.to_series().groupby(px.index.to_period('M')).last()
    mpx=px.loc[end_dates];mpx.index=end_dates.index
    ratios=mpx.shift(2)/mpx.shift(months)-1
    # Map by current calendar month, whose t-2 and t-n anchors are already
    # complete. No dependence on a future date within the current month.
    ans=ratios.reindex(px.index.to_period('M'))
    ans.index=px.index
    return ans

def build_features(px,raw):
    c,s=features(px,raw);r=px.pct_change(fill_method=None)
    v63=r.rolling(63).std();out={};cores=[]
    for n in [252,294]:
        for suffix in ['skip42','risk_recent63','risk_half','risk_aligned']:
            name=f'mom{n}_{suffix}';out[name]=c[name];cores.append(name)
    for n in [12,14]:
        m=calendar_momentum(px,n)
        for tag,z in [('raw',m),('risk',m/v63.replace(0,np.nan)),('half',m/np.sqrt(v63.replace(0,np.nan)))]:
            name=f'calendar{n}_skip2_{tag}';out[name]=z;cores.append(name)
    support=['vol63','vol126','neg_vol63','downside63','neg_downside63','ma200_distance',
             'high252_distance','industry_relative60','acceleration_skip42','path_efficiency63']
    corrections=['cmf21','cmf63','neg_cmf63','signed_volume63','dollar_volume_ratio21_126',
                 'neg_market_relative63','neg_vol_ratio21_126','neg_jump_concentration63']
    out.update({k:s[k] for k in support+corrections})
    # Market-conditioned support remains bounded by the separate correction budget.
    states=market_states(px)*2-1
    for state,feature in [('high_market_vol','neg_cmf63'),('spy_trend','dollar_volume_ratio21_126'),
                          ('tech_relative','industry_relative60'),('high_market_vol','ma200_distance')]:
        name=state+'__'+feature;out[name]=out[feature].mul(states[state],axis=0);corrections.append(name)
    return out,cores,support,corrections,c['mom252_skip42'],states

def pack(feat,dates,symbols,names,norm):
    arr=np.zeros((len(dates),52,len(names)))
    for i,(d,ss) in enumerate(zip(dates,symbols)):
        z=pd.DataFrame({k:feat[k].loc[d,ss] for k in names})
        arr[i,:len(ss)]=engine.normalize(z,norm).to_numpy()
    return arr

def prepare():
    px,raw,members,periods,sources,pools=load_prices()
    feat,cores,support,corrections,mom,states=build_features(px,raw);names=list(feat)
    dates=[p['start'] for p in periods];ends=[p['end'] for p in periods]
    etfs=members['core17'][1]
    warm=px.notna().rolling(295,min_periods=295).sum().eq(295)&px.gt(0)
    metadata={};arrays={}
    for n in [35,32]:
        ss=[sorted({actual(s) for s in pools[str(n)][d]}|etfs) for d in dates]
        elig=np.zeros((69,52),bool);labels=np.zeros_like(elig);gross=np.ones((69,52));complete=np.ones_like(elig)
        for i,(d,e,symbols) in enumerate(zip(dates,ends,ss)):
            nn=len(symbols);elig[i,:nn]=(warm.loc[d,symbols]&mom.loc[d,symbols].gt(0)).to_numpy()
            labels[i,:nn]=[display(s) in periods[i]['holdings'] for s in symbols]
            gross[i,:nn]=(px.loc[e,symbols]/px.loc[d,symbols]).to_numpy()
            complete[i,:nn]=px.loc[d:e,symbols].notna().all().to_numpy()&px.loc[d:e,symbols].gt(0).all().to_numpy()
        for norm in ['percentile','magnitude']:
            key=f'top{n}_{norm}';X=pack(feat,dates,ss,names,norm)
            np.savez_compressed(P/(key+'.npz'),X=X,eligible=elig,labels=labels,gross=gross,complete=complete)
            metadata[key]=dict(symbols=ss)
        arrays[n]=dict(symbols=ss,eligible=elig)
    # Return models get weekly examples, not author holdings; prior five-year cutoffs.
    # Before the new rank archive starts, keep the existing historical Top32 proxy.
    wdates=px.loc['2015-01-01':'2026-08-31'].index.to_series().groupby(px.loc['2015-01-01':'2026-08-31'].index.to_period('W-FRI')).last().tolist()
    wdates=[d for d in wdates if px.index.get_loc(d)+21<len(px)]
    pdates=pd.to_datetime(sorted(pools['35']))
    wss=[];origin=[]
    for d in wdates:
        prior=pdates[pdates<=d]
        if len(prior):
            symbols=sorted({actual(s) for s in pools['35'][str(prior[-1].date())]}|etfs);origin.append('new_Top35_last_available_monthend')
        else:
            m=members['core17'][0];symbols=sorted(m.columns[m.loc[d]]);origin.append('old_historical_Top32_proxy')
        wss.append(symbols)
    # Only raw base factors; regimes enter as two explicit low-dimensional states.
    ml_names=[k for k in names if '__' not in k]
    WX=pack(feat,wdates,wss,ml_names,'percentile')
    Y=np.full((len(wdates),52,2),np.nan);eligible=np.zeros((len(wdates),52),bool);label_ends=[]
    for i,(d,ss) in enumerate(zip(wdates,wss)):
        end=px.index[px.index.get_loc(d)+21];label_ends.append(str(end.date()));nn=len(ss)
        good=(warm.loc[d,ss]&mom.loc[d,ss].gt(0)).to_numpy();eligible[i,:nn]=good
        ret=px.loc[end,ss]/px.loc[d,ss]-1
        Y[i,:nn,0]=(ret-(px.at[end,'SPY']/px.at[d,'SPY']-1)).clip(-1,1)
        Y[i,:nn,1]=ret.where(good).rank(pct=True)-.5
    for name in ['spy_trend','high_market_vol']:
        WX=np.concatenate([WX,np.repeat(states.loc[wdates,name].to_numpy()[:,None,None],52,axis=1)],axis=2)
    monthlyX=pack(feat,dates,arrays[35]['symbols'],ml_names,'percentile')
    for name in ['spy_trend','high_market_vol']:
        monthlyX=np.concatenate([monthlyX,np.repeat(states.loc[dates,name].to_numpy()[:,None,None],52,axis=1)],axis=2)
    np.savez_compressed(P/'ML_INPUTS.npz',X=WX,Y=Y,eligible=eligible,eventX=monthlyX)
    metadata.update(dates=dates,ends=ends,periods=periods,names=names,cores=cores,support=support,corrections=corrections,
                    ml_names=ml_names,weekly_dates=[str(d.date()) for d in wdates],weekly_label_ends=label_ends,weekly_origin=origin,
                    development_mask=[e<='2024-12-31' for e in ends],etfs=sorted(etfs),training_universe_limit='Pre2021 historical proxy; from2021 onward last available reconstructed Top35 monthend. No author holdings in market-return targets.')
    dump('INPUTS.json',metadata);dump('SOURCE_HASHES.json',sources)
    print('PREPARED',len(names),'features',len(wdates),'weekly training dates',flush=True)

def picks(scores,eligible):
    ids=np.argsort(-np.where(eligible,scores,-np.inf),axis=-1,kind='stable')[...,:3]
    good=np.take_along_axis(np.broadcast_to(eligible,scores.shape),ids,axis=-1)
    return ids,good

def train():
    m=read(P/'INPUTS.json');a=np.load(P/'ML_INPUTS.npz');e=np.load(P/'top35_percentile.npz')['eligible']
    configs=[dict(target=t,alpha=al,half_life=h) for t in [0,1] for al in [1.,10.,100.] for h in [0,12]]
    pred=np.zeros((len(configs),69,52));audit=[]
    for year in range(2021,2027):
        ids=[i for i,end in enumerate(m['ends']) if pd.Timestamp(end).year==year]
        cutoff=m['dates'][min(ids)];start=str((pd.Timestamp(cutoff)-pd.DateOffset(years=5)).date())
        training=[i for i,(d,end) in enumerate(zip(m['weekly_dates'],m['weekly_label_ends'])) if start<=d and end<=cutoff]
        XX=[];YY=[];age=[]
        for i in training:
            good=a['eligible'][i]&np.isfinite(a['Y'][i]).all(axis=1)
            XX.append(a['X'][i,good]);YY.append(a['Y'][i,good]);age.extend([(pd.Timestamp(cutoff)-pd.Timestamp(m['weekly_dates'][i])).days/30.4375]*int(good.sum()))
        X=np.vstack(XX);Y=np.vstack(YY);age=np.array(age)
        assert len(X)>1000 and max(m['weekly_label_ends'][i] for i in training)<=cutoff
        for ci,c in enumerate(configs):
            model=fit_weighted_ridge(X,Y[:,c['target']],recency_weights(age,c['half_life']),c['alpha'])
            val=np.einsum('ijk,k->ij',a['eventX'][ids],model['coef'],optimize=False)+model['intercept']
            for j,i in enumerate(ids):
                pred[ci,i,e[i]]=pd.Series(val[j,e[i]]).rank(pct=True).to_numpy()-.5
            audit.append(dict(year=year,config=ci,cutoff=cutoff,train_start=min(m['weekly_dates'][i] for i in training),
                              latest_label_end=max(m['weekly_label_ends'][i] for i in training),rows=len(X),
                              coef=model['coef'].tolist(),intercept=model['intercept'],normal_equation_residual=model['normal_equation_residual']))
        print('TRAINED',year,len(X),'examples',len(configs),'Ridge configurations',flush=True)
    np.savez_compressed(P/'PREDICTIONS.npz',predictions=pred)
    dump('MODEL_CONFIGS.json',configs);dump('TRAINING_AUDIT.json',audit)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','train']);args=ap.parse_args()
    (prepare if args.phase=='prepare' else train)()

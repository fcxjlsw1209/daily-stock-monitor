"""Month-end selection reconstruction; market-return-trained ML, ledger-scored diagnostics.
All writes are confined to this research directory. Does not trade or modify observers.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, itertools, json, sqlite3, sys, time
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent; O=P.parent
sys.path.insert(0,str(O/'monthly_factor_search_v2'))
from data_inputs import load_inputs, read, sha, actual, display, market_states
from search import features


def dump(name,value):
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def protect():
    old=read(O/'monthly_filtered_reconstruction/PROTECTED_BEFORE.json')
    files=[Path(f) for f in old if Path(f).exists()]
    files+=list((O/'monthly_filtered_reconstruction').glob('*'))
    return {str(f):sha(f) for f in files if f.is_file()}

def initialize():
    if not (P/'PLAN.json').exists():
        dump('PROTECTED_BEFORE.json',protect())
        dump('PLAN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
          objective='Reconstruct supplied monthly holdings; no return optimization and no chart inputs.',
          fixed_masks=str(O/'monthly_filtered_reconstruction/PLAN.json'),
          training='Annual refit before first holding month of each year; preceding five years, monthly next-month returns fully matured at cutoff. No author holdings enter model fits.',
          labels=['excess_next_month_vs_SPY_clipped_1','cross_sectional_next_month_rank','excess_div_known_vol63_sqrt21_clipped_5'],
          ridge_alphas=[.1,1.,10.],ridge_recent_half_lives_months=[12,24],
          gbdt=dict(depth=[2,3],max_iter=100,learning_rate=.05,min_samples_leaf=40,l2_regularization=1,early_stopping=False,random_state=1729,weak_gate='SPY close < MA200'),
          scoring='Original fixed candidate pool and gate; core >=50%; total ML <=15%, new layers <=20%; all old/new state+ML+pattern corrections <=20%. Weights come from support, preserving core when possible. Unused weak-market budget returns to support.',
          search='Finite grid of 3 targets x 3 Ridge alphas x 2 recent half-lives x 2 tree depths; base seeds from prior compact leaders, plus pure long momentum. Rank predictions among eligible assets. Original A54/B39 masks frozen.',
          weights=dict(ordinary_ridge=[0,.05,.10],recent_ridge=[0,.025,.05],weak_gbdt=[0,.025,.05],shape=[0,.025,.05]),
          shapes=['all_shapes','volume_shapes','chase_penalty'],
          limitations=['Stock membership/capitalization and sector labels are inherited historical proxies, not certified author universe.',
                       'Exact ETF17, training objective and factor definitions unknown. Model choices are hypotheses.',
                       'Current-close features assume descriptive close-time selection, not proven same-close executable fills.',
                       'No author prior holdings, position P&L, calendar event guesses or intra-month exits used.',
                       'A/B filtered months and hyperparameter search are retrospective diagnostics, not independent OOS.']))


def shape_features(px,raw):
    r=px.pct_change(fill_method=None); c=raw['close']; v=raw['volume']
    # Put OHLC on same adjusted price scale before computing gaps or lookback extrema.
    factor=px/c.where(c>0); h=raw['high']*factor;l=raw['low']*factor;op=raw['opening']*factor
    prev=px.shift(1); spread=(h-l).where(h>l)
    tr=np.maximum(h-l,np.maximum((h-prev).abs(),(l-prev).abs())); atr=tr.rolling(20).mean().replace(0,np.nan)
    relv=v/v.rolling(20).mean().shift(1).replace(0,np.nan)
    gap=(op/prev-1).clip(lower=0)
    location=((px-l)/spread).clip(0,1)
    shake=(l.lt(l.shift(1).rolling(20).min()) & location.gt(.6)).astype(float)*relv.clip(0,4)*location
    compression=(1-(h.rolling(5).max()-l.rolling(5).min())/atr/4).clip(0,1)
    dry=(1-v.rolling(5).mean()/v.rolling(20).mean().shift(1)).clip(0,1)*compression
    failed=(h.gt(prev) & px.lt(prev)).astype(float)*((h-px)/atr).clip(0,3)
    beta=r.rolling(63).cov(r.SPY).div(r.SPY.rolling(63).var(),axis=0).shift(1)
    residual=r-beta.mul(r.SPY,axis=0)
    extreme=r.SPY.lt(r.SPY.rolling(252).quantile(.1).shift(1))
    tail=residual.where(extreme,axis=0).rolling(252,min_periods=10).mean()
    ans=dict(rebound21_atr=(px-l.rolling(21).min())/atr,
             positive_gap5_vol=gap.rolling(5).max()/r.rolling(63).std().replace(0,np.nan),
             shakeout_recovery5=shake.rolling(5).max(),dry_consolidation=dry,
             failed_rebounds5=failed.rolling(5).sum(),market_tail_residual252=tail,
             sector_corr63=r.rolling(63).corr(raw['sector_return']),
             sector_link_stability=-(r.rolling(63).corr(raw['sector_return'])-r.rolling(252).corr(raw['sector_return'])).abs())
    return {k:z.replace([np.inf,-np.inf],np.nan) for k,z in ans.items()}


def prepare():
    initialize();px,raw,members,periods,sources=load_inputs();member,etfs=members['theme17']
    def frame(f):sources[str(f)]=sha(f);return pd.read_csv(f,index_col=0,parse_dates=True,low_memory=False)
    opening=frame(O/'index_union/alias_resolved_inputs/opening.csv.gz')
    for mf in [O/'four_factor_rebound/ETF_SOURCES.json',O/'monthly_reconstruction_v2/ETF_MANIFEST.json',O/'monthly_reconstruction_v2/foreign/ETF_MANIFEST.json']:
        for e in read(mf):
            assert sha(e['file'])==e['sha256']; z=frame(Path(e['file']));opening[e['symbol']]=z.Open.reindex(opening.index)
    sectors=frame(O/'index_union/alias_resolved_inputs/sectors.csv.gz')
    kept=sorted(set(member.columns[member.any()])|{'SPY','QQQ','XLK','XLY','XLV','XLF','XLI','XLE','XLC','XLP','XLU','XLB','XLRE'})
    px=px[kept];raw={k:v.reindex(columns=kept) for k,v in raw.items()};raw['opening']=opening.reindex(index=px.index,columns=kept)
    member=member.reindex(columns=kept,fill_value=False);ret=px.pct_change(fill_method=None)
    mapping={'Technology':'XLK','Consumer Cyclical':'XLY','Healthcare':'XLV','Financial Services':'XLF','Industrials':'XLI','Energy':'XLE','Communication Services':'XLC','Consumer Defensive':'XLP','Utilities':'XLU','Basic Materials':'XLB','Real Estate':'XLRE'}
    sectorret=pd.DataFrame(np.nan,index=px.index,columns=kept)
    for s in kept:
        if s in etfs:sectorret[s]=ret.SPY
        elif s in {'TSM','BABA','NVO','SAP','TM'}:sectorret[s]=ret[{'TSM':'XLK','BABA':'XLY','NVO':'XLV','SAP':'XLK','TM':'XLY'}[s]]
        elif s in sectors:
            for cat,benchmark in mapping.items():sectorret.loc[sectors[s].eq(cat),s]=ret.loc[sectors[s].eq(cat),benchmark]
    raw['sector_return']=sectorret
    core,side=features(px,raw);new=shape_features(px,raw)
    # Avoid exact duplicate log-momentum ranks and sign-inverted copies in ML inputs.
    core_names=[f'mom{n}_{suffix}' for n in [189,252,294] for suffix in ['skip42','risk_recent63','risk_aligned','risk_half']]
    names=core_names+[n for n in side if not n.startswith('neg_')]+list(new)
    allfeat={**core,**side,**new};states=market_states(px)*2-1
    warm=px.notna().rolling(295,min_periods=295).sum().eq(295)&px.gt(0)
    allends=px.index.to_series().groupby(px.index.to_period('M')).last().tolist()
    # Last source month may be partial; never use it as a model training label end.
    allends=allends[:-1]
    dates=[d for d in allends if d>=pd.Timestamp('2014-01-01')]
    vol63=ret.rolling(63).std();spyma200=px.SPY.rolling(200).mean()
    records=read(O/'monthly_filtered_reconstruction/PLAN.json')['records'];event_dates=[r['date'] for r in records]
    N=len(dates);M=49;D=len(names);X=np.zeros((N,M,D*5));eligible=np.zeros((N,M),bool);ys=np.full((N,M,3),np.nan)
    shape=np.zeros((N,M,3));symbols=[];missings=[];weak=np.zeros(N,bool)
    for i,d in enumerate(dates):
        ss=sorted(member.columns[member.loc[d]]);n=len(ss);assert n<=M,(d,n);symbols.append(ss)
        values=pd.DataFrame({k:allfeat[k].loc[d,ss] for k in names})
        q=values.rank(pct=True).fillna(.5)-.5;base=q.to_numpy()
        X[i,:n]=np.column_stack([base]+[base*states.at[d,k] for k in states])
        eligible[i,:n]=(warm.loc[d,ss]&core['mom252_skip42'].loc[d,ss].gt(0)).to_numpy()
        missings.append(dict(date=str(d.date()),missing_by_feature=values.loc[eligible[i,:n]].isna().sum().astype(int).to_dict()))
        weak[i]=px.at[d,'SPY']<spyma200.at[d]
        chase=(-q.rebound21_atr-q.positive_gap5_vol)/2
        volume=(q.shakeout_recovery5+q.dry_consolidation-q.failed_rebounds5)/3
        shape[i,:n]=np.column_stack([(2*chase+3*volume)/5,volume,chase])
        j=allends.index(d)
        if j+1<len(allends):
            end=allends[j+1];rr=px.loc[end,ss]/px.loc[d,ss]-1;excess=rr-(px.at[end,'SPY']/px.at[d,'SPY']-1)
            rank=rr.where(eligible[i,:n]).rank(pct=True)-.5
            ys[i,:n,0]=excess.clip(-1,1);ys[i,:n,1]=rank
            ys[i,:n,2]=(excess/(vol63.loc[d,ss]*np.sqrt(21))).clip(-5,5)
    ds=[str(d.date()) for d in dates];ends=[str(allends[allends.index(d)+1].date()) if allends.index(d)+1<len(allends) else None for d in dates]
    events=[ds.index(d) for d in event_dates]
    old=read(O/'monthly_factor_search_v2/INPUTS.json')['theme17__percentile'];a=np.load(old['file']);sel=a['label_mask']
    assert np.array_equal(eligible[events],a['eligible'][sel]);assert [symbols[i] for i in events]==[s for s,b in zip(old['symbols'],sel) if b]
    for f in [O/'monthly_filtered_reconstruction/PLAN.json',O/'monthly_factor_search_v2/INPUTS.json',Path(old['file'])]:sources[str(f)]=sha(f)
    np.savez_compressed(P/'FEATURES.npz',X=X,eligible=eligible,targets=ys,shape=shape,weak=weak)
    dump('FEATURES.json',dict(dates=ds,ends=ends,symbols=symbols,names=names,expanded_names=names+[state+'__'+n for state in states for n in names],events=events,records=records,shape_names=['all_shapes','volume_shapes','chase_penalty'],weak_events=int(weak[events].sum()),data_last_date=str(px.index[-1].date())))
    dump('MISSING_FEATURES.json',missings);dump('SOURCE_HASHES.json',sources)
    dump('FEATURE_DESCRIPTIONS.json',dict(rebound21_atr='(adjusted close - minimum adjusted low21) / trailing ATR20',positive_gap5_vol='max positive adjusted open/prior adjusted close gap5 / daily volatility63',shakeout_recovery5='max5[low < prior low20 AND close location>0.6] * volume/prior mean20 volume capped4 * close location',dry_consolidation='max(0,1-volmean5/prior volmean20)*clip(1-range5/(4*ATR20),0,1)',failed_rebounds5='sum5[high>prior close AND close<prior close] * clip((high-close)/ATR20,0,3)',market_tail_residual252='mean252 of return minus previous-day beta63*SPY return, only SPY days below previous-day trailing252 10% quantile; >=10 events',sector_corr63='corr63(stock, inherited sector ETF proxy)',sector_link_stability='negative absolute difference corr63 versus corr252',neutral_missing='Missing factor values are neutral centered rank0; their counts are separately reported.',calendar='Not tested: exact monthly event specification and historical event vintages are not known.',profit='Not tested: own-position profit would require recursive portfolio state; source author holdings never used as feature.'))
    print('PREPARED',N,'months',D,'features',X.shape[-1],'expanded; weak author months',weak[events].sum(),flush=True)


def training_rows(meta,arr,cutoff):
    start=str((pd.Timestamp(cutoff)-pd.DateOffset(years=5)).date())
    rows=[i for i,(a,b) in enumerate(zip(meta['dates'],meta['ends'])) if a>=start and b is not None and b<=cutoff]
    X=[];Y=[];age=[];orig=[]
    for i in rows:
        keep=arr['eligible'][i]&np.isfinite(arr['targets'][i]).all(axis=1)
        X.append(arr['X'][i,keep]);Y.append(arr['targets'][i,keep]);orig.extend([(i,j) for j in np.flatnonzero(keep)])
        age.extend([(pd.Timestamp(cutoff)-pd.Timestamp(meta['dates'][i])).days/30.4375]*int(keep.sum()))
    return np.vstack(X),np.vstack(Y),np.array(age),dict(train_start=meta['dates'][min(rows)],train_end=max(meta['ends'][i] for i in rows),months=len(rows),examples=len(age),rows=orig)


def recency_weights(age,half_life):
    w=np.ones_like(age) if half_life==0 else 2**(-age/half_life)
    return w/w.mean()


def fit_weighted_ridge(X,y,weights,alpha):
    # Algebraically identical to sklearn Ridge(fit_intercept=True), avoiding
    # spurious native matmul FP flags observed on this host's NumPy/BLAS build.
    xm=np.einsum('ij,i->j',X,weights,optimize=False)/weights.sum()
    ym=float(np.sum(y*weights)/weights.sum());xc=X-xm;yc=y-ym
    gram=np.einsum('ij,ik,i->jk',xc,xc,weights,optimize=False)+alpha*np.eye(X.shape[1])
    rhs=np.einsum('ij,i,i->j',xc,yc,weights,optimize=False)
    coef=np.linalg.solve(gram,rhs);intercept=ym-float(np.sum(xm*coef))
    residual=float(np.max(np.abs(np.einsum('ij,j->i',gram,coef,optimize=False)-rhs)))
    assert residual<1e-7 and np.isfinite(coef).all()
    return dict(coef=coef,intercept=intercept,normal_equation_residual=residual)


def centered_rank(pred,eligible):
    result=np.zeros_like(pred)
    for i in range(len(pred)):
        result[i,eligible[i]]=pd.Series(pred[i,eligible[i]]).rank(pct=True).to_numpy()-.5
    return result


def train():
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    import sklearn, joblib
    m=read(P/'FEATURES.json');a=np.load(P/'FEATURES.npz');events=m['events'];E=len(events)
    configs=[dict(kind='ridge',target=t,alpha=alpha,half_life=h) for t in range(3) for alpha in [.1,1.,10.] for h in [0,12,24]]
    configs += [dict(kind='gbdt',target=t,depth=d) for t in range(3) for d in [2,3]]
    pred=np.zeros((len(configs),E,49));audit=[];models=P/'models';models.mkdir(exist_ok=True)
    years=sorted({pd.Timestamp(r['end']).year for r in m['records']})
    for year in years:
        event_ids=[i for i,r in enumerate(m['records']) if pd.Timestamp(r['end']).year==year]
        cutoff=min(m['records'][i]['date'] for i in event_ids);X,Y,age,info=training_rows(m,a,cutoff)
        assert info['months']>=36 and info['examples']>500 and info['train_end']<=cutoff
        evX=a['X'][np.array(events)[event_ids]];flat=evX.reshape(-1,evX.shape[-1]);t=time.monotonic()
        for ci,c in enumerate(configs):
            if c['kind']=='ridge':
                w=recency_weights(age,c['half_life']);model=fit_weighted_ridge(X,Y[:,c['target']],w,c['alpha'])
                values=(np.einsum('ij,j->i',flat,model['coef'],optimize=False)+model['intercept']).reshape(len(event_ids),49)
            else:
                model=HistGradientBoostingRegressor(max_depth=c['depth'],max_leaf_nodes=2**c['depth'],max_iter=100,learning_rate=.05,min_samples_leaf=40,l2_regularization=1,early_stopping=False,random_state=1729);w=np.ones(len(Y))
                model.fit(X,Y[:,c['target']],sample_weight=w)
                values=model.predict(flat).reshape(len(event_ids),49)
            assert np.isfinite(values).all()
            pred[ci,event_ids]=values
            f=models/f'{year}_{ci}.joblib';joblib.dump(model,f)
            audit.append(dict(year=year,config_id=ci,config=c,cutoff=cutoff,**{k:v for k,v in info.items() if k!='rows'},training_X_sha256=hashlib.sha256(X.tobytes()).hexdigest(),training_y_sha256=hashlib.sha256(Y[:,c['target']].tobytes()).hexdigest(),sample_weight_min=float(w.min()),sample_weight_max=float(w.max()),model_file=str(f),model_sha256=sha(f)))
        print('TRAINED',year,len(configs),'models',info['examples'],'examples',round(time.monotonic()-t,1),'seconds',flush=True)
        dump('TRAINING_AUDIT.json',audit)
    ranks=np.stack([centered_rank(z,a['eligible'][events]) for z in pred])
    np.savez_compressed(P/'PREDICTIONS.npz',raw=pred,ranks=ranks)
    dump('MODEL_CONFIGS.json',dict(sklearn_version=sklearn.__version__,configs=configs,predictions_sha256=sha(P/'PREDICTIONS.npz')))


def load_eval():
    m=read(P/'FEATURES.json');a=np.load(P/'FEATURES.npz');metadata=read(O/'monthly_factor_search_v2/INPUTS.json');packs={}
    for key in ['theme17__percentile','theme17__magnitude']:
        mm=metadata[key];assert sha(mm['file'])==mm['sha256'];z=np.load(mm['file']);mask=z['label_mask']
        packs[key]=dict(X=z['X'][mask],extra=np.load(O/'monthly_factor_search_v2'/(key+'__correction_features.npz'))['X'],meta=mm)
    ss=[m['symbols'][i] for i in m['events']]
    labels=np.array([[s in {actual(t) for t in r['target']} for s in symbols] for r,symbols in zip(m['records'],ss)])
    masks={'all':np.ones(len(ss),bool),'A':np.array([r['included_A'] for r in m['records']]),'B':np.array([r['included_B'] for r in m['records']]),'dev':np.array([r['end']<'2025-01-01' for r in m['records']]),'later':np.array([r['end']>='2025-01-01' for r in m['records']])}
    return dict(meta=m,eligible=a['eligible'][m['events']],symbols=ss,labels=labels,masks=masks,packs=packs,weak=a['weak'][m['events']],shapes=a['shape'][m['events']])


def seeds(data):
    result=[];seen=set()
    def add(r):
        keep={k:r[k] for k in ['key','recipe','extra','gamma'] if k in r};key=json.dumps(keep,sort_keys=True)
        if key in seen:return
        seen.add(key);X=data['packs'][r['key']]['X'];c,s1,s2,wc,w1,w2=r['recipe']
        core=wc*X[:,:,c];support=w1*X[:,:,max(s1,0)]+w2*X[:,:,max(s2,0)];small=0.
        if 'extra' in r:
            g=r['gamma'];donor=support/(1-wc) if wc<1 else core
            extra=data['packs'][r['key']]['extra'][:,:,r['extra']]
            if wc<1:support=support-g*donor+g*extra
            else:core=(1-g)*core;wc=1-g;support=g*extra
            small=g
        result.append(dict(id=len(result),rule=keep,core=core,support=support,core_weight=wc,small_weight=small))
    cs=read(O/'monthly_filtered_reconstruction/COMPACT_LEADERS.json')
    for key in data['packs']:
        for sl in ['A','B']:
            for r in [r for r in cs if r['key']==key and r['slice']==sl][:10]:add(r)
    for r in read(O/'monthly_filtered_reconstruction/CORRECTION_LEADERS.json'):add(r)
    for key in data['packs']:
        for c in [7,14,21]:add(dict(key=key,recipe=[c,-1,-1,1.,0.,0.]))
    return result


def compose(seed,ridge,recent,tree,shape,weak,weights):
    wr,wt,wg,ws=weights;budget=wr+wt+wg*weak[:,None]+ws
    new=wr*ridge+wt*recent+wg*weak[:,None]*tree+ws*shape
    wc=seed['core_weight']
    if wc<1:return seed['core']+seed['support']*(1-budget/(1-wc))+new
    return seed['core']*(1-budget)+new


def allowed(seed,weights):
    wr,wt,wg,ws=weights;b=sum(weights);wc=seed['core_weight']
    return wr+wt+wg<=.15+1e-9 and b<=.20+1e-9 and (wc==1 or b<=1-wc+1e-9) and (wc<1 or 1-b>=.5) and (seed['small_weight']*(1-b/(1-wc)) if wc<1 else 0)+b<=.20+1e-9


def metrics(data,scores,details=False):
    one=scores.ndim==2
    if one:scores=scores[None]
    order=np.argsort(-np.where(data['eligible'][None],scores,-np.inf),axis=2,kind='stable')
    picks=order[:,:,:3];ok=np.take_along_axis(np.broadcast_to(data['eligible'],scores.shape),picks,axis=2)
    count=ok.sum(axis=2);hit=(np.take_along_axis(np.broadcast_to(data['labels'],scores.shape),picks,axis=2)&ok).sum(axis=2);hit=np.where(count>=2,hit,0)
    out={k:np.column_stack([(hit[:,v]==3).sum(axis=1),hit[:,v].sum(axis=1)]).astype(int) for k,v in data['masks'].items()}
    if not details:return {k:v[0] for k,v in out.items()} if one else out
    assert one
    rows=[]
    for i,r in enumerate(data['meta']['records']):
        chosen=[display(data['symbols'][i][j]) for j,o in zip(picks[0,i],ok[0,i]) if o] if count[0,i]>=2 else []
        ranks={display(s):int(np.flatnonzero(order[0,i]==j)[0])+1 for j,s in enumerate(data['symbols'][i]) if actual(s) in {actual(t) for t in r['target']}}
        rows.append(dict(date=r['date'],end=r['end'],target=r['target'],selected=chosen,shared=int(hit[0,i]),exact=bool(hit[0,i]==3),included_A=r['included_A'],included_B=r['included_B'],weak=bool(data['weak'][i]),target_ranks=ranks))
    return dict(metrics={k:dict(periods=int(data['masks'][k].sum()),exact=int(v[0,0]),shared=int(v[0,1])) for k,v in out.items()},details=rows)


def search():
    data=load_eval();ss=seeds(data);conf=read(P/'MODEL_CONFIGS.json')['configs'];pred=np.load(P/'PREDICTIONS.npz')['ranks'];grid=list(itertools.product([0,.05,.10],[0,.025,.05],[0,.025,.05],[0,.025,.05]))
    dump('SEEDS.json',[{k:v for k,v in s.items() if k not in ['core','support']} for s in ss])
    bundles=[]
    for t,alpha,h,depth in itertools.product(range(3),[.1,1.,10.],[12,24],[2,3]):
        ids=[conf.index(dict(kind='ridge',target=t,alpha=alpha,half_life=hh)) for hh in [0,h]]+[conf.index(dict(kind='gbdt',target=t,depth=depth))]
        bundles.append(dict(target=t,alpha=alpha,half_life=h,depth=depth,model_ids=ids))
    dump('BUNDLES.json',bundles)
    path=P/'TRIALS.sqlite'
    if path.exists():raise RuntimeError('TRIALS.sqlite already exists; use report to inspect without overwriting research history.')
    con=sqlite3.connect(path);con.execute('CREATE TABLE trials (id INTEGER PRIMARY KEY,seed INTEGER,bundle INTEGER,shape INTEGER,wr REAL,wt REAL,wg REAL,ws REAL,all_exact INTEGER,all_shared INTEGER,A_exact INTEGER,A_shared INTEGER,B_exact INTEGER,B_shared INTEGER,dev_exact INTEGER,dev_shared INTEGER,later_exact INTEGER,later_shared INTEGER)')
    total=0;t0=time.monotonic();seen=set()
    for s in ss:
        scores=[];spec=[]
        def flush():
            nonlocal total,scores,spec
            if not scores:return
            met=metrics(data,np.stack(scores));rows=[]
            for k,(bi,shapeid,w) in enumerate(spec):rows.append((total+k,s['id'],bi,shapeid,*w,*[int(q) for sl in ['all','A','B','dev','later'] for q in met[sl][k]]))
            con.executemany('INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',rows);con.commit();total+=len(rows);scores=[];spec=[]
        # Include all-zero ablation and deduplicate weight-zero hyperparameter variants.
        for bi,b in enumerate(bundles):
            for shapeid in range(3):
                for w in grid:
                    if not allowed(s,w):continue
                    ids=b['model_ids'];key=(s['id'],ids[0] if w[0] else -1,ids[1] if w[1] else -1,ids[2] if w[2] else -1,shapeid if w[3] else -1,*w)
                    if key in seen:continue
                    seen.add(key)
                    scores.append(compose(s,*pred[ids],data['shapes'][:,:,shapeid],data['weak'],w));spec.append((bi,shapeid,w))
                    if len(scores)>=128:flush()
        flush()
        if s['id']%5==0:print('SEARCH',s['id']+1,'/',len(ss),'trials',total,'seconds',round(time.monotonic()-t0,1),flush=True)
    con.close();dump('SEARCH_STATUS.json',dict(trials=total,seeds=len(ss),bundles=len(bundles),seconds=time.monotonic()-t0,status='completed'));report()


def report():
    data=load_eval();ss=seeds(data);bundles=read(P/'BUNDLES.json');pred=np.load(P/'PREDICTIONS.npz')['ranks'];con=sqlite3.connect(P/'TRIALS.sqlite');con.row_factory=sqlite3.Row
    def result(row,override=None):
        row=dict(row);w=[row[k] for k in ['wr','wt','wg','ws']] if override is None else override
        s=ss[row['seed']];b=bundles[row['bundle']];sc=compose(s,*pred[b['model_ids']],data['shapes'][:,:,row['shape']],data['weak'],w)
        return dict(trial=row,rule=s['rule'],core_weight=s['core_weight'],bundle=b,weights=w,shape=data['meta']['shape_names'][row['shape']],**metrics(data,sc,True))
    leaders={}
    for sl in ['all','A','B','dev']:
        row=con.execute(f'SELECT * FROM trials ORDER BY {sl}_exact DESC,{sl}_shared DESC,(wr+wt+wg+ws),id LIMIT 1').fetchone();leaders[sl]=result(row)
    ab=[]
    row=leaders['A']['trial'];w=leaders['A']['weights']
    for label,ww in [('base',[0,0,0,0]),('base+ordinary',[w[0],0,0,0]),('base+ordinary+recent',[w[0],w[1],0,0]),('base+ordinary+recent+weak_tree',[w[0],w[1],w[2],0]),('full',w)]:ab.append(dict(label=label,**result(row,ww)))
    # Independent family comparisons are tuning diagnostics, not causal ablations.
    families={}
    for label,where in [('base','wr=0 AND wt=0 AND wg=0 AND ws=0'),('ordinary_ridge_only','wt=0 AND wg=0 AND ws=0'),('recent_ridge_only','wr=0 AND wg=0 AND ws=0'),('weak_tree_only','wr=0 AND wt=0 AND ws=0'),('shape_only','wr=0 AND wt=0 AND wg=0'),('no_tree','wg=0'),('no_shape','ws=0'),('full_architecture','wr>0 AND wt>0 AND wg>0 AND ws>0')]:
        r=con.execute(f'SELECT * FROM trials WHERE {where} ORDER BY A_exact DESC,A_shared DESC,(wr+wt+wg+ws),id LIMIT 1').fetchone();families[label]=result(r)
    dump('LEADERS.json',leaders);dump('ABLATIONS.json',ab);dump('FAMILY_LEADERS.json',families)
    baseline=read(O/'monthly_filtered_reconstruction/CORRECTION_LEADERS.json')[0]
    lines=['月策略多层模型：选股复现研究','', '口径：模型按年用此前五年已成熟的下月收益训练；作者三只持仓仅用于事后评估/选参数，不参与模型训练。固定A54、B39，不再追加删月。不读收益图，不按回报选优。','', '|规则|全69期完全一致|A54期完全一致|B39期完全一致|A中位置命中/162|','|---|---:|---:|---:|---:|']
    old=baseline['evaluations'];lines.append(f"|前轮简洁4因子|{old['all']['exact_periods']}|{old['A']['exact_periods']}|{old['B']['exact_periods']}|{old['A']['shared_positions']}|")
    for label,l in [('本轮A选择',leaders['A']),('本轮B选择',leaders['B']),('本轮全69选择',leaders['all']),('2021–2024选择',leaders['dev'])]:
        m=l['metrics'];lines.append(f"|{label}|{m['all']['exact']}|{m['A']['exact']}|{m['B']['exact']}|{m['A']['shared']}|")
    lines+=['','A选择的完整参数：',json.dumps({k:leaders['A'][k] for k in ['rule','core_weight','bundle','weights','shape']},ensure_ascii=False,indent=2),'','同一基底、同一模型参数的逐层消融：']
    for l in ab:lines.append(f"- {l['label']}: all {l['metrics']['all']['exact']}/69; A {l['metrics']['A']['exact']}/54, B {l['metrics']['B']['exact']}/39; A位置{l['metrics']['A']['shared']}/162")
    lines+=['','限制与未实现项：','- 同一总权重预算，长期动量至少50%；普通/近期Ridge/弱市GBDT总权重至多15%。市场状态和新增量价修正合计最多20%。强市不使用树修正，预算返还基底。','- 新加入8种量价/相对行业/极端市场特征。弱市定义SPY低于200日均线。树模型不是只用熊市样本训练，而是全历史训练、弱市生效。','- 长假、选举、FOMC及当前持仓浮盈未加入：没有证据确认它们属于月策略，也没有足够的原始定义。未把日策略回撤资格、日内退出搬进月策略。','- 股票池/行业历史是既有代理口径，作者ETF17未知；本轮不改池，以隔离模型差异。','- 原作者称“小幅模型修正”的精确幅度、目标、特征、训练方式未知；本轮是可复查的有限假设。','- 模型训练没有未来收益，但本轮组合和参数按已知持仓挑选，是事后拟合。2025后切片也被历史研究多次查看，不是纯样本外。','- 月末完整收盘特征用于描述性排名；没有证明能在同一收盘价成交。没有计算不连续A/B切片年化或Sharpe。','- 原有多系数诊断31/54、30/39依赖46/56个系数和高精度，本轮不与其混作同等简洁可执行模型。','',f"有限网格：{read(P/'SEARCH_STATUS.json')['trials']}个去重组合，33个模型配置×6年=198次年度拟合。",'', '详细训练审计TRAINING_AUDIT.json；逐月持仓MONTHLY_DECISIONS.txt；试验数据库TRIALS.sqlite；输入与保护哈希独立保存。','', '依赖文档：https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.Ridge.html；https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html']
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    rows=['A选择规则的逐期持仓比较（全69期；未在A/B内也照常输出）','日期 | 作者 | 模型 | 命中 | A/B纳入 | 弱市']
    for r in leaders['A']['details']:rows.append(f"{r['date']} | {','.join(r['target'])} | {','.join(r['selected'])} | {r['shared']}/3 | {int(r['included_A'])}/{int(r['included_B'])} | {r['weak']}")
    (P/'MONTHLY_DECISIONS.txt').write_text('\n'.join(rows)+'\n');con.close()
    print('LEADERS',{k:v['metrics'] for k,v in leaders.items()},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','train','search','report']);args=ap.parse_args();globals()[args.phase]()

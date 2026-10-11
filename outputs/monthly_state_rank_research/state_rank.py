"""Remaining planned market-state and top-three ML experiments, isolated research."""
from pathlib import Path
from datetime import datetime, timezone
import sys, json, itertools, argparse, importlib.util
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent; O=P.parent
sys.path.insert(0,str(O/'monthly_quality_research')); import explore as prior
rp,ex,read,sha=prior.rp,prior.ex,prior.read,prior.sha
MODEL_CONFIGS=[dict(target=t,alpha=a,half_life=12) for t in ['return','top3_pair','downside_rank'] for a in [10.,100.]]

def dump(name,z):
    (P/name).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def initialize():
    assert not (P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
      purpose='Finish directions4 market state and5 top-three ML / return-downside ranking. Preserve65.3395543 observer and65.9476507 research control.',
      universe='Exact122 monthly Top32-proxy+theme17 pools, unchanged eligibility. Weekly ML training beforeAug2016 uses inherited historical foreign membership proxy; thereafter last already observed monthlyTop32, not future pool. This inherited proxy is not certified point-in-time universe.',
      state_grid='4profiles sector/quality/lowdownside/rotation x4gates trend_broad/weak_narrow/high_dispersion/breadth_improving x2.5/5/10% from noncore support =48. Breadth thresholds60/40%,SPY200trend; dispersion21 cross-stock std above its prior252session median.',
      ml='Annual trailing5years weekly samples, next21session fully mature labels. Six configs: ordinary-return control, top3-weighted pairwise rank Ridge, downside-adjusted-rank Ridge; alpha10/100; half-life12months. No author labels or identifier features. Features include skip42long momentum,volatility,own-sector relative,volume/price shapes and observed state interactions.',
      targets='return:21session excess vsSPY clip[-1,1]; pair:future return percentile differences with3x weight across realizedtop3 boundary, only pairs touching the past-scoretop8; downside_rank:rank(excess_return-.5*max(0,-minimum21sessionreturn)). Path labels require all21future prices and are used only after label_end<=cutoff.',
      ml_grid='6forecasts x2.5/5/10% support correction xglobal/highvol/weak_narrow =54. State-only48 plus controls2=104 initially. No unchanged daily pullback entry gate introduced.',
      combination='Using first88 only, choose best admissible state and genuinely-newML(target pair/downside) singles. Test their sum plus combinations with fixed2.5%residual; cap total simultaneous correction10%,scale addons proportionately. At most4additional combos. Ordinary-returnRidge only comparator, not mislabeled new.',
      split='Train first88 through2023-11-30; early53 and recent35 constraints. Freeze primary, robust and family controls BEFORE last34 replay. Market-model annual retraining in later periods may use then-mature past prices, never future labels. Repeatedly researched history, not pristine OOS.',
      selection='Same guards as previous round relative to65.34:trainSharpe>=base-.05,DD>=base-.03,earlyCAGR>=base,recent35CAGR>=base-.02. MaxtrainCAGR primary; worst-segmentSharpe robust. Include65.95 residual control as competitor.',
      execution='Long-only,unlevered,max3equal,2half,<2cash; fixed monthly units; core>=50%,unchanged. Zero-cost same-close idealization primary,next-open0cost and5bp/side sensitivities. No full-history tuning after freeze.',
      protection='No changes to observers, plans, ledgers, originalcode/data/automation. No promotion. New outputs only here.'))
    folders=['monthly_quality_research','monthly_stability_research','monthly_2016_extension','monthly_stability_monitor','monthly_industry_monitor','monthly_regime_monitor','monthly_buffer5_monitor','daily_stock_monitor','pressure_shadow_monitor','resilience_shadow_monitor','accel_shadow_monitor','high52_shadow_monitor','accel40_shadow_monitor']
    dump('PROTECTED.json',{str(f):sha(f) for folder in folders for f in (O/folder).rglob('*') if f.is_file() and f.suffix!='.pyc' and f.name!='run.lock'})
    dump('MODEL_CONFIGS.json',MODEL_CONFIGS)

def panel_load(etfs):
    file=O/'monthly_stability_monitor/inputs.py'
    spec=importlib.util.spec_from_file_location('state_rank_readonly_inputs',file);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    panel,info=mod.load_panel(etfs);info['source_hashes'][str(file)]=sha(file)
    return panel,info

def raw_features(px,raw):
    r=px.pct_change(fill_method=None);v63=r.rolling(63).std();v126=r.rolling(126).std();log=np.log(px/px.shift(1))
    down=np.sqrt(r.clip(upper=0).pow(2).rolling(63).mean())
    mom={n:px.shift(42)/px.shift(n)-1 for n in [126,252,294]}
    span=(raw['high']-raw['low']).replace(0,np.nan);loc=(2*raw['close']-raw['high']-raw['low'])/span
    cmf={n:(loc*raw['volume']).rolling(n).sum()/raw['volume'].rolling(n).sum().replace(0,np.nan) for n in [21,63]}
    tr=pd.DataFrame(np.maximum.reduce([span.to_numpy(),(raw['high']-raw['close'].shift()).abs().to_numpy(),(raw['low']-raw['close'].shift()).abs().to_numpy()]),index=px.index,columns=px.columns)
    atr=tr.rolling(21).mean()/raw['close']
    z=dict(mom252=mom[252],mom294=mom[294],risk252=mom[252]/v63.replace(0,np.nan),risk294=mom[294]/v63.replace(0,np.nan),
      accel=px.shift(42)/px.shift(126)-px.shift(126)/px.shift(210),path=np.log(px.shift(42)/px.shift(294))/log.abs().rolling(252).sum().shift(42).replace(0,np.nan),
      vol63=v63,vol126=v126,downside=down,upside=np.sqrt(r.clip(lower=0).pow(2).rolling(126).mean()),
      cmf21=cmf[21],cmf63=cmf[63],signed_volume=(np.sign(r)*raw['volume']).rolling(63).sum()/raw['volume'].rolling(63).sum().replace(0,np.nan),
      volume_ratio=raw['volume'].rolling(21).mean()/raw['volume'].rolling(126).mean().replace(0,np.nan),
      close_location=loc.rolling(5).mean(),dry_consolidation=-v63*(raw['volume'].rolling(5).mean()/raw['volume'].rolling(21).mean().replace(0,np.nan)),
      positive_gap=(raw['open']/raw['close'].shift(1)-1).clip(lower=0).rolling(5).mean()/v63.replace(0,np.nan),
      rebound21=(px/px.shift(21)-1)/atr.replace(0,np.nan),
      pressure_easing=-(raw['volume']*r.clip(upper=0).abs()).rolling(5).sum()/raw['volume'].rolling(5).sum().replace(0,np.nan)+(raw['volume']*r.clip(upper=0).abs()).rolling(21).sum()/raw['volume'].rolling(21).sum().replace(0,np.nan))
    return z,mom

def state_row(px,date,ss,etfs):
    date=pd.Timestamp(date);hist=px.loc[:date];stocks=[s for s in ss if s not in etfs]
    h=hist[stocks].iloc[-515:];avg=h.rolling(200).mean();ok=avg.notna()&h.notna()
    breadth=(h>avg).where(ok).mean(axis=1);b=float(breadth.iloc[-1]);b63=float(breadth.iloc[-64])
    disp=h.pct_change(21,fill_method=None).std(axis=1);med=float(disp.iloc[-253:-1].median());dv=float(disp.iloc[-1])
    spy=hist.SPY;trend=bool(spy.iloc[-1]>spy.iloc[-200:].mean());mv=spy.pct_change(fill_method=None).rolling(63).std();high=bool(mv.iloc[-1]>mv.iloc[-252:].median())
    strong=bool((hist.QQQ.iloc[-1]/hist.QQQ.iloc[-127])>(spy.iloc[-1]/spy.iloc[-127]))
    return dict(breadth=b,breadth63=b63,dispersion=dv,dispersion_prior_median=med,trend=trend,highvol=high,growth=strong,
      trend_broad=trend and b>=.60,weak_narrow=(not trend) or b<.40,high_dispersion=dv>med,breadth_improving=b>b63,global_=True)

def feature_row(date,ss,feat,mom,px,sectors,etfs,state):
    row={k:prior.ranks(v.loc[date,ss]) for k,v in feat.items()}
    bm=ex.t.strict.benchmark_map(date,ss,sectors,etfs)
    for n in [126,252]:row['sector'+str(n)]=prior.ranks(ex.t.strict.relative_row(mom[n],date,ss,bm))
    names=list(row);x=np.column_stack([row[k] for k in names]);st=np.array([2*state['breadth']-1,2*float(state['trend'])-1,2*float(state['highvol'])-1,2*float(state['growth'])-1])
    # State scalars alone do not alter pairwise ordering; interactions do.
    full=np.column_stack([x,x*st[0],x*st[1],x*st[2]])
    prior_score=.52*row['risk294']+.22*row['vol126']+.18*row['accel']+.08*row['cmf63']
    return full,prior_score,names

def forward_labels(px,date,ss,horizon=21):
    i=px.index.get_loc(date);end=px.index[i+horizon];p=px.iloc[i+1:i+horizon+1][ss].div(px.loc[date,ss])-1
    y=(p.iloc[-1]-(px.at[end,'SPY']/px.at[date,'SPY']-1)).clip(-1,1)
    adverse=(-p.min()).clip(lower=0);valid=p.notna().all()&np.isfinite(y)&np.isfinite(adverse)
    return str(end.date()),y.to_numpy(),adverse.to_numpy(),valid.to_numpy()

def prepare():
    m,d,L,x,px,op,f,a=prior.load();panel,info=panel_load(m['etfs']);full=panel['adj']
    for s in px:
        common=full[s].reindex(px.index);np.testing.assert_allclose(common,px[s],equal_nan=True,rtol=1e-12,atol=1e-8)
    print('Input price parity; computing features',flush=True)
    feat,mom=raw_features(full,panel);etfs=set(m['etfs']);sectors=panel['sectors']
    legacyfile=O/'monthly_reconstruction_v2/foreign/membership.csv.gz';legacy=pd.read_csv(legacyfile,index_col=0,parse_dates=True).fillna(False).astype(bool)
    member=panel['member'];warm=full.notna().rolling(295).sum().eq(295)&full.gt(0)
    weekly=full.loc['2011-01-01':'2025-12-31'].index.to_series().groupby(full.loc['2011-01-01':'2025-12-31'].index.to_period('W-FRI')).last()
    X=[];Y=[];A=[];G=[];S=[];counts=[0];dates=[];ends=[];symbols=[];origins=[]
    for i,date in enumerate(weekly):
        previous=member.index[member.index<=date]
        if len(previous):
            origin=str(previous[-1].date());ss=sorted(set(member.columns[member.loc[previous[-1]]])|etfs)
        else:
            origin='inherited_pre2016_historical_proxy';ss=sorted(set(legacy.columns[legacy.loc[date]])|etfs)
        st=state_row(full,date,ss,etfs);xx,score,names=feature_row(date,ss,feat,mom,full,sectors,etfs,st)
        end,y,adverse,valid=forward_labels(full,date,ss)
        good=(warm.loc[date,ss]&mom[252].loc[date,ss].gt(0)).to_numpy()&valid
        X.append(xx);Y.append(y);A.append(adverse);G.append(good);S.append(score);counts.append(counts[-1]+len(ss))
        dates.append(str(date.date()));ends.append(end);symbols.append(ss);origins.append(origin)
        if (i+1)%150==0:print('Weekly samples',i+1,flush=True)
    event=[];states=[]
    for date,ss in zip(m['dates'],m['symbols']):
        st=state_row(full,date,ss,etfs);xx,_,_=feature_row(date,ss,feat,mom,full,sectors,etfs,st);event.append(xx);states.append(st)
    np.savez_compressed(P/'TRAIN_DATA.npz',X=np.vstack(X),Y=np.concatenate(Y),adverse=np.concatenate(A),good=np.concatenate(G),past_score=np.concatenate(S),offsets=np.array(counts),event=np.stack(event))
    dump('TRAIN_DATA.json',dict(dates=dates,ends=ends,symbols=symbols,pool_origins=origins,feature_names=names,feature_interactions=['raw','breadth','SPY200trend','market_highvol'],monthly_states=states))
    addons={'sector':.5*(x['sector126']+x['sector252']), 'quality':.5*f['path']+.25*x['stability']+.25*x['tail'], 'lowdownside':-f['downside'], 'rotation':.5*f['sector']+.25*f['path']+.25*f['flow']}
    features={**a,**{'profile_'+k:v for k,v in addons.items()},**{'state_'+k:np.array([s[k] for s in states]) for k in states[0]}}
    # Keep highvol/global/calm exactly as the original score state.
    features['state_highvol']=a['state_highvol'];features['state_global_']=a['state_global_']
    np.savez_compressed(P/'FEATURES.npz',**features)
    hashes=info['source_hashes'];hashes[str(legacyfile)]=sha(legacyfile)
    for path in [Path(prior.__file__),prior.P/'FEATURES.npz',prior.P/'FROZEN.json',Path(rp.__file__),Path(ex.e.engine.__file__)]:hashes[str(path)]=sha(path)
    dump('INPUT_HASHES.json',hashes);print('Prepared',len(dates),'weekly samples',np.vstack(X).shape,flush=True)

def pair_moments(X,y,past_score):
    """Laplacian sufficient statistics exactly equal weighted pair differences.
    Pool of comparisons: at least one member in past-score top8. Three times
    weight for pairs straddling the realized top3 label boundary (training only).
    """
    n=len(y);anchor=np.zeros(n,bool);anchor[np.argsort(-past_score,kind='stable')[:8]]=True
    top=np.zeros(n,bool);top[np.argsort(-y,kind='stable')[:3]]=True
    weights=(anchor[:,None]|anchor[None,:]).astype(float)*(1+2*np.not_equal(top[:,None],top[None,:]));np.fill_diagonal(weights,0)
    lap=np.diag(weights.sum(axis=1))-weights
    rank=pd.Series(y).rank(pct=True).to_numpy()-.5
    gram=np.einsum('ji,jk,kl->il',X,lap,X,optimize=True);rhs=np.einsum('ji,jk,k->i',X,lap,rank,optimize=True)
    # Divide by pair count to give each weekly cross-section comparable influence.
    scale=weights.sum()/2
    return gram/scale,rhs/scale

def training_indices(meta,cutoff):
    lo=str((pd.Timestamp(cutoff)-pd.DateOffset(years=5)).date())
    return [i for i,(date,end) in enumerate(zip(meta['dates'],meta['ends'])) if lo<=date<=cutoff and end<=cutoff]

def fit_year(data,meta,cutoff,cfg):
    ids=training_indices(meta,cutoff);age=np.array([(pd.Timestamp(cutoff)-pd.Timestamp(meta['dates'][i])).days/30.4375 for i in ids]);weight=2**(-age/cfg['half_life']);weight/=weight.mean()
    if cfg['target']=='top3_pair':
        F=data['X'].shape[1];gram=np.zeros((F,F));rhs=np.zeros(F);n=0
        for i,w in zip(ids,weight):
            sl=slice(data['offsets'][i],data['offsets'][i+1]);good=data['good'][sl];X=data['X'][sl][good];y=data['Y'][sl][good]
            if len(y)<4:continue
            g,r=pair_moments(X,y,data['past_score'][sl][good]);gram+=w*g;rhs+=w*r;n+=len(y)
        coef=np.linalg.solve(gram+cfg['alpha']*np.eye(F),rhs);intercept=0.
    else:
        xs=[];ys=[];ws=[];n=0
        for i,w in zip(ids,weight):
            sl=slice(data['offsets'][i],data['offsets'][i+1]);good=data['good'][sl];X=data['X'][sl][good];y=data['Y'][sl][good]
            if len(y)<4:continue
            if cfg['target']=='downside_rank':y=prior.ranks(y-.5*data['adverse'][sl][good])
            xs.append(X);ys.append(y);ws.extend([w/len(y)]*len(y));n+=len(y)
        weights=np.array(ws);weights*=len(weights)/weights.sum()
        fit=ex.ridge.fit_weighted_ridge(np.vstack(xs),np.concatenate(ys),weights,cfg['alpha']);coef=fit['coef'];intercept=fit['intercept']
    return coef,intercept,dict(training_start=meta['dates'][ids[0]],latest_label_end=max(meta['ends'][i] for i in ids),weekly_samples=len(ids),security_examples=n)

def forecasts():
    m,d,*_=prior.load();data=dict(np.load(P/'TRAIN_DATA.npz'));meta=read(P/'TRAIN_DATA.json');pred={f'ml_{i}':np.zeros_like(d['core']) for i in range(len(MODEL_CONFIGS))};audit=[]
    for year in range(2016,2027):
        ids=[i for i,end in enumerate(m['ends']) if pd.Timestamp(end).year==year];first=min(m['dates'][i] for i in ids)
        cutoff=min(first,str((pd.Timestamp(f'{year}-01-01')-pd.Timedelta(days=1)).date()))
        for k,cfg in enumerate(MODEL_CONFIGS):
            coef,intercept,log=fit_year(data,meta,cutoff,cfg);assert log['latest_label_end']<=cutoff<=first
            y=np.einsum('ijk,k->ij',data['event'][ids],coef,optimize=False)+intercept
            for j,i in enumerate(ids):pred[f'ml_{k}'][i]=prior.ranks(pd.Series(y[j]).where(d['eligible'][i]))
            audit.append(dict(year=year,configuration=k,config=cfg,cutoff=cutoff,**log,coef=coef.tolist(),intercept=float(intercept)))
        print('Causal annual forecasts',year,'labels through',audit[-1]['latest_label_end'],flush=True)
    np.savez_compressed(P/'FORECASTS.npz',**pred);dump('FORECAST_AUDIT.json',audit)

def load():
    for p,h in read(P/'INPUT_HASHES.json').items():assert sha(p)==h,p
    return (*prior.load()[:7],{**dict(np.load(P/'FEATURES.npz')),**dict(np.load(P/'FORECASTS.npz'))})

def score(spec,a,n):
    core=a['core'][:n];support=(a['baseline'][:n]-core)/(1-a['core_weight'][:n])
    budgets=[z['amount']*a['state_'+z['state']][:n,None].astype(float) for z in spec]
    total=sum(budgets) if budgets else np.zeros((n,1));scale=np.minimum(1.,.10/np.maximum(total,1e-12))
    result=a['baseline'][:n].copy()
    for z,b in zip(spec,budgets):result+=b*scale*(a[z['key']][:n]-support)
    return result

def choices(spec,a,m,d,n):
    sc=score(spec,a,n);return rp.choose(sc,d['eligible'][:n],m['symbols'][:n],3),sc

def grid():
    rows=[dict(id=0,family='baseline',spec=[]),dict(id=1,family='residual_control',spec=read(prior.P/'FROZEN.json')['selected']['primary']['spec'])]
    for key,state,amount in itertools.product(['sector','quality','lowdownside','rotation'],['trend_broad','weak_narrow','high_dispersion','breadth_improving'],[.025,.05,.10]):
        rows.append(dict(id=len(rows),family='state',spec=[dict(key='profile_'+key,state=state,amount=amount)]))
    for i,state,amount in itertools.product(range(len(MODEL_CONFIGS)),['global_','highvol','weak_narrow'],[.025,.05,.10]):
        rows.append(dict(id=len(rows),family='return_control' if MODEL_CONFIGS[i]['target']=='return' else 'new_ml',spec=[dict(key=f'ml_{i}',state=state,amount=amount)]))
    return rows

def measure(row,m,d,a,blocks,bounds,valid):
    ch,_=choices(row['spec'],a,m,d,88);rr=rp.fast_returns(ch,blocks,valid)
    train=rp.stats_rr(rr,m['dates'][0],m['ends'][87]);early=rp.stats_rr(rr[:bounds[53]],m['dates'][0],m['ends'][52]);recent=rp.stats_rr(rr[bounds[53]:],m['dates'][53],m['ends'][87])
    robust=min(early['sharpe'],recent['sharpe'])+.3*train['sharpe']+.4*np.log1p(train['cagr'])-.3*abs(train['max_drawdown'])
    return dict(row,training=train,early=early,recent=recent,robust=float(robust)),rr

def train():
    assert not (P/'FROZEN.json').exists()
    m,d,L,x,px,op,f,a=load();blocks,_,bounds,valid=rp.paths(px,m,88);rows=[];returns=[]
    for spec in grid():
        z,rr=measure(spec,m,d,a,blocks,bounds,valid);rows.append(z);returns.append(rr)
    controls={}
    for fam in ['state','new_ml','return_control']:
        good=[r for r in rows if r['family']==fam and prior.allowed(r,rows[0])];controls[fam]=max(good,key=lambda r:(r['training']['cagr'],-r['id'])) if good else rows[0]
        print('Training',fam,'admissible',len(good),'best id',controls[fam]['id'],flush=True)
    seeds=[rows[1]]+[controls[k] for k in ['state','new_ml'] if controls[k]['id']!=0]
    for size in range(2,len(seeds)+1):
        for combo in itertools.combinations(seeds,size):
            z,rr=measure(dict(id=len(rows),family='combined',spec=sum([r['spec'] for r in combo],[])),m,d,a,blocks,bounds,valid);rows.append(z);returns.append(rr)
    good=[r for r in rows if prior.allowed(r,rows[0])]
    selected=dict(primary=max(good,key=lambda r:(r['training']['cagr'],-r['id'])),robust=max(good,key=lambda r:(r['robust'],-r['id'])))
    # Fixed initial grid only; no full-training-selected combinations in prefix selection.
    wf=[];stitched=[];baseparts=[]
    for n,end in [(36,60),(60,76),(76,88)]:
        metrics=[rp.stats_rr(rr[:bounds[n]],m['dates'][0],m['ends'][n-1]) for rr in returns[:104]];b=metrics[0]
        ids=[i for i,z in enumerate(metrics) if z['sharpe']>=b['sharpe']-.05 and z['max_drawdown']>=b['max_drawdown']-.03]
        win=max(ids,key=lambda i:(metrics[i]['cagr'],-i));rr=returns[win][bounds[n]:bounds[end]];br=returns[0][bounds[n]:bounds[end]];stitched.extend(rr);baseparts.extend(br)
        wf.append(dict(train_end=m['ends'][n-1],test_start=m['dates'][n],test_end=m['ends'][end-1],id=win,spec=rows[win]['spec'],metrics=rp.stats_rr(rr,m['dates'][n],m['ends'][end-1]),baseline=rp.stats_rr(br,m['dates'][n],m['ends'][end-1])))
    dump('WALKFORWARD.json',dict(folds=wf,metrics=rp.stats_rr(np.array(stitched),m['dates'][36],m['ends'][87]),baseline=rp.stats_rr(np.array(baseparts),m['dates'][36],m['ends'][87]),note='Sequential selector diagnostic on reused history and already-developed baseline; not pristineOOS.'))
    dump('TRAIN_TRIALS.json',rows);fr=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=selected,controls=controls,baseline=rows[0],residual_control=rows[1],trials=len(rows),admissible=len(good),code_sha256=sha(__file__),data_hashes={n:sha(P/n) for n in ['PLAN.json','TRAIN_DATA.npz','TRAIN_DATA.json','FEATURES.npz','FORECASTS.npz','FORECAST_AUDIT.json']})
    dump('FROZEN.json',fr);(P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n');print('FROZEN',[(k,v['id'],v['training']['cagr']) for k,v in selected.items()],flush=True)

def evaluate():
    fr=read(P/'FROZEN.json');assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip();assert sha(__file__)==fr['code_sha256']
    for n,h in fr['data_hashes'].items():assert sha(P/n)==h,n
    m,d,L,x,px,op,f,a=load();out={};allorders={};blocks,_,bounds,valid=rp.paths(px,m,122)
    for name,row in dict(baseline=fr['baseline'],residual_control=fr['residual_control'],**fr['controls'],**fr['selected']).items():
        ch,sc=choices(row['spec'],a,m,d,122);orders=rp.make_orders(ch,sc,m)
        met,eq,tr,hold=ex.e.engine.replay(px,op,orders,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
        rr=rp.fast_returns(ch,blocks,valid);np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
        out[name]=dict(metrics=met,slices=prior.old.previous.cuts(eq,m),spec=row['spec'],id=row['id'])
        assert abs(out[name]['slices']['training2016_2023']['cagr']-row['training']['cagr'])<1e-10
        if name in ['baseline','residual_control']:
            ref=read(prior.P/'RESULTS.json')['baseline' if name=='baseline' else 'primary']['metrics']
            for k in ['cagr','sharpe','max_drawdown']:assert abs(met[k]-ref[k])<1e-10
        for label,cost in [('next_open_zero',0.),('next_open_cost',.0005)]:
            mm,ee,_,_=ex.e.engine.replay(px,op,orders,cost=cost,execution='open',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]));out[name][label]=dict(metrics=mm,slices=prior.old.previous.cuts(ee,m))
        for suffix,z in [('ORDERS',orders),('TRADES',tr),('HOLDINGS',hold),('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])]:dump(name+'_'+suffix+'.json',z)
        allorders[name]=orders;print(name,'CAGR',met['cagr'],'Sharpe',met['sharpe'],'early',out[name]['slices']['early2016_2020']['cagr'],'later',out[name]['slices']['later34']['cagr'],flush=True)
    q=px.loc[m['dates'][0]:m['ends'][-1],'QQQ'];eq=q/q.iloc[0]*100000;out['qqq']=dict(metrics=ex.e.engine.stats(eq),slices=prior.old.previous.cuts(eq,m));dump('RESULTS.json',out)
    comparisons=[]
    for i,day in enumerate(m['dates']):
        r=dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
        for name,orders in allorders.items():
            ss=orders[i]['symbols'];r[name]=dict(symbols=ss,monthly_return=float((px.loc[m['ends'][i],ss]/px.loc[day,ss]-1).mean()) if ss else 0.)
        comparisons.append(r)
    dump('MONTH_COMPARISON.json',comparisons)
    changed=[p for p,h in read(P/'PROTECTED.json').items() if sha(p)!=h];assert not changed,changed
    dump('VALIDATION.json',dict(both_control122_parity=True,independent_replay_parity=True,train_freeze_parity=True,protected_files=len(read(P/'PROTECTED.json')),protected_changes=changed,pristine_oos=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','prepare','forecasts','train','evaluate']);globals()[ap.parse_args().phase]()

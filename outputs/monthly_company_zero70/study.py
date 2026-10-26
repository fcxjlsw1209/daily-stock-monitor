"""Own-company mature monthly outcomes with a zero prior in the frozen episode slot."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent
O=P.parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

previous=module('companyzero_path',O/'monthly_path70_research/study.py')
old=previous.old
rp,ex,arch,ep,read,sha=old.rp,old.ex,old.arch,old.ep,old.read,old.sha
TRAIN=88
FACTORS=('company_mean_k6','company_mean_k12','company_signal_context_k6','company_signal_context_k12')

def dump(name,value):
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n')

def configs():
    return [dict(id=0,base='original65',factor='none'),dict(id=1,base='incumbent66',factor='none')]+[dict(id=i+2,base='incumbent66',factor=f) for i,f in enumerate(FACTORS)]

def implementations():
    return (P/'study.py',P/'features.py',O/'monthly_company_context/context_research.py',O/'monthly_path70_research/incumbent_adapter.py',Path(previous.__file__),O/'monthly_horizon70_research/study.py',O/'monthly_conditional_model70/study.py',O/'monthly_target70_research/study.py',Path(ep.__file__),Path(ep.old.__file__))

def initialize():
    assert not(P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        objective='Own-company monthly outcomes shrunk tozero, no shared model prior. Fullperiod2016-08-17..2026-09-24 target70percent; no guarantee.',
        fixed='Same122periods,estimatedhistoricalTop32+assumedETF17,original295price/positiveskip42eligibility;3equal/2half/<2cash;longonly,no leverage,monthfixedunits. Originalcore50/52 and prior scores intact.',
        novelty='Owncompany mean/context/signal monthly forecasts existed with globalRidgeprior k6/12; dailyepisode zero-prior also existed for5/10/21session events. This is matchedmonthly-history zero-prior with no sharedRidge. Reuse original monthly labels/shapes/states/signals and annualclock.',
        grid='6fixed:2unchanged controls plus company_mean k6/k12 and company_signal_context k6/k12. Shape bandwidth .5, statebandwidth1,recency24months. Nootherparameter search.',
        clock='Group queries by holding_end.year. Cutoff=min(firstsignal inyeargroup,Dec31previousyear). Onlysame-security originaleligible trainingrows with date<query, ends<=cutoff<=query,date>=cutoff-5years. Reuse180 monthlysnapshots2011..2025 and inherited clipped[-1,1] successor-month simple excessSPYlabels. Never include current outcome.',
        forecast='p=sum(w*y)/(sum(w)+k); w=2^(-age_in_months/24). company_mean uses allsame-security eligible mature rows. signal_context requires matching positive signal A/B/both and multiplies original8shapeGaussian bandwidth.5 and3marketstateGaussian bandwidth1. k6/12 shrinks toward0, no sharedRidge and no fixedtickerbonuses. n_eff onlyaudit,no additional threshold. Missing/no usablehistory=>raw0,availableFalse.',
        score='Onlyavailable predictions add .25*tanh(raw/.05) to incumbent frozen episode effect2. No crosssectionalrank, no increment forunknown ortruezero. Reuse actual select_boundary pretop2 andonly ranks3..5 within .02 ofthird; original .05episodebudget and total oldslots<=.10 unchanged. Maximum additional score .0125. No globalpostscore reranking.',
        selection='First88 through2023-11-30. TrainCAGR>=inc66,Sharpe>=inc-.05,DD>=inc-.03,early53/recent35CAGR>=inc-.025. Primary highestadmittedtrainingCAGR,robust minimumsegmentSharpe; ties lowerid/fallbackinc66. Freezeall4newdiagnostics beforelater34; prefix36/60/76 diagnostic.',
        evaluation='Samecompleteclose/zerocost idealization; nextopenzero and5bpper-side using sameorders; originalcontrolparity, independentfixedunits, month attribution. All history repeatedlyresearched,not pristineOOS. No observerchanges or exactauthorreplicationclaim.',
        hashes={str(file):sha(file) for file in implementations()}))
    dump('GRID.json',configs());dump('PROTECTED.json',read(previous.P/'PROTECTED.json'))

def registered():
    assert read(P/'GRID.json')==configs()
    old.assert_hashes(read(P/'PLAN.json')['hashes'])

def augmented_effects(spec,a,n):
    if spec not in configs():raise ValueError('Unregistered configuration')
    shape=a['baseline'].shape
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or not 0<=n<=shape[0]:raise ValueError('Invalid prefix')
    result=np.zeros(shape,float);result[:n]=np.asarray(a['effects'])[:n]
    if spec['factor']=='none':return result
    raw=np.asarray(a[spec['factor']]);available=np.asarray(a['available_'+spec['factor']])
    if raw.shape!=shape or available.shape!=shape or available.dtype.kind!='b':raise ValueError('Misaligned forecasts')
    value=raw[:n].astype(float);usable=available[:n]&np.isfinite(value)
    increment=np.zeros((n,shape[1]));increment[usable]=.25*np.tanh(value[usable]/.05)
    nonzero=increment!=0.;prefix=result[:n];prefix[nonzero]+=increment[nonzero]
    return result

def choices(spec,a,m,n):
    adapter=module('companyzero_adapter',O/'monthly_path70_research/incumbent_adapter.py')
    return adapter.score_and_choose(spec['base'],0.,np.full_like(a['baseline'],np.nan),a,m,n,pre_score=a['pre_score'],effects=augmented_effects(spec,a,n),episode_policy=read(P/'INCUMBENT_POLICY.json'),boundary_selector=ep.select_boundary)

def prepare():
    registered();assert not(P/'FROZEN.json').exists() and not(P/'DATA.npz').exists()
    m,prior,px,op=previous.load()
    a={k:prior[k].copy() for k in ('baseline','core','core_weight','state_calm10','state_weak_narrow','eligible','pre_score','effects')}
    ctx=module('companyzero_context',O/'monthly_company_context/context_research.py')
    old.assert_hashes(read(ctx.P/'INPUT_HASHES.json'));old.assert_hashes(read(ctx.P/'SIGNAL_SOURCES.json'))
    prior_cache_hashes=read(ctx.P/'FROZEN.json')['data_hashes']
    for filename in ('SIGNALS.npz','SIGNAL_SOURCES.json'):
        assert sha(ctx.P/filename)==prior_cache_hashes[filename],filename
    meta=read(ctx.blend.P/'MONTHLY_TRAIN.json')
    with np.load(ctx.blend.P/'MONTHLY_TRAIN.npz',allow_pickle=False) as cache:
        data={k:cache[k].copy() for k in ('X','Y','good','offsets','event')}
    offsets=data['offsets'];assert len(offsets)==len(meta['dates'])+1
    shape,state=ctx.context_features(data['X'],meta['feature_names'],offsets)
    flat=data['event'].reshape(-1,data['X'].shape[1]);eo=np.arange(0,len(flat)+1,49)
    event_shape,event_state=ctx.context_features(flat,meta['feature_names'],eo)
    event_shape=event_shape.reshape(len(m['dates']),49,8);event_state=event_state.reshape(len(m['dates']),49,3)
    # Signal order comes from unchanged producer monthly meta,not NPZ event dates.
    original_m,_,_,_,_,_=ctx.load()
    assert original_m==m
    with np.load(ctx.P/'SIGNALS.npz',allow_pickle=False) as cache:
        tsig,esig=cache['train'].copy(),cache['event'].copy()
    dates=[];ends=[];symbols=[]
    for i,names in enumerate(meta['symbols']):
        assert len(names)==offsets[i+1]-offsets[i]
        dates.extend([meta['dates'][i]]*len(names));ends.extend([meta['ends'][i]]*len(names));symbols.extend(names)
    features=module('companyzero_features',P/'features.py')
    forecasts,audit=features.annual_forecasts(np.array(dates),np.array(ends),np.array(symbols),data['Y'],data['good'],shape,state,tsig,np.array(m['dates']),np.array(m['ends']),np.array(m['symbols']),event_shape,event_state,esig,a['eligible'])
    for key in FACTORS:
        assert forecasts[key].shape==a['baseline'].shape and np.isfinite(forecasts[key]).all()
        assert forecasts['available_'+key].shape==a['baseline'].shape and forecasts['available_'+key].dtype.kind=='b'
    a.update(forecasts)
    policy=read(previous.P/'INCUMBENT_POLICY.json');dump('INCUMBENT_POLICY.json',policy)
    ch,sc=choices(configs()[1],a,m,len(m['dates']))
    assert rp.make_orders(ch,sc,m)==read(ep.P/'primary_ORDERS.json')
    ch0,sc0=choices(configs()[0],a,m,len(m['dates']))
    assert rp.make_orders(ch0,sc0,m)==read(ep.P/'original65_ORDERS.json')
    effects_audit={}
    for cfg in configs()[2:]:
        cc,ss=choices(cfg,a,m,len(m['dates']));delta=ss-sc
        assert np.max(np.abs(delta))<=.0125+1e-14
        effects_audit[cfg['factor']]=dict(available_cells=int(a['available_'+cfg['factor']].sum()),changed_score_cells=int(np.count_nonzero(delta)),changed_ordered_months=sum(x!=y for x,y in zip(cc,ch)),max_absolute_score_change=float(np.max(np.abs(delta))))
    audit['effect_audit']=effects_audit
    np.savez_compressed(P/'DATA.npz',**a);dump('META.json',m);dump('FEATURE_AUDIT.json',audit)
    dump('PREPARE_VALIDATION.json',dict(original_controls_exact=True,monthly_axes_exact=True,eligibility_unchanged=True))
    hashes=dict(read(previous.P/'INPUT_HASHES.json'));hashes.update(read(ctx.P/'INPUT_HASHES.json'));hashes.update(read(ctx.P/'SIGNAL_SOURCES.json'))
    for f in list(implementations())+[ctx.P/'FROZEN.json',ctx.P/'INPUT_HASHES.json',ctx.P/'SIGNALS.npz',ctx.P/'SIGNAL_SOURCES.json',ctx.blend.P/'MONTHLY_TRAIN.npz',ctx.blend.P/'MONTHLY_TRAIN.json',previous.P/'DATA.npz',previous.P/'META.json',previous.P/'INCUMBENT_POLICY.json',previous.P/'RESULTS.json']:
        hashes[str(f)]=sha(f)
    dump('INPUT_HASHES.json',hashes)
    print('Prepared6zero-prior configurations; controls exact',flush=True)

def load():
    registered();old.assert_hashes(read(P/'INPUT_HASHES.json'))
    m,_,px,op=previous.load();assert m==read(P/'META.json')
    return m,dict(np.load(P/'DATA.npz',allow_pickle=False)),px,op

def admitted(z,incumbent):
    return (z['training']['cagr']>=incumbent['training']['cagr']-1e-12 and z['training']['sharpe']>=incumbent['training']['sharpe']-.05 and z['training']['max_drawdown']>=incumbent['training']['max_drawdown']-.03 and z['early']['cagr']>=incumbent['early']['cagr']-.025 and z['recent']['cagr']>=incumbent['recent']['cagr']-.025)

def select(rows):
    incumbent=rows[1]
    for z in rows:
        z['admitted']=admitted(z,incumbent);z['old_strict_admitted']=old.session.admitted(z,incumbent)
    best=lambda rr:max(rr,key=lambda z:(z['training']['cagr'],-z['spec']['id']))
    qualified=[z for z in rows if z['admitted']] or [incumbent]
    selected=dict(primary=best(qualified),robust=max(qualified,key=lambda z:(min(z['early']['sharpe'],z['recent']['sharpe']),z['training']['cagr'],-z['spec']['id'])),original65=rows[0],incumbent66=rows[1])
    for z in rows[2:]:selected[z['spec']['factor']]=z
    return selected


def train():
    assert not (P/'FROZEN.json').exists()
    m, a, px, _ = load()
    blocks, _, bounds, valid = rp.paths(px,m,TRAIN)
    rows, returns = [], []
    for spec in read(P/'GRID.json'):
        ch, _ = choices(spec,a,m,TRAIN)
        rr = rp.fast_returns(ch,blocks,valid)
        rows.append(dict(spec=spec,**old.session.measure(rr,m,bounds)))
        returns.append(rr)
    selected = select(rows)
    folds, stitched, control = [], [], []
    for n,end in ((36,60),(60,76),(76,88)):
        prefix=[dict(spec=z['spec'],**old.session.measure(rr,m,bounds,n)) for z,rr in zip(rows,returns)]
        winner=select(prefix)['primary']
        stitched.extend(returns[winner['spec']['id']][bounds[n]:bounds[end]])
        control.extend(returns[1][bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=winner['spec']))
    dump('WALKFORWARD.json',dict(folds=folds,pristine_oos=False,metrics=rp.stats_rr(np.array(stitched),m['dates'][36],m['ends'][87]),incumbent=rp.stats_rr(np.array(control),m['dates'][36],m['ends'][87])))
    dump('TRAIN_TRIALS.json',rows)
    np.savez_compressed(P/'TRAIN_RETURNS.npz',returns=np.stack(returns),bounds=np.array(bounds))
    files=('PLAN.json','GRID.json','DATA.npz','META.json','INCUMBENT_POLICY.json','FEATURE_AUDIT.json','INPUT_HASHES.json','PROTECTED.json','study.py','features.py')
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,
        new_admitted=sum(z['admitted'] and z['spec']['factor']!='none' for z in rows),trials=len(rows),
        hashes={file:sha(P/file) for file in files},pristine_oos=False,diagnostics_may_be_unqualified=True,author_labels_used=False))
    (P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('Frozenprimary',selected['primary']['spec'],selected['primary']['training'],flush=True)


def evaluate():
    f=read(P/'FROZEN.json')
    assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
    old.assert_hashes(f['hashes'],P)
    m,a,px,op=load()
    blocks,_,bounds,valid=rp.paths(px,m,len(m['dates']))
    orders={}
    for name,z in f['selected'].items():
        ch,sc=choices(z['spec'],a,m,len(m['dates']))
        orders[name]=rp.make_orders(ch,sc,m)
    orders['qqq']=[dict(date=m['dates'][0],signal_date=m['dates'][0],end=m['ends'][-1],symbols=['QQQ'],scores=[])]
    results,memo={},{}
    for name,oo in orders.items():
        key=json.dumps(oo,sort_keys=True)
        if key not in memo:
            met,eq,tr,hold=ex.e.engine.replay(px,op,oo,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
            for year in met['annual_returns']:
                if year['year'] in (2016,2026):year['partial']=True
            common=dict(metrics=met,slices=arch.old.prior.old.previous.cuts(eq,m))
            for label,cost in (('next_open_zero',0.),('next_open_cost',.0005)):
                mm,ee,_,_=ex.e.engine.replay(px,op,oo,cost=cost,execution='open',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
                for year in mm['annual_returns']:
                    if year['year'] in (2016,2026):year['partial']=True
                common[label]=dict(metrics=mm,slices=arch.old.prior.old.previous.cuts(ee,m))
            memo[key]=common,eq,tr,hold
        common,eq,tr,hold=memo[key]
        if name!='qqq':
            ch=[[m['symbols'][i].index(s) for s in row['symbols']] for i,row in enumerate(oo)]
            rr=rp.fast_returns(ch,blocks,valid)
            np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
            assert abs(common['slices']['training2016_2023']['cagr']-f['selected'][name]['training']['cagr'])<1e-10
        results[name]=dict(common,selection=f['selected'].get(name))
        for suffix,value in (('ORDERS',oo),('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()]),('TRADES',tr),('HOLDINGS',hold)):
            dump(name+'_'+suffix+'.json',value)
        print(name,round(common['metrics']['cagr']*100,3),round(common['metrics']['sharpe'],3),flush=True)
    previous_results=read(previous.P/'RESULTS.json')
    for name in ('original65','incumbent66'):
        for key in ('cagr','sharpe','max_drawdown'):
            assert abs(results[name]['metrics'][key]-previous_results[name]['metrics'][key])<1e-10
    months=[]
    for i,day in enumerate(m['dates']):
        row=dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
        for name,oo in orders.items():
            if name=='qqq':continue
            ss=oo[i]['symbols']
            r=px.loc[m['ends'][i],ss]/px.loc[day,ss]-1 if ss else []
            row[name]=dict(symbols=ss,monthly_return=float(np.mean(r)) if len(r) else 0.)
        months.append(row)
    dump('MONTH_COMPARISON.json',months)
    attribution={}
    for name in f['selected']:
        if name in ('original65','incumbent66'):continue
        changes=[]
        for row in months:
            x,b=row[name],row['incumbent66']
            if set(x['symbols'])!=set(b['symbols']):
                changes.append(dict(date=row['date'],split=row['split'],old=b['symbols'],new=x['symbols'],difference=x['monthly_return']-b['monthly_return'],log_contribution=float(np.log1p(x['monthly_return'])-np.log1p(b['monthly_return']))))
        positive=sorted([z['log_contribution'] for z in changes if z['log_contribution']>0],reverse=True)
        met=results[name]['metrics']
        undo=lambda n:float(((1+met['total_return'])*np.exp(-sum(positive[:n])))**(1/met['years'])-1)
        attribution[name]=dict(changed_months=len(changes),changes=changes,undo_best1_diagnostic_cagr=undo(1),undo_best3_diagnostic_cagr=undo(3))
    dump('ATTRIBUTION.json',attribution)
    exposure={}
    for name,oo in orders.items():
        if name=='qqq':continue
        counts=[len(row['symbols']) for row in oo]
        exposure[name]=dict(cash_months=counts.count(0),two_asset_months=counts.count(2),three_asset_months=counts.count(3),invested_month_fraction=sum(k>=2 for k in counts)/len(counts),cash_dates=[row['date'] for row in oo if not row['symbols']])
    dump('EXPOSURE.json',exposure)
    protected=read(P/'PROTECTED.json')
    changed=[file for file,digest in protected.items() if sha(file)!=digest]
    assert not changed,changed
    dump('RESULTS.json',results)
    dump('VALIDATION.json',dict(original_controls_exact=True,independent_fixed_unit_replay=True,
        training_freeze_parity=True,protected_files=len(protected),protected_changes=changed,
        observer_accounts_changed=False,pristine_oos=False,fixed_configs=6,
        primary_target70_reached=results['primary']['metrics']['cagr']>=.70))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('initialize','prepare','train','evaluate'))
    globals()[parser.parse_args().phase]()

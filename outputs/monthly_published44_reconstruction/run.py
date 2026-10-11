"""Read-only replay against 44 published month-end capitalization rankings."""
from pathlib import Path
import hashlib, itertools, json, sys
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
O = P.parent
sys.path.insert(0, str(O/'monthly_factor_search_v2'))
sys.path.insert(0, str(O/'monthly_multilayer_reconstruction'))
from data_inputs import load_inputs, sha, read, market_states
from search import features
import validate as engine
import research as layered


def dump(name, value):
    (P/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def actual(s):
    return {'META':'research_equity_FB','BRK-B':'BRK.B'}.get(s,s)


def display(s):
    return {'research_equity_FB':'META','BRK.B':'BRK-B'}.get(s,s)


def corrected_stocks(event, exclude_skhy=False):
    rows=event['top32']+event['boundary33_40']
    return [r for r in rows if not (
        r['symbol']=='LTM' and '2020-06-22'<=event['date']<'2024-07-25')
        and not(exclude_skhy and r['symbol']=='SKHY')][:32]


def pack_rows(px, feat, dates, symbols, names, norm):
    X=np.zeros((len(dates),49,len(names)))
    eligible=np.zeros((len(dates),49),bool)
    warm=px.notna().rolling(295,min_periods=295).sum().eq(295)&px.gt(0)
    for i,(d,ss) in enumerate(zip(dates,symbols)):
        assert len(ss)<=49 and len(ss)==len(set(ss))
        tab=pd.DataFrame({n:feat[n].loc[d,ss] for n in names})
        X[i,:len(ss)]=engine.normalize(tab,norm).to_numpy()
        eligible[i,:len(ss)]=(warm.loc[d,ss]&feat['mom252_skip42'].loc[d,ss].gt(0)).to_numpy()
    return X,eligible


def state_extra(spec, X, names, states, dates):
    if 'correction' not in spec:
        return None
    state,name=spec['correction'].split('__',1)
    return X[:,:,names.index(name)]*states.loc[dates,state].to_numpy()[:,None]


def select(scores, eligible, symbols):
    orders=[]
    for i,ss in enumerate(symbols):
        ids=[j for j in np.argsort(-np.where(eligible[i],scores[i],-np.inf),kind='stable')[:3] if eligible[i,j]]
        if len(ids)<2:ids=[]
        orders.append(([ss[j] for j in ids],[float(scores[i,j]) for j in ids]))
    return orders


def matches(orders, periods):
    hit=[len(set(ss)&{actual(s) for s in row['holdings']}) for (ss,_),row in zip(orders,periods)]
    return dict(periods=len(hit),exact=sum(n==3 for n in hit),shared=sum(hit),total_slots=3*len(hit),
                exact_rate=sum(n==3 for n in hit)/len(hit),slot_match_rate=sum(hit)/(3*len(hit)),per_month=hit)


def independent_equity(px, orders):
    nav=100000.;pieces=[]
    for i,o in enumerate(orders):
        z=px.loc[o['date']:o['end'],o['symbols']]
        assert np.isfinite(z.to_numpy()).all() and (z.to_numpy()>0).all()
        eq=z.div(z.iloc[0]).mean(axis=1)*nav if len(o['symbols']) else pd.Series(nav,index=z.index)
        pieces.append(eq if i==0 else eq.iloc[1:]);nav=float(eq.iloc[-1])
    return pd.concat(pieces)


def run():
    px,raw,members,allperiods,sources=load_inputs()
    ranks=read(O/'us_listed_top32_yahoo/PUBLISHED_MONTHLY_RANKINGS.json')['periods']
    dates=[e['date'] for e in ranks];by={r['start']:r for r in allperiods};periods=[by[d] for d in dates]
    assert len(periods)==44 and all(a['end']==b['start'] for a,b in zip(periods,periods[1:]))
    opening=engine.load_open(px,raw)*raw['close']/px
    for s in ['SHEL','SKHY']:
        f=P/'prices'/(s+'.csv');sources[str(f)]=sha(f)
        z=pd.read_csv(f,index_col=0,parse_dates=True)
        px[s]=z['Adj Close'].reindex(px.index)
        for k,c in [('close','Close'),('high','High'),('low','Low'),('volume','Volume')]:raw[k][s]=z[c].reindex(px.index)
        opening[s]=z.Open.reindex(px.index)
    # Include previous universe members too, for the unchanged-pool replay control.
    keep={actual(r['symbol']) for e in ranks for r in e['top32']+e['boundary33_40'] if r['symbol']!='LTM'}
    keep|={s for m,etfs in members.values() for s in etfs}
    keep|={s for m,_ in members.values() for d in dates for s in m.columns[m.loc[d]]}
    keep|={'SPY','QQQ','XLK','XLY','XLV','XLF','XLI','XLE','XLC','XLP','XLU','XLB','XLRE'}
    keep=sorted(keep&set(px.columns))
    px=px[keep];raw={k:v.reindex(index=px.index,columns=keep) for k,v in raw.items()}
    raw['opening']=opening.reindex(index=px.index,columns=keep)
    for s,b in [('SHEL','XLE'),('SKHY','XLK')]:raw['sector60'][s]=px[b].pct_change(60,fill_method=None)
    sectors=pd.read_csv(O/'index_union/alias_resolved_inputs/sectors.csv.gz',index_col=0,parse_dates=True)
    mapping={'Technology':'XLK','Consumer Cyclical':'XLY','Healthcare':'XLV','Financial Services':'XLF','Industrials':'XLI','Energy':'XLE','Communication Services':'XLC','Consumer Defensive':'XLP','Utilities':'XLU','Basic Materials':'XLB','Real Estate':'XLRE'}
    ret=px.pct_change(fill_method=None);sectorret=pd.DataFrame(np.nan,index=px.index,columns=keep)
    for s in keep:
        if s in members['theme17'][1]:sectorret[s]=ret.SPY
        elif s in {'TSM','BABA','NVO','SAP','TM','SHEL','SKHY'}:
            sectorret[s]=ret[{'TSM':'XLK','BABA':'XLY','NVO':'XLV','SAP':'XLK','TM':'XLY','SHEL':'XLE','SKHY':'XLK'}[s]]
        elif s in sectors:
            for cat,b in mapping.items():sectorret.loc[sectors[s].eq(cat),s]=ret.loc[sectors[s].eq(cat),b]
    raw['sector_return']=sectorret
    core,side=features(px,raw);feat={**core,**side,**layered.shape_features(px,raw)}
    names=list(core)+list(side);states=market_states(px)*2-1
    modelmeta=read(O/'monthly_multilayer_reconstruction/FEATURES.json')
    chosen=read(O/'monthly_multilayer_reconstruction/LEADERS.json')['A']
    specs=read(O/'monthly_factor_search_v2/SELECTED_CASES.json')
    specs+=[dict(name='pure_mom252_skip42',key='theme17__percentile',recipe=[names.index('mom252_skip42'),-1,-1,1.,0.,0.])]
    latest=dict(chosen['rule'],name='latest_multilayer')
    correction_plan=read(O/'monthly_factor_search_v2'/(latest['key']+'_CORRECTIONS_PLAN.json'))
    latest['correction']=correction_plan['extra_names'][latest['extra']];specs.append(latest)
    pack={};symbolsets={};membership_audit=[]
    for pool in ['theme17','core17']:
        etfs=members[pool][1]
        for mode in ['published','old','without_skhy']:
            ss=[]
            for e in ranks:
                symbols=sorted({actual(r['symbol']) for r in corrected_stocks(e,mode=='without_skhy')}|etfs) if mode!='old' else sorted(members[pool][0].columns[members[pool][0].loc[e['date']]])
                assert all(s in px for s in symbols),(e['date'],set(symbols)-set(px))
                ss.append(symbols)
            symbolsets[pool,mode]=ss
            for norm in ['percentile','magnitude']:
                pack[pool+'__'+norm,mode]=pack_rows(px,feat,dates,ss,names,norm)
    elig=pack['theme17__percentile','published'][1]
    for i,(e,p) in enumerate(zip(ranks,periods)):
        symbols=symbolsets['theme17','published'][i];target={actual(s) for s in p['holdings']}
        allowed={symbols[j] for j in np.flatnonzero(elig[i])}
        stockrows=corrected_stocks(e)
        membership_audit.append(dict(date=e['date'],author=p['holdings'],stocks=[r['symbol'] for r in stockrows],
            missing_from_pool=sorted(display(s) for s in target-set(symbols)),
            fails_signal_gate=sorted(display(s) for s in (target&set(symbols))-allowed),
            available_target_slots=len(target&allowed),eligible_count=len(allowed),
            ineligible=sorted(display(s) for s in set(symbols)-allowed)))
    dump('POOL_AUDIT.json',membership_audit)
    corrected=dict(source=str(O/'us_listed_top32_yahoo/PUBLISHED_MONTHLY_RANKINGS.json'),
                   adjustments=read(P/'PLAN.json')['universe_corrections'],periods=[dict(date=e['date'],top32=corrected_stocks(e)) for e in ranks])
    dump('CORRECTED_POOLS.json',corrected)
    # Use the already-frozen annual model, changing neither fit nor hyperparameters.
    def ml_scores(mode,X,eligible,symbols):
        import joblib
        meta=modelmeta;pred=np.zeros(X.shape[:2]);qshape=np.zeros_like(pred)
        for i,(d,ss) in enumerate(zip(dates,symbols)):
            vals=pd.DataFrame({n:feat[n].loc[d,ss] for n in meta['names']})
            q=engine.normalize(vals,'percentile');base=q.to_numpy()
            xx=np.column_stack([base]+[base*states.at[d,k] for k in states])
            year=pd.Timestamp(periods[i]['end']).year
            modelpath=O/'monthly_multilayer_reconstruction/models'/('%d_%d.joblib'%(year,chosen['bundle']['model_ids'][1]))
            sources[str(modelpath)]=sha(modelpath);model=joblib.load(modelpath)
            yy=np.einsum('ij,j->i',xx,model['coef'],optimize=False)+model['intercept']
            ids=np.flatnonzero(eligible[i,:len(ss)])
            pred[i,ids]=pd.Series(yy[ids]).rank(pct=True).to_numpy()-.5
            qshape[i,:len(ss)]=(-q.rebound21_atr-q.positive_gap5_vol)/2
        c,a,b,wc,wa,wb=latest['recipe'];cc=X[:,:,c]*wc;support=X[:,:,a]*wa+X[:,:,b]*wb
        extra=state_extra(latest,X,names,states,dates);g=latest['gamma']
        support=support*(1-g/(1-wc))+g*extra
        assert chosen['weights']==[0.,.05,0.,.05] and chosen['shape']=='chase_penalty'
        return cc+support*(1-.10/(1-wc))+.05*pred+.05*qshape
    allresults={};monthly=[];eqs={};controls=[]
    opening_adj=raw['opening']*px/raw['close']
    def evaluate(name, selections, mode, spec=None):
        ordered=[dict(date=p['start'],end=p['end'],signal_date=p['start'],symbols=ss,scores=sc) for p,(ss,sc) in zip(periods,selections)]
        metrics,eq,trades,holds=engine.replay(px,opening_adj,ordered,cost=0.,execution='close',start=pd.Timestamp(dates[0]),end=pd.Timestamp(periods[-1]['end']))
        np.testing.assert_allclose(eq,independent_equity(px,ordered),rtol=1e-11,atol=1e-6)
        match=matches(selections,periods);result=dict(name=name,universe=mode,matches=match,performance=metrics,spec=spec)
        allresults[name+'__'+mode]=result;eqs[name+'__'+mode]=eq
        for i,(p,(ss,sc)) in enumerate(zip(periods,selections)):
            monthly.append(dict(model=name,universe=mode,date=p['start'],end=p['end'],author=p['holdings'],
                selected=[display(s) for s in ss],scores=sc,shared=match['per_month'][i],exact=match['per_month'][i]==3))
        dump(name+'__'+mode+'_HOLDINGS.json',holds);dump(name+'__'+mode+'_TRADES.json',trades)
        return result
    for spec in specs:
        pool=spec['key'].split('__')[0]
        for mode in ['old','published','without_skhy']:
            X,eligible=pack[spec['key'],mode];ss=symbolsets[pool,mode]
            scores=ml_scores(mode,X,eligible,ss) if spec['name']=='latest_multilayer' else engine.score_of(spec,X,state_extra(spec,X,names,states,dates))
            picks=select(scores,eligible,ss)
            if mode=='old' and spec['name']!='pure_mom252_skip42':
                if spec['name']=='latest_multilayer':expected={r['date']:{actual(s) for s in r['selected']} for r in chosen['details']}
                else:expected={r['signal_date']:{actual(s) for s in r['symbols']} for r in read(O/'monthly_factor_search_v2'/(spec['name']+'_close_zero_HOLDINGS.json'))}
                mismatch=[d for d,(selected,_) in zip(dates,picks) if set(selected)!=expected[d]]
                controls.append(dict(model=spec['name'],old_holdings_reproduced=not mismatch,mismatch_dates=mismatch))
                assert not mismatch,(spec['name'],mismatch)
            z=evaluate(spec['name'],picks,mode,spec)
            print(spec['name'],mode,z['matches']['exact'],z['matches']['shared'],round(z['performance']['cagr'],4),flush=True)
    evaluate('author',[(list(map(actual,p['holdings'])),[]) for p in periods],'supplied_ledger')
    # Defined finite search: all three month-decision slots, no return-based selection.
    X,eligible=pack['theme17__percentile','published'];ss=symbolsets['theme17','published']
    labels=np.zeros_like(eligible)
    for i,(p,symbols) in enumerate(zip(periods,ss)):
        labels[i,:len(symbols)]=[display(s) in p['holdings'] for s in symbols]
    cs=[names.index('mom%d_%s'%(n,suffix)) for n in [189,252,294] for suffix in ['skip42','risk_recent63','risk_aligned','risk_half']]
    sn=['vol63','neg_vol63','downside63','neg_downside63','cmf21','neg_cmf21','cmf63','neg_cmf63',
        'industry_relative60','market_relative63','acceleration_skip42','ma200_distance','high252_distance',
        'dollar_volume_ratio21_126','signed_volume63','path_efficiency63']
    si=[names.index(s) for s in sn]
    recipes=[]
    for c in cs:
        recipes.append([c,-1,-1,1.,0.,0.])
        for a in si:
            for w in [.1,.2,.3,.4,.5]:recipes.append([c,a,-1,1-w,w,0.])
        for a,b in itertools.combinations(si,2):
            if names[a]=='neg_'+names[b] or names[b]=='neg_'+names[a]:continue
            for wc,wa,wb in [(.8,.1,.1),(.6,.2,.2),(.5,.25,.25),(.6,.3,.1)]:
                recipes.append([c,a,b,wc,wa,wb])
                if wa!=wb:recipes.append([c,b,a,wc,wa,wb])
    dev=np.array(dates)<'2025-01-01';later=~dev
    trials=[];bestall=None;bestdev=None
    for begin in range(0,len(recipes),128):
        batch=recipes[begin:begin+128];params=np.array(batch);ff=params[:,:3].astype(int);ww=params[:,3:]
        score=np.moveaxis(X[:,:,ff[:,0]],2,0)*ww[:,0,None,None]
        for j in [1,2]:score+=np.moveaxis(X[:,:,np.maximum(ff[:,j],0)],2,0)*ww[:,j,None,None]
        order=np.argsort(-np.where(eligible[None],score,-np.inf),axis=2,kind='stable')[:,:,:3]
        good=np.take_along_axis(np.broadcast_to(eligible,score.shape),order,axis=2)
        hit=(np.take_along_axis(np.broadcast_to(labels,score.shape),order,axis=2)&good).sum(axis=2)
        hit=np.where(good.sum(axis=2)>=2,hit,0)
        for j,recipe in enumerate(batch):
            z=dict(id=begin+j,recipe=recipe,exact=int((hit[j]==3).sum()),shared=int(hit[j].sum()),
                   calibration_exact=int((hit[j,dev]==3).sum()),calibration_shared=int(hit[j,dev].sum()),
                   later_exact=int((hit[j,later]==3).sum()),later_shared=int(hit[j,later].sum()))
            trials.append(z)
            if bestall is None or (z['exact'],z['shared'])>(bestall['exact'],bestall['shared']):bestall=z
            if bestdev is None or (z['calibration_exact'],z['calibration_shared'])>(bestdev['calibration_exact'],bestdev['calibration_shared']):bestdev=z
    dump('MATCHING_TRIALS.json',trials)
    for name,entry in [('retrospective44_match_leader',bestall),('calibration23_match_leader',bestdev)]:
        spec=dict(name=name,key='theme17__percentile',recipe=entry['recipe'],selection=entry)
        score=engine.score_of(spec,X);evaluate(name,select(score,eligible,ss),'published',spec)
    qqq=px.loc[dates[0]:periods[-1]['end'],'QQQ'];qqq=qqq/qqq.iloc[0]*100000.;eqs['QQQ']=qqq
    dump('RESULTS.json',dict(start=dates[0],end=periods[-1]['end'],periods=44,cost_bps=0,
        runs=allresults,qqq=engine.stats(qqq),search_trials=len(trials),controls=controls,
        author_covered_months=sum(not a['missing_from_pool'] and not a['fails_signal_gate'] for a in membership_audit),
        max_possible_shared=sum(a['available_target_slots'] for a in membership_audit),
        normalization='Ranks and magnitude scaling calculated within each month-end candidate pool BEFORE eligibility filtering, inherited unchanged.',
        classification='Historical descriptive replay and matching research. No independent OOS claim; no new live strategy. Frozen-model test isolates pool substitution; finite new search selects by holdings matching only.'))
    dump('MONTHLY_COMPARISON.json',monthly)
    dump('DAILY_EQUITY.json',[dict(date=str(d.date()),**{k:float(eq.at[d]) for k,eq in eqs.items()}) for d in qqq.index])
    for f in [O/'us_listed_top32_yahoo/PUBLISHED_MONTHLY_RANKINGS.json',O/'monthly_multilayer_reconstruction/LEADERS.json',O/'monthly_multilayer_reconstruction/FEATURES.json',O/'monthly_multilayer_reconstruction/TRAINING_AUDIT.json',O/'monthly_factor_search_v2/SELECTED_CASES.json',Path(__file__),P/'PLAN.json']:
        sources[str(f)]=sha(f)
    assert all(sha(f)==h for f,h in sources.items())
    protected=read(P/'PROTECTED_BEFORE.json');assert all(sha(f)==h for f,h in protected.items())
    dump('SOURCE_HASHES.json',sources)
    dump('VALIDATION.json',dict(old_frozen_rules_reproduced=controls,independent_daily_replays=True,
         protected_unchanged=len(protected),complete=True,source_hashes_unchanged=True))
    print('COMPLETE',len(trials),'matching rules',flush=True)


if __name__=='__main__':
    run()

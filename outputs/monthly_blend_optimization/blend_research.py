"""Frozen-split monthly return/downside ensemble research. No observer mutation."""
from pathlib import Path
from datetime import datetime, timezone
import sys, json, itertools, argparse
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_state_rank_research'));import state_rank as old
prior,rp,ex,read,sha=old.prior,old.rp,old.ex,old.read,old.sha
ALPHAS=[10.,100.]

def dump(name,z):
    (P/name).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def initialize():
    assert not (P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
      purpose='Continue optimization through complementary return/downside forecasts and training horizon alignment; no observer changes.',
      universe='Exact122 archived Top32-proxy+theme17 monthly pools,2016-08-17..2026-09-24. Same positive skip42momentum/295warmup gate. Monthly model training uses2011+ inherited historical members beforeAug2016, most recent already-known archivedTop32 afterward.',
      forecast='Reuse existing annual prior5year weekly21session return and downside-rank Ridge. Train separate21session adverse-excursion Ridge; additionally train monthly-close to next-month-close return, adverse excursion and rank(return-.5adverse) Ridge on real monthly sessions. alpha10/100,12monthrecency half-life. Require every label price valid and labelend<=annualcutoff. Same84 observed features/interactions; no author labels.',
      scoring='Keep65.34long-core score unchanged, addfixed2.5%residual in calm10, replace only support with ML. Total addonbudget<=10%, proportionally scale when exceeded. Long core>=50%. Monthly fixedunits,max3equal,2half,<2cash;no intramonthtrades orpullbackgate.',
      grid='Rank ensembles: weekly ormonthly return/downside-rank blend shares0/25/50/75/100%,alphas10/100 independently. Economic utility:rank(predictedexcessreturn-lambda*max(predictedadverse,0)),lambda0/.25/.5/1,weekly/monthly,alphas10/100. Cross-frequency return ensemble25/50/75%. Regime specialist:75%return+25%downside-rank normally,reversed inweak_narrow. Each with2.5/5/7.5/10%budget xglobal/highvol/weak_narrow. Exactduplicate specs removed. Include65.34baseline,65.95residual,66.10previous best controls.',
      freeze='First88through2023-11-30 only; no outerlast34return inspection until code/data/specs freeze. Annual forecasting later may train on then-mature past marketdata. Reused historicalsample,not pristineOOS.',
      guard='Relative66.10 priorcandidate:trainSharpe>=prior-.05,DD>=prior-.03,early53CAGR>=prior,recent35CAGR>=prior-.02. PrimarymaxtrainCAGR. Robustworst-segmentSharpe plusCAGR. Retain each model-family best admissible. Additional sequentialprefix selection on36/60/76 before12/16/12 blocks for diagnostic; priorarchitectureitselfalreadydeveloped.',
      evaluation='Same-close0cost primary, next-open0cost and5bp/side. Independentdailyreplay, baselineparity, closedlabeltiming, prefixes,oldfilehashes. Postfreeze ablation for attribution only; no post-holdout tuning.',
      limitations='Estimated universe, assumedETFlist, proxysectorhistory and earliertrainingmembership remain. No claimauthorreplication or reliablefuture70/80/100%return.',
      protections='Oldsixdaily andmonthlyobservers/rules/ledgers/plans/fingerprints untouched. Newresearch savedhere only.'))
    h=read(old.P/'PROTECTED.json')
    h.update({str(f):sha(f) for f in old.P.rglob('*') if f.is_file() and f.suffix!='.pyc'})
    dump('PROTECTED.json',h)
    files=[old.P/n for n in ['FROZEN.json','TRAIN_DATA.npz','TRAIN_DATA.json','FORECAST_AUDIT.json','FEATURES.npz','FORECASTS.npz','RESULTS.json']]+[Path(old.__file__),Path(rp.__file__),Path(ex.e.engine.__file__)]
    dump('INPUT_HASHES.json',{str(f):sha(f) for f in files})

def label_to_end(px,date,end,ss):
    p=px.loc[date:end,ss].iloc[1:].div(px.loc[date,ss])-1
    y=(p.iloc[-1]-(px.at[end,'SPY']/px.at[date,'SPY']-1)).clip(-1,1)
    adverse=(-p.min()).clip(lower=0);valid=p.notna().all()&np.isfinite(y)&np.isfinite(adverse)
    return y.to_numpy(),adverse.to_numpy(),valid.to_numpy()

def monthly_prepare():
    m,d,L,x,px,op,f,a=old.load();panel,info=old.panel_load(m['etfs']);full=panel['adj'];feat,mom=old.raw_features(full,panel)
    etfs=set(m['etfs']);sectors=panel['sectors'];member=panel['member'];legacyfile=O/'monthly_reconstruction_v2/foreign/membership.csv.gz';legacy=pd.read_csv(legacyfile,index_col=0,parse_dates=True).fillna(False).astype(bool)
    warm=full.notna().rolling(295).sum().eq(295)&full.gt(0)
    monthends=full.index.to_series().groupby(full.index.to_period('M')).last().tolist()
    X=[];Y=[];A=[];G=[];S=[];offsets=[0];dates=[];ends=[];symbols=[];origins=[]
    for i,date in enumerate(monthends[:-1]):
        if date<pd.Timestamp('2011-01-01') or date>pd.Timestamp('2025-12-31'):continue
        previous=member.index[member.index<=date]
        if len(previous):
            origin=str(previous[-1].date());ss=sorted(set(member.columns[member.loc[previous[-1]]])|etfs)
        else:
            origin='inherited_pre2016_historical_proxy';ss=sorted(set(legacy.columns[legacy.loc[date]])|etfs)
        st=old.state_row(full,date,ss,etfs);xx,score,names=old.feature_row(date,ss,feat,mom,full,sectors,etfs,st)
        end=monthends[i+1];y,adverse,valid=label_to_end(full,date,end,ss)
        good=(warm.loc[date,ss]&mom[252].loc[date,ss].gt(0)).to_numpy()&valid
        X.append(xx);Y.append(y);A.append(adverse);G.append(good);S.append(score);offsets.append(offsets[-1]+len(ss));dates.append(str(date.date()));ends.append(str(end.date()));symbols.append(ss);origins.append(origin)
    events=np.load(old.P/'TRAIN_DATA.npz')['event']
    for i in [0,20,53,87,121]:
        day=m['dates'][i];ss=m['symbols'][i];xx,_,_=old.feature_row(day,ss,feat,mom,full,sectors,etfs,old.state_row(full,day,ss,etfs));np.testing.assert_allclose(xx,events[i],atol=1e-12)
    np.savez_compressed(P/'MONTHLY_TRAIN.npz',X=np.vstack(X),Y=np.concatenate(Y),adverse=np.concatenate(A),good=np.concatenate(G),past_score=np.concatenate(S),offsets=np.array(offsets),event=events)
    dump('MONTHLY_TRAIN.json',dict(dates=dates,ends=ends,symbols=symbols,pool_origins=origins,feature_names=names))
    h=read(P/'INPUT_HASHES.json');h.update(info['source_hashes']);h[str(legacyfile)]=sha(legacyfile);dump('INPUT_HASHES.json',h)
    print('Aligned monthly training',len(dates),'cross-sections, rows',len(np.concatenate(Y)),flush=True)

def fit_point(data,meta,cutoff,target,alpha):
    ids=old.training_indices(meta,cutoff);xs=[];ys=[];ws=[];used=[]
    for i in ids:
        sl=slice(data['offsets'][i],data['offsets'][i+1]);good=data['good'][sl];xx=data['X'][sl][good];yy=data['Y'][sl][good]
        if len(yy)<4:continue
        if target=='adverse':yy=data['adverse'][sl][good]
        elif target=='utility':yy=prior.ranks(yy-.5*data['adverse'][sl][good])
        age=(pd.Timestamp(cutoff)-pd.Timestamp(meta['dates'][i])).days/30.4375;w=2**(-age/12)
        xs.append(xx);ys.append(yy);ws.extend([w/len(yy)]*len(yy));used.append(i)
    weights=np.array(ws);weights*=len(weights)/weights.sum();fit=ex.ridge.fit_weighted_ridge(np.vstack(xs),np.concatenate(ys),weights,alpha)
    return fit,dict(training_start=meta['dates'][used[0]],latest_label_end=max(meta['ends'][i] for i in used),cross_sections=len(used),examples=len(weights))

def eligible_rank(values,eligible):
    return np.stack([prior.ranks(pd.Series(row).where(good)) for row,good in zip(values,eligible)])

def forecasts():
    m,d,L,x,px,op,f,a=old.load();datasets={'W':(dict(np.load(old.P/'TRAIN_DATA.npz')),read(old.P/'TRAIN_DATA.json')),'M':(dict(np.load(P/'MONTHLY_TRAIN.npz')),read(P/'MONTHLY_TRAIN.json'))}
    audit=[];raw={f'{mode}_{target}_{int(alpha)}':np.zeros_like(d['core']) for mode,target,alpha in itertools.product(['W','M'],['return','utility','adverse'],ALPHAS)}
    existing=read(old.P/'FORECAST_AUDIT.json')
    for year in range(2016,2027):
        ids=[i for i,end in enumerate(m['ends']) if pd.Timestamp(end).year==year];first=min(m['dates'][i] for i in ids);cutoff=min(first,str((pd.Timestamp(f'{year}-01-01')-pd.Timedelta(days=1)).date()))
        for mode,target,alpha in itertools.product(['W','M'],['return','utility','adverse'],ALPHAS):
            data,meta=datasets[mode];key=f'{mode}_{target}_{int(alpha)}'
            if mode=='W' and target!='adverse':
                cfg=([0,1] if target=='return' else [4,5])[int(alpha==100.)];entry=next(z for z in existing if z['year']==year and z['configuration']==cfg)
                fit=dict(coef=np.array(entry['coef']),intercept=entry['intercept']);log={k:entry[k] for k in ['training_start','latest_label_end','weekly_samples','security_examples']};log['reused_frozen_model']=True
            else:fit,log=fit_point(data,meta,cutoff,target,alpha)
            assert log['latest_label_end']<=cutoff<=first
            pred=np.einsum('ijk,k->ij',data['event'][ids],fit['coef'],optimize=False)+fit['intercept']
            if target=='adverse':pred=pred.clip(0,1)
            raw[key][ids]=pred;audit.append(dict(year=year,key=key,cutoff=cutoff,alpha=alpha,target=target,frequency=mode,**log,coef=fit['coef'].tolist(),intercept=float(fit['intercept'])))
        print('Monthly/weekly annual fits',year,flush=True)
    ranks={k:eligible_rank(v,d['eligible']) for k,v in raw.items()}
    np.savez_compressed(P/'PREDICTIONS.npz',**{k+'_raw':v for k,v in raw.items()},**{k+'_rank':v for k,v in ranks.items()});dump('FORECAST_AUDIT.json',audit)
    for key,oldkey in [('W_return_10','ml_0'),('W_return_100','ml_1'),('W_utility_10','ml_4'),('W_utility_100','ml_5')]:np.testing.assert_allclose(ranks[key],a[oldkey],atol=1e-12)

def recipes():
    recipes=[];seen=set()
    def add(family,terms=None,**kwargs):
        recipe=dict(family=family,**kwargs)
        if terms is not None:
            terms=sorted((k,w) for k,w in terms if w>0);recipe['terms']=terms
        key=json.dumps(recipe,sort_keys=True)
        if key not in seen:seen.add(key);recipes.append(dict(id=len(recipes),**recipe))
    for mode,alpha,beta,share in itertools.product(['W','M'],[10,100],[10,100],[0.,.25,.5,.75,1.]):
        add('weekly_mix' if mode=='W' else 'monthly_mix',[(f'{mode}_return_{alpha}',share),(f'{mode}_utility_{beta}',1-share)])
    for mode,alpha,beta,penalty in itertools.product(['W','M'],[10,100],[10,100],[0.,.25,.5,1.]):
        if penalty==0 and beta==100:continue
        add('economic_utility',mode=mode,return_alpha=alpha,risk_alpha=beta,penalty=penalty)
    for alpha,beta,share in itertools.product([10,100],[10,100],[.25,.5,.75]):
        add('frequency_mix',[(f'W_return_{alpha}',share),(f'M_return_{beta}',1-share)])
    for mode,alpha,beta in itertools.product(['W','M'],[10,100],[10,100]):
        add('regime_mix',mode=mode,return_alpha=alpha,utility_alpha=beta)
    return recipes

def recipe_scores(recipe,pred,a,eligible):
    family=recipe['family']
    if 'terms' in recipe:return sum(w*pred[k+'_rank'] for k,w in recipe['terms'])
    mode=recipe['mode'];alpha=recipe['return_alpha']
    if family=='economic_utility':
        z=pred[f'{mode}_return_{alpha}_raw']-recipe['penalty']*pred[f"{mode}_adverse_{recipe['risk_alpha']}_raw"]
        return eligible_rank(z,eligible)
    if family=='regime_mix':
        r=pred[f'{mode}_return_{alpha}_rank'];u=pred[f"{mode}_utility_{recipe['utility_alpha']}_rank"];share=np.where(a['state_weak_narrow'][:,None],.25,.75)
        return share*r+(1-share)*u
    raise ValueError(family)

def grid(recs):
    rows=[dict(id=0,family='original65',spec=[]),dict(id=1,family='residual65',spec=read(prior.P/'FROZEN.json')['selected']['primary']['spec']),dict(id=2,family='previous66',spec=read(old.P/'FROZEN.json')['selected']['primary']['spec'])]
    residual=rows[1]['spec']
    for rec,budget,state in itertools.product(recs,[.025,.05,.075,.10],['global_','highvol','weak_narrow']):
        rows.append(dict(id=len(rows),family=rec['family'],recipe_id=rec['id'],spec=residual+[dict(key=f"recipe_{rec['id']}",amount=budget,state=state)]))
    return rows

def load():
    for p,h in read(P/'INPUT_HASHES.json').items():assert sha(p)==h,p
    m,d,L,x,px,op,f,a=old.load();pred=dict(np.load(P/'PREDICTIONS.npz'));recs=recipes()
    for rec in recs:a[f"recipe_{rec['id']}"]=recipe_scores(rec,pred,a,d['eligible'])
    return m,d,L,x,px,op,f,a,recs

def measure(row,m,d,a,blocks,bounds,valid):return old.measure(row,m,d,a,blocks,bounds,valid)

def train():
    assert not (P/'FROZEN.json').exists()
    m,d,L,x,px,op,f,a,recs=load();blocks,_,bounds,valid=rp.paths(px,m,88);rows=[];returns=[]
    for spec in grid(recs):
        z,rr=measure(spec,m,d,a,blocks,bounds,valid);rows.append(z);returns.append(rr)
        if len(rows)%200==0:print('Training configurations',len(rows),flush=True)
    base=rows[2];allowed=lambda r:prior.allowed(r,base);good=[r for r in rows if allowed(r)]
    controls={}
    for fam in ['weekly_mix','monthly_mix','economic_utility','frequency_mix','regime_mix']:
        subset=[r for r in good if r['family']==fam];controls[fam]=max(subset,key=lambda r:(r['training']['cagr'],-r['id'])) if subset else base
        print('Family',fam,'admissible',len(subset),'best',controls[fam]['id'],flush=True)
    selected=dict(primary=max(good,key=lambda r:(r['training']['cagr'],-r['id'])),robust=max(good,key=lambda r:(r['robust'],-r['id'])))
    wf=[];stitched=[];prevparts=[];originalparts=[]
    for n,end in [(36,60),(60,76),(76,88)]:
        met=[rp.stats_rr(rr[:bounds[n]],m['dates'][0],m['ends'][n-1]) for rr in returns];b=met[2]
        ids=[i for i,z in enumerate(met) if z['sharpe']>=b['sharpe']-.05 and z['max_drawdown']>=b['max_drawdown']-.03];win=max(ids,key=lambda i:(met[i]['cagr'],-i))
        part=returns[win][bounds[n]:bounds[end]];pb=returns[2][bounds[n]:bounds[end]];ob=returns[0][bounds[n]:bounds[end]];stitched.extend(part);prevparts.extend(pb);originalparts.extend(ob)
        wf.append(dict(train_end=m['ends'][n-1],test_start=m['dates'][n],test_end=m['ends'][end-1],chosen_id=win,spec=rows[win]['spec'],metrics=rp.stats_rr(part,m['dates'][n],m['ends'][end-1]),previous66=rp.stats_rr(pb,m['dates'][n],m['ends'][end-1])))
    dump('WALKFORWARD.json',dict(folds=wf,metrics=rp.stats_rr(np.array(stitched),m['dates'][36],m['ends'][87]),previous66=rp.stats_rr(np.array(prevparts),m['dates'][36],m['ends'][87]),original65=rp.stats_rr(np.array(originalparts),m['dates'][36],m['ends'][87]),pristine_oos=False))
    dump('RECIPES.json',recs);dump('TRAIN_TRIALS.json',rows);np.savez_compressed(P/'TRAIN_RETURNS.npz',returns=np.stack(returns),bounds=np.array(bounds))
    fr=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=selected,family_controls=controls,base=base,original=rows[0],residual=rows[1],trials=len(rows),admissible=len(good),code_sha256=sha(__file__),input_hashes={n:sha(P/n) for n in ['PLAN.json','MONTHLY_TRAIN.npz','MONTHLY_TRAIN.json','PREDICTIONS.npz','FORECAST_AUDIT.json','RECIPES.json','INPUT_HASHES.json']})
    dump('FROZEN.json',fr);(P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n');print('FROZEN',[(k,v['id'],v['training']['cagr'],v['early']['cagr']) for k,v in selected.items()],flush=True)

def evaluate():
    fr=read(P/'FROZEN.json');assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip();assert sha(__file__)==fr['code_sha256']
    for n,h in fr['input_hashes'].items():assert sha(P/n)==h,n
    m,d,L,x,px,op,f,a,recs=load();out={};orders_all={};blocks,_,bounds,valid=rp.paths(px,m,122)
    for name,row in dict(original65=fr['original'],previous66=fr['base'],residual65=fr['residual'],**fr['family_controls'],**fr['selected']).items():
        ch,sc=old.choices(row['spec'],a,m,d,122);orders=rp.make_orders(ch,sc,m)
        met,eq,tr,hold=ex.e.engine.replay(px,op,orders,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
        rr=rp.fast_returns(ch,blocks,valid);np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
        out[name]=dict(metrics=met,slices=prior.old.previous.cuts(eq,m),spec=row['spec'],recipe=recs[row['recipe_id']] if 'recipe_id' in row else None,id=row['id'])
        assert abs(out[name]['slices']['training2016_2023']['cagr']-row['training']['cagr'])<1e-10
        if name in ['original65','previous66']:
            ref=read(old.P/'RESULTS.json')['baseline' if name=='original65' else 'primary']['metrics']
            for k in ['cagr','sharpe','max_drawdown']:assert abs(met[k]-ref[k])<1e-10
        for label,cost in [('next_open_zero',0.),('next_open_cost',.0005)]:
            mm,ee,_,_=ex.e.engine.replay(px,op,orders,cost=cost,execution='open',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]));out[name][label]=dict(metrics=mm,slices=prior.old.previous.cuts(ee,m))
        for suffix,z in [('ORDERS',orders),('TRADES',tr),('HOLDINGS',hold),('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])]:dump(name+'_'+suffix+'.json',z)
        orders_all[name]=orders;print(name,'CAGR',met['cagr'],'Sharpe',met['sharpe'],'early',out[name]['slices']['early2016_2020']['cagr'],'later',out[name]['slices']['later34']['cagr'],flush=True)
    q=px.loc[m['dates'][0]:m['ends'][-1],'QQQ'];eq=q/q.iloc[0]*100000;out['qqq']=dict(metrics=ex.e.engine.stats(eq),slices=prior.old.previous.cuts(eq,m));dump('RESULTS.json',out)
    compare=[]
    for i,day in enumerate(m['dates']):
        r=dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
        for name,orders in orders_all.items():
            ss=orders[i]['symbols'];r[name]=dict(symbols=ss,monthly_return=float((px.loc[m['ends'][i],ss]/px.loc[day,ss]-1).mean()) if ss else 0.)
        compare.append(r)
    dump('MONTH_COMPARISON.json',compare);changed=[p for p,h in read(P/'PROTECTED.json').items() if sha(p)!=h];assert not changed,changed
    dump('VALIDATION.json',dict(independent_replay_parity=True,baseline122_parity=True,freeze_training_parity=True,protected_files=len(read(P/'PROTECTED.json')),protected_changes=changed,pristine_oos=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','monthly_prepare','forecasts','train','evaluate']);globals()[ap.parse_args().phase]()

"""Training-only local/regime mixtures after unsuccessful broad architecture sweep."""
import stability as s
from stability import np,pd,P,O,read,sha,dump,rp,ex,datetime,timezone
import itertools,argparse


def states(px,m,d):
    mv=px.SPY.pct_change(fill_method=None).rolling(63).std()*np.sqrt(252)
    relative=(px.QQQ/px.SPY).pct_change(126,fill_method=None)
    dates=m['dates']
    return dict(global_=np.ones(len(dates),bool),lowvol=~d['high'],highvol=d['high'],
        weak=d['weak'],calm10=mv.loc[dates].to_numpy()<.10,
        calm15=mv.loc[dates].to_numpy()<.15,calm20=mv.loc[dates].to_numpy()<.20,
        growth=relative.loc[dates].to_numpy()>0)


def score(spec,m,d,L,x,f,st,n):
    base=s.score({'baseline':True},m,d,L,x,f,n)
    alt=spec['alternative'];blend=spec['blend']*st[spec['state']][:n].astype(float)[:,None]
    alternative=s.score(alt,m,d,L,x,f,n)
    mixed=(1-blend)*base+blend*alternative
    aw=alt.get('core_weight',.52)
    acore=d['core'][:n] if alt.get('baseline') else aw*f[alt['core']][:n]
    core=(1-blend)*d['core'][:n]+blend*acore
    core_weight=(1-blend)*.52+blend*aw
    budget=spec['addon_budget']*st[spec['addon_state']][:n].astype(float)[:,None]
    result=core+(mixed-core)*(1-budget/(1-core_weight))
    if spec['addon']!='none':
        v=L[spec['addon']]['score'][:n] if spec['addon'] in L else f[spec['addon']][:n]
        result+=budget*v
    return result


def choose(spec,m,d,L,x,f,st,n):
    sc=score(spec,m,d,L,x,f,st,n)
    return rp.choose(sc,d['eligible'][:n],m['symbols'][:n],spec['retain']),sc


def evaluate_train(spec,m,d,L,x,f,st,blocks,valid,bounds):
    ch,_=choose(spec,m,d,L,x,f,st,88);rr=rp.fast_returns(ch,blocks,valid)
    met=rp.stats_rr(rr,m['dates'][0],m['ends'][87])
    early=rp.stats_rr(rr[:bounds[53]],m['dates'][0],m['ends'][52])
    recent=rp.stats_rr(rr[bounds[53]:],m['dates'][53],m['ends'][87])
    rob=min(early['sharpe'],recent['sharpe'])+.3*met['sharpe']+.4*np.log1p(met['cagr'])-.3*abs(met['max_drawdown'])
    return dict(spec=spec,training=met,early=early,recent=recent,robust=float(rob))


def train():
    assert not (P/'LOCAL_FROZEN.json').exists()
    m,d,L,x,px,op,f=s.load();st=states(px,m,d)
    old=read(P/'TRAIN_TRIALS.json');base=old[0];seeds={}
    for family in s.CORES:
        rows=[r for r in old if r['family']==family]
        for criterion in [('training','cagr'),('early','cagr')]:
            z=max(rows,key=lambda z:(z[criterion[0]][criterion[1]],-z['id']))
            seeds[z['id']]=z
    seeds[0]=base
    dump('LOCAL_PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        reason='No architecture in first1441 sweep passed combined constraints. Do not relax final constraints. Test convex mixtures and price-observed market activation using training-derived seeds only.',
        seed_selection='Per core family highest trainingCAGR and highest early-trainingCAGR, plus baseline. Selected from first88 training periods only.',
        candidates='Each seed xblend10/25/50/100% xglobal/relative-lowvol/highvol/weak/SPYannvol<10/15/20%/QQQvsSPY126strong xretain3/5. Then top8admissible distinct configurations xaddon shapes/Ridge/tail/stability x2.5/5/10% xglobal/highvol/lowvol.',
        core='Convex blend retains coreweight>=50%. Addons funded only by noncore support. No date/symbol-specific activation, same-signal rolling market states only.',
        split='Same88training/34later, not pristine OOS. First broad sweep holdout not read. Freeze before evaluating this experiment.',
        selection='Same early/recent/Sharpe/drawdown constraints; maxtrainCAGR primary and worst-blockSharpe-aware robust control. No further search after evaluation.',
        sources_sha256=sha(P/'TRAIN_TRIALS.json')))
    blocks,_,bounds,valid=rp.paths(px,m,88);rows=[];seen=set()
    def add(spec,stage,seed):
        key=json.dumps(spec,sort_keys=True)
        if key in seen:return
        seen.add(key);z=evaluate_train(spec,m,d,L,x,f,st,blocks,valid,bounds)
        rows.append(dict(z,id=len(rows),stage=stage,seed=seed))
    default=dict(alternative={'baseline':True},blend=0.,state='global_',retain=3,
        addon='none',addon_budget=0.,addon_state='global_')
    add(default,'control',0)
    for seed,blend,state,retain in itertools.product(seeds.values(),[.1,.25,.5,1.],list(st),[3,5]):
        if seed['id']==0:continue
        add(dict(default,alternative=seed['spec'],blend=blend,state=state,retain=retain),'mixture',seed['id'])
        if len(rows)%400==0:print('Local training',len(rows),flush=True)
    good=[r for r in rows if s.acceptable(r,base)]
    top=sorted(good,key=lambda z:z['training']['cagr'],reverse=True)[:8]
    for seed,key,budget,state in itertools.product(top,
            ['anti_chase','volume_recovery','pressure_easing','ridge_10_12','tail','stability'],
            [.025,.05,.10],['global_','highvol','lowvol']):
        add(dict(seed['spec'],addon=key,addon_budget=budget,addon_state=state),'addon',seed['id'])
    good=[r for r in rows if s.acceptable(r,base)]
    selected={'return_primary':max(good,key=lambda z:(z['training']['cagr'],-z['id'])),
        'robust_control':max(good,key=lambda z:(z['robust'],-z['id']))}
    dump('LOCAL_TRAIN_TRIALS.json',rows)
    frozen=dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,baseline=rows[0],
        candidate_count=len(rows),admissible_count=len(good),code_sha256=sha(__file__),
        architecture_code_sha256=sha(P/'stability.py'),feature_sha256=sha(P/'FEATURES.npz'))
    dump('LOCAL_FROZEN.json',frozen);(P/'LOCAL_FROZEN.sha256').write_text(sha(P/'LOCAL_FROZEN.json')+'\n')
    for name,z in selected.items():print('FROZEN',name,z,flush=True)
    print('Local total',len(rows),'admissible',len(good),flush=True)


def evaluate():
    frozen=read(P/'LOCAL_FROZEN.json')
    assert sha(P/'LOCAL_FROZEN.json')==(P/'LOCAL_FROZEN.sha256').read_text().strip()
    assert sha(__file__)==frozen['code_sha256']
    assert sha(P/'stability.py')==frozen['architecture_code_sha256']
    assert sha(P/'FEATURES.npz')==frozen['feature_sha256']
    m,d,L,x,px,op,f=s.load();st=states(px,m,d);out={};allorders={}
    for name,row in dict(baseline=frozen['baseline'],**frozen['selected']).items():
        ch,sc=choose(row['spec'],m,d,L,x,f,st,122);orders=rp.make_orders(ch,sc,m)
        met,eq,tr,holds=ex.e.engine.replay(px,op,orders,cost=0.,execution='close',
            start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
        blocks,dates,bounds,valid=rp.paths(px,m,122);rr=rp.fast_returns(ch,blocks,valid)
        np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000.,rtol=1e-11,atol=1e-5)
        z=dict(metrics=met,slices=s.previous.cuts(eq,m),spec=row['spec'])
        assert abs(z['slices']['training2016_2023']['cagr']-row['training']['cagr'])<1e-10
        if name=='baseline':
            prior=read(ex.P/'RESULTS.json')['return_primary']
            for k in ['cagr','sharpe','max_drawdown']:assert abs(met[k]-prior['metrics'][k])<1e-10
            assert [o['symbols'] for o in orders]==[o['symbols'] for o in read(ex.P/'return_primary_ORDERS.json')]
        for label,mode,cost in [('next_open_zero','open',0.),('next_open_cost','open',.0005)]:
            mm,ee,_,_=ex.e.engine.replay(px,op,orders,cost=cost,execution=mode,
                start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
            z[label]=dict(metrics=mm,slices=s.previous.cuts(ee,m))
        for suffix,value in [('ORDERS',orders),('TRADES',tr),('HOLDINGS',holds),
            ('DAILY',[dict(date=str(day.date()),equity=float(v)) for day,v in eq.items()])]:dump(name+'_'+suffix+'.json',value)
        out[name]=z;allorders[name]=orders
        print(name,dict(full=met['cagr'],sharpe=met['sharpe'],dd=met['max_drawdown'],
            early=z['slices']['early2016_2020']['cagr'],recent=z['slices']['original2021_2026']['cagr'],
            later=z['slices']['later34']['cagr']),flush=True)
    q=px.loc[m['dates'][0]:m['ends'][-1],'QQQ'];qe=q/q.iloc[0]*100000.
    out['qqq']=dict(metrics=ex.e.engine.stats(qe),slices=s.previous.cuts(qe,m));dump('RESULTS.json',out)
    compare=[]
    for i,day in enumerate(m['dates']):
        r=dict(date=day,end=m['ends'][i],split='early' if i<53 else 'recent_train' if i<88 else 'later34')
        for name,orders in allorders.items():
            ss=orders[i]['symbols'];r[name]=dict(symbols=ss,monthly_return=float((px.loc[m['ends'][i],ss]/px.loc[day,ss]-1).mean()) if ss else 0.)
        compare.append(r)
    dump('MONTH_COMPARISON.json',compare)
    changes=[p for p,h in read(P/'PROTECTED.json').items() if sha(p)!=h];assert not changes,changes
    dump('VALIDATION.json',dict(baseline_exact=True,independent_path_parity=True,freeze_train_parity=True,
        protected_files=len(read(P/'PROTECTED.json')),protected_changes=changes,author_targets=False,pristine_oos=False))


if __name__=='__main__':
    import json
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['train','evaluate'])
    globals()[ap.parse_args().phase]()

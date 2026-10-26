"""Monthly market-path interactions, isolated from all observer accounts."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util
import itertools
import argparse
import json
import sys
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent;O=P.parent
sp=importlib.util.spec_from_file_location('phase_prior',O/'monthly_opportunity_alignment/study.py')
prev=importlib.util.module_from_spec(sp);sp.loader.exec_module(prev)
sys.path.insert(0,str(P))
ep,rp,arch,ex,read,sha=prev.ep,prev.rp,prev.arch,prev.ex,prev.read,prev.sha
TRAIN=88
BASES=['original65','simple']
PROFILES={'distress_lowdown':('pressure','lowdown'), 'repair_volatility':('repair','volatility'),
          'duration_relative':('underwater','relative')}
PEERS=['peer_participation','peer_broadening']
PEER_CONTROLS={'peer_participation':'market_participation','peer_broadening':'market_broadening'}


def dump(name,value):
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n')


def grid():
    rows=[]
    for base in BASES:
        rows.append(dict(id=len(rows),base=base,profile='none',mode='control',budget=0.))
        for profile,mode,budget in itertools.product(PROFILES,['phase','static'],[.025,.075]):
            rows.append(dict(id=len(rows),base=base,profile=profile,mode=mode,budget=budget))
        for profile,budget in itertools.product(PEERS+list(PEER_CONTROLS.values()),[.025,.075]):
            rows.append(dict(id=len(rows),base=base,profile=profile,mode='peer',budget=budget))
    return rows


def initialize():
    assert not (P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Market drawdown/recovery path interactions vs exactstatic controls; separatelylabelled historical pricecorrelation peer participation, not certifiedindustrybreadth.',
        unchanged='122 monthly estimatedTop32+assumed17ETF pools2016-08-17..2026-09-24; original signal-only eligibility and positive longmomentum295sessionhistory. Longcore skips42sessions; threeequal/twohalf/<2cash; monthlyfixedunits,no intramonthstops,no leverage. Currentrisk andmarketpath supplementaryfeatures permitted.',
        states='Trailing252 SPYadjustedcloseprices including signal. Last maximum, then last observed minimum since that peak. depth=1-current/peak;peak_age;trough_depth=1-min/peak;recovered=(current-min)/(peak-min),flat0;trough_age. pressure=clip(depth/.20,0,1);repair=clip(trough_depth/.20,0,1)*recovered*(1-clip(trough_age/126,0,1));underwater=clip(peak_age/126,0,1). Rollingpeak expiration is causalreferencechange, not permanentcyclelabel. No futuretrough/recovery identification.',
        interactions='distress_lowdown:pressure*rank(-dailylogdownsideRMS126);repair_volatility:repair*existingrankvol126;duration_relative:underwater*existingstock-vs-own-sectorETFskip42relative126/252rankaverage,ETFneutral. Every profile exactstatic no-statecontrol. Missingstate/factor exactzero; never eligibility filter.',
        peers='Available-data historicalNasdaq100/SP100 researchunion, nottoday503survivors andnotstrictPITcertified. Explicitidentitymappings only; atsignal select up-to10 distinctotherissuers withpositive past252return correlation,minimum5. Excludefocalissuerallclasses. Contemporaryeligible peers withpositivecomplete263prices tocompare200MA nowand63sessionsago. Participation=mean(aboveMA200)-.5;broadening=(meanabovenow-meanabove63ago)/2 usingSAMEcurrentselectedpeers. ETFs/unmapped/insufficient exactzero correction, no targeteligibilityfilter. No sectorlabel claim. Datareadiness limitations persisted; peer features not observermigration.',
        scoring='Original65.34 and independent75%rawskip42momentum+25%vol126. Existingcoreunchanged. support=(base-core)/(1-coreweight). score=base+budget*(clip(support+effect,-.5,.5)-support), effect centeredrankfactor*phaseintensity. Budgets2.5/7.5%; all3slotsfree. Samephaseavailability in staticcontrol for fairmissingcoverage.',
        grid='42 fixedconfigs:2bases*(1control+3marketprofiles*2modes*2budgets+2peerprofiles*2budgets*2peer/globalcomparators). No combinations or further grid afterholdout.',
        peer_controls='Exactmatched broaderavailable-data market breadth:allotherdistinctdata-qualifiedissuers instead ofpositivecorrelationtop10. Samefocalissuerexclusion,263completeprices,currentmembership andsamecurrentlist atbothdates. SAMEpeer_validtargetcoverage aslocalpeerfeature; ETFsneutral. Thisisolates peerselectionfromgenericstock-vs-ETF breadthtilting. Addedbefore anyperformancecomparisons.',
        selection='First88 through2023-11-30:trainCAGR>=original65;Sharpe>=base-.05;DD>=base-.03;early53CAGR>=base;recent35CAGR>=base-.02. MaxadmittedtrainingCAGR primaryelseoriginal65. Freeze eachbase/profile bestphasebudget and itsmatchedstatic; diagnosticsmayfailadmission. Last34 replayonlyafterfreeze,no reselection. Prefix36/60/76 sequentialdiagnostic.',
        execution='Primarysame-complete-close0costidealization,nextopen0and5bp/side sensitivity withsameorders. FirstAug2016,lastSep2026partial. Officialdailyfixedunitreplay crosschecked independentvectorengine.',
        limits='Previouslyreusedhistory,notpristineOOS;historicalcap/sectorproxies andunknownauthorETFlistremain. Existing66.36 exactorders comparisononly. Noauthorlabels;noautomaticpromotion;no changes to observers prices plans ledgers or fingerprints.',
        external_references=['https://www.nber.org/papers/w20439'],
        reference_scope='Market-state motivation only. Paper long/short findings do not validate this long-only monthly rule.'))
    dump('GRID.json',grid())
    protected=read(prev.P/'PROTECTED.json')
    protected.update({str(f):sha(f) for f in prev.P.rglob('*') if f.is_file() and f.suffix!='.pyc'})
    dump('PROTECTED.json',protected)


def prepare():
    from market_phases import build_states
    from peer_features import build_features
    m,old,px,op=prev.load();panel,info=arch.old.panel_load(m['etfs'])
    events=list(zip(m['dates'],m['symbols']))
    states,audit=build_states(panel,events)
    peers,peer_audit=build_features(panel,events,m['etfs'])
    arrays={'eligible':old['eligible']}
    for base in BASES:
        for prefix in ['base_','core_','cw_']:arrays[prefix+base]=old[prefix+base]
    for k,v in states.items():arrays['state_'+k]=v
    plain=old['event_plain'].reshape(122,49,10)
    arrays['volatility']=plain[:,:,2];arrays['relative']=plain[:,:,3]
    r=np.log(panel['adj']/panel['adj'].shift(1))
    downside=np.sqrt(r.clip(upper=0).pow(2).rolling(126,min_periods=126).mean())
    raw=np.stack([-downside.loc[day,ss].to_numpy() for day,ss in zip(m['dates'],m['symbols'])])
    arrays['lowdown']=np.stack([arch.rank_values(row) for row in raw])
    arrays['valid_lowdown']=np.isfinite(raw)
    arrays['valid_volatility']=np.isfinite(arrays['volatility'])
    arrays['valid_relative']=np.isfinite(arrays['relative'])
    arrays['lowdown'][~arrays['valid_lowdown']]=0.
    for key in PEERS+list(PEER_CONTROLS.values()):
        arrays[key]=np.where(np.isfinite(peers[key]),peers[key],0.)
        arrays['valid_'+key]=np.asarray(peers['peer_valid'],bool)&np.isfinite(peers[key])
    np.savez_compressed(P/'DATA.npz',**arrays);dump('META.json',dict(monthly=m));dump('FEATURE_AUDIT.json',audit)
    dump('PEER_FEATURE_AUDIT.json',peer_audit)
    hashes=dict(info['source_hashes'])
    hashes.update(peer_audit['source_hashes'])
    for file in [prev.P/'DATA.npz',prev.P/'FORECASTS.npz',prev.P/'FROZEN.json',Path(prev.__file__),
        Path(arch.__file__),Path(ep.__file__),Path(rp.__file__),Path(ex.e.engine.__file__)]:hashes[str(file)]=sha(file)
    dump('INPUT_HASHES.json',hashes)
    oo=rp.make_orders(rp.choose(arrays['base_original65'],arrays['eligible'],m['symbols'],3),arrays['base_original65'],m)
    assert oo==read(ep.P/'original65_ORDERS.json')
    dump('PREPARE_VALIDATION.json',dict(original_orders_exact=True,original_eligibility_unchanged=True,
        market_valid_events=int(states['valid'].sum()),eligible_missing_lowdown=int((arrays['eligible']&~arrays['valid_lowdown']).sum())))
    print('Prepared market-path interactions for122 unchanged snapshots',flush=True)


def load():
    for p,h in read(P/'INPUT_HASHES.json').items():assert sha(p)==h,p
    m,_,px,op=prev.load()
    return m,dict(np.load(P/'DATA.npz')),px,op


def score(row,a,n):
    base=a['base_'+row['base']][:n]
    if row['mode']=='control':return base.copy()
    if row['mode']=='peer':
        factor=row['profile'];valid=a['valid_'+factor][:n];intensity=1.
    else:
        state,factor=PROFILES[row['profile']]
        valid=a['valid_'+factor][:n]&a['state_valid'][:n,None]
        intensity=a['state_'+state][:n,None] if row['mode']=='phase' else 1.
    effect=np.where(valid,a[factor][:n]*intensity,0.)
    core,cw=a['core_'+row['base']][:n],a['cw_'+row['base']][:n]
    support=(base-core)/(1-cw)
    delta=np.where(effect==0,0.,np.clip(support+effect,-.5,.5)-support)
    return base+row['budget']*delta


def choose(row,a,m,n):
    sc=score(row,a,n)
    return rp.choose(sc,a['eligible'][:n],m['symbols'][:n],3),sc


def measure(rr,m,bounds,n=TRAIN):return prev.prev.prev.measure(rr,m,bounds,n)
def admitted(z,b):return prev.prev.prev.admitted(z,b)


def train():
    assert not (P/'FROZEN.json').exists()
    m,a,px,_=load();blocks,_,bounds,valid=rp.paths(px,m,TRAIN);rows=[];returns=[]
    for spec in grid():
        ch,_=choose(spec,a,m,TRAIN);rr=rp.fast_returns(ch,blocks,valid)
        rows.append(dict(spec=spec,**measure(rr,m,bounds)));returns.append(rr)
    baseline=rows[0]
    for z in rows:z['admitted']=admitted(z,baseline)
    best=lambda zs:max(zs,key=lambda z:(z['training']['cagr'],-z['spec']['id']))
    selected=dict(primary=best([z for z in rows if z['admitted']]),original65=baseline,
        simple=next(z for z in rows if z['spec']['base']=='simple' and z['spec']['mode']=='control'))
    for b,profile in itertools.product(BASES,PROFILES):
        win=best([z for z in rows if z['spec']['base']==b and z['spec']['profile']==profile and z['spec']['mode']=='phase'])
        selected[b+'_'+profile]=win
        selected[b+'_'+profile+'_static']=next(z for z in rows if z['spec']['base']==b and z['spec']['profile']==profile and z['spec']['mode']=='static' and z['spec']['budget']==win['spec']['budget'])
    for b,profile in itertools.product(BASES,PEERS):
        win=best([z for z in rows if z['spec']['base']==b and z['spec']['profile']==profile]);selected[b+'_'+profile]=win
        control=PEER_CONTROLS[profile]
        selected[b+'_'+control]=next(z for z in rows if z['spec']['base']==b and z['spec']['profile']==control and z['spec']['budget']==win['spec']['budget'])
    folds=[];parts=[];baseparts=[]
    for n,end in [(36,60),(60,76),(76,88)]:
        metrics=[dict(spec=z['spec'],**measure(rr,m,bounds,n)) for z,rr in zip(rows,returns)]
        win=best([z for z in metrics if admitted(z,metrics[0])])
        parts.extend(returns[win['spec']['id']][bounds[n]:bounds[end]]);baseparts.extend(returns[0][bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=win['spec']))
    dump('WALKFORWARD.json',dict(folds=folds,metrics=rp.stats_rr(np.array(parts),m['dates'][36],m['ends'][87]),baseline=rp.stats_rr(np.array(baseparts),m['dates'][36],m['ends'][87]),pristine_oos=False))
    dump('TRAIN_TRIALS.json',rows)
    files=['PLAN.json','GRID.json','DATA.npz','META.json','INPUT_HASHES.json','FEATURE_AUDIT.json','PEER_FEATURE_AUDIT.json','study.py','market_phases.py','peer_features.py']
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,trials=len(rows),
        admitted_noncontrols=sum(z['admitted'] and z['spec']['mode']!='control' for z in rows),hashes={f:sha(P/f) for f in files},author_labels_used=False,diagnostics_may_be_unqualified=True))
    (P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('Frozen primary:',selected['primary']['spec'],flush=True)


def evaluate():
    fr=read(P/'FROZEN.json');assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
    for f,h in fr['hashes'].items():assert sha(P/f)==h,f
    m,a,px,op=load();blocks,_,bounds,valid=rp.paths(px,m,122);orders={};choices={}
    for name,z in fr['selected'].items():
        ch,sc=choose(z['spec'],a,m,122);choices[name]=ch;orders[name]=rp.make_orders(ch,sc,m)
    orders['previous66']=read(ep.P/'primary_ORDERS.json')
    orders['qqq']=[dict(date=m['dates'][0],signal_date=m['dates'][0],end=m['ends'][-1],symbols=['QQQ'],scores=[])]
    results={}
    for name,oo in orders.items():
        met,eq,tr,ho=ex.e.engine.replay(px,op,oo,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
        for z in met['annual_returns']:
            if z['year'] in [2016,2026]:z['partial']=True
        cuts=arch.old.prior.old.previous.cuts(eq,m)
        if name in choices:
            rr=rp.fast_returns(choices[name],blocks,valid)
            np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
            assert abs(cuts['training2016_2023']['cagr']-fr['selected'][name]['training']['cagr'])<1e-10
        results[name]=dict(metrics=met,slices=cuts,selection=fr['selected'].get(name))
        for label,cost in [('next_open_zero',0.),('next_open_cost',.0005)]:
            mm,ee,_,_=ex.e.engine.replay(px,op,oo,cost=cost,execution='open',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
            for z in mm['annual_returns']:
                if z['year'] in [2016,2026]:z['partial']=True
            results[name][label]=dict(metrics=mm,slices=arch.old.prior.old.previous.cuts(ee,m))
        for suffix,data in [('ORDERS',oo),('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()]),('TRADES',tr),('HOLDINGS',ho)]:dump(name+'_'+suffix+'.json',data)
        print(name,round(met['cagr']*100,3),round(met['sharpe'],3),flush=True)
    for name in ['original65','previous66']:
        ref=read(prev.P/'RESULTS.json')[name]['metrics']
        for k in ['cagr','sharpe','max_drawdown']:assert abs(results[name]['metrics'][k]-ref[k])<1e-10
    comparison=[]
    for i,day in enumerate(m['dates']):
        row=dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34',states={k:float(a['state_'+k][i]) if np.isfinite(a['state_'+k][i]) else None for k in ['depth','peak_age','trough_depth','recovered','trough_age','pressure','repair','underwater']})
        for name,oo in orders.items():
            if name=='qqq':continue
            ss=oo[i]['symbols'];rr=px.loc[m['ends'][i],ss]/px.loc[day,ss]-1 if ss else pd.Series(dtype=float)
            row[name]=dict(symbols=ss,monthly_return=float(rr.mean()) if ss else 0.,stock_contributions={s:float(v/len(ss)) for s,v in rr.items()})
        comparison.append(row)
    attribution={}
    for name,z in fr['selected'].items():
        base=z['spec']['base'];changes=[]
        for r in comparison:
            x,b=r[name],r[base]
            if set(x['symbols'])!=set(b['symbols']):changes.append(dict(date=r['date'],split=r['split'],old=b['symbols'],new=x['symbols'],difference=x['monthly_return']-b['monthly_return'],log_contribution=float(np.log1p(x['monthly_return'])-np.log1p(b['monthly_return']))))
        pos=sorted([z['log_contribution'] for z in changes if z['log_contribution']>0],reverse=True)
        years=results[name]['metrics']['years'];wealth=1+results[name]['metrics']['total_return']
        attribution[name]=dict(base=base,changed_months=len(changes),changes=changes,undo_best1_cagr=float((wealth*np.exp(-sum(pos[:1])))**(1/years)-1),undo_best3_cagr=float((wealth*np.exp(-sum(pos[:3])))**(1/years)-1))
    dump('MONTH_COMPARISON.json',comparison);dump('ATTRIBUTION.json',attribution);dump('RESULTS.json',results)
    changed=[f for f,h in read(P/'PROTECTED.json').items() if sha(f)!=h];assert not changed,changed
    dump('VALIDATION.json',dict(original_controls_exact=True,independent_daily_replay=True,training_freeze_parity=True,protected_files=len(read(P/'PROTECTED.json')),protected_changes=changed,new_observation_accounts=False,pristine_oos=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','prepare','train','evaluate'])
    globals()[ap.parse_args().phase]()

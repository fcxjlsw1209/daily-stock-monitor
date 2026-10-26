"""Predeclared monthly trend cores and matched ordinal/return Ridge research."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import itertools
import json
import sys
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
P=Path(__file__).resolve().parent; O=P.parent
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(sp);sp.loader.exec_module(obj);return obj
prev=module('expansion_previous',O/'monthly_market_phase_research/study.py')
session=module('expansion_session',O/'monthly_session_type_research/study.py')
ep,rp,arch,ex,read,sha=prev.ep,prev.rp,prev.arch,prev.ex,prev.read,prev.sha
TRAIN=88
BASES=['original65','simple']
CORES=['ordinary_sharpe','hac_sharpe','market_residual_sharpe','rank_residence']
ARMS=[('none',0.),('return',.025),('return',.05),('ordinal',.025),('ordinal',.05)]

def dump(name,value):
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n')

def grid():
    rows=[]
    for base in BASES:
        designs=[('original',0.)]+list(itertools.product(CORES,[.5,1.]))
        for (core,blend),(arm,budget) in itertools.product(designs,ARMS):
            rows.append(dict(id=len(rows),base=base,core=core,blend=blend,arm=arm,budget=budget))
    return rows

def initialize():
    assert not (P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        objective='Broaden monthly strategy at its long-trend core, not only micro-adjust the current winner. No author-holding supervision.',
        guidance='Long momentum dominates and omits latest42sessions. Monthend top3equal/top2half/<2cash; positive longtrend and295price eligibility unchanged; no pullback gate, no intramonth stops or rotation, no leverage.',
        data='Same122 monthly estimated historicalTop32+assumedETF17 snapshots2016-08-17..2026-09-24. Initial partialmonth,final partialmonth/year. Pre2016 model pools and historicalsector/cap data remain inherited proxies, not exactauthor/unbiased certifieduniverse.',
        cores='OrdinarySharpe matchedcontrol:mean/sampleSTD of210simple returns endingt-42. HAC:mean/sqrt(gamma0+2sum_l1..5((1-l/6)*gamma_l)), gamma with1/210 denominator; fallbackordinary ifbadvariance. Residual:everyasset inclETF uses prior-day252returncov/varSPY beta; r-beta*SPY,no interceptsubtraction; same210skip42mean/sampleSTD. Nearzero residual includingSPYself givesraw0,notETFneutralization; incompletebeta/nonzerozerovariance fallbackordinary. Missingcompleteformation remainsmissing,never changeseligibility.',
        residence='For currenteligiblecandidate, its prior6calendar-signal cross-sectional raw210skip42momentum percentile history (samehistorical49pool eachdate); pastobservation onlyifthenpositive longtrend and295positiveprices. >=4valid appearances:mean-minuspopulationSTD; otherwisecurrentordinaryrawmomentumrank fallback. No carrying older observations tofillholes; no futuremembership. Initial2016-08-17currentonly, not as later monthlyhistory. Training and event snapshots dateunion, conflictsraise.',
        architectures='Two preexisting bases: original65 core50/52% with sameauxiliaryweights/state, and independent75%rawlongmomentum+25%vol126. Replace half/all weightedcore with cross-sectional rank of newcore; auxiliary remains exactsame. OrdinarySharpe replacement is paireddefinitioncontrol. All49 assets ranked before originaleligibility. No fixed top2, all3slotscompete. Coremissing=>thatasset originalcore fallback; no neweligibilityfilter.',
        ml='Sameexisting10plainfeatures for both targets, annual previousDec31 maturitycutoff and past5years, alpha10,24monthhalf-life,monthbalanced,min18months. Returntarget clipped[-1,1] excessSPY thenmonthcenter; ordinaltarget actualUNCLIPPED nextcalendar-month simple return midrank:2*(rank-.5)/N-1. Entire historicalpasteligible month dropped forboth targets if ANYqualifiedname orSPY lacks required complete positivepricepath. Futurelabel completeness never changes eventeligibility. No ticker/authorlabels or futurestandardization.',
        corrections='None,return2.5/5%,ordinal2.5/5%. bounded effect=.5*tanh(pred/scale), scale return.05,ordinal.25. support=(base-core)/(1-cw); addon=budget*(clip(support+effect,-.5,.5)-support); zeroeffect exactnochange. Each coreblend keepsoriginalcoreweight; ML modifiesonlysupport, notrankexpansionof tinyforecasts.',
        grid='90fixedconfigs=2bases*(originalcore+4coredefinitions*2blends)*(none+2targets*2budgets). No further variants after later-periodevaluation.',
        selection='First88through2023-11-30 only; sameoriginal65 trainCAGR/Sharpe-.05/DD-.03/early53CAGR/recent35CAGR-.02 guards. Highestadmitted trainCAGR primary, fallbackoriginal65. Also freeze eachbase/core noML trainingbestblend and samecore matchedordinary control; bestordinal percore with exactreturn andnone matchedcontrols; all diagnostics show admission. Later34 onlyafterfreeze,noreselection. Prefix36/60/76 chronologicalprocedurediagnostic.',
        validation='Fixedunitdailyengine vs independentvectorreturns; same-close0 primary, nextopen0 and5bp/side sensitivity. Changedmonths/topgains-attribution; bothold65/66controls parity. Freeze source/dependency/input hashes. Preserveexisting observers/rules/plans/ledgers/fingerprints, no promotion.',
        interpretation='Repeatedly researched history is NOT pristineOOS; no guarantee70/80/100% or authorrecovery. Rankprediction may improve rankaccuracy but miss payoffmagnitudes.',
        sources=['https://repub.eur.nl/pub/22252/ResidualMomentum-2011.pdf','https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios'],
        source_scope='Motivation only; neither paper validates this precise long-onlyEW3 implementation. HAC is a signal variant; reportedportfolioSharpe remains same conventionalsqrt252 statistic for comparison.'))
    dump('GRID.json',grid())
    protected=read(prev.P/'PROTECTED.json')
    protected.update({str(f):sha(f) for f in prev.P.rglob('*') if f.is_file() and f.suffix!='.pyc'})
    dump('PROTECTED.json',protected)

def prepare():
    sys.path.insert(0,str(P))
    from core_features import build_features
    from residence import build_features as residence_features
    from ordinal_model import build_targets
    m,old,px,op=session.load()
    panel,info=arch.old.panel_load(m['etfs'])
    tm=read(arch.P/'META.json')['training']
    events=list(zip(m['dates'],m['symbols']))
    raw,audit=build_features(panel,events)
    res,ra=residence_features(panel,tm,events)
    raw.update(res)
    a={k:old[k] for k in ['eligible','dates','ends','event_dates','offsets','train_plain','event_plain']}
    for b in BASES:
        for prefix in ['base_','core_','cw_']:a[prefix+b]=old[prefix+b]
    for key in CORES:
        vals=np.asarray(raw[key],float)
        a['factor_'+key]=np.stack([arch.rank_values(row) for row in vals])
        a['valid_'+key]=np.isfinite(vals)
    # Rebuild signal-time eligibility rather than reuse a future-label mask.
    _,_,good=arch.raw_frames(panel)
    eligible=np.concatenate([good.loc[day,ss].to_numpy(bool) for day,ss in zip(tm['dates'],tm['symbols'])])
    ss=[s for pool in tm['symbols'] for s in pool]
    targets,ta=build_targets(panel['adj'],a['dates'],a['ends'],ss,eligible)
    for key,v in targets.items():a['target_'+key]=v
    a['train_eligible']=eligible
    a['train_good']=eligible&np.isfinite(a['target_return'])&np.isfinite(a['target_ordinal'])
    assert len(eligible)==len(a['dates'])==len(ss)
    np.testing.assert_array_equal(a['eligible'],old['eligible'])
    expected=read(ep.P/'original65_ORDERS.json')
    assert rp.make_orders(rp.choose(a['base_original65'],a['eligible'],m['symbols'],3),a['base_original65'],m)==expected
    # Ordinary control must agree with existing aligned-Sharpe implementation.
    reference=dict(np.load(arch.P/'DATA.npz'))['event_all_aligned_sharpe']
    np.testing.assert_allclose(a['factor_ordinary_sharpe'],reference,atol=1e-12)
    np.savez_compressed(P/'DATA.npz',**a)
    dump('META.json',dict(monthly=m,training=tm))
    dump('FEATURE_AUDIT.json',dict(core=audit,residence=ra,missing_eligible={k:int((a['eligible']&~a['valid_'+k]).sum()) for k in CORES}))
    dump('TARGET_AUDIT.json',ta)
    hashes=dict(info['source_hashes'])
    for f in [session.P/'DATA.npz',session.P/'FORECASTS.npz',session.P/'FROZEN.json',arch.P/'DATA.npz',arch.P/'META.json',
              Path(session.__file__),Path(arch.__file__),Path(ep.__file__),Path(rp.__file__),Path(ex.e.engine.__file__),
              O/'monthly_architecture_restart/residual_model.py',Path(__file__),
              P/'core_features.py',P/'residence.py',P/'ordinal_model.py']:hashes[str(f)]=sha(f)
    dump('INPUT_HASHES.json',hashes)
    dump('PREPARE_VALIDATION.json',dict(original_orders_exact=True,original_eligibility_unchanged=True,ordinary_sharpe_reference_exact=True,training_rows=len(eligible),usable_label_rows=int(a['train_good'].sum())))
    print('Prepared new cores and matched actual-month targets',flush=True)

def forecasts():
    for f,h in read(P/'INPUT_HASHES.json').items():assert sha(f)==h,f
    sys.path.insert(0,str(P))
    from ordinal_model import annual_forecasts
    a=dict(np.load(P/'DATA.npz'))
    pred,audit=annual_forecasts(a['dates'],a['ends'],a['train_plain'],
        {k:a['target_'+k] for k in ['return','ordinal']},a['train_good'],a['event_dates'],a['event_plain'],a['eligible'].reshape(-1))
    np.savez_compressed(P/'FORECASTS.npz',**{k:np.asarray(v).reshape(a['eligible'].shape) for k,v in pred.items()})
    dump('FORECAST_AUDIT.json',audit)
    dump('FORECAST_INPUT_HASHES.json',{f:sha(P/f) for f in ['DATA.npz','ordinal_model.py']})
    print('Prepared annual matched return/ordinal Ridge forecasts',flush=True)

def load():
    for f,h in read(P/'INPUT_HASHES.json').items():assert sha(f)==h,f
    for f,h in read(P/'FORECAST_INPUT_HASHES.json').items():assert sha(P/f)==h,f
    m,_,px,op=session.load()
    return m,{**dict(np.load(P/'DATA.npz')),**{'pred_'+k:v for k,v in dict(np.load(P/'FORECASTS.npz')).items()}},px,op

def score(row,a,n):
    b=row['base'];base=a['base_'+b][:n];core=a['core_'+b][:n];cw=a['cw_'+b][:n]
    newcore=core.copy()
    if row['core']!='original':
        f=a['factor_'+row['core']][:n]
        alternative=np.where(a['valid_'+row['core']][:n],cw*f,core)
        newcore=core+row['blend']*(alternative-core)
    out=base+(newcore-core)
    if row['arm']=='none':return out
    scale=.05 if row['arm']=='return' else .25
    effect=np.where(a['pred_available'][:n],.5*np.tanh(a['pred_'+row['arm']][:n]/scale),0.)
    support=(base-core)/(1-cw)
    delta=np.where(effect==0,0.,np.clip(support+effect,-.5,.5)-support)
    return out+row['budget']*delta

def choose(row,a,m,n):
    sc=score(row,a,n);return rp.choose(sc,a['eligible'][:n],m['symbols'][:n],3),sc

def train():
    assert not (P/'FROZEN.json').exists()
    m,a,px,_=load();blocks,_,bounds,valid=rp.paths(px,m,TRAIN)
    rows=[];returns=[]
    for spec in grid():
        ch,_=choose(spec,a,m,TRAIN);rr=rp.fast_returns(ch,blocks,valid)
        rows.append(dict(spec=spec,**session.measure(rr,m,bounds)));returns.append(rr)
    baseline=rows[0]
    for z in rows:z['admitted']=session.admitted(z,baseline)
    best=lambda zs:max(zs,key=lambda z:(z['training']['cagr'],-z['spec']['id']))
    find=lambda **kw:next(z for z in rows if all(z['spec'][k]==v for k,v in kw.items()))
    selected=dict(primary=best([z for z in rows if z['admitted']]),original65=baseline,simple=find(base='simple',core='original',arm='none'))
    for b,c in itertools.product(BASES,['original']+CORES):
        subset=[z for z in rows if z['spec']['base']==b and z['spec']['core']==c]
        no=best([z for z in subset if z['spec']['arm']=='none'])
        selected[b+'_'+c+'_noML']=no
        z=best([z for z in subset if z['spec']['arm']=='ordinal'])
        label=b+'_'+c
        selected[label+'_ordinal']=z
        selected[label+'_return']=find(base=b,core=c,blend=z['spec']['blend'],arm='return',budget=z['spec']['budget'])
        selected[label+'_matched_noML']=find(base=b,core=c,blend=z['spec']['blend'],arm='none')
        if c not in ['original','ordinary_sharpe']:
            selected[label+'_matched_ordinary']=find(base=b,core='ordinary_sharpe',blend=no['spec']['blend'],arm='none')
    folds=[];parts=[];baseparts=[]
    for n,end in [(36,60),(60,76),(76,88)]:
        metrics=[dict(spec=z['spec'],**session.measure(rr,m,bounds,n)) for z,rr in zip(rows,returns)]
        win=best([z for z in metrics if session.admitted(z,metrics[0])])
        parts.extend(returns[win['spec']['id']][bounds[n]:bounds[end]]);baseparts.extend(returns[0][bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=win['spec']))
    dump('WALKFORWARD.json',dict(folds=folds,metrics=rp.stats_rr(np.array(parts),m['dates'][36],m['ends'][87]),baseline=rp.stats_rr(np.array(baseparts),m['dates'][36],m['ends'][87]),pristine_oos=False))
    dump('TRAIN_TRIALS.json',rows)
    files=['PLAN.json','GRID.json','DATA.npz','META.json','FORECASTS.npz','INPUT_HASHES.json','FEATURE_AUDIT.json','TARGET_AUDIT.json','FORECAST_AUDIT.json','study.py','core_features.py','residence.py','ordinal_model.py']
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,trials=len(rows),admitted_noncontrols=sum(z['admitted'] and (z['spec']['arm']!='none' or z['spec']['core']!='original') for z in rows),hashes={f:sha(P/f) for f in files},author_labels_used=False,diagnostics_may_be_unqualified=True))
    (P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('Frozen primary:',selected['primary']['spec'],flush=True)

def evaluate():
    fr=read(P/'FROZEN.json');assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
    for f,h in fr['hashes'].items():assert sha(P/f)==h,f
    m,a,px,op=load();blocks,_,bounds,valid=rp.paths(px,m,122)
    orders={};choices={}
    for name,z in fr['selected'].items():
        ch,sc=choose(z['spec'],a,m,122);choices[name]=ch;orders[name]=rp.make_orders(ch,sc,m)
    orders['previous66']=read(ep.P/'primary_ORDERS.json')
    orders['qqq']=[dict(date=m['dates'][0],signal_date=m['dates'][0],end=m['ends'][-1],symbols=['QQQ'],scores=[])]
    results={};memo={}
    for name,oo in orders.items():
        # Same frozen spec may have several diagnostic labels: compute once.
        key=json.dumps(oo,sort_keys=True)
        if key not in memo:
            met,eq,tr,ho=ex.e.engine.replay(px,op,oo,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
            for z in met['annual_returns']:
                if z['year'] in [2016,2026]:z['partial']=True
            common=dict(metrics=met,slices=arch.old.prior.old.previous.cuts(eq,m))
            for label,cost in [('next_open_zero',0.),('next_open_cost',.0005)]:
                mm,ee,_,_=ex.e.engine.replay(px,op,oo,cost=cost,execution='open',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
                for z in mm['annual_returns']:
                    if z['year'] in [2016,2026]:z['partial']=True
                common[label]=dict(metrics=mm,slices=arch.old.prior.old.previous.cuts(ee,m))
            memo[key]=(common,eq,tr,ho)
        common,eq,tr,ho=memo[key]
        if name in choices:
            rr=rp.fast_returns(choices[name],blocks,valid)
            np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
            assert abs(common['slices']['training2016_2023']['cagr']-fr['selected'][name]['training']['cagr'])<1e-10
        results[name]=dict(common,selection=fr['selected'].get(name))
        for suffix,data in [('ORDERS',oo),('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()]),('TRADES',tr),('HOLDINGS',ho)]:dump(name+'_'+suffix+'.json',data)
        print(name,round(common['metrics']['cagr']*100,3),round(common['metrics']['sharpe'],3),flush=True)
    for name in ['original65','previous66']:
        ref=read(prev.P/'RESULTS.json')[name]['metrics']
        for k in ['cagr','sharpe','max_drawdown']:assert abs(results[name]['metrics'][k]-ref[k])<1e-10
    comparison=[]
    for i,day in enumerate(m['dates']):
        row=dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
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
    dump('VALIDATION.json',dict(original_controls_exact=True,independent_daily_replay=True,training_freeze_parity=True,protected_files=len(read(P/'PROTECTED.json')),protected_changes=changed,new_observation_accounts=False,pristine_oos=False,unique_replay_specs=len(memo)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','prepare','forecasts','train','evaluate']);globals()[ap.parse_args().phase]()

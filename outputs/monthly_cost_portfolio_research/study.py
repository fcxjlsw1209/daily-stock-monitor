"""Frozen volume-reference and actual-month joint-trio utility experiments."""
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
prev=module('cost_previous',O/'monthly_core_expansion/study.py')
session,ep,rp,arch,ex,read,sha=prev.session,prev.ep,prev.rp,prev.arch,prev.ex,prev.read,prev.sha
adapter=module('cost_base_adapter',O/'monthly_opportunity_alignment/baseline_adapter.py')
TRAIN=88
BASES=['original65','simple']
KEYS=[f'{weight}_{feature}{w}' for weight,feature,w in itertools.product(['volume','time'],['gap','upper_mass'],[126,252])]

def dump(name,value):
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n')

def grid():
    rows=[]
    for b in BASES:
        rows.append(dict(id=len(rows),base=b,family='control',key='none',budget=0.))
        for k,budget in itertools.product(KEYS,[.025,.05]):
            rows.append(dict(id=len(rows),base=b,family='cost',key=k,budget=budget))
        for k,budget in itertools.product(['additive','joint'],[.025,.05,.10]):
            rows.append(dict(id=len(rows),base=b,family='trio',key=k,budget=budget))
    return rows

def initialize():
    assert not (P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Execute volume-reference proxy, then true three-slot portfolio utility, then historical earnings readiness. No author holding labels.',
        guidance='Same122 estimated historicalTop32+assumedETF17 pools2016-08-17..2026-09-24. Positive longtrend/295prices eligibility; core skips42sessions; monthend3equal/2half/<2cash, fixedunits intramonth. No leverage or new pullback gate.',
        bases='Original65.34 score and independent75%rawlongmomentum+25%vol126; exact previous66.36 orders as additional control.',
        cost='126/252 observations endingat signal, exponential63session half-life. Adjusted-unit volume q=rawclose*rawvolume/adjclose. Reference=weightedmean(adjclose,decay*q); gap=log(currentadj/reference); uppermass=weightedfraction historicaladj>currentadj. Gap positive, uppermass negative; no sign search. Matchedtime controls q=1 with identicalprice/volumeavailabilitymask. This is a traded-volume price-location proxy, not true investor cost, outstanding-share turnover, or intraday volume-at-price. Dividend basis and inheritedvendor unit limitations disclosed.',
        cost_correction='Midpoint centeredpercentiles overfinitecurrentpool values;allties0;missing0 withouteligibilitychange. Budget2.5/5%fromnoncore support via clip(support+effect,-.5,.5)-support. Longcoreunchanged.',
        trio='Original eachbase top8eligible atsignal; enumerateall56triples, all3slotscanchange. Baseaverage individualscore plus bounded learnedutility correction. No automatic sector diversification or hard correlation constraint. Prior jointworkfit author holdings or locked top2; here realfuturemature monthlyreturns train allslots.',
        features='Additive10=mean existing10plainstockfeatures minusoriginaltop3mean. Joint13=additive10 plus differences fromoriginaltop3 in meanpaircorrelation126, meandownsidecosine126 and cross-asset dailyvolatilitydispersion126. Risk uses127positiveclose observations endingatsignal. No per-ticker/date features.',
        labels='log(1+arithmeticmean nextcalendar-month simple returns oftrio) minus samequantity oforiginaltop3. Every originallyeligibleasset andSPY requires fullpositivefinite signal-to-end path; missingone invalidates entiretrainingmonth forbotharms, not eventqualification. Trainfeaturesjointcomplete mask sharedbotharms. BaselinecenteredX andtarget0; nointercept.',
        models='FixedannualpreviousDec31 maturitycutoff, previous5years, alpha10,24monthhalf-life,min18months. Eachmonth totalweight1 beforeageweight, not56independentmonths. Same samples additive/joint. Baselineprediction exactly0. Eventend=None nofuturelabels; unavailableforecast effect0.',
        trio_correction='For eachtrio: meanbasescore +budget*(clip(meanbasesupport+.5*tanh(pred/.05),-.5,.5)-meanbasesupport). Budget2.5/5/10%, fixedcore. Zero/unavailable correction preservesoriginaltop3; exacttie favorsoldtop3, then deterministic base-rank order. No re-rankingforecasts or volatility-basedleveraging.',
        grid='46fixedconfigs=2bases*(1control+8costsignals*2budgets+2trioarms*3budgets). No combinations orpost-laterperiodextensions inthisround.',
        selection='First88through2023-11-30 only. Relativeoriginal65 guardstrainCAGR>=base,Sharpe>=base-.05,DD>=base-.03,early53CAGR>=base,recent35CAGR>=base-.02. HighestadmittedtrainCAGR primary elseoriginal65. Freeze eachbase bestvolumecost plus EXACTtimecontrol atsamekey/window/budget; bestjointtrio plus EXACTadditive atsamebudget, andbesttime diagnostic. Diagnostics maynotqualify. Prefix36/60/76 procedurediagnostics. Later34 onlyafterfreeze, no reselection.',
        execution='Zero-cost same-final-close idealized primary; nextopenzero and5bp/side sensitivity. Partialfirst/lastperiods. Allhistory reused and proxyuniverse limitations, notnewpristineOOS orauthorrecovery.',
        earnings='Readinessonly unless independentlysufficient verifiedhistoricalcoverage and preregistered newexperiment. ExistingAVtodaylimit honored, no retries/payments/secretlogging. SourceIRreleasedates/times distinct fromfiscalperiodend orSECacceptance.',
        protection='No changes to existingobserveraccounts/rules/plans/ledgers/codefingerprints. Recordallsource/codehashes, freezechoices, independentdailyenginevalidation; no automaticpromotion.',
        sources=['https://www.nber.org/papers/w8734','https://www.nber.org/papers/w5375'],
        source_scope='Motivation only. No literature paper validates these exactfeatures or concentrated monthlystrategy.'))
    dump('GRID.json',grid())
    protect=read(prev.P/'PROTECTED.json')
    protect.update({str(f):sha(f) for f in prev.P.rglob('*') if f.is_file() and f.suffix!='.pyc'})
    dump('PROTECTED.json',protect)

def centered_rank(values):
    x=np.asarray(values,float);out=np.zeros(x.shape)
    good=np.isfinite(x);n=int(good.sum())
    if n:
        out[good]=(pd.Series(x[good]).rank(method='average').to_numpy()-.5)/n-.5
    return out

def encode_months(meta,scores,eligible,plain,px,training):
    from trio_model import build_month
    records=[]; offset=[0];add=[];joint=[];ys=[];eg=[];lg=[];dates=[];ends=[];audits=[]
    for i,(day,ss) in enumerate(zip(meta['dates'],meta['symbols'])):
        z=build_month(px,day,meta['ends'][i] if training else None,ss,eligible[i],scores[i],plain[i])
        c=len(z['indices']);offset.append(offset[-1]+c)
        records.append(dict(event=i,indices=np.asarray(z['indices'],int).tolist()))
        add.append(z['X_additive']);joint.append(z['X_joint']);ys.extend(z['y'])
        eg.extend(z['event_good']);lg.extend(z['label_good'])
        dates.extend([day]*c);ends.extend([meta['ends'][i]]*c);audits.append(z['audit'])
    return dict(offsets=np.asarray(offset,int),X_additive=np.concatenate(add),X_joint=np.concatenate(joint),
        y=np.asarray(ys,float),event_good=np.asarray(eg,bool),label_good=np.asarray(lg,bool),
        dates=np.array(dates,dtype='datetime64[D]'),ends=np.array(ends,dtype='datetime64[D]')),records,audits

def prepare():
    sys.path.insert(0,str(P))
    from cost_features import build_features
    m,old,px,op=prev.load();panel,info=arch.old.panel_load(m['etfs'])
    tm=read(arch.P/'META.json')['training'];nt=len(tm['dates'])
    raw,audit=build_features(panel,list(zip(m['dates'],m['symbols'])))
    a={k:old[k] for k in ['eligible','train_plain','event_plain']}
    for b in BASES:
        for prefix in ['base_','core_','cw_']:a[prefix+b]=old[prefix+b]
    for k in KEYS:
        vals=np.asarray(raw[k],float);direction=-1 if 'upper_mass' in k else 1
        a[k]=np.stack([centered_rank(direction*v) for v in vals]);a['valid_'+k]=np.isfinite(vals)
    events=list(zip(tm['dates']+m['dates'],tm['symbols']+m['symbols']))
    exact=adapter.prepare(panel,events,m['etfs'])
    np.testing.assert_allclose(exact['score'][nt:],a['base_original65'],atol=1e-12)
    np.testing.assert_array_equal(exact['eligible'][nt:],a['eligible'])
    trainplain=a['train_plain'].reshape(nt,49,10);eventplain=a['event_plain'].reshape(122,49,10)
    records={};ta={}
    for b in BASES:
        train_scores=exact['score'][:nt] if b=='original65' else .75*trainplain[:,:,0]+.25*trainplain[:,:,2]
        tr,trec,traudit=encode_months(tm,train_scores,exact['eligible'][:nt],trainplain,panel['adj'],True)
        er,erec,eraudit=encode_months(m,a['base_'+b],a['eligible'],eventplain,panel['adj'],False)
        for prefix,arrays in [('train',tr),('event',er)]:
            for k,v in arrays.items():a[b+'_'+prefix+'_'+k]=v
        records[b]=dict(training=trec,events=erec)
        ta[b]=dict(training=traudit,events=eraudit)
    np.savez_compressed(P/'DATA.npz',**a)
    dump('META.json',dict(monthly=m,training=tm,trios=records))
    dump('FEATURE_AUDIT.json',dict(cost=audit,missing_eligible={k:int((a['eligible']&~a['valid_'+k]).sum()) for k in KEYS}))
    dump('TRIO_AUDIT.json',ta)
    hashes=dict(info['source_hashes'])
    for f in [prev.P/'DATA.npz',prev.P/'FORECASTS.npz',prev.P/'FROZEN.json',prev.P/'INPUT_HASHES.json',
              arch.P/'META.json',Path(prev.__file__),Path(arch.__file__),Path(ep.__file__),Path(rp.__file__),
              Path(ex.e.engine.__file__),Path(adapter.__file__),*adapter.SOURCE_PATHS,
              O/'monthly_opportunity_alignment/portfolio_model.py',O/'monthly_architecture_restart/residual_model.py',
              Path(__file__),P/'cost_features.py',P/'trio_model.py']:
        hashes[str(f)]=sha(f)
    dump('INPUT_HASHES.json',hashes)
    oo=rp.make_orders(rp.choose(a['base_original65'],a['eligible'],m['symbols'],3),a['base_original65'],m)
    assert oo==read(ep.P/'original65_ORDERS.json')
    dump('PREPARE_VALIDATION.json',dict(original_orders_exact=True,training_and_event_score_parity=True,
         eligibility_unchanged=True,cost_missing_eligible=read(P/'FEATURE_AUDIT.json')['missing_eligible']))
    print('Prepared cost features and matched actual-month trios',flush=True)

def forecasts():
    for f,h in read(P/'INPUT_HASHES.json').items():assert sha(f)==h,f
    sys.path.insert(0,str(P));from trio_model import annual_forecasts
    a=dict(np.load(P/'DATA.npz'));out={};audits={}
    for b,key in itertools.product(BASES,['additive','joint']):
        pre=b+'_train_';ev=b+'_event_'
        z=annual_forecasts(a[pre+'dates'],a[pre+'ends'],a[pre+'X_'+key],a[pre+'y'],
            a[pre+'label_good']&a[pre+'event_good'],a[ev+'dates'],a[ev+'X_'+key],a[ev+'event_good'])
        out[b+'_'+key]=z['pred'];out[b+'_valid_'+key]=z['available'];audits[b+'_'+key]=z['audit']
        print('Forecast',b,key,flush=True)
    np.savez_compressed(P/'FORECASTS.npz',**out);dump('FORECAST_AUDIT.json',audits)
    dump('FORECAST_INPUT_HASHES.json',{f:sha(P/f) for f in ['DATA.npz','META.json','trio_model.py']})

def load():
    for f,h in read(P/'INPUT_HASHES.json').items():assert sha(f)==h,f
    for f,h in read(P/'FORECAST_INPUT_HASHES.json').items():assert sha(P/f)==h,f
    m,_,px,op=prev.load()
    return m,{**dict(np.load(P/'DATA.npz')),**dict(np.load(P/'FORECASTS.npz'))},px,op

def corrected_score(row,a,n):
    b=row['base'];sc=a['base_'+b][:n].copy()
    if row['family']!='cost':return sc
    support=(sc-a['core_'+b][:n])/(1-a['cw_'+b][:n])
    effect=np.where(a['valid_'+row['key']][:n],a[row['key']][:n],0.)
    delta=np.where(effect==0,0.,np.clip(support+effect,-.5,.5)-support)
    return sc+row['budget']*delta

def select_combo(indices,base_scores,core,cw,pred,available,budget):
    ids=np.asarray(indices,int)
    if not len(ids):return None
    mean=base_scores[ids].mean(axis=1)
    support=((base_scores-core)/(1-cw))[ids].mean(axis=1)
    effect=np.where(available,.5*np.tanh(pred/.05),0.)
    bonus=np.where(effect==0,0.,np.clip(support+effect,-.5,.5)-support)*budget
    utility=mean+bonus
    return ids[int(np.argmax(utility))]

def choose(row,a,m,n,records):
    sc=corrected_score(row,a,n);ch=rp.choose(sc,a['eligible'][:n],m['symbols'][:n],3)
    if row['family']!='trio':return ch,sc
    b=row['base'];off=a[b+'_event_offsets']
    for i in range(n):
        lo,hi=off[i:i+2];ids=records[b]['events'][i]['indices']
        selected=select_combo(ids,sc[i],a['core_'+b][i],a['cw_'+b][i],a[b+'_'+row['key']][lo:hi],
            a[b+'_valid_'+row['key']][lo:hi],row['budget'])
        if selected is not None:ch[i]=list(selected)
    return ch,sc

def train():
    assert not (P/'FROZEN.json').exists()
    m,a,px,_=load();records=read(P/'META.json')['trios']
    blocks,_,bounds,valid=rp.paths(px,m,TRAIN);rows=[];returns=[]
    for spec in grid():
        ch,_=choose(spec,a,m,TRAIN,records);rr=rp.fast_returns(ch,blocks,valid)
        rows.append(dict(spec=spec,**session.measure(rr,m,bounds)));returns.append(rr)
    baseline=rows[0]
    for z in rows:z['admitted']=session.admitted(z,baseline)
    best=lambda zs:max(zs,key=lambda z:(z['training']['cagr'],-z['spec']['id']))
    find=lambda **kw:next(z for z in rows if all(z['spec'][k]==v for k,v in kw.items()))
    selected=dict(primary=best([z for z in rows if z['admitted']]),original65=baseline,simple=find(base='simple',family='control'))
    for b in BASES:
        z=best([z for z in rows if z['spec']['base']==b and z['spec']['family']=='cost' and z['spec']['key'].startswith('volume')])
        selected[b+'_volume']=z
        selected[b+'_matched_time']=find(base=b,key=z['spec']['key'].replace('volume','time'),budget=z['spec']['budget'])
        selected[b+'_time_best']=best([z for z in rows if z['spec']['base']==b and z['spec']['family']=='cost' and z['spec']['key'].startswith('time')])
        z=best([z for z in rows if z['spec']['base']==b and z['spec']['key']=='joint'])
        selected[b+'_joint']=z;selected[b+'_matched_additive']=find(base=b,key='additive',budget=z['spec']['budget'])
    folds=[];parts=[];baseparts=[]
    for n,end in [(36,60),(60,76),(76,88)]:
        metrics=[dict(spec=z['spec'],**session.measure(rr,m,bounds,n)) for z,rr in zip(rows,returns)]
        win=best([z for z in metrics if session.admitted(z,metrics[0])])
        parts.extend(returns[win['spec']['id']][bounds[n]:bounds[end]]);baseparts.extend(returns[0][bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=win['spec']))
    dump('WALKFORWARD.json',dict(folds=folds,metrics=rp.stats_rr(np.array(parts),m['dates'][36],m['ends'][87]),baseline=rp.stats_rr(np.array(baseparts),m['dates'][36],m['ends'][87]),pristine_oos=False))
    dump('TRAIN_TRIALS.json',rows)
    files=['PLAN.json','GRID.json','DATA.npz','META.json','INPUT_HASHES.json','FEATURE_AUDIT.json','TRIO_AUDIT.json','FORECASTS.npz','FORECAST_AUDIT.json','FORECAST_INPUT_HASHES.json','study.py','cost_features.py','trio_model.py']
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,trials=len(rows),
        admitted_noncontrols=sum(z['admitted'] and z['spec']['family']!='control' for z in rows),
        hashes={f:sha(P/f) for f in files},author_labels_used=False,diagnostics_may_be_unqualified=True))
    (P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('Frozen primary',selected['primary']['spec'],flush=True)

def evaluate():
    fr=read(P/'FROZEN.json');assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
    for f,h in fr['hashes'].items():assert sha(P/f)==h,f
    m,a,px,op=load();records=read(P/'META.json')['trios'];blocks,_,bounds,valid=rp.paths(px,m,122)
    orders={};choices={}
    for name,z in fr['selected'].items():
        ch,sc=choose(z['spec'],a,m,122,records);choices[name]=ch;orders[name]=rp.make_orders(ch,sc,m)
    orders['previous66']=read(ep.P/'primary_ORDERS.json')
    orders['qqq']=[dict(date=m['dates'][0],signal_date=m['dates'][0],end=m['ends'][-1],symbols=['QQQ'],scores=[])]
    results={};memo={}
    for name,oo in orders.items():
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
    dump('VALIDATION.json',dict(original_controls_exact=True,independent_daily_replay=True,training_freeze_parity=True,
        protected_files=len(read(P/'PROTECTED.json')),protected_changes=changed,new_observation_accounts=False,
        pristine_oos=False,unique_replay_specs=len(memo)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','prepare','forecasts','train','evaluate']);globals()[ap.parse_args().phase]()

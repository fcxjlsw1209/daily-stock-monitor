"""Predeclared monthly trend segments, causal auxiliary budgets and portfolio targets."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import itertools
import json
import sys
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(P))
sp=importlib.util.spec_from_file_location('opportunity_previous',O/'monthly_trend_evidence_research/study.py')
prev=importlib.util.module_from_spec(sp);sp.loader.exec_module(prev)
ep,rp,arch,ex,read,sha=prev.ep,prev.rp,prev.arch,prev.ex,prev.read,prev.sha
TRAIN=88
BASES=['original65','simple']
KEYS=['continuity_floor','quality_improvement','downside_improvement']


def dump(name,value):
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n')


def grid():
    rows=[]
    for b in BASES:
        rows.append(dict(id=len(rows),base=b,family='control',key='none',budget=0.))
        for key,budget in itertools.product(KEYS,[.025,.075]):
            rows.append(dict(id=len(rows),base=b,family='segment',key=key,budget=budget))
        for key,budget in itertools.product(['fixed','dynamic'],[.025,.05]):
            rows.append(dict(id=len(rows),base=b,family='budget',key=key,budget=budget))
        for key,budget in itertools.product(['portfolio','stock'],[.025,.05]):
            rows.append(dict(id=len(rows),base=b,family='pair',key=key,budget=budget))
    return rows


def initialize():
    assert not (P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Three bounded independent monthly hypotheses, no observer modifications.',
        guidance='Long momentum excluding latest42sessions dominates; monthly3equal/2half/<2cash, fixedunits intramonth; no requiredcurrentpullback, stops, leverage or issuer/date exceptions.',
        universe='Same122 estimatedTop32+assumed17ETF monthly snapshots. Initial2016-08-17 through2026-09-24. Historical stock/sector proxies remain; training prior2016 uses inherited proxy. Not exact author universe.',
        segments='Compare126returns t-252..t-126 and84returns t-126..t-42. Signedpath=sumlog/sumabs. Signals min(pathold,pathnew), pathnew-pathold, downsideRMSold-downsideRMSnew. Centered crosssection ranks; missingexactzero. Add2.5/7.5% convex auxiliary correction without changing core or eligibility.',
        budgets='Existing fixed annual pooledRidge forecasts and rank3 vs ranks4/5 gap<=.02, with top2 unchanged. Compare2.5/5%fixed to dynamic upperbudget. Each historical virtual fixedbudget portfolio uses its then-causal saved forecast. At signal: last12 fullymature earlier complete monthly logreturn differences vs unmodifiedbase; excludeinitialhalfmonth; require12 with allfinite; multiplier=max(mean,0)/(abs(mean)+SE), zero/negative->zero. Fixed virtual outcomes drive the gate, never gated executed policy. Coldstart or anymissingrecent evidence no correction; no backfillingoldermonths.',
        portfolio_model='For same baseline ranks3 vs4/5gap<=.02 compare native log change in arithmeticEW3 return after replacing third vs stocklogreturndifference/3. Features20:10 challenger-incumbent existingplain features plus eachdifference timesmean firsttwo feature. Annual prior5years, all labels mature bypreviousDec31, alpha10,24monthdecay,monthbalanced,trainingRMS,nointercept,min18months30pairs. Samevalidlabels matchedtargets. Candidate/ranksfixed before labelavailability; missinglabeldropspair without replacement.',
        pair_selection='Predictedportfoliochange or predictedstockchange/3 scaled via budget*.5*tanh(prediction/(.05/3)). Onlypositiveforecast andbonus>basegap can replace third. Same2.5/5%budgets, top2locked. Availability false nochange.',
        grid='30 fixed configs:2bases*(1control+3segments*2budgets+fixed/dynamic*2budgets+portfolio/stock*2budgets). No combining or extending after evaluation.',
        bases='Original65.34 exact scores and independent75%rawskip42momentum+25%vol126. Previous66.36 exact orders additionalcontrol.',
        selection='Train first88 through2023-11-30: CAGR>=original65, Sharpe>=base-.05,DD>=base-.03,early53CAGR>=base,recent35CAGR>=base-.02. HighesttrainingCAGR admittedprimary elseoriginal65. Familytrainingbest diagnostics may be unqualified; freeze matchedfixed/dynamic andportfolio/stock at chosenbudget. Later34 evaluatedonlyafterfreeze; no reselection. Prefix36/60/76 sequential diagnostics.',
        execution='Primaryzero-costsame-final-close idealization; nextopenzero and5bps/side sensitivity. InitialAug2016 and finalSep2026 partial. Full-history repeated research, not pristine OOS. Author labels not used.',
        validation='Prefix/future tests, model training cutoffs and matched rows, baseline exactscores/orders, independentdailyreplay, protectedhashes. Preserve all existing ledgers plans and fingerprints.'))
    dump('GRID.json',grid())
    protected=read(prev.P/'PROTECTED.json')
    protected.update({str(f):sha(f) for f in prev.P.rglob('*') if f.is_file() and f.suffix!='.pyc'})
    dump('PROTECTED.json',protected)


def ranks_for(score,eligible,symbols):
    return sorted(np.flatnonzero(eligible),key=lambda j:(-float(score[j]),symbols[j]))


def pair_rows(meta,scores,eligible,plain,px=None):
    from portfolio_model import pair_features,pair_targets
    rows=[]; features=[]; targets={'portfolio':[],'stock':[]}; audit=dict(potential_pairs=0,missing_label_pairs=0)
    for i,(day,ss) in enumerate(zip(meta['dates'],meta['symbols'])):
        ranks=ranks_for(scores[i],eligible[i],ss)
        if len(ranks)<3:continue
        A,B,inc=ranks[:3]
        for j in ranks[3:5]:
            if scores[i,inc]-scores[i,j]>.02+1e-12:continue
            audit['potential_pairs']+=1
            rows.append(dict(event=i,inc=int(inc),challenger=int(j),date=day,end=meta['ends'][i]))
            features.append(pair_features(plain[i,inc],plain[i,j],plain[i,A],plain[i,B]))
            if px is not None:
                # Choices are already fixed before examining label completeness.
                series=px.reindex(columns=[ss[k] for k in [A,B,inc,j]]).loc[day:meta['ends'][i]]
                ok=(len(series)>1 and str(series.index[0].date())==day and str(series.index[-1].date())==meta['ends'][i]
                    and np.isfinite(series.to_numpy()).all() and (series.to_numpy()>0).all())
                if ok:
                    r=(series.iloc[-1]/series.iloc[0]-1).to_numpy()
                    labels=pair_targets(*r)
                    for k in targets:targets[k].append(float(labels[k]))
                else:
                    audit['missing_label_pairs']+=1
                    for k in targets:targets[k].append(float('nan'))
    return rows,np.asarray(features,float).reshape(-1,20),{k:np.asarray(v,float) for k,v in targets.items()},audit


def prepare():
    from segment_features import build_features
    from baseline_adapter import prepare as baseline_prepare
    m,old,px,op=prev.load();panel,info=arch.old.panel_load(m['etfs'])
    tm=read(arch.P/'META.json')['training']
    events=list(zip(tm['dates']+m['dates'],tm['symbols']+m['symbols']))
    exact=baseline_prepare(panel,events,m['etfs']);nt=len(tm['dates'])
    np.testing.assert_allclose(exact['score'][nt:],old['base_original65'],atol=1e-12)
    np.testing.assert_array_equal(exact['eligible'][nt:],old['eligible'])
    raw,audit=build_features(panel,list(zip(m['dates'],m['symbols'])))
    arrays={k:old[k] for k in ['eligible','full','available','train_plain','event_plain']}
    for b in BASES:
        for prefix in ['base_','core_','cw_']:arrays[prefix+b]=old[prefix+b]
    for key in KEYS:
        arrays[key]=np.stack([arch.rank_values(row) for row in raw[key]])
        arrays['valid_'+key]=np.isfinite(raw[key]);arrays[key][~arrays['valid_'+key]]=0.
    trainplain=old['train_plain'].reshape(nt,49,10);eventplain=old['event_plain'].reshape(122,49,10)
    records={};training_elig=exact['eligible'][:nt]
    for b in BASES:
        training_scores=exact['score'][:nt] if b=='original65' else .75*trainplain[:,:,0]+.25*trainplain[:,:,2]
        tr,tx,ty,ta=pair_rows(tm,training_scores,training_elig,trainplain,panel['adj'])
        er,xx,_,ea=pair_rows(m,old['base_'+b],old['eligible'],eventplain)
        for key,v in [('X',tx),('event_X',xx),('dates',np.array([z['date'] for z in tr],dtype='datetime64[D]')),
                      ('ends',np.array([z['end'] for z in tr],dtype='datetime64[D]')),
                      ('event_dates',np.array([z['date'] for z in er],dtype='datetime64[D]'))]:arrays[b+'_'+key]=v
        for k,v in ty.items():arrays[b+'_y_'+k]=v
        records[b]=dict(training=tr,events=er,training_audit=ta,event_audit=ea)
    np.savez_compressed(P/'DATA.npz',**arrays)
    dump('META.json',dict(monthly=m,training=tm,pairs=records));dump('FEATURE_AUDIT.json',audit)
    hashes=dict(info['source_hashes'])
    for f in [prev.P/'DATA.npz',prev.P/'FORECASTS.npz',prev.P/'FROZEN.json',arch.P/'META.json',Path(prev.__file__),
              Path(arch.__file__),Path(ep.__file__),Path(rp.__file__),Path(ex.e.engine.__file__),
              O/'monthly_stability_monitor/strategy.py',O/'monthly_stability_monitor/base_strategy.py']:
        hashes[str(f)]=sha(f)
    dump('INPUT_HASHES.json',hashes)
    oo=rp.make_orders(rp.choose(arrays['base_original65'],arrays['eligible'],m['symbols'],3),arrays['base_original65'],m)
    assert oo==read(ep.P/'original65_ORDERS.json')
    dump('PREPARE_VALIDATION.json',dict(original_scores_exact=True,original_orders_exact=True,eligibility_exact=True,
        pair_audits={b:records[b]['training_audit'] for b in BASES},pre2016_membership='inherited historical proxy'))
    print('Prepared segments and causal boundary-pair training data',flush=True)


def forecasts():
    from portfolio_model import annual_pair_forecasts
    a=dict(np.load(P/'DATA.npz'));meta=read(P/'META.json');out={};audits={}
    for b,key in itertools.product(BASES,['portfolio','stock']):
        r=annual_pair_forecasts(a[b+'_dates'],a[b+'_ends'],a[b+'_X'],a[b+'_y_'+key],
            a[b+'_event_dates'],a[b+'_event_X'],np.ones(len(a[b+'_event_dates']),bool))
        audits[b+'_'+key]=r.pop('audit')
        pred=np.zeros((122,49));valid=np.zeros((122,49),bool)
        for z,value,good in zip(meta['pairs'][b]['events'],r['pred'],r['available']):
            pred[z['event'],z['challenger']]=value;valid[z['event'],z['challenger']]=good
        out[b+'_'+key]=pred;out[b+'_valid_'+key]=valid
    np.savez_compressed(P/'FORECASTS.npz',**out);dump('FORECAST_AUDIT.json',audits)
    dump('FORECAST_INPUT_HASHES.json',{f:sha(P/f) for f in ['DATA.npz','META.json','baseline_adapter.py','segment_features.py','portfolio_model.py']})
    print('Annual matched portfolio/stock forecasts ready',flush=True)


def load():
    for p,h in read(P/'INPUT_HASHES.json').items():assert sha(p)==h,p
    for p,h in read(P/'FORECAST_INPUT_HASHES.json').items():assert sha(P/p)==h,p
    m,_,px,op=prev.load()
    return m,{**dict(np.load(P/'DATA.npz')),**dict(np.load(P/'FORECASTS.npz'))},px,op


def score(row,a,n):
    base=a['base_'+row['base']][:n]
    if row['family']!='segment':return base.copy()
    core,cw=a['core_'+row['base']][:n],a['cw_'+row['base']][:n]
    support=(base-core)/(1-cw);effect=np.where(a['valid_'+row['key']][:n],a[row['key']][:n],0.)
    delta=np.where(effect==0,0.,np.clip(support+effect,-.5,.5)-support)
    return base+row['budget']*delta


def choose(row,a,m,n,budgets=None):
    sc=score(row,a,n);chosen=rp.choose(sc,a['eligible'][:n],m['symbols'][:n],3)
    if row['family'] not in ['budget','pair']:return chosen,sc
    if budgets is None:budgets=np.full(n,row['budget'])
    for i in range(n):
        ranks=ranks_for(sc[i],a['eligible'][i],m['symbols'][i])
        if len(ranks)<3:continue
        inc=ranks[2];offers=[]
        for j in ranks[3:5]:
            gap=sc[i,inc]-sc[i,j]
            if gap>.02+1e-12:continue
            if row['family']=='budget':
                left=i*49+inc;right=i*49+j
                available=a['available'][left] and a['available'][right]
                delta=a['full'][right]-a['full'][left];scale=.05
            else:
                available=a[row['base']+'_valid_'+row['key']][i,j]
                delta=a[row['base']+'_'+row['key']][i,j]/(3 if row['key']=='stock' else 1);scale=.05/3
            if not available or not np.isfinite(delta) or delta<=0:continue
            bonus=budgets[i]*.5*np.tanh(delta/scale)
            if bonus>gap+1e-12:offers.append((float(sc[i,j]+bonus),m['symbols'][i][j],j))
        if offers:
            best=sorted(offers,key=lambda z:(-z[0],z[1]))[0]
            chosen[i]=ranks[:2]+[best[2]];sc[i,best[2]]=best[0]
    return chosen,sc


def decisions(row,a,m,n,px):
    if row['family']!='budget' or row['key']!='dynamic':return (*choose(row,a,m,n),None)
    from dynamic_budget import budget_series
    fixed,_=choose({**row,'key':'fixed'},a,m,n)
    base=rp.choose(a['base_'+row['base']][:n],a['eligible'][:n],m['symbols'][:n],3)
    differences=[]
    for i in range(n):
        # Stored fixed forecasts choose first. These realized labels feed ONLY future gates.
        def realized(ch):
            if not ch:return 0.
            ss=[m['symbols'][i][j] for j in ch];p=px.loc[m['dates'][i]:m['ends'][i],ss]
            if len(p)<2 or str(p.index[0].date())!=m['dates'][i] or str(p.index[-1].date())!=m['ends'][i]:return float('nan')
            if not np.isfinite(p.to_numpy()).all() or not (p.to_numpy()>0).all():return float('nan')
            return float((p.iloc[-1]/p.iloc[0]-1).mean())
        x,y=realized(fixed[i]),realized(base[i])
        differences.append(float(np.log1p(x)-np.log1p(y)))
    budgets,audit=budget_series(m['dates'][1:n],m['ends'][1:n],differences[1:],row['budget'])
    budgets=np.r_[0.,budgets]
    audit=[dict(date=m['dates'][0],budget=0.,reason='initial_partial_period_excluded')]+audit
    ch,sc=choose(row,a,m,n,budgets)
    return ch,sc,dict(months=audit,virtual_log_differences=[None if not np.isfinite(z) else z for z in differences])


def train():
    assert not (P/'FROZEN.json').exists()
    m,a,px,_=load();blocks,_,bounds,valid=rp.paths(px,m,TRAIN);rows=[];returns=[]
    for spec in grid():
        ch,_,_=decisions(spec,a,m,TRAIN,px);rr=rp.fast_returns(ch,blocks,valid)
        rows.append(dict(spec=spec,**prev.prev.measure(rr,m,bounds)));returns.append(rr)
    base=rows[0]
    for z in rows:z['admitted']=prev.prev.admitted(z,base)
    best=lambda zs:max(zs,key=lambda z:(z['training']['cagr'],-z['spec']['id']))
    selected=dict(primary=best([z for z in rows if z['admitted']]),original65=base,
        simple=next(z for z in rows if z['spec']['base']=='simple' and z['spec']['family']=='control'))
    for b,key in itertools.product(BASES,KEYS+['dynamic','portfolio']):
        chosen=best([z for z in rows if z['spec']['base']==b and z['spec']['key']==key]);selected[b+'_'+key]=chosen
        match={'dynamic':'fixed','portfolio':'stock'}.get(key)
        if match:selected[b+'_'+match]=next(z for z in rows if z['spec']['base']==b and z['spec']['key']==match and z['spec']['budget']==chosen['spec']['budget'])
    folds=[];parts=[];baseparts=[]
    for n,end in [(36,60),(60,76),(76,88)]:
        metrics=[dict(spec=z['spec'],**prev.prev.measure(rr,m,bounds,n)) for z,rr in zip(rows,returns)]
        win=best([z for z in metrics if prev.prev.admitted(z,metrics[0])])
        parts.extend(returns[win['spec']['id']][bounds[n]:bounds[end]]);baseparts.extend(returns[0][bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=win['spec']))
    dump('WALKFORWARD.json',dict(folds=folds,metrics=rp.stats_rr(np.array(parts),m['dates'][36],m['ends'][87]),baseline=rp.stats_rr(np.array(baseparts),m['dates'][36],m['ends'][87]),pristine_oos=False))
    dump('TRAIN_TRIALS.json',rows)
    files=['PLAN.json','GRID.json','DATA.npz','META.json','INPUT_HASHES.json','FORECASTS.npz','FORECAST_AUDIT.json','FORECAST_INPUT_HASHES.json','study.py','segment_features.py','portfolio_model.py','baseline_adapter.py','dynamic_budget.py']
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,trials=len(rows),
        new_admitted=sum(z['admitted'] and z['spec']['family']!='control' for z in rows),hashes={f:sha(P/f) for f in files},author_labels_used=False,diagnostics_may_be_unqualified=True))
    (P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('Training frozen:',selected['primary']['spec'],flush=True)


def evaluate():
    fr=read(P/'FROZEN.json');assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
    for f,h in fr['hashes'].items():assert sha(P/f)==h,f
    m,a,px,op=load();blocks,_,bounds,valid=rp.paths(px,m,122)
    orders={};choices={};gates={}
    for name,z in fr['selected'].items():
        ch,sc,ga=decisions(z['spec'],a,m,122,px);choices[name]=ch;orders[name]=rp.make_orders(ch,sc,m)
        if ga:gates[name]=ga
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
        for k in ['cagr','sharpe','max_drawdown']:assert abs(results[name]['metrics'][k]-read(prev.P/'RESULTS.json')[name]['metrics'][k])<1e-10
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
    dump('GATE_AUDIT.json',gates);dump('MONTH_COMPARISON.json',comparison);dump('ATTRIBUTION.json',attribution);dump('RESULTS.json',results)
    changes=[f for f,h in read(P/'PROTECTED.json').items() if sha(f)!=h];assert not changes,changes
    dump('VALIDATION.json',dict(old_controls_exact=True,independent_daily_replay=True,training_freeze_parity=True,protected_files=len(read(P/'PROTECTED.json')),protected_changes=changes,new_observation_accounts=False,pristine_oos=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','prepare','forecasts','train','evaluate'])
    globals()[ap.parse_args().phase]()

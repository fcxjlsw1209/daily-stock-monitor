"""Small, predeclared monthly trend-evidence experiments; no observer writes."""
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
P = Path(__file__).resolve().parent
O = P.parent
sys.path.insert(0, str(P))
_spec = importlib.util.spec_from_file_location('trend_evidence_prior', O/'monthly_session_type_research/study.py')
prev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prev)
ep, rp, arch, ex, read, sha = prev.ep, prev.rp, prev.arch, prev.ex, prev.read, prev.sha
TRAIN = 88
KEYS = ['volume_confirmation', 'recovery_speed']
BASES = ['original65', 'simple']


def dump(name, value):
    (P/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else x)+'\n')


def grid():
    rows = []
    for base in BASES:
        rows.append(dict(id=len(rows),base=base,family='control',key='none',budget=0.,k=0.))
        for key,budget in itertools.product(KEYS,[.025,.075]):
            rows.append(dict(id=len(rows),base=base,family='hand',key=key,budget=budget,k=0.))
        for budget,k in itertools.product([.025,.05],[0.,1.,2.]):
            rows.append(dict(id=len(rows),base=base,family='ml',key='block_stability',budget=budget,k=k))
    return rows


def initialize():
    assert not (P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Long-horizon trading-amount confirmation, time spent awaiting a new high, and year-block stability of small annual Ridge boundary corrections.',
        guidance='Long momentum remains dominant and skips latest42sessions. No required current pullback,no intramonth trading/stops,no leverage,no issuer/date exceptions. Three equal,two half,fewer than two cash.',
        inputs='Identical122 cached Top32-estimate plus assumedtheme17 pools and original past-only eligibility;2016-08-17 initialization through2026-09-24. Unknown author ETF17 and market-cap/sector proxies inherited; not exact author recovery.',
        bases='Unchanged registered65.34 score(core50/52%) and independent75%raw skip42 momentum+25%vol126. Prior66.36 exact orders additional control, not a searched base.',
        hand='Two directional signals endingt-42. volume_confirmation:210*sum(w*logreturn)/sum(w)-sum(logreturn), w=clip(rawclose*volume/prior63mean_dollarvolume,.25,4). recovery_speed:-mean(days since latest within-window running high)/210 over211prices t-252..t-42. Rank finite event-pool values centered; missing gives exact zero correction, original eligibility unchanged. Extra volume warmup disclosed.',
        support='Hand adds budget*(clip(originalsupport+factor,-.5,.5)-originalsupport), budgets2.5/7.5%. Original core unchanged, equivalent convex auxiliary replacement. Numerical zero or missing is exact nochange.',
        ml='Pooled pointwise Ridge,10 existing plain features. Annual past5years actual-next-calendar-month mature centered excessSPY labels,month balanced24month decay,alpha10. At least36months and4signal-year blocks; delete each actualyear and refit scaler/weights/model,at least18months per fold. No author labels and no outcome-based eligibility.',
        ml_selection='Start each month with base top3,lock top2; only base ranks4/5 within.02 of third may challenge. Pair delta=full_forecast(challenger)-full_forecast(third). StabilitySE=sqrt((B-1)/B*sum((fold_delta-mean_fold_delta)^2)); lower=delta-kSE. Challenger wins iff lower>0 and budget*.5*tanh(lower/.05)>base-score-gap; largest corrected score then symbol tie-break. Budget2.5/5%,k0/1/2. No prior-month incumbent retention. SE is training-year sensitivity,not a calibrated confidence interval. Unknown forecasts exactly nochange.',
        grid='22 fixed rows:2bases*(1control+2handfactors*2budgets+2MLbudgets*3penalties). No combining families or extending grid after returns.',
        training='First88 through2023-11-30,early53 and recent35 guards relative original65:trainCAGR>=base,trainSharpe>=base-.05,DD>=base-.03,earlyCAGR>=base,recentCAGR>=base-.02. Max admitted training CAGR primary,else original65. Per-base/family best trainingCAGR descriptive diagnostics frozen even if unqualified. ML penalty matched0/1/2 frozen at selected budget. Later34 replay only after freeze; no reselection.',
        execution='Zero-cost full-close signal/same-close idealization primary,next-openzero and5bps/side sensitivity. InitialAug2016 and lastSep2026 partial periods. Independently validate daily fixed-unit replay.',
        reporting='Period/year results,changedmonths and stock contributions,undo best1/3changes diagnostic. Sequential36/60/76 selections withintraining. History repeatedly used,not pristineOOS. No guaranteed80% return.',
        protection='New isolated files only. Do not change old plans,prices,ledgers,strategy fingerprints,automations or live observers.',
        references=[dict(url='https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00280',scope='Volume/momentum relationship motivates a hypothesis; not replication of paper or proof our factor works.'),dict(url='https://arxiv.org/abs/1505.01333',scope='Duration-based ranking motivation only; our mean age is not the paper estimator.')]))
    dump('GRID.json',grid())
    protected=read(prev.P/'PROTECTED.json')
    protected.update({str(f):sha(f) for f in prev.P.rglob('*') if f.is_file() and f.suffix!='.pyc'})
    dump('PROTECTED.json',protected)


def prepare():
    from trend_features import build_features
    m,old,px,op=prev.load()
    panel,info=arch.old.panel_load(m['etfs'])
    raw,audit=build_features(panel,list(zip(m['dates'],m['symbols'])))
    arrays={k:old[k] for k in ['dates','ends','event_dates','y','train_good','eligible','train_plain','event_plain']}
    for base in BASES:
        for prefix in ['base_','core_','cw_']:arrays[prefix+base]=old[prefix+base]
    for key in KEYS:
        arrays['raw_'+key]=raw[key]
        arrays[key]=np.stack([arch.rank_values(row) for row in raw[key]])
        arrays['valid_'+key]=np.isfinite(raw[key])
        arrays[key][~arrays['valid_'+key]]=0.
    np.savez_compressed(P/'DATA.npz',**arrays)
    dump('META.json',read(prev.P/'META.json'))
    hashes=dict(info['source_hashes'])
    for f in [prev.P/'DATA.npz',prev.P/'FORECASTS.npz',prev.P/'FROZEN.json',Path(prev.__file__),
              Path(arch.__file__),arch.P/'residual_model.py',
              Path(ep.__file__),Path(rp.__file__),Path(ex.e.engine.__file__)]:hashes[str(f)]=sha(f)
    dump('INPUT_HASHES.json',hashes)
    orders=rp.make_orders(rp.choose(arrays['base_original65'],arrays['eligible'],m['symbols'],3),arrays['base_original65'],m)
    assert orders==read(ep.P/'original65_ORDERS.json')
    dump('PREPARE_VALIDATION.json',dict(features=audit,original65_orders_exact=True,eligibility_unchanged=True,
        missing_eligible={k:int((arrays['eligible'] & ~arrays['valid_'+k]).sum()) for k in KEYS}))
    print('Prepared122 unchanged monthly pools',flush=True)


def forecasts():
    from uncertainty import annual_block_forecasts
    a=dict(np.load(P/'DATA.npz'))
    pred=annual_block_forecasts(a['dates'],a['ends'],a['train_plain'],a['y'],a['train_good'],
        a['event_dates'],a['event_plain'],a['eligible'].reshape(-1))
    audit=pred.pop('audit')
    np.savez_compressed(P/'FORECASTS.npz',**pred)
    dump('FORECAST_AUDIT.json',audit)
    # Full fit must be exactly the previously tested pooled Ridge where available.
    prior=np.load(prev.P/'FORECASTS.npz')['plain_pooled'].reshape(-1)
    valid=pred['available']
    np.testing.assert_allclose(pred['full'][valid],prior[valid],rtol=1e-10,atol=1e-12)
    dump('FORECAST_VALIDATION.json',dict(previous_pooled_exact=True,available_rows=int(valid.sum())))
    dump('FORECAST_INPUT_HASHES.json',{f:sha(P/f) for f in ['DATA.npz','uncertainty.py']})
    print('Annual block forecasts ready',flush=True)


def load():
    for p,h in read(P/'INPUT_HASHES.json').items():assert sha(p)==h,p
    for p,h in read(P/'FORECAST_INPUT_HASHES.json').items():assert sha(P/p)==h,p
    m,_,px,op=prev.load()
    return m,{**dict(np.load(P/'DATA.npz')),**dict(np.load(P/'FORECASTS.npz'))},px,op


def score(row,a,n):
    base=a['base_'+row['base']][:n]
    if row['family']!='hand':return base.copy()
    core,cw=a['core_'+row['base']][:n],a['cw_'+row['base']][:n]
    support=(base-core)/(1-cw)
    effect=np.where(a['valid_'+row['key']][:n],a[row['key']][:n],0.)
    delta=np.where(effect==0,0.,np.clip(support+effect,-.5,.5)-support)
    return base+row['budget']*delta


def choose(row,a,m,n):
    sc=score(row,a,n)
    chosen=rp.choose(sc,a['eligible'][:n],m['symbols'][:n],3)
    if row['family']!='ml':return chosen,sc
    from uncertainty import pair_evidence
    width=sc.shape[1]
    for i in range(n):
        if len(chosen[i])!=3:continue
        indices=np.flatnonzero(a['eligible'][i])
        ranks=sorted(indices,key=lambda j:(-float(sc[i,j]),m['symbols'][i][j]))
        incumbent=ranks[2]
        offers=[]
        for j in ranks[3:5]:
            gap=float(sc[i,incumbent]-sc[i,j])
            if gap>.02+1e-12:continue
            evidence=pair_evidence(a['full'],a['folds'],a['available'],a['fold_available'],
                i*width+incumbent,i*width+j,k=row['k'])
            if not evidence['available'] or evidence['lower']<=0:continue
            bonus=row['budget']*.5*np.tanh(evidence['lower']/.05)
            if bonus>gap+1e-12:offers.append((float(sc[i,j]+bonus),m['symbols'][i][j],j))
        if offers:
            offer=sorted(offers,key=lambda z:(-z[0],z[1]))[0]
            chosen[i]=ranks[:2]+[offer[2]]
            sc[i,offer[2]]=offer[0]
    return chosen,sc


def train():
    assert not (P/'FROZEN.json').exists()
    m,a,px,_=load();blocks,_,bounds,valid=rp.paths(px,m,TRAIN)
    rows,returns=[],[]
    for spec in grid():
        ch,_=choose(spec,a,m,TRAIN);rr=rp.fast_returns(ch,blocks,valid)
        rows.append(dict(spec=spec,**prev.measure(rr,m,bounds)));returns.append(rr)
    base=rows[0]
    for r in rows:r['admitted']=prev.admitted(r,base)
    best=lambda zs:max(zs,key=lambda z:(z['training']['cagr'],-z['spec']['id']))
    selected=dict(primary=best([z for z in rows if z['admitted']]),original65=base,
        simple=next(z for z in rows if z['spec']['base']=='simple' and z['spec']['family']=='control'))
    for b,key in itertools.product(BASES,KEYS+['block_stability']):
        selected[b+'_'+key]=best([z for z in rows if z['spec']['base']==b and z['spec']['key']==key])
    for b in BASES:
        budget=selected[b+'_block_stability']['spec']['budget']
        for k in [0.,1.,2.]:
            selected[b+'_matched_k'+str(int(k))]=next(z for z in rows if z['spec']['base']==b
                and z['spec']['family']=='ml' and z['spec']['budget']==budget and z['spec']['k']==k)
    folds,parts,baseparts=[],[],[]
    for n,end in [(36,60),(60,76),(76,88)]:
        metrics=[dict(spec=z['spec'],**prev.measure(rr,m,bounds,n)) for z,rr in zip(rows,returns)]
        win=best([z for z in metrics if prev.admitted(z,metrics[0])])
        parts.extend(returns[win['spec']['id']][bounds[n]:bounds[end]])
        baseparts.extend(returns[0][bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=win['spec']))
    dump('WALKFORWARD.json',dict(folds=folds,metrics=rp.stats_rr(np.array(parts),m['dates'][36],m['ends'][87]),
        baseline=rp.stats_rr(np.array(baseparts),m['dates'][36],m['ends'][87]),pristine_oos=False))
    dump('TRAIN_TRIALS.json',rows)
    files=['PLAN.json','GRID.json','DATA.npz','META.json','INPUT_HASHES.json','FORECASTS.npz',
        'FORECAST_AUDIT.json','FORECAST_INPUT_HASHES.json','study.py','trend_features.py','uncertainty.py']
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,trials=len(rows),
        new_admitted=sum(z['admitted'] and z['spec']['family']!='control' for z in rows),
        hashes={f:sha(P/f) for f in files},author_labels_used=False,diagnostics_may_be_unqualified=True))
    (P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('Frozen primary:',selected['primary']['spec'],flush=True)


def evaluate():
    fr=read(P/'FROZEN.json');assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
    for f,h in fr['hashes'].items():assert sha(P/f)==h,f
    m,a,px,op=load();blocks,_,bounds,valid=rp.paths(px,m,122)
    orders={};decisions={}
    for name,z in fr['selected'].items():
        ch,sc=choose(z['spec'],a,m,122);decisions[name]=ch;orders[name]=rp.make_orders(ch,sc,m)
    orders['previous66']=read(ep.P/'primary_ORDERS.json')
    orders['qqq']=[dict(date=m['dates'][0],signal_date=m['dates'][0],end=m['ends'][-1],symbols=['QQQ'],scores=[])]
    results={}
    for name,oo in orders.items():
        met,eq,tr,ho=ex.e.engine.replay(px,op,oo,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
        for z in met['annual_returns']:
            if z['year'] in [2016,2026]:z['partial']=True
        slices=arch.old.prior.old.previous.cuts(eq,m)
        if name in decisions:
            rr=rp.fast_returns(decisions[name],blocks,valid)
            np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
            assert abs(slices['training2016_2023']['cagr']-fr['selected'][name]['training']['cagr'])<1e-10
        results[name]=dict(metrics=met,slices=slices,selection=fr['selected'].get(name))
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
        r=dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
        for name,oo in orders.items():
            if name=='qqq':continue
            ss=oo[i]['symbols'];ret=px.loc[m['ends'][i],ss]/px.loc[day,ss]-1 if ss else pd.Series(dtype=float)
            r[name]=dict(symbols=ss,monthly_return=float(ret.mean()) if ss else 0.,stock_contributions={s:float(v/len(ss)) for s,v in ret.items()})
        comparison.append(r)
    attribution={}
    for name,z in fr['selected'].items():
        base=z['spec']['base'];changes=[]
        for r in comparison:
            x,b=r[name],r[base]
            if set(x['symbols'])!=set(b['symbols']):
                changes.append(dict(date=r['date'],end=r['end'],split=r['split'],old=b['symbols'],new=x['symbols'],
                    difference=x['monthly_return']-b['monthly_return'],
                    log_contribution=float(np.log1p(x['monthly_return'])-np.log1p(b['monthly_return']))))
        positive=sorted([z['log_contribution'] for z in changes if z['log_contribution']>0],reverse=True)
        years=results[name]['metrics']['years'];wealth=1+results[name]['metrics']['total_return']
        attribution[name]=dict(base=base,changed_months=len(changes),changes=changes,
            undo_best1_diagnostic_cagr=float((wealth*np.exp(-sum(positive[:1])))**(1/years)-1),
            undo_best3_diagnostic_cagr=float((wealth*np.exp(-sum(positive[:3])))**(1/years)-1))
    dump('MONTH_COMPARISON.json',comparison);dump('ATTRIBUTION.json',attribution)
    changes=[f for f,h in read(P/'PROTECTED.json').items() if sha(f)!=h];assert not changes,changes
    dump('RESULTS.json',results)
    dump('VALIDATION.json',dict(old_controls_exact=True,independent_daily_replay=True,training_freeze_parity=True,
        protected_files=len(read(P/'PROTECTED.json')),protected_changes=changes,new_observation_accounts=False,pristine_oos=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','prepare','forecasts','train','evaluate'])
    globals()[ap.parse_args().phase]()

"""Company-history conditional analogues, restricted to month-end rank boundary."""
from pathlib import Path
from datetime import datetime, timezone
import sys,json,itertools,argparse
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_boundary_refinement'));import boundary_research as boundary
blend,old,prior,rp,ex,read,sha=boundary.blend,boundary.old,boundary.prior,boundary.rp,boundary.ex,boundary.read,boundary.sha

def dump(name,z):
    (P/name).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def model_configs():
    return [dict(id=0,kind='global_ridge')]+[dict(id=i+1,kind='company_mean',prior_mass=k) for i,k in enumerate([6.,12.])]+[dict(id=i+3,kind='company_context',bandwidth=b,prior_mass=k) for i,(b,k) in enumerate(itertools.product([.25,.5],[6.,12.]))]+[dict(id=7,kind='company_shape',bandwidth=.5,prior_mass=6.)]+[dict(id=8+i,kind=kind,bandwidth=.5,prior_mass=k) for i,(kind,k) in enumerate(itertools.product(['company_signal','company_signal_context'],[6.,12.]))]

def grid():
    rows=[dict(id=0,family='original65'),dict(id=1,family='previous66')]
    for model,ceiling,gap,budget in itertools.product(model_configs(),[4,5],[.01,.02],[.025,.05]):
        rows.append(dict(id=len(rows),family=model['kind'],model_id=model['id'],ceiling=ceiling,gap=gap,budget=budget))
    return rows

def initialize():
    assert not (P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
      purpose='Determine whether same-company historical outcomes conditional on market and security state improve monthly rank3 boundary choices.',
      fixed='Same archived Top32 proxy+theme17, long momentum skip42 and positive-momentum295-session gate, threeequal/twohalf/<2cash, fixed intramonthunits. Previous66.10score and registered65.34control unchanged.',
      inputs='Existing180 calendar-month training snapshots2011..2025, successor-month mature excessSPY return label clipped[-1,1]. Only prior5years and labelend<=annualcutoff. Historical membership/sector proxies inherited. No author labels. Same security IDs used without ad hoc issuer aliases.',
      forecast='Annual globalRidge alpha100,84 causal features, prior5years,12monthrecency, reused existing frozen fits. Own-history analogue empirical mean shrunk to globalcurrentforecast: (sum(weight*label)+prior_mass*global)/(sum(weight)+prior_mass). Prior mass6/12 effective-month units. Past same-company returns only while in archived training pool and historically eligible. All own observations use24monthrecency; no fixed ticker bonuses.',
      similarity='Own shape distance: mean squared rank differences of mom252,risk294,path,vol126,sector126,sector252,cmf63,pressure_easing. Gaussian bandwidth.25/.5. Context additionally multiplies Gaussian kernel of known breadth,SPY200trend,highvol distance (state scaled[-1,1],bandwidth1). Shape-only control and company-mean control isolate market-context contribution. Recover shared market-state scalars from archived feature interactions with exact reconstruction assertions.',
      ranking='Keep originalfirst2. Only originalranks3..ceiling4/5 with gap from originalthird<=.01/.02 compete for finalslot. Score correction2.5/5% times (forecast cross-sectional rank - originalsupport). Total residual+oldML+new correction capped10%, returning to originalsupport. No intramonth exit or ranking turnover filters.',
      trials='2unchangedcontrols +8forecastmodels x2ceilings x2scoregaps x2budgets =66. Fully fixed before training.',
      selection='First88 through2023-11-30 only: guards trainSharpe>=previous-.05,DD>=previous-.03,early53CAGR>=previous,recent35CAGR>=previous-.02. PrimarymaxtrainCAGR; robustsamepriorcriterion. Freeze onebestadmissibleperfamily orfallbackprevious before later34. Sequentialprefix36/60/76 diagnostics. Reused history, not pristineOOS. No full-period rescuing of rejected configs.',
      evaluation='Same-final-close zero-cost idealization2016-08-17..2026-09-24, independent daily replay, nextopenzero and5bp/side sensitivities, change-month attribution and remove-largest-benefit-month diagnostic only.',
      events='Not tested: no complete verified historical earnings/FOMC/news feature table with original availability times. SECfilingtime is not automatically earnings releasetime; future realized event surprise forbidden. Fed calendar/EDGAR data are possible sources requiring dedicated preparation.',
      protection='No changes to old sources, observer rules, plans, ledgers, fingerprints, automations. No promotion. All new artifacts isolated here.'))
    dump('MODEL_CONFIGS.json',model_configs());dump('GRID.json',grid())
    h=read(boundary.P/'PROTECTED.json');h.update({str(f):sha(f) for f in boundary.P.rglob('*') if f.is_file() and f.suffix!='.pyc'});dump('PROTECTED.json',h)
    inputs=[Path(boundary.__file__),Path(blend.__file__),Path(old.__file__),Path(rp.__file__),Path(ex.e.engine.__file__),blend.P/'MONTHLY_TRAIN.npz',blend.P/'MONTHLY_TRAIN.json',blend.P/'PREDICTIONS.npz',blend.P/'FORECAST_AUDIT.json',old.P/'FROZEN.json',old.P/'RESULTS.json']
    dump('INPUT_HASHES.json',{str(f):sha(f) for f in inputs})
    dump('SOURCES.json',dict(checked='2026-10-11',sources=[dict(url='https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html',use='Time order and mature outcome controls; do not randomly mix future observations.'),dict(url='https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data',use='EDGAR company submissions JSON; not an already prepared earnings release feed.'),dict(url='https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',use='Official decisions and minutes have different publication times; event features not implemented here.')]))

def load():
    for p,h in read(P/'INPUT_HASHES.json').items():assert sha(p)==h,p
    if (P/'SIGNAL_SOURCES.json').exists():
        for p,h in read(P/'SIGNAL_SOURCES.json').items():assert sha(p)==h,p
    return boundary.load()

def context_features(X,names,offsets):
    n=len(names);raw=X[:,:n];states=[]
    for j in range(1,4):
        interact=X[:,j*n:(j+1)*n];state=np.empty(len(X))
        for lo,hi in zip(offsets[:-1],offsets[1:]):
            # All securities at a timestamp share these state scalars. Some
            # ineligible rows have every ranked feature zero; recover their
            # state from another row at the SAME date, never another date.
            row,col=np.unravel_index(np.argmax(np.abs(raw[lo:hi])),raw[lo:hi].shape)
            assert abs(raw[lo+row,col])>1e-10
            scalar=interact[lo+row,col]/raw[lo+row,col]
            np.testing.assert_allclose(raw[lo:hi]*scalar,interact[lo:hi],atol=1e-12)
            state[lo:hi]=scalar
        states.append(state)
    shape=['mom252','risk294','path','vol126','sector126','sector252','cmf63','pressure_easing']
    return raw[:,[names.index(k) for k in shape]],np.column_stack(states)


def signal_frames(panel):
    close,high,low,adj=panel['close'],panel['high'],panel['low'],panel['adj']
    span=(high-low).replace(0,np.nan)
    tr=pd.DataFrame(np.maximum.reduce([span.to_numpy(),(high-close.shift()).abs().to_numpy(),(low-close.shift()).abs().to_numpy()]),index=close.index,columns=close.columns)
    atr=tr.rolling(21).mean()/close
    weak=adj.pct_change(5,fill_method=None)<(-.5*atr)
    lowclose=(close-low)/span<=.25
    known=atr.notna()&adj.shift(5).notna()&span.notna()
    return (weak.astype(int)+2*lowclose.astype(int)).where(known,-1)


def prepare_signals(m,meta,data):
    panel,info=old.panel_load(m['etfs']);signals=signal_frames(panel)
    train=np.concatenate([signals.loc[date,ss].to_numpy(dtype=int) for date,ss in zip(meta['dates'],meta['symbols'])])
    event=np.stack([signals.loc[date,ss].to_numpy(dtype=int) for date,ss in zip(m['dates'],m['symbols'])])
    assert len(train)==len(data['X']);np.savez_compressed(P/'SIGNALS.npz',train=train,event=event)
    dump('SIGNAL_SOURCES.json',info['source_hashes']);return train,event


def annual_rows(data,meta,cutoff):
    ids=old.training_indices(meta,cutoff);rows=[];symbols=[];dates=[];ends=[]
    for i in ids:
        lo,hi=data['offsets'][i:i+2]
        good=np.flatnonzero(data['good'][lo:hi])+lo
        rows.extend(good);symbols.extend(meta['symbols'][i][j-lo] for j in good);dates.extend([meta['dates'][i]]*len(good));ends.extend([meta['ends'][i]]*len(good))
    assert rows and max(ends)<=cutoff
    return np.array(rows,int),np.array(symbols),np.array(dates),ends

def analogue(y,dates,shape,state,query_shape,query_state,query_date,global_prior,cfg):
    if len(y)==0:return float(global_prior),dict(observations=0,weight_sum=0.,effective_samples=0.,own_weight=0.)
    age=(pd.Timestamp(query_date)-pd.to_datetime(dates)).days.to_numpy()/30.4375
    assert (age>=0).all();w=2**(-age/24.)
    if cfg['kind'] in ['company_shape','company_context','company_signal_context']:
        ds=np.mean((shape-query_shape)**2,axis=1);w*=np.exp(-.5*ds/cfg['bandwidth']**2)
    if cfg['kind'] in ['company_context','company_signal_context']:
        dm=np.mean((state-query_state)**2,axis=1);w*=np.exp(-.5*dm)
    total=float(w.sum());prediction=(float(w@y)+cfg['prior_mass']*global_prior)/(total+cfg['prior_mass'])
    neff=total**2/float(w@w) if total else 0.
    return float(prediction),dict(observations=len(y),weight_sum=total,effective_samples=neff,own_weight=total/(total+cfg['prior_mass']))

def forecasts():
    m,d,px,op,a,pred=load();data=dict(np.load(blend.P/'MONTHLY_TRAIN.npz'));meta=read(blend.P/'MONTHLY_TRAIN.json');fits=read(blend.P/'FORECAST_AUDIT.json')
    shape,state=context_features(data['X'],meta['feature_names'],data['offsets']);eventflat=data['event'].reshape(-1,data['X'].shape[1]);eventoffsets=np.arange(0,len(eventflat)+1,data['event'].shape[1]);eshape,estate=context_features(eventflat,meta['feature_names'],eventoffsets);eshape=eshape.reshape(122,-1,shape.shape[-1]);estate=estate.reshape(122,-1,3)
    tsig,esig=prepare_signals(m,meta,data)
    values={str(c['id']):np.zeros_like(d['core']) for c in model_configs()};audit=[];details=[]
    for year in range(2016,2027):
        ids=[i for i,end in enumerate(m['ends']) if pd.Timestamp(end).year==year];entry=next(z for z in fits if z['year']==year and z['key']=='M_return_100');cutoff=entry['cutoff'];rows,symbols,dates,ends=annual_rows(data,meta,cutoff)
        assert max(ends)<=cutoff<=min(m['dates'][i] for i in ids)
        # Direct parity with prior global forecast proves identical annual state.
        direct=np.einsum('ijk,k->ij',data['event'][ids],np.array(entry['coef']),optimize=False)+entry['intercept'];np.testing.assert_allclose(direct,pred['M_return_100_raw'][ids],atol=1e-12)
        for i in ids:
            for j,symbol in enumerate(m['symbols'][i]):
                global_prior=float(pred['M_return_100_raw'][i,j]);values['0'][i,j]=global_prior;mask=symbols==symbol;rr=rows[mask]
                for cfg in model_configs()[1:]:
                    selected=np.ones(len(rr),bool)
                    if cfg['kind'] in ['company_signal','company_signal_context']:selected=(tsig[rr]==esig[i,j]) if esig[i,j]>0 else np.zeros(len(rr),bool)
                    used=rr[selected];useddates=dates[mask][selected];usedends=np.array(ends)[mask][selected]
                    p,info=analogue(data['Y'][used],useddates,shape[used],state[used],eshape[i,j],estate[i,j],m['dates'][i],global_prior,cfg);values[str(cfg['id'])][i,j]=p
                    details.append(dict(date=m['dates'][i],symbol=symbol,model_id=cfg['id'],signal_type=int(esig[i,j]),cutoff=cutoff,latest_label_end=max(usedends) if len(used) else None,global_prediction=global_prior,prediction=p,**info))
        audit.append(dict(year=year,cutoff=cutoff,latest_label_end=max(ends),earliest_date=min(dates),security_examples=len(rows),unique_securities=len(set(symbols)),months=len(set(dates)),global_model_key=entry['key']))
        print('Annual company-context forecasts',year,'rows',len(rows),flush=True)
    ranks={k:blend.eligible_rank(v,d['eligible']) for k,v in values.items()}
    np.savez_compressed(P/'FORECASTS.npz',**{k+'_raw':v for k,v in values.items()},**{k+'_rank':v for k,v in ranks.items()});dump('FORECAST_AUDIT.json',audit);dump('COMPANY_CONTEXT_DETAILS.json',details)

def boundary_choices(row,a,preds,m,d,n):
    if row['family']=='original65':sc=a['baseline'][:n].copy();return rp.choose(sc,d['eligible'][:n],m['symbols'][:n],3),sc
    base=old.score(read(old.P/'FROZEN.json')['selected']['primary']['spec'],a,n)
    if row['family']=='previous66':return rp.choose(base,d['eligible'][:n],m['symbols'][:n],3),base
    return select_boundary(base,d['eligible'][:n],m['symbols'][:n],preds[str(row['model_id'])+'_rank'][:n],a,n,row)

def select_boundary(base,eligible,symbols,forecast,a,n,row):
    score=base.copy();ans=[]
    support=(a['baseline'][:n]-a['core'][:n])/(1-a['core_weight'][:n])
    oldbudget=.025*a['state_calm10'][:n]+.05*a['state_weak_narrow'][:n];budget=np.minimum(row['budget'],np.maximum(0.,.10-oldbudget))
    for i,ss in enumerate(symbols):
        rank=sorted(np.flatnonzero(eligible[i]),key=lambda j:(-base[i,j],ss[j]))
        if len(rank)<3:ans.append(rank if len(rank)==2 else []);continue
        third=rank[2];candidates=[j for j in rank[2:row['ceiling']] if base[i,third]-base[i,j]<=row['gap']]
        for j in candidates:score[i,j]+=budget[i]*(forecast[i,j]-support[i,j])
        winner=min(candidates,key=lambda j:(-score[i,j],ss[j]));picked=rank[:2]+[winner]
        ans.append(sorted(picked,key=lambda j:(-score[i,j],ss[j])))
    return ans,score

def measure(row,m,d,a,pred,blocks,bounds,valid):
    ch,_=boundary_choices(row,a,pred,m,d,88);rr=rp.fast_returns(ch,blocks,valid)
    full=rp.stats_rr(rr,m['dates'][0],m['ends'][87]);early=rp.stats_rr(rr[:bounds[53]],m['dates'][0],m['ends'][52]);recent=rp.stats_rr(rr[bounds[53]:],m['dates'][53],m['ends'][87])
    robust=min(early['sharpe'],recent['sharpe'])+.3*full['sharpe']+.4*np.log1p(full['cagr'])-.3*abs(full['max_drawdown'])
    return dict(row,training=full,early=early,recent=recent,robust=float(robust)),rr

def train():
    assert not (P/'FROZEN.json').exists();m,d,px,op,a,_=load();pred=dict(np.load(P/'FORECASTS.npz'));blocks,_,bounds,valid=rp.paths(px,m,88);rows=[];returns=[]
    for row in read(P/'GRID.json'):
        z,rr=measure(row,m,d,a,pred,blocks,bounds,valid);rows.append(z);returns.append(rr)
    good=[z for z in rows if prior.allowed(z,rows[1])];pick=lambda g:max(g,key=lambda z:(z['training']['cagr'],-z['id']));controls={};counts={}
    for family in ['global_ridge','company_mean','company_context','company_shape','company_signal','company_signal_context']:
        g=[z for z in good if z['family']==family];controls[family]=pick(g) if g else rows[1];counts[family]=len(g)
    selected=dict(primary=pick(good),robust=max(good,key=lambda z:(z['robust'],-z['id'])))
    wf=[];stitched=[];baseparts=[]
    for n,end in [(36,60),(60,76),(76,88)]:
        metrics=[rp.stats_rr(rr[:bounds[n]],m['dates'][0],m['ends'][n-1]) for rr in returns];b=metrics[1];ids=[i for i,z in enumerate(metrics) if z['sharpe']>=b['sharpe']-.05 and z['max_drawdown']>=b['max_drawdown']-.03];win=max(ids,key=lambda i:(metrics[i]['cagr'],-i))
        rr=returns[win][bounds[n]:bounds[end]];br=returns[1][bounds[n]:bounds[end]];stitched.extend(rr);baseparts.extend(br);wf.append(dict(train_end=m['ends'][n-1],test_start=m['dates'][n],test_end=m['ends'][end-1],chosen_id=win,metrics=rp.stats_rr(rr,m['dates'][n],m['ends'][end-1]),base=rp.stats_rr(br,m['dates'][n],m['ends'][end-1])))
    dump('WALKFORWARD.json',dict(folds=wf,metrics=rp.stats_rr(np.array(stitched),m['dates'][36],m['ends'][87]),base=rp.stats_rr(np.array(baseparts),m['dates'][36],m['ends'][87]),pristine_oos=False))
    dump('TRAIN_TRIALS.json',rows);np.savez_compressed(P/'TRAIN_RETURNS.npz',returns=np.stack(returns),bounds=np.array(bounds))
    fr=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=selected,family_controls=controls,family_admissible=counts,base=rows[1],original=rows[0],trials=len(rows),admissible=len(good),code_sha256=sha(__file__),data_hashes={n:sha(P/n) for n in ['PLAN.json','GRID.json','INPUT_HASHES.json','MODEL_CONFIGS.json','FORECASTS.npz','FORECAST_AUDIT.json','SIGNALS.npz','SIGNAL_SOURCES.json']})
    dump('FROZEN.json',fr);(P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n');print('Admissible',counts,'Frozen',[(k,v['id'],v['training']['cagr']) for k,v in selected.items()],flush=True)

def check_freeze():
    fr=read(P/'FROZEN.json');assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip();assert sha(__file__)==fr['code_sha256']
    for n,h in fr['data_hashes'].items():assert sha(P/n)==h,n
    return fr

def evaluate():
    fr=check_freeze();m,d,px,op,a,_=load();pred=dict(np.load(P/'FORECASTS.npz'));out={};allorders={};blocks,_,bounds,valid=rp.paths(px,m,122)
    for name,row in dict(original65=fr['original'],previous66=fr['base'],**fr['family_controls'],**fr['selected']).items():
        ch,sc=boundary_choices(row,a,pred,m,d,122);orders=rp.make_orders(ch,sc,m);met,eq,tr,hold=ex.e.engine.replay(px,op,orders,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]));rr=rp.fast_returns(ch,blocks,valid)
        np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5);out[name]=dict(metrics=met,slices=prior.old.previous.cuts(eq,m),configuration=row)
        assert abs(out[name]['slices']['training2016_2023']['cagr']-row['training']['cagr'])<1e-10
        if name in ['original65','previous66']:
            ref=read(old.P/'RESULTS.json')['baseline' if name=='original65' else 'primary']['metrics']
            for k in ['cagr','sharpe','max_drawdown']:assert abs(met[k]-ref[k])<1e-10
        for label,cost in [('next_open_zero',0.),('next_open_cost',.0005)]:
            mm,ee,_,_=ex.e.engine.replay(px,op,orders,cost=cost,execution='open',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]));out[name][label]=dict(metrics=mm,slices=prior.old.previous.cuts(ee,m))
        for suffix,z in [('ORDERS',orders),('TRADES',tr),('HOLDINGS',hold),('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])]:dump(name+'_'+suffix+'.json',z)
        allorders[name]=orders;print(name,'id',row['id'],'CAGR',met['cagr'],'Sharpe',met['sharpe'],'early',out[name]['slices']['early2016_2020']['cagr'],'later',out[name]['slices']['later34']['cagr'],flush=True)
    q=px.loc[m['dates'][0]:m['ends'][-1],'QQQ'];eq=q/q.iloc[0]*100000;out['qqq']=dict(metrics=ex.e.engine.stats(eq),slices=prior.old.previous.cuts(eq,m));dump('RESULTS.json',out)
    comparisons=[]
    for i,day in enumerate(m['dates']):
        r=dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
        for name,orders in allorders.items():
            ss=orders[i]['symbols'];r[name]=dict(symbols=ss,monthly_return=float((px.loc[m['ends'][i],ss]/px.loc[day,ss]-1).mean()) if ss else 0.)
        comparisons.append(r)
    dump('MONTH_COMPARISON.json',comparisons);bad=[p for p,h in read(P/'PROTECTED.json').items() if sha(p)!=h];assert not bad,bad
    dump('VALIDATION.json',dict(independent_replay_parity=True,controls122_parity=True,frozen_training_parity=True,protected_files=len(read(P/'PROTECTED.json')),protected_changes=bad,pristine_oos=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','forecasts','train','evaluate']);globals()[ap.parse_args().phase]()

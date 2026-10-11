"""Research-only month-end retention and model-disagreement experiments."""
from pathlib import Path
from datetime import datetime, timezone
import sys, json, itertools, argparse
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_blend_optimization'))
import blend_research as blend
old,prior,rp,ex,read,sha=blend.old,blend.prior,blend.rp,blend.ex,blend.read,blend.sha


def dump(name,obj):
    (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')


def grid():
    rows=[]
    def add(family,selector=None,shrink=None,base='previous66'):
        rows.append(dict(id=len(rows),family=family,base=base,selector=selector or dict(kind='top3'),shrink=shrink))
    add('original65',base='original65');add('previous66')
    selectors=[dict(kind='buffer',ceiling=k) for k in [4,5]]
    selectors += [dict(kind='gap',ceiling=k,gap=g) for k,g in itertools.product([4,5],[.005,.01,.02])]
    shrinks=[dict(comparator=c,threshold=t,floor=f) for c,t,f in itertools.product(['alpha','frequency','both'],[.15,.30],[0.,.5])]
    for v in selectors:add('retention' if v['kind']=='buffer' else 'switch_gap',selector=v)
    for v in shrinks:add('disagreement',shrink=v)
    for v,w in itertools.product(selectors,shrinks):add('combined',selector=v,shrink=w)
    return rows


def initialize():
    assert not (P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Test small month-end rank-boundary changes using existing forecasts; preserve all registered observers.',
        universe='Exact archived122 Top32-proxy+assumedtheme17 pools. Same long-momentum skip42,positive momentum and295-session warmup eligibility. No author-specific pool reconstruction in this experiment.',
        base='Previous research66.10% score: frozen65.34% baseline +2.5% residual in calm10 +5% weekly utility Ridge in weak_narrow. Original65.34% control.',
        selector='Top4/top5 retention. Alternatively, keep incumbents only within top4/top5 and replace weakest retained name only when strongest challenger exceeds it by .005/.01/.02 total-score units. If below2 eligible, cash; otherwise2half or3equal. Noneligible or out-of-pool incumbent must exit. At each month-end all selected names reset equal; fixed shares inside month.',
        shrink='Compare existing weekly utility alpha10 rank to alpha100 and/or actual-month utility alpha10. If absolute rank disagreement>.15/.30, multiply existing5% ML budget by0/.5 for that security; return unused budget to original support. Both comparator takes max disagreement. A disagreement proxy, not calibrated confidence. No refitting.',
        grid='2controls +2buffer +6gap +12shrink +96selector/shrink combinations =118 configs; fully enumerated before training.',
        split='First88 periods through2023-11-30 choose/freeze, then later34 replay. History repeatedly researched, including later34: not untouchedOOS. No date/ticker special cases; no post-later34 retuning.',
        selection='Guard against previous66:trainSharpe>=base-.05,maxDD>=base-.03,early53CAGR>=base,recent35CAGR>=base-.02. Primary maxtrainCAGR; secondary priorrobustscore. Reportbestadmissibleperfamily, otherwiseprevious66 fallback. Sequentialprefix diagnostic36/60/76, then24/16/12 periods. Original andprevious controls always retained.',
        checks='Independent fixed-unit replay, original122-order parity, prefix causality, forceexit, score-gap tie behavior, prediction disagreement budget, protected-file hashes. Diagnose changed months and replace largest beneficial changed month with baseline only for attribution, never reselection.',
        execution='Primaryzero-costsame-final-close idealized replay2016-08-17..2026-09-24. Next-openzero and5bp/side sensitivity. No real trading.',
        caveats='Estimated historical cap/member pool, assumed ETF list, earlier training-membership proxies, same-close execution idealization, and repeated historical search persist. This does not reproduce the author or establish future80/100% returns.'))
    dump('GRID.json',grid())
    protected=read(blend.P/'PROTECTED.json')
    protected.update({str(f):sha(f) for f in blend.P.rglob('*') if f.is_file() and f.suffix!='.pyc'})
    dump('PROTECTED.json',protected)
    files=[Path(blend.__file__),Path(old.__file__),Path(rp.__file__),Path(ex.e.engine.__file__),blend.P/'PREDICTIONS.npz',old.P/'FROZEN.json',old.P/'RESULTS.json']
    dump('INPUT_HASHES.json',{str(f):sha(f) for f in files})


def load():
    for p,h in read(P/'INPUT_HASHES.json').items():assert sha(p)==h,p
    m,d,L,x,px,op,f,a=old.load()
    pred=dict(np.load(blend.P/'PREDICTIONS.npz'))
    np.testing.assert_allclose(pred['W_utility_10_rank'],a['ml_4'],atol=1e-12)
    return m,d,px,op,a,pred


def disagreement(pred,rule,n):
    base=pred['W_utility_10_rank'][:n]
    da=np.abs(base-pred['W_utility_100_rank'][:n]);df=np.abs(base-pred['M_utility_10_rank'][:n])
    return {'alpha':da,'frequency':df,'both':np.maximum(da,df)}[rule['comparator']]


def score(row,a,pred,n):
    if row['base']=='original65':return a['baseline'][:n].copy()
    spec=read(old.P/'FROZEN.json')['selected']['primary']['spec']
    sc=old.score(spec,a,n)
    if row['shrink'] is not None:
        rule=row['shrink'];scale=np.where(disagreement(pred,rule,n)>rule['threshold'],rule['floor'],1.)
        support=(a['baseline'][:n]-a['core'][:n])/(1-a['core_weight'][:n])
        mlbudget=.05*a['state_weak_narrow'][:n,None]
        sc+=(scale-1.)*mlbudget*(a['ml_4'][:n]-support)
    assert np.isfinite(sc).all()
    return sc


def choose(sc,eligible,symbols,selector):
    if selector['kind']=='top3':return rp.choose(sc,eligible,symbols,3)
    if selector['kind']=='buffer':return rp.choose(sc,eligible,symbols,selector['ceiling'])
    assert selector['kind']=='gap'
    previous=[];ans=[]
    for i,ss in enumerate(symbols):
        ordered=sorted(np.flatnonzero(eligible[i]),key=lambda j:(-sc[i,j],ss[j]))
        ranks={ss[j]:k+1 for k,j in enumerate(ordered)};lookup={t:j for j,t in enumerate(ss)}
        selected=sorted([t for t in previous if ranks.get(t,10**9)<=selector['ceiling']],key=lambda t:ranks[t])[:3]
        for j in ordered:
            if len(selected)>=3:break
            if ss[j] not in selected:selected.append(ss[j])
        # Every swap strictly improves score; previously excluded names cannot cycle.
        while len(selected)==3:
            outside=[j for j in ordered if ss[j] not in selected]
            if not outside:break
            challenger=outside[0];weakest=max(selected,key=lambda t:ranks[t]);inc=lookup[weakest]
            margin=float(sc[i,challenger]-sc[i,inc])
            wins_tie=selector['gap']==0 and margin==0 and ranks[ss[challenger]]<ranks[weakest]
            if margin>selector['gap'] or wins_tie:selected[selected.index(weakest)]=ss[challenger]
            else:break
        if len(selected)<2:selected=[]
        selected=sorted(selected,key=lambda t:ranks[t]);ans.append([lookup[t] for t in selected]);previous=selected
    return ans


def choices(row,a,pred,m,d,n):
    sc=score(row,a,pred,n)
    return choose(sc,d['eligible'][:n],m['symbols'][:n],row['selector']),sc


def measure(row,m,d,a,pred,blocks,bounds,valid):
    ch,_=choices(row,a,pred,m,d,88);rr=rp.fast_returns(ch,blocks,valid)
    train=rp.stats_rr(rr,m['dates'][0],m['ends'][87]);early=rp.stats_rr(rr[:bounds[53]],m['dates'][0],m['ends'][52]);recent=rp.stats_rr(rr[bounds[53]:],m['dates'][53],m['ends'][87])
    robust=min(early['sharpe'],recent['sharpe'])+.3*train['sharpe']+.4*np.log1p(train['cagr'])-.3*abs(train['max_drawdown'])
    return dict(row,training=train,early=early,recent=recent,robust=float(robust)),rr


def train():
    assert not (P/'FROZEN.json').exists()
    m,d,px,op,a,pred=load();blocks,_,bounds,valid=rp.paths(px,m,88);rows=[];returns=[]
    for row in read(P/'GRID.json'):
        z,rr=measure(row,m,d,a,pred,blocks,bounds,valid);rows.append(z);returns.append(rr)
    base=rows[1];good=[z for z in rows if prior.allowed(z,base)]
    pick=lambda zs:max(zs,key=lambda z:(z['training']['cagr'],-z['id']))
    controls={};counts={}
    for family in ['retention','switch_gap','disagreement','combined']:
        g=[z for z in good if z['family']==family];controls[family]=pick(g) if g else base;counts[family]=len(g)
    selected=dict(primary=pick(good),robust=max(good,key=lambda z:(z['robust'],-z['id'])))
    wf=[];stitched=[];baseparts=[]
    for n,end in [(36,60),(60,76),(76,88)]:
        metrics=[rp.stats_rr(rr[:bounds[n]],m['dates'][0],m['ends'][n-1]) for rr in returns];b=metrics[1]
        ids=[i for i,z in enumerate(metrics) if z['sharpe']>=b['sharpe']-.05 and z['max_drawdown']>=b['max_drawdown']-.03]
        win=max(ids,key=lambda i:(metrics[i]['cagr'],-i));rr=returns[win][bounds[n]:bounds[end]];br=returns[1][bounds[n]:bounds[end]];stitched.extend(rr);baseparts.extend(br)
        wf.append(dict(train_end=m['ends'][n-1],test_start=m['dates'][n],test_end=m['ends'][end-1],chosen_id=win,configuration=rows[win],metrics=rp.stats_rr(rr,m['dates'][n],m['ends'][end-1]),base=rp.stats_rr(br,m['dates'][n],m['ends'][end-1])))
    dump('WALKFORWARD.json',dict(folds=wf,metrics=rp.stats_rr(np.array(stitched),m['dates'][36],m['ends'][87]),base=rp.stats_rr(np.array(baseparts),m['dates'][36],m['ends'][87]),pristine_oos=False))
    dump('TRAIN_TRIALS.json',rows);np.savez_compressed(P/'TRAIN_RETURNS.npz',returns=np.stack(returns),bounds=np.array(bounds))
    fr=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=selected,family_controls=controls,family_admissible=counts,base=base,original=rows[0],trials=len(rows),admissible=len(good),code_sha256=sha(__file__),input_hashes={n:sha(P/n) for n in ['PLAN.json','GRID.json','INPUT_HASHES.json']})
    dump('FROZEN.json',fr);(P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('Admissible',counts,flush=True)
    print('Frozen',[(k,z['id'],z['training']['cagr'],z['early']['cagr'],z['recent']['cagr']) for k,z in selected.items()],flush=True)


def check_freeze():
    fr=read(P/'FROZEN.json');assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip();assert sha(__file__)==fr['code_sha256']
    for n,h in fr['input_hashes'].items():assert sha(P/n)==h,n
    return fr


def evaluate():
    fr=check_freeze();m,d,px,op,a,pred=load();out={};allorders={};blocks,_,bounds,valid=rp.paths(px,m,122)
    for name,row in dict(original65=fr['original'],previous66=fr['base'],**fr['family_controls'],**fr['selected']).items():
        ch,sc=choices(row,a,pred,m,d,122);orders=rp.make_orders(ch,sc,m)
        met,eq,tr,hold=ex.e.engine.replay(px,op,orders,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
        rr=rp.fast_returns(ch,blocks,valid);np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
        out[name]=dict(metrics=met,slices=prior.old.previous.cuts(eq,m),configuration=row)
        assert abs(out[name]['slices']['training2016_2023']['cagr']-row['training']['cagr'])<1e-10
        if name in ['original65','previous66']:
            ref=read(old.P/'RESULTS.json')['baseline' if name=='original65' else 'primary']['metrics']
            for k in ['cagr','sharpe','max_drawdown']:assert abs(met[k]-ref[k])<1e-10
        for label,cost in [('next_open_zero',0.),('next_open_cost',.0005)]:
            mm,ee,_,_=ex.e.engine.replay(px,op,orders,cost=cost,execution='open',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]));out[name][label]=dict(metrics=mm,slices=prior.old.previous.cuts(ee,m))
        for suffix,z in [('ORDERS',orders),('TRADES',tr),('HOLDINGS',hold),('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])]:dump(name+'_'+suffix+'.json',z)
        allorders[name]=orders
        print(name,'id',row['id'],'CAGR',met['cagr'],'Sharpe',met['sharpe'],'early',out[name]['slices']['early2016_2020']['cagr'],'later',out[name]['slices']['later34']['cagr'],flush=True)
    q=px.loc[m['dates'][0]:m['ends'][-1],'QQQ'];eq=q/q.iloc[0]*100000;out['qqq']=dict(metrics=ex.e.engine.stats(eq),slices=prior.old.previous.cuts(eq,m));dump('RESULTS.json',out)
    compare=[]
    for i,day in enumerate(m['dates']):
        r=dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
        for name,orders in allorders.items():
            ss=orders[i]['symbols'];r[name]=dict(symbols=ss,monthly_return=float((px.loc[m['ends'][i],ss]/px.loc[day,ss]-1).mean()) if ss else 0.)
        compare.append(r)
    dump('MONTH_COMPARISON.json',compare)
    changes=[p for p,h in read(P/'PROTECTED.json').items() if sha(p)!=h];assert not changes,changes
    dump('VALIDATION.json',dict(independent_daily_replay=True,controls_exact=True,frozen_train_parity=True,protected_files=len(read(P/'PROTECTED.json')),protected_changes=changes,pristine_oos=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','train','evaluate']);globals()[ap.parse_args().phase]()

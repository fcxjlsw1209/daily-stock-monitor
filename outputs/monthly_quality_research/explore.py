"""Ordered residual / quality-interaction / domain-calibration experiments.

All candidate selection uses first88 monthly periods, then specifications freeze
before final34 replay. History was previously researched, not pristine OOS.
"""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,itertools,argparse
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_stability_research'))
import stability as old
import local_refine as local
rp,ex,read,sha=old.rp,old.ex,old.read,old.sha


def dump(name,z):
    (P/name).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')


def initialize():
    assert not (P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        baseline='Frozen monthly_stability_research.robust_control65.3395543%; no registered strategy mutation.',
        ordered_families=['1 residual long momentum: joint rolling market+own-sector betas, prior-day estimates; 126/252beta windows; 210/252formation sessions ending42sessions ago; sum or sum/std; 2.5/5/10% support-only correction; global/calm/highvol.',
          '2 quality-conditioned volatility: path efficiency, positive-day persistence or inverse largest-absolute-day concentration ending42sessions ago; rank volatility126*quality_percentile^1or2; replace half/all volatility weight; global/calm/highvol.',
          '3 stock/ETF calibration: within-class ranks or own trailing756day percentile (min252observations) then cross-sectional ranks; apply core/support/all; blend25/50/100%; global/calm/highvol. No enforced stock/ETF allocation.'],
        combination='After individual training only, retain one admissible maximum-training-CAGR single change per family, test combinations of these three (up to7 nonempty). Keep original baseline option.',
        execution='Long-only,unlevered,max3equal,2half,<2cash; monthend fixed units. Long core>=50%; same frozen122pools/prices/eligibility. Zero-cost same-close primary; next-open0cost and5bp/side diagnostic.',
        selection='First88periods through2023-11-30. Require trainSharpe>=baseline-.05; drawdown no worse than baseline by>.03; early53CAGR>=baseline; recent35CAGR>=baseline-.02. Select maxtrainCAGR primary plus worst-segment-Sharpe robust control; retain best admissible single-family controls. Freeze all before evaluating last34.',
        validation='Causality/prefix invariance; prior-beta availability; skip42for long factors; no future-validity candidate filtering; baseline122decision parity; independent daily/monthly replay; old ledgers/code protected.',
        limits='All history previously researched. Not author reproduction; historical cap/sector proxy and assumedETF17 limitations inherited. No additional search after holdout evaluation. No guarantee80percent.',
        references=['https://pure.eur.nl/en/publications/residual-momentum/','https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2370931']))
    folders=['monthly_stability_research','monthly_2016_extension','monthly_stability_monitor','monthly_industry_monitor','monthly_regime_monitor','monthly_buffer5_monitor','daily_stock_monitor','pressure_shadow_monitor','resilience_shadow_monitor','accel_shadow_monitor','high52_shadow_monitor','accel40_shadow_monitor']
    dump('PROTECTED.json',{str(f):sha(f) for folder in folders for f in (O/folder).rglob('*') if f.is_file() and f.suffix!='.pyc' and f.name!='run.lock'})


def ranks(z):
    return pd.Series(z).replace([np.inf,-np.inf],np.nan).rank(pct=True).fillna(.5).to_numpy()-.5


def group_rank(arr,m):
    out=np.zeros_like(arr);etfs=set(m['etfs'])
    for i,ss in enumerate(m['symbols']):
        mask=np.array([s in etfs for s in ss])
        for group in [mask,~mask]:out[i,group]=ranks(arr[i,group])
    return out


def residual_returns(px,sectors,window):
    """Two-factor coefficients with intercept in fit; retain fitted alpha in residual.
    Never subtract a fitted intercept: residual drift is the momentum of interest.
    All rolling beta coefficients are shifted one session before applying returns.
    """
    r=px.pct_change(fill_method=None);market=r.SPY
    vm=market.rolling(window).var();cm=r.rolling(window).cov(market)
    market_beta=cm.div(vm.replace(0,np.nan),axis=0).shift(1)
    out=r-market_beta.mul(market,axis=0)
    mapping=ex.t.strict.SECTOR;foreign=ex.t.strict.FOREIGN
    # Sector assignment is a historical proxy inherited from the panel, not a
    # claim of point-in-time industry classification. Beta/price dates are causal.
    for b in mapping.values():
        if b not in r:continue
        sector=r[b];vs=sector.rolling(window).var();cms=market.rolling(window).cov(sector)
        cs=r.rolling(window).cov(sector);det=vm*vs-cms*cms
        safe=det.where(det>vm*vs*1e-8)
        bm=(cm.mul(vs,axis=0)-cs.mul(cms,axis=0)).div(safe,axis=0).shift(1)
        bs=(cs.mul(vm,axis=0)-cm.mul(cms,axis=0)).div(safe,axis=0).shift(1)
        joint=r-bm.mul(market,axis=0)-bs.mul(sector,axis=0)
        for s in r:
            if s in foreign:mask=pd.Series(foreign[s]==b,index=r.index)
            elif s in sectors:mask=sectors[s].reindex(r.index).map(mapping).eq(b)
            else:continue
            good=mask&joint[s].notna();out.loc[good,s]=joint.loc[good,s]
    return out


def components(m,d,L,x,px,f):
    st=local.states(px,m,d);frozen=read(old.P/'LOCAL_FROZEN.json')['selected']['robust_control']['spec']
    baseline=local.score(frozen,m,d,L,x,f,st,len(m['dates']))
    calm=st['calm10'][:,None];high=d['high'][:,None]
    cw=np.where(calm,.5,.52)
    core=np.where(calm,.5*f['252_raw'],d['core'])
    vw=np.where(calm,.2,.219375*(1-np.where(high,.10,0)/.48))
    weights=dict(core=cw,vol=vw,accel=np.where(calm,0,.185625*(1-np.where(high,.10,0)/.48)),
        flow=np.where(calm,.075,.075*(1-np.where(high,.10,0)/.48)),
        sector126=np.where(calm,.0875,np.where(high,.025,0)),
        sector252=np.where(calm,.0875,0),path=np.where(calm,.05,0),
        stability=np.where(~calm&high,.05,0),tail=np.where(~calm&high,.025,0))
    arr=dict(core=np.where(calm,f['252_raw'],f['294_v63']),vol=f['vol'],accel=f['accel'],
        flow=f['flow'],sector126=x['sector126'],sector252=x['sector252'],path=f['path'],stability=x['stability'],tail=x['tail'])
    np.testing.assert_allclose(sum(weights[k]*arr[k] for k in arr),baseline,atol=1e-12,rtol=0)
    return baseline,core,cw,vw,weights,arr,st


def make_features(px,m,sectors,d,L,x,f):
    baseline,core,cw,vw,weights,arr,st=components(m,d,L,x,px,f)
    out=dict(baseline=baseline,core=core,core_weight=cw,vol_weight=vw,vol=arr['vol'])
    etfs=set(m['etfs'])
    for beta in [126,252]:
        resid=residual_returns(px,sectors,beta)
        for length in [210,252]:
            total=resid.rolling(length).sum().shift(42)
            for mode,z in [('sum',total),('risk',total/resid.rolling(length).std().shift(42).replace(0,np.nan))]:
                rows=[]
                for day,ss in zip(m['dates'],m['symbols']):
                    a=z.loc[day,ss].copy();a.loc[list(set(ss)&etfs)]=np.nan;rows.append(ranks(a))
                out[f'resid_{beta}_{length}_{mode}']=np.stack(rows)
    r=px.pct_change(fill_method=None);lr=np.log(px/px.shift(1))
    quality=dict(path=np.log(px.shift(42)/px.shift(294))/lr.abs().rolling(252).sum().shift(42).replace(0,np.nan),
        persistence=(lr.gt(0).where(lr.notna()).rolling(252).sum()-lr.lt(0).where(lr.notna()).rolling(252).sum()).shift(42)/252,
        no_jump=-(lr.abs().rolling(252).max()/lr.abs().rolling(252).sum().replace(0,np.nan)).shift(42))
    for name,z in quality.items():
        q=old.rank_rows(z,m)+.5
        for power in [1,2]:out[f'quality_{name}_{power}']=np.stack([ranks((arr['vol'][i]+.5)*q[i]**power) for i in range(len(q))])
    hist_raw=dict(core_calm=px.shift(42)/px.shift(252)-1,
        core_other=(px.shift(42)/px.shift(294)-1)/r.rolling(63).std().replace(0,np.nan),
        vol=r.rolling(126).std(),accel=px.shift(42)/px.shift(126)-px.shift(126)/px.shift(210))
    hist={k:old.rank_rows(z.rolling(756,min_periods=252).rank(pct=True),m) for k,z in hist_raw.items()}
    for method in ['group','history']:
        if method=='group':new={k:group_rank(v,m) for k,v in arr.items()}
        else:
            # Historical calibration of price features only. Industry/CMF terms
            # retain their current cross-sectional definitions.
            new=dict(arr,core=np.where(st['calm10'][:,None],hist['core_calm'],hist['core_other']),vol=hist['vol'],accel=hist['accel'])
        for scope in ['core','support','all']:
            chosen=[k for k in arr if scope=='all' or (k=='core')==(scope=='core')]
            out[f'cal_{method}_{scope}']=sum(weights[k]*(new[k]-arr[k]) for k in chosen)
    return out,st


def prepare():
    m,d,L,x,px,op,f=old.load()
    sectors_file=O/'index_union/alias_resolved_inputs/sectors.csv.gz'
    sectors=pd.read_csv(sectors_file,index_col=0,parse_dates=True,low_memory=False)
    arrays,st=make_features(px,m,sectors,d,L,x,f)
    np.savez_compressed(P/'FEATURES.npz',**arrays,**{'state_'+k:v for k,v in st.items()})
    dump('INPUT_HASHES.json',{str(q):sha(q) for q in [old.P/'LOCAL_FROZEN.json',old.P/'FEATURES.npz',old.P/'INPUT_HASHES.json',Path(old.__file__),Path(local.__file__),sectors_file]})
    print('Prepared',len(arrays),'arrays',flush=True)


def load():
    for file,h in read(P/'INPUT_HASHES.json').items():assert sha(file)==h,file
    return (*old.load(),dict(np.load(P/'FEATURES.npz')))


def active(features,state,n):return features['state_'+state][:n,None].astype(float)


def score(spec,a,n):
    sc=a['baseline'][:n].copy();base=sc.copy()
    for change in spec:
        mask=active(a,change['state'],n);amount=change['amount']*mask;k=change['key']
        if change['family']=='residual':
            sc+=amount*(a[k][:n]-(base-a['core'][:n])/(1-a['core_weight'][:n]))
        elif change['family']=='quality':sc+=amount*a['vol_weight'][:n]*(a[k][:n]-a['vol'][:n])
        elif change['family']=='calibration':sc+=amount*a[k][:n]
    return sc


def choices(spec,a,m,d,n):
    sc=score(spec,a,n);return rp.choose(sc,d['eligible'][:n],m['symbols'][:n],3),sc


def grid(a):
    out=[dict(id=0,family='baseline',spec=[])]
    for family,prefix,amounts in [('residual','resid_',[.025,.05,.10]),('quality','quality_',[.5,1.]),('calibration','cal_',[.25,.5,1.])]:
        for key,amount,state in itertools.product(sorted(k for k in a if k.startswith(prefix)),amounts,['global_','calm10','highvol']):
            out.append(dict(id=len(out),family=family,spec=[dict(family=family,key=key,amount=amount,state=state)]))
    return out


def measure(row,m,d,a,blocks,bounds,valid):
    ch,_=choices(row['spec'],a,m,d,88);rr=rp.fast_returns(ch,blocks,valid)
    full=rp.stats_rr(rr,m['dates'][0],m['ends'][87]);early=rp.stats_rr(rr[:bounds[53]],m['dates'][0],m['ends'][52]);recent=rp.stats_rr(rr[bounds[53]:],m['dates'][53],m['ends'][87])
    robust=min(early['sharpe'],recent['sharpe'])+.3*full['sharpe']+.4*np.log1p(full['cagr'])-.3*abs(full['max_drawdown'])
    return dict(row,training=full,early=early,recent=recent,robust=float(robust))


def allowed(row,base):
    return row['training']['sharpe']>=base['training']['sharpe']-.05 and row['training']['max_drawdown']>=base['training']['max_drawdown']-.03 and row['early']['cagr']>=base['early']['cagr']-1e-12 and row['recent']['cagr']>=base['recent']['cagr']-.02


def train():
    assert not (P/'FROZEN.json').exists()
    m,d,L,x,px,op,f,a=load();blocks,dates,bounds,valid=rp.paths(px,m,88)
    rows=[];family_picks={}
    for family in ['baseline','residual','quality','calibration']:
        for row in grid(a):
            if row['family']==family:rows.append(measure(row,m,d,a,blocks,bounds,valid))
        good=[r for r in rows if r['family']==family and allowed(r,rows[0])]
        family_picks[family]=max(good,key=lambda r:(r['training']['cagr'],-r['id'])) if good else rows[0]
        dump('TRAIN_TRIALS.json',rows)
        print('Finished',family,'trials',sum(r['family']==family for r in rows),'admitted',len(good),'best_train',family_picks[family]['training']['cagr'],flush=True)
    seeds=[r for k,r in family_picks.items() if k!='baseline' and r['id']!=0]
    for size in range(2,len(seeds)+1):
        for combo in itertools.combinations(seeds,size):
            spec=sum([r['spec'] for r in combo],[])
            # Combine score corrections as a convex average to avoid double
            # counting support budgets or negative implicit factor weights.
            spec=[dict(z,amount=z['amount']/size) for z in spec]
            rows.append(measure(dict(id=len(rows),family='combined',spec=spec),m,d,a,blocks,bounds,valid))
    good=[r for r in rows if allowed(r,rows[0])]
    selected=dict(primary=max(good,key=lambda r:(r['training']['cagr'],-r['id'])),robust=max(good,key=lambda r:(r['robust'],-r['id'])))
    frozen=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=selected,family_controls=family_picks,
        trials=len(rows),admissible=len(good),code_sha256=sha(__file__),features_sha256=sha(P/'FEATURES.npz'))
    dump('TRAIN_TRIALS.json',rows);dump('FROZEN.json',frozen);(P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('FROZEN',[(k,v['id'],v['training']['cagr']) for k,v in selected.items()],flush=True)


def evaluate():
    fr=read(P/'FROZEN.json');assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
    assert sha(__file__)==fr['code_sha256'];assert sha(P/'FEATURES.npz')==fr['features_sha256']
    m,d,L,x,px,op,f,a=load();out={}
    selected=dict(fr['family_controls'],**fr['selected'])
    blocks,dates,bounds,valid=rp.paths(px,m,122)
    for name,row in selected.items():
        ch,sc=choices(row['spec'],a,m,d,122);orders=rp.make_orders(ch,sc,m)
        met,eq,tr,hold=ex.e.engine.replay(px,op,orders,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
        rr=rp.fast_returns(ch,blocks,valid);np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
        out[name]=dict(metrics=met,slices=old.previous.cuts(eq,m),spec=row['spec'],training_id=row['id'])
        assert abs(out[name]['slices']['training2016_2023']['cagr']-row['training']['cagr'])<1e-10
        if name=='baseline':
            ref=read(old.P/'RESULTS.json')['robust_control']['metrics']
            for k in ['cagr','sharpe','max_drawdown']:assert abs(met[k]-ref[k])<1e-10
            assert [o['symbols'] for o in orders]==[o['symbols'] for o in read(old.P/'robust_control_ORDERS.json')]
        for label,cost in [('next_open_zero',0.),('next_open_cost',.0005)]:
            mm,_,_,_=ex.e.engine.replay(px,op,orders,cost=cost,execution='open',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]));out[name][label]=mm
        for suffix,z in [('ORDERS',orders),('TRADES',tr),('HOLDINGS',hold),('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])]:dump(name+'_'+suffix+'.json',z)
        print(name,'CAGR',met['cagr'],'Sharpe',met['sharpe'],'early',out[name]['slices']['early2016_2020']['cagr'],'later',out[name]['slices']['later34']['cagr'],flush=True)
    dump('RESULTS.json',out)
    changes=[p for p,h in read(P/'PROTECTED.json').items() if sha(p)!=h];assert not changes,changes
    dump('VALIDATION.json',dict(baseline122_parity=True,independent_replay_parity=True,training_freeze_parity=True,protected_files=len(read(P/'PROTECTED.json')),protected_changes=changes,pristine_oos=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','prepare','train','evaluate']);globals()[ap.parse_args().phase]()

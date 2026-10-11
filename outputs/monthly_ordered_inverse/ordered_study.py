"""Bounded, ordered monthly inverse research. Never writes live monitor files."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, itertools, json, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog

P=Path(__file__).resolve().parent; O=P.parent
sys.path.insert(0,str(O/'monthly_guidance_reset'))
from study import load_prices, actual, display, calendar_momentum, engine
from search import features
from data_inputs import market_states

def read(f): return json.loads(Path(f).read_text())
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(name,x): (P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda z:z.item() if isinstance(z,np.generic) else z)+'\n')

def initialize():
    assert not (P/'PLAN.json').exists(), 'Do not overwrite completed research'
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
      objective='Explain original ordered monthly selections, not maximize historical return.',
      order_assumption='User-approved hypothesis: original list is rank1>rank2>rank3; not independently confirmed by author.',
      primary='Reconstructed monthly US-listed Top32 plus unchanged theme17 ETF assumption.',
      controls=['Top35/theme17','Top32/core17'],
      calibration='First48 holding months ending through2024-12-31; later21 scored only after freezing. Already-viewed history, not pristine OOS.',
      family='16 long-momentum definitions, each skipping42 sessions or2 calendar months; positive/negative126-session volatility, recent and skip42 industry-relative60, small CMF/signed-volume/state corrections; optional skip42 acceleration capped10%.',
      constraints='Nonnegative weights sum1. Core>=50%; volatility<=30%; industry sum<=20%; shape/state/acceleration total<=20%, acceleration<=10%. No pullback entry gate. Positive252-skip42 momentum plus295-session price warmup.',
      fits=256,grid='16 cores x2 normalizations x2 volatility signs x2 acceleration settings x2 regularization levels.',
      loss='Equal weight selected-over-outsider and author-internal-order squared margin losses; each month equal within group; margin.01; regularization0 or.01 toward fixed prior.',
      selection='Calibration only: equal mean of selected-slot recall and internal-order accuracy; ties by exact ordered months, set months, then lower objective and earlier trial.',
      diagnostics='Refit selected specification to all69 for retrospective explanatory upper check, and solve per-month linear feasibility for the same family. This is not original author code recovery or proof against other models.',
      execution='Three equal positions, two half, fewer cash; monthend same-close zero-cost descriptive replay, and next-open control; fixed shares between rebalances.',
      protected='All original accounts, plans, ledgers, fingerprints and prior research remain unchanged; no automation edits.'))
    dirs=['daily_stock_monitor','pressure_shadow_monitor','resilience_shadow_monitor','accel_shadow_monitor','high52_shadow_monitor','accel40_shadow_monitor','paired_stock_monitor','monthly_buffer5_monitor','monthly_regime_monitor']
    files={str(f):sha(f) for d in dirs for f in (O/d).rglob('*') if f.is_file() and '__pycache__' not in str(f) and f.suffix!='.pyc'}
    dump('PROTECTED_BEFORE.json',files)

def panels(px,raw):
    c,s=features(px,raw);r=px.pct_change(fill_method=None);v=r.rolling(63).std()
    out={};cores=[]
    for n in [252,294]:
        for suffix in ['skip42','risk_recent63','risk_aligned','risk_half']:
            k=f'mom{n}_{suffix}';out[k]=c[k];cores.append(k)
    for n in [12,14]:
        m=calendar_momentum(px,n)
        for tag,z in [('raw',m),('risk',m/v.replace(0,np.nan)),('half',m/np.sqrt(v.replace(0,np.nan)))]:
            k=f'calendar{n}_skip2_{tag}';out[k]=z;cores.append(k)
    for n in [252,294]:
        k=f'mean_sharpe{n}_skip42';out[k]=c[k];cores.append(k)
    out.update({k:s[k] for k in ['vol126','industry_relative60','cmf21','cmf63','signed_volume63','acceleration_skip42']})
    out['industry_lagged60']=s['industry_relative60'].shift(42)
    states=market_states(px)*2-1
    # Normalize first, multiply by a market state afterwards. Otherwise the
    # sign reversal could treat tied/neutral missing features asymmetrically.
    return out,cores,states

def make_pack(px,raw,periods,pools,members,n,etf_mode,feat,cores,states,norm):
    symbols=[sorted({actual(s) for s in pools[str(n)][p['start']]}|members[etf_mode][1]) for p in periods]
    names=list(feat)+['highvol_cmf63','trend_industry60']
    size=max(map(len,symbols));X=np.zeros((len(periods),size,len(names)))
    eligible=np.zeros(X.shape[:2],bool);labels=np.zeros_like(eligible);order=np.full((len(periods),3),-1,int);coverage=[]
    warm=px.notna().rolling(295).sum().eq(295)&px.gt(0)
    m=px.shift(42)/px.shift(252)-1
    for i,(p,ss) in enumerate(zip(periods,symbols)):
        d=p['start'];tab=pd.DataFrame({k:z.loc[d,ss] for k,z in feat.items()})
        q=engine.normalize(tab,norm)
        q['highvol_cmf63']=q.cmf63*states.at[d,'high_market_vol']
        q['trend_industry60']=q.industry_relative60*states.at[d,'spy_trend']
        X[i,:len(ss)]=q[names].to_numpy()
        eligible[i,:len(ss)]=(warm.loc[d,ss]&m.loc[d,ss].gt(0)).to_numpy()
        for j,a in enumerate(p['holdings']):
            a=actual(a)
            if a in ss:
                k=ss.index(a);labels[i,k]=True
                if eligible[i,k]:order[i,j]=k
        coverage.append(dict(date=d,industry_missing=[display(s) for s in ss if pd.isna(tab.at[s,'industry_relative60'])],
                             author_unavailable=[a for j,a in enumerate(p['holdings']) if order[i,j]<0]))
    return dict(X=X,eligible=eligible,labels=labels,order=order,symbols=symbols,names=names,cores=cores,norm=norm,coverage=coverage)

def prepare():
    px,raw,members,periods,sources,pools=load_prices();feat,cores,states=panels(px,raw)
    meta=dict(periods=periods,cores=cores,development=[p['end']<='2024-12-31' for p in periods],packs={})
    for n,e in [(32,'theme17'),(35,'theme17'),(32,'core17')]:
        for norm in ['percentile','magnitude']:
            key=f'top{n}_{e}_{norm}';d=make_pack(px,raw,periods,pools,members,n,e,feat,cores,states,norm)
            np.savez_compressed(P/(key+'.npz'),**{k:d[k] for k in ['X','eligible','labels','order']})
            meta['packs'][key]={k:d[k] for k in ['symbols','names','norm','coverage']}
    dump('INPUTS.json',meta);dump('SOURCE_HASHES.json',sources)
    print('Prepared',len(periods),'months',len(cores),'cores',flush=True)

def pack(key,meta=None):
    meta=meta or read(P/'INPUTS.json');return dict(np.load(P/(key+'.npz')),**meta['packs'][key])

def basis(d,core,sign,acceleration):
    spec=[(core,1.,'core'),('vol126',sign,'vol'),('industry_relative60',1.,'industry'),('industry_lagged60',1.,'industry')]
    for n in ['cmf21','cmf63','signed_volume63','highvol_cmf63','trend_industry60']:
        spec.extend([(n,1.,'shape'),(n,-1.,'shape')])
    if acceleration:spec.append(('acceleration_skip42',1.,'accel'))
    X=np.stack([d['X'][:,:,d['names'].index(n)]*s for n,s,g in spec],axis=-1)
    return X,spec

def limits(spec):
    n=len(spec);rows=[];caps=[]
    for gs,cap in [({'industry'},.2),({'shape','accel'},.2)]:
        rows.append([float(g in gs) for _,_,g in spec]);caps.append(cap)
    bounds=[(.5,1.) if g=='core' else (0.,.3) if g=='vol' else (0.,.1) if g=='accel' else (0.,.2) for _,_,g in spec]
    prior=np.zeros(n);prior[0]=.65;prior[1]=.2;prior[2:4]=.05;prior[4]=.05
    return np.array(rows),np.array(caps),bounds,prior

def pairs(X,d,mask):
    ds=[];ws=[];groups=[];months=[]
    for i in np.flatnonzero(mask):
        available=[int(k) for k in d['order'][i] if k>=0]
        other=np.flatnonzero(d['eligible'][i]&~d['labels'][i])
        pairs0=[(a,int(b)) for a in available for b in other]
        pairs1=list(itertools.combinations(available,2))
        for group,ps in enumerate([pairs0,pairs1]):
            for a,b in ps:
                ds.append(X[i,a]-X[i,b]);ws.append(1/len(ps));groups.append(group);months.append(int(i))
    D=np.array(ds);w=np.array(ws);g=np.array(groups)
    for group in [0,1]:w[g==group]*=.5/w[g==group].sum()
    return D,w,g,np.array(months)

def objective(w,D,pw,reg,prior,margin=.01):
    h=np.maximum(margin-np.einsum('ij,j->i',D,w,optimize=False),0)
    return float(np.sum(pw*h*h)+reg*((w-prior)**2).sum()), -2*np.einsum('ij,i->j',D,pw*h,optimize=False)+2*reg*(w-prior)

def fit(X,d,mask,spec,reg):
    D,pw,g,ids=pairs(X,d,mask);A,b,bounds,prior=limits(spec)
    result=minimize(objective,prior,args=(D,pw,reg,prior),jac=True,method='SLSQP',bounds=bounds,
       constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(len(w))},
                    {'type':'ineq','fun':lambda w:b-np.einsum('ij,j->i',A,w,optimize=False),'jac':lambda w:-A}],
       options={'maxiter':400,'ftol':1e-11})
    assert result.success,result.message
    assert abs(result.x.sum()-1)<1e-8 and np.all(np.einsum('ij,j->i',A,result.x,optimize=False)<=b+1e-8)
    return result.x,float(result.fun)

def ranks_and_picks(scores,eligible):
    order=np.argsort(-np.where(eligible,scores,-np.inf),axis=1,kind='stable')
    ranks=np.empty_like(order);np.put_along_axis(ranks,order,np.arange(1,order.shape[1]+1)[None,:],axis=1)
    chosen=[]
    for i,row in enumerate(order):
        ids=[int(j) for j in row[:3] if eligible[i,j]];chosen.append(ids if len(ids)>=2 else [])
    return ranks,chosen

def metrics(scores,d,mask):
    ranks,chosen=ranks_and_picks(scores,d['eligible']);m=dict(months=0,set_exact=0,order_exact=0,top1_exact=0,shared_slots=0,position_exact=0,pairs_correct=0,pairs_total=0,unavailable_author_slots=0)
    for i in np.flatnonzero(mask):
        ours=chosen[i];au=list(d['order'][i]);m['months']+=1;m['unavailable_author_slots']+=sum(a<0 for a in au)
        m['set_exact']+=bool(len(ours)==3 and set(ours)==set(au));m['order_exact']+=ours==au
        m['top1_exact']+=bool(ours and ours[0]==au[0]);m['shared_slots']+=sum(bool(d['labels'][i,j]) for j in ours)
        m['position_exact']+=sum(a==b for a,b in zip(ours,au))
        for a,b in itertools.combinations(au,2):
            if a>=0 and b>=0:m['pairs_total']+=1;m['pairs_correct']+=bool(ranks[i,a]<ranks[i,b])
    m['selection_score']=.5*m['shared_slots']/(3*m['months'])+.5*m['pairs_correct']/max(m['pairs_total'],1)
    return m

def select_key(row):
    m=row['calibration'];return m['selection_score'],m['order_exact'],m['set_exact'],-row['objective'],-row['id']

def train():
    meta=read(P/'INPUTS.json');mask=np.array(meta['development']);assert mask.sum()==48
    rows=[]
    for norm,core,sign,accel,reg in itertools.product(['percentile','magnitude'],meta['cores'],[1.,-1.],[False,True],[0.,.01]):
        key='top32_theme17_'+norm;d=pack(key,meta);X,spec=basis(d,core,sign,accel)
        w,obj=fit(X,d,mask,spec,reg);score=np.einsum('ijk,k->ij',X,w,optimize=False)
        row=dict(id=len(rows),key=key,core=core,vol_sign=sign,acceleration=accel,regularization=reg,spec=spec,weights=w.tolist(),objective=obj,calibration=metrics(score,d,mask))
        rows.append(row)
        if len(rows)%32==0:print('Fitted',len(rows),'best calibration',max(rows,key=select_key)['calibration'],flush=True)
    assert len(rows)==256
    dump('TRIALS.json',rows)
    # Freeze before any later-month metrics or returns are computed.
    winner=max(rows,key=select_key)
    dump('FROZEN.json',dict(winner=winner,trial_count=len(rows),selection='calibration ordered similarity only',created_at=datetime.now(timezone.utc).isoformat()))
    print('FROZEN',winner['id'],winner['core'],winner['calibration'],flush=True)

def feasibility(X,d,spec,i,include_order=True):
    if (d['order'][i]<0).any():return {'status':'author_unavailable'}
    mask=np.zeros(len(d['eligible']),bool);mask[i]=True;D,pw,g,ids=pairs(X,d,mask)
    if not include_order:D=D[g==0]
    A,b,bounds,prior=limits(spec)
    # Maximize a common margin. Strict positive margin proves this local
    # linear family can order every relevant pair, even if global weights fail.
    n=len(spec);constraints=np.vstack([np.column_stack([-D,np.ones(len(D))]),np.column_stack([A,np.zeros(len(A))])])
    bound=np.r_[np.zeros(len(D)),b]
    result=linprog(np.r_[np.zeros(n),-1.],A_ub=constraints,b_ub=bound,A_eq=np.array([np.r_[np.ones(n),0.]]),b_eq=[1.],bounds=bounds+[(None,None)],method='highs')
    assert result.success,result.message
    return dict(status='solved',max_margin=float(result.x[-1]),strictly_feasible=bool(result.x[-1]>1e-8),weights=result.x[:-1].tolist())

def independent(px,orders):
    nav=100000.;parts=[]
    for i,o in enumerate(orders):
        z=px.loc[o['date']:o['end'],o['symbols']]
        assert np.isfinite(z.to_numpy()).all() and z.gt(0).all().all()
        eq=z.div(z.iloc[0]).mean(axis=1)*nav if o['symbols'] else pd.Series(nav,index=z.index)
        parts.append(eq if not i else eq.iloc[1:]);nav=float(eq.iloc[-1])
    return pd.concat(parts)

def evaluate():
    meta=read(P/'INPUTS.json');frozen=read(P/'FROZEN.json');frozen_sha=sha(P/'FROZEN.json');winner=frozen['winner']
    px,raw,members,periods,sources,pools=load_prices();dev=np.array(meta['development']);allmask=np.ones(len(dev),bool)
    results={};curves={};monthrows={};ordersets={}
    def run(name,key,scores,weights=None,spec=None,execution='close',explicit=None):
        d=pack(key,meta);ranks,picks=ranks_and_picks(scores,d['eligible']);orders=[];rows=[]
        for i,p in enumerate(periods):
            ss=[d['symbols'][i][j] for j in picks[i]] if explicit is None else explicit[i]
            scorelist=[float(scores[i,j]) for j in picks[i]] if explicit is None else []
            orders.append(dict(date=p['start'],end=p['end'],signal_date=p['start'],symbols=ss,scores=scorelist))
            row=dict(date=p['start'],end=p['end'],author=p['holdings'],ours=list(map(display,ss)),
                     author_ranks=[int(ranks[i,j]) if j>=0 else None for j in d['order'][i]],
                     unavailable=d['coverage'][i]['author_unavailable'])
            if weights is not None:
                X,_=basis(d,winner['core'],winner['vol_sign'],winner['acceleration'])
                candidates=[]
                for j,s in enumerate(d['symbols'][i]):
                    if s not in ss and display(s) not in p['holdings']:continue
                    parts={}
                    for k,(f,sign,group) in enumerate(spec):parts[group]=parts.get(group,0.)+float(X[i,j,k]*weights[k]*100)
                    candidates.append(dict(symbol=display(s),rank=int(ranks[i,j]) if d['eligible'][i,j] else None,score=float(scores[i,j]*100),contributions=parts))
                row['candidate_contributions']=candidates
            rows.append(row)
        met,eq,trades,holds=engine.replay(px,raw['opening']*px/raw['close'],orders,cost=0,execution=execution,start=pd.Timestamp(periods[0]['start']),end=pd.Timestamp(periods[-1]['end']))
        if execution=='close':np.testing.assert_allclose(eq,independent(px,orders),rtol=1e-11,atol=1e-6)
        match={label:metrics(scores,d,mask) for label,mask in [('all',allmask),('calibration48',dev),('later21',~dev)]}
        slices={}
        for label,a,b in [('calibration48','2020-12-31','2024-12-31'),('later21','2024-12-31','2026-09-24')]:
            z=eq.loc[a:b];slices[label]=engine.stats(z/z.iloc[0]*100000.)
        results[name]=dict(metrics=met,matching=None if explicit else match,slices=slices)
        curves[name]=eq;monthrows[name]=rows;ordersets[name]=orders
        dump(name+'_HOLDINGS.json',holds);dump(name+'_TRADES.json',trades)
        print(name,'CAGR',round(met['cagr']*100,2),'Sharpe',round(met['sharpe'],3),'matching',match['all'] if explicit is None else 'author',flush=True)
    key=winner['key'];d=pack(key,meta);X,spec=basis(d,winner['core'],winner['vol_sign'],winner['acceleration']);w=np.array(winner['weights'])
    score=np.einsum('ijk,k->ij',X,w,optimize=False)
    run('ordered_frozen48',key,score,w,spec)
    run('ordered_frozen48_next_open',key,score,w,spec,execution='open')
    for k in ['top35_theme17_'+d['norm'],'top32_core17_'+d['norm']]:
        dd=pack(k,meta);xx,_=basis(dd,winner['core'],winner['vol_sign'],winner['acceleration'])
        run(k,key=k,scores=np.einsum('ijk,k->ij',xx,w,optimize=False),weights=w,spec=spec)
    # Retrospective explanatory fit only; architecture chosen on first48.
    wa,obj=fit(X,d,allmask,spec,winner['regularization']);allscore=np.einsum('ijk,k->ij',X,wa,optimize=False)
    dump('RETROSPECTIVE_WEIGHTS.json',dict(weights=wa.tolist(),objective=obj,warning='Fitted to all69 original holding lists; retrospective explanation, not forecast validation.'))
    run('ordered_retrospective69',key,allscore,wa,spec)
    # Frozen old score control, recomputed exactly from prior audit to avoid
    # changing old normalizations or rounding its picks.
    audit=read(O/'monthly_decision_audit/MONTHS.json');old=np.zeros_like(score)
    for i,m in enumerate(audit):
        rr={actual(r['symbol']):r for r in m['rows']}
        for j,s in enumerate(d['symbols'][i]):old[i,j]=(rr[s]['score']-50)/100
    run('previous_candidate',key,old)
    for i,row in enumerate(monthrows['previous_candidate']):assert row['ours']==audit[i]['ours']
    run('author',key,score,explicit=[[actual(s) for s in p['holdings']] for p in periods])
    feasible=[]
    for i,p in enumerate(periods):
        feasible.append(dict(date=p['start'],set_only=feasibility(X,d,spec,i,False),ordered=feasibility(X,d,spec,i,True)))
    dump('MONTH_FEASIBILITY.json',feasible)
    # Selected architecture ablations do not trigger a new search or choose by return.
    ablations={}
    for group in ['vol','industry','shape','accel']:
        keep=np.array([g!=group for _,_,g in spec]);ww=w*keep
        sc=np.einsum('ijk,k->ij',X,ww,optimize=False)
        ablations[group]=metrics(sc,d,allmask)
    dump('ABLATIONS.json',ablations)
    dump('RESULTS.json',results);dump('MONTHS.json',monthrows)
    dump('DAILY_EQUITY.json',[dict(date=str(date.date()),**{name:float(eq.at[date]) for name,eq in curves.items()}) for date in curves['author'].index])
    assert sha(P/'FROZEN.json')==frozen_sha
    protected=read(P/'PROTECTED_BEFORE.json');changed=[f for f,h in protected.items() if sha(f)!=h];assert not changed,changed
    dump('VALIDATION.json',dict(independent_replay_passed=True,old69_holdings_parity=True,frozen_hash_unchanged=True,protected_files=len(protected),protected_changes=changed))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['init','prepare','train','evaluate']);a=ap.parse_args()
    {'init':initialize,'prepare':prepare,'train':train,'evaluate':evaluate}[a.phase]()

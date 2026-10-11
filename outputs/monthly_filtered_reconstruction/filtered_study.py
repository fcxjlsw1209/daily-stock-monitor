"""Retrospective holdings reconstruction on fixed user-requested diagnostic slices.
Never optimize returns; never write an observer, ledger, plan or original research cache.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys,argparse,time
import numpy as np
import pandas as pd
from scipy.optimize import linprog,milp,Bounds,LinearConstraint,minimize
from scipy.sparse import csr_matrix,hstack,vstack,eye
P=Path(__file__).resolve().parent;O=P.parent;OLD=O/'monthly_factor_search_v2'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(f,v):(P/f).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def actual(s):return 'research_equity_FB' if s=='META' else s
def display(s):return 'META' if s=='research_equity_FB' else s

def prepare():
    files=[OLD/'INPUTS.json',OLD/'return_leader_open_cost_HOLDINGS.json',O/'monthly_supplied_ledger_audit/PERIODS.json',O/'monthly_ledger_reconstruction/UNIVERSE_AUDIT.json',O/'monthly_ledger_reconstruction/BOUNDARY_AUDIT.json']
    sources={str(p):sha(p) for p in files}
    ledger=read(files[2]);base={r['signal_date']:r for r in read(files[1])}
    missing={r['date']:r['missing_from_proxy'] for r in read(files[3]) if r['pool']=='theme17'}
    border=read(files[4])['positions'];rank={(r['date'],r['symbol']):r['rank'] for r in border if r['strategy']=='new_monthly'}
    records=[]
    for r in ledger:
        d=r['start'];chosen=base[d]['symbols'];etfs=sorted(set(chosen)&{'ARKK','SOXX'})
        edge={s:rank.get((d,s)) for s in chosen if rank.get((d,s)) is not None and 28<=rank[(d,s)]<=32}
        records.append(dict(date=d,end=r['end'],target=r['holdings'],baseline=chosen,excluded_etf=etfs,missing_target=missing[d],baseline_boundary=edge,
                            included_A=not etfs and not missing[d],included_B=not etfs and not missing[d] and not edge))
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose='Match all three source-ledger holdings in each retained period; retrospective rule inference, not OOS performance.',
        slices={'A':'Exclude months where frozen return_leader selected ARKK/SOXX or source target outside original theme17+Top32 proxy.',
                'B':'If A exact matching remains below90%, exclude baseline-selected stocks ranked28-32; freeze this mask before any new fit.'},
        counts={'all':len(records),'A':sum(r['included_A'] for r in records),'B':sum(r['included_B'] for r in records)},
        boundary='Estimated issuer cap ranks at original monthend signal; ETFs not cap-ranked. The baseline is fixed and never replaced by a refit for inclusion.',
        constraints=['Same49 candidates and eligibility; ARKK/SOXX remain candidates. Only months are excluded.',
                     'Single uniform scoring formula: nonnegative long momentum core>=50%, noncore<=50%, small market state/Ridge correction<=10%.',
                     'No ticker/date-specific coefficients, future returns, original chart, or repeated exclusions until fit.',
                     'No change to eight existing paper observers. No CAGR calculated from disconnected retained months.'],records=records)
    if (P/'PLAN.json').exists():
        old=read(P/'PLAN.json');assert old['records']==records;plan=old
    else:dump('PLAN.json',plan)
    metadata=read(OLD/'INPUTS.json');packs={}
    for key in ['theme17__percentile','theme17__magnitude']:
        m=metadata[key];sources[m['file']]=sha(m['file']);assert sources[m['file']]==m['sha256']
        a=np.load(m['file']);sel=a['label_mask'];ds=np.array(m['dates'])[sel].tolist();assert ds==[r['date'] for r in records]
        correction=OLD/(key+'_CORRECTIONS_PLAN.json');cf=OLD/(key+'__correction_features.npz');sources[str(correction)]=sha(correction);sources[str(cf)]=sha(cf)
        cp=read(correction);assert cp['dates']==ds;extra=np.load(cf)['X'];X=a['X'][sel];eligible=a['eligible'][sel];symbols=[s for s,b in zip(m['symbols'],sel) if b]
        y=np.array([[s in {actual(t) for t in r['target']} for s in ss] for ss,r in zip(symbols,records)])
        assert np.array_equal(y,a['labels'][sel])
        core=[i for i in m['core_ids'] if 7<=i<28];side=m['side_ids']
        design=np.concatenate([X[:,:,core+side],extra],axis=2)
        names=[m['feature_names'][i] for i in core+side]+cp['extra_names']
        packs[key]=dict(key=key,X=X,extra=extra,design=design,names=names,core=len(core),side=len(side),small=extra.shape[2],eligible=eligible,labels=y,symbols=symbols,records=records,
                        dates=ds,core_ids=core,side_ids=side,feature_names=m['feature_names'])
        assert np.isfinite(design).all()
    dump('SOURCE_HASHES.json',sources);return plan,packs

def mask(data,slice_):
    return np.ones(len(data['dates']),bool) if slice_=='all' else np.array([r['included_'+slice_] for r in data['records']],bool)

def pick(scores,eligible):
    return np.argsort(-np.where(eligible,scores,-np.inf),axis=1,kind='stable')[:,:3]

def evaluate(data,scores,slice_):
    picked=pick(scores,data['eligible']);selected=mask(data,slice_);detail=[]
    for i in np.flatnonzero(selected):
        syms=[data['symbols'][i][j] for j in picked[i]];target=data['records'][i]['target'];overlap=len(set(syms)&{actual(s) for s in target})
        order=np.argsort(-np.where(data['eligible'][i],scores[i],-np.inf),kind='stable');ranks={s:int(np.where(order==j)[0][0])+1 for j,s in enumerate(data['symbols'][i]) if s in {actual(z) for z in target}}
        detail.append(dict(date=data['dates'][i],end=data['records'][i]['end'],target=target,selected=[display(s) for s in syms],shared=overlap,exact=overlap==3,target_score_ranks={display(s):v for s,v in ranks.items()}))
    return dict(periods=len(detail),exact_periods=sum(r['exact'] for r in detail),shared_positions=sum(r['shared'] for r in detail),total_positions=3*len(detail),details=detail)

def baseline_results(data):
    out={}
    for name in ['return_leader','holdings_leader','development_leader','simple_leader']:
        hs={r['signal_date']:r for r in read(OLD/(name+'_open_cost_HOLDINGS.json'))};details=[]
        for r in data['records']:
            symbols=hs[r['date']]['symbols'];n=len(set(symbols)&set(r['target']));details.append(dict(date=r['date'],shared=n,exact=n==3,selected=symbols))
        out[name]={}
        for sl in ['all','A','B']:
            kept=[r for r,m in zip(details,mask(data,sl)) if m];out[name][sl]=dict(periods=len(kept),exact_periods=sum(r['exact'] for r in kept),shared_positions=sum(r['shared'] for r in kept),total_positions=3*len(kept))
    return out

def pair_design(data,slice_):
    chunks=[];groups=[];indices=[]
    for g,i in enumerate(np.flatnonzero(mask(data,slice_))):
        yes=np.flatnonzero(data['labels'][i]);no=np.flatnonzero(data['eligible'][i]&~data['labels'][i])
        assert len(yes)==3 and data['eligible'][i,yes].all()
        x=data['design'][i];D=(x[yes,None,:]-x[no][None,:,:]).reshape(-1,x.shape[1]);chunks.append(D);groups.extend([g]*len(D));indices.append(i)
    return np.vstack(chunks),np.array(groups),indices

def budget(data):
    n=len(data['names']);core=np.zeros(n);core[:data['core']]=1
    small=np.zeros(n);small[-data['small']:]=1
    return np.vstack([-core,small]),np.array([-.5,.1])

def feasibility(data,slice_):
    D,groups,indices=pair_design(data,slice_);n=D.shape[1];B,b=budget(data)
    A=np.vstack([np.column_stack([-D,np.ones(len(D))]),np.column_stack([B,np.zeros(len(B))])])
    obj=np.r_[np.zeros(n),-1.]
    sol=linprog(obj,A_ub=A,b_ub=np.r_[np.zeros(len(D)),b],A_eq=np.array([np.r_[np.ones(n),0.]]),b_eq=[1.],bounds=[(0,None)]*n+[(-1,1)],method='highs')
    assert sol.success,sol.message
    w=sol.x[:-1];diff=np.einsum('ij,j->i',D,w,optimize=False)
    assert diff.min()>=sol.x[-1]-1e-7 and w[:data['core']].sum()>=.5-1e-8 and w[-data['small']:].sum()<=.1+1e-8
    scores=np.einsum('mnj,j->mn',data['design'],w,optimize=False)
    return dict(key=data['key'],slice=slice_,family='long-momentum positive mixture + signed support + <=10% regime/causal Ridge',margin=float(sol.x[-1]),strict_full_fit=bool(sol.x[-1]>1e-7),
                coefficients={k:float(v) for k,v in zip(data['names'],w) if v>1e-8},evaluations={sl:evaluate(data,scores,sl) for sl in ['all','A','B']},weights=w.tolist(),solver_message=sol.message)

def fit_soft(data,slice_,penalty):
    D,groups,indices=pair_design(data,slice_);B,b=budget(data);n=D.shape[1]
    # Weighted squared pairwise hinge: each month equally weighted, regularized coefficients.
    weights=1/np.bincount(groups)[groups]/len(indices);norm=lambda v:np.einsum('ij,j->i',D,v,optimize=False)
    def fun(v):
        err=np.maximum(.025-norm(v),0)
        return float(np.sum(weights*err**2)+penalty*np.sum(v*v)), -2*np.einsum('ij,i->j',D,weights*err,optimize=False)+2*penalty*v
    seed=np.zeros(n);seed[:data['core']]=1/data['core']
    constraints=[dict(type='eq',fun=lambda v:v.sum()-1,jac=lambda v:np.ones(n)),dict(type='ineq',fun=lambda v:b-np.einsum('ij,j->i',B,v,optimize=False),jac=lambda v:-B)]
    sol=minimize(fun,seed,jac=True,bounds=[(0,1)]*n,constraints=constraints,method='SLSQP',options={'ftol':1e-10,'maxiter':800})
    assert sol.success,sol.message
    scores=np.einsum('mnj,j->mn',data['design'],sol.x,optimize=False)
    return dict(key=data['key'],slice=slice_,kind='regularized_pairwise',penalty=penalty,loss=float(sol.fun),weights=sol.x.tolist(),coefficients={k:float(v) for k,v in zip(data['names'],sol.x) if v>1e-7},evaluations={sl:evaluate(data,scores,sl) for sl in ['all','A','B']})

def fit_max_months(data,slice_,seconds=60,max_features=None):
    D,groups,indices=pair_design(data,slice_);n=D.shape[1];months=len(indices);B,b=budget(data)
    # z=1 relaxes all inequalities of an entire period. Exact feasible periods stay z=0.
    M=1.001;eps=1e-5;Z=csr_matrix((-np.full(len(D),M),(np.arange(len(D)),groups)),shape=(len(D),months))
    A=hstack([csr_matrix(-D),Z]);upper=np.full(len(D),-eps)
    A=vstack([A,hstack([csr_matrix(B),csr_matrix((len(B),months))]),csr_matrix(np.r_[np.ones(n),np.zeros(months)][None,:])]);lo=np.r_[np.full(len(D)+len(B),-np.inf),1.];hi=np.r_[upper,b,1.]
    obj=np.r_[np.zeros(n),np.ones(months)];integrality=np.r_[np.zeros(n),np.ones(months)];lb=np.zeros(n+months);ub=np.ones(n+months)
    if max_features is not None:
        # Feature activation is only a sparsity diagnostic, with the same market-data predictors.
        A=hstack([A,csr_matrix((A.shape[0],n))]);A=vstack([A,hstack([eye(n),csr_matrix((n,months)),-eye(n)]),csr_matrix(np.r_[np.zeros(n+months),np.ones(n)][None,:])]);lo=np.r_[lo,np.full(n+1,-np.inf)];hi=np.r_[hi,np.zeros(n),max_features]
        obj=np.r_[obj,np.full(n,1e-6)];integrality=np.r_[integrality,np.ones(n)];lb=np.r_[lb,np.zeros(n)];ub=np.r_[ub,np.ones(n)]
    sol=milp(obj,integrality=integrality,bounds=Bounds(lb,ub),constraints=LinearConstraint(A,lo,hi),options={'time_limit':seconds,'mip_rel_gap':0.0})
    if sol.x is None:return dict(key=data['key'],slice=slice_,kind='maximum_months',status=int(sol.status),message=sol.message,seconds=seconds,max_features=max_features)
    w=sol.x[:n];scores=np.einsum('mnj,j->mn',data['design'],w,optimize=False);diff=np.einsum('ij,j->i',D,w,optimize=False);z=sol.x[n:n+months]
    assert np.min(diff+M*z[groups])>=eps-1e-6
    assert w[:data['core']].sum()>=.5-1e-7 and w[-data['small']:].sum()<=.1+1e-7
    return dict(key=data['key'],slice=slice_,kind='maximum_months',status=int(sol.status),message=sol.message,seconds=seconds,max_features=max_features,mip_gap=float(sol.mip_gap),
                objective=float(sol.fun),objective_lower_bound=float(sol.mip_dual_bound),weights=w.tolist(),coefficients={k:float(v) for k,v in zip(data['names'],w) if v>1e-7},
                evaluations={sl:evaluate(data,scores,sl) for sl in ['all','A','B']})

def run(mode):
    plan,packs=prepare();dump('BASELINES.json',baseline_results(packs['theme17__percentile']))
    if mode=='feasibility':
        trials=[]
        for sl in ['A','B']:
            for data in packs.values():
                r=feasibility(data,sl);trials.append(r);dump('FEASIBILITY.json',trials)
                print('FEASIBILITY',data['key'],sl,'margin',r['margin'],'exact',r['evaluations'][sl]['exact_periods'],flush=True)
    elif mode=='fit':
        trials=[]
        for sl in ['A','B']:
            for data in packs.values():
                for penalty in [.0001,.001,.01]:
                    r=fit_soft(data,sl,penalty);trials.append(r);dump('FITS.json',trials);print('SOFT',data['key'],sl,penalty,r['evaluations'][sl]['exact_periods'],flush=True)
                r=fit_max_months(data,sl,45);trials.append(r);dump('FITS.json',trials);print('MIP',data['key'],sl,r.get('evaluations',{}).get(sl,{}).get('exact_periods'),r['status'],flush=True)
    print('DONE',mode,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['feasibility','fit'],default='feasibility');args=p.parse_args();run(args.mode)

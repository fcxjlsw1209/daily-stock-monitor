"""All selections use only the fixed first48 months, never later-check metrics."""
import itertools, json, sqlite3
import numpy as np
from study import P,read,dump,picks

def evaluate(scores,data,mask):
    ids,good=picks(scores,data['eligible'])
    hit=(np.take_along_axis(np.broadcast_to(data['labels'],scores.shape),ids,axis=-1)&good).sum(axis=-1)
    gross=np.take_along_axis(np.broadcast_to(data['gross'],scores.shape),ids,axis=-1)
    complete=np.take_along_axis(np.broadcast_to(data['complete'],scores.shape),ids,axis=-1)
    active=good.sum(axis=-1)>=2
    period=np.where(active,np.where(good,gross,0).sum(axis=-1)/np.maximum(good.sum(axis=-1),1),1)
    hit=hit*active
    valid=(((complete|~good).all(axis=-1)|~active)&np.isfinite(period)&(period>0))[...,mask].all(axis=-1)
    return dict(exact=(hit[...,mask]==3).sum(axis=-1),shared=hit[...,mask].sum(axis=-1),
                log_return=np.log(np.where(period>0,period,1))[...,mask].sum(axis=-1),valid=valid)

def score_rule(rule,data,pred=None):
    sc=np.zeros(data['eligible'].shape)
    for name,w in rule['weights'].items():sc+=w*data['X'][:,:,data['names'].index(name)]
    if rule.get('ml_weight',0):sc+=rule['ml_weight']*pred[rule['ml_id']]
    return sc

def main():
    meta=read(P/'INPUTS.json');names=meta['names'];dev=np.array(meta['development_mask']);assert dev.sum()==48
    pred=np.load(P/'PREDICTIONS.npz')['predictions']
    packs={n:dict(np.load(P/f'top35_{n}.npz'),names=names) for n in ['percentile','magnitude']}
    con=sqlite3.connect(P/'TRIALS.sqlite');con.execute('CREATE TABLE IF NOT EXISTS trials (id INTEGER PRIMARY KEY,phase TEXT,rule TEXT,dev_exact INT,dev_shared INT,dev_log_return REAL,valid INT)')
    assert con.execute('SELECT COUNT(*) FROM trials').fetchone()[0]==0,'Do not overwrite prior experiment'
    rows=[];rules=[]
    def add(rule,phase):
        d=packs[rule['normalization']];z=evaluate(score_rule(rule,d,pred),d,dev)
        row=dict(id=len(rows),phase=phase,rule=rule,dev_exact=int(z['exact']),dev_shared=int(z['shared']),dev_log_return=float(z['log_return']),valid=bool(z['valid']))
        rows.append(row);rules.append(rule)
        return row
    def top(criterion,count,subset=None):
        rr=[r for r in (subset if subset is not None else rows) if r['valid']]
        key=(lambda r:(r['dev_exact'],r['dev_shared'],-len(r['rule']['weights']),-r['id'])) if criterion=='matches' else (lambda r:(r['dev_log_return'],-len(r['rule']['weights']),-r['id']))
        return sorted(rr,key=key,reverse=True)[:count]
    for norm in packs:
        for core in meta['cores']:
            add(dict(normalization=norm,core=core,weights={core:1.}),'core')
            for a in meta['support']:
                for w in [.1,.2,.3,.4,.5]:
                    add(dict(normalization=norm,core=core,weights={core:1-w,a:w}),'support')
            for a,b in itertools.combinations(meta['support'],2):
                if a=='neg_'+b or b=='neg_'+a:continue
                for wc,wa,wb in [(.5,.25,.25),(.6,.2,.2),(.7,.2,.1),(.5,.35,.15)]:
                    add(dict(normalization=norm,core=core,weights={core:wc,a:wa,b:wb}),'support')
    print('BASE GRID',len(rows),'best matches',top('matches',1)[0]['dev_exact'],flush=True)
    base=list(rows)
    seeds={r['id']:r for criterion in ['matches','return'] for r in top(criterion,6,base)}
    for seed in seeds.values():
        r=seed['rule'];core=r['core'];wc=r['weights'][core]
        for correction in meta['corrections']:
            for gamma in [.05,.10]:
                if wc>=1:weights={core:1-gamma,correction:gamma}
                else:
                    weights={n:(w if n==core else w*(1-gamma/(1-wc))) for n,w in r['weights'].items()}
                    weights[correction]=weights.get(correction,0)+gamma
                add(dict(normalization=r['normalization'],core=core,weights=weights,seed_id=seed['id'],correction_budget=gamma),'shape_state')
    # Forecasts trained on market outcomes only. Combine up to10% predictive rank
    # with existing seeds; together with state/shape corrections never exceed20%.
    cseeds={r['id']:r for criterion in ['matches','return'] for r in top(criterion,6)}
    for seed in cseeds.values():
        r=seed['rule'];core=r['core'];wc=r['weights'][core]
        for mi in range(len(pred)):
            for gamma in [.05,.10]:
                if wc>=1:weights={core:1-gamma}
                else:weights={n:(w if n==core else w*(1-gamma/(1-wc))) for n,w in r['weights'].items()}
                add(dict(normalization=r['normalization'],core=core,weights=weights,ml_id=mi,ml_weight=gamma,seed_id=seed['id']), 'market_ridge')
    # Inverse diagnostic: learn a small global score from prior author preferences.
    # Not described as the author's own Ridge objective or as independent OOS.
    from scipy.optimize import minimize
    for seed in top('matches',6,base):
        r=seed['rule'];core=r['core'];d=packs[r['normalization']]
        cols=list(r['weights'])+[n for n in ['cmf21','neg_market_relative63','high_market_vol__neg_cmf63','tech_relative__industry_relative60'] if n not in r['weights']]
        coreid=cols.index(core);support_ids=[i for i,n in enumerate(cols) if n in r['weights'] and n!=core]
        correction_ids=[i for i,n in enumerate(cols) if n not in r['weights']]
        xx=d['X'][:,:,[names.index(n) for n in cols]];pairs=[];pw=[]
        for i in np.flatnonzero(dev):
            yy=np.flatnonzero(d['eligible'][i]&d['labels'][i]);nn=np.flatnonzero(d['eligible'][i]&~d['labels'][i])
            for a in yy:
                for b in nn:pairs.append(xx[i,a]-xx[i,b]);pw.append(1/max(len(yy)*len(nn),1))
        D=np.array(pairs);pw=np.array(pw);pw/=pw.sum();w0=np.array([r['weights'].get(n,0) for n in cols])
        for reg in [.01,.1,1.]:
            def fun(w):
                gap=.02-np.einsum('ij,j->i',D,w,optimize=False);hinge=np.maximum(gap,0)
                value=float(np.sum(pw*hinge**2)+reg*np.sum((w-w0)**2))
                grad=-2*np.einsum('ij,i->j',D,pw*hinge,optimize=False)+2*reg*(w-w0)
                return value,grad
            constraints=[{'type':'eq','fun':lambda w:w.sum()-1},
                         {'type':'ineq','fun':lambda w:.2-w[correction_ids].sum()}]
            bounds=[(.5,1.) if i==coreid else (0.,1.) for i in range(len(cols))]
            opt=minimize(fun,w0,jac=True,bounds=bounds,constraints=constraints,method='SLSQP',options={'maxiter':300,'ftol':1e-10})
            assert opt.success,opt.message
            weights={n:float(w) for n,w in zip(cols,opt.x) if w>1e-9}
            add(dict(normalization=r['normalization'],core=core,weights=weights,seed_id=seed['id'],inverse_regularization=reg,
                     fitted_to='author_pairwise_preferences_first48_only',objective=float(opt.fun)),'inverse_diagnostic')
    con.executemany('INSERT INTO trials VALUES (?,?,?,?,?,?,?)',[(r['id'],r['phase'],json.dumps(r['rule']),r['dev_exact'],r['dev_shared'],r['dev_log_return'],int(r['valid'])) for r in rows]);con.commit();con.close()
    normal=[r for r in rows if r['phase']!='inverse_diagnostic']
    inverse=[r for r in rows if r['phase']=='inverse_diagnostic']
    leaders=dict(return_selected=top('return',1,normal)[0],match_selected=top('matches',1,normal)[0],inverse_selected=top('matches',1,inverse)[0])
    # Freeze only development-selected rules before later-check performance.
    dump('FROZEN_RULES.json',dict(selection_period='2020-12-31 to2024-12-31',trial_count=len(rows),leaders=leaders,
                               base_leaders={k:top(k,6,base) for k in ['matches','return']},
                               shortlist={k:top(k,6,normal) for k in ['matches','return']}))
    print('FROZEN',len(rows),{k:(v['dev_exact'],v['dev_shared'],round(np.expm1(v['dev_log_return']/4),4)) for k,v in leaders.items()},flush=True)

if __name__=='__main__':main()

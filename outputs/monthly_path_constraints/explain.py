"""Can one author-compatible score explain latent whole-curve paths?

This is an explanatory ordering test. A bounded, arbitrary correction envelope
is only a necessary feasibility test; it is NOT a trained model or transaction.
"""
import json
import numpy as np
import pandas as pd
from study import P, O, load, read, dump, dates_and_events, sha, START, END

FEATURES = ['volatility', 'industry_relative', 'cmf21', 'volume_ratio', 'near_high']


def explanations_for(explanations,path_id):
    return {f'{path_id}__{family}':explanations[f'{path_id}__{family}'] for family in ['raw','risk','mixed']}


def project(w, radius=.5):
    if np.abs(w).sum() <= radius: return w
    u = np.sort(np.abs(w))[::-1]; c = np.cumsum(u)
    rho = np.flatnonzero(u-(c-radius)/np.arange(1, len(u)+1) > 0)[-1]
    theta = (c[rho]-radius)/(rho+1)
    return np.sign(w)*np.maximum(np.abs(w)-theta, 0)


def fit_pairwise(X, b):
    assert np.isfinite(X).all() and np.isfinite(b).all()
    assert np.abs(X).max()<=1+1e-12 and np.abs(b).max()<=.5+1e-12
    w = np.zeros(X.shape[1]); a = w.copy(); v = w.copy(); best = w.copy(); minimum = float('inf')
    history = []
    for t in range(1, 1501):
        assert np.isfinite(w).all() and np.abs(w).sum()<=.5+1e-12
        # Explicit bounded reductions avoid platform BLAS floating-point flags.
        with np.errstate(divide='raise',over='raise',invalid='raise'):
            residual = np.minimum(b+np.einsum('ij,j->i',X,w,optimize=False), 0.)
        loss = float(np.mean(residual**2)+1e-5*(w*w).sum())
        if loss < minimum: minimum = loss; best = w.copy()
        if t % 100 == 0: history.append(dict(iteration=t, loss=loss, coefficients=w.tolist()))
        g = 2*np.einsum('ij,i->j',X,residual,optimize=False)/len(X)+2e-5*w
        a = .9*a+.1*g; v = .999*v+.001*g*g
        w = project(w-.01*(a/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8))
    return best, minimum, history


def frames(px, pools):
    _, events = dates_and_events(px)
    fr = O / 'index_union/alias_resolved_inputs'
    fs = {k: pd.read_csv(fr / f'{k}.csv.gz', index_col=0, parse_dates=True) for k in ['close', 'high', 'low', 'volume', 'sector60']}
    sources = read(O / 'four_factor_rebound/ETF_SOURCES.json') + read(O / 'monthly_reconstruction_v2/ETF_MANIFEST.json')
    for row in sources:
        z = pd.read_csv(row['file'], index_col=0, parse_dates=True)
        for key, col in [('close', 'Close'), ('high', 'High'), ('low', 'Low'), ('volume', 'Volume')]:
            fs[key][row['symbol']] = z[col].reindex(px.index)
        fs['sector60'][row['symbol']] = px.SPY.pct_change(60, fill_method=None)
    for k in fs: fs[k] = fs[k].reindex(index=px.index, columns=px.columns)
    ret = px.pct_change(fill_method=None); vol = ret.rolling(63).std()
    m12 = px.shift(42)/px.shift(252)-1; m6 = px.shift(42)/px.shift(126)-1
    c, h, l, v = (fs[k] for k in ['close', 'high', 'low', 'volume'])
    flow = ((2*c-h-l)/(h-l).replace(0, np.nan)).fillna(0)
    raw = dict(volatility=vol,
        industry_relative=(1+px.pct_change(60, fill_method=None))/(1+fs['sector60'])-1,
        cmf21=(flow*v).rolling(21).sum()/v.rolling(21).sum(),
        volume_ratio=v.rolling(21).mean()/v.rolling(126).mean(),
        near_high=px/px.rolling(252).max())
    out = {}
    for pool, member in pools.items():
        r12 = m12.where(member).rank(axis=1, pct=True)
        r6 = m6.where(member).rank(axis=1, pct=True)
        risk = (m12/vol).where(member).rank(axis=1, pct=True)
        cores = dict(raw=.8*r12+.2*r6, risk=.8*risk+.2*r12)
        cores['mixed'] = .5*cores['raw']+.5*cores['risk']
        f = {k: val.where(member).rank(axis=1, pct=True).fillna(.5)-.5 for k, val in raw.items()}
        eligible = member&m12.gt(0)&px.notna().rolling(253, min_periods=253).sum().eq(253)&px.gt(0)
        rows = []
        for day in events:
            cols = sorted(px.columns[eligible.loc[day]])
            rows.append(dict(day=day, symbols=cols,
                X=np.column_stack([f[k].loc[day, cols] for k in FEATURES]),
                cores={k: val.loc[day, cols].to_numpy() for k, val in cores.items()}))
        out[pool] = rows
    return out


def replay(px, rows, coef, core):
    nav = pd.Series(index=px.loc[START:END].index, dtype=float); value = 1.; selections = []
    for i, z in enumerate(rows):
        score=.5*z['cores'][core]+np.einsum('ij,j->i',z['X'],coef,optimize=False)
        ids=np.argsort(-score, kind='stable')[:3]
        chosen=[z['symbols'][j] for j in ids] if len(ids)>=2 else []
        end=rows[i+1]['day'] if i+1<len(rows) else END
        f=px.loc[z['day']:end, chosen]
        if chosen:
            if not f.notna().all().all() or not f.gt(0).all().all():
                return None, None, f'Unpriceable held path at {z["day"]}'
            seg=value*(f/f.iloc[0]).mean(axis=1)
        else: seg=pd.Series(value, index=px.loc[z['day']:end].index)
        nav.loc[seg.index]=seg;value=float(seg.iloc[-1]);selections.append(set(chosen))
    return nav,selections,None


def run():
    px,pools,_=load();data=frames(px,pools);out={};curves={}
    for file in sorted(P.glob('*_RESULT.json')):
        ident=file.name.replace('_RESULT.json','');source=read(file);pool=source['pool']
        meta=read(P/f'META_{pool}.json');states=np.load(P/'internal_states'/f'{ident}_states.npz')['final']
        targets=[]
        for m,z in enumerate(meta):
            cc=np.load(z['file'])['combos'][states[m]]
            targets.append({z['symbols'][j] for j in cc})
        rows=data[pool]
        for family in ['raw','risk','mixed']:
            xs=[];bs=[];bounds=[]
            for row, selected in zip(rows,targets):
                chosen=[j for j,s in enumerate(row['symbols']) if s in selected]
                other=[j for j,s in enumerate(row['symbols']) if s not in selected]
                assert len(chosen)==len(selected)
                for j in chosen:
                    xs.extend(row['X'][j]-row['X'][other]);bs.extend(.5*(row['cores'][family][j]-row['cores'][family][other]))
                bounds.append((chosen,other))
            X=np.asarray(xs);b=np.asarray(bs);coef,loss,trace=fit_pairwise(X,b)
            deficits=[];same=0
            for row,sel,(chosen,other) in zip(rows,targets,bounds):
                scores=.5*row['cores'][family]+np.einsum('ij,j->i',row['X'],coef,optimize=False)
                selected={row['symbols'][j] for j in np.argsort(-scores,kind='stable')[:3]}
                same+=selected==sel
                deficits.append(max(0.,float(scores[other].max()-scores[chosen].min()))/2 if chosen and other else 0.)
            nav,selection,error=replay(px,rows,coef,family)
            key=f'{ident}__{family}'
            metric=None
            if nav is not None:
                r=nav.pct_change().dropna();q=px.loc[nav.index,'QQQ'].pct_change().dropna()
                metric=dict(cagr_252_returns=float(nav.iloc[-1]**(252/len(r))-1),
                    cagr_252_nav_count=float(nav.iloc[-1]**(252/len(nav))-1),sharpe=float(r.mean()/r.std()*np.sqrt(252)),
                    max_dd=float((nav/nav.cummax()-1).min()),correlation=float(r.corr(q)),wealth=float(nav.iloc[-1]))
                curves[key]=nav
            out[key]=dict(core_family=family, core_weight=.5, side_weights=dict(zip(FEATURES,coef.tolist())),
                pairwise_loss=loss, optimization_trace=trace, exact_target_months=same,total_months=len(rows),
                months_requiring_more_than_10_percent_full_score_range_correction=int(sum(v>.05+1e-12 for v in deficits)),
                minimum_per_security_correction_to_force_target=dict(maximum=float(max(deficits)),median=float(np.median(deficits)),p90=float(np.quantile(deficits,.9))),
                meaning='Correction bounds allow hindsight arbitrary per-security +/-0.05 adjustments for feasibility ONLY. This is not a trained model or claimed author mechanism.',
                replay_metrics=metric,replay_failure=error,full_sample_explanatory_fit=True,
                author_rule_found=False,potential_transactions_exported=False)
            print('EXPLAIN',key,'matched',same,'large_correction',sum(v>.05 for v in deficits),'metrics',json.dumps(metric),flush=True)
    dump('EXPLANATION.json',out);pd.DataFrame(curves).to_csv(P/'EXPLANATION_NAV.csv.gz')
    print('EXPLAIN_DONE',flush=True)


if __name__=='__main__':run()

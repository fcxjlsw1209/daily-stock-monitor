"""Early-month rule refinement only; no return evaluation."""
import json
from itertools import product
import numpy as np
from scipy.optimize import linprog
from decisions import P, read, dump, sha, CORE, SIDE, FEATURES, STRICT, display

PAIR_FEATURES=list(product(range(len(CORE)),range(len(CORE),len(FEATURES))))


def load_first_months(limit=12):
    decisions=read(P/'DECISIONS.json')[:limit]
    dates={r['date'] for r in decisions}
    snaps={d:[] for d in dates}
    for row in read(P/'SNAPSHOTS.json'):
        if row['date'] in dates:snaps[row['date']].append(row)
    out=[]
    for d in decisions:
        z=sorted(snaps[d['date']],key=lambda r:r['symbol'])
        X={f:np.array([[r['normalized'][f][k] for k in FEATURES] for r in z]) for f in z[0]['normalized']}
        out.append(dict(date=d['date'],symbols=[r['symbol'] for r in z],y=np.array([r['selected'] for r in z]),X=X))
    return out


def matrix(row,family,interaction):
    X=row['X'][family]
    if not interaction:return X
    Z=np.column_stack([2*X[:,i]*X[:,j] for i,j in PAIR_FEATURES])
    assert np.abs(Z).max()<=.5+1e-12
    return np.column_stack([X,Z])


def solve(rows,family,dominance,interaction):
    nc=len(CORE);ns=len(SIDE)+(len(PAIR_FEATURES) if interaction else 0)
    d=[]
    for row in rows:
        X=matrix(row,family,interaction)
        yes,no=np.flatnonzero(row['y']),np.flatnonzero(~row['y'])
        d.append((X[yes,None]-X[no][None,:]).reshape(-1,nc+ns))
    D=np.vstack(d);n=nc+2*ns+1
    A=np.column_stack([-D[:,:nc],-D[:,nc:],D[:,nc:],np.ones(len(D))]);b=np.zeros(len(D))
    extras=[];rhs=[]
    if dominance=='half':
        z=np.zeros(n);z[:nc]=-1;extras.append(z);rhs.append(-.5)
    else:
        # Four groups in the author's prose. Count each side coefficient's
        # absolute-value upper bound, not only its signed sum.
        for indices in [[0,1,2,3],[4],[5,6,7,8,9]]:
            z=np.zeros(n);z[:nc]=-1
            for j in indices:z[nc+j]=1;z[nc+ns+j]=1
            extras.append(z);rhs.append(0.)
    if interaction:
        z=np.zeros(n)
        z[nc+len(SIDE):nc+ns]=1;z[nc+ns+len(SIDE):nc+2*ns]=1
        extras.append(z);rhs.append(.1)
    A=np.vstack([A,extras]);b=np.r_[b,rhs]
    eq=np.ones((1,n));eq[0,-1]=0;c=np.zeros(n);c[-1]=-1
    fit=linprog(c,A_ub=A,b_ub=b,A_eq=eq,b_eq=[1.],bounds=[(0,None)]*(n-1)+[(-1,1)],
        method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    assert fit.success,fit.message
    v=fit.x;w=np.r_[v[:nc],v[nc:nc+ns]-v[nc+ns:nc+2*ns]]
    margins=np.einsum('ij,j->i',D,w,optimize=False)
    assert np.isfinite(margins).all() and margins.min()>=v[-1]-1e-7
    dual=float(np.einsum('i,i->',b,fit.ineqlin.marginals,optimize=False)+fit.eqlin.marginals.sum()-fit.lower.marginals[-1]+fit.upper.marginals[-1])
    primal=float(np.einsum('i,i->',c,v,optimize=False))
    assert abs(primal-dual)<1e-7
    if interaction:assert np.abs(w[len(FEATURES):]).sum()<=.1+1e-8
    return dict(weights=w.tolist(),margin=float(v[-1]),core_mass=float(w[:nc].sum()),
        model_l1=float(np.abs(w[len(FEATURES):]).sum()) if interaction else 0.,
        primal_dual_gap=abs(primal-dual),feature_names=FEATURES+([f'{FEATURES[i]}*{FEATURES[j]}' for i,j in PAIR_FEATURES] if interaction else []))


def choices(row,fit,family,interaction):
    scores=np.einsum('ij,j->i',matrix(row,family,interaction),fit['weights'],optimize=False)
    assert np.isfinite(scores).all()
    ids=np.argsort(-scores,kind='stable')[:3]
    return [row['symbols'][i] for i in ids],scores


def main():
    rows=load_first_months()
    protected=read(P/'SOURCE_HASHES.json')
    for f in ['DECISIONS.json','SNAPSHOTS.json']:protected[str(P/f)]=sha(P/f)
    results={}
    for family,dominance,interaction in product(['universe_percentile','raw_magnitude','asset_class_percentile'],['half','largest_group'],[False,True]):
        name=f'{family}__{dominance}__'+('small_interaction' if interaction else 'linear')
        history=[];prior=None;exact_prefix=0
        for i,row in enumerate(rows):
            prediction=None
            if prior:
                pick,_=choices(row,prior,family,interaction)
                prediction=dict(trained_through=rows[i-1]['date'],picks=pick,overlap=len(set(pick)&set(np.array(row['symbols'])[row['y']].tolist())))
            fit=solve(rows[:i+1],family,dominance,interaction)
            entry=dict(date=row['date'],prior_prediction=prediction,fit=fit)
            if fit['margin']>STRICT:
                for before in rows[:i+1]:
                    actual,_=choices(before,fit,family,interaction)
                    assert set(actual)==set(np.array(before['symbols'])[before['y']])
                exact_prefix=i+1;prior=fit
            history.append(entry)
            if fit['margin']<=STRICT:break
        results[name]=dict(family=family,dominance=dominance,interaction=interaction,
            exact_prefix_months=exact_prefix,history=history,
            first_conflict=None if exact_prefix==len(rows) else history[-1]['date'])
        print(name,'exact prefix',exact_prefix,'last margin',fit['margin'],flush=True)
        dump('FRONTIER_RESULTS.json',results)
    best=max(results,key=lambda k:(results[k]['exact_prefix_months'],-results[k]['interaction'],results[k]['history'][results[k]['exact_prefix_months']-1]['fit']['margin'] if results[k]['exact_prefix_months'] else -1))
    chosen=results[best]
    lines=['从第一月开始复现决策：规则推进记录','',
        '本轮不做全期收益回测。将原持仓按用户假设作为目标，逐月检查买入三只及所有排除项。',
        '8月17日仅作初始持仓，从2016年8月31日这一常规月底决策开始。',
        '“动量主导”按两种解释核对：至少50%；或只要求它为最大的因子组。后者未给其他组任意压倒动量的权限。',
        '小模型仅测试来自既有特征的二阶交互，总绝对权重不超过10%，每只影响不超过±0.05；没有日期或股票代码特征。','',
        f"目前能共同解释的最长前缀：{chosen['exact_prefix_months']}个月；模型假设：{best}。",
        f"下一处尚未解释的月末决策：{chosen['first_conflict'] or '前12个月均有共同解释'}。",'',
        '假设 | 连续共同解释月数 | 首个冲突日']
    for name,r in results.items():lines.append(f"{name} | {r['exact_prefix_months']} | {r['first_conflict'] or '无（仅检验前12个月）'}")
    lines+=['','逐月推进（最佳前缀假设）']
    for entry in chosen['history']:
        i=next(i for i,row in enumerate(rows) if row['date']==entry['date']);row=rows[i]
        targets=[display(s) for s,yes in zip(row['symbols'],row['y']) if yes]
        lines += ['',f"{row['date']}：目标{'、'.join(targets)}。"]
        if entry['prior_prediction']:
            p=entry['prior_prediction'];lines.append(f"  先用仅解释此前月份的规则，选{'、'.join(map(display,p['picks']))}，命中{p['overlap']}/3只。")
        f=entry['fit']
        lines.append(f"  纳入本月选择约束后，共同最小分差{f['margin']:+.6f}；动量权重{f['core_mass']:.2%}；小模型权重上限使用量{f['model_l1']:.2%}。")
        lines.append('  状态：'+('有一组统一系数同时解释目前全部月份，但仍需下月验证。' if f['margin']>STRICT else '当前统一规则家族仍有冲突，不能把本月标为已复现。'))
        weights=[(n,w) for n,w in zip(f['feature_names'],f['weights']) if abs(w)>1e-7]
        lines.append('  拟合的非零系数：'+'；'.join(f'{n} {w:+.4f}' for n,w in weights))
    lines+=['','本轮检验的是截至当前月份能否存在统一规则，并未声称存在唯一参数；没有把每月独立规则接起来，没有使用后续收益选择参数。',
        '所有本月候选对照与分数贡献见months逐月卡片。十二种规则解释是假设空间，不是十二套收益回测。']
    (P/'FRONTIER_REPORT.txt').write_text('\n'.join(lines)+'\n')
    for f,h in protected.items():assert sha(f)==h,f
    dump('FRONTIER_STATUS.json',dict(selected=best,exact_prefix_months=chosen['exact_prefix_months'],
        first_conflict=chosen['first_conflict'],full_period_backtest_run=False,labels_changed=False,
        arbitrary_monthly_weights_concatenated=False,sources_unchanged=len(protected)))
    dump('CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})


if __name__=='__main__':main()

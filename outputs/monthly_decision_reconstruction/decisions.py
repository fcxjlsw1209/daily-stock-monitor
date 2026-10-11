"""Chronological decision dossiers, without return or NAV backtesting."""
from pathlib import Path
from collections import Counter
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog

P = Path(__file__).resolve().parent
O = P.parent
sys.path.insert(0,str(O/'monthly_rule_inference'))
from infer import CORE, SIDE, feature_frames
from study import load, sha, read

FEATURES = CORE+SIDE
FAMILIES = ['universe_percentile','raw_magnitude','asset_class_percentile']
STRICT = 1e-6
CN = dict(mom12_2='年度动量',mom9_2='9个月动量',mom6_2='半年动量',
    risk12_2='风险调整年度动量',risk9_2='风险调整9个月动量',risk6_2='风险调整半年动量',
    vol21='月波动',vol63='季度波动',vol126='半年波动',downside63='近期下行波动',
    industry_relative60='行业相对强弱',cmf21='21日资金流',cmf63='63日资金流',
    volume_ratio21_126='量比',near_high252='接近年内高点',ma126_distance='偏离半年均线')


def display(s): return {'research_equity_FB':'META/FB'}.get(s,s)


def dump(name,obj):
    (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def snapshot_data():
    px,pools,sources=load()
    source=O/'01a0b6b1-511d-7401-b4f7-4dd2aaa895a7/monthly_inverse_holdings/records.json'
    records=read(source);sources.update(records['sources']);sources[str(source)]=sha(source)
    raw={}
    for k in ['close','high','low','volume','sector60']:
        f=O/'index_union/alias_resolved_inputs'/f'{k}.csv.gz'
        raw[k]=pd.read_csv(f,index_col=0,parse_dates=True);sources[str(f)]=sha(f)
    manifests=read(O/'four_factor_rebound/ETF_SOURCES.json')+read(O/'monthly_reconstruction_v2/ETF_MANIFEST.json')
    etfs={r['symbol'] for r in manifests}
    for entry in manifests:
        z=pd.read_csv(entry['file'],index_col=0,parse_dates=True)
        for field,col in [('close','Close'),('high','High'),('low','Low'),('volume','Volume')]:
            raw[field][entry['symbol']]=z[col].reindex(px.index)
        raw['sector60'][entry['symbol']]=px.SPY.pct_change(60,fill_method=None)
    raw={k:v.reindex(index=px.index,columns=px.columns) for k,v in raw.items()}
    feats=feature_frames(px,raw)
    member=pools['theme17']
    warm=px.notna().rolling(253,min_periods=253).sum().eq(253)&px.gt(0)
    # Verify the earliest full-month inputs directly against truncated histories.
    for cut in ['2016-08-31','2016-09-30','2016-10-31']:
        limited=feature_frames(px.loc[:cut],{k:v.loc[:cut] for k,v in raw.items()})
        for k in FEATURES: assert np.allclose(limited[k].iloc[-1],feats[k].loc[cut],equal_nan=True,atol=1e-12,rtol=1e-12),(cut,k)
    rows=[]
    # Explicitly do not carry any period_return, end_nav or subsequent prices
    # from the input record into the inference dataset.
    for event in records['monthly'][1:]:
        date=pd.Timestamp(event['rebalance_date'])
        universe=sorted(px.columns[member.loc[date]])
        raw_table=pd.DataFrame({k:feats[k].loc[date,universe] for k in FEATURES})
        q=raw_table.rank(pct=True).fillna(.5)
        lo,hi=raw_table.quantile(.1),raw_table.quantile(.9)
        magnitude=((raw_table-lo)/(hi-lo).replace(0,np.nan)).clip(0,1).fillna(.5)
        separate=raw_table.copy()*np.nan
        for flag in [False,True]:
            members=[s for s in universe if (s in etfs)==flag]
            separate.loc[members]=raw_table.loc[members].rank(pct=True).fillna(.5)
        symbols=[s for s in universe if bool(warm.at[date,s]) and feats['mom12_2'].at[date,s]>0]
        y=np.array([s in event['source_holdings'] for s in symbols])
        assert y.sum()==3
        rows.append(dict(date=str(date.date()),month=event['holding_month'],symbols=symbols,y=y,
            universe=universe,excluded=[dict(symbol=s,reason='预热或价格不完整' if not warm.at[date,s] else 'R12-2不为正') for s in universe if s not in symbols],
            selected=sorted(event['source_holdings']),new=event['new_buys'],sold=event['liquidations'],
            retained=event['retained_adjustments'],
            X=dict(universe_percentile=q.loc[symbols].to_numpy()-.5,
                   raw_magnitude=magnitude.loc[symbols].to_numpy()-.5,
                   asset_class_percentile=separate.loc[symbols].to_numpy()-.5),
            raw=raw_table.loc[symbols],percentiles=q.loc[symbols]))
    return rows,sources,records['monthly'][0]


def differences(row,family):
    yes,no=np.flatnonzero(row['y']),np.flatnonzero(~row['y'])
    X=row['X'][family]
    D=(X[yes,None]-X[no][None,:]).reshape(-1,len(FEATURES))
    pairs=[dict(date=row['date'],chosen=row['symbols'][i],unchosen=row['symbols'][j]) for i in yes for j in no]
    return D,pairs


def maximum_margin(D):
    core=len(CORE);side=len(SIDE);n=core+2*side+1
    A=np.column_stack([-D[:,:core],-D[:,core:],D[:,core:],np.ones(len(D))])
    # Chosen score >= unchosen score + margin, positive momentum mass >= 0.5.
    dominance=np.zeros(n);dominance[:core]=-1
    A=np.vstack([A,dominance]);b=np.r_[np.zeros(len(D)),-.5]
    eq=np.ones((1,n));eq[0,-1]=0
    objective=np.zeros(n);objective[-1]=-1
    solved=linprog(objective,A_ub=A,b_ub=b,A_eq=eq,b_eq=[1.],
        bounds=[(0,None)]*(n-1)+[(-1,1)],method='highs',
        options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    assert solved.success,solved.message
    v=solved.x
    w=np.r_[v[:core],v[core:core+side]-v[core+side:core+2*side]]
    margins=np.einsum('ij,j->i',D,w,optimize=False)
    assert np.isfinite(margins).all()
    assert margins.min()>=v[-1]-1e-7
    assert w[:core].min()>=-1e-8 and w[:core].sum()>=.5-1e-8
    assert np.abs(w).sum()<=1+1e-8
    primal=float(np.einsum('i,i->',objective,v,optimize=False))
    # HiGHS dual certificate includes RHS and active variable bounds.
    dual=float(np.einsum('i,i->',b,solved.ineqlin.marginals,optimize=False)+np.sum(solved.eqlin.marginals))
    dual+=float(-solved.lower.marginals[-1]+solved.upper.marginals[-1])
    assert abs(primal-dual)<1e-7,(primal,dual)
    return dict(w=w,margin=float(v[-1]),core_mass=float(w[:core].sum()),
        side_l1=float(np.abs(w[core:]).sum()),primal_dual_gap=abs(primal-dual),
        margin_residual=float(margins.min()-v[-1]),binding=np.flatnonzero(margins<=v[-1]+1e-7).tolist(),
        oracle_minimum_correction=max(0.,float(-v[-1]/2)),
        status='exact_local_explanation' if v[-1]>STRICT else ('tie_boundary' if v[-1]>=-STRICT else 'not_separable_in_family'))


def picked(row,w,family):
    scores=np.einsum('ij,j->i',row['X'][family],w,optimize=False)
    assert np.isfinite(scores).all()
    ids=np.argsort(-scores,kind='stable')[:3]
    return sorted(row['symbols'][j] for j in ids),scores


def serialize(solved):
    return {k:v for k,v in solved.items() if k not in ['w','binding']}|{'weights':dict(zip(FEATURES,solved['w'].tolist()))}


def conflict_subset(history):
    subset=list(history)
    # Month-level irreducible explanation conflict, not a claim of global
    # minimal cardinality. Retains the latest decision that exposed it.
    for candidate in history[:-1]:
        if len(subset)<=2:break
        rest=[r for r in subset if r is not candidate]
        if maximum_margin(np.vstack([r['D'] for r in rest]))['margin']<=STRICT:subset=rest
    fit=maximum_margin(np.vstack([r['D'] for r in subset]))
    return dict(months=[r['date'] for r in subset],fit=serialize(fit))


def make_card(row,results,representative):
    line=[f"{row['month']}持仓：{row['date']}收盘决策",'',
        f"目标三只：{'、'.join(map(display,row['selected']))}，每只1/3。",
        f"新买入：{'、'.join(row['new']) or '无'}；清仓：{'、'.join(row['sold']) or '无'}；保留调整：{'、'.join(row['retained']) or '无'}。",
        f"当时代理池{len(row['universe'])}只，沿用正R12-2及预热条件后{len(row['symbols'])}只。以下只使用截至该收盘的数据。",'',
        '一、先解释选中标的的特征，不参考之后涨跌',
        '标的 | R12-2 | R6-2 | 半年年化波动 | 行业相对60日 | CMF21',]
    baseline=.8*row['X']['universe_percentile'][:,0]+.2*row['X']['universe_percentile'][:,2]
    rivals=[row['symbols'][i] for i in np.argsort(-baseline,kind='stable') if not row['y'][i]][:4]
    for s in row['selected']+rivals:
        r=row['raw'].loc[s]
        industry='缺失（中性处理）' if pd.isna(r.industry_relative60) else f'{r.industry_relative60:+.2%}'
        line.append(f"{display(s)}{'〔目标〕' if s in row['selected'] else '〔对照〕'} | {r.mom12_2:+.2%} | {r.mom6_2:+.2%} | {r.vol126*np.sqrt(252):.2%} | {industry} | {r.cmf21:+.3f}")
    line+=['','二、逐个排除候选的可行性']
    for family,res in results.items():
        z=res['local']
        status={'exact_local_explanation':'存在能精确选中三只的本月解释','tie_boundary':'只能达到同分边界','not_separable_in_family':'本因子/预算家族无法严格选中三只'}[z['status']]
        line.append(f"{family}：{status}；最小选中-落选分差最优值{z['margin']:+.6f}。")
        if res['prior_rule']:
            p=res['prior_rule']
            line.append(f"  只沿用截至{p['trained_through']}的前序共同规则，会选{'、'.join(map(display,p['picks']))}；命中{p['overlap']}/3只。")
        if res['cumulative']['status']=='compatible':
            line.append('  本月加入后，仍存在同时解释从2016-08-31至本月全部选择的同一组权重。')
        else:
            line.append(f"  跨月共同规则仍未解决：最早在{res['cumulative']['first_conflict_date']}出现冲突；不按月偷偷换权重。")
    if representative:
        family,fit=representative
        _,scores=picked(row,fit['w'],family)
        other=np.flatnonzero(~row['y']);rival=other[np.argmax(scores[other])]
        line+=['','三、一个本月可行解释（用本月标签推导，尚不能当作已恢复的作者规则）',
            f"采用{family}，动量权重合计{fit['core_mass']:.1%}，侧因子绝对权重合计{fit['side_l1']:.1%}。",
            '系数：'+'；'.join(f"{CN[k]} {v:+.4f}" for k,v in zip(FEATURES,fit['w']) if abs(v)>1e-7),
            f"分数最高的落选者：{display(row['symbols'][rival])}。每只目标股必须都超过它："]
        for i in np.flatnonzero(row['y']):
            contribution=(row['X'][family][i]-row['X'][family][rival])*fit['w']
            plus=np.argsort(-contribution)[:2];minus=int(np.argmin(contribution))
            support='、'.join(f'{CN[FEATURES[j]]}贡献{contribution[j]:+.4f}' for j in plus)
            counter=f'{CN[FEATURES[minus]]}贡献{contribution[minus]:+.4f}'
            line.append(f"  {display(row['symbols'][i])}：总分差{scores[i]-scores[rival]:+.6f}；支持：{support}；反向项：{counter}。")
    else:
        line+=['','三、尚未复现本月选择',
            '三种标准化下均未找到严格排序。需要核对具体特征定义、资格过滤或小模型修正；不得将逐月硬编码当成决策规则。']
    line+=['','四、记录与时序',
        '三个本月解释只用于检查这一次选择是否可能。跨月共用权重单独检验；不将每个月各自拟合的权重拼接成收益回测。',
        '月末持有股数、等权与资格条件是既有假设。本文件没有计算持有期收益、累计收益、CAGR或Sharpe。']
    return '\n'.join(line)+'\n'


def main():
    rows,sources,initial=snapshot_data()
    dump('SOURCE_HASHES.json',sources)
    dump('INITIAL_ALLOCATION.json',dict(date=initial['rebalance_date'],holdings=initial['holdings'],
        classification='non_month_end_initial_condition_not_regular_monthly_rule_label'))
    (P/'months').mkdir(exist_ok=True)
    histories={k:[] for k in FAMILIES};prior={};conflicts={};all_results=[];snapshots=[]
    maxdual=0.
    for ix,row in enumerate(rows):
        result={};representative=None
        for family in FAMILIES:
            D,pairs=differences(row,family)
            previous=None
            if family in prior:
                selected,_=picked(row,prior[family]['w'],family)
                previous=dict(trained_through=prior[family]['date'],picks=selected,
                    overlap=len(set(selected)&set(row['selected'])))
            local=maximum_margin(D);maxdual=max(maxdual,local['primal_dual_gap'])
            if local['margin']>STRICT:
                actual,_=picked(row,local['w'],family)
                assert actual==row['selected']
                if representative is None:representative=(family,local)
            histories[family].append(dict(date=row['date'],D=D))
            if family not in conflicts:
                common=maximum_margin(np.vstack([h['D'] for h in histories[family]]))
                maxdual=max(maxdual,common['primal_dual_gap'])
                if common['margin']>STRICT:
                    for earlier in rows[:ix+1]:
                        actual,_=picked(earlier,common['w'],family)
                        assert actual==earlier['selected']
                    prior[family]=dict(w=common['w'],date=row['date'])
                    cumulative=dict(status='compatible',through=row['date'],fit=serialize(common))
                else:
                    conflict=conflict_subset(histories[family])
                    conflicts[family]=dict(first_conflict_date=row['date'],**conflict)
                    cumulative=dict(status='conflict',**conflicts[family])
                    # The last genuinely compatible rule remains available as
                    # a frozen counterfactual. Never replace it by a local fit.
            else:cumulative=dict(status='conflict',**conflicts[family])
            result[family]=dict(local={**serialize(local),'binding_pairs':[pairs[j] for j in local['binding']]},
                prior_rule=previous,cumulative=cumulative)
        current=dict(date=row['date'],holding_month=row['month'],target=row['selected'],
            universe=row['universe'],eligible=row['symbols'],excluded=row['excluded'],
            universe_count=len(row['universe']),eligible_count=len(row['symbols']),families=result,
            has_exact_local_explanation=representative is not None,
            selected_representative_family=None if representative is None else representative[0])
        all_results.append(current)
        for i,s in enumerate(row['symbols']):
            snapshots.append(dict(date=row['date'],symbol=s,selected=bool(row['y'][i]),
                raw={k:None if pd.isna(row['raw'].at[s,k]) else float(row['raw'].at[s,k]) for k in FEATURES},
                normalized={family:dict(zip(FEATURES,row['X'][family][i].tolist())) for family in FAMILIES}))
        card=make_card(row,result,representative)
        (P/'months'/f"{row['date']}.txt").write_text(card)
        if ix<5 or (ix+1)%20==0:print(ix+1,row['date'],{k:round(v['local']['margin'],6) for k,v in result.items()},flush=True)
    dump('DECISIONS.json',all_results);dump('SNAPSHOTS.json',snapshots);dump('CONFLICTS.json',conflicts)
    lines=['逐月决策复现：按决策日逐一验证','',
        '按用户假设将目标持仓视为作者交易。本轮不计算全期收益，只分析选择与排除的逻辑。',
        '2016-08-17单列初始持仓，常规月末决策从2016-08-31开始。每月详细卡片位于months目录。',
        '本月存在可行解释 ≠ 已找到跨月统一规则。不得将各月拟合参数拼成策略。','',
        '持有月份 | 月末决策日 | 目标持仓 | 单月可行解释 | 前序共同规则状态']
    for r in all_results:
        joint=[k for k,v in r['families'].items() if v['cumulative']['status']=='compatible']
        lines.append(f"{r['holding_month']} | {r['date']} | {','.join(map(display,r['target']))} | {'存在' if r['has_exact_local_explanation'] else '待解决'} | {','.join(joint) if joint else '三种尺度均有未解冲突'}")
    lines+=['','最初三个月的详细分析','']
    for row in rows[:3]:lines.append((P/'months'/f"{row['date']}.txt").read_text())
    (P/'DECISION_JOURNAL.txt').write_text('\n'.join(lines))
    for source,digest in sources.items():assert sha(source)==digest,source
    summary=dict(month_end_decisions=len(rows),normalization_hypotheses=len(FAMILIES),
        locally_exact_by_family={f:sum(r['families'][f]['local']['status']=='exact_local_explanation' for r in all_results) for f in FAMILIES},
        locally_exact_any_family=sum(r['has_exact_local_explanation'] for r in all_results),
        first_conflicts={f:v['first_conflict_date'] for f,v in conflicts.items()},
        full_period_return_backtest_run=False,future_returns_used=False,monthly_weights_concatenated=False,
        maximum_lp_primal_dual_gap=maxdual,feature_prefix_checks=48,
        sources_unchanged=len(sources),original_accounts_unchanged=True,
        common_author_rule_recovered=False,first_decision=rows[0]['date'],last_decision=rows[-1]['date'])
    dump('STATUS.json',summary);dump('CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

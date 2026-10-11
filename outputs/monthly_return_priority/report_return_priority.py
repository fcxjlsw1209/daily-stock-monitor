"""Summarize frozen results without choosing or modifying rules after validation."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(P))
import return_priority as r
read=lambda n:json.loads((P/n).read_text())
R=read('RESULTS.json');F=read('FROZEN.json');V=read('VALIDATION.json');E=json.loads((O/'monthly_exact35/RESULTS.json').read_text())['momentum_residual_tree']
assert r.sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
assert F['code_sha256']==r.sha(P/'return_priority.py')
assert all(r.sha(p)==h for p,h in read('SOURCE_HASHES.json').items())
assert all(r.sha(p)==h for p,h in read('PROTECTED_BEFORE.json').items())
assert F['primary']=='balanced' and not F['author_labels_used']
assert len({z['spec']['id'] for z in F['selected'].values()})==1
labels={'balanced':'本轮收益/风险筛选规则','old_candidate_original':'此前高收益简洁候选（历史搜索选出）','old_formula_top36':'旧因子思路在本次Top36特征重算','fixed70_base':'固定70%动量基础分','author':'作者原记录重放','qqq':'QQQ买入持有'}
L=['月策略：收益与风险优先，不再要求作者持仓完全一致','生成时间：'+datetime.now(timezone.utc).isoformat(),'',
'结论：完成1152组规则的训练期搜索及冻结后验证。新的收益/风险规则没有超越此前高收益简洁候选；不替换现有观察策略。',
'持仓及名单次序只作诊断：没有进入本轮损失函数、目标函数或选择标准。没有用作者收益曲线作为优化目标。',
'核心仍跳过最近约两个月，月末集中持有三只、月内份额固定，只做多或现金。',
'',
'一、统一日期与主口径',
'全期：2020-12-31收盘—2026-09-24。前35个持有月用于设计选择，后34个持有月为本轮冻结后检查。',
'后34个月：2023-11-30收盘—2026-09-24；最后一个月不完整。',
'零费率、零滑点、月末最终收盘价计算信号并同价成交。该成交方式是理想化研究口径。',
'Sharpe为日收益均值/样本标准差×sqrt(252)，未扣无风险利率；CAGR按实际日历时长；回撤按日终净值。',
'历史排名是估算代理，17ETF名单仍是研究假设。此轮统一沿用Top36＋theme17，未搜索股票池宽度。',
'旧简洁候选原始版保留旧Top32池及完整原选股，单独标注；其系数来自此前多轮历史搜索，不能当作这轮35/34首次验证。',
'旧因子Top36重算同时使用本轮归一化/资格数据，因此并非严格只改变股票池宽度的一变量消融。',
'后34个月此前已多次查看，这轮按前35个月冻结只能减少本轮额外过拟合，不能抹除先前研究暴露。',
'',
'二、全期及后34个月比较',
'模型 | 全期年化 | 全期Sharpe | 全期最大回撤 | 后34年化 | 后34Sharpe | 后34最大回撤',
]
for key in ['balanced','old_candidate_original','old_formula_top36','fixed70_base','author','qqq']:
 a=R[key]['metrics'];b=R[key]['slices']['validation34'];L.append(f'{labels[key]} | {a["cagr"]:.2%} | {a["sharpe"]:.3f} | {a["max_drawdown"]:.2%} | {b["cagr"]:.2%} | {b["sharpe"]:.3f} | {b["max_drawdown"]:.2%}')
a=E['metrics'];b=E['slices']['validation34'];L.append(f'上一轮35个月精确拟合主模型 | {a["cagr"]:.2%} | {a["sharpe"]:.3f} | {a["max_drawdown"]:.2%} | {b["cagr"]:.2%} | {b["sharpe"]:.3f} | {b["max_drawdown"]:.2%}')
L += ['',
'旧高收益候选全期仍是77.62%/1.857，而不是训练样本完全一致的模型。它只匹配作者9/69个月三股集合。',
'本轮规则只匹配作者1/69个月集合，但全期Sharpe达到1.729。这印证了持仓相似度和策略绩效必须分开衡量。',
'本轮没有同时提高收益和Sharpe：后段相对精确拟合模型Sharpe提高，但年化下降；相对旧候选两者仍较低。',
'',
'三、本轮唯一选中的规则（3种预先指定选择标准恰好选到同一个ID509）',
'训练稳健目标、训练Sharpe最高、训练CAGR最高均选中同一条规则；不是3个不同的新策略。',
'每个月末对有效候选计算百分位排名，采用下列分数：',
'50% × 跳过最近两个月的14日历月锚点动量（P[t-2月]/P[t-14月]-1，即此前12个月收益）；',
'17.5% × 下行波动较低的排名（负downside63）；',
'12.5% × 同样跳过两个月的动量加速；',
'12.5% × 个股相对于自己所属行业ETF的252/42交易日强弱；',
'7.5% × 市场状态量价修正：SPY63日波动高于其过去252日中位数时用负CMF63，否则用正CMF21。',
'所有单因子是居中百分位，不是把原始单位直接相加。ETF采用现有明确基准映射。',
'该quality分支本来就是低下行波动，risk_aware中的弱市替换不再额外改变波动项；真正起作用的状态变化在CMF。',
'当月持仓在新排名前5以内则保留；空缺按分数从高到低补齐，最多3只。月末重新等权，月内份额固定。',
'2只时各半，少于2只则现金；无短期回撤资格要求、无月内止盈止损、无杠杆、无作者持仓注入。',
'',
'四、怎么选的',
'固定搜索：8种长期动量 ×4种辅助因子组合 ×3种动量权重 ×3种市场模式 ×2种动量资格 ×2种持仓保留门槛＝1152配置。配置间可产生相同交易，不是1152个独立发现。',
'动量权重50%/65%/80%；市场模式固定、弱市低风险、SPY低于200日均线空仓；资格不加门槛/核心动量排名前半；保留前3/前5。',
'仅前35个月计算目标：0.4全段Sharpe＋0.4最差分段Sharpe＋0.2分段平均Sharpe＋0.25ln(1+CAGR)－0.5最大回撤绝对值。',
'分段为前12个月、次12个月、最后11个月；这是训练期稳健性约束，不能把用来选参数的分段叫独立验证。',
'另做前12选规则看后6、前18选规则看后6、前24选规则看后11的向前滚动检查；每段选择只用当时前缀数据。',
]
w=read('TRAIN_WALKFORWARD.json');z=w['stitched'];L.append(f'该内部滚动检查合并2021-12-31至2023-11-30：年化{z["cagr"]:.2%}、Sharpe {z["sharpe"]:.3f}、最大回撤{z["max_drawdown"]:.2%}。')
L += ['滚动检查采用当时选中虚拟规则的后续持仓路径；没有把它当成新实盘账户或作为另一个调参目标。',
'然后用全部前35个月选规则，冻结代码、参数和输入校验和，再看后34个月。不按后段收益更换主模型。',
'',
'五、执行敏感性（本轮ID509，后34个月）',
]
for key,label in [('next_open_zero','下个交易日开盘，零成本'),('next_open_cost','下个交易日开盘，每边0.05%成本')]:
 z=R['balanced'][key]['slices']['validation34'];L.append(f'{label}：年化{z["cagr"]:.2%}、Sharpe {z["sharpe"]:.3f}、最大回撤{z["max_drawdown"]:.2%}。')
L += ['', '六、验证和研究决定',
'9项测试通过：固定份额收益、前五保留、池内移除、现金资格、缺行情拒绝、未来特征/价格隔离、回撤起点、独立回放。',
'训练向量化回放、完整逐日引擎、独立固定份额回放一致；旧简洁候选原版的年化、Sharpe、回撤精确复核。',
f'{len(read("PROTECTED_BEFORE.json"))}个受保护文件哈希未变；原观察账户、计划、账本及上一轮精确拟合结果均未修改。',
'后续研究以此前简洁候选作为性能基准，优化收益/风险表现；持仓匹配只用于理解机制。',
'本次不新增观察账户、不改现有参数，也不宣称恢复作者公式或达到作者收益。',
'冻结参数SHA256：'+r.sha(P/'FROZEN.json'),
]
(P/'REPORT.txt').write_text('\n'.join(L)+'\n')
months={k:read(k+'_MONTHS.json') for k in ['balanced','old_candidate_original','author']}
lines=['月度比较：本轮规则 vs 旧简洁候选 vs 作者，全部原顺序保留。','']
for i,z in enumerate(months['balanced']):
 lines.append(f'{i+1:02d} {z["split"]} {z["signal_date"]} → {z["holding_end"]}')
 for k in months:
  x=months[k][i];lines.append(f'{labels[k]}：{" > ".join(x["selected"]) or "现金"}；收益{x["monthly_return"]:.2%}；共同持仓{x["shared_slots"]}/3')
 lines.append('')
(P/'MONTHLY_COMPARISON.txt').write_text('\n'.join(lines)+'\n')
meta,data,px,op=r.load();s=F['selected']['balanced']['spec'];ch,sc=r.decisions(s,meta,data);expl=[]
g=lambda n:data['X'][:,:,meta['features'].index(n)]
parts={'skip2_long_momentum':.5*g('core_calendar14_v0'),'low_downside_volatility':-.175*g('downside63'),'skip2_acceleration':.125*g('acceleration_skip42'),'stock_vs_own_sector':.125*g('own_sector_skip42_252'),'market_conditioned_cmf':.075*np.where(g('market_highvol')>0,-g('cmf63'),g('cmf21'))}
np.testing.assert_allclose(sum(parts.values()),sc,atol=1e-14)
previous=set()
for i,ss in enumerate(meta['symbols']):
 rank=sorted(np.flatnonzero(data['eligible'][i]),key=lambda j:(-sc[i,j],ss[j]));rankmap={j:k+1 for k,j in enumerate(rank)}
 selected=[ss[j] for j in ch[i]]
 expl.append(dict(signal_date=meta['dates'][i],previous_holdings=sorted(map(r.e.display,previous)),selected=list(map(r.e.display,selected)),candidates=[dict(ticker=r.e.display(ss[j]),rank=rankmap[j],selected=j in ch[i],retained=ss[j] in previous and j in ch[i],score=float(sc[i,j]),contributions={k:float(a[i,j]) for k,a in parts.items()}) for j in rank]))
 previous=set(selected)
r.dump('PRIMARY_EXPLANATIONS.json',expl)
r.dump('STATUS.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),complete=True,candidate_configurations=1152,unique_selected_rules=1,primary_id=509,primary=R['balanced'],previous_candidate=R['old_candidate_original'],author=R['author'],improved_over_previous_candidate=False,author_holdings_used_for_selection=False,observation_accounts_changed=False,unit_tests_passed=9,protected_changes=[],pristine_oos=False,reports=['REPORT.txt','MONTHLY_COMPARISON.txt','PRIMARY_EXPLANATIONS.json']))
print('Report and69monthly comparisons written; frozen inputs and protected files unchanged.')

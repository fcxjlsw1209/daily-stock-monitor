from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
P=Path(__file__).resolve().parent
O=P.parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
families=[('放开权重与资格限制',P,'core_primary'),('直接拟合整组选择概率',P/'listwise','core_primary'),('月底自有持仓状态',P/'stateful','core_primary'),('量价形态＋小幅Ridge',P/'minor_layers','core_primary')]
rows=[]
for label,root,category in families:
 r=read(root/'RESULTS.json')[category];check=read(root/'VALIDATION.json')
 assert check['protected_changes']==[] and check['independent_replay_verified']
 rows.append(dict(label=label,source=str(root/'RESULTS.json'),candidate=category,configurations=len(read(root/'INNER_TRIALS.json')),full=r['metrics'],training=r['slices']['training35'],validation=r['slices']['validation34'],matching=r['matching'],next_open=r['next_open'],next_open_validation=r['next_open_slices']['validation34'],rule=r['rule']))
previous=read(O/'monthly_score_architecture/RESULTS.json')['overall']
author=read(P/'RESULTS.json')['author']
old=read(O/'monthly_ordered_inverse/RESULTS.json')['previous_candidate']
print('Author schema',list(author),'previous schema',list(old))
feas=read(P/'FEASIBILITY.json');order=read(P/'ORDER_AUDIT.json')
changes=[f for f,h in read(P/'PROTECTED_BEFORE.json').items() if sha(f)!=h];assert not changes,changes
status=dict(created_at=datetime.now(timezone.utc).isoformat(),objective_status='not_reproduced',user_boundary='Only author-supplied holdings and confirmed author guidance are immutable. Numerical factor budgets, proxies, normalizations, gates, objectives and retention assumptions are hypotheses.',configurations=sum(r['configurations'] for r in rows),families=rows,tests_passed=23,tests_command='work/venv/bin/python -m unittest discover -s outputs/monthly_open_hypotheses -p test_openhyp_*.py',test_repairs='Corrected pandas Series alignment during pattern preparation; renamed tests to avoid existing imports with same test-module basename. No failed-run output used as successful result.',author=author,previous_replication=previous,previous_high_return=old,feasibility=dict(systems=len(feas),strictly_separable=sum(x['strict_common_score_separable'] for x in feas),scope='Common fresh-ranking linear scores in tested base/regime/hinge feature bases; covers only available author targets, training and full sets. Not a general impossibility statement.'),protected_changes=changes,order_audit={k:v for k,v in order.items() if k!='changes'},next_research='Use saved month/name contributions to identify missing condition-dependent decisions or annual forecasting targets. Do not optimize newer experiments against reused validation as if it were fresh, and do not promote a holdings lookup or year/month-specific fit.',limitations=['Original17ETF list unknown','Top32 market-cap reconstruction approximate; Top35 remains boundary sensitivity','First35/later34 split but latter history repeatedly viewed; not independent out-of-sample','Monthend final close simultaneously used for signal/fill is an idealized reproduction convention; next-open sensitivity exported','Inverse holdings calibration is not the author’s original annual market-return model','Ridge reused matured-label annual forecasts, with inherited proxy universe and gate limitations'])
(P/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
text=['月策略开放假设研究 — 2026-10-10','',
'结论：本轮完成，但尚未彻底复现，也没有刷新此前最佳复现候选。作者记录和guidance保持不变。',
'已移除继承的50/30/20/10数值预算、和为1限制、固定295天连续预热、正动量硬门槛。行业相对表现增加近期窗口；核心动量仍跳过两个月。名单顺序分别作为约束和待检验假设。',
'累计4836个配置（不是4836个独立策略），只用前35个月选择与拟合，再看后34个月。零成本、月末等权、月内固定股数。',
'全期：2020-12-31收盘—2026-09-24；训练：前35个持有月；验证：2023-11-30收盘—2026-09-24。验证历史已反复查看，不能称独立样本外。','',
'模型 | 全期年化 | 全期Sharpe | 验证年化 | 验证Sharpe | 验证组合完全匹配 | 验证顺序匹配']
for r in rows:
 f,v,mm=r['full'],r['validation'],r['matching']['validation'];text.append(f"{r['label']} | {f['cagr']:.2%} | {f['sharpe']:.3f} | {v['cagr']:.2%} | {v['sharpe']:.3f} | {mm['set_exact']}/34 | {mm['order_exact']}/34")
f=previous['metrics'];v=previous['slices']['validation34'];mm=previous['matching']['validation']
text.append(f"此前最佳复现候选 | {f['cagr']:.2%} | {f['sharpe']:.3f} | {v['cagr']:.2%} | {v['sharpe']:.3f} | {mm['set_exact']}/34 | {mm['order_exact']}/34")
text+=['','作者原持仓回放：同期全期年化93.7523%、Sharpe1.9216；后34个月年化133.6892%、Sharpe2.0917。',
'另保留以前全期年化77.6221%、Sharpe1.8573的高收益候选；它的历史选择口径与这轮前35/后34不同，不能冒充这轮严格验证结果。',
'量价＋Ridge候选：核心仍贡献最大单组评分差异，采用5%市场收益预测排名修正。前35个月组合匹配12/35，验证7/34；全69个月19组完全一致、仅2组顺序一致。不能据此称已复现。',
'新增持仓状态模型从空仓自行演化成本、峰值和持仓年龄；作者状态只用于训练前缀，验证及自主执行未读入作者前期持仓。',
'48个共同评分可分性检查：当前特征基下均不能让所有已覆盖作者目标严格领先所有未选项。这限定于测试的共同线性/交互/分段基和新选前三名规则，不排除其他非线性、额外信息或状态依赖规则。',
'名单顺序检查：相邻月留存股票对75次可比，发生25次顺序翻转；54次组合变化中仅20次符合“留存顺序保留、新股追加”。不支持把全部顺序解释成机械记账顺序，也不能证明就是评分排序。',
'股票池仍是近似：Top32与Top35敏感性分开；作者17ETF清单未知。不能通过针对某个月强行加入作者股票来宣称复现。',
'23项测试通过；四轮模拟均与独立固定股数回放一致；原数据/作者记录/观察账户保护哈希无变化。',
'修复记录：量价prepare中Series对齐错误已修复；测试文件名与旧目录冲突已改为独有前缀。失败输出未作为结果。','',
'文件：', '  STATUS.json：完整指标、来源、规则和限制', '  core_primary_MONTHS.json：第一轮逐月排名及贡献', '  listwise/core_primary_MONTHS.json：整组概率版本', '  stateful/core_primary_MONTHS.json：持仓状态版本', '  minor_layers/core_primary_MONTHS.json：量价＋Ridge逐月选股', '  */*_TRADES.json、*_DAILY.json：完整交易和净值', '  FEASIBILITY.json：可分性诊断', '  ORDER_AUDIT.json：名单顺序诊断','',
'下一步方向：针对逐月主要冲突，检验作者年度收益预测目标、条件化因子定义和股票池来源；没有证据支持继续把更高复杂度或更高开发期收益当作成功。']
(P/'REPORT.txt').write_text('\n'.join(text)+'\n')
print('Wrote report;',status['configurations'],'configurations;',len(changes),'protected changes.')

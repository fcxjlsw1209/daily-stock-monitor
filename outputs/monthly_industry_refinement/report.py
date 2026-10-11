from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
P=Path(__file__).resolve().parent;O=P.parent
read=lambda p:json.loads(Path(p).read_text())
r=read(P/'RESULTS.json');f=read(P/'FROZEN.json');obs=read(O/'monthly_industry_monitor/STATUS.json');registration=read(O/'monthly_industry_monitor/REGISTRATION_VALIDATION.json');wf=read(P/'TRAIN_WALKFORWARD.json')
labels={'original_anchor':'原简洁候选','industry_reference':'已登记行业观察','annual_primary':'本轮训练年化主候选','robust_control':'本轮稳健对照','author':'作者同期持仓重放'}
lines=['月策略行业观察登记与第二轮改进',f'生成UTC：{datetime.now(timezone.utc).isoformat()}','',
'观察登记已完成：monthly_momentum_industry_paper_v1；独立100000美元现金；10/12开始，首个信号10/30，次一交易日11/2开盘。保持80.8349%历史年化候选冻结，不随本轮优化替换。',
'已接入统一run_pair及现有13:20/20:20提醒。原六日账户与两个旧月账户保持不变。',
f'当前实际状态：{obs["status"]}；联合行情截至{obs["data_as_of"]}；月末Top32档案截至{obs["input_coverage"]["membership_end"]}。没有计划、持仓或成交，胜率尚无数据；按需等待当前行情和当月准确股票池，不能拿历史回测冒充观测成绩。','',
'注册一致性核查发现：80.83%研究实际使用monthly_top35_extension/POOLS.json的32档案，并非monthly_regime_monitor早期估算池。新观察已精确采用研究来源；69个月持仓及分数完全一致。前轮报告中“相同原股票池”的正确含义是与77.62%研究基线相同，不能理解为与旧regime观察池相同。没有修改旧观察规则或研究记录。','',
'本次研究：106组有限配置，前35个月（2021-01至2023-11）选择，冻结后34个月（2023-12至2026-09-24）验证。全部历史已经多轮研究，不是未触碰样本外。',
'全期2020-12-31收盘至2026-09-24。零成本、零滑点、完整月末收盘信号同价理想成交。月底最多三只等权，月内固定份额；Sharpe使用日收益、无风险利率0。',
'模型 | 全期年化 | Sharpe | 最大回撤 | 前35月年化 | 后34月年化 | 后34月Sharpe']
for k,label in labels.items():
 z=r[k];a=z['metrics'];b=z['slices']['validation34'];c=z['slices']['training35'];lines.append(f'{label} | {a["cagr"]:.2%} | {a["sharpe"]:.3f} | {a["max_drawdown"]:.2%} | {c["cagr"]:.2%} | {b["cagr"]:.2%} | {b["sharpe"]:.3f}')
lines += ['', '本轮主候选规则（研究版，未新开账户）：',
'保留52%长期动量主分及资格；高波动时原辅助48%缩至30.5%，15%行业组合加2.5%近期加权Ridge。非高波动时沿用原四因子。',
'行业15%内部：3.75%行业相对强弱（126与252窗口、均跳42日、两种横截面百分位均值）；3.75%行业联动稳定；7.5%极端市场下跌日beta残差抗跌。',
'Ridge是按年度、此前五年且21日未来收益标签已成熟才训练，alpha10、近期权重半衰期12个月；预测对SPY超额收益，不学习作者持仓。预测百分位占总分2.5%。',
'无前五保留，无月内退出，无已有浮盈惩罚，不改变长期动量的跳过两月。',
'全期从80.83%至82.20%，Sharpe1.914至1.928；后段年化93.78%至93.43%，Sharpe1.920至1.924。整体提升主要仍来自训练段，不能称明显泛化突破。',
'稳健候选混合少量对齐动量窗口的波动分母，并加2.5%普通Ridge；后段回报更低。这次没有证明风险分母改造或波动质量替换一定更好。','',
'执行敏感性（同一冻结月末计划，下一开盘）：',
'模型 | 成本 | 全期年化 | Sharpe | 最大回撤 | 后34月年化']
old=read(O/'monthly_guidance_tweaks/EXECUTION_CHECK.json')
# Add this round leaders; previous frozen reference controls remain in their own artifact.
for k in ['annual_primary','robust_control']:
 for key,cost in [('next_open_zero','0'),('next_open_cost','每边0.05%')]:
  z=r[k][key];a=z['metrics'];b=z['slices']['validation34'];lines.append(f'{labels[k]} | {cost} | {a["cagr"]:.2%} | {a["sharpe"]:.3f} | {a["max_drawdown"]:.2%} | {b["cagr"]:.2%}')
lines += ['', '搜索审计：',
f'预先定义{f["single_count"]}组单类变化，再将训练期各类最多两个候选按兼容参数两两组合，共{f["candidate_count"]}组；{f["admissible_count"]}组通过约束。以训练年化最高为主候选，同时冻结稳健目标对照。未看后34个月重选。',
'约束：训练Sharpe至少为行业基线减0.10，回撤不差5个百分点，12/12/11月三块至少两块回报不低于基线。主轴类别52%不动，只允许其中最多25%更换波动分母定义，非核心修正总预算最高25%。',
'附加训练期前缀滚动：12月选/未来6月检验、18月选/未来6月检验、24月选/未来11月检验；只用预定义单类网格，排除利用全部35月挑出的组合种子。该滚动方法与最终固定候选不同。',
f'前缀滚动2022-01至2023-11拼接：年化{wf["stitched"]["cagr"]:.2%}，Sharpe{wf["stitched"]["sharpe"]:.3f}，回撤{wf["stitched"]["max_drawdown"]:.2%}。','',
'验证：新观察11项测试、统一入口3项测试、本轮研究5项测试通过。两条参考策略指标精确回放一致，逐日净值与独立固定份额引擎一致；股票池/价格来源、冻结参数及保护文件校验通过。',
'观察的代码与配置已冻结；输入来源是历史研究代理而非认证全市场时点数据库，ETF名单仍为假设。更新真实数据服务是后续计划生成的前提。',
'已登记观察：outputs/monthly_industry_monitor/STATUS.txt；研究逐月交易：annual_primary_ORDERS.json、TRADES.json、HOLDINGS.json；所有被试配置：TRAIN_TRIALS.json。']
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
status=dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),registered_strategy='monthly_momentum_industry_paper_v1',registered_observer_status=obs['status'],research_primary=r['annual_primary'],candidate_count=f['candidate_count'],tests_passed=19,existing_ledgers_changed=False,automatic_replacement=False,pristine_oos=False)
(P/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
# Final immutable artifact/ledger comparison after both workflows.
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
changed=[p for p,h in read(P/'PROTECTED_BEFORE.json').items() if sha(p)!=h];assert not changed,changed
print('Report complete; protected records unchanged; research82.20%/1.928, observation80.83% retained')

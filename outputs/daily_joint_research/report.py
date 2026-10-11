# -*- coding: utf-8 -*-
"""Validate and report every run, including unsuccessful and post-diagnostic variants."""
from pathlib import Path
import json,hashlib,collections
import experiment as e

P=Path(__file__).resolve().parent
labels={
 'baseline':'原日策略', 'cost_margin':'上一轮成本门槛对照',
 'context_return':'市场／行业／残差环境特征', 'asset_split':'股票与ETF分开建模',
 'context_split':'环境特征＋分开建模', 'joint_path':'收益与涨跌路径联合评分',
 'joint_tail_gate':'联合评分＋不利路径概率门槛', 'joint_cost_gate':'联合评分＋预测收益门槛',
 'joint_split_gate':'分组联合模型＋两个门槛', 'stocks_only':'只允许股票入场',
 'horizon10':'预测目标改成十日', 'rules_only':'取消拟合模型，只用四因子',
 'half_ml':'模型评分权重减半', 'validated_base':'过去验证有效才启用原模型',
 'validated_pool':'从四个模型中按过去验证选用',
}
main=json.loads((P/'SUMMARY.json').read_text())
adaptive=json.loads((P/'adaptive/SUMMARY.json').read_text())
stable=json.loads((P/'stability/SUMMARY.json').read_text())
checks=json.loads((P/'CHECKS.json').read_text())
assert len(main)==22 and len(adaptive)==12 and len(stable)==4
assert checks['label_maturity'] and checks['logistic_nonconvergence']==0
hashes=json.loads((P/'SOURCE_HASHES.json').read_text())
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in hashes.items())
result_hashes={}
for folder,summary in [(P,main),(P/'adaptive',adaptive),(P/'stability',stable)]:
 for key,item in summary.items():
  assert item['status']=='complete'
  file=folder/(key+'.json');out=json.loads(file.read_text())
  assert out['status']=='complete' and out['performance_publishable'] is False
  assert all(v['signal_date']<v['session'] for v in out['decisions'])
  assert all(v['cash']>=-1e-6 for v in out['daily'])
  assert out['metrics']['equity']==item['ending_equity']
  result_hashes[str(file.relative_to(P))]=hashlib.sha256(file.read_bytes()).hexdigest()

def table(summary,cost,kinds):
 rows=['方案 | 全期年化 | 最大日终回撤 | 剔除最近252日年化 | 最近252日收益 | 2019—2023年化 | 平均仓位']
 for k in kinds:
  v=summary[f'{k}_cost{cost}']
  values=[v['full']['cagr'],v['full']['max_drawdown'],v['ex_recent']['cagr'],
          v['last252']['return_'],v['development']['cagr'],v['mean_exposure']]
  rows.append(labels.get(k,k)+' | '+' | '.join(f'{x:.2%}' for x in values))
 return rows

lines=[
 '日策略继续优化：联合预测、模型分组与历史有效性检查',
 '完成：2026-10-08；回测范围：2019-01-02至2026-10-06。',
 '初始10万美元，历史指数并集内可用市值Top32＋17 ETF，只做多，最多三只，少于两只空仓。',
 '四组仍各25%。原选股资格、0.75评分门槛、10日上限、ATR止盈止损及排名换股保留；只允许股票入场版本明确改变了入场池。',
 '本轮共38次执行：第一阶段11版本×2成本，诊断后第二阶段6版本×2成本，相邻验证窗口2版本×2成本。所有结果保留，未自动修改观察账户。',
 '最近252日：2025-10-06至2026-10-06；剔除最近252日后的区间截至2025-10-03。',
 '严格历史数据认证仍未完成，全部performance_publishable=false；多轮查看过的历史不能称为未触碰的样本外。',
 '',
 '实际结论',
 '新增“过去验证有效才启用原模型”值得继续保留为研究候选：全期年化21.61%，原版18.58%；剔除最近252日16.37%，原版13.30%；最近252日64.04%，原版61.46%。',
 '新增复杂联合模型没有超过原版。上一轮成本门槛版本仍是另一项有效对照：年化19.27%，回撤26.94%，但最近252日38.73%。',
 '没有找到近期100%或长期50%年化的新日策略；不通过提高杠杆、改变股票池或事后择优掩盖这一结果。',
 '',
]
for cost in [10,30]:
 lines += [f'第一阶段：买卖合计成本{cost / 100:.2f}%']
 lines += table(main,cost,json.loads((P/'CONFIG.json').read_text())['variants'])+['']
 lines += [f'第二阶段：买卖合计成本{cost / 100:.2f}%']
 lines += table(adaptive,cost,json.loads((P/'adaptive/CONFIG.json').read_text())['variants'])+['']

lines += [
 '候选的明确规则',
 '先沿用原日策略四大因子及入场／退出规则。原模型最多占总评分12.5%，属于反弹力度组的一半；其他组没有增加或减少权重。',
 '每天计算过去预测收益与实际五日收益的横截面Spearman相关性，仅在当天至少五个回撤候选都有预测和结果时计算；标签延迟六个交易日才进入验证序列。',
 '每月第一个信号日，用前一完整交易日已知的验证结果，分别检查过去126日和252日的平均相关性，两者必须都为正；有效观察数分别至少40和60。',
 '两窗口均为正时，当月继续使用原Ridge评分；否则当月完全使用四因子历史评分。月内不按后来收益重新选择。',
 '没有足够历史验证时默认不用模型。此正相关门槛是固定启发式规则，不是统计显著性检验。',
 '',
 '相邻窗口检查（看到候选结果后追加，保留原126／252日设定，不重新选最佳参数）',
 '窗口 | 成本 | 年化 | 最大日终回撤 | 剔除最近252日年化 | 最近252日收益',
]
for name,title in [('shorter','84／168日'),('candidate','126／252日'),('longer','168／336日')]:
 for cost in [10,30]:
  v=adaptive[f'validated_base_cost{cost}'] if name=='candidate' else stable[f'{name}_cost{cost}']
  values=[v['full']['cagr'],v['full']['max_drawdown'],v['ex_recent']['cagr'],v['last252']['return_']]
  lines.append(title+f' | {cost / 100:.2f}% | '+' | '.join(f'{x:.2%}' for x in values))
lines += [
 '相邻窗口的全期年化均仍高于原版，但较短窗口回撤变大。支持继续研究这个方向，不构成未来稳定增益的证明。',
 '',
 '增益来自哪里、还有什么限制',
 '2019—2023开发期，原版年化13.00%，候选13.21%，提升很小；2024以后则由29.47%升到38.53%。不能把改善描述成所有时期都明显胜出。',
 '成本提高至0.30%后，全期年化由原版13.73%升至16.26%；但开发期两者为8.12%和8.06%，候选没有在该段成本压力下胜出。',
 '平均仓位从61.25%升至65.44%，部分改善来自持仓时间变化；没有使用杠杆。胜率从55.27%至55.67%，不是依靠大幅提高胜率。',
 '模型原先约占总评分12.5%，并非整个策略。彻底去掉模型或把其权重减半都弱于原版；从多个复杂模型中自动挑选也没有改善。',
 '',
 '预测分组诊断',
 '新增环境模型在2019—2023最低至最高预测五分位的实际五日平均净收益约为0.03%、0.22%、0.14%、0.25%、0.48%；到2024以后变成0.33%、0.31%、0.35%、0.37%、0.22%。',
 '这显示其收益排序没有稳定延续。分位是事后诊断统计，五日标签存在重叠，没有用它们产生过去买单或宣称独立显著性。',
 '不利路径逻辑回归的Brier误差在2024以后也未优于按该评估期实际发生率构建的常数诊断；该常数使用事后发生率，不是可交易基准。',
 '',
 '研究及实现',
 '模型仍使用带正则化的Ridge；新增风险概率用正则化逻辑回归。环境特征包含市场回撤、滞后beta残差、行业相对涨幅、成交量变化和事前固定的交互项。',
 '每月只用过去1008个交易日的已成熟标签训练；股票与ETF模型样本不足300时退回合并模型。所有逻辑回归拟合均收敛。',
 '特征交互这一研究方向参考Gu、Kelly、Xiu的论文摘要：https://www.nber.org/papers/w25398 。该论文不验证本项目的Top32短线规则，不能用来为本次收益背书。',
 '第二阶段在看过第一阶段预测分组诊断后提出；相邻窗口检查在第二阶段候选出现后提出。三个阶段各自在运行前固定参数，仍属于同一已被反复研究的历史。',
 '',
 '逐年收益，2026仅截至10月6日',
 '年份 | 原版 | 成本门槛 | 历史验证后启用模型',
]
annual={}
for name in ['baseline','cost_margin','validated_base']:
 out=json.loads((P/'adaptive'/f'{name}_cost10.json').read_text())
 annual[name]={str(year):e.r.period_metrics(out['daily'],f'{year}-01-01',f'{year}-12-31')['return_'] for year in range(2019,2027)}
for year in range(2019,2027):lines.append(str(year)+' | '+' | '.join(f'{annual[k][str(year)]:.2%}' for k in annual))
(P/'ANNUAL_RETURNS.json').write_text(json.dumps(annual,indent=2)+'\n')
lines += [
 '', '验证',
 '38次运行全部完成；原版和上一轮成本门槛版本在两档成本下的成交订单及净值均复现。所有交易信号早于成交，模拟现金非负。',
 '六项新增测试通过：Ridge／逻辑回归数值验证、完整未来路径标签、未来价格不改变过去特征、未成熟标签不进入训练、未成熟结果不改变当前模型选择、负验证效果关闭模型。RuntimeWarning被当作错误处理。',
 '原策略实现、输入manifest和月观察冻结清单哈希保持一致；没有改变六个旧观察账户或月策略账户。',
 '初始数值检查发现本机矩阵乘法实现警告，改为显式einsum并重新通过全部测试后才运行真实回测；没有忽略数值警告。',
 '', '数据局限',
 '股票池按历史指数并集与可用的过去市值快照构建，不是完整认证的历史精确Top32。市值快照混合频率、原始披露时点、早期成员／退市、行业和公司行动仍继承此前研究限制。',
 '没有把缺失的2008—2009拼进本轮，也没有把本轮历史回测算入前向观察业绩。现金不计息，未计个人税。',
 '', '文件',
 'CONFIG.json、adaptive/CONFIG.json、stability/CONFIG.json：各阶段固定参数与先后关系。',
 'SUMMARY.json、adaptive/SUMMARY.json、stability/SUMMARY.json：所有结果；每个*_cost*.json包含逐日净值、决策、成交及平仓。',
 'CALIBRATION.json：预测分组和风险概率诊断；adaptive/CHOICES.json：每月实际选择及当时可用的验证值。',
 'TRAINING_AUDIT.json、CHECKS.json、FINAL_VALIDATION.json：训练成熟时间、复现与结果校验。',
]
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
validation=dict(completed_runs=len(result_hashes),results_sha256=result_hashes,
 code_sha256={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['experiment.py','adaptive.py','check_stability.py','report.py']},
 tests_passed=6,baseline_and_cost_margin_reproduced=True,labels_mature=True,
 logistic_nonconvergence=0,protected_sources_unchanged=True,data_certified=False)
(P/'FINAL_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
print('All 38 runs validated; report saved.')

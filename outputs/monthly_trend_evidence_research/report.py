"""Render frozen experiment outputs without any new model selection."""
from pathlib import Path
import json
import re

P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text())
r,a,fr,v=map(read,['RESULTS.json','ATTRIBUTION.json','FROZEN.json','VALIDATION.json'])
labels={
    'original65':'已登记原月策略', 'previous66':'此前最佳研究候选',
    'original65_volume_confirmation':'原策略＋长期成交额确认',
    'original65_recovery_speed':'原策略＋较短前高等待时间',
    'primary':'原策略＋边界Ridge，未扣稳定性（训练冻结主候选）',
    'original65_matched_k1':'同预算Ridge，扣1倍年份敏感度',
    'original65_matched_k2':'同预算Ridge，扣2倍年份敏感度（无换股）',
    'simple':'独立75%动量＋25%波动',
    'simple_volume_confirmation':'独立底座＋长期成交额确认',
    'simple_recovery_speed':'独立底座＋较短前高等待时间',
    'simple_block_stability':'独立底座＋边界Ridge稳定性修正',
    'qqq':'QQQ买入持有'}

lines=['月策略研究：长期成交额确认、前高等待时间、年度模型稳定性', '',
    '已完成22个事前固定配置。本轮没有提高此前最佳研究候选66.36%或已登记观察规则65.34%。没有新增/替换观察账户，也没有在查看后段结果后重新选择参数。', '',
    '共同口径：2016-08-17—2026-09-24，零成本，完整收盘信号并以同收盘价成交的理想化回测。只做多、不加杠杆，月末三只等权、两只各半、少于两只现金；月内固定复权总回报单位。初期2016-08-17非月底，最后2026-09-24为不完整月份。',
    'Sharpe按每日收益、sqrt(252)、无风险利率0；最大回撤按日终净值。股票池为继承的历史Top32估算＋自拟17ETF，并未获得作者精确名单。后34个月此前已反复研究，不能称全新样本外。', '',
    '一、全期结果',
    '版本 | 年化 | Sharpe | 最大回撤 | 下一开盘每边0.05%成本年化']
for name,label in labels.items():
    z=r[name];m=z['metrics']
    lines.append(f"{label} | {m['cagr']:.2%} | {m['sharpe']:.3f} | {m['max_drawdown']:.2%} | {z['next_open_cost']['metrics']['cagr']:.2%}")
lines+=['', '二、与作者思想的对应',
    '1. 两个手工因子都只看t−252..t−42：长期动量主轴未变，最近两个月没有进入这两个新因子。它们只是2.5%/7.5%的辅助修正，不要求当月正在回撤。',
    '成交额确认：过去210日的成交额加权log收益减普通log收益。每日权重为raw close×raw volume相对前63日均值，限制0.25到4；测上涨是否集中在交易活跃的日子。美元成交额避免原始拆股价格/成交量单位相反变化。',
    '前高等待时间：211个价格中，每天距离最近一次窗口内新高的天数，取负平均值/210。测长趋势是不是经常陷在水下；不是“今天必须超卖”的入场规则。',
    '2. 模型仍是年度过去五年Ridge，只用已实现的下一自然月收益。固定10个已有特征、alpha10、24月样本半衰期，至少36个月。每次删掉一个训练年份后重训，测两只候选预测收益差是否稳定。',
    '从当月基础前三出发，前两只保持；只有基础第4/5名且距第3名总分≤0.02才可能替换。预测差扣0/1/2倍年份敏感度后必须仍为正，且2.5%/5%小幅修正足以覆盖原分差。它是训练年份敏感度，不是经校准的置信区间。',
    '3. 月内无止盈止损、没有日策略回撤资格、没有ticker或月份硬编码。两只候选的删除年份预测差直接计算，保留预测相关性。', '',
    '三、冻结选择与关键发现']
lines.append('版本 | 早期2016—2020年化 | 训练88期年化/Sharpe | 后34期年化/Sharpe')
for name in ['original65','primary','original65_matched_k1','original65_matched_k2']:
    s=r[name]['slices'];t=s['training2016_2023'];later=s['later34']
    lines.append(f"{labels[name]} | {s['early2016_2020']['cagr']:.2%} | {t['cagr']:.2%}/{t['sharpe']:.3f} | {later['cagr']:.2%}/{later['sharpe']:.3f}")
lines += [
    '主候选由前88期训练表现选出，恰为k=0未扣稳定性的对照，不能把它说成稳定性模型胜出。训练年化仅53.4916%→53.5408%，全期却降至61.5812%。',
    '同5%预算下，k=1减少但未消除不利变更；k=2在整个122期没有一次换股，65.34%完全来自原策略。它没有提供额外收益。',
    '独立简单底座的稳定性修正37.81%→39.01%，仅来自早期3个月，后34期零换股；没有新的后段增益证据，不能转移为现有高收益规则的提升。', '',
    '四、逐月换股归因（完整明细见MONTH_COMPARISON.json/ATTRIBUTION.json）']
for name in ['original65_volume_confirmation','original65_recovery_speed','primary','original65_matched_k1','original65_matched_k2','simple_block_stability']:
    z=a[name];up=sum(c['difference']>0 for c in z['changes']);down=sum(c['difference']<0 for c in z['changes'])
    lines.append(f"{labels[name]}：{z['changed_months']}个月组合改变，{up}次改善/{down}次变差；撤销最大1/3次正贡献的事后集中度诊断年化{z['undo_best1_diagnostic_cagr']:.2%}/{z['undo_best3_diagnostic_cagr']:.2%}。")
lines += ['', '主候选全部改变月份：']
for c in a['primary']['changes']:
    display=lambda ss:','.join('META' if x=='research_equity_FB' else x for x in ss)
    lines.append(f"{c['date']}→{c['end']}：{display(c['old'])} → {display(c['new'])}；组合当月差{c['difference']*100:+.3f}个百分点。")
wf=read('WALKFORWARD.json')
lines += ['',
    '研究解释：本轮数据不支持“辅助信号越多，越能找到下一月赢家”。成交额因子17次改变中9次较好，但少数较大的损失超过其改善；较短前高等待时间也会排除随后持续上涨的强者。这里只是换股结果的诊断，不是据此针对AVGO/BABA等名称增加特殊规则。',
    f"训练内部按36/60/76期选择后的后续段拼接：年化{wf['metrics']['cagr']:.2%}、Sharpe{wf['metrics']['sharpe']:.3f}；同段原规则{wf['baseline']['cagr']:.2%}、{wf['baseline']['sharpe']:.3f}。仍未显示稳定优势。", '',
    '五、验证与限制',
    '34项测试通过：因子手算、skip42、未来扰动、常数权重、拆股美元成交额、前高龄路径、缺值、年度标签成熟、删年模型独立标准化、成对预测共变、原计划一致及选择规则。',
    '17个原合格股票—月份因额外成交额预热不足或输入缺失，维持原评分，没有剔除股票；前高等待因子在合格位置无缺失。',
    '完整Ridge预测与上一轮pooled模型精确一致；34项测试与冻结后的独立日级回放分别验证数值和交易。原65/66指标精确复现。',
    f"{v['protected_files']}个受保护文件指纹未变；无旧计划、行情、账本、策略或自动化修改。已补齐数值和排名直接依赖代码hash。",
    '同收盘价主要用于用户要求的零成本研究比较，含完整收盘信号的执行理想化。实际执行敏感性单独列出。数据代理、反复研究历史、初末残缺月份等限制仍在，不能把上述年化视为未来承诺。', '',
    '研究背景（只启发假设，不证明本实现或复制论文策略）：',
    'Lee & Swaminathan, Price Momentum and Trading Volume: https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00280',
    'Damien Challet, Sharper asset ranking from total drawdown durations: https://arxiv.org/abs/1505.01333', '',
    '复现环境：现有work/venv/bin/python，PYTHONPATH=work/monthly-ml-deps:work/monthly-decision-deps:work/research-plot-deps，OPENBLAS_NUM_THREADS=1，OMP_NUM_THREADS=1。',
    '冻结回放：python outputs/monthly_trend_evidence_research/study.py evaluate。不要覆盖冻结训练。',
    '结果：RESULTS.json；逐月股票收益贡献：MONTH_COMPARISON.json；全部22组训练结果：TRAIN_TRIALS.json；规则指纹：FROZEN.json。']
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
test=(P/'TEST_RESULTS.txt').read_text()
assert re.search(r'Ran 34 tests',test) and test.rstrip().endswith('OK')
completion=dict(completed=True,trials=fr['trials'],tests_passed=34,
    primary_cagr=r['primary']['metrics']['cagr'],primary_sharpe=r['primary']['metrics']['sharpe'],
    primary_k=fr['selected']['primary']['spec']['k'],
    beats_original65=r['primary']['metrics']['cagr']>r['original65']['metrics']['cagr'],
    beats_previous66=r['primary']['metrics']['cagr']>r['previous66']['metrics']['cagr'],
    observers_changed=False,protected_files=v['protected_files'],evaluation_kind='reused_historical_research')
(P/'COMPLETION.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(completion,ensure_ascii=False))

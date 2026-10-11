"""Summarize completed experiments without selecting or fitting another model."""
import hashlib
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent
ROOT = P.parent.parent
def read(name):
    return json.loads((P/name).read_text())
def save(name, value):
    (P/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

r = read('RESULTS.json')['runs']
best = 'prior_return_leader_top32_theme17'
monthly = [m for m in read('MONTHLY_COMPARISON.json') if m['model']==best]
daily = read('DAILY_EQUITY.json')
ret = {k:np.diff(np.log([d[k] for d in daily])) for k in ['author',best]}
diagnostics = {
    'retained_research_baseline':best,
    'selection_provenance':'Previously selected by log_later DESC on already viewed history; this replay is not independent validation.',
    'new_search_improves_prior_baseline':False,
    'new_search_trials':read('RESULTS.json')['trial_count'],
    'annual_ridge_fits':len(read('TRAINING_AUDIT.json')),
    'cagr_gap_to_author':r['author']['metrics']['cagr']-r[best]['metrics']['cagr'],
    'same_rule_etf_list_cagr_difference':r[best]['metrics']['cagr']-r['prior_return_leader_top32_core17']['metrics']['cagr'],
    'same_rule_top32_minus_top35_cagr':r[best]['metrics']['cagr']-r['prior_return_leader_top35_theme17']['metrics']['cagr'],
    'daily_log_return_correlation':float(np.corrcoef(ret['author'],ret[best])[0,1]),
    'worst_relative_months':sorted(monthly,key=lambda m:m['log_relative_return'])[:10],
    'best_relative_months':sorted(monthly,key=lambda m:m['log_relative_return'],reverse=True)[:10],
    'limits':['ETF17 unknown; both sets are hypotheses.','Historical ranks reconstructed, not a complete contemporaneous market-cap vintage.','Same-close replay is descriptive; next-open is separately tested.','All periods have been examined before; no pristine out-of-sample result.','Author ledger treated as supplied simulated holdings, not independently verified broker records.'],
}
save('DIAGNOSTICS.json',diagnostics)
protected=read('PROTECTED_BEFORE.json')
changed=[str(path) for path,h in protected.items() if not (ROOT/path).exists() or sha(ROOT/path)!=h]
assert not changed,changed
validation=read('REPLAY_VALIDATION.json')
validation.update(protected_files_rechecked=len(protected),tests_passed=8,
    tests_command="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 work/venv/bin/python -m unittest discover -s outputs/monthly_guidance_reset -p 'test_*.py' -v",
    report_generated_from_saved_results_only=True)
save('VALIDATION.json',validation)
save('STATUS.json',{'status':'bounded_study_completed','author_strategy_replicated':False,
    'author_performance_matched':False,'new_rule_promoted':False,'research_baseline':best,
    'remaining':'Unknown ETF list and author training/feature definitions; no further fitting after the later21 check.',
    'protected_accounts_unchanged':True})
keys=['author','baseline_top32','baseline_top35',best,'prior_return_leader_top35_theme17',
      'prior_return_leader_top32_core17','return_selected_top35','match_selected_top35']
labels=['作者持仓回放','原简洁 / Top32 / core17','原简洁 / Top35 / core17',
        '旧收益候选 / Top32 / theme17','旧收益候选 / Top35 / theme17',
        '旧收益候选 / Top32 / core17','本轮收益候选 / Top35 / core17','本轮匹配候选 / Top35 / core17']
lines=['月策略 guidance 重建研究：结果与决策','',
 '结论：本轮从 guidance 重建、年度 Ridge 和逆向偏好拟合，没有超过此前简洁候选。将更早已经冻结的收益候选迁移到最新股票池后，得到更强的研究基线；这不是本轮新发现的规则，也不是作者交易的完整复现。',
 '', '比较区间：2020-12-31 收盘至 2026-09-24 收盘，69个持仓期，全部零交易成本。',
 '日收益 Sharpe = 均值 / 样本标准差 × sqrt(252)，无风险利率设为0；最大回撤按日终净值。金额从10万美元开始。',
 '作者持仓来自用户提供的学习账本，以下为同一行情上的回放。股票池采用现有全美上市公司月度Top32/35重建；历史市值快照并非严格完整的当时可知版本。',
 '', '同收盘成交回放：','候选 | CAGR | Sharpe | 最大回撤 | 完整匹配月份 | 匹配股票槽位']
for k,label in zip(keys,labels):
    m=r[k]['metrics'];lines.append(f"{label} | {m['cagr']:.2%} | {m['sharpe']:.3f} | {m['max_drawdown']:.2%} | {m['exact_months']}/69 | {m['shared_slots']}/207")
lines += ['', '可执行时点敏感性（仍为零成本）：']
for k,m in read('PRIOR_CANDIDATE_EXECUTION.json').items():
    lines.append(f"{k}: CAGR {m['cagr']:.2%}, Sharpe {m['sharpe']:.3f}, 最大回撤 {m['max_drawdown']:.2%}。")
lines += ['', '冻结旧收益候选的明确规则：',
 '1. 月末股票Top32（另测Top35）与theme17 ETF并集，至少295个连续有效收盘价；252至42交易日前动量为正。选前3名等权、2名各半、少于2名现金。每月调仓后固定股数。',
 '2. 因子在当月完整候选池内做居中百分位排名，然后执行资格筛选。',
 '3. 52%：P[t-42]/P[t-294]-1 除以最近63日收益波动率。',
 '4. 21.9375%：最近126日波动率（正向偏好）。',
 '5. 18.5625%：跳过最近42日的动量加速，即 P[t-42]/P[t-126] - P[t-126]/P[t-210]。',
 '6. 7.5%：负CMF63排名乘以市场波动状态。SPY最近63日波动率高于其过去252日中位数时状态为+1，否则为-1。',
 '7. ETF假设theme17：SPY QQQ IWM TLT GLD XLE XLF XLI XLK XLV XLY SMH SOXX IGV VGT MTUM ARKK。它不是已确认的作者名单。',
 '8. 该规则的原筛选标准是 log_later DESC，即曾使用后段历史收益择优。不能把现在的77.62%解释成独立验证成功；也不能因本轮未改规则就抹去原选择偏差。',
 '', '本轮实际做了什么：',
 '预先冻结研究计划；用前48个月校准，在冻结后查看后21个月，但后21个月也已在过往研究中见过，并非干净样本外。',
 f"共记录 {diagnostics['new_search_trials']} 项试验：动量定义和归一化、支持因子、有限量价/状态修正、年度五年训练Ridge、小权重预测，以及受限作者偏好逆向拟合。", 
 '年度Ridge共72次拟合。每年只使用此前五年且21日未来收益标签已经成熟的训练样本。年度模型不使用作者持仓标签；逆向偏好轨道使用前48个月持仓并单独标记。',
 '新收益候选年化61.90%、Sharpe1.50；更贴近持仓的候选年化48.08%、Sharpe1.16。更多匹配不能自动换来更高收益。没有按后21个月结果继续扩展搜索。',
 '', '现在最有解释力的差异：',
 f"同一旧规则，Top32下core17换成theme17，CAGR相差 {diagnostics['same_rule_etf_list_cagr_difference']*100:.2f} 个百分点；相同theme17下Top32和Top35仅差 {diagnostics['same_rule_top32_minus_top35_cagr']*100:.2f} 个百分点。ETF差异应作为未确定的模型设定，不能归功于新增因子。", 
 '2021：原简洁Top32收益8.03%，旧收益候选66.89%，作者90.11%。2024及2026仍有明显差距。',
 '与作者收益差距最大的月份（不是按这些月份定制规则）：']
for a in diagnostics['worst_relative_months'][:5]:
    lines.append(f"{a['date']} 至 {a['end']}: 候选 {'/'.join(a['selected'])} {a['return_']:.2%}；作者 {'/'.join(a['author'])} {a['author_return']:.2%}。")
lines += ['', '决定：保留旧冻结候选作为研究比较基线；不把本轮新模型加入观察账户。后续若继续研究，优先分辨ETF集合、训练目标与动量定义的差异，而不是追逐单月作者股票或持续在同一后段挑高收益参数。',
 '未达成：没有复现作者全部交易，也没有达到作者同期93.75% CAGR。77.62%仅为历史筛选候选回放。',
 '', '验证：8项针对性测试通过；固定股数独立净值核对通过；作者、原简洁候选、旧收益候选旧股票池均逐日复现原存档；823个受保护文件校验和未变。',
 '相关文件：PLAN.json、CONTROL_AMENDMENT.json、TRIALS.sqlite、FROZEN_RULES.json、RESULTS.json、DIAGNOSTICS.json、TRAINING_AUDIT.json、VALIDATION.json，以及逐日净值、逐月持仓和交易导出。']
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
save('CODE_HASHES.json',{f.name:sha(f) for f in sorted(P.glob('*.py'))})
print(json.dumps({'report':str(P/'REPORT.txt'),'protected_files_unchanged':len(protected),'trials':diagnostics['new_search_trials']},ensure_ascii=False))

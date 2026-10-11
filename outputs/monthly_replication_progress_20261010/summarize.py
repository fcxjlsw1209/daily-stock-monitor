from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
P=Path(__file__).resolve().parent;O=P.parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
experiments=['monthly_adaptive_core','monthly_causal_ridge','monthly_core_etf_control','monthly_score_architecture','monthly_close_exit_recheck','monthly_regime_tree_inverse','monthly_pattern_recheck','monthly_budget_sensitivity']
rows=[];counts={};checks={}
for ex in experiments:
 root=O/ex;res=read(root/'RESULTS.json');validation=read(root/'VALIDATION.json');counts[ex]=len(read(root/'INNER_TRIALS.json'))
 assert not validation['protected_changes']
 sources=read(root/'SOURCE_HASHES.json');assert all(sha(f)==h for f,h in sources.items()),ex
 protected=read(root/'PROTECTED_BEFORE.json');assert all(sha(f)==h for f,h in protected.items()),ex
 checks[ex]=dict(validated=True,protected_files=len(protected),result_sha256=sha(root/'RESULTS.json'),frozen_sha256=sha(root/'FROZEN.json'))
 for name,r in res.items():
  if name in ['author','qqq']:continue
  rows.append(dict(experiment=ex,category=name,full=r['metrics'],training=r['slices']['training35'],validation=r['slices']['validation34'],matching=r['matching'],source=str(root/'RESULTS.json')))
author=read(O/'monthly_supplied_ledger_audit/AUDIT.json');ref=read(O/'monthly_style_inverse/RESULTS.json');old=read(O/'monthly_ordered_inverse/RESULTS.json')['previous_candidate']
lead=read(O/'monthly_score_architecture/RESULTS.json')['overall'];r=lead['rule'];groups={g:float(sum(w for (_,_,kind),w in zip(r['rule']['spec'],r['rule']['base']) if kind==g)) for g in ['core','vol','industry','shape','accel']}
summary=dict(created_at=datetime.now(timezone.utc).isoformat(),objective_status='not_reproduced',
 acceptance='A common guidance-consistent rule recovers all69 monthly ordered three-name holdings and changes, with returns/accounting independently reconciled. Lookup tables and per-month ex-post fits do not count.',
 actual_work=dict(candidate_policy_configurations=sum(counts.values()),counts=counts,meaningful_tests_passed=39,tests_note='5adaptive+5Ridge+5ETF-control+6architecture+8position-engine+3tree+3patterns+4budget; repaired initial import/mask errors before final runs.'),
 execution=dict(full_start='2020-12-31',full_end='2026-09-24',training_months=35,training_end='2023-11-30',validation_months=34,validation_start='2023-11-30',zero_cost=True,fixed_shares_intramonth=True,same_close_idealization=True,validation_not_pristine=True),
 comparisons=rows,author=dict(full=author['replay'],slices=ref['author']['slices']),previous_return_candidate=old,
 notable_candidate=dict(experiment='monthly_score_architecture',category='overall',reason='Highest later34 exact-set count among this batch frozen candidates; descriptive comparison, not a new validation-selected promoted strategy.',full=lead['metrics'],validation=lead['slices']['validation34'],matching=lead['matching'],rule=r,group_weights=groups),
 known_replay=dict(monthly_returns_matching_display_precision=author['return_rounding_matches'],max_return_error_bp=author['max_return_error_bp'],final_accounting_difference_usd=author['replay_accounting_minus_supplied']['ending'],qqq_final_period_rounding_unresolved=True),
 unresolved=['Top32 excludes5author slots; uniform Top35 resolves4, COST2022-05-31 remains proxy rank36. Original historical capitalizations are not known.', 'Author complete17ETF list, precise factor transformations and ML training target/coefficients unknown.', 'Many author combinations and orderings still disagree even when all three names are eligible; missing universe is not the sole explanation.', 'Author order-as-rank is a user hypothesis, not established by the supplied ledger. It remains the main target; no silent relaxation.', 'Known holdings replay matches69 rounded monthly returns but final dollars differ64.43 and QQQ last-period comparison remains unresolved.'],
 findings=['Annual causal market-return Ridge received effectively0 scoring weight in selected fit; not evidence of improved reconstruction.', 'Alternate ETF17 list, market-state trees, anti-chasing price-volume patterns and weight-cap relaxation did not materially improve heldout matching.', 'Fixed/adaptive monthend exit policies did not improve validation; no evidence to introduce intramonth trading.', 'Numeric weight caps were research assumptions; sensitivity explicitly separated long-core50 and dominant40 controls.'],
 next_work='Resolve exact momentum/risk transformation and original universe before further similar weight grids. Keep every failed trial logged; do not retrofit validation or promote an inverse fit.',checks=checks,observation_accounts_unchanged=True)
(P/'STATUS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
v=lead['slices']['validation34'];f=lead['metrics'];av=ref['author']['slices']['validation34']
lines=[
'月策略复现进展：尚未达到完整复现',
'本轮完成4308个候选配置（含对照与重复行为），39项相关测试通过；原观察账户、冻结计划和源数据校验和未变。',
'口径：前35个月训练，后34个月验证。零交易成本、月末收盘理想成交、月内固定股数。验证段此前已经反复查看，不是独立样本外。',
'',
'全期 2020-12-31 至 2026-09-24：',
f"作者已知持仓回放：年化{author['replay']['cagr']:.2%}，Sharpe{author['replay']['daily_sharpe_rf0_sqrt252_ddof1']:.3f}。",
f"此前高收益简洁候选：年化{old['metrics']['cagr']:.2%}，Sharpe{old['metrics']['sharpe']:.3f}，完整组合9/69个月一致；保持原样。",
f"本轮动量资格＋排序候选：年化{f['cagr']:.2%}，Sharpe{f['sharpe']:.3f}，完整组合19/69个月一致，完整顺序1/69。",
'',
'后34个月 2023-11-30 至 2026-09-24：',
f"作者：年化{av['cagr']:.2%}，Sharpe{av['sharpe']:.3f}。",
f"动量资格＋排序：年化{v['cagr']:.2%}，Sharpe{v['sharpe']:.3f}，66/102个持仓名额一致，8/34个月三只全对，0/34个月完整顺序一致。",
'该候选在每轮训练内部冻结后才计算后34个月结果；这里的跨轮对照是事后研究摘要，不用于自动推广。',
'',
'新增假说的实际结论：年度Ridge、ETF名单对照、额外量价形态、市场状态树、月末退出、研究权重限制放宽均已跑完。没有找到可以解释全部月份的共同规则。',
'重点候选仍采用Top35研究对照、跳过最近两个月的风险调整动量、动量前半资格和月末评分保留；不是宣称恢复了作者Top32或原算法。',
'“选到同样的组合”和“收益接近”分别评估，任何一项都不能冒充彻底复现。',
'',
'剩余实质问题：',
'1. 原始Top32和ETF17名单未确定：Top35可覆盖4个边界缺口，但2022-05-31 COST仍为代理排名36。',
'2. 全部作者股票可用的月份仍存在大量组合、次序和换仓差异，不能只归因于股票池。',
'3. 动量/波动公式、标准化、作者模型训练目标和排序细则没有原始定义。当前都是可检验假说。',
'4. 作者69个月收益均可在两位小数显示精度内回放；最终美元仍差64.43，QQQ最后一期显示精度差异未完全核对。',
'',
'详细逐月对照：'+str(O/'monthly_score_architecture/MONTHLY_EXPLANATIONS.txt'),
'所有候选和统计：'+str(P/'STATUS.json'),
'本轮没有新增观察账户，也没有替换原策略或回填交易。']
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
print('Saved',P/'REPORT.txt');print('Configurations',sum(counts.values()),'tests39; protection verified for',len(checks),'experiments')

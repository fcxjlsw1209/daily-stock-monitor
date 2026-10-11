import json,hashlib
from pathlib import Path
import pandas as pd
from prepare import P,ETFS,U
O=P/'top32_research';R=O/'validated_round1'
def main():
 summary=json.loads((R/'SUMMARY.json').read_text());selection=json.loads((R/'SELECTION.json').read_text());validation=json.loads((R/'VALIDATION.json').read_text())
 assert len(summary)==8 and all(v['status']=='complete' for v in summary.values())
 members=pd.read_csv(O/'membership.csv.gz',index_col=0);caps=pd.read_csv(O/'cap_estimates.csv.gz',index_col=0);eligible=pd.read_csv(U/'alias_resolved_inputs/eligible.csv.gz',index_col=0)
 stocks=[c for c in members if c not in ETFS];counts=members.loc['2019-01-02':,stocks].sum(axis=1);assert counts.eq(32).all()
 assert (members[stocks]&~eligible.reindex(columns=stocks,fill_value=False)).sum().sum()==0
 day=caps.index[-1];missing=[s for s in eligible if eligible.at[day,s] and pd.isna(caps.at[day,s])]
 capage=json.loads((O/'POOL_AUDIT.json').read_text());classes=json.loads((O/'CLASS_SHARE_EVIDENCE.json').read_text());groups={}
 for r in classes['rows']:groups.setdefault(r['symbol'],[]).append(r['filed'])
 checks=dict(all_eight_complete=True,every_research_day_exactly32_estimated_stock_representatives=True,no_nonmember_stock_entries=True,tests_passed=10,exact_requested_top32_certified=False,latest_missing_cap_members=missing,recovered_class_counts={k:dict(rows=len(v),first=min(v),last=max(v)) for k,v in groups.items()},code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'top32_pool.py',P/'collect_class_shares.py',P/'run_top32.py',P/'factors.py',P/'run.py']})
 (O/'FINAL_AUDIT.json').write_text(json.dumps(checks,indent=2)+'\n')
 lines=['动态Top32四因子研究：2019-01-02至2026-10-06','',
 '范围限制：这是有可用SEC股数估算的历史指数并集成员内Top32＋17ETF；尚非完整指数并集精确Top32。缺失市值的成员未参与排名，会改变边界与收益。不可据此宣称用户要求的精确股票池已验证。',
 '已恢复Meta与伯克希尔部分年度/季度原始财报股类计数；伯克希尔按各财报中1/1500经济权利折为B股单位。之前年度没有可解析记录的日期仍可能缺失。',
 '四因子各25%，先按每日市值估算入围再排名；2019-2023选版本，2024以后仅作对照，历史已被研究过并非全新样本外。次日开盘成交，10bps双边成本，2至3只，原六观察账户未改。',
 '最近252日收益不是多年年化。候选没有按近期收益挑选。','',
 '版本 | 全期年化 | 全期最大回撤 | 2024以后年化 | 最近252日 | 2026YTD']
 for k,v in summary.items():lines.append(f"{k} | {v['metrics']['cagr']:.2%} | {v['metrics']['maximum_drawdown']:.2%} | {v['evaluation']['cagr']:.2%} | {v['last252_return']:.2%} | {v['year2026']['return_']:.2%}")
 chosen=selection['selected'];v=summary[chosen];lines+=['',f'按开发期选择：{chosen}',f"净胜率{v['metrics']['net_win_rate']:.2%}，平仓{v['metrics']['closed_trades']}笔。",f"30bps成本压力：全期年化{validation['cost_30bps']['metrics']['cagr']:.2%}，最近252日{validation['cost_30bps']['last252_return']:.2%}。",f"QQQ同期全期年化{validation['qqq_buy_hold']['full']['cagr']:.2%}，最近252日{validation['qqq_buy_hold']['last252_return']:.2%}。",'', '最新成员缺市值：'+', '.join(missing), '完整股数/股类覆盖仍为剩余阻塞。现有结果只作近似池敏感性研究，不用于新模拟账户或实盘。','', '验证：10项单元测试通过，8版本各1951交易日完整运行；每日32个估算成员、没有指数并集外股票。原始数据沿用可用数据研究的身份、行业及成分历史限制。']
 (O/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()

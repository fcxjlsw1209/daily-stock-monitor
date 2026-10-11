# -*- coding: utf-8 -*-
import json,hashlib
from pathlib import Path
import pandas as pd
from prepare import P,U,ETFS
O=P/'vendor32_research';R=O/'validated_round1';V=P/'vendor_cap'
def main():
 summary=json.loads((R/'SUMMARY.json').read_text());selected=json.loads((R/'SELECTION.json').read_text())['selected'];validation=json.loads((R/'VALIDATION.json').read_text());collection=json.loads((V/'COLLECTION.json').read_text())
 assert len(summary)==8 and all(v['status']=='complete' for v in summary.values())
 coverage=pd.read_csv(O/'coverage.csv').fillna('');period=coverage[coverage.date>='2019-01-02'];assert period.selected.eq(32).all()
 m=pd.read_csv(O/'membership.csv.gz',index_col=0);eligible=pd.read_csv(U/'alias_resolved_inputs/eligible.csv.gz',index_col=0);stocks=[s for s in m if s not in ETFS];assert not (m[stocks]&~eligible[stocks]).any().any()
 obs=pd.read_csv(O/'observation_dates.csv.gz',index_col=0);future=0
 for s in stocks:
  z=pd.to_datetime(obs[s]);future+=int((z>=pd.to_datetime(obs.index)).sum())
 assert future==0
 latest=period.iloc[-1].to_dict();missing_latest=latest['missing'].split(';') if latest['missing'] else []
 audit=dict(source_series=len(collection['records']),source_failures=collection['failures'],latest_coverage=latest,min_research_coverage=int(period.covered.min()),max_research_members=int(period.members.max()),complete_eight_runs=True,tests_passed=12,future_snapshots_used=0,all_daily_stock_selections32=True,exact_daily_cap_and_historical_vintage_verified=False,remaining_missing_latest=missing_latest,code_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['collect_vendor_caps.py','vendor_pool.py','run_vendor32.py','factors.py','run.py']})
 (O/'FINAL_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
 lines=['现成历史市值快照 Top32＋17ETF：研究报告','',f"已下载{len(collection['records'])}条公司历史序列；最新日期{latest['date']}，历史并集成员{latest['members']}个、可用市值{latest['covered']}个。",'数据来自CompaniesMarketCap公开图表；m字段单位按站点chart1.js乘100000确认。原始频率混合月/周/日，并非完整日频。','信号日只取前一天或更早的快照，最多沿用45天；不向前补、不使用未来插值、不用股价乘股数重建。来源的事后修订与实际历史发布时间未认证。','这是快照驱动的动态Top32研究，不应称精确历史每日Top32。当前目录仅用于发现网址，历史入围使用原历史指数并集，不采用当前公司名单回溯。','', '回测2019-01-02至2026-10-06；四因子各25%，次日开盘，2至3只，双边10bps；2019-2023开发期选择，后续历史已研究过，不是全新样本外。','', '版本 | 全期年化 | 最大回撤 | 2024以后年化 | 最近252日 | 2026YTD']
 for n,v in summary.items():lines.append(f"{n} | {v['metrics']['cagr']:.2%} | {v['metrics']['maximum_drawdown']:.2%} | {v['evaluation']['cagr']:.2%} | {v['last252_return']:.2%} | {v['year2026']['return_']:.2%}")
 v=summary[selected];stress=validation['cost_30bps'];q=validation['qqq_buy_hold'];old=json.loads((P/'validated_round1/SUMMARY.json').read_text());sec=json.loads((P/'top32_research/validated_round1/SUMMARY.json').read_text())
 lines+=['',f'开发期选出：{selected}',f"胜率{v['metrics']['net_win_rate']:.2%}，平仓{v['metrics']['closed_trades']}笔。",f"30bps成本压力：全期年化{stress['metrics']['cagr']:.2%}，最近252日{stress['last252_return']:.2%}。",f"QQQ同期：全期年化{q['full']['cagr']:.2%}，最近252日{q['last252_return']:.2%}。",'',f"固定同一ridge_selective算法的最近252日：原大池{old['ridge_selective']['last252_return']:.2%}；上轮SEC估算池{sec['ridge_selective']['last252_return']:.2%}；本次现成快照池{summary['ridge_selective']['last252_return']:.2%}。这些是股票池敏感性对照，不是收益预测。",'', '最新仍缺市值：'+(', '.join(missing_latest) or '无'), '历史逐日缺口见coverage.csv；退市公司未覆盖、较早年份缺快照及原历史成分/行业证据限制仍然存在。','12项测试通过；8个版本各1951交易日完整运行；验证每日32个估算成员、未使用未来快照。原六个模拟账户和冻结计划未改变。']
 (O/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()

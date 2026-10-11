"""Identification experiment, not a trading strategy or author-holdings claim.
Take holdings from our own replay and synthesize exact equal-weight fixed-share
adjusted-price paths. Compare exact data with screenshot-like downsampling and
pixel quantization. Every selected month is disclosed.
"""
from pathlib import Path
from itertools import combinations
import pandas as pd,numpy as np,json,hashlib,sys
P=Path(__file__).resolve().parent;O=P.parent
MONTHS=['2017-06','2018-02','2020-03','2020-11','2021-01','2022-06','2023-11','2024-01','2024-07','2025-03','2026-05','2026-08']

def solve(paths,observed,positions):
 rows=[]
 for k in [2,3]:
  for combo in combinations(range(paths.shape[1]),k):rows.append(combo)
 # Include cash; all investment candidates enter equally weighted at the month end.
 fit=np.stack([paths[:,x].mean(axis=1) for x in rows]+[np.ones(len(paths))]);combos=rows+[()]
 residual=np.log(fit[:,positions])-np.log(observed[positions])[None,:]
 err=np.sqrt((residual**2).mean(axis=1));order=np.argsort(err,kind='stable')
 return combos,err,order

def main():
 sources=[O/'index_union/alias_resolved_inputs/adj.csv.gz',O/'daily_rebuild_research/price_scaled_membership.csv.gz',O/'daily_rebuild_research/diagnostics/runs/monthly_scaled_same_close_diagnostic_cost0.json']
 px=pd.read_csv(sources[0],index_col=0,parse_dates=True)
 for r in json.loads((O/'four_factor_rebound/ETF_SOURCES.json').read_text()):
  p=Path(r['file']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];q=pd.read_csv(p,index_col=0,parse_dates=True);px[r['symbol']]=q['Adj Close'].reindex(px.index);sources.append(p)
 membership=pd.read_csv(sources[1],index_col=0,parse_dates=True);replay=json.loads(sources[2].read_text());orders=pd.DataFrame(replay['orders']);daily=pd.DataFrame(replay['daily']).set_index('date')
 logpixel=np.log(3000)/318;results=[]
 config=dict(months=MONTHS,provenance='Known holdings from OUR monthly replay, synthetic equal-initial-weight adjusted-price paths. Not extracted author holdings.',candidate_rule='Existing historical index union Top32 +17ETF membership at entry date. Complete month prices needed only for retrospective identification; never a prospective entry condition.',image_approx=dict(width_px=1280,time_axis_px=1046,wealth_panel_height_px=318,wealth_range=[1,3000],trading_sessions=2540,log_wealth_error_per_pixel=logpixel),limitations=['Author universe and17ETFs not known','Adjusted-price path approximates dividend reinvestment, unlike cash dividends in live share accounting','Quantization experiment assumes dates and chart calibration known; actual screenshot can be worse','No annual-return optimization, no inferred holdings used as executable backtest picks'])
 (P/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
 for month in MONTHS:
  period=pd.Period(month,freq='M');days=px.index[px.index.to_period('M')==period];start=px.index[px.index.get_loc(days[0])-1];end=days[-1]
  truth=sorted(orders.loc[(orders.date==str(start.date()))&(orders.side=='buy'),'symbol'].tolist())
  if len(truth) not in [2,3]:raise ValueError((month,truth))
  frame=px.loc[start:end];eligible=membership.loc[start];cols=sorted([s for s in px if bool(eligible.get(s,False)) and frame[s].notna().all() and frame[s].gt(0).all()])
  assert set(truth)<=set(cols)
  paths=(frame[cols]/frame[cols].iloc[0]).to_numpy();indices=tuple(cols.index(s) for s in truth);target=paths[:,indices].mean(axis=1)
  allpositions=np.arange(1,len(frame));combos,err,order=solve(paths,target,allpositions);truth_i=combos.index(indices);assert set(combos[order[0]])==set(indices)
  # Screenshot has ~8-9 horizontal pixels per month. Offset uses the known
  # portfolio's initial wealth solely to model global log-axis quantization.
  wealth0=float(daily.at[str(start.date()),'equity'])/100000
  levels=np.round(np.log(wealth0*target)/logpixel)*logpixel
  coarse=np.exp(levels-levels[0]);n=max(2,round((end-start).days/(365.25*10.103/1046)))
  positions=np.unique(np.rint(np.linspace(1,len(frame)-1,n)).astype(int));cc,ce,co=solve(paths,coarse,positions)
  consistent=int((ce<=logpixel).sum());best=cc[co[0]]
  results.append(dict(month=month,start=str(start.date()),end=str(end.date()),candidate_symbols=len(cols),combinations=len(combos),known_truth=truth,exact_best=[cols[i] for i in combos[order[0]]],exact_error=float(err[order[0]]),screenshot_points=len(positions),coarse_best=[cols[i] for i in best],coarse_recovers_truth=set(best)==set(indices),truth_rank_under_coarse=int(np.flatnonzero(co==truth_i)[0])+1,candidates_within_one_log_pixel_RMSE=consistent,coarse_best_error=float(ce[co[0]]),truth_coarse_error=float(ce[truth_i])))
 summary=dict(experiments=len(results),exact_recovery=sum(set(r['known_truth'])==set(r['exact_best']) for r in results),coarse_recovery=sum(r['coarse_recovers_truth'] for r in results),median_candidates_within_one_pixel=float(np.median([r['candidates_within_one_log_pixel_RMSE'] for r in results])),log_pixel_fraction=float(np.exp(logpixel)-1),conclusion='Monthly inverse search is computationally feasible; the screenshot alone generally cannot uniquely identify exact holdings. Demonstration on own synthetic paths, not recovered author trades.')
 (P/'RESULTS.json').write_text(json.dumps(dict(summary=summary,months=results),indent=2)+'\n');(P/'SOURCE_HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
 lines=['月策略持仓反向识别：可行性小试验','',f"12个月，在口径完全一致的精确净值条件下找回{summary['exact_recovery']}/12组真实测试持仓；降到近似截图精度后第一名找回{summary['coarse_recovery']}/12组。每月误差落在一像素RMSE以内的候选组合数中位数为{summary['median_candidates_within_one_pixel']:.0f}。",'注意：使用的是我们自己的已知持仓及合成价格路径，没有识别出截图作者的实际持仓。这个实验验证工具的可识别性，不是策略收益回测。',f"1280像素宽截图，十年价格图区约1046像素，每月约8—9个横向点。对数净值1—3000倍压在约318像素，纵向一像素约{summary['log_pixel_fraction']:.2%}净值变化。数值仅为图像分辨率量级估计。",'', '月 | 已知持仓 | 降精度第一候选 | 正确组合名次 | 一像素内组合数']
 for r in results:lines.append(f"{r['month']} | {','.join(r['known_truth'])} | {','.join(r['coarse_best']) or 'cash'} | {r['truth_rank_under_coarse']} | {r['candidates_within_one_log_pixel_RMSE']}")
 lines += ['', '后续优先次序：先从紫色净值曲线、净值/QQQ比值和回撤三幅图交叉提取月内约束，保留多组候选与误差区间；若有逐日净值或逐月放大图，显著提升识别力。用高置信候选月份的入选/退出模式反推月末可计算的动量与过滤规则，再到未参与反推的月份验证。', '某月利用月内已发生收益拟合出的最像组合仅供事后归因，不能作为那个月事前能选出的组合，也不能据此声称有80%年化可交易策略。', '读数和估计不确定时不指认具体股票。不改变现有观察账户。']
 (P/'REPORT.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()

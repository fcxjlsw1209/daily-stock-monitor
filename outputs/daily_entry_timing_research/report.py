"""Verify every finite trial and emit the full research record, including failures."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
import numpy as np
import pandas as pd
import timing as t
P=t.P
NAMES={'fixed10':'固定分差0.10对照','stable_control':'上一轮动态门槛对照','armed_up3':'回撤后等上涨且收在上半区','armed_break3':'回撤后等收盘突破前日高点','hybrid_recovery':'即时买入＋延迟确认补位','intraday_gate':'额外要求三日盘中卖压','intraday_shape':'盘中卖压替代半个形态因子','shock_pause2':'异常放量低开后暂停两日','stop_pause3':'止损后暂停重买三日','fast_highvol':'高波动标的五日到期','shock_days1':'异常低开后暂停一日','shock_days3':'异常低开后暂停三日','stop_pause2':'止损后暂停两日','stop_pause4':'止损后暂停四日','shock_plus_stop':'异常低开＋止损双暂停','shock_fixed10':'异常低开暂停＋固定分差0.10'}

def main():
 summaries={};loaded={};hashes={}
 for folder,count in [('',20),('robustness',21)]:
  where=P/folder;s=json.loads((where/'SUMMARY.json').read_text());check=json.loads((where/'CHECKS.json').read_text())
  assert len(s)==count and check['complete'];summaries[folder]=s
  for key,metric in s.items():
   file=where/(key+'.json');x=json.loads(file.read_text());assert x['status']=='complete' and not x['performance_publishable']
   assert x['metrics']['equity']==metric['ending_equity']
   assert all(v['signal_date']<v['session'] for v in x['decisions'])
   assert all(v['cash']>=-1e-6 and v['positions']<=3 and v['exposure']<=1+1e-8 for v in x['daily'])
   for dec in x['decisions']:
    pairs=dec.get('replacement_pairs',[]);assert len({v['buy'] for v in pairs})==len(pairs)
   relative=str(file.relative_to(P));loaded[relative]=x;hashes[relative]=hashlib.sha256(file.read_bytes()).hexdigest()
 protected=json.loads((P/'SOURCE_HASHES.json').read_text());assert protected==t.protected_hashes()
 selection=json.loads((P/'SELECTION.json').read_text());s=summaries[''];rb=summaries['robustness']
 kinds=['stable_control','shock_pause2','stop_pause3'];annual={};concentration={};uncertainty={}
 for k in kinds:
  x=loaded[k+'_cost10.json'];annual[k]={str(y):t.r.period_metrics(x['daily'],f'{y}-01-01',f'{y}-12-31')['return_'] for y in range(2019,2027)}
  frame=pd.DataFrame(x['daily']).set_index('date');z=np.log1p(frame.loc['2025-10-06':,'return']);best=z.nlargest(10)
  concentration[k]=dict(actual=float(np.expm1(z.sum())),without_best5=float(np.expm1(z.sum()-best.iloc[:5].sum())),without_best10=float(np.expm1(z.sum()-best.sum())),best5_dates=list(best.index[:5]),description='Hindsight diagnostic only; not an executable alternative portfolio or expected return.')
 base=pd.DataFrame(loaded['stable_control_cost10.json']['daily']).set_index('date')
 for k in kinds[1:]:
  x=pd.DataFrame(loaded[k+'_cost10.json']['daily']).set_index('date');assert x.index.equals(base.index)
  difference=np.log1p(x['return'])-np.log1p(base['return'])
  for label,start,end in [('full','2019-01-02','2026-10-06'),('development','2019-01-02','2023-12-31')]:
   v=difference.loc[start:end].to_numpy();n=len(v);rng=np.random.RandomState(818);starts=rng.randint(0,n,(2000,(n+19)//20))
   sampled=v[(starts[:,:,None]+np.arange(20))%n].reshape(2000,-1)[:,:n].mean(1)
   low,high=np.expm1(np.quantile(sampled,[.025,.975])*252)
   uncertainty[k+'_'+label]=dict(annualized_growth_ratio_minus1=float(np.expm1(v.mean()*252)),lower95=float(low),upper95=float(high),block_sessions=20,resamples=2000,note='Descriptive paired block interval; not adjusted for repeated research. Not a CAGR percentage-point interval.')
 for name,value in [('ANNUAL_RETURNS',annual),('CONCENTRATION',concentration),('UNCERTAINTY',uncertainty)]:
  (P/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
 def table(summary,keys,cost):
  lines=[f'买卖合计成本{cost/100:.2f}%','方案 | 全期年化 | 最大日终回撤 | 剔除最近252日年化 | 最近252日收益 | 开发期年化 | 平均仓位']
  for key in keys:
   v=summary[f'{key}_cost{cost}'];metrics=[v['full']['cagr'],v['full']['max_drawdown'],v['ex_recent']['cagr'],v['last252']['return_'],v['development']['cagr'],v['mean_exposure']]
   lines.append(NAMES[key]+' | '+' | '.join(f'{z:.2%}' for z in metrics))
  return lines
 lines=['日策略第六轮：入场时机、异常低开和止损冷却','',f'生成UTC：{datetime.now(timezone.utc).isoformat()}。',
 '完成10版本×2费用共20次第一阶段回测，以及21次后续检验，共41次；全部保存。',
 '区间2019-01-02—2026-10-06；最近252日为2025-10-06—2026-10-06，剔除后截至2025-10-03。',
 '初始10万美元，现金不计息、未计个人税，只做多，无杠杆。研究股票池仍为历史指数并集内可用市值Top32＋17只无杠杆ETF。四大组各25%，最多三只、目标少于两只空仓。',
 '数据历史认证未完成，市值快照历史发布时间和修订不确定性仍在；performance_publishable=false。全部历史已经反复研究，下面的开发期/后来时期都不代表新获得的独立样本外。',
 '', '结论',
 f"异常放量低开暂停两日：全期年化{s['shock_pause2_cost10']['full']['cagr']:.2%}，对照{s['stable_control_cost10']['full']['cagr']:.2%}；最大回撤分别{s['shock_pause2_cost10']['full']['max_drawdown']:.2%}与{s['stable_control_cost10']['full']['max_drawdown']:.2%}；最近252日{s['shock_pause2_cost10']['last252']['return_']:.2%}，对照{s['stable_control_cost10']['last252']['return_']:.2%}。",
 f"成本0.30%时，该版本年化{s['shock_pause2_cost30']['full']['cagr']:.2%}，对照{s['stable_control_cost30']['full']['cagr']:.2%}；平均仓位{s['shock_pause2_cost10']['mean_exposure']:.2%}，低于对照{s['stable_control_cost10']['mean_exposure']:.2%}。该观察增益没有依靠提高仓位。",
 '事前声明的开发期标准实际选出止损后暂停三日，未选择上面的异常低开版本。后者只是同时通过开发期及成本门槛的另一个研究候选，不能把它冒称为事前唯一选中者。',
 f"止损暂停三日的开发期年化{s['stop_pause3_cost10']['development']['cagr']:.2%}较好，但全期只有{s['stop_pause3_cost10']['full']['cagr']:.2%}，最近252日{s['stop_pause3_cost10']['last252']['return_']:.2%}，最大回撤{s['stop_pause3_cost10']['full']['max_drawdown']:.2%}。不能因为开发期更好就替换现有方案。",
 '等待止跌确认、要求盘中卖压、重写形态因子均未改善综合表现；高波动五日退出减少回撤，但也减少收益。完整失败结果如下。',
 '', '异常低开暂停的具体规则',
 '在信号日收盘，用当天开盘/前收盘−1与此前20交易日收益标准差比较：低开超过2倍标准差，且当天成交量超过此前20日平均的1.5倍，视为异常低开。两个比较基准都不含当天。',
 '该信号日和后续两个交易日不产生该证券的新买入建议；最早根据第三个后续交易日收盘信号、第四日开盘再买。已有持仓仍按原退出规则，不自动强平。',
 '这是量价异常代理，不能据此认定财报、内幕消息或真实新闻事件；不使用当日收盘信息回填当日开盘交易。',
 '保留原Ridge/历史四因子缓存评分、0.75入场门槛、2ATR止损、3ATR止盈、10交易日上限以及因果月度换股分差策略。退出仍是收盘判断、下个开盘代理成交，跳空可能穿越止损。',
 '', '第一阶段全部结果']
 for cost in [10,30]:lines+=['']+table(s,t.KINDS,cost)
 lines+=['','开发期选择记录',json.dumps(selection,ensure_ascii=False),
 '', '后续相邻参数与组合：在查看第一阶段结果后固定，保留原选择，不再根据近期收益选最优参数']
 keys=['shock_days1','shock_pause2','shock_days3','stop_pause2','stop_pause3','stop_pause4','shock_plus_stop','shock_fixed10']
 all_s={**s,**rb}
 for cost in [10,30]:lines+=['']+table(all_s,keys,cost)
 lines+=['','2024-01-02从10万美元现金重新起步；先前成熟模型可使用，不搬入旧仓位；这段历史已经看过','方案 | 成本 | 2024后年化 | 最大回撤']
 for k in kinds:
  for cost in [10,30]:
   v=rb[f'fresh2024_{k}_cost{cost}'];lines.append(f'{NAMES[k]} | {cost/100:.2f}% | {v["full"]["cagr"]:.2%} | {v["full"]["max_drawdown"]:.2%}')
 lines+=['','0.50%成本压力','方案 | 全期年化 | 最大回撤 | 最近252日']
 for k in kinds:
  v=rb[k+'_cost50'];lines.append(f'{NAMES[k]} | {v["full"]["cagr"]:.2%} | {v["full"]["max_drawdown"]:.2%} | {v["last252"]["return_"]:.2%}')
 lines+=['','逐年收益：2026仅至10/6','年份 | 上一轮对照 | 异常低开暂停 | 止损暂停']
 for y in range(2019,2027):lines.append(str(y)+' | '+' | '.join(f'{annual[k][str(y)]:.2%}' for k in kinds))
 lines+=['','最近252日收益集中度：仅为事后敏感性，不是可交易的移除方案','方案 | 实际 | 最佳五天置零 | 最佳十天置零']
 for k,v in concentration.items():lines.append(f'{NAMES[k]} | {v["actual"]:.2%} | {v["without_best5"]:.2%} | {v["without_best10"]:.2%}')
 lines+=['','相对上一轮对照的描述性不确定性；20日分块、2000次固定种子，未校正多轮筛选']
 for key,v in uncertainty.items():lines.append(f'{key}: 年化增长比减一{v["annualized_growth_ratio_minus1"]:.2%}，95%描述区间[{v["lower95"]:.2%},{v["upper95"]:.2%}]。不是年化差额百分点。')
 recent_max=max(v['last252']['return_'] for phase in summaries.values() for v in phase.values())
 lines+=['',f'本轮全部41次中最近252日最高收益{recent_max:.2%}。该最大值只是试验记录，不据此重新选策略。',
 '', '研究动机来源',
 'Reversals and the Returns to Liquidity Provision：https://www.nber.org/papers/w30917 ——研究波动、流动性与反转速度之间的关系。',
 'Evaporating Liquidity：https://www.nber.org/papers/w17653 ——研究反转收益与市场流动性环境。',
 'Short-Term Return Reversals and Intraday Transactions：https://doi.org/10.1142/S2010139219500022 ——研究盘中与隔夜收益的反转差异。',
 '以上只支持研究方向，不支持本项目具体阈值或收益。本轮是独立假设检验，不是原论文策略复现。',
 '', '验证与账户',
 '9项新增测试通过：等待期与过期、最近设置覆盖、失去资格排除、不追涨、未来信息隔离、低开暂停、执行日信息隔离、止损冷却、入场波动锁定及持仓使用当前分数（部分测试同时覆盖多个断言）。',
 '旧对照两版本×两档成本逐笔订单、每日净值复现。全部41组完成，信号早于成交、现金非负、最多三只、不使用杠杆。缓存评分复用，没有新增训练。',
 '六账户和月策略观察账户的账本、计划、配置/冻结文件及受保护旧研究输入哈希均未变化；没有新增观察账户或更换策略。',
 '文件：CONFIG.json与robustness/CONFIG.json、SUMMARY.json、各版本JSON的全部交易和净值、SELECTION.json、ANNUAL_RETURNS.json、CONCENTRATION.json、UNCERTAINTY.json、FINAL_VALIDATION.json。']
 (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 final=dict(complete=True,completed_runs=len(loaded),new_tests_passed=9,related_prior_tests_passed=10,
  controls_reproduced=True,protected_unchanged=True,signals_before_fills=True,cached_predictions=True,data_certified=False,
  performance_publishable=False,result_sha256=hashes,code_sha256={file:hashlib.sha256((P/file).read_bytes()).hexdigest() for file in ['timing.py','robustness.py','test_timing.py','report.py']},highest_recent_tested=recent_max,original_selection=selection['selected'])
 (P/'FINAL_VALIDATION.json').write_text(json.dumps(final,indent=2)+'\n')
 print('VALIDATED',len(loaded),'runs; highest recent',recent_max);print(json.dumps(uncertainty,indent=2))

if __name__=='__main__':main()

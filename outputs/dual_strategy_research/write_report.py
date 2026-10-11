import json
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent
s=json.loads((P/'SUMMARY.json').read_text());sel=json.loads((P/'SELECTION_EXTENDED.json').read_text());c=json.loads((P/'COMBINATIONS_EXTENDED.json').read_text());v=json.loads((P/'FINAL_VALIDATION.json').read_text())
audit={}
for name in ['m_base_positive','x_m_252_risk_gateFalse']:
 r=json.loads((P/(name+'.json')).read_text());dates=pd.DatetimeIndex([x['date'] for x in r['daily']]);ends=set(str(x.date()) for x in dates[:-1][dates[:-1].to_period('M')!=dates[1:].to_period('M')]);assert all(x['date'] in ends for x in r['orders']);assert all(x['signal_date']<x['session'] for x in r['decisions']);audit[name]=dict(order_count=len(r['orders']),all_orders_month_end=True,signals_before_execution=True,corporate_actions=len(r['actions']),performance_publishable=r['performance_publishable'])
(P/'EXECUTION_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
rows=[]
for label,name in [('开发期选出月策略','m_base_positive'),('事后最近一年最佳月策略','x_m_252_risk_gateFalse')]:
 x=s[name];rows.append(f"{label}: 最近252日 {x['last252_return']:.2%}; 全期CAGR {x['metrics']['cagr']:.2%}; 最大回撤 {x['metrics']['maximum_drawdown']:.2%}")
for label,key in [('开发期选出组合',sel['selected']),('事后最近一年最佳组合',sel['retrospective_best_recent'])]:
 x=c[key];rows.append(f"{label}: 最近252日 {x['last252_return']:.2%}; 全期CAGR {x['full']['cagr']:.2%}; 最大回撤 {x['full']['max_drawdown']:.2%}; {key}")
text='''日/月双策略探索研究结果
区间：2019-01-02 至 2026-10-06。最近一年定义为最后252个交易日，不是全期年化。
重要：以下是有数据限制的探索性模拟，performance_publishable=false；不能作为已通过严格历史数据准入的业绩。

'''+ '\n'.join(rows)+f'''

成本压力测试：买卖合计由0.10%提高到0.30%。开发期选出组合最近252日 {v['development_selected']['last252_return']:.2%}；事后最近一年最佳组合 {v['retrospective_recent_best']['last252_return']:.2%}。

资金口径：组合是一笔10万美元初始分配25%给日策略、75%给月策略，各自运行，之后不转移资金，所以两边权重会漂移。收益不直接相加；组合最多可能持有六个不同标的，并非全账户最多三个。无杠杆、只做多或空仓。现金不计利息，未计个人税。

规则：
日策略四组（强度、历史反弹、反弹路径/空间、回撤形态）各25%；Ridge替代反弹组的一半。年度动量为正且价格高于100日均线，3日标准化回撤低于-0.35或收盘位于当日区间下20%，排除z<-3.5。评分至少0.75，选最多3只；不足两只时新建仓为空仓。基于收盘信号下一开盘执行。止盈止损是收盘检查、下一开盘成交，并非盘中保证成交。
开发期选中日策略：无市场均线门槛、行业最多3只、最长10交易日、止盈3ATR、止损2ATR，排名恶化或候选明显优于持仓时退出。实际规则见research.py。
事后最高收益组合中的日策略：SPY高于200日均线、行业最多2只、最长5日、止盈2ATR/止损2ATR，合格候选不足2只就清仓。
开发期选中月策略：跳过最近42交易日的252日动量排名50%，该动量/63日波动排名20%，行业相对强度15%，CMF10%，近252日高点位置5%；核心动量必须为正，行业最多2只。
事后最近一年最佳月策略：63%风险调整后的252日跳过42日动量，13.5%行业相对强度，9%CMF，4.5%波动空间，10%Ridge预测排名；行业最多3只、无SPY均线门槛、核心动量为正。
月策略使用月末前一交易日信号，在月末收盘卖出旧仓再等权买入最多3只，只有2只则各半，少于2只则现金；月内固定股数（拆股等公司行动除外），没有日策略止盈止损。第一笔调仓在2019-01-31。即使目标相同也计全额换仓成本。
模型每月滚动训练，仅使用已完全实现标签，训练窗口1008交易日，Ridge正则强度10；树模型仅为浅树修正，不等同于外部作者原版模型。

验证与局限：
共42个独立规则变体及1320个相关资金组合；137组合最近252日>=100%，不是137个独立证据。开发期按2019—2023 CAGR最大且回撤不超过40%选择；延伸搜索已受到本轮结果启发，因此2024年后也不是完全未触碰的留出样本。超过100%的近期最佳版本属于事后筛选，需要独立前向观察。
股票池为历史Nasdaq-100与S&P 100并集内按供应商市值快照选Top32，加17只无杠杆ETF。快照包含月/周频，不是完整每日历史市值；滞后一日可用、最长45日保鲜，不用未来插值。供应商历史值并不具备当时可知版本认证，缺失成员仍受available-data口径限制。因此严格准入未通过，不将这些数字描述为可信的纯样本外业绩。
新研究4项测试、既有因子/池12项测试通过。188个训练批次标签日期早于预测起点；两个月策略的全部普通订单均为月末、信号日期早于执行日期，动作另行处理。未修改原六观察账户或下实盘单。
完整参数、全部候选和成本结果见同目录CONFIG.json、EXTENSION_CONFIG.json、SUMMARY.json、COMBINATIONS_EXTENDED.json、FINAL_VALIDATION.json、EXECUTION_AUDIT.json。
'''
(P/'REPORT.txt').write_text(text)
print(text[:1500])

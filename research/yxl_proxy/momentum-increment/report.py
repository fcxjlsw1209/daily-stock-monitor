from pathlib import Path
import json
import pandas as pd

P=Path(__file__).resolve().parent
NAMES={'baseline':'原基准','momentum':'新增动能改善10%','pressure':'既有压力影子','resilience':'既有抗跌影子','constant_control':'常数因子（事后权重对照）'}


def table(df, columns, percent=()):
    d=df[list(columns)].copy()
    if 'model' in d:d['model']=d['model'].map(NAMES)
    for c in percent:
        if c in d:d[c]=d[c].map(lambda v:f'{v*100:.2f}%' if pd.notna(v) else '—')
    return d.rename(columns=columns).to_html(index=False,border=0,escape=True)


def main():
    m=pd.read_csv(P/'metrics.csv');y=pd.read_csv(P/'yearly.csv')
    corr=pd.read_csv(P/'correlations.csv');diag=pd.read_csv(P/'factor_diagnostics.csv')
    boot=pd.DataFrame(json.loads((P/'bootstrap.json').read_text()))
    control=pd.read_csv(P/'weight_control_metrics.csv')
    controlboot=pd.DataFrame(json.loads((P/'weight_control_bootstrap.json').read_text()))
    allm=m.query("window=='all'")
    a=allm.query("model=='baseline' and cost_bps==10").iloc[0]
    b=allm.query("model=='momentum' and cost_bps==10").iloc[0]
    changes=dict(cagr_difference_pp=(b.cagr-a.cagr)*100,
                 drawdown_improvement_pp=(b.max_drawdown-a.max_drawdown)*100,
                 win_difference_pp=(b.trade_win-a.trade_win)*100,
                 stop_difference_pp=(b.stop_rate-a.stop_rate)*100,
                 exposure_difference_pp=(b.average_exposure-a.average_exposure)*100)
    yy=y.query("cost_bps==10 and model in ['baseline','momentum']").pivot(index='year',columns='model',values='total_return')
    yy['difference']=yy.momentum-yy.baseline
    yy.to_csv(P/'paired_years.csv')
    summary=dict(conclusion='Exploratory improvement; insufficient evidence to promote or create another forward account.',
                 period=['2019-01-02','2026-09-18'],delta=changes,
                 full_year_wins=int((yy.loc[2019:2025,'difference']>0).sum()),full_years=7,
                 current_partial_year_difference=float(yy.loc[2026,'difference']),
                 original_baseline_reproduced=True,unit_tests_passed=4,
                 factor_definition='(MACD_hist[t]-MACD_hist[t-2])/SMA_ATR14[t]',
                 bootstrap=json.loads(boot.to_json(orient='records')),
                 posthoc_weight_control=json.loads(control.to_json(orient='records')),
                 posthoc_weight_control_bootstrap=json.loads(controlboot.to_json(orient='records')),
                 diagnostics=json.loads(diag.to_json(orient='records')),
                 correlations=json.loads(corr.to_json(orient='records')))
    (P/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>动能改善因子增量检验</title>
<style>body{font-family:system-ui,sans-serif;max-width:1160px;margin:36px auto;padding:0 22px;line-height:1.6;color:#18212b}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}th,td{padding:9px;border-bottom:1px solid #d5dce2;text-align:right}th:first-child,td:first-child{text-align:left}h2{margin-top:36px}.table{overflow-x:auto}a{color:#155e9b}.note{background:#fff3da;padding:16px}</style>
<h1>动能改善：对已有策略的单因子增量检验</h1>
<p>2026-09-29 · 历史区间2019-01-02至2026-09-18 · 不修改前向模拟账户</p>
<p class="note">结论：历史组合表现略有改善，但因子直接预测力弱、抽样区间跨零。保留研究候选，暂不加入正式策略，也未新增第四账户。</p>
<p>F=(MACD柱[t]−MACD柱[t−2])/原ATR14[t]。MACD12/26/9；ATR为原策略14日简单平均TR。只提高F较大股票的排序，不要求F为正或MACD柱为负。保持原候选集合、原分数≥0.65的资格、R12–1权重30%、每只10%、最多5只、行业2只、guarded10退出。新分数=0.30×R12–1排名+(6/7)×原其余分数+0.10×F排名。其余四个因子按比例缩减，因此实验检验的是这一整套重新加权，不是新因子的纯因果效果。</p>
<p>四组均从2019年初独立10万美元开始，收盘信号后次日开盘模拟，分红按旧仓资格，现金零利息。压力/抗跌列是离线历史对照，不能当成真实启用后的纸面账本。未叠加多个新因子。</p>''']
    cols={'model':'模型','cost_bps':'双边成本bp','cagr':'年化收益','max_drawdown':'最大日终回撤','trade_win':'净胜率','closed_trades':'已平仓数','stop_rate':'止损退出比例','average_exposure':'平均仓位','mean_mae':'平均持仓价格MAE'}
    parts+=['<h2>全区间表现</h2>',table(allm,cols,['cagr','max_drawdown','trade_win','stop_rate','average_exposure','mean_mae'])]
    parts+=['<p>MAE：入场日至卖出前一日最低价与卖出当日开盘的最差跌幅，不使用卖出后的低点，不含分红；止损比例以已平仓交易为分母。新增版未改善平均MAE，止损比例仅小幅下降。</p>']
    parts+=['<h2>追加归因检查：只改权重是否也会改善</h2>',table(control.query("window=='all'"),cols,['cagr','max_drawdown','trade_win','stop_rate','average_exposure','mean_mae']),'<p>这是看到首轮结果后追加的归因检查：将新因子排名固定为0.5，保留同样的权重缩减。双边10bp下，常数对照年化11.88%，原基准11.39%，动能版12.55%。因此观察到的改善有一部分来自重新分配权重；动能版超过常数对照约0.66个百分点年化，仍不足以认定稳定增量。</p>',table(controlboot,{'cost_bps':'双边bp','annual_mean_log_advantage':'动能相对常数对照：年化平均对数收益优势','low95':'95%下界','high95':'95%上界'},['annual_mean_log_advantage','low95','high95']),'<p>两个成本下的区间仍跨0。见 <a href="ATTRIBUTION_CONTROL.md">追加检查说明</a>；此检查不是独立样本外检验，不据此自动改变任何账户。</p>']
    parts+=['<h2>基准与新增版本逐年配对</h2>',table(yy.reset_index(),{'year':'年份','baseline':'原基准收益','momentum':'新增动能收益','difference':'收益差'},['baseline','momentum','difference']),'<p>2026仅截至9月18日；其余为完整年。7个完整年4年改善、3年下降，2026当前下降。收益沿用连续持仓，不在年初重置。</p>']
    parts+=['<h2>历史前后分段</h2>',table(m.query("model in ['baseline','momentum'] and cost_bps==10"),{'model':'模型','window':'分段','cagr':'年化','max_drawdown':'回撤','mean_trade_return':'平均净单笔收益'},['cagr','max_drawdown','mean_trade_return']),'<p>earlier=2019–2021；recent=2022–2026/9/18。所有历史都已反复探索，不是独立样本外检验。</p>']
    parts+=['<h2>候选内预测诊断</h2>',table(diag,{'window':'分段','sample':'子样本','days':'有效日期数','mean_ic':'平均日Spearman IC','top_mean_net':'高三分位5日净收益','bottom_mean_net':'低三分位5日净收益','top_minus_bottom':'高减低'},['top_mean_net','bottom_mean_net','top_minus_bottom']),'<p>每日≥5只原候选，信号后次日开盘入场，第5个后续交易日开盘退出，双边10bp；标签按分段边界剔除跨界交易。negative_hist只作MACD柱为负的描述性子样本，不改变组合规则。重叠标签不独立。全样本IC约0.002，近期为负，未显示稳定单调预测力。</p>']
    parts+=['<h2>与已有因子的重合度</h2>',table(corr,{'factor':'原有因子','days':'有效日期数','mean_daily_spearman':'平均每日秩相关','median_daily_spearman':'中位每日秩相关'}),'<p>在原候选内计算。低平均相关不等于统计独立，也不保证有收益增量。与压力/抗跌平均相关约−0.07/−0.14；与原适度回调排名约−0.38。</p>']
    parts+=['<h2>配对分块抽样</h2>',table(boot,{'cost_bps':'双边bp','annual_mean_log_advantage':'年化平均对数收益优势','low95':'95%区间下界','high95':'95%区间上界'},['annual_mean_log_advantage','low95','high95']),'<p>20交易日移动块、2000次、固定种子。此处不是CAGR差的区间；区间跨0，只作为探索性不确定度参考，未校正历史反复选择及幸存者偏差。</p>']
    parts+=['''<h2>验证与限制</h2><p>原基准在10/20bp成本下的年化、回撤、胜率、交易数精确重现旧研究；新因子候选mask完全一致；2300/3100/4100行前缀检验通过；8次组合模拟现金/已实现/未实现对账通过；4项专用单元测试通过。导入的原计算与源数据SHA256未变，不读写实盘/模拟账本。</p>
<p>当前2026年大市值100家公司名单存在幸存者和事后名单偏差，行业也是当前映射；Yahoo历史未逐条独立核验。未做参数搜索，未声称样本外有效。仍需真正的未来记录来确认。</p>
<p><a href="PROTOCOL.md">事前实验规则</a> · <a href="study.py">计算代码</a> · <a href="test_study.py">单元测试</a> · <a href="metrics.csv">完整指标</a> · <a href="trades.csv.gz">逐笔成交</a> · <a href="daily.csv.gz">逐日权益</a> · <a href="validation.json">验证记录</a> · <a href="summary.json">结构化结论</a></p></html>''']
    (P/'RESULTS.html').write_text(''.join('<div class="table">'+x+'</div>' if x.startswith('<table') else x for x in parts))
    print(json.dumps(changes,indent=2));print('Full years better:',summary['full_year_wins'],'of 7')


if __name__=='__main__':main()

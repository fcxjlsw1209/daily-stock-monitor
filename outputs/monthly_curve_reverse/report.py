"""Summarize attribution uncertainty and transparent rule diagnostics."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from extract import P,O,START,END,read_prices,sha
from hypotheses import metrics

NAMES={'r12_2_replace':'R12-2，月末重选前三','r12_2_keep5':'R12-2，前五名保留','risk12_2_replace':'风险调整 R12-2','mix_raw_risk_replace':'动量/风险调整各半','momentum_highvol_keep5':'动量75%+波动25%，前五保留'}

def main():
 testlog=(P/'TESTS.txt').read_text();assert 'Ran 10 tests' in testlog and testlog.rstrip().endswith('OK')
 cal=json.loads((P/'CALIBRATION.json').read_text());summary=json.loads((P/'SUMMARY.json').read_text());cand=json.loads((P/'MONTHLY_CANDIDATES.json').read_text());rules=json.loads((P/'RULE_COMPARISON.json').read_text());orders=json.loads((P/'RULE_HOLDINGS.json').read_text());sensitivity=json.loads((P/'SENSITIVITY.json').read_text());monthly=json.loads((P/'APPROX_MONTHLY_RETURNS.json').read_text());selected=rules['selected_by_development_shape'];nav=pd.read_csv(P/'RULE_NAV.csv',index_col=0,parse_dates=True);px,_,_=read_prices();obs=pd.read_csv(P/'DIGITIZED_WEALTH.csv');w=obs[obs.panel=='wealth'];axis=cal['wealth'];start=nav.index[0];xstart=axis['x0']+axis['width']*(start-START).days/(END-START).days;target0=np.interp(xstart,w.x,w.log_wealth)
 q=px.loc[nav.index,'QQQ'];q=q/q.iloc[0];qqq=metrics(q)
 report=['月策略截图反推：第一轮完整研究','日期：2026-10-09','',
 '结论：已从作者原图提取紫色净值、相对QQQ和回撤三条线，完成2016-09至2026-09共121个月的候选持仓搜索；只辨认出部分股票线索，尚未复现作者月策略。截图没有足够信息唯一确定每月完整持仓，更不能据此确定模型/权重。',
 '比较了8组固定因子规则×两种月末保留方式，共16个诊断回放。按2016-2021月内曲线形状选择，R12-2月末重选前三最贴近，但仍远低于作者展示的80.64%年化。此结果不证明原图错误，也不证明我们恢复了其策略。','',
 '一、原始证据与校准',
 '原图：SOURCE.jpg；作者区间2016-08-17—2026-09-24；作者披露零成本、同收盘成交、月内固定股数、历史样本重复使用、历史市值代理。17只ETF名单和完整Top32定义未知。',
 f"缓存QQQ同期净值倍数为{cal['qqq_wealth']:.8f}，与原图6.78倍一致。QQQ颜色独立校准时间轴，保留像素验证RMSE={axis['validation_rmse']:.3f}px。紫线没有参与时间轴参数选择。",
 f"对数净值纵轴1像素≈{np.expm1(1/axis['log_scale']):.2%}；时间轴1像素≈{(END-START).days/axis['width']:.2f}个日历日。每月通常只有约8—9个横向像素。回撤图区比例为每像素{100/cal['dd']['scale']:.3f}个百分点，可改善识别，但多个交易日会压成同一列。", 
 'EXTRACTION_QA.png叠加了检测到的紫色像素和已知QQQ路径，供人工检查；被其他曲线遮住的像素保留缺失。相对QQQ和净值来自同一序列，不能当成统计独立证据。','',
 '二、反推方法与不确定性',
 '每月独立枚举本项目近似历史Top32＋17只无杠杆ETF中的3只、2只或空仓。入场等值，之后固定复权价格单位；复权价代表股息再投资近似。2只时各50%，不是留1/3现金。该股票池并非已验证的作者股票池。',
 '联合匹配月内净值与相对净值，回撤使用像素列覆盖区间，避免把压缩后的竖线当成一个精确日收盘点。月初净值及过去最高净值限制在原图读数±1.5像素内；不能为拟合回撤而编造此前峰值。',
 '识别使用当月已发生的收益，属于事后归因。候选表不是事前选股，不拼接成“策略收益”。价格完整性在反推中只决定某条路径能否计算，不作为规则回放的未来筛选条件。',
 f"121个月均完成拟合。将时间轴左右移动0.35像素（约{.35*(END-START).days/axis['width']:.2f}日历日）后，保持完整组合不变的月份为{summary['stable_combinations']}/121。仅满足自定义score≤最优+1的候选组合数中位数为{summary['median_ambiguity']:.0f}。这是敏感性与歧义量度，不是统计置信区间或持仓概率。",
 '以下仅列比较有辨识度的线索。比例指该月近似拟合集合中包含该股票的比例，不是作者持有该股票的概率。','',
 '月份 | 较突出线索 | 近似集合出现比例 | 其他限制']
 clues={'2020-08':['TSLA'],'2025-09':['ORCL'],'2026-04':['AMD','AVGO'],'2026-05':['MU'],'2026-06':['AMAT','MU']}
 for month,tickers in clues.items():
  r=next(v for v in cand if v['month']==month);freq=', '.join(f"{s} {r['near_ticker_fractions'].get(s,0):.1%}" for s in tickers)
  report.append(f"{month} | {', '.join(tickers)} | {freq} | 完整组合仍不确定，最低图像评分{r['best']['score']:.2f}")
 report+=['','2020-08若要求股票原先动量为正，候选示例为ADBE/NFLX/TSLA；2025-09增加强势条件后示例为ORCL/PLTR/TSLA。它们都是条件化的候选，不能当作交易记录。',
 '已做27个额外敏感性拟合：9个月×（未按价格外推的旧市值快照池、正R12-2、正R12-2且动量排名前半）。2020-07仅改变市值代理就会改变首选组合，说明池定义不能省略。具体逐项保留在SENSITIVITY.json。','',
 '三、事前可计算的简单规则检验',
 '目的：检验从曲线和文字得到的规则假设，不以80%/100%收益为优化目标。R12-2定义为P[t-42]/P[t-252]-1；风险调整版本除以过去63日收益标准差。各因子先在当时股票池内排名。所有规则要求R12-2为正、253个连续价格观测；不使用未来回报选股。',
 '8组：R12-2、风险调整R12-2、R6-2、R12-1时钟对照、70/30长短动量、50/50动量与风险调整、75/25动量与高波动、75/25动量与52周高位；分别每月重选前三或保留排名前五的已有持仓。最多3只、少于2只空仓，无行业上限。',
 '回放区间2016-08-31—2026-09-24，从第一个完整月边界起；因此比原图少最初约两周。同收盘信号/同收盘价、零成本、复权价近似。这个乐观研究口径尚无真实收盘竞价执行验证，不是扣费收益，也不是原观察账户口径。所有16个结果均保留，不只显示赢家。',
 '规则 | 252交易日年化 | 最大回撤 | 最终倍数']
 for name in ['r12_2_replace','risk12_2_replace','mix_raw_risk_replace','momentum_highvol_keep5','r12_2_keep5']:
  v=rules['results'][name];report.append(f"{NAMES[name]} | {v['cagr_252']:.2%} | {v['max_dd']:.2%} | {v['wealth']:.2f}x")
 report.append(f"QQQ同期零成本买持 | {qqq['cagr_252']:.2%} | {qqq['max_dd']:.2%} | {qqq['wealth']:.2f}x")
 chosen=rules['results'][selected];report += ['',f"按2016-2021月内形状误差选出的规则：{selected}。开发段月内log误差RMSE={chosen['development_month_shape_log_rmse']:.4f}；后段2022-2026为{chosen['later_month_shape_log_rmse']:.4f}。后段没有被用于这一次规则选择，但此前已经研究过，不能称独立样本外。净值全路径仍明显偏离原图。", 
 '退出逻辑结论：只凭这幅图，无法区分每月彻底重排、保留前五，或模型调分造成的留仓；16个规则的相似度也不足以认定原作者的退出标准。按文字约束可知月内没有日策略式止盈止损，月末才能决定换仓。','',
 '四、下一步优先方向',
 '优先检查分歧最大的月份及其候选集合，而非继续增加所有指标。先确认作者ETF名单是否包含行业/主题/杠杆ETF、Top32是否含我们指数并集之外的公司，再区分纯动量排名与模型残差修正。现有图像支持局部线索，尚不足以推断Ridge目标、正则化强度或选股权重。',
 '如以后取得逐日净值或放大的单月图，现有工具可直接复用；并不以取得它们为继续研究的前提。任何从当月收益反推的持仓，只能用于建立假设，验证规则必须只使用月末当时可见的特征。','',
 '验证：10项测试通过，包括固定股数/2只等权/空仓/日历插值/精确已知组合恢复/同路径不可识别/未来价格不改变先前交易/持仓缺价报错/收益统计/QQQ图像校准。原六账户、月观察账户及相关输入共94个保护文件与本轮前校验和一致。',
 '文件：CALIBRATION.json，DIGITIZED_WEALTH.csv，MONTHLY_CANDIDATES.json，SENSITIVITY.json，RULE_COMPARISON.json，RULE_HOLDINGS.json，RULE_NAV.csv，TESTS.txt，VALIDATION.json。',
 '复现：先extract.py，再inverse.py、hypotheses.py、sensitivity.py、单元测试与report.py。使用work/venv/bin/python及PYTHONPATH=work/research-plot-deps，绘图设置MPLCONFIGDIR=/private/tmp/daily-rebuild-mpl。',
 '所有运行仅写本目录；没有新增观察账户、覆盖冻结计划、下单或修改原策略。']
 (P/'REPORT.txt').write_text('\n'.join(report)+'\n')
 # Identify monthly gaps; original figure estimates remain explicitly uncertain.
 gaps=[]
 for row in monthly:
  m=row['month'];r=next(a for a in cand if a['month']==m);s=pd.Timestamp(r['start']);e=pd.Timestamp(r['end']);actual=nav.at[e,selected]/nav.at[s,selected]-1
  o=next(a for a in orders[selected] if a['date']==r['start'])
  gaps.append(dict(month=m,figure_estimated_return=row['estimated_return'],rule_return=float(actual),return_gap=float(row['estimated_return']-actual),rule_holdings=o['holdings'],inverse_best_candidate=r['best']['holdings'],inverse_is_expost=True))
 (P/'MONTHLY_DIFFERENCES.json').write_text(json.dumps(sorted(gaps,key=lambda a:-a['return_gap']),indent=2)+'\n')
 # Research figure: approximated source target versus actual diagnostic paths.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,1,figsize=(13,12));target_dates=START+pd.to_timedelta((w.x-axis['x0'])/axis['width']*(END-START).days,unit='D');ok=target_dates>=start
 axes[0].plot(target_dates[ok],np.exp(w.loc[ok,'log_wealth']-target0),color='#8e519b',lw=1.7,label='Author monthly, digitized (approximate)')
 for name,color in [('r12_2_replace','#146bb1'),('risk12_2_replace','#23a090')]:axes[0].plot(nav.index,nav[name],color=color,label=f"{NAMES_EN[name]} ({rules['results'][name]['cagr_252']:.1%} CAGR)")
 axes[0].plot(q.index,q,color='#b0881d',label=f"QQQ ({qqq['cagr_252']:.1%} CAGR)");axes[0].set_yscale('log');axes[0].set_ylabel('Wealth (rebased at 2016-08-31)');axes[0].legend(loc='upper left',fontsize=9);axes[0].grid(alpha=.2)
 mask=np.load(P/'purple_dd_mask.npy');yy,xx=np.where(mask);dt=START+pd.to_timedelta((xx-axis['x0'])/axis['width']*(END-START).days,unit='D');dd=(cal['dd']['y0']-yy)/cal['dd']['scale'];axes[1].scatter(dt,dd*100,s=2,color='#8e519b',alpha=.2,label='Author visible drawdown pixels')
 for name,color in [('r12_2_replace','#146bb1'),('risk12_2_replace','#23a090')]:axes[1].plot(nav.index,100*(nav[name]/nav[name].cummax()-1),lw=.8,color=color,label=NAMES_EN[name])
 axes[1].set_ylabel('Drawdown (%)');axes[1].legend(loc='lower left',fontsize=9);axes[1].grid(alpha=.2)
 t=pd.to_datetime([r['month'] for r in monthly]);lo=np.array([r['illustrative_low'] for r in monthly])*100;hi=np.array([r['illustrative_high'] for r in monthly])*100;target=np.array([r['estimated_return'] for r in monthly])*100;actual=np.array([r['rule_return'] for r in gaps])*100
 axes[2].fill_between(t,lo,hi,color='#8e519b',alpha=.13,label='Figure boundary pixel sensitivity (not confidence interval)');axes[2].plot(t,target,color='#8e519b',lw=1,label='Figure estimated monthly return');axes[2].plot(t,actual,color='#146bb1',lw=1,label='R12-2 diagnostic return');axes[2].axhline(0,color='grey',lw=.7);axes[2].set_ylabel('Monthly return (%)');axes[2].legend(fontsize=8,loc='upper left');axes[2].grid(alpha=.2)
 fig.suptitle('Monthly curve reverse study: partial clues, not a replicated strategy',fontsize=15);fig.text(.07,.012,'2016-08-31 to 2026-09-24 | Hypotheses: same-close, zero cost, assumed Top32 +17 ETFs. No independent OOS.\nImage fits are retrospective attribution; inferred combinations are not tradable recommendations.',fontsize=9);fig.tight_layout(rect=[0,.055,1,.965]);fig.savefig(P/'COMPARISON.png',dpi=150);plt.close(fig)
 protected=json.loads((P/'PROTECTED_BEFORE.json').read_text());differences=[f for f,h in protected.items() if sha(f)!=h];sources=json.loads((P/'SOURCE_HASHES.json').read_text());source_diff=[f for f,h in sources.items() if Path(f).exists() and sha(f)!=h]
 validation=dict(tests_passed=10,months_completed=len(cand),rule_diagnostics=len(rules['results']),rule_failures=[n for n,v in rules['results'].items() if 'failed'in v],additional_sensitivity_fits=sum(map(len,sensitivity.values())),protected_files=len(protected),protected_unchanged=not differences,protected_differences=differences,sources_unchanged=not source_diff,source_differences=source_diff,screenshot_copy_sha256=sha(P/'SOURCE.jpg'),original_image_sha256=cal['image_sha256'],author_holdings_verified=False,author_strategy_reproduced=False,independent_oos=False,notes='Ten tests are numerical/accounting/calibration checks, not proof that identified author holdings are correct.')
 assert not differences and not source_diff and validation['screenshot_copy_sha256']==validation['original_image_sha256']
 (P/'VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n');(P/'CODE_HASHES.json').write_text(json.dumps({v.name:sha(v) for v in P.glob('*.py')},indent=2)+'\n');print(json.dumps(validation,indent=2))
NAMES_EN={'r12_2_replace':'R12-2 top3 monthly','risk12_2_replace':'Risk-adjusted R12-2'}
if __name__=='__main__':main()

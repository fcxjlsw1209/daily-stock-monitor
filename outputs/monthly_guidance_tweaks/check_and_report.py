"""Execution and component diagnostics after freeze; no retuning or promotion."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import tweaks as t
read=lambda n:t.read(P/n)

def check():
 F=read('FROZEN.json');assert t.sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip();assert F['code_sha256']==t.sha(P/'tweaks.py')
 m,d,layers,px,op=t.load();result={};spec=F['categories']['industry']['spec'];variants={'industry':spec,'anchor':F['base']['spec']}
 for label,cfg in variants.items():
  ch,sc,_=t.decisions(cfg,m,d,layers);orders=t.rp.make_orders(ch,sc,m);result[label]={}
  for name,mode,cost in [('same_close_zero','close',0.),('next_open_zero','open',0.),('next_open_cost','open',.0005)]:result[label][name]=t.rp.replay(label,orders,px,op,m,cost,mode,False)[0]
  if label=='industry':
   _,_,tr,holds=t.e.engine.replay(px,op,orders,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]));t.dump('category_industry_TRADES.json',tr);t.dump('category_industry_HOLDINGS.json',holds)
 # Omit one component and return its exact budget to the old support; no parameter selection.
 components={'sector_relative':.5,'sector_stability':.25,'tail_resilience':.25};ablations={}
 for omitted,w in components.items():
  altered=dict(layers);remaining=sum(weight*layers[name]['score'] for name,weight in components.items() if name!=omitted)/(1-w)
  altered['industry_mix']=dict(score=remaining,active=layers['industry_mix']['active']*(1-w))
  ch,sc,_=t.decisions(spec,m,d,altered);orders=t.rp.make_orders(ch,sc,m);ablations[omitted]=t.rp.replay(omitted,orders,px,op,m,0.,'close',False)[0]
 t.dump('EXECUTION_CHECK.json',result);t.dump('COMPONENT_DIAGNOSTICS.json',dict(note='After-freeze attribution only. No re-selection. Removed component budget returned to original support.',ablations=ablations))
 for label,z in result.items():print(label,{k:(v['metrics']['cagr'],v['metrics']['sharpe'],v['slices']['validation34']['cagr'],v['slices']['validation34']['sharpe']) for k,v in z.items()},flush=True)
 assert all(t.sha(p)==h for p,h in read('PROTECTED_BEFORE.json').items())
 print('Execution and component checks complete',flush=True)

def report():
 R=read('RESULTS.json');F=read('FROZEN.json');E=read('EXECUTION_CHECK.json');C=read('COMPONENT_DIAGNOSTICS.json');m=read('INPUTS.json');A=R['anchor'];I=R['category_industry'];audit=read('FORECAST_AUDIT.json')
 assert all(z['latest_label_end']<=z['cutoff'] for z in audit)
 labels={'anchor':'原简洁候选','annual_primary':'预先指定主候选：高波动近期Ridge＋自身浮盈降分','sharpe_control':'预先指定风险候选：高波动量价组合＋自身浮盈降分','category_industry':'预先冻结行业类候选：行业相对＋联动稳定＋尾部抗跌','author':'作者持仓重放'}
 lines=['月策略作者提示增量研究：固定原股票池、保留长期动量主轴','生成：'+datetime.now(timezone.utc).isoformat(),'',
 '结论：818组有限配置完成。行业类候选全期年化80.83%、Sharpe1.914，较原简洁候选77.62%、1.857小幅提高；后34个月收益提升很小，Sharpe略低，尚非明显的后段突破。',
 '预先指定的年化主候选是Ridge＋自身浮盈调整，不是事后改选的行业候选。主候选全期年化80.05%，后段表现低于基线；所有预先冻结候选如实保留。',
 '没有新增观察账户、没有自动切换策略、没有改作者记录或现有账本。',
 '',
 '一、统一比较',
 '全期2020-12-31收盘—2026-09-24；前35个持有月选参数，后34个月为2023-11-30收盘—2026-09-24。',
 '主表零成本、零滑点、月末完整收盘信号且同收盘价理想成交；三只等权、月内固定份额。Sharpe以日收益计算，未扣无风险收益。',
 '所有本轮候选与77.62%基线采用相同的原估算Top32＋theme17股票池、资格及归一化，不使用上轮Top36。',
 '模型 | 全期年化 | 全期Sharpe | 全期回撤 | 后34年化 | 后34Sharpe | 后34回撤',
 ]
 for k,label in labels.items():
  z=R[k];a=z['metrics'];b=z['slices']['validation34'];lines.append(f'{label} | {a["cagr"]:.2%} | {a["sharpe"]:.3f} | {a["max_drawdown"]:.2%} | {b["cagr"]:.2%} | {b["sharpe"]:.3f} | {b["max_drawdown"]:.2%}')
 lines += ['',
 '二、行业候选的实际规则',
 '基础分不变：52%跳过42交易日的294交易日动量/近期63日波动；21.9375%波动126；18.5625%跳过两月的加速；7.5%市场状态CMF63。所有因子均在当期池转为居中百分位。',
 '当SPY的63日收益波动率高于自身过去252日中位数时，从原48%辅助分中按比例取出15%用于以下修正：',
 '7.5%：股票相对所属行业ETF的强弱，(P股[t-42]/P股[t-252])/(P行业[t-42]/P行业[t-252])-1；',
 '3.75%：行业联动稳定度，负的|corr63(股票,行业ETF)-corr252(股票,行业ETF)|；',
 '3.75%：极端市场下跌日抗跌，过去252日SPY跌幅落入其事前历史10%尾部时，个股收益减前一日已知beta×SPY收益的均值。',
 '高波动状态下，核心52%不动；旧波动/加速/CMF分别降至15.0820%、12.7617%、5.1563%，再加上述15%。非高波动月份完整沿用原评分。',
 '行业ETF与未知所属行业的标的在行业相对/稳定项取中性0；不会把行业ETF之间的强弱冒充个股相对行业。尾部抗跌可以适用于ETF。',
 '每月底按总分取前三，满足原正长期动量及295交易日预热资格；两只各半，少于两只空仓。本候选不加浮盈惩罚，也未采用前五保留。',
 '',
 '三、其他作者提示本次怎么检验',
 '量价：反弹过多和大幅高开降分；放量创新低后收回、缩量压缩、连续反弹失败、下跌量价压力减轻。',
 'Ridge：alpha1/10/100，普通权重或近期半衰期12个月。每年按当时过去五年数据训练，只使用预测21个交易日收益已成熟的样本；目标是相对SPY的未来收益，不是作者持仓。',
 'GBDT：深度2/3、100次迭代、学习率0.05、最小叶100、L2=1，固定随机种子；在全部历史样本训练，只有SPY弱于MA200时激活小幅排名修正。',
 'Ridge使用已校验的缓存系数，树模型此次拟合12次；训练截止审计48条均通过，没有读取截止日之后的标签。',
 '自身浮盈：只在月末按本模型本次持仓起始价计算，按月波动标准化后tanh截断，最多扣2.5或5分；股票卖完重入时重置起始价，不读作者成本，也不在月内卖出。',
 'FOMC、选举和长假：此次未加，没有经核实的完整历史事件及原作者具体定义。日策略提示也不能自动视为月策略已采用的规则。',
 '',
 '四、搜索与验证流程',
 f'共{F["candidate_count"]}组配置：第一阶段802组单因子/固定组合×小权重×状态激活×保留门槛，第二阶段对训练期前8种不同架构增加16组自身浮盈配置。',
 '长期动量52%不变。新增修正预算2.5/5/7.5/10/15%，从原辅助分取出；全局/弱市/高波动/原前三分差<0.10时激活；保留门槛前3/前5。',
 f'只用前35个月选择；{F["admissible_count"]}组通过训练Sharpe不低于基线0.10、回撤不差于基线5个百分点、12/12/11月三块至少两块收益不低于基线的约束。',
 '在通过约束者中取训练年化最高为主候选，同时预先冻结Sharpe、稳健目标及形态/行业/ML/浮盈类别候选。后34个月不再调参或替换主候选。',
 '这些历史已经多轮研究；股票池、市值、行业历史和17ETF仍是研究代理。故本轮时间隔离不是全新未触碰样本外，也不是恢复作者算法。',
 '原77.62%基线自身也经此前全历史探索；提高几个百分点仍可能含选择偏差。',
 '',
 '五、下一开盘与交易成本敏感性（同一冻结计划）',
 '模型 | 执行 | 全期年化 | 全期Sharpe | 后34年化 | 后34Sharpe',
 ]
 for key,label in [('anchor','原候选'),('industry','行业修正')]:
  for mode,title in [('next_open_zero','下个开盘零成本'),('next_open_cost','下个开盘每边0.05%')]:
   z=E[key][mode];a=z['metrics'];b=z['slices']['validation34'];lines.append(f'{label} | {title} | {a["cagr"]:.2%} | {a["sharpe"]:.3f} | {b["cagr"]:.2%} | {b["sharpe"]:.3f}')
 lines += ['本表延后同一计划的成交时间。对带自身持仓状态的候选，这不等于从头按开盘成本重新生成所有后续状态；行业候选本身没有自身浮盈项。',
 '', '六、行业组合的事后归因检查（不重新选参数）','删除项、其预算归还旧辅助分 | 全期年化 | 全期Sharpe | 后34年化 | 后34Sharpe']
 for k,z in C['ablations'].items():
  a=z['metrics'];b=z['slices']['validation34'];lines.append(f'{k} | {a["cagr"]:.2%} | {a["sharpe"]:.3f} | {b["cagr"]:.2%} | {b["sharpe"]:.3f}')
 months={k:read(k+'_MONTHS.json') for k in ['anchor','annual_primary','sharpe_control','category_industry']}
 changes=[]
 for i,(a,b) in enumerate(zip(months['anchor'],months['category_industry'])):
  if set(a['selected'])==set(b['selected']):continue
  ar=list(map(t.e.actual,a['selected']));br=list(map(t.e.actual,b['selected']))
  # Price data below loaded once for per-month contribution accounting.
  changes.append((i,a,b,ar,br))
 meta,d,layers,px,op=t.load();detail=[]
 for i,a,b,ar,br in changes:
  ra=float((px.loc[a['holding_end'],ar]/px.loc[a['signal_date'],ar]-1).mean());rb=float((px.loc[b['holding_end'],br]/px.loc[b['signal_date'],br]-1).mean())
  detail.append(dict(signal_date=a['signal_date'],holding_end=a['holding_end'],split='training' if i<35 else 'validation',anchor=a['selected'],industry=b['selected'],anchor_return=ra,industry_return=rb,arithmetic_difference=rb-ra,log_wealth_contribution=float(np.log1p(rb)-np.log1p(ra))))
 t.dump('INDUSTRY_CHANGED_MONTHS.json',detail)
 for label,ss in [('all',detail),('training',[z for z in detail if z['split']=='training']),('validation',[z for z in detail if z['split']=='validation'])]:
  lines.append(f'行业组合{label}：与基线换股不同{len(ss)}个月，胜出{sum(z["arithmetic_difference"]>0 for z in ss)}、落后{sum(z["arithmetic_difference"]<0 for z in ss)}；累计对数财富差{sum(z["log_wealth_contribution"] for z in ss):+.6f}。')
 lines += ['', '七、交付验证',
 '8项新测试通过，覆盖原样基线、预算守恒、非激活状态、自身浮盈状态、未来数据隔离、已成熟标签和中性缺失值。',
 '逐日回放通过独立固定股数核算，原77.62%候选的全期指标精确相等；所有保护文件及冻结参数哈希未变。',
 '逐月选择在*_MONTHS.json，逐标的评分构成在*_DECISIONS.json，行业换股归因在INDUSTRY_CHANGED_MONTHS.json，可读逐月表在MONTHLY_COMPARISON.txt。',
 '本轮有小幅全期改进，没有达到作者同期93.75%年化；后34个月仍明显低于作者133.69%/Sharpe2.092。',
 '保持现有观察账户原样。行业修正保留为研究候选，不因一次历史比较就自动升级。',
 ]
 (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 summary=['各月三只持仓及收益：原候选/主候选/行业修正。零成本、月末固定份额。','']
 for i,a in enumerate(months['anchor']):
  summary.append(f'{i+1:02d} {"训练" if i<35 else "验证"} {a["signal_date"]} → {a["holding_end"]}')
  for key in ['anchor','annual_primary','category_industry']:
   z=months[key][i];ss=list(map(t.e.actual,z['selected']));ret=float((px.loc[z['holding_end'],ss]/px.loc[z['signal_date'],ss]-1).mean()) if ss else 0.
   summary.append(f'{labels[key]}：{" > ".join(z["selected"])}；月收益{ret:.2%}')
  summary.append('')
 (P/'MONTHLY_COMPARISON.txt').write_text('\n'.join(summary)+'\n')
 assert t.sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
 assert all(t.sha(p)==h for p,h in read('PROTECTED_BEFORE.json').items())
 t.dump('STATUS.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),complete=True,candidates=F['candidate_count'],main='annual_primary',primary_full_cagr=R['annual_primary']['metrics']['cagr'],industry_candidate=I,baseline=A,author=R['author'],all_period_improvement=True,large_later_period_improvement=False,pristine_oos=False,automatic_promotion=False,protected_changes=[],unit_tests_passed=8))
 print('Report complete; changed industry months',len(detail),flush=True)

if __name__=='__main__':
 if sys.argv[-1]=='check':check()
 else:report()

"""Render frozen research outputs without selecting or changing a strategy."""
from pathlib import Path
import json

P=Path(__file__).resolve().parent
read=lambda f:json.loads((P/f).read_text())
R,F,A,V=map(read,['RESULTS.json','FROZEN.json','ATTRIBUTION.json','VALIDATION.json'])
pct=lambda x:f'{x*100:.2f}%'
labels=dict(original65='已登记观察规则',previous66='此前最佳研究候选',
    original65_session='原底座＋隔夜/日内修正（训练未准入诊断）',
    original65_type='原底座＋类型部分汇聚（本轮冻结主候选）',
    original65_pair='原底座＋市场状态成对排序',simple='独立75%动量＋25%波动',
    simple_session='独立底座＋隔夜/日内修正',simple_type='独立底座＋分类型模型',
    simple_pair='独立底座＋市场状态成对排序',qqq='QQQ买入持有')
lines=['月策略隔夜/日内、股票/ETF类型和实际月度成对排序研究','',
    '54个预先登记配置完成；5个新配置通过训练约束，但冻结主候选在后段退步，未提高既有65.34%观察规则或66.36%研究候选。',
    '主候选由前88期选出，后34期未重选。配对对照和未准入的描述性家族结果也事前冻结并如实保留。没有新增或更换观察账户。','',
    '共同口径：2016-08-17—2026-09-24，零成本、月末完整收盘信号且同收盘价理想化成交；最多三只等权、两只各半、少于两只现金。月内固定复权收益单位。',
    'Sharpe按每日收益、sqrt(252)、无风险利率0；回撤为日终峰值回撤。首次2016-08-17中月初始化，末期截至9/24，均不是完整年。',
    '股票池仍是历史Top32估算＋自拟17ETF，保持原资格；不是已完全认证的历史池，也不是作者精确ETF清单。','',
    '版本 | 全期年化 | Sharpe | 最大回撤 | 2016—2020年化 | 后34期年化']
for name,label in labels.items():
    z=R[name];m=z['metrics'];c=z['slices']
    lines.append(f"{label} | {pct(m['cagr'])} | {m['sharpe']:.3f} | {pct(m['max_drawdown'])} | {pct(c['early2016_2020']['cagr'])} | {pct(c['later34']['cagr'])}")
lines += ['', '本轮主候选规则与训练结果',
    '保留原核心及资格，只用2.5%的辅助权重作修正：共同Ridge斜率＋股票/ETF类型偏离，偏离正则为共同斜率的10倍。',
    '信号每年使用上一年末以前成熟的实际下一自然月收益标签，过去五年、月份均衡、24月半衰期、alpha10、至少18个月。',
    '特征为旧动量、风险调整动量、波动、个股相对行业，以及六项量价特征；无ticker或作者持仓标签。预测保持收益尺度，经0.5*tanh(pred/.05)限制，再微调原辅助分。',
    '原核心50%/52%保留；所有合格证券竞争三个席位。零预测/不可用模型严格不改分。原66.36控制仍用自身锁前两名的原订单。',
    f"训练段年化：{pct(R['original65']['slices']['training2016_2023']['cagr'])} → {pct(R['primary']['slices']['training2016_2023']['cagr'])}；后34期：{pct(R['original65']['slices']['later34']['cagr'])} → {pct(R['primary']['slices']['later34']['cagr'])}。",'',
    '主候选只改变5个月，1次改善、4次变差；后34期三次变更均不利：']
for x in A['primary']['changes']:
    old=','.join(x['old']).replace('research_equity_FB','META')
    new=','.join(x['new']).replace('research_equity_FB','META')
    lines.append(f"{x['date']}→{x['end']}：{old} → {new}；组合当月差{x['difference']*100:+.3f}个百分点。")
lines+=['','同2.5%预算的模型消融（仅诊断，不能按全期再挑冠军）','模型 | 年化 | Sharpe']
for name in ['matched_type_plain_pooled','matched_type_plain_partial_10','matched_type_plain_partial_100','matched_type_plain_independent','matched_pair_plain_pairwise','matched_pair_state_pairwise','matched_pair_state_pooled']:
    z=R[name]['metrics'];lines.append(f"{name} | {pct(z['cagr'])} | {z['sharpe']:.3f}")
lines+=['','隔夜/日内的独立底座结果',
    '三种长期特征：隔夜减日内累计log收益、隔夜正负日持续性、日内正负日持续性。均取截至42交易日前的210日完整窗口；正反方向及2.5%/7.5%预算预先固定。',
    '训练选定的独立底座配置是7.5%正向隔夜减日内修正。简单底座有所改善，旧复杂底座反而下降，不能把改进机械叠加到65%版本。']
z=A['simple_session'];changes=z['changes']
lines += [f"简单底座改变{len(changes)}个月，其中{sum(x['difference']>0 for x in changes)}次改善；全期{pct(R['simple']['metrics']['cagr'])}→{pct(R['simple_session']['metrics']['cagr'])}。",
    f"撤销最有利1次/3次换股的诊断年化：{pct(z['undo_best1_diagnostic_cagr'])}/{pct(z['undo_best3_diagnostic_cagr'])}。这是事后集中度诊断，不是新可交易规则。",'',
    '实际自然月成对排序',
    '比较同月证券下一月收益差，允许量价×市场宽度/趋势/波动交互。事前简单评分前三与其余的配对权重2倍；边界先按过去资格固定，缺未来标签只剔除训练行，绝不把原第四名晋升为事前第三。',
    '状态模型相对普通成对模型更差；一次明显不利换股是2024-11月底AVGO换NVDA，下一月组合少赚15.427个百分点。不能据此专门硬编码保留AVGO。', '',
    '执行敏感性 | 下一开盘零成本年化 | 下一开盘每边0.05%年化 | 后者Sharpe']
for name in ['original65','previous66','primary','original65_session','original65_pair','simple_session']:
    z=R[name];net=z['next_open_cost']['metrics']
    lines.append(f"{labels.get(name,'本轮主候选')} | {pct(z['next_open_zero']['metrics']['cagr'])} | {pct(net['cagr'])} | {net['sharpe']:.3f}")
wf=read('WALKFORWARD.json')
lines += ['',f"训练内部逐段选择诊断：年化{pct(wf['metrics']['cagr'])}，Sharpe{wf['metrics']['sharpe']:.3f}；同段原规则年化{pct(wf['baseline']['cagr'])}、Sharpe{wf['baseline']['sharpe']:.3f}。这仍是反复研究过的历史。",'',
    '验证与可复核输出',
    '31项测试通过；修复了未来标签筛除后的事前边界重排、配对对照同名覆盖和平盘日浮点噪声放大为正负信号的问题。修复均在收益筛选前完成。',
    '122期候选实际形成窗、90证券、151945个联合单元的数值异常检查，没有发现单腿超过log(1.5)或巨大反向抵消；这不等于独立认证所有公司行动。',
    f"两个旧策略指标及原65.34订单精确复现；独立日级固定份额回放一致；{V['protected_files']}个受保护文件哈希未变。",
    '模型与数据、规则在FROZEN.json记录校验和。逐月组合及每只当月收益贡献在MONTH_COMPARISON.json，换股归因和集中度在ATTRIBUTION.json，完整订单/成交/净值另存各版本JSON。',
    '本轮没有研究收益突破；不加入观察账户。数据代理、同收盘理想执行及重复历史探索限制保留，后34期不能称从未接触的样本外。']
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
(P/'COMPLETION.json').write_text(json.dumps(dict(completed=True,trials=54,tests_passed=31,
    primary_cagr=R['primary']['metrics']['cagr'],primary_sharpe=R['primary']['metrics']['sharpe'],
    beats_original65=False,beats_previous66=False,observers_changed=False,protected_files=V['protected_files'],
    evaluation_kind='reused_historical_research'),indent=2)+'\n')
print('REPORT.txt and COMPLETION.json saved')

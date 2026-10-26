# -*- coding: utf-8 -*-
"""Serialized, same-snapshot runs of independently frozen paper accounts."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,json,subprocess,sys
P=Path(__file__).resolve().parent
ROOT=P.parent.parent
ACCOUNTS={'baseline':P.parent/'daily_stock_monitor','shadow':P.parent/'pressure_shadow_monitor','resilience':P.parent/'resilience_shadow_monitor'}

CORE_ACCOUNTS=dict(ACCOUNTS)
ACCOUNTS.update({'accel':P.parent/'accel_shadow_monitor','high52':P.parent/'high52_shadow_monitor','accel40':P.parent/'accel40_shadow_monitor'})

def invoke(folder,args):
    from execution_registry import active_account_paths
    route=active_account_paths(folder)
    proc=subprocess.run([sys.executable,str(route['adapter']),'--account',str(folder),*args],cwd=ROOT,text=True,capture_output=True)
    try:report=json.loads(proc.stdout)
    except Exception:raise RuntimeError(f'{folder.name}: invalid response; exit={proc.returncode}; {proc.stderr[-1000:]}')
    if proc.returncode or report.get('status')=='ERROR_NO_NEW_PLAN':raise RuntimeError(f'{folder.name}: {report}')
    return report

def observe_monthly(folder="monthly_buffer5_monitor"):
    """Separate dynamic-universe monthly account, never merged with static100 orders."""
    proc=subprocess.run([sys.executable,str(P.parent/folder/'observe.py')],cwd=ROOT,text=True,capture_output=True)
    try:
        result=json.loads(proc.stdout)
        if proc.returncode and result.get('status')!='ERROR_NO_NEW_PLAN':
            raise ValueError('Nonzero monthly observer exit')
        from monthly_status import enrich_monthly_status
        return enrich_monthly_status(result,P.parent/folder)
    except Exception:
        return {'status':'ERROR_NO_NEW_PLAN','reason':f'Monthly observer failed; exit={proc.returncode}; {proc.stderr[-500:]}','plan':None}

def publish(reports,monthly=None,monthly_regime=None,monthly_industry=None,monthly_stability=None):
    a,b=reports['baseline'],reports['shadow']
    if set(reports)!=set(ACCOUNTS):raise RuntimeError('Missing account report')
    if any((a.get('data_as_of'),a.get('snapshot_id'),a.get('phase'))!=(r.get('data_as_of'),r.get('snapshot_id'),r.get('phase')) for r in reports.values()):raise RuntimeError('Account snapshot/phase mismatch; no combined recommendation')
    out={'status':'paired_success','generated_utc':datetime.now(timezone.utc).isoformat(),'data_as_of':a['data_as_of'],'accounts':reports}
    out['monthly_observation']=monthly
    out['monthly_observations']={'buffer5':monthly,'regime':monthly_regime,'industry':monthly_industry,'stability':monthly_stability}
    migration_path=P.parent/'dynamic_top100'/'STATUS.json'
    if migration_path.exists():out['universe_migration']=json.loads(migration_path.read_text())
    out['shadow_minus_baseline_return_pp']=100*(b['statistics']['cumulative_return']-a['statistics']['cumulative_return'])
    from cohort import summarize_cohort
    out['cohort']=summarize_cohort(CORE_ACCOUNTS,a['data_as_of'])
    out['expanded_cohort']=summarize_cohort(ACCOUNTS,a['data_as_of'])
    from qqq_benchmark import summarize as summarize_qqq
    out['qqq_buy_hold']=summarize_qqq(ROOT/'work/stock_monitor_cache'/a['snapshot_id'],a['data_as_of'])
    names=list(reports);labels={'baseline':'基准','shadow':'压力影子','resilience':'抗跌影子','accel':'动量加速20日','high52':'52周高点20日','accel40':'动量40/10 20日'}
    lines=['# 六策略独立模拟每日比较','',f"生成时间：{out['generated_utc']}；行情截至：{a['data_as_of']}。",'', '独立模拟账户，按冻结规则连续跟踪，包含已授权的漏跑日顺序补算；不是实盘或纯样本外绩效。账户启用日期不同，全历史收益不可直接比较。','', '|项目|'+'|'.join(labels[n] for n in names)+'|','|---|'+'---:|'*len(names)]
    if out.get('universe_migration',{}).get('paper_migration_complete') is False:
        lines[1:1]=['', '**股票池说明：以下仍是2026-09-18固定Top100旧版本。每日当期Top100迁移等待可验证数据，尚未生效。**', '']
    def win(st):return '尚无数据' if st['win_rate'] is None else f"{st['win_rate']:.2%}"
    fields=[('运行状态',lambda r:r['status']),('账本截至',lambda r:r['statistics']['as_of'] or '尚未开始'),('净值',lambda r:f"${r['statistics']['equity']:,.2f}"),('累计净收益',lambda r:f"{r['statistics']['cumulative_return']:.2%}"),('已平仓笔数',lambda r:str(r['statistics']['closed_trades'])),('净胜率',lambda r:win(r['statistics'])),('已实现盈亏',lambda r:f"${r['statistics']['realized_pnl']:,.2f}"),('未实现盈亏',lambda r:f"${r['statistics']['unrealized_pnl']:,.2f}"),('最大日终回撤',lambda r:f"{-r['statistics']['max_close_drawdown']:.2%}"),('持仓数',lambda r:str(r['statistics']['open_positions'])),('未补齐交易日',lambda r:str(r['statistics']['missed_sessions'])),('已补算交易日',lambda r:str(r['statistics'].get('replayed_sessions',0)))]
    for label,fmt in fields:lines.append('|'+label+'|'+'|'.join(fmt(reports[n]) for n in names)+'|')
    lines.extend(['',f"影子相对基准累计收益差：{out['shadow_minus_baseline_return_pp']:+.2f}个百分点。",''])
    cohort=out['cohort'];lines += ['','## 共同观察区间',f"起点：{cohort['start']}；共同完整交易日：{cohort['common_complete_days']}（仅事前计划）；共同补算日：{cohort.get('common_replayed_days',0)}；达到初步评估门槛：{cohort['initial_review_ready']}。",'收益含继承持仓影响；胜率仅计共同起点后入场的已平仓交易。','', '|账户|同期收益|同期最大回撤|新入场后已平仓|净胜率|继承持仓平仓|','|---|---:|---:|---:|---:|---:|']
    for name,c in cohort['accounts'].items():
        rate='尚无数据' if c['win_rate'] is None else f"{c['win_rate']:.2%}"
        lines.append(f"|{labels[name]}|{c['return']:.2%}|{-c['max_drawdown']:.2%}|{c['closed_trades']}|{rate}|{c['inherited_closed_trades']}|")
    lines += ['', '## 同期 QQQ Buy & Hold', '回溯比较基准：各起点开盘满仓买入10万美元，买入成本0.05%；持有未卖出，不扣卖出费用。复权价格近似股息再投资；不是新增模拟账户，风险敞口与约半仓策略不同。', '', '|起点|截至|净收益|最大日终回撤|', '|---|---|---:|---:|']
    for start,q in out['qqq_buy_hold']['periods'].items():
        lines.append(f"|{start}|{q['as_of']}|{q['return']:.3%}|{-q['max_drawdown']:.3%}|")
    expanded=out['expanded_cohort']
    lines += ['', '## 新观察组共同区间', f"起点：{expanded['start']}；截至：{expanded['as_of']}；完整共同交易日：{expanded['common_complete_days']}。", '新三账户每只20%、最多100%入场预算；原三账户每只10%、最多50%。收益不代表同等风险比较。原9/28共同区间保留。']
    if expanded['as_of']<expanded['start']:lines.append('新观察组尚未开始；无模拟成交，胜率尚无数据。')
    else:
        lines += ['', '|账户|同期收益|最大回撤|新入场平仓|净胜率|', '|---|---:|---:|---:|---:|']
        for name,c in expanded['accounts'].items():
            rate='尚无数据' if c['win_rate'] is None else f"{c['win_rate']:.2%}"
            lines.append(f"|{labels[name]}|{c['return']:.3%}|{-c['max_drawdown']:.3%}|{c['closed_trades']}|{rate}|")
    for name,r in reports.items():
        lines += [f'## {name}', '', '盘前计划（无计划不等于无入围股票）：','```json',json.dumps(r.get('plan'),ensure_ascii=False,indent=2),'```','', '事后重算（不计入前向账本）：','```json',json.dumps(r.get('retrospective_plan'),ensure_ascii=False,indent=2),'```','', '当前持仓：','```json',json.dumps(r['statistics']['positions'],ensure_ascii=False,indent=2),'```','']
    lines += ['## 月观察执行与数据状态', '', '未到首次信号日与数据更新成功是不同状态。以下现金仅在只读账本核验未发生交易后展示；未补造日终账本。', '', '|账户|今日需交易|首次信号|首次执行|现金|数据状态|', '|---|---|---|---|---:|---|']
    monthly_labels={'buffer5':'前五名保留','regime':'市场状态','industry':'行业修正','stability':'稳定性'}
    for name,r in out['monthly_observations'].items():
        if r is None:continue
        due='无需（尚未到首次信号）' if r.get('no_trade_due_today') is True else '以账户原始状态为准'
        cash=f"${r['cash']:,.2f}" if r.get('execution_ledger_audit',{}).get('untouched_cash_verified') else '未核实'
        lines.append(f"|{monthly_labels[name]}|{due}|{r.get('first_signal_date','未核实')}|{r.get('first_execution_date','未核实')}|{cash}|{r.get('data_health',r.get('status'))}；截至{r.get('data_as_of','未知')}|")
    for name,r in reports.items():
        for revision in r.get('verified_entry_price_revisions',[]):
            lines += ['', f"报价修订已核实：{labels[name]} {revision['symbol']} 的 {revision['entry_date']} 开盘报价由 ${revision['recorded_entry_price']:.2f} 修订至 ${revision['accepted_provider_open']:.2f}。原模拟成交价、股数和历史记录保留；该核验仅解除已知修订的计划阻塞。"]
    if monthly is not None:
        lines += ['## 月策略独立观察：差额调仓＋前五名保留','',
                  '动态历史指数并集内Top32＋17 ETF，数据与原六账户不同，独立展示，不合并买单或收益。',
                  f"状态：{monthly.get('status')}；行情截至：{monthly.get('data_as_of')}；下一月末：{monthly.get('next_rebalance')}。",
                  monthly.get('reason',''),
                  '```json',json.dumps({'statistics':monthly.get('statistics'),'plan':monthly.get('plan')},ensure_ascii=False,indent=2),'```',
                  '详细状态：outputs/monthly_buffer5_monitor/STATUS.txt','']
    if monthly_regime is not None:
        lines += ['## 月策略独立观察：长期动量＋波动状态修正','',
                  '估算当期Top32＋自拟theme17 ETF；月末完整收盘信号、下一交易日开盘差额调仓。独立账本，不合并原六账户或另一月策略。',
                  f"状态：{monthly_regime.get('status')}；行情截至：{monthly_regime.get('data_as_of')}；信号日：{monthly_regime.get('required_signal_date')}；开盘调仓日：{monthly_regime.get('next_rebalance')}。",
                  monthly_regime.get('reason',''),
                  '```json',json.dumps({'statistics':monthly_regime.get('statistics'),'plan':monthly_regime.get('plan')},ensure_ascii=False,indent=2),'```',
                  '平仓统计按卖出片段计，包含部分减仓；不是完整持仓轮次。模拟数量是复权总回报单位。',
                  '详细状态：outputs/monthly_regime_monitor/STATUS.txt','']
    if monthly_industry is not None:
        lines += ['## 月策略独立观察：长期动量＋行业辅助修正','',
                  '估算Top32＋theme17；高波动状态15%行业相对/联动稳定/尾部抗跌修正；独立10万美元，不合并买单或收益。',
                  f"状态：{monthly_industry.get('status')}；行情截至：{monthly_industry.get('data_as_of')}；信号日：{monthly_industry.get('required_signal_date')}；开盘调仓日：{monthly_industry.get('next_rebalance')}。",
                  monthly_industry.get('reason',''),
                  '```json',json.dumps({'statistics':monthly_industry.get('statistics'),'plan':monthly_industry.get('plan')},ensure_ascii=False,indent=2),'```',
                  '月末收盘信号、下一开盘执行；复权总回报单位，卖出片段包含部分减仓。',
                  '详细状态：outputs/monthly_industry_monitor/STATUS.txt','']
    if monthly_stability is not None:
        lines += ['## 月策略独立观察：低波动状态修正（65.34%研究候选）','',
                  '独立10万美元；估算Top32＋theme17；月末收盘信号、下一开盘执行，不合并其他账户。',
                  f"状态：{monthly_stability.get('status')}；行情截至：{monthly_stability.get('data_as_of')}；信号日：{monthly_stability.get('required_signal_date')}；开盘调仓日：{monthly_stability.get('next_rebalance')}。",
                  monthly_stability.get('reason',''),
                  '```json',json.dumps({'statistics':monthly_stability.get('statistics'),'plan':monthly_stability.get('plan')},ensure_ascii=False,indent=2),'```',
                  '65.34%是零成本同收盘价历史研究结果；本观察按每边0.05%成本，月内固定复权单位，不回填历史收益。',
                  '详细状态：outputs/monthly_stability_monitor/STATUS.txt','']
    (P/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    (P/'COMPARISON.md').write_text('\n'.join(lines))
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['auto','morning','close'],default='auto');parser.add_argument('--no-refresh',action='store_true');args=parser.parse_args()
    with (P/'run.lock').open('a') as lock:
      monthly=None;monthly_regime=None;monthly_industry=None;monthly_stability=None
      try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        monthly=observe_monthly()
        monthly_regime=observe_monthly("monthly_regime_monitor")
        monthly_industry=observe_monthly("monthly_industry_monitor")
        monthly_stability=observe_monthly("monthly_stability_monitor")
        for folder in ACCOUNTS.values():invoke(folder,['--phase','status'])
        options=['--phase',args.phase]+(['--no-refresh'] if args.no_refresh else [])
        baseline=invoke(ACCOUNTS['baseline'],options)
        snapshot=ROOT/'work/stock_monitor_cache'/baseline['snapshot_id']
        reports={'baseline':baseline}
        for name,folder in ACCOUNTS.items():
            if name!='baseline':reports[name]=invoke(folder,['--phase',args.phase,'--snapshot',str(snapshot)])
        result=publish(reports,monthly,monthly_regime,monthly_industry,monthly_stability)
        print(json.dumps(result,ensure_ascii=False,indent=2))
      except Exception as exc:
        result={'status':'ERROR_NO_NEW_RECOMMENDATIONS','error':str(exc),'generated_utc':datetime.now(timezone.utc).isoformat(),'monthly_observation':monthly,'monthly_observations':{'buffer5':monthly,'regime':monthly_regime,'industry':monthly_industry,'stability':monthly_stability}}
        (P/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        (P/'COMPARISON.md').write_text('# 本次运行失败\n\n'+result['generated_utc']+'\n\n'+str(exc)+'\n\n不发布新买入建议；不要引用旧摘要。已保存的不可变计划和账本保留。\n')
        print(json.dumps(result,ensure_ascii=False));raise SystemExit(1)
if __name__=='__main__':main()

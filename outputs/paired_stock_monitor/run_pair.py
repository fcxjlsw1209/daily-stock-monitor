# -*- coding: utf-8 -*-
"""Serialized, same-snapshot runs of independently frozen paper accounts."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,json,subprocess,sys
P=Path(__file__).resolve().parent
ROOT=P.parent.parent
ACCOUNTS={'baseline':P.parent/'daily_stock_monitor','shadow':P.parent/'pressure_shadow_monitor','resilience':P.parent/'resilience_shadow_monitor'}

def invoke(folder,args):
    proc=subprocess.run([sys.executable,str(P/'evening_adapter.py'),'--account',str(folder),*args],cwd=ROOT,text=True,capture_output=True)
    try:report=json.loads(proc.stdout)
    except Exception:raise RuntimeError(f'{folder.name}: invalid response; exit={proc.returncode}; {proc.stderr[-1000:]}')
    if proc.returncode or report.get('status')=='ERROR_NO_NEW_PLAN':raise RuntimeError(f'{folder.name}: {report}')
    return report

def publish(reports):
    a,b=reports['baseline'],reports['shadow']
    if set(reports)!=set(ACCOUNTS):raise RuntimeError('Missing account report')
    if any((a.get('data_as_of'),a.get('snapshot_id'),a.get('phase'))!=(r.get('data_as_of'),r.get('snapshot_id'),r.get('phase')) for r in reports.values()):raise RuntimeError('Account snapshot/phase mismatch; no combined recommendation')
    out={'status':'paired_success','generated_utc':datetime.now(timezone.utc).isoformat(),'data_as_of':a['data_as_of'],'accounts':reports}
    out['shadow_minus_baseline_return_pp']=100*(b['statistics']['cumulative_return']-a['statistics']['cumulative_return'])
    from cohort import summarize_cohort
    out['cohort']=summarize_cohort(ACCOUNTS,a['data_as_of'])
    names=list(reports);labels={'baseline':'基准','shadow':'压力影子','resilience':'抗跌影子'}
    lines=['# 三策略独立模拟每日比较','',f"生成时间：{out['generated_utc']}；行情截至：{a['data_as_of']}。",'', '独立模拟账户，包含已授权补记；不是实盘或纯样本外绩效。账户启用日期不同，全历史收益不可直接比较。','', '|项目|'+'|'.join(labels[n] for n in names)+'|','|---|'+'---:|'*len(names)]
    def win(st):return '尚无数据' if st['win_rate'] is None else f"{st['win_rate']:.2%}"
    fields=[('运行状态',lambda r:r['status']),('账本截至',lambda r:r['statistics']['as_of'] or '尚未开始'),('净值',lambda r:f"${r['statistics']['equity']:,.2f}"),('累计净收益',lambda r:f"{r['statistics']['cumulative_return']:.2%}"),('已平仓笔数',lambda r:str(r['statistics']['closed_trades'])),('净胜率',lambda r:win(r['statistics'])),('已实现盈亏',lambda r:f"${r['statistics']['realized_pnl']:,.2f}"),('未实现盈亏',lambda r:f"${r['statistics']['unrealized_pnl']:,.2f}"),('最大日终回撤',lambda r:f"{-r['statistics']['max_close_drawdown']:.2%}"),('持仓数',lambda r:str(r['statistics']['open_positions'])),('错过交易日',lambda r:str(r['statistics']['missed_sessions']))]
    for label,fmt in fields:lines.append('|'+label+'|'+'|'.join(fmt(reports[n]) for n in names)+'|')
    lines.extend(['',f"影子相对基准累计收益差：{out['shadow_minus_baseline_return_pp']:+.2f}个百分点。",''])
    cohort=out['cohort'];lines += ['','## 共同观察区间',f"起点：{cohort['start']}；共同完整交易日：{cohort['common_complete_days']}；达到初步评估门槛：{cohort['initial_review_ready']}。",'收益含继承持仓影响；胜率仅计共同起点后入场的已平仓交易。','', '|账户|同期收益|同期最大回撤|新入场后已平仓|净胜率|继承持仓平仓|','|---|---:|---:|---:|---:|---:|']
    for name,c in cohort['accounts'].items():
        rate='尚无数据' if c['win_rate'] is None else f"{c['win_rate']:.2%}"
        lines.append(f"|{labels[name]}|{c['return']:.2%}|{-c['max_drawdown']:.2%}|{c['closed_trades']}|{rate}|{c['inherited_closed_trades']}|")
    for name,r in reports.items():
        lines += [f'## {name}', '', '盘前计划（无计划不等于无入围股票）：','```json',json.dumps(r.get('plan'),ensure_ascii=False,indent=2),'```','', '事后重算（不计入前向账本）：','```json',json.dumps(r.get('retrospective_plan'),ensure_ascii=False,indent=2),'```','', '当前持仓：','```json',json.dumps(r['statistics']['positions'],ensure_ascii=False,indent=2),'```','']
    (P/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    (P/'COMPARISON.md').write_text('\n'.join(lines))
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['auto','morning','close'],default='auto');parser.add_argument('--no-refresh',action='store_true');args=parser.parse_args()
    with (P/'run.lock').open('a') as lock:
      try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for folder in ACCOUNTS.values():invoke(folder,['--phase','status'])
        options=['--phase',args.phase]+(['--no-refresh'] if args.no_refresh else [])
        baseline=invoke(ACCOUNTS['baseline'],options)
        snapshot=ROOT/'work/stock_monitor_cache'/baseline['snapshot_id']
        reports={'baseline':baseline}
        for name,folder in ACCOUNTS.items():
            if name!='baseline':reports[name]=invoke(folder,['--phase',args.phase,'--snapshot',str(snapshot)])
        result=publish(reports)
        print(json.dumps(result,ensure_ascii=False,indent=2))
      except Exception as exc:
        result={'status':'ERROR_NO_NEW_RECOMMENDATIONS','error':str(exc),'generated_utc':datetime.now(timezone.utc).isoformat()}
        (P/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        (P/'COMPARISON.md').write_text('# 本次运行失败\n\n'+result['generated_utc']+'\n\n'+str(exc)+'\n\n不发布新买入建议；不要引用旧摘要。已保存的不可变计划和账本保留。\n')
        print(json.dumps(result,ensure_ascii=False));raise SystemExit(1)
if __name__=='__main__':main()

"""Portable command line entry point; scheduling is configured separately."""
import argparse,json,os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(prog='stock-monitor')
    parser.add_argument('--home',help='Runtime directory (default .stock-monitor, or STOCK_MONITOR_HOME)')
    sub=parser.add_subparsers(dest='command',required=True)
    setup=sub.add_parser('init',help='Create a new paper account starting at a future session')
    setup.add_argument('--config',required=True,help='Example strategy/universe JSON')
    setup.add_argument('--start',help='Future NYSE trading date YYYY-MM-DD; default next session')
    run=sub.add_parser('run',help='Refresh free data and run morning/close workflow')
    run.add_argument('--phase',choices=['auto','morning','close'],default='auto')
    run.add_argument('--no-refresh',action='store_true')
    sub.add_parser('status',help='Read paper-account statistics without network access')
    args=parser.parse_args()
    if args.home:os.environ['STOCK_MONITOR_HOME']=str(Path(args.home).expanduser().resolve())
    from .paths import runtime_dir
    from . import monitor
    from .market_data import schedule
    import pandas as pd
    home=runtime_dir()
    try:
        if args.command=='init':
            from .factors import MAP
            cfg=json.loads(Path(args.config).read_text())
            if cfg.get('mode')!='paper_only':raise ValueError('Only paper_only is supported')
            if cfg.get('admission') not in ['broad','strict','industry'] or cfg.get('exit') not in ['fixed5','rebound5','guarded10']:raise ValueError('Unknown strategy')
            expected={'R12-1':.30,'relative_strength':.20,'pullback':.25,'stability':.10,'CMF21':.15}
            if cfg.get('weights')!=expected or cfg.get('sector_etfs')!=MAP:raise ValueError('Version v1 requires the documented fixed factors and sector ETF map')
            symbols=cfg['symbols']
            if len(symbols)!=len(set(symbols)) or not symbols or not set(symbols)<=set(cfg['sectors']):raise ValueError('Invalid symbol/sector map')
            if not all(cfg['sectors'][s] in MAP for s in symbols):raise ValueError('Unsupported sector')
            if not (0<cfg['allocation']<=.10 and cfg['max_positions']==5 and cfg['sector_cap']==2 and cfg['initial_capital']>0 and 0<=cfg['cost_bps']<=100):raise ValueError('Invalid account constraints')
            now=pd.Timestamp.now(tz='UTC');today=pd.Timestamp(now.tz_convert('America/New_York').date());calendar=schedule(now)
            start=pd.Timestamp(args.start) if args.start else calendar.index[calendar.index>today][0]
            if start<=today or start not in calendar.index:raise ValueError('Start must be a future NYSE session; backdating is not supported')
            cfg['paper_start']=str(start.date())
            home.mkdir(parents=True,exist_ok=True)
            if (home/'paper_ledger.sqlite').exists():raise ValueError('Account already exists; select a new --home')
            with (home/'config.json').open('x') as stream:json.dump(cfg,stream,indent=2)
            db=monitor.connect();monitor.init(db,cfg);result={'status':'initialized','home':str(home),'paper_start':cfg['paper_start'],'statistics':monitor.statistics(db)};db.close()
        elif args.command=='status':
            cfg=json.loads((home/'config.json').read_text());db=monitor.connect();monitor.init(db,cfg)
            result={'status':'status_only','statistics':monitor.statistics(db)};db.close()
        else:result=monitor.run(args.phase,not args.no_refresh)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    except Exception as exc:
        print(json.dumps({'status':'ERROR_NO_NEW_PLAN','error':str(exc)},ensure_ascii=False));raise SystemExit(1)

if __name__=='__main__':main()

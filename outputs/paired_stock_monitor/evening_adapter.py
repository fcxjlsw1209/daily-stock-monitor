"""Execution schedule v2; strategy fingerprints and historical rows stay intact."""
from pathlib import Path
import argparse,sys,json,hashlib
import pandas as pd
P=Path(__file__).resolve().parent

def target_session(calendar,cutoff):
    return calendar.index[calendar.index>pd.Timestamp(cutoff)][0]

def plan_mode(now,opening):
    return 'prospective' if pd.Timestamp(now)<pd.Timestamp(opening)-pd.Timedelta(minutes=1) else 'retrospective'

def run(folder,phase,refresh,snapshot):
    sys.path.insert(0,str(folder))
    import monitor as m
    from market_data import load_snapshot,CACHE
    cfg=json.loads((folder/'config.json').read_text())
    db=m.connect();m.init(db,cfg)
    digest=hashlib.sha256(Path(__file__).read_bytes()+(P/'reconciliation_v2.py').read_bytes()+(P/'snapshot_guard.py').read_bytes()+(P/'secondary_prices.py').read_bytes()).hexdigest()
    row=db.execute("SELECT value FROM meta WHERE key='evening_adapter_v2_hash'").fetchone()
    if row and row[0]!=digest:raise RuntimeError('Execution adapter changed; explicit migration required')
    if not row:
        with db:
            db.execute('INSERT INTO meta VALUES (?,?)',('evening_adapter_v2_hash',digest))
            db.execute('INSERT INTO meta VALUES (?,?)',('evening_adapter_v2_migration',m.dumps({'at_utc':pd.Timestamp.now(tz='UTC').isoformat(),'description':'User approved prior-close planning; no historical rows or original fingerprints rewritten'})))
    if phase=='status':
        result={'status':'status_only','statistics':m.statistics(db)};db.close();return result
    db.close()
    from reconciliation_v2 import reconcile
    m.reconcile=lambda db,cfg,frames,manifest,now:reconcile(m,db,cfg,frames,manifest,now)
    from snapshot_guard import prepare_snapshot,fallback_seed
    cutoff=m.completed_session(pd.Timestamp.now(tz='UTC'))
    if snapshot:source=Path(snapshot)
    elif refresh:
        try:source=m.download(cutoff)
        except Exception as exc:source=fallback_seed(CACHE,cutoff,exc)
    else:source=Path(json.loads((CACHE/'latest.json').read_text())['folder'])
    snapshot=prepare_snapshot(source,m.schedule(pd.Timestamp.now(tz='UTC')).index)
    report=m.run('close',False,snapshot)
    now=pd.Timestamp.now(tz='UTC');calendar=m.schedule(now);cutoff=pd.Timestamp(report['data_as_of'])
    target=target_session(calendar,cutoff);opening=calendar.at[target,'open']
    snap=Path(snapshot) if snapshot else CACHE/report['snapshot_id']
    data,raw,frames,manifest=load_snapshot(snap)
    db=m.connect();m.init(db,cfg);state=m.latest_state(db)
    report['phase']='evening_v2';report['execution_version']='evening_v2';report['target_session']=str(target.date())
    old=db.execute('SELECT payload FROM plans WHERE session=?',(str(target.date()),)).fetchone()
    if old:
        report['plan']=json.loads(old[0]);report['status']='existing_immutable_plan'
        report['plan_origin']=report['plan'].get('origin','prospective')
    elif target<pd.Timestamp(cfg['paper_start']):report['status']='paper_start_pending'
    else:
        for symbol,position in state['positions'].items():
            if abs(float(data.opening.at[pd.Timestamp(position['entry_date']),symbol])/position['entry_price']-1)>.001:
                raise RuntimeError('Unprocessed corporate action or price revision: '+symbol)
        f=m.features(data,raw)
        decision=m.plan(state,target,calendar.index,data.close,f['ma5'],f['atr'],f['masks'][cfg['admission']],f['score'],cfg['sectors'],cfg)
        decision.update(strategy_version=cfg['version'],data_as_of=manifest['as_of'],cost_bps=cfg['cost_bps'],fill_model='next_session_open_proxy_not_actual_broker_fill')
        mode=plan_mode(pd.Timestamp.now(tz='UTC'),opening)
        if mode=='prospective':
            m.save_plan(db,decision,pd.Timestamp.now(tz='UTC'),snap,opening)
            report.update(plan=decision,status='next_open_plan_frozen',plan_origin='prospective')
        else:
            report.update(status='retrospective_only_no_forward_plan',plan=None,retrospective_plan=decision,plan_origin='retrospective')
            dest=folder/'retrospective';dest.mkdir(exist_ok=True)
            file=dest/(str(target.date())+'.json')
            if not file.exists():
                file.write_text(m.dumps({'created_utc':now.isoformat(),'data_as_of':manifest['as_of'],'snapshot_id':manifest['snapshot_id'],'plan':decision,'note':'Computed after deadline from prior-session data. Not a prospective plan or forward-account fill.'}))
            report['retrospective_plan']=json.loads(file.read_text())['plan']
    report['fill_status']='Completed-session ledger only; intraday opening fills are pending close confirmation.'
    report['statistics']=m.statistics(db)
    report['statistics']['mode']='paper_simulation_with_recorded_provenance'
    report['statistics']['retrospective_sessions']=db.execute("SELECT count(*) FROM sessions WHERE status='user_authorized_retrospective_executed'").fetchone()[0]
    m.exports(db,report);db.close();return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--account',required=True);p.add_argument('--phase',default='auto');p.add_argument('--no-refresh',action='store_true');p.add_argument('--snapshot');a=p.parse_args()
    try:print(json.dumps(run(Path(a.account),a.phase,not a.no_refresh,a.snapshot),ensure_ascii=False,indent=2,allow_nan=False))
    except Exception as e:print(json.dumps({'status':'ERROR_NO_NEW_PLAN','error':str(e)}));raise SystemExit(1)

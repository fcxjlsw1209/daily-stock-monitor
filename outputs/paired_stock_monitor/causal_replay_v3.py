"""Authorized scheduled-rule simulation, including causal reconstruction of missed runs.

Frozen strategy modules and original v2 ledgers are not modified. Each account has
an independent v3 ledger/export directory, routed by ACTIVE_EXECUTION.json.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import copy
import math
import pandas as pd

P=Path(__file__).resolve().parent
VERSION='causal_replay_v3'

def execution_fingerprint():
    names=['causal_replay_v3.py','reconciliation_v3.py','price_revision_guard.py',
           'snapshot_guard.py','secondary_prices.py','execution_registry.py',
           'EXECUTION_POLICY_V3.json']
    return hashlib.sha256(b''.join((P/name).read_bytes() for name in names)).hexdigest()

def confined(folder,value):
    path=Path(value).resolve();folder=Path(folder).resolve()
    try:path.relative_to(folder)
    except ValueError:raise ValueError('Execution ledger/export path escapes account')
    return path

def run(folder,phase='auto',refresh=True,snapshot=None,ledger=None,export_dir=None):
    folder=Path(folder).resolve()
    sys.path.insert(0,str(folder))
    import monitor as m
    from market_data import load_snapshot,CACHE
    from execution_registry import active_account_paths
    from reconciliation_v3 import reconcile,build_decision
    from snapshot_guard import prepare_snapshot,fallback_seed
    from price_revision_guard import validate_entry_basis
    if ledger is None:
        route=active_account_paths(folder)
        if route['version']!=VERSION:raise ValueError('Execution v3 has not been activated')
        ledger,export_dir=route['ledger'],route['exports']
    elif export_dir is None:
        raise ValueError('Explicit migration ledger needs an export directory')
    ledger=confined(folder,ledger);export_dir=confined(folder,export_dir)
    if ledger!=folder/'execution_v3/paper_ledger.sqlite' or export_dir!=folder/'execution_v3':
        raise ValueError('V3 must use the dedicated independent ledger/export directory')
    if not ledger.is_file():raise ValueError('Missing initialized execution-v3 ledger')
    if (folder/'paper_ledger.sqlite').exists() and ledger.samefile(folder/'paper_ledger.sqlite'):
        raise ValueError('V3 ledger aliases original ledger')
    cfg=json.loads((folder/'config.json').read_text())
    original_connect=m.connect
    m.connect=lambda path=None:original_connect(path if path is not None else ledger)
    db=m.connect();m.init(db,cfg)
    expected=execution_fingerprint()
    row=db.execute("SELECT value FROM meta WHERE key='execution_v3_hash'").fetchone()
    if row is None or row[0]!=expected:raise ValueError('Execution-v3 fingerprint mismatch; explicit migration required')
    row=db.execute("SELECT value FROM meta WHERE key='execution_v3_policy'").fetchone()
    policy=json.loads(row[0]) if row else None
    if not isinstance(policy,dict) or policy.get('version')!=VERSION:raise ValueError('Missing causal-replay authorization')
    if phase=='status':
        report={'status':'status_only','execution_version':VERSION,'ledger_path':str(ledger),'statistics':m.statistics(db)}
        db.close();return report
    db.close()
    original_exports=m.exports
    def export(db,report):
        report['execution_version']=VERSION;report['ledger_path']=str(ledger)
        st=report['statistics'];st['mode']='paper_simulation_with_authorized_causal_replay'
        st['replayed_sessions']=db.execute("SELECT count(*) FROM sessions WHERE status='authorized_causal_replay_executed'").fetchone()[0]
        st['legacy_retrospective_sessions']=db.execute("SELECT count(*) FROM sessions WHERE status='user_authorized_retrospective_executed'").fetchone()[0]
        st['prospective_sessions']=db.execute("SELECT count(*) FROM sessions WHERE status='recorded_plan_executed'").fetchone()[0]
        export_dir.mkdir(parents=True,exist_ok=True)
        original_p=m.P
        try:
            m.P=export_dir;original_exports(db,report)
        finally:m.P=original_p
        path=export_dir/'DAILY_STATUS.md'
        text=path.read_text().replace('模式：事前记录的模拟账户；不是实际成交或历史回测。',
            '模式：按冻结规则连续模拟，包含获授权的漏跑日顺序补算；不是实盘成交或纯前向样本外绩效。')
        text+=f"\n- 已补算交易日：{st['replayed_sessions']}\n- 原逐日授权补记：{st['legacy_retrospective_sessions']}\n"
        path.write_text(text)
    m.exports=export
    def checked_reconcile(db,cfg,frames,manifest,now):
        previous_state=m.latest_state(db)
        for symbol,position in previous_state['positions'].items():
            check_position=copy.deepcopy(position)
            pending=frames[symbol].loc[frames[symbol].index>pd.Timestamp(previous_state['last_date']),'Stock Splits']
            if not all(math.isfinite(float(v)) and float(v)>=0 for v in pending):
                raise ValueError('Invalid pending corporate action: '+symbol)
            factor=float(pending.where(pending>0,1.).prod())
            if not math.isfinite(factor) or factor<=0:raise ValueError('Invalid pending split factor')
            # Guard in the provider's current split units; do not mutate account state.
            check_position['entry_price']/=factor
            validate_entry_basis(db,symbol,check_position,frames[symbol],manifest)
        return reconcile(m,db,cfg,frames,manifest,now)
    m.reconcile=checked_reconcile
    now=pd.Timestamp.now(tz='UTC');cutoff=m.completed_session(now)
    if snapshot:source=Path(snapshot)
    elif refresh:
        try:source=m.download(cutoff)
        except Exception as exc:source=fallback_seed(CACHE,cutoff,exc)
    else:source=Path(json.loads((CACHE/'latest.json').read_text())['folder'])
    snapshot=prepare_snapshot(source,m.schedule(now).index)
    report=m.run('close',False,snapshot)
    data,raw,frames,manifest=load_snapshot(snapshot)
    db=m.connect();m.init(db,cfg);state=m.latest_state(db)
    now=pd.Timestamp.now(tz='UTC');calendar=m.schedule(now)
    target=calendar.index[calendar.index>pd.Timestamp(manifest['as_of'])][0]
    opening=calendar.at[target,'open'];revisions=[]
    for symbol,position in state['positions'].items():
        revision=validate_entry_basis(db,symbol,position,frames[symbol],manifest)
        if revision is not None:revisions.append(revision)
    report.update(phase='evening_v3',target_session=str(target.date()),verified_entry_price_revisions=revisions)
    old=db.execute('SELECT payload FROM plans WHERE session=?',(str(target.date()),)).fetchone()
    if old:
        decision=json.loads(old[0])
        if decision.get('decision_state_sha256')!=hashlib.sha256(m.dumps(state).encode()).hexdigest():
            raise ValueError('Saved v3 next-session plan is based on a different account state')
        report.update(plan=decision,status='existing_immutable_plan',plan_origin=decision.get('origin','prospective'))
    elif target<pd.Timestamp(cfg['paper_start']):report.update(status='paper_start_pending',plan=None)
    elif now<opening-pd.Timedelta(minutes=1):
        decision=build_decision(m,state,target,calendar,frames,manifest,cfg)
        decision['origin']='prospective'
        decision['execution_version']=VERSION
        m.save_plan(db,decision,now,Path(snapshot),opening)
        report.update(plan=decision,status='next_open_plan_frozen',plan_origin='prospective')
    else:
        # Current/incomplete sessions are reconciled after complete OHLC is verified.
        report.update(plan=None,status='awaiting_completed_session_for_causal_replay')
    report['fill_status']='Opening-price proxy; completed sessions only; no broker connection.'
    report['statistics']=m.statistics(db)
    m.exports(db,report);db.close();return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--account',required=True);p.add_argument('--phase',default='auto')
    p.add_argument('--no-refresh',action='store_true');p.add_argument('--snapshot');p.add_argument('--ledger');p.add_argument('--export-dir')
    a=p.parse_args()
    try:print(json.dumps(run(Path(a.account),a.phase,not a.no_refresh,a.snapshot,a.ledger,a.export_dir),ensure_ascii=False,indent=2,allow_nan=False))
    except Exception as exc:
        print(json.dumps({'status':'ERROR_NO_NEW_PLAN','error':str(exc),'execution_version':VERSION},ensure_ascii=False));raise SystemExit(1)

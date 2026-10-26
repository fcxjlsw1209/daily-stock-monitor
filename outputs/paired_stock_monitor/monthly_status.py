"""Read-only execution context for independently frozen monthly observers.

This presentation layer never turns stale inputs into a successful observer run.
It proves the narrow pre-first-signal, untouched-cash case from the actual ledger.
It creates neither plans, daily valuations nor transactions.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

import exchange_calendars as xc
import pandas as pd


_MODES = {
    'monthly_momentum_buffer5_net_v1': 'prior_session_to_monthend_close',
    'monthly_momentum_regime_paper_v1': 'monthend_to_next_open',
    'monthly_momentum_industry_paper_v1': 'monthend_to_next_open',
    'monthly_momentum_stability_paper_v1': 'monthend_to_next_open',
}
_SCHEMA = {
    'meta': {'key', 'value'},
    'plans': {'session', 'created_utc', 'payload'},
    'daily': {'session', 'payload'},
    'fills': {'id', 'payload'},
    'closed': {'id', 'payload'},
}


def _read_json(path):
    return json.loads(path.read_text())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _amount(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Invalid numeric ' + label)
    return float(value)


def _equal_amount(value, expected, label):
    if abs(_amount(value, label) - expected) > max(1e-7, abs(expected) * 1e-10):
        raise ValueError('Cash-state mismatch: ' + label)


def _cash_state(state, initial, label, *, positions_required=True):
    if not isinstance(state, dict):
        raise ValueError('Unknown cash-state schema: ' + label)
    if positions_required and 'positions' not in state:
        raise ValueError('Missing positions in ' + label)
    if positions_required and 'positions' in state and state['positions'] != {}:
        raise ValueError('Nonempty or unknown positions in ' + label)
    for key in ('cash', 'equity'):
        if key not in state:
            raise ValueError('Missing ' + key + ' in ' + label)
        _equal_amount(state[key], initial, label + '.' + key)
    for key in ('initial', 'initial_capital'):
        if key in state:
            _equal_amount(state[key], initial, label + '.' + key)
    for key in ('fees', 'realized_pnl', 'dividends'):
        if key in state:
            _equal_amount(state[key], 0., label + '.' + key)


def _first_schedule(folder, config):
    strategy_id = config.get('strategy_id')
    if strategy_id not in _MODES:
        raise ValueError('Unknown monthly strategy/schema')
    mode = _MODES[strategy_id]
    rules = (folder / 'RULES.md').read_text()
    if mode == 'prior_session_to_monthend_close':
        match = re.search(r'首个计划调仓日为\s*(\d{4}-\d{2}-\d{2})[，,]\s*信号日为\s*(\d{4}-\d{2}-\d{2})', rules)
        if not match:
            raise ValueError('Cannot verify first buffer5 schedule in RULES.md')
        execution_text, signal_text = match.groups()
        for key, expected in [('first_signal', signal_text), ('first_execution', execution_text)]:
            if key in config and config[key] != expected:
                raise ValueError('Config/RULES schedule mismatch: ' + key)
    else:
        signal_text, execution_text = config.get('first_signal'), config.get('first_execution')
        if not all(isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value)
                   and value in rules for value in (signal_text, execution_text)):
            raise ValueError('Cannot verify first monthly schedule in config and RULES.md')
    signal, execution = pd.Timestamp(signal_text), pd.Timestamp(execution_text)
    cal = xc.get_calendar('XNYS')
    if not cal.is_session(signal) or not cal.is_session(execution):
        raise ValueError('First signal/execution is not an XNYS session')
    if signal < pd.Timestamp(config['start']):
        raise ValueError('First signal precedes account start')
    if mode == 'prior_session_to_monthend_close':
        if cal.previous_session(execution) != signal or cal.next_session(execution).month == execution.month:
            raise ValueError('Invalid prior-session/monthend execution schedule')
    elif cal.next_session(signal) != execution or signal.month == execution.month:
        raise ValueError('Invalid monthend/next-open execution schedule')
    # Verify the configuration against the frozen source record, without changing it.
    frozen = _read_json(folder / 'FREEZE.json')
    if not isinstance(frozen, dict):
        raise ValueError('Unknown monthly freeze manifest schema')
    for name in ('config.json', 'RULES.md'):
        matches = [digest for path, digest in frozen.items()
                   if Path(path).parts[-2:] == (folder.name, name)]
        if name == 'config.json' and len(matches) != 1:
            raise ValueError('Missing or ambiguous frozen monthly config')
        if matches and (len(matches) != 1 or _sha(folder / name) != matches[0]):
            raise ValueError('Frozen monthly input changed: ' + name)
    return signal.date(), execution.date(), mode


def _ledger_audit(folder, config, report):
    """A snapshot read; mode=ro must not create a missing SQLite database."""
    path = folder / 'paper_ledger.sqlite'
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=2.)
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        expected = set(_SCHEMA)
        if _MODES[config['strategy_id']] == 'prior_session_to_monthend_close':
            expected.remove('closed')
        if tables != expected:
            raise ValueError('Unknown monthly ledger tables: ' + ','.join(sorted(tables)))
        for table in sorted(tables):
            columns = {row[1] for row in db.execute('PRAGMA table_info(' + table + ')')}
            if columns != _SCHEMA[table]:
                raise ValueError('Unknown monthly ledger columns: ' + table)
        counts = {table: db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
                  for table in sorted(tables)}
        meta_rows = db.execute('SELECT key,value FROM meta').fetchall()
        if len({key for key, _ in meta_rows}) != len(meta_rows):
            raise ValueError('Duplicate monthly ledger metadata')
        meta = dict(meta_rows)
        if meta.get('strategy_id') != config['strategy_id']:
            raise ValueError('Monthly ledger strategy mismatch')
        if set(meta) - {'strategy_id', 'last_result', 'anchors'}:
            raise ValueError('Unknown monthly ledger metadata keys')
        audit = {'read_only': True, 'table_counts': counts, 'empty_trade_ledger': False,
                 'untouched_cash_verified': False}
        if counts['plans'] or counts['fills'] or counts.get('closed', 0):
            return audit
        initial = _amount(config.get('initial_capital'), 'initial_capital')
        if initial <= 0:
            raise ValueError('Nonpositive initial capital')
        if 'anchors' in meta and json.loads(meta['anchors']) != {}:
            raise ValueError('Unexpected normalization anchors in untouched account')
        if 'last_result' in meta:
            previous = json.loads(meta['last_result'])
            if not isinstance(previous, dict):
                raise ValueError('Unknown prior-result schema')
            if previous.get('plan') is not None or previous.get('orders', []):
                raise ValueError('Trade activity in prior result but not ledger')
            _cash_state(previous.get('state'), initial, 'meta.last_result.state')
        for session, payload in db.execute('SELECT session,payload FROM daily'):
            row = json.loads(payload)
            if not isinstance(row, dict) or row.get('date') != session:
                raise ValueError('Unknown daily payload schema')
            if _MODES[config['strategy_id']] == 'prior_session_to_monthend_close':
                if type(row.get('positions')) is not int or row['positions'] != 0:
                    raise ValueError('Nonempty/unknown buffer5 daily positions')
                _cash_state(row, initial, 'daily.' + session, positions_required=False)
                for key in ('invested', 'fees_cumulative', 'dividends_cumulative', 'return', 'exposure'):
                    if key in row:
                        _equal_amount(row[key], 0., 'daily.' + key)
            else:
                _cash_state(row, initial, 'daily.' + session)
        if report.get('plan') is not None or report.get('orders', []):
            raise ValueError('Reported trade activity but empty ledger')
        if 'state' in report:
            _cash_state(report['state'], initial, 'report.state')
        if report.get('strategy_id', config['strategy_id']) != config['strategy_id']:
            raise ValueError('Reported monthly strategy mismatch')
        audit.update(empty_trade_ledger=True, untouched_cash_verified=True,
                     cash=initial, initial_capital=initial)
        return audit
    finally:
        db.close()


def enrich_monthly_status(report, account_folder, now=None):
    """Return an enriched copy, preserving original observer status and errors.

    ``no_trade_due_today`` is only True before the first signal date when the
    ledger proves untouched cash. None means this helper makes no such claim;
    the original frozen observer remains authoritative after that boundary.
    """
    out = deepcopy(report)
    out.update(execution_phase='UNVERIFIED', no_trade_due_today=None)
    out.pop('execution_status_error', None)
    out.pop('execution_ledger_audit', None)
    out.setdefault('data_health', 'STALE' if str(out.get('status', '')).startswith('WAITING_CURRENT_')
                   else 'ERROR' if str(out.get('status', '')).startswith('ERROR') else 'NOT_ASSESSED')
    try:
        now = now or datetime.now(timezone.utc)
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError('Timezone-aware current datetime required')
        today = now.astimezone(ZoneInfo('America/New_York')).date()
        folder = Path(account_folder)
        config = _read_json(folder / 'config.json')
        if not isinstance(config, dict):
            raise ValueError('Unknown monthly configuration schema')
        signal, execution, mode = _first_schedule(folder, config)
        out.update(first_signal_date=signal.isoformat(), first_execution_date=execution.isoformat(),
                   execution_schedule_mode=mode, execution_checked_utc=now.astimezone(timezone.utc).isoformat())
        if today < signal:
            out.update(next_signal_date=signal.isoformat(), next_execution_date=execution.isoformat())
        else:
            for target, source in [('next_signal_date', 'required_signal_date'), ('next_execution_date', 'next_rebalance')]:
                out.pop(target, None)
                if isinstance(out.get(source), str):
                    out[target] = out[source]
        audit = _ledger_audit(folder, config, out)
        out['execution_ledger_audit'] = audit
        if today >= signal:
            out['execution_phase'] = 'FIRST_SIGNAL_REACHED'
            return out
        if not audit['untouched_cash_verified']:
            out['execution_phase'] = 'EXISTING_LEDGER_ACTIVITY'
            return out
        out.update(execution_phase='WAITING_FIRST_SIGNAL', no_trade_due_today=True,
                   cash=audit['cash'], initial_capital=audit['initial_capital'])
        return out
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error) as exc:
        out['execution_status_error'] = str(exc)
        return out

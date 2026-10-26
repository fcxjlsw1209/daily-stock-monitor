"""Round172: frozen Local67 plus annual operating-margin confirmation.

All actual feature calculations and returns require a prospectively bound PLAN.
The test phase uses only synthetic arrays. No model is fitted in this experiment.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from decimal import Decimal

import operating_margin172_v4_common as sources
import ptr169_rebuild_evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_operating_margin172'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_price_distribution171'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67', 'annual_margin_confirmation', 'annual_income_confirmation')
RAW = {NAMES[1]: 'candidate', NAMES[2]: 'control'}
TRAIN, DOSE = 88, .01
TEST_RECEIPT = ROOT / 'work/operating_margin172_TESTS.json'
PREPARED_FILES = {'FEATURES.npz', 'RANKS_DELTAS.npz', 'FEATURE_AUDIT.json',
                  'ORDERS.json', 'META.json', 'SOURCE_PARITY.json', 'INDEPENDENT_PREPARE.json', 'SOURCE_RECORDS.json'}
sha, read, require, dump = ev.sha, ev.read, ev.require, ev.dump


def now():
    return datetime.now(timezone.utc).isoformat()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def local():
    return module('operating_margin172_original', LOCAL / 'study.py')


def contract():
    return dict(arms=list(NAMES), candidate=NAMES[1], control_nonpromotable=NAMES[2],
        period=['2016-08-17', '2026-09-24'], training_months=88, later_months=34,
        hypothesis='Small confirmation from same-company reported annual operating-margin improvement can distinguish durable long-term momentum.',
        measurement='Candidate=sign(OI_current/Revenue_current - OI_comparative/Revenue_comparative); control=sign(OI_current/days_current - OI_comparative/days_comparative), exact rational signs from the SAME annual consolidated report.',
        source_scope={'MSFT': list(range(2015,2027)), 'ABBV': list(range(2013,2026)), 'AAPL': list(range(2016,2026))},
        availability='Latest report known at signal close using publication-next-session clock; max400 calendar days inclusive since known_session; no fallback from held latest report. Both annual periods must have330..400days, positive revenue and finite signed OI. AAPL FY2016 unknown comparative start stays unavailable. Other companies/ETFs neutral.',
        scoring='Add fixed0.01*sign to frozen Local67 pre-score then exactly one original episode. No cross-sectional re-ranking of signs, no eligibility filter in feature computation; first partial allocation gets zero delta.',
        portfolio='Unchanged long-momentum main score, skip42 core, historical proxy Top32+theme17, original eligibility and EW3/two halves/under2 cash. Fixed adjusted units intramonth; difference rebalance, no terminal liquidation.',
        selection=dict(first88_only=True, minimum_cagr_gain=.005, open_5bp_noninferior=True, sharpe_tolerance=.05,
            drawdown_tolerance=.03, era_cagr_tolerance=.025, changed_set_months_min=3, changed_set_months_each_era_min=1,
            require_noninferior_to_income_only=True, later34_for_selection=False),
        limitation='Conditional historical research, reused development sample and estimated membership; retrospectively retrieved original-dated documents, no pristine first-public archive or complete amendment history. Same-table currency units cancel in ratios, ISO not inferred. Reported margin is affected by acquisitions, disposals, FX and accounting estimates; not organic or constant-scope profitability.',
        historical_best_tracking='First88-only frozen selection. Historical best uses full-period descriptive performance among training-selected candidates, not independent validation or observer promotion.',
        new_model_fits=0, new_arms=2, registered_trials=3, execution_paths=9, no_observer_writes=True,
        original_pool_preserved=True, no_parameter_search=True)


def source_files():
    original = ROOT / 'outputs/monthly_operating_margin172_sources'
    return [sources.OUT / 'PLAN.json', sources.OUT / 'independent/REVIEW.json',
            sources.OUT / 'independent/COMPLETION.json', original / 'v2/html/EXTRACTION.json',
            original / 'aapl/REPORTS.json', original / 'aapl/VISUAL_QA.json']


def source_ready():
    files = source_files()
    p = sources.gate(sha(files[0])); review = read(files[1]); completion = read(files[2])
    require(completion['passed'] is True and completion['review_sha256'] == sha(files[1]), 'Independent source review not bound')
    require(review['passed'] is True and review['source_reports'] == 35 and review['checked_annual_cells'] == 140,
            'Complete independent source review required')
    verify_source_review(review)
    require(review['plan_sha256'] == sha(files[0]), 'Source review PLAN mismatch')
    require(review['producer_outputs'] == {str(files[3]): sha(files[3]), str(files[4]): sha(files[4])}, 'Source outputs changed')
    visual = read(files[5])
    require(visual['passed'] is True and visual['pages_viewed'] == 10 and
            visual['annual_cells_compared'] == 40 and visual['reports_sha256'] == sha(files[4]),
            'Apple original page review incomplete')
    require(len(visual['records']) == 10 and all(r['matches_report'] is True for r in visual['records']),
            'Incomplete per-page visual review')
    for row in visual['records']:
        require(sha(ROOT / row['viewed_path']) == row['viewed_sha256'], 'Reviewed PDF render changed')
    return p, review


def verify_source_review(review):
    expected = {(s, y) for s, years in contract()['source_scope'].items() for y in years}
    rows = review.get('reports', [])
    require(review.get('independent_of_production_extractors') is True and len(rows) == 35,
            'Independent complete per-report evidence required')
    require({(r.get('issuer'), r.get('fiscal_year')) for r in rows} == expected,
            'Independent source scope differs')
    require(all(r.get('passed') is True and r.get('checked_annual_cells') == 4 and
                r.get('next_session_clock_rederived') is True for r in rows),
            'Incomplete independent per-report cell/clock checks')


def source_records():
    _, review = source_ready(); files = source_files()
    html, apple = read(files[3])['reports'], read(files[4])['records']
    checked = {(r['issuer'], r['fiscal_year']): r for r in review['reports']}
    records = []
    for row in html + apple:
        symbol = row.get('issuer', row.get('symbol')); year = row['fiscal_year']; checked_row = checked[symbol, year]
        fields = row['fields'] if symbol == 'AAPL' else row['cells']
        record = dict(symbol=symbol, fiscal_year=year,
            report_id='AAPL_FY'+str(year) if symbol == 'AAPL' else row['report_id'],
            release_date=checked_row['release_date'], known_session=checked_row['known_session'],
            source_sha256=checked_row['source_sha256'], hold_reasons=list(row.get('holds', [])))
        for role in ('current','comparative'):
            period = fields['revenue'][role]['period'] if symbol == 'AAPL' else row['periods'][role]
            record[role] = dict(revenue=fields['revenue'][role]['amount_decimal'],
                operating_income=fields['operating_income'][role]['amount_decimal'], days=period.get('days_inclusive'))
        record['usable'] = all(record[role]['days'] is not None for role in ('current','comparative'))
        records.append(record)
    require(len(records) == 35 and [(r['symbol'],r['fiscal_year']) for r in records if not r['usable']] == [('AAPL',2016)], 'Unexpected source qualification scope')
    return records, html, apple, review


def code_files():
    return [ROOT / 'work' / n for n in ('operating_margin172_host.py',
        'test_operating_margin172_host.py', 'operating_margin172_kernel.py',
        'test_operating_margin172_kernel.py', 'operating_margin172_factor_independent.py',
        'test_operating_margin172_factor_independent.py', 'ptr169_rebuild_evaluate.py',
        'ptr169_results_independent.py', 'price_distribution171_independent.py')]


def verify(pins):
    require(isinstance(pins, dict) and pins, 'Nonempty source pins required')
    for name, digest in pins.items():
        path = Path(name)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require('CloudStorage' not in path.parts and (path.is_relative_to(ROOT) or
                str(path).startswith('/Library/Developer/CommandLineTools/')), 'Unapproved source path')
        require(path.is_file() and sha(path) == digest, 'Changed or missing source: ' + str(path))


def bind(pins, path, digest=None):
    key, actual = str(path), sha(path)
    require(digest is None or actual == digest, 'Source differs from declared digest: ' + key)
    require(key not in pins or pins[key] == actual, 'Conflicting inherited source digest: ' + key)
    pins[key] = actual


def state_check(state):
    require(len(state['completed']) == 169 and sum(r['trials'] for r in state['completed']) == 1022,
            'Expected169 completed rounds /1022 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round171 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_OPERATING_MARGIN172_' + tag + '_' + expected + '.json')
    if not backup.exists():
        with backup.open('xb') as stream:
            stream.write(before)
    require(sha(backup) == expected, 'Central backup mismatch')
    with tempfile.NamedTemporaryFile(dir=CENTRAL.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(after, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
        stream.flush(); os.fsync(stream.fileno())
    try:
        require(CENTRAL.read_bytes() == before, 'Concurrent central update before commit')
        os.replace(temporary, CENTRAL)
    finally:
        temporary.unlink(missing_ok=True)


def tests():
    names = ('test_operating_margin172_host.py', 'test_operating_margin172_kernel.py',
             'test_operating_margin172_factor_independent.py')
    modules = [module('test_om172_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    require(result.wasSuccessful(), 'Synthetic validation failed')
    receipt = TEST_RECEIPT
    require(not receipt.exists(), 'Test receipt is write-once')
    dump(receipt, dict(passed=True, synthetic_only=True, tests=result.testsRun,
        runtime=ev.runtime(), code_sha256={str(p): sha(p) for p in code_files()}))


def register():
    require(not OUT.exists(), 'Fresh experiment directory required')
    source_plan, _ = source_ready()
    before = CENTRAL.read_bytes(); state = json.loads(before); state_check(state)
    require(not state['registered_queue'], 'Existing registered experiment')
    predecessor = read(PRIOR / 'COMPLETION.json')
    require(state['completed'][-1]['completion']['sha256'] == sha(PRIOR / 'COMPLETION.json') and
            predecessor['result_completion_sha256'] == sha(PRIOR / 'results/COMPLETION.json'),
            'Predecessor differs from registered completion')
    require(predecessor['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED' and
            predecessor['plan_sha256'] == sha(PRIOR / 'PLAN.json'), 'Predecessor not verified')
    validation = TEST_RECEIPT; t = read(validation)
    require(t['passed'] is True and t['synthetic_only'] is True and t['runtime'] == ev.runtime(), 'Synthetic tests required')
    require(t['code_sha256'] == {str(p): sha(p) for p in code_files()}, 'Code changed since tests')
    pins = dict(read(PRIOR / 'PLAN.json')['input_sha256'])
    raw = read(LOCAL / 'PLAN.json')['raw_source_hashes']
    calendar_sources = read(CALENDAR)['provenance']['source_hashes']
    for p in code_files() + [PRIOR / 'PLAN.json', PRIOR / 'COMPLETION.json', validation,
                            CALENDAR, PROTECTED, LOCAL / 'PLAN.json']:
        bind(pins, p)
    for mapping in (raw, calendar_sources):
        for name, digest in mapping.items(): bind(pins, Path(name), digest)
    for name in ('DATA.npz', 'META.json', 'local_linear_ORDERS.json', 'INCUMBENT_POLICY.json'):
        bind(pins, LOCAL / name)
    for name, digest in ev.ENGINE_HASHES.items(): bind(pins, ev.ENGINE_DIR / name, digest)
    for p in source_files(): bind(pins, p)
    for p in list(source_plan['input_sha256']): bind(pins, Path(p), source_plan['input_sha256'][p])
    for p in list(ROOT.glob('work/operating_margin172*common.py')): bind(pins, p)
    verify(pins); protected = read(PROTECTED); verify(protected)
    require(len(protected) == 21444, 'Incomplete protected archive manifest')
    OUT.mkdir()
    with (OUT / 'STATE_BEFORE.json').open('xb') as stream: stream.write(before)
    plan = dict(schema='operating_margin172_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, raw_source_hashes=raw,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_feature_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_new_returns_before_registration=0, protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_financial_features',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare fixed annual-margin confirmation and independent audit; select only on first88 before later34 diagnostics.')
    commit_state(before, state, 'register')
    print(json.dumps(dict(status='REGISTERED', plan_sha256=sha(OUT / 'PLAN.json'))), flush=True)


def gate(expected):
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['contract'] == contract() and p['runtime'] == ev.runtime(), 'Frozen contract/runtime changed')
    verify(p['input_sha256']); source_ready()
    require(sha(OUT / 'STATE_BEFORE.json') == p['prior_central_sha256'], 'Changed pre-registration state')
    state = read(CENTRAL)
    require(state['registered_queue'] == [dict(directory=OUT.name, plan_sha256=expected)], 'Central registration mismatch')
    state_check(state)
    return p


def rank_and_delta(values, n):
    require(set(values) == {'candidate','control','available'}, 'Feature keys changed')
    available = np.asarray(values['available'])
    require(available.ndim == 2 and available.dtype == np.dtype(bool) and type(n) is int and 0 <= n <= len(available), 'Invalid mask/prefix')
    ranks, deltas = {}, {}
    for arm, key in RAW.items():
        raw = np.asarray(values[key]); require(raw.shape == available.shape, 'Feature shape mismatch')
        part = np.asarray(raw[:n], float)
        require(np.array_equal(np.isfinite(part), available[:n]) and not np.isinf(part).any(), 'Common availability mismatch')
        require(np.isin(part[available[:n]], [-1.,0.,1.]).all(), 'Only exact signs permitted')
        ranks[arm] = np.where(available[:n], part, 0.)
        deltas[arm] = DOSE * ranks[arm]
        if n: deltas[arm][0] = 0.
    return ranks, deltas


def decisions(study, arrays, meta, values, n):
    policy = read(LOCAL / 'INCUMBENT_POLICY.json')
    choices, scores = study.choices(study.configs()[4], arrays, meta, n, policy=policy)
    orders = {NAMES[0]: study.rp.make_orders(choices, scores, meta)}
    ranks, deltas = rank_and_delta(values, n)
    for arm in NAMES[1:]:
        pre = study.modified_pre_score(study.configs()[4], arrays, n)
        view = pre[:n].copy(); changed = deltas[arm] != 0.
        view[changed] += deltas[arm][changed]; pre[:n] = view
        chosen, score = study.adapter.score_and_choose('incumbent66', 0., np.full(arrays['baseline'].shape, np.nan),
            arrays, meta, n, pre_score=pre, effects=arrays['effects'], episode_policy=policy,
            boundary_selector=study.ep.select_boundary)
        orders[arm] = study.rp.make_orders(chosen, score, meta)
    return orders, ranks, deltas


def require_source_audit(check):
    require(check.get('passed') is True and check.get('checked_records') == 35 and
            check.get('source_reports') == 35 and check.get('checked_annual_cells') == 140,
            'Independent normalized source coverage incomplete')
    for key in ('exact_source_cells', 'exact_source_clocks', 'exact_source_durations',
                'source_hold_preservation', 'compares_pinned_objects_only'):
        require(check.get(key) is True, 'Independent source evidence missing ' + key)
    require(check.get('unavailable_records') == [['AAPL', 2016]] and
            check.get('financial_ratios_computed') is False, 'Source scope changed')


def require_feature_audit(check, values):
    require(check.get('passed') is True, 'Independent feature audit failed')
    for key in ('rational_scalar_oracle', 'exact_common_availability', 'independent_raw_signs_equal', 'no_future_or_outcome_inputs', 'audit_checked'):
        require(check.get(key) is True, 'Independent feature evidence missing ' + key)
    require(check.get('events') == values['available'].shape[0] == 122 and
            check.get('symbols_per_event') == values['available'].shape[1] == 49, 'Independent feature coverage')
    require(check.get('available_cells') == int(values['available'].sum()) and
            check.get('unavailable_cells') == int((~values['available']).sum()), 'Independent mask coverage')
    require(check.get('computes_returns') is False, 'Feature audit must not compute returns')


def require_rank_audit(check, deltas):
    require(check.get('passed') is True and check.get('full_candidate_universe') is True and
            check.get('initial_delta_zero') is True, 'Independent rank audit failed')
    for key in ('common_available_only', 'no_eligibility_or_model_filter', 'no_cross_sectional_ranking'):
        require(check.get(key) is True, 'Independent rank evidence missing ' + key)
    require(check.get('dose') == DOSE, 'Independent rank dose differs')
    require(check.get('changed_cells') == {k: int(np.count_nonzero(v)) for k, v in deltas.items()}, 'Independent rank coverage')


def require_outcome_audit(check):
    for key in ('passed', 'independent_accounts', 'independent_of_production_model_and_engine'):
        require(check.get(key) is True, 'Independent outcome audit missing ' + key)
    require(check.get('checked_paths') == 9 and check.get('checked_slices') == 45 and
            check.get('terminal_liquidation') is False, 'Independent outcome coverage')
    require(check.get('accounting_source_sha256') == sha(ROOT / 'work/ptr169_results_independent.py'), 'Accounting source changed')
    require(set(check.get('paths', {})) == set(NAMES), 'Missing independent arms')
    for paths in check['paths'].values():
        require(set(paths) == {p[0] for p in ev.PATHS}, 'Missing independent paths')
        for path in paths.values():
            require(path.get('passed') is True and path.get('monthly_executions') == 122 and
                    set(path.get('slices', {})) == {'full122', 'first88', 'early53', 'recent35', 'later34'}, 'Incomplete path audit')


def prepare(expected):
    plan = gate(expected); require(not (OUT / 'prepared').exists(), 'Preparation is write-once')
    import operating_margin172_kernel as kernel
    import operating_margin172_factor_independent as independent
    study = local(); meta, arrays, px, op = study.load()
    require(len(meta['dates']) == 122 and arrays['baseline'].shape == (122, 49), 'Original122x49 required')
    require(all(len(s) == 49 for s in meta['symbols']), 'Original monthly candidate widths changed')
    before = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}
    source, info = study.arch.old.panel_load(meta['etfs'])
    raw_hashes = study.raw_source_hashes(info)
    require(raw_hashes == plan['raw_source_hashes'], 'Original raw source lineage changed')
    verify(raw_hashes)
    parity = study.verify_panel_parity(source, px, op)
    records, html, apple, review = source_records()
    source_check = independent.verify_source_records(records, html, apple, review)
    require_source_audit(source_check)
    events = list(zip(meta['dates'], meta['symbols']))
    values, audit = kernel.build_features(records, events)
    check = independent.verify_features(records, events, values, audit)
    require_feature_audit(check, values)
    orders, ranks, deltas = decisions(study, arrays, meta, values, 122)
    require(orders[NAMES[0]] == read(LOCAL / 'local_linear_ORDERS.json'), 'Original full order parity failed')
    rank_check = independent.verify_rank_deltas(values, ranks, deltas)
    require_rank_audit(rank_check, deltas)
    for n in range(123):
        small, _ = kernel.build_features(records, events, limit=n)
        for key in values: np.testing.assert_array_equal(small[key], values[key][:n])
        oo, rr, dd = decisions(study, arrays, meta, values, n)
        require(all(oo[k] == orders[k][:n] for k in NAMES), 'Prefix order parity failed')
        for k in RAW:
            np.testing.assert_array_equal(rr[k], ranks[k][:n]); np.testing.assert_array_equal(dd[k], deltas[k][:n])
    require(before == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}, 'Original arrays modified')
    gate(expected)
    dest = OUT / 'prepared'; dest.mkdir()
    np.savez_compressed(dest / 'FEATURES.npz', **values)
    np.savez_compressed(dest / 'RANKS_DELTAS.npz', **{'signal_' + k: v for k, v in ranks.items()}, **{'delta_' + k: v for k, v in deltas.items()})
    dump(dest / 'SOURCE_RECORDS.json', records)
    dump(dest / 'FEATURE_AUDIT.json', audit); dump(dest / 'ORDERS.json', orders); dump(dest / 'META.json', meta)
    dump(dest / 'SOURCE_PARITY.json', dict(raw_source_hashes=raw_hashes, execution_prices=parity))
    dump(dest / 'INDEPENDENT_PREPARE.json', dict(source=source_check, feature=check, rank=rank_check, prefixes=123,
        original_orders_exact=True, original_arrays_unchanged=True, new_model_fits=0))
    dump(dest / 'COMPLETION.json', dict(status='COMPLETE', plan_sha256=expected, created_utc=now(),
        artifacts_sha256={p.name: sha(p) for p in dest.iterdir()}, new_returns_computed=False, new_model_fits=0))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED', completion_sha256=sha(dest / 'COMPLETION.json'))), flush=True)


def verify_prepared_evidence(c, values, deltas, audit):
    require(set(c['artifacts_sha256']) == PREPARED_FILES, 'Prepared artifact set incomplete')
    require_source_audit(audit['source'])
    require_feature_audit(audit['feature'], values)
    require_rank_audit(audit['rank'], deltas)
    require(audit.get('prefixes') == 123 and audit.get('original_orders_exact') is True and
            audit.get('original_arrays_unchanged') is True and audit.get('new_model_fits') == 0,
            'Prepared invariants not verified')


def prepared(expected):
    gate(expected); dest = OUT / 'prepared'; c = read(dest / 'COMPLETION.json')
    require(c['status'] == 'COMPLETE' and c['plan_sha256'] == expected, 'Preparation incomplete')
    require(set(c['artifacts_sha256']) == PREPARED_FILES, 'Prepared artifact set incomplete')
    for name, digest in c['artifacts_sha256'].items(): require(sha(dest / name) == digest, 'Changed prepared artifact ' + name)
    with np.load(dest / 'FEATURES.npz') as data: values = {k: data[k] for k in data.files}
    with np.load(dest / 'RANKS_DELTAS.npz') as data: deltas = {k: data['delta_' + k] for k in RAW}
    verify_prepared_evidence(c, values, deltas, read(dest / 'INDEPENDENT_PREPARE.json'))
    return read(dest / 'META.json'), read(dest / 'ORDERS.json')


def train_metrics(engine, eq, meta):
    answer = {}
    for name, lo, hi in (('training', 0, 87), ('early', 0, 52), ('recent', 53, 87)):
        part = eq.loc[meta['dates'][lo]:meta['ends'][hi]]
        answer[name] = engine.stats(part / part.iloc[0] * 100000.)
    return answer


def guard(candidate, reference, gain):
    return (candidate['training']['cagr'] >= reference['training']['cagr'] + gain - 1e-12 and
        candidate['training']['sharpe'] >= reference['training']['sharpe'] - .05 and
        candidate['training']['max_drawdown'] >= reference['training']['max_drawdown'] - .03 and
        all(candidate[e]['cagr'] >= reference[e]['cagr'] - .025 for e in ('early', 'recent')))


def select_first88(metrics, orders):
    changed = [i for i in range(TRAIN) if set(orders[NAMES[1]][i]['symbols']) != set(orders[NAMES[0]][i]['symbols'])]
    guards = dict(close_vs_incumbent=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[0]]['same_close_zero'], .005),
        open_vs_incumbent=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[0]]['next_open_cost'], 0.),
        close_vs_income_only=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_income_only=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
        changed_support=len(changed) >= 3 and any(i < 53 for i in changed) and any(i >= 53 for i in changed))
    return dict(primary=NAMES[1] if all(guards.values()) else NAMES[0], admitted=all(guards.values()),
        guards=guards, changed_months=changed, first88_only=True, later34_used=False)


def select(expected):
    meta, orders = prepared(expected); require(not (OUT / 'selected').exists(), 'Selection is write-once')
    px, op = ev.prices(); engine = ev.engine(); metrics = {}
    for arm in NAMES:
        metrics[arm] = {}
        for path, mode, cost in (ev.PATHS[0], ev.PATHS[2]):
            _, eq, _, _ = engine.replay(px, op, orders[arm][:TRAIN], cost=cost, execution=mode,
                start=pd.Timestamp(meta['dates'][0]), end=pd.Timestamp(meta['ends'][87]))
            require(str(eq.index[-1].date()) == meta['ends'][87], 'Selection escaped first88 cutoff')
            metrics[arm][path] = train_metrics(engine, eq, meta)
    result = select_first88(metrics, orders); gate(expected)
    dest = OUT / 'selected'; dest.mkdir(); dump(dest / 'TRAIN_RESULTS.json', metrics)
    dump(dest / 'FROZEN.json', dict(created_utc=now(), plan_sha256=expected,
        prepared_sha256=sha(OUT / 'prepared/COMPLETION.json'), orders_sha256=sha(OUT / 'prepared/ORDERS.json'),
        training_sha256=sha(dest / 'TRAIN_RESULTS.json'), selection=result))
    print(json.dumps(result), flush=True)


def frozen_selection(expected, orders):
    frozen = read(OUT / 'selected/FROZEN.json')
    require(frozen['plan_sha256'] == expected and frozen['orders_sha256'] == sha(OUT / 'prepared/ORDERS.json') and
        frozen['training_sha256'] == sha(OUT / 'selected/TRAIN_RESULTS.json') and
        frozen['prepared_sha256'] == sha(OUT / 'prepared/COMPLETION.json'), 'Frozen selection dependencies changed')
    require(select_first88(read(OUT / 'selected/TRAIN_RESULTS.json'), orders) == frozen['selection'], 'Selection mismatch')
    return frozen


def verify_training_against_audit(training, audit):
    import operating_margin172_factor_independent as independent
    helper = independent.accounting()
    require(set(training) == set(NAMES), 'Training arms differ')
    for arm in NAMES:
        require(set(training[arm]) == {'same_close_zero', 'next_open_cost'}, 'Training paths differ')
        for path, slices in training[arm].items():
            require(set(slices) == {'training', 'early', 'recent'}, 'Training slices differ')
            for name, full_name in (('training', 'first88'), ('early', 'early53'), ('recent', 'recent35')):
                helper.check_metrics(audit['paths'][arm][path]['slices'][full_name], slices[name],
                                     'Frozen training/' + arm + '/' + path + '/' + name)


def diagnose(expected):
    meta, orders = prepared(expected); frozen = frozen_selection(expected, orders)
    require(not (OUT / 'results').exists(), 'Diagnostics are write-once')
    px, op = ev.prices(); engine = ev.engine(); dest = OUT / 'results'; dest.mkdir(); results, exports = {}, {}
    for arm in NAMES:
        results[arm], exports[arm] = {}, {}
        for path, mode, cost in ev.PATHS:
            result, eq, trades, holds, positions = ev.run_path(engine, px, op, orders[arm], meta, mode, cost)
            result['slices'] = dict(full122=result['metrics'], **ev.split_stats(engine, eq, meta))
            results[arm][path] = result
            data = dict(DAILY=[dict(date=str(d.date()), equity=float(v)) for d, v in eq.items()],
                TRADES=trades, HOLDINGS=holds, DAILY_POSITIONS=positions)
            exports[arm][path] = data
            for kind, value in data.items(): dump(dest / (arm + '_' + path + '_' + kind + '.json'), value)
    import operating_margin172_factor_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, conditional_original_document_vintages=True, observer_promotion=False))
    gate(expected)
    dump(dest / 'COMPLETION.json', dict(status='COMPLETE', created_utc=now(), plan_sha256=expected,
        frozen_sha256=sha(OUT / 'selected/FROZEN.json'), paths=9, slices=45,
        artifacts_sha256={p.name: sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='COMPLETE', primary=frozen['selection']['primary'],
        full_cagr={n: results[n]['same_close_zero']['metrics']['cagr'] for n in NAMES},
        completion_sha256=sha(dest / 'COMPLETION.json'))), flush=True)


def close(expected):
    gate(expected); completion = read(OUT / 'results/COMPLETION.json')
    require(completion['status'] == 'COMPLETE' and completion['plan_sha256'] == expected and
            completion['paths'] == 9 and completion['slices'] == 45, 'Incomplete diagnostics')
    for name, digest in completion['artifacts_sha256'].items(): require(sha(OUT / 'results' / name) == digest, 'Changed result ' + name)
    require_outcome_audit(read(OUT / 'results/INDEPENDENT_OUTCOME.json'))
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), read(OUT / 'results/INDEPENDENT_OUTCOME.json'))
    _, orders = prepared(expected); frozen = frozen_selection(expected, orders)
    require(completion['frozen_sha256'] == sha(OUT / 'selected/FROZEN.json'), 'Frozen selection receipt changed')
    results = read(OUT / 'results/RESULTS.json')
    require(results['selection'] == frozen['selection'], 'Closure selection mismatch')
    before = CENTRAL.read_bytes(); state = json.loads(before); state_check(state)
    protected = read(PROTECTED); verify(protected); require(len(protected) == 21444, 'Incomplete protection')
    require(not (OUT / 'COMPLETION.json').exists(), 'Closure is write-once')
    best = state['best_training_selected_candidate']; pick = results['selection']['primary']; arm = results['arms'][pick]
    metrics = arm['same_close_zero']['metrics']
    if results['selection']['admitted'] and metrics['cagr'] > best['cagr']:
        best = dict(directory=OUT.name, spec=dict(arm=pick), cagr=metrics['cagr'], sharpe=metrics['sharpe'],
            max_drawdown=metrics['max_drawdown'], next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],
            selected_on_first88=True, observer_promotion=False, limitations=contract()['limitation'])
    receipt = dict(status='COMPLETED_INDEPENDENTLY_VERIFIED', created_utc=now(), plan_sha256=expected,
        result_completion_sha256=sha(OUT / 'results/COMPLETION.json'), new_trials=3, new_model_fits=0,
        protected_files=len(protected), best_training_selected_candidate=best,
        target_reached=best['cagr'] >= .7, observers_modified=False)
    dump(OUT / 'COMPLETION.json', receipt)
    state['completed'].append(dict(directory=OUT.name, trials=3, completed_utc=now(), primary=pick,
        primary_cagr=metrics['cagr'], primary_sharpe=metrics['sharpe'], new_admitted=int(results['selection']['admitted']),
        independent_execution_paths=9, protected_files=len(protected), target_reached=receipt['target_reached'],
        completion=dict(path=str(OUT / 'COMPLETION.json'), sha256=sha(OUT / 'COMPLETION.json'))))
    state.update(best_training_selected_candidate=best, latest_training_selected_candidate=dict(directory=OUT.name,
        arm=pick, selection=results['selection'], full_period_cagr=metrics['cagr'], full_period_sharpe=metrics['sharpe'], selected_on_first88=True),
        best_tracking_note=contract()['historical_best_tracking'], registered_queue=[], current_status='completed_independently_verified',
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Review annual-margin confirmation results before a distinct hypothesis; preserve frozen observers.')
    require(len(state['completed']) == 170 and sum(r['trials'] for r in state['completed']) == 1025, 'Wrong final counters')
    commit_state(before, state, 'close'); print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['test', 'register', 'prepare', 'select', 'diagnose', 'close'])
    parser.add_argument('--plan-sha256'); args = parser.parse_args()
    if args.phase == 'test': tests()
    elif args.phase == 'register': register()
    else:
        require(bool(args.plan_sha256), 'An explicit registered PLAN SHA is required')
        globals()[args.phase](args.plan_sha256)

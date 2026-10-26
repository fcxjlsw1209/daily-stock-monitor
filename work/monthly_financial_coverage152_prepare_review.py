"""Independent coverage152 prepare audit, with no current producer imports.

The old149 earnings cache is evidence, never recomputed. New NI/assets signals
use integer decimal alignment; source time selection precedes validity checks.
Only an explicit registered PLAN plus completed preparation enables real I/O.
"""
from bisect import bisect_right
from collections import Counter
from datetime import timedelta
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import numpy as np
import pandas as pd
import monthly_cash_quality146_prepare_review as legacy
import monthly_financial_coverage150_prepare_review as prior
import monthly_financial_coverage151_prepare_review as previous
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
old, require, day = legacy.old, legacy.require, legacy.day
STUDY = 'outputs/monthly_financial_coverage152'
LOCAL = old.LOCAL
LEGACY = 'outputs/monthly_stock_comp149'
PREVIOUS_PLAN = '9b0095d8112cb4579e63f85e2f6b837fe8d7ebb5eef5e399fd926f41c682735d'
PREVIOUS_DONE = '0811e2607b350ce4df89c32616548e17c17e65a2c752c916e67580633343f7c6'
CONCEPT = 'work/after149_coverage_extension_concept.json'
CONCEPT_SHA = '60c3afd837cd32b75a70733b70c777084a9c124cf917e8052628947f1cb7d6ee'
METHOD = 'expanded_annual_earnings_coverage_v1'
DEFINITION = 'current_annual_net_income_average_assets_v1'
NAMES = ('incumbent_local67', 'original_earnings', 'batch1_earnings', 'batch2_earnings', 'expanded_earnings')
EXTRA = ('earnings_original', 'earnings_batch1', 'earnings_batch2', 'earnings_expanded', 'original_available', 'batch1_available', 'batch2_available', 'expanded_available')
FIELDS = ('annual_net_income', 'current_total_assets', 'prior_total_assets')
NEW = frozenset(('KO', 'LRCX'))
PRIOR = frozenset(('AAPL', 'AMD', 'MSFT', 'NVDA', 'AMZN', 'CSCO', 'INTC', 'KLAC'))
INPUTS = ('proposal', 'meta', 'policy', 'original_orders', 'prices', 'sessions', 'protected_manifest')
CACHE_FILES, ATOL, CALENDAR_SHA = legacy.CACHE_FILES, legacy.ATOL, legacy.CALENDAR_SHA
LEGACY_REFS = {k: LEGACY+'/'+v for k, v in (
    ('plan', 'PLAN.json'), ('cache', 'CACHE.json'), ('completion', 'COMPLETION.json'), ('outcome', 'INDEPENDENT_OUTCOME.json'))}
COVERAGE_REFS = {k: 'outputs/monthly_financial_coverage_admission_batch1_v1/'+v for k, v in (
    ('dataset', 'DATASET.json'), ('admission', 'ADMISSION.json'), ('coverage', 'COVERAGE.json'), ('manifest', 'MANIFEST.json'))}
COVERAGE_REFS.update(policy='work/after149_batch1_admission_policy.json', review='work/after149_batch1_independent_review.json')
BATCH = 'outputs/monthly_financial_coverage_batch1_v1'
CENSUS = 'outputs/monthly_financial_coverage_census_v1/CENSUS.json'
BATCH2 = 'outputs/monthly_financial_coverage_batch2_v1'
BATCH2_CONTEXT = 'work/after150_next_source_context.json'
BATCH3 = 'outputs/monthly_financial_coverage_batch3_v1'
BATCH3_CONTEXT = 'work/after151_next_source_context.json'
ACCOUNTING = 'work/after151_batch3_accounting_review.json'
PRIOR151 = 'outputs/monthly_financial_coverage151'
PRIOR151_PLAN = '6f8aa84dadee4d2499e683ecf93a6e82b09ea4f54ec05df5047ef19b8bafcac4'
PRIOR151_DONE = '7dd764a81ff112d2706e4f4216685832830e899ca19af65b77e5dbc5a6648cf5'
PRIOR151_REFS = {k:v.replace(LEGACY,PRIOR151) for k,v in LEGACY_REFS.items()}
BATCH3_REFS = {k:v.replace('batch1','batch3').replace('after149','after151') for k,v in COVERAGE_REFS.items()}
IDENTITY_ADDED = {'issuer_id','security_id','identity_valid_from','identity_valid_through','scope','certified_daily_security_master'}
ISSUERS = {'KO':'The Coca-Cola Company','LRCX':'Lam Research Corporation'}
ASSUMPTIONS = {'KO':'KO_calendar_annual_boundaries_v1','LRCX':'LRCX_consecutive_annual_boundaries_v1'}
PRIOR150 = 'outputs/monthly_financial_coverage150'
PRIOR150_PLAN = '1d14bd0e34fcaaa27be9fe290d5b6e9f27aa882723f5c521eacdd294160f55db'
PRIOR150_DONE = '28546cec6459a8ca73df5b434d06878ca7bb16008fc0e13b1a6b870cee8ffb92'
PRIOR150_REFS = {k: v.replace(LEGACY, PRIOR150) for k,v in LEGACY_REFS.items()}
BATCH2_REFS = {k: v.replace('batch1','batch2').replace('after149','after150') for k,v in COVERAGE_REFS.items()}
CHECKS = ('original27_arrays_byte_identical', 'all_features_independently_reconstructed',
          'all_five_score_matrices_and_ordered_choices_independently_verified',
          'one_episode_and_prefix_isolation_verified', 'zero_and_missing_exact_fallback',
          'original149_earnings_cache_and_mask_verified', 'coverage_source_admission_verified',
          'old44_source_records_and_holds_preserved', 'latest_catalog_blocking_independently_verified',
          'prior150_batch1_earnings_cache_and_mask_verified', 'batch1_source_records_and_holds_preserved', 'prior151_batch2_earnings_cache_and_mask_verified', 'batch2_source_records_and_holds_preserved')


def source_files(root=None):
    root = ROOT if root is None else Path(root).resolve()
    return sorted(set(previous.source_files(root)+[root/'work'/n for n in (
        'monthly_financial_coverage152_prepare_review.py', 'test_monthly_financial_coverage152_prepare_review.py')]))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def first_session(released, sessions):
    d = day(released)
    require(sessions and sessions == sorted(set(sessions)), 'Ordered unique sessions')
    if d < day(sessions[0]) or d >= day(sessions[-1]):
        return None
    return sessions[bisect_right(sessions, released)]


def scaled(cell):
    a, e = legacy.decimal_integer(cell['value_decimal'])
    b, f = legacy.decimal_integer(cell['scale'])
    require(b > 0, 'Positive scale')
    return a*b, e+f


def bounded_earnings(values):
    """Independent integer closed form: 2 NI / (A0 + A1 + abs(2 NI))."""
    require(len(values) == 3, 'Three amounts')
    power = min(e for _, e in values)
    ni, current, prior = [a*10**(e-power) for a, e in values]
    require(current > 0 and prior > 0, 'Positive asset endpoints')
    numerator = 2*ni
    return numerator/(current+prior+abs(numerator))


def empty(reason, key=None):
    return dict(available=False, reason=reason, event_id=key, earnings_bounded=None)


def report_oracle(event, signal):
    result = empty('unavailable', event.get('event_id'))
    def fail(reason):
        return dict(result, reason=reason)
    gate = event.get('ni_assets_admission')
    if type(gate) is not dict or gate.get('definition') != DEFINITION:
        return fail('missing_or_wrong_admission')
    if gate.get('admitted') is not True or gate.get('reasons') != []:
        return fail('latest_event_held')
    if event.get('symbol') not in NEW:
        return fail('unsupported_symbol')
    try:
        now, available = day(signal), day(event['available_from'])
        release = day(event.get('release_date', event.get('publication_date')))
        if not release < available <= now:
            return fail('invalid_available_time')
        source, fields = event['source'], event['fields']
        require(set(fields) == set(FIELDS), 'Three fields')
        period = fields[FIELDS[0]]['period']
        start, end = day(period['period_start']), day(period['period_end'])
        length = (end-start).days+1
        if not 330 <= length <= 400 or period['days_inclusive'] != length:
            return fail('invalid_annual_period')
        if end > release:
            return fail('release_before_period_end')
        identity = event['identity']
        require(type(identity['issuer_id']) is str and identity['issuer_id'].strip(), 'Issuer identity')
        require(identity['security_id'] == event['symbol'], 'Security identity')
        require(day(identity['identity_valid_from']) == available and
                day(identity['identity_valid_through']) == end+timedelta(days=550), 'Identity bounds')
        if (now-end).days > 550:
            return fail('stale_latest_event')
        require(available <= now <= day(identity['identity_valid_through']), 'Identity window')
        amounts, currencies = [], set()
        for name in FIELDS:
            cell = fields[name]
            require(type(cell) is dict and cell['source']['path'] == source['path'] and
                    cell['source']['sha256'] == source['sha256'], 'Same report')
            require(cell['available_from'] == event['available_from'] and cell['currency_literal'] == '$', 'Clock/currency')
            require(day(cell['as_of']) == (start-timedelta(days=1) if name == FIELDS[2] else end), 'Asset endpoints')
            currencies.add(cell.get('currency_iso'))
            amounts.append(scaled(cell))
        require(len(currencies) == 1, 'Same currency')
        value = bounded_earnings(amounts)
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return fail('invalid_required_source_field')
    return dict(result, available=True, reason='available', earnings_bounded=value,
                available_from=event['available_from'], period_end=end.isoformat())


def latest_oracle(catalog, events, symbol, signal):
    """Group known records before testing any gate, availability or financial field."""
    now = day(signal)
    by_id = {}
    for row in catalog:
        if row.get('symbol') != symbol:
            continue
        release = row.get('release_date', row.get('publication_date'))
        available = row.get('available_from', row.get('available_session'))
        clock = row.get('clock_review', {})
        if clock.get('verified') is True:
            require(day(available) > day(release) and clock.get('first_pinned_session') == available, 'Catalog clock')
            known = day(available) <= now
        else:
            known = day(release) < now
        if not known:
            continue
        key = row['event_id']
        if key in by_id and by_id[key] != row:
            return empty('conflicting_catalog_event', key)
        by_id[key] = row
    if not by_id:
        return empty('no_known_annual_event')
    last = max(r.get('release_date', r.get('publication_date')) for r in by_id.values())
    chosen = [r for r in by_id.values() if r.get('release_date', r.get('publication_date')) == last]
    if len(chosen) != 1:
        return empty('ambiguous_latest_annual_event')
    row = chosen[0]
    key = row['event_id']
    if row.get('clock_review', {}).get('verified') is not True:
        return empty('latest_annual_event_clock_unverified', key)
    if key not in events:
        return empty('latest_annual_event_unprocessed', key)
    event = events[key]
    require(event['symbol'] == symbol and event['available_from'] == row.get('available_from', row.get('available_session')) and
            event.get('release_date', event.get('publication_date')) == last, 'Catalog/report binding')
    return report_oracle(event, signal)


def features_oracle(meta, original, batch1, batch2, dataset):
    require(dataset.get('schema_version') == 1 and dataset.get('method') == METHOD, 'Source schema')
    dates,pools,etfs = meta['dates'],meta['symbols'],set(meta['etfs'])
    require(len(etfs) == 17 and not etfs & (NEW|PRIOR), 'ETF identity')
    require([day(d) for d in dates] == sorted(set(day(d) for d in dates)) and len(pools) == len(dates), 'Signal axes')
    for array in (original,batch1,batch2):
        require(array.dtype == np.float64 and array.shape == (len(dates),49) and not np.isinf(array).any(), 'Control axes')
    events = {e['event_id']:e for e in dataset['events']}
    catalog = dataset['latest_event_catalog']
    require(len(events) == len(dataset['events']) and len({c['event_id'] for c in catalog}) == len(catalog), 'Unique source IDs')
    require(all(e['symbol'] in NEW|{'AMZN','CSCO','INTC','KLAC'} for e in events.values()) and all(c['symbol'] in NEW|{'AMZN','CSCO','INTC','KLAC'} for c in catalog), 'Cumulative source scope')
    expanded,cells = batch2.copy(),[]
    for i,(d,names) in enumerate(zip(dates,pools)):
        require(len(names) == len(set(names)) == 49 and etfs <= set(names), '49 pool entries')
        for j,name in enumerate(names):
            require(not np.isfinite(original[i,j]) or name in {'AAPL','AMD','MSFT','NVDA'}, 'Original149 scope')
            require(not np.isfinite(batch1[i,j]) or name in {'AAPL','AMD','MSFT','NVDA','AMZN','CSCO'}, 'Prior150 scope')
            if name not in {'AMZN','CSCO'}:
                require(original[i:i+1,j:j+1].tobytes() == batch1[i:i+1,j:j+1].tobytes(), 'Prior150 changed unrelated cell')
            require(not np.isfinite(batch2[i,j]) or name in PRIOR, 'Prior151 scope')
            if name not in {'INTC','KLAC'}:
                require(batch1[i:i+1,j:j+1].tobytes() == batch2[i:i+1,j:j+1].tobytes(), 'Prior151 changed unrelated cell')
            if name in NEW:
                item=latest_oracle(catalog,events,name,d)
                if item['available']:
                    expanded[i,j]=item['earnings_bounded']
            else:
                item=empty('etf_not_applicable' if name in etfs else 'preserved_original_coverage' if name in {'AAPL','AMD','MSFT','NVDA'} else 'preserved_batch1_coverage' if name in {'AMZN','CSCO'} else 'preserved_batch2_coverage' if name in {'INTC','KLAC'} else 'no_new_source_coverage')
            cells.append(dict(signal_index=i,pool_index=j,signal_date=d,symbol=name,**item))
    return dict(arrays=dict(earnings_original=original.copy(),earnings_batch1=batch1.copy(),earnings_batch2=batch2.copy(),earnings_expanded=expanded,
                           original_available=np.isfinite(original),batch1_available=np.isfinite(batch1),batch2_available=np.isfinite(batch2),expanded_available=np.isfinite(expanded)),cells=cells)


def score_oracle(arrays, meta, original, batch1, batch2, expanded, n=None):
    helper = old.original_helper()
    state = helper.original_prefix(arrays, meta, n)
    n, shape = state['n'], state['core'].shape
    scores, choices, deltas = {}, {}, {}
    for name, hand in zip(NAMES, (None, original, batch1, batch2, expanded)):
        before, delta = state['local'].copy(), np.zeros(shape)
        if hand is not None:
            hand = np.asarray(hand)[:n]
            require(hand.shape == shape and hand.dtype.kind in 'fiu' and not np.isinf(hand).any(), 'Hand shape/type')
            require((np.abs(hand[np.isfinite(hand)]) <= 1).all(), 'Hand bounds')
            for i, pool in enumerate(meta['symbols'][:n]):
                require(all(not np.isfinite(hand[i, j]) for j, s in enumerate(pool) if s in meta['etfs']), 'ETF hand')
            active = state['eligible'] & np.isfinite(hand) & (hand != 0)
            delta[active] = .01*hand[active]
            before[active] += delta[active]
        chosen, final, _ = helper.episode(before, state)
        scores[name], choices[name], deltas[name] = final, chosen, delta
    return scores, choices, deltas


def ordered_orders(meta, scores, choices):
    return {name: [dict(date=d, signal_date=d, end=meta['ends'][i],
                       symbols=[meta['symbols'][i][j] for j in choice],
                       scores=[float(scores[name][i, j]) for j in choice])
                   for i, (d, choice) in enumerate(zip(meta['dates'], choices[name]))] for name in NAMES}


def verify_original_arrays(original, prepared):
    require(len(original) == 27 and set(prepared) == set(original)|set(EXTRA), '27 original plus eight arrays')
    for key, value in original.items():
        actual = prepared[key]
        require(actual.dtype == value.dtype and actual.shape == value.shape and actual.tobytes() == value.tobytes(), 'Original array '+key)


def verify_scores_and_orders(actual_scores, actual_orders, meta, scores, choices):
    require(set(actual_scores) == set(actual_orders) == set(NAMES), 'Five exact arms')
    orders = ordered_orders(meta, scores, choices)
    for name in NAMES:
        require(actual_scores[name].dtype == scores[name].dtype and actual_scores[name].shape == scores[name].shape, 'Score axes/dtype')
        np.testing.assert_allclose(actual_scores[name], scores[name], rtol=0, atol=ATOL)
        require(len(actual_orders[name]) == len(orders[name]), 'Complete orders')
        for actual, expected in zip(actual_orders[name], orders[name]):
            require(set(actual) == set(expected) and all(actual[k] == expected[k] for k in ('date', 'signal_date', 'end', 'symbols')), 'Ordered holding')
            np.testing.assert_allclose(actual['scores'], expected['scores'], rtol=0, atol=ATOL)


def verify_audit(saved, expected, plan):
    a=expected['arrays']
    wanted = dict(method=METHOD,original_cells=int(a['original_available'].sum()),batch1_cells=int(a['batch1_available'].sum()),
        batch2_cells=int(a['batch2_available'].sum()),expanded_cells=int(a['expanded_available'].sum()),added_cells=int(a['expanded_available'].sum()-a['batch2_available'].sum()),
        original_input_unchanged=True,batch1_input_unchanged=True,batch2_input_unchanged=True,new_model_fits=0,returns_computed=False,
        reasons=dict(Counter(c['reason'] for c in expected['cells'])),cells=expected['cells'],
        legacy149=plan['legacy149'],prior151=plan['prior151'],coverage_sources=plan['coverage_sources'],
        prior151_batch2_earnings_cache_and_mask_verified=True,prior150_batch1_earnings_cache_and_mask_verified=True,original149_earnings_cache_and_mask_verified=True,
        original149_control_scores_and_orders_byte_exact=True,prior150_control_scores_and_orders_byte_exact=True,prior151_control_scores_and_orders_byte_exact=True)
    exact(saved,wanted)


def exact(actual, wanted):
    require(type(actual) is type(wanted), 'Exact type')
    if type(wanted) is dict:
        require(set(actual) == set(wanted), 'Exact object keys')
        for key in wanted:
            exact(actual[key], wanted[key])
    elif type(wanted) is list:
        require(len(actual) == len(wanted), 'Exact list length')
        for a, b in zip(actual, wanted):
            exact(a, b)
    else:
        require(actual == wanted, 'Exact value')


def selection_contract():
    return dict(total_months=122, training_months=88, early_months=53, recent_months=35, later_months=34,
                walkforward=[[36,60],[60,76],[76,88]], execution_paths=[dict(execution=x, cost=c, key=k) for x,c,k in
                (('close',0.,'same_close'),('open',0.,'next_open_zero'),('open',.0005,'next_open_cost'))],
                same_close_cagr='strict_greater_than_incumbent_original_batch1_and_batch2_earnings',
                next_open_cost_cagr='noninferior_incumbent_strict_greater_than_original_batch1_and_batch2_earnings', strict_cagr_epsilon=1e-12,
                max_sharpe_degradation=.05, max_drawdown_degradation=.03, max_early_cagr_degradation=.025,
                max_recent_cagr_degradation=.025, new_model_fits=0, intramonth_trades=False, later34_for_selection=False,
                financial_formula='NI_current/((assets_current+assets_prior)/2)', bounded='h(x)=x/(1+abs(x))', dose=.01,
                coverage_masks='distinct_original_batch1_batch2_and_expanded_masks', unavailable='exact_incumbent_fallback_for_that_hand',
                old_coverage='frozen149_earnings_control_including_SBC_common_mask',
                batch1_coverage='frozen150_expanded_earnings_and_mask', batch2_coverage='frozen151_expanded_earnings_and_mask',
                episode='one_original_episode_selector_after_same_original_pre_score')


def measurement_contract():
    return dict(method=METHOD, definition=DEFINITION, earnings='NI_current/((assets_current+assets_prior)/2)',
                bounded='h(x)=x/(1+abs(x))', exact_arithmetic='scaled_Fraction_then_one_bounded_float',
                original_control='verified_final149_CACHE_earnings_bounded_and_common_available',
                old_events=44, old_admitted=31, old_held=13, new_symbols=['KO','LRCX'], source_events=55, annual_catalog_events=96,
                prior150_control='verified_final150_CACHE_earnings_expanded_and_expanded_available',
                batch1_events=26,batch2_events=19,batch3_events=10,batch1_admitted=17,batch2_admitted=14,batch3_admitted=5,
                prior151_control='verified_final151_CACHE_earnings_expanded_and_expanded_available',
                masks='distinct_original_batch1_batch2_expanded_with_all_three_frozen_controls_preserved',
                time='current_release_next_pinned_session_330_400_day_current_year_550_day_identity_window',
                latest_event_before_validity=True, invalid_latest_blocks_fallback=True,
                placement='add_0.01_h_to_original_pre_score_then_one_original_episode_selector',
                unchanged_original_arrays=27, monthly_equal_weight_positions=3, intramonth_trades=False,
                etf_financial_features=False, new_model_fits=0, no_parameter_search=True)


def bound_ref(reader, ref, pins, name=None):
    require(type(ref) is dict and set(ref) == {'path','sha256'}, 'Exact source ref')
    path = reader.path(ref['path'])
    require(name is None or path == reader.path(name), 'Dedicated source path')
    require(reader.sha(path) == ref['sha256'] == pins.get(str(path)), 'Registered source hash')
    return path


def verify_proposal(reader, plan, pins):
    p = reader.json(bound_ref(reader, plan['inputs']['proposal'], pins))
    require(p.get('round') == 152 and p.get('arms') == list(NAMES) and p.get('primary_challenger') == NAMES[4]
            and p.get('dose') == .01 and p.get('status') == 'FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION', 'Prospective152')
    exact(p['selection_contract'], selection_contract())
    exact(p['measurement'], measurement_contract())
    require(p['sources'] == dict(legacy149=plan['legacy149'], prior151=plan['prior151'], coverage_sources=plan['coverage_sources']), 'Prospective refs')
    require(p['concept']['sha256'] == CONCEPT_SHA, 'Fixed concept')
    bound_ref(reader, p['concept'], pins, CONCEPT)
    for key in ('actual_new_factors','actual_new_orders','actual_new_returns'):
        require(type(p.get(key)) is int and p[key] == 0, 'Prospective no computation')
    return p


def closure(reader, mapping, pins):
    result = reader.verify(mapping)
    require(all(pins.get(k) == v for k,v in result.items()), 'Entire closure registered')
    return result


def objects(reader, refs, paths, pins):
    require(set(refs) == set(paths), 'Exact reference set')
    return {k: reader.json(bound_ref(reader, refs[k], pins, path)) for k,path in paths.items()}


def verify_legacy(reader, plan, pins):
    refs = plan['legacy149']
    docs = objects(reader, refs, LEGACY_REFS, pins)
    p, cache, done, review = [docs[k] for k in ('plan','cache','completion','outcome')]
    require(refs['plan']['sha256'] == PREVIOUS_PLAN and refs['completion']['sha256'] == PREVIOUS_DONE, 'Fixed finalized149')
    require(p['round'] == done['round'] == 149 and p['status'] == 'REGISTERED', 'Registered predecessor')
    require(done['plan_sha256'] == review['plan_sha256'] == PREVIOUS_PLAN and
            done['independent_outcome_sha256'] == refs['outcome']['sha256'] and
            review['cache_sha256'] == refs['cache']['sha256'] and review.get('passed') is True and review.get('stage') == 'evaluate', '149 completed audit chain')
    closure(reader, p['hashes'], pins)
    require(set(cache) == set(CACHE_FILES), 'Original six cache files')
    cachepins = closure(reader, {LEGACY+'/'+k:v for k,v in cache.items()}, pins)
    audited = closure(reader, review['artifact_hashes'], pins)
    require(all(audited.get(k) == v for k,v in cachepins.items()) and review['orders_sha256'] == cache['PRETRAIN_ORDERS.json'], 'Final149 cache independently audited')
    closure(reader, review['source_hashes'], pins)
    for key in INPUTS[1:]:
        require(reader.path(p['inputs'][key]['path']) == reader.path(plan['inputs'][key]['path']) and
                p['inputs'][key]['sha256'] == plan['inputs'][key]['sha256'], 'Original named input '+key)
    require(plan['raw_source_hashes'] == p['raw_source_hashes'], 'Original raw64 preserved')
    return p


def verify_prior151(reader,plan,pins):
    docs=objects(reader,plan['prior151'],PRIOR151_REFS,pins)
    p,cache,done,review=(docs[k] for k in ('plan','cache','completion','outcome'))
    require(plan['prior151']['plan']['sha256']==PRIOR151_PLAN and plan['prior151']['completion']['sha256']==PRIOR151_DONE,'Fixed completed151')
    require(p['round']==done['round']==151 and p['status']=='REGISTERED' and review['passed'] is True and review['stage']=='evaluate','Prior151 completed audit')
    require(done['plan_sha256']==review['plan_sha256']==PRIOR151_PLAN and done['independent_outcome_sha256']==plan['prior151']['outcome']['sha256'] and review['cache_sha256']==plan['prior151']['cache']['sha256'],'Prior151 chain')
    require(p['legacy149']==plan['legacy149'] and p['coverage_sources']=={k:plan['coverage_sources'][k] for k in ('batch1','batch2')},'Prior151 sources unchanged')
    closure(reader,p['hashes'],pins)
    require(set(cache)==set(CACHE_FILES),'Prior151 six cache files')
    cp=closure(reader,{PRIOR151+'/'+k:v for k,v in cache.items()},pins)
    audited=closure(reader,review['artifact_hashes'],pins)
    require(all(audited.get(k)==v for k,v in cp.items()) and review['orders_sha256']==cache['PRETRAIN_ORDERS.json'],'Prior151 independently checked cache')
    closure(reader,review['source_hashes'],pins)
    for k in INPUTS[1:]: require(p['inputs'][k]==plan['inputs'][k],'Prior151 input drift '+k)
    require(p['raw_source_hashes']==plan['raw_source_hashes'],'Prior151 raw64 drift')
    previous.verify_prior150(reader,dict(plan,prior150=p['prior150']),pins)
    return p



def source_gate(event,context,sessions,physical,raw_identity):
    """Independent source eligibility; no earnings ratio or market data."""
    sym=event['symbol']; require(sym in NEW,'New issuer')
    require(all(event[k]==context[k] for k in ('event_id','symbol','fiscal_year')),'Context identity')
    src=event['source'];require(src==context['source'] and src['path']==context['evidence_path'] and src['sha256']==context['evidence_sha256'] and src['url']==context['source_url'],'Context source')
    require(event['context_holds']==context['context_holds'] and event['accounting_context_numbers_redacted']==context['accounting_context_numbers_redacted'],'Context preserved')
    release,available=event['publication_date'],event['available_from']
    require(release==context['publication_date'] and available==context['available_session'] and day(release)<day(available),'Context clock')
    derived=first_session(release,sessions);clock=derived is not None and derived==available
    require(physical['clock_verified'] is clock and physical['issuer_id']==ISSUERS[sym] and physical['security_id']==sym and physical['identity_verified'] is True,'Independent clock/identity')
    reasons=[]
    if not clock:reasons.append('unverified_exchange_calendar' if derived is None else 'availability_not_first_pinned_session')
    boundary=physical.get('boundary_assumption_verified') is True and physical.get('assumption_id')==ASSUMPTIONS[sym]
    if not boundary:reasons.append('annual_boundary_assumption_unverified')
    acc=physical['accounting_review']
    if acc.get('status')!='resolved':reasons.append('accounting_comparability_unresolved')
    else:
        require(acc['source']==src and acc['scope']=='current_annual_net_income_same_report_assets' and acc['current_prior_common_basis'] is True and acc['reasons']==[],'Resolved accounting scope')
        require(set(acc['field_checks'])==set(FIELDS) and all(x is True for x in acc['field_checks'].values()) and acc['basis'] and acc['evidence'],'Resolved accounting evidence')
        for item in acc['evidence']:require(item['path']==src['path'] and item['sha256']==src['sha256'] and item['locator'] and item['quote'],'Accounting source')
    known={'monetary_cells_and_sign_geometry_not_yet_independently_verified','currency_ISO_and_first_public_version_not_certified','publication_outside_pinned_calendar_range','accounting_notes_require_field_specific_independent_review','proposed_annual_start_boundary_assumption_not_yet_approved'}
    require(type(event['context_holds']) is list,'Explicit holds')
    for hold in event['context_holds']:
        if hold not in known:reasons.append('retained_context_hold:'+hold)
    require(set(event['fields'])==set(FIELDS),'Three source fields')
    for name in FIELDS:
        if event['fields'][name] is None:reasons.append('missing_'+name)
    start=end=None
    try:
        p=event['fields'][FIELDS[0]]['period'];start,end=day(p['period_start']),day(p['period_end']);n=(end-start).days+1
        require(type(p['days_inclusive']) is int and n==p['days_inclusive'] and 330<=n<=400,'Annual duration')
        c=context['periods'];require(str(end)==context['official_actual_period_end']==c['current_end'] and str(start)==c['proposed_current_start'] and start-timedelta(days=1)==day(c['prior_end']) and end<day(release),'Annual endpoints')
        require(event['periods']['current']==p and event['periods']['comparison']['period_end']==c['prior_end'] and event['periods']['start_basis']=='derived_same_report_prior_annual_end_plus_one_not_directly_quoted' and event['periods']['start_assumption_certified'] is False,'Derived annual boundary')
        require(physical['derived_start']==str(start) and physical['start_is_source_quoted'] is False,'Independent boundary')
        year=event['fiscal_year']
        if sym=='KO':require(n in (365,366) and str(start)==f'{year}-01-01' and str(end)==f'{year}-12-31' and c['prior_end']==f'{year-1}-12-31','KO annual calendar')
        else:require(n in (364,371) and end.year==year and day(c['prior_end']).year==year-1,'LRCX fiscal year')
    except (KeyError,TypeError,ValueError):reasons.append('invalid_current_annual_period')
    for name,cell in event['fields'].items():
        if cell is None:continue
        try:
            require(cell['source']==src and cell['available_from']==available and cell['fact_admitted'] is False and cell['displayed_amount'].strip() and cell['locator'],'Cell provenance')
            a,e=scaled(cell);b,f=legacy.decimal_integer(cell['scale']);require(b*10**f==(1000000 if sym=='KO' else 1000) and (name==FIELDS[0] or a>0),'Scale/positive assets')
            require(cell['currency_literal']=='$' and cell.get('currency_iso') is None,'Dollar assumption')
            require(start is not None and end is not None and day(cell['as_of'])==(start-timedelta(days=1) if name==FIELDS[2] else end),'Cell endpoint')
            require(cell.get('period')==(event['periods']['current'] if name==FIELDS[0] else None),'Cell period')
        except (KeyError,TypeError,ValueError,ArithmeticError):reasons.append('invalid_'+name)
    identity=dict(raw_identity,issuer_id=ISSUERS[sym],security_id=sym,identity_valid_from=available,identity_valid_through=str(end+timedelta(days=550)) if end else None,scope='source_local_report_binding_and_use_window_not_continuous_security_master',certified_daily_security_master=False)
    exact(event['identity'],identity)
    exact(event['clock_review'],dict(first_pinned_session=derived,verified=clock))
    exact(event['source_review'],dict(event_id=physical['event_id'],metadata_verified=True,all_three_positions_reviewed=True,boundary_assumption_verified=boundary,accounting_review=acc,original_context_holds_preserved=True))
    return dict(admitted=not reasons,reasons=list(dict.fromkeys(reasons)),definition=DEFINITION)



def verify_coverage(reader,plan,pins,legacy_plan,sessions):
    require(set(plan['coverage_sources']) == {'batch1','batch2','batch3'}, 'Three exact source packages')
    combined2,batch1=previous.verify_coverage(reader,dict(plan,coverage_sources={k:plan['coverage_sources'][k] for k in ('batch1','batch2')}),pins,legacy_plan,sessions)
    refs=plan['coverage_sources']['batch3']
    docs=objects(reader,refs,BATCH3_REFS,pins)
    data,admission,physical,policy=(docs[k] for k in ('dataset','admission','review','policy'))
    require(data['schema_version'] == 1 and data['method'] == METHOD and admission['passed'] is True and admission['stage'] == 'conditional_source_admission_not_strategy_registration', 'Conditional source package')
    require(docs['manifest']['status'] == 'conditional_source_package_no_strategy_registration' and docs['manifest']['outputs'] == {n+'.json':refs[k]['sha256'] for k,n in (('dataset','DATASET'),('admission','ADMISSION'),('coverage','COVERAGE'))}, 'Batch3 manifest')
    require(docs['manifest']['input_hashes'] == admission['input_hashes'],'Source closure equality')
    sourcepins=closure(reader,admission['input_hashes'],pins)
    reviewpins=closure(reader,physical['input_hashes'],pins)
    closure(reader,physical['source_hashes'],pins)
    sourceplan,extracted=reader.json(BATCH3+'/PLAN.json'),reader.json(BATCH3+'/EXTRACTIONS.json')
    for key in ('inputs','code'): closure(reader,sourceplan[key],pins)
    psha,esha=reader.sha(BATCH3+'/PLAN.json'),reader.sha(BATCH3+'/EXTRACTIONS.json')
    require(data['source_plan_sha256'] == admission['source_plan_sha256'] == physical['plan_sha256'] == extracted['plan_sha256'] == psha and physical['extractions_sha256'] == esha,'Batch3 extraction links')
    for path in (BATCH3+'/PLAN.json',BATCH3+'/EXTRACTIONS.json',BATCH3_CONTEXT):
        require(sourcepins.get(str(reader.path(path))) == reviewpins.get(str(reader.path(path))) == reader.sha(path),'Reviewed source input')
    for key in ('policy','review'):
        require(sourcepins.get(str(reader.path(refs[key]['path']))) == refs[key]['sha256'],'Source mandatory input')
    for key,name in (('policy_sha256','policy'),('independent_review_sha256','review')):
        require(data[key] == admission[key] == refs[name]['sha256'],'Policy/review links')
    require(policy['source_scope']['symbols'] == ['KO','LRCX'] and policy['required_fields'] == list(FIELDS) and policy['known_time']['maximum_age_days'] == 550 and policy['source_scope']['no_automatic_admission'] is True,'Source policy')
    setting=policy['periods']['LRCX_52_53_week_assumption']
    require(setting['enabled'] is True and setting['not_a_quoted_fact'] is True and setting['assumption_id'] == 'LRCX_consecutive_annual_boundaries_v1','Frozen boundary assumption')
    for flag in ('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'):
        require(data[flag] is policy['preservation'][flag] is False,'No source certification upgrade')
    for container,keys in ((data,('scores_computed','ratios_computed','returns_computed','new_strategy_trials')),(admission,('actual_factors_calculated','actual_returns_calculated','new_strategy_trials'))):
        for key in keys: require(type(container[key]) is int and container[key]==0,'Source-only preparation')
    events=data['events'];ids=[e['event_id'] for e in events]
    require(len(ids)==len(set(ids))==10 and Counter(e['symbol'] for e in events)=={'KO':6,'LRCX':4},'Batch3 fixed10')
    require(ids == [e['event_id'] for e in extracted['events']] == [e['event_id'] for e in sourceplan['records']],'Fixed extraction order')
    require([strip_added_event(e,r) for e,r in zip(events,extracted['events'])]==extracted['events'],'Preserved source values and metadata')
    require(physical['passed'] is True,'Physical review passed')
    reviews={e['event_id']:e for e in physical['event_reviews']}
    require(len(reviews)==len(physical['event_reviews'])==10 and set(reviews)==set(ids),'Every source event reviewed')
    context=reader.json(BATCH3_CONTEXT);contexts={e['event_id']:e for e in context['events']}
    require(policy['source_scope']['source_context'] == sourceplan['catalog_ref'] == dict(path=BATCH3_CONTEXT,sha256=reader.sha(BATCH3_CONTEXT)),'Source context binding')
    accounting=reader.json(ACCOUNTING)
    require(sourcepins.get(str(reader.path(ACCOUNTING)))==reviewpins.get(str(reader.path(ACCOUNTING)))==reader.sha(ACCOUNTING),'Accounting file pinned')
    closure(reader,accounting['input_hashes'],pins)
    require(accounting['passed'] is True and accounting['stage']=='independent_field_specific_accounting_review' and accounting['plan_sha256']==psha and accounting['extractions_sha256']==esha,'Accounting review scope')
    ar={r['event_id']:r for r in accounting['events']}
    require(len(ar)==len(accounting['events'])==10 and set(ar)==set(ids),'Complete accounting review')
    for event,raw in zip(events,extracted['events']):
        rr=reviews[event['event_id']]
        require(rr['passed'] is rr['metadata_verified'] is rr['identity_verified'] is True and set(rr['fields'])==set(FIELDS),'Physical metadata review')
        require(rr['source']==event['source'] and rr['retained_context_holds']==event['context_holds'] and rr['accounting_context_numbers_redacted']==event['accounting_context_numbers_redacted'] and rr['available_from']==event['available_from'] and rr['release_date']==event['publication_date'],'Review provenance')
        require(all(event[k] is False for k in ('fact_admitted','model_admitted','strategy_admitted')),'Raw flags preserved')
        exact(rr['accounting_review'],ar[event['event_id']])
        for name in FIELDS:
            null,fr=event['fields'][name] is None,rr['fields'][name]
            require(fr['is_null'] is null,'Physical null flag')
            if null:require(fr['null_reason_verified'] is True and fr['null_classification'],'Classified source null')
            else:require(rr['periods_verified'] is rr['currency_verified'] is True and all(fr[k] is True for k in ('verified','source_value_present','value_and_sign_verified','units_verified','period_verified','source_locator_verified')),'Physical field proof')
        ctx=contexts[event['event_id']];src=ctx['source']
        for path,digest in ((src['path'],src['sha256']),(src['receipt_path'],src['receipt_sha256'])):
            require(reviewpins.get(str(reader.path(path)))==sourcepins.get(str(reader.path(path)))==reader.sha(path)==digest,'Raw report/receipt proof')
        exact(event['ni_assets_admission'],source_gate(event,ctx,sessions,rr,raw['identity']))
        if rr.get('boundary_assumption_verified') is True:
            require(set(rr['boundary_evidence'])==set(FIELDS),'Complete boundary proof')
            for name in FIELDS:
                point=rr['boundary_evidence'][name];end=event['periods']['comparison' if name==FIELDS[2] else 'current']['period_end']
                require(type(point['table']) is int and point['headers'] and point['date']==end,'Boundary physical locator')
                if event['fields'][name] is not None:require(event['fields'][name]['as_of']==end,'Nonnull endpoint')
    catalog=data['latest_event_catalog'];census=reader.json(CENSUS)
    require([{k:v for k,v in e.items() if k not in {'clock_review','processed_in_batch'}} for e in catalog] == [e for e in census['annual_documents'] if e['symbol'] in NEW],'Full batch3 catalog preserved')
    require(len(catalog)==len({e['event_id'] for e in catalog})==32 and Counter(e['symbol'] for e in catalog)=={'KO':16,'LRCX':16},'Batch3 full32')
    require(sourceplan['catalog_ids']==extracted['catalog_ids']==[e['event_id'] for e in context['events']]==[e['event_id'] for e in catalog],'Catalog order unchanged')
    for row in catalog:
        derived=first_session(row['publication_date'],sessions)
        require(row['fiscal_quarter']==4 and row['processed_in_batch'] is (row['event_id'] in ids),'Catalog annual processing flag')
        exact(row['clock_review'],dict(first_pinned_session=derived,verified=derived is not None and derived==row.get('available_session')))
    admitted=sum(e['ni_assets_admission']['admitted'] for e in events)
    require(admitted==5,'Fixed batch3 admission count')
    exact(admission['counts'],dict(events=10,admitted=5,held=5,reviewed_nonnull_fields=27,reviewed_null_fields=3,annual_catalog_events=32,unprocessed_annual_catalog_events=22,original_events=44,original_admitted=31,original_held=13,prior_batch1_events=26,prior_batch1_admitted=17,prior_batch1_held=9,prior_batch2_events=19,prior_batch2_admitted=14,prior_batch2_held=5))
    require(data['original_four_issuer_reference']==batch1['original_four_issuer_reference'],'Old44 unchanged')
    for key,prior_data,counts in (('batch1',batch1,[26,17,9,32]),('batch2',reader.json(plan['coverage_sources']['batch2']['dataset']['path']),[19,14,5,32])):
        ref=data['prior_'+key+'_reference'];br=plan['coverage_sources'][key]
        for pkey,hkey,name in (('path','sha256','dataset'),('admission_path','admission_sha256','admission'),('manifest_path','manifest_sha256','manifest'),('coverage_path','coverage_sha256','coverage')):
            require(reader.path(ref[pkey])==reader.path(br[name]['path']) and ref[hkey]==br[name]['sha256'],'Preserved prior reference')
        require([ref[k] for k in ('events','admitted','held','annual_catalog_events')]==counts and ref['preserved_without_modification'] is True and ref['records_canonical_sha256']==canonical(prior_data['events']) and ref['catalog_canonical_sha256']==canonical(prior_data['latest_event_catalog']),'Prior records and holds preserved')
    require(all(docs['coverage']['counts'][k]==v for k,v in dict(months=122,stocks=73,stock_slots=3904).items()),'Coverage population unchanged')
    combined=dict(schema_version=1,method=METHOD,events=combined2['events']+events,latest_event_catalog=combined2['latest_event_catalog']+catalog)
    require(len(combined['events'])==len({e['event_id'] for e in combined['events']})==55 and len(combined['latest_event_catalog'])==len({e['event_id'] for e in combined['latest_event_catalog']})==96,'Cumulative source identities')
    return combined,combined2,batch1


def strip_added_event(event,raw):
    result={k:deepcopy(v) for k,v in event.items() if k not in {'ni_assets_admission','source_review','clock_review'}}
    expected=set(raw['identity'])|IDENTITY_ADDED
    require(set(event['identity'])==expected,'No hidden identity change')
    result['identity']={k:deepcopy(event['identity'][k]) for k in raw['identity']}
    return result


def verify_prior151_reference(original,prepared,cached,reference,mask,batch1,bmask):
    require(set(cached)==set(original)|set(previous.EXTRA),'Prior151 array schema')
    for key,value in original.items():require(cached[key].dtype==value.dtype and cached[key].shape==value.shape and cached[key].tobytes()==value.tobytes(),'Prior151 original27 '+key)
    for key,value in (('earnings_original',reference),('original_available',mask),('earnings_batch1',batch1),('batch1_available',bmask)):
        require(cached[key].dtype==value.dtype and cached[key].shape==value.shape and cached[key].tobytes()==value.tobytes(),'Prior151 inherited control '+key)
    hand,b2mask=cached['earnings_expanded'],cached['expanded_available']
    require(hand.dtype==np.float64 and b2mask.dtype==bool and hand.shape==b2mask.shape==(122,49) and np.array_equal(np.isfinite(hand),b2mask) and not np.isinf(hand).any(),'Prior151 hand/mask')
    require(prepared['earnings_batch2'].tobytes()==hand.tobytes() and prepared['batch2_available'].tobytes()==b2mask.tobytes(),'Prior151 exact control bytes')
    return hand,b2mask


def verify_reference(original, prepared, cached):
    require(len(original) == 27 and set(cached) == set(original)|{'earnings_bounded','cash_flow_bounded','comp_adjusted_cash_flow_bounded','common_available'}, 'Original149 data schema')
    for key,value in original.items():
        require(cached[key].dtype == value.dtype and cached[key].shape == value.shape and cached[key].tobytes() == value.tobytes(), '149 original array '+key)
    earnings, mask = cached['earnings_bounded'], cached['common_available']
    require(earnings.shape == mask.shape == (122,49) and earnings.dtype == np.float64 and mask.dtype == bool
            and np.array_equal(np.isfinite(earnings),mask) and not np.isinf(earnings).any(), '149 exact hand/mask')
    require(prepared['earnings_original'].tobytes() == earnings.tobytes() and prepared['original_available'].tobytes() == mask.tobytes(), '149 hand/mask bytes')
    return earnings,mask


def verify_prior_reference(original,prepared,cached,reference,mask):
    require(set(cached)==set(original)|set(prior.EXTRA),'Prior150 array schema')
    for key,value in original.items():
        require(cached[key].dtype==value.dtype and cached[key].shape==value.shape and cached[key].tobytes()==value.tobytes(),'Prior150 original27 '+key)
    require(cached['earnings_original'].tobytes()==reference.tobytes() and cached['original_available'].tobytes()==mask.tobytes(),'Prior150 original control')
    hand,bmask=cached['earnings_expanded'],cached['expanded_available']
    require(hand.dtype==np.float64 and bmask.dtype==bool and hand.shape==bmask.shape==(122,49) and np.array_equal(np.isfinite(hand),bmask) and not np.isinf(hand).any(),'Prior150 hand/mask')
    require(prepared['earnings_batch1'].tobytes()==hand.tobytes() and prepared['batch1_available'].tobytes()==bmask.tobytes(),'Prior150 hand/mask exact bytes')
    return hand,bmask


def verify_prefixes(arrays,meta,reference,batch1,batch2,dataset,expected,scores,choices):
    for n in range(len(meta['dates'])+1):
        subset=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
        keep=lambda e:n and e.get('release_date',e.get('publication_date'))<meta['dates'][n-1]
        history=dict(dataset,events=[e for e in dataset['events'] if keep(e)],latest_event_catalog=[e for e in dataset['latest_event_catalog'] if keep(e)])
        prefix=features_oracle(subset,reference[:n],batch1[:n],batch2[:n],history)
        exact(prefix['cells'],expected['cells'][:n*49])
        for key in EXTRA: np.testing.assert_array_equal(prefix['arrays'][key],expected['arrays'][key][:n])
        ss,cc,_=score_oracle(arrays,meta,prefix['arrays'][EXTRA[0]],prefix['arrays'][EXTRA[1]],prefix['arrays'][EXTRA[2]],prefix['arrays'][EXTRA[3]],n=n)
        for name in NAMES:
            require(ss[name].tobytes()==scores[name][:n].tobytes() and cc[name]==choices[name][:n],'Score/order prefix')
    for fill in (0.,np.nan):
        hand=np.full(reference.shape,np.nan)
        for i,pool in enumerate(meta['symbols']):
            for j,s in enumerate(pool):
                if s not in meta['etfs']: hand[i,j]=fill
        ss,cc,_=score_oracle(arrays,meta,hand,hand,hand,hand)
        for name in NAMES[1:]:
            require(ss[name].tobytes()==scores[NAMES[0]].tobytes() and cc[name]==choices[NAMES[0]],'Exact zero/missing fallback')


def audit(root, plan_path, plan_sha):
    require(type(plan_sha) is str and re.fullmatch('[0-9a-f]{64}',plan_sha), 'Explicit PLAN SHA before I/O')
    reader = old.Reader(root)
    directory = reader.path(STUDY)
    require(reader.path(plan_path) == directory/'PLAN.json' and reader.sha(plan_path) == plan_sha, 'Dedicated frozen152 PLAN')
    plan = reader.json(plan_path)
    require(plan.get('round') == 152 and plan.get('status') == 'REGISTERED' and plan.get('arms') == list(NAMES)
            and plan.get('primary_challenger') == NAMES[4] and plan.get('dose') == .01, 'Fixed152 arms/dose')
    require(reader.path(plan['output_directory']) == directory and plan['runtime'] == dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__), 'Runtime/output')
    require(not any((directory/n).exists() for n in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','WALKFORWARD.json','FROZEN.json','RESULTS.json')), 'Prepare before outcomes')
    pins = reader.verify(plan['hashes'])
    require(set(plan['inputs']) == set(INPUTS), 'Named inputs')
    inputs = {k:bound_ref(reader,v,pins) for k,v in plan['inputs'].items()}
    verify_proposal(reader,plan,pins)
    for key in ('actual_model_fits_before_registration','actual_feature_calculations_before_registration','actual_returns_before_registration'):
        require(type(plan.get(key)) is int and plan[key] == 0, 'No actual preregistration calculation')
    require(plan['tolerances']['feature_atol'] == plan['tolerances']['scores_atol'] == ATOL and plan['tolerances']['scores_rtol'] == 0, 'Fixed tolerances')
    sources = {str(path):reader.sha(path) for path in source_files(root)}
    require(closure(reader,plan['independent_prepare_source_hashes'],pins) == sources, 'Independent source closure')
    legacy_plan = verify_legacy(reader,plan,pins)
    verify_prior151(reader,plan,pins)
    raw = closure(reader,plan['raw_source_hashes'],pins)
    require(raw == reader.verify(reader.json(LOCAL+'/PLAN.json')['raw_source_hashes']) and len(raw) == 64, 'Original raw64')
    original_cache = reader.verify(reader.json(LOCAL+'/CACHE_HASHES.json'), reader.path(LOCAL))
    require(all(pins.get(k) == v for k,v in original_cache.items()), 'Original cache bindings')
    for key,file in (('meta','META.json'),('policy','INCUMBENT_POLICY.json'),('original_orders','local_linear_ORDERS.json')):
        require(inputs[key] == reader.path(LOCAL+'/'+file), 'Original input '+key)
    require(inputs['prices'] == reader.path(old.PRICES), 'Original prices')
    helper = old.original_helper()
    policy = reader.json(inputs['policy'])
    require(all(policy.get(k) == v for k,v in helper.POLICY.items()), 'Frozen episode policy')
    cache = reader.json(directory/'CACHE.json')
    require(set(cache) == set(CACHE_FILES), 'Six cache files')
    reader.verify(cache,directory)
    meta = reader.json(directory/'META.json')
    require(meta == reader.json(inputs['meta']) == reader.json(LEGACY+'/META.json'), 'Identical original metadata')
    require(len(meta['dates']) == len(meta['symbols']) == 122 and meta['train_months'] == 88 and meta['early_months'] == 53
            and len(meta['etfs']) == 17 and all(len(row) == 49 for row in meta['symbols']), 'Original122x49')
    original, arrays = reader.npz(LOCAL+'/DATA.npz'), reader.npz(directory/'DATA.npz')
    verify_original_arrays(original,arrays)
    reference,mask = verify_reference(original,arrays,reader.npz(LEGACY+'/DATA.npz'))
    prior_arrays=reader.npz(PRIOR150+'/DATA.npz')
    batch1,bmask=verify_prior_reference(original,arrays,prior_arrays,reference,mask)
    batch2_arrays=reader.npz(PRIOR151+'/DATA.npz')
    batch2,b2mask=verify_prior151_reference(original,arrays,batch2_arrays,reference,mask,batch1,bmask)
    require(meta==reader.json(PRIOR150+'/META.json')==reader.json(PRIOR151+'/META.json'),'Prior metadata')
    require(reader.sha(old.LOADER) == old.LOADER_SHA, 'Frozen raw price loader')
    spec = importlib.util.spec_from_file_location('coverage152_independent_price_loader',reader.path(old.LOADER))
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    panel,info = loader.load_panel(meta['etfs'])
    used = dict(info['source_hashes'])
    used[str(reader.path(old.LOADER))] = old.LOADER_SHA
    require({str(reader.path(k)):v for k,v in used.items()} == raw, 'Exact raw64 consumption')
    reader.verify(used)
    prices = reader.npz(inputs['prices'])
    require(set(prices) == {'dates','symbols','close','opening'}, 'Price cache schema')
    parity = old.price_parity(panel,prices)
    old.compare(reader.json(directory/'PRICE_PARITY.json'),parity,'PRICE_PARITY')
    calendar_path = inputs['sessions'].parent/'CALENDAR_PLAN.json'
    require(reader.sha(calendar_path) == CALENDAR_SHA, 'Pinned calendar')
    calendar = reader.json(calendar_path)
    require(calendar['start'] == '2015-01-01' and calendar['end'] == '2027-12-31', 'Calendar coverage')
    closure(reader,calendar['bindings'],pins)
    sessions = [str(d.date()) for d in old.official_sessions(reader,calendar['end']) if str(d.date()) >= calendar['start']]
    saved = reader.json(inputs['sessions'])
    require(saved['plan_sha256'] == CALENDAR_SHA and saved['sessions'] == sessions, 'Official session overlay')
    dataset,batch2dataset,batch1dataset = verify_coverage(reader,plan,pins,legacy_plan,sessions)
    old_expected=prior.features_oracle(meta,reference,batch1dataset)
    for key in prior.EXTRA:
        np.testing.assert_array_equal(prior_arrays[key],old_expected['arrays'][key])
    previous_expected=previous.features_oracle(meta,reference,batch1,batch2dataset)
    for key in previous.EXTRA: np.testing.assert_array_equal(batch2_arrays[key],previous_expected['arrays'][key])
    expected = features_oracle(meta,reference,batch1,batch2,dataset)
    verify_audit(reader.json(directory/'FEATURE_AUDIT.json'),expected,plan)
    for key in EXTRA:
        require(arrays[key].dtype == expected['arrays'][key].dtype and arrays[key].shape == (122,49), 'Feature axes/dtype')
        np.testing.assert_array_equal(arrays[key],expected['arrays'][key])
    scores,choices,deltas = score_oracle(arrays,meta,expected['arrays'][EXTRA[0]],expected['arrays'][EXTRA[1]],expected['arrays'][EXTRA[2]],expected['arrays'][EXTRA[3]])
    saved_scores,saved_orders = reader.npz(directory/'SCORES.npz'), reader.json(directory/'PRETRAIN_ORDERS.json')
    verify_scores_and_orders(saved_scores,saved_orders,meta,scores,choices)
    prior_scores,prior_orders = reader.npz(LEGACY+'/SCORES.npz'), reader.json(LEGACY+'/PRETRAIN_ORDERS.json')
    for name,oldname in zip(NAMES[:2],('incumbent_local67','earnings_control')):
        require(saved_scores[name].tobytes() == prior_scores[oldname].tobytes() and saved_orders[name] == prior_orders[oldname], 'Exact149 score/order preservation')
    helper.assert_exact_tree(saved_orders[NAMES[0]],reader.json(inputs['original_orders']))
    batch_scores,batch_orders=reader.npz(PRIOR150+'/SCORES.npz'),reader.json(PRIOR150+'/PRETRAIN_ORDERS.json')
    for name,oldname in zip(NAMES[:3],prior.NAMES):
        require(saved_scores[name].tobytes()==batch_scores[oldname].tobytes() and saved_orders[name]==batch_orders[oldname],'Exact150 score/order preservation')
    old_scores,old_orders=reader.npz(PRIOR151+'/SCORES.npz'),reader.json(PRIOR151+'/PRETRAIN_ORDERS.json')
    for name,oldname in zip(NAMES[:4],previous.NAMES):
        require(saved_scores[name].tobytes()==old_scores[oldname].tobytes() and saved_orders[name]==old_orders[oldname],'Exact151 score/order preservation')
    require(saved_scores[NAMES[0]].tobytes() == scores[NAMES[0]].tobytes(), 'Exact incumbent score bytes')
    verify_prefixes(arrays,meta,reference,batch1,batch2,dataset,expected,scores,choices)
    required = {str(directory/name):reader.sha(directory/name) for name in (*CACHE_FILES,'PLAN.json','CACHE.json')}
    allowed = dict(pins,**required)
    for path,digest in list(reader.seen.items()):
        require(reader.sha(path) == digest == allowed.get(path), 'Unbound consumed artifact '+path)
    return dict(schema_version=1,stage='prepare',passed=True,plan_sha256=plan_sha,
                cache_sha256=reader.sha(directory/'CACHE.json'),orders_sha256=reader.sha(directory/'PRETRAIN_ORDERS.json'),
                **{k:True for k in CHECKS},source_hashes=sources,artifact_hashes=dict(reader.seen),
                producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,new_model_fits=0,
                orders_checked=610,source_cells_checked=5978,stock_cells_checked=3904,
                original_available_stock_cells=int(mask.sum()),batch1_available_stock_cells=int(bmask.sum()),batch2_available_stock_cells=int(b2mask.sum()),available_stock_cells=int(expected['arrays']['expanded_available'].sum()),
                exact_integer_closed_form_arithmetic=True,source_admission_bound=True,
                raw_prices_independently_verified=True,calendar_overlay_independently_verified=True,all_signal_prefixes_verified=True,
                max_score_error=max(float(np.max(np.abs(saved_scores[name]-scores[name]))) for name in NAMES),
                limitations=['Conditional source research only; original source assumptions and all inherited holds remain.',
                             'Original149 earnings and common mask are read from its finalized cache, not recalculated.',
                             'No returns, models or selection metrics are calculated during preparation review.'])


def synthetic_receipt(result,root=None):
    root = ROOT if root is None else Path(root).resolve()
    files = source_files(root)
    reviewed = [root/'work'/name for name in ('monthly_financial_coverage152_runner.py','monthly_financial_coverage152_factor.py',
                                             'monthly_financial_coverage152_proposal.json','monthly_financial_coverage152_selection.py')]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(schema_version=1,stage='synthetic_prepare_review_only',passed=result.wasSuccessful(),tests_run=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
                source_hashes={str(p):digest(p) for p in files},reviewed_contract_hashes={str(p):digest(p) for p in reviewed},
                runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),
                method=METHOD,arms=list(NAMES),features=list(EXTRA),required_checks=list(CHECKS),cache_files=list(CACHE_FILES),
                required_inputs=list(INPUTS),actual_preparation_checked=False,actual_factors_calculated=0,actual_returns_calculated=0,
                producer_modules_imported=False,source_gate_separate_from_signal_day_validity=True,
                actual_api='./run_local.sh work/monthly_financial_coverage152_prepare_review.py --plan <PLAN.json> --plan-sha <SHA256> --prepare-complete',
                actual_output=STUDY+'/INDEPENDENT_PREPARE.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe',action='store_true')
    parser.add_argument('--plan-sha')
    parser.add_argument('--plan',type=Path,default=ROOT/STUDY/'PLAN.json')
    parser.add_argument('--prepare-complete',action='store_true')
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(dict(stage='oracles_and_registered_audit_ready',method=METHOD,arms=NAMES,features=EXTRA,checks=CHECKS,actual_checks_run=False)))
        return
    require(args.prepare_complete,'Explicit completed preparation required')
    result = audit(ROOT,args.plan,args.plan_sha)
    path = ROOT/STUDY/'INDEPENDENT_PREPARE.json'
    with path.open('x') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True,output=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))

if __name__ == '__main__':
    main()

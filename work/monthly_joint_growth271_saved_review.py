"""Read saved271 evidence without a panel/NPZ load or portfolio replay.

The separately authored pure271 oracle rechecks saved253-price inputs. This is
not a fresh source/core reconstruction. The reader author also wrote the271
producer, so numerical independence comes from that pinned separate oracle.
Only --self-test runs before actual successful phase terminals are supplied.
Actual review prints JSON; the caller publishes it only after a clean terminal.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_joint_growth271'
PRIOR = ROOT / 'outputs/monthly_forecast_spacing269'
SOURCE = ROOT / 'outputs/monthly_basket_core251'
CONTEXT = ROOT / 'work/monthly_joint_growth271_context.json'
TESTS = ROOT / 'work/monthly_joint_growth271_TESTS.json'
TEST_TERMINAL = ROOT / 'work/monthly_joint_growth271_TEST_TERMINAL.json'
TEST_LOG = ROOT / 'work/monthly_joint_growth271_FORMAL_TESTS.log'
REVIEW = ROOT / 'work/monthly_joint_growth271_CODE_REVIEW.json'
SOURCE_REVIEW = ROOT / 'work/basket_core251_SOURCE_CONTRACT.json'
ARMS = ('incumbent_local67', 'member_log_growth', 'joint_log_growth')
PATHS = ('same_close_zero', 'next_open_zero', 'next_open_cost')
KINDS = ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')
PREFIXES = (0, 1, 53, 88)
FORMAL = tuple('work/' + name for stem in ('host', 'adapter', 'kernel', 'independent')
               for name in ('monthly_joint_growth271_' + stem + '.py',
                            'test_monthly_joint_growth271_' + stem + '.py'))
PREP = {'SOURCE_FEATURES.json', 'INPUTS.json', 'ORDERS.json', 'META.json',
        'AUDITS.json', 'PROOFS.json', 'SOURCE_PARITY.json', 'SCORES.npz', 'PRE_SCORES.npz'}
RESULT = {'RESULTS.json', 'INDEPENDENT_OUTCOME.json'} | {
    a + '_' + p + '_' + k + '.json' for a in ARMS for p in PATHS for k in KINDS}
FIXED = {
    'work/monthly_monotone_tree266_saved_reader.py': 'eb117fa95c4d7940b96797b1a5a2bd08320b64f3663660a9569915f624e56a8d',
    'work/monthly_boundary_pair267_saved_reader.py': '29eb059828074123afdf6365c914b5ccb004fa3e0b1636a77dbef4050f73a06d',
    'work/monthly_gain_concentration265_saved_reader.py': '8990f2e15e423808546da661ae36717a6dd83dd71fee377f918d55599a2decd8',
    'work/ptr169_results_independent.py': '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59',
}
SOURCE_FLAGS = ('passed', 'full_features_supplied', 'original_core_checked', 'price_clocks_checked',
                'original_raw_rank_parity_checked', 'all_Eold_required_sources_checked',
                'Eold_false_new_source_unconsumed', 'visible_listing_bounds_checked', 'exact_discrete')
INCUMBENT_FLAGS = ('passed', 'full_scores_supplied', 'full_pre_scores_supplied',
                   'independent_original_pre_checked', 'exact_incumbent_copy_checked',
                   'original_eligibility_unchanged', 'single_original_episode')
SELECTION_COUNT_KEYS = ('checked_events', 'checked_required_cells', 'checked_trios',
                        'exact_scalar_checks', 'log_display_checks')
WORK_KEYS = ('events', 'symbols_per_event', 'source_feature_calls', 'independent_source_feature_calls',
             'producer_original_episode_calls', 'independent_original_episode_calls', 'candidate_episode_calls',
             'consumed_panel_copy_cells', 'producer_original_history_price_observations',
             'independent_original_history_price_observations', 'producer251_required_price_observations',
             'independent251_required_price_observations', 'producer251_momentum_evaluations',
             'independent251_momentum_evaluations', 'producer251_block_return_evaluations',
             'independent251_block_return_evaluations', 'adapter_required_price_observations',
             'independent_window_price_observations', 'exact_window_value_comparisons',
             'producer_selection_counts', 'independent_selection_counts', 'new_model_fits', 'predictive_model_fits')


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def pinned_module(path, expected, name):
    need(sha(path) == expected, 'Frozen helper changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# These modules only define metadata/accounting checks at import time.
old = pinned_module(ROOT / 'work/monthly_monotone_tree266_saved_reader.py',
                    FIXED['work/monthly_monotone_tree266_saved_reader.py'], 'saved271_account_rules')
evidence_module = pinned_module(ROOT / 'work/monthly_boundary_pair267_saved_reader.py',
                                FIXED['work/monthly_boundary_pair267_saved_reader.py'], 'saved271_evidence_rules')
Evidence = evidence_module.Evidence
exact, integer, finite, local = old.exact, old.integer, old.finite, old.local


def roles(value):
    need(type(value) is dict and set(value) == set(ARMS), 'Three exact role keys')
    return {b: value[a] for a, b in zip(ARMS, old.ARMS)}


class Accounts:
    def __init__(self, evidence):
        self.evidence = evidence
        self.paths = {old.OUT / 'results' / (a + '_' + p + '_' + k + '.json'):
                      OUT / 'results' / (b + '_' + p + '_' + k + '.json')
                      for a, b in zip(old.ARMS, ARMS) for p in PATHS for k in KINDS}

    def read(self, path):
        need(path in self.paths, 'Exact saved account path')
        return self.evidence.read(self.paths[path])


def accounting(evidence, result, certificate, training):
    adapted_result = dict(result, arms=roles(result['arms']))
    adapted_certificate = dict(certificate, paths=roles(certificate['paths']))
    report, exports = old.accounting(Accounts(evidence), adapted_result, adapted_certificate, roles(training))
    renamed = dict(zip(old.ARMS, ARMS))
    return report, {(renamed[a], p): data for (a, p), data in exports.items()}


def required_metrics(metrics, incumbent):
    need(type(metrics) is dict and set(metrics) == set(PATHS[:2]), 'Exactly two ZERO training paths')
    undefined = []
    for path, groups in metrics.items():
        need(type(groups) is dict and set(groups) == {'training', 'early', 'recent'}, 'First88 groups only')
        for group in groups:
            need(finite(groups[group]['cagr']), 'Finite required CAGR')
        row = groups['training']
        need(finite(row['max_drawdown']), 'Finite training drawdown')
        if row['sharpe'] is None:
            need(not incumbent and finite(row.get('daily_return_sd')) and row['daily_return_sd'] == 0.
                 and row.get('sharpe_status') == 'undefined_sharpe'
                 and row.get('sharpe_reason') == 'zero_daily_return_sd', 'Undefined Sharpe evidence')
            undefined.append(path)
        else:
            need(finite(row['sharpe']), 'Finite required Sharpe')
            if 'daily_return_sd' in row:
                need(finite(row['daily_return_sd']) and row['daily_return_sd'] > 0.
                     and row.get('sharpe_status') == 'defined' and row.get('sharpe_reason') is None,
                     'Defined Sharpe evidence')
    return undefined


def guard(candidate, reference, gain):
    a, b = candidate['training'], reference['training']
    return (a['cagr'] >= b['cagr'] + gain - 1e-12 and a['sharpe'] >= b['sharpe'] - .05
            and a['max_drawdown'] >= b['max_drawdown'] - .03
            and all(candidate[k]['cagr'] >= reference[k]['cagr'] - .025 for k in ('early', 'recent')))


def selection(metrics, orders):
    need(type(metrics) is dict and set(metrics) == set(orders) == set(ARMS), 'Selection roles')
    required_metrics(metrics[ARMS[0]], True)
    need(all(len(orders[a]) >= 88 for a in ARMS), 'First88 orders required')
    for arm in ARMS:
        for i in range(88):
            symbols = orders[arm][i]['symbols']
            need(type(symbols) is list and len(symbols) in (0, 2, 3)
                 and all(type(s) is str and s for s in symbols) and len(set(symbols)) == len(symbols),
                 'Well-formed first88 holdings')
    accepted, guards, changes, undefined = [], {}, {}, {}
    for arm in ARMS[1:]:
        undefined[arm] = required_metrics(metrics[arm], False)
        changes[arm] = [i for i in range(88) if set(orders[arm][i]['symbols']) != set(orders[ARMS[0]][i]['symbols'])]
        guards[arm] = dict(
            close='same_close_zero' not in undefined[arm]
            and guard(metrics[arm]['same_close_zero'], metrics[ARMS[0]]['same_close_zero'], .005),
            open='next_open_zero' not in undefined[arm]
            and guard(metrics[arm]['next_open_zero'], metrics[ARMS[0]]['next_open_zero'], 0.),
            support=len(changes[arm]) >= 3 and any(i < 53 for i in changes[arm])
            and any(i >= 53 for i in changes[arm]))
        if all(guards[arm].values()):
            accepted.append(arm)
    primary = ARMS[0]
    if accepted:
        primary = sorted(accepted, key=lambda a: (-metrics[a]['same_close_zero']['training']['cagr'],
                                                  -metrics[a]['same_close_zero']['training']['sharpe'],
                                                  ARMS.index(a)))[0]
    return dict(primary=primary, admitted=bool(accepted), admissible_arms=accepted, guards=guards,
                changed_months=changes, undefined_sharpe=undefined,
                first88_only=True, later34_used=False, cost_used=False)


def terminal_records(record, plan_sha, hashes):
    need(record['plan_sha256'] == plan_sha and set(record['phases']) == {'prepare', 'select', 'diagnose'},
         'Three actual phase terminals')
    for phase, name in (('prepare', 'prepared/COMPLETION.json'), ('select', 'selected/FROZEN.json'),
                        ('diagnose', 'results/COMPLETION.json')):
        row = record['phases'][phase]
        need(set(row) == {'session_id', 'exit_code', 'chunk_id', 'artifact_path', 'artifact_sha256'}, 'Terminal fields')
        integer(row['exit_code'], 'Actual exit', 0)
        need(row['session_id'] is None or type(row['session_id']) is int and row['session_id'] > 0,
             'Actual session identity')
        need(type(row['chunk_id']) is str and bool(row['chunk_id'].strip()), 'Actual terminal chunk')
        need(local(row['artifact_path']) == OUT / name and row['artifact_sha256'] == hashes[phase],
             'Actual terminal artifact digest')


def source_gate(check, n):
    expected = dict.fromkeys(SOURCE_FLAGS, True)
    expected.update(checked_events=n, C_source_atol=1e-12, C_source_rtol=0., continuous_atol=2e-10,
                    continuous_rtol=2e-10, rank_tolerance=0., producer_modules_imported=False,
                    computes_returns=False, candidate_selection_computed=False, episode_calls=0,
                    new_target_ranks_computed=False, new_basket_selection_computed=False)
    exact(check, expected, 'Saved original source gate')


def incumbent_gate(check, n):
    expected = dict.fromkeys(INCUMBENT_FLAGS, True)
    expected.update(checked_events=n, ordered_choices_checked=n, numerical_tolerance=0.,
                    producer_modules_imported=False, computes_returns=False, candidate_selection_computed=False)
    exact(check, expected, 'Saved original incumbent gate')


def work_check(work, features, audits, proofs, width=49):
    need(type(work) is dict and set(work) == set(WORK_KEYS), 'Exact adapter work fields')
    for key in WORK_KEYS:
        if key not in ('producer_selection_counts', 'independent_selection_counts'):
            integer(work[key], 'Adapter work ' + key)
    n = len(features)
    required = sum(len(row['required_indices']) for row in features if not row['initial_partial'])
    for key, expected in (
        ('events', n), ('symbols_per_event', width), ('source_feature_calls', 1),
        ('independent_source_feature_calls', 1), ('producer_original_episode_calls', 1),
        ('independent_original_episode_calls', 1), ('candidate_episode_calls', 0),
        ('new_model_fits', 0), ('predictive_model_fits', 0)):
        integer(work[key], key, expected)
    for key in ('producer251_required_price_observations', 'independent251_required_price_observations',
                'adapter_required_price_observations', 'independent_window_price_observations',
                'exact_window_value_comparisons'):
        integer(work[key], key, 253 * required)
    for prefix in ('producer251', 'independent251'):
        integer(work[prefix + '_momentum_evaluations'], prefix + ' momentum', required)
        integer(work[prefix + '_block_return_evaluations'], prefix + ' blocks', 12 * required)
    exact(work['producer_selection_counts'], {a: audits[a]['counts'] for a in ARMS}, 'Producer work audit binding')
    exact(work['independent_selection_counts'],
          {a: {k: proofs[a][k] for k in SELECTION_COUNT_KEYS} for a in ARMS}, 'Independent work proof binding')
    need(work['consumed_panel_copy_cells'] == work['producer_original_history_price_observations'],
         'Original copied/history cells')
    if not any(not f['initial_partial'] for f in features):
        for key in ('consumed_panel_copy_cells', 'producer_original_history_price_observations',
                    'independent_original_history_price_observations'):
            integer(work[key], key, 0)
    # The complete past panel index is not saved. Do not infer its row count
    # from official-calendar query_position or fabricate an exact source scan.
    return dict(prefix=n, required_cells=required, exact_required_window_values=253 * required,
                original_history_observation_counts='Bound saved claims; full past index not independently reconstructed')


def prefix_audit(full, n):
    """Project already independently verified events and derive all23 counters."""
    out = deepcopy(full)
    out['events'] = out['events'][:n]
    c = dict.fromkeys(full['counts'], 0)
    c.update(events=n, symbols_per_event=len(out['events'][0]['symbols']) if n else 0, selection_calls=1)
    for row in out['events']:
        c['initial_partial_events' if row['initial_partial'] else 'formal_events'] += 1
        if row['branch'] in ('initial_partial', 'incumbent_copy'):
            c['incumbent_copy_events'] += 1
            continue
        required, trios = len(row['measurements']), len(row['trios'])
        c['required_cells'] += required
        c['price_observations'] += 253 * required
        c['momentum_evaluations'] += required
        c['block_gross_evaluations'] += 12 * required
        c['positive_momentum_cells'] += len(row['new_eligible_indices'])
        c['excluded_momentum_cells'] += len(row['excluded_momentum_indices'])
        c['source_rankings'] += 1
        c['shortlist_cells'] += len(row['shortlist_indices'])
        c['trio_evaluations'] += trios
        c['objective_comparisons'] += max(trios - 1, 0)
        c['exact_amgm_block_checks'] += 12 * trios
        c['exact_amgm_total_checks'] += trios
        c[{'cash': 'cash_events', 'two_names': 'two_name_events', 'optimized': 'optimized_events'}[row['branch']]] += 1
    out['counts'] = c
    return out


def prefix_proof(fullproof, choices, audit):
    """Count the exact structural comparisons performed by the pure oracle."""
    exact_count, display_count = 0, 0
    logs = {'control_log', 'joint_log', 'direct_log_gap', 'log_gap_from_exact_ratios'}
    def walk(value, display=False):
        nonlocal exact_count, display_count
        if type(value) is dict:
            for key, item in value.items():
                walk(item, key in logs)
        elif type(value) is list:
            for item in value:
                walk(item)
        elif display:
            display_count += 1
        else:
            exact_count += 1
    walk(choices); walk(audit)
    result = dict(fullproof)
    result.update(checked_events=len(choices), checked_required_cells=audit['counts']['required_cells'],
                  checked_trios=audit['counts']['trio_evaluations'], exact_scalar_checks=exact_count,
                  log_display_checks=display_count)
    return result


def saved_choices(events, orders, meta, arm):
    need(len(events) == len(orders[arm]), 'Decision length')
    result = []
    for i, (event, order) in enumerate(zip(events, orders[arm])):
        need(order['date'] == order['signal_date'] == meta['dates'][i] == event['date']
             and order['end'] == meta['ends'][i], 'Order metadata clock')
        need(event['symbols'] == meta['symbols'][i], 'Input/source symbols')
        names = order['symbols']
        need(type(names) is list and len(names) in (0, 2, 3) and len(set(names)) == len(names)
             and len(order['scores']) == len(names) and all(finite(v) for v in order['scores']), 'EW3 order')
        result.append([event['symbols'].index(name) for name in names])
        if arm != ARMS[0] and event['date'] != '2016-08-17':
            exact(order['scores'], [float(j) for j in range(len(names), 0, -1)], 'Candidate score is order display')
        if event['date'] == '2016-08-17':
            exact(order, orders[ARMS[0]][i], 'Initial partial exact incumbent')
    return result


def source_metadata(features, events, meta, source_review, axis):
    need(len(features) == len(events) == len(meta['dates']), 'Source/input event count')
    positions = {day: i for i, day in enumerate(axis)}
    clocks = {row['query_index']: row for row in source_review['clock_events']}
    need(len(clocks) == len(source_review['clock_events']), 'Unique source clock IDs')
    for i, (f, event) in enumerate(zip(features, events)):
        need(f['query_index'] == i and f['date'] == meta['dates'][i] == event['date']
             and f['symbols'] == event['symbols'] == meta['symbols'][i], 'Saved source identity axis')
        t = positions[f['date']]
        need(type(f['query_position']) is int and f['query_position'] == t and t >= 294
             and t + 1 < len(axis), 'Official original source clock')
        partial = f['date'] == '2016-08-17'
        need(f['initial_partial'] is partial and (partial or axis[t + 1][:7] != f['date'][:7]), 'Partial/monthend scope')
        expected = dict(query_index=i, date=f['date'], initial_partial=partial,
                        numeric_evaluation_required=not partial, symbols=f['symbols'], query_position=t,
                        next_official_session=axis[t + 1], price_start=axis[t - 294], price_end=axis[t - 42],
                        source_start_position=t - 294, source_end_position=t - 42, price_count=253,
                        return_count=252, block_count=12, block_sessions=21,
                        block_boundary_dates=[axis[t - 294 + 21 * k] for k in range(13)])
        for k, v in expected.items():
            exact(clocks[i][k], v, 'Bound source clock ' + k)
        for k in ('next_official_session', 'price_start', 'price_end', 'block_boundary_dates'):
            exact(f[k], expected[k], 'Saved source clock ' + k)
        exact(f['original_eligible_indices'], [j for j, yes in enumerate(event['eligible']) if yes], 'Complete original E')
        exact(f['required_indices'], [] if partial else f['original_eligible_indices'], 'All E required before rank')
        for k in ('new_eligible_indices', 'excluded_momentum_indices', 'shortlist_indices', 'trios', 'common_trio_indices', 'winners'):
            need(f[k] is None, 'Old251 candidate finalizer not consumed')
        need(f['panel_first_session'] == source_review['panel_first_session'], 'Full original history origin claim')
        if not partial:
            need(finite(f['core_source_max_abs_error']) and 0. <= f['core_source_max_abs_error'] <= 1e-12,
                 'Original weighted-core source tolerance')
            need(len(f['cells']) == len(event['symbols']), 'Full original cell inventory')
        for j, value in enumerate(event['prices253']):
            if partial or not event['eligible'][j]:
                need(value is None, 'Unconsumed required-price body absent')


def preparation(proof, features, events, audits, orders, meta, oracle):
    need(set(proof) == {'source_proof', 'incumbent_proof', 'selection_proofs', 'cached_prefixes',
                       'fresh_physical_prefixes', 'full_prefix_reused', 'work', 'original_arrays_unchanged'},
         'Prepared proof exact fields')
    n = len(events)
    need(n == 122 and proof['cached_prefixes'] == list(range(123)) and proof['full_prefix_reused'] == 122
         and proof['original_arrays_unchanged'] is True, '123 cached prefixes and reused full preparation')
    source_gate(proof['source_proof'], n)
    incumbent_gate(proof['incumbent_proof'], n)
    need(set(audits) == set(orders) == set(proof['selection_proofs']) == set(ARMS), 'Three preparation roles')
    choices = {a: saved_choices(events, orders, meta, a) for a in ARMS}
    checked = {a: oracle.verify_selection(events, a, choices[a], audits[a], n=n) for a in ARMS}
    exact(checked, proof['selection_proofs'], 'Actual saved independent pure selection')
    work = [work_check(proof['work'], features, audits, checked)]
    fresh = proof['fresh_physical_prefixes']
    need([x['prefix'] for x in fresh] == list(PREFIXES), 'Four fresh physical prefixes')
    for row in fresh:
        q = row['prefix']
        need(set(row) == {'prefix', 'source_proof', 'incumbent_proof', 'selection_proofs', 'exact', 'work'}
             and row['exact'] is True, 'Exact physical proof fields')
        source_gate(row['source_proof'], q)
        incumbent_gate(row['incumbent_proof'], q)
        # The full oracle has checked every event. Project those exact events,
        # independently derive counts and compare the actual saved prefix proof.
        short_audits = {}
        short_proofs = {}
        for arm in ARMS:
            short_choices = choices[arm][:q]
            short_audit = prefix_audit(audits[arm], q)
            short_audits[arm] = short_audit
            short_proofs[arm] = prefix_proof(checked[arm], short_choices, short_audit)
        exact(short_proofs, row['selection_proofs'], 'Physical independent proof')
        work.append(work_check(row['work'], features[:q], short_audits, short_proofs))
    return dict(main_and_four_fresh_prefix_work=work, cached_prefixes=list(range(123)),
                cached_prefix_source='Bound host loop and saved inventory; no per-prefix receipt or fresh source scan is claimed',
                full122_reused=True, source_gate_calls=5, original_producer_episode_calls=5,
                original_independent_episode_calls=5, new_model_fits=0,
                historical_panel_counts_independently_recomputed=False,
                saved_math_rechecks=3, saved_math_scope='3 main oracle checks; 4x3 prefix proofs derived from verified events/counters; no raw panel load')


def formal(evidence, plan):
    for path, expected in FIXED.items():
        evidence.pin(path, expected)
        need(evidence.registered(plan, path) == expected, 'Registered frozen helper')
    context = evidence.read(CONTEXT, evidence.registered(plan, CONTEXT))
    exact(plan['context'], context, 'Embedded frozen context')
    need(context['schema'] == 'monthly_joint_growth271_context_v1' and context['frozen'] is True
         and context['actual_data_started'] is False and context['arms'] == list(ARMS)
         and context['registered_trials'] == 3 and context['new_predictive_fits'] == 0
         and context['fresh_physical_prefixes'] == list(PREFIXES) and context['full_prefix_reused'] == 122
         and context['source270_included'] is False, 'Prospective fixed definition')
    need(plan['context_sha256'] == sha(CONTEXT), 'Context digest')
    expected = {p: evidence.pin(p, evidence.registered(plan, p)) for p in FORMAL}
    exact(context['code_sha256'], expected, 'Eight formal code pins')
    tests = evidence.read(TESTS, evidence.registered(plan, TESTS))
    exact(tests['code_sha256'], expected, 'Formal tests bind current code')
    need(tests['passed'] is True and tests['synthetic_only'] is True and type(tests['tests']) is int
         and tests['tests'] == 112 and tests['runtime'] == plan['runtime']
         == dict(python='3.9.6', numpy='2.0.2', pandas='2.3.3'), '112 formal synthetic tests and runtime')
    review = evidence.read(REVIEW, evidence.registered(plan, REVIEW))
    exact(review['code_sha256'], expected, 'Pre-registration code review pins')
    need(review['passed'] is True and review['unresolved_blockers'] == []
         and review['context_sha256'] == sha(CONTEXT) and review['actual_returns_computed'] is False,
         'Pre-registration review scope')
    for paper in (context, review):
        for p, h in paper['input_sha256'].items():
            need(evidence.registered(plan, p) == h, 'Registered paper dependency')
            evidence.pin(p, h)
    terminal = evidence.read(TEST_TERMINAL)
    need(terminal['passed'] is True and terminal['synthetic_only'] is True and terminal['tests'] == 112
         and terminal['actual_prepare_calls'] == 0 and terminal['actual_returns_computed'] is False,
         'Observed formal synthetic scope')
    need(terminal['command'] == './run_local.sh work/monthly_joint_growth271_host.py test'
         and terminal['terminal'] == dict(session_id=53310, exit_code=0, chunk_id='6026ea'),
         'Actual112 synthetic terminal')
    need(local(terminal['tests_receipt']['path']) == TESTS and terminal['tests_receipt']['sha256'] == sha(TESTS)
         and local(terminal['log']['path']) == TEST_LOG, 'Actual test artifact references')
    evidence.pin(TEST_LOG, terminal['log']['sha256'])
    need('Ran 112 tests' in TEST_LOG.read_text() and '\nOK\n' in TEST_LOG.read_text(), 'Actual112 test log')
    return context, tests, terminal


def source_evidence(evidence, plan, source, meta):
    original = evidence.read(old.LOCAL / 'PLAN.json', evidence.registered(plan, old.LOCAL / 'PLAN.json'))
    rawmaps = [old.canonical_source_map(m) for m in
               (source['raw_source_hashes'], plan['raw_source_hashes'], original['raw_source_hashes'])]
    exact(rawmaps[0], rawmaps[1], 'Raw64 host source map')
    exact(rawmaps[1], rawmaps[2], 'Original raw64 identity')
    for folder in (PRIOR, SOURCE):
        parent = evidence.read(folder / 'PLAN.json', evidence.registered(plan, folder / 'PLAN.json'))
        for p, h in parent['input_sha256'].items():
            # Metadata equality includes external runtime pins without opening
            # or silently updating those files during this saved review.
            q = Path(p)
            if not q.is_absolute() or q.is_relative_to(ROOT):
                need(evidence.registered(plan, p) == h, 'Complete inherited dependency map')
            else:
                need(plan['input_sha256'].get(p) == h, 'Inherited external runtime declaration')
        done = evidence.read(folder / 'COMPLETION.json', evidence.registered(plan, folder / 'COMPLETION.json'))
        need(done['plan_sha256'] == sha(folder / 'PLAN.json') and done['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED',
             'Closed inherited source/predecessor')
    old.full_original_chain(evidence, plan)
    parity = evidence.read(old.REBUILD / 'PARITY.json', evidence.registered(plan, old.REBUILD / 'PARITY.json'))
    need(len(parity['artifacts_sha256']) == 12, 'Complete twelve169 cache members')
    for name, h in parity['artifacts_sha256'].items():
        need(Path(name).name == name and evidence.registered(plan, old.REBUILD / name) == h, 'Registered169 cache')
        evidence.pin(old.REBUILD / name, h)
    exact(meta, evidence.read(old.REBUILD / 'MONTHLY_META.json')['monthly'], 'Original metadata including static extras')
    need(len(source['original_array_fingerprints']) == 27, 'Original27 cache fingerprints')
    for row in source['original_array_fingerprints'].values():
        need(set(row) == {'dtype', 'shape', 'sha256'} and row['shape'][0] == 122
             and type(row['dtype']) is str and old.sha_value(row['sha256']), 'Opaque original array fingerprint')
    price = source['execution_prices']
    need(type(price) is dict and set(price) == {'adjusted_levels_constant_scale', 'adjusted_opening_parity',
         'missing_price_masks_exact', 'positive_price_masks_exact', 'symbols', 'scales', 'all_missing_symbols'},
         'Bound execution-price parity receipt')
    need(all(price[k] is True for k in ('adjusted_levels_constant_scale', 'adjusted_opening_parity',
                                      'missing_price_masks_exact', 'positive_price_masks_exact')),
         'Execution-price parity flags')
    integer(price['symbols'], 'Execution-scale symbols', len(price['scales']))
    need(all(finite(v) and v > 0 for v in price['scales'].values())
         and len(price['all_missing_symbols']) == len(set(price['all_missing_symbols']))
         and not set(price['all_missing_symbols']).intersection(price['scales']), 'Execution-scale inventory')
    source_review = evidence.read(SOURCE_REVIEW, evidence.registered(plan, SOURCE_REVIEW))
    need(source['source_review_sha256'] == sha(SOURCE_REVIEW), 'Source metadata review digest')
    calendar = evidence.read(old.CALENDAR, evidence.registered(plan, old.CALENDAR))
    axis = calendar['sessions']
    need(type(axis) is list and axis == sorted(set(axis)), 'Official session calendar')
    need(source_review['schema'] == 'monthly_basket_core251_metadata_source_review_v1'
         and source_review['passed'] is True and source_review['metadata_only'] is True
         and source_review['events'] == 122 and source_review['width'] == 49
         and source_review['formal_monthends'] == 121 and source_review['panel_first_session'] == '2008-01-02'
         and source_review['calendar_sha256'] == sha(old.CALENDAR), 'Frozen source review scope')
    exact(old.canonical_source_map(source_review['raw_source_hashes']), rawmaps[0], 'Source-review raw64 identity')
    for key in ('source_admitted', 'strict_pit', 'registered_strategy', 'prices_loaded', 'volumes_loaded',
                'raw_source_values_decoded', 'npz_decoded', 'actual_features_orders_returns_computed'):
        need(source_review[key] is False, 'No source-review overclaim ' + key)
    for path, h in source_review['input_sha256'].items():
        need(evidence.registered(plan, path) == h, 'Registered source-review input')
    return source_review, axis


def run(args):
    need(platform.python_version() == '3.9.6', 'Pinned runtime')
    for name in ('plan', 'prepared', 'frozen', 'results', 'terminals'):
        need(old.sha_value(getattr(args, name + '_sha256')), 'Explicit immutable digest ' + name)
    evidence = Evidence()
    evidence.pin(Path(__file__))
    plan = evidence.read(OUT / 'PLAN.json', args.plan_sha256)
    context, tests, test_terminal = formal(evidence, plan)
    for key in ('actual_features_before_registration', 'actual_orders_before_registration',
                'actual_returns_before_registration', 'new_predictive_fits'):
        integer(plan[key], 'PLAN pre-registration work', 0)
    before = evidence.read(OUT / 'STATE_BEFORE.json', plan['prior_central_sha256'])
    need(len(before['completed']) == 265 and sum(x['trials'] for x in before['completed']) == 1306
         and before['registered_queue'] == [] and before['completed'][-1]['directory'] == PRIOR.name,
         'Central pre-state and predecessor')
    prior = evidence.read(PRIOR / 'COMPLETION.json', evidence.registered(plan, PRIOR / 'COMPLETION.json'))
    need(before['completed'][-1]['completion']['sha256'] == sha(PRIOR / 'COMPLETION.json')
         and prior['protected_files'] == 21444 and prior['observers_modified'] is False
         and prior['target_reached'] is False, 'Predecessor closure')
    priorreview = ROOT / 'work/monthly_forecast_spacing269_RESULT_REVIEW.json'
    need(prior['result_review_sha256'] == evidence.pin(priorreview, evidence.registered(plan, priorreview)),
         'Predecessor saved review')
    protected = evidence.read(old.PROTECTED, evidence.registered(plan, old.PROTECTED))
    need(len(protected) == plan['protected_files'] == 21444, 'Registered protected inventory')
    terminals = evidence.read(OUT / 'PHASE_TERMINALS.json', args.terminals_sha256)
    terminal_records(terminals, args.plan_sha256,
                     dict(prepare=args.prepared_sha256, select=args.frozen_sha256, diagnose=args.results_sha256))
    pc = evidence.phase(OUT / 'prepared', args.prepared_sha256, args.plan_sha256, PREP)
    need(pc['events'] == 122 and pc['new_returns_computed'] is False and pc['new_predictive_fits'] == 0,
         'Complete preparation scope')
    frozen = evidence.read(OUT / 'selected/FROZEN.json', args.frozen_sha256)
    training = evidence.read(OUT / 'selected/TRAIN_RESULTS.json', frozen['training_sha256'])
    need({p.name for p in (OUT / 'selected').iterdir()} == {'FROZEN.json', 'TRAIN_RESULTS.json'}
         and frozen['plan_sha256'] == args.plan_sha256 and frozen['prepared_sha256'] == args.prepared_sha256
         and frozen['orders_sha256'] == pc['artifacts_sha256']['ORDERS.json'], 'Frozen first88 selection bindings')
    rc = evidence.phase(OUT / 'results', args.results_sha256, args.plan_sha256, RESULT)
    need(rc['frozen_sha256'] == args.frozen_sha256 and rc['paths'] == 9 and rc['slices'] == 45,
         'Nine-path diagnostics follow frozen selection')
    need(pc['created_utc'] <= frozen['created_utc'] <= rc['created_utc'], 'Saved phase time ordering')
    meta = evidence.read(OUT / 'prepared/META.json')
    need(len(meta['dates']) == len(meta['ends']) == len(meta['symbols']) == 122
         and meta['dates'][0] == '2016-08-17' and meta['ends'][-1] == '2026-09-24'
         and meta['ends'][:-1] == meta['dates'][1:] and meta['ends'][52] == '2020-12-31'
         and meta['ends'][87] == '2023-11-30' and all(len(x) == len(set(x)) == 49 for x in meta['symbols']),
         'Fixed event and segment axes')
    orders = evidence.read(OUT / 'prepared/ORDERS.json')
    exact(orders[ARMS[0]], evidence.read(old.LOCAL / 'local_linear_ORDERS.json',
          evidence.registered(plan, old.LOCAL / 'local_linear_ORDERS.json')), 'Original Local67 orders')
    source = evidence.read(OUT / 'prepared/SOURCE_PARITY.json')
    review, axis = source_evidence(evidence, plan, source, meta)
    features = evidence.read(OUT / 'prepared/SOURCE_FEATURES.json')
    events = evidence.read(OUT / 'prepared/INPUTS.json')
    source_metadata(features, events, meta, review, axis)
    audits = evidence.read(OUT / 'prepared/AUDITS.json')
    proof = evidence.read(OUT / 'prepared/PROOFS.json')
    oracle_path = ROOT / 'work/monthly_joint_growth271_independent.py'
    oracle = pinned_module(oracle_path, context['code_sha256']['work/monthly_joint_growth271_independent.py'],
                           'saved271_separate_math_oracle')
    prep_review = preparation(proof, features, events, audits, orders, meta, oracle)
    chosen = selection(training, orders)
    exact(chosen, frozen['selection'], 'Independent first88 choice')
    result = evidence.read(OUT / 'results/RESULTS.json')
    certificate = evidence.read(OUT / 'results/INDEPENDENT_OUTCOME.json')
    exact(result['selection'], chosen, 'Irrevocable selection in result')
    for key in ('reused_sample', 'zero_cost_selection', 'next_open_cost_diagnostic_only'):
        need(result[key] is True, 'Declared research scope ' + key)
    for key in ('strict_pit', 'observer_promotion'):
        need(result[key] is False, 'No additional admission claim ' + key)
    need(certificate['accounting_source_sha256'] == FIXED['work/ptr169_results_independent.py'],
         'Independent accounting source')
    account, exports = accounting(evidence, result, certificate, training)
    for arm in ARMS:
        for path in PATHS:
            exact([r['date'] for r in exports[arm, path]['DAILY']],
                  [d for d in axis if meta['dates'][0] <= d <= meta['ends'][-1]], 'Official saved daily axis')
            if path != 'next_open_cost':
                for m in (result['arms'][arm][path]['metrics'], result['arms'][arm][path]['slices']['full122']):
                    need(finite(m['cost_per_side']) and m['cost_per_side'] == 0.
                         and finite(m['fees_paid']) and m['fees_paid'] == 0., 'Literal ZERO accounting')
    prc = evidence.read(PRIOR / 'results/COMPLETION.json', prior['result_completion_sha256'])
    previous = evidence.read(PRIOR / 'results/RESULTS.json', prc['artifacts_sha256']['RESULTS.json'])
    oldfees = None
    for path in PATHS:
        for kind in KINDS:
            name = ARMS[0] + '_' + path + '_' + kind + '.json'
            priorrows = evidence.read(PRIOR / 'results' / name, prc['artifacts_sha256'][name])
            if path == 'next_open_cost' and kind == 'HOLDINGS':
                oldfees = priorrows
            else:
                exact(exports[ARMS[0], path][kind], priorrows, 'Incumbent economic export ' + path + '/' + kind)
    fee = old.diagnostic_summary_parity(result['arms'][ARMS[0]], previous['arms'][ARMS[0]],
                                       exports[ARMS[0], 'next_open_cost']['HOLDINGS'], oldfees)
    evidence.finish()
    return dict(schema='monthly_joint_growth271_saved_review_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                passed=True, unresolved_blockers=[], plan_sha256=args.plan_sha256, context_sha256=sha(CONTEXT),
                prepared_completion_sha256=args.prepared_sha256, results_completion_sha256=args.results_sha256,
                frozen_sha256=args.frozen_sha256, phase_terminals_sha256=args.terminals_sha256,
                actual_phase_terminals=terminals, formal_synthetic_tests=tests['tests'], formal_test_terminal=test_terminal,
                preparation_review=prep_review, selection=chosen, saved_accounting_evidence=account,
                incumbent_economic_exports_exact=True, incumbent_summary_exact=fee['exact'], summary_fee_display_review=fee,
                full_results={a: {p: result['arms'][a][p]['metrics'] for p in PATHS} for a in ARMS},
                segmented_results={a: {p: result['arms'][a][p]['slices'] for p in PATHS} for a in ARMS},
                npz_decoded=False, saved_price_windows_math_rechecked=True, source_panel_loaded=False,
                fresh_source_or_original_core_reconstruction=False, portfolio_replay=False, returns_recomputed=False,
                phase_rerun=False, original_files_modified=False, observers_modified=False,
                source270_used=False, strict_pit=False, reused_sample=True,
                protection_and_close=dict(status='pending_root_close', protected_members_rehashed=False,
                                          expected_post_close_completed=266, expected_post_close_trials=1309),
                limitations=['Reader author also authored the271 producer. Saved rational math uses the separately authored pinned oracle.',
                             'Source/core and NPZ numerical truth depend on the bound actual251 source/incumbent gates; this reader hashes arrays but does not decode them or load the original panel.',
                             'Full historical source observation counts are bound saved claims because the complete past panel index is not saved. Exact253-window counters are independently derived.',
                             '123 cached checks are a bound host execution statement; no saved per-cached-prefix receipt exists. Four fresh proofs and full122 reuse are checked separately.',
                             'Independent account statistics and fixed-unit fills are those from the bound accountant; this reader reconciles all36 exports and metric groups without recomputing NAV.',
                             'The21,444 protected-file rehash and central closure remain root close responsibilities; no strategy/observer promotion is performed.'],
                input_sha256=evidence.pins)


def synthetic_metrics():
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for path in PATHS[:2]:
            row = dict(cagr=.20 if arm == ARMS[0] else .21, sharpe=1., max_drawdown=-.2,
                       daily_return_sd=.01, sharpe_status='defined', sharpe_reason=None)
            result[arm][path] = dict(training=deepcopy(row), early={'cagr': .2}, recent={'cagr': .2})
    orders = {a: [{'symbols': ['A', 'B', 'C']} for _ in range(88)] for a in ARMS}
    for arm in ARMS[1:]:
        for i in (2, 3, 60):
            orders[arm][i] = {'symbols': ['A', 'B', 'D']}
    return result, orders


class SyntheticChecks(unittest.TestCase):
    def test_prefix_counters_and_scalar_counts_match_separate_oracle(self):
        oracle_path = ROOT / 'work/monthly_joint_growth271_independent.py'
        oracle = pinned_module(oracle_path, sha(oracle_path), 'saved271_synthetic_only_oracle')
        names = ['A', 'B', 'C', 'D']
        paths = [[float(2 + j) * (1 + k * .001 * (j + 1)) for k in range(253)] for j in range(4)]
        events = [dict(date='2016-08-17', symbols=names, eligible=[True] * 4,
                       prices253=[None] * 4, incumbent_choices=[0, 1, 2]),
                  dict(date='2016-08-31', symbols=names, eligible=[True] * 4,
                       prices253=paths, incumbent_choices=[0, 1, 2]),
                  dict(date='2016-09-30', symbols=names, eligible=[True, True, False, False],
                       prices253=paths[:2] + [None, None], incumbent_choices=[0, 1]),
                  dict(date='2016-10-31', symbols=names, eligible=[True] * 4,
                       prices253=[[2.] * 253 for _ in names], incumbent_choices=[0, 1, 2])]
        for arm in ARMS:
            choices, audit = oracle._reference(events, arm, None)
            proof = oracle.verify_selection(events, arm, choices, audit)
            for n in range(5):
                with self.subTest(arm=arm, n=n):
                    prefix_choices, expected = oracle._reference(events, arm, n)
                    got = prefix_audit(audit, n)
                    exact(got, expected, 'Synthetic prefix count projection')
                    exact(prefix_proof(proof, choices[:n], got),
                          oracle.verify_selection(events, arm, prefix_choices, expected, n=n),
                          'Synthetic scalar/check-count projection')

    def test_tie_member_control_and_sharpe_then_exact_cagr(self):
        m, o = synthetic_metrics()
        self.assertEqual(selection(m, o)['primary'], ARMS[1])
        m[ARMS[2]][PATHS[0]]['training']['sharpe'] = 1.1
        self.assertEqual(selection(m, o)['primary'], ARMS[2])
        m[ARMS[1]][PATHS[0]]['training']['cagr'] += 1e-14
        self.assertEqual(selection(m, o)['primary'], ARMS[1])

    def test_each_guard_and_unsupported_fallback(self):
        base, o = synthetic_metrics()
        mutations = [('training', 'cagr', .19), ('training', 'sharpe', .94),
                     ('training', 'max_drawdown', -.231), ('early', 'cagr', .174), ('recent', 'cagr', .174)]
        for path in PATHS[:2]:
            for group, key, value in mutations:
                with self.subTest(path=path, key=key, group=group):
                    m = deepcopy(base)
                    for a in ARMS[1:]:
                        m[a][path][group][key] = value
                    self.assertEqual(selection(m, o)['primary'], ARMS[0])
        for a in ARMS[1:]:
            o[a][60] = deepcopy(o[ARMS[0]][60])
        self.assertEqual(selection(base, o)['primary'], ARMS[0])

    def test_undefined_candidate_and_invalid_incumbent(self):
        m, o = synthetic_metrics()
        for a in ARMS[1:]:
            m[a][PATHS[1]]['training'].update(sharpe=None, daily_return_sd=0.,
                                            sharpe_status='undefined_sharpe', sharpe_reason='zero_daily_return_sd')
        self.assertEqual(selection(m, o)['primary'], ARMS[0])
        m[ARMS[0]][PATHS[0]]['training']['sharpe'] = None
        with self.assertRaises(ValueError):
            selection(m, o)

    def test_later_orders_never_accessed(self):
        class Poison:
            def __getitem__(self, key): raise AssertionError('Future access')
        m, o = synthetic_metrics()
        before = selection(m, o)
        for a in ARMS:
            o[a].extend([Poison()] * 34)
        exact(selection(m, o), before, 'First88 only')

    def test_source_and_incumbent_schema_no_flag_only_acceptance(self):
        for checker in (source_gate, incumbent_gate):
            with self.assertRaises((ValueError, KeyError)):
                checker({'passed': True}, 0)

    def test_phase_terminal_failures(self):
        hashes = dict(prepare='1' * 64, select='2' * 64, diagnose='3' * 64)
        record = dict(plan_sha256='a' * 64, phases={})
        for phase, name in (('prepare', 'prepared/COMPLETION.json'), ('select', 'selected/FROZEN.json'),
                            ('diagnose', 'results/COMPLETION.json')):
            record['phases'][phase] = dict(session_id=1, exit_code=0, chunk_id='synthetic',
                                           artifact_path=str(OUT / name), artifact_sha256=hashes[phase])
        terminal_records(record, 'a' * 64, hashes)
        for key, value in (('exit_code', 1), ('exit_code', False), ('session_id', 0),
                           ('chunk_id', ''), ('artifact_sha256', '0' * 64)):
            bad = deepcopy(record)
            bad['phases']['prepare'][key] = value
            with self.assertRaises(ValueError):
                terminal_records(bad, 'a' * 64, hashes)

    def test_all36_account_paths_and_no_global_mutation(self):
        class Fake:
            def __init__(self): self.got = []
            def read(self, path): self.got.append(path); return []
        before = (old.OUT, old.ARMS)
        fake = Fake(); bridge = Accounts(fake)
        self.assertEqual(len(bridge.paths), 36)
        for path in bridge.paths:
            bridge.read(path)
        self.assertEqual(len(set(fake.got)), 36)
        self.assertTrue(all(p.parent == OUT / 'results' for p in fake.got))
        self.assertEqual((old.OUT, old.ARMS), before)
        with self.assertRaises(ValueError):
            bridge.read(OUT / 'results/unknown.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    for key in ('plan', 'prepared', 'frozen', 'results', 'terminals'):
        parser.add_argument('--' + key + '-sha256')
    options = parser.parse_args()
    if options.self_test:
        need(all(getattr(options, key + '_sha256') is None for key in
                 ('plan', 'prepared', 'frozen', 'results', 'terminals')), 'Self-tests cannot receive actual digests')
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticChecks)
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if outcome.wasSuccessful() else 1)
    print(json.dumps(run(options), indent=2, sort_keys=True, allow_nan=False))

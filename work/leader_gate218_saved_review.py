"""Review closed round218 saved JSON/hashes only; never replay or fit.

Run only after the root closure, through ./run_local.sh, with its explicit SHA.
NPZ archives are byte-hashed but never decoded. This script recounts saved
orders/audit records; independent portfolio accounting remains a separate receipt.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import re

R = Path(__file__).resolve().parents[1]
O = R / 'outputs/monthly_leader_gate218'
P = R / 'outputs/monthly_persistence217'
CENTRAL = R / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
BASELINE = R / 'outputs/monthly_model_scope70/original65_ORDERS.json'
LOCAL = R / 'outputs/monthly_local_linear70'
CONTEXT = R / 'work/monthly_leader_gate218_context.json'
TESTS = R / 'work/monthly_leader_gate218_TESTS.json'
CODE = R / 'work/leader_gate218_CODE_REVIEW.json'
SOURCE = R / 'work/leader_gate218_SOURCE_REVIEW.json'
CONTEXT_SHA = 'faec461c4716c65c5bd0d4c7f695df5ca8b3bec395d04b416c671d846bfa1a8c'
ARMS = ('incumbent_local67', 'leader6_gate', 'baseline_score')
PATHS = {'same_close_zero': ('close', 0.), 'next_open_zero': ('open', 0.),
         'next_open_cost': ('open', .0005)}
SLICES = {'full122': (0, 122), 'first88': (0, 88), 'early53': (0, 53),
          'recent35': (53, 88), 'later34': (88, 122)}
PREPARED_FILES = {'SHORTLIST.npz', 'SELECTION_AUDIT.json', 'ORDERS.json', 'META.json',
                  'SOURCE_PARITY.json', 'INDEPENDENT_PREPARE.json', 'BASELINE_PARITY.json'}
RESULT_FILES = {a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS
                for k in ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')}
RESULT_FILES |= {'RESULTS.json', 'INDEPENDENT_OUTCOME.json'}
COUNT_FIELDS = ('query_months', 'eligible_cells', 'shortlisted_cells', 'gated_months',
    'tie_expansion_months', 'tie_added_cells', 'changed_holding_months',
    'changed_ordered_months', 'cash_months', 'two_holding_months', 'three_holding_months')
ZERO_FIELDS = ('new_model_fits', 'predictive_model_fits', 'new_ridge_fits',
               'new_tree_fits', 'new_parameters')
EXTERNAL_RUNTIME = {
    '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':
    '6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def source_path(name, digest):
    path = Path(name)
    path = (path if path.is_absolute() else R/path).resolve()
    need('CloudStorage' not in path.parts, 'Cloud dependency forbidden')
    need(path.is_relative_to(R) or EXTERNAL_RUNTIME.get(str(path)) == digest,
         'Unexpected external source '+str(path))
    need(path != CENTRAL.resolve(), 'Mutable central cannot be a registered input')
    return path


def add_pin(inputs, path, digest=None):
    path = Path(path).resolve()
    actual = sha(path)
    need(digest is None or actual == digest, 'Hash mismatch: '+str(path))
    key = str(path.relative_to(R)) if path.is_relative_to(R) else str(path)
    need(key not in inputs or inputs[key] == actual, 'Conflicting evidence pin')
    inputs[key] = actual
    return actual


def record_counts(records):
    result = dict.fromkeys(COUNT_FIELDS, 0)
    result['query_months'] = len(records)
    for row in records:
        k, q = row['eligible_count'], row['shortlist_count']
        result['eligible_cells'] += k
        result['shortlisted_cells'] += q
        result['gated_months'] += int(k > 6)
        result['tie_expansion_months'] += int(row['tie_added_count'] > 0)
        result['tie_added_cells'] += row['tie_added_count']
        result['changed_holding_months'] += int(row['changed_holdings'])
        result['changed_ordered_months'] += int(row['changed_order'])
        result['cash_months'] += int(k < 2)
        result['two_holding_months'] += int(k == 2)
        result['three_holding_months'] += int(k >= 3)
    return result


def saved_metric_near(actual, expected):
    # Preserve the independently registered169 accounting tolerance.
    return (math.isfinite(actual) and math.isfinite(expected)
            and abs(actual-expected) <= 1e-7 + 2e-11*abs(expected))


def guard(candidate, reference, gain):
    return (candidate['training']['cagr'] >= reference['training']['cagr'] + gain - 1e-12
        and candidate['training']['sharpe'] >= reference['training']['sharpe'] - .05
        and candidate['training']['max_drawdown'] >= reference['training']['max_drawdown'] - .03
        and all(candidate[e]['cagr'] >= reference[e]['cagr'] - .025 for e in ('early', 'recent')))


def selection_recount(training, changes):
    candidate, incumbent, control = (training[k] for k in (ARMS[1], ARMS[0], ARMS[2]))
    changed = [i for i in changes['leader6_gate_vs_incumbent_local67']['indices'] if i < 88]
    changed_control = [i for i in changes['leader6_gate_vs_baseline_score']['indices'] if i < 88]
    def support(indices):
        return len(indices) >= 3 and any(i < 53 for i in indices) and any(i >= 53 for i in indices)
    guards = dict(
        close_vs_incumbent=guard(candidate['same_close_zero'], incumbent['same_close_zero'], .005),
        open_vs_incumbent=guard(candidate['next_open_cost'], incumbent['next_open_cost'], 0.),
        close_vs_baseline_control=guard(candidate['same_close_zero'], control['same_close_zero'], 0.),
        open_vs_baseline_control=guard(candidate['next_open_cost'], control['next_open_cost'], 0.),
        changed_support=support(changed), changed_support_vs_control=support(changed_control))
    return dict(primary=ARMS[1] if all(guards.values()) else ARMS[0], admitted=all(guards.values()),
                guards=guards, changed_months=changed, changed_months_vs_control=changed_control,
                first88_only=True, later34_used=False)


def main(expected):
    need(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected),
         'Explicit lowercase closure SHA-256 required')
    need(sha(O/'COMPLETION.json') == expected, 'Explicit final closure SHA required')
    targets = (O/'RESEARCH_REVIEW.json', O/'RESEARCH_REVIEW.md')
    need(not any(path.exists() for path in targets), 'Saved review is write-once')
    inputs = {}
    closure, plan = read(O/'COMPLETION.json'), read(O/'PLAN.json')
    prepared, closed = read(O/'prepared/COMPLETION.json'), read(O/'results/COMPLETION.json')
    plan_sha = add_pin(inputs, O/'PLAN.json')
    need(closure['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED'
         and closure['protected_files'] == 21444 and closure['observers_modified'] is False,
         'Final closure scope')
    for receipt in (closure, prepared, closed):
        need(receipt['plan_sha256'] == plan_sha, 'PLAN lineage')
    need(closure['result_completion_sha256'] == sha(O/'results/COMPLETION.json'), 'Result-to-closure chain')
    need(prepared['status'] == closed['status'] == 'COMPLETE', 'Incomplete saved stages')
    need(set(prepared['artifacts_sha256']) == PREPARED_FILES, 'Exact seven preparation artifacts required')
    need(set(closed['artifacts_sha256']) == RESULT_FILES and closed['paths'] == 9 and closed['slices'] == 45,
         'Exact 38 result artifacts / nine paths / 45 slices required')
    need(isinstance(plan['input_sha256'], dict) and plan['input_sha256'], 'Registered inputs required')
    for name, digest in plan['input_sha256'].items():
        add_pin(inputs, source_path(name, digest), digest)
    for sub, receipt in (('prepared', prepared), ('results', closed)):
        for name, digest in receipt['artifacts_sha256'].items():
            add_pin(inputs, O/sub/name, digest)

    context = read(CONTEXT)
    need(sha(CONTEXT) == CONTEXT_SHA == plan['contract']['prospective_context_sha256'], 'Context identity')
    need(plan['contract']['context'] == context and plan['contract']['arms'] == list(ARMS), 'Frozen contract')
    tests, review, source_review = read(TESTS), read(CODE), read(SOURCE)
    for path in (CONTEXT, TESTS, CODE, SOURCE):
        need(plan['input_sha256'].get(str(path)) == sha(path), 'Registered review/test/context pin')
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests'] == 78
         and tests['runtime'] == plan['runtime'], 'Formal 78-test receipt')
    for name, digest in tests['code_sha256'].items():
        path = source_path(name, digest)
        need(plan['input_sha256'].get(str(path)) == digest, 'Formal tested code not registered')
        add_pin(inputs, path, digest)
    need(review['passed'] is True and review['unresolved_blockers'] == []
         and review['context_sha256'] == CONTEXT_SHA, 'Static numerical review')
    need(len(review['code_sha256']) == 6, 'Six reviewed numerical/test files')
    for name, digest in review['code_sha256'].items():
        path = source_path(name, digest)
        need(tests['code_sha256'].get(str(path)) == digest, 'Reviewed/tested source mismatch')
        add_pin(inputs, path, digest)
    need(plan['numerical_code_review']['sha256'] == sha(CODE)
         and plan['cache_source_binding']['sha256'] == sha(SOURCE), 'Registered independent-review lineage')
    need(source_review['passed'] is True and source_review['context_sha256'] == CONTEXT_SHA
         and source_review['no_fresh_raw_pit_claim'] is True, 'Source-review scope')
    # The independent inspected-source list and prospective-context list differ
    # in scope. Verify each registered list, including shared-pin conflicts.
    binding = plan['cache_source_binding']
    evidence_pins = {}
    for label, document in (('review', source_review), ('context', context)):
        evidence = document['evidence_sha256']
        need(isinstance(evidence, dict) and evidence
             and evidence == binding[label+'_evidence_sha256'],
             'Registered '+label+' evidence list differs')
        for name, digest in evidence.items():
            path = source_path(name, digest)
            key = str(path)
            need(key not in evidence_pins or evidence_pins[key] == digest,
                 'Shared source/context evidence hash conflict')
            evidence_pins[key] = digest
            need(binding['input_sha256'].get(key) == digest
                 and plan['input_sha256'].get(key) == digest,
                 'Evidence pin missing from source/PLAN binding')
            add_pin(inputs, path, digest)

    prep = read(O/'prepared/INDEPENDENT_PREPARE.json')
    audit = read(O/'prepared/SELECTION_AUDIT.json')
    baseline = read(O/'prepared/BASELINE_PARITY.json')
    source = read(O/'prepared/SOURCE_PARITY.json')
    outcome = read(O/'results/INDEPENDENT_OUTCOME.json')
    meta, orders = read(O/'prepared/META.json'), read(O/'prepared/ORDERS.json')
    need(meta == read(LOCAL/'META.json') and len(meta['dates']) == len(meta['ends']) == len(meta['symbols']) == 122,
         'Original 122-month identities')
    need(meta['dates'][0] == '2016-08-17' and meta['ends'][-1] == '2026-09-24'
         and meta['ends'][:-1] == meta['dates'][1:], 'Continuous saved period')
    need(set(orders) == set(ARMS), 'Three saved order streams')
    need(orders[ARMS[0]] == read(LOCAL/'local_linear_ORDERS.json'), 'Original Local67 orders changed')
    need(orders[ARMS[2]] == read(BASELINE), 'Baseline-score control differs from historical original65 orders')
    for key in ('passed', 'original_selector_orders_exact', 'historical_original65_orders_exact',
                'baseline_name_is_not_a_performance_claim'):
        need(baseline[key] is True, 'Baseline parity flag '+key)
    need(baseline['checked_months'] == 122 and baseline['historical_orders_sha256'] == sha(BASELINE),
         'Historical baseline parity lineage')
    for arm in ARMS:
        need(len(orders[arm]) == 122, 'Order coverage')
        for i, order in enumerate(orders[arm]):
            need(set(order) == {'date', 'end', 'signal_date', 'symbols', 'scores'}
                 and order['date'] == order['signal_date'] == meta['dates'][i]
                 and order['end'] == meta['ends'][i], 'Order schema or interval')
            names = order['symbols']
            need(isinstance(names, list) and len(names) in (0, 2, 3) and len(set(names)) == len(names)
                 and set(names) <= set(meta['symbols'][i]) and len(order['scores']) == len(names)
                 and all(type(x) in (int, float) and math.isfinite(x) for x in order['scores']),
                 'Invalid saved holdings or scores')

    need(audit['method'] == 'eligible_core_top6_then_baseline_top3', 'Saved selection method')
    records = audit['records']
    need(len(records) == 122 and [r['row'] for r in records] == list(range(122)), 'Audit monthly axis')
    for i, row in enumerate(records):
        k, q = row['eligible_count'], row['shortlist_count']
        need(type(k) is int and type(q) is int and 0 <= q <= k <= 49, 'Saved candidate counts')
        need(row['date'] == meta['dates'][i] and row['leader6_symbols'] == orders[ARMS[1]][i]['symbols']
             and row['baseline_symbols'] == orders[ARMS[2]][i]['symbols'], 'Audit/order monthly identity')
        need(row['changed_holdings'] == (set(row['leader6_symbols']) != set(row['baseline_symbols']))
             and row['changed_order'] == (row['leader6_symbols'] != row['baseline_symbols']), 'Change flags')
        need(row['tie_added_count'] == max(q-6, 0), 'Tie expansion arithmetic')
        need((k <= 6 and q == k and row['sixth_core_threshold'] is None and row['sixth_tie_count'] == 0)
             or (k > 6 and q >= 6 and math.isfinite(row['sixth_core_threshold'])
                 and 1 <= row['sixth_tie_count'] <= k), 'Saved cutoff evidence')
    counts = record_counts(records)
    need(counts == audit['counts'] == prep['selection_counts'] == prepared['selection_counts']
         == closure['selection_counts'] == prep['selection']['counts'], 'Selection-count lineage')
    need(prep['selection']['passed'] is True and prep['selection']['producer_modules_imported'] is False
         and prep['selection']['computes_returns'] is False, 'Independent scalar selection scope')
    for key in ZERO_FIELDS:
        need(type(audit[key]) is int and audit[key] == 0 and prep['selection'][key] == 0,
             'No new fitting or learned parameters: '+key)
    need(prepared['new_model_fits'] == prepared['new_parameters'] == 0
         and prepared['new_returns_computed'] is False, 'No preparation fits/returns')
    need(prep['original_array_count'] == prepared['original_array_count'] == closure['original_array_count'] == 27,
         'Original cache count')
    for key in ('original_orders_exact', 'baseline_control_orders_exact', 'historical_baseline_orders_exact',
                'original_arrays_unchanged', 'original_incumbent_policy_and_episode_preserved',
                'paired_initial_partial_uses_formula', 'physical_arrays_truncated', 'physical_metadata_truncated'):
        need(prep[key] is True, 'Preparation scope '+key)
    for key in ('paired_arms_use_ml', 'paired_arms_use_episode', 'training_labels_consumed'):
        need(prep[key] is False, 'New-arm scope '+key)
    need(prep['calendar_check']['passed'] is True and prep['calendar_check']['formal_events'] == 121,
         'Formal calendar evidence')
    need(prep['prefixes'] == 123 and prep['physical_prefixes'] == list(range(123)), 'All 123 prefix checks')
    proofs = prep['prefix_evidence']
    need(len(proofs) == 123 and [x['prefix'] for x in proofs] == list(range(123)), 'Prefix proof axis')
    for n, proof in enumerate(proofs):
        need(proof['query_rows'] == 49*n and proof['query_date'] == (meta['dates'][n-1] if n else None),
             'Prefix query identity')
        need(all(proof[k] is True for k in ('independent_passed', 'arrays_physically_truncated',
             'metadata_physically_truncated', 'baseline_parity')), 'Prefix proof flags')
        need(proof['counts'] == record_counts(records[:n]), 'Saved prefix counts independently recounted')
    need(source['original_identity_exact'] is True and source['original_cache_bytes_bound'] is True
         and source['original_core_neutral_semantics_preserved'] is True, 'Original-cache provenance')
    need(source['raw_source_hashes'] == plan['raw_source_hashes']
         and source['cache_source_binding'] == plan['cache_source_binding'], 'Source lineage changed')
    for key in ('training_labels_consumed', 'source_forecasts_consumed_by_new_arms', 'new_raw_features',
                'strict_pit', 'fresh_raw_training_reconstruction'):
        need(source[key] is False, 'Source limitation '+key)

    prior_receipt = P/'results/COMPLETION.json'
    prior_results = P/'results/RESULTS.json'
    need(plan['predecessor_binding']['input_sha256'][str(prior_receipt)] == sha(prior_receipt),
         'Registered predecessor result manifest')
    prior_done = read(prior_receipt)
    need(prior_done['status'] == 'COMPLETE' and prior_done['paths'] == 9 and prior_done['slices'] == 45,
         'Predecessor independently completed')
    add_pin(inputs, prior_results, prior_done['artifacts_sha256']['RESULTS.json'])
    add_pin(inputs, prior_receipt)
    prior = read(prior_results)
    frozen, result = read(O/'selected/FROZEN.json'), read(O/'results/RESULTS.json')
    training = read(O/'selected/TRAIN_RESULTS.json')
    need(frozen['plan_sha256'] == plan_sha and frozen['prepared_sha256'] == sha(O/'prepared/COMPLETION.json')
         and frozen['orders_sha256'] == sha(O/'prepared/ORDERS.json')
         and frozen['training_sha256'] == sha(O/'selected/TRAIN_RESULTS.json'), 'Frozen first88 dependencies')
    need(closed['frozen_sha256'] == sha(O/'selected/FROZEN.json')
         and frozen['selection'] == result['selection'], 'Frozen-to-results selection chain')
    need(result['selection']['first88_only'] is True and result['selection']['later34_used'] is False,
         'Selection temporal scope')
    need(result['strict_pit'] is False and result['reused_sample'] is True
         and result['observer_promotion'] is False and result['paired_arms_use_ml'] is False
         and result['paired_arms_use_episode'] is False, 'Result limitations')
    need(set(result['arms']) == set(training) == set(ARMS), 'Three metric arms')
    need(outcome['passed'] is True and outcome['checked_paths'] == 9 and outcome['checked_slices'] == 45
         and outcome['independent_accounts'] is True and outcome['independent_of_production_model_and_engine'] is True
         and outcome['terminal_liquidation'] is False, 'Independent account evidence')
    need(outcome['accounting_source_sha256'] == sha(R/'work/ptr169_results_independent.py'), 'Accounting source pin')
    need(set(outcome['paths']) == set(ARMS), 'Independent arm coverage')
    metrics, fees = {}, {}
    for arm in ARMS:
        need(set(result['arms'][arm]) == set(outcome['paths'][arm]) == set(PATHS), 'Three execution paths')
        need(set(training[arm]) == {'same_close_zero', 'next_open_cost'}, 'Only two registered training paths')
        metrics[arm], fees[arm] = {}, {}
        for path, (execution, cost) in PATHS.items():
            data = result['arms'][arm][path]
            metric, proof = data['metrics'], outcome['paths'][arm][path]
            need(metric['execution'] == execution and metric['cost_per_side'] == cost, 'Execution and fee label')
            need(metric['start'] == '2016-08-17' and metric['end'] == '2026-09-24', 'Full period identity')
            need(proof['passed'] is True and proof['monthly_executions'] == 122
                 and set(proof['slices']) == set(data['slices']) == set(SLICES), 'Five independent slices')
            need(data['accounting']['terminal_liquidation'] is False
                 and data['accounting']['exported_fill_ledger_matches'] is True, 'No terminal liquidation / fills')
            metrics[arm][path] = {}
            for label, (lo, hi) in SLICES.items():
                observed, independent = data['slices'][label], proof['slices'][label]
                need(observed['start'] == meta['dates'][lo] and observed['end'] == meta['ends'][hi-1],
                     'Saved slice dates')
                values = {}
                for key in ('cagr', 'sharpe', 'max_drawdown'):
                    value = observed[key]
                    need(type(value) in (int, float) and math.isfinite(value), 'Finite saved metric')
                    need(saved_metric_near(value, independent[key]), 'Saved metric disagrees with independent receipt')
                    values[key] = value
                metrics[arm][path][label] = dict(values, start=observed['start'], end=observed['end'])
            if path in training[arm]:
                need(set(training[arm][path]) == {'training', 'early', 'recent'}, 'Training subperiods')
                for label, slice_name in (('training', 'first88'), ('early', 'early53'), ('recent', 'recent35')):
                    for key in ('cagr', 'sharpe', 'max_drawdown'):
                        need(saved_metric_near(training[arm][path][label][key], data['slices'][slice_name][key]),
                             'Frozen first88 metric identity')
            if arm == ARMS[0]:
                for key in ('cagr', 'sharpe', 'max_drawdown', 'final_equity', 'total_return',
                            'trading_sessions', 'fees_paid', 'dollar_turnover', 'trade_rows'):
                    need(metric[key] == prior['arms'][arm][path]['metrics'][key], 'Incumbent exact metric parity: '+key)
            trades = read(O/'results'/f'{arm}_{path}_TRADES.json')
            holdings = read(O/'results'/f'{arm}_{path}_HOLDINGS.json')
            need(len(holdings) == 122, 'Monthly holdings coverage')
            if cost == 0.:
                need(metric['fees_paid'] == 0., 'Zero-cost total fees')
                need(all(row['cost'] == 0. for row in trades), 'Zero-cost trade charge')
                need(all(row['fee'] == 0. for row in holdings), 'Zero-cost monthly charge')
            fees[arm][path] = dict(cost_per_side=cost, total_fees=metric['fees_paid'],
                                  trade_rows=len(trades), monthly_records=len(holdings),
                                  all_trade_and_month_fees_zero=cost == 0.)

    changes, monthly_changes = {}, []
    pairs = ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[1], ARMS[2]))
    for left, right in pairs:
        key = left+'_vs_'+right
        indices = [i for i in range(122) if set(orders[left][i]['symbols']) != set(orders[right][i]['symbols'])]
        changes[key] = dict(indices=indices, dates=[meta['dates'][i] for i in indices],
            counts={label: sum(lo <= i < hi for i in indices) for label, (lo, hi) in SLICES.items()})
    for i, day in enumerate(meta['dates']):
        monthly_changes.append(dict(row=i, date=day,
            symbols={arm: orders[arm][i]['symbols'] for arm in ARMS},
            changed={left+'_vs_'+right: i in changes[left+'_vs_'+right]['indices'] for left, right in pairs}))
    need(selection_recount(training, changes) == frozen['selection'], 'First88 gate recounted from saved data')
    need(changes['leader6_gate_vs_baseline_score']['counts']['full122'] == counts['changed_holding_months'],
         'Portfolio-change count identity')
    state = read(CENTRAL)
    need(len(state['completed']) == 216 and sum(x['trials'] for x in state['completed']) == 1165
         and not state['registered_queue'], 'Closed central counters')
    need(state['completed'][-1]['directory'] == O.name
         and state['completed'][-1]['completion']['sha256'] == expected, 'Central closure identity')
    need(closure['best_training_selected_candidate'] == state['best_training_selected_candidate'], 'Best-candidate lineage')
    need(isinstance(plan['contract']['limitation'], list) and plan['contract']['limitation'], 'Declared source limitations')
    for path in (O/'COMPLETION.json', O/'prepared/COMPLETION.json', O/'results/COMPLETION.json',
                 O/'selected/FROZEN.json', O/'selected/TRAIN_RESULTS.json', Path(__file__), P/'RESEARCH_REVIEW.json'):
        add_pin(inputs, path)
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        scope='Saved JSON/hash/fee-label review only; saved-order/audit counters recounted. No NPZ decoded, no new feature, fit, price return, NAV, or security selection computed.',
        input_sha256=inputs, completion_sha256=expected, plan_sha256=plan_sha,
        plan_input_hashes_reverified=len(plan['input_sha256']), formal_synthetic_tests=78,
        source_evidence_lists_independently_verified=True,
        source_evidence_counts=dict(review=len(source_review['evidence_sha256']),
                                   context=len(context['evidence_sha256']), union=len(evidence_pins)),
        selection=frozen['selection'], selection_recounted=True, metrics=metrics,
        changed_holdings=changes, monthly_holdings_comparison=monthly_changes,
        selection_counts=counts, prefix_count_rechecks=123, physical_prefix_evidence_verified=True,
        physical_prefixes_reexecuted_in_this_review=False, shortlist_npz_decoded=False,
        independent_execution_paths=9, independent_metric_slices=45,
        saved_independent_metric_tolerance=dict(absolute=1e-7, relative=2e-11),
        source_model_refits=0, new_parameters=0,
        source_training_rebuilt=False, strict_pit=False, zero_cost_trade_and_month_fees_checked=True,
        fee_labels=fees, incumbent_metrics_match_round217_exactly=True,
        historical_original65_orders_exact=True, original_arrays_preserved_by_preparer=27,
        protected_files_verified_by_closer=21444, full_protected_archive_reread_in_this_review=False,
        completed_entries=216, trials=1165, central_sha256_at_review=sha(CENTRAL),
        best=state['best_training_selected_candidate'], limitations=plan['contract']['limitation'], observers_modified=False)
    lines = ['# 第218轮：长期动量前六名内再选三只', '',
        '候选通过前88期筛选。' if report['selection']['admitted'] else '候选未通过前88期筛选，保留原Local67。', '',
        '在原合格名单内，按长期动量核心分先取前六名，第六名精确并列者全部保留；再按原基础评分选三只。固定对照直接按原基础评分选三只。两只时各半，少于两只空仓；初始部分月份也按同一公式。两个新分支都不使用累积的机器学习或episode修正，原Local67基准保持完整规则。月末调仓、月内固定持仓。', '']
    labels = {'full122': '全期122期', 'first88': '筛选前88期', 'early53': '前53期',
              'recent35': '随后35期', 'later34': '后34期'}
    for path, title in (('same_close_zero', '同收盘零成本'), ('next_open_zero', '次开盘零成本'),
                        ('next_open_cost', '次开盘每边0.05%成本')):
        lines += ['## '+title, '', '|区间|原Local67：年化 / Sharpe / 最大回撤|前六再选三|原基础评分对照|',
                  '|---|---:|---:|---:|']
        for label in SLICES:
            cells = []
            for arm in ARMS:
                metric = metrics[arm][path][label]
                cells.append(f"{metric['cagr']:.2%} / {metric['sharpe']:.3f} / {metric['max_drawdown']:.2%}")
            lines.append('|'+labels[label]+'|'+'|'.join(cells)+'|')
        lines.append('')
    failed = [key for key, value in report['selection']['guards'].items() if not value]
    lines += ['未通过门槛：'+('、'.join(failed) if failed else '无')+'。', '',
        '门槛只使用前88期：同收盘年化至少高于原Local67 0.5个百分点，次开盘扣费年化不低于原Local67，且两条路径均不劣于基础评分对照；同时核对Sharpe、回撤、前53期与随后35期及两个阶段的持仓变化。', '',
        f"全期与原Local67不同持仓：前六再选三 {changes['leader6_gate_vs_incumbent_local67']['counts']['full122']} 期、基础评分对照 {changes['baseline_score_vs_incumbent_local67']['counts']['full122']} 期；前六再选三与基础评分对照不同 {counts['changed_holding_months']} 期。第六名并列扩容 {counts['tie_expansion_months']} 期，新增训练和拟合参数均为0。", '',
        '78项合成测试、123个物理前缀证明、9条执行路径及45个区间核验的哈希链完整；两条零成本路径的逐笔费用和逐月费用均为0。原Local67三条路径的关键指标与第217轮精确一致；基础评分对照的订单与历史original65订单精确一致。', '',
        '本文件只读取已保存结果并复核哈希、费用与计数，不重新计算收益。保护21444件档案由正式关闭流程验证，本次不重新遍历完整保护档案。该架构属于此前动量硬门槛家族的固定前六名变体，不宣称是全新因子。', '',
        '后34期与全期仍是重复使用的研究样本，不是独立前向收益。历史股票池和行情版本仍有代理限制；原核心缺失值中性处理保持不变，完整哈希和因果时钟不等于严格历史时点数据认证。未重建原始训练，也未修改观察账户。']
    json_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    markdown = '\n'.join(lines)+'\n'
    need(not any(path.exists() for path in targets), 'Concurrent saved-review write')
    with targets[0].open('x') as stream:
        stream.write(json_text)
    with targets[1].open('x') as stream:
        stream.write(markdown)
    print(json.dumps(dict(passed=True, review_sha256=sha(targets[0]), report_sha256=sha(targets[1]),
        selection=report['selection'], same_close={arm: metrics[arm]['same_close_zero']['full122'] for arm in ARMS}),
        ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--completion-sha256', required=True)
    main(parser.parse_args().completion_sha256)

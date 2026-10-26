"""Saved-JSON and byte-integrity review for CLOSED round210 only.

The reader author also wrote the production kernel. This is not a second
mathematical oracle: exact transformations are certified by the separately
authored, hash-bound independent receipt. No NPZ is loaded, no strategy is
imported, and no support/rank/delta/order/return is rebuilt. Saved metrics and
order lists are summarized; all output requires an explicit root closure SHA.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/monthly_asset_aux210'
PRIOR = ROOT/'outputs/monthly_relative_path209'
CENTRAL = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
LOCAL = ROOT/'outputs/monthly_local_linear70'
ARMS = ('incumbent_local67', 'asset_aux_rank', 'asset_aux_affine_control')
PRIOR_ARMS = ('incumbent_local67', 'relative_path_shortfall_ridge', 'relative_terminal_shortfall_ridge')
PATHS = ('same_close_zero', 'next_open_zero', 'next_open_cost')
BOUNDS = {'full122':(0,122), 'first88':(0,88), 'early53':(0,53), 'recent35':(53,88), 'later34':(88,122)}
PREPARED_FILES = {'FEATURES.npz', 'TRANSFORMS_DELTAS.npz', 'FEATURE_AUDIT.json', 'ORDERS.json',
                  'META.json', 'SOURCE_PARITY.json', 'INDEPENDENT_PREPARE.json'}
RESULT_FILES = {'RESULTS.json', 'INDEPENDENT_OUTCOME.json'} | {
    a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS
    for k in ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')}
PRIOR_RESULT_FILES = {'RESULTS.json', 'INDEPENDENT_OUTCOME.json'} | {
    a+'_'+p+'_'+k+'.json' for a in PRIOR_ARMS for p in PATHS
    for k in ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')}
CONTEXT = 'work/monthly_asset_aux210_context.json'
CONTEXT_SHA = '650b34956368e6df3ce1e542975dc7388048d6df044d6eb2a59f5b22ec865832'
COVERAGE = 'work/after209_asset_aux_coverage.json'
COVERAGE_SHA = '88d0d883d83dde4fc6ca54b855a33455906626f9c2cc252b5f1dd50f9b7e754a'
TESTS = 'work/monthly_asset_aux210_TESTS.json'
CODE_REVIEW = 'work/asset_aux210_CODE_REVIEW.json'
HELPER = 'work/aux_rank204_close_review_v2.py'
HELPER_SHA = '36acc8b196e29d19266d9a4e95fa65e9c652a416293231018331230c58512c70'
CODE_FILES = {f'work/{prefix}monthly_asset_aux210_{kind}.py'
              for prefix in ('', 'test_') for kind in ('host', 'kernel', 'independent')}
TEST_FILES = sorted(x for x in CODE_FILES if Path(x).name.startswith('test_'))
COUNTS = ('rank_rows', 'constant_rows', 'two_level_rows', 'active_forecast_cells',
          'grouped_rank_rows', 'stock_cells', 'etf_cells', 'empty_groups')
STAT_KEYS = {'start', 'end', 'years', 'final_equity', 'cagr', 'sharpe',
             'max_drawdown', 'trading_sessions', 'total_return'}
EXECUTIONS = {'same_close_zero':('close',0.), 'next_open_zero':('open',0.),
              'next_open_cost':('open',.0005)}
FEATURE_FLAGS = ('passed', 'independent_scalar_rebuild', 'full_auxiliary_support_checked',
    'core_weights_checked', 'full_candidate_universe', 'average_tie_ranks_checked',
    'affine_span_control_checked', 'two_level_equivalence_checked', 'constant_rows_unchanged',
    'first_partial_unchanged', 'effective_support_weight_checked', 'no_eligibility_rank_filter',
    'no_new_availability_filter', 'no_future_or_outcome_inputs', 'audit_checked',
    'asset_type_identity_checked', 'within_type_rank_domains_checked', 'empty_groups_checked',
    'constant_groups_unchanged')
DELTA_FLAGS = ('passed', 'independent_delta_rebuild', 'effective_support_weight_checked',
    'full_auxiliary_replacement_checked', 'first_partial_unchanged', 'constant_rows_unchanged',
    'core_weights_checked', 'no_eligibility_delta_filter', 'source_values_checked',
    'asset_type_identity_checked', 'group_degenerate_noop_checked', 'constant_groups_unchanged')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value, label):
    require(type(value) is int and value >= 0, label)
    return value


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label)
    return value


def helpers():
    path = ROOT/HELPER
    require(sha(path) == HELPER_SHA, 'Frozen saved-evidence helper changed')
    spec = importlib.util.spec_from_file_location('asset_aux210_saved_helper', path)
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    h.ROOT, h.OUT, h.PRIOR, h.CENTRAL, h.ARMS = ROOT, OUT, PRIOR, CENTRAL, ARMS
    return h


def declared_tests(source):
    """Count the exact simple unittest methods without importing/executing tests."""
    tree = ast.parse(source)
    count = 0
    require(not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name=='load_tests'
                    for n in ast.walk(tree)), 'Dynamic test loaders are outside this certificate')
    for node in tree.body:
        if not isinstance(node, ast.ClassDef): continue
        methods = [x for x in node.body if isinstance(x, ast.FunctionDef) and x.name.startswith('test')]
        if not methods: continue
        direct = any((isinstance(b, ast.Attribute) and isinstance(b.value, ast.Name)
                      and b.value.id=='unittest' and b.attr=='TestCase')
                     or (isinstance(b, ast.Name) and b.id=='TestCase') for b in node.bases)
        require(direct and len(node.bases)==1, 'Unexpected inherited/dynamic unittest cases')
        require(len({x.name for x in methods})==len(methods), 'Duplicate test methods')
        decorators = node.decorator_list + [d for x in methods for d in x.decorator_list]
        require(not any('skip' in ast.unparse(d).lower() for d in decorators), 'Skipped tests not accepted')
        count += len(methods)
    require(count > 0, 'No declared synthetic tests')
    return count


def test_code_binding(e, h, plan, tests, code):
    tests_sha = plan['input_sha256'][str(ROOT/TESTS)]
    require(code['formal_test_receipt_sha256']==tests_sha, 'Code review binds a different test receipt')
    require(tests['passed'] is True and tests['synthetic_only'] is True and
            code['passed'] is True and code['unresolved_blockers']==[], 'Formal evidence not passed')
    require(code['context_sha256']==CONTEXT_SHA and code['input_sha256'].get(COVERAGE)==COVERAGE_SHA
            and plan['input_sha256'].get(str(ROOT/COVERAGE))==COVERAGE_SHA,
            'Reviewed context/coverage differs')
    require(tests['runtime']==plan['runtime']==dict(python='3.9.6', numpy='2.0.2', pandas='2.3.3'),
            'Pinned runtime differs')
    require(set(code['code_sha256'])==CODE_FILES, 'Exactly six reviewed numerical files required')
    for name, pin in code['code_sha256'].items():
        require(tests['code_sha256'].get(str(ROOT/name))==plan['input_sha256'].get(str(ROOT/name))==pin,
                'Tested/reviewed/registered code hashes differ')
        e.verify(name, pin)
    for name, pin in tests['code_sha256'].items():
        path = h.input_path(name, pin)
        require(path!=CENTRAL and plan['input_sha256'].get(str(path))==pin, 'Unbound tested dependency')
        e.verify(name, pin)
    methods = {name:declared_tests((ROOT/name).read_text()) for name in TEST_FILES}
    recorded = {str(h.local_path(name).relative_to(ROOT)):n for name,n in code['synthetic_test_methods'].items()}
    require(recorded==methods, 'Test file/method certificate differs')
    total = sum(methods.values())
    require(integer(tests['tests'], 'Formal test count')==integer(code['synthetic_tests'], 'Reviewed test count')==total,
            'Formal tests must equal the bound three-file method count')
    for key in ('actual_features_computed', 'actual_model_fits_computed', 'actual_portfolio_returns_computed',
                'production_files_modified', 'strategy_registered_by_reviewer'):
        require(code[key] is False, 'Code review crossed its static boundary')
    for name, pin in code['input_sha256'].items():
        require(h.input_path(name, pin)!=CENTRAL, 'Mutable central cannot be a review input')
        require(plan['input_sha256'].get(str(h.input_path(name, pin)))==pin, 'Unregistered review dependency')
        e.verify(name, pin)
    return dict(tests=total, test_methods=methods, six_numerical_files=code['code_sha256'])


def inspect_groups(audit, meta):
    """Validate recorded JSON geometry/counts; never recalculate support or ranks."""
    n = integer(audit['events'], 'Event count'); width = integer(audit['symbols_per_event'], 'Width')
    require(width>0 and audit['checked_cells']==n*width and len(audit['event_history'])==n,
            'Group audit axes')
    names, etfs = meta['symbols'], meta['etfs']
    require(isinstance(names, list) and len(names)==n and isinstance(etfs, list)
            and all(isinstance(x, str) and x for x in etfs) and len(set(etfs))==len(etfs), 'Explicit ETF identity')
    counts = {k:0 for k in COUNTS}
    maxima = {a:0. for a in ARMS[1:]}
    for i, row in enumerate(audit['event_history']):
        symbols = names[i]
        require(isinstance(symbols, list) and len(symbols)==width and len(set(symbols))==width
                and all(isinstance(x,str) and x for x in symbols), 'Unique monthly identities')
        require(row['query_index']==i and type(row['query_index']) is int and row['symbols']==symbols
                and len(row['groups'])==2, 'Group event identity')
        active = 0; event_max = dict(candidate=0., control=0., difference=0.)
        for kind, g in zip(('stock','etf'), row['groups']):
            ids = [j for j,s in enumerate(symbols) if (s in etfs)==(kind=='etf')]
            size = len(ids)
            require(g['asset_type']==kind and g['indices']==ids and g['symbols']==[symbols[j] for j in ids]
                    and integer(g['size'],'Group size')==size, 'Recorded group membership')
            require(all(type(j) is int for j in g['indices']), 'Integer membership indices')
            counts['stock_cells' if kind=='stock' else 'etf_cells'] += size
            levels = integer(g['distinct_levels'], 'Distinct group levels')
            require(levels<=size and type(g['constant']) is bool and type(g['two_levels']) is bool
                    and g['constant']==(levels==1) and g['two_levels']==(levels==2), 'Group degeneracy flags')
            local_active = integer(g['active_forecast_count'], 'Active funding count')
            require(local_active<=size, 'Group funding count'); active += local_active
            if not size:
                require(levels==0 and all(g[k] is None for k in ('support_min','support_max','candidate_min',
                        'candidate_max','control_min','control_max')), 'Empty group descriptors')
                counts['empty_groups'] += 1
            else:
                require(1<=levels and (size!=1 or levels==1), 'Nonempty group levels')
                counts['rank_rows'] += 1; counts['grouped_rank_rows'] += 1
                counts['constant_rows'] += int(g['constant']); counts['two_level_rows'] += int(g['two_levels'])
                for field in ('support','candidate','control'):
                    lo, hi = finite(g[field+'_min'],'Finite extrema'), finite(g[field+'_max'],'Finite extrema')
                    bound = .5+1e-12 if field=='support' else .5
                    require(-bound<=lo<=hi<=bound, 'Saved group bounds')
                require((g['support_min']==g['support_max'])==g['constant'], 'Constant support extrema')
                if g['constant']:
                    require(all(g[k]==0. for k in ('candidate_min','candidate_max','control_min','control_max')),
                            'Constant group target descriptors')
                else:
                    require(g['candidate_min']==g['control_min'] and g['candidate_max']==g['control_max']
                            and g['candidate_min']<g['candidate_max'], 'Matched affine endpoints')
            difference = finite(g['max_absolute_candidate_control_difference'],'Group difference')
            require(0.<=difference<=1. and (not g['two_levels'] or difference==0.), 'Two-level identity descriptor')
            if size<=1 or g['constant']: require(difference==0., 'Degenerate difference')
            event_max['difference'] = max(event_max['difference'], difference)
            for short, arm in zip(('candidate','control'), ARMS[1:]):
                maximum = finite(g['max_absolute_'+short+'_delta'],'Group delta maximum')
                require(0.<=maximum<=.5, 'Delta bound')
                if not i or size<=1 or g['constant']: require(maximum==0., 'Initial/degenerate group changed')
                event_max[short] = max(event_max[short], maximum); maxima[arm] = max(maxima[arm], maximum)
            if g['two_levels']:
                require(g['max_absolute_candidate_delta']==g['max_absolute_control_delta'], 'Two-level delta descriptors')
        require(integer(row['active_forecast_count'],'Row active count')==active, 'Row active aggregation')
        counts['active_forecast_cells'] += active
        for short in ('candidate','control'):
            require(row['max_absolute_'+short+'_delta']==event_max[short], 'Row delta aggregation')
        require(row['max_absolute_candidate_control_difference']==event_max['difference'], 'Row difference aggregation')
    require(all(integer(audit[k], 'Aggregate '+k)==v for k,v in counts.items()), 'Aggregate group counts')
    for key in ('new_model_fits', 'predictive_model_fits'): require(integer(audit[key],key)==0, 'Unexpected fits')
    for key in ('ranking_eligibility_filtered','new_availability_filter','portfolio_returns_calculated'):
        require(audit[key] is False, 'Wrong grouping scope')
    return counts, maxima


def integration_receipts(validation, prepared, closure, counts, maxima):
    for key in FEATURE_FLAGS: require(validation['feature'].get(key) is True, 'Feature proof '+key)
    for key in DELTA_FLAGS: require(validation['delta'].get(key) is True, 'Delta proof '+key)
    f, d = validation['feature'], validation['delta']
    require(f['events']==122 and f['symbols_per_event']==49 and f['checked_cells']==5978
            and f['computes_returns'] is False and f['numerical_tolerance']==0.
            and f['support_bound_tolerance']==1e-12, 'Exact scalar oracle coverage/tolerances')
    require(d['max_absolute_delta']==.5 and d['per_arm_max_absolute_delta']==maxima
            and d['observed_max_absolute_delta']==max(maxima.values()), 'Independent delta maxima')
    require(set(d['changed_cells'])==set(ARMS[1:]), 'Independent changed-cell arms')
    for value in d['changed_cells'].values(): require(integer(value,'Changed cells')<=5978, 'Changed-cell bound')
    require(validation['prefixes']==123 and validation['feature_prefixes']==list(range(123)), 'All123 prefixes')
    for key in ('feature_prefix_sources_truncated','feature_prefix_metadata_truncated','asset_type_metadata_checked',
                'original_cache_exact','original_orders_exact','original_arrays_unchanged',
                'original_core_metadata_and_support_preserved','single_original_episode'):
        require(validation[key] is True, 'Original invariant/prefix proof '+key)
    for record in (f, validation, prepared, closure):
        require(all(integer(record[k],k)==v for k,v in counts.items()), 'Cross-receipt group counts')
        for key in ('new_model_fits','predictive_model_fits'): require(integer(record[key],key)==0, 'Nonzero new fits')
    for record in (validation, prepared, closure):
        require(record['original_array_count']==27 and record['new_predictive_model_fits']==0, 'Original27/zero-fit proof')
    require(validation['model_fit_accounting']==prepared['model_fit_accounting']==closure['model_fit_accounting'], 'Fit accounting wording')
    require(prepared['new_returns_computed'] is False, 'Preparation evaluated portfolio returns')
    return dict(changed_cells=d['changed_cells'],per_arm_max_absolute_delta=maxima,
        grouped_json_descriptors_checked=True, saved_npz_hash_verified_only=True,
        original_arrays_not_loaded=True, numerical_transform_not_recomputed=True,
        formula_evidence='Separately authored exact scalar oracle; this reader only reconciles recorded JSON and hashes.')


def zero_fee_evidence(result, trades, holdings):
    # Non-full slices are engine.stats dictionaries and do not contain fees.
    require(finite(result['metrics']['fees_paid'],'Full fees')==0.
            and finite(result['slices']['full122']['fees_paid'],'Full-slice fees')==0.,
            'Zero-cost full metrics must have zero fees')
    require(all(finite(x['fees_paid'],'Optional slice fees')==0.
                for x in result['slices'].values() if 'fees_paid' in x), 'Nonzero declared slice fees')
    require(all(type(x['cost']) in (int,float) and x['cost']==0. for x in trades), 'Zero-cost fill fees')
    require(all(type(x['fee']) in (int,float) and x['fee']==0. for x in holdings), 'Zero-cost holding fees')
    return dict(fees_paid=0., fill_rows=len(trades), holding_periods=len(holdings), all_saved_fees_exact_zero=True)


def complete_metric_evidence(h, independent, saved, label):
    require(isinstance(independent,dict) and set(independent)==STAT_KEYS,
            label+'/independent statistics must contain exactly nine fields')
    h.metric_evidence(independent,saved,label)


def execution_evidence(metrics, path):
    execution,cost=EXECUTIONS[path]
    require(metrics['execution']==execution and finite(metrics['cost_per_side'],'Execution cost')==cost,
            'Saved execution/cost differs from registered path '+path)


def review(completion_sha, check_only=False):
    require(isinstance(completion_sha,str) and len(completion_sha)==64
            and all(c in '0123456789abcdef' for c in completion_sha), 'Explicit closed completion SHA required')
    require(not (OUT/'RESEARCH_REVIEW.json').exists() and not (OUT/'RESEARCH_REVIEW.md').exists(), 'Write-once report')
    h=helpers(); e=h.Evidence()
    closure=e.load(OUT/'COMPLETION.json',completion_sha)
    require(closure['status']=='COMPLETED_INDEPENDENTLY_VERIFIED','Close must precede review')
    plan_sha=closure['plan_sha256']; plan=e.load(OUT/'PLAN.json',plan_sha); context=e.load(CONTEXT,CONTEXT_SHA)
    require(plan['schema']=='monthly_asset_aux210_v1' and plan['contract']['arms']==list(ARMS)
        and plan['contract']['candidate']==ARMS[1] and plan['contract']['control_nonpromotable']==ARMS[2]
        and plan['contract']['context']==context and plan['contract']['prospective_context_sha256']==CONTEXT_SHA
        and plan['contract']['coverage_sha256']==COVERAGE_SHA and plan['contract']['max_absolute_delta']==.5,
        'Fixed210 contract')
    require(len(plan['raw_source_hashes'])==64, 'Original raw64 closure')
    for mapping in (plan['input_sha256'],plan['raw_source_hashes'],plan['calendar_source_hashes']):
        for name,pin in mapping.items():
            path=h.input_path(name,pin)
            require(path!=CENTRAL and plan['input_sha256'].get(str(path))==pin,'Immutable complete PLAN inputs')
            e.verify(name,pin)
    e.load(COVERAGE,COVERAGE_SHA)
    for key in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration',
        'actual_cached_transforms_before_registration','actual_ranks_before_registration','actual_new_returns_before_registration'):
        require(type(plan[key]) is int and plan[key]==0,'Prospective registration boundary')
    before=e.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    require(len(before['completed'])==207 and sum(x['trials'] for x in before['completed'])==1138,'Predecessor counters')
    tests=e.load(TESTS,plan['input_sha256'][str(ROOT/TESTS)])
    review_binding=plan['numerical_code_review']
    require(h.local_path(review_binding['path'])==ROOT/CODE_REVIEW and
        review_binding['sha256']==plan['input_sha256'][str(ROOT/CODE_REVIEW)],'Registered code review')
    code=e.load(CODE_REVIEW,review_binding['sha256']); test_evidence=test_code_binding(e,h,plan,tests,code)
    rc=e.manifest(OUT/'results',closure['result_completion_sha256'],plan_sha,RESULT_FILES)
    require(rc['paths']==9 and rc['slices']==45,'Nine paths45 slices')
    frozen=e.load(OUT/'selected/FROZEN.json',rc['frozen_sha256'])
    prepared=e.manifest(OUT/'prepared',frozen['prepared_sha256'],plan_sha,PREPARED_FILES)
    require(frozen['plan_sha256']==plan_sha and frozen['orders_sha256']==prepared['artifacts_sha256']['ORDERS.json'], 'Frozen selection inputs')
    training=e.load(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    saved={name:e.load(OUT/'prepared'/name) for name in PREPARED_FILES if name.endswith('.json')}
    meta,orders,validation=saved['META.json'],saved['ORDERS.json'],saved['INDEPENDENT_PREPARE.json']
    cache=h.cache_binding(e,plan,saved['SOURCE_PARITY.json'],meta)
    binding=plan['cache_source_binding']
    require(binding['explicit_asset_type_metadata_checked'] is True and binding['etf_ids']==meta['etfs']
        and len(meta['etfs'])==17 and binding['stock_cells']==3904 and binding['etf_cells']==2074,'Bound asset types')
    require(orders[ARMS[0]]==e.load(LOCAL/'local_linear_ORDERS.json',plan['input_sha256'][str(LOCAL/'local_linear_ORDERS.json')]), 'Original Local67 orders')
    counts,maxima=inspect_groups(saved['FEATURE_AUDIT.json'],meta)
    require(counts['rank_rows']==counts['grouped_rank_rows']==244 and counts['stock_cells']==3904
        and counts['etf_cells']==2074 and counts['empty_groups']==0,'Production244groups/32stock/17ETF')
    require(all([g['size'] for g in r['groups']]==[32,17] for r in saved['FEATURE_AUDIT.json']['event_history']), 'Production monthly membership sizes')
    integration=integration_receipts(validation,prepared,closure,counts,maxima)
    require(validation['model_fit_accounting']==plan['model_fit_accounting']==plan['contract']['model_fit_accounting'], 'Registered zero-fit accounting')
    results=e.load(OUT/'results/RESULTS.json'); outcome=e.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    for key in ('full_pre_episode_auxiliary_rank','asset_type_calibration','original_support_retained_for_episode'):
        require(results[key] is True,'Result interpretation '+key)
    require(results['observer_promotion'] is False and results['strict_pit'] is False,'Result limitations')
    require(set(orders)==set(training)==set(results['arms'])==set(outcome['paths'])==set(ARMS),'Three arms')
    selected=h.selection_from_saved(training,orders)
    require(results['selection']==frozen['selection']==selected,'Exact first88 selection')
    require(outcome['passed'] is True and outcome['checked_paths']==9 and outcome['checked_slices']==45
        and outcome['independent_accounts'] is True and outcome['independent_of_production_model_and_engine'] is True
        and outcome['terminal_liquidation'] is False,'Independent nine-path evidence')
    e.verify('work/ptr169_results_independent.py',outcome['accounting_source_sha256'])
    summaries={}; slices=groups=0; zero_fees={}
    for arm in ARMS:
        require(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates']
            and [o['end'] for o in orders[arm]]==meta['ends'],'Saved order calendar')
        require(set(results['arms'][arm])==set(outcome['paths'][arm])==set(PATHS),'Execution paths')
        summaries[arm]={}
        for path in PATHS:
            actual,ind=results['arms'][arm][path],outcome['paths'][arm][path]
            require(ind['passed'] is True and ind['monthly_executions']==122
                and set(actual['slices'])==set(ind['slices'])==set(BOUNDS)
                and actual['metrics']==actual['slices']['full122'],'Complete independent slices')
            execution_evidence(actual['metrics'],path)
            for part in BOUNDS: complete_metric_evidence(h,ind['slices'][part],actual['slices'][part],arm+'/'+path+'/'+part); slices+=1
            summaries[arm][path]={part:{k:actual['slices'][part][k] for k in h.METRICS} for part in BOUNDS}
            if path.endswith('_zero'):
                zero_fees[arm+'/'+path]=zero_fee_evidence(actual,e.load(OUT/'results'/(arm+'_'+path+'_TRADES.json')),
                    e.load(OUT/'results'/(arm+'_'+path+'_HOLDINGS.json')))
        require(set(training[arm])=={'same_close_zero','next_open_cost'},'Selection execution paths')
        for path in training[arm]:
            require(set(training[arm][path])=={'training','early','recent'},'Selection eras')
            for short,full in (('training','first88'),('early','early53'),('recent','recent35')):
                complete_metric_evidence(h,outcome['paths'][arm][path]['slices'][full],training[arm][path][short],arm+'/'+path+'/'+short); groups+=1
    require(slices==45 and groups==18,'45 slices18 training groups')
    predecessor=plan['predecessor_binding']; require(predecessor['directory']==PRIOR.name,'209 predecessor')
    old=e.load(PRIOR/'COMPLETION.json',predecessor['input_sha256'][str(PRIOR/'COMPLETION.json')])
    old_rc=e.load(PRIOR/'results/COMPLETION.json',old['result_completion_sha256'])
    require(old['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and old_rc['status']=='COMPLETE'
        and old_rc['plan_sha256']==old['plan_sha256'] and set(old_rc['artifacts_sha256'])==PRIOR_RESULT_FILES,'Prior exact38 closure')
    old_results=e.load(PRIOR/'results/RESULTS.json',old_rc['artifacts_sha256']['RESULTS.json'])
    incumbent=h.incumbent_comparison(e,old_rc,results,old_results)
    incumbent=json.loads(json.dumps(incumbent).replace('previous203','previous209').replace('current204','current210'))
    changed={name:h.changes(orders,a,b) for name,a,b in (('candidate_vs_incumbent',ARMS[0],ARMS[1]),
        ('control_vs_incumbent',ARMS[0],ARMS[2]),('candidate_vs_control',ARMS[2],ARMS[1]))}
    protected=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
    require(len(e.load(protected,plan['input_sha256'][str(protected)]))==closure['protected_files']==21444
        and closure['observers_modified'] is False and closure['new_trials']==3,'Protected closure')
    central=e.load(CENTRAL)
    require(central['current']==OUT.name and central['current_status']=='completed_independently_verified'
        and central['registered_queue']==[] and len(central['completed'])==208
        and sum(x['trials'] for x in central['completed'])==1141,'Immediate closed210 counters')
    last=central['completed'][-1]; best=closure['best_training_selected_candidate']; oldbest=before['best_training_selected_candidate']
    require(last['directory']==OUT.name and last['completion']['sha256']==completion_sha
        and h.local_path(last['completion']['path'])==OUT/'COMPLETION.json' and last['primary']==selected['primary']
        and last['new_admitted']==int(selected['admitted']) and last['trials']==3
        and central['best_training_selected_candidate']==best,'Central closure linkage')
    pick=selected['primary']; metrics=results['arms'][pick]['same_close_zero']['metrics']
    update=selected['admitted'] and metrics['cagr']>oldbest['cagr']
    if update:
        require(best['directory']==OUT.name and best['spec']==dict(arm=pick)
            and all(best[k]==metrics[k] for k in ('cagr','sharpe','max_drawdown'))
            and best['next_open_5bp_cagr']==results['arms'][pick]['next_open_cost']['metrics']['cagr']
            and best['selected_on_first88'] is True and best['observer_promotion'] is False,'Recorded best update')
    else: require(best==oldbest,'Unqualified historical-best change')
    latest=central['latest_training_selected_candidate']
    require(latest['directory']==OUT.name and latest['arm']==pick and latest['selection']==selected
        and latest['full_period_cagr']==metrics['cagr'] and latest['full_period_sharpe']==metrics['sharpe']
        and latest['selected_on_first88'] is True,'Central latest selection')
    require(closure['target_reached']==(best['cagr']>=.7),'Target flag')
    e.verify(HELPER,HELPER_SHA); e.verify(__file__); e.verify('work/test_asset_aux210_close_review.py')
    for name,pin in list(e.pins.items()): e.verify(name,pin)
    report=dict(schema='monthly_asset_aux210_research_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,result='ADMITTED' if selected['admitted'] else 'NOT_ADOPTED',selection=selected,
        failed_training_guards=[k for k,v in selected['guards'].items() if not v],summary=summaries,
        holding_changes=changed,group_counts=counts,cache_source_coverage=cache,integration=integration,
        zero_fee_evidence=zero_fees,incumbent209_export_diagnostic=incumbent,source_refs=e.pins,
        phase_sha256=dict(plan=plan_sha,prepared=frozen['prepared_sha256'],selected=rc['frozen_sha256'],
            results=closure['result_completion_sha256'],closure=completion_sha),formal_test_evidence=test_evidence,
        best_training_selected_candidate=best,historical_best_unchanged=best==oldbest,target_reached=closure['target_reached'],
        validation=dict(prepared_artifacts=7,result_artifacts=38,independent_cells=5978,grouped_rank_rows=244,
            stock_cells=3904,etf_cells=2074,independent_paths=9,independent_slices=45,training_metric_groups=18,
            order_prefixes=123,physical_cache_and_metadata_prefixes=list(range(123)),original_arrays=27,
            new_model_fits=0,protected_files=21444,completed_entries=208,trials=1141,
            protection_scope='Hash-bound original manifest and successful registration/closure receipts; no archive rescan.'),
        reviewer_was_production_kernel_author=True,independent_of_production_kernel=False,
        independent_mathematical_evidence='Separate monthly_asset_aux210_independent oracle and bound INDEPENDENT_PREPARE/INDEPENDENT_OUTCOME',
        review_scope='Saved JSON assertions and byte hashes; derived counts/holding changes/metric summary only, no numerical feature reconstruction.',
        limitations=context['limitations'],npz_loaded=False,raw_price_arrays_loaded=False,production_modules_imported=False,
        new_features_computed=False,new_models_fitted=False,new_orders_computed=False,new_portfolio_returns_computed=False,
        central_modified=False,observers_modified=False)
    if not check_only:
        with (OUT/'RESEARCH_REVIEW.json').open('x') as stream: json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')
        with (OUT/'RESEARCH_REVIEW.md').open('x') as stream: stream.write(markdown(report))
    return report


def markdown(report):
    lines=['# 第210轮：股票与ETF辅助分分组校准','',
        '候选'+('通过' if report['selection']['admitted'] else '未通过')+'前88期筛选。',
        '', '组内平均并列排名与端点匹配仿射控制，共同替换原pre-episode有效辅助贡献；原长期核心及episode支持项保持。', '']
    for path,title in (('same_close_zero','同收盘零成本'),('next_open_cost','次开盘、每边5基点')):
        lines += ['## '+title,'','|区间|Local67 CAGR / Sharpe / DD|组内排名|仿射控制|','|---|---:|---:|---:|']
        for period in BOUNDS:
            cells=[]
            for arm in ARMS:
                x=report['summary'][arm][path][period]
                cells.append(f"{x['cagr']:.2%} / {x['sharpe']:.3f} / {x['max_drawdown']:.2%}")
            lines.append('|'+period+'|'+'|'.join(cells)+'|')
        lines.append('')
    lines += ['未通过门槛：'+('、'.join(report['failed_training_guards']) or '无'),'','持仓集合变化（全期／前88／后34）：']
    for name,item in report['holding_changes'].items():
        lines.append(f"- {name}: {item['counts']['full122']} / {item['counts']['first88']} / {item['counts']['later34']}")
    lines += ['', '已核对244个资产组、3904个股票单元、2074个ETF单元、7份准备产物、38份结果、9条路径／45段／18组训练指标及全部123个缓存和订单前缀。',
        '', 'Local67指标、总费用、成交和每日权益/持仓与209精确比较；仅次开盘成本路径的逐期HOLDINGS.fee允许≤1e-10浮点差。零成本路径费用必须为零。',
        '', '本reader作者也是生产kernel作者；独立数学证据来自另行实现的oracle。本检查只读JSON和字节哈希，未加载NPZ、重建因子或回测。后34期为重复使用的诊断样本，未改观察账户。','']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion-sha256',required=True); parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(); report=review(args.completion_sha256,args.check_only)
    print(json.dumps(dict(passed=True,result=report['result'],check_only=args.check_only,
        output=str(OUT/'RESEARCH_REVIEW.json'),sha256=None if args.check_only else sha(OUT/'RESEARCH_REVIEW.json'))))

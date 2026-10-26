"""Saved-JSON and byte-integrity review for CLOSED round211 only.

The reader author also wrote the production model. This is not a second
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
OUT = ROOT/'outputs/monthly_forecast_blend211'
PRIOR = ROOT/'outputs/monthly_asset_aux210'
CENTRAL = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
LOCAL = ROOT/'outputs/monthly_local_linear70'
ARMS = ('incumbent_local67', 'annual_convex_blend', 'fixed_half_control')
PRIOR_ARMS = ('incumbent_local67', 'asset_aux_rank', 'asset_aux_affine_control')
PATHS = ('same_close_zero', 'next_open_zero', 'next_open_cost')
BOUNDS = {'full122':(0,122), 'first88':(0,88), 'early53':(0,53), 'recent35':(53,88), 'later34':(88,122)}
PREPARED_FILES = {'FORECASTS.npz', 'RANKS_DELTAS.npz', 'MODEL_AUDIT.json', 'ORDERS.json',
                  'META.json', 'SOURCE_PARITY.json', 'INDEPENDENT_PREPARE.json'}
RESULT_FILES = {'RESULTS.json', 'INDEPENDENT_OUTCOME.json'} | {
    a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS
    for k in ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')}
PRIOR_RESULT_FILES = {'RESULTS.json', 'INDEPENDENT_OUTCOME.json'} | {
    a+'_'+p+'_'+k+'.json' for a in PRIOR_ARMS for p in PATHS
    for k in ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')}
CONTEXT = 'work/monthly_forecast_blend211_context.json'
CONTEXT_SHA = '1cc5934776614b5d338a590ba442573c0d1eab07e32a8a623ef4ae6c8bb6c2fb'
COVERAGE = 'work/after210_forecast_blend_coverage.json'
COVERAGE_SHA = '6d3179937baaa2d469e9a8958c886165eb26d2176d43eceb55897e564e5fc4bf'
TESTS = 'work/monthly_forecast_blend211_TESTS.json'
CODE_REVIEW = 'work/forecast_blend211_CODE_REVIEW.json'
HELPER = 'work/aux_rank204_close_review_v2.py'
HELPER_SHA = '36acc8b196e29d19266d9a4e95fa65e9c652a416293231018331230c58512c70'
CODE_FILES = {f'work/{prefix}monthly_forecast_blend211_{kind}.py'
              for prefix in ('', 'test_') for kind in ('host', 'model', 'independent')}
TEST_FILES = sorted(x for x in CODE_FILES if Path(x).name.startswith('test_'))
SOURCE = ROOT/'outputs/monthly_identity_transfer206'
REBUILD = ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
SOURCE_REVIEW = 'work/forecast_blend211_SOURCE_REVIEW.json'
SOURCE_SHA = '245505c6c2939a5ecd9f5493f6b917862b4f3d13ff0ece5c1e6f3a791a59b855'
COUNTS = ('annual_coefficient_fits','cold_start_years','zero_denominator_years',
          'clipped_coefficient_years','calibration_months','calibration_rows',
          'joined_label_rows','invalid_label_rows','small_month_rows','cached_forecast_rows')
STAT_KEYS = {'start', 'end', 'years', 'final_equity', 'cagr', 'sharpe',
             'max_drawdown', 'trading_sessions', 'total_return'}
EXECUTIONS = {'same_close_zero':('close',0.), 'next_open_zero':('open',0.),
              'next_open_cost':('open',.0005)}
MODEL_FLAGS = ('passed','independent_scalar_rebuild','exact_identity_join_checked','cached_forecasts_reused',
    'annual_prior_december_cutoffs_checked','five_year_mature_window_checked','four_target_common_cohort_checked',
    'historical_query_support_checked','month_equal_losses_checked','convex_coefficient_checked',
    'half_fallback_checked','no_local_history_filter','availability_unchanged','original_targets_unchanged',
    'no_future_or_outcome_inputs','source_forecasts_not_refitted','audit_checked')
SLOT_FLAGS = ('passed','full_original_universe','includes_eligible_etfs','no_local_history_filter',
    'replacement_not_additive_local_slot','paired_whole_row_readiness','initial_partial_unchanged','exact_rank_rebuild')


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
    spec = importlib.util.spec_from_file_location('forecast_blend211_saved_helper', path)
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


def count_record(record, years=None, query_rows=None):
    require(isinstance(record,dict) and set(record)==set(COUNTS), 'Exactly ten calibration counters')
    for key,value in record.items(): integer(value,key)
    if years is not None:
        require(sum(record[k] for k in ('annual_coefficient_fits','cold_start_years','zero_denominator_years'))==years,
                'Annual fit/fallback accounting')
    require(record['clipped_coefficient_years']<=record['annual_coefficient_fits'], 'Clipped fit count')
    require(record['joined_label_rows']==record['invalid_label_rows']+record['small_month_rows']+record['calibration_rows'],
            'All joined labels accounted')
    require(record['calibration_rows']>=2*record['calibration_months'], 'Common month support')
    if query_rows is not None: require(record['cached_forecast_rows']<=query_rows, 'Forecast row count')


def training_identities(training, cutoff):
    require(len(training['dates'])==len(training['ends'])==len(training['symbols']), 'Training metadata axes')
    identities=[]
    for day,end,names in zip(training['dates'],training['ends'],training['symbols']):
        if end<=cutoff: identities.extend((day,symbol,end) for symbol in names)
    require(len(set(identities))==len(identities),'Unique original training triples')
    return identities


def model_history(audit, meta, training):
    """Reconcile saved descriptors and identities only; no new losses, coefficients or forecasts."""
    days,ends,names=meta['dates'],meta['ends'],meta['symbols'];n=len(days)
    width=integer(audit['symbols_per_event'],'Query width')
    require(width>0 and len(ends)==len(names)==n and audit['query_months']==n and audit['query_rows']==n*width,
            'Model query axes')
    require(all(len(x)==width and len(set(x))==width for x in names),'Monthly query identities')
    require(audit['method']=='annual_mature_asof_convex_forecast_blend' and audit['arms']==list(ARMS)
        and audit['min_months']==18 and audit['lookback_calendar_years']==5,'Fixed calibration definition')
    require(audit['month_equal_weights'] is True,'Equal month loss weights')
    for key in ('source_forecasts_refitted','labels_recomputed','local_available_filtered'):
        require(audit[key] is False,'Calibration scope '+key)
    years=sorted({int(d[:4]) for d in days})
    require([a['year'] for a in audit['annual']]==years,'Annual query clock')
    identities=training_identities(training,f'{years[-1]-1:04d}-12-31') if years else []
    counts={k:0 for k in COUNTS};summaries=[]
    for row in audit['annual']:
        y=integer(row['year'],'Year');cutoff=f'{y-1:04d}-12-31';start=f'{y-6:04d}-12-31'
        require(row['cutoff']==cutoff and row['lookback_exclusive_start']==start,'Prior December five-year boundary')
        expected=[i for i,(d,e) in enumerate(zip(days,ends)) if d>start and e<=cutoff and d!='2016-08-17']
        require([m['date'] for m in row['months']]==[days[i] for i in expected],'Exact mature query month list')
        joined=invalid=small=accepted=accepted_rows=0;latest=None
        for i,m in zip(expected,row['months']):
            require(m['date']==days[i] and m['end']==ends[i],'Recorded source month identity')
            used=integer(m['usable_rows'],'Usable historical rows');sr=integer(m['source_rows'],'Historical source rows')
            bad=integer(m['invalid_label_rows'],'Invalid label rows')
            require(used+bad==sr and sr<=width and type(m['accepted']) is bool and m['accepted']==(used>=2),
                    'Shared cohort month counts')
            vectors=('source_row_ids','training_row_ids','symbols','targets','own_predictions','peer_predictions')
            require(all(isinstance(m[k],list) and len(m[k])==used for k in vectors),'Saved historical row vectors')
            require(len(set(m['source_row_ids']))==used and len(set(m['training_row_ids']))==used,'Unique recorded row IDs')
            for sid,tid,symbol in zip(m['source_row_ids'],m['training_row_ids'],m['symbols']):
                require(type(sid) is int and i*width<=sid<(i+1)*width and names[i][sid%width]==symbol,
                        'Source query exact identity')
                require(type(tid) is int and 0<=tid<len(identities) and identities[tid]==(days[i],symbol,ends[i]),
                        'Mature training triple exact identity')
            for k in ('targets','own_predictions','peer_predictions'):
                for v in m[k]: finite(v,'Recorded finite '+k)
            require(len(m['numerator_terms'])==len(m['denominator_terms'])==(used if used>=2 else 0),
                    'Accepted-month loss term coverage')
            for v in m['numerator_terms']:finite(v,'Recorded numerator term')
            require(all(finite(v,'Recorded squared difference')>=0 for v in m['denominator_terms']), 'Squared differences nonnegative')
            if used>=2:
                finite(m['numerator_mean'],'Recorded monthly loss numerator')
                require(finite(m['denominator_mean'],'Recorded monthly loss denominator')>=0,'Monthly denominator')
                accepted+=1;accepted_rows+=used;latest=ends[i]
            else:
                require(m['numerator_mean'] is None and m['denominator_mean'] is None,'Unaccepted month is not a loss sample')
                small+=used
            joined+=sr;invalid+=bad
        for key,value in dict(joined_label_rows=joined,invalid_label_rows=invalid,small_month_rows=small,
            calibration_rows=accepted_rows,calibration_months=accepted).items():
            require(integer(row[key],key)==value,'Annual count '+key);counts[key]+=value
        require(row['latest_label_end']==latest,'Latest accepted mature endpoint')
        initial_capacity=sum(width for d,e in zip(days,ends) if d=='2016-08-17' and d>start and e<=cutoff)
        require(integer(row['skipped_initial_rows'],'Initial skipped rows')<=initial_capacity,'Initial history excluded')
        p=integer(row['prediction_rows'],'Available prediction rows')
        require(p<=width*sum(int(d[:4])==y for d in days),'Year prediction coverage');counts['cached_forecast_rows']+=p
        finite(row['numerator'],'Recorded annual loss numerator');den=finite(row['denominator'],'Recorded annual loss denominator')
        require(den>=0,'Annual denominator nonnegative')
        status='cold_start' if accepted<18 else 'zero_denominator' if den==0 else 'learned'
        require(row['status']==status and type(row['clipped']) is bool,'Fit/fallback classification')
        alpha=finite(row['alpha'],'Recorded own weight');require(0<=alpha<=1,'Convex own weight')
        if status!='learned':
            require(alpha==.5 and row['unclipped_alpha'] is None and row['clipped'] is False,'Exact half fallback')
        else:
            unconstrained=finite(row['unclipped_alpha'],'Recorded unconstrained coefficient')
            require(alpha==min(1.,max(0.,unconstrained)) and row['clipped']==(alpha!=unconstrained),'Recorded clipping descriptors')
        counts[{'learned':'annual_coefficient_fits','cold_start':'cold_start_years','zero_denominator':'zero_denominator_years'}[status]]+=1
        counts['clipped_coefficient_years']+=int(row['clipped'])
        summaries.append({k:row[k] for k in ('year','cutoff','lookback_exclusive_start','status','alpha','unclipped_alpha',
            'clipped','prediction_rows','calibration_months','calibration_rows','latest_label_end')})
    count_record(audit['counts'],len(years),n*width)
    require(audit['counts']==counts,'Main-pass count aggregation')
    for k in ('new_model_fits','predictive_model_fits'):
        require(integer(audit[k],k)==counts['annual_coefficient_fits'],'Scalar learned coefficients count as fits')
    for k in ('new_ridge_fits','new_tree_fits'):require(integer(audit[k],k)==0,'No source model refits')
    return counts,summaries


def integration_receipts(validation, prepared, closure, counts, meta, training):
    for key in MODEL_FLAGS:require(validation['model'].get(key) is True,'Model oracle proof '+key)
    for key in SLOT_FLAGS:require(validation['slot'].get(key) is True,'Slot oracle proof '+key)
    m,s=validation['model'],validation['slot'];n=len(meta['dates']);width=len(meta['symbols'][0]);years=len({d[:4] for d in meta['dates']})
    require(m['computes_returns'] is False and m['producer_modules_imported'] is False and m['atol']==0.
        and m['query_rows']==n*width and m['annual_cohorts']==years,'Exact oracle scope')
    require(m['counts']==counts,'Independent calibration accounting')
    for k in ('new_model_fits','predictive_model_fits'):require(integer(m[k],k)==counts['annual_coefficient_fits'],'Oracle fit accounting')
    for k in ('new_ridge_fits','new_tree_fits'):require(integer(m[k],k)==0,'Oracle source fits')
    require(set(s['ready_rows'])==set(s['changed_cells'])==set(ARMS[1:]),'Paired slot arms')
    require(len(set(s['ready_rows'].values()))==1,'Common whole-row readiness')
    for v in s['ready_rows'].values():require(integer(v,'Ready rows')<=n,'Readiness bound')
    for v in s['changed_cells'].values():require(integer(v,'Changed cells')<=n*width,'Changed-cell bound')
    require(validation['prefixes']==n+1 and validation['model_prefixes']==list(range(n+1)),'All physical prefixes')
    for key in ('model_prefix_sources_truncated','model_prefix_metadata_truncated','model_prefix_original_arrays_truncated',
        'model_prefix_future_labels_removed','source_history_starts_at_original_zero','original_orders_exact',
        'original_arrays_unchanged','original_core_metadata_and_support_preserved','single_original_episode','no_new_score_budget'):
        require(validation[key] is True,'Original invariant/prefix proof '+key)
    evidence=validation['model_prefix_evidence']
    require([x['prefix'] for x in evidence]==list(range(n+1)),'Prefix evidence completeness')
    for x in evidence:
        j=x['prefix'];day=meta['dates'][j-1] if j else None;cutoff=f'{int(day[:4])-1:04d}-12-31' if j else None
        ids=training_identities(training,cutoff) if j else []
        latest=max((item[2] for item in ids),default=None)
        require(x['query_rows']==j*width and integer(x['training_rows'],'Prefix training rows')==len(ids)
            and x['query_date']==day and x['cutoff']==cutoff and x['latest_label_end']==latest
            and x['source_history_starts_at_original_zero'] is True,'Physical source maturity/identity prefix')
        count_record(x['counts'],len({d[:4] for d in meta['dates'][:j]}),j*width)
        if not j:require(all(v==0 for v in x['counts'].values()),'Empty prefix counts')
    require(evidence[-1]['counts']==counts,'Full prefix is main-pass accounting')
    for x in (validation,prepared,closure):
        require(x['model_counts']==counts and x['original_array_count']==27,'Original27/main-pass model counts')
    require(prepared['new_returns_computed'] is False and prepared['source_model_refits']==0,'Preparation scope')
    require(prepared['model_fit_accounting']==closure['model_fit_accounting'],'Fit-accounting text')
    return dict(changed_cells=s['changed_cells'],ready_rows=s['ready_rows'],saved_npz_hash_verified_only=True,
        original_arrays_not_loaded=True,numerical_loss_or_forecast_not_recomputed=True,
        formula_evidence='Separately authored exact scalar oracle; this reader reconciles saved descriptors, identities and hashes.')


def source_binding(e,h,plan,context,parity,meta):
    binding=plan['model_source_binding'];r=e.load(SOURCE_REVIEW,SOURCE_SHA)
    require(h.local_path(binding['path'])==ROOT/SOURCE_REVIEW and binding['sha256']==SOURCE_SHA,
            'Registered source-review identity')
    require(r['passed'] is True and r['context_sha256']==CONTEXT_SHA and r['evidence_sha256']==context['evidence_sha256'],
            'Source context/evidence lineage')
    for k in ('cached_forecasts_are_prior_annual_asof_replays','not_realtime_archival_vintages','numeric_calibration_coverage_not_yet_computed'):
        require(r[k] is True,'Source scope '+k)
    for k in ('uses_fitted_residuals_as_historical_errors','source_npz_decoded','observers_modified'):
        require(r[k] is False,'Source prohibited operation '+k)
    for k in ('actual_new_losses','actual_new_coefficients','actual_new_orders','actual_new_returns'):
        require(type(r[k]) is int and r[k]==0,'Source prospective boundary')
    require(r['matched_formal_months']==113 and len(r['source_annual'])==11,'Existing vintage history only')
    for name,pin in context['evidence_sha256'].items():
        require(plan['input_sha256'].get(str(ROOT/name))==pin,'Context evidence registered');e.verify(name,pin)
    for name,pin in binding['input_sha256'].items():
        require(h.input_path(name,pin)!=CENTRAL and plan['input_sha256'].get(str(h.input_path(name,pin)))==pin,
                'Immutable source dependency registered');e.verify(name,pin)
    require(binding['directory']==SOURCE.name and binding['source_model_refits']==0,'Cached206 source identity')
    source_close=e.load(SOURCE/'COMPLETION.json',context['evidence_sha256'][str((SOURCE/'COMPLETION.json').relative_to(ROOT))])
    require(source_close['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and source_close['plan_sha256']==binding['source_plan_sha256'],'Closed206 lineage')
    e.manifest(SOURCE/'prepared',context['evidence_sha256'][str((SOURCE/'prepared/COMPLETION.json').relative_to(ROOT))],binding['source_plan_sha256'],PREPARED_FILES)
    for k in ('original_cached_training_and_queries','prior_annual_asof_forecasts'):require(binding[k] is True,'Source lineage '+k)
    for k in ('fresh_training_array_comparison','uses_fitted_residuals_as_historical_errors','new_labels','new_raw_features','strict_pit'):
        require(binding[k] is False,'Source claim '+k)
    pairs={'source_plan_sha256':SOURCE/'PLAN.json','source_forecasts_sha256':SOURCE/'prepared/FORECASTS.npz',
        'source_query_meta_sha256':SOURCE/'prepared/META.json','training_inputs_sha256':REBUILD/'TRAIN_INPUTS.npz',
        'training_meta_sha256':REBUILD/'MONTHLY_META.json','saved_training_parity_sha256':REBUILD/'PARITY.json'}
    for k,path in pairs.items():require(binding[k]==plan['input_sha256'][str(path)],'Source digest field '+k);e.verify(path,binding[k])
    require(parity['model_source_binding']==binding and parity['raw_source_hashes']==plan['raw_source_hashes'], 'Prepared source lineage')
    for k in ('original_training_cache_exact','original_identity_exact','cached206_source_vectors_exact','cached206_control_constant_exact',
        'source_availability_exact','no_source_model_refits','no_fitted_residual_error_inputs'):require(parity[k] is True,'Prepared source proof '+k)
    for k in ('fresh_training_array_comparison','new_labels','new_raw_features'):require(parity[k] is False,'Prepared source scope '+k)
    require(parity['saved_training_parity_sha256']==binding['saved_training_parity_sha256'] and
        parity['original_training_cache_exact_basis']=='Pinned TRAIN_INPUTS bytes and saved169/206 parity certificates reused; not a fresh original-array reconstruction',
        'Historical parity certificate must not be represented as a fresh numerical rebuild')
    data=e.load(REBUILD/'MONTHLY_META.json',binding['training_meta_sha256'])
    require(meta==data['monthly']==e.load(SOURCE/'prepared/META.json',binding['source_query_meta_sha256'])
        ==e.load(LOCAL/'META.json',plan['input_sha256'][str(LOCAL/'META.json')]),'Exact historical query identities')
    return data['training'],dict(source_model_refits=0,fresh_training_array_comparison=False,
        causal_annual_replay_not_realtime_vintages=True,source_history_not_extended=True)


def selection_from_saved(h,training,orders):
    result=h.selection_from_saved(training,orders)
    result['guards']['close_vs_half_control']=result['guards'].pop('close_vs_affine_control')
    result['guards']['open_vs_half_control']=result['guards'].pop('open_vs_affine_control')
    return result


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
    require(plan['schema']=='monthly_forecast_blend211_v1' and plan['contract']['arms']==list(ARMS)
        and plan['contract']['candidate']==ARMS[1] and plan['contract']['control_nonpromotable']==ARMS[2]
        and plan['contract']['context']==context and plan['contract']['prospective_context_sha256']==CONTEXT_SHA
        and plan['contract']['coverage_sha256']==COVERAGE_SHA and plan['contract']['max_absolute_delta']==.0125,
        'Fixed211 contract')
    require(len(plan['raw_source_hashes'])==64, 'Original raw64 closure')
    for mapping in (plan['input_sha256'],plan['raw_source_hashes']):
        for name,pin in mapping.items():
            path=h.input_path(name,pin)
            require(path!=CENTRAL and plan['input_sha256'].get(str(path))==pin,'Immutable complete PLAN inputs')
            e.verify(name,pin)
    e.load(COVERAGE,COVERAGE_SHA)
    calendar=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
    e.load(calendar,plan['input_sha256'][str(calendar)])
    for key in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration',
        'actual_new_losses_before_registration','actual_new_coefficients_before_registration','actual_new_orders_before_registration','actual_new_returns_before_registration'):
        require(type(plan[key]) is int and plan[key]==0,'Prospective registration boundary')
    before=e.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    require(len(before['completed'])==208 and sum(x['trials'] for x in before['completed'])==1141,'Predecessor counters')
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
    training_meta,cache=source_binding(e,h,plan,context,saved['SOURCE_PARITY.json'],meta)
    require(len(meta['dates'])==122 and all(len(x)==49 for x in meta['symbols']),'Original122x49 queries')
    require(orders[ARMS[0]]==e.load(LOCAL/'local_linear_ORDERS.json',plan['input_sha256'][str(LOCAL/'local_linear_ORDERS.json')]), 'Original Local67 orders')
    counts,annual=model_history(saved['MODEL_AUDIT.json'],meta,training_meta)
    integration=integration_receipts(validation,prepared,closure,counts,meta,training_meta)
    require(prepared['model_fit_accounting']==plan['contract']['model_fit_accounting']==context['formation']['fit_accounting'], 'Registered scalar-fit accounting')
    results=e.load(OUT/'results/RESULTS.json'); outcome=e.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    for key in ('annual_convex_forecast_blend','same_support_both_arms'):
        require(results[key] is True,'Result interpretation '+key)
    require(results['observer_promotion'] is False and results['strict_pit'] is False and results['cached_source_models_refitted'] is False and results['reused_sample'] is True,'Result limitations')
    require(set(orders)==set(training)==set(results['arms'])==set(outcome['paths'])==set(ARMS),'Three arms')
    selected=selection_from_saved(h,training,orders)
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
    predecessor=plan['predecessor_binding']; require(predecessor['directory']==PRIOR.name,'210 predecessor')
    old=e.load(PRIOR/'COMPLETION.json',predecessor['input_sha256'][str(PRIOR/'COMPLETION.json')])
    old_rc=e.load(PRIOR/'results/COMPLETION.json',old['result_completion_sha256'])
    require(old['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and old_rc['status']=='COMPLETE'
        and old_rc['plan_sha256']==old['plan_sha256'] and set(old_rc['artifacts_sha256'])==PRIOR_RESULT_FILES,'Prior exact38 closure')
    old_results=e.load(PRIOR/'results/RESULTS.json',old_rc['artifacts_sha256']['RESULTS.json'])
    incumbent=h.incumbent_comparison(e,old_rc,results,old_results)
    incumbent=json.loads(json.dumps(incumbent).replace('previous203','previous210').replace('current204','current211'))
    changed={name:h.changes(orders,a,b) for name,a,b in (('candidate_vs_incumbent',ARMS[0],ARMS[1]),
        ('control_vs_incumbent',ARMS[0],ARMS[2]),('candidate_vs_control',ARMS[2],ARMS[1]))}
    protected=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
    require(len(e.load(protected,plan['input_sha256'][str(protected)]))==closure['protected_files']==21444
        and closure['observers_modified'] is False and closure['new_trials']==3,'Protected closure')
    central=e.load(CENTRAL)
    require(central['current']==OUT.name and central['current_status']=='completed_independently_verified'
        and central['registered_queue']==[] and len(central['completed'])==209
        and sum(x['trials'] for x in central['completed'])==1144,'Immediate closed211 counters')
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
    e.verify(HELPER,HELPER_SHA); e.verify(__file__); e.verify('work/test_forecast_blend211_close_review.py')
    for name,pin in list(e.pins.items()): e.verify(name,pin)
    report=dict(schema='monthly_forecast_blend211_research_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,result='ADMITTED' if selected['admitted'] else 'NOT_ADOPTED',selection=selected,
        failed_training_guards=[k for k,v in selected['guards'].items() if not v],summary=summaries,
        holding_changes=changed,model_counts=counts,annual_calibration=annual,cache_source_coverage=cache,integration=integration,
        zero_fee_evidence=zero_fees,incumbent210_export_diagnostic=incumbent,source_refs=e.pins,
        phase_sha256=dict(plan=plan_sha,prepared=frozen['prepared_sha256'],selected=rc['frozen_sha256'],
            results=closure['result_completion_sha256'],closure=completion_sha),formal_test_evidence=test_evidence,
        best_training_selected_candidate=best,historical_best_unchanged=best==oldbest,target_reached=closure['target_reached'],
        validation=dict(prepared_artifacts=7,result_artifacts=38,independent_query_rows=5978,independent_paths=9,independent_slices=45,training_metric_groups=18,
            order_prefixes=123,physical_cache_and_metadata_prefixes=list(range(123)),original_arrays=27,
            new_model_fits=counts['annual_coefficient_fits'],new_ridge_fits=0,new_tree_fits=0,protected_files=21444,completed_entries=209,trials=1144,
            protection_scope='Hash-bound original manifest and successful registration/closure receipts; no archive rescan.'),
        reviewer_was_production_model_author=True,independent_of_production_model=False,
        independent_mathematical_evidence='Separate monthly_forecast_blend211_independent oracle and bound INDEPENDENT_PREPARE/INDEPENDENT_OUTCOME',
        review_scope='Saved JSON assertions and byte hashes; derived counts/holding changes/metric summary only, no loss/coefficient/forecast reconstruction.',
        limitations=context['limitations'],npz_loaded=False,raw_price_arrays_loaded=False,production_modules_imported=False,
        new_features_computed=False,new_losses_computed=False,new_coefficients_fitted=False,new_models_fitted=False,new_orders_computed=False,new_portfolio_returns_computed=False,
        central_modified=False,observers_modified=False)
    if not check_only:
        with (OUT/'RESEARCH_REVIEW.json').open('x') as stream: json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')
        with (OUT/'RESEARCH_REVIEW.md').open('x') as stream: stream.write(markdown(report))
    return report


def markdown(report):
    lines=['# 第211轮：年度成熟预测误差的凸组合','',
        '候选'+('通过' if report['selection']['admitted'] else '未通过')+'前88期筛选。',
        '', '复用206因果年度预测，以过去已成熟月份的等月权重误差学习年度own权重；控制固定为二分之一。原ML四分之一槽位及一次episode保持。','']
    for path,title in (('same_close_zero','同收盘零成本'),('next_open_cost','次开盘、每边5基点')):
        lines += ['## '+title,'','|区间|Local67 CAGR / Sharpe / DD|年度凸组合|固定半权重|','|---|---:|---:|---:|']
        for period in BOUNDS:
            cells=[]
            for arm in ARMS:
                x=report['summary'][arm][path][period]
                cells.append(f"{x['cagr']:.2%} / {x['sharpe']:.3f} / {x['max_drawdown']:.2%}")
            lines.append('|'+period+'|'+'|'.join(cells)+'|')
        lines.append('')
    lines += ['未通过门槛：'+('、'.join(report['failed_training_guards']) or '无'),'',
        '已核对7份准备产物、38份结果、9条路径／45段／18组训练指标及全部123个训练、查询、元数据、原数组和订单前缀。',
        '', '年度系数学习计为模型拟合；零新增Ridge/树拟合不等于零新增模型拟合。缺少2016年前预测不作填补。',
        '', 'Local67指标、总费用、成交和每日权益/持仓与210精确比较；仅次开盘成本路径的逐期HOLDINGS.fee允许≤1e-10浮点差。零成本路径费用必须为零。',
        '', '本reader作者也是生产model作者；独立数学证据来自另行实现的oracle。本检查只读JSON和字节哈希，未加载NPZ、重算损失/系数或回测。后34期为重复使用的诊断样本，未改观察账户。','']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion-sha256',required=True); parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(); report=review(args.completion_sha256,args.check_only)
    print(json.dumps(dict(passed=True,result=report['result'],check_only=args.check_only,
        output=str(OUT/'RESEARCH_REVIEW.json'),sha256=None if args.check_only else sha(OUT/'RESEARCH_REVIEW.json'))))

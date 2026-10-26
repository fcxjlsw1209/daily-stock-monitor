"""Round233 metadata-only source binding; never decode numeric arrays or prices.

Verifies the original sources and saved192 control certificate. Actual233 support,
raw-control/availability/order parity are mandatory later preparation checks.
"""
from datetime import date, datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'work/sector_signedrank233_SOURCE_REVIEW.json'
CONTEXT = ROOT/'work/monthly_sector_signedrank233_context.json'
CONTEXT_SHA = '565c11f6a0343f9c321c57816a0becb68ee236f403a9a802098d26fa1d0d87ab'
PRIOR = ROOT/'outputs/monthly_volume_slope232'
PRIOR_PLAN = '268c385aef1845ecdf6b244c7a3fe99c5a3f7fff22b73d315d54a8bdf6034256'
PRIOR_CLOSE = '8f85e2fbba3151af26f279d02c02552dfaa34d2c86ad1c24161d8fc523e8dd81'
PRIOR_REVIEW = 'ba29648ec9e6cacda1cb5b6631c1cd1d2b196c9af5a6977a0444e332d1630494'
CONTROL = ROOT/'outputs/monthly_sector_recurrence192'
CONTROL_PLAN = '36b65d71662805a2662b89e22077a6af6a831dfc3e9df4c7a4ad834838762259'
CONTROL_CLOSE = '8fe405ab72b1a1b4828d2a71e6626467479792056c7e557310fc06d6ec562281'
CONTROL_PREPARED = 'fe34887e4ee2e91a7045fb7d89183ab75f9ba9d4b3fd137b0a2b442a575bd6c3'
LOCAL = ROOT/'outputs/monthly_local_linear70'
CALENDAR = ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
CALENDAR_SHA = '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
MAPPER = ROOT/'outputs/monthly_strict_relative/strict_study.py'
MAPPER_SHA = 'ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
SECTORS = ROOT/'outputs/index_union/alias_resolved_inputs/sectors.csv.gz'
ETF_MANIFESTS = ('outputs/four_factor_rebound/ETF_SOURCES.json',
    'outputs/monthly_reconstruction_v2/ETF_MANIFEST.json',
    'outputs/monthly_reconstruction_v2/foreign/ETF_MANIFEST.json')
SECTOR_ETFS = ('XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY')
PREPARED_FILES = {'FEATURES.npz','TRANSFORMS_DELTAS.npz','FEATURE_AUDIT.json',
    'ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
PROTECTED = ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
CENTRAL = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
EXTERNAL = {'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':
    '6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PINS = {}


def need(condition, message):
    if not condition: raise ValueError(message)


def resolve(value):
    p = Path(value); p = (p if p.is_absolute() else ROOT/p).resolve()
    need('CloudStorage' not in p.parts, 'Cloud source forbidden')
    need(p.is_relative_to(ROOT) or str(p) in EXTERNAL, 'External source forbidden')
    need(p != CENTRAL, 'Mutable central cannot become immutable input pin')
    return p


def sha(value):
    h = hashlib.sha256()
    with resolve(value).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def bind(value, expected=None):
    p = resolve(value); key = str(p)
    if key in PINS:
        need(expected is None or PINS[key] == expected, 'Conflicting pin: '+key)
        return PINS[key]
    actual = sha(p)
    need(expected is None or actual == expected, 'Source bytes changed: '+key)
    if key in EXTERNAL: need(actual == EXTERNAL[key], 'Pinned runtime changed')
    PINS[key] = actual
    return actual


def read(value, expected=None):
    bind(value, expected)
    return json.loads(resolve(value).read_text())


def text_check(value, tokens):
    bind(value); content = resolve(value).read_text()
    for token in tokens: need(token in content, 'Missing source definition: '+str(value)+' / '+token)


def review():
    need(not OUT.exists(), 'Source receipt is write-once')
    central_bytes = CENTRAL.read_bytes(); state = json.loads(central_bytes)
    need(len(state['completed']) == 228 and sum(x['trials'] for x in state['completed']) == 1201,
        'Expected232 closed,228 strategy entries and1201 trials')
    need(state['completed'][-1]['directory'] == PRIOR.name and
        state['completed'][-1]['completion']['sha256'] == PRIOR_CLOSE, 'Predecessor232 central identity')
    need(not state['registered_queue'] and state['best_training_selected_candidate']['directory'] == LOCAL.name,
        'Empty queue and unchanged Local67 incumbent required')
    context = read(CONTEXT, CONTEXT_SHA)
    need(context['round'] == 233 and context['name'] == 'monthly_sector_signedrank233', 'Context identity')
    need(context['arms'] == ['incumbent_local67','sector_signed_magnitude','sector_sign_control'], 'Fixed arm identity')
    need(context['formation']['parameters'] == dict(blocks=12,block_sessions=21,skip_sessions=42,
        price_count=253,neutral_pseudo_blocks=4,doubled_rank_denominator=208,control_denominator=16), 'Formation contract')
    need(context['integration']['dose'] == .025 and context['integration']['maximum_absolute_delta'] == .01875, 'Fixed direct dose')
    for key in ('actual_new_features_computed','actual_new_orders_computed','actual_returns_computed'):
        need(context[key] is False, 'Prospective context scope')
    evidence = {p:bind(p,h) for p,h in context['evidence_sha256'].items()}
    prior = read(PRIOR/'PLAN.json', PRIOR_PLAN)
    prior_close = read(PRIOR/'COMPLETION.json', PRIOR_CLOSE)
    prior_review = read(PRIOR/'RESEARCH_REVIEW.json', PRIOR_REVIEW)
    need(prior_close['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED' and prior_close['plan_sha256'] == PRIOR_PLAN
        and prior_close['protected_files'] == 21444 and prior_close['observers_modified'] is False, 'Prior closure')
    need(prior_review['passed'] is True and prior_review['plan_sha256'] == PRIOR_PLAN
        and prior_review['completion_sha256'] == PRIOR_CLOSE, 'Prior independent saved review')
    for p,h in prior['input_sha256'].items(): bind(p,h)
    old_plan = read(CONTROL/'PLAN.json', CONTROL_PLAN)
    old_close = read(CONTROL/'COMPLETION.json', CONTROL_CLOSE)
    old_prepared = read(CONTROL/'prepared/COMPLETION.json', CONTROL_PREPARED)
    need(old_close['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED' and old_close['plan_sha256'] == CONTROL_PLAN
        and old_close['protected_files'] == 21444 and old_close['observers_modified'] is False, '192 closed certificate')
    need(old_prepared['status'] == 'COMPLETE' and old_prepared['plan_sha256'] == CONTROL_PLAN
        and set(old_prepared['artifacts_sha256']) == PREPARED_FILES, '192 exact prepared artifact set')
    for name,h in old_prepared['artifacts_sha256'].items(): bind(CONTROL/'prepared'/name,h)
    for p,h in old_plan['input_sha256'].items(): bind(p,h)
    old_parity = read(CONTROL/'prepared/SOURCE_PARITY.json')
    old_check = read(CONTROL/'prepared/INDEPENDENT_PREPARE.json')
    f = old_check['feature']; d = old_check['delta']
    for key in ('passed','independent_scalar_rebuild','own_sector_mapping_checked','exact_rational_relative_signs_checked',
        'twelve_nonoverlapping_blocks_checked','full_253_price_support_checked','official_session_windows_checked',
        'monthend_queries_checked','exact_common_availability','audit_checked','no_future_or_outcome_inputs'):
        need(f[key] is True, 'Prior192 feature certificate: '+key)
    need(f['events'] == 122 and f['symbols_per_event'] == 49 and f['checked_cells'] == 5978
        and f['available_cells'] == 3705 and f['unavailable_cells'] == 2273, 'Prior192 saved dimensions/support')
    need(f['relative_sign_observations'] == old_prepared['relative_sign_observations'] == 44460,
        'Prior192 relative sign certificate')
    need(old_check['prefixes'] == 123 and old_check['feature_prefixes'] == [0,1,53,88,122]
        and old_check['feature_prefix_sources_truncated'] is True and old_check['original_arrays_unchanged'] is True,
        'Prior192 prefix and original-array proof')
    need(d['passed'] is True and d['dose'] == .025 and d['initial_delta_zero'] is True
        and d['no_cross_sectional_ranking'] is True and d['direct_additive_pre_score'] is True,
        'Prior192 direct integration contract')
    local_plan = read(LOCAL/'PLAN.json','0016175aee1df43b734010d80bd543b881c57036a7f8c6ab73428340203afc96')
    original_inputs = read(LOCAL/'INPUT_HASHES.json','5ac2b639fa8c944688787eba47ea1b420bfbe9c6e1e051a3d827d8b2b32b264d')
    raw = local_plan['raw_source_hashes']
    need(len(raw) == 64 and raw == prior['raw_source_hashes'] == old_plan['raw_source_hashes'] == old_parity['raw_source_hashes'],
        'Original64 raw source map equality')
    for p,h in raw.items(): need(original_inputs.get(p) == h, 'Raw source absent from original inputs'); bind(p,h)
    meta = read(LOCAL/'META.json','3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4')
    need(meta == read(CONTROL/'prepared/META.json'), '192/Local exact metadata axes')
    need(len(meta['dates']) == len(meta['ends']) == len(meta['symbols']) == 122 and
        meta['dates'] == sorted(set(meta['dates'])) and meta['ends'][:-1] == meta['dates'][1:], 'Original event axes')
    need(meta['dates'][0] == '2016-08-17' and meta['dates'][-1] == '2026-08-31' and meta['ends'][-1] == '2026-09-24', 'Original date bounds')
    need(len(meta['etfs']) == len(set(meta['etfs'])) == 17 and all(len(ss) == len(set(ss)) == 49 for ss in meta['symbols']), 'Original49/17 identities')
    for ss in meta['symbols']:
        need(all(type(s) is str and s and s.strip() == s for s in ss), 'Exact security identities')
    cache = read(LOCAL/'CACHE_HASHES.json','e3b440f8e2f2669d535cb92d33a2302be7ceec4f54792ff4078a7d976356460b')
    for name,h in cache.items(): bind(LOCAL/name,h)
    bind(LOCAL/'local_linear_ORDERS.json','c0e9286cdfae5e586a83ee0c412447a607ca21d1a5d6bbe47d40bd494b9d947e')
    policy = read(LOCAL/'INCUMBENT_POLICY.json')
    need(all(policy[k] == v for k,v in dict(budget=.05,ceiling=5,gap=.02).items()), 'Original episode policy')
    bind(MAPPER,MAPPER_SHA)
    need(raw.get(str(SECTORS)) == bind(SECTORS), 'Original sector classification source')
    manifests = {}; entries = {}
    for p in ETF_MANIFESTS:
        manifest = ROOT/p; contents = read(manifest); manifests[str(manifest)] = PINS[str(manifest)]
        need(raw.get(str(manifest)) == PINS[str(manifest)], 'Original ETF manifest pin')
        for entry in contents:
            symbol = entry['symbol']
            if symbol not in SECTOR_ETFS: continue
            need(symbol not in entries, 'Ambiguous own-sector ETF declaration')
            path = resolve(entry['file']); digest = bind(path,entry['sha256'])
            need(raw.get(str(path)) == digest, 'ETF data absent from original64')
            entries[symbol] = dict(path=str(path),sha256=digest,manifest=str(manifest))
    need(set(entries) == set(SECTOR_ETFS), 'All11 own-sector ETF sources')
    sector_binding = dict(mapping_source=dict(path=str(MAPPER),sha256=MAPPER_SHA),
        sectors_source=dict(path=str(SECTORS),sha256=PINS[str(SECTORS)]),manifest_sha256=manifests,
        own_sector_etfs=list(SECTOR_ETFS),sources=entries,price_column='Adj Close',loader='original study.arch.old.panel_load')
    need(sector_binding == old_plan['sector_source_binding'] == old_parity['sector_source_binding'], 'Exact192 sector binding')
    calendar = read(CALENDAR,CALENDAR_SHA)
    for p,h in calendar['provenance']['source_hashes'].items(): bind(p,h)
    sessions = calendar['sessions']
    need(sessions == sorted(set(sessions)) and calendar['start'] == '2008-01-01' and calendar['end'] == '2035-12-31', 'Full official calendar metadata')
    for value in sessions: need(type(value) is str and date.fromisoformat(value).isoformat() == value, 'Canonical calendar date')
    positions = {v:i for i,v in enumerate(sessions)}; windows = []
    for start,end in zip(meta['dates'],meta['ends']):
        need(start in positions and end in positions and end > start, 'Official interval clocks')
        t = positions[start]; need(t >= 294 and t+1 < len(sessions), 'Formation calendar bound')
        if start != '2016-08-17': need(start[:7] != sessions[t+1][:7], 'Formal monthend query')
        formation = sessions[t-294:t-41]; need(len(formation) == 253, 'Full253 official date slots')
        windows.append(dict(date=start,end=end,price_start=formation[0],price_end=formation[-1],
            price_count=253,return_span=252,blocks=12,block_sessions=21,
            endpoint_dates=[formation[j] for j in range(0,253,21)],
            actual_prices_read=False,actual_233_support='UNKNOWN',initial_partial_delta_zero=start == '2016-08-17'))
    text_check(MAPPER,('def benchmark_map(date, symbols, sectors, etfs):','if s in etfs:','ans[s] = None','elif s in FOREIGN:'))
    text_check('work/monthly_sector_recurrence192_kernel.py',('relative_sum / DENOMINATOR',"sectors.loc[:day]",'np.isfinite(converted) & (converted > 0.)'))
    text_check('work/monthly_sector_recurrence192_host.py',('np.savez_compressed(dest / \'FEATURES.npz\', **values)',
        'def decisions(study, arrays, meta, values, n):','pre_score=pre, effects=arrays[\'effects\']'))
    text_check('outputs/monthly_local_linear70/study.py',('def modified_pre_score(spec,a,n):','def verify_panel_parity(panel,px,op):','len(hashes)==64'))
    text_check('outputs/monthly_stability_monitor/inputs.py',("panel['adj'].loc[gap,s]=z.loc[gap,'Adj Close']*ratio",'def load_panel(etfs):'))
    protected = read(PROTECTED); need(len(protected) == 21444, 'Complete protected manifest')
    implementation = bind(Path(__file__))
    direct = dict(raw); direct[str(CALENDAR)] = CALENDAR_SHA; direct[str(MAPPER)] = MAPPER_SHA
    direct[str(LOCAL/'META.json')] = PINS[str(LOCAL/'META.json')]
    for p,h in calendar['provenance']['source_hashes'].items(): direct[p] = h
    report = dict(schema='sector_signedrank233_source_metadata_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,source_only=True,metadata_only=True,source_admission_metadata_only=True,
        source_npz_decoded=False,npz_decoded=False,price_rows_read=False,source_loader_executed=False,source_models_imported=False,
        actual_support_computed=False,actual_source_coverage_recomputed=False,actual_new_features_computed=False,
        actual_features_computed=False,actual_orders_computed=False,actual_new_orders_computed=False,actual_returns_computed=False,
        actual_model_fits=0,actual_new_estimates=0,observers_modified=False,central_modified=False,frozen_files_modified=False,
        strict_pit=False,source_publication_clock_certified=False,issuer_identity_certified=False,
        metadata_axes_verified=True,calendar_clock_verified=True,raw_source_hashes_verified=True,raw_source_map_matches_original=True,
        sector_source_binding_verified=True,prior192_control_artifacts_bound=True,integration_source_verified=True,
        source_sha256=dict(sorted(PINS.items())),input_sha256=dict(sorted(PINS.items())),evidence_sha256=evidence,
        raw_source_hashes=raw,raw_source_count=64,direct_feature_source_sha256=dict(sorted(direct.items())),
        context_sha256=CONTEXT_SHA,sector_source_binding=sector_binding,
        source_identity=dict(query_months=122,query_width=49,query_rows=5978,etfs=17,own_sector_etfs=11,
            original_cached_array_count=27,cache_count_basis='Existing frozen certificates; no array decoding',
            unique_candidate_symbols=len({s for ss in meta['symbols'] for s in ss}),first_query=meta['dates'][0],
            last_query=meta['dates'][-1],last_end=meta['ends'][-1],monthly_axes_exact192=True),
        calendar=dict(path=str(CALENDAR),sha256=CALENDAR_SHA,session_count=len(sessions),windows=windows,
            complete_date_windows_verified=True,actual_per_security_price_support='UNKNOWN'),
        control192=dict(plan_sha256=CONTROL_PLAN,completion_sha256=CONTROL_CLOSE,prepared_completion_sha256=CONTROL_PREPARED,
            artifact_sha256=old_prepared['artifacts_sha256'],feature_array_keys=['candidate','control','available'],
            feature_key_to_reuse='candidate',availability_key_to_reuse='available',order_arm_to_reuse='sector_relative_recurrence',
            prior_certificate_available_cells=3705,prior_certificate_unavailable_cells=2273,prior_certificate_relative_sign_observations=44460,
            support_counts_are_saved192_certificate_not_actual233_computation=True,
            source_and_metadata_bindings_match=True,actual233_control_parity_computed=False,
            value_availability_order_parity_required_after_registration=True),
        provenance_scope=dict(new_input_sources_required=False,new_data_collection_required=False,
            inputs='Only original adjusted stock/own-sector ETF prices, original query-date sector mapping, official calendar and current122x49 identity axes',
            prior232_input_pins_role='Immutable predecessor lineage; unrelated labels, models and volume are not233 new feature inputs',
            current_local67_models_reused_without_refit=True,host_kernel_oracle_not_pinned_to_avoid_cycle=True),
        mapping_policy=dict(query_mapping_held_through_formation=True,candidate_etfs_always_neutral=True,
            unknown_mapping_neutral=True,foreign_overrides_inherited=True,historical_classification_certified=False,
            no_sector_etf_substitution=True,no_cross_issuer_alias_inference=True),
        price_policy=dict(adjusted_price_source='Original frozen panel loader; retrospectively retrieved vendor versions',
            price_unit_semantics='Inherited normalization/alias providers and leading-gap price splices; exact raw fingerprints preserved',
            no_volume_or_dollar_turnover_used=True,complete_actions_or_first_vintage_certified=False,
            actual_adjusted_and_execution_price_parity_required_in_prepare=True),
        integration=dict(core_and_27_cached_arrays_unchanged=True,original_core_weights=[.5,.52],skip42_unchanged=True,
            dose=.025,max_absolute_delta=.01875,initial_partial_delta_zero=True,cross_sectional_rank_added=False,
            episode_calls_per_arm=1,original_episode_policy={k:policy[k] for k in ('budget','ceiling','gap')},
            raw_control_definition='192 relative-sign candidate, not192 absolute-sign control',
            unsupported_assets_remain_in_original_pool_with_zero_delta=True,new_ml_fits=0),
        required_prepare_checks=[
            'Reverify full64 loader map and27 cached-array bytes, incumbent orders and execution price parity after PLAN.',
            'Rebuild233 common full253 stock/sector support with exact original mapping, all49 cells, no fill or benchmark substitution.',
            '233 control and available must match192 FEATURES candidate and available exactly, including raw float dtype/NaNs.',
            '233 control orders must match192 sector_relative_recurrence; firstpartialdelta0 and one original episode.',
            'Independent exact Fraction relative-move signs/absolute ordering/doubled ties, finite arithmetic,208/16 denominator and bounded raw.',
            '123 decision prefixes and5 physically truncated price-and-sector prefixes;9 paths45 slices independently verified.'],
        protected_manifest=dict(path=str(PROTECTED),sha256=PINS[str(PROTECTED)],members=21444,
            all_members_rehashed_by_this_metadata_review=False,prior_terminal_closure_proves_archive_check=True),
        predecessor232=dict(plan_sha256=PRIOR_PLAN,completion_sha256=PRIOR_CLOSE,review_sha256=PRIOR_REVIEW),
        central_snapshot=dict(sha256=hashlib.sha256(central_bytes).hexdigest(),persistent_input=False,
            completed_entries=228,trials=1201,last_strategy=PRIOR.name,registered_queue_empty=True),
        implementation=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=implementation),
        limitations=context['limitations']+[
            'Metadata-only admission does not prove any new233 feature value, support cell or control-order identity; those remain mandatory postregistration checks.',
            'The saved192 support counts and certificates are inherited evidence, not a fresh computation of233 coverage.',
            'Canonical research symbols and query-date industry mapping are inherited proxies, not independently certified issuer continuity or historical sector membership.'])
    need(CENTRAL.read_bytes() == central_bytes, 'Central changed during metadata review')
    with OUT.open('x') as stream: json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(passed=True,receipt_sha256=sha(OUT),implementation_sha256=implementation,
        source_pins=len(PINS),raw_sources=64,query_months=122,sector_etfs=11,
        prior192_available_cells=3705,actual233_support_computed=False,npz_decoded=False)),flush=True)


if __name__ == '__main__': review()

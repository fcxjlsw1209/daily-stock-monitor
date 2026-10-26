"""Write-once prospective249 contracts from code/metadata and byte hashes only.

No strategy imports, price/volume decoding, NPZ access, outcome reads or fitting.
Use run_local.sh. The actual source and incumbent gates remain future work.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'work/volume_breadth249_SOURCE_REVIEW.json'
CONTEXT = ROOT / 'work/monthly_volume_breadth249_context.json'
TRAINING_CAGR_COMPARISON_ATOL = 1e-12  # Existing guard only; all other guards strict.
EXPECTED = {
 'work/after248_volume_breadth_definition_draft.json':'c95851e4eb2c1d523b27241c543fa9a5cd587b866d947596b708e33201671e61',
 'work/after248_volume_breadth_DEFINITION_REVIEW.json':'2f0cd21d0f66d0f982c2726e3e2bb1b72c20d93fcd647b22a53e0de969a01452',
 'work/after248_volume_breadth_POLICY_DECISION.json':'76746876ca853fd0d1c91c666f5e340bdcdf4038150589ba085c4715d0e40b86',
 'work/after248_volume_breadth_source_preflight.json':'5b189d6a52f7b05197081727809673dde12253218ff3e0a47e8846e60277f306',
 'work/after248_volume_breadth_TRANSITION_LEDGER.json':'4b35ee2b539e83e4502741fbba2edefa41c874b2082d37f06f927f6098146db8',
 'work/monthly_volume_breadth249_ADAPTER_REVIEW.json':'555244f3b2136e7a5ff6b6eb715ed614e7a576b5c6509c8c635ee768392d3379',
 'work/monthly_volume_breadth249_math.py':'c632e74a76fbc1975fff3e56d96b4ff1fccd314828e3099bd9495d7424a73ee9',
 'work/test_monthly_volume_breadth249_math.py':'5b0b2291775b9bde7947926b3cbfecad1fcba90107943f47f162a31b5f6271ad',
 'work/monthly_volume_breadth249_math_independent.py':'4e416cc897b63f453a0f2fbcdedc833684924affaa5c02be92e2de099150bc3b',
 'work/test_monthly_volume_breadth249_math_independent.py':'840423ec775159890cf43e337504eda9efc548e3145e9fd4413d6c9322931e4c',
 'work/monthly_volume_breadth249_adapter.py':'4794e3d2a6b6a3196a0bb5602b781cf324e4cf34a5cb42ba9a3c19e40f7ee2c0',
 'work/test_monthly_volume_breadth249_adapter.py':'47f59cb6a113efd49d7d48a778cb1090d7bc5af71cddf135822872fc3a0e7e09',
 'work/calendar_clock248_ORIGIN_DIAGNOSTIC.json':'d250fc4d42b6089c62f4b1c4a032b872db7a1dd3cc99f9c133b8c8421c8c0eaf',
 'work/after248_v2_strategy_checkpoint.json':'d4717959714625b34413bb0d89c869e9fd24231f5cb39d107a137a29c2dadef8',
 'outputs/monthly_daily_episodes/episodes.py':'8ae91e55e0d662cb6dee9cba6d1422aa3713b779976de69b4560194d311ae16a',
 'outputs/monthly_local_linear70/study.py':'e8cfe0c27a6805cbee4b59d24ce9771ece7caa01c29f1a30aa2d5acf9837f3fb',
 'outputs/monthly_stability_monitor/inputs.py':'dd97fb4a9f7b2102d2c7f497825526a484628fa86ee9891312bfb5abad7c8ded',
 'outputs/index_union/price_units.py':'353e2f039d022a03cf90be06028d680b8dd389fad7bb216727cc2ecbda226291',
}


def path(name):
    p = Path(name)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    assert p.is_relative_to(ROOT) and 'CloudStorage' not in p.parts
    return p


def sha(name):
    digest = hashlib.sha256()
    with path(name).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(name):
    return json.loads(path(name).read_text())


def bind(pins, name, expected=None):
    name = str(path(name).relative_to(ROOT))
    actual = sha(name)
    assert expected is None or expected == actual, 'Changed evidence: ' + name
    assert name not in pins or pins[name] == actual, 'Conflicting evidence: ' + name
    pins[name] = actual
    return actual


def write_new(name, value):
    with name.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(name.relative_to(ROOT)), sha256=sha(name))), flush=True)


def main():
    assert not SOURCE.exists() and not CONTEXT.exists(), 'Both outputs must be new'
    pins = {}
    for name, digest in EXPECTED.items():bind(pins, name, digest)
    draft = read('work/after248_volume_breadth_definition_draft.json')
    definition = read('work/after248_volume_breadth_DEFINITION_REVIEW.json')
    policy = read('work/after248_volume_breadth_POLICY_DECISION.json')
    pre = read('work/after248_volume_breadth_source_preflight.json')
    ledger = read('work/after248_volume_breadth_TRANSITION_LEDGER.json')
    adapter_review = read('work/monthly_volume_breadth249_ADAPTER_REVIEW.json')
    origin = read('work/calendar_clock248_ORIGIN_DIAGNOSTIC.json')
    assert definition['verdict']['passed'] and not definition['verdict']['algebraic_or_api_blockers']
    assert adapter_review['passed'] and adapter_review['unresolved_blockers'] == []
    assert policy['registered'] is False and policy['source_admitted'] is False and policy['strict_pit'] is False
    for name, digest in pre['input_sha256'].items():bind(pins, name, digest)
    for name, digest in ledger['evidence_sha256'].items():bind(pins, name, digest)
    for name, detail in draft['evidence_sha256'].items():bind(pins, name, detail['sha256'])
    for field in ['code_sha256', 'input_sha256']:
        for name, digest in adapter_review[field].items():bind(pins, name, digest)
    for name in ['work/calendar_clock248_origin_diagnostic.py',
                 'outputs/index_union/assemble_input_diagnostics.py',
                 'outputs/index_union/assemble_alias_resolved_inputs.py',
                 'outputs/monthly_state_rank_research/state_rank.py',
                 'outputs/monthly_local_linear70/PLAN.json',
                 'outputs/monthly_local_linear70/META.json',
                 'outputs/monthly_paired_target143/CALENDAR.json',
                 'work/prepare_volume_breadth249_contract.py']:
        bind(pins, name)
    local_plan = read('outputs/monthly_local_linear70/PLAN.json')
    raw = {str(path(name)):digest for name,digest in local_plan['raw_source_hashes'].items()}
    pre_raw = {str(path(item['path'])):item['sha256'] for item in pre['original64']['source_bindings']}
    assert len(raw) == 64 and raw == pre_raw
    for name,digest in raw.items():bind(pins,name,digest)
    manifest = read('outputs/index_union/alias_resolved_inputs/MANIFEST.json')
    assert origin['status'] == 'ORIGINAL_LOADER_AXIS_DIAGNOSTIC' and origin['raw_sources_exact'] is True
    assert origin['raw_sources'] == 64 and origin['actual_origin'] == '2008-01-02'
    assert origin['actual_end'] == '2026-10-06' and origin['actual_rows'] == 4720
    assert manifest['start'] == origin['manifest_start'] == '2010-01-04'
    assert origin['loader_path'] == str(path('outputs/monthly_state_rank_research/state_rank.py'))
    calendar_path = pre['calendar']['path'];calendar = read(calendar_path)['sessions']
    meta = read('outputs/monthly_local_linear70/META.json')
    assert len(meta['dates']) == len(pre['calendar']['events']) == 122
    assert all(len(names) == len(set(names)) == 49 for names in meta['symbols'])
    lookup = {day:i for i,day in enumerate(calendar)}
    assert len(lookup) == len(calendar)
    for i,event in enumerate(pre['calendar']['events']):
        day = meta['dates'][i];t = lookup[day]
        assert event['query_index'] == i and event['date'] == day and event['symbols'] == meta['symbols'][i]
        assert event['query_position'] == t and event['next_official_session'] == calendar[t+1]
        assert event['price_start'] == calendar[t-294] and event['price_end'] == calendar[t-42]
        assert event['volume_start'] == calendar[t-356] and event['volume_end'] == calendar[t-42]
        assert event['price_count'] == 253 and event['volume_count'] == 315
        assert event['initial_partial'] == (day == '2016-08-17')
        if i:assert day[:7] != calendar[t+1][:7]
    bounds = {item['symbol']:item for item in pre['listing']['bounds']}
    cells = pre['listing']['identity_cells']
    expected_cells = {(i,j) for i,names in enumerate(meta['symbols']) for j,s in enumerate(names) if s in bounds}
    assert len(cells) == len(expected_cells) == 5
    assert {(c['query_index'],c['column_index']) for c in cells} == expected_cells
    for cell in cells:
        event = pre['calendar']['events'][cell['query_index']]
        bound = bounds[cell['symbol']]
        assert event['symbols'][cell['column_index']] == cell['symbol'] and cell['date'] == event['date']
        assert all(cell[key] == bound[key] for key in ['issuer_key','available_from','not_before'])
        assert all(cell[key] == event[key] for key in ['price_start','price_end','volume_start','volume_end'])
        assert cell['bound_visible'] == (cell['available_from'] <= cell['date'])
    assert ledger['automatic_known_unresolved_transition_fallback_ids'] == []
    assert len(ledger['undated_known_loader_rules']) == 3
    assert all(not record['known_unresolved_fallback_eligible'] for record in ledger['records'])
    now = datetime.now(timezone.utc).isoformat()
    source_policy = {
      'mode':'DECLARED_ASSUMPTION_ARCHIVED_MIXED_STORED_VOLUME',
      'strict_pit':False,'unit_continuity':'UNKNOWN','actual_common49_coverage':'UNKNOWN_NOT_SCANNED',
      'zero_volume_rule':'Observed zero is a hard admission-contract failure, not proof of a vendor error.',
      'original_control_precedence':'Original adjusted-price query/required columns and consumed official t-294..t row axis must exist and align exactly; missing original control structure hardfails before auxiliary fallback. Retain original full-history pandas NaN/Inf sanitation and zero-sigma semantics; do not add a full-history positivity filter or reindex the original clock.',
      'ordinary_missing':'After original source/control structure passes, absent required volume symbol/session or None/pandas.NA/real numeric NaN in auxiliary253/315 windows gives whole-event fallback after all49 scans. A price row/column needed for original control is not eligible for this fallback. No survivor re-ranking or imputation.',
      'invalid_source':'Boolean/non-real (including numeric strings/NaT), failed or overflowing binary64 cast, infinity or finite<=0: hard failure, even if another source/member is missing.',
      'identity':'For a same-security bound with available_from<=query, any present positive required price OR volume observation before not_before is a hard contradiction. Check253-price and315-volume windows separately; a different missing cell or transition fallback cannot hide it.',
      'known_transition':'Only the exact pinned ledger eligible unresolved entries can cause whole-event fallback. This ledger contains zero such dated entries. Do not treat informational rename/mask/corporate-event dates as unit transitions.',
      'unknown_transition':'The3 undated early bridges and other unrecorded action/ADR/provider continuity stay UNKNOWN under the explicit research assumption. No fabricated effective date, numeric discontinuity detector, adjusted/close-derived volume multiplier, source substitution or retrospective outcome-driven repair.',
      'no_short_circuit':'Inspect every required49 member and all present source slots for hard numeric/identity errors before ordinary missing or known-transition fallback.',
      'source_observation':'Use actual observed source slots on the official required dates; loader reindex-created NaN remains missing. No row compression, forward fill or zero fill.',
      'named_initial_partial':'2016-08-17 skips all new price/volume numeric feature reads and copies exact incumbent artifacts.',
      'freeze':'New transition evidence requires separately reviewed/versioned source policy; this registered definition cannot acquire outcome-dependent exceptions.'}
    source = dict(schema='monthly_volume_breadth249_metadata_source_review_v1', created_utc=now,
      status='METADATA_REVIEW_COMPLETE_NUMERIC_ADMISSION_PENDING',passed=True,metadata_only=True,
      source_admitted=False,strict_pit=False,registered_strategy=False,
      prices_loaded=False,volumes_loaded=False,npz_decoded=False,
      actual_features_orders_returns_computed=False,
      raw_source_count=64,raw_source_hashes=raw,
      raw_source_verification='All64 original files were streamed for SHA only; no bar or cache decoded.',
      panel_first_session='2008-01-02',panel_axis_end=origin['actual_end'],panel_axis_rows=origin['actual_rows'],
      panel_origin_evidence=dict(path='work/calendar_clock248_ORIGIN_DIAGNOSTIC.json',sha256=pins['work/calendar_clock248_ORIGIN_DIAGNOSTIC.json'],
        scope='Previously saved original-loader axis diagnostic, not numeric coverage or core parity.'),
      manifest_reported_start=manifest['start'],manifest_reported_start_is_panel_origin=False,
      panel_manifest_path='outputs/index_union/alias_resolved_inputs/MANIFEST.json',
      panel_manifest_sha256=pins['outputs/index_union/alias_resolved_inputs/MANIFEST.json'],
      calendar_path=calendar_path,calendar_sha256=pins[calendar_path],
      metadata_path='outputs/monthly_local_linear70/META.json',metadata_sha256=pins['outputs/monthly_local_linear70/META.json'],
      events=122,width=49,formal_monthends=121,etfs=deepcopy(meta['etfs']),
      clock_events=deepcopy(pre['calendar']['events']),
      price_sessions=253,volume_sessions=315,return_sessions=252,volume_extra_leading_sessions=62,
      listing_bounds=deepcopy(pre['listing']['bounds']),identity_cells=deepcopy(cells),
      listing_scope=pre['listing']['limitation'],unknown_other_listing_bounds=True,
      source_classes=deepcopy(pre['source_classes']),source_lineage=deepcopy(pre['source_lineage']),
      upstream_lineage_sha_scope='Metadata leads unless independently present in the original64 closure; no fresh upstream row/action validation.',
      transition_ledger_path='work/after248_volume_breadth_TRANSITION_LEDGER.json',
      transition_ledger_sha256=pins['work/after248_volume_breadth_TRANSITION_LEDGER.json'],
      transition_records=deepcopy(ledger['records']),undated_loader_rules=deepcopy(ledger['undated_known_loader_rules']),
      automatic_known_unresolved_transition_fallback_ids=[],empty_transition_list_certifies_continuity=False,
      source_policy=source_policy,actual_cell_coverage='UNKNOWN_UNTIL_REGISTERED_PREPARE',
      original_U0_and_core_parity='UNKNOWN_UNTIL_REGISTERED_PREPARE',
      actual_unit_continuity='UNKNOWN_DECLARED_ASSUMPTION',
      source_scope=dict(VOX_used=False,sector_mapping_used=False,new_downloads=False,extra_sources=[],original49_including_ETFs=True,eligibility_filter_used=False),
      definition_path='work/after248_volume_breadth_definition_draft.json',definition_sha256=pins['work/after248_volume_breadth_definition_draft.json'],
      definition_review_path='work/after248_volume_breadth_DEFINITION_REVIEW.json',definition_review_sha256=pins['work/after248_volume_breadth_DEFINITION_REVIEW.json'],
      preflight_path='work/after248_volume_breadth_source_preflight.json',preflight_sha256=pins['work/after248_volume_breadth_source_preflight.json'],
      policy_path='work/after248_volume_breadth_POLICY_DECISION.json',policy_sha256=pins['work/after248_volume_breadth_POLICY_DECISION.json'],
      input_sha256=deepcopy(pins))
    arms = ['incumbent_local67','formation_volume_confirmation_breadth','formation_volume_confirmation_mean_control']
    assert draft['arms'] == arms and draft['funding_and_selector']['budget'] == .01
    context = dict(schema='monthly_volume_breadth249_context_v1',created_utc=now,
      status='PROSPECTIVE_NOT_REGISTERED',arms=arms,registered_trials=3,
      primary_candidate=arms[1],matched_control=arms[2],only_candidate_promotable=True,
      predecessor='monthly_calendar_clock248_v2',predecessor_completed_rounds=244,predecessor_trials=1245,
      completed_rounds_after=245,trials_after=1248,
      predecessor_gate='Future host must bind the independently reviewed completed predecessor PLAN/COMPLETION/result-review receipts and exact central state at registration. This builder does not read predecessor results or certify closure.',
      predecessor_checkpoint=dict(path='work/after248_v2_strategy_checkpoint.json',sha256=pins['work/after248_v2_strategy_checkpoint.json'],scope='Root-reported terminal closure/checkpoint; byte hash only here, content/outcomes not decoded.'),
      new_model_fits=0,new_predictive_model_fits=0,registered=False,source_admitted=False,strict_pit=False,
      hypothesis=draft['hypothesis'],coverage=deepcopy(draft['coverage']),
      clock=dict(query='Official month-end, except named2016-08-17 initial partial.',
        prices='253 official sessions t-294..t-42 inclusive',volumes='315 official sessions t-356..t-42 inclusive',
        return_days='252 days t-293..t-42 inclusive',volume_extra_leading_sessions=62,
        current_volume_index='V[q+63]',previous63_volume_indices='V[q:q+63], current excluded',
        median_index=31,blocks=12,block_returns=21,latest_auxiliary_source='t-42',
        original_signals='Original states/risk/models retain their frozen clocks; do not claim every selector input ends at t-42.'),
      source=dict(review_path=str(SOURCE.relative_to(ROOT)),review_sha256=None,panel_fields=['adj','volume'],
        panel_first_session='2008-01-02',manifest_reported_start='2010-01-04',
        common_domain='Exactly all49 original per-event names, includingETFs and Efalse names, before eligibility. Both experimental arms share one complete-event source mask.',
        all_missing_fallback='Any ordinary required source absence or known eligible unresolved ledger transition makes both experiments copy incumbent for the entire event. Continue full hard-error scan first.',
        source_policy=deepcopy(source_policy)),
      exact_math=deepcopy(definition['exact_arithmetic_contract']),
      identification=deepcopy(definition['identification_review']),
      funding=dict(budget=.01,scope='All existing P-C auxiliary contributions including ML, calm extension and episode.',
        core='Original weighted C and common cw=.50 calm/.52 otherwise remain unchanged.',
        canonical='coefficient=.01/(1-cw); alpha=1-coefficient; correction=.01*Q-coefficient*(P-C); copyP then add only nonzero correction.',
        original_episode='e0=min(.05,max(0,.10-(.025*calm+.05*weak)))',
        requested_episode_budget='active:alpha*e0; inactive/control:.05, always shape(n,).',
        original_support='(original baseline-C)/(1-cw), cw explicitly broadcast(n,49).',
        selector='One unchanged SHA-bound ep.select_boundary call perarm, includingn0; funded pre defines own hook; originalE/effects/support, ceiling5,gap.02; return original choices/order, never globally rerank final scores.',
        bypass='Named partial, missing or known transition: inactive; exact original P/F/orders compared and copied by host. No target numeric consumption.',
        constant_target='FiniteQ0 or all-tied1/98 remain active and fund old auxiliaries; only computed correction0 skipsaddition.',
        bound='Only .01*Q is bounded by .005; no universal bound is claimed for total score change or unchanged holdings.',
        adapter_api=deepcopy(adapter_review['api']),
        inherited_support_validation_atol=1e-12,tie_tolerance=0.),
      phase_order=[
        'Register immutable context/source/implementation/test/review/runtime/input hashes and predecessor closure before actual249 features.',
        'Load original64 sources and first verify original full-history source/core reproduction and official t-294..t consumed axis. Original control structure failure cannot be swallowed by auxiliary missing fallback. Then independently verify official253/315 source windows, full49 identities, complete scan and transition policy without any candidate scoring.',
        'Reconstruct original Local67 config4 P and one original episode F/ordered choices; independently require exact frozen incumbent control before either experimental score/order.',
        'Use independently verified exact rational block features to form both49 target ranks and score integration after the incumbent gate; feature statistics may be prepared earlier without candidate scores or orders. Inactive events retain source gaps and original artifacts.',
        'For each experimental arm call the new source-free adapter once, preserve original27 arrays and E, compare/copy bypass artifacts, and independently verify full features/pre/final/order/audit.',
        'Complete123 cached logical/order prefixes and five physical source/calendar/cache/meta prefixes before preparation receipt.',
        'Select candidate only on first88 and fixed zero-fee guards; freeze selection before later34/full diagnostics; independently verify9paths/45slices and close only after saved-evidence review.'],
      validation=dict(events=122,width=49,formal_events=121,logical_prefixes=list(range(123)),physical_prefixes=[0,1,53,88,122],
        physical_source_end='At most query date; calendar at most one next-session month-end witness; n0 no consumed source numeric.',
        original_cache_arrays=27,original_E_unchanged=True,original_incumbent_orders_exact=True,
        exact_rational_statistics=True,rank_tolerance=0.,source_masks_exact=True,discrete_choices_exact=True,
        original_weighted_core_source_atol=1e-12,original_weighted_core_source_rtol=0.,
        original_core_tolerance_scope='Inherited original-core provenance check only; no rounding/epsilon for U0, ranks, new rational statistics or ties.',
        paths=9,slices=45,zero_fee_training_groups=18,protected_files=21444,
        independent_implementation='Do not import producer math/adapter as oracle; reconstruct source/masks/Fraction statistics/ranks, originalpre, allaux funding and episode support independently.',
        admission_pending='Integrated source producer/oracle/host tests, final code review and prospective registration remain required. Standalone pure-math/adapter reviews are not strategy/source admission.'),
      selection=dict(training_events=88,early_events=53,recent_events=35,
        primary_candidate=arms[1],references=[arms[0],arms[2]],
        same_close_zero_cagr_gain_over_incumbent_min=.005,same_close_zero_cagr_gain_over_mean_min=0.,
        next_open_zero_cagr_gain_over_each_reference_min=0.,sharpe_loss_max=.05,max_drawdown_loss_max=.03,
        each_training_era_cagr_loss_max=.025,changed_holdings_min_each_reference=3,changed_holdings_each_training_era_min_each_reference=1,
        guarded_paths=['same_close_zero','next_open_zero'],later34_used_for_selection=False,fee_path_used_for_selection=False,
        training_cagr_comparison_atol=TRAINING_CAGR_COMPARISON_ATOL,
        training_cagr_comparison='On each zero-fee training path and both references: candidate CAGR >= reference CAGR + required_gain -1e-12.',
        sharpe_comparison_atol=0.,max_drawdown_comparison_atol=0.,era_cagr_comparison_atol=0.,
        comparison_tolerance_scope='Only aggregate first88 training CAGR guard comparisons use1e-12. Sharpe>=reference-.05, max_drawdown>=reference-.03 and each era CAGR>=reference-.025 are strict literal comparisons. Never apply performance tolerance to source/masks/Fraction signs/ranks/ties.',
        reject_action='Retain incumbent; the mean comparator cannot become a promoted backup.'),
      execution=dict(primary='same_close_zero',diagnostics=['next_open_zero','next_open_cost_0.0005_each_side'],
        start='2016-08-17',end='2026-09-24',long_only=True,leverage=1.,maximum_positions=3,
        three_positions='equal thirds',two_eligible='half each',fewer_than_two='cash',fixed_intramonth_units=True,
        difference_rebalance=True,terminal_liquidation=False),
      limitations=deepcopy(draft['limitations'])+[
        'The3 undated early bridges cannot support an automatic dated-window gate. Their continuity remains an explicit research assumption.',
        'An empty dated unresolved-transition list is not evidence of continuity; all raw64 hash matches are byte identity only.',
        'The2008 axis diagnostic certifies saved loader axis metadata, not actual complete249 windows or candidate support.',
        'History is repeatedly researched; later34 is reused diagnostic, not pristine out-of-sample evidence.'],
      input_sha256=deepcopy(pins),
      builder_scope=dict(metadata_and_code_only=True,raw_source_files_sha_only=64,price_volume_rows_decoded=False,
        npz_decoded=False,predecessor_returns_read=False,source_loader_executed=False,
        actual_features_computed=False,actual_orders_computed=False,actual_returns_computed=False,
        tests_run_by_builder=0,network_requests=0,central_state_written=False,registered_trials_by_builder=0,old_files_modified=0))
    write_new(SOURCE,source)
    context['source']['review_sha256'] = sha(SOURCE)
    context['input_sha256'][str(SOURCE.relative_to(ROOT))] = sha(SOURCE)
    write_new(CONTEXT,context)
    print(json.dumps(dict(status='PROSPECTIVE_CONTRACTS_WRITTEN_NOT_REGISTERED',raw_sources=64,
        metadata_events=122,listing_identity_cells=5,undated_bridges=3,dated_unresolved_transition_rules=0,
        input_pins=len(pins),source_admitted=False,actual_numeric_decoded=False)),flush=True)


if __name__ == '__main__':
    main()

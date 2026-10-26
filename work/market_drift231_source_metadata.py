"""Round231 metadata-only source proof, executed under the offline wrapper.

Hashes source bytes and reads immutable JSON/source text. Never imports a model,
loader or array package and never decodes prices/NPZs or evaluates source support.
"""
from datetime import date, datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'work/market_drift231_SOURCE_REVIEW.json'
CONTEXT = ROOT/'work/monthly_market_drift231_context.json'
CONTEXT_SHA = 'bfa2fc65db4a3e1a50cda8979eef65e513049f196a70cc0647864e0d50b66de9'
PROPOSAL = ROOT/'work/market_drift231_PROPOSAL.json'
PROPOSAL_SHA = '46556aa1b5d6176c6960ae08fbfe5e1190b7042c15d8f321eaa4d1f7e6463b73'
CLARIFICATIONS = ROOT/'work/market_drift231_CLARIFICATIONS.json'
CLARIFICATIONS_SHA = '40675e741285248f15abc47114ba4105268451d6b1f9eb883af7f35b271dbf9c'
PRIOR = ROOT/'outputs/monthly_separate_history230'
PRIOR_PLAN = 'f8e0c1baceee92b2975117825ab077909b4cdabe0c7ab02fb5f59a9dc50e8b5c'
PRIOR_CLOSE = 'ea29a346fccc2a6fdc05696377d3cb6c487f557a7d87c0e92e420db81f14b23f'
PRIOR_REVIEW = '548e0bad94a450a7790efa26f4016b7dd69e64b975f1a0fc4da966c7018146ef'
LOCAL = ROOT/'outputs/monthly_local_linear70'
CALENDAR = ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
CALENDAR_SHA = '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
PROTECTED = ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
CENTRAL = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
EXTERNAL = {'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':
 '6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PINS = {}


def need(condition, message):
    if not condition: raise ValueError(message)


def resolve(value):
    p = Path(value); p = (p if p.is_absolute() else ROOT/p).resolve()
    need('CloudStorage' not in p.parts, 'No cloud source')
    need(p.is_relative_to(ROOT) or str(p) in EXTERNAL, 'Unexpected external source')
    need(p != CENTRAL, 'Mutable central must not become a persistent source pin')
    return p


def sha(value):
    h = hashlib.sha256()
    with resolve(value).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def bind(value, expected=None):
    p = resolve(value); key = str(p)
    if key in PINS:
        need(expected is None or PINS[key] == expected, 'Conflicting source hash: '+key)
        return PINS[key]
    actual = sha(p)
    need(expected is None or actual == expected, 'Source hash changed: '+key)
    if key in EXTERNAL: need(actual == EXTERNAL[key], 'Runtime hash changed')
    PINS[key] = actual
    return actual


def read(value, expected=None):
    bind(value, expected)
    return json.loads(resolve(value).read_text())


def day(value):
    need(type(value) is str and date.fromisoformat(value).isoformat() == value, 'Canonical ISO date required')
    return value


def source_text(value, expected, tokens):
    bind(value, expected); text = resolve(value).read_text()
    for token in tokens: need(token in text, 'Source implementation token missing: '+str(value)+' / '+token)
    return text


def review():
    need(not OUTPUT.exists(), 'New source receipt is write-once')
    central_bytes = CENTRAL.read_bytes(); state = json.loads(central_bytes)
    need(len(state['completed']) == 226 and sum(x['trials'] for x in state['completed']) == 1195, 'Expected226 closed strategies /1195trials')
    need(state['completed'][-1]['directory'] == PRIOR.name and state['completed'][-1]['completion']['sha256'] == PRIOR_CLOSE, 'Closed230 central identity')
    need(not state['registered_queue'] and state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Empty queue / original incumbent required')
    context = read(CONTEXT, CONTEXT_SHA); proposal = read(PROPOSAL, PROPOSAL_SHA)
    clarification = read(CLARIFICATIONS, CLARIFICATIONS_SHA)
    need(context['round'] == proposal['round'] == 231 and context['name'] == proposal['name'] == 'monthly_market_drift231', 'Prospective231 context')
    need(context['arms'] == proposal['arms'] == ['incumbent_local67','market_conditioned_drift','unconditional_drift_control'], 'Fixed three arms')
    need(clarification['proposal_sha256'] == PROPOSAL_SHA and clarification['stage'] == 'unregistered_before_actual_features_or_outcomes', 'Prospective clarifications')
    definition = context['definition']; formation = definition['formation_window']
    need(formation['returns_calm10'] == 210 and formation['returns_other_states'] == 252, 'Distinct formation lengths')
    need(definition['measurement']['minimum_matching_state_days'] == 42 and definition['measurement']['no_forward_labels'] is True and definition['measurement']['no_new_model_fits'] is True, 'No predictive model/forward labels')
    for record in (context, proposal, clarification):
        for key in ('actual_new_features_computed','actual_new_parameters_computed','actual_new_orders_computed','actual_new_returns_computed'):
            need(record[key] is False, 'Prospective design computation boundary')
    evidence = {p:bind(p,h) for p,h in context['evidence_sha256'].items()}
    need(evidence[str(PROPOSAL.relative_to(ROOT))] == PROPOSAL_SHA and evidence[str(CLARIFICATIONS.relative_to(ROOT))] == CLARIFICATIONS_SHA, 'Definition evidence pins')
    predecessor = read(PRIOR/'PLAN.json', PRIOR_PLAN)
    closed = read(PRIOR/'COMPLETION.json', PRIOR_CLOSE)
    saved = read(PRIOR/'RESEARCH_REVIEW.json', PRIOR_REVIEW)
    results = read(PRIOR/'results/COMPLETION.json', closed['result_completion_sha256'])
    need(closed['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED' and closed['plan_sha256'] == PRIOR_PLAN and closed['protected_files'] == 21444 and closed['observers_modified'] is False, 'Terminal230 receipt')
    need(saved['passed'] is True and saved['plan_sha256'] == PRIOR_PLAN and saved['completion_sha256'] == PRIOR_CLOSE and saved['completed_entries'] == 226 and saved['trials'] == 1195, 'Independent230 review lineage')
    need(results['status'] == 'COMPLETE' and results['plan_sha256'] == PRIOR_PLAN and results['paths'] == 9 and results['slices'] == 45, 'Closed230 outcome completion')
    # These inherited paths establish immutable lineage, not new-feature inputs.
    for p,h in predecessor['input_sha256'].items(): bind(p,h)
    inherited_pins = dict(predecessor['input_sha256'])
    local_plan = read(LOCAL/'PLAN.json','0016175aee1df43b734010d80bd543b881c57036a7f8c6ab73428340203afc96')
    old_inputs = read(LOCAL/'INPUT_HASHES.json','5ac2b639fa8c944688787eba47ea1b420bfbe9c6e1e051a3d827d8b2b32b264d')
    raw = local_plan['raw_source_hashes']
    need(len(raw) == 64 and raw == predecessor['raw_source_hashes'], 'Original64 raw-source map')
    for p,h in raw.items():
        need(old_inputs.get(p) == h, 'Raw source absent from original registered input map')
        bind(p,h)
    meta = read(LOCAL/'META.json','3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4')
    need(len(meta['dates']) == len(meta['ends']) == len(meta['symbols']) == 122 and meta['dates'] == sorted(set(meta['dates'])), 'Original122 monthly axes')
    need(meta['dates'][0] == '2016-08-17' and meta['dates'][-1] == '2026-08-31' and meta['ends'][-1] == '2026-09-24', 'Original event endpoints')
    need(len(meta['etfs']) == len(set(meta['etfs'])) == 17 and all(len(x) == len(set(x)) == 49 for x in meta['symbols']), 'Original49candidate/17ETF axes')
    need(all(type(s) is str and s and s.strip() == s for ss in meta['symbols'] for s in ss), 'Canonical original security IDs')
    need(meta['ends'][:-1] == meta['dates'][1:] and all(e>d for d,e in zip(meta['dates'],meta['ends'])), 'Original monthly intervals')
    cache = read(LOCAL/'CACHE_HASHES.json','e3b440f8e2f2669d535cb92d33a2302be7ceec4f54792ff4078a7d976356460b')
    for p,h in cache.items(): bind(LOCAL/p,h)
    bind(LOCAL/'local_linear_ORDERS.json','c0e9286cdfae5e586a83ee0c412447a607ca21d1a5d6bbe47d40bd494b9d947e')
    policy = read(LOCAL/'INCUMBENT_POLICY.json')
    need(all(policy[k] == v for k,v in dict(budget=.05,ceiling=5,gap=.02).items()), 'Frozen original episode policy')
    calendar = read(CALENDAR,CALENDAR_SHA)
    for p,h in calendar['provenance']['source_hashes'].items():bind(p,h)
    sessions = calendar['sessions']
    need(type(sessions) is list and sessions == sorted(set(sessions)) and calendar['start'] == '2008-01-01' and calendar['end'] == '2035-12-31', 'Full143 official calendar')
    for value in sessions:day(value)
    positions = {d:i for i,d in enumerate(sessions)}; windows = []
    for d,e in zip(meta['dates'],meta['ends']):
        day(d);day(e); need(d in positions and e in positions,'Decision/end is not an official session')
        t = positions[d]; need(t >= 493 and t+1 < len(sessions),'Calendar covers maximum preceding state window')
        if d != '2016-08-17':need(d[:7] != sessions[t+1][:7], 'Formal query is not a month end')
        branches = {}
        for name,L in (('calm10',210),('other_states',252)):
            stock = sessions[t-42-L:t-41]
            historical = sessions[t-42-L-199:t-42]
            current = sessions[t-199:t+1]
            all_spy = sessions[t-42-L-199:t+1]
            need(len(stock)==L+1 and len(historical)==L+199 and len(current)==200 and len(all_spy)==L+242,'Official metadata window lengths')
            need(set(historical)|set(current)==set(all_spy),'Historical and current SPY windows form exact contiguous union')
            need(stock[-1] == sessions[t-42] and historical[-1] == sessions[t-43] and current[-1] == d,'Information cutoffs')
            branches[name] = dict(return_intervals=L,stock_price_count=L+1,stock_price_start=stock[0],first_stock_return_date=stock[1],stock_price_end=stock[-1],
                historical_state_first_date=stock[0],historical_state_last_date=sessions[t-43],historical_state_count=L,
                historical_spy_start=historical[0],historical_spy_end=historical[-1],historical_spy_unique_prices=L+199,
                current_spy200_start=current[0],current_spy200_end=d,required_spy_union_start=all_spy[0],required_spy_union_end=d,
                required_spy_union_count=L+242,earliest_spy_offset=-(L+241))
        windows.append(dict(date=d,end=e,initial_partial_copies_incumbent=d=='2016-08-17',branches=branches,
            current_calm63_price_start=sessions[t-63],current_calm63_price_end=d,current_calm63_prices=64))
    code = {}
    definitions = {
      'outputs/monthly_local_linear70/study.py':('e8cfe0c27a6805cbee4b59d24ce9771ece7caa01c29f1a30aa2d5acf9837f3fb',(
        'def load():','m,_,px,op=scope.load();assert m==read(P/\'META.json\')','def modified_pre_score(spec,a,n):','def verify_panel_parity(panel,px,op):',
        'np.testing.assert_allclose(source[valid],raw[valid]*scale,rtol=1e-10,atol=1e-10)','expected=px*opening/close')),
      'outputs/monthly_stability_monitor/inputs.py':('dd97fb4a9f7b2102d2c7f497825526a484628fa86ee9891312bfb5abad7c8ded',(
        "('adj','Adj Close')","panel[k][entry['symbol']]=z[col].reindex(panel[k].index)","gap=a.isna()&(a.index<first)","panel['adj'].loc[gap,s]=z.loc[gap,'Adj Close']*ratio","for s,cache in [('CHL','CHL'),('PTR','PTRCY')]")),
      'outputs/monthly_state_rank_research/state_rank.py':('53866e5ffcb06146f73bb409747df4bb6f5b7864c1bfe0edebb639d551aa4c6d',(
        'def panel_load(etfs):',"file=O/'monthly_stability_monitor/inputs.py'","panel,info=mod.load_panel(etfs);info['source_hashes'][str(file)]=sha(file)")),
      'outputs/monthly_stability_research/local_refine.py':('5207909c5e777b1918676942a130e8be62a7a90a6bea1469938cae9aedcab949',(
        'mv=px.SPY.pct_change(fill_method=None).rolling(63).std()*np.sqrt(252)','calm10=mv.loc[dates].to_numpy()<.10')),
      'outputs/monthly_stability_research/stability.py':('2d22b1bcef8925c8f98d6778142968f2ee3e2f0847c46caf2aea8c6de2b72ad4',(
        'mom = {n: px.shift(42) / px.shift(n) - 1 for n in [252, 294]}',"(63, 0, 1., 'v63')")),
      'outputs/monthly_quality_research/explore.py':('5af173a8d6a659153278eedc4e38bc2c95f18dee5b10ac4d67cf44bdeb1ddfde',(
        'cw=np.where(calm,.5,.52)',"core=np.where(calm,.5*f['252_raw'],d['core'])")),
      'outputs/monthly_path70_research/incumbent_adapter.py':('5619bfc3512a0aa057e79042256a379360d6baa1bc0c448f2f301b84d6ec3e2b',(
        'support = (baseline-core)/(1-cw)','old_budget = .025*calm[:, None]+.05*weak[:, None]',
        'episode_budget = np.minimum(.05, np.maximum(0., OLD_LIMIT-old_budget))','return boundary_selector(score, good.copy(), [list(ss) for ss in symbols]')),
      'outputs/monthly_daily_episodes/episodes.py':('8ae91e55e0d662cb6dee9cba6d1422aa3713b779976de69b4560194d311ae16a',(
        'support = (a[\'baseline\'][:n] - a[\'core\'][:n]) / (1 - a[\'core_weight\'][:n])',
        "target = np.clip(support[i, j] + effects[i, j], -.5, .5)",'score[i, j] += budget[i] * (target - support[i, j])')),
      'outputs/index_union/price_units.py':('353e2f039d022a03cf90be06028d680b8dd389fad7bb216727cc2ecbda226291',(
        'Dividend-adjusted close is kept untouched for total-return momentum.','def raw_to_split_adjusted(frame):'))}
    for p,(digest,tokens) in definitions.items():source_text(p,digest,tokens);code[p]=digest
    loader=ROOT/'outputs/monthly_stability_monitor/inputs.py'
    need(raw.get(str(loader))==code[str(loader.relative_to(ROOT))], 'Original64 loader identity')
    manifest=read('outputs/index_union/alias_resolved_inputs/MANIFEST.json','b48b3e337c9789bd6effbdb5ee533fddfbcfe124dfb2a07d330b9f1da43ee5ce')
    need(manifest['start']=='2010-01-04' and manifest['end']=='2026-10-06','Inherited panel metadata horizon')
    # This range is only a manifest claim, not per-cell or security support.
    need(all(manifest['start']<=row['branches']['other_states']['required_spy_union_start'] and row['date']<=manifest['end'] for row in windows),'Calendar windows lie within inherited panel declared range')
    market=[]
    for p in ('outputs/four_factor_rebound/ETF_SOURCES.json','outputs/monthly_reconstruction_v2/ETF_MANIFEST.json','outputs/monthly_reconstruction_v2/foreign/ETF_MANIFEST.json'):
        for item in read(p):
            if item['symbol']=='SPY':
                need(raw.get(item['file'])==item['sha256'],'SPY must belong to original64')
                bind(item['file'],item['sha256'])
                market.append(dict(manifest=p,manifest_sha256=PINS[str(ROOT/p)],file=item['file'],sha256=item['sha256'],field='Adj Close'))
    need(len(market)==1 and market[0]['sha256']=='9fbc17731dfadfb74382d076a281dcc935f1971f74f8d08cfe860726b69edff5','Unique fixed SPY source')
    for folder in ('monthly_published44_reconstruction','monthly_top35_extension','monthly_2016_extension'):
        need(not (ROOT/'outputs'/folder/'prices/SPY.csv').exists(),'Unexpected later SPY override')
    protected=read(PROTECTED);need(len(protected)==21444,'Exact protected manifest member count')
    implementation=bind(Path(__file__))
    direct=dict(raw)
    for p in (CALENDAR,LOCAL/'META.json',loader,ROOT/'outputs/index_union/alias_resolved_inputs/MANIFEST.json'):
        direct[str(p)]=PINS[str(p)]
    for p,h in calendar['provenance']['source_hashes'].items():direct[p]=h
    report=dict(schema='market_drift231_source_metadata_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,source_only=True,metadata_only=True,source_npz_decoded=False,npz_decoded=False,price_rows_read=False,
        source_models_imported=False,source_loader_executed=False,actual_support_computed=False,actual_features_computed=False,
        actual_labels_computed=False,actual_forecasts_computed=False,actual_returns_computed=False,actual_orders_computed=False,
        actual_model_fits=0,actual_new_parameters_computed=False,observers_modified=False,central_modified=False,frozen_files_modified=False,
        source_sha256=dict(sorted(PINS.items())),input_sha256=dict(sorted(PINS.items())),evidence_sha256=evidence,
        direct_feature_source_sha256=dict(sorted(direct.items())),raw_source_hashes=raw,raw_source_count=64,
        context_sha256=CONTEXT_SHA,proposal_sha256=PROPOSAL_SHA,clarifications_sha256=CLARIFICATIONS_SHA,
        source_bytes_reverified=True,raw_source_map_matches_original=True,metadata_axes_verified=True,calendar_clock_verified=True,
        core_calm_source_definition_reviewed=True,single_episode_budget_source_reviewed=True,price_unit_source_fingerprints_bound=True,
        actual_source_coverage='UNKNOWN until registered preparation',strict_pit=False,host_not_pinned_to_avoid_cycle=True,
        mutable231_model_and_oracle_not_pinned=True,
        predecessor230=dict(plan_sha256=PRIOR_PLAN,completion_sha256=PRIOR_CLOSE,saved_review_sha256=PRIOR_REVIEW,
            result_completion_sha256=closed['result_completion_sha256'],closed=True),
        provenance_scope=dict(inherited_predecessor_input_count=len(inherited_pins),inherited_input_role='Immutable historical provenance only; source169/206/227/229 forecasts and labels are not new drift feature inputs',
            new_feature_inputs='Original adjusted-close panel, exact current monthly IDs/eligibility/calm flag, official sessions',
            incumbent_cached_model_reused=True,new_feature_model_or_forward_labels_needed=False),
        source_identity=dict(query_months=122,query_width=49,query_rows=5978,original_cached_array_count=27,
            cached_array_count_basis='Prior230 registered/closed/independent receipts; NPZ not decoded here',
            etf_count=17,unique_candidate_symbols=len({s for row in meta['symbols'] for s in row}),first_query=meta['dates'][0],last_query=meta['dates'][-1],last_end=meta['ends'][-1]),
        calendar=dict(path=str(CALENDAR.relative_to(ROOT)),sha256=CALENDAR_SHA,session_count=len(sessions),first_session=sessions[0],last_session=sessions[-1],
            full_calendar_checked=True,window_scope='Both possible210/252 branches per query, irrespective of actual calm state; no prices or feature availability read',windows=windows),
        market_source=dict(symbol='SPY',entries=market,loader=str(loader.relative_to(ROOT)),loader_sha256=PINS[str(loader)],later_override_absent=True,price_rows_read=False),
        price_units=dict(signal_price='Frozen dividend-adjusted close; no new split or currency conversion',execution_parity_required='Original study.verify_panel_parity: identical finite/positive masks, constant per-security scale and adjusted-open ratio, rtol=atol1e-10',
            alias_manifest_status=manifest['status'],unit_helper_is_provenance_only=True,new_unit_conversion_executed=False,new_PIT_or_corporate_action_certification=False,
            early_splice='Frozen loader fills only leading gaps with first-overlap scale; inherited retrospective-vintage limitation'),
        integration=dict(core_preserved=True,calm_core_weight=.5,noncalm_core_weight=.52,calm_return_count=210,other_return_count=252,skip_stock_sessions=42,
            calm_parity_required='Exact cached bool equality with prior63 sample std of SPY simple returns times sqrt252 <0.10; invalid sources never silently zero',
            support='(original_baseline-original_core)/(1-original_core_weight)',budget='min(.05,max(0,.10-.025*calm-.05*weak))',
            fixed_original_episode_policy={k:policy[k] for k in ('budget','ceiling','gap')},
            supported_selector='New full-eligible selector replacing the single episode hook, uses original pre/support/clamp/budget; does not run original boundary afterward',
            fallback_selector='Original incumbent selector, exact initial/cold/full-month fallback',
            extra_score_budget=False,source_score_or_selection_recomputed=False,independent_host_code_and_synthetic_checks_required=True),
        required_prepare_checks=['Read original27 cache arrays only after PLAN and snapshot all input bytes before/after.',
            'Recheck actual original64 loader map and execution/raw adjusted-price unit parity.',
            'Use official calendar positions and require all stock and SPY prices for selected branch; do not forward-fill or remove an eligible security.',
            'Exactly verify cachedcalm10 using current63 SPY sample volatility; keep cached core/support/eligibility unchanged.',
            'Historical SPY state uses d-1 trailing200; query state uses currentt trailing200, equality non-up.',
            'Require42 matched days and at least2 original eligible names, but constant raw values remain supported with zero effect.',
            'Physically truncate all inputs at current query for causal prefix checks; never use outcome rows as feature eligibility.',
            'Use one detached replacement selector per arm; no extra original episode on supported rows.',
            'Original incumbent orders exact; nine independent execution paths and45slices; two zero-cost paths drive selection,5bp sensitivity only.'],
        reviewed_code_sha256=code,implementation=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=implementation),
        protected_manifest=dict(path=str(PROTECTED.relative_to(ROOT)),sha256=sha(PROTECTED),members=21444,
            full_hash_checks_proved_by_prior_closure=True,all_members_rehashed_by_this_metadata_review=False),
        central_snapshot=dict(sha256=hashlib.sha256(central_bytes).hexdigest(),persistent_input=False,completed_strategies=226,trials=1195,last_strategy=PRIOR.name,registered_queue_empty=True),
        limitations=context['limitations']+['Source hashes and declared date ranges do not prove complete prices for every actual231 eligible stock/window; support is UNKNOWN here.',
            'No strict PIT, company-identity, split/restatement or source publication-time certification is added.',
            'The inherited source closure includes older model files for immutable lineage only; this new feature consumes no labels or those historical forecasts.',
            'Price unit functions are pinned as provenance, not executed. Current source/output correctness is still subject to registered preparation and independent checks.'])
    need(CENTRAL.read_bytes()==central_bytes,'Central changed during metadata review')
    with OUTPUT.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,receipt_sha256=sha(OUTPUT),implementation_sha256=implementation,source_pins=len(PINS),raw_sources=64,
        queries=122,branches_per_query=2,actual_support_computed=False,npz_decoded=False)),flush=True)


if __name__=='__main__':review()

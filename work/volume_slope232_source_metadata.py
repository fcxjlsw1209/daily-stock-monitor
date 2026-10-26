"""Round232 source-only metadata review. Hashes raw bytes without decoding them.

No array packages, loader imports, prices, factor arithmetic or execution returns.
The declared admission is for a stored-volume proxy, not certified share units.
"""
from datetime import date, datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'work/volume_slope232_SOURCE_REVIEW.json'
CONTEXT = ROOT/'work/monthly_volume_slope232_context.json'
CONTEXT_SHA = 'd541a8533a6694dd6a74d19a23dda4d97a5a104d7ac85e9beeb733e6df9df6d8'
COVERAGE = ROOT/'work/volume_slope232_COVERAGE.json'
COVERAGE_SHA = '58fa25bde4c040140c10fee0f1931d5e5d5c8760b95e2e15a2da87314adca46a'
PRIOR = ROOT/'outputs/monthly_market_drift231'
PRIOR_PLAN = 'ff0699567e0ebb6781bdb6386ad11df7d734d22e3dfff0584afee3fa8f8afeff'
PRIOR_CLOSE = '7a0060a5db6f9ed784b25e40d16955c1f06129f02617a5da9f6022865bf5fda5'
PRIOR_REVIEW = '92013baa0d874f16ccc9c3cffd2175b4326b450ec252ffc30b1557b207cd226e'
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
    central_bytes=CENTRAL.read_bytes(); state=json.loads(central_bytes)
    need(len(state['completed'])==227 and sum(x['trials'] for x in state['completed'])==1198,'Expected231 closed:227 strategies/1198 trials')
    need(state['completed'][-1]['directory']==PRIOR.name and state['completed'][-1]['completion']['sha256']==PRIOR_CLOSE,'Closed231 central identity')
    need(not state['registered_queue'] and state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Original incumbent and empty queue')
    context=read(CONTEXT,CONTEXT_SHA); coverage=read(COVERAGE,COVERAGE_SHA)
    need(context['round']==coverage['round']==232 and context['name']=='monthly_volume_slope232','Prospective232 identity')
    need(context['arms']==['incumbent_local67','volume_rank_slope','unweighted_slope_control'],'Fixed arms')
    need(context['integration']['dose']==.01 and context['integration']['max_absolute_delta']==.005,'Fixed auxiliary dose')
    need('253 consecutive official-session adjusted closes AND253 reported stored volumes at t-294..t-42' in context['formation']['window'],'Fixed common window')
    need('No dollar-volume' in context['formation']['weights'] and 'exactly one unchanged original daily-episode boundary selector' in context['integration']['formula'],'Stored-volume/one unchanged episode contract')
    for record in (context,coverage):
        for key in ('actual_new_features_computed','actual_new_parameters_computed','actual_new_orders_computed','actual_new_returns_computed'):
            need(record[key] is False,'Prospective computation boundary')
    evidence={p:bind(p,h) for p,h in context['evidence_sha256'].items()}
    for row in coverage['comparisons']:bind(row['path'],row['sha256'])
    for p,h in coverage['supporting_text_sha256'].items():bind(p,h)
    for row in coverage['absent_or_missing_originals']:
        need(resolve(row['path']).exists()==row['exists'],'Coverage source availability changed')
    predecessor=read(PRIOR/'PLAN.json',PRIOR_PLAN)
    closed=read(PRIOR/'COMPLETION.json',PRIOR_CLOSE)
    saved=read(PRIOR/'RESEARCH_REVIEW.json',PRIOR_REVIEW)
    results=read(PRIOR/'results/COMPLETION.json',closed['result_completion_sha256'])
    need(closed['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and closed['plan_sha256']==PRIOR_PLAN and closed['protected_files']==21444 and closed['observers_modified'] is False,'Terminal231 closure')
    need(saved['passed'] is True and saved['plan_sha256']==PRIOR_PLAN and saved['completion_sha256']==PRIOR_CLOSE and saved['completed_entries']==227 and saved['trials']==1198,'Independent231 saved review')
    need(results['status']=='COMPLETE' and results['plan_sha256']==PRIOR_PLAN and results['paths']==9 and results['slices']==45,'231 completed accounting receipt')
    for p,h in predecessor['input_sha256'].items():bind(p,h)
    inherited_count=len(predecessor['input_sha256'])
    local_plan=read(LOCAL/'PLAN.json','0016175aee1df43b734010d80bd543b881c57036a7f8c6ab73428340203afc96')
    original_inputs=read(LOCAL/'INPUT_HASHES.json','5ac2b639fa8c944688787eba47ea1b420bfbe9c6e1e051a3d827d8b2b32b264d')
    raw=local_plan['raw_source_hashes']; need(len(raw)==64 and raw==predecessor['raw_source_hashes'],'Original64 raw map')
    for p,h in raw.items():need(original_inputs.get(p)==h,'Raw missing from original input map');bind(p,h)
    meta=read(LOCAL/'META.json','3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and meta['dates']==sorted(set(meta['dates'])),'Original122 axes')
    need(meta['dates'][0]=='2016-08-17' and meta['dates'][-1]=='2026-08-31' and meta['ends'][-1]=='2026-09-24','Original event endpoints')
    need(len(meta['etfs'])==len(set(meta['etfs']))==17 and all(len(ss)==len(set(ss))==49 for ss in meta['symbols']),'Original49/17 axes')
    need(all(type(s) is str and s and s.strip()==s for ss in meta['symbols'] for s in ss),'Exact original identities')
    need(meta['ends'][:-1]==meta['dates'][1:] and all(e>d for d,e in zip(meta['dates'],meta['ends'])),'Original intervals')
    cache=read(LOCAL/'CACHE_HASHES.json','e3b440f8e2f2669d535cb92d33a2302be7ceec4f54792ff4078a7d976356460b')
    for p,h in cache.items():bind(LOCAL/p,h)
    bind(LOCAL/'local_linear_ORDERS.json','c0e9286cdfae5e586a83ee0c412447a607ca21d1a5d6bbe47d40bd494b9d947e')
    policy=read(LOCAL/'INCUMBENT_POLICY.json')
    need(all(policy[k]==v for k,v in dict(budget=.05,ceiling=5,gap=.02).items()),'Frozen episode policy')
    calendar=read(CALENDAR,CALENDAR_SHA)
    for p,h in calendar['provenance']['source_hashes'].items():bind(p,h)
    sessions=calendar['sessions']
    need(type(sessions) is list and sessions==sorted(set(sessions)) and calendar['start']=='2008-01-01' and calendar['end']=='2035-12-31','Full official143 calendar')
    for value in sessions:day(value)
    positions={d:i for i,d in enumerate(sessions)}; windows=[]
    for d,e in zip(meta['dates'],meta['ends']):
        day(d);day(e);need(d in positions and e in positions,'Official event sessions')
        t=positions[d];need(t>=294 and t+1<len(sessions),'Calendar covers complete formation')
        if d!='2016-08-17':need(d[:7]!=sessions[t+1][:7],'Formal monthend')
        dates=sessions[t-294:t-41]
        need(len(dates)==253 and dates[-1]==sessions[t-42] and dates[0]==sessions[t-294],'Exact253 formation metadata')
        windows.append(dict(date=d,end=e,start=dates[0],last=dates[-1],price_count=253,volume_count=253,
            price_and_volume_calendar_identical=True,first_offset=-294,last_offset=-42,return_span=252,
            initial_partial_delta_zero=d=='2016-08-17',source_values_read=False,actual_common_support='UNKNOWN'))
    code={}
    definitions={
      'outputs/monthly_local_linear70/study.py':('e8cfe0c27a6805cbee4b59d24ce9771ece7caa01c29f1a30aa2d5acf9837f3fb',(
        'def load():','def modified_pre_score(spec,a,n):','def verify_panel_parity(panel,px,op):','expected=px*opening/close','len(hashes)==64')),
      'outputs/monthly_stability_monitor/inputs.py':('dd97fb4a9f7b2102d2c7f497825526a484628fa86ee9891312bfb5abad7c8ded',(
        "('volume','Volume')","panel[k][entry['symbol']]=z[col].reindex(panel[k].index)","panel['adj'].loc[gap,s]=z.loc[gap,'Adj Close']*ratio","for s,cache in [('CHL','CHL'),('PTR','PTRCY')]")),
      'outputs/monthly_state_rank_research/state_rank.py':('53866e5ffcb06146f73bb409747df4bb6f5b7864c1bfe0edebb639d551aa4c6d',('def panel_load(etfs):',"info['source_hashes'][str(file)]=sha(file)")),
      'outputs/monthly_stability_research/local_refine.py':('5207909c5e777b1918676942a130e8be62a7a90a6bea1469938cae9aedcab949',('calm10=mv.loc[dates].to_numpy()<.10',)),
      'outputs/monthly_stability_research/stability.py':('2d22b1bcef8925c8f98d6778142968f2ee3e2f0847c46caf2aea8c6de2b72ad4',('mom = {n: px.shift(42) / px.shift(n) - 1 for n in [252, 294]}',)),
      'outputs/monthly_quality_research/explore.py':('5af173a8d6a659153278eedc4e38bc2c95f18dee5b10ac4d67cf44bdeb1ddfde',('cw=np.where(calm,.5,.52)',)),
      'outputs/monthly_path70_research/incumbent_adapter.py':('5619bfc3512a0aa057e79042256a379360d6baa1bc0c448f2f301b84d6ec3e2b',('support = (baseline-core)/(1-cw)','episode_budget = np.minimum(.05, np.maximum(0., OLD_LIMIT-old_budget))')),
      'outputs/monthly_daily_episodes/episodes.py':('8ae91e55e0d662cb6dee9cba6d1422aa3713b779976de69b4560194d311ae16a',("target = np.clip(support[i, j] + effects[i, j], -.5, .5)",'score[i, j] += budget[i] * (target - support[i, j])')),
      'outputs/index_union/price_units.py':('353e2f039d022a03cf90be06028d680b8dd389fad7bb216727cc2ecbda226291',("out['volume']=pd.to_numeric(frame['volume'],errors='coerce')/multiplier","signal['volume']=pd.to_numeric(frame['volume'],errors='coerce')*multiplier"))}
    for p,(digest,tokens) in definitions.items():source_text(p,digest,tokens);code[p]=digest
    explanatory={
      'outputs/index_union/assemble_input_diagnostics.py':("d=pd.read_csv(P/r['signal_file']", "('volume','volume')"),
      'outputs/index_union/assemble_alias_resolved_inputs.py':('alias_resolution_diagnostic_not_admitted',),
      'outputs/index_union/normalize_new_prices.py':('yahoo_split_adjusted_ohlcv_dividend_adjusted_close','tiingo_raw_ohlcv_and_separate_adjusted_close'),
      'outputs/index_union/build_execution_candidates.py':('identity_verified=False,action_completeness_verified=False',),
      'outputs/index_union/build_raw_source_candidates.py':('identity_verified=False,action_completeness_verified=False','admission=False'),
      'outputs/index_union/consolidate_security_aliases.py':('for field in PRICE_FIELDS:out[field][identity]=data[field][provider]',),
      'outputs/four_factor_rebound/prepare.py':('auto_adjust=False,actions=True',),
      'outputs/monthly_2016_extension/collect.py':('auto_adjust=False,actions=True',)}
    for p,tokens in explanatory.items():
        digest=bind(p);source_text(p,digest,tokens);code[p]=digest
    loader=ROOT/'outputs/monthly_stability_monitor/inputs.py'
    need(raw.get(str(loader))==code[str(loader.relative_to(ROOT))],'Original loader identity')
    manifest=read('outputs/index_union/alias_resolved_inputs/MANIFEST.json','b48b3e337c9789bd6effbdb5ee533fddfbcfe124dfb2a07d330b9f1da43ee5ce')
    need(manifest['start']=='2010-01-04' and manifest['end']=='2026-10-06','Base source declared horizon')
    need(all(manifest['start']<=x['start']<=x['last']<=manifest['end'] for x in windows),'Formation dates within declared base range, not per-cell support')
    etf_entries=[]
    for p in ('outputs/four_factor_rebound/ETF_SOURCES.json','outputs/monthly_reconstruction_v2/ETF_MANIFEST.json','outputs/monthly_reconstruction_v2/foreign/ETF_MANIFEST.json'):
        for item in read(p):
            need(raw.get(item['file'])==item['sha256'],'Manifest source must match original64')
            if item['symbol'] in meta['etfs']:
                etf_entries.append(dict(symbol=item['symbol'],manifest=p,manifest_sha256=PINS[str(ROOT/p)],
                    file=item['file'],sha256=item['sha256'],price_field='Adj Close',volume_field='Volume',
                    observed_rows_read=False,complete_window_support='UNKNOWN',volume_units_certified=False))
    need(set(x['symbol'] for x in etf_entries)==set(meta['etfs']) and len(etf_entries)==17,'All17ETF source declarations only')
    protected=read(PROTECTED);need(len(protected)==21444,'Protected manifest exact21444')
    implementation=bind(Path(__file__))
    direct=dict(raw)
    for p in (CALENDAR,LOCAL/'META.json',loader):direct[str(p)]=PINS[str(p)]
    for p,h in calendar['provenance']['source_hashes'].items():direct[p]=h
    report=dict(schema='volume_slope232_source_metadata_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,source_only=True,metadata_only=True,conditional_proxy_admission=True,volume_units_certified=False,strict_pit=False,
        source_npz_decoded=False,npz_decoded=False,price_rows_read=False,volume_rows_read=False,source_models_imported=False,
        source_loader_executed=False,actual_support_computed=False,actual_source_coverage_recomputed=False,
        actual_features_computed=False,actual_new_features_computed=False,actual_labels_computed=False,
        actual_forecasts_computed=False,actual_returns_computed=False,actual_orders_computed=False,
        actual_model_fits=0,actual_new_parameters_computed=False,observers_modified=False,central_modified=False,frozen_files_modified=False,
        source_sha256=dict(sorted(PINS.items())),input_sha256=dict(sorted(PINS.items())),evidence_sha256=evidence,
        direct_feature_source_sha256=dict(sorted(direct.items())),raw_source_hashes=raw,raw_source_count=64,
        context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,source_bytes_reverified=True,raw_source_map_matches_original=True,
        metadata_axes_verified=True,calendar_clock_verified=True,core_calm_source_definition_reviewed=True,
        single_episode_budget_source_reviewed=True,price_unit_source_fingerprints_bound=True,volume_source_semantics_reviewed=True,
        actual_source_coverage='UNKNOWN until registered preparation',host_not_pinned_to_avoid_cycle=True,
        mutable232_model_and_oracle_not_pinned=True,
        predecessor231=dict(plan_sha256=PRIOR_PLAN,completion_sha256=PRIOR_CLOSE,saved_review_sha256=PRIOR_REVIEW,
            result_completion_sha256=closed['result_completion_sha256'],closed=True),
        provenance_scope=dict(inherited_predecessor_input_count=inherited_count,
            inherited_input_role='Immutable historical lineage only; old labels and forecasts are not232 auxiliary inputs',
            new_feature_inputs='Original adj and stored volume fields, exact current monthly IDs/eligibility, official sessions',
            incumbent_cached_model_reused=True,new_feature_model_or_forward_labels_needed=False),
        source_identity=dict(query_months=122,query_width=49,query_rows=5978,original_cached_array_count=27,
            cached_array_count_basis='Prior231 closure and original cache manifests, no NPZ decoding',etf_count=17,
            unique_candidate_symbols=len({s for ss in meta['symbols'] for s in ss}),first_query=meta['dates'][0],
            last_query=meta['dates'][-1],last_end=meta['ends'][-1]),
        calendar=dict(path=str(CALENDAR.relative_to(ROOT)),sha256=CALENDAR_SHA,session_count=len(sessions),
            first_session=sessions[0],last_session=sessions[-1],full_calendar_checked=True,
            window_scope='Same253 official sessions for price and stored volume, every query; no numerical support assessed',windows=windows),
        etf_source_declarations=etf_entries,
        volume_semantics=dict(admission='Conditional stored-volume proxy only; not certified shares, flow, dollar turnover or strictPIT',
            main_branch='alias_resolved main panel reads signal_file OHLCV; documented split-only normalization is conditional on source action completeness',
            yahoo_overlay='Three manifests and additional CSVs copy vendor Volume directly with vendor OHLC/Adj Close; no new unit conversion',
            early_splice='Leading gaps rescale adjusted prices at first overlap while copying source OHLC andVolume without the adjusted-price ratio',
            tiingo_override='CHL/PTR override with raw Tiingo volume and distinct adjClose before2021',
            alias_rule='Whole-provider fields copied together; alias overlap checks raw price, not volume',
            split_rule='Signal price=raw/multiplier; signal volume=raw_volume*multiplier. Reverse conversion reciprocal. Functions are provenance-only here.',
            constant_positive_scale_preserves_within_window_rank=True,in_window_unit_changes_certified=False,
            action_completeness_certified=False,source_publication_clock_certified=False,
            all17_etfs_have_declared_volume_field=True,complete_numeric_windows_known=False,
            raw_dollar_volume_computed=False,adjusted_price_times_volume_used=False,
            existing_execution_parity_certifies_volume=False,base_manifest_status=manifest['status']),
        integration=dict(core_preserved=True,calm_core_weight=.5,noncalm_core_weight=.52,original_calm_return_count=210,
            original_other_return_count=252,new_auxiliary_return_span=252,skip_stock_sessions=42,dose=.01,max_absolute_delta=.005,
            location='config4 modified_pre_score: add only nonzero .01 centered-rank deltas, then one unchanged original episode',
            original_episode_policy={k:policy[k] for k in ('budget','ceiling','gap')},
            original_effects_baseline_core_support_budget_unchanged=True,initial_partial_zero_delta=True,
            common_missing_cells_neutral_without_pool_removal=True,source_score_or_selection_recomputed=False),
        required_prepare_checks=['After PLAN, snapshot all27 original arrays and verify original64 loader map plus execution-price parity.',
            'Use exact253 official-session adj andVolume observations, finite positive nonboolean numeric sources, originaleligible cells only.',
            'No filling, no source substitution, no source-unit inference; incomplete window gives both arms neutral correction and preserves pool.',
            'Independent within-window exact-tie volume ranks, bounded weights, weighted and matched unweighted slopes; reject nonfinite derived arithmetic.',
            'Cross-section ranks only common supported eligible cells; initialpartial/under2/equal scalars neutral; constant volume exactcandidate-control copy.',
            'Both arms add fixed .01 rank to original config4 pre-score and call original episode exactly once.',
            'All123 decision prefixes and5 physically truncated source prefixes; no future source/window access.',
            'Incumbent orders exact;9paths45slices independent accounting;zero-cost first88 selection frozen before later34 diagnostic.'],
        reviewed_code_sha256=code,implementation=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=implementation),
        protected_manifest=dict(path=str(PROTECTED.relative_to(ROOT)),sha256=sha(PROTECTED),members=21444,
            full_hash_checks_proved_by_prior_closure=True,all_members_rehashed_by_this_metadata_review=False),
        central_snapshot=dict(sha256=hashlib.sha256(central_bytes).hexdigest(),persistent_input=False,
            completed_strategies=227,trials=1198,last_strategy=PRIOR.name,registered_queue_empty=True),
        limitations=context['limitations']+['Metadata date coverage and field declarations do not establish any actual232 common source support.',
            'No simple source-certified share-basis mask was found; even normalizedsignal source candidates retain action_completeness_verified=false.',
            'Coverage compares earlier price-level and other trend/volume families; final232 log-price/rank-weighted formulation is fixed bycontext, not certified universally novel.',
            '64source identities are rehashed but price/volume values and NPZ arrays are never decoded by this review.'])
    need(CENTRAL.read_bytes()==central_bytes,'Central changed during metadata review')
    with OUTPUT.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,receipt_sha256=sha(OUTPUT),implementation_sha256=implementation,
        source_pins=len(PINS),raw_sources=64,queries=122,formation_observations_per_source=253,etfs=17,
        conditional_proxy_admission=True,volume_units_certified=False,actual_support_computed=False,npz_decoded=False)),flush=True)


if __name__=='__main__':review()

"""Write two prospective252 metadata contracts; no numerical research inputs.

Use run_local.sh. Read only metadata/code and hash raw/cache bytes; no source
loader, array decoder, feature calculation, strategy imports or returns.
"""
from copy import deepcopy
from datetime import date,datetime,timezone
from fractions import Fraction
import hashlib,json,platform
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'work/monthly_peer_path252_SOURCE_CONTRACT.json'
CONTEXT=ROOT/'work/monthly_peer_path252_CONTEXT.json'
SCHEMA='monthly_peer_path252_metadata_source_contract_v1'
ARMS=('incumbent_local67','daily_peer_path_relative','endpoint_peer_relative_control')
CLOCK_KEYS=('query_index','date','initial_partial','numeric_evaluation_required','symbols','query_position','next_official_session','price_start','price_end','source_start_position','source_end_position','price_count','return_count','stock_indices')
MAPPING_KEYS=('query_index','column_index','date','symbol','issuer_key','original_sector','sector','mapping_origin','mapping_repair','mapping_evidence')
IDENTITY_KEYS=('query_index','column_index','date','symbol','issuer_key','not_before','available_from','bound_visible','price_start','price_end','price_window_dates_before_bound')
EXPECTED={
 'work/after251_peer_path252_DRAFT_01.json':'421501f86333141ebc81f99637e259834cccd2e65b870243f9decc8ebd5eddd2',
 'work/after251_peer_path252_DRAFT_02.json':'4e6becc7e25e0b4771bdf53a605d870ef4ad54ee10508bbdc0777658c994cd94',
 'work/after251_peer_draft252_REVIEW_v2.json':'0272e7777f9674b028b697df262af243ae33e8501af8a3ec4002d2c2633b45be',
 'work/after251_peer_path_source_map.json':'e15fa2a81df431e4eb3da19c76520a9f7487c3f9b54fe20e8999f54bbbb45311',
 'work/peer239_maturity_policy.json':'3bbcbcbe141fb0d2d96cd64ca74f2ee1e9450e12c1f2594c7cb84fb98bf64862',
 'work/ownsector238_repair_source_policy.json':'bb743a4ca456c43219436f5e39f0387b3b721e65d2344012de699ba79d6ec7de',
 'work/peer239_listing_gap_evidence.json':'3a02401016eb1254c35ef46ee6dbf2200e2b24a3aa9b5a15ea1804dd071e6a30',
 'outputs/monthly_peer239_maturity_audit/SOURCE_PLAN.json':'8867b2029400afdd96a3f34a940d467849903b3acc6cce0997defcc183366dee',
 'outputs/monthly_peer239_maturity_audit/REPORT.json':'5138025ced5531b6d3447cc9bb9835e886b3f45157b6b4e03a54e888fcc97f7c',
 'outputs/monthly_peer239_maturity_audit/INDEPENDENT_REVIEW.json':'4fefe123e2c2856e93dd3670a68f89170f15652c51f5f423d1558dcf6366f10e',
 'outputs/monthly_peer239_maturity_audit/COMPLETION.json':'fdeaf5afdd7a2d2823d0ee66bcc00a13b4d767c88b1eded2f9b575f1f17b2dc8',
 'outputs/monthly_local_linear70/PLAN.json':'0016175aee1df43b734010d80bd543b881c57036a7f8c6ab73428340203afc96',
 'outputs/monthly_local_linear70/META.json':'3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
 'outputs/monthly_paired_target143/CALENDAR.json':'8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
 'work/calendar_clock248_ORIGIN_DIAGNOSTIC.json':'d250fc4d42b6089c62f4b1c4a032b872db7a1dd3cc99f9c133b8c8421c8c0eaf',
 'work/basket_core251_SOURCE_CONTRACT.json':'16e78872660c0ee5dad15854333f6cf0d2fad8a1be179463f10cf18ae81f304c',
 'work/monthly_calendar_clock248_v2_independent.py':'c7d172a9abec73f81d1d07ce8a422cd709a4a9bfebd55e6211b471c8d315663c',
 'outputs/monthly_path70_research/incumbent_adapter.py':'5619bfc3512a0aa057e79042256a379360d6baa1bc0c448f2f301b84d6ec3e2b',
 'outputs/monthly_daily_episodes/episodes.py':'8ae91e55e0d662cb6dee9cba6d1422aa3713b779976de69b4560194d311ae16a',
 'outputs/monthly_model_scope70/study.py':'b13619cf253a0d1cab7079b9da88b8df464e9a322836380ae0a039037ba9f711',
 'outputs/monthly_local_linear70/study.py':'e8cfe0c27a6805cbee4b59d24ce9771ece7caa01c29f1a30aa2d5acf9837f3fb',
 'work/monthly_aux_rank204_kernel.py':'c44dd6b24ce060513a92890148675d7e091a5e0d91061da9d64637ecd160ac65',
 'outputs/monthly_basket_core251/PLAN.json':'c6b96bf7f1af7e2f4036e92157841abfe326ceadb6bbf7a658cf2305b18c5fbd',
 'outputs/monthly_basket_core251/COMPLETION.json':'c8a204d27c99f93ecf0c104673a7a980c10d26ed7d2f19f16bbef96826f4d81b',
 'work/monthly_basket_core251_RESULT_REVIEW.json':'770b1d61de2a199e43e5e79b70442fe6dc41d264be96fe54b88f43307a7152f3',
 'work/after251_strategy_checkpoint.json':'5772d6ec198631319d0faa2972cab4cdc8ce43d1fbf7ea5a0777179efb2db89a'}

def need(ok,label):
    if not ok:raise ValueError(label)

def path(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    need(p.is_relative_to(ROOT) and 'CloudStorage' not in p.parts,'Local metadata/byte hashes only')
    need(p.name!='CONTINUATION_STATUS.json','Mutable central not an immutable input')
    return p

def sha(value):
    h=hashlib.sha256()
    with path(value).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def read(value):
    p=path(value);need(p.suffix=='.json','JSON metadata only');return json.loads(p.read_text())

def bind(pins,value,want=None):
    p=path(value);key=str(p.relative_to(ROOT));got=sha(p)
    need(want is None or got==want,'Digest mismatch '+key)
    need(key not in pins or pins[key]==got,'Conflicting pin '+key);pins[key]=got;return got

def merge(pins,mapping):
    for p,h in mapping.items():bind(pins,p,h)

def write(p,value):
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    need(platform.python_version()=='3.9.6','Pinned offline wrapper required')
    need(not SOURCE.exists() and not CONTEXT.exists(),'Both contracts write-once')
    pins={};merge(pins,EXPECTED);bind(pins,Path(__file__))
    draft=read('work/after251_peer_path252_DRAFT_02.json');paper=read('work/after251_peer_draft252_REVIEW_v2.json');smap=read('work/after251_peer_path_source_map.json')
    need(draft['arms']==list(ARMS) and paper['blocking_mathematical_findings']==[] and paper['status']=='PAPER_REVIEW_PASS_FOR_FINAL_CONTRACT_NOT_REGISTERED','Prospective paper review')
    need(smap['passed'] is True and smap['source_admitted'] is False and smap['metadata_only'] is True,'Metadata source map only')
    merge(pins,smap['input_sha256']);merge(pins,paper['input_sha256'])
    original=read('outputs/monthly_local_linear70/PLAN.json')
    raw={str(path(p).relative_to(ROOT)):h for p,h in original['raw_source_hashes'].items()}
    need(len(raw)==64 and raw==smap['raw_source_contract']['raw_source_hashes'],'Original64 maps identical')
    merge(pins,raw)
    for group in ('hashes','source_hashes'):merge(pins,original[group])
    cache=read('outputs/monthly_local_linear70/CACHE_HASHES.json');bind(pins,'outputs/monthly_local_linear70/CACHE_HASHES.json')
    for name,h in cache.items():need(Path(name).name==name,'Cache basename');bind(pins,ROOT/'outputs/monthly_local_linear70'/name,h)
    for p in ('outputs/monthly_local_linear70/INCUMBENT_POLICY.json','outputs/monthly_local_linear70/local_linear_ORDERS.json','work/monthly_peer239_host.py','work/monthly_ownsector_gate238_host.py','work/ptr169_results_independent.py','work/ptr169_rebuild_evaluate.py','outputs/monthly_factor_search_v2/validate.py'):
        bind(pins,p)
    sourceplan=read('outputs/monthly_peer239_maturity_audit/SOURCE_PLAN.json');oldreport=read('outputs/monthly_peer239_maturity_audit/REPORT.json');done=read('outputs/monthly_peer239_maturity_audit/COMPLETION.json')
    external_runtime=[]
    for legacy_path,expected in sourceplan['input_sha256'].items():
        candidate=Path(legacy_path)
        if candidate.is_absolute() and not candidate.is_relative_to(ROOT):
            need(str(candidate)=='/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9' and expected=='6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a','Only inherited fixed external Python runtime allowed')
            got=hashlib.sha256(candidate.read_bytes()).hexdigest();need(got==expected,'Inherited external runtime exactSHA')
            external_runtime.append(dict(path=str(candidate),sha256=expected,verified=True,scope='Inherited239 runtime pin retained inside the hash-bound SOURCE_PLAN; not a repository-relative input. Host still verifies complete oldsource closure.'))
        else:bind(pins,legacy_path,expected)
    need(oldreport['schema']=='peer239_mature_source_report_v1' and oldreport['admitted'] is True and oldreport['gaps']==[] and oldreport['source_only'] is True,'Old endpoint report closure')
    need(done['status']=='COMPLETED_MATURE_SOURCE_AUDIT_INDEPENDENTLY_VERIFIED' and done['report_sha256']==EXPECTED['outputs/monthly_peer239_maturity_audit/REPORT.json'] and done['plan_sha256']==EXPECTED['outputs/monthly_peer239_maturity_audit/SOURCE_PLAN.json'],'Old source completion binding')
    need(oldreport['counts']==done['counts'] and oldreport['counts']['stock_cells']==3904,'Old complete inventory counts, not daily admission')
    policy=read('work/peer239_maturity_policy.json');repair=read('work/ownsector238_repair_source_policy.json');merge(pins,repair['source_input_sha256'])
    meta=read('outputs/monthly_local_linear70/META.json');calendar=read('outputs/monthly_paired_target143/CALENDAR.json')['sessions'];positions={d:i for i,d in enumerate(calendar)}
    need(calendar==sorted(set(calendar)) and all(date.fromisoformat(d).isoformat()==d for d in calendar),'Official canonical dates')
    need(len(meta['dates'])==len(meta['symbols'])==len(meta['ends'])==122 and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24' and meta['ends'][:-1]==meta['dates'][1:],'Fixed122 interval')
    need(len(meta['etfs'])==len(set(meta['etfs']))==17,'Explicit ETF17')
    records=[];clocks=[];bounds=deepcopy(policy['securities']);bybound={b['symbol']:b for b in bounds};identities=[]
    need(len(bounds)==2 and set(bybound)=={'SPCX','SKHY'},'Original two admitted visible bounds')
    allrecords=oldreport['records'];need(len(allrecords)==3904,'Full32 inventory records')
    for q,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        need(len(names)==len(set(names))==49 and set(meta['etfs']).issubset(names),'Original49 query inventory')
        stock=[j for j,s in enumerate(names) if s not in meta['etfs']];need(len(stock)==32,'Full32 direct stocks')
        t=positions[day];partial=day=='2016-08-17';need(t>=252 and t+1<len(calendar) and (not partial or q==0),'Official211 clock')
        need(partial or calendar[t+1][:7]!=day[:7],'Official month-end witness')
        dates=calendar[t-252:t-41];need(len(dates)==211,'Exact211 official sessions')
        c=dict(query_index=q,date=day,initial_partial=partial,numeric_evaluation_required=not partial,symbols=deepcopy(names),query_position=t,next_official_session=calendar[t+1],price_start=dates[0],price_end=dates[-1],source_start_position=t-252,source_end_position=t-42,price_count=211,return_count=210,stock_indices=stock)
        need(tuple(c)==CLOCK_KEYS,'Clock schema');clocks.append(c)
        rr=[r for r in allrecords if r['query_index']==q]
        need([r['column_index'] for r in rr]==stock,'Exact old fullstock record domain')
        issuers=[]
        for r in rr:
            j=r['column_index'];need(r['symbol']==names[j] and r['date']==day and isinstance(r['issuer_key'],str) and r['issuer_key'],'Whole-issuer identity')
            issuers.append(r['issuer_key']);records.append({k:deepcopy(r[k]) for k in MAPPING_KEYS})
            if r['symbol'] in bybound:
                b=bybound[r['symbol']];need(r['issuer_key']==b['issuer_key'],'Reviewed listing issuer match')
                x=dict(query_index=q,column_index=j,date=day,symbol=r['symbol'],issuer_key=r['issuer_key'],not_before=b['not_before'],available_from=b['available_from'],bound_visible=b['available_from']<=day,price_start=dates[0],price_end=dates[-1],price_window_dates_before_bound=sum(d<b['not_before'] for d in dates))
                need(tuple(x)==IDENTITY_KEYS,'Identity schema');identities.append(x)
        need(len(issuers)==len(set(issuers)),'No arbitrary same-issuer representative')
    need(len(records)==3904 and len(identities)==5,'Complete metadata inventory')
    origin=read('work/calendar_clock248_ORIGIN_DIAGNOSTIC.json');need(origin['actual_origin']=='2008-01-02' and origin['raw_sources_exact'] is True,'Actual original axis evidence')
    source_policy=dict(
      original_control='Independent original full49/fullpast pandas core reconstruction with origin2008-01-02 and original current63 risk path; last295 official rows and query/required columns must match. C_source_atol1e-12/rtol0 only. No new positive-price restriction on old U0 and no unrelated older-window listing gate. This gate precedes new daily source and is separate from the211 identity scope.',
      domain='Every formal direct stock in original32, before originalE, peer support, ranking or selection. ETF17 has no new source/statistic target but remains in original strategy. No deletion or substitute security.',
      clock='Exactly official t-252..t-42 inclusive,211 prices/210 daily ratios. No compressed rows, fill, interpolation, adjusted alternate provider or date snapping.',
      source_types='Consumed present scalars must be real nonbool, convertible to stored binary64, finite and strictly positive. Reject bool/nonreal/Inf/nonpositive/overflowing cast/nonzero-tozero cast. Missing scalar means NaN/NA/None; absent official row or symbol is also missing, never filled.',
      mature='No visible same-issuer bound with start<not_before means mature: every211 scalar must be valid; any missing or invalid is hard failure independent ofE or minpeer support.',
      warmup='Only exact symbol+whole issuer with available_from<=query and price_start<not_before permits official_pre_inception. Scan entire211 dates, not just endpoints. Beforebound: missing may be explicitly deferred, any finitepositive quote is a hard identity contradiction; any other present malformed/nonpositive/nonfinite value also hardfails. At/afterbound every price must be present valid positivefinite. Equality atnot_before is post-bound.',
      mapping='Keep complete pinned239 ten-field mapping projection. Only visible structuralwarmup can defer absentsector. Mature missing/invalidsector, missingissuer or any duplicate issuer is hardfailure. Repaired business proxies are assumptions, not official historicalGICS certification.',
      cohort='Reconstruct mature directstock same-sector peers excluding wholeissuer, fixed atcurrentquery for210days, beforeE; require>=2 distinct other issuers for both targets. Structurally unsupported mature stocks still require full211 source validation.',
      partial='Named2016-08-17 retains metadata/identity/mapping checks but reads no NEW211 numeric values and allarms copy exact incumbent. Original inherited control may read its own source.',
      no_fallback='Mature or post-bound sourcefailure aborts. Structural support insufficiency or N=0 is declared targetabsence, not source failure; no whole-event sourcefallback, tickerdrop or targetfill.',
      prefixes='Consumed q<n only. Future record traversal checks canonical nonnegative integer query_index then ignores futurecontents. Fullsource_report can be supplied to logicalprefixes; physicalsource/cache/meta/price/calendar prefixes must match. n0 retains the original positive symbol width and consumes no new source values.',
      old_certificate='239 observed_full_window refers to endpoints; its availability/support counts must never certify211 coverage. Full239 report and original/deferred gaps stay frozen and reviewable.')
    source=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),status='METADATA_CONTRACT_READY_DAILY_ADMISSION_PENDING',passed=True,source_only=True,metadata_only=True,source_admitted=False,registered_strategy=False,strict_pit=False,
      actual_features_computed=False,actual_returns_computed=False,actual_orders_computed=False,prices_loaded=False,volumes_loaded=False,npz_decoded=False,full_daily_source_coverage='UNKNOWN',
      raw_source_count=64,raw_source_hashes=raw,raw_source_bytes_rehashed=True,raw_source_values_decoded=False,external_runtime_bindings=external_runtime,
      panel_first_session='2008-01-02',panel_origin_evidence=dict(path='work/calendar_clock248_ORIGIN_DIAGNOSTIC.json',sha256=EXPECTED['work/calendar_clock248_ORIGIN_DIAGNOSTIC.json']),
      manifest_reported_start='2010-01-04',manifest_reported_start_is_panel_origin=False,
      calendar_path='outputs/monthly_paired_target143/CALENDAR.json',calendar_sha256=EXPECTED['outputs/monthly_paired_target143/CALENDAR.json'],metadata_path='outputs/monthly_local_linear70/META.json',metadata_sha256=EXPECTED['outputs/monthly_local_linear70/META.json'],
      events=122,width=49,formal_monthends=121,direct_stocks_per_query=32,etfs=deepcopy(meta['etfs']),price_sessions=211,return_intervals=210,price_fields=['adj'],volume_used=False,sector_labels_used=True,sector_ETF_prices_used=False,
      clock_event_keys=list(CLOCK_KEYS),clock_events=clocks,mapping_record_keys=list(MAPPING_KEYS),mapping_records=records,listing_bounds=bounds,identity_cell_keys=list(IDENTITY_KEYS),identity_cells=identities,
      source239=dict(path='outputs/monthly_peer239_maturity_audit/REPORT.json',sha256=EXPECTED['outputs/monthly_peer239_maturity_audit/REPORT.json'],schema='peer239_mature_source_report_v1',purpose='Identity,query-date sector mapping and visible history metadata only; old endpoints/peer/support are NOT new daily admission.'),
      source_policy=source_policy,actual_original_core_parity='UNKNOWN_UNTIL_REGISTERED_PREPARE',actual_mature211_coverage='UNKNOWN_UNTIL_REGISTERED_PREPARE',
      transition_policy=dict(inherits_volume_rules=False,VOX_price_used=False,reference_ETF_source_required=False,unknown_continuity=True,empty_known_break_list_is_not_certification=True),
      limitations=['Current-query sector cohort and repaired business proxies are retrospectively assembled; strictPITfalse.','Known visible listing bounds cover onlySPCX/SKHY; other historical identity/unit/provider continuity remains inherited and incompletely certified.','Original64 byte equality is provenance, not economic or211source validity.','Old239 endpoint/source status and peer counts are not reused as daily coverage.'],input_sha256=dict(pins))
    # Budget paper check uses exact nominal constants, never market values.
    budgets=[]
    for calm,weak,A in ((False,False,False),(False,True,False),(True,False,False),(True,False,True),(True,True,False)):
        cw=Fraction(1,2) if calm else Fraction(13,25);res=Fraction(1,40)*calm;ml=Fraction(1,20)*weak;ext=Fraction(1,100)*A
        e=min(Fraction(1,20),max(Fraction(0),Fraction(1,10)-res-ml));remain=1-cw-res-ml-ext-e-Fraction(1,20)
        need(remain>=Fraction(33,100),'Nominal .33 residual support lower bound')
        budgets.append(dict(calm=calm,weak=weak,A=A,remaining_numerator=str(remain.numerator),remaining_denominator=str(remain.denominator)))
    prior=ROOT/'outputs/monthly_basket_core251';prior_done=read(prior/'COMPLETION.json')
    need(prior_done['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior_done['plan_sha256']==EXPECTED['outputs/monthly_basket_core251/PLAN.json'] and prior_done['result_review_sha256']==EXPECTED['work/monthly_basket_core251_RESULT_REVIEW.json'],'Predecessor independently closed')
    closure={}
    for rel in ('prepared/COMPLETION.json','selected/FROZEN.json','results/COMPLETION.json'):
        closure[rel]=bind(pins,prior/rel)
    need(prior_done['result_completion_sha256']==closure['results/COMPLETION.json'],'Prior results completion close binding')
    # Source and context use one fixed metadata snapshot. No prior outcomes are decoded.
    state=json.loads((ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json').read_text())
    need(len(state['completed'])==247 and sum(x['trials'] for x in state['completed'])==1254 and state['registered_queue']==[] and state['completed'][-1]['directory']==prior.name,'Expected closed predecessor and empty queue')
    context=dict(schema='monthly_peer_path252_context_v1',created_utc=datetime.now(timezone.utc).isoformat(),status='PROSPECTIVE_FINAL_CONTRACT_NOT_REGISTERED',arms=list(ARMS),registered_trials=3,new_model_fits=0,new_predictive_model_fits=0,
      hypothesis='Compare aggregation/reference-weight reset of a synthetic fixed-cohort daily peer median with an endpoint peer median, using identical211-price clocks, supported stock domain and original-support funding.',
      causal_scope='This is NOT an isolated trend-persistence test. With2 peers, daily median gross is daily-rebalanced equal-weight gross; endpoint median gross is initial-equal-weight buy-and-hold endpoint. More peers additionally change median identity/order over time. No reference basket is actually traded.',
      predecessor=prior.name,prior_completed_rounds=247,prior_trials=1254,expected_completed_rounds=248,expected_trials=1257,predecessor_closure=dict(plan_sha256=EXPECTED['outputs/monthly_basket_core251/PLAN.json'],completion_sha256=EXPECTED['outputs/monthly_basket_core251/COMPLETION.json'],result_review_sha256=EXPECTED['work/monthly_basket_core251_RESULT_REVIEW.json'],phase_sha256=closure),
      definition=dict(path='work/after251_peer_path252_DRAFT_02.json',sha256=EXPECTED['work/after251_peer_path252_DRAFT_02.json'],hypothesis_wording_refined_by_final_contract=True),paper_review=dict(path='work/after251_peer_draft252_REVIEW_v2.json',sha256=EXPECTED['work/after251_peer_draft252_REVIEW_v2.json']),
      original_control=dict(configuration='Exact Local67 configs()[4]',arrays_unchanged=27,core_coefficients=[.50,.52],E_unchanged=True,pre_and_effects_and_states_unchanged=True,source_core_atol=1e-12,source_core_rtol=0.,source_origin='2008-01-02',full_history_pandas_rolling_preserved=True,original_controls_exact=True),
      exact_math=dict(clock='211prices t-252..t-42 /210daily intervals',focal_gross='Gi=Fraction(P_i_end)/Fraction(P_i_start)',daily_peer_gross='Product over210 dates of conventional exact median_j(P_j[d]/P_j[d-1]) over fixed same-sector other-issuer mature peers',endpoint_peer_gross='Conventional exact median_j(P_j_end/P_j_start)',candidate_raw='Gi/Gdaily-1',control_raw='Gi/Gendpoint-1',even_median='Exact rational sum of middle two divided by2',price_conversion='Validate stored positivefinite binary64 first, then exact Fraction of each float; do not cast raw rational results tofloat.',rank='Fraction(2*less+equal+1-N,2*N) on common supported stocks, exact ties and onlyfinalboundedfloat conversion',all_tie='1/(2N), not0; do not introduce extra centering',N0='No rank denominator evaluated; allnewtargets absent and both experimentalarms exactincumbent.',zero_raw='Valid ordinary rawvalue, includingQ==0; correction is funded unless delta exactly0.',overflow='Exact rational intermediate growth is not source missing or an excuse to approximate/drop; numeric input cast overflow/nonzero underflow is hardfailure.',reference_is_not_traded=True),
      source=dict(path=str(SOURCE.relative_to(ROOT)),sha256=None,schema=SCHEMA,source_policy=source_policy,price_count=211,return_count=210,all_original_inventory_preserved=True,domain='Formal all32 direct stocks beforeE',full_daily_source_coverage='UNKNOWN',mapping_field_keys=list(MAPPING_KEYS)),
      scoring=dict(S='(original_baselineB-original_weightedcoreC)/(1-cw)',hand_budget=.05,delta='.05*(Q-S) at supported formal cells; exactly nonzero delta only is added to originalLocal67 P',partial='Named2016-08-17 allarms copy original P/F/orders',unsupported='No new target/correction. ETF andstructurally unsupported names retain original score but may still be selected.',episode='Exactly one original ep.select_boundary per arm, includingn0; original B/C/cw/state/E/effects/policy passed unmodified, modifiedpre determines top2 andrank3..5gap.02. No final global rerank.',episode_policy=dict(id=7,family='level',model_id=2,budget=.05,ceiling=5,gap=.02),funding_kind='OriginalS substitution; NOT allauxalpha scaling, not reestimated core, not replacingF-C.',new_core_weights=False),
      budget=dict(A='calm & ~weak & extension_available & forecast_available & original_eligible & finite(rank_log_excess)',existing_residual='.025*calm',existing_weak_ML='.05*weak',existing_extension='.01*A',local_linear='Replaces25% inside existing ML/extension slots, adds no nominal budget.',effective_episode='min(.05,max(0,.10-.025*calm-.05*weak))',request_episode=.05,remaining_support='(1-cw)-(.025calm+.05weak+.01A)-effective_episode-.05*supported_formal',conservative_episode_mask='Reserve e on everycell; actual originalepisode only rank3..5 candidates, never greater.',nominal_minimum='33/100',legal_state_extremes=budgets,validation_tolerance=1e-12,tolerance_scope='Inherited boundedsource/coefficient validation only; no exactFraction/rank/tie/source-identity tolerance.',does_not_claim_realized_auxiliary_amplitude_bound=True,extra_A_gate_preserved=True),
      interfaces=dict(build_source='kernel.build_source(panel,meta,calendar,arrays,source239,limit=None) -> daily_source_report',build_features='kernel.build_features(panel,meta,calendar,arrays,daily_source_report,limit=None) -> values,audit',panel={'adj':'original DataFrame'},source239='Full hash-pinned239 mature report for metadata only; contract ten-key mapping projection is an independent consistency aid.',values=['mature','supported','path_target','endpoint_target'],host_decisions='OriginalP + frozen hand support adapter and singleepisode per arm; originalcontrol gate precedes new features.'),
      phase_order=['Immutable contract/code/syntheticreview andPLAN before anyactual252source/feature.','Independent original full49/fullpast source/core gate beforenew daily source.','Producer build_source scans full32 paths and independent verify_source validates all211 andcomplete cohort; no newgross ratios/ranks/target features yet.','Original Local67 singleepisode and exact frozen originalorders plus independent incumbent gate.','Only after bothsource andinc gates compute exact daily/endpoint references andrank; independent features gate before new funded candidate scores/orders.','Two candidate originalepisodes, fullindependent selection, all123cachedprefixes and5physicalprefixes; original27preservation.','First88 ZERO training only, freeze candidate admission/primary.','Then9paths45slices18traininggroups andexactZEROfee checks; independent savedresultreview; full21444protected audit beforeCASclose;noobservermutation.'],
      selection=dict(training_months=88,early_months=[0,53],recent_months=[53,88],later_months=[88,122],candidate_only=ARMS[1],diagnostic_control=ARMS[2],control_can_be_promoted=False,close_cagr_gain_vs_incumbent=.005,close_cagr_gain_vs_control=0.,next_open_zero_cagr_gain_vs_both=0.,training_sharpe_loss_max=.05,training_max_drawdown_loss_max=.03,early_recent_cagr_loss_max=.025,required_changed_sets_vs_each=dict(minimum=3,at_least_one_in_each_training_subperiod=True),cagr_guard_tolerance=1e-12,tolerance_scope='TrainingCAGR comparison only; Sharpe/DD/early/recent guards strict frozen238 semantics.',all_guards_required=True,primary_if_fail=ARMS[0],later_full_or_cost_may_reselect=False,undefined_training_sharpe='Only None plus explicit zero-return-SD reason is admissible undefined evidence: either candidate or endpoint-control required trainingSharpe undefined makes candidateinadmissible. Incumbent undefined is a baseline-parity hardfailure. All other required metrics must be finite; NaN/Inf or invalid reason hardfails. Finite metric guards remain unchanged.'),
      execution=dict(paths=[dict(id='same_close_zero',execution='close',cost_per_side=0.),dict(id='next_open_zero',execution='open',cost_per_side=0.),dict(id='next_open_cost',execution='open',cost_per_side=.0005)],headline='same_close_zero',cost_diagnostic_only=True,positions=[0,2,3],weights='3equal/2halves/<2cash from originalepisode',intramonth_rebalance=False,terminal_liquidation=False,actual_fixed_adjusted_return_units=True),
      validation=dict(cached_prefixes=list(range(123)),physical_prefixes=[0,1,53,88,122],physical_clock='Price/cache/meta throughlastquery; calendar atmostnextofficialsession witness, n0 no numeric read; retain originalwidth.',exact_source_masks_rationals_ranks_choices=True,raw_rational_serialization='Reduced numerator and positive denominator canonical integer decimal strings.',original_C_atol=1e-12,original_C_rtol=0.,source_gate_before_features=True,incumbent_gate_before_features=True,new_feature_gate_before_candidate_selection=True,training_groups=18,independent_paths=9,independent_slices=45,zero_cost_streams=6,zero_fee_fields_exact=True,financial_stat_rule='Frozen ptr169 near(actual independent,expected saved):1e-7+2e-11*abs(reference), no new tolerance.',new_model_fits=0),
      external_runtime_bindings=external_runtime,source_coverage_before_registration='UNKNOWN',strict_pit=False,reused_sample=True,observer_promotion=False,actual_prices_loaded=False,actual_features_computed=False,actual_orders_computed=False,actual_returns_computed=False,registered=False,
      limitations=deepcopy(source['limitations'])+['Currentquery fixed peer membership is retrospective feature formation, not point-in-time daily tradeable membership.','Daily reference aggregation/reset is the only claimed comparison; not isolated persistence, lead-lag, statisticalindependence or author replication.','Repeatedly researched first88/later34 archive; later34 is not pristineOOS.','Only auxiliary clock endsat t-42; inherited originalP/core/current63 states retain their frozen clocks.','Exact rescaling invariance applies only when stored binary64 per-security paths remain exactly proportional.','Bounded nominal coefficient accounting does not establish bounded realized F-C or an economic benefit.','Same-close zero-cost fills remain idealized; openzero and5bp/side diagnostics are separate.','Global exactnoveltyUNKNOWN because someold definitions unavailable; no guarantee70%.'])
    # Do not add inherited outcome numbers or mutablecoordinator bytes to contracts.
    for p,h in pins.items():need(sha(p)==h,'Inputs changed during metadata build '+p)
    write(SOURCE,source);bind(pins,SOURCE);context['source']['sha256']=pins[str(SOURCE.relative_to(ROOT))];context['input_sha256']=pins
    write(CONTEXT,context)
    print(json.dumps(dict(status='252_PROSPECTIVE_CONTRACTS_READY_NOT_REGISTERED',source_sha256=sha(SOURCE),context_sha256=sha(CONTEXT),source_input_pins=len(source['input_sha256']),context_input_pins=len(pins),clock_events=len(clocks),mapping_records=len(records),identity_cells=len(identities),new_actual_source_coverage='UNKNOWN')),flush=True)

if __name__=='__main__':main()

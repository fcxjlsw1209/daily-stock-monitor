"""Write-once prospective251 context from frozen definitions/source metadata.

Only metadata and byte hashes: no strategy import, prices/NPZ decoding, model,
orders or returns calculation. Existing250 artifacts are hashed for closure,
not decoded as numerical research inputs. Execute through run_local.sh.
"""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json
import platform
import prepare_basket_core251_source as metadata

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/monthly_basket_core251_context.json'
SOURCE=ROOT/'work/basket_core251_SOURCE_CONTRACT.json'
SOURCE_SHA='16e78872660c0ee5dad15854333f6cf0d2fad8a1be179463f10cf18ae81f304c'
DEFINITION='work/after250_basket_core251_definition_v2.json'
DEFINITION_SHA='3fd3acdcff6e5f0ce782b5647ed77b96921a82717ad247c91832a61a7d476a17'
PRIOR=ROOT/'outputs/monthly_reference_gain250'
ARMS=('incumbent_local67','long_momentum_basket_core','long_momentum_diagonal_core')
need,read,sha,bind=metadata.need,metadata.read,metadata.sha,metadata.bind
PRIOR_PINS={
 'outputs/monthly_reference_gain250/PLAN.json':'30ee9c3206ed0f3039e9e1f02115216739bcb4dcb7fb1f1bd4a2c1c8c0e14d0c',
 'outputs/monthly_reference_gain250/prepared/COMPLETION.json':'72d92e0aafe4b1d3d5098cb4bb0a92db51bdf6f69f96f1e5c920408961816754',
 'outputs/monthly_reference_gain250/selected/FROZEN.json':'7b2365e3b8f15af16129f3dcef0d3f767c2e107d4cffe4f65abfc88cb387645f',
 'outputs/monthly_reference_gain250/results/COMPLETION.json':'cf63cd8b2779814c052dcb03b5cb1644f3286fe31784ca684bb361abaaa002d9',
 'outputs/monthly_reference_gain250/COMPLETION.json':'19a1268dba496a0641bce23fa9db6c91f20c59e80d1eaa4b9ad674757a520b2c',
 'work/monthly_reference_gain250_RESULT_REVIEW.json':'a9557b13524a04a9c6e16b3a6e308676de864b40a1a3947ce927e053444e4595',
 'work/monthly_reference_gain250_TESTS.json':'da0af961f2d86dee31c7b7327d5e50d589d524f24e0e4448e2082f69979c9d0e',
 'work/monthly_reference_gain250_CODE_REVIEW.json':'365a43031967be9294279ee971448cbc18248d98f8a7d72567b20ec2610bcb6a',
 'work/monthly_reference_gain250_context.json':'5aa6a7a80455e226167dcb93f0f685f346c4d38e0d4682e67585d0f83b4e317a',
 'work/reference_gain250_SOURCE_CONTRACT.json':'85a277ec7de395790b4d8d82778df762e385de53f651a70b6e20980b5e6d9d19',
 'work/reference_gain250_METADATA_ERRATUM.json':'808c905d4e6c5289cf2ac793de6556dad0ee5cbfab0caf16e15e2b4528f2aa55',
 'work/monthly_calendar_clock248_v2_kernel.py':'b580b2a2389201f48d81cd78ddecb641699fdbd919c79202746665789b5a6bcb',
 'work/monthly_calendar_clock248_v2_independent.py':'c7d172a9abec73f81d1d07ce8a422cd709a4a9bfebd55e6211b471c8d315663c'}

def closure(pins):
    for p,h in PRIOR_PINS.items():bind(pins,p,h)
    plan=read(PRIOR/'PLAN.json');done=read(PRIOR/'COMPLETION.json')
    prepared=read(PRIOR/'prepared/COMPLETION.json');frozen=read(PRIOR/'selected/FROZEN.json');results=read(PRIOR/'results/COMPLETION.json')
    need(done['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and done['protected_files']==21444 and done['new_trials']==3,'Prior complete three-arm closure')
    ph=PRIOR_PINS['outputs/monthly_reference_gain250/PLAN.json']
    need(done['plan_sha256']==prepared['plan_sha256']==frozen['plan_sha256']==results['plan_sha256']==ph,'Prior phase PLAN chain')
    need(plan['context_sha256']==PRIOR_PINS['work/monthly_reference_gain250_context.json'],'Prior context binding')
    need(frozen['prepared_sha256']==PRIOR_PINS['outputs/monthly_reference_gain250/prepared/COMPLETION.json'],'Prior prepared freeze binding')
    need(results['frozen_sha256']==PRIOR_PINS['outputs/monthly_reference_gain250/selected/FROZEN.json'],'Prior diagnostic freeze binding')
    need(done['result_completion_sha256']==PRIOR_PINS['outputs/monthly_reference_gain250/results/COMPLETION.json'],'Prior outcome close binding')
    need(done['result_review_sha256']==PRIOR_PINS['work/monthly_reference_gain250_RESULT_REVIEW.json'],'Prior independent result review binding')
    need(prepared['status']==results['status']=='COMPLETE' and len(prepared['artifacts_sha256'])==8 and len(results['artifacts_sha256'])==38,'Prior phase cardinalities')
    for folder,receipt in [('prepared',prepared),('results',results)]:
        for name,h in receipt['artifacts_sha256'].items():
            need(Path(name).name==name,'Prior artifact basename');bind(pins,PRIOR/folder/name,h)
    need(frozen['orders_sha256']==prepared['artifacts_sha256']['ORDERS.json'],'Prior orders freeze identity')
    bind(pins,PRIOR/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    # No numerical field in FEATURES,orders,NPZ,TRAIN_RESULTS orRESULTS is decoded.
    tests=read('work/monthly_reference_gain250_TESTS.json');review=read('work/monthly_reference_gain250_CODE_REVIEW.json')
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==206,'Prior formal synthetic proof')
    need(review['passed'] is True and review['unresolved_blockers']==[] and review['code_sha256']==tests['code_sha256'],'Prior code-review chain')
    need(len(tests['code_sha256'])==12,'Prior complete code closure')
    for p,h in tests['code_sha256'].items():bind(pins,p,h)
    return dict(experiment=PRIOR.name,plan_sha256=ph,completion_sha256=sha(PRIOR/'COMPLETION.json'),
      prepared_completion_sha256=sha(PRIOR/'prepared/COMPLETION.json'),frozen_sha256=sha(PRIOR/'selected/FROZEN.json'),
      result_completion_sha256=sha(PRIOR/'results/COMPLETION.json'),result_review_sha256=PRIOR_PINS['work/monthly_reference_gain250_RESULT_REVIEW.json'],
      prepared_artifacts_hashed=8,result_artifacts_hashed=38,training_results_hashed=True,prior_code_files_hashed=12,
      actual_prior_price_cache_or_outcome_arrays_decoded=False,protected_archive_rehashed_by_context_builder=False)

def main():
    need(platform.python_version()=='3.9.6','Use pinned run_local.sh')
    need(not OUT.exists(),'Context write-once')
    pins={};bind(pins,DEFINITION,DEFINITION_SHA);bind(pins,SOURCE,SOURCE_SHA)
    d=read(DEFINITION);s=read(SOURCE)
    need(d['arms']==list(ARMS) and d['new_trials']==3 and d['new_predictive_model_fits']==0,'Exact prospective arms')
    need(s['schema']==metadata.SCHEMA and s['passed'] is True and s['metadata_only'] is True and s['source_admitted'] is False,'Prospective source schema')
    need(s['definition_sha256']==DEFINITION_SHA and s['actual_cell_coverage']=='UNKNOWN_UNTIL_REGISTERED_PREPARE','Source definition and numeric scope')
    need(s['clock_event_keys']==list(metadata.CLOCK_KEYS) and s['identity_cell_keys']==list(metadata.IDENTITY_KEYS),'Producer/oracle metadata schema')
    for p,h in s['input_sha256'].items():bind(pins,p,h)
    bind(pins,Path(__file__));bind(pins,Path(metadata.__file__))
    for p,h in metadata.EXPECTED.items():bind(pins,p,h)
    reviews={}
    for kind in ('definition','math','source'):
        p=s[kind+'_review_path'];r=read(p)
        need(r['passed'] is True and r['unresolved_blockers']==[],'All three definition reviews')
        bind(pins,p,s[kind+'_review_sha256']);reviews[kind]=dict(path=p,sha256=s[kind+'_review_sha256'])
    prior=closure(pins)
    # Reuse original immutable implementation/cache inputs; hash only, never import.
    original=read('outputs/monthly_local_linear70/PLAN.json')
    for group in ('hashes','source_hashes'):
        for p,h in original[group].items():bind(pins,p,h)
    cache=read('outputs/monthly_local_linear70/CACHE_HASHES.json')
    bind(pins,'outputs/monthly_local_linear70/CACHE_HASHES.json')
    for name,h in cache.items():
        need(Path(name).name==name,'Original cache member basename');bind(pins,ROOT/'outputs/monthly_local_linear70'/name,h)
    for p in ('outputs/monthly_core_timing70_research/core_adapter.py','outputs/monthly_path70_research/incumbent_adapter.py',
              'outputs/monthly_local_linear70/INCUMBENT_POLICY.json','outputs/monthly_local_linear70/local_linear_ORDERS.json',
              'outputs/monthly_stability_research/stability.py','outputs/monthly_guidance_tweaks/tweaks.py',
              'outputs/monthly_quality_research/explore.py','outputs/monthly_stability_monitor/base_strategy.py',
              'outputs/monthly_stability_monitor/strategy.py','work/ptr169_results_independent.py',
              'work/ptr169_rebuild_evaluate.py','work/monthly_ownsector_gate238_host.py'):
        bind(pins,p)
    # Mutable coordinator state is checked as metadata, never frozen as an input.
    central=json.loads((ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json').read_text())
    need(len(central['completed'])==246 and sum(x['trials'] for x in central['completed'])==1251,'Expected predecessor counters')
    need(central['registered_queue']==[] and central['completed'][-1]['directory']==PRIOR.name,'No registered successor')
    last=central['completed'][-1]['completion']
    need(Path(last['path'])==PRIOR/'COMPLETION.json' and last['sha256']==prior['completion_sha256'],'Central predecessor closure')
    context=dict(schema='monthly_basket_core251_context_v1',created_utc=datetime.now(timezone.utc).isoformat(),
      status='PROSPECTIVE_CONTEXT_READY_BEFORE_ACTUAL_FEATURES_REGISTRATION_PENDING',
      hypothesis=d['hypothesis'],registered_trials=3,new_predictive_model_fits=0,arms=list(ARMS),
      predecessor=PRIOR.name,prior_completed_rounds=246,prior_trials=1251,expected_completed_rounds=247,expected_trials=1254,
      predecessor_closure=prior,prospective_definition=deepcopy(d),definition_sha256=DEFINITION_SHA,definition_reviews=reviews,
      source=dict(review_path=str(SOURCE.relative_to(ROOT)),review_sha256=SOURCE_SHA,schema=metadata.SCHEMA,
        panel_first_session=s['panel_first_session'],price_count=253,return_count=252,block_count=12,block_sessions=21,
        clock_event_keys=deepcopy(s['clock_event_keys']),identity_cell_keys=deepcopy(s['identity_cell_keys']),
        source_policy=deepcopy(s['source_policy']),actual_required253_support='UNKNOWN_UNTIL_REGISTERED_PREPARE',
        volume_used=False,sector_mapping_used=False,VOX_used=False,all49_universe_preserved=True,new_numeric_domain='FORMAL_EOLD_TRUE_ONLY'),
      original_control=dict(configuration='Local67 config4',original_arrays_preserved=27,original_E_preserved=True,
        originalcore_source_atol=1e-12,originalcore_source_rtol=0.,incumbent_order_pre_and_final_parity='EXACT',
        original_history_origin='2008-01-02',original_rolling_history='Frozen fullpast pandas control semantics; no resetting rolling63 or new positive-price filtering for originalU0.',
        episode_calls_per_decision_call=1,episode_including_n0=True,experimental_episode_calls=0,
        original_eligible_is_supplied='Do not rederive Eold approximately; verify registered source/cache and immutable incumbent first.'),
      experimental_domain=dict(Enew=d['new_domain'],shortlist=deepcopy(d['shortlist']),source_required='All formal Eoldtrue names before newM/ranking, not just eventualTop6.',
        common_basket_support=d['common_basket_support'],empty_common_domain=d['empty_common_basket_domain'],partial=d['partial_initial_event'],
        historical_positive_gate='Exact M252>0 with equality excluded; no state-dependent threshold.',
        no_original_score_terms=deepcopy(d['no_extra_score_terms'])),
      exact_math=dict(price_clock=deepcopy(d['price_clock']),block_return=d['block_return'],basket_statistics=deepcopy(d['basket_statistics']),
        objective=deepcopy(d['objectives']),arithmetic=d['exact_math'],positive_mean_invariant=d['positive_mean_invariant'],ties=d['ties'],
        scenario_interpretation=d['scenario_interpretation'],feature_comparison_tolerance=0.,canonical_rational_pair='Reduced integer numerator and positive denominator as decimal strings.',
        never_float_for_decision='No float M/objective or square-root approximation for domain, ranking, ties or chosen triples.'),
      direct_export=dict(
        selection='Direct exact winning triple or declared2/cash/zero-variance fallback. No episode, oldP/F score rerank or top2lock in either experimental arm.',
        selected_member_order='Descending exactM252 then ascending symbol.',
        display_formula='S - zero_based_position in rawM-ordered shortlist, S=len(shortlist)<=6.',
        selected_values='Finite exact float integers1..6 for each selected member.',unselected_value=0.,
        experimental_pres_scores='Two equal display-only matrices for each experimental arm. They never choose membership or weights and are not originalpre/final scores.',
        incumbent_pres_scores='Original incumbent fullP and fullF exact, with original order serialization.',
        partial='Named2016-08-17 allarms exact incumbent P/F/orders; it is the explicit exception to experimental display matrices.',
        cash='No selectedsymbols/scores, full display matrices zero.',
        exact_statistics='Unbounded exactM/N/V remain canonical-rational audit evidence; never cast them tofloat for export.'),
      selection=deepcopy(d['selection']),execution=dict(contract=d['actual_execution'],path_export_mapping=deepcopy(d['path_export_mapping']),
        registered_paths=[dict(id='same_close_zero',execution='close',cost_per_side=0.),dict(id='next_open_zero',execution='open',cost_per_side=0.),dict(id='next_open_cost',execution='open',cost_per_side=.0005)],
        signal_clock='Official formal month-end; namedinitialpartial preserved.',position_counts=[0,2,3],weights='EW3 / twoequalhalves / cash',
        terminal_liquidation=False,within_month_rebalance=False,observer_promotion=False),
      phase_order=[
        'Formal synthetic tests and independent CODE_REVIEW then immutable PLAN registration before any251 actual new source/statistics.',
        'Verify original64byte hashes,27cache,source identity/axis,fullpast originalcore; scan every formal Eoldtrue253price cell. Only exact per-securityM and12blocks may be prepared for independent source verification; no new shortlist or trio selection yet.',
        'Pass independent source callback, then reconstruct original Local67 incumbent P/F/orders with originalepisode once and exact frozen-original order check plus independent incumbent callback.',
        'Only after both gates derive Enew/rawMTop6/commontrios and directly choose joint anddiagonal baskets. No experimentalepisode or allauxfunding.',
        'Independently verify fullfeatures/source/selection and all123cached plus5physical prefixes, original27unchanged and namedpartial exact.',
        'Compute first88 zero-fee training metrics; separately admit both experimental arms vsinc and freeze selectedprimary with defined CAGR/Sharpe/diagonal tie policy.',
        'After FROZEN only: diagnose9paths45slices18traininggroups, verify everyzero-fee field and immutableincumbent reproduction.',
        'Independent saved-result review and full21444protected audit then append3trials with CAS central update; later/full cannot rewriteFROZEN, no observer mutation.'
      ],
      validation=dict(cached_prefixes=list(range(123)),physical_prefixes=[0,1,53,88,122],
        physical_clock='Prices through finalquery only; officialcalendar may retain one next-session month-end witness. n0 accepts empty consumed axes.',
        feature_math_masks_ranks_winners='EXACT canonical rationals and exact discrete equality; no tolerancebased ties.',
        originalcore_source_atol=1e-12,originalcore_source_rtol=0.,actual_orders_incumbent='Exact original ordered choices/metadata before experimentalselection',
        source_gate='Original core plus per-requiredM/blocks only; new_basket_selection_computed=False.',
        mandatory_callbacks='Formalprepare requires source_validator,incumbent_validator and expected_incumbent_orders; all123cacheddecisions require incumbent gate and expectedoriginalorders.',
        outcome_paths=9,outcome_slices=45,training_metric_groups=18,zero_cost_streams=6,
        code_binding='New251 module/test bytes remain unfrozen while authored; final formalTESTS,CODE_REVIEW and PLAN pin every required newmodule and transitive dependency. This context imports no251 implementation.'),
      coverage=deepcopy(d['coverage']),limitations=deepcopy(d['limits'])+[
        'Both experimental arms being admissible is prospective in-sample architecture selection; no pristine out-of-sample or covariance-success claim from diagonal admission.',
        'Only joint-versus-diagonal isolates denominator covariance on common triples; either versusLocal67 changes domain and completeobjective together.',
        'Metadata checks do not establish actual Eold-required source coverage or exhaustive identity/corporate-action continuity.',
        'Inherited raw64 closure includes unused new-factor volume/sector inputs for control protection; new basketmath consumes adjustedprices only.'
      ],input_sha256=pins,builder_scope=dict(metadata_only=True,prices_loaded=False,volumes_loaded=False,npz_decoded=False,
        actual_features_orders_returns_computed=False,registered_strategy=False,central_modified=False,observers_modified=False,old_files_modified=False,
        predecessor_artifacts_hashed_only=True,mutable_central_bound_as_input=False),
      previous_goal_turn_classification='progress',previous_goal_turn_evidence='250 completed and independently reviewed;251 definitions/reviews fixed prospectively with actual251support UNKNOWN.')
    for p,h in pins.items():need(sha(p)==h,'Input changed during context build '+p)
    with OUT.open('x') as f:json.dump(context,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='CONTEXT_READY_NOT_REGISTERED',path=str(OUT.relative_to(ROOT)),sha256=sha(OUT),input_pins=len(pins),prior_completed=246,prior_trials=1251)),flush=True)

if __name__=='__main__':main()

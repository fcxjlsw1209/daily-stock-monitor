"""Prospective unknown-zero availability version; preserve failed249 byte evidence.
Only saved metadata and byte digests, never numeric bars/NPZ or outcome fitting.
"""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import prepare_volume_breadth249_contract as prior
ROOT=prior.ROOT
SOURCE=ROOT/'work/volume_breadth249_v2_SOURCE_REVIEW.json'
CONTEXT=ROOT/'work/monthly_volume_breadth249_v2_context.json'
EXPECTED={
 'work/volume_breadth249_ZERO_POLICY_REVIEW.json':'f87370db418f1f46aeafe77a298ce19a3891af63149024bae8dad929bdd600a1',
 'work/monthly_volume_breadth249_context.json':'b1d2a6c2de07cc10fdd5f1ba542e6558ebea7ff3a5dfd3ca10b67ba84a34cb93',
 'work/volume_breadth249_SOURCE_REVIEW.json':'b07b6342161f2e90767dc08d059b5710915998b93412ce27877d6e903f2b47ea',
 'outputs/monthly_volume_breadth249/PLAN.json':'f3f4a6d9f00c2aacf6877fd39f0763b446873afb0588806972013735a4d8a9f8',
 'work/volume_breadth249_SOURCE_DIAGNOSTIC.json':'d3c876fe5a03a095ed2eed8d59184143cfd5aa905ff636282b476b11cc5b6f07',
 'work/volume_breadth249_FAILURE_REVIEW.json':'70ba41f08901f2950f02f4f8e0f7f09a80b096bffcd0d3f27715b09af36af6cd'}

def main():
    assert not SOURCE.exists() and not CONTEXT.exists()
    source=deepcopy(prior.read('work/volume_breadth249_SOURCE_REVIEW.json'))
    context=deepcopy(prior.read('work/monthly_volume_breadth249_context.json'))
    pins={}
    for name,digest in context['input_sha256'].items():prior.bind(pins,name,digest)
    for name,digest in EXPECTED.items():prior.bind(pins,name,digest)
    archive_path='outputs/monthly_volume_breadth249/ABORTED.json'
    policy_path='work/volume_breadth249_ZERO_POLICY_REVIEW.json'
    for name in (archive_path,policy_path,'work/archive_volume_breadth249_source_failure.py','work/volume_breadth249_source_diagnostic.py','work/prepare_volume_breadth249_v2_contract.py'):
        prior.bind(pins,name)
    archive=prior.read(archive_path);diagnostic=prior.read('work/volume_breadth249_SOURCE_DIAGNOSTIC.json');policy=prior.read(policy_path)
    assert archive['status']=='ABORTED_DURING_SOURCE_PREPARATION' and archive['completed_rounds_unchanged']==244 and archive['completed_trial_count_unchanged']==1245
    assert archive['plan_sha256']==EXPECTED['outputs/monthly_volume_breadth249/PLAN.json']
    assert archive['actual_candidate_features']==archive['actual_candidate_orders']==archive['actual_candidate_returns']==0
    assert diagnostic['source_admitted'] is False and diagnostic['strict_pit'] is False and diagnostic['passed'] is True
    assert diagnostic['summary']['hard_invalid_events']==20 and diagnostic['summary']['ordinary_missing_events']==3
    assert policy['passed'] is True and policy['status']=='PROSPECTIVE_249_V2_UNRESOLVED_ZERO_FALLBACK_SELECTED'
    assert policy['selected_new_version_policy']['version']=='monthly_volume_breadth249_v2' and policy['selected_new_version_policy']['policy']=='unresolved_zero_volume whole-event fallback'
    rule='unresolved_zero_volume_whole_event_incumbent_fallback'
    # Root's sole repair decision precedes any new candidate result. Uniform, no performance alternative.
    sp=deepcopy(source['source_policy'])
    sp.update(zero_volume_rule=rule,actual_common49_coverage='SOURCE_ONLY_DIAGNOSTIC_98_POSITIVE_COMPLETE_20_ZERO_3_MISSING_NOT_ADMISSION',
      zero_volume_detail='Any observed native numeric volume zero anywhere in the required315 slots of any of the49 names is explicitly unavailable. Preserve raw0 and a separate volume_zeros slot list; no conversion to NaN, positive epsilon, row deletion or ticker removal. Both experimental arms copy the entire exact incumbent event after every hard-source and identity check.',
      invalid_source='Boolean/non-real, infinity, failed/overflowing binary64 cast, negative volume, any price<=0, or nonzero real volume that casts to0 remain hard failures even when another member has missing/zero. Native observed volume0 alone is explicit unavailable, not an economic-data assertion.',
      identity='Query-visible same-security positive price OR volume before not_before is a hard contradiction. Prelisting volume0 is unavailable under the uniform zero policy and cannot supply history. Complete price253 and volume315 scans before any fallback.',
      no_short_circuit='Scan all required49 members and every present source slot for hard numeric/identity errors before zero, ordinary-missing or known-transition fallback.',
      fallback_branch_precedence=['initial_partial','zero_volume','missing_source','known_transition','available'],
      zero_economic_semantics='UNKNOWN: native stored CSV token0 confirmed, not certified no-trade or missing placeholder.',
      freeze='Newversion prospectively frozen after source-only diagnosis and before any candidate ranks/funded scores/orders/returns; no outcome-dependent exceptions. Original249 contract remains unchanged.')
    now=datetime.now(timezone.utc).isoformat()
    source.update(schema='monthly_volume_breadth249_v2_metadata_source_review_v1',created_utc=now,
      source_policy=sp,actual_cell_coverage='SOURCE_DIAGNOSTIC_ONLY_NOT_REGISTERED_PREPARE',
      actual_common49_source_scan_summary=deepcopy(diagnostic['summary']),
      diagnostic_path='work/volume_breadth249_SOURCE_DIAGNOSTIC.json',diagnostic_sha256=pins['work/volume_breadth249_SOURCE_DIAGNOSTIC.json'],
      repair_archive_path=archive_path,repair_archive_sha256=pins[archive_path],
      original_policy_path=source['policy_path'],original_policy_sha256=source['policy_sha256'],
      policy_path=policy_path,policy_sha256=pins[policy_path],
      metadata_review_scope='Builder verifies saved metadata/digests only. A separately executed source-only diagnostic already scanned actual windows; it did not compute249 features, ranks, funded scores or strategy outcomes. Current builder does not reread numeric rows.',
      actual_cell_coverage_limitation='20 zero-volume events and3 ordinary-missing events from source-only diagnostic are known availability observations, not registered pipeline preparation or source-semantic certification.',
      original_U0_and_core_parity='UNKNOWN_UNTIL_REGISTERED_PREPARE',input_sha256=deepcopy(pins))
    context.update(schema='monthly_volume_breadth249_v2_context_v1',created_utc=now,
      repair=dict(failed_directory='monthly_volume_breadth249',failed_plan_sha256=EXPECTED['outputs/monthly_volume_breadth249/PLAN.json'],
        failure_review_sha256=pins['work/volume_breadth249_FAILURE_REVIEW.json'],archive_sha256=pins[archive_path],
        diagnostic_sha256=pins['work/volume_breadth249_SOURCE_DIAGNOSTIC.json'],zero_policy_review_sha256=pins[policy_path],
        sole_change='Uniform unknown native-zero source unavailability, explicitly separated from ordinary missing, with complete-event incumbent fallback.',
        unchanged=['original64 raw bytes','original27 arrays','all49 inventory and eligibility','253/315 clocks','exact Fraction math','common49 ranks','three arms','.01 budget','core and episode funding','training88 guards','execution and fee definitions'],
        new_features_before_version_registration=0,new_candidate_returns_before_version_registration=0,
        raw_source_diagnosis_before_registration=True,failed_attempt_retained=True),
      input_sha256=deepcopy(pins))
    context['source'].update(review_path=str(SOURCE.relative_to(ROOT)),review_sha256=None,source_policy=deepcopy(sp),
       all_missing_fallback='Any ordinary required absence, explicit native-volume zero or known eligible unresolved transition makes both experiments copy incumbent for the entire event; scan all hard errors first. Zeros remain explicit distinct metadata.')
    context['funding']['bypass']='Named partial, explicit zero volume, ordinary missing or known transition: inactive; exact original P/F/orders copied and compared. No target numeric consumption.'
    context['validation']['admission_pending']='New v2 host/producer/independent tests, review and prospective registration required. Source-only diagnosis is not completed preparation or strategy admission.'
    context['builder_scope'].update(source_diagnostic_metadata_read=True,source_diagnostic_numeric_rows_reloaded=False,
      prior_failed_strategy_returns_exist=False,old_files_modified=0)
    context['limitations'] += [
      'Eight native ARKK source-zero dates affect20 early formal events. Their economic meaning remains unknown; complete-event incumbent fallback avoids pretending that missing flow evidence supports new ranks.',
      'Three late formal events retain original ordinary-missing fallback. The diagnostic98 complete-positive events do not by themselves establish registered strategy admission.',
      'Fallback scope is data-driven source availability specified before candidate outcomes; reported performance must disclose its20 early and3 late affected events.']
    prior.write_new(SOURCE,source)
    context['source']['review_sha256']=prior.sha(SOURCE)
    context['input_sha256'][str(SOURCE.relative_to(ROOT))]=prior.sha(SOURCE)
    prior.write_new(CONTEXT,context)
    print('PROSPECTIVE249_V2_CONTRACTS_WRITTEN_NOT_REGISTERED',flush=True)

if __name__=='__main__':main()

"""228 source-only provenance review of frozen226 and source-only227.

Reads JSON/code and byte hashes only. It never decodes NPZ, imports strategy
modules, constructs risk inputs or estimates shared historical support/returns.
"""
from pathlib import Path
from datetime import date, datetime, timezone
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'work/expanded_error228_SOURCE_REVIEW.json'
PRIOR=ROOT/'outputs/monthly_joint_error226'
EXPANDED=ROOT/'outputs/monthly_ineligible_source227'
OLD=ROOT/'outputs/monthly_identity_transfer206'
LOCAL=ROOT/'outputs/monthly_local_linear70'
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
PRIOR_PLAN='996a6f80084cb95c48427949786caff84b6935c8e73126433bd12980fec26d25'
PRIOR_CLOSE='80e1f20981a27348b4d2623d9103d3f3425d90d0ce083c4a76fb88937e68b414'
EXPANDED_PLAN='d2b8eb195a7c54fc61a3acb910653ad3f86dabbbb6d23cdf41c187260ab20ae1'
EXPANDED_CLOSE='d4669e7254e15bbf799dc7f5aca69b6069ffc83094b3d98f839a8c8b897249e0'
SOURCE227_REVIEW='4610aacb5941fe67b8496ce607cd2152c642835b761f48b295e74c33d5f370ba'
CONTEXT227='196773953dbaa5caccae64d3df2fd2b5e2fad149291d4db8c318a28eb022a9e7'
EXTERNAL={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PREPARED227={'FORECASTS.npz','LABELS.npz','META.json','SOURCE_AUDIT.json','INDEPENDENT.json'}
PREPARED206={'FORECASTS.npz','RANKS_DELTAS.npz','MODEL_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
FORECAST_FLAGS=('passed','independent_annual_control_flow','shared_frozen_primitives_disclosed','old_forecasts_exact','old_masks_exact','old_forecasts_copied','same_training_cohort','four_targets_unchanged','annual_prior_december_cutoffs','five_year_mature_window','minimum_18_months','same_security_local_history','local_history_not_query_filter','separate_expansion_replay','warmup_mask_enforced','no_query_rerank','fit_counts_checked','audit_checked')
LABEL_FLAGS=('passed','independent_label_rebuild','exact_original_identity','original_centering_cohort_unchanged','fixed_four_clips_checked','old_targets_exact_copy','old_common_mask_preserved','original_reconstruction_checked','only_previously_ineligible_added','finite_plain10_checked','full_295_history_checked','volatility63_checked','full_official_future_path_checked','official_endpoints_checked','no_added_training_rows','audit_checked')
CONTEXT228=ROOT/'work/monthly_expanded_error228_context.json'
CONTEXT228_SHA='5831d860f78d87cf47bee1dac8267f180615a2add1c64b01b787ed4bc5dcf747'
PINS={}

def need(ok, message):
    if not ok:raise ValueError(message)

def resolve(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    need('CloudStorage' not in p.parts,'Cloud input forbidden')
    need(p.is_relative_to(ROOT) or str(p) in EXTERNAL,'Unexpected external input')
    need(p!=CENTRAL,'Mutable central must not be a persistent pin')
    return p

def sha(value):
    h=hashlib.sha256()
    with resolve(value).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def bind(value, expected=None):
    p=resolve(value);actual=sha(p)
    need(expected is None or actual==expected,'SHA mismatch: '+str(p))
    need(str(p) not in PINS or PINS[str(p)]==actual,'Input changed during metadata review')
    if str(p) in EXTERNAL:need(actual==EXTERNAL[str(p)],'External runtime changed')
    PINS[str(p)]=actual
    return actual

def read(value, expected=None):
    bind(value,expected)
    return json.loads(resolve(value).read_text())

def day(value):
    need(type(value) is str and date.fromisoformat(value).isoformat()==value,'Noncanonical metadata day')
    return value

def identity(meta, initial=False):
    days,ends,names=meta['dates'],meta['ends'],meta['symbols']
    need(len(days)==len(ends)==len(names) and days==sorted(set(days)),'Monthly identity axes')
    triples=[];months=[]
    for d,e,ids in zip(days,ends,names):
        day(d);day(e)
        need(e>d and len(ids)==49 and len(set(ids))==49,'Security width/next-month endpoint')
        need(all(type(s) is str and s and s.strip()==s for s in ids),'Invalid security identity')
        m=int(d[:4])*12+int(d[5:7]);em=int(e[:4])*12+int(e[5:7])
        partial=initial and d=='2016-08-17' and e=='2016-08-31'
        need(partial or em==m+1,'Wrong target month')
        if not partial:months.append(m)
        triples.extend((d,s,e) for s in ids)
    need(len(triples)==len(set(triples)) and len(months)==len(set(months)),'Duplicate identities/months')
    return triples

def review(saved_sha):
    need(not OUTPUT.exists(),'Metadata receipt is write-once')
    need(len(saved_sha)==64 and all(c in '0123456789abcdef' for c in saved_sha),'Explicit saved-review SHA required')
    context228=read(CONTEXT228,CONTEXT228_SHA)
    need(context228['round']==228 and context228['name']=='monthly_expanded_error228','Prospective228 context identity')
    for p,h in context228['evidence_sha256'].items():bind(p,h)
    prior=read(PRIOR/'PLAN.json',PRIOR_PLAN);prior_close=read(PRIOR/'COMPLETION.json',PRIOR_CLOSE)
    need(prior_close['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior_close['plan_sha256']==PRIOR_PLAN,'226 terminal predecessor')
    plan=read(EXPANDED/'PLAN.json',EXPANDED_PLAN);close=read(EXPANDED/'COMPLETION.json',EXPANDED_CLOSE)
    need(close['status']=='SOURCE_COMPLETED_INDEPENDENTLY_VERIFIED' and close['plan_sha256']==EXPANDED_PLAN,'227 terminal source lineage')
    for k in ('source_only','original_strategy_unchanged'):need(close[k] is True,'Source terminal flag '+k)
    for k in ('observers_modified','new_portfolio_returns'):need(close[k] is False,'Source terminal scope '+k)
    need(close['strategy_trials']==0,'Source-only work cannot count as strategy trial')
    for mapping in (prior['input_sha256'],plan['input_sha256']):
        for p,h in mapping.items():bind(p,h)
    for p,h in prior['input_sha256'].items():need(plan['input_sha256'].get(p)==h,'227 did not preserve prior226 closure pin')
    context=read(ROOT/'work/monthly_ineligible_source227_context.json',CONTEXT227)
    prerequisite=read(ROOT/'work/ineligible_source227_SOURCE_REVIEW.json',SOURCE227_REVIEW)
    need(context==plan['contract'] and prerequisite['context_sha256']==CONTEXT227 and prerequisite['passed'] is True and prerequisite['metadata_only'] is True,'227 preregistration source proof')
    for p,h in prerequisite['input_sha256'].items():bind(p,h)
    raw=plan['raw_source_hashes']
    need(len(raw)==64 and raw==prior['raw_source_hashes']==read(LOCAL/'PLAN.json')['raw_source_hashes']==read(OLD/'PLAN.json')['raw_source_hashes'],'Original64 source map')
    for p,h in raw.items():bind(p,h)
    protection=read(PROTECTED)
    need(len(protection)==21444 and close['protected_files']==prior_close['protected_files']==plan['protected_files']==21444,'Full protected archive certificate')
    old=read(OLD/'prepared/COMPLETION.json')
    need(old['status']=='COMPLETE' and set(old['artifacts_sha256'])==PREPARED206,'Original206 seven-member source manifest')
    for p,h in old['artifacts_sha256'].items():bind(OLD/'prepared'/p,h)
    prepared=read(EXPANDED/'prepared/COMPLETION.json',close['prepared_sha256'])
    need(prepared['status']=='COMPLETE' and prepared['plan_sha256']==EXPANDED_PLAN and set(prepared['artifacts_sha256'])==PREPARED227,'Exact227 five-member manifest')
    need(prepared['source_only'] is True and prepared['portfolio_returns'] is False and prepared['strategy_trials']==0,'Prepared source-only scope')
    for p,h in prepared['artifacts_sha256'].items():bind(EXPANDED/'prepared'/p,h)
    saved=read(EXPANDED/'RESEARCH_REVIEW.json',saved_sha)
    need(saved['passed'] is True and saved['plan_sha256']==EXPANDED_PLAN and saved['completion_sha256']==EXPANDED_CLOSE and saved['prepared_sha256']==sha(EXPANDED/'prepared/COMPLETION.json'),'Saved227 independent closure review')
    need(saved['source_only'] is True and saved['strategy_trials']==0 and saved['source_npz_decoded'] is False and saved['portfolio_returns_computed'] is False,'Saved review source-only scope')
    for p,h in saved['input_sha256'].items():bind(p,h)
    meta=read(EXPANDED/'prepared/META.json');original=read(REBUILD/'MONTHLY_META.json')
    need(meta==original and meta['monthly']==read(LOCAL/'META.json')==read(OLD/'prepared/META.json'),'Expanded source preserves all exact original169/206/Local67 metadata')
    query=identity(meta['monthly'],True);training=identity(meta['training'])
    need(len(query)==5978 and len(training)==8820,'Original query/training dimensions')
    sessions=read(CALENDAR)['sessions'];need(sessions==sorted(set(sessions)),'Original calendar ordering')
    for d in sessions:day(d)
    official=set(sessions);last={d[:7]:d for d in sessions}
    for d,e in zip(meta['training']['dates'],meta['training']['ends']):
        need(d in official and e in official and last[d[:7]]==d and last[e[:7]]==e,'Training endpoints exact calendar monthend')
    for d,e in zip(meta['monthly']['dates'],meta['monthly']['ends']):
        need(d in official and e in official and (d=='2016-08-17' or last[d[:7]]==d),'Query official clock')
        need(e=='2026-09-24' or last[e[:7]]==e,'Query endpoint clock')
    audit=read(EXPANDED/'prepared/SOURCE_AUDIT.json');independent=read(EXPANDED/'prepared/INDEPENDENT.json')
    need(close['forecasts']==audit['forecasts'] and close['labels']==audit['labels'],'Terminal source audit binding')
    for k in ('original_array_bytes_unchanged','no_new_orders','no_new_portfolio_returns'):need(audit[k] is True,'Source audit flag '+k)
    need(audit['strict_PIT'] is False,'Do not claim strict PIT')
    for name,flags in (('forecasts',FORECAST_FLAGS),('labels',LABEL_FLAGS)):
        for k in flags:need(independent[name].get(k) is True,'Missing saved independent flag '+name+'/'+k)
        need(independent[name]['counts']==audit[name]['counts'] and independent[name]['numerical_tolerance']==0. and independent[name]['producer_modules_imported'] is False and independent[name]['computes_returns'] is False,'Saved independent scope/counts')
    need(independent['forecasts']['shared_numerical_primitives'] is True,'Disclose reused frozen numeric primitives')
    need(independent['labels']['original_reconstruction_atol']==1e-12,'Frozen original reconstruction tolerance')
    f=audit['forecasts'];l=audit['labels']
    for k in ('old_parity_exact','original_forecasts_copied','new_queries_separate_replay','original_training_unchanged'):need(f[k] is True,'Forecast guarantee '+k)
    for k in ('features_recomputed','reranked'):need(f[k] is False,'No new feature ranks')
    for k in ('original_targets_copied','original_common_good_preserved','original_centers_only','added_rows_never_train','full_official_paths_required_for_added','original_strict_exceptions_retained'):need(l[k] is True,'Label guarantee '+k)
    need([p['prefix'] for p in audit['forecast_prefixes']]==[0,1,53,88,122] and [p['cutoff'] for p in audit['label_prefixes']]==['2019-12-31','2023-12-31','2025-12-31'],'Physical saved source prefixes')
    before=CENTRAL.read_bytes();state=json.loads(before)
    need(len(state['completed'])==224 and sum(x['trials'] for x in state['completed'])==1189,'Current strategy count224/1189')
    need(state['completed'][-1]['directory']==PRIOR.name and not state['registered_queue'] and state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Strategy predecessor/queue/incumbent')
    registered=[r for r in state.get('source_experiments',[]) if r['directory']==EXPANDED.name]
    need(len(registered)==1 and registered[0]['strategy_trials']==0 and registered[0]['completion']['sha256']==EXPANDED_CLOSE and Path(registered[0]['completion']['path']).resolve()==EXPANDED/'COMPLETION.json','Source227 separately registered')
    reviewed_code={}
    for name in ('work/monthly_joint_error226_host.py','work/monthly_joint_error226_model.py','work/monthly_joint_error226_independent.py','work/monthly_ineligible_source227_host.py','work/monthly_ineligible_source227_model.py','work/monthly_ineligible_source227_independent.py','work/ineligible_source227_saved_review.py'):
        reviewed_code[name]=bind(ROOT/name)
    implementation=bind(Path(__file__))
    result=dict(schema='expanded_error228_source_metadata_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,source_only=True,metadata_only=True,
        source_npz_decoded=False,training_npz_decoded=False,npz_decoded=False,source_models_imported=False,
        actual_support_computed=False,actual_joint_history_support_computed=False,actual_features_computed=False,actual_risks_computed=False,actual_labels_computed=False,actual_forecasts_computed=False,actual_returns_computed=False,actual_orders_computed=False,actual_model_fits=0,
        observers_modified=False,central_modified=False,frozen_files_modified=False,source_bytes_reverified=True,metadata_axes_verified=True,calendar_clock_verified=True,
        raw_source_hashes=raw,source_sha256=dict(sorted(PINS.items())),input_sha256=dict(sorted(PINS.items())),
        predecessor226=dict(plan_sha256=PRIOR_PLAN,completion_sha256=PRIOR_CLOSE,source_review_sha256=sha(ROOT/'work/joint_error226_SOURCE_REVIEW.json')),
        expanded227=dict(plan_sha256=EXPANDED_PLAN,completion_sha256=EXPANDED_CLOSE,prepared_sha256=sha(EXPANDED/'prepared/COMPLETION.json'),saved_review_sha256=saved_sha,context_sha256=CONTEXT227,prepared_artifacts_sha256=prepared['artifacts_sha256'],source_only=True,strategy_trials=0),
        source_identity=dict(query_months=122,query_width=49,query_rows=5978,training_months=180,training_width=49,training_rows=8820,metadata227_equals169=True,monthly_metadata_equals206_and_local67=True,exact_signal_security_label_end=True,original_identity_order_preserved=True,numeric_offsets_not_decoded=True,initial_partial='2016-08-17',last_partial_label_end='2026-09-24',identity_scope='Original canonical security IDs and inherited alias assumptions; not new issuer/PIT certification'),
        original_source_guarantees=dict(original_forecasts_exact_copy=True,original_forecast_masks_preserved=True,original_targets_exact_copy=True,original_common_good_subset_preserved=True,original_clipped_centering_cohort_preserved=True,added_labels_used_in_original_models=False,query_features_recomputed=False,query_feature_reranking=False,source227_fit_work_already_performed=True,new_source_model_refits_required_in228=False),
        saved_source_counts=dict(basis='Previously saved source227 counts only; no new228 complete-history support computed',forecasts=f['counts'],labels=l['counts']),
        interfaces=dict(forecasts227_keys=['raw_source_log','raw_local_constant','raw_local_linear','available','local_available','expansion_allowed'],labels227_keys=['target_simple_excess','target_log_excess','target_vol_scaled','target_loss_utility','common_good','added','vol63','dates','ends','offsets'],metadata227_keys=list(meta),risk_prediction_key='raw_local_linear',risk_target_key='target_log_excess',risk_only_training_overrides=['common_good','target_simple_excess','target_log_excess','target_vol_scaled','target_loss_utility'],unchanged_training_fields=['dates','ends','offsets','train_plain'],field_schema_basis='Frozen227 writer source; NPZ field names/values not decoded by this metadata review'),
        required_prepare_checks=['Retain original206 seven-field cached parity separately from expanded227 six-field source; never overwrite original27 strategy arrays.',
            'Verify227 LABELS dates/ends/offsets exact dtype/value against169; risk-only copy supplies227 common_good/four targets plus unchanged169 train_plain.',
            'Verify original forecast rows and original labels remain exact copies; added rows never fit a source model in228.',
            'Historical-use mask is expanded only for risk history. Current eligible/pre/top2/boundary/episode and original strategy arrays remain unchanged.',
            'Both risk arms share the complete B history, prior-Dec31 maturity/five-year window/min18 and all physical prefix checks.',
            'Legacy226 audit field names mentioning original universe are inherited labels, not a claim that228 historical-use domain equals original U. Add explicit228 expansion evidence.',
            'Do not recompute label centering over expanded rows, use local_available as eligibility, fill missing rows, or claim fresh PIT/raw-source admission.'],
        protected_manifest=dict(path=str(PROTECTED.relative_to(ROOT)),sha256=sha(PROTECTED),members=21444,registered_and_closed_full_checks=True,all_members_rehashed_by_this_review=False),
        central_snapshot=dict(sha256=hashlib.sha256(before).hexdigest(),persistent_input=False,completed_strategies=224,trials=1189,last_strategy=PRIOR.name,source227_separate=True,registered_queue_empty=True),
        reviewed_code_sha256=reviewed_code,context_sha256=CONTEXT228_SHA,evidence_sha256=context228['evidence_sha256'],context228_not_pinned=False,host_must_bind_final228_context=True,host_not_pinned_to_avoid_cycle=True,strict_pit=False,
        limitations=['227 is a frozen source-only extension, not demonstrated portfolio improvement;228 joint complete history remains uncomputed.',
            'Cached10 inputs can contain neutral ranks from missing raw feature values;295 adjusted-close support does not certify all rawOHLCV fields.',
            'Historical annual as-of replay uses later-acquired vendor data, inherited aliases and2016 leading-gap splices; not live archived vintages or strictPIT.',
            'Added labels use original clipped monthly centers. Errors measure centered relative-prediction error risk, not absolute realized portfolio volatility.',
            'Full21444 byte verification is attested by frozen registration/close; this review checks its manifest identity and count without repeating all protected bytes.'],
        implementation=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=implementation))
    need(CENTRAL.read_bytes()==before,'Central changed during metadata review')
    with OUTPUT.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,output=str(OUTPUT.relative_to(ROOT)),sha256=sha(OUTPUT),source_pins=len(PINS),npz_decoded=False,actual_support_computed=False)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source227-saved-review-sha256',required=True);a=p.parse_args()
    review(a.source227_saved_review_sha256)

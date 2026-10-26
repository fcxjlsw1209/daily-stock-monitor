"""227 source-only byte/metadata review. No array decode, fit or label construction."""
from pathlib import Path
from datetime import date, datetime, timezone
import argparse
import ast
import hashlib
import json

R=Path(__file__).resolve().parents[1]
OUT=R/'work/ineligible_source227_SOURCE_REVIEW.json'
CONTEXT=R/'work/monthly_ineligible_source227_context.json'
PRIOR=R/'outputs/monthly_joint_error226'
PRIOR_PLAN_SHA='996a6f80084cb95c48427949786caff84b6935c8e73126433bd12980fec26d25'
PRIOR_CLOSE_SHA='80e1f20981a27348b4d2623d9103d3f3425d90d0ce083c4a76fb88937e68b414'
CHECKPOINT=R/'work/after226_research_checkpoint.json'
CHECKPOINT_SHA='98b4e7086eb73257ac23875687bf7519581a3f45a34cc4e50988981c9edcd728'
SOURCE=R/'outputs/monthly_identity_transfer206'
LOCAL=R/'outputs/monthly_local_linear70'
REBUILD=R/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
CALENDAR=R/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=R/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
EXTERNAL={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PIN={}
PREPARED={'FORECASTS.npz','RANKS_DELTAS.npz','ORDERS.json','MODEL_AUDIT.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
PLAIN10=['raw','recent_risk','vol','sector','cmf21','cmf63','signed_volume','volume_ratio','close_location21','compression']

def need(value,message):
    if not value:raise ValueError(message)

def local_path(value):
    p=Path(value);p=p if p.is_absolute() else R/p
    need('CloudStorage' not in str(p),'Cloud path forbidden')
    need((p.is_relative_to(R) and p.resolve().is_relative_to(R)) or str(p) in EXTERNAL,'Nonlocal source')
    need(p!=R/'outputs/monthly_target70_research/CONTINUATION_STATUS.json','Do not pin mutable central')
    return p

def digest(value):
    p=local_path(value);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def bind(value,expected=None):
    p=local_path(value);actual=digest(p)
    need(expected is None or actual==expected,'Source SHA mismatch '+str(p))
    need(str(p) not in PIN or PIN[str(p)]==actual,'Source changed during review')
    if str(p) in EXTERNAL:need(actual==EXTERNAL[str(p)],'External runtime changed')
    PIN[str(p)]=actual
    return actual

def read(value,expected=None):
    bind(value,expected);return json.loads(local_path(value).read_text())

def canonical(day):
    need(type(day) is str and date.fromisoformat(day).isoformat()==day,'Noncanonical date')
    return day

def axes(meta,initial=False):
    ds,es,ss=meta['dates'],meta['ends'],meta['symbols']
    need(len(ds)==len(es)==len(ss) and ds==sorted(set(ds)),'Metadata axes')
    identities=[]
    for d,e,names in zip(ds,es,ss):
        canonical(d);canonical(e)
        need(e>d and len(names)==49 and len(set(names))==49,'Identity width/label end')
        need(all(type(s) is str and s and s.strip()==s for s in names),'Invalid exact security ID')
        y,m=map(int,d[:7].split('-'));ny,nm=(y+1,1) if m==12 else (y,m+1)
        need((initial and d=='2016-08-17' and e=='2016-08-31') or e[:7]==f'{ny:04d}-{nm:02d}','Next-month label identity')
        identities.extend((d,s,e) for s in names)
    need(len(identities)==len(set(identities)),'Duplicate exact identity triple')
    return identities

def review(context_sha):
    need(not OUT.exists(),'Source review is write-once')
    need(len(context_sha)==64 and all(c in '0123456789abcdef' for c in context_sha),'Explicit final context SHA required')
    context=read(CONTEXT,context_sha)
    plan=read(PRIOR/'PLAN.json',PRIOR_PLAN_SHA);closed=read(PRIOR/'COMPLETION.json',PRIOR_CLOSE_SHA)
    need(closed['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and closed['plan_sha256']==PRIOR_PLAN_SHA,'Prior closure')
    checkpoint=read(CHECKPOINT,CHECKPOINT_SHA)
    need(checkpoint['completed_round']=='monthly_joint_error226' and checkpoint['completion_sha256']==PRIOR_CLOSE_SHA,'Checkpoint lineage')
    inherited=plan['input_sha256']
    for p,h in inherited.items():bind(p,h)
    evidence=context.get('evidence_sha256',{})
    need(type(evidence) is dict,'Context evidence map')
    for p,h in evidence.items():bind(p,h)
    raw=plan['raw_source_hashes'];need(len(raw)==64,'Original raw64 map')
    local_plan=read(LOCAL/'PLAN.json',inherited[str(LOCAL/'PLAN.json')])
    source_plan=read(SOURCE/'PLAN.json',inherited[str(SOURCE/'PLAN.json')])
    need(raw==local_plan['raw_source_hashes']==source_plan['raw_source_hashes'],'Original raw map mismatch')
    for p,h in raw.items():need(inherited.get(p)==h,'Raw pin not inherited');bind(p,h)
    protection=read(PROTECTED,inherited[str(PROTECTED)])
    need(len(protection)==21444 and plan['protected_files']==closed['protected_files']==21444,'Full protection manifest identity')
    done=read(SOURCE/'COMPLETION.json',inherited[str(SOURCE/'COMPLETION.json')])
    prep=read(SOURCE/'prepared/COMPLETION.json',inherited[str(SOURCE/'prepared/COMPLETION.json')])
    need(done['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prep['status']=='COMPLETE','206 incomplete')
    need(done['plan_sha256']==prep['plan_sha256']==digest(SOURCE/'PLAN.json'),'206 plan lineage')
    need(set(prep['artifacts_sha256'])==PREPARED and prep['original_array_count']==27,'Exact206 seven artifact set')
    for filename,h in prep['artifacts_sha256'].items():bind(SOURCE/'prepared'/filename,h)
    source_parity=read(SOURCE/'prepared/SOURCE_PARITY.json')
    need(source_parity['raw_source_hashes']==raw,'206 raw map')
    for k in ('original_training_cache_exact','original_identity_exact','original_control_constant_exact'):
        need(source_parity[k] is True,'206 saved source parity '+k)
    for k in ('new_labels','new_raw_features'):need(source_parity[k] is False,'206 altered source '+k)
    parity=read(REBUILD/'PARITY.json',inherited[str(REBUILD/'PARITY.json')])
    need(parity['passed'] is True,'169 saved control parity failed')
    field_shapes=dict(dates=[8820],ends=[8820],event_dates=[5978],offsets=[181],
        train_plain=[8820,10],event_plain=[5978,10],train_good=[8820],train_eligible=[8820],eligible=[122,49],
        common_good=[8820],vol63=[8820],target_simple_excess=[8820],target_log_excess=[8820],
        target_vol_scaled=[8820],target_loss_utility=[8820])
    for key,shape in field_shapes.items():
        proof=parity['checks']['monthly'][key]
        need(proof['shape']==shape and (proof.get('exact') is True or proof.get('max_abs_error')==0.),'169 saved field proof '+key)
    bind(REBUILD/'TRAIN_INPUTS.npz',inherited[str(REBUILD/'TRAIN_INPUTS.npz')])
    m=read(SOURCE/'prepared/META.json');meta=read(REBUILD/'MONTHLY_META.json',inherited[str(REBUILD/'MONTHLY_META.json')])
    need(m==meta['monthly']==read(LOCAL/'META.json',inherited[str(LOCAL/'META.json')]),'Original query metadata mismatch')
    need(len(m['dates'])==122 and len(meta['training']['dates'])==180,'Metadata dimensions')
    training_ids=axes(meta['training']);query_ids=axes(m,True)
    calendar=read(CALENDAR,inherited[str(CALENDAR)]);sessions=calendar['sessions']
    need(sessions==sorted(set(sessions)),'Calendar sessions order/uniqueness')
    for d in sessions:canonical(d)
    last={d[:7]:d for d in sessions};session_set=set(sessions)
    for d,e in zip(meta['training']['dates'],meta['training']['ends']):
        need(d in session_set and e in session_set and last[d[:7]]==d and last[e[:7]]==e,'Training clock')
    for d,e in zip(m['dates'],m['ends']):
        need(d in session_set and e in session_set,'Query clock')
        need(d=='2016-08-17' or last[d[:7]]==d,'Formal query clock')
        need(e=='2026-09-24' or last[e[:7]]==e,'Query target end clock')
    code_paths=['work/ptr169_rebuild.py','outputs/monthly_session_type_research/study.py',
        'outputs/monthly_architecture_restart/study.py','outputs/monthly_core_expansion/study.py',
        'outputs/monthly_core_expansion/ordinal_model.py','outputs/monthly_target_definition70/study.py',
        'outputs/monthly_target_definition70/labels.py','outputs/monthly_local_linear70/model.py',
        'work/monthly_identity_transfer206_host.py','outputs/monthly_stability_monitor/inputs.py']
    code={}
    for p in code_paths:bind(p,inherited.get(str(R/p)));code[p]=local_path(p).read_text()
    need("['raw','recent_risk','vol','sector']" in code['work/ptr169_rebuild.py'],'Plain core order changed')
    need("AUX = ['cmf21', 'cmf63', 'signed_volume', 'volume_ratio', 'close_location21', 'compression']" in code['outputs/monthly_architecture_restart/study.py'],'Plain auxiliary order changed')
    need("train_good=train_eligible&np.isfinite(targets['return'])&np.isfinite(targets['ordinal'])" in code['work/ptr169_rebuild.py'],'Training eligibility lineage')
    need('base_cohort = eligible & cached_good & finite_X' in code['outputs/monthly_target_definition70/labels.py'],'Original common-cohort semantics')
    need('targets[arm][selected] = clipped - clipped.mean()' in code['outputs/monthly_target_definition70/labels.py'],'Original label centering')
    need('s.rank(pct=True).fillna(.5).to_numpy() - .5' in code['outputs/monthly_architecture_restart/study.py'],'Raw missing rank neutral semantics')
    implementation=bind(Path(__file__))
    result=dict(schema='ineligible_source227_metadata_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,source_only=True,metadata_only=True,source_admission_metadata_only=True,context_sha256=context_sha,evidence_sha256=evidence,
        source_bytes_reverified=True,metadata_axes_verified=True,calendar_clock_verified=True,
        raw_source_hashes_verified=True,raw_source_map_matches_original=True,raw_source_hashes=raw,
        predecessor226=dict(plan_sha256=PRIOR_PLAN_SHA,completion_sha256=PRIOR_CLOSE_SHA,checkpoint_sha256=CHECKPOINT_SHA,inherited_input_count=len(inherited)),
        source206_prepared_manifest=dict(path=str((SOURCE/'prepared/COMPLETION.json').relative_to(R)),sha256=digest(SOURCE/'prepared/COMPLETION.json'),
            artifacts_sha256=prep['artifacts_sha256'],original_array_count=27),
        source169_saved_parity=dict(path=str((REBUILD/'PARITY.json').relative_to(R)),sha256=digest(REBUILD/'PARITY.json'),
            field_proofs={key:parity['checks']['monthly'][key] for key in field_shapes},numeric_parity_reexecuted=False),
        metadata_identity_review=dict(query_months=122,query_width=49,query_rows=len(query_ids),training_months=180,
            training_width=49,training_rows=len(training_ids),query_metadata_exact=True,exact_unique_signal_security_end_triples=True,
            flat_order='month-major original metadata symbols order; training offsets are the cumulative original widths',
            numeric_offsets_not_decoded=True,security_scope='exact inherited canonical security strings; no new issuer or alias certification',
            initial_partial='2016-08-17',last_partial_label_end='2026-09-24'),
        plain10_definition=dict(columns=PLAIN10,training_metadata_feature_names_are_not_plain10_columns=True,
            normalization='original all49 contemporaneous pool rank; no newly eligible-only normalization or reference-set expansion',
            raw_missing_values_rank_fill_neutral=True,finite_plain10_proves_raw_ohlc_completeness=False,
            sources=[dict(path=p,sha256=PIN[str(R/p)]) for p in code_paths]),
        train_good_embeds_original_eligibility=True,train_good_preserves_original_eligible_full_month_path_rule=True,
        original_common_cohort_unchanged_required=True,original_label_clip_and_center_unchanged_required=True,
        calendar_clock_review=dict(path=str(CALENDAR.relative_to(R)),sha256=digest(CALENDAR),
            start=calendar['start'],end=calendar['end'],sessions=len(sessions),all_training_dates_ends_monthend=True,
            all_query_dates_ends_in_calendar=True,no_price_support_recomputed=True),
        protected_manifest=dict(path=str(PROTECTED.relative_to(R)),sha256=digest(PROTECTED),members=21444,
            manifest_verified=True,all_protected_member_bytes_rehashed_by_this_review=False,
            required_future_checks='Registration and source closure must verify the full protected21444 member hashes'),
        source_npz_decoded=False,training_npz_decoded=False,npz_decoded=False,source_code_imported=False,
        actual_new_features=False,actual_new_forecasts=False,actual_new_labels=False,actual_new_support_counts=False,
        actual_new_model_fits=False,actual_new_orders=False,actual_new_returns=False,actual_source_coverage_recomputed=False,
        inherited_input_pins_reverified=len(inherited),source_sha256=dict(sorted(PIN.items())),input_sha256=dict(sorted(PIN.items())),
        strict_pit=False,observers_modified=False,central_modified=False,frozen_files_modified=False,
        new_strategy_registered=False,host_not_pinned_to_avoid_cycle=True,
        limitations=['Cached all49 query extensions do not certify raw source completeness, missing values may already be neutral ranks.',
            'Training cohort, scaler, original reference clipping and centering must stay fixed; new noneligible labels need separately validated source construction.',
            'Retrospective vendor/alias/2016 leading-gap splices remain inherited research assumptions; no live archived or strict PIT claim.',
            'Only the original49 per-month identities are in scope; the broad alias archive diagnostic is not newly admitted.',
            'No newly supported rows, expanded forecasts, labels, estimates, or strategy performance were computed by this metadata receipt.'],
        implementation=dict(path=str(Path(__file__).relative_to(R)),sha256=implementation))
    with OUT.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,output=str(OUT.relative_to(R)),sha256=digest(OUT),source_pins=len(PIN),npz_decoded=False)),flush=True)

def snapshot_context():
    """Evaluate only the pure contract AST, never import a numerical host/module."""
    need(not CONTEXT.exists(),'Context is write-once')
    text=(R/'work/monthly_ineligible_source227_host.py').read_text();tree=ast.parse(text)
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('PREFIXES','LABEL_CUTOFFS'):
            constants[node.targets[0].id]=ast.literal_eval(node.value)
    funcs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='contract']
    need(len(funcs)==1 and set(constants)=={'PREFIXES','LABEL_CUTOFFS'},'Pure contract not found')
    need(all(isinstance(node.func,ast.Name) and node.func.id in ('dict','list') for node in ast.walk(funcs[0]) if isinstance(node,ast.Call)),'Unexpected impure contract expression')
    scope={'__builtins__':{'dict':dict,'list':list},**constants}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<source-only contract snapshot>','exec'),scope)
    value=scope['contract']()
    with CONTEXT.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(context_path=str(CONTEXT.relative_to(R)),context_sha256=digest(CONTEXT),source_code_imported=False)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--context-sha256');parser.add_argument('--snapshot-context',action='store_true')
    args=parser.parse_args()
    if args.snapshot_context:
        need(args.context_sha256 is None,'Snapshot and review are separate phases');snapshot_context()
    else:
        need(args.context_sha256 is not None,'Explicit final context SHA required');review(args.context_sha256)


"""Source229 byte/JSON metadata review; never decode arrays or run a model."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import ast
import json
import ineligible_source227_source_metadata as base

R=Path(__file__).resolve().parents[1]
OUT=R/'work/early_history229_SOURCE_REVIEW.json'
CONTEXT=R/'work/monthly_early_history229_context.json'
CHECKPOINT=R/'work/after228_research_checkpoint.json'
PRIOR=R/'outputs/monthly_expanded_error228'
SOURCE227=R/'outputs/monthly_ineligible_source227'
REBUILD=base.REBUILD
CONTEXT_SHA='bdcddb7e70b12ede9c42464a3ed0d7be230ec7b467720b0262c3358b2126670f'
CHECKPOINT_SHA='02cdb05a519fec822bc9e88aaee470829ce4ca084a4bdd19b9ee5d0b08a91d99'
PRIOR_PLAN_SHA='8360603efb9116377401a3b45d879b66944df7d10911d8f73338ec6a92813654'
PRIOR_CLOSE_SHA='8d65e817a0c34e79950b957f3263d651b792fbaf6c128681a98f1be1ed59df7f'
PRIOR_REVIEW_SHA='0c04840c9199d4e60419872e548bf62ce7f8505b373d5b922322952c5ac21952'
SOURCE_PLAN_SHA='d2b8eb195a7c54fc61a3acb910653ad3f86dabbbb6d23cdf41c187260ab20ae1'
SOURCE_CLOSE_SHA='d4669e7254e15bbf799dc7f5aca69b6069ffc83094b3d98f839a8c8b897249e0'
SOURCE_REVIEW_SHA='5837808950b5e340d9a5db6cbf52953d1337b928e07b9f003164b5a0d7378929'
SOURCE_METADATA_SHA='4610aacb5941fe67b8496ce607cd2152c642835b761f48b295e74c33d5f370ba'
PREPARED={'FORECASTS.npz','LABELS.npz','META.json','SOURCE_AUDIT.json','INDEPENDENT.json'}
need,read,bind,digest=base.need,base.read,base.bind,base.digest


def contract_ast():
    """Evaluate only literal contract AST; do not import the numerical host."""
    tree=ast.parse((R/'work/monthly_early_history229_host.py').read_text())
    fs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='contract']
    ps=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PREFIXES' for t in n.targets)]
    need(len(fs)==len(ps)==1,'Contract AST schema')
    need(all(isinstance(n.func,ast.Name) and n.func.id in ('dict','list') for n in ast.walk(fs[0]) if isinstance(n,ast.Call)),'Impure contract AST')
    scope={'__builtins__':{'dict':dict,'list':list},'PREFIXES':ast.literal_eval(ps[0].value)}
    exec(compile(ast.Module(body=fs,type_ignores=[]),'<pure229contract>','exec'),scope)
    return scope['contract']()


def review():
    need(not OUT.exists(),'Source review is write-once')
    context=read(CONTEXT,CONTEXT_SHA)
    need(context==contract_ast(),'Host/context contract mismatch')
    checkpoint=read(CHECKPOINT,CHECKPOINT_SHA)
    need(checkpoint['completed_round']==PRIOR.name and checkpoint['completion_sha256']==PRIOR_CLOSE_SHA,'Checkpoint lineage')
    need(checkpoint['completed_strategy_entries']==225 and checkpoint['total_strategy_trials']==1192,'Checkpoint counts')
    for p,h in checkpoint['evidence_sha256'].items():bind(p,h)
    prior_plan=read(PRIOR/'PLAN.json',PRIOR_PLAN_SHA)
    prior_close=read(PRIOR/'COMPLETION.json',PRIOR_CLOSE_SHA)
    prior_review=read(PRIOR/'RESEARCH_REVIEW.json',PRIOR_REVIEW_SHA)
    need(prior_close['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior_close['plan_sha256']==PRIOR_PLAN_SHA,'228 closure')
    need(prior_review['passed'] is True and prior_review['completion_sha256']==PRIOR_CLOSE_SHA,'228 saved review')
    inherited=prior_plan['input_sha256']
    for p,h in inherited.items():bind(p,h)
    source_plan=read(SOURCE227/'PLAN.json',SOURCE_PLAN_SHA)
    source_close=read(SOURCE227/'COMPLETION.json',SOURCE_CLOSE_SHA)
    source_review=read(SOURCE227/'RESEARCH_REVIEW.json',SOURCE_REVIEW_SHA)
    need(source_close['status']=='SOURCE_COMPLETED_INDEPENDENTLY_VERIFIED' and source_close['plan_sha256']==SOURCE_PLAN_SHA,'227 closure')
    need(source_close['source_only'] is True and source_close['strategy_trials']==0 and source_close['new_portfolio_returns'] is False,'227 source scope')
    need(source_review['passed'] is True and source_review['completion_sha256']==SOURCE_CLOSE_SHA,'227 saved review')
    source_metadata=read(R/'work/ineligible_source227_SOURCE_REVIEW.json',SOURCE_METADATA_SHA)
    need(source_metadata['passed'] is True and source_metadata['metadata_only'] is True,'227 metadata proof')
    for p,h in source_metadata['input_sha256'].items():bind(p,h)
    prep=read(SOURCE227/'prepared/COMPLETION.json',source_close['prepared_sha256'])
    need(prep['status']=='COMPLETE' and prep['plan_sha256']==SOURCE_PLAN_SHA and set(prep['artifacts_sha256'])==PREPARED,'227 exact prepared set')
    for n,h in prep['artifacts_sha256'].items():bind(SOURCE227/'prepared'/n,h)
    source_meta=read(SOURCE227/'prepared/META.json')
    meta=read(REBUILD/'MONTHLY_META.json',inherited[str(REBUILD/'MONTHLY_META.json')])
    need(source_meta==meta,'227 original training/current query identities')
    tm=meta['training'];ids=base.axes(tm);base.axes(meta['monthly'],True)
    need(len(tm['dates'])==180 and len(ids)==8820 and len(meta['monthly']['dates'])==122,'Original dimensions')
    early_ids=[i for i,d in enumerate(tm['dates']) if d<'2016-08-17']
    need(early_ids==list(range(67)),'Exact67 early month prefix')
    dates=tm['dates'][:67];ends=tm['ends'][:67];symbols=tm['symbols'][:67]
    need(dates[0]=='2011-01-31' and dates[-1]=='2016-07-29','Early endpoints')
    need(ends[-1]=='2016-08-31' and sum(map(len,symbols))==3283,'Early row/label identity')
    expected_months=[f'{year:04d}-{month:02d}' for year in range(2011,2017) for month in range(1,13) if (year,month)<=(2016,7)]
    need([d[:7] for d in dates]==expected_months,'No early natural month omitted')
    origins=tm['pool_origins'][:67]
    need(len(origins)==67 and set(origins)=={'inherited_pre2016_historical_proxy'},'Inherited proxy scope changed')
    calendar=read(base.CALENDAR,inherited[str(base.CALENDAR)])
    sessions=calendar['sessions'];need(sessions==sorted(set(sessions)),'Calendar unique sorted')
    for d in sessions:base.canonical(d)
    last={d[:7]:d for d in sessions};present=set(sessions)
    for d,e in zip(dates,ends):
        need(d in present and e in present and last[d[:7]]==d and last[e[:7]]==e,'Early official month-end clock')
    ceilings=[]
    for year in range(2011,2017):
        cutoff=f'{year-1:04d}-12-31';start=f'{year-6:04d}-12-31'
        possible=[d for d,e in zip(tm['dates'],tm['ends']) if d>start and e<=cutoff]
        ceilings.append(dict(year=year,cutoff=cutoff,lookback_exclusive_start=start,
            metadata_month_ceiling=len(possible),necessarily_cold_start=len(possible)<18,
            actual_common_cohort_support_computed=False))
    need([r['metadata_month_ceiling'] for r in ceilings]==[0,11,23,35,47,59],'Mature metadata upper bounds')
    raw=prior_plan['raw_source_hashes'];need(len(raw)==64 and raw==source_plan['raw_source_hashes'],'Original raw64 mismatch')
    for p,h in raw.items():bind(p,h)
    parity=read(REBUILD/'PARITY.json',inherited[str(REBUILD/'PARITY.json')]);need(parity['passed'] is True,'169 saved parity')
    fields=source_metadata['source169_saved_parity']['field_proofs']
    need(all(parity['checks']['monthly'][k]==proof for k,proof in fields.items()),'169 unchanged cached field proofs')
    for name in ('TRAIN_INPUTS.npz','MONTHLY_META.json'):bind(REBUILD/name,inherited[str(REBUILD/name)])
    model_path=R/'outputs/monthly_local_linear70/model.py';source_path=R/'outputs/monthly_target_definition70/model.py'
    bind(model_path,inherited[str(model_path)]);bind(source_path,inherited[str(source_path)])
    model_text=model_path.read_text();source_text=source_path.read_text()
    need('MIN_MONTHS = 18' in model_text and 'MIN_OWN_MONTHS = 18' in model_text,'Original minimums')
    need('events = event_good & np.isfinite(event_X).all(axis=1)' in source_text,'Signal-time query guard')
    need('train = usable & (dates > start) & (ends <= cutoff)' in source_text,'Annual mature clock')
    protection=read(base.PROTECTED,inherited[str(base.PROTECTED)])
    need(len(protection)==21444 and prior_close['protected_files']==21444 and source_close['protected_files']==21444,'Protection identity')
    bind(Path(base.__file__),source_metadata['implementation']['sha256']);implementation=bind(Path(__file__))
    result=dict(schema='early_history229_source_metadata_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,metadata_only=True,source_only=True,context_sha256=CONTEXT_SHA,checkpoint_sha256=CHECKPOINT_SHA,
        predecessor228=dict(plan_sha256=PRIOR_PLAN_SHA,completion_sha256=PRIOR_CLOSE_SHA,review_sha256=PRIOR_REVIEW_SHA),
        source227=dict(plan_sha256=SOURCE_PLAN_SHA,completion_sha256=SOURCE_CLOSE_SHA,review_sha256=SOURCE_REVIEW_SHA,
            metadata_review_sha256=SOURCE_METADATA_SHA,prepared_completion_sha256=digest(SOURCE227/'prepared/COMPLETION.json'),
            artifacts_sha256=prep['artifacts_sha256']),
        metadata_identity_review=dict(training_months=180,training_rows=8820,current_query_months=122,current_query_rows=5978,
            early_query_months=67,early_query_rows=3283,first_signal=dates[0],last_signal=dates[-1],last_label_end=ends[-1],
            original_month_major_ids_exact=True,width49=True,early_original_indices=early_ids,
            pool_origin_counts=dict(Counter(origins)),new_identities_added=False,numeric_offsets_decoded=False),
        source169_saved_field_proofs=fields,plain10_definition=source_metadata['plain10_definition'],
        query_mask_contract='finite train_plain10 AND (original train_eligible OR verified295pastclosewarmup)',
        future_label_validity_must_not_gate_queries=True,original_training_common_good_and_four_targets_unchanged=True,
        copied227_labels_role='Risk-error outcomes only, exact early rows; never additional source Ridge training',
        initial_training_clock=dict(minimum_mature_months=18,metadata_only_upper_bounds=ceilings,
            earliest_possible_non_cold_year=2013,actual_earliest_fit_year_not_computed=True),
        annual_replay_contract=dict(prior_december_cutoff=True,five_calendar_years=True,own_minimum18=True,
            full_training_archive_not_available_at_earliest_years=True,cold_starts_must_remain_unavailable=True),
        raw_source_hashes=raw,calendar_sha256=digest(base.CALENDAR),calendar_monthends_verified=True,
        protected_manifest=dict(path=str(base.PROTECTED),sha256=digest(base.PROTECTED),members=21444,
            all_member_bytes_rehashed_here=False,registered_host_and_closer_must_reverify=True),
        source_bytes_reverified=True,source_npz_decoded=False,training_npz_decoded=False,
        numerical_source_code_imported=False,actual_new_forecasts=False,actual_new_features=False,actual_new_labels=False,
        actual_new_support_counts=False,actual_new_model_fits=False,actual_new_orders=False,actual_new_returns=False,
        strategy_trials=0,strict_PIT=False,observers_modified=False,central_modified=False,frozen_files_modified=False,
        host_not_pinned_to_avoid_cycle=True,input_sha256=dict(sorted(base.PIN.items())),
        implementation=dict(path=str(Path(__file__).relative_to(R)),sha256=implementation),
        limitations=['The first67 metadata rows are inherited pre2016 historical proxies, not contemporaneous certified constituent archives.',
            'Finite plain10 may retain neutral-filled raw missing features; price warmup cannot certify all OHLCV feature inputs.',
            '2011 and2012 are structurally cold; metadata ceilings do not establish usable common cohorts or actual predictions.',
            'Retrospective adjustedprice/alias and inherited pool assumptions remain; no strict PIT or archived real-time prediction claim.',
            'Source-only admission creates no strategy or return result; a separate PLAN is required to consume earlier errors.'])
    with OUT.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT.relative_to(R)),sha256=digest(OUT),input_pins=len(base.PIN),early_metadata_rows=3283,npz_decoded=False)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();review()

"""Metadata/hash-only source review for separate-history230.

No numerical modules are imported and no NPZ members are decoded. Source227 and
source229 counts are read from their saved certificates, not recomputed for230.
"""
from pathlib import Path
from datetime import date, datetime, timezone
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'work/separate_history230_SOURCE_REVIEW.json'
CONTEXT=ROOT/'work/monthly_separate_history230_context.json'
CONTEXT_SHA='badb7ce7a17e46c9f44a50e3dd7916f23bc92780610075989f16f223586bf31e'
PRIOR=ROOT/'outputs/monthly_expanded_error228'
PRIOR_PLAN='8360603efb9116377401a3b45d879b66944df7d10911d8f73338ec6a92813654'
PRIOR_CLOSE='8d65e817a0c34e79950b957f3263d651b792fbaf6c128681a98f1be1ed59df7f'
PRIOR_REVIEW='0c04840c9199d4e60419872e548bf62ce7f8505b373d5b922322952c5ac21952'
EXPANDED=ROOT/'outputs/monthly_ineligible_source227'
EARLY=ROOT/'outputs/monthly_early_history229'
LOCAL=ROOT/'outputs/monthly_local_linear70'
OLD=ROOT/'outputs/monthly_identity_transfer206'
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
PREPARED={'FORECASTS.npz','LABELS.npz','META.json','SOURCE_AUDIT.json','INDEPENDENT.json'}
PREPARED206={'FORECASTS.npz','RANKS_DELTAS.npz','MODEL_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
EXTERNAL={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
EXPECTED={
 227:dict(directory=EXPANDED,plan='d2b8eb195a7c54fc61a3acb910653ad3f86dabbbb6d23cdf41c187260ab20ae1',
  completion='d4669e7254e15bbf799dc7f5aca69b6069ffc83094b3d98f839a8c8b897249e0',
  prepared='ebaab972da636d59aaf186f52016a7a41494cf20bde62b1dc664165f1a095bb2',
  review='5837808950b5e340d9a5db6cbf52953d1337b928e07b9f003164b5a0d7378929'),
 229:dict(directory=EARLY,plan='7cd73009ffaae9c6ab4418221049306066da94dc48ca9223f13ab9d24335f1e6',
  completion='32920689375b500ff645d243181b8984dbeca7ce686749b5ba083fc9693377cd',
  prepared='927787a77d99c738cf979edda2bc872b18c408ccd390a73fe973097417b8994a',
  review='c600d0ba429dc08a15748e1c869b755155e31a11bc320f0fd32b8a9e5d0d4157')}
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
    p=resolve(value)
    if str(p) in PINS:
        need(expected is None or expected==PINS[str(p)],'Conflicting immutable SHA: '+str(p))
        return PINS[str(p)]
    actual=sha(p)
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

def source_package(number):
    spec=EXPECTED[number];root=spec['directory']
    plan=read(root/'PLAN.json',spec['plan']);close=read(root/'COMPLETION.json',spec['completion'])
    prepared=read(root/'prepared/COMPLETION.json',spec['prepared']);saved=read(root/'RESEARCH_REVIEW.json',spec['review'])
    need(close['status']=='SOURCE_COMPLETED_INDEPENDENTLY_VERIFIED' and close['plan_sha256']==prepared['plan_sha256']==spec['plan'] and close['prepared_sha256']==spec['prepared'],'Terminal source chain '+str(number))
    need(close['source_only'] is True and close['original_strategy_unchanged'] is True and close['strategy_trials']==0 and close['new_portfolio_returns'] is False and close['observers_modified'] is False and close['protected_files']==21444,'Source-only closure scope '+str(number))
    need(prepared['status']=='COMPLETE' and prepared['source_only'] is True and prepared['strategy_trials']==0 and prepared['portfolio_returns'] is False and set(prepared['artifacts_sha256'])==PREPARED,'Exact source five-member manifest '+str(number))
    need(saved['passed'] is True and saved['source_only'] is True and saved['strategy_trials']==0 and saved['source_npz_decoded'] is False and saved['portfolio_returns_computed'] is False and saved['plan_sha256']==spec['plan'] and saved['completion_sha256']==spec['completion'] and saved['prepared_sha256']==spec['prepared'],'Independent saved source review '+str(number))
    for mapping in (plan['input_sha256'],saved['input_sha256']):
        for path,digest in mapping.items():bind(path,digest)
    for name,digest in prepared['artifacts_sha256'].items():bind(root/'prepared'/name,digest)
    return dict(plan=plan,close=close,prepared=prepared,saved=saved,
        meta=read(root/'prepared/META.json'),audit=read(root/'prepared/SOURCE_AUDIT.json'),
        independent=read(root/'prepared/INDEPENDENT.json'))


def review():
    need(not OUTPUT.exists(),'Metadata receipt is write-once')
    context=read(CONTEXT,CONTEXT_SHA)
    need(context['round']==230 and context['name']=='monthly_separate_history230','Prospective230 context')
    for path,digest in context['evidence_sha256'].items():bind(path,digest)
    prior=read(PRIOR/'PLAN.json',PRIOR_PLAN);prior_close=read(PRIOR/'COMPLETION.json',PRIOR_CLOSE)
    prior_saved=read(PRIOR/'RESEARCH_REVIEW.json',PRIOR_REVIEW)
    need(prior_close['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior_close['plan_sha256']==PRIOR_PLAN and prior_saved['passed'] is True,'Closed228 strategy predecessor')
    for path,digest in prior['input_sha256'].items():bind(path,digest)
    expanded=source_package(227);early=source_package(229)
    need(all(early['plan']['input_sha256'].get(p)==h for p,h in prior['input_sha256'].items()),'229 inherited228 source pins unchanged')
    raw=read(LOCAL/'PLAN.json')['raw_source_hashes']
    need(len(raw)==64 and raw==prior['raw_source_hashes']==expanded['plan']['raw_source_hashes']==early['plan']['raw_source_hashes']==read(OLD/'PLAN.json')['raw_source_hashes'],'Original64 raw map')
    for path,digest in raw.items():bind(path,digest)
    protected=read(PROTECTED)
    need(len(protected)==21444 and prior_close['protected_files']==21444,'Full21444 protected archive lineage')
    old=read(OLD/'prepared/COMPLETION.json')
    need(old['status']=='COMPLETE' and set(old['artifacts_sha256'])==PREPARED206,'Original206 prepared7')
    for name,digest in old['artifacts_sha256'].items():bind(OLD/'prepared'/name,digest)
    original=read(REBUILD/'MONTHLY_META.json')
    need(expanded['meta']==original and original['monthly']==read(LOCAL/'META.json')==read(OLD/'prepared/META.json'),'Exact current122 and full180training identities')
    tm=original['training'];qm=original['monthly'];hm=early['meta']['history']
    expected={k:(v[:67] if isinstance(v,list) and len(v)==len(tm['dates']) else v) for k,v in tm.items()}
    need(hm==expected and early['meta']['current_query_axis_unchanged'] is True and early['meta']['training_reference']==str(REBUILD/'MONTHLY_META.json'),'Exact original first67 historical metadata')
    training=identity(tm);current=identity(qm,True);history_early=identity(hm)
    need(len(training)==8820 and len(current)==5978 and len(history_early)==3283,'Original180/122/67 by49 dimensions')
    need(history_early==training[:3283] and hm['dates'][0]=='2011-01-31' and hm['dates'][-1]=='2016-07-29','Original early TRAIN prefix, no appended labels')
    need(hm['dates'][-1]<qm['dates'][0]=='2016-08-17' and not(set(history_early)&set(current)),'Separate query axes no overlap')
    combined_dates=hm['dates']+qm['dates']
    need(combined_dates==sorted(set(combined_dates)) and len(combined_dates)==189 and combined_dates[67]=='2016-08-17','Metadata-only189group historical axis')
    sessions=read(CALENDAR)['sessions'];need(sessions==sorted(set(sessions)),'Original official session ordering')
    for d in sessions:day(d)
    official=set(sessions);last={d[:7]:d for d in sessions}
    for d,_,e in training+current:
        need(d in official and e in official and (d=='2016-08-17' or last[d[:7]]==d),'Official source signal clock')
        need(e=='2026-09-24' or last[e[:7]]==e,'Official source outcome clock')
    ea,ei=early['audit'],early['independent'];xa,xi=expanded['audit'],expanded['independent']
    need(early['close']['forecasts']==ea['forecasts'] and ea['forecasts']['counts']==ei['counts']==early['saved']['forecasts'],'Saved229 forecast audit linkage')
    need(ea['copied_label_exact'] is True and ea['copied_label_rows']==3283 and ea['source227_label_sha256']==expanded['prepared']['artifacts_sha256']['LABELS.npz'],'229 labels exact source227 early prefix certificate')
    for key in ('original_arrays_unchanged','no_new_orders','no_portfolio_returns'):need(ea[key] is True,'229 saved scope '+key)
    need(ea['new_history_months']==67 and ea['original_trading_queries']==122 and ea['strict_PIT'] is False,'Separate source229 scope')
    need(len(early['saved']['independent_flags'])==20 and all(ei[k] is True for k in early['saved']['independent_flags']),'22920 independent flags')
    need(ei['numerical_tolerance']==1e-12 and ei['producer_modules_imported'] is False and ei['shared_numerical_primitives'] is False and ei['computes_returns'] is False,'229 independent augmentedLS scope')
    need([x['prefix'] for x in ea['forecast_prefixes']]==[0,12,24,48,67],'229 source physical prefixes')
    for name in ('forecasts','labels'):
        saved_counts=expanded['saved'][name]['counts'] if name=='labels' else expanded['saved'][name]
        need(expanded['close'][name]==xa[name] and xi[name]['passed'] is True and
            xi[name]['counts']==xa[name]['counts']==saved_counts,'Saved227 '+name+' lineage')
    need(xa['forecasts']['original_training_unchanged'] is True and xa['forecasts']['original_forecasts_copied'] is True and xa['forecasts']['reranked'] is False,'227 source model/feature preservation')
    for key in ('original_targets_copied','original_common_good_preserved','original_centers_only','added_rows_never_train'):
        need(xa['labels'][key] is True,'227 original center/training guarantee '+key)
    proposal=read(ROOT/'work/separate_history230_PROPOSAL.json')
    need(proposal['strategy_trials_before_registration']==0 and proposal['source_fits_in230']==0 and proposal['source_training_changed'] is False and proposal['current_pool_changed'] is False and proposal['current_features_changed'] is False and proposal['observer_writes'] is False,'Prospective separate-history proposal scope')
    need(proposal['created_utc']<early['close']['created_utc'],'Proposal predates229 source closure; full chronological result claim comes from its saved evidence')
    before=CENTRAL.read_bytes();state=json.loads(before)
    need(len(state['completed'])==225 and sum(x['trials'] for x in state['completed'])==1192 and state['completed'][-1]['directory']==PRIOR.name,'Current225entries1192trials predecessor')
    need(not state['registered_queue'] and state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Empty registration queue/original incumbent')
    for number in (227,229):
        spec=EXPECTED[number];entries=[r for r in state.get('source_experiments',[]) if r['directory']==spec['directory'].name]
        need(len(entries)==1 and entries[0]['strategy_trials']==0 and entries[0]['completion']['sha256']==spec['completion'] and Path(entries[0]['completion']['path']).resolve()==spec['directory']/'COMPLETION.json','Separate central source'+str(number))
    code={}
    for name in ('work/monthly_joint_error226_host.py','work/monthly_joint_error226_model.py','work/monthly_joint_error226_independent.py',
        'work/monthly_expanded_error228_host.py','work/monthly_expanded_error228_adapter.py','work/monthly_expanded_error228_independent.py',
        'work/monthly_ineligible_source227_host.py','work/monthly_ineligible_source227_model.py','work/monthly_ineligible_source227_independent.py',
        'work/monthly_early_history229_host.py','work/monthly_early_history229_model.py','work/monthly_early_history229_independent.py',
        'work/early_history229_saved_review.py'):
        code[name]=bind(ROOT/name)
    implementation=bind(Path(__file__))
    def package_record(number,p):
        spec=EXPECTED[number]
        return dict(plan_sha256=spec['plan'],completion_sha256=spec['completion'],prepared_sha256=spec['prepared'],saved_review_sha256=spec['review'],
            prepared_artifacts_sha256=p['prepared']['artifacts_sha256'],source_only=True,strategy_trials=0)
    result=dict(schema='separate_history230_source_metadata_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,source_only=True,metadata_only=True,
        source_npz_decoded=False,training_npz_decoded=False,npz_decoded=False,source_models_imported=False,
        actual_support_computed=False,actual_joint_history_support_computed=False,actual_features_computed=False,actual_risks_computed=False,
        actual_labels_computed=False,actual_forecasts_computed=False,actual_returns_computed=False,actual_orders_computed=False,actual_model_fits=0,
        observers_modified=False,central_modified=False,frozen_files_modified=False,source_bytes_reverified=True,metadata_axes_verified=True,calendar_clock_verified=True,
        source_sha256=dict(sorted(PINS.items())),input_sha256=dict(sorted(PINS.items())),raw_source_hashes=raw,
        context_sha256=CONTEXT_SHA,evidence_sha256=context['evidence_sha256'],host_not_pinned_to_avoid_cycle=True,strict_pit=False,
        predecessor228=dict(plan_sha256=PRIOR_PLAN,completion_sha256=PRIOR_CLOSE,saved_review_sha256=PRIOR_REVIEW),
        expanded227=package_record(227,expanded),early229=package_record(229,early),
        source_identity=dict(query_months=122,query_width=49,query_rows=5978,training_months=180,training_width=49,training_rows=8820,
            early_history_months=67,early_history_rows=3283,combined_history_months=189,combined_history_rows=9261,initial_partial_history_index=67,
            early_metadata_is_exact_training_prefix=True,metadata227_equals169=True,monthly_metadata_equals206_and_local67=True,
            exact_signal_security_label_end=True,current_identity_order_preserved=True,numeric_offsets_not_decoded=True,
            identity_scope='Original canonical security IDs and inherited aliases/proxy pre2016 pools; no new issuer or strictPIT proof'),
        original_source_guarantees=dict(original_training_cohort_and_weights_unchanged=True,source227_original_forecasts_exact_copy=True,
            original_targets_and_centers_preserved=True,early_labels_are_exact_existing227_prefix=True,no_labels_appended_or_duplicated_in_training=True,
            new_source_model_refits_required_in230=False,query_features_recomputed=False,query_feature_reranking=False,
            early_and_current_history_are_separate_from_current_trading_axis=True),
        saved_source_counts=dict(basis='Previously saved source227/source229 counts only; no actual230 complete-history support computed',
            source227_forecasts=xa['forecasts']['counts'],source227_labels=xa['labels']['counts'],source229_forecasts=ea['forecasts']['counts']),
        interfaces=dict(forecasts227_keys=['raw_source_log','raw_local_constant','raw_local_linear','available','local_available','expansion_allowed'],
            forecasts229_keys=['raw_source_log','raw_local_constant','raw_local_linear','available','local_available','query_eligible','expansion_allowed'],
            labels_common_keys=['target_simple_excess','target_log_excess','target_vol_scaled','target_loss_utility','common_good','added','vol63','dates','ends','offsets'],
            risk_prediction_key='raw_local_linear',risk_target_key='target_log_excess',
            historical_use_early='(available AND query_eligible) OR (available AND NOT query_eligible AND expansion_allowed); must equal available',
            historical_use_current='originalU OR newly_available227; unchanged228 semantics',current_available_shape=[122,49],
            field_schema_basis='Frozen source writer and saved JSON; NPZ schemas/values are not decoded here'),
        required_prepare_checks=['Recheck exact227/229 label-prefix dtype, shape, bytes including vol63; do not append early outcomes to full training.',
            'Keep current27 arrays and122query identity/pre/eligibility fixed. Concatenate early67 and current122 only into detached history inputs.',
            'Check early query union from saved query_eligible/expansion_allowed/available, and retain original228 current historical mask.',
            'Skip initial2016-08-17 wherever it occurs in the separate history; monthly prefix size is not a history prefix index.',
            'Use exact matured security/date/end joins and whole-domain commonH, priorDecember cutoff/five years/min18 unchanged.',
            'No source-model refits, feature/rank rebuilding, outcome recentering, future-label query gate or local_available eligibility.',
            'Forecast availability and earlier query count do not certify actual230 common-history support or improve strategy performance.'],
        protected_manifest=dict(path=str(PROTECTED.relative_to(ROOT)),sha256=sha(PROTECTED),members=21444,
            registered_and_closed_full_checks=True,all_members_rehashed_by_this_review=False),
        central_snapshot=dict(sha256=hashlib.sha256(before).hexdigest(),persistent_input=False,completed_strategies=225,trials=1192,
            last_strategy=PRIOR.name,source227_separate=True,source229_separate=True,registered_queue_empty=True),
        reviewed_code_sha256=code,implementation=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=implementation),
        limitations=context['limitations']+['本回执只核来源字节和保存metadata；实际230共同支持、预测误差、风险、排名和收益仍未计算。',
            '229 copied-label equality依赖冻结host逐字段字节核验及完成证书，本次不解码标签NPZ。',
            '21444成员全量验证由冻结注册/收尾回执证明，本次仅重核manifest与成员数量。'])
    need(CENTRAL.read_bytes()==before,'Central changed during metadata review')
    with OUTPUT.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,sha256=sha(OUTPUT),source_pins=len(PINS),raw_sources=len(raw),training_rows=len(training),current_rows=len(current),
        early_rows=len(history_early),historical_groups=len(combined_dates),actual_support_computed=False,npz_decoded=False)),flush=True)


if __name__=='__main__':review()

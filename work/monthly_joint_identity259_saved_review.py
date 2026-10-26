"""Saved review of terminal259 saved evidence after explicit authorization.

Run via run_local.sh with the four immutable phase digests. Standard library
only: no strategy/producer/oracle imports, array decoding, prices, volume,
feature mathematics or account replay. Saved DAILY scalar metrics are independently recomputed;
statistics and guard comparisons use already frozen rules. One new report.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import platform

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_joint_identity259'
PRIOR=ROOT/'outputs/monthly_downside_basket258'
LOCAL=ROOT/'outputs/monthly_local_linear70'
REPORT=ROOT/'work/monthly_joint_identity259_RESULT_REVIEW.json'
CONTEXT=ROOT/'work/monthly_joint_identity259_context.json'
TESTS=ROOT/'work/monthly_joint_identity259_TESTS.json'
CODE_REVIEW=ROOT/'work/monthly_joint_identity259_CODE_REVIEW.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
PREVIOUS_REVIEW=ROOT/'work/monthly_downside_basket258_RESULT_REVIEW.json'
ARMS=('incumbent_local67','sequential_identity','joint_identity')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
SLICES={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FORECASTS.npz','LABELS.npz','MODEL_AUDIT.json','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
PREFIXES=(0,1,53,88,122)
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
FIXED={
 'work/monthly_joint_identity259_context.json':'0b71847a2fbfc3f44e5b0e8b86b2ecfcfaaf15a94d2fa88406f8995a46471527',
 'work/monthly_joint_identity259_IMPLEMENTATION_CONTRACT.json':'72565c3821be025ae22a2e6fc3827a8129eea002b08f135fbb93705779e37cf4',
 'work/after258_joint_source_review.json':'eee0453844527feaf5f1282feeac2785fdce887ccf6005a35d0121d5793b3b7e',
 'work/monthly_downside_basket258_RESULT_REVIEW.json':'2c099c759ba2edda6d788be3ff07521ba4b44d299d4878aeb85de85890eeafcc',
 'work/ptr169_results_independent.py':'6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59',
 'work/monthly_ownsector_gate238_host.py':'94f70ab8c0d9fc1cd891bf1ce3eaa30212a54ef4f9cdb1eb4e237bd82bed30a5'}
TERMINALS=OUT/'PHASE_TERMINALS.json'
EXPECTED_FORMAL_TESTS=97
ACCOUNT_ATOL=1e-7
ACCOUNT_RTOL=2e-11
PERIOD_FEE_ATOL=1e-10
# Existing evidence: ptr169_results_independent.py:37-41,85-91 (saved-stat
# comparison); previous250 RESULT_REVIEW.accounting_review.comparison_rule
# and incumbent_reproduction.period_fee_exception (ONLY cost HOLDINGS.fee <=1e-10).


CONTRACT=ROOT/'work/monthly_joint_identity259_IMPLEMENTATION_CONTRACT.json'
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
NUMERICAL_VERIFICATION=dict(continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.0,selection_tolerance=0.0,independent_numeric_system='augmented_least_squares_SVD_and_canonical_normal_equations')

def need(ok,message):
    if not ok:raise ValueError(message)


def finite(v):return type(v) in (int,float) and math.isfinite(v)


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def local(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    need(p.is_relative_to(ROOT) and 'CloudStorage' not in p.parts,'Only local evidence')
    need(p!=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json','Do not bind mutable central')
    return p


def canonical_source_map(mapping):
    """Path spelling equivalence only; exactly64 complete path/digest pairs.

    No files are opened, no basename/suffix matching, and no changed digest
    can pass. Reject ambiguous aliases even when their digests agree.
    """
    need(type(mapping) is dict and len(mapping)==64,'Exactly64 source-map entries')
    result={}
    for key,value in mapping.items():
        need(type(key) is str and bool(key),'Nonempty source-map path')
        raw=Path(key)
        need('CloudStorage' not in raw.parts and '..' not in raw.parts,'Forbidden raw source-map path')
        need(not raw.is_absolute() or raw.is_relative_to(ROOT),'Absolute source path must begin at current research root')
        path=local(key)
        need(path!=ROOT,'Source-map key is a file path')
        canonical=str(path.relative_to(ROOT))
        need(canonical not in result,'Canonical source-map collision')
        need(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Canonical bare SHA256 source value')
        result[canonical]=value
    return result

class Evidence:
    def __init__(self):self.pins={}
    def pin(self,p,want=None):
        p=local(p);got=digest(p);need(want is None or got==want,'Digest mismatch: '+str(p))
        key=str(p.relative_to(ROOT));need(key not in self.pins or self.pins[key]==got,'Evidence changed: '+key)
        self.pins[key]=got;return got
    def read(self,p,want=None):
        p=local(p);need(p.suffix=='.json','JSON only; never decode arrays/raw bars');self.pin(p,want)
        def invalid(x):raise ValueError('Nonfinite JSON constant '+x)
        return json.loads(p.read_text(),parse_constant=invalid)
    def registered(self,plan,p):
        p=local(p)
        # Compare path spellings only: inherited pins can include external runtime
        # metadata. This reader never opens or rehashes those external files.
        found=[]
        for k,h in plan['input_sha256'].items():
            key=Path(k);key=(key if key.is_absolute() else ROOT/key).resolve()
            if key==p:found.append(h)
        need(len(found)>=1 and len(set(found))==1,'Missing/conflicting registered pin: '+str(p));return found[0]
    def phase(self,folder,want,plan_sha,names):
        c=self.read(folder/'COMPLETION.json',want)
        need(c['status']=='COMPLETE' and c['plan_sha256']==plan_sha and set(c['artifacts_sha256'])==names,'Exact phase manifest')
        need({p.name for p in folder.iterdir()}==names|{'COMPLETION.json'},'Unexpected/missing phase file')
        for name,h in c['artifacts_sha256'].items():
            need(Path(name).name==name,'Artifact traversal');self.pin(folder/name,h)
        return c


def constants(path,names,seed=None):
    """Parse only literal schema tuples from verified source; never import it."""
    found=dict(seed or {})
    def value(n):
        if isinstance(n,ast.Constant):return n.value
        if isinstance(n,ast.Name):return found[n.id]
        if isinstance(n,(ast.Tuple,ast.List)):return tuple(value(x) for x in n.elts)
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Add):return value(n.left)+value(n.right)
        raise ValueError('Nonliteral schema')
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in names:found[target.id]=value(node.value)
    need(set(names)<=set(found),'Missing frozen schema constants');return {k:found[k] for k in names}



def digest_record(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha_value(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def integer(value,label,expected=None):
    need(type(value) is int and value>=0,label+' nonnegative native int')
    if expected is not None:need(value==expected,label+' exact value')


def counts_check(value,fields,label):
    need(type(value) is dict and set(value)==set(fields),label+' exact fields')
    for key,item in value.items():integer(item,label+'/'+key)


def solve_counts(value,label):
    counts_check(value,('pooled_log_solves','local_linear_solves','total_regression_solves'),label)
    need(value['total_regression_solves']==value['pooled_log_solves']+value['local_linear_solves'],label+' sum')


def producer_counts(value,label):
    counts_check(value,('target_fits','log_refits','local_evaluations','local_solves','actual_regression_solves'),label)
    need(value['target_fits']==4*value['log_refits'] and value['local_solves']<=value['local_evaluations'],label+' fit scope')
    need(value['actual_regression_solves']==value['target_fits']+value['log_refits']+value['local_solves'],label+' solve sum')


def model_counts_check(value,n):
    ind=SimpleNamespace(**SCHEMAS,**FLAGS)
    counts_check(value,ind.MODEL_COUNT_FIELDS,'Model counts')
    need(value['valid_training_rows']<=value['training_rows'],'Model cohort counts')
    need(value['control_target_fits']==4*value['control_log_refits'] and value['control_local_solves']<=value['control_local_evaluations'],'Original model solve scope')
    need(value['annual_design_builds']==value['control_log_refits'],'One centered dummy design per original annual context')
    need(value['diagnostic_pooled_fits']==value['candidate_pooled_fits']==0,'No separate new pooled solve; joint global refit counted as joint solve')
    need(value['diagnostic_dummy_solves']==value['candidate_joint_solves'],'Matched active annual solve cohort')
    need(sum(value[k] for k in ('no_qualified_columns_years','no_qualified_query_years','zero_centered_design_years','candidate_joint_solves'))==value['annual_design_builds'],'Mutually exclusive annual fit/copy statuses')
    need(value['qualified_query_rows']+value['cold_query_rows']==value['control_local_evaluations'],'Complete original query cohort')
    need(value['candidate_joint_solves']<=value['qualified_query_rows'] and value['candidate_joint_solves']<=value['qualified_security_columns'],'Active solve has qualified query/column')
    need(value['actual_new_model_solves']==value['diagnostic_dummy_solves']+value['candidate_joint_solves'],'New solve sum')
    need(value['actual_regression_solves']==sum(value[k] for k in ('control_target_fits','control_log_refits','control_local_solves','actual_new_model_solves')),'All producer solves')
    if n==0:need(not any(value.values()),'Empty prefix no training rows or model solves')

def model_preservation(model,labels,n):
    ind=SimpleNamespace(**SCHEMAS,**FLAGS)
    need(model['schema']=='monthly_joint_identity259_model_audit_v1','Model audit schema')
    c=model['counts'];model_counts_check(c,n)
    keys=('original_row_order_preserved','original_common_good_preserved','all_targets_preserved','diagnostic_source_log_copied','candidate_source_log_copied','diagnostic_local_constant_copied','candidate_local_constant_copied','availability_preserved','local_support_preserved','common_design_weights_penalties','security_intercepts_only','candidate_global_coefficients_refit')
    for key in keys:need(model.get(key) is True,'Model preservation:'+key)
    need(model.get('computes_returns') is False,'Models before returns')
    expected={'schema','rows','valid_rows','target_keys','target_sha256','common_good_sha256','labels_unchanged','common_good_unchanged','original_row_order_preserved','recentered','raw_target_reconstructed'}
    need(type(labels) is dict and set(labels)==expected and labels['schema']=='monthly_joint_identity259_labels_preserved_v1','Exact unchanged-label proof schema')
    integer(labels['rows'],'Preserved label rows',c['training_rows']);integer(labels['valid_rows'],'Preserved valid rows',c['valid_training_rows'])
    need(labels['target_keys']==list(ind.TARGETS) and type(labels['target_sha256']) is dict and set(labels['target_sha256'])==set(ind.TARGETS),'All four unchanged targets')
    for digest in list(labels['target_sha256'].values())+[labels['common_good_sha256']]:
        need(type(digest) is str and len(digest)==64 and all(x in '0123456789abcdef' for x in digest),'Canonical target/good SHA')
    for key in ('labels_unchanged','common_good_unchanged','original_row_order_preserved'):need(labels[key] is True,'Label preservation:'+key)
    for key in ('recentered','raw_target_reconstructed'):need(labels[key] is False,'No new label construction:'+key)

def receipt(check,n,flags,kind,*,width=None):
    ind=SimpleNamespace(**SCHEMAS,**FLAGS)
    need(type(check) is dict,'Independent receipt object')
    kind={'model':'models'}.get(kind,kind)
    standard={'control':ind.CONTROL_FLAGS,'models':ind.MODEL_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,'selection':ind.SELECTION_FLAGS}
    need(kind in standard,'Known receipt kind');integer(n,'Requested events');integer(check.get('checked_events'),'Independent events',n)
    need(check.get('passed') is True and all(check.get(k) is True for k in tuple(flags)+tuple(standard[kind])),'Independent flags:'+kind)
    need(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Oracle independence:'+kind)
    def flag(key):need(check.get(key) is True,'Complete independent proof:'+kind+'/'+key)
    def tolerance(key,value):
        actual=check.get(key);need(type(actual) is float and math.isfinite(actual) and actual==value,'Exact prospective tolerance:'+kind+'/'+key)
    if kind in ('control','models'):
        for key in ('continuous_atol','continuous_rtol','rank_tolerance'):tolerance(key,NUMERICAL_VERIFICATION[key])
        flag('exact_discrete');flag('full_control_supplied')
        cc=check.get('independent_canonical_solve_counts')
        counts_check(cc,('pooled_log_solves','sequential_dummy_solves','joint_block_solves','total_regression_solves'),'Independent canonical solves')
        need(cc['total_regression_solves']==sum(cc[k] for k in ('pooled_log_solves','sequential_dummy_solves','joint_block_solves')),'Canonical solve sum')
        integer(check.get('independent_design_builds'),'Independent centered designs')
    if kind=='control':
        need(check.get('candidate_labels_computed') is False and check.get('candidate_models_computed') is False,'Control precedes both new fits')
        integer(check.get('episode_calls'),'Control episode calls',0)
        pc=check.get('producer_equivalent_fit_counts');sc=check.get('independent_solve_counts')
        producer_counts(pc,'Control producer equivalent');solve_counts(sc,'Control actual SVD')
        need(sc['pooled_log_solves']==pc['log_refits'] and sc['local_linear_solves']==pc['local_solves'],'Actual control SVD scope')
        need(not any(cc.values()) and check['independent_design_builds']==0,'Original gate no new canonical solves/designs')
        if n==0:need(not any(pc.values()) and not any(sc.values()),'Empty control no fits')
    elif kind=='models':
        for key in ('full_bundle_supplied','full_labels_supplied','full_diagnostic_supplied','full_candidate_supplied'):flag(key)
        need(check.get('independent_numeric_system')==NUMERICAL_VERIFICATION['independent_numeric_system'],'Independent systems declaration')
        c=check.get('counts');model_counts_check(c,n)
        counts_check(check.get('producer_equivalent_counts'),ind.MODEL_COUNT_FIELDS,'Producer equivalent counts')
        need(check['producer_equivalent_counts']==c,'Equivalent count binding')
        sc=check.get('independent_solve_counts');need(type(sc) is dict and set(sc)=={'control','sequential','joint','total_regression_solves'},'Full actual SVD ledger')
        solve_counts(sc['control'],'Actual control SVD')
        need(sc['control']['pooled_log_solves']==c['control_log_refits'] and sc['control']['local_linear_solves']==c['control_local_solves'],'Control actual solve scope')
        for role,key,source in [('sequential','dummy_solves','diagnostic_dummy_solves'),('joint','block_solves','candidate_joint_solves')]:
            counts_check(sc[role],(key,'total_regression_solves'),role+' actual SVD')
            need(sc[role][key]==sc[role]['total_regression_solves']==c[source],'New actual SVD scope:'+role)
        integer(sc['total_regression_solves'],'Independent SVD total',sum(sc[r]['total_regression_solves'] for r in ('control','sequential','joint')))
        need(cc['pooled_log_solves']==c['control_log_refits'] and cc['sequential_dummy_solves']==c['diagnostic_dummy_solves'] and cc['joint_block_solves']==c['candidate_joint_solves'],'Canonical rechecks actually counted')
        integer(check['independent_design_builds'],'Independent designs',c['annual_design_builds'])
    else:
        for key in ('full_scores_supplied','full_pre_scores_supplied'):flag(key)
        tolerance('numerical_tolerance',0.)
        integer(check.get('ordered_choices_checked'),'Ordered choices',n if kind=='incumbent' else 3*n)
        if kind=='incumbent':need(check.get('candidate_selection_computed') is False,'Original decision before new ranks')
        else:
            for key in ('full_bundle_supplied','full_audit_supplied'):flag(key)
            c=check.get('counts');counts_check(c,ind.COUNT_FIELDS,'Decision counts')
            need(c['events']==n and c['symbols_per_event']>0 and c['initial_partial_events']+c['formal_events']==n,'Decision scope')
            if width is not None:integer(c['symbols_per_event'],'Exact decision width',width)
            need(c['incumbent_episode_calls']==c['diagnostic_episode_calls']==c['candidate_episode_calls']==1 and c['new_model_fits']==c['predictive_model_fits']==0,'Decision-only original episode once per arm including n0')

def fit_binding(control,models,model_audit,n):
    c=model_audit['counts'];model_counts_check(c,n)
    need(models['counts']==c,'Independent/producer model counts binding')
    need(control['independent_solve_counts']==models['independent_solve_counts']['control'],'Repeated independent control solve scope')
    pc=control['producer_equivalent_fit_counts']
    for key in ('target_fits','log_refits','local_evaluations','local_solves'):
        need(pc[key]==c['control_'+key],'Control producer-equivalent fit count binding')


def execution_accounting(e):
    invocations=[('main',122,e['control'],e['models'],e['main_model_counts'])]
    invocations += [('physical',v['prefix'],v['control'],v['model'],v['model_counts']) for v in e['physical_prefixes'] if v['prefix']!=122]
    entries=[]
    for scope,n,control,models,audit in invocations:
        fit_binding(control,models,audit,n)
        a=audit['counts'];svd_control=control['independent_solve_counts']['total_regression_solves'];svd_model=models['independent_solve_counts']['total_regression_solves']
        canonical_control=control['independent_canonical_solve_counts']['total_regression_solves'];canonical_model=models['independent_canonical_solve_counts']['total_regression_solves']
        entries.append(dict(scope=scope,events=n,producer_regression_solves=a['actual_regression_solves'],producer_new_model_solves=a['actual_new_model_solves'],producer_sequential_dummy_solves=a['diagnostic_dummy_solves'],producer_joint_block_solves=a['candidate_joint_solves'],independent_control_gate_svd_solves=svd_control,independent_model_check_svd_solves=svd_model,independent_control_gate_canonical_solves=canonical_control,independent_model_check_canonical_solves=canonical_model,total_regression_solves=a['actual_regression_solves']+svd_control+svd_model+canonical_control+canonical_model,producer_centered_design_builds=a['annual_design_builds'],independent_centered_design_builds=control['independent_design_builds']+models['independent_design_builds']))
    keys=tuple(k for k in entries[0] if k not in ('scope','events'))
    return dict(schema='monthly_joint_identity259_fit_execution_accounting_v1',model_invocations=len(entries),physical122_reuses_main=True,entries=entries,totals={k:sum(v[k] for v in entries) for k in keys},decision_only_fits=0,centered_design_builds_are_predictive_fits=False)

def preparation_evidence(e,model_audit,decision_audit):
    ind=SimpleNamespace(**SCHEMAS,**FLAGS)
    kinds=[('control',ind.CONTROL_FLAGS),('models',ind.MODEL_FLAGS),('incumbent',ind.INCUMBENT_FLAGS),('selection',ind.SELECTION_FLAGS)]
    for key,flags in kinds:receipt(e[key],122,flags,key,width=49)
    need(e['all123_order_prefixes'] is True and e['original_arrays_unchanged'] is True and e['original_training_unchanged'] is True,'Preservation proofs')
    need(e['main_model_counts']==model_audit['model'],'Saved main model audit binding')
    model_preservation(model_audit['model'],model_audit['labels'],122)
    fit_binding(e['control'],e['models'],model_audit['model'],122)
    need(decision_audit['schema']=='monthly_joint_identity259_decisions_v1' and len(decision_audit['events'])==122 and decision_audit['counts']==e['selection']['counts'],'Saved decision audit scope')
    need(decision_audit['incumbent_gate']==e['incumbent'],'Saved pre-candidate incumbent receipt')
    for key in ('original_arrays_unchanged','original_eligibility_unchanged','original_core_support_preserved','single_original_episode_per_arm','full_original_local_quarter','partial_exact'):
        need(decision_audit[key] is True,'Decision preservation:'+key)
    need(decision_audit['computes_returns'] is False,'Decisions before outcomes')
    cached=e['cached_prefixes'];physical=e['physical_prefixes']
    need(type(cached) is list and len(cached)==123,'All cached prefixes')
    for n,v in enumerate(cached):
        integer(v['prefix'],'Cached prefix',n);need(v['exact'] is True,'Exact cached prefix')
        receipt(v['incumbent'],n,ind.INCUMBENT_FLAGS,'incumbent')
        receipt(v['selection'],n,ind.SELECTION_FLAGS,'selection',width=49)
    need(type(physical) is list and len(physical)==len(PREFIXES),'All model prefixes')
    for n,v in zip(PREFIXES,physical):
        integer(v['prefix'],'Physical prefix',n);need(v['exact'] is True,'Exact physical prefix')
        for key,flags in kinds:receipt(v['model' if key=='models' else key],n,flags,key,width=49)
        fit_binding(v['control'],v['model'],v['model_counts'],n)
        if n==122:
            need(v['model_recomputed'] is False and v.get('reuses_main_verified_full') is True,'Physical122 reuse, no new fit')
            for key,_ in kinds:need(v['model' if key=='models' else key]==e[key],'Physical122 main receipt reuse')
            need(v['model_counts']==e['main_model_counts'],'Physical122 main counts reuse')
        else:
            need(v['model_recomputed'] is True and v.get('reuses_main_verified_full',False) is False,'Physically recomputed prefix')
    need(e['fit_execution_accounting']==execution_accounting(e),'Actual solve accounting with122 excluded from repeat count')




def schema_flags():
    prior=constants(ROOT/'work/monthly_asset_target255_independent.py',('TARGETS','CONTROL_FLAGS','SELECTION_FLAGS'))
    flags={'CONTROL_FLAGS':prior['CONTROL_FLAGS']}
    flags.update(constants(ROOT/'work/monthly_joint_identity259_independent.py',('MODEL_FLAGS',),seed=flags))
    flags['INCUMBENT_FLAGS']=constants(ROOT/'work/monthly_calendar_clock248_v2_independent.py',('INCUMBENT_GATE_FLAGS',))['INCUMBENT_GATE_FLAGS']
    inherited=tuple(k for k in prior['SELECTION_FLAGS'] if k!='weak_only_five_percent_slot_checked')+('weak_only_local_linear_delta_checked','original_pooled_rank_unchanged')
    flags['SELECTION_FLAGS']=tuple(k for k in inherited if k not in ('weak_only_local_linear_delta_checked','calm_nonweak_slot_unchanged'))+('full_original_local_quarter_checked','new_rank_minus_old_local_checked','three_arms_single_episode_checked')
    schemas=constants(ROOT/'work/monthly_joint_identity259_kernel.py',('MODEL_COUNT_FIELDS','COUNT_FIELDS','EVENT_KEYS','ANNUAL_KEYS','FORECAST_AUDIT_KEYS'))
    schemas.update(constants(ROOT/'work/monthly_local_distance256_kernel.py',('LABEL_KEYS',)))
    schemas['TARGETS']=prior['TARGETS']
    need([len(schemas[k]) for k in ('LABEL_KEYS','MODEL_COUNT_FIELDS','COUNT_FIELDS','EVENT_KEYS','ANNUAL_KEYS','FORECAST_AUDIT_KEYS')]==[11,19,18,15,28,14],'259 exact schemas')
    return schemas,flags

SCHEMAS,FLAGS=schema_flags()

def vector(v,size,label,positive=False):
    need(type(v) is list and len(v)==size and all(finite(x) and (not positive or x>0) for x in v),label)

def annual_summary(r,role,meta):
    need(type(r) is dict and set(r)==set(SCHEMAS['ANNUAL_KEYS']),'Complete annual record')
    integer(r['year'],'Annual year');need(r['cutoff']==str(r['year']-1)+'-12-31','Prior December31 cutoff')
    for key in ('training_row_ids','query_row_ids','qualified_query_row_ids','unqualified_query_row_ids'):
        ids=r[key];need(type(ids) is list and all(type(x) is int and x>=0 for x in ids) and ids==sorted(set(ids)),'Canonical annual IDs/'+key)
    ids=r['query_row_ids'];q=r['qualified_query_row_ids'];cold=r['unqualified_query_row_ids']
    need(sorted(q+cold)==ids and not set(q)&set(cold),'Exact qualified/cold partition')
    firms=r['qualified_symbols'];own=r['own_month_counts']
    need(type(own) is dict and all(type(k) is str and k and type(v) is int and v>0 for k,v in own.items()),'Native firm/month inventory')
    need(firms==sorted(k for k,v in own.items() if v>=18),'All training-qualified columns, not query subset')
    for j in ids:
        i,k=divmod(j,49);need(i<len(meta['dates']) and int(meta['dates'][i][:4])==r['year'],'Annual query identity')
        need((meta['symbols'][i][k] in firms)==(j in set(q)),'Query identity eligibility independent of rank universe')
    J=len(firms);N=len(r['training_row_ids']);need(sum(own.values())==N,'Unique security-month inventory covers every retained row');vector(r['weights'],N,'Positive original W',True)
    need(finite(r['weight_sum']) and r['weight_sum']>0,'Finite W total')
    for key in ('source_mean','source_beta','global_beta'):vector(r[key],10,'Ten original coordinates/'+key)
    vector(r['source_scale'],10,'Positive effective source scales',True)
    for key in ('dummy_mean','dummy_delta','weighted_dummy_sum'):vector(r[key],J,'Full dummy coordinates/'+key)
    need(finite(r['source_intercept']),'Finite original intercept')
    need(type(r['feature_dummy_cross']) is list and len(r['feature_dummy_cross'])==10,'Ten feature/dummy cross rows')
    for row in r['feature_dummy_cross']:vector(row,J,'Finite saved cross block')
    for key in ('raw_dummy_sha256','centered_dummy_sha256','standardized_design_sha256'):need(sha_value(r[key]),'Saved design digest')
    need(type(r['centered_dummy_zero']) is bool and type(r['fit_performed']) is bool,'Exact branch flags')
    status='no_qualified_columns' if not J else 'no_qualified_query' if not q else 'zero_centered_design' if r['centered_dummy_zero'] else 'fit'
    need(r['status']==status and r['fit_performed']==(status=='fit'),'Prospective exact annual skip priority')
    penalty=[12.]*J if role=='diagnostic' else [10.]*10+[12.]*J
    need(r['penalty']==penalty,'Fixed matched penalty units')
    if status=='fit':
        K=len(penalty);vector(r['rhs'],K,'Finite saved RHS');need(type(r['gram']) is list and len(r['gram'])==K,'Full saved normal matrix')
        for row in r['gram']:vector(row,K,'Finite saved normal matrix row')
    else:
        need(r['gram'] is None and r['rhs'] is None and r['global_beta']==r['source_beta'] and r['dummy_delta']==[0.]*J,'Copy branch has no fitted coefficients')
    if role=='diagnostic':need(r['global_beta']==r['source_beta'],'Sequential keeps source beta exactly')
    return status

def model_summary(audit,meta):
    need(set(audit)=={'model','labels','control','diagnostic','candidate'},'Complete model audit roles')
    m=audit['model'];c=m['counts'];model_preservation(m,audit['labels'],122)
    parity=m['original_cache_parity'];need(parity['events']==122 and all(parity[k] is True for k in ('passed','original_raw_exact','original_availability_exact','original_ranks_exact')) and parity['candidate_models_computed'] is False,'Original exact model gate')
    a=audit['control'];annual=a['annual'];rows=a['local']
    actual=dict(target_fits=4*sum(x['status']=='fit' and x['prediction_rows']>0 for x in a['source']['annual']),log_refits=sum(x['status']=='fit' and x['prediction_rows']>0 for x in annual),local_evaluations=len(rows),local_solves=sum(x['status']=='fit' for x in rows))
    need(all(c['control_'+k]==v for k,v in actual.items()),'Original solves from saved records')
    need(a['feature_count']==10 and a['alpha']==10. and a['half_life_months']==12. and a['min_months']==a['min_own_months']==18,'Original model configuration')
    for r in annual:
        need(r['cutoff']==str(r['year']-1)+'-12-31' and (r['latest_label_end'] is None or r['latest_label_end']<=r['cutoff']),'Original label maturity')
    fitted=[r for r in annual if r['status']=='fit' and r['prediction_rows']>0]
    common=set(SCHEMAS['ANNUAL_KEYS'])-{'global_beta','dummy_delta','gram','rhs','penalty','fit_performed'}
    statuses={}
    for role in ('diagnostic','candidate'):
        v=audit[role];need(set(v)==set(SCHEMAS['FORECAST_AUDIT_KEYS']) and v['schema']=='monthly_joint_identity259_forecast_audit_v1' and v['role']==role,'Exact new forecast audit')
        need(v['feature_count']==10 and v['target']=='unchanged cached centered log_excess' and v['global_penalty']==10. and v['dummy_penalty']==12.,'Fixed identity fit units')
        for k in ('source_log_copied','local_constant_copied','availability_preserved','local_support_preserved','security_intercepts_only','common_weights'):need(v[k] is True,'New forecast preservation/'+k)
        need(v['source_global_parameters_fixed']==(role=='diagnostic'),'Joint global coefficients may change')
        need(len(v['annual'])==len(fitted)==c['annual_design_builds'],'One design per original context')
        statuses[role]=Counter()
        for r,old in zip(v['annual'],fitted):
            need(r['year']==old['year'] and r['cutoff']==old['cutoff'],'Same original annual context')
            statuses[role][annual_summary(r,role,meta)]+=1
    for seq,joint in zip(audit['diagnostic']['annual'],audit['candidate']['annual']):
        need({k:seq[k] for k in common}=={k:joint[k] for k in common},'Matched complete rows/design/W and query domain')
    rs=audit['candidate']['annual'];need(statuses['diagnostic']==statuses['candidate'],'Matched active and copy years')
    need(c['qualified_security_columns']==sum(len(r['qualified_symbols']) for r in rs) and c['qualified_query_rows']==sum(len(r['qualified_query_row_ids']) for r in rs) and c['cold_query_rows']==sum(len(r['unqualified_query_row_ids']) for r in rs),'Actual identity/query counts')
    for k,s in [('no_qualified_columns_years','no_qualified_columns'),('no_qualified_query_years','no_qualified_query'),('zero_centered_design_years','zero_centered_design'),('candidate_joint_solves','fit')]:need(c[k]==statuses['candidate'][s],'Actual annual branch count/'+s)
    return dict(counts=c,label_preservation=audit['labels'],annual_status_counts={r:dict(v) for r,v in statuses.items()},all_saved_annual_records_checked=True,labels_and_forecasts_decoded=False,normal_equations_or_models_recomputed=False,source_log_copies_do_not_mean_joint_beta_is_fixed=True)

def index_list(value,width,label,ordered=False):
    need(type(value) is list and all(type(v) is int and 0<=v<width for v in value) and len(value)==len(set(value)),label)
    if not ordered:need(value==sorted(value),label+' ascending')


def decision_summary(audit,meta,orders):
    c=audit['counts'];counts_check(c,SCHEMAS['COUNT_FIELDS'],'Decision counts');need(c['events']==122 and c['symbols_per_event']==49,'Full122x49')
    ra=audit['rank_delta'];need(set(ra)==set(ARMS[1:]),'Two new rank audits')
    for arm,r in ra.items():
        need(r['schema']=='monthly_joint_identity259_rank_delta_v1' and r['arm']==arm and r['events']==122,'Rank audit scope')
        for key in ('original_ranking_population','no_group_ranking','no_own_history_filter','pooled_ranks_preserved','full_original_local_quarter','partial_exact','calm_quarter_included','no_post_rerank'):need(r[key] is True,'Rank scope/'+key)
        need(len(r['ready_rows'])==len(r['active_indices'])==len(r['coefficient'])==122 and all(type(x) is bool for x in r['ready_rows']),'Full rank metadata')
    counts=Counter()
    for i,e in enumerate(audit['events']):
        need(set(e)==set(SCHEMAS['EVENT_KEYS']) and e['query_index']==i and e['date']==meta['dates'][i],'Event identities')
        need(all(type(e[k]) is bool for k in ('initial_partial','weak','calm')) and e['initial_partial']==(e['date']=='2016-08-17'),'Formal/market flags')
        rank,active=e['ranking_indices'],e['active_indices']
        for name,ids in (('rank',rank),('active',active)):index_list(ids,49,name)
        need(set(active)<=set(rank),'Original U subset');counts['ranking_cells']+=len(rank);counts['initial_partial_events']+=e['initial_partial']
        counts['weak_applicable_cells']+=len(active) if e['weak'] else 0;counts['calm_applicable_cells']+=len(active) if e['calm'] and not e['weak'] else 0
        for role,arm in zip(('diagnostic','candidate'),ARMS[1:]):
            r=ra[arm];d=e[role+'_delta'];changed=e[role+'_changed_delta_indices'];index_list(changed,49,'Changed delta')
            vector(d,49,'Finite saved score delta');need(all(abs(x)<=.0125 for x in d),'Exact bounded quarter delta')
            need(changed==[j for j,x in enumerate(d) if x!=0.] and set(changed)<=set(active),'Exact nonzero delta support')
            need(active==r['active_indices'][i],'Common active scope');co=r['coefficient'][i];vector(co,49,'Finite slot coefficient')
            legal=(.0125,) if e['weak'] else (0.,.0025) if e['calm'] else (0.,)
            need(all(v in legal for v in co),'Full weak/calm coefficient, not weak-only')
            need(not active or (not e['initial_partial'] and r['ready_rows'][i] and all(co[j]>0 for j in active)),'Formal complete-row application')
            if not changed:need(orders[arm][i]==orders[ARMS[0]][i],'Zero delta exact original order')
            counts[role+'_changed_delta_cells']+=len(changed)
        for arm,key in zip(ARMS,('incumbent_choices','diagnostic_choices','candidate_choices')):
            ids=e[key];index_list(ids,49,'Choices',True);need(len(ids) in (0,2,3) and orders[arm][i]['symbols']==[meta['symbols'][i][j] for j in ids],'Order choices exact')
        counts['diagnostic_changed_holdings_events']+=set(e['incumbent_choices'])!=set(e['diagnostic_choices'])
        counts['candidate_changed_holdings_events']+=set(e['incumbent_choices'])!=set(e['candidate_choices'])
        counts['candidate_vs_diagnostic_changed_holdings_events']+=set(e['candidate_choices'])!=set(e['diagnostic_choices'])
        need(finite(e['episode_budget']) and 0<=e['episode_budget']<=.05,'Original episode budget')
    need(c['initial_partial_events']==1 and c['formal_events']==121,'Initial partial scope')
    for k,v in counts.items():need(c[k]==v,'Saved event count/'+k)
    for role,arm in zip(('diagnostic','candidate'),ARMS[1:]):need(ra[arm]['changed_cells']==c[role+'_changed_delta_cells'] and ra[arm]['ranking_cells']==c['ranking_cells'],'Rank/event counters')
    need(c['original_eligible_cells']>=c['ranking_cells'],'U within original E')
    return dict(counts=c,rank_delta=ra,full_arrays_decoded=False,original_eligibility_proven_by_bound_independent_certificate=True)

def saved_selection(train,changed):
    need(set(train)==set(ARMS),'Three fixed arms');invalid={a:required_metrics(train[a],a==ARMS[0]) for a in ARMS}
    candidate=ARMS[2];guards={};byref={}
    for ref in ARMS[:2]:
        ids=[r['index'] for r in changed[candidate+'_vs_'+ref]['months'] if r['index']<88];byref[ref]=ids
        close=not invalid[candidate] and not invalid[ref] and guard(train[candidate]['same_close_zero'],train[ref]['same_close_zero'],.005 if ref==ARMS[0] else 0.)
        opening=not invalid[candidate] and not invalid[ref] and guard(train[candidate]['next_open_zero'],train[ref]['next_open_zero'],0.)
        support=(len(ids)>=3 and any(i<53 for i in ids) and any(i>=53 for i in ids)) if ref==ARMS[0] else len(ids)>=1
        guards[ref]=dict(close=bool(close),open=bool(opening),support=bool(support))
    admitted=all(all(g.values()) for g in guards.values())
    return dict(primary=candidate if admitted else ARMS[0],admitted=admitted,guards=guards,changed_months_by_reference=byref,undefined_sharpe=invalid,first88_only=True,later34_used=False,cost_used=False,diagnostic_arm_never_promoted=True)

def changes(orders,reference,candidate):
    rows=[]
    for i,(a,b) in enumerate(zip(orders[reference],orders[candidate])):
        if set(a['symbols'])!=set(b['symbols']):rows.append(dict(index=i,date=a['date'],end=a['end'],reference=a['symbols'],candidate=b['symbols'],removed=sorted(set(a['symbols'])-set(b['symbols'])),added=sorted(set(b['symbols'])-set(a['symbols']))))
    return dict(counts={s:sum(lo<=r['index']<hi for r in rows) for s,(lo,hi) in SLICES.items()},months=rows,by_year=dict(Counter(r['date'][:4] for r in rows)))


def guard(c,r,gain):
    return c['training']['cagr']>=r['training']['cagr']+gain-1e-12 and c['training']['sharpe']>=r['training']['sharpe']-.05 and c['training']['max_drawdown']>=r['training']['max_drawdown']-.03 and all(c[e]['cagr']>=r[e]['cagr']-.025 for e in ('early','recent'))


def required_metrics(metrics,incumbent):
    need(set(metrics)==set(PATHS[:2]),'Exactly two ZERO paths for training')
    undefined=[]
    for path,groups in metrics.items():
        need(set(groups)=={'training','early','recent'},'First88 groups only')
        for group in groups:need(finite(groups[group]['cagr']),'Finite required CAGR')
        m=groups['training'];need(finite(m['max_drawdown']),'Finite required training drawdown')
        if m['sharpe'] is None:
            need(not incumbent and finite(m.get('daily_return_sd')) and m['daily_return_sd']==0 and m.get('sharpe_status')=='undefined_sharpe' and m.get('sharpe_reason')=='zero_daily_return_sd','Experimental undefined Sharpe evidence')
            undefined.append(path)
        else:
            need(finite(m['sharpe']),'Finite required Sharpe')
            need(finite(m.get('daily_return_sd')) and m['daily_return_sd']>0 and m.get('sharpe_status')=='defined' and m.get('sharpe_reason') is None,'Defined training Sharpe evidence')
    return undefined


def compare_metrics(independent,saved,label,residuals):
    need(set(independent)==STAT_KEYS and STAT_KEYS<=set(saved),'Nine saved metric fields')
    for key,value in independent.items():
        other=saved[key]
        if isinstance(value,str) or value is None or key=='trading_sessions':need(other==value,'Exact metric '+label+'/'+key)
        else:near(value,other,label+'/'+key,residuals)


def near(actual,expected,label,residuals):
    need(finite(actual) and finite(expected) and abs(actual-expected)<=ACCOUNT_ATOL+ACCOUNT_RTOL*abs(expected),'Frozen saved-accounting tolerance '+label)
    if actual!=expected:residuals.append(dict(label=label,actual=actual,reference=expected,difference=actual-expected))


def accounting(ev,results,ind,training):
    need(ind['passed'] is True and ind['checked_paths']==9 and ind['checked_slices']==45 and ind['checked_training_groups']==18,'9/45/18 independent accounting')
    need(ind['independent_accounts'] is True and ind['independent_of_production_model_and_engine'] is True and ind['terminal_liquidation'] is False,'Independent fixed-unit accounting')
    need(set(ind['paths'])==set(results['arms'])==set(training)==set(ARMS),'Exactly3 result arms')
    residuals=[];fee_fields=0;metric_ops=0;exports={}
    for arm in ARMS:
        need(set(results['arms'][arm])==set(ind['paths'][arm])==set(PATHS),'Exactly3 paths')
        need(set(training[arm])==set(PATHS[:2]),'Only zero-cost training')
        for p in PATHS:
            cert=ind['paths'][arm][p];saved=results['arms'][arm][p]
            need(cert['passed'] is True and cert['monthly_executions']==122 and set(cert['slices'])==set(saved['slices'])==set(SLICES),'Five saved slices')
            compare_metrics(cert['slices']['full122'],saved['metrics'],arm+'/'+p+'/metrics',residuals);metric_ops+=9
            for split in SLICES:compare_metrics(cert['slices'][split],saved['slices'][split],arm+'/'+p+'/'+split,residuals);metric_ops+=9
            for k in ('fees_paid','dollar_turnover','trade_rows'):near(cert[k],saved['metrics'][k],arm+'/'+p+'/'+k,residuals)
            need(saved['accounting']['exported_fill_ledger_matches'] is True and saved['accounting']['terminal_liquidation'] is False,'Saved account declarations')
            exports[arm,p]={k:ev.read(OUT/'results'/(arm+'_'+p+'_'+k+'.json')) for k in KINDS}
            data=exports[arm,p]
            need(len(data['HOLDINGS'])==122 and len(data['TRADES'])==cert['trade_rows'],'Full saved ledger rows')
            need([x['date'] for x in data['DAILY']]==[x['date'] for x in data['DAILY_POSITIONS']],'Saved NAV/positions dates')
            if p!='next_open_cost':
                need(all(finite(v) and v==0 for v in (saved['metrics']['fees_paid'],saved['slices']['full122']['fees_paid'],cert['fees_paid'])),'Zero-fee aggregate and full122 exact0')
                for k,field in (('TRADES','cost'),('HOLDINGS','fee')):
                    for x in data[k]:need(type(x[field]) in (int,float) and x[field]==0,'Every zero-fee field exact0');fee_fields+=1
            if p in PATHS[:2]:
                need(set(training[arm][p])=={'training','early','recent'},'Training3 groups')
                for short,split in (('training','first88'),('early','early53'),('recent','recent35')):
                    compare_metrics(cert['slices'][split],training[arm][p][short],arm+'/'+p+'/train/'+short,residuals);metric_ops+=9
                    # Compare the two existing summaries directly using the inherited
                    # near rule with the saved full slice as the reference.
                    compare_metrics({k:training[arm][p][short][k] for k in STAT_KEYS},saved['slices'][split],arm+'/'+p+'/train-full/'+short,residuals);metric_ops+=9
    return dict(independent_paths=9,independent_slices=45,training_groups=18,metric_fields_compared=metric_ops,zero_cost_streams=6,zero_cost_fee_fields_exact_zero=fee_fields,comparison_rule='ptr169_results_independent.py:37-41,85-91: absolute1e-7+relative2e-11*abs(reference), exact strings/None/trading_sessions',actual_saved_metric_residuals=residuals,returns_recomputed=False),exports

def ordered_period_fees(rows):
    """Frozen validate.replay accumulation: initial0.0, chronological +=fee.

    This is a saved-ledger identity check, not an account or return replay.
    Do not replace with fsum or an aggregate tolerance.
    """
    need(len(rows)==122,'Full incumbent holdings')
    dates=[r['execution_date'] for r in rows]
    need(all(type(d) is str for d in dates) and all(a<b for a,b in zip(dates,dates[1:])),'Chronological saved fee accumulation')
    total=0.0
    for row in rows:
        fee=row['fee'];need(finite(fee) and fee>=0,'Finite nonnegative period fee')
        total+=fee;need(math.isfinite(total),'Finite saved fee accumulation')
    return total


def compare_incumbent_saved(before_paths,after_paths,before_exports,after_exports,before_cert,after_cert):
    """Pure comparison of saved summaries/ledgers; no file or strategy access."""
    need(set(before_paths)==set(after_paths)==set(before_exports)==set(after_exports)==set(before_cert)==set(after_cert)==set(PATHS),'Exact incumbent path domain')
    exceptions=[];aggregate=None
    for p in PATHS:
        before=before_paths[p];after=after_paths[p]
        need(set(before_exports[p])==set(after_exports[p])==set(KINDS),'Exact incumbent export domain')
        need(before_cert[p]['passed'] is True and after_cert[p]['passed'] is True,'Both saved independent incumbent certificates passed')
        for key in ('metrics','slices'):
            left=before[key];right=after[key]
            if p=='next_open_cost':
                # Remove ONLY the two explicitly reconciled aggregate fields.
                if key=='metrics':
                    need('fees_paid' in left and 'fees_paid' in right,'Cost aggregate present')
                    left={k:v for k,v in left.items() if k!='fees_paid'}
                    right={k:v for k,v in right.items() if k!='fees_paid'}
                else:
                    need(set(left)==set(right)==set(SLICES),'Exact cost slice domain')
                    need('fees_paid' in left['full122'] and 'fees_paid' in right['full122'],'Full122 cost aggregate present')
                    left={k:({f:v for f,v in m.items() if f!='fees_paid'} if k=='full122' else m) for k,m in left.items()}
                    right={k:({f:v for f,v in m.items() if f!='fees_paid'} if k=='full122' else m) for k,m in right.items()}
            need(left==right,'Incumbent '+p+'/'+key+' exact except explicitly reconciled cost aggregate')
        for kind in KINDS:
            a=before_exports[p][kind];b=after_exports[p][kind]
            if kind!='HOLDINGS':need(a==b,'Incumbent saved '+kind+' exact');continue
            need(len(a)==len(b)==122,'Full incumbent holdings')
            for i,(x,y) in enumerate(zip(a,b)):
                need(set(x)==set(y) and {k:v for k,v in x.items() if k!='fee'}=={k:v for k,v in y.items() if k!='fee'},'Holding nonfee fields exact')
                need(finite(x['fee']) and finite(y['fee']) and x['fee']>=0 and y['fee']>=0,'Finite nonnegative period fees')
                if x['fee']!=y['fee']:
                    d=y['fee']-x['fee'];need(p=='next_open_cost' and abs(d)<=PERIOD_FEE_ATOL,'Only inherited cost period fee exception')
                    exceptions.append(dict(path=p,index=i,signal_date=y['signal_date'],previous=x['fee'],current=y['fee'],difference=d))
        if p!='next_open_cost':
            need(before['metrics']['fees_paid']==after['metrics']['fees_paid']==before_cert[p]['fees_paid']==after_cert[p]['fees_paid']==0,'Both rounds zero-cost aggregate exact0')
            for data in (before_exports[p],after_exports[p]):
                for kind,field in (('HOLDINGS','fee'),('TRADES','cost')):
                    need(all(finite(row[field]) and row[field]==0 for row in data[kind]),'Both rounds every zero-fee field exact0')
            continue
        totals={}
        for label,saved,data in (('previous',before,before_exports[p]),('current',after,after_exports[p])):
            total=ordered_period_fees(data['HOLDINGS'])
            for scope,metric in (('metrics',saved['metrics']),('slices.full122',saved['slices']['full122'])):
                need(finite(metric['fees_paid']) and metric['fees_paid']==total,'Cost '+label+'/'+scope+' fees_paid exact chronological HOLDINGS accumulation')
            totals[label]=total
        previous_ind=before_cert[p]['fees_paid'];current_ind=after_cert[p]['fees_paid']
        need(finite(previous_ind) and finite(current_ind) and previous_ind>=0 and previous_ind==current_ind,'Independent incumbent cost fee totals exact across rounds')
        aggregate=dict(path=p,fields=['metrics.fees_paid','slices.full122.fees_paid'],previous=totals['previous'],current=totals['current'],difference=totals['current']-totals['previous'],
          previous_independent=previous_ind,current_independent=current_ind,independent_totals_exact=True,
          each_round_exact_to_own_ordered_holdings=True,accumulation='total=0.0; for row in chronological HOLDINGS: total+=row[fee]',
          source='outputs/monthly_factor_search_v2/validate.py:109-123 replay totalcost=0.; totalcost+=fee; fees_paid=totalcost',
          aggregate_tolerance_added=False,cross_round_aggregate_bitwise_equal=totals['previous']==totals['current'])
    return dict(all_incumbent_metrics_slices_exact=aggregate['cross_round_aggregate_bitwise_equal'],
      zero_cost_metrics_slices_exact=True,all_other_incumbent_metrics_slices_exact=True,daily_trades_positions_exact=True,holding_nonfee_fields_exact=True,
      cost_aggregate_reconciliation=aggregate,
      period_fee_exception=dict(absolute_bound=PERIOD_FEE_ATOL,scope='ONLY next_open_cost individual HOLDINGS.fee; cumulative fields require exact own-ledger accumulation and exact cross-round independent certificates, not a tolerance',
        inherited_scope='ONLY next_open_cost individual HOLDINGS.fee; no cumulative metric exception',evidence='work/monthly_reference_gain250_RESULT_REVIEW.json/incumbent_reproduction/period_fee_exception',actual_differences=exceptions))


def previous_reference(ev,results,exports,current_ind,prior,review_path,current_arm,reference_arm):
    review=ev.read(review_path,FIXED[str(review_path.relative_to(ROOT))])
    need(review['passed'] is True and review['unresolved_blockers']==[],'Reference independently accepted')
    previous_completion=ev.read(prior/'results/COMPLETION.json',review['results_completion_sha256'])
    previous=ev.read(prior/'results/RESULTS.json',previous_completion['artifacts_sha256']['RESULTS.json'])
    previous_ind=ev.read(prior/'results/INDEPENDENT_OUTCOME.json',previous_completion['artifacts_sha256']['INDEPENDENT_OUTCOME.json'])
    need(previous_ind['passed'] is True and current_ind['passed'] is True and previous_ind['accounting_source_sha256']==current_ind['accounting_source_sha256'],'Same passed frozen accounting source')
    prior_exports={};zero_hashes=[];cost_exact_hashes=[]
    for p in PATHS:
        prior_exports[p]={}
        for kind in KINDS:
            before_name=reference_arm+'_'+p+'_'+kind+'.json';after_name=current_arm+'_'+p+'_'+kind+'.json'
            prior_exports[p][kind]=ev.read(prior/'results'/before_name,previous_completion['artifacts_sha256'][before_name])
            if p!='next_open_cost' or kind!='HOLDINGS':
                need(ev.pins[str((prior/'results'/before_name).relative_to(ROOT))]==ev.pins[str((OUT/'results'/after_name).relative_to(ROOT))],'Reference unchanged export bytes: '+after_name)
                (zero_hashes if p!='next_open_cost' else cost_exact_hashes).append(after_name)
    check=compare_incumbent_saved(previous['arms'][reference_arm],results['arms'][current_arm],prior_exports,{p:exports[current_arm,p] for p in PATHS},previous_ind['paths'][reference_arm],current_ind['paths'][current_arm])
    check.update(prior=prior.name,current_arm=current_arm,reference_arm=reference_arm,zero_cost_export_sha_exact=zero_hashes,cost_daily_trades_positions_sha_exact=cost_exact_hashes)
    return check



def terminals_check(record,expected_plan,hashes):
    need(record.get('plan_sha256')==expected_plan and set(record.get('phases',{}))=={'prepare','select','diagnose'},'Three root-owned successful phase terminals')
    for phase,relative in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]:
        p=record['phases'][phase];need(set(p)=={'session_id','exit_code','chunk_id','artifact_path','artifact_sha256'},'Terminal schema')
        need(p['session_id'] is None or (type(p['session_id']) is int and p['session_id']>0),'Terminal session identity')
        integer(p['exit_code'],'Terminal exit',0);need(type(p['chunk_id']) is str and bool(p['chunk_id'].strip()),'Terminal chunk')
        need(local(p['artifact_path'])==OUT/relative and p['artifact_sha256']==hashes[phase],'Terminal artifact exact')
    return record

def nav_metrics(days,values,initial=100000.):
    import statistics
    from datetime import date
    need(len(days)==len(values)>1 and days==sorted(set(days)),'Valid saved NAV interval')
    need(all(finite(v) and v>0 for v in values),'Positive saved NAV')
    years=(date.fromisoformat(days[-1])-date.fromisoformat(days[0])).days/365.2425;need(years>0,'Positive interval')
    returns=[b/a-1. for a,b in zip(values,values[1:])];sd=statistics.stdev(returns) if len(returns)>1 else 0.
    peak,dd=initial,0.
    for v in values:peak=max(peak,v);dd=min(dd,v/peak-1.)
    return dict(start=days[0],end=days[-1],years=years,final_equity=values[-1],cagr=(values[-1]/initial)**(1./years)-1.,sharpe=statistics.mean(returns)/sd*math.sqrt(252.) if sd>0 else None,max_drawdown=dd,trading_sessions=len(days),total_return=values[-1]/initial-1.),sd

def saved_nav_slices(daily,meta):
    days=[r['date'] for r in daily];values=[r['equity'] for r in daily];out={};sds={}
    for name,(lo,hi) in SLICES.items():
        left,right=days.index(meta['dates'][lo]),days.index(meta['ends'][hi-1]);v=values[left:right+1]
        if name!='full122':v=[x/values[left]*100000. for x in v]
        out[name],sds[name]=nav_metrics(days[left:right+1],v)
    return out,sds

def exported_ledger_check(data,orders,axis,path,residuals):
    days=[r['date'] for r in data['DAILY']];pos=data['DAILY_POSITIONS']
    need(days==[d for d in axis if orders[0]['date']<=d<=orders[-1]['end']] and days==[r['date'] for r in pos],'Full official exported daily axis')
    mapping={d:i for i,d in enumerate(axis)}
    execution=[o['date'] if path=='same_close_zero' else axis[mapping[o['date']]+1] for o in orders]
    holdings=data['HOLDINGS'];need([h['execution_date'] for h in holdings]==execution,'Exact monthly execution calendar')
    schedule=dict(zip(execution,orders));prev=None;chosen=set();dust=[];filtered_targets=[]
    for row,p in zip(data['DAILY'],pos):
        need(finite(p['cash']) and p['cash']>-1e-5 and finite(p['equity']) and p['equity']>0,'Frozen exported cash/NAV domain')
        units,values=p['adjusted_units'],p['market_values'];need(set(units)==set(values),'Position/value security domain')
        need(all(finite(v) and abs(v)>1e-9 and v>-1e-7 for v in units.values()) and all(finite(v) for v in values.values()),'Frozen exported units/dust domain')
        need(all(values[k]*units[k]>=0 for k in units),'Saved valuation sign matches position')
        need(p['cash']+sum(values.values())==p['equity'],'Saved exported position NAV arithmetic exact')
        # Direct exported NAV pair: frozen ptr169_rebuild_evaluate.py:278, not metric or cross-round comparison.
        difference=p['equity']-row['equity'];need(abs(difference)<=1e-5+1e-10*abs(row['equity']),'Frozen DAILY/position NAV bound')
        if difference:residuals.append(dict(field='DAILY/position NAV',actual=p['equity'],reference=row['equity'],difference=difference,existing_absolute_bound=1e-5,existing_relative_bound=1e-10))
        if row['date'] in schedule:chosen=set(schedule[row['date']]['symbols'])
        elif prev is not None:need(units==prev['adjusted_units'] and p['cash']==prev['cash'],'No intra-month units/cash trading')
        need(all(s in chosen or abs(q)<=1e-7 for s,q in units.items()),'Frozen non-target residual units bound')
        need(all(q>0 for s,q in units.items() if s in chosen),'Positive retained target units')
        for symbol,q in units.items():
            if symbol not in chosen:dust.append(dict(date=row['date'],symbol=symbol,adjusted_units=q,market_value=values[symbol]))
        if chosen-set(units):filtered_targets.append(dict(date=row['date'],symbols=sorted(chosen-set(units))))
        prev=p
    for h,o in zip(holdings,orders):
        need(h['signal_date']==o['date'] and h['symbols']==['META' if s=='research_equity_FB' else s for s in o['symbols']] and h['scores']==o['scores'],'Holding/ordered choice identity')
        need(finite(h['fee']) and h['fee']>=0,'Nonnegative saved period fee')
    seen=set();trades=data['TRADES'];need([r['date'] for r in trades]==sorted(r['date'] for r in trades),'Ordered fills')
    fee=turnover=0.
    for t in trades:
        need(t['date'] in schedule and t['side'] in ('buy','sell') and (t['date'],t['symbol']) not in seen,'Unique monthly net trade')
        seen.add((t['date'],t['symbol']))
        need(all(finite(t[k]) and t[k]>0 for k in ('adjusted_units','adjusted_price','dollars')) and finite(t['cost']) and t['cost']>=0,'Finite positive saved fill')
        near(t['adjusted_units']*t['adjusted_price'],t['dollars'],'Saved fill dollar identity',residuals)
        expected=t['dollars']*(.0005 if path=='next_open_cost' else 0.)
        need(abs(t['cost']-expected)<=1e-9+ACCOUNT_RTOL*abs(expected),'Frozen ptr169 fill-fee bound')
        if path!='next_open_cost':need(t['cost']==0.,'ZERO trade fee exact')
        fee+=t['cost'];turnover+=t['dollars']
    if path!='next_open_cost':need(all(h['fee']==0. for h in holdings),'ZERO holding fee exact')
    return dict(trade_rows=len(trades),saved_fill_cost_sum=fee,saved_dollar_turnover=turnover,ordered_period_fees=ordered_period_fees(holdings),no_intra_month_units_or_cash_changes=True,export_dust_policy=dict(source='ptr169_rebuild_evaluate.py:273-280 and ptr169_results_independent.py:174-203',export_abs_units_above=1e-9,minimum_exported_units_exclusive=-1e-7,maximum_non_target_absolute_units=1e-7,actual_non_target_dust=dust,targets_absent_after_export_filter=filtered_targets,absent_tiny_target_units_verified_by_bound_independent_source_price_certificate=True))

def recompute_saved_metrics(result,exports,orders,meta,calendar,training):
    residuals=[];allstats={};newtrain={};ledgers={}
    for arm in ARMS:
        allstats[arm]={};newtrain[arm]={};ledgers[arm]={}
        for path in PATHS:
            data=exports[arm,path];stats,sds=saved_nav_slices(data['DAILY'],meta);allstats[arm][path]=stats
            ledger=exported_ledger_check(data,orders[arm],calendar['sessions'],path,residuals);ledgers[arm][path]=ledger
            saved=result['arms'][arm][path]
            compare_metrics(stats['full122'],saved['metrics'],arm+'/'+path+'/fresh_saved_nav',residuals)
            for split in SLICES:compare_metrics(stats[split],saved['slices'][split],arm+'/'+path+'/fresh/'+split,residuals)
            need(saved['metrics']['fees_paid']==saved['slices']['full122']['fees_paid']==ledger['ordered_period_fees'],'Each saved cost aggregate exact ordered HOLDINGS sum')
            near(ledger['saved_fill_cost_sum'],saved['metrics']['fees_paid'],'Saved fill versus period fee',residuals)
            near(ledger['saved_dollar_turnover'],saved['metrics']['dollar_turnover'],'Saved turnover sum',residuals)
            need(ledger['trade_rows']==saved['metrics']['trade_rows'],'Saved trade count')
            if path in PATHS[:2]:
                groups={}
                for short,split in [('training','first88'),('early','early53'),('recent','recent35')]:
                    groups[short]=dict(stats[split]);compare_metrics(groups[short],training[arm][path][short],arm+'/'+path+'/fresh_training/'+short,residuals)
                sd=sds['first88'];groups['training'].update(daily_return_sd=sd,sharpe_status='defined' if sd>0 else 'undefined_sharpe',sharpe_reason=None if sd>0 else 'zero_daily_return_sd')
                near(sd,training[arm][path]['training']['daily_return_sd'],'Saved training SD',residuals);newtrain[arm][path]=groups
    return dict(paths=9,slices=45,training_groups=18,metrics=allstats,ledger_checks=ledgers,actual_saved_metric_residuals=residuals,method='Independent stdlib statistics from saved DAILY; ptr169 scalar formula; saved fill/holding/position identities. No raw prices, model fitting or portfolio replay.'),newtrain

def run(args):
    need(not REPORT.exists(),'Report write-once');need(platform.python_version()=='3.9.6','Pinned offline wrapper required')
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256)
    for p,h in FIXED.items():ev.pin(p,h);need(ev.registered(plan,p)==h,'Registered evidence '+p)
    context=ev.read(CONTEXT);contract=ev.read(CONTRACT);tests=ev.read(TESTS,ev.registered(plan,TESTS));code=ev.read(CODE_REVIEW,ev.registered(plan,CODE_REVIEW))
    need(plan['context']==context and plan['context_sha256']==digest(CONTEXT) and plan['implementation_contract']==contract and plan['implementation_contract_sha256']==digest(CONTRACT)==code['implementation_contract_sha256'],'Prospective exact context/contract')
    need(plan['numerical_verification']==NUMERICAL_VERIFICATION and context['arms']==list(ARMS) and context['only_promotable']==ARMS[2] and context['diagnostic_only']==ARMS[1] and context['registered_trials']==3,'Three-arm original design')
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==EXPECTED_FORMAL_TESTS and tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'97 formal tests/runtime')
    need(code['passed'] is True and code['unresolved_blockers']==[] and code['code_sha256']==tests['code_sha256'] and code['context_sha256']==digest(CONTEXT) and code['actual_returns_computed'] is False and code['tests']['count']==97,'Prospective code review')
    need(len(tests['code_sha256'])==6,'All six final codes')
    for p,h in tests['code_sha256'].items():need(ev.registered(plan,p)==h,'Registered code');ev.pin(p,h)
    for p in ('work/monthly_asset_target255_independent.py','work/monthly_calendar_clock248_v2_independent.py','work/monthly_local_distance256_kernel.py'):ev.pin(p,ev.registered(plan,p))
    state=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256']);need(len(state['completed'])==254 and sum(r['trials'] for r in state['completed'])==1273 and state['registered_queue']==[] and state['completed'][-1]['directory']==PRIOR.name,'Original prestate254/1273')
    protected=ev.read(PROTECTED,ev.registered(plan,PROTECTED));need(len(protected)==plan['protected_files']==21444,'Bound protected inventory')
    terminal=ev.read(TERMINALS);terminals_check(terminal,args.plan_sha256,dict(prepare=args.prepared_sha256,select=args.frozen_sha256,diagnose=args.results_sha256))
    pc=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP);need(pc['events']==122 and pc['new_returns_computed'] is False,'Prepared before returns')
    frozen=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256);training=ev.read(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==args.prepared_sha256 and frozen['orders_sha256']==pc['artifacts_sha256']['ORDERS.json'],'Immutable first88 chain')
    need({p.name for p in (OUT/'selected').iterdir()}=={'FROZEN.json','TRAIN_RESULTS.json'},'Exact selected files')
    rc=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT);need(rc['frozen_sha256']==args.frozen_sha256 and rc['paths']==9 and rc['slices']==45,'Full after frozen selection')
    meta=ev.read(OUT/'prepared/META.json');orders=ev.read(OUT/'prepared/ORDERS.json')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Fixed122 months')
    need(meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30','Frozen splits')
    need(all(len(s)==49 and len(set(s))==49 for s in meta['symbols']) and set(orders)==set(ARMS),'Original49 and three arms')
    for arm in ARMS:
        need(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and [o['end'] for o in orders[arm]]==meta['ends'],'Order clocks')
        need(all(o['signal_date']==o['date'] and len(o['symbols']) in (0,2,3) and len(o['symbols'])==len(set(o['symbols'])) and len(o['scores'])==len(o['symbols']) and all(finite(s) for s in o['scores']) for o in orders[arm]),'Finite monthly3/2/cash orders')
    need(orders[ARMS[0]]==ev.read(LOCAL/'local_linear_ORDERS.json',ev.registered(plan,LOCAL/'local_linear_ORDERS.json')),'Exact original orders')
    calendar=ev.read(CALENDAR,ev.registered(plan,CALENDAR));axis=calendar['sessions'];need(axis==sorted(set(axis)),'Official calendar metadata')
    positions={d:i for i,d in enumerate(axis)}
    for i,d in enumerate(meta['dates']):
        t=positions[d];need(i==0 or (t+1<len(axis) and axis[t+1][:7]!=d[:7]),'Formal monthend')
    original=ev.read(LOCAL/'PLAN.json',ev.registered(plan,LOCAL/'PLAN.json'));parity=ev.read(OUT/'prepared/SOURCE_PARITY.json')
    maps=[canonical_source_map(v) for v in (parity['raw_source_hashes'],original['raw_source_hashes'],plan['raw_source_hashes'])];need(all(v==maps[0] for v in maps),'Original64 complete path/SHA identity')
    for key,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:
        p=REBUILD/name;need(parity[key]==ev.registered(plan,p),'Original source cache pin');ev.pin(p,parity[key])
    need(parity['fresh_training_array_reconstruction'] is False and parity['cached_training_basis']=='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.','No fresh source claim')
    need(all(parity[k] is True for k in ('original_arrays_unchanged','original_training_unchanged','original_orders_exact')),'Original cache preservation certificate')
    ma=ev.read(OUT/'prepared/MODEL_AUDIT.json');audit=ev.read(OUT/'prepared/DECISION_AUDIT.json');e=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json')
    preparation_evidence(e,ma,audit);models=model_summary(ma,meta);decisions=decision_summary(audit,meta,orders)
    changed={a+'_vs_'+b:changes(orders,b,a) for a,b in ((ARMS[2],ARMS[0]),(ARMS[2],ARMS[1]),(ARMS[1],ARMS[0]))}
    selection=saved_selection(training,changed);need(selection==frozen['selection'],'First88 dual-reference admission exact')
    result=ev.read(OUT/'results/RESULTS.json');ind=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(result['selection']==selection and result['reused_sample'] is True and result['strict_pit'] is False and result['zero_cost_selection'] is True and result['next_open_cost_diagnostic_only'] is True and result['observer_promotion'] is False,'Honest result labels')
    need(ind['accounting_source_sha256']==FIXED['work/ptr169_results_independent.py'],'Pinned independent source-price accounting')
    ac,exports=accounting(ev,result,ind,training);fresh,newtrain=recompute_saved_metrics(result,exports,orders,meta,calendar,training)
    need(saved_selection(newtrain,changed)==selection,'First88 decision from independently recomputed saved NAV metrics')
    inc=previous_reference(ev,result,exports,ind,PRIOR,PREVIOUS_REVIEW,ARMS[0],ARMS[0])
    for p,h in {'work/basket_core251_saved_result_reader_v2.py':'35ab84ab4e6401e1858ac1e522c6e4c58483cce2a65b7d89d6da23014f97f555','work/monthly_reference_gain250_RESULT_REVIEW.json':'a9557b13524a04a9c6e16b3a6e308676de864b40a1a3947ce927e053444e4595'}.items():ev.pin(p,h)
    fee=ev.read('work/monthly_reference_gain250_RESULT_REVIEW.json')['incumbent_reproduction']['period_fee_exception'];need(fee['absolute_bound']==PERIOD_FEE_ATOL and fee['scope']=='ONLY next_open_cost individual HOLDINGS.fee; no cumulative metric exception','Inherited fee scope')
    ev.pin(ROOT/'work/ptr169_rebuild_evaluate.py',ev.registered(plan,ROOT/'work/ptr169_rebuild_evaluate.py'))
    ev.pin(ROOT/'outputs/monthly_factor_search_v2/validate.py',ev.registered(plan,ROOT/'outputs/monthly_factor_search_v2/validate.py'));ev.pin(Path(__file__))
    for p,h in ev.pins.items():need(digest(local(p))==h,'Evidence changed during audit: '+p)
    summary={a:{p:result['arms'][a][p]['metrics'] for p in PATHS} for a in ARMS}
    report=dict(schema='monthly_joint_identity259_saved_result_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
      reviewer_role='Lifecycle host author; saved-evidence reconciliation independent of producer/oracle model implementation. Not a second model fit.',
      scope='Saved source/manifest/audit/solve counters and scalar NAV metrics, first88 guards, exported positions/fills and prior-incumbent equality. No raw data or NPZ decoding; model numerics rely on separately authored bound certificates.',
      plan_sha256=args.plan_sha256,context_sha256=digest(CONTEXT),implementation_contract_sha256=digest(CONTRACT),prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,phase_terminals_sha256=digest(TERMINALS),root_owned_terminal_receipts=terminal,
      formal_synthetic_tests=97,bindings=dict(code_files=6,prepared_files=10,result_files=38,all_checked_hashes_match=True,raw64_canonical_maps_exact=True,raw64_files_rehashed=False,full_registered_input_closure_rehashed=False),
      source=parity,model=models,decision=decisions,preparation=dict(all123_cached_selection_and_incumbent_gates=True,physical_prefixes=list(PREFIXES),physical_model_recomputed=[0,1,53,88],physical122_reuses_main=True,fit_execution_accounting=e['fit_execution_accounting'],main_model_counts=ma['model']['counts'],independent_svd_counts=e['models']['independent_solve_counts'],independent_canonical_counts=e['models']['independent_canonical_solve_counts'],original27_arrays_preservation_verified_by_bound_source_and_numerical_certificates=True),
      selection=selection,changed_holdings=changed,accounting_review=ac,saved_nav_recalculation=fresh,incumbent_reproduction=inc,full_results=summary,segmented_results={a:{p:result['arms'][a][p]['slices'] for p in PATHS} for a in ARMS},
      target_reached_by_saved_selected_result=summary[selection['primary']]['same_close_zero']['cagr']>=.7,cost_label_answer=dict(headline_is_same_close_zero=True,includes_transaction_costs=False,cost_path_used_for_selection=False,cost_rate_per_side=.0005),
      protection_and_close=dict(status='READY_FOR_ROOT_CLOSE_NOT_YET_CLOSED',registered_pre_state_completed=254,registered_pre_state_trials=1273,expected_post_close_completed=255,expected_post_close_trials=1276,protected_manifest_hash_and_count_checked=True,protected_members_rehashed_by_reviewer=False,final_root_close_verified=False,mutable_central_read_or_bound=False),
      actual_models_computed_by_reviewer=False,actual_model_solve_count=0,npz_decoded=False,source_prices_loaded=False,actual_portfolio_replay=False,saved_nav_metrics_recomputed=True,phase_rerun=False,frozen_files_modified=False,observers_modified=False,strict_pit=False,reused_sample=True,
      historical_best_reporting=dict(first88_primary=selection['primary'],first88_admitted=selection['admitted'],prior_historical_best=state['best_training_selected_candidate'],later_diagnostics_did_not_reselect=True,root_postclose_best_not_yet_verified=True),
      reader_execution=dict(script=str(Path(__file__).relative_to(ROOT)),sha256=digest(Path(__file__)),phase_digests=vars(args),status='REPORT_WRITTEN_BEFORE_PROCESS_EXIT',terminal_exit_and_chunk='Reported separately only after the owning agent personally polls terminal; not asserted prospectively here'),
      limitations=['Complete forecast/label/score NPZ bytes are hashed but not decoded. Full-array equality, model fits and exact ranks rely on bound independent preparation receipts; no fresh original27 numeric reread is claimed.',
       'SVD and canonical normal rechecks are distinct actual work; original early gate duplicates control SVD intentionally. Physical122 reuse is excluded from repeated-fit totals; cached decisions fit no model.',
       'Saved DAILY metrics and position/fill arithmetic are recomputed without new price inputs. Source-price execution correctness remains the independently bound diagnose certificate.',
       'Both new arms retain full original ranking support and weak+calm local quarter. Joint changes shared coefficients; source_log copy does not imply unchanged joint global beta. Cold queries copy original pooled forecast.',
       'Only joint can pass frozen first88 dual-reference gates. Later34/full/cost cannot rescue rejection, and sequential is never promoted.',
       'Prior cost comparison preserves only the existing period1e-10 allowance, with each aggregate exactly matching own chronological HOLDINGS and both independent total fees exactly equal. No new tolerance.',
       'Archived reused sample, source vintage and universe proxies remain strict_pit=false. No observer promotion or future return claim.'],input_sha256=ev.pins)
    with REPORT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='SAVED_EVIDENCE_REVIEW_PASSED',report_sha256=digest(REPORT),primary=selection['primary'],admitted=selection['admitted'],full_same_close_zero_cagr={a:summary[a]['same_close_zero']['cagr'] for a in ARMS},fit_execution=e['fit_execution_accounting']['totals'])),flush=True)

def self_test():
    import copy
    import unittest
    from datetime import date,timedelta
    def model_count():
        c=dict.fromkeys(SCHEMAS['MODEL_COUNT_FIELDS'],0)
        c.update(training_rows=36,valid_training_rows=36,control_target_fits=4,control_log_refits=1,control_local_evaluations=2,control_local_solves=2,annual_design_builds=1,qualified_security_columns=2,qualified_query_rows=2,diagnostic_dummy_solves=1,candidate_joint_solves=1,actual_new_model_solves=2,actual_regression_solves=9)
        return c
    def model_receipts(n=122):
        c=model_count() if n else dict.fromkeys(SCHEMAS['MODEL_COUNT_FIELDS'],0)
        pc={k:c['control_'+k] for k in ('target_fits','log_refits','local_evaluations','local_solves')};pc['actual_regression_solves']=sum(pc[k] for k in ('target_fits','log_refits','local_solves'))
        sc=dict(pooled_log_solves=c['control_log_refits'],local_linear_solves=c['control_local_solves'],total_regression_solves=c['control_log_refits']+c['control_local_solves'])
        cc=dict(pooled_log_solves=0,sequential_dummy_solves=0,joint_block_solves=0,total_regression_solves=0)
        control=dict(**{k:True for k in FLAGS['CONTROL_FLAGS']},checked_events=n,producer_modules_imported=False,computes_returns=False,continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,exact_discrete=True,full_control_supplied=True,candidate_labels_computed=False,candidate_models_computed=False,episode_calls=0,producer_equivalent_fit_counts=pc,independent_solve_counts=sc,independent_canonical_solve_counts=cc,independent_design_builds=0)
        svd=dict(control=sc,sequential=dict(dummy_solves=c['diagnostic_dummy_solves'],total_regression_solves=c['diagnostic_dummy_solves']),joint=dict(block_solves=c['candidate_joint_solves'],total_regression_solves=c['candidate_joint_solves']),total_regression_solves=sc['total_regression_solves']+c['actual_new_model_solves'])
        canon=dict(pooled_log_solves=c['control_log_refits'],sequential_dummy_solves=c['diagnostic_dummy_solves'],joint_block_solves=c['candidate_joint_solves'],total_regression_solves=c['control_log_refits']+c['actual_new_model_solves'])
        full=dict(**{k:True for k in FLAGS['MODEL_FLAGS']},checked_events=n,producer_modules_imported=False,computes_returns=False,continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,exact_discrete=True,full_control_supplied=True,full_bundle_supplied=True,full_labels_supplied=True,full_diagnostic_supplied=True,full_candidate_supplied=True,independent_numeric_system=NUMERICAL_VERIFICATION['independent_numeric_system'],counts=c,producer_equivalent_counts=c.copy(),independent_solve_counts=svd,independent_canonical_solve_counts=canon,independent_design_builds=c['annual_design_builds'])
        return control,full,dict(counts=c)
    def training():
        result={}
        for arm,cagr in zip(ARMS,(.2,.21,.23)):
            result[arm]={}
            for p in PATHS[:2]:result[arm][p]=dict(training=dict(cagr=cagr,sharpe=1.,max_drawdown=-.1,daily_return_sd=.01,sharpe_status='defined',sharpe_reason=None),early=dict(cagr=cagr),recent=dict(cagr=cagr))
        return result
    def changes_fixture():return {ARMS[2]+'_vs_'+r:dict(months=[dict(index=i) for i in (2,54,75)]) for r in ARMS[:2]}
    def terminal_fixture():
        return dict(plan_sha256='p',phases={p:dict(session_id=1,exit_code=0,chunk_id='x',artifact_path=str(OUT/f),artifact_sha256=p) for p,f in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]})
    def annual_fixture():
        meta=dict(dates=['2020-01-31'],symbols=[['A','B']+['X'+str(i) for i in range(47)]])
        r=dict(year=2020,cutoff='2019-12-31',training_row_ids=list(range(36)),query_row_ids=[0,1],qualified_symbols=['A','B'],qualified_query_row_ids=[0,1],unqualified_query_row_ids=[],own_month_counts=dict(A=18,B=18),weight_sum=36.,weights=[1.]*36,source_mean=[0.]*10,source_scale=[1.]*10,source_intercept=0.,source_beta=[0.]*10,dummy_mean=[.5,.5],raw_dummy_sha256='a'*64,centered_dummy_sha256='b'*64,standardized_design_sha256='c'*64,weighted_dummy_sum=[0.,0.],feature_dummy_cross=[[0.,0.] for _ in range(10)],centered_dummy_zero=False,status='fit',global_beta=[0.]*10,dummy_delta=[.1,-.1],gram=[[12.,0.],[0.,12.]],rhs=[1.2,-1.2],penalty=[12.,12.],fit_performed=True)
        return r,meta
    def decision_fixture():
        dates=['2016-08-17']+[str(date(2017,1,1)+timedelta(days=i)) for i in range(121)]
        meta=dict(dates=dates,symbols=[['S'+str(j) for j in range(49)] for _ in dates]);orders={a:[dict(symbols=[],scores=[]) for _ in dates] for a in ARMS}
        counts=dict.fromkeys(SCHEMAS['COUNT_FIELDS'],0);counts.update(events=122,symbols_per_event=49,initial_partial_events=1,formal_events=121,original_eligible_cells=244,ranking_cells=244,weak_applicable_cells=2,calm_applicable_cells=2,diagnostic_changed_delta_cells=2,candidate_changed_delta_cells=2,incumbent_episode_calls=1,diagnostic_episode_calls=1,candidate_episode_calls=1)
        rank={a:dict(schema='monthly_joint_identity259_rank_delta_v1',arm=a,events=122,ready_rows=[True]*122,active_indices=[([0,1] if i in (1,2) else []) for i in range(122)],coefficient=[([.0125]*49 if i==1 else [.0025]*49 if i==2 else [0.]*49) for i in range(122)],ranking_cells=244,changed_cells=2,**{k:True for k in ('original_ranking_population','no_group_ranking','no_own_history_filter','pooled_ranks_preserved','full_original_local_quarter','partial_exact','calm_quarter_included','no_post_rerank')}) for a in ARMS[1:]}
        events=[]
        for i,d in enumerate(dates):
            delta=([.00125]+[0.]*48) if i==1 else ([.00025]+[0.]*48) if i==2 else [0.]*49
            events.append(dict(query_index=i,date=d,initial_partial=i==0,weak=i==1,calm=i==2,ranking_indices=[0,1],active_indices=[0,1] if i in (1,2) else [],diagnostic_changed_delta_indices=[0] if i in (1,2) else [],candidate_changed_delta_indices=[0] if i in (1,2) else [],diagnostic_delta=delta.copy(),candidate_delta=delta.copy(),incumbent_choices=[],diagnostic_choices=[],candidate_choices=[],episode_budget=.05))
        return dict(counts=counts,rank_delta=rank,events=events),meta,orders
    class Tests(unittest.TestCase):
        def test_annual_full_training_inventory(self):
            r,m=annual_fixture();self.assertEqual(annual_summary(r,'diagnostic',m),'fit')
        def test_annual_row_inventory_not_query_only(self):
            r,m=annual_fixture();r['own_month_counts']['B']=17
            with self.assertRaises(ValueError):annual_summary(r,'diagnostic',m)
        def test_annual_query_partition_and_cutoff(self):
            for key,value in [('unqualified_query_row_ids',[1]),('cutoff','2020-12-31'),('source_global_parameters_fixed',False)]:
                r,m=annual_fixture();r[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):annual_summary(r,'diagnostic',m)
        def test_sequential_must_preserve_global_beta(self):
            r,m=annual_fixture();r['global_beta'][0]=1e-12
            with self.assertRaises(ValueError):annual_summary(r,'diagnostic',m)
        def test_annual_zero_copy_no_fake_solve(self):
            r,m=annual_fixture();r.update(centered_dummy_zero=True,status='zero_centered_design',fit_performed=False,gram=None,rhs=None,dummy_delta=[0.,0.]);self.assertEqual(annual_summary(r,'diagnostic',m),'zero_centered_design')
            r['fit_performed']=True
            with self.assertRaises(ValueError):annual_summary(r,'diagnostic',m)
        def test_full_quarter_saved_decision(self):
            a,m,o=decision_fixture();self.assertEqual(decision_summary(a,m,o)['counts']['calm_applicable_cells'],2)
        def test_calm_coefficient_or_delta_tamper(self):
            for mode in ('coefficient','indices','bound'):
                a,m,o=decision_fixture()
                if mode=='coefficient':a['rank_delta'][ARMS[2]]['coefficient'][2]=[0.]*49
                elif mode=='indices':a['events'][2]['candidate_changed_delta_indices']=[]
                else:a['events'][2]['candidate_delta'][0]=.01250000000001
                with self.subTest(mode=mode),self.assertRaises(ValueError):decision_summary(a,m,o)
        def test_initial_partial_cannot_consume_delta(self):
            a,m,o=decision_fixture();a['events'][0]['active_indices']=[0];a['rank_delta'][ARMS[1]]['active_indices'][0]=[0]
            with self.assertRaises(ValueError):decision_summary(a,m,o)
        def test_legal_export_dust_and_reject_outside_bound(self):
            axis=[str(date(2020,1,1)+timedelta(days=i)) for i in range(123)]
            orders=[dict(date=d,end=axis[i+1],signal_date=d,symbols=[],scores=[]) for i,d in enumerate(axis[:122])]
            for q in (2e-9,-2e-9,1e-7,-1e-7,1.0000001e-7):
                values={'OLD':q*10.};nav=100000.+sum(values.values())
                d=dict(DAILY=[dict(date=x,equity=nav) for x in axis],DAILY_POSITIONS=[dict(date=x,equity=nav,cash=100000.,adjusted_units={'OLD':q},market_values=values.copy()) for x in axis],HOLDINGS=[dict(execution_date=o['date'],signal_date=o['date'],symbols=[],scores=[],fee=0.) for o in orders],TRADES=[])
                if q in (2e-9,-2e-9,1e-7):self.assertEqual(len(exported_ledger_check(d,orders,axis,'same_close_zero',[])['export_dust_policy']['actual_non_target_dust']),123)
                else:
                    with self.assertRaises(ValueError):exported_ledger_check(d,orders,axis,'same_close_zero',[])
        def test_existing_position_nav_tolerance_is_field_specific(self):
            axis=[str(date(2020,1,1)+timedelta(days=i)) for i in range(123)];orders=[dict(date=d,end=axis[i+1],signal_date=d,symbols=[],scores=[]) for i,d in enumerate(axis[:122])]
            d=dict(DAILY=[dict(date=x,equity=100000.) for x in axis],DAILY_POSITIONS=[dict(date=x,equity=100000.+1.5e-5,cash=100000.+1.5e-5,adjusted_units={},market_values={}) for x in axis],HOLDINGS=[dict(execution_date=o['date'],signal_date=o['date'],symbols=[],scores=[],fee=0.) for o in orders],TRADES=[])
            exported_ledger_check(d,orders,axis,'same_close_zero',[])
            d['DAILY_POSITIONS'][0]['cash']+=1e-6
            with self.assertRaisesRegex(ValueError,'arithmetic exact'):exported_ledger_check(d,orders,axis,'same_close_zero',[])
        def test_literal_schema(self):self.assertEqual((len(SCHEMAS['MODEL_COUNT_FIELDS']),len(SCHEMAS['COUNT_FIELDS'])),(19,18))
        def test_model_count_normal_and_empty(self):model_counts_check(model_count(),122);model_counts_check(dict.fromkeys(SCHEMAS['MODEL_COUNT_FIELDS'],0),0)
        def test_count_tampering(self):
            for k in ('diagnostic_pooled_fits','candidate_pooled_fits','actual_new_model_solves','annual_design_builds','cold_query_rows','actual_regression_solves'):
                c=model_count();c[k]+=1
                with self.subTest(k=k),self.assertRaises(ValueError):model_counts_check(c,122)
        def test_bool_count_rejected(self):
            c=model_count();c['no_qualified_columns_years']=False
            with self.assertRaises(ValueError):model_counts_check(c,122)
        def test_control_and_model_receipt(self):
            c,m,a=model_receipts();receipt(c,122,(),'control');receipt(m,122,(),'models');fit_binding(c,m,a,122)
        def test_missing_flags(self):
            for kind,i in [('control',0),('models',1)]:
                r=model_receipts()[i];r.pop(FLAGS['CONTROL_FLAGS'][0])
                with self.assertRaises(ValueError):receipt(r,122,(),kind)
        def test_canonical_work_not_hidden(self):
            r=model_receipts()[1];r['independent_canonical_solve_counts']['total_regression_solves']=0
            with self.assertRaises(ValueError):receipt(r,122,(),'models')
        def test_control_precedes_new_design(self):
            r=model_receipts()[0];r['independent_design_builds']=1
            with self.assertRaises(ValueError):receipt(r,122,(),'control')
        def test_tolerance_cannot_change(self):
            r=model_receipts()[1];r['continuous_atol']=1e-8
            with self.assertRaises(ValueError):receipt(r,122,(),'models')
        def test_physical122_not_counted_twice(self):
            c,m,a=model_receipts();e=dict(control=c,models=m,main_model_counts=a,physical_prefixes=[])
            for n in PREFIXES:
                cc,mm,aa=model_receipts(n);e['physical_prefixes'].append(dict(prefix=n,control=cc,model=mm,model_counts=aa))
            r=execution_accounting(e);self.assertEqual(r['model_invocations'],5);self.assertEqual(r['totals']['total_regression_solves'],80)
        def test_joint_only_promotion(self):self.assertEqual(saved_selection(training(),changes_fixture())['primary'],ARMS[2])
        def test_best_diagnostic_cannot_promote(self):
            t=training();t[ARMS[1]]['same_close_zero']['training']['cagr']=.9
            self.assertEqual(saved_selection(t,changes_fixture())['primary'],ARMS[0])
        def test_later_changes_cannot_rescue(self):
            changes=changes_fixture();changes[ARMS[2]+'_vs_'+ARMS[0]]['months']=[dict(index=i) for i in (89,100,121)]
            self.assertFalse(saved_selection(training(),changes)['admitted'])
        def test_need_both_segments_and_reference_change(self):
            for ref,ids in [(ARMS[0],[1,2,3]),(ARMS[1],[])]:
                c=changes_fixture();c[ARMS[2]+'_vs_'+ref]['months']=[dict(index=i) for i in ids];self.assertFalse(saved_selection(training(),c)['admitted'])
        def test_undefined_comparator_sharpe_rejects(self):
            t=training();t[ARMS[1]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd');self.assertFalse(saved_selection(t,changes_fixture())['admitted'])
        def test_incumbent_undefined_is_error(self):
            t=training();t[ARMS[0]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
            with self.assertRaises(ValueError):saved_selection(t,changes_fixture())
        def test_cost_cannot_enter_training(self):
            t=training();t[ARMS[2]]['next_open_cost']=t[ARMS[2]]['same_close_zero']
            with self.assertRaises(ValueError):saved_selection(t,changes_fixture())
        def test_terminal_exact(self):terminals_check(terminal_fixture(),'p',{k:k for k in ('prepare','select','diagnose')})
        def test_terminal_missing_badexit_and_wrongartifact(self):
            for key,val in [('exit_code',1),('exit_code',False),('artifact_path',str(OUT/'selected/TRAIN_RESULTS.json')),('artifact_sha256','bad'),('chunk_id','')]:
                r=terminal_fixture();r['phases']['prepare'][key]=val
                with self.subTest(key=key),self.assertRaises(ValueError):terminals_check(r,'p',{k:k for k in ('prepare','select','diagnose')})
            r=terminal_fixture();r['phases'].pop('select')
            with self.assertRaises(ValueError):terminals_check(r,'p',{})
        def test_saved_scalar_metrics(self):
            m,sd=nav_metrics(['2020-01-01','2020-01-02','2021-01-01'],[100.,110.,99.],initial=100.)
            self.assertAlmostEqual(m['total_return'],-.01);self.assertAlmostEqual(m['max_drawdown'],-.1);self.assertAlmostEqual(m['sharpe'],0.,places=12);self.assertGreater(sd,0.)
        def test_flat_nav_undefined(self):
            m,sd=nav_metrics(['2020-01-01','2020-01-02','2021-01-01'],[100.,100.,100.],initial=100.);self.assertIsNone(m['sharpe']);self.assertEqual(sd,0.)
        def test_nav_bad_values(self):
            for v in (0.,float('nan'),float('inf'),True):
                with self.subTest(v=v),self.assertRaises(ValueError):nav_metrics(['2020-01-01','2021-01-01'],[100.,v],initial=100.)
        def test_metrics_change_rejected(self):
            m,_=nav_metrics(['2020-01-01','2020-01-02','2021-01-01'],[100.,110.,99.],100.);bad=dict(m,cagr=m['cagr']+.001)
            with self.assertRaises(ValueError):compare_metrics(m,bad,'mutation',[])
        def test_source_complete_paths_and_sha(self):
            a={f'work/fake{i}':'a'*64 for i in range(64)};b={str(ROOT/k):v for k,v in a.items()};self.assertEqual(canonical_source_map(a),canonical_source_map(b));self.assertEqual(len(a),64)
            for k in ('../escape','/tmp/external','work/CloudStorage/x'):
                bad=a.copy();del bad['work/fake0'];bad[k]='a'*64
                with self.assertRaises(ValueError):canonical_source_map(bad)
        def test_source_collision_and_count(self):
            a={f'work/fake{i}':'a'*64 for i in range(63)};a[str(ROOT/'work/fake0')]='a'*64
            with self.assertRaises(ValueError):canonical_source_map(a)
            with self.assertRaises(ValueError):canonical_source_map({})
        def test_ordered_fees_exact_order_and_bool(self):
            rows=[dict(execution_date=str(date(2020,1,1)+timedelta(days=i)),fee=1. if i else 1e16) for i in range(122)]
            total=0.
            for r in rows:total+=r['fee']
            self.assertEqual(ordered_period_fees(rows),total);rows[1]['fee']=True
            with self.assertRaises(ValueError):ordered_period_fees(rows)
        def test_zero_fee_not_near_zero(self):
            # Exact zero requirement is independent of numeric comparison tolerance.
            axis=['2020-01-01','2020-01-02'];o=[dict(date=axis[0],end=axis[1],signal_date=axis[0],symbols=[],scores=[])]
            d=dict(DAILY=[dict(date=x,equity=100000.) for x in axis],DAILY_POSITIONS=[dict(date=x,equity=100000.,cash=100000.,adjusted_units={},market_values={}) for x in axis],HOLDINGS=[dict(execution_date=axis[0],signal_date=axis[0],symbols=[],scores=[],fee=1e-20)],TRADES=[])
            with self.assertRaisesRegex(ValueError,'ZERO holding fee'):exported_ledger_check(d,o,axis,'same_close_zero',[])
        def test_no_midmonth_units_or_cash_change(self):
            axis=['2020-01-01','2020-01-02'];o=[dict(date=axis[0],end=axis[1],signal_date=axis[0],symbols=[],scores=[])]
            d=dict(DAILY=[dict(date=x,equity=100000.) for x in axis],DAILY_POSITIONS=[dict(date=x,equity=100000.,cash=100000.,adjusted_units={},market_values={}) for x in axis],HOLDINGS=[],TRADES=[])
            d['HOLDINGS']=[dict(execution_date=axis[0],signal_date=axis[0],symbols=[],scores=[],fee=0.)];d['DAILY_POSITIONS'][1]['cash']+=1.
            with self.assertRaises(ValueError):exported_ledger_check(d,o,axis,'same_close_zero',[])
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    for name in ('plan','prepared','frozen','results'):parser.add_argument('--'+name+'-sha256')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        need(all(getattr(args,n+'_sha256') for n in ('plan','prepared','frozen','results')),'All four explicitly authorized phase digests required')
        run(args)

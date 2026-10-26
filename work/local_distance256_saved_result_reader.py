"""Saved review of terminal256 saved evidence after explicit authorization.

Run via run_local.sh with the four immutable phase digests. Standard library
only: no strategy/producer/oracle imports, array decoding, prices, volume,
feature mathematics, account replay or return recomputation. Saved scalar
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
OUT=ROOT/'outputs/monthly_local_distance256'
PRIOR=ROOT/'outputs/monthly_asset_target255'
LOCAL=ROOT/'outputs/monthly_local_linear70'
REPORT=ROOT/'work/monthly_local_distance256_RESULT_REVIEW.json'
CONTEXT=ROOT/'work/monthly_local_distance256_context.json'
TESTS=ROOT/'work/monthly_local_distance256_TESTS.json'
CODE_REVIEW=ROOT/'work/monthly_local_distance256_CODE_REVIEW.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
PREVIOUS_REVIEW=ROOT/'work/monthly_asset_target255_RESULT_REVIEW.json'
ARMS=('incumbent_local67','diag_mass_local67')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
SLICES={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FORECASTS.npz','LABELS.npz','MODEL_AUDIT.json','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
PREFIXES=(0,1,53,88,122)
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
FIXED={'work/monthly_local_distance256_context.json': '485f0b47a6cec7c578e27969424da2dfde576de903e2de6247acd99d22a8784b', 'work/monthly_local_distance256_IMPLEMENTATION_CONTRACT.json': 'beada676bdce0cdffe0bddf5e09b593f1284be2f8e837d74987639835151847e', 'work/monthly_asset_target255_RESULT_REVIEW.json': 'b18a2fb08b12abdbde75665ede52788524f5ee51216e99bb0e112b89911cc3f4', 'work/monthly_asset_target255_independent.py': '01c0750c54594657857668b5fb59f22a02ef0f5f7b166ecc09a17326d4b9a9e2', 'work/monthly_calendar_clock248_v2_independent.py': 'c7d172a9abec73f81d1d07ce8a422cd709a4a9bfebd55e6211b471c8d315663c', 'work/ptr169_results_independent.py': '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59', 'work/monthly_ownsector_gate238_host.py': '94f70ab8c0d9fc1cd891bf1ce3eaa30212a54ef4f9cdb1eb4e237bd82bed30a5'}
MASS_RTOL=16*2.220446049250313e-16
EXPECTED_FORMAL_TESTS=94
ACCOUNT_ATOL=1e-7
ACCOUNT_RTOL=2e-11
PERIOD_FEE_ATOL=1e-10
# Existing evidence: ptr169_results_independent.py:37-41,85-91 (saved-stat
# comparison); previous250 RESULT_REVIEW.accounting_review.comparison_rule
# and incumbent_reproduction.period_fee_exception (ONLY cost HOLDINGS.fee <=1e-10).


CONTRACT=ROOT/'work/monthly_local_distance256_IMPLEMENTATION_CONTRACT.json'
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
NUMERICAL_VERIFICATION=dict(continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.0,selection_tolerance=0.0,independent_numeric_system='augmented_least_squares_SVD')

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
    need(value['candidate_pooled_fits']==value['candidate_log_refits']==0,'Candidate pooled/log solves prohibited')
    need(value['candidate_local_evaluations']==value['control_local_evaluations'],'Same candidate query cohort')
    need(value['candidate_local_solves']<=value['candidate_kernel_evaluations']<=value['candidate_local_evaluations'],'Candidate local/kernel scope')
    need(value['candidate_equal_scale_evaluations']+value['candidate_changed_weight_evaluations']<=value['candidate_kernel_evaluations'],'Equal-scale copies and changed weights are disjoint')
    total=value['control_target_fits']+value['control_log_refits']+value['control_local_solves']+value['candidate_local_solves']
    need(value['actual_regression_solves']==total,'Model solve sum')
    if n==0:need(all(v==0 for v in value.values()),'Empty prefix no training rows or model fits')


def model_preservation(model,labels,n):
    ind=SimpleNamespace(**SCHEMAS,**FLAGS)
    need(model['schema']=='monthly_local_distance256_model_audit_v1','Model audit schema')
    c=model['counts'];model_counts_check(c,n)
    for key in ('original_row_order_preserved','original_common_good_preserved','all_targets_preserved','candidate_pooled_copied','candidate_local_constant_copied','candidate_local_linear_only','equal_scale_original_weights_exact','residuals_are_mature_in_sample'):
        need(model.get(key) is True,'Model preservation:'+key)
    for key in ('residuals_are_oof','computes_returns'):need(model.get(key) is False,'Model scope:'+key)
    need(type(model.get('mass_relative_tolerance')) is float and model['mass_relative_tolerance']==MASS_RTOL,'Prospective weight mass tolerance')
    expected={'schema','rows','valid_rows','target_keys','target_sha256','common_good_sha256','labels_unchanged','common_good_unchanged','original_row_order_preserved','recentered','raw_target_reconstructed'}
    need(type(labels) is dict and set(labels)==expected and labels['schema']=='monthly_local_distance256_labels_preserved_v1','Exact unchanged-label proof schema')
    integer(labels['rows'],'Preserved label rows',c['training_rows']);integer(labels['valid_rows'],'Preserved valid label rows',c['valid_training_rows'])
    need(labels['target_keys']==list(ind.TARGETS) and type(labels['target_sha256']) is dict and set(labels['target_sha256'])==set(ind.TARGETS),'All four unchanged targets')
    for digest in list(labels['target_sha256'].values())+[labels['common_good_sha256']]:
        need(type(digest) is str and len(digest)==64 and all(x in '0123456789abcdef' for x in digest),'Canonical preserved target/good SHA')
    for key in ('labels_unchanged','common_good_unchanged','original_row_order_preserved'):need(labels[key] is True,'Labels preservation:'+key)
    for key in ('recentered','raw_target_reconstructed'):need(labels[key] is False,'No new labels:'+key)



def receipt(check,n,flags,kind,*,width=None):
    ind=SimpleNamespace(**SCHEMAS,**FLAGS)
    need(type(check) is dict,'Independent receipt object')
    kind={'model':'models'}.get(kind,kind)
    need(kind in ('control','models','incumbent','selection'),'Known receipt kind')
    integer(n,'Requested events');integer(check.get('checked_events'),'Independent events',n)
    standard_flags={'control':ind.CONTROL_FLAGS,'models':ind.MODEL_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,'selection':ind.SELECTION_FLAGS}[kind]
    need(check.get('passed') is True and all(check.get(k) is True for k in tuple(flags)+tuple(standard_flags)),'Independent flags:'+kind)
    need(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Oracle independence:'+kind)
    def flag(key):need(check.get(key) is True,'Full independent proof:'+kind+'/'+key)
    def tolerance(key,value):
        actual=check.get(key)
        need(type(actual) is float and math.isfinite(actual) and actual==value,'Exact prospective tolerance:'+kind+'/'+key)
    if kind in ('control','models'):
        for key in ('continuous_atol','continuous_rtol','rank_tolerance'):tolerance(key,NUMERICAL_VERIFICATION[key])
        flag('exact_discrete');flag('full_control_supplied')
    if kind=='control':
        need(check.get('candidate_labels_computed') is False and check.get('candidate_models_computed') is False,'Control before new candidate')
        integer(check.get('episode_calls'),'Control episode calls',0)
        pc=check.get('producer_equivalent_fit_counts');sc=check.get('independent_solve_counts')
        producer_counts(pc,'Control producer-equivalent counts');solve_counts(sc,'Control independent solves')
        need(sc['pooled_log_solves']==pc['log_refits'] and sc['local_linear_solves']==pc['local_solves'],'Control independent actual solve scope')
        if n==0:need(not any(pc.values()) and not any(sc.values()),'Empty control no model fits')
    elif kind=='models':
        for key in ('full_bundle_supplied','full_labels_supplied','full_candidate_supplied'):flag(key)
        need(check.get('independent_numeric_system')==NUMERICAL_VERIFICATION['independent_numeric_system'],'Independent numerical method')
        c=check.get('counts');model_counts_check(c,n)
        counts_check(check.get('producer_equivalent_counts'),ind.MODEL_COUNT_FIELDS,'Producer-equivalent model counts')
        need(check['producer_equivalent_counts']==c,'Equivalent counts binding')
        sc=check.get('independent_solve_counts')
        need(type(sc) is dict and set(sc)=={'control','candidate','total_regression_solves'},'Independent model solve counts')
        for role in ('control','candidate'):
            solve_counts(sc[role],role+' independent solves')
            need(sc[role]['pooled_log_solves']==c[role+'_log_refits'] and sc[role]['local_linear_solves']==c[role+'_local_solves'],'Independent solves versus actual fit scope')
        integer(sc['total_regression_solves'],'Independent total solves',sum(sc[r]['total_regression_solves'] for r in ('control','candidate')))
    else:
        for key in ('full_scores_supplied','full_pre_scores_supplied'):flag(key)
        tolerance('numerical_tolerance',0.)
        integer(check.get('ordered_choices_checked'),'Ordered choices',n if kind=='incumbent' else 2*n)
        if kind=='incumbent':need(check.get('candidate_selection_computed') is False,'Original checked before candidate selection')
        else:
            for key in ('full_bundle_supplied','full_audit_supplied'):flag(key)
            c=check.get('counts');counts_check(c,ind.COUNT_FIELDS,'Decision counts')
            need(c['events']==n and c['symbols_per_event']>0 and c['initial_partial_events']+c['formal_events']==n,'Decision event scope')
            if width is not None:integer(c['symbols_per_event'],'Exact decision width',width)
            need(c['incumbent_episode_calls']==c['candidate_episode_calls']==1 and c['new_model_fits']==c['predictive_model_fits']==0,'Decision-only single episode scope')


def fit_binding(control,models,model_audit,n):
    c=model_audit['counts'];model_counts_check(c,n)
    need(models['counts']==c,'Independent/producer model counts binding')
    need(control['independent_solve_counts']==models['independent_solve_counts']['control'],'Repeated independent control solve scope')
    pc=control['producer_equivalent_fit_counts']
    for key in ('target_fits','log_refits','local_evaluations','local_solves'):
        need(pc[key]==c['control_'+key],'Control producer-equivalent fit count binding')


def execution_accounting(e):
    entries=[]
    invocations=[('main',122,e['control'],e['models'],e['main_model_counts'])]
    invocations += [('physical',v['prefix'],v['control'],v['model'],v['model_counts']) for v in e['physical_prefixes'] if v['prefix']!=122]
    for scope,n,control,models,audit in invocations:
        fit_binding(control,models,audit,n)
        a=audit['counts']['actual_regression_solves'];b=control['independent_solve_counts']['total_regression_solves'];c=models['independent_solve_counts']['total_regression_solves']
        entries.append(dict(scope=scope,events=n,producer_regression_solves=a,independent_control_gate_solves=b,independent_model_check_solves=c,total_regression_solves=a+b+c))
    sums={key:sum(v[key] for v in entries) for key in ('producer_regression_solves','independent_control_gate_solves','independent_model_check_solves','total_regression_solves')}
    return dict(schema='monthly_local_distance256_fit_execution_accounting_v1',model_invocations=len(entries),physical122_reuses_main=True,entries=entries,totals=sums,decision_only_fits=0)


def preparation_evidence(e,model_audit,decision_audit):
    ind=SimpleNamespace(**SCHEMAS,**FLAGS)
    kinds=[('control',ind.CONTROL_FLAGS),('models',ind.MODEL_FLAGS),('incumbent',ind.INCUMBENT_FLAGS),('selection',ind.SELECTION_FLAGS)]
    for key,flags in kinds:receipt(e[key],122,flags,key,width=49)
    need(e['all123_order_prefixes'] is True and e['original_arrays_unchanged'] is True and e['original_training_unchanged'] is True,'Preservation proofs')
    need(e['main_model_counts']==model_audit['model'],'Saved main model audit binding')
    model_preservation(model_audit['model'],model_audit['labels'],122)
    fit_binding(e['control'],e['models'],model_audit['model'],122)
    need(decision_audit['schema']=='monthly_local_distance256_decisions_v1' and len(decision_audit['events'])==122 and decision_audit['counts']==e['selection']['counts'],'Saved decision audit scope')
    need(decision_audit['incumbent_gate']==e['incumbent'],'Saved pre-candidate incumbent receipt')
    for key in ('original_arrays_unchanged','original_eligibility_unchanged','original_core_support_preserved','single_original_episode_per_arm','calm_slot_preserved','partial_exact'):
        need(decision_audit[key] is True,'Decision preservation:'+key)
    need(decision_audit['computes_returns'] is False,'Decisions before outcomes')
    cached=e['cached_prefixes'];physical=e['physical_prefixes']
    need(type(cached) is list and len(cached)==123,'All cached prefixes')
    for n,v in enumerate(cached):
        integer(v['prefix'],'Cached prefix',n);need(v['exact'] is True,'Exact cached prefix')
        receipt(v['incumbent'],n,ind.INCUMBENT_FLAGS,'incumbent')
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
    previous=constants(ROOT/'work/monthly_asset_target255_independent.py',('TARGETS','CONTROL_FLAGS','SELECTION_FLAGS'))
    flags={'CONTROL_FLAGS':previous['CONTROL_FLAGS']}
    flags.update(constants(ROOT/'work/monthly_local_distance256_independent.py',('MODEL_FLAGS',),seed=flags))
    flags['INCUMBENT_FLAGS']=constants(ROOT/'work/monthly_calendar_clock248_v2_independent.py',('INCUMBENT_GATE_FLAGS',))['INCUMBENT_GATE_FLAGS']
    flags['SELECTION_FLAGS']=tuple(k for k in previous['SELECTION_FLAGS'] if k!='weak_only_five_percent_slot_checked')+('weak_only_local_linear_delta_checked','original_pooled_rank_unchanged')
    schemas=constants(ROOT/'work/monthly_local_distance256_kernel.py',('LABEL_KEYS','MODEL_COUNT_FIELDS','COUNT_FIELDS','EVENT_KEYS','GEOMETRY_KEYS','CANDIDATE_ANNUAL_KEYS','CANDIDATE_LOCAL_KEYS','CANDIDATE_AUDIT_KEYS'))
    schemas['TARGETS']=previous['TARGETS']
    need([len(schemas[k]) for k in ('LABEL_KEYS','MODEL_COUNT_FIELDS','COUNT_FIELDS','EVENT_KEYS')]==[11,14,13,11],'256 exact label/model/decision schemas')
    return schemas,flags


SCHEMAS,FLAGS=schema_flags()


def geometry_summary(g,own_rows):
    need(type(g) is dict and set(g)==set(SCHEMAS['GEOMETRY_KEYS']),'Full saved geometry fields')
    for key in ('source_scale','coefficients'):
        need(type(g[key]) is list and len(g[key])==10 and all(finite(v) and v>0 for v in g[key]),'Positive finite ten-dimensional geometry/'+key)
    for key in ('old_distance','distance','original_weights','unscaled_weights','weights'):
        need(type(g[key]) is list and len(g[key])==own_rows and all(finite(v) and (v>=0 if 'distance' in key else v>0) for v in g[key]),'Saved own-history geometry/'+key)
    for key in ('original_mass','unscaled_mass','mass_multiplier','matched_mass','effective_samples','original_effective_samples'):
        need(finite(g[key]) and g[key]>0,'Positive saved mass/support/'+key)
    need(finite(g['mass_absolute_error']) and g['mass_absolute_error']==abs(g['matched_mass']-g['original_mass']) and g['mass_absolute_error']<=MASS_RTOL*g['original_mass'],'Declared16epsilon mass identity')
    need(finite(g['maximum_absolute_weight_change']) and g['maximum_absolute_weight_change']>=0,'Saved nonnegative maximum weight change')
    need(type(g['equal_scales']) is bool and g['equal_scales']==all(v==g['source_scale'][0] for v in g['source_scale']),'Exact equal-scale branch classification')
    need(type(g['weights_exact_original']) is bool and g['weights_exact_original']==(g['weights']==g['original_weights']),'Exact saved positive weight copy declaration')
    if g['equal_scales']:
        need(g['weights_exact_original'] and g['weights']==g['unscaled_weights'] and g['distance']==g['old_distance'] and g['coefficients']==[.1]*10 and g['mass_multiplier']==1. and g['matched_mass']==g['unscaled_mass']==g['original_mass'] and g['mass_absolute_error']==g['maximum_absolute_weight_change']==0.,'Equal-scale original weights exact')
    return dict(equal_scale=g['equal_scales'],changed_weights=not g['weights_exact_original'],mass_absolute_error=g['mass_absolute_error'])


def model_summary(audit,meta):
    m=audit['model'];c=m['counts'];model_preservation(m,audit['labels'],122)
    parity=m['original_cache_parity']
    need(parity['events']==122 and all(parity[k] is True for k in ('passed','original_raw_exact','original_availability_exact','original_ranks_exact')) and parity['candidate_models_computed'] is False,'Original exact cache gate')
    a=audit['control'];annual=a['annual'];rows=a['local'];candidate=audit['candidate'];cr=candidate['local']
    actual=dict(target_fits=4*sum(x['status']=='fit' and x['prediction_rows']>0 for x in a['source']['annual']),log_refits=sum(x['status']=='fit' and x['prediction_rows']>0 for x in annual),local_evaluations=len(rows),local_solves=sum(x['status']=='fit' for x in rows))
    need(all(c['control_'+k]==v for k,v in actual.items()),'Saved actual original producer fit scope')
    need(a['feature_count']==10 and a['alpha']==10. and a['half_life_months']==12. and a['min_months']==a['min_own_months']==18,'Frozen model configuration')
    for r in annual:
        need(r['cutoff']==str(r['year']-1)+'-12-31' and (r['latest_label_end'] is None or r['latest_label_end']<=r['cutoff']),'Annual mature prior-Dec31 cutoff')
    need(set(candidate)==set(SCHEMAS['CANDIDATE_AUDIT_KEYS']) and candidate['schema']=='monthly_local_distance256_candidate_audit_v1' and candidate['feature_count']==10,'Candidate local-only audit schema')
    for key in ('pooled_copied','local_constant_copied','local_linear_only'):need(candidate[key] is True,'Candidate preservation/'+key)
    need(candidate['target']=='unchanged cached centered log_excess' and candidate['residuals']=='same mature in-sample pooled residuals as control','No new label/OOF claim')
    ca=candidate['annual'];fitted=[r for r in annual if r['status']=='fit' and r['prediction_rows']>0]
    need(len(ca)==len(fitted)==c['control_log_refits'],'Candidate reused annual contexts')
    for r,old in zip(ca,fitted):
        need(set(r)==set(SCHEMAS['CANDIDATE_ANNUAL_KEYS']) and r['year']==old['year'] and r['cutoff']==old['cutoff'],'Candidate original annual cutoff')
        for key in ('training_row_ids','query_row_ids'):
            need(type(r[key]) is list and all(type(v) is int and v>=0 for v in r[key]) and r[key]==sorted(set(r[key])),'Canonical saved context row IDs')
        need(type(r['source_scale']) is list and len(r['source_scale'])==10 and all(finite(v) and v>0 for v in r['source_scale']),'Effective original ten scales')
    need(len(cr)==len(rows)==c['candidate_local_evaluations'],'Same local query inventory')
    gs=[]
    for r,old in zip(cr,rows):
        need(set(r)==set(SCHEMAS['CANDIDATE_LOCAL_KEYS']),'Full candidate local audit')
        for key in ('event_row','date','symbol','year','cutoff','own_rows','own_months','latest_label_end','constant_correction'):
            need(r[key]==old[key],'Same original local row/support/'+key)
        j=r['event_row'];integer(j,'Local event row');q,k=divmod(j,49)
        need(q<122 and r['date']==meta['dates'][q] and r['symbol']==meta['symbols'][q][k],'Local query identity')
        need(r['cutoff']==str(int(r['date'][:4])-1)+'-12-31' and (r['latest_label_end'] is None or r['latest_label_end']<=r['cutoff']),'Local maturity')
        need(type(r['own_row_ids']) is list and len(r['own_row_ids'])==r['own_rows'] and all(type(v) is int and v>=0 for v in r['own_row_ids']) and r['own_row_ids']==sorted(set(r['own_row_ids'])),'Exact own row IDs inventory')
        if r['geometry'] is not None:gs.append(geometry_summary(r['geometry'],r['own_rows']))
        else:need(r['status']=='insufficient_own_history','No unreported geometry for eligible local fit')
    need(c['candidate_local_solves']==sum(r['status']=='fit' for r in cr) and c['candidate_kernel_evaluations']==len(gs) and c['candidate_equal_scale_evaluations']==sum(g['equal_scale'] for g in gs) and c['candidate_changed_weight_evaluations']==sum(g['changed_weights'] for g in gs),'Saved candidate local/geometry counts')
    return dict(counts=c,label_preservation=audit['labels'],local_status_counts={role:dict(Counter(r['status'] for r in audit[role]['local'])) for role in ('control','candidate')},saved_geometry_records_checked=len(gs),maximum_saved_mass_error=max((g['mass_absolute_error'] for g in gs),default=0.),candidate_pooled_and_log_fits_zero=True,all_saved_local_fit_records_checked=True,labels_and_forecasts_decoded=False,weights_or_models_recomputed=False,raw_training_reconstructed=False)


def index_list(value,width,label,ordered=False):
    need(type(value) is list and all(type(v) is int and 0<=v<width for v in value) and len(value)==len(set(value)),label)
    if not ordered:need(value==sorted(value),label+' ascending')


def decision_summary(audit,meta,orders):
    c=audit['counts'];counts_check(c,SCHEMAS['COUNT_FIELDS'],'Decision counts')
    need(c['events']==122 and c['symbols_per_event']==49,'Decision full axis')
    ra=audit['rank_delta'];need(ra['schema']=='monthly_local_distance256_rank_delta_v1' and ra['events']==122,'Rank audit axis')
    for key in ('original_ranking_population','no_group_ranking','no_own_history_filter','pooled_ranks_preserved','weak_only','partial_exact','calm_slot_preserved','no_post_rerank'):need(ra[key] is True,'Rank scope '+key)
    need(len(ra['ready_rows'])==len(ra['active_indices'])==122 and all(type(x) is bool for x in ra['ready_rows']),'Common row readiness')
    ranks=active=changed=held=partial=0
    for i,e in enumerate(audit['events']):
        need(tuple(e)==SCHEMAS['EVENT_KEYS'] and e['query_index']==i and e['date']==meta['dates'][i],'Decision event identity')
        need(type(e['initial_partial']) is bool and e['initial_partial']==(e['date']=='2016-08-17') and type(e['weak']) is bool,'State and partial flags')
        r,a,d=e['ranking_indices'],e['active_indices'],e['changed_delta_indices']
        for name,ids in (('Ranking',r),('Active',a),('Changed delta',d)):index_list(ids,49,name)
        need(set(d)<=set(a)<=set(r) and a==ra['active_indices'][i],'Common original ranking/active domain')
        need(not a or (e['weak'] and not e['initial_partial'] and ra['ready_rows'][i]),'Weak-only formal application')
        need(len(e['delta'])==49 and all(finite(x) and abs(x)<=.0125+1e-12 for x in e['delta']),'Bounded saved delta')
        need([j for j,x in enumerate(e['delta']) if x!=0]==d,'Exact nonzero delta indices')
        need(finite(e['episode_budget']) and 0<=e['episode_budget']<=.05,'Original effective episode budget')
        for arm,key in zip(ARMS,('incumbent_choices','candidate_choices')):
            ids=e[key];index_list(ids,49,'Selected indices',True)
            need(len(ids) in (0,2,3) and orders[arm][i]['symbols']==[meta['symbols'][i][j] for j in ids],'Saved order identity')
        if not d:need(orders[ARMS[0]][i]==orders[ARMS[1]][i],'Zero delta preserves full original order record')
        ranks+=len(r);active+=len(a);changed+=len(d);held+=set(e['incumbent_choices'])!=set(e['candidate_choices']);partial+=e['initial_partial']
    need(c['initial_partial_events']==partial==1 and c['formal_events']==121,'Named initial partial')
    need(c['ranking_cells']==ra['ranking_cells']==ranks and c['weak_applicable_cells']==active and c['changed_delta_cells']==ra['changed_cells']==changed and c['changed_holdings_events']==held,'Decision counts from saved events')
    need(c['original_eligible_cells']>=ranks,'Ranks stay inside E; exact E proof bound in oracle receipt')
    return dict(counts=c,rank_delta=ra,original_eligibility_count_verified_by_bound_oracle=True,full_arrays_decoded=False)


def saved_selection(train,changed):
    need(set(train)==set(ARMS),'Two fixed arms')
    undefined={a:required_metrics(train[a],a==ARMS[0]) for a in ARMS}
    ids=[r['index'] for r in changed[ARMS[1]+'_vs_'+ARMS[0]]['months'] if r['index']<88]
    c=train[ARMS[1]];r=train[ARMS[0]]
    guards=dict(same_close='same_close_zero' not in undefined[ARMS[1]] and guard(c['same_close_zero'],r['same_close_zero'],.005),next_open='next_open_zero' not in undefined[ARMS[1]] and guard(c['next_open_zero'],r['next_open_zero'],0.),support=len(ids)>=3 and any(i<53 for i in ids) and any(i>=53 for i in ids))
    admitted=all(guards.values())
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=guards,changed_months=ids,undefined_sharpe=undefined,first88_only=True,later34_used=False)


def saved_month_impacts(changed,exports):
    rows=[];maps={a:{r['date']:r['equity'] for r in exports[a,'same_close_zero']['DAILY']} for a in ARMS}
    for row in changed['months']:
        growth={}
        for a in ARMS:
            first,last=maps[a][row['date']],maps[a][row['end']]
            need(finite(first) and finite(last) and first>0 and last>0,'Positive saved NAV endpoints')
            growth[a]=last/first-1.
        rows.append(dict(row,source='Saved same-close ZERO DAILY endpoint NAV ratios only; no prices or portfolio replay',saved_interval_returns=growth,candidate_minus_incumbent=growth[ARMS[1]]-growth[ARMS[0]],relative_growth_ratio=(1+growth[ARMS[1]])/(1+growth[ARMS[0]])))
    return dict(months=rows,attribution_scope='Descriptive saved NAV growth over signal-date to next signal/end. Not a newly estimated factor, causal effect, or recomputed portfolio.',new_portfolio_replay=False)


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
    need(ind['passed'] is True and ind['checked_paths']==6 and ind['checked_slices']==30 and ind['checked_training_groups']==12,'6/30/12 independent accounting')
    need(ind['independent_accounts'] is True and ind['independent_of_production_model_and_engine'] is True and ind['terminal_liquidation'] is False,'Independent fixed-unit accounting')
    need(set(ind['paths'])==set(results['arms'])==set(training)==set(ARMS),'Exactly2 result arms')
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
    return dict(independent_paths=6,independent_slices=30,training_groups=12,metric_fields_compared=metric_ops,zero_cost_streams=4,zero_cost_fee_fields_exact_zero=fee_fields,comparison_rule='ptr169_results_independent.py:37-41,85-91: absolute1e-7+relative2e-11*abs(reference), exact strings/None/trading_sessions',actual_saved_metric_residuals=residuals,returns_recomputed=False),exports


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



def run(args):
    need(not REPORT.exists(),'Report write-once');need(platform.python_version()=='3.9.6','Use pinned run_local wrapper')
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256)
    for p,h in FIXED.items():ev.pin(p,h);need(ev.registered(plan,p)==h,'Registered frozen evidence '+p)
    context=ev.read(CONTEXT);contract=ev.read(CONTRACT);tests=ev.read(TESTS,ev.registered(plan,TESTS));code=ev.read(CODE_REVIEW,ev.registered(plan,CODE_REVIEW))
    need(plan['context']==context and plan['context_sha256']==FIXED[str(CONTEXT.relative_to(ROOT))],'Registered exact context')
    need(plan['implementation_contract']==contract and plan['implementation_contract_sha256']==FIXED[str(CONTRACT.relative_to(ROOT))]==code['implementation_contract_sha256'],'Implementation contract binding')
    need(plan['numerical_verification']==NUMERICAL_VERIFICATION,'Prospective fixed numerical verification')
    need(context['arms']==list(ARMS) and context['promotable_arms']==[ARMS[1]] and context['registered_trials']==2,'Two arms only candidate promotable')
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==EXPECTED_FORMAL_TESTS and tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Exact formal suite runtime')
    need(code['passed'] is True and code['unresolved_blockers']==[] and code['tests']==EXPECTED_FORMAL_TESTS and code['code_sha256']==tests['code_sha256'] and code['context_sha256']==FIXED[str(CONTEXT.relative_to(ROOT))] and code['actual_returns_computed'] is False,'Prospective code review')
    need(len(tests['code_sha256'])==6,'All6 frozen code files')
    for p,h in tests['code_sha256'].items():need(ev.registered(plan,p)==h,'Registered6code');ev.pin(p,h)
    state=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(state['completed'])==251 and sum(x['trials'] for x in state['completed'])==1265 and state['registered_queue']==[] and state['completed'][-1]['directory']==PRIOR.name,'Prestate251/1265 no pending')
    protection=ev.read(PROTECTED,ev.registered(plan,PROTECTED));need(len(protection)==plan['protected_files']==21444,'Protected manifest count only')
    pc=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP)
    need(pc['events']==122 and pc['new_returns_computed'] is False,'Preparation without returns')
    frozen=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256);training=ev.read(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==args.prepared_sha256 and frozen['orders_sha256']==pc['artifacts_sha256']['ORDERS.json'],'Frozen first88 chain')
    need({p.name for p in (OUT/'selected').iterdir()}=={'FROZEN.json','TRAIN_RESULTS.json'},'Exact selected files')
    rc=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT)
    need(rc['frozen_sha256']==args.frozen_sha256 and rc['paths']==6 and rc['slices']==30,'Frozen selection before diagnostics')
    meta=ev.read(OUT/'prepared/META.json');orders=ev.read(OUT/'prepared/ORDERS.json')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Fixed interval')
    need(meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30','Training split clocks')
    need(all(len(s)==49 and len(set(s))==49 for s in meta['symbols']),'Full49 query identities')
    need(set(orders)==set(ARMS),'Exactly2 order arms')
    for arm in ARMS:
        need(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and [o['end'] for o in orders[arm]]==meta['ends'],'Order date axis')
        need(all(o['signal_date']==o['date'] and len(o['symbols']) in (0,2,3) and len(o['symbols'])==len(set(o['symbols'])) and len(o['scores'])==len(o['symbols']) and all(finite(s) for s in o['scores']) for o in orders[arm]),'3/2/cash finite saved orders')
    need(orders[ARMS[0]]==ev.read(LOCAL/'local_linear_ORDERS.json',ev.registered(plan,LOCAL/'local_linear_ORDERS.json')),'Exact Local67 incumbent orders')
    calendar=ev.read(CALENDAR,ev.registered(plan,CALENDAR));axis=calendar['sessions'];need(axis==sorted(set(axis)),'Official calendar metadata')
    positions={d:i for i,d in enumerate(axis)}
    for i,d in enumerate(meta['dates']):
        t=positions[d];need(i==0 or (t+1<len(axis) and axis[t+1][:7]!=d[:7]),'Formal official monthends')
    originalplan=ev.read(LOCAL/'PLAN.json',ev.registered(plan,LOCAL/'PLAN.json'))
    parity=ev.read(OUT/'prepared/SOURCE_PARITY.json')
    maps=[canonical_source_map(x) for x in (parity['raw_source_hashes'],originalplan['raw_source_hashes'],plan['raw_source_hashes'])]
    need(all(x==maps[0] for x in maps[1:]),'Original64 canonical path/SHA identity')
    for key,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:
        path=REBUILD/name;need(parity[key]==ev.registered(plan,path),'Saved cached source binding '+name);ev.pin(path,parity[key])
    need(parity['fresh_training_array_reconstruction'] is False and parity['cached_training_basis']=='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.','Honest cached training provenance')
    need(all(parity[k] is True for k in ('original_arrays_unchanged','original_training_unchanged','original_orders_exact')),'Original source preservation')
    ma=ev.read(OUT/'prepared/MODEL_AUDIT.json');audit=ev.read(OUT/'prepared/DECISION_AUDIT.json');e=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json')
    preparation_evidence(e,ma,audit)
    model=model_summary(ma,meta);decisions=decision_summary(audit,meta,orders)
    prep=dict(all123_cached_control_gates=True,all123_order_prefixes=True,physical_prefixes=list(PREFIXES),physical_model_recomputed=[0,1,53,88],physical122_reuses_main=True,empty_prefix_actual_fits=0,all_independent_flags_counts_and_fit_accounting_checked=True,fit_execution_accounting=e['fit_execution_accounting'],main_producer_model_counts=ma['model']['counts'],main_independent_control_solves=e['control']['independent_solve_counts'],main_independent_model_solves=e['models']['independent_solve_counts'],continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,selection_tolerance=0.)
    changed={ARMS[1]+'_vs_'+ARMS[0]:changes(orders,ARMS[0],ARMS[1])}
    selection=saved_selection(training,changed);need(selection==frozen['selection'],'Exact frozen first88 guards')
    result=ev.read(OUT/'results/RESULTS.json');ind=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(result['selection']==selection and result['reused_sample'] is True and result['strict_pit'] is False and result['zero_cost_selection'] is True and result['next_open_cost_diagnostic_only'] is True and result['observer_promotion'] is False,'Result policy labels')
    need(ind['accounting_source_sha256']==FIXED['work/ptr169_results_independent.py'],'Frozen independent accounting source')
    ac,exports=accounting(ev,result,ind,training)
    inc=previous_reference(ev,result,exports,ind,PRIOR,PREVIOUS_REVIEW,ARMS[0],ARMS[0])
    impacts=saved_month_impacts(changed[ARMS[1]+'_vs_'+ARMS[0]],exports)
    fee_path=ROOT/'work/monthly_reference_gain250_RESULT_REVIEW.json'
    fee_evidence=ev.read(fee_path,'a9557b13524a04a9c6e16b3a6e308676de864b40a1a3947ce927e053444e4595')['incumbent_reproduction']['period_fee_exception']
    need(fee_evidence['absolute_bound']==PERIOD_FEE_ATOL and fee_evidence['scope']=='ONLY next_open_cost individual HOLDINGS.fee; no cumulative metric exception','Inherited1e-10 single-period scope only')
    engine=ROOT/'outputs/monthly_factor_search_v2/validate.py';ev.pin(engine,ev.registered(plan,engine))
    ev.pin(Path(__file__));ev.pin(ROOT/'work/test_local_distance256_saved_result_reader.py')
    for p,h in ev.pins.items():need(digest(local(p))==h,'Evidence changed during review '+p)
    summary={a:{p:result['arms'][a][p]['metrics'] for p in PATHS} for a in ARMS}
    report=dict(schema='monthly_local_distance256_saved_result_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
      reviewer_role='Author of256 lifecycle host and saved-evidence reader; not independent256 model or numerical oracle author. Numerical geometry/model/selection/accounting proofs come from the separately authored bound independent oracle.',
      scope='Saved hashes, source/cache provenance, producer and independent actual solve counts, original control and selection receipts, first88 guards,6path saved accounts and historical incumbent comparison. No labels/models/NPZ decoding or strategy replay.',
      plan_sha256=args.plan_sha256,context_sha256=FIXED[str(CONTEXT.relative_to(ROOT))],implementation_contract_sha256=FIXED[str(CONTRACT.relative_to(ROOT))],prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,
      formal_synthetic_tests=EXPECTED_FORMAL_TESTS,bindings=dict(code_files=6,prepared_files=10,result_files=26,all_checked_hashes_match=True,raw64_registered_map_equal=True,raw64_path_normalization_only=True,raw64_files_rehashed=False,full_registered_input_closure_rehashed=False),
      preparation=prep,source=parity,model=model,decision=decisions,selection=selection,changed_holdings=changed,saved_changed_month_impacts=impacts,accounting_review=ac,incumbent_reproduction=inc,full_results=summary,
      segmented_results={a:{p:result['arms'][a][p]['slices'] for p in PATHS} for a in ARMS},
      holdings_counts={a:{str(k):sum(len(o['symbols'])==k for o in orders[a]) for k in (0,2,3)} for a in ARMS},
      target_reached_by_saved_selected_result=summary[selection['primary']]['same_close_zero']['cagr']>=.7,
      cost_label_answer=dict(headline_is_same_close_zero=True,includes_transaction_costs=False,cost_path_used_for_selection=False,cost_rate_per_side=.0005),
      protection_and_close=dict(status='READY_FOR_ROOT_CLOSE_NOT_YET_CLOSED',registered_pre_state_completed=251,registered_pre_state_trials=1265,expected_post_close_completed=252,expected_post_close_trials=1267,protected_manifest_hash_and_count_checked=True,protected_members_rehashed_by_reviewer=False,final_root_close_verified=False,post_close_central_verified=False,mutable_central_read_or_bound=False),
      source_prices_loaded=False,source_volumes_loaded=False,npz_decoded=False,actual_labels_computed_by_reviewer=False,actual_models_computed_by_reviewer=False,actual_features_computed_by_reviewer=False,actual_orders_computed_by_reviewer=False,actual_portfolio_returns_computed_by_reviewer=False,
      formal_tests_reexecuted=False,phase_rerun=False,frozen_files_modified=False,observers_modified=False,observer_promotion=False,reused_sample=True,pristine_out_of_sample_claim=False,strict_pit=False,
      historical_best_reporting=dict(first88_primary=selection['primary'],first88_admitted=selection['admitted'],prior_historical_best=state['best_training_selected_candidate'],later_diagnostics_did_not_reselect=True,root_postclose_best_not_yet_verified=True),
      limitations=['Labels, forecasts and full score arrays are only hash checked; numerical correctness relies on bound independent preparation certificates. Saved preservation/geometries/counts/cutoffs are checked without model or target recomputation.',
        'Training source is the previously rebuilt pinned cache, not a fresh raw-input reconstruction. All original four targets and common_good are preserved; no label transformation, new data or new cohort is certified by this saved-only review.',
        'Production solve counts and actual independent SVD counts differ intentionally; repeated physical model prefixes are counted, full122 reuse is not. Decision-only fits are zero, model preparation fits are not.',
        'Only weak formal .0125*(new_local_rank-old_local_rank) changes; pooled and local_constant predictions are copied. Original50/52 core,calm1%slot,E,monthly execution and original episode remain. Same-security residuals are mature in-sample,not OOF.',
        'Cost cumulative fees may differ only when each round exactly equals its own chronological HOLDINGS sum and cross-round independent totals remain exact. All other metrics/slices and unchanged exports exact; single-period bound1e-10 is inherited.',
        'First88 guards alone decide admission; later34/full/cost cannot rescue failure. Reused sample,proxy universe and source vintages imply strict_pit=false. No observer promotion or future performance certified.'],input_sha256=ev.pins)
    with REPORT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='SAVED_EVIDENCE_REVIEW_PASSED',report_sha256=digest(REPORT),input_pins=len(ev.pins),selection=selection['primary'],admitted=selection['admitted'],full_same_close_zero_cagr={a:summary[a]['same_close_zero']['cagr'] for a in ARMS},changed_month_counts={k:v['counts'] for k,v in changed.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','frozen','results'):parser.add_argument('--'+name+'-sha256',required=True)
    run(parser.parse_args())

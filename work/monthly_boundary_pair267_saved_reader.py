"""Prospective267 saved-evidence review, with no fitting or NPZ decoding.

Numerical verification belongs to the separately authored preparation verifier.
This reader reconciles its saved evidence with registered files and accounting.
The role adapters reuse immutable266 pure review rules without mutating globals.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import monthly_monotone_tree266_saved_reader as inherited

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_boundary_pair267'
PRIOR = ROOT / 'outputs/monthly_monotone_tree266'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
REBUILD = ROOT / 'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
CONTEXT = ROOT / 'work/monthly_boundary_pair267_context.json'
CONTRACT = ROOT / 'work/monthly_boundary_pair267_IMPLEMENTATION_CONTRACT.json'
TESTS = ROOT / 'work/monthly_boundary_pair267_TESTS.json'
REVIEW = ROOT / 'work/monthly_boundary_pair267_CODE_REVIEW.json'
TERMINALS = OUT / 'PHASE_TERMINALS.json'
ARMS = ('incumbent_local67', 'top5_all_pairs_control', 'original_p34_pair_candidate')
PATHS = ('same_close_zero', 'next_open_zero', 'next_open_cost')
KINDS = ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')
PREFIXES = (0, 1, 53, 88, 122)
PREP = {'MODELS.json', 'EFFECTS.npz', 'SCORES.npz', 'PRE_SCORES.npz', 'ORDERS.json',
        'META.json', 'DECISION_AUDIT.json', 'SOURCE_PARITY.json',
        'INDEPENDENT_PREPARE.json', 'SOURCE_AUDIT.json'}
RESULT = {'RESULTS.json', 'INDEPENDENT_OUTCOME.json'} | {
    a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
FORMAL = tuple('work/'+n for stem in ('host','kernel','independent','source','admission')
               for n in ('monthly_boundary_pair267_'+stem+'.py',
                         'test_monthly_boundary_pair267_'+stem+'.py'))
SUPPLEMENTAL = ('work/monthly_boundary_pair267_saved_reader.py',
                'work/test_monthly_boundary_pair267_saved_reader.py')
FIXED = {
    'work/monthly_monotone_tree266_saved_reader.py': 'eb117fa95c4d7940b96797b1a5a2bd08320b64f3663660a9569915f624e56a8d',
    'work/monthly_monotone_tree266_admission.py': '1b4199a384ac9ffaa43eb34775dd2c3ae094b743b55a3622ee6ac4f472050baf',
    'work/ptr169_results_independent.py': '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59',
    'work/monthly_gain_concentration265_saved_reader.py': '8990f2e15e423808546da661ae36717a6dd83dd71fee377f918d55599a2decd8',
}
need, digest, local = inherited.need, inherited.digest, inherited.local
finite, exact, integer = inherited.finite, inherited.exact, inherited.integer


def strict_pairs(items):
    result = {}
    for k,v in items:
        need(k not in result, 'Duplicate JSON key')
        result[k] = v
    return result


class Evidence(inherited.Evidence):
    """Index registered names once; all indexed maps and pinned bytes rechecked."""
    def __init__(self):
        super().__init__()
        self.registered_indexes = {}
    def read(self,p,want=None):
        p=local(p);need(p.suffix=='.json','JSON only; no numeric array decoding')
        self.pin(p,want)
        def invalid(value): raise ValueError('Nonfinite JSON constant')
        return json.loads(p.read_text(),object_pairs_hook=strict_pairs,parse_constant=invalid)
    def registered(self,plan,p):
        p=local(p);key=id(plan)
        if key not in self.registered_indexes:
            mapping=plan['input_sha256'];need(type(mapping) is dict,'Registration map')
            index={}
            for name,value in mapping.items():
                need(type(name) is str and inherited.sha_value(value),'Registered path/SHA')
                q=Path(name);q=(q if q.is_absolute() else ROOT/q).resolve()
                need(q not in index or index[q]==value,'Conflicting registered path aliases')
                index[q]=value
            # Retain the object so Python cannot recycle its id during this review.
            self.registered_indexes[key]=(plan,dict(mapping),index)
        saved,mapping,index=self.registered_indexes[key]
        need(saved is plan and p in index,'Missing registered file')
        return index[p]
    def finish(self):
        for plan,original,index in self.registered_indexes.values():
            need(plan['input_sha256']==original,'Registered map changed during reader')
        for p,value in self.pins.items():
            need(digest(local(p))==value,'Evidence bytes changed during reader')


def role_keys(value, *, to_original):
    source,target=(ARMS,inherited.ARMS) if to_original else (inherited.ARMS,ARMS)
    need(type(value) is dict and set(value)==set(source),'Three exact role keys')
    return {b:value[a] for a,b in zip(source,target)}


def saved_selection(training,orders):
    result=inherited.saved_selection(role_keys(training,to_original=True),
                                     role_keys(orders,to_original=True))
    names=dict(zip(inherited.ARMS,ARMS))
    result['primary']=names[result['primary']]
    for key in ('guards','changed_months_by_reference','undefined_sharpe'):
        result[key]={names[k]:v for k,v in result[key].items()}
    return result


class AccountEvidence:
    """Only36 explicit export paths are adapted; no catch-all path rewrite."""
    def __init__(self,ev):
        self.ev=ev
        self.paths={inherited.OUT/'results'/(old+'_'+p+'_'+k+'.json'):
                    OUT/'results'/(new+'_'+p+'_'+k+'.json')
                    for old,new in zip(inherited.ARMS,ARMS) for p in PATHS for k in KINDS}
    def read(self,p):
        need(p in self.paths,'Unapproved accounting path adaptation')
        return self.ev.read(self.paths[p])


def accounting(ev,results,independent,training):
    result=dict(results);result['arms']=role_keys(results['arms'],to_original=True)
    cert=dict(independent);cert['paths']=role_keys(independent['paths'],to_original=True)
    report,exports=inherited.accounting(AccountEvidence(ev),result,cert,
                                         role_keys(training,to_original=True))
    names=dict(zip(inherited.ARMS,ARMS))
    return report,{(names[a],p):v for (a,p),v in exports.items()}


def terminals_check(record,expected,hashes):
    need(record.get('plan_sha256')==expected and set(record.get('phases',{}))=={'prepare','select','diagnose'},'Root terminal phases')
    for phase,name in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]:
        row=record['phases'][phase]
        need(type(row) is dict and set(row)=={'session_id','exit_code','chunk_id','artifact_path','artifact_sha256'},'Exact terminal fields')
        need(row['session_id'] is None or type(row['session_id']) is int and row['session_id']>0,'Terminal session')
        integer(row['exit_code'],'Real terminal exit',0)
        need(type(row['chunk_id']) is str and bool(row['chunk_id'].strip()),'Real terminal chunk')
        need(local(row['artifact_path'])==OUT/name and row['artifact_sha256']==hashes[phase],'Terminal artifact binding')
    return record


def formal_bindings(ev,plan):
    for p,h in FIXED.items():
        need(ev.registered(plan,p)==h,'Immutable shared helper pin');ev.pin(p,h)
    context=ev.read(CONTEXT,ev.registered(plan,CONTEXT));contract=ev.read(CONTRACT,ev.registered(plan,CONTRACT))
    exact(plan['context'],context,'Embedded context');exact(plan['implementation_contract'],contract,'Embedded contract')
    need(plan['context_sha256']==digest(CONTEXT)==contract['context_sha256'] and plan['implementation_contract_sha256']==digest(CONTRACT),'Final contract hash')
    for d in (context,contract):
        need(d['frozen'] is True and d['registration_authorized'] is True and d['actual_data_started'] is False,'Prospective frozen definition')
    need(context['arms']==list(ARMS) and context['only_promotable']==ARMS[2] and context['diagnostic_only']==ARMS[1] and context['registered_trials']==3,'Fixed economic roles')
    need(contract['code_files']==list(FORMAL) and contract['supplemental_input_files']==list(SUPPLEMENTAL),'Exact code inventory')
    tests=ev.read(TESTS,ev.registered(plan,TESTS));review=ev.read(REVIEW,ev.registered(plan,REVIEW))
    need(tests['passed'] is tests['synthetic_only'] is True and type(tests['tests']) is int and tests['tests']>0,'Formal synthetics')
    need(tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Pinned runtime')
    need(review['passed'] is True and review['unresolved_blockers']==[] and review['actual_returns_computed'] is False,'Prospective code review')
    need(review['code_sha256']==tests['code_sha256'] and review['context_sha256']==digest(CONTEXT) and review['implementation_contract_sha256']==digest(CONTRACT),'Reviewed exact files')
    mapped={local(k):v for k,v in tests['code_sha256'].items()}
    need(len(mapped)==len(tests['code_sha256'])==10 and set(mapped)=={ROOT/p for p in FORMAL},'Ten formal sources')
    for p,h in mapped.items():
        need(ev.registered(plan,p)==h,'Registered test source');ev.pin(p,h)
    for p in SUPPLEMENTAL:
        h=ev.registered(plan,p);ev.pin(p,h)
        need(context['input_sha256'].get(p)==contract['input_sha256'].get(p)==h,'Supplemental files in both papers')
    return context,contract,tests


def packed_array(value,shape,label):
    need(type(value) is dict and set(value)=={'__ndarray__','dtype','shape','values_hex'}
         and value['__ndarray__'] is True and value['dtype']=='<f8' and value['shape']==list(shape),label+' typedfloat64 schema')
    count=math.prod(shape)
    need(type(value['values_hex']) is list and len(value['values_hex'])==count,label+' complete encoded array')
    numbers=[]
    for text in value['values_hex']:
        need(type(text) is str,label+' hex string')
        try:x=float.fromhex(text)
        except ValueError:raise ValueError(label+' invalid hex') from None
        need(math.isfinite(x) and x.hex()==text,label+' canonical finite binary64')
        numbers.append(x)
    return numbers


def float_audit(value,evidence,label):
    need(evidence==dict(dtype=value['dtype'],shape=value['shape'],values=value['values_hex']),label+' matching exact float evidence')


def models_check(record,meta,n):
    need(type(record) is dict and set(record)=={'schema','annual','annual_years','producer_counts'} and record['schema']=='monthly_boundary_pair267_saved_models_v1','Saved model document fields')
    years=sorted({int(d[:4]) for d in meta['dates'][:n] if d!='2016-08-17'})
    need(record['annual_years']==years and len(record['annual'])==len(years),'Consumed query years only')
    total_solves=total_control=total_candidate=total_endpoints=0
    summaries=[]
    for year,bundle in zip(years,record['annual']):
        need(bundle['schema']=='monthly_boundary_pair267_models_v1' and bundle['query_year']==year
             and bundle['pair_loss_set_only_difference'] is True and bundle['actual_coverage_claim'] is False,'Original caller-bound model bundle')
        c=bundle['cohort'];need(c['schema']=='monthly_boundary_pair267_cohort_v1' and c['query_year']==year,'Cohort year/schema')
        need(c['cutoff']==str(year-1)+'-12-31' and c['signal_lower_exclusive']==str(year-6)+'-12-31' and c['minimum_months']==18,'Historical annual clock')
        need(c['training_gap_filter'] is False and c['actual_coverage_claim'] is False,'No new traininggap or unauthenticated coverage claim')
        months=c['months'];need(type(months) is list and [m['source_index'] for m in months]==list(range(len(months))),'Original chronological month headers')
        dates=[m['signal'] for m in months];need(dates==sorted(set(dates)),'Unique month header dates')
        valid=[]
        for month in months:
            signal=month['signal'];end=month['end'];eligible_dates=signal!='2016-08-17' and signal>c['signal_lower_exclusive'] and end<=c['cutoff']
            need(signal<end,'Training endpoint after signal')
            if not eligible_dates:
                wanted='initial_partial' if signal=='2016-08-17' else 'outside_window' if signal<=c['signal_lower_exclusive'] else 'not_mature'
                need(month['status']==wanted,'Maturity gate before body')
                continue
            need(month['status'] in ('fewer_than_four_eligible','invalid_selected_name','common_valid_month'),'Explicit mature month validity')
            if month['status']=='common_valid_month':
                k=min(5,month['eligible_count']);need(k in (4,5),'Eligible TopK')
                ids=month['selected_indices'];symbols=month['selected_symbols']
                need(len(ids)==len(set(ids))==len(symbols)==len(set(symbols))==k
                     and all(type(i) is int and i>=0 for i in ids),'Selected original identities')
                need(month['boundary_pair_indices']==ids[2:4],'Original3/4 pair')
                valid.append(month)
        M=len(valid);need(c['common_months']==M and c['mathematical_weight_mass']==M and c['common_source_indices']==[m['source_index'] for m in valid],'Derived common cohort')
        ready=M>=18;need(c['status']==('ready' if ready else 'cold'),'Eighteen common months')
        ages=[(year-1-int(m['signal'][:4]))*12+12-int(m['signal'][5:7]) for m in valid]
        need(c['decay_ages']==ages and all(a>=1 for a in ages),'Mature decay calendar ages')
        d=c['decays'];need(len(d)==M and all(finite(v) and v>0 for v in d),'Positive stored decays')
        mass=math.fsum(d);weights=[(M*v)/mass for v in d] if M else []
        need(c['month_weights']==weights and c['month_weights_hex']==[v.hex() for v in weights],'Exact stored-decay normalization')
        need(set(c['arms'])==set(bundle['models'])==set(ARMS[1:]),'Two fixed Ridge roles')
        endpoints=sum(len(m['selected_indices']) for m in valid);total_endpoints+=endpoints
        sizes={}
        for arm in ARMS[1:]:
            bucket=c['arms'][arm];pairs=bucket['pairs'];expected=[];expected_weights=[]
            for mi,m in enumerate(valid):
                positions=[(2,3)] if arm==ARMS[2] else [(i,j) for i in range(len(m['selected_indices'])) for j in range(i+1,len(m['selected_indices']))]
                for i,j in positions:
                    expected.append(dict(source_index=m['source_index'],signal=m['signal'],end=m['end'],indices=[m['selected_indices'][i],m['selected_indices'][j]],symbols=[m['selected_symbols'][i],m['selected_symbols'][j]],original_ranks=[i+1,j+1],common_month_index=mi))
                    expected_weights.append(weights[mi]/len(positions))
            exact(pairs,expected,'Original pair set/order')
            rows=len(pairs);sizes[arm]=rows
            values={k:packed_array(bucket[k],(rows,10) if k=='X' else (rows,),arm+'/'+k) for k in ('X','y','weights')}
            need(all(-1.<=v<=1. for v in values['X']),'Unscaled centered-rank differences')
            need([v.hex() for v in values['weights']]==[v.hex() for v in expected_weights],'Exact pair mass split')
            wm=math.fsum(expected_weights)
            need(bucket['weight_mass']==wm and bucket['weight_mass_hex']==wm.hex(),'Actual rounded pair mass')
            for key in ('X','y','weights'):float_audit(bucket[key],bucket['float_evidence'][key],arm+'/'+key)
            model=bundle['models'][arm]
            need(model['schema']=='monthly_boundary_pair267_model_v1' and model['arm']==arm and model['query_year']==year and model['status']==c['status'] and model['common_months']==M,'Model identity/support')
            need(model['alpha']==10. and model['intercept']==0. and model['rescaled'] is False and model['reverse_augmented'] is False,'Fixed original feature-unit Ridge')
            need(model['rows']==rows and model['actual_solve_calls']==int(ready),'Actual model solve count')
            beta=packed_array(model['beta'],(10,),arm+'/beta');float_audit(model['beta'],model['float_evidence']['beta'],arm+'/beta')
            if ready:
                for key,shape in [('gram',(10,10)),('rhs',(10,))]:
                    packed_array(model[key],shape,arm+'/'+key);float_audit(model[key],model['float_evidence'][key],arm+'/'+key)
            else:
                need(all(v.hex()==0.0.hex() for v in beta) and model['gram'] is model['rhs'] is None and model['float_evidence']['gram'] is model['float_evidence']['rhs'] is None,'Cold model explicit zero no solve')
        solves=2*int(ready);need(bundle['actual_solve_calls']==solves,'Paired actual solves')
        total_solves+=solves;total_control+=sizes[ARMS[1]];total_candidate+=sizes[ARMS[2]]
        summaries.append(dict(year=year,common_months=M,status=c['status'],control_rows=sizes[ARMS[1]],candidate_rows=sizes[ARMS[2]],producer_solves=solves,selected_endpoint_returns=endpoints))
    return dict(annual=summaries,annual_cohort_builds=len(years),producer_ridge_solves=total_solves,control_pair_rows=total_control,candidate_pair_rows=total_candidate,endpoint_return_evaluations=total_endpoints,json_array_evidence_checked=True,decay_exp2_not_recomputed=True,no_model_solve_or_inference_by_reader=True)


def declarations():
    ind=inherited.constants(ROOT/'work/monthly_boundary_pair267_independent.py',('WORK_FIELDS','MODEL_FLAGS','ADJUSTMENT_FLAGS','SELECTION_FLAGS'))
    src=inherited.constants(ROOT/'work/monthly_boundary_pair267_source.py',('WORK_FIELDS',))
    producer=inherited.constants(ROOT/'work/monthly_boundary_pair267_host.py',('PRODUCER_FIELDS',))
    return dict(independent=ind,source=src['WORK_FIELDS'],producer=producer['PRODUCER_FIELDS'])


def add_counts(rows,fields):
    out=dict.fromkeys(fields,0)
    for row in rows:
        inherited.counts_check(row,fields,'Actual saved work')
        for k,v in row.items():out[k]+=v
    return out


def check_receipt(p,kind,schema,*,n=None,width=None,year=None):
    need(type(p) is dict and p.get('schema')=='monthly_boundary_pair267_independent_'+kind+'_v1' and p.get('passed') is True,'Actual independent '+kind)
    need(p.get('actual_coverage_claim') is False,'No synthetic coverage admission claim')
    flags=schema[{'models':'MODEL_FLAGS','adjustment':'ADJUSTMENT_FLAGS','selection':'SELECTION_FLAGS'}[kind]]
    need(all(p.get(k) is True for k in flags),'Complete independent flags')
    inherited.counts_check(p['work'],schema['WORK_FIELDS'],'Independent work')
    if year is not None:integer(p['query_year'],'Receipt year',year)
    if n is not None:integer(p['events'],'Receipt events',n)
    if width is not None:integer(p['width'],'Receipt width',width)
    if kind=='models':
        need(p['continuous_atol']==p['continuous_rtol']==2e-10 and p['exact_cohort_bits'] is True and p['canonical_equations_exact'] is True,'Only continuous SVD tolerance')
        need(set(p['model_sha256'])==set(p['svd_beta'])==set(ARMS[1:]),'Both independent solutions')
        need(all(inherited.sha_value(h) for h in p['model_sha256'].values()),'Model identity hashes')
        for beta in p['svd_beta'].values():
            if p['status']=='cold':need(beta is None,'Cold no independent solution')
            else:
                need(type(beta) is dict and set(beta)=={'dtype','shape','values'},'Packed independent solution')
                packed_array(dict(__ndarray__=True,dtype=beta['dtype'],shape=beta['shape'],values_hex=beta['values']),(10,),'Independent beta')
        integer(p['work']['independent_svd_solves'],'Independent paired solves',2*int(p['status']=='ready'))
    elif kind=='selection':
        need(p['numerical_tolerance']==0. and p['original_episode_calls']==1 and p['new_model_fits']==0,'Exact single original selector')
        need(p['policy']==dict(budget=.05,ceiling=5,gap=.02),'Frozen original boundary policy')
    else:
        need(p['numeric_tolerance']==0. and p['requires_prior_independent_model_gate'] is True and p['model_fitting_repeated'] is False,'Exact deployment after model gate')
    return p['work']


def prefix_evidence(entry,n,width,meta,schemas,*,cached=False):
    need(set(entry)=={'prefix','model_recomputed','reuses_main_verified_full','original_source','original_selection','source_audit','source_integrity_work','model_proofs','model_bindings','adjustment_proofs','selection_proofs','producer_counts'},'Complete source/model prefix proof')
    integer(entry['prefix'],'Prefix',n)
    c=entry['producer_counts'];inherited.counts_check(c,schemas['producer'],'Producer counts')
    need(c['original_local_fits']==c['original_other_target_fits']==0 and c['original_selector_calls']==3 and c['selector_events']==3*n and c['original_pre_score_calls']==1,'Original preserved and three single selectors')
    op=entry['original_source'];need(op['passed'] is True and op['events']==n and op['width']==width and op['model_fits']==0 and op['source_numeric_cells_compared']==2*n*width,'Original P/E source scope')
    for field in ('full_local67_p_exact','original_eligibility_exact','source_arrays_unchanged'):need(op[field] is True,'Original source '+field)
    sa=entry['source_audit'];need(sa['schema']=='monthly_boundary_pair267_source_audit_v1' and sa['events']==n and sa['width']==width,'Source audit identity')
    for key in ('labels_computed','orders_computed','portfolio_returns_computed','source_files_authenticated','npz_decoded','month_gap_filter'):need(sa[key] is False,'Exact source scope '+key)
    need(sa['models_fitted']==0 and sa['train_query_features_bitwise'] is True and sa['original_eligibility_bitwise'] is True and sa['canonical_label_price_source']=="panel['adj']",'Source units and parity')
    inherited.counts_check(sa['counts'],schemas['source'],'Source consumed counts')
    need(sa['consumed_month_indices']==sorted(set(sa['consumed_month_indices'])),'Mature source union')
    years=sorted({int(d[:4]) for d in meta['dates'][:n] if d!='2016-08-17'})
    need(sa['consumed_years']==years and set(entry['model_bindings'])=={str(y) for y in years},'Consumed years match query clock')
    need(all(inherited.sha_value(h) for h in entry['model_bindings'].values()),'Model bundle hashes')
    need(set(entry['selection_proofs'])==set(ARMS) and entry['original_selection']==entry['selection_proofs'][ARMS[0]],'Aliased incumbent proof')
    work=[]
    for proof in entry['selection_proofs'].values():work.append(check_receipt(proof,'selection',schemas['independent'],n=n,width=width))
    need(set(entry['adjustment_proofs'])==set(ARMS[1:]),'Both adjustments')
    for arm,proofs in entry['adjustment_proofs'].items():
        need([p['signal'] for p in proofs]==[d for d in meta['dates'][:n] if d!='2016-08-17'] and all(p['arm']==arm for p in proofs),'Every ordinary adjustment')
        for p in proofs:work.append(check_receipt(p,'adjustment',schemas['independent']))
    need(c['effect_adjustments']==2*sum(d!='2016-08-17' for d in meta['dates'][:n]),'Adjustment event work')
    if cached:
        need(entry['model_recomputed'] is False and entry['model_proofs']=={} and sa['mode']=='cached_projection','Cached projection without new model verification')
        for key in ('annual_cohort_builds','producer_ridge_solves','control_pair_rows','candidate_pair_rows','endpoint_return_evaluations'):integer(c[key],'Cached no '+key,0)
    else:
        need(set(entry['model_proofs'])==set(entry['model_bindings']),'Every annual model verified')
        for y,p in entry['model_proofs'].items():work.append(check_receipt(p,'models',schemas['independent'],year=int(y)))
        integer(c['annual_cohort_builds'],'Annual cohort work',len(years))
        integer(c['producer_ridge_solves'],'Ready paired solves',2*sum(p['status']=='ready' for p in entry['model_proofs'].values()))
    integrity=entry['source_integrity_work'];need(set(integrity)=={'checks','counts'},'Explicit integrity projection work')
    integer(integrity['checks'],'Source integrity calls',1 if cached else len(years)+1)
    source_work=add_counts([sa['counts'],integrity['counts']],schemas['source'])
    return dict(producer=c,independent=add_counts(work,schemas['independent']['WORK_FIELDS']),source=source_work)


def preparation_check(proof,models,audit,source,meta,*,prefixes=PREFIXES):
    n=len(meta['dates']);width=len(meta['symbols'][0]);schemas=declarations()
    need(set(proof)=={'schema','main','cached_prefixes','physical_prefixes','all123_order_prefixes','source_arrays_unchanged','source_training_unchanged','source_panel_unchanged','original_reference_unchanged','fit_execution_accounting'},'Full preparation fields')
    need(proof['schema']=='monthly_boundary_pair267_independent_prepare_v1','Preparation schema')
    for k in ('all123_order_prefixes','source_arrays_unchanged','source_training_unchanged','source_panel_unchanged','original_reference_unchanged'):need(proof[k] is True,'Actual preparation '+k)
    need([x['prefix'] for x in proof['cached_prefixes']]==list(range(n+1)),'All cached prefixes')
    need([x['prefix'] for x in proof['physical_prefixes']]==list(prefixes),'Fixed physical prefixes')
    expected=deepcopy(proof['main']);expected.update(model_recomputed=False,reuses_main_verified_full=True)
    exact(proof['physical_prefixes'][-1],expected,'Full physical aliases main')
    groups={'main':[proof['main']],'cached':proof['cached_prefixes'],'physical_fresh':proof['physical_prefixes'][:-1]}
    counted={}
    for group,entries in groups.items():
        rows=[]
        for entry in entries:
            if group=='physical_fresh':need(entry['model_recomputed'] is True and entry['reuses_main_verified_full'] is False,'Physical fresh work')
            rows.append(prefix_evidence(entry,entry['prefix'],width,meta,schemas,cached=group=='cached'))
            for year,binding in entry['model_bindings'].items():need(binding==proof['main']['model_bindings'][year],'Exact annual prefix model')
        counted[group]=dict(executions=len(entries),**{k:add_counts([r[k] for r in rows],f) for k,f in [('producer',schemas['producer']),('independent',schemas['independent']['WORK_FIELDS']),('source',schemas['source'])]})
    counted['total']={k:add_counts([counted[g][k] for g in groups],f) for k,f in [('producer',schemas['producer']),('independent',schemas['independent']['WORK_FIELDS']),('source',schemas['source'])]}
    actual=proof['fit_execution_accounting']
    for group,row in counted.items():exact(actual[group],row,'Actual separate-unit work '+group)
    need(actual['physical122_reuses_main'] is True and type(actual['count_definition']) is str,'Accounting reuse disclosure')
    summary=models_check(models,meta,n)
    need(models['producer_counts']==proof['main']['producer_counts'],'Saved annual model counts bound')
    for field in ('annual_cohort_builds','producer_ridge_solves','control_pair_rows','candidate_pair_rows','endpoint_return_evaluations'):integer(models['producer_counts'][field],'Derived model work '+field,summary[field])
    for row in summary['annual']:
        p=proof['main']['model_proofs'][str(row['year'])]
        need(p['status']==row['status'] and p['common_months']==row['common_months'],'Independent shared cohort support')
    need(audit['schema']=='monthly_boundary_pair267_decision_audit_v1' and audit['events']==n and audit['width']==width and set(audit['arms'])==set(ARMS) and audit['counts']==models['producer_counts'],'Full decision audit')
    for arm,record in audit['arms'].items():
        need([r['signal'] for r in record['events']]==meta['dates'] and record['selector']['original_selector_calls']==1,'Decision chronology and single selector')
        if arm==ARMS[0]:need(all(r['status']=='incumbent_identity' for r in record['events']),'Original identity decisions')
        else:
            proofs=iter(proof['main']['adjustment_proofs'][arm])
            for event in record['events']:
                if event['signal']=='2016-08-17':need(event['status']=='initial_partial' and event['q']==0.,'Partial identity');continue
                p=next(proofs);need(p['status']==event['status'] and p['signal']==event['signal'],'Exact adjustment audit status')
                if event['predicted_return_gap'] is not None:need(p['independent_svd_prediction_checked'] is True,'Actual prediction independently checked')
                need(finite(event['q']) and abs(event['q'])<=.5 and set(event['changed_indices'])<=set(event['pair_indices']),'Only original pair adjustment')
    exact(source['main'],proof['main']['source_audit'],'Main source manifest')
    for key,field in [('physical','physical_prefixes'),('cached','cached_prefixes')]:exact(source[key],[dict(prefix=x['prefix'],audit=x['source_audit']) for x in proof[field]],'Source '+key+' bindings')
    return summary,counted


def source_proof(ev,plan,source):
    original=ev.read(LOCAL/'PLAN.json',ev.registered(plan,LOCAL/'PLAN.json'))
    maps=[inherited.canonical_source_map(v) for v in (source['raw_source_hashes'],plan['raw_source_hashes'],original['raw_source_hashes'])]
    need(maps[0]==maps[1]==maps[2],'Original64 source lineage')
    for key,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:
        need(source[key]==ev.registered(plan,REBUILD/name),'Registered source pin');ev.pin(REBUILD/name,source[key])
    for key in ('original_arrays_unchanged','original_training_unchanged','original_orders_exact','original_scores_exact','original_pre_scores_exact','original_eligibility_exact','original_gate_before_new_labels','independent_model_before_deployment','whole_files_decoded'):need(source[key] is True,'Exact source '+key)
    need(source['original_other_target_fits']==source['original_local_fits']==0 and source['fresh_training_array_reconstruction'] is False and source['canonical_label_price_source']=="panel['adj']",'Original cached scope')
    closure=source['original169_source_closure'];need(closure==plan['original169_source_closure'],'Registered169 closure')
    for p,h in closure['chain_sha256'].items():need(ev.registered(plan,p)==h,'169 chain registered');ev.pin(p,h)
    parity=ev.read(REBUILD/'PARITY.json');p=ev.read(REBUILD.parent/'PLAN.json')
    need(parity['passed'] is True and parity['returns_computed'] is False and parity['source_plan_sha256']==digest(REBUILD.parent/'PLAN.json'),'Original169 source parity')
    need(len(parity['artifacts_sha256'])==12 and parity['artifacts_sha256']==closure['artifacts_sha256'],'All12 source artifacts')
    for name,h in parity['artifacts_sha256'].items():need(Path(name).name==name and ev.registered(plan,REBUILD/name)==h,'Registered cache member');ev.pin(REBUILD/name,h)
    need(closure['source_input_count']==len(p['input_sha256']),'169 source inventory')
    for path,h in p['input_sha256'].items():need(ev.registered(plan,path)==h,'Inherited169 input binding')
    inherited.full_original_chain(ev,plan)
    return dict(original64_maps_exact=True,original169_artifact_count=12,original169_artifacts_rehashed=True,npz_decoded=False,source_cache_rebuilt=False,raw64_rehashed=False,inherited_source_input_digests_bound=True,original_gate_before_new_labels=True,training_endpoint_source="panel['adj']")


def run(args):
    need(platform.python_version()=='3.9.6','Pinned wrapper runtime')
    for name in ('plan','prepared','frozen','results','terminals'):need(inherited.sha_value(getattr(args,name+'_sha256')),'Explicit phase digest '+name)
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256);context,contract,tests=formal_bindings(ev,plan)
    runtime=ev.read(inherited.RUNTIME,ev.registered(plan,inherited.RUNTIME));need(len(runtime['input_sha256'])==3309,'Numeric package inventory')
    for path,h in runtime['input_sha256'].items():need(ev.registered(plan,path)==h,'Registered runtime closure')
    state=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256']);need(len(state['completed'])==262 and sum(r['trials'] for r in state['completed'])==1297 and state['registered_queue']==[] and state['completed'][-1]['directory']==PRIOR.name,'Registered predecessor state')
    prior=ev.read(PRIOR/'COMPLETION.json',ev.registered(plan,PRIOR/'COMPLETION.json'));need(prior['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior['protected_files']==21444 and prior['target_reached'] is False,'Closed predecessor unmet goal')
    need(prior['plan_sha256']==ev.pin(PRIOR/'PLAN.json',ev.registered(plan,PRIOR/'PLAN.json')) and context['predecessor']['completion_sha256']==digest(PRIOR/'COMPLETION.json'),'Predecessor PLAN/context')
    priorreview=ROOT/'work/monthly_monotone_tree266_RESULT_REVIEW.json';pr=ev.read(priorreview,ev.registered(plan,priorreview))
    need(pr['passed'] is True and pr['unresolved_blockers']==[] and prior['result_review_sha256']==context['predecessor']['result_review_sha256']==digest(priorreview),'Prior saved review')
    protected=ev.read(inherited.PROTECTED,ev.registered(plan,inherited.PROTECTED));need(len(protected)==plan['protected_files']==21444,'Protected inventory')
    terminal=ev.read(TERMINALS,args.terminals_sha256);terminals_check(terminal,args.plan_sha256,dict(prepare=args.prepared_sha256,select=args.frozen_sha256,diagnose=args.results_sha256))
    pc=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP);need(pc['events']==122 and pc['new_returns_computed'] is False and pc['training_endpoint_returns_computed'] is True,'Training labels vs portfolio returns scope')
    frozen=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256);training=ev.read(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==args.prepared_sha256 and frozen['orders_sha256']==pc['artifacts_sha256']['ORDERS.json'],'Immutable first88')
    need({p.name for p in (OUT/'selected').iterdir()}=={'FROZEN.json','TRAIN_RESULTS.json'},'Exact selected files')
    rc=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT);need(rc['frozen_sha256']==args.frozen_sha256 and rc['paths']==9 and rc['slices']==45,'Complete nine outcomes')
    meta=ev.read(OUT/'prepared/META.json');orders=ev.read(OUT/'prepared/ORDERS.json')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Fixed full122 chronology')
    need(meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30','Fixed53/35/34 segments')
    need(all(len(s)==49 and len(set(s))==49 for s in meta['symbols']) and set(orders)==set(ARMS),'Original49 and three roles')
    for arm in ARMS:
        need(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and [o['end'] for o in orders[arm]]==meta['ends'],'Order clock')
        for o in orders[arm]:need(o['signal_date']==o['date'] and len(o['symbols']) in (0,2,3) and len(o['symbols'])==len(set(o['symbols'])) and len(o['scores'])==len(o['symbols']) and all(finite(v) for v in o['scores']),'Original ordered selections')
    source=ev.read(OUT/'prepared/SOURCE_PARITY.json');source_summary=source_proof(ev,plan,source)
    priorprep=ev.read(PRIOR/'prepared/COMPLETION.json',source['prior_prepared_sha256']);need(priorprep['status']=='COMPLETE' and priorprep['plan_sha256']==prior['plan_sha256'] and source['prior_prepared_sha256']==ev.registered(plan,PRIOR/'prepared/COMPLETION.json'),'Prior source manifest')
    for name in ('META.json','ORDERS.json','SCORES.npz','PRE_SCORES.npz'):ev.pin(PRIOR/'prepared'/name,priorprep['artifacts_sha256'][name])
    exact(meta,ev.read(PRIOR/'prepared/META.json'),'Prior identities');exact(orders[ARMS[0]],ev.read(PRIOR/'prepared/ORDERS.json')[ARMS[0]],'Prior incumbent orders')
    exact(orders[ARMS[0]],ev.read(LOCAL/'local_linear_ORDERS.json',ev.registered(plan,LOCAL/'local_linear_ORDERS.json')),'Original Local67 ordered source')
    metadata=ev.read(REBUILD/'MONTHLY_META.json',ev.registered(plan,REBUILD/'MONTHLY_META.json'));exact(meta,metadata['monthly'],'Original169 monthly identities')
    calendar=ev.read(inherited.CALENDAR,ev.registered(plan,inherited.CALENDAR));need(source['calendar_sha256']==plan['calendar_sha256']==digest(inherited.CALENDAR),'Calendar binding')
    axis=calendar['sessions'];need(axis==sorted(set(axis)),'Official calendar');positions={d:i for i,d in enumerate(axis)}
    for i,d in enumerate(meta['dates']):need(d in positions and (i==0 or positions[d]+1<len(axis) and axis[positions[d]+1][:7]!=d[:7]),'Official monthends')
    models=ev.read(OUT/'prepared/MODELS.json');audit=ev.read(OUT/'prepared/DECISION_AUDIT.json');proof=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json');sa=ev.read(OUT/'prepared/SOURCE_AUDIT.json')
    summary,work=preparation_check(proof,models,audit,sa,meta)
    exact(sa['join']['monthly'],{k:meta[k] for k in ('dates','ends','symbols','etfs')},'Source join monthly axes');exact(sa['join']['training'],{k:metadata['training'][k] for k in ('dates','ends','symbols','etfs') if k in metadata['training']},'Source join training axes')
    selected=saved_selection(training,orders);exact(selected,frozen['selection'],'First88 dual-reference selection')
    result=ev.read(OUT/'results/RESULTS.json');ind=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json');exact(result['selection'],selected,'Immutable outcome selection')
    for k in ('reused_sample','zero_cost_selection','next_open_cost_diagnostic_only'):need(result[k] is True,'Declared research '+k)
    need(result['strict_pit'] is False and result['observer_promotion'] is False and ind['accounting_source_sha256']==FIXED['work/ptr169_results_independent.py'],'Research/accounting source')
    for arm in ARMS:
        for path in PATHS[:2]:
            for m in (result['arms'][arm][path]['metrics'],result['arms'][arm][path]['slices']['full122']):need(m['cost_per_side']==0. and m['fees_paid']==0.,'Exact ZERO headline')
    account,exports=accounting(ev,result,ind,training)
    for arm in ARMS:
        for path in PATHS:exact([r['date'] for r in exports[arm,path]['DAILY']],[d for d in axis if meta['dates'][0]<=d<=meta['ends'][-1]],'Official daily dates')
    prc=ev.read(PRIOR/'results/COMPLETION.json',prior['result_completion_sha256']);need(prc['status']=='COMPLETE' and prc['plan_sha256']==prior['plan_sha256'] and prc['paths']==9 and prc['slices']==45,'Prior economic manifest')
    oldresult=ev.read(PRIOR/'results/RESULTS.json',prc['artifacts_sha256']['RESULTS.json']);priorcost=None
    for path in PATHS:
        for kind in KINDS:
            filename=ARMS[0]+'_'+path+'_'+kind+'.json';previous=ev.read(PRIOR/'results'/filename,prc['artifacts_sha256'][filename]);current=exports[ARMS[0],path][kind]
            if path=='next_open_cost' and kind=='HOLDINGS':priorcost=previous
            else:exact(current,previous,'Original economic export '+path+'/'+kind)
    fees=inherited.diagnostic_summary_parity(result['arms'][ARMS[0]],oldresult['arms'][ARMS[0]],exports[ARMS[0],'next_open_cost']['HOLDINGS'],priorcost)
    ev.finish()
    return dict(schema='monthly_boundary_pair267_saved_reader_evidence_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
        reviewer_role='Root source/host integration review plus separately authored numerical verifier; non-blind roles disclosed.',
        plan_sha256=args.plan_sha256,context_sha256=digest(CONTEXT),implementation_contract_sha256=digest(CONTRACT),prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,phase_terminals_sha256=args.terminals_sha256,root_owned_terminals=terminal,formal_synthetic_tests=tests['tests'],source=source_summary,model_summary=summary,execution_accounting=work,selection=selected,
        changed_holdings={a+'_vs_'+b:inherited.changes(orders,b,a) for a,b in ((ARMS[2],ARMS[0]),(ARMS[2],ARMS[1]),(ARMS[1],ARMS[0]))},saved_accounting_evidence=account,incumbent_economic_exports_exact=True,incumbent_summary_exact=fees['exact'],summary_fee_display_review=fees,
        full_results={a:{p:result['arms'][a][p]['metrics'] for p in PATHS} for a in ARMS},segmented_results={a:{p:result['arms'][a][p]['slices'] for p in PATHS} for a in ARMS},
        bindings=dict(code_files=10,prepared_files=len(PREP),result_files=len(RESULT),all_collected_pins_rechecked=True,package3309_manifest_bound=True,package_members_rehashed=False,full_registered_closure_rehashed=False),
        protection_and_close=dict(status='pending_root_close',protected_inventory_count=21444,protected_members_rehashed=False,final_root_close_verified=False,registered_pre_state_completed=262,registered_pre_state_trials=1297,expected_post_close_completed=263,expected_post_close_trials=1300),
        actual_models_computed_by_reviewer=False,actual_model_solve_count=0,npz_decoded=False,source_prices_loaded=False,actual_portfolio_replay=False,saved_nav_metrics_recomputed=False,phase_rerun=False,frozen_files_modified=False,observers_modified=False,strict_pit=False,reused_sample=True,
        limitations=['Reader inspects saved exact hexadecimal cohort arrays and evidence but does not refit models or recompute forecasts, labels, decay exponentials or source-price returns; those depend on the bound actually invoked source and independent verifier.',
        'Full source NPZ and panel were decoded by admitted producer. Physical prefix checks restrict numerical consumption; they are not a claim that future file bytes were never loaded.',
        'Canonical Gram/rhs share declared NumPy contractions. Separate augmented-SVD numerical algorithm and exact supplied-beta mapping are disclosed, not blind independent data collection.',
        'All later/full/cost results are diagnostic for this selection; repeated historical exploration is not untouched out-of-sample. Root close still must verify21444 protected files and advance counters.'],
        reader_execution=dict(script=str(Path(__file__).relative_to(ROOT)),sha256=digest(Path(__file__)),phase_digests=vars(args),report_written=False,terminal_exit_claimed=False),input_sha256=ev.pins)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','frozen','results','terminals'):parser.add_argument('--'+name+'-sha256',required=True)
    print(json.dumps(run(parser.parse_args()),sort_keys=True,indent=2,allow_nan=False))

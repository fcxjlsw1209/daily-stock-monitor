"""Independent149 prepare audit: old integer oracle plus same-report SBC.

No new factor, runner or admission producer import. Actual audit requires the
registered PLAN SHA and completed preparation; synthetic oracles do no I/O.
"""
from pathlib import Path
from collections import Counter, defaultdict
from decimal import Decimal
import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import numpy as np
import pandas as pd
import monthly_cash_quality146_prepare_review as legacy
import monthly_stock_comp149_review_contract as contract

ROOT=Path(__file__).resolve().parents[1]
STUDY='outputs/monthly_stock_comp149'
old=legacy.old
LOCAL=old.LOCAL
NAMES=contract.NAMES
METHOD=legacy.METHOD
DEFINITION=legacy.DEFINITION
MEASUREMENT=contract.MEASUREMENT
SBC_DEFINITION=contract.SBC_DEFINITION
EXTRA=('earnings_bounded','cash_flow_bounded','comp_adjusted_cash_flow_bounded','common_available')
CHECKS=legacy.CHECKS+('stock_comp_source_admission_verified','common_mask_subset_of146_verified')
ATOL=legacy.ATOL
INPUTS=legacy.INPUTS
CACHE_FILES=legacy.CACHE_FILES
CALENDAR_SHA=legacy.CALENDAR_SHA
SIDES=legacy.SIDES
FIELDS=legacy.FIELDS
REQUIRED=legacy.REQUIRED
require=legacy.require
day=legacy.day
text=legacy.text
decimal_integer=legacy.decimal_integer
verify_admission=legacy.verify_admission
META=('event_id','symbol','fiscal_year','release_date','available_from','identity','periods','source',
      'inherited_rejection_codes','cash_quality_admission')


def source_files(root=None):
    root=ROOT if root is None else Path(root).resolve()
    return sorted(set(legacy.source_files(root)+[root/'work'/name for name in (
        'monthly_stock_comp149_prepare_review.py','test_monthly_stock_comp149_prepare_review.py',
        'monthly_stock_comp149_review_contract.py','stock_comp_source_admission_v1_independent.py',
        'test_stock_comp_source_admission_v1_independent.py')]))


def scaled_integer(cell,event,side):
    require(type(cell) is dict,'Required cell missing')
    require(cell.get('source')==event['source'],'Cell source mismatch')
    require(cell.get('period')==event['periods'][side],'Cell period mismatch')
    require(cell.get('available_from')==event['available_from'],'Cell availability mismatch')
    require(cell.get('currency_literal')=='$','Same-report dollar literal required')
    for key in ('value_decimal','scale'):
        require(text(cell[key]) and re.fullmatch(r'-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',cell[key]),
                'Explicit finite decimal string required')
    a,e=decimal_integer(cell['value_decimal']);b,f=decimal_integer(cell['scale'])
    require(b>0,'Positive declared scale required')
    return a*b,e+f


def bounded_from_cells(values):
    """Five scaled integer pairs -> E, C, C-SBC with one final division."""
    require(len(values)==5,'Five current/prior source amounts required')
    exponent=min(e for _,e in values)
    cash,earnings,current,previous,sbc=[c*10**(e-exponent) for c,e in values]
    require(current>0 and previous>0,'stock_compensation_nonpositive_assets')
    denominator=current+previous
    nums=(2*earnings,2*cash,2*(cash-sbc))
    return tuple(n/(denominator+abs(n)) for n in nums)


def adjusted_oracle(original,candidate):
    gate=candidate.get('stock_compensation_admission')
    require(type(gate) is dict and set(gate)=={'admitted','reasons','definition'},'stock_compensation_admission_required')
    require(gate['admitted'] is True,'current_not_stock_compensation_admitted')
    require(gate['definition']==SBC_DEFINITION,'wrong_stock_compensation_definition')
    require(type(gate['reasons']) is list and not gate['reasons'],'stock_compensation_admission_has_reasons')
    require(all(k in original and k in candidate and candidate[k]==original[k] for k in META),'stock_compensation_event_metadata_mismatch')
    if 'current_release_available_from' in candidate:
        require(candidate['current_release_available_from']==original['available_from'],'stock_compensation_event_availability_mismatch')
    component=candidate.get('stock_compensation')
    require(type(component) is dict and component.get('field')=='stock_compensation_addback','stock_compensation_raw_field_required')
    require(type(component.get('gaps')) is list and not component['gaps'],'stock_compensation_raw_field_has_gaps')
    cell=component.get('value');require(type(cell) is dict,'stock_compensation_value_missing')
    require(cell.get('source_sha256')==original['source']['sha256'],'stock_compensation_source_sha_mismatch')
    try:
        sbc=scaled_integer(cell,original,'current')
        cfo=original['fields']['current']['operating_cash_flow']
        require(Decimal(cell['scale'])==Decimal(cfo['scale']),'stock_compensation_cashflow_scale_mismatch')
        require(cell['currency_literal']==cfo['currency_literal'],'stock_compensation_cashflow_currency_mismatch')
        values=[scaled_integer(original['fields'][side][field],original,side) for side,field in REQUIRED]
    except (KeyError,TypeError,ArithmeticError) as exc:
        raise ValueError('stock_compensation_invalid_monetary_cell') from exc
    return bounded_from_cells(values+[sbc])[2]


def features_oracle(meta,dataset,sbc_dataset):
    original=legacy.features_oracle(meta,dataset)
    require(type(sbc_dataset) is dict and type(sbc_dataset.get('schema_version')) is int and
            sbc_dataset['schema_version']==1 and sbc_dataset.get('method')==MEASUREMENT,'Explicit SBC dataset schema required')
    indexed=[]
    for events in (dataset['events'],sbc_dataset.get('events')):
        require(type(events) is list,'Complete event list required')
        group=defaultdict(list)
        for event in events:
            require(type(event) is dict and all(text(event.get(k)) for k in ('event_id','symbol')),'Declared source event identity required')
            group[(event['symbol'],event['event_id'])].append(event)
        indexed.append(group)
    originals,components=indexed
    arrays={key:np.full(original['arrays']['common_available'].shape,np.nan) for key in EXTRA[:-1]}
    arrays['common_available']=np.zeros(original['arrays']['common_available'].shape,bool)
    cells=[]
    for prior in original['cells']:
        cell={k:v for k,v in prior.items() if k!='low_accrual_bounded'}
        cell.update(measurement=MEASUREMENT,original_cash_quality_available=prior['available'],
                    comp_adjusted_cash_flow_bounded=None,stock_compensation_event_id=None,available=False)
        for key in EXTRA[:-1]:cell[key]=None
        if prior['available']:
            key=(cell['symbol'],cell['current_event_id'])
            chosen=[e for e in originals[key] if e['available_from']==cell['current_available_from']]
            require(chosen and all(e==chosen[0] for e in chosen),'Unique original146 selected report required')
            candidates=components.get(key,[])
            if not candidates:cell['reason']='current_stock_compensation_event_missing'
            elif any(e!=candidates[0] for e in candidates):cell['reason']='conflicting_stock_compensation_event'
            else:
                try:value=adjusted_oracle(chosen[0],candidates[0])
                except ValueError as exc:cell['reason']=str(exc)
                else:
                    i,j=cell['signal_index'],cell['pool_index']
                    cell.update(available=True,reason='available',stock_compensation_event_id=cell['current_event_id'],
                                comp_adjusted_cash_flow_bounded=value)
                    arrays['common_available'][i,j]=True;arrays[EXTRA[2]][i,j]=value
                    for name in EXTRA[:2]:arrays[name][i,j]=original['arrays'][name][i,j];cell[name]=prior[name]
        cells.append(cell)
    require(not np.any(arrays['common_available'] & ~original['arrays']['common_available']),'Common mask must narrow original146')
    return dict(arrays=arrays,cells=cells,original_cash_quality_available_cells=int(original['arrays']['common_available'].sum()))


def score_oracle(arrays,meta,earnings,cashflow,adjusted,n=None):
    result=legacy.score_oracle(arrays,meta,earnings,cashflow,adjusted,n=n)
    return tuple({new:group[prior] for new,prior in zip(NAMES,legacy.NAMES)} for group in result)


def verify_original_arrays(original,prepared):
    require(len(original)==27 and set(prepared)==set(original)|set(EXTRA),'Original27 plus four new arrays')
    for key,value in original.items():
        candidate=prepared[key]
        require(value.shape==candidate.shape and value.dtype==candidate.dtype and value.tobytes()==candidate.tobytes(),'Original array changed: '+key)


def verify_audit(saved,expected,*,dataset_sha=None,admission_sha=None,sbc_refs=None):
    require(type(saved.get('new_model_fits')) is int and saved['new_model_fits']==0 and saved.get('returns_or_nav_computed') is False,'No new fits/returns')
    for key in ('original_arrays_unchanged','original_auxiliary_states_and_core_coefficients_preserved','latest_event_before_validity_filter',
                'same_current_release_for_all_cells','earnings_and_cashflow_controls_identical_to146_on_common_mask',
                'common_mask_subset_of146','same_selected_current_event_for_all_fields','signed_stock_compensation_preserved',
                'stock_compensation_raw_flags_do_not_substitute_new_admission','adjusted_cashflow_is_research_measure_not_reported_GAAP_cashflow'):
        require(saved.get(key) is True,'Required audit claim '+key)
    for key in ('annual_duration_normalization','comparison_flow_values_used'):require(saved.get(key) is False,'Forbidden transformation '+key)
    for key,value in (('measurement',MEASUREMENT),('stock_compensation_definition',SBC_DEFINITION),
        ('arithmetic_method','exact_fraction_scaled_cashflow_minus_signed_sbc_over_average_assets_then_bounded_float'),
        ('source_verification','caller_verified_original146_and_new_stock_compensation_admission_closures_required'),
        ('currency_scope','same_report_dollar_literal_assumption_not_ISO_certification')):
        require(saved.get(key)==value,'Audit method '+key)
    if dataset_sha is not None or admission_sha is not None:
        require(dataset_sha and admission_sha and saved.get('admitted_dataset_sha256')==dataset_sha and saved.get('source_admission_sha256')==admission_sha,'Original source audit binding')
    if sbc_refs is not None:
        require(set(sbc_refs)==set(contract.SBC_REFS),'All four source bindings')
        for key,ref in sbc_refs.items():require(saved.get(key+'_sha256')==ref['sha256'],'SBC source audit binding '+key)
    require(saved.get('common_available_cells')==int(expected['arrays']['common_available'].sum()) and
            saved.get('original_cash_quality_available_cells')==expected['original_cash_quality_available_cells'],'Availability counts')
    require(saved.get('original_months')==len(expected['cells'])//49 and saved.get('original_width')==49,'Audit axes')
    require(saved.get('reasons')==dict(Counter(c['reason'] for c in expected['cells'])),'Full reason census')
    require(len(saved.get('cells',[]))==len(expected['cells']),'All cells audited')
    for actual,wanted in zip(saved['cells'],expected['cells']):
        require(set(actual)==set(wanted),'Exact audit cell schema')
        for key,value in wanted.items():
            require(type(actual.get(key)) is type(value),'Audit cell type '+key)
            if type(value) is float:require(math.isclose(actual[key],value,abs_tol=ATOL,rel_tol=0),'Audit cell value '+key)
            else:require(actual[key]==value,'Audit cell value '+key)


def ordered_orders(meta,scores,choices):
    old_scores={oldname:scores[new] for new,oldname in zip(NAMES,legacy.NAMES)}
    old_choices={oldname:choices[new] for new,oldname in zip(NAMES,legacy.NAMES)}
    result=legacy.ordered_orders(meta,old_scores,old_choices)
    return {new:result[oldname] for new,oldname in zip(NAMES,legacy.NAMES)}


def verify_scores_and_orders(actual_scores,actual_orders,meta,expected_scores,expected_choices):
    require(set(actual_scores)==set(actual_orders)==set(NAMES),'Exactly four saved arms')
    orders=ordered_orders(meta,expected_scores,expected_choices)
    for name in NAMES:
        require(actual_scores[name].shape==expected_scores[name].shape and actual_scores[name].dtype==expected_scores[name].dtype,'Full score matrix axes/dtype')
        np.testing.assert_allclose(actual_scores[name],expected_scores[name],rtol=0,atol=ATOL)
        require(len(actual_orders[name])==len(orders[name]),'Every month ordered')
        for actual,wanted in zip(actual_orders[name],orders[name]):
            require(set(actual)==set(wanted) and all(actual[k]==wanted[k] for k in ('date','signal_date','end','symbols')),'Ordered holding mismatch')
            np.testing.assert_allclose(actual['scores'],wanted['scores'],rtol=0,atol=ATOL)


def verify_prefixes(arrays,meta,dataset,sbc_dataset,expected,scores,choices):
    for n in range(len(meta['dates'])+1):
        subset=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
        historical=dict(dataset,events=[e for e in dataset['events'] if n and e['available_from']<=meta['dates'][n-1]])
        historical_sbc=dict(sbc_dataset,events=[e for e in sbc_dataset['events'] if n and e['available_from']<=meta['dates'][n-1]])
        prefix=features_oracle(subset,historical,historical_sbc)
        require(prefix['cells']==expected['cells'][:n*49],'Source prefix isolation')
        for key in EXTRA:np.testing.assert_array_equal(prefix['arrays'][key],expected['arrays'][key][:n])
        ss,cc,_=score_oracle(arrays,meta,*(prefix['arrays'][k] for k in EXTRA[:3]),n=n)
        for name in NAMES:require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(),'Score/holding prefix isolation')
    for fill in (np.nan,0.):
        hand=np.full((len(meta['dates']),len(meta['symbols'][0])),np.nan)
        for i,names in enumerate(meta['symbols']):
            for j,symbol in enumerate(names):
                if symbol not in meta['etfs']:hand[i,j]=fill
        ss,cc,_=score_oracle(arrays,meta,hand,hand,hand)
        for name in NAMES[1:]:require(ss[name].tobytes()==scores[NAMES[0]].tobytes() and cc[name]==choices[NAMES[0]],'Exact zero/missing incumbent fallback')
def audit(root,plan_path,plan_sha):
 require(type(plan_sha)is str and re.fullmatch('[0-9a-f]{64}',plan_sha),'Explicit PLAN SHA required before any file read')
 reader=old.Reader(root);directory=reader.path(STUDY)
 require(reader.path(plan_path)==directory/'PLAN.json','Dedicated round149 PLAN')
 require(reader.sha(plan_path)==plan_sha,'Frozen PLAN hash');plan=reader.json(plan_path)
 require(plan.get('round')==149 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES) and plan.get('primary_challenger')==NAMES[3],'Round149 frozen arms')
 require(reader.path(plan['output_directory'])==directory and plan['dose']==.01,'Output and dose policy')
 require(plan['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),'Pinned runtime')
 require(not any((directory/name).exists() for name in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','WALKFORWARD.json','FROZEN.json','RESULTS.json')),'Preparation audit must precede outcomes')
 pins=reader.verify(plan['hashes']);require(set(plan['inputs'])==set(INPUTS),'Named inputs')
 inputs={}
 for name,ref in plan['inputs'].items():
  path=reader.path(ref['path']);require(reader.sha(path)==ref['sha256']==pins.get(str(path)),'Bound input: '+name);inputs[name]=path
 contract.verify_proposal(reader,plan,pins)
 for key in ('actual_model_fits_before_registration','actual_feature_calculations_before_registration','actual_returns_before_registration'):
  require(type(plan.get(key))is int and plan[key]==0,'Prospective registration')
 require(plan['tolerances']['feature_atol']==ATOL and plan['tolerances']['scores_atol']==ATOL and plan['tolerances']['scores_rtol']==0,'Exact preparation tolerances')
 sources={str(path):reader.sha(path) for path in source_files(root)}
 require(reader.verify(plan['independent_prepare_source_hashes'])==sources and all(pins.get(k)==v for k,v in sources.items()),'Independent source closure')
 oldplan=reader.json(LOCAL+'/PLAN.json');raw=reader.verify(plan['raw_source_hashes'])
 require(raw==reader.verify(oldplan['raw_source_hashes']) and len(raw)==64 and all(pins.get(k)==v for k,v in raw.items()),'Original raw64 unchanged')
 original_cache=reader.verify(reader.json(LOCAL+'/CACHE_HASHES.json'),reader.path(LOCAL))
 require(all(pins.get(k)==v for k,v in original_cache.items()),'Original cache bindings')
 for name,file in (('meta','META.json'),('policy','INCUMBENT_POLICY.json'),('original_orders','local_linear_ORDERS.json')):
  require(inputs[name]==reader.path(LOCAL+'/'+file),'Original local67 input: '+name)
 require(inputs['prices']==reader.path(old.PRICES),'Original prices')
 helper=old.original_helper();policy=reader.json(inputs['policy'])
 require(all(policy.get(k)==v for k,v in helper.POLICY.items()),'Frozen episode policy')
 cache=reader.json(directory/'CACHE.json');require(set(cache)==set(CACHE_FILES),'Six preparation cache artifacts');reader.verify(cache,directory)
 meta=reader.json(directory/'META.json');require(meta==reader.json(inputs['meta']),'Exact original metadata')
 require(len(meta['dates'])==len(meta['symbols'])==122 and meta['train_months']==88 and meta['early_months']==53 and len(meta['etfs'])==17 and all(len(row)==49 for row in meta['symbols']),'Original122x49 shape')
 original=reader.npz(LOCAL+'/DATA.npz');arrays=reader.npz(directory/'DATA.npz');verify_original_arrays(original,arrays)
 require(reader.sha(old.LOADER)==old.LOADER_SHA,'Frozen raw price loader')
 spec=importlib.util.spec_from_file_location('cash_quality146_independent_price_loader',reader.path(old.LOADER));loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
 panel,info=loader.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(reader.path(old.LOADER))]=old.LOADER_SHA
 require({str(reader.path(k)):v for k,v in used.items()}==raw,'Exactly the registered raw64 consumed');reader.verify(used)
 prices=reader.npz(inputs['prices']);require(set(prices)=={'dates','symbols','close','opening'},'Original price cache schema')
 parity=old.price_parity(panel,prices);old.compare(reader.json(directory/'PRICE_PARITY.json'),parity,'PRICE_PARITY')
 calendar_path=inputs['sessions'].parent/'CALENDAR_PLAN.json';require(reader.sha(calendar_path)==CALENDAR_SHA,'Pinned source calendar')
 calendar=reader.json(calendar_path);require(calendar['start']=='2015-01-01' and calendar['end']=='2027-12-31','Calendar coverage')
 require(all(pins.get(k)==v for k,v in reader.verify(calendar['bindings']).items()),'Calendar provenance')
 sessions=old.official_sessions(reader,calendar['end']);sessions=[str(d.date()) for d in sessions if str(d.date())>=calendar['start']]
 saved=reader.json(inputs['sessions']);require(saved['plan_sha256']==CALENDAR_SHA and saved['sessions']==sessions,'Complete official session overlay')
 dataset=verify_admission(reader,plan,pins,sessions)
 sbc_dataset=contract.verify_stock_comp_source(reader,plan,pins,dataset)
 expected=features_oracle(meta,dataset,sbc_dataset)
 verify_audit(reader.json(directory/'FEATURE_AUDIT.json'),expected,
  dataset_sha=reader.sha(inputs['admitted_dataset']),admission_sha=reader.sha(inputs['source_admission']),
  sbc_refs={key:plan[key] for key in contract.SBC_REFS})
 for key in EXTRA:
  require(arrays[key].shape==(122,49) and arrays[key].dtype==expected['arrays'][key].dtype,'New feature dtype/shape')
  np.testing.assert_array_equal(arrays[key],expected['arrays'][key])
 scores,choices,deltas=score_oracle(arrays,meta,*(expected['arrays'][key] for key in EXTRA[:3]))
 saved_scores=reader.npz(directory/'SCORES.npz');saved_orders=reader.json(directory/'PRETRAIN_ORDERS.json')
 verify_scores_and_orders(saved_scores,saved_orders,meta,scores,choices)
 helper.assert_exact_tree(saved_orders[NAMES[0]],reader.json(inputs['original_orders']))
 require(saved_scores[NAMES[0]].tobytes()==scores[NAMES[0]].tobytes(),'Incumbent score bytes')
 verify_prefixes(arrays,meta,dataset,sbc_dataset,expected,scores,choices)
 required={str(directory/name):reader.sha(directory/name) for name in (*CACHE_FILES,'PLAN.json','CACHE.json')};allowed=dict(pins,**required)
 for path,digest in list(reader.seen.items()):require(reader.sha(path)==digest and allowed.get(path)==digest,'Unbound consumed artifact: '+path)
 return dict(schema_version=1,stage='prepare',passed=True,plan_sha256=plan_sha,cache_sha256=reader.sha(directory/'CACHE.json'),orders_sha256=reader.sha(directory/'PRETRAIN_ORDERS.json'),
  **{key:True for key in CHECKS},source_hashes=sources,artifact_hashes=dict(reader.seen),
  producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,new_model_fits=0,
  orders_checked=488,source_cells_checked=5978,stock_cells_checked=3904,available_stock_cells=int(expected['arrays']['common_available'].sum()),
  exact_integer_closed_form_arithmetic=True,source_admission_bound=True,source_union_preservation_verified=True,
  raw_prices_independently_verified=True,calendar_overlay_independently_verified=True,all_signal_prefixes_verified=True,
  max_score_error=max(float(np.max(np.abs(saved_scores[name]-scores[name]))) for name in NAMES),
  limitations=['Raw source assumptions and inherited holds remain; the new gate is conditional research admission.',
   'Integer-scaled bounded ratios and all four ordered score paths are checked without importing the new producer; no strategy returns are computed.'])


def synthetic_receipt(result,root=None):
    """Only test results and source hashes; never evaluate real observations."""
    root=ROOT if root is None else Path(root).resolve()
    sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files(root)}
    reviewed=[root/'work'/name for name in (
        'monthly_stock_comp149_runner.py','monthly_stock_comp149_factor.py',
        'monthly_stock_comp149_proposal.json','after148_next_mechanism_review.json',
        'monthly_cash_quality146_factor.py','monthly_forward_guidance132_runner.py')]
    return dict(schema_version=1,stage='synthetic_prepare_review_only',passed=result.wasSuccessful(),
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),source_hashes=sources,
        reviewed_contract_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in reviewed},
        method=METHOD,measurement=MEASUREMENT,arms=list(NAMES),features=list(EXTRA),required_checks=list(CHECKS),
        cache_files=list(CACHE_FILES),required_inputs=list(INPUTS),
        required_source_policy_reference=['path','sha256'],required_stock_comp_references=contract.SBC_REFS,
        actual_preparation_checked=False,actual_factors_calculated=0,actual_returns_calculated=0,
        producer_modules_imported=False,source_gate_separate_from_signal_day_validity=True,
        actual_api='./run_local.sh work/monthly_stock_comp149_prepare_review.py --plan <PLAN.json> --plan-sha <SHA256> --prepare-complete',
        actual_output=STUDY+'/INDEPENDENT_PREPARE.json',
        pending_requirements=['Root supplies frozen round149 PLAN SHA after registration and completed preparation.',
            'PLAN.hashes binds all four SBC source files, their complete source closure, the original146 source gate, raw64, original cache, and independent source closure.'],
        scope_limits=['Synthetic evidence only: no real factors, orders, returns, or source eligibility changes.',
            'The three ratios share the selected latest report and one common mask; inherited source holds and certification limits remain.'])


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--describe',action='store_true');parser.add_argument('--plan-sha')
 parser.add_argument('--plan',type=Path,default=ROOT/STUDY/'PLAN.json');parser.add_argument('--prepare-complete',action='store_true')
 parser.add_argument('--output',type=Path,default=ROOT/STUDY/'INDEPENDENT_PREPARE.json');args=parser.parse_args()
 if args.describe:
  print(json.dumps(dict(stage='oracles_and_registered_audit_ready',method=METHOD,arms=NAMES,features=EXTRA,checks=CHECKS,actual_checks_run=False,producer_modules_imported=False)));return
 require(args.prepare_complete,'Explicit completed preparation required')
 result=audit(ROOT,args.plan,args.plan_sha)
 require(args.output.resolve()==ROOT/STUDY/'INDEPENDENT_PREPARE.json','Dedicated independent output')
 with args.output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(passed=True,output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))

if __name__=='__main__':main()

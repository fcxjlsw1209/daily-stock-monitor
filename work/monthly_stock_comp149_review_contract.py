"""Independent round149 bindings; no strategy or factor producer imports."""
import sys
import stock_comp_source_admission_v1_independent as source_oracle

CONCEPT='work/after148_next_mechanism_review.json'
CONCEPT_SHA='dc4237f5c4382f40800bb8cbb2b8268c069e80ba10703fac60d860408f70c1de'
DATA_SHA='3725b384ab4f48a3cabee1fc1adb1acd896be3f4967e8b6cd649a9a690c953d5'
NAMES=('incumbent_local67','earnings_control','cashflow_control','comp_adjusted_cashflow')
MEASUREMENT='current_annual_stock_compensation_adjusted_cash_flow_v1'
SBC_DEFINITION='current_annual_reported_stock_compensation_addback_v1'
PATHS=(('close',0.,'same_close'),('open',0.,'next_open_zero'),('open',.0005,'next_open_cost'))
SBC_REFS={key:'outputs/monthly_stock_comp_source_admission_v2/'+file for key,file in (
    ('sbc_dataset','DATASET.json'),('sbc_admission','ADMISSION.json'),('sbc_policy','POLICY.json'),('sbc_review','INDEPENDENT_REVIEW.json'))}


def require(value,message):
    if not value:raise ValueError(message)


def exact(a,b):
    require(type(a) is type(b),'Contract type mismatch')
    if type(a) is dict:
        require(set(a)==set(b),'Contract keys mismatch')
        for key in a:exact(a[key],b[key])
    elif type(a) is list:
        require(len(a)==len(b),'Contract length mismatch')
        for x,y in zip(a,b):exact(x,y)
    else:require(a==b,'Contract value mismatch')


def proposal_contract():
    return dict(total_months=122,training_months=88,early_months=53,recent_months=35,later_months=34,
        walkforward=[[36,60],[60,76],[76,88]],execution_paths=[dict(execution=x,cost=c,key=k) for x,c,k in PATHS],
        same_close_cagr='strict_greater_than_all_three_controls',
        next_open_cost_cagr='noninferior_incumbent_strict_greater_than_two_controls',strict_cagr_epsilon=1e-12,
        max_sharpe_degradation=.05,max_drawdown_degradation=.03,max_early_cagr_degradation=.025,
        max_recent_cagr_degradation=.025,new_model_fits=0,intramonth_trades=False,later34_for_selection=False)


def measurement_contract():
    return dict(method=MEASUREMENT,average_assets='(A_current+A_comparison)/2',
        earnings='NI_current/average_assets',cashflow='CFO_current/average_assets',
        comp_adjusted_cashflow='(CFO_current-SBC_current)/average_assets',bounded='h(x)=x/(1+abs(x))',
        exact_arithmetic='scaled_Fraction_then_one_bounded_float',
        common_mask='original146_valid_and_same_selected_latest_event_SBC_admitted',
        unavailable='all_three_features_NaN_then_exact_incumbent_fallback',
        placement='add_0.01_h_to_original_pre_score_then_one_original_episode_selector',
        time='current_release_next_pinned_session_contiguous_330_400_day_years_550_day_expiry',
        invalid_latest_blocks_fallback=True,unchanged_original_arrays=27,monthly_equal_weight_positions=3,
        intramonth_trades=False,etf_financial_features=False,no_parameter_search=True)


def bound_ref(reader,ref,pins,message):
    require(type(ref) is dict and set(ref)=={'path','sha256'},message+' reference')
    path=reader.path(ref['path'])
    require(reader.sha(path)==ref['sha256']==pins.get(str(path)),message+' digest')
    return path


def verify_proposal(reader,plan,pins):
    proposal=reader.json(bound_ref(reader,plan['inputs']['proposal'],pins,'Proposal'))
    require(proposal.get('round')==149 and proposal.get('arms')==list(NAMES) and
            proposal.get('primary_challenger')==NAMES[3] and proposal.get('dose')==.01,'Fixed149 arms/dose')
    exact(proposal.get('selection_contract'),proposal_contract())
    exact(proposal.get('measurement'),measurement_contract())
    for key in ('concept','novelty'):
        ref=proposal.get(key)
        require(type(ref) is dict and reader.path(ref.get('path'))==reader.path(CONCEPT) and
                ref.get('sha256')==CONCEPT_SHA,'Fixed prospective source hypothesis')
        reader.json(bound_ref(reader,ref,pins,key))
    require(proposal.get('status')=='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION','Prospective stage')
    for key in ('actual_new_factors','actual_new_orders','actual_new_returns'):
        require(type(proposal.get(key)) is int and proposal[key]==0,'Prospective computations')
    sources=proposal.get('sources');require(type(sources) is dict,'Proposal sources')
    refs=dict(policy=plan['source_policy'],dataset=plan['inputs']['admitted_dataset'],admission=plan['inputs']['source_admission'],
              **{key:plan[key] for key in SBC_REFS})
    for key,ref in refs.items():
        actual=sources.get(key)
        require(type(actual) is dict and set(actual)=={'path','sha256'} and
                reader.path(actual['path'])==reader.path(ref['path']) and actual['sha256']==ref['sha256'],'Proposal source '+key)
        bound_ref(reader,ref,pins,key)
    receipt=reader.json(plan['inputs']['source_admission']['path'])
    require(sources.get('full_inventory')==receipt.get('events')==44 and
            sources.get('admitted')==receipt.get('admitted_events')==31 and
            sources.get('held')==receipt.get('held_events')==13,'Original31/13 source census')
    return proposal


def verify_stock_comp_source(reader,plan,pins,data):
    require(plan['inputs']['admitted_dataset']['sha256']==DATA_SHA,'Unchanged146 dataset')
    objects={}
    for key,path in SBC_REFS.items():
        ref=plan[key]
        require(reader.path(ref['path'])==reader.path(path),'Exact SBC source path')
        objects[key]=reader.json(bound_ref(reader,ref,pins,key))
    dataset,receipt,policy,review=(objects[k] for k in SBC_REFS)
    source_oracle.policy_contract(policy)
    require(policy.get('runtime')==sys.version,'SBC source runtime')
    require(dataset.get('schema_version')==1 and dataset.get('method')==MEASUREMENT and
            dataset.get('conditional_research_only') is True and
            dataset.get('policy_sha256')==receipt.get('policy_sha256')==plan['sbc_policy']['sha256'],'SBC dataset/policy identity')
    require(receipt.get('passed') is True and receipt.get('stage')=='conditional_stock_compensation_source_admission'
            and receipt.get('data_sha256')==plan['sbc_dataset']['sha256'],'SBC source admission')
    require(review.get('passed') is True and review.get('stage')=='independent_conditional_stock_compensation_source_review'
            and review.get('source_only') is True and review.get('policy_sha256')==plan['sbc_policy']['sha256'],'Independent SBC review')
    for obj,keys in ((receipt,('financial_ratios','factor_calculations','model_fits','return_calculations','observer_changes')),
                     (review,('financial_ratios','factor_calculations','return_calculations'))):
        for key in keys:require(type(obj.get(key)) is int and obj[key]==0,'Source-only zero counter '+key)
        for key in source_oracle.CERTS:require(obj.get(key) is False,'Uncertified source assumption '+key)
    for key in ('all_source_candidates_preserved','original_rejections_preserved','independent_decisions_match',
                'three_patch_cells_bound_to_original_sources','all_present_values_match_independent_physical_or_visual_proof'):
        require(review.get(key) is True,'Independent review check '+key)
    require(review.get('producer_admission_code_imported') is False,'Independent source oracle required')
    require(receipt.get('source_values_independently_reviewed') is True and receipt.get('accounting_flags_preserved') is True
            and receipt.get('accounting_flag_disposition')==policy.get('accounting_changes'),'Explicit accounting disposition')
    admitted_pins=reader.verify(receipt['input_hashes']);reviewed_pins=reader.verify(review['input_hashes'])
    codes=reader.verify(policy['source_code_hashes'])
    require(all(pins.get(k)==v for group in (admitted_pins,reviewed_pins,codes) for k,v in group.items()),'Full registered SBC source closure')
    needed=dict(admitted_pins,**{str(reader.path(plan[k]['path'])):plan[k]['sha256'] for k in ('sbc_dataset','sbc_admission','sbc_policy')})
    require(all(reviewed_pins.get(k)==v for k,v in needed.items()),'Independent SBC review omitted dependency')
    refs=policy.get('inputs',{})
    require(set(refs)=={'raw','source_plan','source_manifest','html_review','pdf_visual','patch_plan','patch','patch_audit',
                        'patch_manifest','old_dataset','old_admission','old_review','sessions','tests'},'Exact SBC source input set')
    evidence={}
    for key,ref in refs.items():
        path=bound_ref(reader,ref,pins,'SBC input '+key)
        require(admitted_pins.get(str(path))==reviewed_pins.get(str(path))==ref['sha256'],'Source review and admission share input '+key)
        evidence[key]=reader.json(path)
    require(evidence['old_dataset']==data,'Same old raw dataset')
    for name,target in (('old_dataset',plan['inputs']['admitted_dataset']),('old_admission',plan['inputs']['source_admission']),
                        ('sessions',plan['inputs']['sessions'])):
        require(reader.path(refs[name]['path'])==reader.path(target['path']) and refs[name]['sha256']==target['sha256'],'Shared input '+name)
    candidates=source_oracle.patched_candidates(evidence['raw']['events'],evidence['patch']['events'])
    originals=source_oracle.index(data['events']);verified=set()
    html=evidence['html_review'];require(html.get('passed') is True,'HTML source review')
    rows=source_oracle.index(html['event_reviews'])
    require(set(rows)==set(originals),'HTML review full census')
    for eid,row in rows.items():
        if row['status']=='HTML_physical_cell_verified':
            require(row['cell']['verified'] is True and originals[eid]['symbol']!='AAPL' and
                    row['cell']['source_sha256']==originals[eid]['source']['sha256'],'HTML source proof')
            verified.add(eid)
    visual=evidence['pdf_visual'];vrows=source_oracle.index(visual['events'])
    require(visual.get('passed_visual_identification') is True and visual.get('pages_reviewed')==10 and
            set(vrows)=={eid for eid,e in originals.items() if e['symbol']=='AAPL'},'All ten Apple source proofs')
    for eid,row in vrows.items():
        value=candidates[eid]['stock_compensation']['value']
        require(value is not None and all(value[k]==row[k] for k in ('value_decimal','source','scale')) and
                value['period']['period_end']==row['current_period_end'],'PDF visual candidate agreement')
        verified.add(eid)
    admitted,holds=source_oracle.verify_output_events(data['events'],candidates,dataset['events'],verified)
    for obj in (receipt,review):
        require(obj.get('events')==44 and obj.get('admitted_event_ids')==admitted and obj.get('holds')==holds and
                obj.get('admitted_events')==len(admitted) and obj.get('held_events')==len(holds),'Exact SBC decision partition')
    return dataset

"""Independent round147 semantic bindings; no producer imports or market I/O."""
from pathlib import Path

CONCEPT = 'work/monthly_asset_growth147_concept.json'
CONCEPT_SHA = '0a08b1818dfafe93e4267a35ee599fb1365741098d1e99ff3d3b30ed598c69d3'
DATA_SHA = '3725b384ab4f48a3cabee1fc1adb1acd896be3f4967e8b6cd649a9a690c953d5'
NAMES = ('incumbent_local67', 'earnings_control', 'low_accrual_control', 'low_asset_growth')
MEASUREMENT = 'same_report_total_asset_expansion_v1'
PATHS = (('close', 0., 'same_close'), ('open', 0., 'next_open_zero'),
         ('open', .0005, 'next_open_cost'))


def require(value, message):
    if not value:
        raise ValueError(message)


def proposal_contract():
    return dict(total_months=122, training_months=88, early_months=53, recent_months=35,
        later_months=34, walkforward=[[36,60],[60,76],[76,88]],
        execution_paths=[dict(execution=x,cost=c,key=k) for x,c,k in PATHS],
        same_close_cagr='strict_greater_than_all_three_controls',
        next_open_cost_cagr='noninferior_incumbent_strict_greater_than_two_controls',
        strict_cagr_epsilon=1e-12, max_sharpe_degradation=.05,
        max_drawdown_degradation=.03, max_early_cagr_degradation=.025,
        max_recent_cagr_degradation=.025, new_model_fits=0,
        intramonth_trades=False, later34_for_selection=False)


def bound_ref(reader, ref, pins, message):
    require(type(ref) is dict and set(ref)=={'path','sha256'}, message+' reference')
    path=reader.path(ref['path'])
    require(reader.sha(path)==ref['sha256']==pins.get(str(path)), message+' digest')
    return path


def verify_proposal(reader, plan, pins):
    ref=plan['inputs']['proposal'];path=bound_ref(reader,ref,pins,'Proposal')
    proposal=reader.json(path)
    require(proposal.get('round')==147 and proposal.get('arms')==list(NAMES)
        and proposal.get('primary_challenger')==NAMES[3] and proposal.get('dose')==.01,
        'Fixed147 proposal arms and dose')
    # Type-exact comparison prevents bool/int equivalence in nested gates.
    def exact(a,b):
        require(type(a) is type(b),'Selection contract type')
        if type(a) is dict:
            require(set(a)==set(b),'Selection contract keys')
            for key in a:exact(a[key],b[key])
        elif type(a) is list:
            require(len(a)==len(b),'Selection contract lengths')
            for x,y in zip(a,b):exact(x,y)
        else:require(a==b,'Selection contract value')
    exact(proposal.get('selection_contract'),proposal_contract())
    cref=proposal.get('concept')
    require(type(cref) is dict and set(cref)=={'path','sha256'}
        and reader.path(cref['path'])==reader.path(CONCEPT)
        and cref['sha256']==CONCEPT_SHA,'Fixed147 concept')
    bound_ref(reader,cref,pins,'Concept');reader.json(cref['path'])
    require(proposal.get('status')=='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',
        'Prospective proposal stage')
    for key in ('actual_new_factors','actual_new_orders','actual_new_returns'):
        require(type(proposal.get(key)) is int and proposal[key]==0,'Prospective computations')
    sources=proposal.get('sources');require(type(sources) is dict,'Proposal sources')
    required={'policy':plan['source_policy'],'dataset':plan['inputs']['admitted_dataset'],
        'admission':plan['inputs']['source_admission'],
        'asset_use_policy':plan['asset_use_policy'],'asset_use_review':plan['asset_use_review']}
    for key,ref in required.items():
        actual=sources.get(key)
        require(type(actual) is dict and set(actual)=={'path','sha256'}
            and reader.path(actual['path'])==reader.path(ref['path'])
            and actual['sha256']==ref['sha256'],'Proposal source '+key)
        bound_ref(reader,ref,pins,key)
    receipt=reader.json(plan['inputs']['source_admission']['path'])
    require(sources.get('full_inventory')==receipt.get('events')==44
        and sources.get('admitted')==receipt.get('admitted_events')==31
        and sources.get('held')==receipt.get('held_events')==13,'Fixed31/13 source census')
    return proposal


def verify_asset_use(reader, plan, pins, dataset):
    policy_ref,review_ref=plan['asset_use_policy'],plan['asset_use_review']
    policy=reader.json(bound_ref(reader,policy_ref,pins,'Asset-use policy'))
    review=reader.json(bound_ref(reader,review_ref,pins,'Asset-use review'))
    require(plan['inputs']['admitted_dataset']['sha256']==DATA_SHA,'Unchanged146 dataset')
    require(policy.get('schema_version')==1 and policy.get('round')==147
        and policy.get('stage')=='prospective_source_use_policy'
        and policy.get('purpose')==MEASUREMENT
        and policy.get('dataset_method')==dataset['method']=='current_annual_cash_quality_v1',
        'New purpose on existing dataset')
    for key in ('same_current_report','current_release_next_pinned_session','contiguous_periods',
                'latest_event_before_validity','invalid_latest_blocks_older_fallback',
                'same_report_currency_literal_assumption','old_source_policy_unchanged'):
        require(policy.get(key) is True,'Asset-use policy '+key)
    for key in ('factor_calculations_permitted_in_source_review','first_public_version_verified',
                'complete_correction_history_verified','certified_daily_security_master',
                'iso_currency_newly_certified','organic_investment_or_CMA_replication'):
        require(policy.get(key) is False,'Asset-use scope '+key)
    require(type(policy.get('new_events_admitted')) is int and policy['new_events_admitted']==0,
        'No new source admissions')
    require(policy.get('annual_days')==[330,400] and policy.get('max_age_days')==550
        and policy.get('required_endpoints')==['current.total_assets','comparison.total_assets'],
        'Asset-use endpoints and time gates')
    require(policy.get('measurement_definition')==
        'g=(A_current-A_comparison)/A_comparison; later registered factor uses -g/(1+abs(g)). No growth values computed in source review.',
        'Asset-use formula')
    require(policy.get('accounting')==
        'As-reported consolidated total-asset expansion; acquisitions, disposals, currency translation and changing business perimeter are included, not adjusted or selectively excluded.',
        'As-reported scope, no retrospective event exclusions')
    require(policy.get('restatement')==
        'MSFT FY2018 uses both asset columns from its current release on the explicitly restated revenue/lease basis; availability is that release, never backfilled.',
        'Restatement release scope')
    require(review.get('schema_version')==1 and review.get('round')==147
        and review.get('stage')=='independent_asset_growth_source_use_review'
        and review.get('passed') is True and review.get('conditional_source_use_only') is True
        and review.get('policy_sha256')==policy_ref['sha256']
        and review.get('dataset_sha256')==DATA_SHA,'Bound source-use review')
    for key in ('financial_ratios','asset_growth_values_computed','factor_calculations','model_fits',
                'portfolio_calculations','return_calculations','newly_admitted_events'):
        require(type(review.get(key)) is int and review[key]==0,'Source-use computation isolation')
    for key in ('first_public_version_verified','complete_correction_history_verified',
                'certified_daily_security_master'):
        require(review.get(key) is False,'No upgraded source certification')
    checks=('frozen146_dataset_and_full_source_closure_verified',
        'all44_records_and_exact31_13_gate_preserved','same_current_report_positive_assets_and_units',
        'annual_periods_availability_identity_rechecked','current_restatement_basis_supported_without_backfill')
    require(type(review.get('checks')) is dict and all(review['checks'].get(k) is True for k in checks),
        'Full purpose-review checks')
    closure=reader.verify(review['input_hashes'])
    require(all(pins.get(k)==v for k,v in closure.items()),'Purpose-review full closure in PLAN')
    for ref in (policy_ref,plan['inputs']['admitted_dataset'],plan['inputs']['source_admission'],plan['source_policy']):
        require(closure.get(str(reader.path(ref['path'])))==ref['sha256'],'Purpose-review required binding')
    require(set(policy.get('inputs',{}))=={'dataset','admission','prior_policy','independent',
        'restatement','accounting','sessions'},'Exact purpose source set')
    for key,ref in policy['inputs'].items():
        bound_ref(reader,ref,pins,'Purpose source')
        require(closure.get(str(reader.path(ref['path'])))==ref['sha256'],
            'Purpose review consumed source '+key)
    for key,ref in (('dataset',plan['inputs']['admitted_dataset']),
                    ('admission',plan['inputs']['source_admission']),
                    ('prior_policy',plan['source_policy']),
                    ('sessions',plan['inputs']['sessions'])):
        actual=policy['inputs'][key]
        require(reader.path(actual['path'])==reader.path(ref['path'])
            and actual['sha256']==ref['sha256'],'Purpose shares original '+key)
    codes=reader.verify(policy['source_code_hashes'])
    require(all(pins.get(k)==v==closure.get(k) for k,v in codes.items()),'Purpose code closure')
    events=dataset['events'];findings=review.get('findings')
    require(type(findings) is list and len(events)==len(findings)==44,'All purpose events')
    admitted=[];holds=[]
    for event,row in zip(events,findings):
        gate=event['cash_quality_admission']
        require(row.get('event_id')==event['event_id']
            and row.get('conditional_asset_use') is gate['admitted']
            and row.get('old_admission')==gate,'Same purpose eligibility, no new mask')
        for key in ('inherited_rejection_codes','source','periods','identity','release_date'):
            require(row.get(key)==event[key],'Purpose event unchanged '+key)
        for side in ('current','comparison'):
            present=event['fields'][side]['total_assets'] is not None
            require(row['asset_endpoints'][side]==dict(present=present,positive_if_present=present),
                'Purpose endpoint presence')
        if gate['admitted']:admitted.append(event['event_id'])
        else:holds.append(dict(event_id=event['event_id'],**gate))
    require(len(admitted)==31 and len(holds)==13 and review.get('events')==44
        and review.get('admitted_events')==31 and review.get('held_events')==13
        and review.get('admitted_event_ids')==admitted and review.get('holds')==holds,
        'Exact existing31/13 purpose population')
    return review

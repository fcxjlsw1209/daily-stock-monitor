"""Independent filing-association preparation audit; no new producer imports.

Only an explicit registered PLAN permits real-data review. Pure functions and
synthetic tests never open market files. Source eligibility remains conditional
on the declared filing-date and issuer-continuity reconstruction assumptions.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import math
from collections import Counter
from io import StringIO
import unittest
import numpy as np
import pandas as pd
import monthly_cash_quality146_prepare_review as legacy
from after156_association_measure_review import decimal_oracle

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_filing_association157'
SOURCE = 'outputs/monthly_filing_association157_source'
HANDS = ('whole_formation_beta_residual_momentum', 'filing_window_density',
         'filing_vs_nonfiling_beta_residual_contrast')
NAMES = ('incumbent_local67', *HANDS)
EXTRA = HANDS + ('common_available',)
CHECKS = legacy.CHECKS + ('filing_source_grid_verified', 'original_source_cutoffs_verified',
          'all_three_filing_features_common_mask', 'preformation_beta_and_counts_verified',
          'post_price_common_coverage_floor_passed')
CACHE_FILES = legacy.CACHE_FILES
old, require = legacy.old, legacy.require
INPUTS = ('proposal','meta','policy','original_orders','prices','protected_manifest',
          'source_manifest','source_grid_review','source_plan','source_calendar')
SOURCE_REVIEW_CHECKS = ('package_matches_source_review','original_5978_grid_preserved',
    'all_output_and_input_hashes_verified','stock_3904_etf_2074_preserved',
    'all_rows_preserve_evidence_and_eligibility')
CLOCK = 'second_XNYS_session_strictly_after_filingDate_v1'
COUNTERS = ('nuisance_beta_estimates_attempted','nuisance_beta_estimates_completed','new_predictive_model_fits')
ERAS = (('early53',0,53),('recent35',53,88),('first88',0,88),('later34',88,122),('all122',0,122))
FORBIDDEN = ('monthly_filing_association157_runner','monthly_filing_association157_lifecycle',
             'after156_association_measure')
SOURCE_CONTRACT = {
    'meta': ('outputs/monthly_local_linear70/META.json',
        '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4'),
    'source_plan': ('work/after156_association_protocol.json',
        '9906d41f2aad9f5981030b4264540cdca0f40f8fb5c723e1060ddc32b00fa88e'),
    'source_calendar': ('outputs/monthly_paired_target143/CALENDAR.json',
        '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622')}


def validate_source_review(review):
    require(review.get('passed') is True and review.get('actual_grid_verified') is True
            and review.get('research_eligibility_verified') is True
            and review.get('exact_acceptance_certified') is False,
            'Independent actual source-grid verification required')
    checks = review.get('checks', {})
    require(all(checks.get(key) is True for key in SOURCE_REVIEW_CHECKS),
            'Independent source calendar, original identities and evidence checks required')


def independent_imports():
    require(not any(name in sys.modules for name in FORBIDDEN),'New producer modules must not be imported')


def validate_prereview_proposal(proposal):
    require(proposal.get('round')==157 and proposal.get('arms')==list(NAMES) and
            proposal.get('primary_challenger')==NAMES[3] and proposal.get('dose')==.01 and
            proposal.get('status')=='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',
            'Final prospective four-arm proposal required')
    for key in ('actual_new_factors','actual_new_orders','actual_new_returns',
                'actual_new_nuisance_estimates','actual_new_predictive_model_fits'):
        require(type(proposal.get(key)) is int and proposal[key]==0,'No preregistration work: '+key)
    measurement=proposal.get('measurement',{})
    expected=dict(clock_policy_id=CLOCK,exact_acceptance_certified=False,beta_prices=[-504,-252],
        formation_prices=[-252,-42],aligned_prices=463,beta='centered_252_return_OLS_slope_no_formation_intercept',
        whole='21*mean(all210_beta_residuals)',density='nE/(nE+nN)',
        contrast='21*(mean(E_beta_residuals)-mean(N_beta_residuals))',
        return_hand='h(raw/0.05)',density_hand='raw_density',h='x/(1+abs(x))',
        minimum_full19_bundles=2,minimum_E=38,minimum_N=80,original_candidate_axis=[122,49],
        unchanged_original_arrays=27,original_eligibility_unchanged=True,
        placement='add_0.01_h_to_original_pre_score_then_one_original_episode_selector',
        common_mask='same_source_partition_price_beta_availability_for_all_three_hands',
        no_parameter_search=True,new_predictive_model_fits=0,etf_features=False,
        monthly_equal_weight_positions=3,intramonth_trades=False)
    require(all(measurement.get(k)==v for k,v in expected.items()),'Independent fixed measurement contract')


def validate_source_manifest(manifest):
    require(type(manifest.get('schema_version')) is int and manifest['schema_version']==1 and
            manifest.get('source_only') is True and manifest.get('strategy_registration') is False and
            manifest.get('clock_policy_id')==CLOCK and
            set(manifest.get('outputs',{}))=={'ROWS.json','COVERAGE_GATE.json'},
            'Exact conditional filing-date source package required')


def source_files(root=None):
    root = ROOT if root is None else Path(root).resolve()
    return sorted(set(legacy.source_files(root) + [root/'work'/name for name in (
        'monthly_filing_association157_prepare_review.py',
        'test_monthly_filing_association157_prepare_review.py',
        'after156_association_measure_review.py')]))


def source_inputs(reader, plan, pins):
    """Honor the independent source oracle, checking its whole byte closure."""
    refs = plan['inputs']
    manifest_path = reader.path(refs['source_manifest']['path'])
    require(manifest_path == reader.path(SOURCE+'/MANIFEST.json'), 'Canonical source manifest')
    manifest = reader.json(manifest_path)
    validate_source_manifest(manifest)
    needed = reader.verify(manifest['input_hashes'])
    needed[str(manifest_path)] = reader.sha(manifest_path)
    for name, ref in manifest['outputs'].items():
        path = reader.path(SOURCE+'/'+name)
        require(reader.sha(path) == ref['sha256'] and path.stat().st_size == ref['bytes'], 'Source output binding')
        needed[str(path)] = ref['sha256']
    require(reader.path(refs['source_grid_review']['path']) == reader.path('work/after156_association_package_review.json'),
            'Exact independent package review')
    review = reader.json(refs['source_grid_review']['path'])
    validate_source_review(review)
    reviewed = reader.verify(review['input_hashes'])
    require(all(reviewed.get(p) == h for p,h in needed.items()), 'Independent source coverage incomplete')
    require(all(pins.get(p) == h for p,h in reviewed.items()), 'Unregistered source closure')
    for key, (canonical, expected) in SOURCE_CONTRACT.items():
        path = reader.path(refs[key]['path'])
        require(path == reader.path(canonical) and reader.sha(path) == expected
                and needed.get(str(path)) == expected, 'Original source contract: '+key)
    rows=reader.json(SOURCE+'/ROWS.json');meta=reader.json(refs['meta']['path'])
    gate=coverage_oracle(meta,rows)
    saved=reader.json(SOURCE+'/COVERAGE_GATE.json')
    require(gate['coverage_floor_passed'] and all(saved.get(k)==v for k,v in gate.items()),
            'Independent prospective source breadth floor')
    calendar=reader.json(refs['source_calendar']['path'])
    implementation='outputs/monthly_earnings_repair/research_calendar.py'
    require(str(reader.path(implementation)) in pins,'Bound original exchange-calendar implementation')
    spec=importlib.util.spec_from_file_location('filing157_independent_calendar',reader.path(implementation))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    schedule=module.market_calendar(calendar['start'],calendar['end']).schedule
    days=[str(day.date()) for day in schedule.index]
    require(days==calendar['sessions'],'Original full calendar including exceptional closures')
    cutoffs={day:int(close.value) for day,close in zip(days,schedule['close'])}
    validate_rows(meta,rows,calendar['sessions'],cutoffs)
    return rows,calendar['sessions'],cutoffs


def validate_rows(meta,rows,sessions=None,cutoffs=None):
    total=len(meta['dates'])
    require(len(meta['symbols'])==total and len(rows)==total*49,'Full original source axis')
    require(meta['dates']==sorted(set(meta['dates'])) and len(set(meta['etfs']))==len(meta['etfs'])==17,'Unique dates and17 ETFs')
    position=None if sessions is None else {day:i for i,day in enumerate(sessions)}
    if position is not None:require(len(position)==len(sessions) and sessions==sorted(sessions),'Full session axis')
    for k,row in enumerate(rows):
        i,j=divmod(k,49);names=meta['symbols'][i]
        require(len(names)==len(set(names))==49 and sum(s not in meta['etfs'] for s in names)==32,'Original32 stocks17 ETFs')
        require(type(row['signal_index']) is int and type(row['candidate_index']) is int and
            (row['signal_index'],row['candidate_index'],row['symbol'],row['decision_date'])==
            (i,j,names[j],meta['dates'][i]),'Every original slot in order')
        require(type(row['research_eligible']) is bool and type(row['reasons']) is list and
            all(type(reason) is str and reason for reason in row['reasons']),'Explicit eligibility and reasons')
        if row['research_eligible']:
            require(names[j] not in meta['etfs'] and not row['reasons'] and
                type(row.get('issuer_id')) is str and bool(row['issuer_id']),'Usable stock issuer')
        else:require(bool(row['reasons']),'Unavailable reason required')
        clock=pd.Timestamp(row['decision_cutoff'])
        require(clock.tzinfo is not None and type(row['decision_cutoff_ns']) is int and
            int(clock.value)==row['decision_cutoff_ns'] and
            str(clock.tz_convert('America/New_York').date())==meta['dates'][i],'Cutoff date and nanoseconds')
        if cutoffs is not None:require(cutoffs.get(meta['dates'][i])==row['decision_cutoff_ns'],'Original exchange close required')
        if position is not None:
            require(type(row['signal_session_index']) is int and position.get(meta['dates'][i])==row['signal_session_index'],
                'Signal full-calendar position')


def coverage_oracle(meta,rows,mask=None):
    validate_rows(meta,rows)
    require(len(meta['dates'])==122,'Full122 coverage denominator')
    source=np.asarray([r['research_eligible'] for r in rows],dtype=bool).reshape(122,49)
    if mask is None:mask=source
    require(isinstance(mask,np.ndarray) and mask.dtype==bool and mask.shape==(122,49) and
        not np.any(mask & ~source),'Common mask cannot expand source eligibility')
    eras={}
    for key,start,end in ERAS:
        month_issuers=[set() for _ in range(start,end)];usable=denominator=0;issuers=set()
        for i in range(start,end):
            for j,symbol in enumerate(meta['symbols'][i]):
                if symbol in meta['etfs']:continue
                denominator+=1
                if mask[i,j]:
                    issuer=rows[49*i+j]['issuer_id'];issuers.add(issuer)
                    month_issuers[i-start].add(issuer);usable+=1
        eras[key]=dict(stock_slots=denominator,usable_slots=usable,fraction=usable/denominator,
            distinct_issuers=len(issuers),months_with_at_least8_issuers=sum(len(v)>=8 for v in month_issuers),
            minimum_issuers_per_month=min(map(len,month_issuers)))
    e,r,f=(eras[k] for k in ('early53','recent35','first88'))
    checks=dict(first88_fraction=f['fraction']>=.5,early53_fraction=e['fraction']>=.4,
        recent35_fraction=r['fraction']>=.4,first88_issuers=f['distinct_issuers']>=12,
        early53_issuers=e['distinct_issuers']>=8,recent35_issuers=r['distinct_issuers']>=8,
        early53_months=e['months_with_at_least8_issuers']>=40,recent35_months=r['months_with_at_least8_issuers']>=26)
    return dict(schema_version=1,original_candidate_rows=5978,original_stock_slots=3904,
        clock_policy_id=CLOCK,exact_acceptance_certified=False,coverage_floor_passed=all(checks.values()),checks=checks,eras=eras)


def partition_oracle(row):
    """Reconstruct latest-owner labels per day, independently of interval painting."""
    t,p=row['signal_session_index'],row['partition']
    require(type(t) is int and t>=504 and type(p) is dict,'Full warmup and reviewed partition')
    left,right=t-251,t-42
    require(p['first_day']==left and p['last_day']==right,'Fixed210 formation endpoints')
    bundles=p['bundles'];require(type(bundles) is list and bool(bundles),'Original event bundles')
    anchors=[b['anchor'] for b in bundles]
    require(all(type(a) is int and t-271<=a<=right for a in anchors) and anchors==sorted(set(anchors)),
        'Unique ordered anchors in complete source interval')
    ids=[]
    for b in bundles:
        require(type(b['event_ids']) is list and bool(b['event_ids']) and len(b['event_ids'])==len(b['forms']),
            'Original bundle identities and forms')
        require(all(f in ('10-Q','10-K','10-Q/A','10-K/A') for f in b['forms']),'Only original periodic forms and amendments')
        require(type(b['amendment']) is bool and b['amendment']==any(f.endswith('/A') for f in b['forms']),
            'Amendment classification')
        ids+=b['event_ids']
    require(len(ids)==len(set(ids)) and len(row['event_ids'])==len(set(row['event_ids'])) and
        set(ids)==set(row['event_ids']),'All actual event IDs exactly once')
    labels=[];owners=[]
    for day in range(left,right+1):
        previous=[b for b in bundles if b['anchor']<=day]
        owner=previous[-1] if previous else None
        if owner is None or day-owner['anchor']>20:labels.append('N');owners.append(None)
        else:
            labels.append('B' if owner['amendment'] or day-owner['anchor']<2 else 'E');owners.append(owner['anchor'])
    full=0
    for b in bundles:
        owned=[left+k for k,label in enumerate(labels) if label=='E' and owners[k]==b['anchor']]
        complete=not b['amendment'] and owned==list(range(b['anchor']+2,b['anchor']+21))
        require(b['owned_E_days']==owned and b['full19'] is complete,'Reconstructed complete19-day bundle')
        full+=complete
    counts=dict(Counter(labels));counts={lab:counts.get(lab,0) for lab in 'EBN'}
    require(p['labels']==''.join(labels) and p['counts']==counts and p['owner_anchor_indices']==owners and
        p['full19_original_bundles']==full and full>=2 and counts['E']>=38 and counts['N']>=80,
        'Full partition and fixed minima')
    return ''.join(labels)


def unavailable(reason,attempts=0):
    return dict(available=False,reason=reason,beta=None,raw=dict.fromkeys(HANDS),hands=dict.fromkeys(HANDS,float('nan')),
        nuisance_beta_estimates_attempted=attempts,nuisance_beta_estimates_completed=0,new_predictive_model_fits=0)


def measure_oracle(stock,market,labels):
    require(len(stock)==len(market)==463 and len(labels)==210 and set(labels)<=set('EBN'), 'Exact old price and label shapes')
    require(labels.count('E')>=38 and labels.count('N')>=80,'Fixed day minima')
    for i,pair in enumerate(zip(stock,market)):
        require(all(isinstance(v,(int,float,np.integer,np.floating)) and not isinstance(v,(bool,np.bool_)) for v in pair),
            'Numeric real prices only')
        if not all(math.isfinite(v) and v>0 for v in pair):
            return unavailable('invalid_required_'+('beta' if i<=252 else 'formation')+'_price')
    # The registered float price/log convention may be exactly degenerate even
    # when Decimal sees tiny roundoff in the supplied geometric price sequence.
    logs=[math.log(b/a) if 0<b/a<math.inf else math.log(b)-math.log(a) for a,b in zip(market[:252],market[1:253])]
    mean=math.fsum(logs)/252
    if math.fsum((x-mean)**2 for x in logs)==0:return unavailable('zero_preformation_market_variance',1)
    beta,raw,hands=decimal_oracle(stock,market,labels)
    require(all(math.isfinite(v) for v in [beta,*raw,*hands]) and all(abs(v)<=1 for v in hands),'Finite bounded independent arithmetic')
    return dict(available=True,reason=None,beta=beta,raw=dict(zip(HANDS,raw)),hands=dict(zip(HANDS,hands)),
        nuisance_beta_estimates_attempted=1,nuisance_beta_estimates_completed=1,new_predictive_model_fits=0)


def features_oracle(meta,rows,sessions,prices,cutoffs,n=None,numeric_cache=None):
    """Read only fixed old price slices; every finite hand shares the same mask."""
    total=len(meta['dates']);n=total if n is None else n
    require(type(n) is int and 0<=n<=total,'Valid signal prefix')
    validate_rows(meta,rows,sessions,cutoffs)
    require(isinstance(prices,pd.DataFrame) and prices.columns.is_unique and 'SPY' in prices.columns,'Price columns and SPY')
    dates=pd.DatetimeIndex(prices.index)
    require(dates.tz is None and dates.equals(dates.normalize()) and dates.is_unique and dates.is_monotonic_increasing,'Uncompressed daily price axis')
    arrays={key:np.full((n,49),np.nan) for key in HANDS};arrays['common_available']=np.zeros((n,49),bool)
    cells=[];totals=dict.fromkeys(COUNTERS,0)
    for row in rows[:49*n]:
        i,j=row['signal_index'],row['candidate_index'];window=None
        result=unavailable(';'.join(row['reasons']))
        if row['research_eligible']:
            labels=partition_oracle(row);t=row['signal_session_index'];window=sessions[t-504:t-41]
            require(len(window)==463 and row['symbol'] in prices.columns,'Required463 old observations')
            index=pd.DatetimeIndex(window);require(index.isin(dates).all(),'Missing calendar dates cannot compress windows')
            stock=prices.loc[index,row['symbol']].to_numpy();market=prices.loc[index,'SPY'].to_numpy()
            # Content-keyed reuse makes123 prefix audits practical without
            # allowing a later changed price or partition to reuse an answer.
            key=(labels,stock.dtype.str,stock.tobytes(),market.dtype.str,market.tobytes())
            if numeric_cache is not None and key in numeric_cache:result=numeric_cache[key]
            else:
                result=measure_oracle(stock,market,labels)
                if numeric_cache is not None:numeric_cache[key]=result
            for hand in HANDS:arrays[hand][i,j]=result['hands'][hand]
            arrays['common_available'][i,j]=result['available']
        for counter in COUNTERS:totals[counter]+=result[counter]
        cells.append(dict(signal_index=i,candidate_index=j,symbol=row['symbol'],issuer_id=row.get('issuer_id'),
            decision_date=row['decision_date'],source_research_eligible=row['research_eligible'],available=result['available'],
            reason=result['reason'],beta=result['beta'],raw=result['raw'],price_window=None if window is None else [window[0],window[-1]],
            **{key:result[key] for key in COUNTERS}))
    require(all(np.array_equal(np.isfinite(arrays[key]),arrays['common_available']) for key in HANDS),'Independent common mask')
    return dict(arrays=arrays,cells=cells,**totals)


def score_oracle(arrays,meta,hands,n=None):
    groups = legacy.score_oracle(arrays,meta,*(hands[key] for key in HANDS),n=n)
    return tuple({new:group[prior] for new,prior in zip(NAMES,legacy.NAMES)} for group in groups)


def order_oracle(meta,scores,choices):
    return {name:[dict(date=meta['dates'][i],signal_date=meta['dates'][i],end=meta['ends'][i],
        symbols=[meta['symbols'][i][j] for j in picked],scores=[float(scores[name][i,j]) for j in picked])
        for i,picked in enumerate(choices[name])] for name in NAMES}


def audit(root,plan_path,plan_sha):
    independent_imports()
    require(type(plan_sha) is str and re.fullmatch('[0-9a-f]{64}',plan_sha), 'Explicit PLAN SHA')
    reader = old.Reader(root); directory=reader.path(OUT)
    require(reader.path(plan_path)==directory/'PLAN.json' and reader.sha(plan_path)==plan_sha,'Exact registered PLAN')
    plan=reader.json(plan_path)
    require(plan.get('round')==157 and plan.get('status')=='REGISTERED' and plan['arms']==list(NAMES)
            and plan['primary_challenger']==NAMES[3] and plan['dose']==.01,'Four fixed prospective arms')
    require(plan['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),'Pinned runtime')
    require(reader.path(plan['output_directory'])==directory and set(plan['inputs'])==set(INPUTS),
            'Exact output and named inputs')
    require(plan['tolerances']['feature_atol']==1e-12 and plan['tolerances']['scores_atol']==1e-12
            and plan['tolerances']['scores_rtol']==0., 'Registered preparation tolerances')
    require(not any((directory/name).exists() for name in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json',
            'WALKFORWARD.json','FROZEN.json','DIAGNOSE_STARTED.json','RESULTS.json')),
            'Preparation audit precedes results')
    pins=reader.verify(plan['hashes'])
    sources={str(path):reader.sha(path) for path in source_files(root)}
    require(reader.verify(plan['independent_prepare_source_hashes'])==sources,'Independent implementation binding')
    for key in ('actual_model_fits_before_registration','actual_feature_calculations_before_registration',
                'actual_returns_before_registration','actual_nuisance_estimates_before_registration'):
        require(type(plan.get(key)) is int and plan[key]==0,'Prospective activity count')
    inputs={name:reader.path(ref['path']) for name,ref in plan['inputs'].items()}
    for name,ref in plan['inputs'].items():
        require(reader.sha(inputs[name])==ref['sha256']==pins.get(str(inputs[name])),'Named input binding')
    proposal=reader.json(inputs['proposal'])
    require(type(proposal.get('actual_new_predictive_model_fits')) is int
            and proposal['actual_new_predictive_model_fits']==0,'No preregistration predictive fits')
    for key, filename in (('policy','INCUMBENT_POLICY.json'),('original_orders','local_linear_ORDERS.json')):
        require(inputs[key]==reader.path(old.LOCAL+'/'+filename),'Original local67 input: '+key)
    require(inputs['prices']==reader.path(old.PRICES),'Original price cache')
    policy=reader.json(inputs['policy'])
    require(all(policy.get(key)==value for key,value in old.original_helper().POLICY.items()),
            'Frozen incumbent episode policy')
    rows,sessions,cutoffs=source_inputs(reader,plan,pins)
    cache=reader.json(directory/'CACHE.json'); require(set(cache)==set(CACHE_FILES),'Full preparation cache')
    reader.verify(cache,directory)
    meta=reader.json(directory/'META.json'); require(meta==reader.json(inputs['meta']),'Original metadata')
    require(len(meta['dates'])==122 and meta['train_months']==88 and meta['early_months']==53,'Fixed segments')
    original=reader.npz(old.LOCAL+'/DATA.npz'); arrays=reader.npz(directory/'DATA.npz')
    require(len(original)==27 and set(arrays)==set(original)|set(EXTRA),'Original27 plus four arrays')
    for key,value in original.items():
        require(value.dtype==arrays[key].dtype and value.shape==arrays[key].shape
                and value.tobytes()==arrays[key].tobytes(),'Original array changed '+key)
    prices=reader.npz(inputs['prices'])
    frame=pd.DataFrame(prices['close'],index=pd.to_datetime(prices['dates']),columns=prices['symbols'])
    numeric_cache={}
    expected=features_oracle(meta,rows,sessions,frame,cutoffs,numeric_cache=numeric_cache)
    for key in EXTRA:
        require(arrays[key].shape==expected['arrays'][key].shape
                and arrays[key].dtype==expected['arrays'][key].dtype,'Feature shape and dtype')
        np.testing.assert_allclose(arrays[key],expected['arrays'][key],rtol=0,atol=1e-12,equal_nan=True)
    feature=reader.json(directory/'FEATURE_AUDIT.json')
    require(feature.get('source_manifest_sha256')==reader.sha(inputs['source_manifest'])
            and feature.get('source_grid_review_sha256')==reader.sha(inputs['source_grid_review']),
            'Feature audit source bindings')
    require(feature['rows']==5978 and feature['usable_rows']==int(expected['arrays']['common_available'].sum()),'Feature audit counts')
    require('new_model_fits' not in feature,'Do not hide nuisance estimates behind unqualified zero fits')
    for key in COUNTERS:
        require(type(feature.get(key)) is int and feature[key]==expected[key],'Audited estimator counts: '+key)
    require(feature['clock_policy_id']==CLOCK and feature['exact_acceptance_certified'] is False and
            feature['strategy_returns_or_nav_computed'] is False,'Association scope and no strategy outcomes')
    old.compare(feature['cells'],expected['cells'],'filing-cells')
    for key in ('original_eligibility_unchanged','filing_source_grid_verified'):
        require(feature.get(key) is True,'Feature check '+key)
    common_gate=coverage_oracle(meta,rows,expected['arrays']['common_available'])
    require(common_gate['coverage_floor_passed'],'Post-price common finite-mask breadth must pass before selection')
    old.compare(feature['common_coverage'],common_gate,'post-price-coverage')
    require(feature['missing_reason_counts']==dict(Counter(c['reason'] for c in expected['cells'] if not c['available'])),
            'All missing reasons counted')
    scores,choices,_=score_oracle(arrays,meta,expected['arrays'])
    saved_scores=reader.npz(directory/'SCORES.npz'); saved_orders=reader.json(directory/'PRETRAIN_ORDERS.json')
    require(set(saved_scores)==set(saved_orders)==set(NAMES),'Four prepared arms')
    wanted=order_oracle(meta,scores,choices)
    for name in NAMES:
        require(saved_scores[name].shape==scores[name].shape and saved_scores[name].dtype==scores[name].dtype,
                'Prepared score shape and dtype')
        np.testing.assert_allclose(saved_scores[name],scores[name],rtol=0,atol=1e-12)
        old.compare(saved_orders[name],wanted[name],name)
    require(saved_orders[NAMES[0]]==reader.json(inputs['original_orders']),'Exact frozen incumbent orders')
    for n in range(123):
        # Only past rows and fixed consumed price slices enter the prefix.
        prefix=features_oracle(meta,rows,sessions,frame,cutoffs,n,numeric_cache)
        for key in EXTRA:np.testing.assert_array_equal(prefix['arrays'][key],expected['arrays'][key][:n])
        ss,cc,_=score_oracle(arrays,meta,prefix['arrays'],n=n)
        for name in NAMES:
            require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(),'Causal prefix')
    for fill in (0.,np.nan):
        hand=np.full((122,49),np.nan)
        for i,symbols in enumerate(meta['symbols']):
            for j,symbol in enumerate(symbols):
                if symbol not in meta['etfs']:hand[i,j]=fill
        ss,cc,_=score_oracle(arrays,meta,dict.fromkeys(HANDS,hand))
        for name in NAMES[1:]:
            require(cc[name]==choices[NAMES[0]] and ss[name].tobytes()==scores[NAMES[0]].tobytes(),'Exact absent/zero fallback')
    require(reader.sha(old.LOADER)==old.LOADER_SHA,'Frozen raw price loader')
    spec=importlib.util.spec_from_file_location('filing157_independent_price_loader',reader.path(old.LOADER))
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    panel,info=loader.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(reader.path(old.LOADER))]=old.LOADER_SHA
    raw=reader.verify(plan['raw_source_hashes'])
    require({str(reader.path(p)):h for p,h in used.items()}==raw and len(raw)==64,'Exact raw64')
    reader.verify(used)
    parity=old.price_parity(panel,prices)
    old.compare(reader.json(directory/'PRICE_PARITY.json'),parity,'price-parity')
    required={str(directory/name):reader.sha(directory/name) for name in (*CACHE_FILES,'PLAN.json','CACHE.json')}
    allowed=dict(pins,**required)
    for p,h in list(reader.seen.items()):
        require(reader.sha(p)==h and allowed.get(p)==h,'Unbound or changed input '+p)
    independent_imports()
    return dict(stage='prepare',passed=True,plan_sha256=plan_sha,cache_sha256=reader.sha(directory/'CACHE.json'),
        orders_sha256=reader.sha(directory/'PRETRAIN_ORDERS.json'),source_hashes=sources,artifact_hashes=dict(reader.seen),
        **dict.fromkeys(CHECKS,True),producer_modules_imported=False,strategy_returns_or_nav_computed=False,
        raw_prices_independently_verified=True,calendar_overlay_independently_verified=True,
        all_signal_prefixes_verified=True,
        source_clock_verification='original_exchange_close_and_hash_bound_conditional_source_review',
        exact_acceptance_certified=False,
        orders_checked=488,candidate_cells_checked=5978,months_checked=122,prefixes_checked=123,
        filing_feature_matrices_verified=3,**{key:expected[key] for key in COUNTERS},
        available_cells=int(expected['arrays']['common_available'].sum()),
        max_score_error=max(float(np.max(np.abs(saved_scores[name]-scores[name]))) for name in NAMES),
        limitations=['Filing-date association and issuer continuity use explicit retrospective reconstruction assumptions; no exact acceptance certification.',
                     'Independent package review binds source closure and history; original cutoffs, partitions, beta, features, coverage and scores/orders are rechecked here.'])


def prereview(root=ROOT):
    """Fresh CLI process, synthetic tests, and final metadata/code bindings only."""
    independent_imports();root=Path(root).resolve()
    import test_monthly_filing_association157_prepare_review as test_module
    log=StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_module))
    require(result.wasSuccessful() and result.testsRun>0 and not result.skipped,'Synthetic prepare tests failed: '+log.getvalue())
    independent_imports();reader=old.Reader(root)
    proposal='work/monthly_filing_association157_proposal.json'
    validate_prereview_proposal(reader.json(proposal))
    implementations=['work/'+name for name in ('after156_association_measure.py',
        'monthly_filing_association157_runner.py','monthly_filing_association157_lifecycle.py',
        'prepare_filing_association157_proposal.py')]
    inputs=[proposal,'work/after156_association_protocol.json',SOURCE+'/MANIFEST.json',
            'work/after156_association_package_review.json']
    report=dict(passed=True,round=157,stage='preregistration_prepare_review',synthetic_only=True,
        proposal_sha256=reader.sha(proposal),actual_factors_calculated=0,actual_returns_calculated=0,
        actual_orders_calculated=0,actual_predictive_model_fits=0,actual_nuisance_estimates=0,
        producer_modules_imported=False,tests=result.testsRun,test_log=log.getvalue(),
        source_hashes={str(p):reader.sha(p) for p in source_files(root)},
        implementation_hashes={str(reader.path(p)):reader.sha(p) for p in implementations},
        input_hashes={str(reader.path(p)):reader.sha(p) for p in inputs},
        expected_feature_matrices=3,expected_monthly_orders=488,expected_signal_prefixes=123,
        reviewer_scope='Independent Decimal beta/features, full-calendar partition and cutoff checks, common-mask breadth and original score/selector tests; actual audit remains pending.',
        actual_audit_pending=True)
    for path,expected in list(reader.seen.items()):require(reader.sha(path)==expected,'Post-prereview binding')
    independent_imports()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan');p.add_argument('--plan-sha');p.add_argument('--prereview',action='store_true')
    p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args()
    if args.prereview:
        if args.plan is not None or args.plan_sha is not None:p.error('--prereview cannot be combined with --plan or --plan-sha')
        result=prereview(args.root);path=args.root/'work/monthly_filing_association157_prepare_prereview.json'
    else:
        if args.plan is None or args.plan_sha is None:p.error('Actual review requires --plan and --plan-sha')
        result=audit(args.root,args.plan,args.plan_sha);path=args.root/OUT/'INDEPENDENT_PREPARE.json'
    with path.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))


if __name__=='__main__':main()

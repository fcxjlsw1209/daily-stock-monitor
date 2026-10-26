"""Four prospectively fixed filing-date association arms.

Import and synthetic tests do not load source data or market prices. Actual
phases require a registered PLAN, a complete source manifest and a positive
independent research-eligibility receipt before constructing the old backend.
This F-only reconstruction never claims a certified acceptance/publication time.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import monthly_forward_guidance132_runner as base
import monthly_stock_comp149_runner as prior
import monthly_revenue_persistence156_runner as previous
import after156_association_measure as factor

require, sha, read = base.require, base.sha, base.read
NAMES = ('incumbent_local67', *factor.NAMES)
HANDS = factor.NAMES
TRAIN, MONTHS, WIDTH, DOSE = 88, 122, 49, .01
OUT = 'outputs/monthly_filing_association157'
SOURCE_OUT = 'outputs/monthly_filing_association157_source'
CLOCK = 'second_XNYS_session_strictly_after_filingDate_v1'
PROTOCOL = 'work/after156_association_protocol.json'
PROTOCOL_SHA = '9906d41f2aad9f5981030b4264540cdca0f40f8fb5c723e1060ddc32b00fa88e'
CACHE_FILES, FROZEN_FILES = base.CACHE_FILES, prior.FROZEN_FILES
TOLERANCES = dict(prior.TOLERANCES)
INPUTS = ('proposal', 'meta', 'policy', 'original_orders', 'prices', 'protected_manifest',
          'source_manifest', 'source_grid_review', 'source_plan', 'source_calendar')
SOURCE_FILES = ('ROWS.json', 'COVERAGE_GATE.json')
CHECKS = (*base.CHECKS, 'filing_source_grid_verified', 'original_source_cutoffs_verified',
          'all_three_filing_features_common_mask', 'preformation_beta_and_counts_verified',
          'post_price_common_coverage_floor_passed')
FIXED_INPUTS = dict(meta=previous.FIXED_INPUTS['meta'],
                   source_calendar=previous.FIXED_INPUTS['source_calendar'],
                   source_plan=(PROTOCOL, PROTOCOL_SHA))
COUNTERS = ('nuisance_beta_estimates_attempted', 'nuisance_beta_estimates_completed', 'new_predictive_model_fits')
corrected = prior.corrected
ERAS = (('early53', 0, 53), ('recent35', 53, 88), ('first88', 0, 88),
        ('later34', 88, 122), ('all122', 0, 122))


def selection_contract():
    contract = prior.selection_contract()
    contract.pop('new_model_fits')
    contract.update(new_predictive_model_fits=0,
        nuisance_estimates='preformation_beta_attempts_and_completions_counted_separately')
    return contract


def measurement_contract():
    return dict(clock_policy_id=CLOCK, exact_acceptance_certified=False,
        interpretation='administrative_filing_date_association_under_declared_reconstruction_assumptions',
        beta_prices=[-504, -252], formation_prices=[-252, -42], aligned_prices=463,
        beta='centered_252_return_OLS_slope_no_formation_intercept',
        whole='21*mean(all210_beta_residuals)', density='nE/(nE+nN)',
        contrast='21*(mean(E_beta_residuals)-mean(N_beta_residuals))',
        return_hand='h(raw/0.05)', density_hand='raw_density', h='x/(1+abs(x))',
        common_mask='same_source_partition_price_beta_availability_for_all_three_hands',
        minimum_full19_bundles=2, minimum_E=38, minimum_N=80,
        placement='add_0.01_h_to_original_pre_score_then_one_original_episode_selector',
        original_candidate_axis=[122,49], unchanged_original_arrays=27,
        original_eligibility_unchanged=True, no_parameter_search=True,
        nuisance_estimates='attempts_and_completions_counted_per_source_eligible_stock_slot',
        new_predictive_model_fits=0, etf_features=False, monthly_equal_weight_positions=3,
        intramonth_trades=False)


def validate_proposal(proposal):
    require(proposal.get('round') == 157 and proposal.get('arms') == list(NAMES)
            and proposal.get('primary_challenger') == NAMES[3] and proposal.get('dose') == DOSE,
            'Wrong prospective filing association experiment')
    require(proposal.get('selection_contract') == selection_contract(), 'Selection contract changed')
    require(proposal.get('measurement') == measurement_contract(), 'Measurement contract changed')
    require(proposal.get('status') == 'FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION', 'Final proposal required')
    for key in ('actual_new_factors', 'actual_new_orders', 'actual_new_returns', 'actual_new_nuisance_estimates',
                'actual_new_predictive_model_fits'):
        require(type(proposal.get(key)) is int and proposal[key] == 0, 'No preregistration calculation')


def validate_rows(meta, rows):
    require(len(meta['dates']) == len(meta['symbols']) == MONTHS and len(rows) == MONTHS*WIDTH, 'Original122x49 grid')
    require(meta['dates'] == sorted(set(meta['dates'])) and len(meta['etfs']) == len(set(meta['etfs'])) == 17, 'Original axes')
    for i, names in enumerate(meta['symbols']):
        require(len(names) == len(set(names)) == WIDTH and len(set(names)-set(meta['etfs'])) == 32, 'Original32stocks17ETFs')
        for j, symbol in enumerate(names):
            row = rows[i*WIDTH+j]
            require(type(row.get('signal_index')) is int and type(row.get('candidate_index')) is int
                    and (row['signal_index'],row['candidate_index'],row['symbol'],row['decision_date']) == (i,j,symbol,meta['dates'][i]), 'Source row identity/order')
            require(type(row.get('research_eligible')) is bool and type(row.get('reasons')) is list, 'Explicit research gate and reasons')
            require(all(type(v) is str and v for v in row['reasons']), 'Reason strings')
            if row['research_eligible']:
                require(symbol not in meta['etfs'] and not row['reasons'] and type(row.get('issuer_id')) is str and row['issuer_id'], 'Research-eligible stock issuer')
            else:
                require(bool(row['reasons']), 'Unavailable row needs reasons')


def coverage_gate(meta, rows, mask=None):
    """Eight frozen breadth floors; original stock denominator, unique issuers."""
    validate_rows(meta, rows)
    source_mask = np.array([r['research_eligible'] for r in rows], dtype=bool).reshape(MONTHS,WIDTH)
    if mask is None:
        mask = source_mask
    else:
        require(isinstance(mask,np.ndarray) and mask.dtype == bool and mask.shape == source_mask.shape
                and not (mask & ~source_mask).any(), 'Price mask cannot expand source eligibility')
    eras = {}
    for name, lo, hi in ERAS:
        group = [r for r in rows if lo <= r['signal_index'] < hi and r['symbol'] not in meta['etfs']]
        selected = [r for r in group if mask[r['signal_index'],r['candidate_index']]]
        per_month = {i:len({r['issuer_id'] for r in selected if r['signal_index']==i}) for i in range(lo,hi)}
        eras[name] = dict(stock_slots=len(group), usable_slots=len(selected), fraction=len(selected)/len(group),
            distinct_issuers=len({r['issuer_id'] for r in selected}),
            months_with_at_least8_issuers=sum(n>=8 for n in per_month.values()),
            minimum_issuers_per_month=min(per_month.values()))
    e,r,f = (eras[k] for k in ('early53','recent35','first88'))
    checks = dict(first88_fraction=f['fraction']>=.5, early53_fraction=e['fraction']>=.4,
        recent35_fraction=r['fraction']>=.4, first88_issuers=f['distinct_issuers']>=12,
        early53_issuers=e['distinct_issuers']>=8, recent35_issuers=r['distinct_issuers']>=8,
        early53_months=e['months_with_at_least8_issuers']>=40,
        recent35_months=r['months_with_at_least8_issuers']>=26)
    return dict(schema_version=1, original_candidate_rows=5978, original_stock_slots=3904,
                clock_policy_id=CLOCK, exact_acceptance_certified=False,
                coverage_floor_passed=all(checks.values()), checks=checks, eras=eras)


def validate_partition(row):
    """Rebuild ownership from the already reviewed bundles, without prices."""
    t, p = row['signal_session_index'], row['partition']
    require(type(t) is int and t>=504 and type(p) is dict, 'Full-calendar warmup and partition required')
    require(type(row.get('event_ids')) is list and bool(row['event_ids'])
            and all(type(v) is str and v for v in row['event_ids'])
            and len(row['event_ids'])==len(set(row['event_ids'])), 'Unique consumed event identifiers')
    left,right = t-251,t-42
    require(p.get('first_day')==left and p.get('last_day')==right, 'Formation boundaries')
    bundles = p.get('bundles')
    require(type(bundles) is list and bool(bundles), 'Reviewed bundles required')
    labels, owners, ids, last = ['N']*210,[None]*210,[],None
    for b in bundles:
        a=b.get('anchor')
        require(type(a) is int and t-271<=a<=right and (last is None or a>last), 'Sorted unique anchors')
        last=a
        require(type(b.get('event_ids')) is list and b['event_ids'] and len(b['event_ids'])==len(b.get('forms',[])), 'Bundle event/form IDs')
        require(all(f in ('10-Q','10-K','10-Q/A','10-K/A') for f in b['forms']), 'Periodic forms only')
        amended=any(f.endswith('/A') for f in b['forms'])
        require(type(b.get('amendment')) is bool and b['amendment']==amended, 'Amendment bundle')
        ids.extend(b['event_ids'])
        for d in range(max(left,a),min(right,a+20)+1):
            labels[d-left]='B' if amended or d-a<2 else 'E'
            owners[d-left]=a
    require(len(ids)==len(set(ids)) and set(ids)==set(row['event_ids']), 'Exact unique consumed event IDs')
    full=0
    for b in bundles:
        owned=[left+k for k,(lab,a) in enumerate(zip(labels,owners)) if lab=='E' and a==b['anchor']]
        complete=not b['amendment'] and owned==list(range(b['anchor']+2,b['anchor']+21))
        require(b.get('owned_E_days')==owned and b.get('full19') is complete, 'Full bundle ownership')
        full+=complete
    counts={lab:labels.count(lab) for lab in 'EBN'}
    require(p.get('labels')==''.join(labels) and p.get('owner_anchor_indices')==owners and p.get('counts')==counts
            and p.get('full19_original_bundles')==full and full>=2 and counts['E']>=38 and counts['N']>=80,
            'Reviewed full partition or minima changed')
    return ''.join(labels)


def counters(audit):
    values={key:audit.get(key) for key in COUNTERS}
    require(all(type(v) is int and v>=0 for v in values.values()), 'Exact integer nuisance counters')
    require(values['new_predictive_model_fits']==0 and values[COUNTERS[1]]<=values[COUNTERS[0]]<=3904, 'Nuisance count bounds')
    return values


def build_features(meta, rows, sessions, px):
    """Called only after the external source gate and registration; also synthetic-testable."""
    validate_rows(meta,rows)
    require(isinstance(px,pd.DataFrame) and px.columns.is_unique and 'SPY' in px.columns, 'Unique price columns and SPY')
    dates=pd.DatetimeIndex(px.index)
    require(dates.tz is None and dates.equals(dates.normalize()) and dates.is_unique and dates.is_monotonic_increasing, 'Uncompressed original daily prices')
    position={day:k for k,day in enumerate(sessions)}
    require(len(position)==len(sessions) and sessions==sorted(sessions), 'Unique full-calendar session axis')
    arrays={key:np.full((MONTHS,WIDTH),np.nan) for key in HANDS}
    arrays['common_available']=np.zeros((MONTHS,WIDTH),bool)
    cells=[]; totals=dict.fromkeys(COUNTERS,0)
    for row in rows:
        i,j=row['signal_index'],row['candidate_index']
        result=factor.missing(';'.join(row['reasons']))
        window=None
        if row['research_eligible']:
            t=row['signal_session_index']
            require(position.get(row['decision_date'])==t, 'Signal calendar index changed')
            labels=validate_partition(row)
            window=sessions[t-504:t-41]
            require(len(window)==463 and row['symbol'] in px.columns, 'Required source-reviewed price axis missing')
            expected=pd.DatetimeIndex(window)
            require(expected.isin(dates).all(), 'Missing calendar dates cannot be compressed')
            # .loc selects only the fixed old window. No post-t-42 value is read.
            result=factor.measure(px.loc[expected,row['symbol']].tolist(),px.loc[expected,'SPY'].tolist(),labels)
            for key in HANDS:
                arrays[key][i,j]=result['hands'][key]
            arrays['common_available'][i,j]=result['available']
        for key in COUNTERS:
            totals[key]+=result[key]
        cells.append(dict(signal_index=i,candidate_index=j,symbol=row['symbol'],issuer_id=row.get('issuer_id'),
            decision_date=row['decision_date'],source_research_eligible=row['research_eligible'],
            available=result['available'],reason=result['reason'],beta=result['beta'],raw=result['raw'],
            price_window=None if window is None else [window[0],window[-1]],
            **{key:result[key] for key in COUNTERS}))
    gate=coverage_gate(meta,rows,arrays['common_available'])
    audit=dict(rows=5978,usable_rows=int(arrays['common_available'].sum()),cells=cells,
        clock_policy_id=CLOCK,exact_acceptance_certified=False,common_coverage=gate,
        missing_reason_counts=dict(Counter(c['reason'] for c in cells if not c['available'])),
        original_eligibility_unchanged=True,strategy_returns_or_nav_computed=False,**totals)
    counters(audit)
    return dict(arrays=arrays,audit=audit)


def rename(value, mapping):
    """Translate only exact arm identifiers at the frozen156 accounting boundary."""
    if isinstance(value,dict):
        return {mapping.get(k,k):rename(v,mapping) for k,v in value.items()}
    if isinstance(value,list):
        return [rename(v,mapping) for v in value]
    return mapping.get(value,value) if isinstance(value,str) else value


TO_PREVIOUS=dict(zip(NAMES,previous.NAMES))
FROM_PREVIOUS=dict(zip(previous.NAMES,NAMES))


def select(rows, opened):
    require(all([r['name'] for r in group]==list(NAMES) for group in (rows,opened)), 'Four exact ordered arms required')
    return rename(previous.select(rename(rows,TO_PREVIOUS),rename(opened,TO_PREVIOUS)),FROM_PREVIOUS)


def choose(name, arrays, meta, n, backend, policy):
    require(name in NAMES,'Undeclared arm')
    return backend.baseline(arrays,meta,n,policy) if name==NAMES[0] else corrected(
        backend.pre(arrays,n),arrays[name],arrays,meta,n,backend,policy)


def panels(arrays, meta, backend, policy):
    # Only metadata names change; frozen156 verifies all123 prefixes, zero/NaN
    # exact fallback and byte preservation using the same original selector.
    aliases={key:value for key,value in arrays.items() if key not in HANDS}
    aliases.update({old:arrays[new] for old,new in zip(previous.HANDS,HANDS)})
    orders,scores=previous.panels(aliases,meta,backend,policy)
    return rename(orders,FROM_PREVIOUS),{FROM_PREVIOUS[k]:v for k,v in scores.items()}


class Backend(previous.Backend):
    def training(self,orders,meta,px,op):
        return tuple(rename(v,FROM_PREVIOUS) for v in super().training(rename(orders,TO_PREVIOUS),meta,px,op))

    def diagnostics(self,orders,meta,px,op,writer):
        def translated(name,value):
            for old,new in FROM_PREVIOUS.items():
                if name.startswith(old+'_'):
                    return writer(new+name[len(old):],value)
            raise ValueError('Unexpected frozen diagnostic filename')
        return rename(super().diagnostics(rename(orders,TO_PREVIOUS),meta,px,op,translated),FROM_PREVIOUS)


class Runner(base.Runner):
    path=previous.Runner.path

    def __init__(self,root,plan_path,plan_sha,backend_factory=Backend):
        self.root,self.plan_path=Path(root).resolve(),Path(plan_path).resolve()
        self.plan_sha=plan_sha
        require(sha(self.plan_path)==plan_sha,'Frozen PLAN changed')
        self.plan=plan=read(self.plan_path)
        require(plan.get('round')==157 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES)
                and plan.get('primary_challenger')==NAMES[3],'Round157 registration required')
        require(plan.get('dose')==DOSE and plan.get('tolerances')==TOLERANCES,'Declared numerical policy')
        require(plan.get('runtime')==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),'Pinned runtime')
        for key in ('actual_feature_calculations_before_registration','actual_returns_before_registration',
                    'actual_model_fits_before_registration','actual_nuisance_estimates_before_registration',
                    'new_market_downloads'):
            require(type(plan.get(key)) is int and plan[key]==0,'Unexpected preregistration activity')
        require(plan.get('no_observer_writes') is True,'Observer protection')
        self.pins=self.verify(plan['hashes'])
        require(set(plan['inputs'])==set(INPUTS),'Exact named input schema')
        self.inputs={key:self.bound_path(ref['path']) for key,ref in plan['inputs'].items()}
        require(all(sha(self.inputs[key])==ref['sha256'] for key,ref in plan['inputs'].items()),'Named input binding')
        for key,(path,expected) in FIXED_INPUTS.items():
            require(self.inputs[key]==self.root/path and sha(self.inputs[key])==expected,'Original axes/protocol/calendar')
        validate_proposal(read(self.inputs['proposal']))
        for path,expected in source_files(self.root).items():
            require(self.pins.get(path)==expected,'Current implementation not registered')
        for key in ('raw_source_hashes','independent_prepare_source_hashes','independent_outcome_source_hashes'):
            require(all(self.pins.get(path)==digest for path,digest in self.verify(plan[key]).items()),'Incomplete registered '+key)
        require(len(plan['raw_source_hashes'])==64,'Original64 raw price sources')
        self.out=self.root/OUT
        require(self.plan_path==self.out/'PLAN.json' and self.path(plan['output_directory'])==self.out,'Dedicated new output')
        self.backend_factory=backend_factory
        self.require_source_grid()

    def require_source_grid(self):
        manifest_path=self.inputs['source_manifest']
        require(manifest_path==self.root/SOURCE_OUT/'MANIFEST.json','Canonical source manifest')
        manifest=read(manifest_path)
        require(manifest.get('schema_version')==1 and manifest.get('source_only') is True and manifest.get('strategy_registration') is False
                and manifest.get('clock_policy_id')==CLOCK,'Filing association source manifest')
        closure=self.verify(manifest['input_hashes'])
        require(all(self.pins.get(p)==v for p,v in closure.items()),'Unregistered source evidence')
        closure[str(manifest_path)]=sha(manifest_path)
        require(set(manifest['outputs'])==set(SOURCE_FILES),'Exact source output set')
        files={}
        for name,ref in manifest['outputs'].items():
            path=self.bound_path(self.root/SOURCE_OUT/name)
            require(sha(path)==ref['sha256'] and path.stat().st_size==ref['bytes'],'Source output changed')
            files[name]=path;closure[str(path)]=ref['sha256']
        review=read(self.inputs['source_grid_review'])
        require(review.get('passed') is True and review.get('research_eligibility_verified') is True
                and review.get('actual_grid_verified') is True and review.get('exact_acceptance_certified') is False
                and review.get('clock_policy_id')==CLOCK,'Independent reconstructed-source review required')
        reviewed=self.verify(review['input_hashes'])
        require(all(reviewed.get(p)==v for p,v in closure.items()) and all(self.pins.get(p)==v for p,v in reviewed.items()),'Independent full source closure')
        meta,rows=read(self.inputs['meta']),read(files['ROWS.json'])
        gate=coverage_gate(meta,rows)
        require(read(files['COVERAGE_GATE.json'])==gate and gate['coverage_floor_passed'],'Source coverage floor must pass')
        calendar=read(self.inputs['source_calendar'])
        require(self.pins.get(str(self.inputs['source_plan']))==PROTOCOL_SHA and read(self.inputs['source_plan'])['clock']['id']==CLOCK,'Explicit F-only protocol')
        for path,digest in calendar['provenance']['source_hashes'].items():
            require(self.pins.get(str(self.path(path)))==digest,'Unbound calendar implementation/evidence')
        module=base.module('filing157_calendar',self.bound_path(self.root/'outputs/monthly_earnings_repair/research_calendar.py'))
        schedule=module.market_calendar(calendar['start'],calendar['end']).schedule
        sessions=[str(day.date()) for day in schedule.index]
        require(sessions==calendar['sessions'],'Full original calendar axis')
        cutoffs={day:int(close.value) for day,close in zip(sessions,schedule['close'])}
        previous.verify_cutoffs(meta,rows,cutoffs)
        positions={day:k for k,day in enumerate(sessions)}
        for row in rows:
            require(type(row.get('signal_session_index')) is int and row['signal_session_index']==positions[row['decision_date']],'Original full-calendar signal index')
            if row['research_eligible']:
                validate_partition(row)
                require(type(row.get('context_original_event_ids')) is list and type(row.get('source_refs')) is list and row['source_refs'],'Explicit consumed source provenance')
                require(all(type(v) is str and v for v in row['context_original_event_ids'])
                        and len(row['context_original_event_ids'])==len(set(row['context_original_event_ids'])), 'Unique support identifiers')
                for ref in row['source_refs']:
                    path=self.bound_path(ref['path'])
                    require(closure.get(str(path))==ref['sha256'],'Usable source row outside reviewed closure')
        self.refresh()
        return meta,rows,sessions

    def prepare(self):
        self.refresh()
        require(not any((self.out/name).exists() for name in (*CACHE_FILES,'CACHE.json','PREPARE_FAILURE.json')),'Preparation already started')
        source=self.require_source_grid()
        backend=self.backend_factory(self.root,self)
        meta,original,px,op=backend.load()
        base.validate_meta(meta)
        require(len(original)==27 and meta==source[0],'Exact original27 cache')
        parity=backend.price_parity(meta,px,op,self.plan['raw_source_hashes'])
        result=build_features(*source,px)
        result['audit'].update(filing_source_grid_verified=True,source_manifest_sha256=sha(self.inputs['source_manifest']),
            source_grid_review_sha256=sha(self.inputs['source_grid_review']))
        if not result['audit']['common_coverage']['coverage_floor_passed']:
            self.write('FEATURE_AUDIT.json',result['audit'])
            self.write('PREPARE_FAILURE.json',dict(reason='POST_PRICE_COMMON_COVERAGE_FLOOR_FAILED',plan_sha256=self.plan_sha,
                feature_audit_sha256=sha(self.out/'FEATURE_AUDIT.json'),strategy_returns_or_nav_computed=False,**counters(result['audit'])))
            raise ValueError('Post-price common coverage floor failed; registered failure preserved, no selection')
        require(not set(original)&set(result['arrays']),'New features cannot replace original arrays')
        arrays=dict(original,**result['arrays'])
        orders,scores=panels(arrays,meta,backend,read(self.inputs['policy']))
        require(orders[NAMES[0]]==read(self.inputs['original_orders']),'Incumbent changed')
        self.refresh()
        for name,values in (('DATA.npz',arrays),('SCORES.npz',scores)):
            with (self.out/name).open('xb') as handle:
                np.savez_compressed(handle,**values)
        for name,value in (('META.json',meta),('PRETRAIN_ORDERS.json',orders),('PRICE_PARITY.json',parity),('FEATURE_AUDIT.json',result['audit'])):
            self.write(name,value)
        self.write('CACHE.json',{name:sha(self.out/name) for name in CACHE_FILES})

    def loaded(self):
        self.cache();self.require_source_grid()
        backend=self.backend_factory(self.root,self)
        meta,original,px,op=backend.load()
        require(meta==read(self.out/'META.json') and len(original)==27,'Original metadata/cache changed')
        with np.load(self.out/'DATA.npz',allow_pickle=False) as data:
            require(set(data.files)==set(original)|set(HANDS)|{'common_available'},'Changed array names')
            require(all(data[k].tobytes()==v.tobytes() and data[k].dtype==v.dtype and data[k].shape==v.shape for k,v in original.items()),'Original27 changed')
        return backend,meta,px,op

    def require_prepare_review(self):
        self.cache()
        report=read(self.out/'INDEPENDENT_PREPARE.json');audit=read(self.out/'FEATURE_AUDIT.json')
        require(report.get('passed') is True and report.get('stage')=='prepare' and all(report.get(key) is True for key in CHECKS),'Independent preparation checks incomplete')
        require(report.get('producer_modules_imported') is False and report.get('strategy_returns_or_nav_computed') is False,'Independent preparation isolation')
        for key,expected in (('orders_checked',488),('candidate_cells_checked',5978),('months_checked',122),('prefixes_checked',123),('filing_feature_matrices_verified',3)):
            require(type(report.get(key)) is int and report[key]==expected,'Independent preparation scope: '+key)
        require(counters(report)==counters(audit) and audit['common_coverage']['coverage_floor_passed'],'Independently matched beta counters and common coverage')
        require(report['plan_sha256']==self.plan_sha and report['cache_sha256']==sha(self.out/'CACHE.json') and report['orders_sha256']==sha(self.out/'PRETRAIN_ORDERS.json'),'Review binding changed')
        reviewed=self.verify(report['artifact_hashes'])
        require(all(reviewed.get(str(self.out/name))==sha(self.out/name) for name in ('PLAN.json','CACHE.json',*CACHE_FILES)),'Unreviewed prepared artifact')
        require(self.verify(report['source_hashes'])==self.verify(self.plan['independent_prepare_source_hashes']),'Independent reviewer changed')

    def select_first88(self):
        self.refresh();self.require_prepare_review()
        require(not any((self.out/name).exists() for name in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','WALKFORWARD.json','FROZEN.json')),'Selection already started')
        backend,meta,px,op=self.loaded()
        rows,opened,walk=backend.training(read(self.out/'PRETRAIN_ORDERS.json'),meta,px,op)
        for name,value in (('TRAIN_TRIALS.json',rows),('TRAIN_OPEN_TRIALS.json',opened),('WALKFORWARD.json',walk)):
            self.write(name,value)
        self.write('FROZEN.json',dict(selection=select(rows,opened),pristine_oos=False,all_diagnostic_orders_frozen=True,
            hashes={name:sha(self.out/name) for name in FROZEN_FILES}))

    def diagnose_later34(self):
        self.refresh();self.require_prepare_review()
        frozen=read(self.out/'FROZEN.json')
        require(set(frozen['hashes'])==set(FROZEN_FILES),'Frozen artifact list changed')
        self.verify({str(self.out/name):expected for name,expected in frozen['hashes'].items()})
        require(frozen['selection']==select(read(self.out/'TRAIN_TRIALS.json'),read(self.out/'TRAIN_OPEN_TRIALS.json')),'First88 selection changed')
        require(not (self.out/'DIAGNOSE_STARTED.json').exists(),'Diagnostics already started')
        self.write('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,frozen_sha256=sha(self.out/'FROZEN.json'),started_utc=datetime.now(timezone.utc).isoformat()))
        backend,meta,px,op=self.loaded()
        results=backend.diagnostics(read(self.out/'PRETRAIN_ORDERS.json'),meta,px,op,self.write)
        for filename,mode in (('TRAIN_TRIALS.json','same_close'),('TRAIN_OPEN_TRIALS.json','next_open_cost')):
            for row in read(self.out/filename):
                require(abs(results[row['name']][mode]['slices']['training2016_2023']['cagr']-row['training']['cagr'])<TOLERANCES['training_cagr'],'Training replay parity')
        self.write('RESULTS.json',results)
        self.write('VALIDATION.json',dict(selection=frozen['selection'],no_observer_writes=True,no_intramonth_trades=True,
            independent_outcome_review_pending=True,pristine_oos=False,**counters(read(self.out/'FEATURE_AUDIT.json'))))


def source_files(root):
    root=Path(root)
    paths=[Path(__file__),Path(factor.__file__),root/'work/test_monthly_filing_association157_runner.py',
           root/'work/test_after156_association_measure.py']
    return dict(previous.source_files(root),**{str(path.resolve()):sha(path) for path in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','select_first88','diagnose_later34'))
    parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    getattr(Runner(args.root,args.plan,args.plan_sha),args.phase)()

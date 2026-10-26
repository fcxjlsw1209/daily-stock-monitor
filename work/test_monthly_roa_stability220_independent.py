"""Synthetic-only independent source clock, rational feature and slot tests."""
import copy
from datetime import date, timedelta
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import monthly_roa_stability220_independent as oracle


def fixture():
    start, end = date(2011,1,1), date(2023,1,6)
    calendar = []
    while start <= end:
        if start.weekday() < 5:
            calendar.append(start.isoformat())
        start += timedelta(days=1)
    dataset = dict(method=oracle.METHOD, conditional_research_only=True,
        identity_registry=[dict(issuer_id='issuer-A',security_id='security-A',symbol_aliases=['A'])],
        catalog=[],reports=[],edges=[],chains=[])
    for year, ni in zip((2017,2018,2019),('10','20','-10')):
        publication = f'{year+1}-01-15'
        available = next(s for s in calendar if s > publication)
        event_id, report_id = f'A-FY{year}', f'report-A-{year}'
        period = dict(period_start=f'{year}-01-01',period_end=f'{year}-12-31',
            days_inclusive=(date(year,12,31)-date(year,1,1)).days+1)
        source = dict(path=f'original-A-{year}.html',sha256=hashlib.sha256(str(year).encode()).hexdigest())
        fields = {}
        for name, value in zip(oracle.FIELDS,(ni,'100','100')):
            point = f'{year-1}-12-31' if name == oracle.FIELDS[1] else period['period_end']
            cell = dict(value_decimal=value,scale='1000000',currency_literal='$',currency_iso='USD',
                source=copy.deepcopy(source),available_from=available,locator={'row':1})
            if name == oracle.FIELDS[0]:
                cell['period'] = copy.deepcopy(period)
            fields[name] = dict(original_field_path='fields.'+name,as_of=point,source_cell=cell,
                ni_attribution='reported_consolidated_total_NI_before_NCI_deduction' if name == oracle.FIELDS[0] else None)
        dataset['catalog'].append(dict(event_id=event_id,canonical_symbol='A',fiscal_year=year,
            period_start=period['period_start'],period_end=period['period_end'],publication_date=publication,
            assumed_available_session=available,clock_status=oracle.VERIFIED,clock={'next_session_rule_verified':True},
            kind='original_annual_event',issuer_id='issuer-A',security_id='security-A',
            original_field_state='admitted',report_version_id=report_id))
        dataset['reports'].append(dict(event_id=event_id,report_version_id=report_id,
            issuer_id='issuer-A',security_id='security-A',fiscal_year=year,period=period,
            source=source,fields=fields,known_at=available,available_from=available,
            inherited_single_report_fields_admitted=True,corrected_original_values_substituted=False,
            supersedes_version_id=None))
    report_ids = [r['report_version_id'] for r in dataset['reports']]
    for i in range(2):
        dataset['edges'].append(dict(edge_version_id=f'edge-A-{i}',issuer_id='issuer-A',security_id='security-A',
            earlier_report_version_id=report_ids[i],later_report_version_id=report_ids[i+1],
            known_at=dataset['reports'][i+1]['known_at'],conditional_allowed=True,supersedes_version_id=None,
            evidence_refs=[dict(assumed_available_from=dataset['reports'][j]['known_at'],
                report_version_id=dataset['reports'][j]['report_version_id'],event_id=dataset['reports'][j]['event_id'],
                source=copy.deepcopy(dataset['reports'][j]['source'])) for j in (i,i+1)]))
    references = [dict(report_version_id=r['report_version_id'],field=f,source_sha256=r['source']['sha256'],
        event_id=r['event_id'],original_field_path=r['fields'][f]['original_field_path'])
        for r in dataset['reports'] for f in oracle.FIELDS]
    dataset['chains'].append(dict(chain_version_id='chain-A',report_version_ids=report_ids,
        edge_version_ids=[r['edge_version_id'] for r in dataset['edges']],issuer_id='issuer-A',security_id='security-A',
        fiscal_years=[2017,2018,2019],known_at=dataset['reports'][-1]['known_at'],conditional_allowed=True,
        nine_required_fields=references))
    meta = dict(dates=['2020-01-31','2020-02-28','2020-03-31'],
        symbols=[['A','FUND','UNKNOWN']]*3,etfs=['FUND'])
    return dataset,meta,calendar


def physical_prefix(args,n):
    d,m,c = copy.deepcopy(args)
    m['dates'],m['symbols'] = m['dates'][:n],m['symbols'][:n]
    cutoff = m['dates'][-1] if n else None
    for key in ('catalog','reports','edges','chains'):
        d[key] = [r for r in d[key] if oracle.known_by(r,cutoff,key=='catalog')]
    if cutoff:
        # One later official session establishes month-end; no future price data.
        k = c.index(cutoff)
        c = c[:k+2]
    return d,m,c


def first_cell(args):
    _,audit = oracle.independent_features(*args)
    return audit['event_history'][0]['cells'][0]


class RationalTests(unittest.TestCase):
    def test_manual_ddof1_and_negative_ni(self):
        rows = [(Fraction(n),Fraction(100),Fraction(100)) for n in (10,20,-10)]
        x = oracle.arithmetic(rows)
        self.assertEqual(x['roa'], (Fraction(1,10),Fraction(1,5),Fraction(-1,10)))
        self.assertEqual(x['mean'], Fraction(1,15))
        self.assertEqual(x['sample_variance'], Fraction(7,300))
        self.assertEqual(x['sample_std'], math.sqrt(float(Fraction(7,300))))
        self.assertLess(x['candidate'],x['control'])

    def test_constant_positive_negative_zero_exactcopy(self):
        for n in (-2,0,3):
            x=oracle.arithmetic([(Fraction(n),Fraction(1),Fraction(1))]*3)
            self.assertEqual(x['candidate'],x['control'])
            self.assertTrue(x['zero_dispersion'])
            self.assertEqual(x['sample_std'],0.)

    def test_same_report_denominator_and_exact_scaling(self):
        rows=[(Fraction(10),Fraction(50),Fraction(150)),(Fraction(20),Fraction(100),Fraction(300)),
            (Fraction(1),Fraction(1),Fraction(19))]
        self.assertEqual(oracle.arithmetic(rows)['roa'],(Fraction(1,10),)*3)
        self.assertEqual(oracle.arithmetic(rows),oracle.arithmetic([tuple(x*Fraction(10**100+i) for x in r) for i,r in enumerate(rows)]))

    def test_h_underflow_both_signs_fails(self):
        for sign in (-1,1):
            with self.assertRaises(FloatingPointError):
                oracle.arithmetic([(Fraction(sign,10**400),Fraction(1),Fraction(1))]*3)

    def test_variance_underflow_and_overflow_fail(self):
        for x in (Fraction(1,10**400),Fraction(10**400)):
            with self.assertRaises(FloatingPointError):
                oracle.arithmetic([(n,Fraction(1),Fraction(1)) for n in (-x,Fraction(0),x)])

    def test_subnormal_variance_and_h_saturation(self):
        x=Fraction(1,10**160)
        r=oracle.arithmetic([(n,Fraction(1),Fraction(1)) for n in (-x,Fraction(0),x)])
        self.assertGreater(r['sample_std'],0.)
        for sign in (-1,1):
            r=oracle.arithmetic([(Fraction(sign*10**400),Fraction(1),Fraction(1))]*3)
            self.assertEqual(r['control'],float(sign))

    def test_malformed_and_nonpositive_assets_fail(self):
        good=(Fraction(1),Fraction(2),Fraction(2))
        for rows in ([],[good]*2,[good]*4,[(1,Fraction(2),Fraction(2))]*3,
            [(True,Fraction(2),Fraction(2))]*3,[(Fraction(1),Fraction(0),Fraction(2))]*3,
            [(Fraction(1),Fraction(2),Fraction(-1))]*3):
            with self.assertRaises(ValueError):oracle.arithmetic(rows)
        for bad in ('NaN','Infinity','oops'):
            with self.assertRaises(ValueError):oracle.exact_amount(bad,'1')


class SourceTests(unittest.TestCase):
    def test_available_originals_exact_audit_and_etfs(self):
        args=fixture();before=copy.deepcopy(args)
        values,audit=oracle.independent_features(*args)
        receipt=oracle.verify_features(*args,values,audit)
        self.assertTrue(all(receipt[k] for k in oracle.FEATURE_FLAGS))
        self.assertEqual([receipt[k] for k in oracle.COUNT_FIELDS],[9,3,6,9,3,3,6])
        self.assertEqual(audit['event_history'][0]['cells'][1]['reason'],'ETF_no_financial_feature')
        self.assertEqual(audit['event_history'][0]['cells'][2]['reason'],'unsupported_issuer')
        self.assertEqual(audit['event_history'][0]['cells'][0]['measurement']['mean_exact'],'1/15')
        self.assertEqual(args,before)

    def test_held_latest_blocks_no_older_fallback(self):
        d,m,c=fixture()
        latest=copy.deepcopy(d['catalog'][-1]);latest.update(event_id='newheld',fiscal_year=2020,
            period_end='2020-01-15',original_field_state='held',report_version_id=None)
        d['catalog'].append(latest)
        self.assertEqual(first_cell((d,m,c))['reason'],'latest_report_held_or_unprocessed')

    def test_latest_fy_not_latest_publication(self):
        d,m,c=fixture()
        earlier=d['catalog'][0]
        earlier['publication_date']='2020-01-20';earlier['assumed_available_session']='2020-01-21'
        # We only assert anchor here; report clock becomes inconsistent later.
        self.assertEqual(first_cell((d,m,c))['latest_event_id'],'A-FY2019')

    def test_ambiguous_latest_and_preceding_year(self):
        for at,reason in ((2,'ambiguous_latest_fiscal_anchor'),(0,'missing_or_ambiguous_preceding_year')):
            d,m,c=fixture();x=copy.deepcopy(d['catalog'][at]);x['event_id']+='duplicate';d['catalog'].append(x)
            self.assertEqual(first_cell((d,m,c))['reason'],reason)
        d,m,c=fixture();d['catalog'].pop(1)
        self.assertEqual(first_cell((d,m,c))['reason'],'missing_or_ambiguous_preceding_year')

    def test_unknown_timing_anchor_kind_identity(self):
        for mutation,reason in (({'publication_date':None},'unknown_catalog_timing'),
            ({'fiscal_year':None},'unknown_fiscal_anchor'),({'kind':'mystery'},'unknown_event_kind'),
            ({'issuer_id':'another'},'latest_identity_unverified')):
            d,m,c=fixture();d['catalog'][-1].update(mutation)
            self.assertEqual(first_cell((d,m,c))['reason'],reason)

    def test_unknown_clock_publication_same_day_not_yet_blocker(self):
        d,m,c=fixture();x=copy.deepcopy(d['catalog'][-1]);x.update(event_id='future_unknown',
            fiscal_year=2020,period_end='2020-01-15',clock_status='unknown',publication_date='2020-01-31')
        d['catalog'].append(x)
        _,r=oracle.independent_features(d,m,c)
        self.assertTrue(r['event_history'][0]['cells'][0]['available'])
        self.assertEqual(r['event_history'][1]['cells'][0]['reason'],'latest_clock_unverified')

    def test_age550_boundary_and_old_reports_not_expired_now(self):
        d,m,c=fixture();self.assertTrue(first_cell((d,m,c))['available'])
        end=date(2019,12,31)
        for age,available in ((550,True),(551,False)):
            signal=(end+timedelta(days=age)).isoformat()
            # source-cell contract can verify exact days separately from month-end query contract.
            result,_=oracle.source_cell(d,'A',signal,c)
            self.assertEqual(result['available'],available)
            if not available:self.assertEqual(result['reason'],'latest_report_stale')

    def test_held_preceding_and_age_at_own_release(self):
        d,m,c=fixture();d['catalog'][0]['original_field_state']='held'
        self.assertEqual(first_cell((d,m,c))['reason'],'invalid_required_source_field:held_report')
        d,m,c=fixture()
        row=d['catalog'][0];row['publication_date']='2019-08-01';row['assumed_available_session']='2019-08-02'
        report=d['reports'][0];report['known_at']=report['available_from']='2019-08-02'
        d['edges'][0]['known_at']='2019-08-02';d['edges'][0]['evidence_refs'][0]['assumed_available_from']='2019-08-02'
        self.assertEqual(first_cell((d,m,c))['reason'],'invalid_required_source_field:report_age')

    def test_corrections_never_replace_old_anchor(self):
        for target,expected in (('report-A-2019','known_correction_requires_new_chain_review'),
            (None,'unlinked_known_correction'),('unrelated-2010','available')):
            d,m,c=fixture();x=copy.deepcopy(d['catalog'][0]);x.update(event_id='correction',kind='correction',
                fiscal_year=2010,corrects_report_id=target,publication_date='2020-01-20',assumed_available_session='2020-01-21')
            d['catalog'].append(x)
            self.assertEqual(first_cell((d,m,c))['reason'],expected)

    def test_late_correction_and_supersession_do_not_change_prefix(self):
        for key,idfield in (('reports','report_version_id'),('edges','edge_version_id')):
            d,m,c=fixture();row=copy.deepcopy(d[key][0]);row[idfield]='later-version'
            row['known_at']='2020-02-03';row['supersedes_version_id']=d[key][0][idfield];d[key].append(row)
            _,r=oracle.independent_features(d,m,c)
            self.assertTrue(r['event_history'][0]['cells'][0]['available'])
            self.assertEqual(r['event_history'][1]['cells'][0]['reason'],
                'report_version_superseded' if key=='reports' else 'edge_version_superseded')

    def test_edge_future_evidence_and_binding_fail(self):
        for key,value,reason in (('known_at','2020-02-03','edge_version_not_known'),
            ('earlier_report_version_id','wrong','edge_binding_mismatch'),
            ('conditional_allowed',False,'edge_not_conditionally_admitted')):
            d,m,c=fixture();d['edges'][0][key]=value
            self.assertEqual(first_cell((d,m,c))['reason'],reason)
        d,m,c=fixture();d['edges'][0]['evidence_refs'].append(dict(assumed_available_from='2020-02-01',
            source=dict(path='extra-note',sha256='note-sha')))
        self.assertEqual(first_cell((d,m,c))['reason'],'edge_evidence_clock')

    def test_edge_evidence_dates_require_canonical_iso_before_maximum(self):
        for bad in (None, '2020-1-1', 'garbage'):
            d,m,c=fixture()
            d['edges'][0]['evidence_refs'].append(dict(assumed_available_from=bad,
                source=dict(path='additional-footnote',sha256='footnote-sha')))
            cell=first_cell((d,m,c))
            self.assertFalse(cell['available'])
            self.assertEqual(cell['reason'],'edge_evidence_clock')
            values,audit=oracle.independent_features(d,m,c)
            self.assertFalse(values['available'].any())
            self.assertEqual(audit['annual_ratio_evaluations'],0)

    def test_report_and_field_fidelity_no_future_comparison_substitution(self):
        cases=[('ni','ni_attribution','parent_only','NI_attribution'),
            ('ni','as_of','2018-12-31','field_period_endpoint'),
            ('cell','value_decimal','NaN','Finite source amount required'),
            ('cell','scale','0','positive_scale'),
            ('cell','available_from','2020-01-17','field_clock'),
            ('cell','currency_literal','EUR','currency_literal')]
        for category,key,value,reason in cases:
            d,m,c=fixture();f=d['reports'][-1]['fields'][oracle.FIELDS[0]]
            target=f if category=='ni' else f['source_cell'];target[key]=value
            self.assertEqual(first_cell((d,m,c))['reason'],'invalid_required_source_field:'+reason)
        d,m,c=fixture();d['reports'][-1]['corrected_original_values_substituted']=True
        self.assertEqual(first_cell((d,m,c))['reason'],'invalid_required_source_field:replaced_original_values')

    def test_nine_source_bindings_and_contiguous_period(self):
        d,m,c=fixture();d['chains'][0]['nine_required_fields'][0]['source_sha256']='wrong'
        self.assertEqual(first_cell((d,m,c))['reason'],'invalid_required_source_field:nine_field_binding_mismatch')
        d,m,c=fixture();d['chains'][0]['nine_required_fields'].pop()
        self.assertEqual(first_cell((d,m,c))['reason'],'invalid_required_source_field:nine_field_bindings')
        d,m,c=fixture();r=d['reports'][1];r['period']['period_start']='2018-01-02';r['period']['days_inclusive']-=1
        d['catalog'][1]['period_start']='2018-01-02'
        r['fields'][oracle.FIELDS[0]]['source_cell']['period']['period_start']='2018-01-02'
        r['fields'][oracle.FIELDS[1]]['as_of']='2018-01-01'
        self.assertEqual(first_cell((d,m,c))['reason'],'invalid_required_source_field:annual_boundaries_not_contiguous')

    def test_calendar_and_metadata_validation(self):
        for mode in ('duplicate','nonmonth','outside','duplicate-symbol','bad-symbol'):
            d,m,c=fixture()
            if mode=='duplicate':m['dates'][1]=m['dates'][0]
            if mode=='nonmonth':m['dates'][0]='2020-01-30'
            if mode=='outside':m['dates'][0]='2010-01-29'
            if mode=='duplicate-symbol':m['symbols'][0]=['A','A','UNKNOWN']
            if mode=='bad-symbol':m['symbols'][0]=[' A','FUND','UNKNOWN']
            with self.assertRaises(ValueError):oracle.independent_features(d,m,c)

    def test_allphysical_prefixes_and_future_poison(self):
        args=fixture();v,r=oracle.independent_features(*args)
        for n in range(4):
            p=physical_prefix(args,n);x,a=oracle.independent_features(*p)
            for key in x:
                expected=np.empty((0,0),bool if key=='available' else float) if n==0 else v[key][:n]
                np.testing.assert_array_equal(x[key],expected)
            self.assertEqual(a['event_history'],r['event_history'][:n])
        d,m,c=fixture();bad=dict(canonical_symbol='A',clock_status=oracle.VERIFIED,
            assumed_available_session='2021-01-15',publication_date='2021-01-14',event_id='future')
        d['catalog'].append(bad)
        x,a=oracle.independent_features(d,m,c)
        self.assertEqual(a,r)
        for key in x:np.testing.assert_array_equal(x[key],v[key])

    def test_fiscal_end_monotonicity_cannot_be_overridden_by_publication(self):
        d,m,c=fixture();d['catalog'][0]['period_end']='2020-01-01'
        self.assertEqual(first_cell((d,m,c))['reason'],'inconsistent_fiscal_anchors')

    def test_original_edge_evidence_omission_duplicate_or_wrong_hash(self):
        for mode in ('absent','duplicate','hash','event','date','extra-unlocated'):
            d,m,c=fixture();refs=d['edges'][0]['evidence_refs']
            if mode=='absent':refs.pop()
            elif mode=='duplicate':refs.append(copy.deepcopy(refs[0]))
            elif mode=='hash':refs[0]['source']['sha256']='wrong'
            elif mode=='event':refs[0]['event_id']='wrong'
            elif mode=='date':refs[0]['assumed_available_from']='2019-01-01'
            else:refs.append(dict(assumed_available_from='2019-01-01'))
            self.assertEqual(first_cell((d,m,c))['reason'],'edge_original_evidence_binding')

    def test_null_currency_iso_is_conditional_not_false_usd_certification(self):
        d,m,c=fixture()
        for report in d['reports']:
            for field in report['fields'].values():field['source_cell']['currency_iso']=None
        self.assertTrue(first_cell((d,m,c))['available'])
        d['reports'][0]['fields'][oracle.FIELDS[0]]['source_cell']['currency_iso']='USD'
        self.assertEqual(first_cell((d,m,c))['reason'],'invalid_required_source_field:same_report_currency')

    def test_source_field_hash_and_positive_assets_fail(self):
        d,m,c=fixture();d['reports'][1]['fields'][oracle.FIELDS[1]]['source_cell']['source']['sha256']='newer'
        self.assertEqual(first_cell((d,m,c))['reason'],'invalid_required_source_field:field_original_source')
        for bad in ('0','-1'):
            d,m,c=fixture();d['reports'][1]['fields'][oracle.FIELDS[1]]['source_cell']['value_decimal']=bad
            self.assertEqual(first_cell((d,m,c))['reason'],'invalid_required_source_field:positive_asset_endpoint')

    def test_bad_clock_receipt_and_wrong_next_session(self):
        d,m,c=fixture();d['catalog'][1]['clock']['next_session_rule_verified']=False
        self.assertEqual(first_cell((d,m,c))['reason'],'invalid_required_source_field:missing_clock_receipt')
        d,m,c=fixture();d['catalog'][1]['assumed_available_session']='2019-01-17'
        self.assertEqual(first_cell((d,m,c))['reason'],'invalid_required_source_field:wrong_next_session')

    def test_numeric_overflow_does_not_become_unavailable_neutral(self):
        d,m,c=fixture()
        for i,value in enumerate(('1e400','0','-1e400')):
            d['reports'][i]['fields'][oracle.FIELDS[0]]['source_cell']['value_decimal']=value
        with self.assertRaises(FloatingPointError):oracle.independent_features(d,m,c)

    def test_52_53_week_annual_periods_and_short_period_rejection(self):
        for durations in ((364,371,364),(371,364,371)):
            d,m,c=fixture();ending=date(2019,12,31)
            periods=[]
            for duration in reversed(durations):
                beginning=ending-timedelta(days=duration-1)
                periods.append((beginning,ending,duration));ending=beginning-timedelta(days=1)
            for i,(beginning,ending,duration) in enumerate(reversed(periods)):
                p=dict(period_start=beginning.isoformat(),period_end=ending.isoformat(),days_inclusive=duration)
                d['reports'][i]['period']=p
                d['catalog'][i].update(period_start=p['period_start'],period_end=p['period_end'])
                for name,f in d['reports'][i]['fields'].items():
                    f['as_of']=(beginning-timedelta(days=1)).isoformat() if name==oracle.FIELDS[1] else ending.isoformat()
                    if name==oracle.FIELDS[0]:f['source_cell']['period']=copy.deepcopy(p)
            self.assertTrue(first_cell((d,m,c))['available'])
        d,m,c=fixture();d['reports'][0]['period']['days_inclusive']=90
        self.assertEqual(first_cell((d,m,c))['reason'],'invalid_required_source_field:report_period')

    def test_tampered_audit_value_mask_and_sign_rejected(self):
        args=fixture();v,r=oracle.independent_features(*args)
        z=copy.deepcopy(v);z['candidate'][0,0]=np.nextafter(z['candidate'][0,0],math.inf)
        with self.assertRaises(ValueError):oracle.verify_features(*args,z,r)
        a=copy.deepcopy(r);a['annual_ratio_evaluations']+=1
        with self.assertRaises(ValueError):oracle.verify_features(*args,v,a)
        z=copy.deepcopy(v);z['available'][0,0]=False
        with self.assertRaises(ValueError):oracle.verify_features(*args,z,r)


class DeltaTests(unittest.TestCase):
    def test_fixed_dose_eligibility_missing_zero_and_negative(self):
        v=dict(available=np.array([[True,True,True,False]],bool),
            candidate=np.array([[0.,-.5,.25,np.nan]]),control=np.array([[0.,.5,.5,np.nan]]))
        a=dict(eligible=np.array([[True,True,False,True]],bool))
        targets={arm:v[key].copy() for arm,key in oracle.RAW.items()}
        deltas={arm:np.array([[0., .01*v[key][0,1],0.,0.]]) for arm,key in oracle.RAW.items()}
        receipt=oracle.verify_deltas(v,a,targets,deltas)
        self.assertTrue(all(receipt[k] for k in oracle.DELTA_FLAGS))
        self.assertEqual(receipt['actual_max_absolute_delta'][oracle.ARMS[1]],.005)
        bad=copy.deepcopy(deltas);bad[oracle.ARMS[1]][0,2]=.0025
        with self.assertRaises(ValueError):oracle.verify_deltas(v,a,targets,bad)

    def test_target_not_ranked_or_shrunk_and_zero_sign_preserved(self):
        v=dict(available=np.ones((1,1),bool),candidate=np.array([[-0.]]),control=np.array([[0.]]))
        a=dict(eligible=np.ones((1,1),bool));targets={arm:v[key].copy() for arm,key in oracle.RAW.items()}
        deltas={arm:np.zeros((1,1)) for arm in oracle.RAW}
        oracle.verify_deltas(v,a,targets,deltas)
        targets[oracle.ARMS[1]][0,0]=0.
        with self.assertRaises(ValueError):oracle.verify_deltas(v,a,targets,deltas)

    def test_no_producer_import_and_pinned_accounting(self):
        text=Path(oracle.__file__).read_text()
        for name in ('monthly_roa_stability220_kernel','monthly_roa_stability220_scoring','roa_stability_preflight_math'):
            self.assertNotIn('import '+name,text)
        helper=oracle.frozen_helper();self.assertTrue(callable(helper.audit_outcomes))
        expected={'checked_paths':9,'checked_slices':45,'paths':{x:{} for x in oracle.ARMS}}
        with patch.object(oracle,'frozen_helper') as load:
            load.return_value.audit_outcomes.return_value=copy.deepcopy(expected)
            got=oracle.audit_outcomes(None,None,None,None,None,None)
            self.assertEqual(got['checked_paths'],9)
            self.assertEqual(load.return_value.ARMS,oracle.ARMS)


class BilateralTests(unittest.TestCase):
    def test_unique_both_fixtures_allprefix_fullaudit_and_scoring(self):
        # Deliberate black-box production calls only in this synthetic test.
        import monthly_roa_stability220_kernel as producer
        import monthly_roa_stability220_scoring as scoring
        fixtures=[fixture()]
        try:
            from test_monthly_roa_stability220_kernel import fixture as producer_fixture
        except ImportError:
            producer_fixture=None
        if producer_fixture is not None:
            pd, pc = producer_fixture()
            pm = dict(dates=['2020-02-28','2020-03-31'],symbols=[['AAA','FUND'],['AAA','FUND']],etfs=['FUND'])
            fixtures.append((pd,pm,pc))
        for args in fixtures:
            for n in range(len(args[1]['dates'])+1):
                d,m,c=physical_prefix(args,n)
                v,a=producer.build_features(d,m,c)
                oracle.verify_features(d,m,c,v,a)
                if n and v['available'].shape[1]:
                    original=dict(baseline=np.zeros(v['available'].shape),eligible=np.ones(v['available'].shape,bool))
                    targets,deltas=scoring.scale_and_delta(v,original,n)
                    oracle.verify_deltas(v,original,targets,deltas)


if __name__=='__main__':
    unittest.main()

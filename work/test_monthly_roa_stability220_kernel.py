"""Synthetic sources only: no real financial dataset or monthly META read."""
from bisect import bisect_right
from copy import deepcopy
from datetime import date, timedelta
import unittest
import numpy as np
import monthly_roa_stability220_kernel as k


def calendar():
    start=date(2015,1,1); end=date(2027,12,31)
    return [(start+timedelta(days=i)).isoformat() for i in range((end-start).days+1)
            if (start+timedelta(days=i)).weekday()<5]


def fixture():
    sessions=calendar()
    data=dict(method=k.METHOD,conditional_research_only=True,
        identity_registry=[dict(issuer_id='Issuer A',security_id='AAA',symbol_aliases=['AAA'])],
        catalog=[],reports=[],edges=[],chains=[])
    for offset,year in enumerate([2017,2018,2019]):
        event='event'+str(year); rid='report'+str(year)
        start=str(year)+'-01-01'; end=str(year)+'-12-31'
        publication=str(year+1)+'-02-03'; available=sessions[bisect_right(sessions,publication)]
        row=dict(event_id=event,canonical_symbol='AAA',issuer_id='Issuer A',security_id='AAA',
            fiscal_year=year,period_start=start,period_end=end,publication_date=publication,
            assumed_available_session=available,clock_status=k.VERIFIED,
            report_version_id=rid,original_field_state='admitted',kind='original_annual_event',
            clock=dict(next_session_rule_verified=True))
        source=dict(path='synthetic/'+event,sha256='sha-'+event)
        fields={}
        for name,value in zip(k.FIELDS,[str(10+offset), '100', '100']):
            endpoint=str(year-1)+'-12-31' if name=='opening_total_assets' else end
            cell=dict(value_decimal=value,scale='1000000',source=source.copy(),
                available_from=available,currency_literal='$',currency_iso='USD',
                locator={'line':offset+1})
            if name=='annual_net_income':cell['period']=dict(period_start=start,period_end=end)
            fields[name]=dict(original_field_path='fields.'+name,as_of=endpoint,source_cell=cell,
                ni_attribution='reported_consolidated_total_NI_before_NCI_deduction' if name=='annual_net_income' else None)
        report=dict(report_version_id=rid,event_id=event,issuer_id='Issuer A',security_id='AAA',
            fiscal_year=year,period=dict(period_start=start,period_end=end,
                days_inclusive=(date(year,12,31)-date(year,1,1)).days+1),
            available_from=available,known_at=available,inherited_single_report_fields_admitted=True,
            corrected_original_values_substituted=False,source=source,fields=fields,supersedes_version_id=None)
        data['catalog'].append(row);data['reports'].append(report)
    for i in [0,1]:
        earlier,later=data['reports'][i:i+2]
        data['edges'].append(dict(edge_version_id='edge'+str(i),issuer_id='Issuer A',security_id='AAA',
            earlier_report_version_id=earlier['report_version_id'],later_report_version_id=later['report_version_id'],
            conditional_allowed=True,known_at=later['known_at'],supersedes_version_id=None,
            evidence_refs=[dict(assumed_available_from=x['known_at'],event_id=x['event_id'],report_version_id=x['report_version_id'],source=deepcopy(x['source'])) for x in [earlier,later]]))
    data['chains']=[dict(chain_version_id='chain',issuer_id='Issuer A',security_id='AAA',
        fiscal_years=[2017,2018,2019],report_version_ids=[x['report_version_id'] for x in data['reports']],
        edge_version_ids=['edge0','edge1'],conditional_allowed=True,known_at=data['reports'][-1]['known_at'],
        nine_required_fields=[dict(report_version_id=r['report_version_id'],field=name,
            source_sha256=r['source']['sha256'],event_id=r['event_id'],original_field_path=r['fields'][name]['original_field_path'])
            for r in data['reports'] for name in k.FIELDS])]
    return data,sessions


def status(data,signal='2020-02-28'):
    answer=k.select_chain(data,'AAA',signal,calendar())
    return answer[0] if isinstance(answer,tuple) else answer


class SelectionTests(unittest.TestCase):
    def test_valid_originals_and_old_reports_do_not_expire_at_query(self):
        data,sessions=fixture(); out,amounts=k.select_chain(data,'AAA','2020-02-28',sessions)
        self.assertTrue(out['available']);self.assertEqual(out['report_version_ids'],['report2017','report2018','report2019'])
        self.assertEqual([row[0]/row[1] for row in amounts],[k.Fraction(1,10),k.Fraction(11,100),k.Fraction(3,25)])
        self.assertGreater((date(2020,2,28)-date(2017,12,31)).days,550)

    def test_latest_550_inclusive_551_expired(self):
        data,_=fixture(); end=date(2019,12,31)
        self.assertTrue(status(data,(end+timedelta(days=550)).isoformat())['available'])
        self.assertEqual(status(data,(end+timedelta(days=551)).isoformat())['reason'],'latest_report_stale')

    def test_held_unprocessed_unknown_identity_latest_blocks(self):
        for state in ['held','unprocessed','unknown_identity']:
            data,_=fixture();row=deepcopy(data['catalog'][-1]);row.update(event_id='new2020',fiscal_year=2020,
                period_start='2020-01-01',period_end='2020-12-31',publication_date='2021-02-01',
                assumed_available_session='2021-02-02',report_version_id=None,original_field_state=state)
            if state=='unknown_identity':row['issuer_id']=None
            data['catalog'].append(row)
            self.assertFalse(status(data,'2021-02-26')['available'])
            self.assertEqual(status(data,'2021-02-26')['latest_event_id'],'new2020')
            self.assertTrue(status(data,'2020-02-28')['available'])

    def test_unknown_latest_clock_cannot_fall_back(self):
        data,_=fixture();data['catalog'][-1]['clock_status']='outside_pinned_calendar'
        self.assertEqual(status(data)['reason'],'latest_clock_unverified')

    def test_unknown_timing_and_anchor_block(self):
        data,_=fixture();data['catalog'][-1]['publication_date']=None;data['catalog'][-1]['clock_status']='unknown'
        self.assertEqual(status(data)['reason'],'unknown_catalog_timing')
        data,_=fixture();data['catalog'][0]['fiscal_year']=None
        self.assertEqual(status(data)['reason'],'unknown_fiscal_anchor')

    def test_unknown_clock_same_publication_date_not_admitted(self):
        data,_=fixture();data['catalog'][-1]['clock_status']='unknown'
        result=status(data,'2020-02-03')
        self.assertFalse(result['available'])
        self.assertEqual(status(data,'2020-02-04')['reason'],'latest_clock_unverified')

    def test_duplicate_latest_fiscal_year_ambiguous_even_later_release(self):
        data,_=fixture();new=deepcopy(data['catalog'][-1]);new.update(event_id='competing',publication_date='2020-02-10',assumed_available_session='2020-02-11')
        data['catalog'].append(new)
        self.assertEqual(status(data)['reason'],'ambiguous_latest_fiscal_anchor')

    def test_future_comparative_evidence_does_not_backdate_chain(self):
        data,_=fixture();data['edges'][-1]['evidence_refs'].append(dict(assumed_available_from='2020-03-10',source=dict(path='footnote',sha256='footnote-sha')))
        self.assertEqual(status(data)['reason'],'edge_evidence_clock')
        data['edges'][-1]['known_at']='2020-03-10';data['chains'][0]['known_at']='2020-03-10'
        self.assertFalse(status(data)['available']);self.assertTrue(status(data,'2020-03-31')['available'])

    def test_malformed_extra_evidence_date_blocks(self):
        for value in [None,'2020-1-1','garbage']:
            data,_=fixture();data['edges'][0]['evidence_refs'].append(dict(assumed_available_from=value,source=dict(path='note',sha256='note-sha')))
            self.assertEqual(status(data)['reason'],'edge_evidence_clock')

    def test_later_edge_denial_does_not_rewrite_earlier_prefix(self):
        data,_=fixture();reference=status(data)
        revised=deepcopy(data['edges'][-1]);revised.update(edge_version_id='revision',known_at='2020-03-10',
            conditional_allowed=False,supersedes_version_id='edge1');data['edges'].append(revised)
        self.assertEqual(status(data),reference)
        self.assertEqual(status(data,'2020-03-31')['reason'],'edge_version_superseded')
        self.assertEqual(status(k.truncate_sources(data,'2020-02-28')),reference)

    def test_old_fy_late_correction_never_becomes_latest_anchor(self):
        data,_=fixture();correction=deepcopy(data['catalog'][0]);correction.update(event_id='correct2017',kind='correction',
            publication_date='2020-03-02',assumed_available_session='2020-03-03',corrects_report_id='report2017')
        data['catalog'].append(correction)
        self.assertTrue(status(data)['available'])
        result=status(data,'2020-03-31');self.assertEqual(result['latest_event_id'],'event2019')
        self.assertEqual(result['reason'],'known_correction_requires_new_chain_review')
        correction['corrects_report_id']='unrelated-old-report';correction['fiscal_year']=2010
        self.assertTrue(status(data,'2020-03-31')['available'])

    def test_unlinked_correction_blocks(self):
        data,_=fixture();correction=deepcopy(data['catalog'][0]);correction.update(event_id='correction',kind='correction')
        data['catalog'].append(correction)
        self.assertEqual(status(data)['reason'],'unlinked_known_correction')

    def test_missing_year_and_ticker_reuse_block(self):
        data,_=fixture();data['catalog']=data['catalog'][1:]
        self.assertEqual(status(data)['reason'],'missing_or_ambiguous_preceding_year')
        data,_=fixture();data['catalog'][0]['issuer_id']='different issuer'
        self.assertEqual(status(data)['reason'],'mixed_or_unknown_issuer')
        data,_=fixture();data['catalog'][-1]['security_id']='different class'
        self.assertEqual(status(data)['reason'],'latest_identity_unverified')

    def test_future_report_version_not_used(self):
        data,_=fixture();revision=deepcopy(data['reports'][0]);revision.update(report_version_id='rev',known_at='2020-03-03',supersedes_version_id='report2017')
        data['reports'].append(revision)
        self.assertTrue(status(data)['available'])
        self.assertEqual(status(data,'2020-03-31')['reason'],'report_version_superseded')

    def test_invalid_endpoint_scope_source_value_is_unavailable(self):
        for modify in [lambda r:r['fields']['opening_total_assets'].update(as_of='2017-01-01'),
                       lambda r:r['fields']['annual_net_income'].update(ni_attribution='net_income_parent_only'),
                       lambda r:r['fields']['closing_total_assets']['source_cell']['source'].update(sha256='wrong'),
                       lambda r:r['fields']['annual_net_income']['source_cell'].update(value_decimal='NaN'),
                       lambda r:r['fields']['opening_total_assets']['source_cell'].update(value_decimal='0'),
                       lambda r:r['fields']['annual_net_income']['source_cell'].update(currency_iso='EUR')]:
            data,_=fixture();modify(data['reports'][0]);answer=status(data)
            self.assertFalse(answer['available']);self.assertTrue(answer['reason'].startswith('invalid_required_source_field:'))

    def test_wrong_next_session_and_old_availability_age(self):
        data,_=fixture();data['catalog'][0]['assumed_available_session']='2018-02-06'
        self.assertFalse(status(data)['available'])
        data,_=fixture();data['catalog'][0]['publication_date']='2019-10-01'
        data['catalog'][0]['assumed_available_session']='2019-10-02'
        data['reports'][0]['known_at']=data['reports'][0]['available_from']='2019-10-02'
        self.assertFalse(status(data)['available'])

    def test_broken_contiguity_and_duplicate_original_rejected(self):
        data,_=fixture();data['reports'][1]['period']['period_start']='2018-01-02'
        self.assertFalse(status(data)['available'])
        data,_=fixture();data['catalog'][1]['report_version_id']='report2017'
        self.assertEqual(status(data)['reason'],'three_distinct_originals_required')

    def test_missing_or_wrong_nine_field_binding(self):
        data,_=fixture();data['chains'][0]['nine_required_fields'].pop()
        self.assertFalse(status(data)['available'])
        data,_=fixture();data['chains'][0]['nine_required_fields'][0]['source_sha256']='wrong'
        self.assertFalse(status(data)['available'])

    def test_edge_original_refs_required_and_bound(self):
        for mode in ['empty','wrongsha','wrongid','wrongclock']:
            data,_=fixture();refs=data['edges'][0]['evidence_refs']
            if mode=='empty':refs[:]=[]
            if mode=='wrongsha':refs[0]['source']['sha256']='wrong'
            if mode=='wrongid':refs[0]['report_version_id']='unrelated'
            if mode=='wrongclock':refs[0]['assumed_available_from']='2017-01-01'
            self.assertEqual(status(data)['reason'],'edge_original_evidence_binding')

    def test_fiscal_year_and_period_end_inconsistency_blocks(self):
        data,_=fixture();data['catalog'][0]['period_end']='2021-12-31'
        self.assertEqual(status(data)['reason'],'inconsistent_fiscal_anchors')

    def test_52_and_53_week_years_not_annualized(self):
        data,sessions=fixture()
        periods=[('2016-09-25','2017-09-30'),('2017-10-01','2018-09-29'),('2018-09-30','2019-09-28')]
        for row,report,(start,end) in zip(data['catalog'],data['reports'],periods):
            row['period_start'],row['period_end']=start,end
            report['period']=dict(period_start=start,period_end=end,days_inclusive=(date.fromisoformat(end)-date.fromisoformat(start)).days+1)
            for name,field in report['fields'].items():
                field['as_of']=(date.fromisoformat(start)-timedelta(days=1)).isoformat() if name=='opening_total_assets' else end
                if name=='annual_net_income':field['source_cell']['period']=dict(period_start=start,period_end=end)
        out,amounts=k.select_chain(data,'AAA','2020-02-28',sessions)
        self.assertTrue(out['available']);self.assertEqual(amounts[0][0]/amounts[0][1],k.Fraction(1,10))

    def test_invalid_meta_calendar_labels_fail(self):
        data,sessions=fixture();meta=dict(dates=['2020-02-28'],symbols=[['AAA']],etfs=[])
        for changed in [dict(meta,etfs='ETF'),dict(meta,etfs=['ETF','ETF']),dict(meta,symbols=[[' AAA']])]:
            with self.assertRaises(ValueError):k.build_features(data,changed,sessions)
        with self.assertRaises(ValueError):k.build_features(data,meta,sorted(sessions+['2020-02-29']))

    def test_no_cross_year_currency_mixing(self):
        data,_=fixture()
        for field in data['reports'][1]['fields'].values():field['source_cell']['currency_iso']='EUR'
        self.assertFalse(status(data)['available'])

    def test_no_data_mutation_and_physical_future_prefix(self):
        data,sessions=fixture();before=deepcopy(data)
        meta=dict(dates=['2020-02-28','2020-03-31'],symbols=[['AAA','ETF','OTHER']]*2,etfs=['ETF'])
        values,audit=k.build_features(data,meta,sessions)
        self.assertEqual(data,before)
        self.assertEqual(audit['available_cells'],2);self.assertEqual(audit['annual_ratio_evaluations'],6)
        self.assertTrue(values['available'][:,0].all());self.assertFalse(values['available'][:,1:].any())
        for n in [0,1,2]:
            prefix=k.truncate_sources(data,meta['dates'][n-1] if n else None)
            short=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
            out,check=k.build_features(prefix,short,sessions)
            self.assertEqual(check['event_history'],audit['event_history'][:n])
            for key in values:
                if n:np.testing.assert_array_equal(out[key],values[key][:n])
                else:self.assertEqual(out[key].shape,(0,0))

    def test_negative_NI_valid_and_numerical_underflow_hard_failure(self):
        data,sessions=fixture()
        for report in data['reports']:
            report['fields']['annual_net_income']['source_cell']['value_decimal']='-10'
        meta=dict(dates=['2020-02-28'],symbols=[['AAA']],etfs=[])
        values,_=k.build_features(data,meta,sessions)
        self.assertLess(values['candidate'][0,0],0)
        for report in data['reports']:
            report['fields']['annual_net_income']['source_cell']['value_decimal']='1e-500'
        with self.assertRaises(FloatingPointError):k.build_features(data,meta,sessions)

    def test_non_monthend_query_rejected(self):
        data,sessions=fixture()
        with self.assertRaises(ValueError):
            k.build_features(data,dict(dates=['2020-02-27'],symbols=[['AAA']],etfs=[]),sessions)


if __name__=='__main__':unittest.main(verbosity=2)

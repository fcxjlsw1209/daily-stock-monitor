"""Synthetic source tests: no financial efficacy test, source request or real extraction."""
from copy import deepcopy
from decimal import Decimal
import json
import unittest

import profitability141_local_admission as m

CIK = '0000123456'
ACC = CIK + '-23-000001'
SESSIONS = ['2023-02-16', '2023-02-17', '2023-02-21', '2023-02-22']


def facts():
    row = dict(accn=ACC, filed='2023-02-16', form='10-K', start='2022-01-01', end='2022-12-31', val=4, fp='FY', fy=2022)
    asset = dict(row, val=10);asset.pop('start')
    return dict(cik=int(CIK), entityName='Synthetic Company', facts={'us-gaap': {'GrossProfit': {'units': {'USD': [row]}}, 'Assets': {'units': {'USD': [asset]}}}})


def document(obj):
    raw = json.dumps(obj).encode()
    return raw, {'path': 'synthetic.json', 'sha256': m.source.sha(raw)}


def extract(obj=None, catalog=None):
    return m.extract_company(CIK, [document(obj or facts())], SESSIONS, catalog, '2023-02-01')[0]


class SourcePairs(unittest.TestCase):
    def test_exact_fields_source_locators_and_no_admission(self):
        row = extract()[0]
        self.assertEqual((row['gross_profit_decimal'], row['assets_decimal']), ('4', '10'))
        self.assertTrue(row['structurally_complete'])
        self.assertFalse(row['admitted'])
        self.assertFalse(row['original_filing_context_verified'])
        self.assertEqual(row['evidence']['GrossProfit'][0]['json_pointer'], '/facts/us-gaap/GrossProfit/units/USD/0')
        self.assertNotIn('ratio', row)
        self.assertEqual(row['first_usable_session'], '2023-02-17')

    def test_zero_and_negative_gp_preserved(self):
        for value in [0, -1]:
            obj = facts();obj['facts']['us-gaap']['GrossProfit']['units']['USD'][0]['val'] = value
            row = extract(obj)[0]
            self.assertTrue(row['structurally_complete']);self.assertEqual(row['gross_profit_decimal'], str(value))

    def test_assets_nonpositive_preserved_but_blocked(self):
        for value in [0, -1]:
            obj = facts();obj['facts']['us-gaap']['Assets']['units']['USD'][0]['val'] = value
            row = extract(obj)[0];self.assertIn('nonpositive_assets', row['structural_blockers']);self.assertEqual(row['assets_decimal'], str(value))

    def test_boolean_amounts_blocked(self):
        for tag in m.TAGS:
            obj = facts();obj['facts']['us-gaap'][tag]['units']['USD'][0]['val'] = True
            self.assertIn('invalid_' + tag + '_amount', extract(obj)[0]['structural_blockers'])

    def test_decimal_precision_is_not_rounded(self):
        raw = json.dumps(facts()).replace('"val": 4', '"val": 4.1234567890123456789').encode()
        row = m.extract_company(CIK, [(raw, {'path': 'x', 'sha256': m.source.sha(raw)})], SESSIONS)[0][0]
        self.assertEqual(row['gross_profit_decimal'], '4.1234567890123456789')

    def test_duplicate_agreement_and_conflict(self):
        for tag in m.TAGS:
            obj = facts();rows = obj['facts']['us-gaap'][tag]['units']['USD'];rows.append(deepcopy(rows[0]))
            self.assertTrue(extract(obj)[0]['structurally_complete'])
            rows[1]['val'] += 1
            self.assertIn('conflicting_' + tag + '_values', extract(obj)[0]['structural_blockers'])

    def test_conflict_across_provenance_backed_sources(self):
        obj = facts();obj['facts']['us-gaap']['GrossProfit']['units']['USD'][0]['val'] = 8
        rows, _ = m.extract_company(CIK, [document(facts()), document(obj)], SESSIONS)
        self.assertIn('conflicting_GrossProfit_values', rows[0]['structural_blockers'])

    def test_period_start_conflict(self):
        obj = facts();rows=obj['facts']['us-gaap']['GrossProfit']['units']['USD'];rows.append(dict(rows[0], start='2022-01-02'))
        self.assertIn('ambiguous_annual_duration', extract(obj)[0]['structural_blockers'])

    def test_join_dimensions_never_crossed(self):
        for change in [dict(accn=CIK+'-23-000002'), dict(filed='2023-02-17'), dict(end='2021-12-31'), dict(form='10-Q'), dict(start='2022-01-01')]:
            obj=facts();obj['facts']['us-gaap']['Assets']['units']['USD'][0].update(change)
            self.assertIn('missing_same_filing_instant_assets', extract(obj)[0]['structural_blockers'])
        obj=facts();obj['facts']['us-gaap']['Assets']['units']['EUR']=obj['facts']['us-gaap']['Assets']['units'].pop('USD')
        self.assertIn('missing_same_filing_instant_assets', extract(obj)[0]['structural_blockers'])

    def test_no_quarter_amendment_or_ifrs_substitution(self):
        for change in [dict(start='2022-10-01'), dict(form='10-K/A'), dict(form='20-F'), dict(filed='2022-12-01')]:
            obj=facts();obj['facts']['us-gaap']['GrossProfit']['units']['USD'][0].update(change);self.assertEqual(extract(obj), [])
        obj=facts();obj['facts']['ifrs-full']=obj['facts'].pop('us-gaap');self.assertEqual(extract(obj), [])

    def test_duration_inclusive_bounds(self):
        from datetime import date,timedelta
        for days, expected in [(329, False),(330,True),(400,True),(401,False)]:
            obj=facts();obj['facts']['us-gaap']['GrossProfit']['units']['USD'][0]['start']=(date(2022,12,31)-timedelta(days=days-1)).isoformat()
            self.assertEqual(bool(extract(obj)),expected)

    def test_optional_fiscal_metadata_is_checked_not_used_as_period(self):
        obj=facts();obj['facts']['us-gaap']['Assets']['units']['USD'][0]['fy']=2023
        self.assertIn('conflicting_filing_fiscal_years', extract(obj)[0]['structural_blockers'])
        obj=facts();obj['facts']['us-gaap']['GrossProfit']['units']['USD'][0]['fp']='Q4'
        self.assertIn('nonannual_fiscal_period_metadata', extract(obj)[0]['structural_blockers'])
        obj=facts()
        for tag in m.TAGS:obj['facts']['us-gaap'][tag]['units']['USD'][0]['fy']=2023
        self.assertTrue(extract(obj)[0]['structurally_complete'])

    def test_strict_json_identity_and_no_nonfinite_values(self):
        for raw in [b'[]', b'{"cik":123456,"entityName":"x","facts":{},"bad":NaN}', b'{"cik":123456,"cik":123456,"facts":{}}']:
            with self.assertRaises((ValueError, TypeError)):m.extract_company(CIK, [(raw, {'path':'x','sha256':'x'})], SESSIONS)
        obj=facts();obj['cik']=123457
        with self.assertRaises(ValueError):extract(obj)


class TimingAndFilingEvents(unittest.TestCase):
    def test_after_not_on_filed_day_and_holiday_weekend(self):
        self.assertEqual(m.first_session_after('2023-02-17',SESSIONS),'2023-02-21')
        self.assertEqual(m.first_session_after('2023-02-18',SESSIONS),'2023-02-21')
        self.assertEqual(m.first_session_after('2023-02-20',SESSIONS),'2023-02-21')
        self.assertIsNone(m.first_session_after('2023-02-22',SESSIONS))
        self.assertIsNone(m.first_session_after('2022-01-01',SESSIONS))
        self.assertEqual(m.first_session_after('2023-02-15',SESSIONS,'2023-02-01'),'2023-02-16')

    def test_later_comparative_keeps_its_filed_availability(self):
        cat={ACC:[dict(form='10-K',filingDate='2023-02-16',reportDate='2023-12-31')]}
        row=extract(catalog=cat)[0]
        self.assertEqual(row['period_role'],'comparative_period')
        self.assertFalse(m.temporally_available(row,'2023-02-16'))
        self.assertTrue(m.temporally_available(row,'2023-02-17'))
        self.assertFalse(row['admitted'])

    def test_filing_vintages_remain_separate(self):
        obj=facts()
        for tag in m.TAGS:
            rs=obj['facts']['us-gaap'][tag]['units']['USD'];rs.append(dict(rs[0],accn=CIK+'-23-000002',filed='2023-02-17'))
        rows=extract(obj);self.assertEqual(len(rows),2);self.assertEqual({r['first_usable_session'] for r in rows},{'2023-02-17','2023-02-21'})

    def test_missing_new_annual_filing_event_is_retained(self):
        newer=CIK+'-23-000002';cats={CIK:{ACC:[dict(form='10-K',filingDate='2023-02-16',reportDate='2022-12-31')],newer:[dict(form='10-K',filingDate='2023-02-17',reportDate='2023-01-31')]}}
        events=m.annual_filing_events([CIK],cats,extract(),SESSIONS)
        self.assertEqual(len(events),2)
        event=next(r for r in events if r['accession']==newer)
        self.assertEqual(event['current_period_candidate_count'],0)
        self.assertIn('no_current_annual_GP_Assets_pair',event['admission_blockers'])

    def test_catalog_not_context_and_conflicting_catalog_blocked(self):
        cat={ACC:[dict(form='10-K',filingDate='2023-02-16',reportDate='2022-12-31')]}
        row=extract(catalog=cat)[0];self.assertTrue(row['filing_catalog_verified']);self.assertFalse(row['original_filing_context_verified'])
        cat[ACC].append(dict(form='10-K',filingDate='2023-02-16',reportDate='2021-12-31'))
        row=extract(catalog=cat)[0];self.assertIn('filing_catalog_report_date_conflict',row['structural_blockers']);self.assertFalse(row['filing_catalog_verified'])

    def test_amended_or_foreign_annual_event_remains_a_blocked_event(self):
        cats={CIK:{ACC:[dict(form='10-K/A',filingDate='2023-02-17',reportDate='2022-12-31')]}}
        events=m.annual_filing_events([CIK],cats,[],SESSIONS)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['form'],'10-K/A')
        self.assertIn('annual_form_outside_10K_contract',events[0]['admission_blockers'])

    def test_period_after_catalog_report_date_is_blocked(self):
        cat={ACC:[dict(form='10-K',filingDate='2023-02-16',reportDate='2021-12-31')]}
        self.assertIn('period_after_catalog_report_date',extract(catalog=cat)[0]['structural_blockers'])

    def test_receipt_hash_url_and_timezone(self):
        raw=document(facts())[0];good=dict(sha256=m.source.sha(raw),source=m.source.endpoint(CIK),retrieved_at='2026-10-01T10:00:00+00:00')
        self.assertEqual(m.receipt_check(raw,CIK,good),[])
        for change in [dict(sha256='x'),dict(source='https://wrong'),dict(retrieved_at='2026-10-01T10:00:00')]:
            bad=dict(good,**change);self.assertTrue(m.receipt_check(raw,CIK,bad))

    def test_catalog_columns_and_cik(self):
        obj=dict(cik=int(CIK),filings={'recent':dict(accessionNumber=[ACC],filingDate=['2023-02-16'],form=['10-K'],reportDate=['2022-12-31'])})
        rows=m.catalog_rows(json.dumps(obj).encode(),{'path':'x','sha256':'x'},CIK);self.assertEqual(len(rows),1)
        obj['filings']['recent']['form']=[]
        with self.assertRaises(ValueError):m.catalog_rows(json.dumps(obj).encode(),{'path':'x','sha256':'x'},CIK)

    def test_catalog_receipt_timestamp_required(self):
        raw=b'{}';binding={'path':'registry_sec/0000123456_submissions.json'}
        origin=dict(sha256=m.source.sha(raw),source='https://data.sec.gov/submissions/CIK'+CIK+'.json',retrieved_at='2026-10-01T00:00:00+00:00')
        self.assertEqual(m.catalog_receipt_check(raw,binding,CIK,origin),[])
        origin['retrieved_at']='2026-10-01T00:00:00'
        self.assertIn('catalog_origin_retrieved_at_invalid',m.catalog_receipt_check(raw,binding,CIK,origin))

    def test_invalid_catalog_annual_rows_preserved_and_blocked(self):
        for field,value,reason in [('reportDate','bad','invalid_catalog_report_date'),('reportDate','2024-01-01','catalog_report_date_after_filed'),('filingDate','bad','invalid_catalog_filing_date')]:
            table=dict(accessionNumber=[ACC],filingDate=['2023-02-16'],form=['10-K'],reportDate=['2022-12-31'])
            table[field]=[value]
            entries=m.catalog_rows(json.dumps(table).encode(),{'path':'x','sha256':'x'},CIK)
            self.assertIn(reason,entries[0]['validation_blockers'])
            events=m.annual_filing_events([CIK],{CIK:{ACC:entries}},[],SESSIONS)
            self.assertEqual(len(events),1)
            self.assertIn(reason,events[0]['admission_blockers'])
            if field=='reportDate':
                row=extract(catalog={ACC:entries})[0]
                self.assertFalse(row['filing_catalog_verified'])
                self.assertIn('invalid_filing_catalog_row',row['structural_blockers'])

    def test_catalog_required_columns_missing_is_not_empty_success(self):
        with self.assertRaisesRegex(ValueError,'required columns'):
            m.catalog_rows(b'{"accessionNumber":[]}',{'path':'x','sha256':'x'},CIK)


    def test_archive_references_missing_and_unverified_are_explicit(self):
        name='CIK'+CIK+'-submissions-001.json'
        raw=json.dumps({'filings':{'files':[{'name':name}]}}).encode()
        names,reasons=m.referenced_archives(raw,CIK)
        self.assertEqual(names,[name]);self.assertEqual(reasons,[])
        primary={'cik':CIK,'raw':{'path':'primary.json'},'accepted':True,'referenced_archive_files':names,'archive_reference_blockers':[]}
        coverage=m.archive_coverage([primary])[0]
        self.assertEqual(coverage['referenced_archive_files_missing'],[name])
        secondary=dict(primary,raw={'path':'history/'+name},accepted=False,referenced_archive_files=[])
        coverage=m.archive_coverage([primary,secondary])[0]
        self.assertEqual(coverage['referenced_archive_files_missing'],[])
        self.assertEqual(coverage['referenced_archive_files_unverified'],[name])
        secondary['accepted']=True
        self.assertEqual(m.archive_coverage([primary,secondary])[0]['referenced_archive_files_unverified'],[])

    def test_archive_path_or_other_cik_reference_rejected(self):
        raw=json.dumps({'filings':{'files':[{'name':'../../x'}, {'name':'CIK9999999999-submissions-001.json'}]}}).encode()
        names,reasons=m.referenced_archives(raw,CIK)
        self.assertEqual(names,[]);self.assertIn('invalid_catalog_archive_reference',reasons)


    def test_malformed_catalog_container_blocks_cleanly(self):
        for raw in [b'[]',b'{"filings":[]}',b'{"filings":{"recent":[]}}']:
            with self.assertRaises(ValueError):m.catalog_rows(raw,{'path':'x','sha256':'x'},CIK)
        for raw in [b'[]',b'{"filings":[]}']:
            self.assertIn('invalid_catalog_archive_container',m.referenced_archives(raw,CIK)[1])


if __name__ == '__main__':
    unittest.main()

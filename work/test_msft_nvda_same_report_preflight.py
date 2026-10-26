"""Generated HTML/metadata only; no actual issuer cells or financial calculations."""
from copy import deepcopy
from datetime import date,timedelta
import io
from pathlib import Path
import sys
import tempfile
import unittest
import msft_nvda_same_report_preflight as m


def fixture(year=2021,symbol='MSFT'):
    label='Gross margin'if symbol=='MSFT'else'Gross profit'
    end=f'{year}-06-30';prev=f'{year-1}-06-30';release=f'{year}-07-20';available=f'{year}-07-21'
    html=f'''<html><p>July 20, {year} Microsoft (Nasdaq MSFT)</p>
    <table><caption>CONSOLIDATED STATEMENTS OF INCOME (In millions)</caption>
    <tr><td></td><td>Twelve Months Ended June 30, {year}</td><td>Twelve Months Ended June 30, {year-1}</td></tr>
    <tr><td>Revenue</td><td>$ 100</td><td>$ 90</td></tr>
    <tr><td>Cost of revenue</td><td>50</td><td>50</td></tr>
    <tr><td>{label}</td><td>$ 50</td><td>$ 40</td></tr></table>
    <table><caption>CONSOLIDATED BALANCE SHEETS (In millions)</caption>
    <tr><td></td><td>June 30, {year}</td><td>June 30, {year-1}</td></tr>
    <tr><td>Assets</td><td></td><td></td></tr>
    <tr><td>Total assets</td><td>$ 200</td><td>$ 180</td></tr></table></html>'''.encode()
    def period(y):
        start=f'{y-1}-07-01';end=f'{y}-06-30'
        return dict(start=start,end=end,days_inclusive=(date.fromisoformat(end)-date.fromisoformat(start)).days+1,
            derived_not_direct_quote=True,evidence=[dict(path='old.html',sha256='a'*64,release_date=f'{y}-07-20')])
    current=period(year);previous=period(year-1)
    if year==2015:previous.update(start=None,days_inclusive=None)
    rows=[]
    for kind,ti,ri,lbl in [('annual_gross_amount',0,3,label),('year_end_assets',1,2,'Total assets')]:
        rows.append(dict(kind=kind,table_index_zero_based=ti,row_index_zero_based=ri,row_label=lbl,
            matching_cells=[dict(source_cell_index=1,grid_columns=[1])],units=dict(literal='(In millions)')))
    spec=dict(symbol=symbol,fiscal_year=year,release_date=release,release_date_literal=f'July 20, {year}',
        source_url='https://www.microsoft.com/synthetic',raw_path='synthetic.html',raw_sha256=m.digest(html),
        receipt_path='synthetic.meta.json',receipt_sha256='a'*64,receipt_entry_key=None,
        current_period=current,comparison_period=previous,comparison_end=prev,identity_anchor='Microsoft (Nasdaq MSFT)',rows=rows)
    return spec,html,[f'{year}-07-19',release,available,f'{year}-07-22']


class CellTests(unittest.TestCase):
    def setUp(self):self.s,self.raw,self.sessions=fixture()
    def event(self):return m.extract_event(self.s,self.raw,self.sessions)
    def test_four_exact_cells_and_original_scales(self):
        e=self.event();self.assertTrue(e['four_cells_extracted']);self.assertEqual(e['rejection_codes'],[])
        self.assertEqual([e[r][k]['value_decimal']for r in ('current','comparison')for k in ('gross_profit','assets')],['50','200','40','180'])
        self.assertTrue(all(e[r][k]['scale']=='1000000'for r in ('current','comparison')for k in ('gross_profit','assets')))
    def test_same_current_availability_for_prior_values(self):
        e=self.event()
        self.assertEqual(e['comparison']['gross_profit']['available_from'],'2021-07-21')
        self.assertNotEqual(e['comparison']['gross_profit']['period_end'],e['available_from'])
    def test_prior_cell_locator_distinct_from_current(self):
        e=self.event()
        self.assertEqual(e['comparison']['gross_profit']['locator']['physical_cell_index_zero_based'],2)
        self.assertEqual(e['current']['gross_profit']['locator']['physical_cell_index_zero_based'],1)
    def test_source_hash_and_parsed_cell_digest_retained(self):
        e=self.event();c=e['comparison']['assets'];self.assertEqual(c['source']['raw_sha256'],self.s['raw_sha256'])
        self.assertRegex(c['parsed_cell_sha256'],'^[0-9a-f]{64}$')
    def test_wrong_current_locator_is_rejected_without_dropping_event(self):
        self.s['rows'][0]['matching_cells'][0]['source_cell_index']=2;e=self.event()
        self.assertIsNone(e['current']['gross_profit']);self.assertIsNotNone(e['comparison']['gross_profit'])
        self.assertIn('cell_unresolved:current:gross_profit',e['rejection_codes'])
    def test_wrong_row_label_is_rejected(self):
        self.s['rows'][0]['row_label']='Net income';self.assertFalse(self.event()['four_cells_extracted'])
    def test_duplicate_row_label_is_rejected(self):
        self.raw=self.raw.replace(b'</table>',b'<tr><td>Gross margin</td><td>1</td><td>1</td></tr></table>',1)
        self.assertIsNone(self.event()['comparison']['gross_profit'])
    def test_prior_date_mismatch_is_not_filled_from_another_report(self):
        self.raw=self.raw.replace(b'Twelve Months Ended June 30, 2020',b'Twelve Months Ended June 30, 2019')
        e=self.event();self.assertIsNone(e['comparison']['gross_profit']);self.assertIsNotNone(e['current']['gross_profit'])
    def test_quarter_column_is_not_annual(self):
        self.raw=self.raw.replace(b'Twelve Months Ended June 30, 2020',b'Three Months Ended June 30, 2020')
        self.assertIsNone(self.event()['comparison']['gross_profit'])
    def test_percent_is_not_monetary(self):
        self.raw=self.raw.replace(b'$ 40',b'40%');self.assertIsNone(self.event()['comparison']['gross_profit'])
    def test_missing_prior_amount_is_not_zero(self):
        self.raw=self.raw.replace(b'$ 40',b'');self.assertIsNone(self.event()['comparison']['gross_profit'])
    def test_nonfinite_amount_is_rejected(self):
        for value in (b'NaN',b'Infinity',b'1e2'):
            self.raw=fixture()[1].replace(b'$ 40',value)
            self.assertIsNone(self.event()['comparison']['gross_profit'])
    def test_zero_and_negative_gross_profit_are_preserved(self):
        for text,want in [(b'0','0'),(b'(40)','-40')]:
            self.raw=fixture()[1].replace(b'$ 40',text)
            self.assertEqual(self.event()['comparison']['gross_profit']['value_decimal'],want)
    def test_missing_declared_scale_rejected(self):
        self.s['rows'][0]['units']['literal']='(In thousands)';self.assertIsNone(self.event()['current']['gross_profit'])
    def test_non_gaap_statement_is_rejected(self):
        self.raw=self.raw.replace(b'CONSOLIDATED STATEMENTS OF INCOME',b'NON-GAAP CONSOLIDATED STATEMENTS OF INCOME')
        self.assertIsNone(self.event()['current']['gross_profit'])
    def test_current_and_prior_different_scales_not_silently_normalized(self):
        self.raw=self.raw.replace(b'BALANCE SHEETS (In millions)',b'BALANCE SHEETS (In thousands)')
        self.s['rows'][1]['units']['literal']='(In thousands)';e=self.event()
        self.assertEqual(e['current']['assets']['scale'],'1000');self.assertEqual(e['current']['gross_profit']['scale'],'1000000')
    def test_no_currency_iso_is_fabricated(self):self.assertIsNone(self.event()['comparison']['assets']['currency_iso'])
    def test_all_admission_flags_false(self):
        e=self.event()
        for k in ('fact_admitted','model_admitted','strategy_admitted','first_public_version_verified','complete_correction_history_verified'):self.assertIs(e[k],False)


class PeriodIdentityTests(unittest.TestCase):
    def test_unknown_2015_prior_start_retained_with_four_raw_cells(self):
        s,raw,axis=fixture(2015);e=m.extract_event(s,raw,axis)
        self.assertTrue(e['four_cells_extracted']);self.assertIsNone(e['comparison']['period']['start'])
        self.assertIn('unknown_comparison_period_start',e['rejection_codes'])
    def test_unknown_2015_start_cannot_be_invented(self):
        s,raw,axis=fixture(2015);s['comparison_period'].update(start='2013-07-01',days_inclusive=365)
        with self.assertRaisesRegex(ValueError,'unknown boundary'):m.extract_event(s,raw,axis)
    def test_future_boundary_evidence_rejected(self):
        s,raw,axis=fixture();s['comparison_period']['evidence'][0]['release_date']='2022-01-01'
        with self.assertRaisesRegex(ValueError,'Future'):m.extract_event(s,raw,axis)
    def test_wrong_annual_period_day_count_rejected(self):
        s,raw,axis=fixture();s['comparison_period']['days_inclusive']=365
        with self.assertRaisesRegex(ValueError,'duration'):m.extract_event(s,raw,axis)
    def test_noncontiguous_period_rejected(self):
        s,raw,axis=fixture();s['current_period']['start']='2020-07-02'
        with self.assertRaisesRegex(ValueError,'Noncontiguous'):m.extract_event(s,raw,axis)
    def test_period_year_mismatch_rejected(self):
        s,raw,axis=fixture();s['fiscal_year']=2022
        with self.assertRaisesRegex(ValueError,'year'):m.extract_event(s,raw,axis)
    def test_wrong_or_unknown_spec_field_rejected(self):
        s,raw,axis=fixture();s['future_hindsight']=True
        with self.assertRaisesRegex(ValueError,'specification fields'):m.extract_event(s,raw,axis)
    def test_missing_identity_anchor_rejects_but_retains_cells(self):
        s,raw,axis=fixture();s['identity_anchor']='Future Security TICK';e=m.extract_event(s,raw,axis)
        self.assertIn('own_issuer_ticker_anchor_missing',e['rejection_codes']);self.assertTrue(e['four_cells_extracted'])
    def test_publication_date_literal_required(self):
        s,raw,axis=fixture();s['release_date_literal']='July 21, 2021';e=m.extract_event(s,raw,axis)
        self.assertIn('current_publication_date_literal_missing',e['rejection_codes'])
    def test_identity_is_own_availability_to_fixed_end_plus550(self):
        s,raw,axis=fixture();e=m.extract_event(s,raw,axis)
        self.assertEqual(e['identity']['valid_from'],e['available_from'])
        self.assertEqual(e['identity']['valid_through'],(date(2021,6,30)+timedelta(days=550)).isoformat())
    def test_expired_late_publication_rejected(self):
        s,raw,axis=fixture();s['release_date']='2023-06-30';axis=['2023-06-29','2023-06-30','2023-07-03']
        with self.assertRaisesRegex(ValueError,'identity expiry'):m.extract_event(s,raw,axis)
    def test_current_restatement_never_backdates(self):
        s,raw,axis=fixture(2018);e=m.extract_event(s,raw,axis)
        self.assertIn('accounting_context_requires_independent_review',e['rejection_codes'])
        self.assertEqual(e['comparison']['gross_profit']['available_from'],'2018-07-21')
    def test_strict_next_session_and_closed_day(self):
        self.assertEqual(m.next_session('2021-07-20',['2021-07-19','2021-07-22','2021-07-23']),'2021-07-22')
    def test_duplicate_or_incomplete_calendar_rejected(self):
        for axis in (['2021-07-20','2021-07-20'],['2021-07-19','2021-07-20']):
            with self.assertRaises(ValueError):m.next_session('2021-07-20',axis)


class ProvenanceTests(unittest.TestCase):
    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw in ('{"x":1,"x":2}','{"x":NaN}'):
            with self.assertRaises(ValueError):m.decode(raw)
    def test_hash_mutation_detected(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'x';p.write_bytes(b'a');m.verify(d,{'x':m.digest(b'a')});p.write_bytes(b'b')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):m.verify(d,{'x':m.digest(b'a')})
    def test_exclusive_outputs_never_overwrite(self):
        with tempfile.TemporaryDirectory()as d:
            m.publish(d,'x.json',{'x':1})
            with self.assertRaises(FileExistsError):m.publish(d,'x.json',{'x':2})
            self.assertEqual(m.read(d,'x.json'),{'x':1})
    def test_cloud_parent_escape_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            (Path(d)/'x').write_bytes(b'x');(Path(d)/'link').symlink_to(Path(d)/'x')
            for p in ('../x','/tmp/x','CloudStorage/x','link'):
                with self.assertRaises(ValueError):m.path(d,p)
    def test_schema_is_source_only_and_scope_includes_sentinels(self):
        s=m.schema();self.assertFalse(s['financial_calculations_permitted']);self.assertFalse(s['first_public_version_verified'])
        self.assertEqual(len(m.SCOPE),24);self.assertIn(('NVDA',2015),m.SCOPE);self.assertIn(('MSFT',2015),m.SCOPE)


if __name__=='__main__':
    write='--write-report'in sys.argv
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    text=stream.getvalue();print(text,end='')
    if write:
        report=dict(passed=result.wasSuccessful(),synthetic_only=True,tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),real_source_cells_extracted=0,
            source_code_hashes={p:m.sha(m.ROOT,p)for p in(m.SELF,m.TEST)},log=text)
        m.publish(m.ROOT,m.OUT+'/TESTS.json',report)
    raise SystemExit(0 if result.wasSuccessful()else 1)

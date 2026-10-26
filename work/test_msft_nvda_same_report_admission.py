"""Pure generated source/review fixtures; no real mappings or market calculations."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
import msft_nvda_same_report_admission as m
import msft_nvda_same_report_preflight as pre
from test_msft_nvda_same_report_preflight import fixture


def generated():
    rows=[];sessions=[];bindings={'old.html':'a'*64};review_rows=[]
    for symbol in m.YEARS:
        for year in m.YEARS[symbol]:
            s,raw,axis=fixture(year,symbol);s['raw_path']=f'synthetic/{symbol}/{year}.html';s['receipt_path']=f'synthetic/{symbol}/{year}.meta.json'
            e=pre.extract_event(s,raw,axis);rows.append(e);sessions+=axis
            bindings[s['raw_path']]=s['raw_sha256'];bindings[s['receipt_path']]=s['receipt_sha256']
            codes=list(e['rejection_codes'])
            accounting=dict(status='no_flag',resolution_code=None,scope=None,evidence=[])
            if (symbol,year)in m.ACCOUNTING:
                accounting=dict(status='resolved',resolution_code='synthetic_source_context_verified',scope='Both compared consolidated statements',
                    evidence=[dict(path=s['raw_path'],sha256=s['raw_sha256'],quote='Synthetic comparative context',locator={'paragraph':1})])
                codes.remove('accounting_context_requires_independent_review')
            review_rows.append(dict(event_id=e['event_id'],symbol=symbol,fiscal_year=year,cells_verified=4,raw_cells_verified=True,availability_verified=True,
                identity_verified_under_assumptions=True,periods_verified=True,remaining_rejection_codes=codes,accounting=accounting))
    review=dict(schema_version=1,stage='independent_same_report_source_review',method=m.METHOD,passed=True,source_only=True,
        financial_calculations=0,model_fits=0,ranks_computed=0,returns_computed=0,source_hashes=bindings,
        preflight_plan_sha256=m.PINS[m.PRE+'/PLAN.json'],events_sha256=m.PINS[m.PRE+'/EVENTS.json'],events_reviewed=24,cells_reviewed=96,events=review_rows,
        first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False)
    policy=m.policy({'assumptions':{k:'Synthetic original assumption'for k in ('calendar','currency','identity_window','period_boundaries','vintage','status','stop')}})
    return {'events':rows},review,sorted(set(sessions)),bindings,policy


class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.template=generated()
    def setUp(self):self.p,self.r,self.axis,self.bindings,self.policy=deepcopy(self.template)
    def build(self):return m.build_dataset(self.p,self.r,self.axis,self.bindings,self.policy,'b'*64)
    def event(self):return self.p['events'][1]
    def test_all24_events_all96_values_preserved(self):
        before=deepcopy(self.p);d=self.build();self.assertEqual(self.p,before)
        self.assertEqual(d['counts'],dict(events=24,conditionally_admitted=22,rejected=2,raw_values=96))
        for e,o in zip(self.p['events'],d['events']):
            for role in ('current','comparison'):
                for k in ('gross_profit','assets'):
                    self.assertEqual(o['same_report'][role][k]['value'],e[role][k]['value_decimal'])
                    self.assertEqual(o['same_report'][role][k]['scale'],e[role][k]['scale'])
    def test_two_unknown_2015_durations_remain_rejected(self):
        for e in self.build()['events']:
            if e['fiscal_year']==2015:
                self.assertFalse(e['admitted']);self.assertIsNone(e['same_report']['comparison']['period_start'])
    def test_unresolved_accounting_is_preserved(self):
        for r in self.r['events']:
            if (r['symbol'],r['fiscal_year'])in m.ACCOUNTING:
                r['accounting']['status']='unresolved';r['remaining_rejection_codes']=['accounting_context_requires_independent_review']
        self.assertEqual(self.build()['counts']['conditionally_admitted'],18)
    def test_resolution_without_evidence_cannot_clear_flag(self):
        self.r['events'][3]['accounting']['evidence']=[]
        with self.assertRaisesRegex(ValueError,'evidence'):self.build()
    def test_other_source_cannot_resolve_accounting(self):
        self.r['events'][3]['accounting']['evidence'][0]['path']='other.html'
        with self.assertRaisesRegex(ValueError,'Unbound accounting'):self.build()
    def test_bound_other_source_still_cannot_resolve_accounting(self):
        self.r['source_hashes']['other.html']=self.r['events'][3]['accounting']['evidence'][0]['sha256']
        self.r['events'][3]['accounting']['evidence'][0]['path']='other.html'
        with self.assertRaisesRegex(ValueError,'this current source'):self.build()
    def test_additional_accounting_evidence_must_be_bound(self):
        self.r['events'][3]['accounting']['evidence'].append(dict(path='unbound.html',sha256='b'*64,quote='x',locator=1))
        with self.assertRaisesRegex(ValueError,'Unbound accounting'):self.build()
    def test_missing_resolution_explanation_rejected(self):
        self.r['events'][3]['accounting']['scope']=''
        with self.assertRaisesRegex(ValueError,'explanation'):self.build()
    def test_review_cannot_remove_duration_rejection(self):
        self.r['events'][0]['remaining_rejection_codes']=[]
        with self.assertRaisesRegex(ValueError,'sentinels'):self.build()
    def test_95_reviewed_cells_not_enough(self):
        self.r['cells_reviewed']=95
        with self.assertRaisesRegex(ValueError,'counter'):self.build()
    def test_single_cell_review_failure_blocks_batch(self):
        self.r['events'][4]['cells_verified']=3
        with self.assertRaisesRegex(ValueError,'counter'):self.build()
    def test_source_review_must_be_passed_and_source_only(self):
        for k in ('passed','source_only'):
            old=self.r[k];self.r[k]=False
            with self.assertRaises(ValueError):self.build()
            self.r[k]=old
    def test_all_event_review_booleans_required(self):
        for k in ('raw_cells_verified','availability_verified','identity_verified_under_assumptions','periods_verified'):
            self.r['events'][1][k]=False
            with self.assertRaisesRegex(ValueError,'event check'):self.build()
            self.r['events'][1][k]=True
    def test_bool_or_nonzero_counters_rejected(self):
        for k in ('financial_calculations','model_fits','ranks_computed','returns_computed'):
            for v in (True,1):
                self.r[k]=v
                with self.assertRaisesRegex(ValueError,'counter'):self.build()
            self.r[k]=0
    def test_wrong_preflight_hash_rejected(self):
        self.r['events_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'binding'):self.build()
    def test_missing_reordered_or_extra_event_rejected(self):
        self.p['events'].reverse()
        with self.assertRaisesRegex(ValueError,'ordered'):self.build()
    def test_review_event_wrong_issuer_rejected(self):
        self.r['events'][1]['symbol']='NVDA'
        with self.assertRaisesRegex(ValueError,'issuer'):self.build()
    def test_unknown_preflight_field_rejected(self):
        self.event()['future_value']=123
        with self.assertRaisesRegex(ValueError,'event fields'):self.build()
    def test_unknown_amount_field_rejected(self):
        self.event()['current']['assets']['extra']=1
        with self.assertRaisesRegex(ValueError,'cell fields'):self.build()
    def test_no_first_version_certification(self):
        self.r['first_public_version_verified']=True
        with self.assertRaisesRegex(ValueError,'certify'):self.build()
    def test_wrong_current_availability_rejected(self):
        self.event()['available_from']='2016-07-20'
        with self.assertRaisesRegex(ValueError,'next-session'):self.build()
    def test_prior_amount_not_backdated(self):
        self.event()['comparison']['gross_profit']['available_from']='2015-07-20'
        with self.assertRaisesRegex(ValueError,'availability'):self.build()
    def test_future_boundary_evidence_rejected(self):
        self.event()['comparison']['period']['evidence'][0]['release_date']='2026-07-20'
        with self.assertRaisesRegex(ValueError,'Future boundary'):self.build()
    def test_invalid_period_or_noncontiguous_dates_rejected(self):
        self.event()['current']['period']['days_inclusive']=20
        with self.assertRaisesRegex(ValueError,'duration'):self.build()
    def test_missing_identity_anchor_rejected(self):
        self.event()['identity']['anchor_in_current_source']=False
        with self.assertRaisesRegex(ValueError,'identity anchor'):self.build()
    def test_identity_window_cannot_extend(self):
        self.event()['identity']['valid_through']='2099-01-01'
        with self.assertRaisesRegex(ValueError,'identity window'):self.build()
    def test_source_mismatch_between_cells_rejected(self):
        self.event()['comparison']['gross_profit']['source']['raw_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Cell source'):self.build()
    def test_unbound_raw_source_rejected(self):
        self.bindings[self.event()['source']['raw_path']]='0'*64
        with self.assertRaisesRegex(ValueError,'Unbound'):self.build()
    def test_nonfinite_float_and_bool_values_rejected(self):
        for value in ('NaN','Infinity','1e2',1.0,True):
            self.event()['current']['gross_profit']['value_decimal']=value
            with self.assertRaises(ValueError):self.build()
    def test_negative_zero_gross_profit_preserved(self):
        for v in ('0','-20'):
            self.event()['current']['gross_profit']['value_decimal']=v
            self.assertEqual(self.build()['events'][1]['same_report']['current']['gross_profit']['value'],v)
    def test_nonpositive_assets_rejected(self):
        self.event()['current']['assets']['value_decimal']='0'
        with self.assertRaisesRegex(ValueError,'Nonpositive'):self.build()
    def test_currency_iso_not_inferred_and_scale_not_arithmetic(self):
        self.event()['current']['assets']['currency_iso']='USD'
        with self.assertRaisesRegex(ValueError,'literal currency'):self.build()
    def test_latest_invalid_is_not_filtered_before_selection(self):
        d=self.build();rows=d['events'];rows[1]['admitted']=False
        self.assertFalse(m.base.latest_metadata(rows,'MSFT',rows[1]['available_from'])['usable'])
    def test_expired_latest_does_not_fallback(self):
        d=self.build();row=d['events'][-1]
        self.assertEqual(m.base.latest_metadata(d['events'],'NVDA','2099-01-01')['reason'],'identity_expired')
    def test_models_and_strategy_remain_unadmitted(self):
        d=self.build();self.assertFalse(d['model_admitted']);self.assertFalse(d['strategy_admitted'])
        self.assertEqual(d['ratios_computed'],0);self.assertEqual(d['ranks_computed'],0)
    def test_evidence_carries_original_rejections_and_accounting(self):
        d=self.build();self.assertTrue(d['events'][3]['admitted'])
        self.assertEqual(d['events'][3]['evidence']['original_rejection_codes'],['accounting_context_requires_independent_review'])


class BindingTests(unittest.TestCase):
    def test_omitted_or_bad_cli_hashes_rejected_before_reading(self):
        with tempfile.TemporaryDirectory()as d:
            for value in (None,'','xyz','A'*64,True):
                with self.assertRaisesRegex(ValueError,'Explicit SHA256'):m.load_plan(d,value)
                with self.assertRaisesRegex(ValueError,'Explicit SHA256'):m.bind_review(d,'a'*64,value)
                with self.assertRaisesRegex(ValueError,'Explicit SHA256'):m.products(d,'a'*64,value)
                with self.assertRaisesRegex(ValueError,'Explicit SHA256'):m.verify_outputs(d,'a'*64,value)
    def test_conflicting_review_source_hashes_rejected(self):
        with self.assertRaisesRegex(ValueError,'Conflicting'):m.merge({'a':'x'},{'a':'y'})
    def test_missing_review_hash_cannot_read(self):
        with tempfile.TemporaryDirectory()as d:
            Path(d,'x').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):m.read(d,'x','0'*64)
    def test_exclusive_binding_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory()as d:
            m.publish(d,'binding.json',{'sha':'first'})
            with self.assertRaises(FileExistsError):m.publish(d,'binding.json',{'sha':'second'})
    def test_policy_requires_review_before_mapping(self):
        p=generated()[-1];self.assertTrue(p['frozen_review_binding_required_before_mapping']);self.assertEqual(p['review_gate']['cells'],96)


if __name__=='__main__':
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    text=stream.getvalue();print(text,end='')
    if '--write-report'in sys.argv:
        m.publish(m.ROOT,m.OUT+'/TESTS.json',dict(passed=r.wasSuccessful(),synthetic_only=True,tests_run=r.testsRun,failures=len(r.failures),errors=len(r.errors),real_source_mappings=0,
            source_code_hashes={p:m.sha(m.ROOT,p)for p in (m.SELF,m.TEST)},log=text))
    raise SystemExit(0 if r.wasSuccessful()else 1)

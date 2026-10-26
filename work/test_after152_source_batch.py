"""Pure synthetic source-batch transport/provenance checks; no real extracts."""
import copy
import io
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest.mock import patch

import after152_source_batch as b
import after152_mu_extract as mu
import after152_pep_extract as pep
import test_after152_mu_extract as fixtures


def one(symbol='MU',year=None):
    raw,c=fixtures.fixture(mu if symbol=='MU' else pep,year=year)
    c['available_session_verified']=True
    return c,(mu if symbol=='MU' else pep).extract_document(raw,c)


def catalog():
    docs=[]
    for s,years in [('MU',range(2019,2026)),('PEP',range(2010,2026))]:
        for y in years:
            chosen=(s=='MU' and y>=2024) or (s=='PEP' and y<=2016)
            if chosen:c,_=one(s,y)
            else:
                c,_=one(s)
                c.update(fiscal_year=y,event_id=s+'_unprocessed_'+str(y),context_status='unprocessed',fields={},periods=None)
                c['source']['path']='synthetic/'+s+'_'+str(y)+'.html'
            if s=='PEP' and y>=2019:
                c.update(source_kind='index_only',original_report_bytes_verified=False)
                c['source'].update(receipt_path=None,receipt_sha256=None)
            docs.append(c)
    return dict(events=docs,input_hashes={})


def input_hashes():
    pins={p:'c'*64 for p in b.BASE_INPUTS};pins[b.CONTEXT]=b.CONTEXT_SHA;pins[b.FEASIBILITY]=b.FEASIBILITY_SHA
    return pins


class TransportTests(unittest.TestCase):
    def test_real_parsers_on_nine_synthetic_contexts(self):
        for s,years in [('MU',(2024,2025)),('PEP',range(2010,2017))]:
            for y in years:
                c,v=one(s,y);b.validate_record(v,c)
                self.assertFalse(v['fact_admitted'])

    def test_signed_income_zero_no_profit_filter(self):
        for amount in ('-10','0','10'):
            c,v=one();v['fields']['annual_net_income']['value_decimal']=amount
            b.validate_record(v,c)

    def test_catalog_full23_order_and_exact9_selection(self):
        ctx=catalog();self.assertEqual(len(b.records(ctx)),9)
        self.assertEqual(len(b.catalog_metadata(ctx)),23)
        for docs in (ctx['events'][:-1],ctx['events'][::-1],ctx['events']+[ctx['events'][0]]):
            with self.assertRaises(ValueError):b.records(dict(events=docs))
        bad=copy.deepcopy(ctx);bad['events'][0]['context_status']=b.READY
        with self.assertRaises(ValueError):b.records(bad)

    def test_index_entries_never_selected_as_originals(self):
        ctx=catalog();chosen=b.records(ctx);chosen[0]['source_kind']='index_only'
        with self.assertRaises(ValueError):b.records(ctx)

    def test_catalog_holds_source_kind_and_missing_receipt_preserved(self):
        ctx=catalog();before=copy.deepcopy(ctx);out=b.catalog_metadata(ctx)
        self.assertEqual(out[-1]['source'],ctx['events'][-1]['source']);self.assertIsNone(out[-1]['source']['receipt_path'])
        self.assertEqual(out[0]['context_holds'],ctx['events'][0]['context_holds']);self.assertEqual(ctx,before)

    def test_exact_policy_contract_and_bindings(self):
        ctx=catalog();policy=b.make_policy(ctx,input_hashes(),'synthetic-time')
        b.validate_policy(policy,ctx,b.CONTEXT_SHA)
        self.assertEqual(policy['periods']['boundary_assumptions']['PEP']['days_inclusive'],[364,371])
        self.assertIs(policy['preservation']['first_public_time_verified'],False)
        self.assertEqual(policy['source_scope']['html_context_reviewed_events'],9)
        self.assertEqual(policy['source_scope']['catalog_events'],23)

    def test_policy_tamper_rejected(self):
        ctx=catalog();base=b.make_policy(ctx,input_hashes(),'synthetic-time')
        for edit in (lambda p:p['known_time'].update(maximum_age_days=551),
                     lambda p:p['periods']['boundary_assumptions']['MU'].update(days_inclusive=[365]),
                     lambda p:p['preservation'].update(first_public_time_verified=True),
                     lambda p:p['fixed_catalog_metadata'].pop()):
            p=copy.deepcopy(base);edit(p)
            with self.assertRaises(ValueError):b.validate_policy(p,ctx,b.CONTEXT_SHA)

    def test_missing_or_bad_policy_pin_rejected(self):
        pins=input_hashes();pins.pop(b.CONTEXT)
        with self.assertRaises(ValueError):b.make_policy(catalog(),pins)
        pins=input_hashes();pins[b.FEASIBILITY]='z'*64
        with self.assertRaises(ValueError):b.make_policy(catalog(),pins)

    def test_record_identity_clock_holds_source_unchanged(self):
        for key,value in [('symbol','OTHER'),('event_id','OTHER'),('fiscal_year',2001),('source',{}),
                          ('context_holds',[]),('publication_date','2000-01-01'),('available_from','2000-01-01'),
                          ('accounting_review_flags',[]),('accounting_context_numbers_redacted',[])]:
            c,v=one();v[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):b.validate_record(v,c)

    def test_record_cannot_upgrade_certification_or_admission(self):
        for key in ('fact_admitted','model_admitted','strategy_admitted','first_public_version_verified',
                    'first_public_time_verified','complete_correction_history_verified','certified_daily_security_master'):
            c,v=one();v[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):b.validate_record(v,c)
        for value in (True,1,None):
            c,v=one();v['financial_calculations']=value
            with self.assertRaises(ValueError):b.validate_record(v,c)

    def test_field_literal_locator_scale_clock_currency_binding(self):
        for key,value in [('source',{}),('available_from','2000-01-01'),('value_decimal','NaN'),('scale','Infinity'),
                          ('scale','1000'),('scale',1000000),('displayed_amount',''),('locator',{}),
                          ('currency_literal','EUR'),('currency_iso','USD'),('as_of','2000-01-01'),
                          ('fact_admitted',True),('period',{})]:
            c,v=one();v['fields']['annual_net_income'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):b.validate_record(v,c)

    def test_assets_positive_instant_only(self):
        for field in ('current_total_assets','prior_total_assets'):
            for key,value in [('value_decimal','-1'),('value_decimal','0'),('period',{'bad':True})]:
                c,v=one();v['fields'][field][key]=value
                with self.assertRaises(ValueError):b.validate_record(v,c)

    def test_null_needs_field_specific_reason_and_completeness_false(self):
        c,v=one();v['fields']['annual_net_income']=None
        with self.assertRaises(ValueError):b.validate_record(v,c)
        v['gaps']=[dict(field='annual_net_income',reason='synthetic unsupported geometry')];v['complete_required_fields']=False
        b.validate_record(v,c)
        self.assertEqual(v['context_holds'],c['context_holds'])

    def test_nonnull_with_gap_duplicate_or_unknown_gap_rejected(self):
        for gaps in ([dict(field='annual_net_income',reason='contradiction')],
                     [dict(field='UNKNOWN',reason='bad')],
                     [dict(field='prior_total_assets',reason='bad')]*2):
            c,v=one();v['gaps']=gaps
            with self.assertRaises(ValueError):b.validate_record(v,c)

    def test_period_or_assumption_tamper_rejected(self):
        for key,value in [('start_basis','directly_quoted'),('start_assumption_certified',True),('assumption_id','OTHER')]:
            c,v=one();v['periods'][key]=value
            with self.assertRaises(ValueError):b.validate_record(v,c)
        c,v=one();v['periods']['current']['days_inclusive']=365
        with self.assertRaises(ValueError):b.validate_record(v,c)

    def test_pin_collision_and_bad_hash_fail(self):
        with patch.object(b.shared,'relative',side_effect=lambda p:p),patch.object(b.shared,'sha',return_value='a'*64):
            with self.assertRaises(ValueError):b._pins({'synthetic':'a'*64},{'synthetic':'b'*64})
            with self.assertRaises(ValueError):b._pins({'synthetic':'b'*64},{})

    def test_wrong_plan_hash_fails_before_data_read(self):
        with patch.object(b.shared,'sha',return_value='a'*64),patch.object(b.shared,'load',side_effect=AssertionError('No read')):
            with self.assertRaises(ValueError):b.run('b'*64)

    def test_freeze_preserves_catalog_and_never_calls_extractors(self):
        ctx=catalog();pins=input_hashes();policy=b.make_policy(ctx,pins,'synthetic-time')
        parser_files=['work/'+m+'.py' for m in b.MODULES.values()]+['work/test_'+m+'.py' for m in b.MODULES.values()]
        review_pins={p:'d'*64 for p in parser_files}
        hashes=dict(pins);hashes.update(review_pins);hashes[b.POLICY]='e'*64
        hashes[b.PREDECESSOR]='f'*64;hashes['work/synthetic_review.json']='a'*64
        for event in ctx['events']:
            s=event['source'];hashes[s['path']]=s['sha256']
            if s['receipt_path'] is not None:hashes[s['receipt_path']]=s['receipt_sha256']
        # All shared synthetic issuer source paths are assigned their actual
        # latest fixture digest: pin each source independently for this test.
        for i,event in enumerate(ctx['events']):
            event['source']['path']='synthetic/report'+str(i)+'.html';hashes[event['source']['path']]=event['source']['sha256']
        policy=b.make_policy(ctx,pins,'synthetic-time')
        state={'completed':[{}]*151+[dict(directory='monthly_financial_coverage152',completion={'sha256':'f'*64})],'registered_queue':[]}
        docs={b.CONTEXT:ctx,b.POLICY:policy,'outputs/monthly_target70_research/CONTINUATION_STATUS.json':state,
              'work/synthetic_review.json':dict(passed=True,input_hashes=review_pins)}
        captured=[]
        with ExitStack() as stack:
            stack.enter_context(patch.object(b.shared,'sha',side_effect=lambda p:hashes[str(p)]))
            stack.enter_context(patch.object(b.shared,'relative',side_effect=lambda p:str(p)))
            stack.enter_context(patch.object(b.shared,'load',side_effect=lambda p:docs[str(p)]))
            stack.enter_context(patch.object(b.shared,'code_closure',return_value=review_pins))
            stack.enter_context(patch.object(b.shared,'write_new',side_effect=lambda p,o:captured.append(copy.deepcopy(o))))
            stack.enter_context(patch.object(b,'synthetic_tests',return_value=[dict(tests=1,passed=True)]))
            stack.enter_context(patch.object(mu,'extract_document',side_effect=AssertionError('No monetary extraction during freeze')))
            stack.enter_context(patch.object(pep,'extract_document',side_effect=AssertionError('No monetary extraction during freeze')))
            hashes[str(b.OUT/'PLAN.json')]='9'*64
            with redirect_stdout(io.StringIO()):b.freeze('f'*64,'work/synthetic_review.json','a'*64)
        self.assertEqual(len(captured),1);plan=captured[0]
        self.assertEqual(len(plan['records']),9);self.assertEqual(len(plan['catalog_ids']),23)
        self.assertEqual(plan['catalog_metadata'],b.catalog_metadata(ctx));self.assertTrue(plan['no_admission_by_extraction'])


if __name__=='__main__':unittest.main(verbosity=2)

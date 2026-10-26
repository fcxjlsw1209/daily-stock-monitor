"""Synthetic-only regression, geometry, and three-event phase-contract tests."""
from copy import deepcopy
from datetime import datetime, timezone
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import stock_comp_pdf_repair_v2 as m
import stock_comp_pdf_repair_v2_runner as r
from test_cashflow_pdf_repair_v4 import FakePage, page, locator

LABEL = 'Share-based compensation expense'


def loc():
    result = locator()
    result.update(format='pdf', row_label=LABEL, page_1based=1, line_1based=1,
                  scale='1000000', currency_literal='$', period_end='2030-09-28')
    return result


class TextPage(FakePage):
    def close(self): pass


class PdfPage:
    def __init__(self, textpage): self.textpage = textpage
    def get_textpage(self): return self.textpage
    def close(self): pass


class PdfDocument:
    def __init__(self, textpage): self.textpage = textpage
    def __getitem__(self, index):
        assert index == 0
        return PdfPage(self.textpage)
    def close(self): pass


class PdfModule:
    def __init__(self, textpage): self.textpage = textpage
    def PdfDocument(self, raw):
        assert raw == b'%PDF-synthetic'
        return PdfDocument(self.textpage)


def extract(tail='2, 3 4 5  6,789', values=('2,345', '6,789'), textpage=None, location=None):
    sample = textpage or page(values, label=LABEL)
    tp = TextPage(); tp.chars = sample.chars
    with patch.object(m.base, 'configure_pdf', return_value=(None, PdfModule(tp))):
        return m.extract_cell(b'%PDF-synthetic', [LABEL+'    '+tail], location or loc(), {})


def synthetic_events():
    ids = list(r.TARGETS)+['SYNTHETIC:'+str(i) for i in range(41)]
    specs, events = [], []
    for i, event_id in enumerate(ids):
        spec = dict(event_id=event_id, symbol='AAPL' if i < 3 else 'MOCK', fiscal_year=2030,
                    release_date='2030-10-01', available_from='2030-10-02',
                    identity={'security':'AAPL'}, periods={'current': {'period_end':'2030-09-28'}},
                    source={'path':'synthetic.pdf', 'sha256':'a'*64}, inherited_rejection_codes=[],
                    cash_quality_admission={'admitted': not 3 <= i < 16, 'reasons':[]},
                    locator_inventory={'format':'pdf', 'candidates':[loc()], 'gaps':[]})
        event = deepcopy(spec); event.pop('locator_inventory')
        event.update(fact_admitted=False, model_admitted=False, strategy_admitted=False,
                     stock_compensation=dict(value=None, gaps=[dict(reason='invalid_or_ambiguous_current_monetary_cell', error_type='ValueError')],
                                             fact_admitted=False, model_admitted=False, strategy_admitted=False))
        specs.append(spec); events.append(event)
    return specs, events


class PartitionTests(unittest.TestCase):
    def test_exact_legacy_columns_unchanged(self):
        cells, proof = m.pypdf_columns('2,345 6,789', 2)
        self.assertEqual(cells, ['2,345','6,789']); self.assertFalse(proof['whitespace_rejoined'])

    def test_space_after_comma_and_between_digits(self):
        cells, proof = m.pypdf_columns('2,       3     4   5       6,789', 2)
        self.assertEqual(cells, ['2,345','6,789']); self.assertEqual(proof['column_token_counts'], [4,1])

    def test_both_columns_fragmented(self):
        self.assertEqual(m.pypdf_columns('2,3 4 5      6, 7 8 9', 2)[0], ['2,345','6,789'])

    def test_only_comparison_fragmented(self):
        self.assertEqual(m.pypdf_columns('12,345       6,7 8 9', 2)[0], ['12,345','6,789'])

    def test_unique_partition_independent_of_gap_width(self):
        self.assertEqual(m.pypdf_columns('2,            3 4 5 6,789', 2)[0], ['2,345','6,789'])

    def test_ambiguous_digit_partitions_rejected(self):
        with self.assertRaisesRegex(ValueError,'unique_syntax_only'): m.pypdf_columns('1 2 3',2)

    def test_no_valid_partition_rejected(self):
        with self.assertRaisesRegex(ValueError,'unique_syntax_only'): m.pypdf_columns('1,2 34,56',2)

    def test_extra_valid_column_is_ambiguous_or_invalid(self):
        with self.assertRaises(ValueError): m.pypdf_columns('2,345 6,789 9,876',2)

    def test_zero_and_negative_values_supported(self):
        self.assertEqual(m.pypdf_columns('0 -123',2)[0], ['0','-123'])
        self.assertEqual(m.pypdf_columns('- 2,3 4 5 (6,789)',2)[0], ['-2,345','(6,789)'])

    def test_parentheses_preserved(self):
        self.assertEqual(m.pypdf_columns('(2, 3 4 5) 0',2)[0], ['(2,345)','0'])

    def test_decimal_preserved(self):
        self.assertEqual(m.pypdf_columns('2,3 4 5.60 0',2)[0], ['2,345.60','0'])

    def test_currency_handling_matches_frozen_helper(self):
        self.assertEqual(m.pypdf_columns('/dollarsign 2,345 $ 6,789',2)[0], ['2,345','6,789'])

    def test_no_new_symbol_normalization(self):
        for tail in ('2,345 EUR6,789','2,345 −6,789','2,345 6,789%','2,345 6,789*','2,345 !6,789'):
            with self.subTest(tail=tail), self.assertRaises(ValueError): m.pypdf_columns(tail,2)

    def test_broken_parentheses_and_double_negative_rejected(self):
        for tail in ('(2,345 6,789','(-2,345) 6,789','2,345 6,789)'):
            with self.subTest(tail=tail), self.assertRaises(ValueError): m.pypdf_columns(tail,2)

    def test_columns_fixed_at_two(self):
        for n in (0,1,3,True,2.0):
            with self.subTest(n=n), self.assertRaises(ValueError): m.pypdf_columns('2,345 6,789',n)

    def test_token_bound_and_empty(self):
        for tail in ('', '2345', ' '.join(['1']*65)):
            with self.subTest(tail=tail), self.assertRaises(ValueError): m.pypdf_columns(tail,2)


class GeometryTests(unittest.TestCase):
    def test_fragmented_text_passes_full_frozen_geometry(self):
        out = extract()
        self.assertEqual(out['value_decimal'],'2345')
        self.assertTrue(out['pypdf_column_reconstruction']['whitespace_rejoined'])
        self.assertTrue(out['pdfium_geometry']['numeric_requires_consecutive_source_character_indices'])

    def test_legacy_result_matches_all_shared_fields(self):
        tp = TextPage(); tp.chars = page(('2,345','6,789'),label=LABEL).chars
        with patch.object(m.base,'configure_pdf',return_value=(None,PdfModule(tp))):
            old = m.pdf.extract_cell(b'%PDF-synthetic',[LABEL+' 2,345 6,789'],loc(),{})
            new = m.extract_cell(b'%PDF-synthetic',[LABEL+' 2,345 6,789'],loc(),{})
        for key in new:
            if key != 'pypdf_column_reconstruction': self.assertEqual(new[key],old[key])

    def test_wrong_amount_not_repaired_from_pdfium(self):
        with self.assertRaisesRegex(ValueError,'disagree'): extract(tail='2, 3 4 6 6,789')

    def test_swapped_columns_rejected(self):
        with self.assertRaisesRegex(ValueError,'disagree'): extract(values=('6,789','2,345'))

    def test_identical_duplicate_rows_rejected(self):
        p = page(('2,345','6,789'),label=LABEL).add(LABEL,20,370).add('2,345',380,370).add('6,789',480,370)
        with self.assertRaisesRegex(ValueError,'unique_exact'): extract(textpage=p)

    def test_different_duplicate_rows_rejected(self):
        p = page(('2,345','6,789'),label=LABEL).add(LABEL,20,370).add('9,876',380,370).add('5,432',480,370)
        with self.assertRaisesRegex(ValueError,'unique_exact'): extract(textpage=p)

    def test_malformed_competing_financial_row_rejected(self):
        p = page(('2,345','6,789'),label=LABEL).add(LABEL,20,370).add('2,345bad',380,370).add('6,789',480,370)
        with self.assertRaisesRegex(ValueError,'malformed_competing'): extract(textpage=p)

    def test_large_shift_crossing_column_rejected(self):
        p = page(('2,345','6,789'),label=LABEL)
        p.chars = [(c,(b[0]-50,b[1],b[2]-50,b[3])) if b[1]==400 and b[0]>=480 else (c,b) for c,b in p.chars]
        with self.assertRaisesRegex(ValueError,'boundary'): extract(textpage=p)

    def test_nonconsecutive_pdfium_indices_stay_rejected(self):
        p = page(('2,3 45','6,789'),label=LABEL)
        with self.assertRaises(ValueError): extract(textpage=p)

    def test_unknown_glyph_stays_rejected(self):
        p = page(('2,345','6,789'),label=LABEL).add('@',550,400)
        with self.assertRaises(ValueError): extract(textpage=p)

    def test_large_overlap_stays_rejected(self):
        p = page(('2,345','6,789'),label=LABEL)
        p.chars = [(c,(b[0],b[1],b[2]+4,b[3])) if b[1]==400 and b[0]==380 else (c,b) for c,b in p.chars]
        with self.assertRaises(ValueError): extract(textpage=p)

    def test_small_kerning_still_passes(self):
        p = page(('2,345','6,789'),label=LABEL)
        p.chars = [(c,(b[0],b[1],385.1359,b[3])) if b[1]==400 and b[0]==380 else (c,b) for c,b in p.chars]
        self.assertEqual(extract(textpage=p)['value_decimal'],'2345')

    def test_wrong_date_header_rejected(self):
        location=loc(); location['header_literals'][0]='September 27, September 30,'
        with self.assertRaisesRegex(ValueError,'date_header'): extract(location=location)

    def test_ambiguous_pypdf_never_uses_pdfium_to_select_partition(self):
        with patch.object(m.base,'configure_pdf',side_effect=AssertionError('Do not consult geometry')):
            with self.assertRaisesRegex(ValueError,'unique_syntax_only'):
                m.extract_cell(b'%PDF-synthetic',[LABEL+' 1 2 3'],loc(),{})

    def test_comparison_selection_and_non_sbc_labels_rejected(self):
        for key,value in (('financial_column_index_0based',1),('row_label','Net income'),('format','html')):
            location=loc(); location[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): extract(location=location)


class ResultTests(unittest.TestCase):
    def inv(self):
        return dict(format='pdf',source_sha256='a'*64,candidates=[loc()],gaps=[],accounting_review_flags=[{'code':'retained'}])

    def run_result(self, inv=None, error=None):
        inv=inv or self.inv()
        parsed=extract()
        with patch.object(m.old,'inventory',return_value=deepcopy(inv)), patch.object(m.base,'pdf_texts',return_value=[]), \
             patch.object(m,'extract_cell',side_effect=error,return_value=parsed):
            return m.extract_current(b'%PDF-synthetic',{'current':{'period_end':'2030-09-28'}},inv,{})

    def test_full_result_stays_candidate_only(self):
        out=self.run_result()
        self.assertTrue(out['candidate_only'])
        for key in ('fact_admitted','model_admitted','strategy_admitted','first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'):
            self.assertIs(out[key],False)
        self.assertIsNone(out['value']['currency_iso'])
        self.assertIs(out['value']['fact_admitted'],False)
        self.assertEqual(out['accounting_review_flags'],[{'code':'retained'}])

    def test_comparison_text_and_raw_row_not_emitted(self):
        value=self.run_result()['value']
        self.assertNotIn('row_text',value)
        self.assertTrue(all('text' not in cell for cell in value['pdfium_geometry']['pdfium_all_row_cells']))
        self.assertEqual(value['pdfium_geometry']['selected_cell']['text'],'2,345')

    def test_failure_preserves_explicit_diagnostic(self):
        out=self.run_result(error=ValueError('unique_syntax_only_pypdf_column_partition_required'))
        self.assertIsNone(out['value'])
        self.assertEqual(out['gaps'][0]['error_message'],'unique_syntax_only_pypdf_column_partition_required')

    def test_frozen_inventory_mismatch_rejected(self):
        inv=self.inv()
        with patch.object(m.old,'inventory',return_value=dict(inv,source_sha256='b'*64)):
            with self.assertRaisesRegex(ValueError,'inventory changed'): m.extract_current(b'%PDF-synthetic',{},inv,{})

    def test_existing_gap_blocks_extraction(self):
        inv=self.inv(); inv['gaps']=[{'reason':'missing'}]
        with patch.object(m.old,'inventory',return_value=inv), patch.object(m,'extract_cell',side_effect=AssertionError('Do not extract')):
            out=m.extract_current(b'%PDF-synthetic',{},inv,{})
        self.assertIsNone(out['value']); self.assertEqual(out['gaps'],inv['gaps'])

    def test_html_not_accepted(self):
        with self.assertRaisesRegex(ValueError,'original_pdf'): m.extract_current(b'<html>',{},self.inv(),{})


class RunnerContractTests(unittest.TestCase):
    def test_only_three_declared_targets_selected(self):
        specs,events=synthetic_events(); saved=deepcopy((specs,events))
        out=r.validate_targets(specs,events)
        self.assertEqual([x['event_id'] for x in out],list(r.TARGETS)); self.assertEqual((specs,events),saved)

    def test_duplicate_or_missing_event_rejected(self):
        for kind in ('duplicate','missing'):
            specs,events=synthetic_events()
            if kind=='duplicate': specs[-1]=specs[-2]
            else: events.pop()
            with self.subTest(kind=kind),self.assertRaises(ValueError): r.validate_targets(specs,events)

    def test_old_nonnull_value_cannot_be_replaced(self):
        specs,events=synthetic_events();events[0]['stock_compensation']['value']={'value_decimal':'2345'}
        with self.assertRaisesRegex(ValueError,'null failures'): r.validate_targets(specs,events)

    def test_wrong_gap_cannot_be_patched(self):
        specs,events=synthetic_events();events[0]['stock_compensation']['gaps'][0]['reason']='different_failure'
        with self.assertRaisesRegex(ValueError,'null failures'): r.validate_targets(specs,events)

    def test_target_metadata_cannot_change(self):
        for key,value in (('available_from','2030-10-03'),('source',{'path':'other'}),('periods',{}),('inherited_rejection_codes',['held'])):
            specs,events=synthetic_events();events[0][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'metadata changed'): r.validate_targets(specs,events)

    def test_thirteen_holds_preserved(self):
        specs,events=synthetic_events();specs[-1]['cash_quality_admission']['admitted']=False
        with self.assertRaisesRegex(ValueError,'Original13'): r.validate_targets(specs,events)

    def test_admission_flags_cannot_change(self):
        for key in ('fact_admitted','model_admitted','strategy_admitted'):
            specs,events=synthetic_events();events[0][key]=True
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'No inherited'): r.validate_targets(specs,events)

    def test_missing_plan_hash_fails_before_read(self):
        with patch.object(r.base,'read',side_effect=AssertionError('Must reject before read')):
            with self.assertRaises(ValueError): r.load_plan(Path('/synthetic'),None)

    def test_wrong_stage_or_policy_rejected(self):
        plan={'stage':'draft','runtime':sys.version,'policy':r.policy()}
        with patch.object(r.base,'read',return_value=plan):
            with self.assertRaisesRegex(ValueError,'Frozen patch'): r.load_plan(Path('/synthetic'),'a'*64)

    def test_source_policy_disallows_financial_calculations(self):
        policy=r.policy();self.assertEqual(policy['event_ids'],list(r.TARGETS));self.assertEqual(policy['maximum_new_cells'],3)
        self.assertIs(policy['financial_calculations_permitted'],False);self.assertIs(policy['strategy_registration_permitted'],False)

    def test_publication_is_exclusive(self):
        with tempfile.TemporaryDirectory() as root:
            r.base.publish(root,'a.json',{'a':1})
            with self.assertRaises(FileExistsError): r.base.publish(root,'a.json',{'a':2})
            self.assertEqual(r.base.read(root,'a.json'),{'a':1})

    def test_freeze_cannot_replace_existing_plan(self):
        with tempfile.TemporaryDirectory() as root:
            r.base.publish(root,r.OUT+'/PLAN.json',{'frozen':True})
            with patch.object(r,'verified_inputs',side_effect=AssertionError('Do not read sources')):
                with self.assertRaisesRegex(ValueError,'already frozen'): r.freeze(Path(root),'review.json')

    def test_build_cannot_replace_existing_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            r.base.publish(root,r.OUT+'/PATCH.json',{'frozen':True})
            with patch.object(r,'load_plan',return_value={}),patch.object(r,'products',side_effect=AssertionError('Do not extract')):
                with self.assertRaisesRegex(ValueError,'write-once'): r.build(Path(root),'a'*64)

    def test_failed_synthetic_receipt_rejected(self):
        with patch.object(r.base,'read',return_value={'passed':False}):
            with self.assertRaisesRegex(ValueError,'synthetic receipt'): r.verify_tests(Path('/synthetic'))


class PhaseIntegrationTests(unittest.TestCase):
    def setup_sources(self, root):
        raw=b'%PDF-synthetic'
        (root/'synthetic.pdf').write_bytes(raw)
        specs,_=synthetic_events();specs=specs[:3]
        for spec in specs:
            spec['source']['sha256']=r.safe.digest(raw)
        return specs

    def freeze_context(self, root, *, valid_review=True):
        specs=self.setup_sources(root)
        (root/'work').mkdir()
        (root/r.DIAGNOSTIC_CODE).write_text('synthetic diagnostic\n')
        diagnostic=dict(plan_sha256=r.V1_PLAN_SHA,manifest_sha256=r.V1_MANIFEST_SHA,
                        new_values_published=0,new_parser_executed=False,
                        diagnostic_code_sha256=r.base.sha(root,r.DIAGNOSTIC_CODE),
                        findings=[dict(event_id=e,frozen_extractor_error=dict(type='ValueError',message='exact_pypdf_column_count_required')) for e in r.TARGETS])
        r.base.publish(root,r.DIAGNOSTIC,diagnostic)
        code={r.DIAGNOSTIC_CODE:r.base.sha(root,r.DIAGNOSTIC_CODE)}
        review=dict(passed=True,source_plan_sha256=r.V1_PLAN_SHA,
                    source_code_hashes=code if valid_review else {'wrong':'b'*64},real_source_cells_extracted=0)
        r.base.publish(root,'review.json',review)
        original=dict(pdf_runtime={'synthetic':True},pdf_runtime_hashes={'synthetic':'a'*64})
        return specs,code,original

    def test_freeze_only_inventories_and_never_calls_new_extractor(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);specs,code,original=self.freeze_context(root)
            with patch.object(r,'verified_inputs',return_value=(original,specs,{})), \
                 patch.object(r,'verify_tests',return_value={}),patch.object(m,'code_hashes',return_value=code), \
                 patch.object(m.old,'inventory',return_value=specs[0]['locator_inventory']) as inventory, \
                 patch.object(m,'extract_current',side_effect=AssertionError('No real extraction before PLAN')):
                result=r.freeze(root,'review.json')
            self.assertEqual(result['real_source_cells_extracted'],0)
            self.assertEqual(inventory.call_count,3)
            plan=r.base.read(root,r.OUT+'/PLAN.json',result['plan_sha256'])
            self.assertEqual(plan['events'],specs)
            self.assertEqual(plan['source_code_hashes'],code)

    def test_freeze_requires_review_of_exact_final_code(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);specs,code,original=self.freeze_context(root,valid_review=False)
            with patch.object(r,'verified_inputs',return_value=(original,specs,{})), \
                 patch.object(r,'verify_tests',return_value={}),patch.object(m,'code_hashes',return_value=code), \
                 patch.object(m.old,'inventory',side_effect=AssertionError('No inventory after rejected review')):
                with self.assertRaisesRegex(ValueError,'Independent pre-build'):r.freeze(root,'review.json')
            self.assertFalse((root/r.OUT/'PLAN.json').exists())

    def value_result(self,spec):
        return dict(value=dict(value_decimal='2345',source_sha256=spec['source']['sha256'],
                               period=deepcopy(spec['periods']['current']),fact_admitted=False),
                    gaps=[],fact_admitted=False,model_admitted=False,strategy_admitted=False)

    def test_patch_contains_only_targets_and_preserves_value_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);specs=self.setup_sources(root)
            result=self.value_result(specs[0])
            with patch.object(m,'extract_current',return_value=result):
                out=r.products(root,{'events':specs},'b'*64)
            self.assertEqual([x['event_id'] for x in out['events']],list(r.TARGETS))
            self.assertEqual(out['counts']['extracted_new_cells'],3)
            for row,spec in zip(out['events'],specs):
                self.assertEqual(row['result']['value']['source'],spec['source'])
                self.assertEqual(row['result']['value']['available_from'],spec['available_from'])
            self.assertTrue(all(out['counts'][k]==0 for k in r.ZERO_COUNTS))

    def test_partial_patch_keeps_gap_not_assumed_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);specs=self.setup_sources(root)
            result=dict(value=None,gaps=[{'reason':'ambiguous'}],fact_admitted=False,model_admitted=False,strategy_admitted=False)
            with patch.object(m,'extract_current',return_value=result):
                out=r.products(root,{'events':specs},'b'*64)
            self.assertEqual(out['counts']['extracted_new_cells'],0);self.assertEqual(out['counts']['gap_cells'],3)

    def test_build_verify_exact_reproduction_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);specs=self.setup_sources(root)
            refs={'synthetic.pdf':r.base.sha(root,'synthetic.pdf')}
            plan=dict(events=specs,source_hashes=refs,source_code_hashes=refs)
            with patch.object(r,'load_plan',return_value=plan),patch.object(m,'extract_current',return_value=self.value_result(specs[0])):
                saved=r.build(root,'b'*64)
                self.assertEqual(saved,r.build(root,'b'*64,verify=True))
                path=root/r.OUT/'PATCH.json';path.write_text(path.read_text()+'\n')
                with self.assertRaises(ValueError):r.build(root,'b'*64,verify=True)


if __name__ == '__main__':
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    log=stream.getvalue();print(log,end='')
    if '--write-report' in sys.argv:
        r.require(not r.safe.local(r.ROOT,r.OUT+'/PLAN.json').exists(),'Cannot replace tests after freeze')
        log_path='work/stock_comp_pdf_repair_v2_tests.log'
        path=r.safe.local(r.ROOT,log_path)
        with path.open('x') as target: target.write(log)
        report=dict(passed=result.wasSuccessful(),synthetic_only=True,tests_run=result.testsRun,
                    errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),
                    real_source_cells_extracted=0,runtime=sys.version,created_utc=datetime.now(timezone.utc).isoformat(),
                    source_code_hashes=m.code_hashes(r.ROOT),log={'path':log_path,'sha256':r.base.sha(r.ROOT,log_path)})
        r.base.publish(r.ROOT,r.TESTS,report)
    raise SystemExit(not result.wasSuccessful())

"""Synthetic metadata integration only; stop before any prepared/result phase.

Uses the pinned stdlib Evidence implementation with fabricated PLAN/receipts and
seven source files. No actual round264 artifacts, prices, arrays or outcomes.
"""
import ast
import copy
import math
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import monthly_gain_survival264_saved_reader as legacy
import monthly_gain_survival264_saved_reader_v4 as frozen_v4
import monthly_gain_survival264_saved_reader_v5 as reader

BASE_SHA='e1cb3f142c068762eff25593f1ee9b85e3ab60f08369d1a284cfbf62c7b50cdf'
MAPPING_OLD="path=ROOT/('work/monthly_gain_survival264_'+name+'.json')"
MAPPING_NEW="path=ROOT/('work/monthly_gain_survival264_'+{'TESTS':'TESTS_V2','CODE_REVIEW':'CODE_REVIEW'}[name]+'.json')"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PhaseBoundary(Exception):
    pass


class RegisteredEvidenceIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='gain264-reader-v5-',dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.out=self.root/'outputs/monthly_gain_survival264'
        self.work=self.root/'work';self.work.mkdir();self.out.mkdir(parents=True)
        self.parent=reader.load_parent()  # Pinned stdlib source only, no data read.
        for relative in ('work/monthly_weak_tree261_saved_reader.py',
                         'work/monthly_tracking_risk262_saved_reader_v2.py',
                         'outputs/monthly_factor_search_v2/validate.py'):
            target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(reader.ROOT/relative,target)
        for module in (reader,legacy):
            shutil.copyfile(module.__file__,self.work/Path(module.__file__).name)
        self.code={}
        for name in ('host','test_host','kernel','test_kernel','independent','test_independent','sources'):
            file=self.work/('synthetic_'+name+'.py');file.write_text('# synthetic '+name+'\n')
            self.code[str(file.relative_to(self.root))]=sha(file)
        self.tests_path=self.work/'monthly_gain_survival264_TESTS_V2.json'
        self.old_path=self.work/'monthly_gain_survival264_TESTS.json'
        self.review_path=self.work/'monthly_gain_survival264_CODE_REVIEW.json'
        self.context_path=self.work/'monthly_gain_survival264_context.json'
        self.contract_path=self.work/'monthly_gain_survival264_IMPLEMENTATION_CONTRACT.json'
        self.runtime=dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3')
        self.context=dict(frozen=True,arms=list(reader.ARMS))
        self.write(self.context_path,self.context)
        self.write(self.contract_path,dict(frozen=True,context=dict(sha256=sha(self.context_path))))
        self.tests=dict(passed=True,synthetic_only=True,tests=141,runtime=self.runtime,code_sha256=self.code)
        self.review=dict(passed=True,unresolved_blockers=[],actual_returns_computed=False,code_sha256=self.code)
        self.write(self.tests_path,self.tests);self.write(self.review_path,self.review)
        self.write(self.old_path,dict(passed=True,tests=139,code_sha256={k:'0'*64 for k in self.code}))
        self.plan=dict(context=self.context,context_sha256=sha(self.context_path),
            implementation_contract_sha256=sha(self.contract_path),runtime=self.runtime,
            input_sha256={str(p.relative_to(self.root)):sha(p) for p in (self.tests_path,self.review_path)})
        self.instances=[];self.phase_calls=[]
        owner=self
        class Evidence(self.parent.Evidence):
            def __init__(self):
                super().__init__();owner.instances.append(self)
            def phase(self,*args):
                owner.phase_calls.append(args)
                raise PhaseBoundary('No phase artifacts are read in this synthetic test')
        self.evidence_type=Evidence

    def write(self,path,value):
        path.write_text(json.dumps(value,sort_keys=True,allow_nan=False))

    def refresh(self,path,value):
        self.write(path,value);self.plan['input_sha256'][str(path.relative_to(self.root))]=sha(path)

    def run_prefix(self,module=reader):
        self.write(self.out/'PLAN.json',self.plan)
        args=SimpleNamespace(plan_sha256=sha(self.out/'PLAN.json'),prepared_sha256='1'*64,
            frozen_sha256='2'*64,results_sha256='3'*64,terminals_sha256='4'*64)
        with patch.object(self.parent,'ROOT',self.root),patch.object(self.parent,'Evidence',self.evidence_type),\
             patch.object(module,'ROOT',self.root),patch.object(module,'OUT',self.out),\
             patch.object(module,'PARENT',self.work/'monthly_weak_tree261_saved_reader.py'),\
             patch.object(module,'__file__',str(self.work/Path(module.__file__).name)),\
             patch.object(module,'load_parent',return_value=self.parent):
            module.run(args)

    def test_v5_preserves_every_v4_byte_except_authorized_additions(self):
        base=Path(frozen_v4.__file__).read_text()
        self.assertEqual(hashlib.sha256(base.encode()).hexdigest(),'432add7a13d18cb2625a8aa343554f91e9577822862c7551c33dbd24a72e0fcd')
        source=Path(reader.__file__).read_text()
        tree=ast.parse(source)
        node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='diagnostic_summary_parity')
        helper=ast.get_source_segment(source,node)
        self.assertEqual(source.count(helper),1)
        source=source.replace(helper+'\n\n','',1)
        added="""    prior_cost_name=ARMS[0]+'_next_open_cost_HOLDINGS.json'
    prior_cost_holdings=ev.read(prior_dir/'results'/prior_cost_name,prior_result['artifacts_sha256'][prior_cost_name])
    summary_fee_display_review=diagnostic_summary_parity(results['arms'][ARMS[0]],prior['arms'][ARMS[0]],exports[ARMS[0],'next_open_cost']['HOLDINGS'],prior_cost_holdings)"""
        original="    need(results['arms'][ARMS[0]]==prior['arms'][ARMS[0]],'Original67.67 all saved path summary objects unchanged')"
        self.assertEqual(source.count(added),1)
        source=source.replace(added,original).replace("schema='monthly_gain_survival264_root_saved_review_v5'","schema='monthly_gain_survival264_root_saved_review_v4'")
        source=source.replace("original_incumbent_summary_metrics_exact=summary_fee_display_review['exact'],economic_summary_metrics_exact=True,summary_fee_display_review=summary_fee_display_review,",'original_incumbent_summary_metrics_exact=True,')
        self.assertEqual(source,base)

    def test_legacy_path_absent_from_plan_reproduces_original_failure(self):
        self.assertNotIn(str(self.old_path.relative_to(self.root)),self.plan['input_sha256'])
        with self.assertRaisesRegex(ValueError,'Missing/conflicting registered pin: .*TESTS.json'):
            self.run_prefix(legacy)
        self.assertEqual(self.phase_calls,[])

    def test_registered_v2_and_all_seven_current_source_pins_reach_phase_boundary(self):
        with self.assertRaises(PhaseBoundary):self.run_prefix()
        self.assertEqual(len(self.phase_calls),1)
        self.assertEqual(self.phase_calls[0][0],self.out/'prepared')
        pins=self.instances[-1].pins
        self.assertEqual(pins[str(self.tests_path.relative_to(self.root))],sha(self.tests_path))
        self.assertEqual(pins[str(self.review_path.relative_to(self.root))],sha(self.review_path))
        for path,digest in self.code.items():self.assertEqual(pins[path],digest)
        self.assertNotIn(str(self.old_path.relative_to(self.root)),pins)

    def test_registered_legacy_receipt_cannot_replace_missing_v2_pin(self):
        self.plan['input_sha256'].pop(str(self.tests_path.relative_to(self.root)))
        self.plan['input_sha256'][str(self.old_path.relative_to(self.root))]=sha(self.old_path)
        with self.assertRaisesRegex(ValueError,'Missing/conflicting registered pin: .*TESTS_V2.json'):
            self.run_prefix()
        self.assertEqual(self.phase_calls,[])

    def test_registered_v2_digest_mismatch_rejected_before_phase(self):
        self.plan['input_sha256'][str(self.tests_path.relative_to(self.root))]='0'*64
        with self.assertRaisesRegex(ValueError,'Digest mismatch: .*TESTS_V2.json'):self.run_prefix()
        self.assertEqual(self.phase_calls,[])

    def test_each_current_source_pin_is_enforced(self):
        for relative in self.code:
            path=self.root/relative;before=path.read_bytes();path.write_bytes(before+b'# changed\n')
            try:
                with self.subTest(path=relative),self.assertRaisesRegex(ValueError,'Digest mismatch: .*synthetic_'):
                    self.run_prefix()
            finally:path.write_bytes(before)
        self.assertEqual(self.phase_calls,[])

    def test_old_source_digest_in_registered_v2_cannot_pass(self):
        stale=dict(self.tests,code_sha256=dict(self.code));stale['code_sha256'][next(iter(self.code))]='0'*64
        self.refresh(self.tests_path,stale)
        with self.assertRaisesRegex(ValueError,'Digest mismatch: .*synthetic_'):self.run_prefix()
        self.assertEqual(self.phase_calls,[])

    def test_synthetic_and_runtime_guards_unchanged(self):
        for changes in (dict(synthetic_only=False),dict(runtime=dict(self.runtime,python='wrong'))):
            self.refresh(self.tests_path,dict(self.tests,**changes))
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'Formal synthetic/runtime evidence'):
                self.run_prefix()
        self.assertEqual(self.phase_calls,[])

    def test_code_review_guards_unchanged(self):
        for changes in (dict(passed=False),dict(unresolved_blockers=['blocked']),dict(actual_returns_computed=True)):
            self.refresh(self.review_path,dict(self.review,**changes))
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'Formal CODE_REVIEW|Prospective code review'):
                self.run_prefix()
        self.assertEqual(self.phase_calls,[])


def holdings(fees=None):
    fees=[1.]+[0.]*121 if fees is None else fees
    start=date(2001,1,1)
    return [dict(signal_date=(start+timedelta(days=31*i)).isoformat(),
        execution_date=(start+timedelta(days=31*i+1)).isoformat(),symbols=['A','B','C'],scores=[1.,.9,.8],fee=fee)
        for i,fee in enumerate(fees)]


def summaries(rows):
    total=0.
    for row in rows:total+=row['fee']
    result={}
    for path in reader.PATHS:
        fee=total if path=='next_open_cost' else 0.
        metrics=dict(fees_paid=fee,cost_per_side=.0005 if path=='next_open_cost' else 0.,
            cagr=.15,sharpe=.8,total_return=2.,max_drawdown=-.3,final_equity=300000.,
            dollar_turnover=1000.,trade_rows=30,execution='open' if path!='same_close_zero' else 'close')
        slices={name:dict(metrics) for name in ('full122','first88','early53','recent35','later34')}
        for name in slices:
            if name!='full122':slices[name]['fees_paid']=0.
        result[path]=dict(metrics=metrics,slices=slices,accounting=dict(exported_fill_ledger_matches=True,terminal_liquidation=False))
    return result


class DiagnosticSummaryParity(unittest.TestCase):
    def setUp(self):
        self.previous_rows=holdings();self.current_rows=copy.deepcopy(self.previous_rows)
        self.current_rows[0]['fee']=math.nextafter(1.,math.inf)
        self.previous=summaries(self.previous_rows);self.current=summaries(self.current_rows)

    def review(self):
        return reader.diagnostic_summary_parity(self.current,self.previous,self.current_rows,self.previous_rows)

    def test_lowbit_fee_only_accepted_with_honest_exact_flag(self):
        review=self.review()
        self.assertFalse(review['exact']);self.assertTrue(review['economic_summary_metrics_exact'])
        self.assertTrue(review['own_totals_exact']);self.assertTrue(review['full122_duplicate_exact'])
        self.assertFalse(review['aggregate_tolerance_used'])
        self.assertEqual(len(review['display_fee_review']['differences']),1)
        self.assertEqual(review['allowed_difference_paths'],['next_open_cost.metrics.fees_paid','next_open_cost.slices.full122.fees_paid'])

    def test_identical_summary_and_fees_are_exact(self):
        self.current=copy.deepcopy(self.previous);self.current_rows=copy.deepcopy(self.previous_rows)
        self.assertTrue(self.review()['exact'])

    def test_identical_totals_can_coexist_with_lowbit_display_difference(self):
        self.previous_rows=holdings([1.,1.]+[0.]*120);self.current_rows=copy.deepcopy(self.previous_rows)
        self.current_rows[0]['fee']=math.nextafter(1.,math.inf)
        self.previous=summaries(self.previous_rows);self.current=summaries(self.current_rows)
        review=self.review()
        self.assertTrue(review['exact']);self.assertFalse(review['display_fee_review']['exact'])

    def test_falsified_own_sum_rejected_even_one_ulp(self):
        for side in ('current','previous'):
            saved=copy.deepcopy(getattr(self,side));bad=copy.deepcopy(saved)
            fee=bad['next_open_cost']['metrics']['fees_paid'];fee=math.nextafter(fee,math.inf)
            bad['next_open_cost']['metrics']['fees_paid']=fee;bad['next_open_cost']['slices']['full122']['fees_paid']=fee
            setattr(self,side,bad)
            with self.subTest(side=side),self.assertRaisesRegex(ValueError,'Exact own chronological'):self.review()
            setattr(self,side,saved)

    def test_full122_duplicate_must_equal_own_sum(self):
        for side in ('current','previous'):
            saved=copy.deepcopy(getattr(self,side));getattr(self,side)['next_open_cost']['slices']['full122']['fees_paid']=0.
            with self.subTest(side=side),self.assertRaisesRegex(ValueError,'Exact own chronological'):self.review()
            setattr(self,side,saved)

    def test_all_nonfee_summary_changes_rejected(self):
        mutations=[('same_close_zero','metrics','fees_paid'),('next_open_zero','metrics','cagr'),
                   ('next_open_cost','metrics','cagr'),('next_open_cost','metrics','dollar_turnover'),
                   ('next_open_cost','slices','first88','fees_paid'),('next_open_cost','slices','full122','cagr')]
        for path in mutations:
            saved=copy.deepcopy(self.current);target=self.current
            for key in path[:-1]:target=target[key]
            target[path[-1]]=math.nextafter(target[path[-1]],math.inf)
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'Every non-display-fee'):self.review()
            self.current=saved

    def test_path_keys_and_fee_leaf_presence_strict(self):
        for change in ('rename','extra','missing_metrics_fee','missing_duplicate'):
            saved=copy.deepcopy(self.current)
            if change=='rename':self.current['renamed']=self.current.pop('same_close_zero')
            elif change=='extra':self.current['extra']=copy.deepcopy(self.current['same_close_zero'])
            elif change=='missing_metrics_fee':del self.current['next_open_cost']['metrics']['fees_paid']
            else:del self.current['next_open_cost']['slices']['full122']['fees_paid']
            with self.subTest(change=change),self.assertRaises(ValueError):self.review()
            self.current=saved

    def test_fee_rate_not_tolerated_even_both_changed(self):
        for rate in (0.,math.nextafter(.0005,math.inf),True,float('nan'),float('inf'),'0.0005'):
            for side in ('current','previous'):getattr(self,side)['next_open_cost']['metrics']['cost_per_side']=rate
            with self.subTest(rate=rate),self.assertRaises(ValueError):self.review()

    def test_bad_summary_fee_numbers_rejected(self):
        for bad in (True,False,None,'1',-1.,float('nan'),float('inf'),float('-inf')):
            for side in ('current','previous'):
                saved=copy.deepcopy(getattr(self,side))
                for container in ('metrics','full122'):
                    item=getattr(self,side)['next_open_cost'];item=item['metrics'] if container=='metrics' else item['slices']['full122']
                    item['fees_paid']=bad
                    with self.subTest(bad=bad,side=side,container=container),self.assertRaises(ValueError):self.review()
                    setattr(self,side,copy.deepcopy(saved))

    def test_bad_holdings_fee_numbers_and_sum_overflow_rejected(self):
        for bad in (True,None,'1',-1.,float('nan'),float('inf')):
            saved=self.current_rows[0]['fee'];self.current_rows[0]['fee']=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):self.review()
            self.current_rows[0]['fee']=saved
        huge=float.fromhex('0x1.fffffffffffffp+1023')
        self.previous_rows=holdings([huge,huge]);self.current_rows=copy.deepcopy(self.previous_rows)
        with self.assertRaisesRegex(ValueError,'Finite chronological display fee total'):self.review()

    def test_dates_order_and_nonfee_holdings_are_exact(self):
        for field,value in (('signal_date','2000-12-31'),('execution_date','2001-01-03'),('symbols',['X']),('scores',[1.,1.,1.])):
            saved=copy.deepcopy(self.current_rows);self.current_rows[0][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Non-fee holdings'):self.review()
            self.current_rows=saved
        self.current_rows[0],self.current_rows[1]=self.current_rows[1],self.current_rows[0]
        with self.assertRaisesRegex(ValueError,'Non-fee holdings'):self.review()

    def test_both_reordered_duplicate_or_malformed_dates_rejected(self):
        for change in ('reordered','duplicate','malformed','execution_before_signal'):
            saved=(copy.deepcopy(self.current_rows),copy.deepcopy(self.previous_rows))
            for rows in (self.current_rows,self.previous_rows):
                if change=='reordered':rows[0],rows[1]=rows[1],rows[0]
                elif change=='duplicate':rows[1]['execution_date']=rows[0]['execution_date']
                elif change=='malformed':rows[0]['signal_date']='2001-1-1'
                else:rows[0]['execution_date']='2000-12-31'
            with self.subTest(change=change),self.assertRaises(ValueError):self.review()
            self.current_rows,self.previous_rows=saved

    def test_existing_perperiod_bound_and_zero_rules_unchanged(self):
        for fee in (0.,2.,1.+2**-45):
            self.current_rows[0]['fee']=fee;self.current=summaries(self.current_rows)
            with self.subTest(fee=fee),self.assertRaisesRegex(ValueError,'Zero display|binary64 bound'):self.review()

    def test_chronological_plus_equal_not_fsum(self):
        self.previous_rows=holdings([2.**53,1.,1.]);self.current_rows=copy.deepcopy(self.previous_rows)
        self.previous=summaries(self.previous_rows);self.current=copy.deepcopy(self.previous)
        review=self.review();self.assertEqual(review['own_chronological_fee_totals']['current'],2.**53)
        self.assertNotEqual(math.fsum(row['fee'] for row in self.current_rows),2.**53)
        for side in ('current','previous'):
            getattr(self,side)['next_open_cost']['metrics']['fees_paid']=2.**53+2.
            getattr(self,side)['next_open_cost']['slices']['full122']['fees_paid']=2.**53+2.
        with self.assertRaisesRegex(ValueError,'Exact own chronological'):self.review()


def load_tests(loader, standard_tests, pattern):
    # Preserve the registered reader's 21 synthetic checks, against V5 only.
    # The frozen test file remains byte-for-byte unchanged.
    import test_monthly_gain_survival264_saved_reader as previous_tests
    previous_tests.reader=reader
    standard_tests.addTests(loader.loadTestsFromModule(previous_tests))
    return standard_tests


if __name__=='__main__':unittest.main()

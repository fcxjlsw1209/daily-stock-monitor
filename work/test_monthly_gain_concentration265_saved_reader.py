"""Synthetic metadata integration only; stop before any prepared/result phase.

Uses the pinned stdlib Evidence implementation with fabricated PLAN/receipts and
twelve source files. No actual round265 artifacts, prices, arrays or outcomes.
"""
import ast
import copy
import math
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from copy import deepcopy
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import monthly_gain_survival264_saved_reader_v5 as frozen_v5
import monthly_gain_concentration265_saved_reader as reader


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PhaseBoundary(Exception):
    pass


class RegisteredEvidenceIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='gain265-reader-',dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.out=self.root/'outputs/monthly_gain_concentration265'
        self.work=self.root/'work';self.work.mkdir();self.out.mkdir(parents=True)
        self.parent=reader.load_parent()  # Pinned stdlib source only, no data read.
        for relative in ('work/monthly_weak_tree261_saved_reader.py',
                         'work/monthly_tracking_risk262_saved_reader_v2.py',
                         'outputs/monthly_factor_search_v2/validate.py'):
            target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(reader.ROOT/relative,target)
        for module in (reader,):
            shutil.copyfile(module.__file__,self.work/Path(module.__file__).name)
        self.code={}
        for name in reader.CODE_FILES:
            file=self.work/name;file.write_text('# synthetic '+name+'\n')
            self.code[str(file.relative_to(self.root))]=sha(file)
        self.tests_path=self.work/'monthly_gain_concentration265_TESTS.json'
        self.old_path=self.work/'monthly_gain_concentration265_TESTS_V2.json'
        self.review_path=self.work/'monthly_gain_concentration265_CODE_REVIEW.json'
        self.context_path=self.work/'monthly_gain_concentration265_context.json'
        self.contract_path=self.work/'monthly_gain_concentration265_IMPLEMENTATION_CONTRACT.json'
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

    def test_fee_helpers_match_v5_exact_source(self):
        base=ast.parse(Path(frozen_v5.__file__).read_text());current=ast.parse(Path(reader.__file__).read_text())
        for name in ('diagnostic_holdings_parity','diagnostic_summary_parity'):
            old=next(x for x in base.body if isinstance(x,ast.FunctionDef) and x.name==name)
            new=next(x for x in current.body if isinstance(x,ast.FunctionDef) and x.name==name)
            self.assertEqual(ast.dump(old),ast.dump(new))

    def test_registered_tests_and_all_twelve_code_pins_reach_phase_boundary(self):
        with self.assertRaises(PhaseBoundary):self.run_prefix()
        self.assertEqual(len(self.phase_calls),1)
        self.assertEqual(self.phase_calls[0][0],self.out/'prepared')
        pins=self.instances[-1].pins
        self.assertEqual(pins[str(self.tests_path.relative_to(self.root))],sha(self.tests_path))
        self.assertEqual(pins[str(self.review_path.relative_to(self.root))],sha(self.review_path))
        for path,digest in self.code.items():self.assertEqual(pins[path],digest)
        self.assertNotIn(str(self.old_path.relative_to(self.root)),pins)

    def test_unregistered_v2_receipt_cannot_replace_missing_tests_pin(self):
        self.plan['input_sha256'].pop(str(self.tests_path.relative_to(self.root)))
        self.plan['input_sha256'][str(self.old_path.relative_to(self.root))]=sha(self.old_path)
        with self.assertRaisesRegex(ValueError,'Missing/conflicting registered pin: .*TESTS.json'):
            self.run_prefix()
        self.assertEqual(self.phase_calls,[])

    def test_registered_tests_digest_mismatch_rejected_before_phase(self):
        self.plan['input_sha256'][str(self.tests_path.relative_to(self.root))]='0'*64
        with self.assertRaisesRegex(ValueError,'Digest mismatch: .*TESTS.json'):self.run_prefix()
        self.assertEqual(self.phase_calls,[])

    def test_each_current_source_pin_is_enforced(self):
        for relative in self.code:
            path=self.root/relative;before=path.read_bytes();path.write_bytes(before+b'# changed\n')
            try:
                with self.subTest(path=relative),self.assertRaisesRegex(ValueError,'Digest mismatch: .*265'):
                    self.run_prefix()
            finally:path.write_bytes(before)
        self.assertEqual(self.phase_calls,[])

    def test_stale_code_digest_cannot_pass(self):
        stale=dict(self.tests,code_sha256=dict(self.code));stale['code_sha256'][next(iter(self.code))]='0'*64
        self.refresh(self.tests_path,stale)
        with self.assertRaisesRegex(ValueError,'Digest mismatch: .*265'):self.run_prefix()
        self.assertEqual(self.phase_calls,[])

    def test_missing_extra_or_alias_code_pin_rejected(self):
        for mode in ('missing','extra','alias','wrong_name'):
            saved=deepcopy(self.tests);changed=deepcopy(saved);pins=changed['code_sha256']
            key=next(iter(pins));digest=pins[key]
            if mode=='missing':del pins[key]
            elif mode=='extra':pins['work/unregistered_extra.py']=digest
            elif mode=='alias':pins[str(self.root/key)]=digest
            else:pins['work/wrong265.py']=pins.pop(key)
            self.refresh(self.tests_path,changed)
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'twelve formal code'):self.run_prefix()
            self.refresh(self.tests_path,saved)
        self.assertEqual(self.phase_calls,[])

    def test_absolute_code_paths_supported_and_bound(self):
        changed=deepcopy(self.tests);changed['code_sha256']={str(self.root/key):value for key,value in self.code.items()}
        self.refresh(self.tests_path,changed)
        with self.assertRaises(PhaseBoundary):self.run_prefix()
        self.assertEqual(len(self.phase_calls),1)

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



def saved_evidence():
    """Fabricated receipts only: no host, independent module or market input."""
    width=2
    def counts(n):
        h=max(n-1,0)*width
        c=dict.fromkeys(reader.COUNT_FIELDS,0)
        c.update(events=n,symbols_per_event=width,original_episode_calls=3,eligible_cells=n*width,
            initial_partial_events=int(n>0),active_events=max(n-1,0),original_core_cells=h,
            original_finite_cells=h,required_cells=h,price_observations=253*h,gain_return_observations=252*h,
            gain_statistics=h,concentration_evaluations=2*h,quality_normalization_evaluations=2*h,
            quality_float_conversions=2*h,positive_gain_days=h,positive_raw_multiplications=2*h,
            available_events=max(n-1,0),daily_ranked_cells=h,block_ranked_cells=h)
        return c
    def proof(n,kind):
        c=counts(n);formal=max(n-1,0);rows=300+n if formal else 0
        result=dict.fromkeys(reader.WORK_FIELDS,0)
        result.update(passed=True,checked_events=n,producer_modules_imported=False,computes_returns=False)
        if kind=='original':
            events=[]
            for q in range(n):
                e=dict(query_index=q,date='2016-08-17' if q==0 else 'synthetic'+str(q),initial_partial=q==0,calm=False,u0_indices=[] if q==0 else [0,1])
                for key in ('original_raw','original_rank','frozen_core','rebuilt_core','core_abs_error','original_sigma63'):e[key]=None if q==0 else [1.,1.]
                events.append(e)
            result.update(original_events=events,u0_indices=[e['u0_indices'] for e in events],original_checked_cells=width*formal,
                original_source_cells=width*formal,original_source_price_observations=sum((301+q)*width for q in range(1,n)))
        elif kind=='incumbent':result['original_episode_calls']=1
        else:
            full=kind in ('source','full');selecting=kind in ('selection','full')
            # Keep the fixture arithmetic separate from the reader's work helper.
            result.update(full_source_slot_inspections=c['price_observations'] if full else 0,cached_slot_validations=0 if full else c['price_observations'],return_evaluations=c['gain_return_observations'],statistic_evaluations=c['gain_statistics'],concentration_evaluations=c['concentration_evaluations'],quality_normalization_evaluations=c['quality_normalization_evaluations'],quality_float_conversions=c['quality_float_conversions'],rank_evaluations=2*c['available_events'] if selecting else 0,scalar_rank_evaluations=2*c['required_cells'] if selecting else 0,original_episode_calls=3 if selecting else 0)
            for k in ('positive_gain_days','positive_raw_multiplications'):result[k]=c[k]
            if kind in ('source','cached_source'):result.update(source_rebuilt=full,feature_counts=c)
            else:result['counts']=c
        if kind in ('original','source','full'):result.update(original_source_snapshot_sha256=reader.canonical_digest(['synthetic snapshot',n]),original_source_snapshot_price_observations=rows*width,original_source_snapshot_hashes=int(formal>0))
        return result
    def item(n,cached=False):
        c=counts(n);original=proof(122 if cached else n,'original');cert_hash=reader.canonical_digest(original)
        x=dict(prefix=n,counts=c,exact=True,gates=dict(incumbent=proof(n,'incumbent'),source=proof(n,'cached_source' if cached else 'source')))
        x['selection' if cached else 'full']=proof(n,'selection' if cached else 'full')
        if cached:x['original_source_certificate_sha256']=cert_hash
        else:
            x.update(original_source=original,original_source_anchor=dict(schema='monthly_gain_concentration265_original_source_anchor_v1',checked_events=n,certificate_sha256=cert_hash,original_event_sha256=[reader.canonical_digest(e) for e in original['original_events']]),producer_original_source_work=dict(fullpast_rows=300+n if n>1 else 0,union_symbols=['A','B'] if n>1 else [],original_source_price_observations=width*(300+n) if n>1 else 0,original_source_cells=width*max(n-1,0)),reused_full=False,source_recomputed=True)
            for r in (x['gates']['source'],x['full']):r.update(original_source_proof_sha256=cert_hash,original_source_proof_events=n)
        x['consumed_original_source_sha256']=reader.canonical_digest(dict(schema='monthly_gain_concentration265_original_source_prefix_v1',checked_events=n,original_events=original['original_events'][:n],u0_indices=original['u0_indices'][:n]));return x
    main=item(122);physical=[item(n) for n in (0,1,53,88)]+[dict(deepcopy(main),reused_full=True,source_recomputed=False)];cached=[item(n,True) for n in range(123)]
    e=dict(main=main,physical_prefixes=physical,cached_prefixes=cached,full_original_source_certificate_sha256=reader.canonical_digest(main['original_source']),full_source_certificate_sha256=reader.canonical_digest(main['gates']['source']))
    actual=[main]+physical[:-1];producer=dict.fromkeys(reader.WORK_FIELDS,0);independent=dict.fromkeys(reader.WORK_FIELDS,0)
    for it in actual+cached:
        n=it['prefix'];full='full' in it;p=proof(n,'full' if full else 'selection')
        if full:
            p['original_source_price_observations']=it['producer_original_source_work']['original_source_price_observations'];p['original_source_cells']=it['counts']['original_core_cells']
            p['original_source_snapshot_price_observations']=p['original_source_snapshot_hashes']=0
        for k in reader.WORK_FIELDS:producer[k]+=p[k]
        receipts=[it['gates']['incumbent'],it['gates']['source'],it['full' if full else 'selection']]+([it['original_source']] if full else [])
        for r in receipts:
            for k in reader.WORK_FIELDS:independent[k]+=r[k]
    total={k:producer[k]+independent[k] for k in reader.WORK_FIELDS}
    e['execution_accounting']=dict(producer=producer,independent=independent,total=total,physical_preparations=5,cached_decisions=123,original_source_gate_calls=5,original_core_checked_cells=sum(it['counts']['original_core_cells'] for it in actual),return_vector_steps_computed=total['return_evaluations'],new_model_fits=0,predictive_model_fits=0,physical_122_reused=True,source_and_cached_statistic_recomputation_disclosed=True)
    rank_full=sum(x['counts']['daily_ranked_cells']+x['counts']['block_ranked_cells'] for x in actual)
    rank_cached=sum(x['counts']['daily_ranked_cells']+x['counts']['block_ranked_cells'] for x in cached)
    e['execution_accounting']['rank_work']={owner:dict(full_scalar_rank_evaluations=rank_full,cached_scalar_rank_evaluations=rank_cached,scalar_rank_evaluations=values['scalar_rank_evaluations']) for owner,values in (('producer',producer),('independent',independent))}
    e['execution_accounting']['original_certificate_metadata_work']=dict(full_certificate_validations=5,full_certificate_hashes=5,full_event_hashes=sum(x['prefix'] for x in actual),prefix_projection_validations=128,prefix_event_hashes=sum(x['prefix'] for x in actual+cached),prefix_projection_hashes=128,cached_full_certificate_validations=0,cached_full_certificate_hashes=0,cached_event_hashes=sum(x['prefix'] for x in cached),cached_projection_hashes=123,scope='Synthetic saved metadata; no numeric source access.')
    return e


class SavedSourceEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.evidence=saved_evidence()
    def check(self,e=None):return reader.source_evidence_review(self.evidence if e is None else e,width=2)
    def test_original_raw_full_snapshot_counts_are_separate(self):
        r=self.check();self.assertTrue(r['passed']);self.assertEqual(r['independent_numeric_snapshot_passes'],9)
        self.assertEqual(r['independent_source_snapshot_observations'],3*2*sum(300+n for n in (122,53,88)))
        self.assertEqual(r['cached_source_price_reads'],0);self.assertFalse(r['source_snapshot_sha_recomputed_by_reader'])
        self.assertGreater(r['independent_original_reconstruction_observations'],r['independent_source_snapshot_observations'])
    def test_copied_json_receipts_are_supported(self):
        import json
        self.check(json.loads(json.dumps(self.evidence)))
    def test_stable_host_synthetic_receipt_schema_compatible(self):
        from test_monthly_gain_concentration265_host import prepared_fixture
        evidence,*_=prepared_fixture()
        result=reader.source_evidence_review(json.loads(json.dumps(evidence)))
        self.assertTrue(result['passed']);self.assertEqual(result['independent_numeric_snapshot_passes'],12)
        self.assertTrue(result['rank_and_certificate_metadata_work_checked'])

    def test_reject_original_raw_full_snapshot_mismatch(self):
        for label in ('original_source','source','full'):
            e=deepcopy(self.evidence);it=e['physical_prefixes'][2];p=it['gates']['source'] if label=='source' else it[label]
            p['original_source_snapshot_sha256']='f'*64
            with self.subTest(label=label),self.assertRaises(ValueError):self.check(e)
    def test_no_zero_or_bool_numeric_snapshot_count_on_formal_prefix(self):
        for field,value in (('original_source_snapshot_price_observations',0),('original_source_snapshot_hashes',0),('original_source_snapshot_hashes',True)):
            e=deepcopy(self.evidence);e['physical_prefixes'][2]['full'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.check(e)
    def test_n0_and_partial_have_no_numeric_snapshot_pass(self):
        for i in (0,1):
            self.assertEqual(self.evidence['physical_prefixes'][i]['full']['original_source_snapshot_hashes'],0)
            e=deepcopy(self.evidence);e['physical_prefixes'][i]['full']['original_source_snapshot_hashes']=1
            with self.assertRaises(ValueError):self.check(e)
    def test_cached_and_incumbent_cannot_claim_source_reads(self):
        for location in ('cached','incumbent'):
            e=deepcopy(self.evidence);p=e['cached_prefixes'][4]['selection'] if location=='cached' else e['main']['gates']['incumbent']
            p['original_source_snapshot_price_observations']=1
            with self.subTest(location=location),self.assertRaises(ValueError):self.check(e)
    def test_raw_full_do_not_repeat_pandas_reconstruction(self):
        for key in ('original_source_price_observations','original_source_cells'):
            e=deepcopy(self.evidence);e['physical_prefixes'][2]['full'][key]=1
            with self.assertRaises(ValueError):self.check(e)
    def test_physical_union_size_and_original_binding_checked(self):
        for mode in ('union','proof','prefix','missing'):
            e=deepcopy(self.evidence);it=e['physical_prefixes'][2]
            if mode=='union':it['producer_original_source_work']['union_symbols'].append('C')
            elif mode=='proof':it['full']['original_source_proof_sha256']='f'*64
            elif mode=='prefix':it['consumed_original_source_sha256']='f'*64
            else:it['full'].pop('original_source_snapshot_hashes')
            with self.subTest(mode=mode),self.assertRaises((ValueError,KeyError)):self.check(e)
    def test_full122_reuse_is_not_a_sixth_physical_execution(self):
        e=deepcopy(self.evidence);e['physical_prefixes'][-1]['reused_full']=False
        with self.assertRaises(ValueError):self.check(e)
        e=deepcopy(self.evidence);e['execution_accounting']['physical_preparations']=6
        with self.assertRaises(ValueError):self.check(e)
    def test_owner_totals_cannot_omit_snapshot_or_doublecount_reuse(self):
        for owner in ('producer','independent','total'):
            e=deepcopy(self.evidence);e['execution_accounting'][owner]['original_source_snapshot_price_observations']+=1
            with self.subTest(owner=owner),self.assertRaises(ValueError):self.check(e)
    def test_cached_original_certificate_or_count_tampering(self):
        for mode in ('certificate','counts','snapshot'):
            e=deepcopy(self.evidence);it=e['cached_prefixes'][53]
            if mode=='certificate':it['original_source_certificate_sha256']='f'*64
            elif mode=='counts':it['counts']['price_observations']-=1
            else:it['gates']['source']['original_source_snapshot_hashes']=1
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.check(e)
    def test_reader_has_no_host_producer_or_independent_import(self):
        tree=ast.parse(Path(reader.__file__).read_text());names=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):names.extend(a.name for a in n.names)
            elif isinstance(n,ast.ImportFrom):names.append(n.module or '')
        self.assertFalse(any('265_host' in n or '265_kernel' in n or '265_independent' in n for n in names))

class SavedSelection(unittest.TestCase):
    def setUp(self):
        self.p=reader.load_parent()
        def metrics(c):
            m=dict(cagr=c,sharpe=1.,max_drawdown=-.2,daily_return_sd=.01,sharpe_status='defined',sharpe_reason=None)
            return {p:{g:dict(m) for g in ('training','early','recent')} for p in reader.PATHS[:2]}
        self.m={a:metrics(.2 if a!=reader.ARMS[2] else .21) for a in reader.ARMS}
        self.o={a:[dict(symbols=['A','B','C']) for _ in range(122)] for a in reader.ARMS}
        for i in (2,4,60):self.o[reader.ARMS[2]][i]=dict(symbols=['A','B','D'])
    def pick(self):return reader.selection(self.p,self.m,self.o)
    def test_candidate_only_when_both_references_pass(self):
        self.assertTrue(self.pick()['admitted']);self.assertEqual(self.pick()['primary'],reader.ARMS[2])
        self.m[reader.ARMS[1]]['same_close_zero']['training']['cagr']=.3
        self.assertFalse(self.pick()['admitted']);self.assertEqual(self.pick()['primary'],reader.ARMS[0])
    def test_open_zero_must_pass(self):
        self.m[reader.ARMS[2]]['next_open_zero']['training']['cagr']=.19
        self.assertFalse(self.pick()['admitted'])
    def test_risk_guard(self):
        self.m[reader.ARMS[2]]['same_close_zero']['training']['sharpe']=.949
        self.assertFalse(self.pick()['admitted'])
    def test_recent_era_required(self):
        self.m[reader.ARMS[2]]['same_close_zero']['recent']['cagr']=.174
        self.assertFalse(self.pick()['admitted'])
    def test_permutations_do_not_count(self):
        for i in (2,4,60):self.o[reader.ARMS[2]][i]=dict(symbols=['C','A','B'])
        self.assertFalse(self.pick()['admitted'])
    def test_later_only_cannot_rescue(self):
        for i in (2,4,60):self.o[reader.ARMS[2]][i]=dict(symbols=['A','B','C'])
        for i in range(88,122):self.o[reader.ARMS[2]][i]=dict(symbols=['A','B','D'])
        self.assertFalse(self.pick()['admitted'])
    def test_undefined_candidate_sharpe_is_inadmissible(self):
        self.m[reader.ARMS[2]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        self.assertFalse(self.pick()['admitted'])
    def test_cost_diagnostic_not_selection_input(self):
        self.m[reader.ARMS[2]]['next_open_cost']={}
        with self.assertRaises(ValueError):self.pick()



class ConcentrationReceipts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.evidence=saved_evidence()

    def test_declared_schema_has_no_inherited_survival_work(self):
        self.assertEqual(len(reader.WORK_FIELDS),17);self.assertEqual(len(reader.COUNT_FIELDS),37)
        for name in reader.WORK_FIELDS+reader.COUNT_FIELDS:
            self.assertNotIn('survival',name);self.assertNotIn('endpoint',name);self.assertNotIn('ratio_evaluations',name)
        self.assertEqual(len(reader.ZERO_SOURCE_COUNTS),7)

    def test_each_new_work_counter_is_bound(self):
        for field in ('concentration_evaluations','quality_normalization_evaluations','quality_float_conversions','positive_gain_days','positive_raw_multiplications'):
            for where in ('source','full'):
                evidence=deepcopy(self.evidence);item=evidence['physical_prefixes'][2]
                receipt=item['gates']['source'] if where=='source' else item['full'];receipt[field]+=1
                with self.subTest(field=field,where=where),self.assertRaises(ValueError):reader.source_evidence_review(evidence,width=2)

    def test_new_counts_are_not_renamed_legacy_ratios(self):
        for change in ('extra','missing','bool','false_zero_float','false_positive_day'):
            c=deepcopy(self.evidence['main']['counts'])
            if change=='extra':c['quality_ratio_evaluations']=0
            elif change=='missing':del c['quality_float_conversions']
            elif change=='bool':c['concentration_evaluations']=True
            elif change=='false_zero_float':c['quality_float_conversions']=0
            else:c['positive_gain_days']=0
            with self.subTest(change=change),self.assertRaises(ValueError):reader.counts_review(c,122,2)

    def test_neutral_and_nonpositive_still_count_quality_conversions(self):
        c=deepcopy(self.evidence['main']['counts']);h=c['required_cells']
        c.update(no_positive_gain_cells=h,positive_gain_days=0,concentration_evaluations=0,
            quality_normalization_evaluations=0,nonpositive_original_cells=h,positive_raw_multiplications=0)
        reader.counts_review(c,122,2)
        work=reader._new_work(c,True,True)
        self.assertEqual(work['quality_float_conversions'],2*h);self.assertEqual(work['statistic_evaluations'],h)
        self.assertEqual(work['concentration_evaluations'],0);self.assertEqual(work['positive_raw_multiplications'],0)
        c['daily_zero_quality_cells']=1
        with self.assertRaises(ValueError):reader.counts_review(c,122,2)

    def test_raw_source_cannot_claim_unchanged_target_ranks(self):
        for key in ('daily_rank_unchanged_events','block_rank_unchanged_events'):
            evidence=deepcopy(self.evidence);item=evidence['physical_prefixes'][2]
            # The fixture aliases c in its records; explicitly separate raw receipt.
            item['gates']['source']['feature_counts']=deepcopy(item['counts'])
            item['gates']['source']['feature_counts'][key]=1
            with self.subTest(key=key),self.assertRaises(ValueError):reader.source_evidence_review(evidence,width=2)

    def test_rank_and_certificate_work_never_silent_or_bool(self):
        for mode in ('rank','rankbool','metadata','metadatabool'):
            evidence=deepcopy(self.evidence);account=evidence['execution_accounting']
            if mode.startswith('rank'):
                item=account['rank_work']['producer'];key='full_scalar_rank_evaluations'
            else:item=account['original_certificate_metadata_work'];key='cached_full_certificate_hashes'
            item[key]=True if mode.endswith('bool') else item[key]+1
            with self.subTest(mode=mode),self.assertRaises(ValueError):reader.source_evidence_review(evidence,width=2)

    def test_reader_imports_only_stdlib_and_no_numerical_execution_modules(self):
        tree=ast.parse(Path(reader.__file__).read_text());names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names.extend(x.name.split('.')[0] for x in node.names)
            elif isinstance(node,ast.ImportFrom):names.append((node.module or '').split('.')[0])
        self.assertEqual(set(names),{'argparse','datetime','hashlib','importlib','json','math','fractions','pathlib'})

if __name__=='__main__':unittest.main(verbosity=2)

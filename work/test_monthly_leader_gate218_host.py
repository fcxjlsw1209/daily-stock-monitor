"""Round218 synthetic host orchestration, source gates and saved-evidence checks."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import importlib.util
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np

P=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('tested_leader218_host',P/'monthly_leader_gate218_host.py')
h=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(h)
import monthly_leader_gate218_model as model

def fixture():
    shape=(4,8)
    a=dict(baseline=np.tile(np.arange(8.),(4,1)),core=np.tile(np.arange(8.,0.,-1.),(4,1)),
           core_weight=np.full((4,1),.5),eligible=np.ones(shape,bool))
    a['core'][1]=0. # all exact boundary ties included
    a['eligible'][2,2:]=False;a['eligible'][3,1:]=False
    meta=dict(dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31'],
              ends=['2016-08-31','2016-09-30','2016-10-31','2016-11-30'],
              symbols=[list('ABCDEFGH') for _ in range(4)],etfs=['H'])
    return a,meta

class FakeStudy:
    def __init__(self):
        rp=h.module('leader218_original_selector',h.ROOT/'outputs/monthly_return_priority/return_priority.py')
        self.rp=rp
        self.choices=Mock(side_effect=self.choose)
        self.adapter=SimpleNamespace(score_and_choose=Mock(side_effect=AssertionError('New-arm episode must not run')))
    @staticmethod
    def configs():return [dict(id=i) for i in range(5)]
    def choose(self,spec,a,meta,n,policy):
        assert spec==dict(id=4) and policy=={'original_policy':True}
        score=-a['baseline']
        return self.rp.choose(score,a['eligible'],meta['symbols'],retain=3),score

def selection_fixture():
    metric={era:dict(cagr=.5,sharpe=1.5,max_drawdown=-.3) for era in ('training','early','recent')}
    metrics={name:{path:copy.deepcopy(metric) for path in ('same_close_zero','next_open_cost')} for name in h.NAMES}
    orders={name:[dict(symbols=['A','B','C']) for _ in range(122)] for name in h.NAMES}
    for path in metrics[h.NAMES[1]]:metrics[h.NAMES[1]][path]['training']['cagr']+=.01
    for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['A','B','D']
    return metrics,orders

def receipt(built):
    return dict(**dict.fromkeys(h.PREPARATION_FLAGS,True),counts=copy.deepcopy(built['audit']['counts']),
        orders_checked={k:len(v) for k,v in built['orders'].items()},computes_returns=False,producer_modules_imported=False,
        **{k:0 for k in ('new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits','new_parameters')})

class WiringTests(unittest.TestCase):
    def test_pair_orders_remain_producer_outputs_and_incumbent_full_choices(self):
        a,m=fixture();built=model.build(a,m,4);study=FakeStudy()
        with patch.object(h,'read',return_value={'original_policy':True}):orders=h.decisions(study,a,m,built,4)
        self.assertEqual(set(orders),set(h.NAMES));study.choices.assert_called_once()
        self.assertEqual(study.choices.call_args.args[0],dict(id=4))
        for arm in h.NAMES[1:]:self.assertEqual(orders[arm],built['orders'][arm])
        self.assertNotEqual(orders[h.NAMES[1]][0],orders[h.NAMES[0]][0])
        study.adapter.score_and_choose.assert_not_called()
    def test_baseline_original_selector_and_historical_full_order_parity(self):
        a,m=fixture();built=model.build(a,m,4);study=FakeStudy()
        old=copy.deepcopy(built['orders'][h.NAMES[2]])
        with patch.object(h,'sha',return_value='historical_digest'):
            proof=h.baseline_parity(study,a,m,built,4,old)
        self.assertTrue(proof['original_selector_orders_exact']);self.assertTrue(proof['historical_original65_orders_exact'])
        self.assertEqual(proof['checked_months'],4)
        for mutation in ('symbols','scores','signal_date'):
            bad=copy.deepcopy(old)
            if mutation=='scores':bad[0]['scores'][0]+=1e-15
            elif mutation=='symbols':bad[0]['symbols']=list(reversed(bad[0]['symbols']))
            else:bad[0]['signal_date']='wrong'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):h.baseline_parity(study,a,m,built,4,bad)
    def test_baseline_cannot_silently_use_incumbent_or_wrong_rank(self):
        a,m=fixture();built=model.build(a,m,4);built['orders'][h.NAMES[2]]=copy.deepcopy(built['orders'][h.NAMES[1]])
        with self.assertRaises(ValueError):h.baseline_parity(FakeStudy(),a,m,built,4)
    def test_all_physical_cache_metadata_prefixes_and_input_immutability(self):
        a,m=fixture()
        for i in range(23):a['unused_'+str(i)]=np.zeros((4,2))
        self.assertEqual(len(a),27);before=copy.deepcopy((a,m));full=model.build(a,m,4)
        with patch.object(h,'read',return_value={'original_policy':True}):
            orders=h.decisions(FakeStudy(),a,m,full,4)
            for n in range(5):
                aa,mm=h.prefix_inputs(a,m,n);self.assertEqual(len(aa),27)
                self.assertTrue(all(x.shape[0]==n for x in aa.values()))
                self.assertEqual(len(mm['dates']),n);self.assertEqual(len(mm['ends']),n);self.assertEqual(len(mm['symbols']),n)
                tiny=model.build(aa,mm,n);got=h.decisions(FakeStudy(),aa,mm,tiny,n)
                for name in h.NAMES:self.assertEqual(got[name],orders[name][:n])
                np.testing.assert_array_equal(tiny['shortlist'],full['shortlist'][:n])
        self.assertEqual(m,before[1])
        for key in a:np.testing.assert_array_equal(a[key],before[0][key])
    def test_future_poison_removed_before_model_and_selection(self):
        a,m=fixture();full=model.build(a,m,4)
        for n in range(5):
            aa,mm=copy.deepcopy(a),copy.deepcopy(m)
            for k in ('baseline','core','core_weight'):aa[k][n:]=np.nan
            aa['eligible']=aa['eligible'].astype(object);aa['eligible'][n:]='future poison'
            for k in ('dates','ends','symbols'):mm[k][n:]=[object()]*(4-n)
            tiny,meta=h.prefix_inputs(aa,mm,n);result=model.build(tiny,meta,n)
            np.testing.assert_array_equal(result['shortlist'],full['shortlist'][:n])
            for arm in h.NAMES[1:]:self.assertEqual(result['orders'][arm],full['orders'][arm][:n])
    def test_prefix_rejects_bad_axes_or_consumed_nonboolean(self):
        for mutation in ('n','array','metadata','mask'):
            a,m=fixture();n=2
            if mutation=='n':n=True
            elif mutation=='array':a['core']=np.zeros((3,8))
            elif mutation=='metadata':m['ends']=m['ends'][:-1]
            else:a['eligible']=a['eligible'].astype(object);a['eligible'][0,0]=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):h.prefix_inputs(a,m,n)
    def test_initial_partial_new_arms_use_same_formula_and_two_cash(self):
        a,m=fixture();b=model.build(a,m,4)
        self.assertEqual(b['orders'][h.NAMES[1]][0]['symbols'],['F','E','D'])
        self.assertEqual(b['orders'][h.NAMES[2]][0]['symbols'],['H','G','F'])
        for arm in h.NAMES[1:]:
            self.assertEqual(len(b['orders'][arm][2]['symbols']),2);self.assertEqual(b['orders'][arm][3]['symbols'],[])
    def test_independent_receipt_exact_counts_flags_and_no_fits(self):
        a,m=fixture();b=model.build(a,m,4);r=receipt(b);h.require_preparation_audit(r,b,4,8)
        for key in h.PREPARATION_FLAGS:
            bad=copy.deepcopy(r);bad.pop(key)
            with self.subTest(flag=key),self.assertRaises(ValueError):h.require_preparation_audit(bad,b,4,8)
        for key in h.COUNT_FIELDS:
            bad=copy.deepcopy(b);check=copy.deepcopy(r);bad['audit']['counts'][key]+=1;check['counts'][key]+=1
            with self.subTest(count=key),self.assertRaises(ValueError):h.require_preparation_audit(check,bad,4,8)
        for key in ('new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits','new_parameters'):
            bad=copy.deepcopy(b);check=copy.deepcopy(r);bad['audit'][key]=check[key]=1
            with self.subTest(fit=key),self.assertRaises(ValueError):h.require_preparation_audit(check,bad,4,8)
    def test_independent_scope_and_order_coverage_required(self):
        a,m=fixture();b=model.build(a,m,4);r=receipt(b)
        for key,value in [('computes_returns',True),('producer_modules_imported',True),('orders_checked',{h.NAMES[1]:3,h.NAMES[2]:4})]:
            bad=copy.deepcopy(r);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_preparation_audit(bad,b,4,8)
        bad=copy.deepcopy(b);bad['shortlist']=bad['shortlist'].astype(int)
        with self.assertRaises(ValueError):h.require_preparation_audit(r,bad,4,8)
    def test_audit_row_symbols_and_mask_counts_cannot_drift(self):
        a,m=fixture();b=model.build(a,m,4);r=receipt(b)
        for change in ('row','symbols','count','date'):
            bad=copy.deepcopy(b)
            if change=='row':bad['audit']['records'][0]['row']=99
            elif change=='symbols':bad['audit']['records'][0]['leader6_symbols']=['A']
            elif change=='count':bad['audit']['records'][0]['shortlist_count']+=1
            else:bad['audit']['records'][0]['date']='wrong'
            with self.subTest(change=change),self.assertRaises(ValueError):h.require_preparation_audit(r,bad,4,8)
    def test_source_scope_flags_rejected_before_frozen_cache_reads(self):
        review=dict(passed=True,context_sha256=h.CONTEXT_SHA,evidence_sha256={},input_sha256={'work/fake':'digest'},**dict.fromkeys(h.SOURCE_FLAGS,True))
        def reader(path):
            if path==h.SOURCE_REVIEW:return review
            if path==h.CONTEXT:return {'evidence_sha256':{}}
            raise AssertionError('Source reads preceded rejection')
        with patch.object(h,'sha',return_value=h.SOURCE_REVIEW_SHA),patch.object(h,'read',side_effect=reader):
            for key in ('passed',*h.SOURCE_FLAGS):
                old=review[key];review[key]=False
                with self.subTest(key=key),self.assertRaises(ValueError):h.cache_source_binding({})
                review[key]=old
    def test_production_has_no_training_or_forecast_npz_load(self):
        text=Path(h.__file__).read_text();tree=ast.parse(text)
        fn={n.name:ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertNotIn('TRAIN_INPUTS.npz',text);self.assertNotIn('FORECASTS.npz',text)
        self.assertNotIn('study.load()',fn['prepare']);self.assertIn("ev.load_arrays(LOCAL/'DATA.npz')",fn['prepare'])
        self.assertLess(fn['prepare'].index('gate(expected)'),fn['prepare'].index('model.build'))
    def test_current_counters_write_once_and_protection_order(self):
        rows=[dict(trials=1,directory='old') for _ in range(215)];rows[0]['trials']=948;rows[-1]['directory']=h.PRIOR.name
        state=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'));h.state_check(state)
        state['completed'][-1]['directory']='wrong'
        with self.assertRaises(ValueError):h.state_check(state)
        text=Path(h.__file__).read_text();tree=ast.parse(text);fn={n.name:ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertIn("len(state['completed']) == 216",fn['close']);self.assertIn('== 1165',fn['close'])
        for marker in ('cache_source_binding(raw)','code_review_binding()','len(protected)==21444'):
            self.assertLess(fn['register'].index(marker),fn['register'].index('OUT.mkdir()'))
        self.assertIn('require_result_manifest(',fn['close']);self.assertEqual(fn['commit_state'].count('CENTRAL.read_bytes() == before'),2)
    def test_actual_independent_preparation_interface_with_synthetic_arrays(self):
        import monthly_leader_gate218_independent as independent
        a,m=fixture()
        for n in range(5):
            aa,mm=h.prefix_inputs(a,m,n);built=model.build(aa,mm,n)
            proof=independent.audit_preparation(aa,mm,n,built);h.require_preparation_audit(proof,built,n,8)

class BindingAndSelectionTests(unittest.TestCase):

    def test_bind_adds_or_accepts_exact_digest_and_rejects_conflict(self):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'synthetic-source'; path.write_bytes(b'synthetic only\n')
                pins = {}; h.bind(pins, path)
                self.assertEqual(pins, {str(path):h.sha(path)})
                h.bind(pins, path)
                old = dict(pins); path.write_bytes(b'changed synthetic\n')
                with self.assertRaisesRegex(ValueError, 'Conflicting inherited'):
                    h.bind(pins, path)
                self.assertEqual(pins, old)

    def test_source_review_and_context_evidence_can_be_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            for name in ('shared','review_only','context_only'):(root/name).write_text(name)
            review={'evidence_sha256':{name:h.sha(root/name) for name in ('shared','review_only')}}
            context={'evidence_sha256':{name:h.sha(root/name) for name in ('shared','context_only')}}
            self.assertNotEqual(review['evidence_sha256'],context['evidence_sha256'])
            pins={}
            with patch.object(h,'ROOT',root):h.bind_source_evidence(pins,review,context)
            self.assertEqual(set(pins),{str(root/name) for name in ('shared','review_only','context_only')})
            with patch.object(h,'ROOT',root):h.verify(pins)

    def test_source_evidence_shared_conflict_and_missing_file_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();p=root/'shared';p.write_text('unchanged synthetic source')
            review={'evidence_sha256':{'shared':h.sha(p)}}
            context={'evidence_sha256':{'shared':'0'*64}}
            with patch.object(h,'ROOT',root):
                with self.assertRaises(ValueError):h.bind_source_evidence({},review,context)
                context={'evidence_sha256':{'missing':'0'*64}}
                with self.assertRaises((ValueError,FileNotFoundError)):h.bind_source_evidence({},review,context)

    def test_selection_ignores_later_orders_and_extra_report_only_metrics(self):
            metrics, orders = selection_fixture()
            result = h.select_first88(metrics, orders)
            self.assertEqual(result['primary'], h.NAMES[1])
            for name in h.NAMES:
                metrics[name]['later34'] = dict(cagr=1000. if name != h.NAMES[1] else -1.)
                metrics[name]['full122'] = dict(cagr=1000. if name != h.NAMES[1] else -1.)
                for i in range(88, 122):
                    orders[name][i]['symbols'] = [name+str(i)]
            self.assertEqual(h.select_first88(metrics, orders), result)

    def test_baseline_control_cannot_be_promoted_and_candidate_clears_both_references(self):
            metrics, orders = selection_fixture()
            metrics[h.NAMES[2]]['same_close_zero']['training']['cagr'] = 2.
            result = h.select_first88(metrics, orders)
            self.assertEqual(result['primary'], h.NAMES[0])
            self.assertFalse(result['guards']['close_vs_baseline_control'])
            metrics, orders = selection_fixture()
            metrics[h.NAMES[2]]['next_open_cost']['training']['cagr'] = 2.
            result = h.select_first88(metrics, orders)
            self.assertEqual(result['primary'], h.NAMES[0])
            self.assertFalse(result['guards']['open_vs_baseline_control'])

    def test_first88_changed_set_support_cannot_come_from_later34(self):
            metrics, orders = selection_fixture()
            for i in (1, 2, 53):
                orders[h.NAMES[1]][i]['symbols'] = ['C','B','A']
            for i in (90, 100, 110):
                orders[h.NAMES[1]][i]['symbols'] = ['A','B','D']
            result = h.select_first88(metrics, orders)
            self.assertEqual(result['changed_months'], [])
            self.assertEqual(result['primary'], h.NAMES[0])
            self.assertFalse(result['guards']['changed_support'])

class PrefixAndContractTests(unittest.TestCase):

    def test_calendar_rejects_nonformal_signals_except_initial_partial(self):
            cal={'sessions':['2016-08-17','2016-08-30','2016-08-31','2016-09-29','2016-09-30']}
            with patch.object(h,'sha',return_value='digest'):
                self.assertEqual(h.verify_calendar_dates({'dates':['2016-08-17','2016-08-31','2016-09-30']},cal)['formal_events'],2)
                for dates in (['2016-08-17','2016-08-30'],['2016-08-31'],['2016-08-17','2016-09-30','2016-08-31']):
                    with self.subTest(dates=dates),self.assertRaises(ValueError):h.verify_calendar_dates({'dates':dates},cal)

    def test_candidate_must_change_vs_control_in_both_training_eras(self):
            metrics,orders=selection_fixture();orders[h.NAMES[2]]=copy.deepcopy(orders[h.NAMES[1]])
            selection=h.select_first88(metrics,orders)
            self.assertFalse(selection['admitted']);self.assertFalse(selection['guards']['changed_support_vs_control'])

    def test_exact_38_results_manifest(self):
            good={name:'synthetic' for name in h.RESULT_FILES};self.assertEqual(len(good),38);h.require_result_manifest(good)
            for kind in ('missing','extra','completion'):
                bad=dict(good)
                if kind=='missing':bad.pop(next(iter(bad)))
                else:bad['COMPLETION.json' if kind=='completion' else 'EXTRA.json']='bad'
                with self.subTest(kind=kind),self.assertRaises(ValueError):h.require_result_manifest(bad)

    def test_only_exact_registered_external_runtime_allowed(self):
            runtime,digest=next(iter(h.EXTERNAL_RUNTIME.items()))
            with patch.object(h,'sha',return_value=digest),patch.object(Path,'is_file',return_value=True):
                h.verify({runtime:digest})
                for path,value in ((runtime,'wrong'),('/Library/Developer/CommandLineTools/other.py',digest),('/tmp/other.py',digest)):
                    with self.subTest(path=path,value=value),self.assertRaises(ValueError):h.verify({path:value})

    def test_mutable_central_cannot_be_persistent_pin(self):
            with patch.object(h,'sha',return_value='synthetic'):
                with self.assertRaisesRegex(ValueError,'Mutable central'):h.verify({str(h.CENTRAL):'synthetic'})

    def test_test_receipt_is_write_once(self):
            with tempfile.TemporaryDirectory() as directory:
                receipt=Path(directory)/'receipt.json';receipt.write_text('keep')
                empty=SimpleNamespace()
                with patch.object(h,'TEST_RECEIPT',receipt),patch.object(h,'module',return_value=empty):
                    with self.assertRaisesRegex(ValueError,'write-once'):h.tests()
                self.assertEqual(receipt.read_text(),'keep')

    def test_central_compare_and_swap_rejects_concurrent_change(self):
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'state.json';path.write_bytes(b'changed')
                with patch.object(h,'CENTRAL',path):
                    with self.assertRaisesRegex(ValueError,'Concurrent central'):h.commit_state(b'prior',{},'test')
                self.assertEqual(path.read_bytes(),b'changed')

class ReceiptTests(unittest.TestCase):

    def test_first88_freeze_binds_all_prerequisites(self):
            metrics,orders=selection_fixture();selection=h.select_first88(metrics,orders)
            frozen=dict(plan_sha256='plan',orders_sha256='digest',training_sha256='digest',prepared_sha256='digest',selection=selection)
            def reader(path):return frozen if path.name=='FROZEN.json' else metrics
            with patch.object(h,'sha',return_value='digest'),patch.object(h,'read',side_effect=reader):
                self.assertEqual(h.frozen_selection('plan',orders),frozen);frozen['prepared_sha256']='changed'
                with self.assertRaises(ValueError):h.frozen_selection('plan',orders)

class PhaseGateTests(unittest.TestCase):

    def test_prepare_gate_precedes_model_or_data_loading(self):
            with patch.object(h,'gate',side_effect=ValueError('unregistered')),patch.object(h,'local') as loader:
                with self.assertRaisesRegex(ValueError,'unregistered'):h.prepare('bad')
                loader.assert_not_called()

    def test_prepared_manifest_is_exact_before_opening_arrays(self):
            receipt=dict(status='COMPLETE',plan_sha256='plan',artifacts_sha256={name:'digest' for name in h.PREPARED_FILES})
            for change in ('missing','extra'):
                bad=copy.deepcopy(receipt)
                if change=='missing':bad['artifacts_sha256'].pop(next(iter(bad['artifacts_sha256'])))
                else:bad['artifacts_sha256']['COMPLETION.json']='extra'
                with patch.object(h,'gate'),patch.object(h,'read',return_value=bad),patch.object(h.ev,'load_arrays') as loader:
                    with self.subTest(change=change),self.assertRaises(ValueError):h.prepared('plan')
                    loader.assert_not_called()

    def test_code_review_requires_no_actual_fits_and_nonempty_immutable_inputs(self):
            files=[h.ROOT/'work'/('synthetic'+str(i)) for i in range(6)]
            receipt=dict(passed=True,unresolved_blockers=[],context_sha256=h.CONTEXT_SHA,code_sha256={str(p.relative_to(h.ROOT)):'digest' for p in files},input_sha256={'work/fake':'digest'})
            for key in ('actual_features_computed','actual_model_fits_computed','actual_portfolio_returns_computed','production_files_modified','strategy_registered_by_reviewer'):receipt[key]=False
            with patch.object(h,'code_files',return_value=files),patch.object(h,'read',return_value=receipt),patch.object(h,'sha',return_value='digest'),patch.object(h,'bind'),patch.object(h,'verify'):
                h.code_review_binding();receipt['context_sha256']='wrong'
                with self.assertRaises(ValueError):h.code_review_binding()
                receipt['context_sha256']=h.CONTEXT_SHA;receipt['actual_model_fits_computed']=True
                with self.assertRaises(ValueError):h.code_review_binding()
                receipt['actual_model_fits_computed']=False;receipt['input_sha256']={}
                with self.assertRaises(ValueError):h.code_review_binding()

if __name__=="__main__":unittest.main(verbosity=2)

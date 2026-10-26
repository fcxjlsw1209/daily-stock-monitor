"""Synthetic host boundaries and real model/oracle interface, no historical arrays."""
from pathlib import Path
import ast
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_early_history229_host as h
import monthly_early_history229_model as model
import monthly_early_history229_independent as oracle


def fixture():
    dates=pd.date_range('2011-01-01','2016-09-30',freq='BME').strftime('%Y-%m-%d').tolist()
    nexts=pd.date_range('2011-02-01','2016-10-31',freq='BME').strftime('%Y-%m-%d').tolist()
    symbols=[['A','B','C'] if i%2==0 else ['C','A','B'] for i in range(len(dates))]
    n=3*len(dates);row=np.arange(n,dtype=float)
    train=dict(dates=np.repeat(dates,3),ends=np.repeat(nexts,3),offsets=np.arange(len(dates)+1,dtype=np.int64)*3,
        train_plain=.3*np.sin(row[:,None]*.17+np.arange(10)[None,:]*.21),
        train_eligible=np.tile([True,True,False],len(dates)),train_good=np.ones(n,bool),common_good=np.ones(n,bool),vol63=np.full(n,.03))
    for i,k in enumerate(h.TARGETS):train['target_'+k]=.08*np.sin(row*.37+i)
    tm=dict(dates=dates,ends=nexts,symbols=symbols,pool_origins=['synthetic']*len(dates),feature_names=['f%d'%i for i in range(10)])
    labels={k:train[k].copy() for k in ('dates','ends','offsets','vol63','common_good')}
    labels['added']=np.zeros(n,bool)
    for k in h.TARGETS:labels['target_'+k]=train['target_'+k].copy()
    return train,tm,labels


def state_fixture():
    entries=[dict(trials=1,directory='prior') for _ in range(225)]
    entries[0]['trials']=968;entries[-1]['directory']=h.PRIOR.name
    return dict(completed=entries,best_training_selected_candidate=dict(directory='monthly_local_linear70'),registered_queue=[],target_reached=False)


def synthetic_receipt():
    check={k:True for k in oracle.FORECAST_FLAGS}
    check.update(numerical_tolerance=1e-12,producer_modules_imported=False,computes_returns=False,shared_numerical_primitives=False,
        counts={k:0 for k in oracle.COUNT_FIELDS})
    return check,dict(audit=dict(counts=copy.deepcopy(check['counts'])))


class InputBoundaryTests(unittest.TestCase):
    def test_exact67_month_prefix_preserves_original_order_and_metadata(self):
        train,tm,_=fixture();args,hm=h.model_inputs(train,tm)
        self.assertEqual(len(hm['dates']),67);self.assertEqual(len(args[6]),201)
        self.assertEqual(hm['dates'][0],'2011-01-31');self.assertEqual(hm['dates'][-1],'2016-07-29')
        self.assertEqual(hm['symbols'],tm['symbols'][:67]);self.assertEqual(hm['feature_names'],tm['feature_names'])
        np.testing.assert_array_equal(args[9],train['train_eligible'][:201]);self.assertEqual(len(args[0]),len(train['dates']))
    def test_union_uses_original_signal_eligibility_not_future_good(self):
        train,tm,_=fixture();train['train_good'][:]=False;train['common_good'][:]=False
        for k in h.TARGETS:train['target_'+k][:]=np.nan
        args,_=h.model_inputs(train,tm)
        np.testing.assert_array_equal(args[9],train['train_eligible'][:201]);self.assertTrue(args[9].any())
    def test_physical_training_cutoffs_all_five_prefixes(self):
        train,tm,_=fixture()
        for n in h.PREFIXES:
            with self.subTest(n=n):
                args,hm=h.model_inputs(train,tm,limit=n);self.assertEqual(len(args[6]),3*n)
                if n:
                    cutoff=str(int(hm['dates'][-1][:4])-1)+'-12-31'
                    expected=np.asarray(train['ends'],dtype='datetime64[D]')<=np.datetime64(cutoff)
                    np.testing.assert_array_equal(args[0],train['dates'][expected]);np.testing.assert_array_equal(args[3],train['train_plain'][expected])
                    self.assertTrue(all(str(d)<=cutoff for d in args[1]))
                else:self.assertEqual(args[3].shape,(0,10));self.assertEqual(args[8].shape,(0,10));self.assertEqual(hm['dates'],[])
    def test_future_numeric_poison_physically_excluded(self):
        train,tm,_=fixture();base,_=h.model_inputs(train,tm,limit=24)
        future=np.asarray(train['ends'],dtype='datetime64[D]')>np.datetime64('2011-12-31')
        for k in h.TARGETS:train['target_'+k][future]=np.inf
        got,_=h.model_inputs(train,tm,limit=24)
        for k in h.TARGETS:np.testing.assert_array_equal(got[4][k],base[4][k]);self.assertTrue(np.isfinite(got[4][k]).all())
    def test_inputs_detached_and_no_mutation(self):
        train,tm,_=fixture();before=h.snapshot(train);args,hm=h.model_inputs(train,tm,limit=48)
        self.assertEqual(before,h.snapshot(train))
        for i,k in ((0,'dates'),(1,'ends'),(3,'train_plain'),(5,'common_good'),(6,'dates'),(8,'train_plain'),(9,'train_eligible')):
            self.assertFalse(np.shares_memory(args[i],train[k]))
        for k in h.TARGETS:self.assertFalse(np.shares_memory(args[4][k],train['target_'+k]))
    def test_corrupt_flat_identity_offsets_and_eligibility_rejected(self):
        for kind in ('date','end','offset','eligible','features'):
            train,tm,_=fixture()
            if kind=='date':train['dates'][0]='2010-01-31'
            elif kind=='end':train['ends'][0]='2012-01-31'
            elif kind=='offset':train['offsets'][1]+=1
            elif kind=='eligible':train['train_eligible']=train['train_eligible'].astype(int)
            else:train['train_plain']=train['train_plain'][:,:9]
            with self.subTest(kind=kind),self.assertRaises((ValueError,AssertionError)):h.model_inputs(train,tm)
    def test_bad_prefix_and_metadata_order_rejected(self):
        train,tm,_=fixture()
        for limit in (-1,68,1.5,True):
            with self.subTest(limit=limit),self.assertRaises(ValueError):h.early_meta(tm,limit)
        tm['dates'][0],tm['dates'][1]=tm['dates'][1],tm['dates'][0]
        with self.assertRaises(ValueError):h.early_meta(tm)


class LabelCopyTests(unittest.TestCase):
    def test_exact_label_copy_each_prefix_and_no_aliases(self):
        train,tm,labels=fixture();before=h.snapshot(labels)
        for n in h.PREFIXES:
            hm=h.early_meta(tm,n);got=h.slice_labels(labels,train,hm);h.verify_label_copy(got,labels,hm)
            self.assertEqual(got['offsets'].tolist(),labels['offsets'][:n+1].tolist())
            self.assertEqual(len(got['common_good']),n*3)
            for k in labels:self.assertFalse(np.shares_memory(got[k],labels[k]))
        self.assertEqual(before,h.snapshot(labels))
    def test_original_nan_and_signedzero_payloads_preserved(self):
        train,tm,labels=fixture();x=labels['target_log_excess'];x[0]=-0.;x[1]=np.nan;x.view(np.uint64)[1]=0x7ff8000000000023
        got=h.slice_labels(labels,train,h.early_meta(tm,1))
        self.assertEqual(got['target_log_excess'].tobytes(),x[:3].tobytes())
    def test_equal_values_wrong_identity_dtype_rejected(self):
        train,tm,labels=fixture();labels['dates']=labels['dates'].astype('U12')
        with self.assertRaisesRegex(ValueError,'Exact source227 label axis'):h.slice_labels(labels,train,h.early_meta(tm))
    def test_missing_extra_fields_and_wrong_dimensions_rejected(self):
        for kind in ('missing','extra','shape','length'):
            train,tm,labels=fixture()
            if kind=='missing':del labels['added']
            elif kind=='extra':labels['other']=np.zeros(len(train['dates']))
            elif kind=='shape':labels['vol63']=labels['vol63'][:,None]
            else:labels['vol63']=labels['vol63'][:-1]
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.slice_labels(labels,train,h.early_meta(tm))
    def test_tampered_copy_or_shared_view_rejected(self):
        train,tm,labels=fixture();hm=h.early_meta(tm,1)
        got=h.slice_labels(labels,train,hm);got['target_log_excess'][0]+=1.
        with self.assertRaisesRegex(ValueError,'bytes'):h.verify_label_copy(got,labels,hm)
        got=h.slice_labels(labels,train,hm);got['vol63']=labels['vol63'][:3]
        with self.assertRaisesRegex(ValueError,'detached'):h.verify_label_copy(got,labels,hm)


class ReceiptTests(unittest.TestCase):
    def test_full_receipt_is_accepted(self):
        c,f=synthetic_receipt();h.require_check(c,f)
    def test_every_missing_or_false_flag_rejected(self):
        for key in oracle.FORECAST_FLAGS:
            for mode in ('missing','false'):
                c,f=synthetic_receipt()
                if mode=='missing':del c[key]
                else:c[key]=False
                with self.subTest(key=key,mode=mode),self.assertRaises(ValueError):h.require_check(c,f)
    def test_count_mismatch_and_unknown_count_rejected(self):
        c,f=synthetic_receipt();c['counts']['query_rows']=1
        with self.assertRaises(ValueError):h.require_check(c,f)
        c,f=synthetic_receipt();c['counts']['unexpected']=0;f['audit']['counts']['unexpected']=0
        with self.assertRaises(ValueError):h.require_check(c,f)
    def test_primitive_return_and_tolerance_scope_rejected(self):
        for key,value in (('numerical_tolerance',1e-8),('producer_modules_imported',True),('computes_returns',True),('shared_numerical_primitives',True)):
            c,f=synthetic_receipt();c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_check(c,f)
    def test_actual_synthetic_model_independent_host_interface(self):
        train,tm,_=fixture();args,_=h.model_inputs(train,tm,limit=48);allowed=np.ones(len(args[6]),bool)
        result=model.forecasts(*args,expansion_allowed=allowed)
        check=oracle.check_forecasts(*args,expansion_allowed=allowed,result=result);h.require_check(check,result)
        self.assertGreater(check['counts']['actual_regression_solves'],0)
        self.assertTrue(result['available'][72:].any());self.assertFalse(result['available'][:72].any())
    def test_full_synthetic_five_physical_prefix_model_parity(self):
        train,tm,_=fixture();args,_=h.model_inputs(train,tm);full=model.forecasts(*args,expansion_allowed=np.ones(len(args[6]),bool))
        for n in h.PREFIXES:
            a,_=h.model_inputs(train,tm,limit=n);q=len(a[6]);small=model.forecasts(*a,expansion_allowed=np.ones(q,bool))
            for key in ('available','local_available'):np.testing.assert_array_equal(small[key],full[key][:q])
            for key in small['raw']:np.testing.assert_array_equal(small['raw'][key],full['raw'][key][:q])
    def test_warmup_physical_price_prefix_before_model(self):
        cal=pd.bdate_range('2014-01-01',periods=310);px=pd.DataFrame({'A':100.+np.arange(310)},index=cal)
        day=cal[294];base=h.prior_source.signal_support(px,[day],['A'],cal)
        poison=px.astype(object);poison.loc[cal[295]:,'A']='future';short=poison.loc[:day]
        np.testing.assert_array_equal(h.prior_source.signal_support(short,[day],['A'],cal),base)
        self.assertTrue(base[0])


class PhaseTests(unittest.TestCase):
    def test_contract_source_only_counts_and_no_strategy_mutation(self):
        c=h.contract();self.assertEqual(c['prefixes'],[0,12,24,48,67]);self.assertTrue(c['source_only']);self.assertEqual(c['strategy_trials'],0)
        for k in ('new_portfolio_returns','new_ranks','new_labels','strict_PIT','observer_writes','original_strategy_changed','parameter_search'):self.assertIs(c[k],False)
        self.assertEqual(len(h.code_files()),6);self.assertEqual(len(set(h.code_files())),6)
    def test_state225_1192_and_source_does_not_count_as_strategy(self):
        s=state_fixture();h.state_check(s);s['source_experiments']=[dict(strategy_trials=0)];h.state_check(s)
        for kind in ('rows','trials','latest','best','target'):
            bad=copy.deepcopy(s)
            if kind=='rows':bad['completed'].pop()
            elif kind=='trials':bad['completed'][0]['trials']+=1
            elif kind=='latest':bad['completed'][-1]['directory']='other'
            elif kind=='best':bad['best_training_selected_candidate']['directory']='other'
            else:bad['target_reached']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.state_check(bad)
    def test_wrong_explicit_plan_before_data_read(self):
        with patch.object(h,'sha',return_value='wrong'),patch.object(h,'read') as reader:
            with self.assertRaisesRegex(ValueError,'Explicit PLAN'):h.gate('expected')
            reader.assert_not_called()
    def test_prepare_gate_before_source_load(self):
        with patch.object(h,'gate',side_effect=ValueError('not registered')),patch.object(h.previous,'local') as load:
            with self.assertRaisesRegex(ValueError,'not registered'):h.prepare('expected')
            load.assert_not_called()
    def test_gate_queue_contract_runtime_and_pins(self):
        s=state_fixture();s['registered_queue']=[dict(directory=h.OUT.name,plan_sha256='digest')]
        p=dict(runtime=h.ev.runtime(),contract=h.contract(),input_sha256={},prior_central_sha256='digest')
        def reader(path):return p if path==h.OUT/'PLAN.json' else h.contract() if path==h.CONTEXT else s
        with patch.object(h,'sha',return_value='digest'),patch.object(h,'read',side_effect=reader),patch.object(h.previous,'verify'):
            self.assertEqual(h.gate('digest'),p);s['registered_queue']=[]
            with self.assertRaisesRegex(ValueError,'Registration'):h.gate('digest')
            s['registered_queue']=[dict(directory=h.OUT.name,plan_sha256='digest')];p['runtime']={}
            with self.assertRaisesRegex(ValueError,'Runtime'):h.gate('digest')
    def test_changed_input_pin_rejected(self):
        p=dict(runtime=h.ev.runtime(),contract=h.contract(),input_sha256={},prior_central_sha256='digest')
        with patch.object(h,'sha',return_value='digest'),patch.object(h,'read',side_effect=lambda path:p if path==h.OUT/'PLAN.json' else h.contract()),patch.object(h.previous,'verify',side_effect=ValueError('changed pin')):
            with self.assertRaisesRegex(ValueError,'changed pin'):h.gate('digest')
    def test_writeonce_register_and_test_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);receipt=out/'TESTS.json';receipt.write_text('preserve')
            with patch.object(h,'OUT',out),patch.object(h,'read') as reader:
                with self.assertRaisesRegex(ValueError,'New source'):h.register()
                reader.assert_not_called()
            with patch.object(h,'TESTS',receipt),patch.object(h.previous,'module') as loader:
                with self.assertRaisesRegex(ValueError,'Test receipt'):h.test()
                loader.assert_not_called()
            self.assertEqual(receipt.read_text(),'preserve')
    def test_register_failures_before_source_import_and_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);central=root/'CENTRAL.json';central.write_text(json.dumps(state_fixture()))
            tests=dict(passed=True,synthetic_only=True,runtime=h.ev.runtime(),code_sha256={str(p):'digest' for p in h.code_files()})
            review=dict(passed=True,metadata_only=True,context_sha256='digest')
            code=dict(passed=True,unresolved_blockers=[],context_sha256='digest',code_sha256=tests['code_sha256'].copy())
            def reader(path):
                if path==h.TESTS:return tests
                if path==h.REVIEW:return review
                if path==h.CODE_REVIEW:return code
                if path==h.CONTEXT:return h.contract()
                raise AssertionError('Unregistered source read')
            with patch.object(h,'OUT',root/'new'),patch.object(h,'CENTRAL',central),patch.object(h,'sha',return_value='digest'),patch.object(h,'read',side_effect=reader):
                for kind in ('test','source','context','review','code'):
                    tests['passed']=kind!='test';review['metadata_only']=kind!='source';review['context_sha256']='bad' if kind=='context' else 'digest'
                    code['unresolved_blockers']=['bad'] if kind=='review' else [];code['code_sha256']={} if kind=='code' else tests['code_sha256'].copy()
                    with self.subTest(kind=kind),self.assertRaises(ValueError):h.register()
                    self.assertFalse((root/'new').exists())
            self.assertEqual(json.loads(central.read_text()),state_fixture())
    def test_incomplete_manifest_blocks_close(self):
        with tempfile.TemporaryDirectory() as directory:
            d=dict(status='COMPLETE',source_only=True,plan_sha256='expected',artifacts_sha256={'FORECASTS.npz':'h'})
            with patch.object(h,'OUT',Path(directory)),patch.object(h,'gate',return_value={}),patch.object(h,'read',return_value=d),patch.object(h,'dump') as save:
                with self.assertRaisesRegex(ValueError,'artifact set'):h.close('expected')
                save.assert_not_called()
    def test_existing_closure_is_writeonce(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'COMPLETION.json').write_text('keep')
            with patch.object(h,'OUT',out),patch.object(h,'gate',return_value={}),patch.object(h,'read') as read:
                with self.assertRaisesRegex(ValueError,'write-once'):h.close('expected')
                read.assert_not_called()
    def test_source_only_phase_order_and_full_protection_static(self):
        text=Path(h.__file__).read_text();tree=ast.parse(text);fs={n.name:ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
        p=fs['prepare'];self.assertLess(p.index('gate(expected)'),p.index('study.load()'))
        self.assertLess(p.index('small_support='),p.index('small=model.forecasts'))
        self.assertIn('require_check(check,forecast)',p);self.assertIn('snapshot(x) for x in (train,a,labels,cached)',p)
        self.assertIn('risk_labels=slice_labels(labels,train,hm)',p);self.assertNotIn('model_inputs(labels',p)
        for name in ('register','close'):self.assertIn('previous.verify(protected)',fs[name]);self.assertIn('21444',fs[name])
        self.assertIn('source_experiments',fs['close']);self.assertNotIn("['completed'].append",fs['close'])
        for bad in ('.decisions(','.make_orders(','.replay(','.evaluate(','.select(','.build_labels('):self.assertNotIn(bad,p)

if __name__=='__main__':unittest.main()

"""Synthetic source-host boundaries only; no historical input arrays or returns."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_ineligible_source227_host as h


def price_fixture():
    sessions=pd.bdate_range('2015-01-01',periods=330)
    price=pd.DataFrame({'A':100.+np.arange(len(sessions)), 'B':np.full(len(sessions),20.)},index=sessions)
    return price,sessions


def training_fixture():
    dates=['2017-10-31','2017-11-30','2017-12-29']
    ends=['2017-11-30','2017-12-29','2018-01-31']
    symbols=[['A','B'],['B','A','C'],['A']]
    widths=np.array([2,3,1]);n=int(widths.sum())
    meta=dict(dates=dates,ends=ends,symbols=symbols,pool_origins=['synthetic']*3,feature_names=['f%02d'%i for i in range(10)])
    train=dict(dates=np.repeat(dates,widths),ends=np.repeat(ends,widths),offsets=np.r_[0,np.cumsum(widths)],
        train_plain=np.arange(n*10.).reshape(n,10),train_good=np.ones(n,bool),train_eligible=np.ones(n,bool),
        common_good=np.ones(n,bool),target_log_excess=np.arange(n,dtype=float),event_plain=np.zeros((8,10)))
    return train,meta


def state_fixture():
    entries=[dict(trials=1,directory='prior') for _ in range(224)]
    entries[0]['trials']=966;entries[-1]['directory']='monthly_joint_error226'
    return dict(completed=entries,best_training_selected_candidate=dict(directory='monthly_local_linear70'),
        registered_queue=[],target_reached=False)


class SignalSupportTests(unittest.TestCase):
    def test_exact295_official_past_sessions_and_no_extra_warmup(self):
        px,calendar=price_fixture()
        np.testing.assert_array_equal(h.signal_support(px,[calendar[293],calendar[294],calendar[295]],['A']*3,calendar),[False,True,True])
        # Extra pre-calendar observations cannot replace a missing official session.
        missing=px.drop(calendar[0]);extra=pd.DataFrame({'A':[10.],'B':[10.]},index=[calendar[0]-pd.Timedelta(days=1)])
        missing=pd.concat([extra,missing]).sort_index()
        self.assertFalse(h.signal_support(missing,[calendar[294]],['A'],calendar)[0])
    def test_each_invalid_past_price_blocks_only_affected_identity(self):
        for value in (np.nan,np.inf,-np.inf,0.,-1.):
            px,calendar=price_fixture();px.loc[calendar[100],'A']=value
            with self.subTest(value=value):np.testing.assert_array_equal(h.signal_support(px,[calendar[294]]*2,['A','B'],calendar),[False,True])
        px,calendar=price_fixture();px.loc[calendar[294],'A']=0.
        self.assertFalse(h.signal_support(px,[calendar[294]],['A'],calendar)[0])
    def test_physical_future_prefix_and_future_nonnumeric_poison(self):
        px,calendar=price_fixture();day=calendar[300]
        expected=h.signal_support(px,[day,day],['A','B'],calendar)
        np.testing.assert_array_equal(h.signal_support(px.loc[:day],[day,day],['A','B'],calendar),expected)
        poisoned=px.astype(object);poisoned.loc[calendar[301]:,:]='future poison'
        np.testing.assert_array_equal(h.signal_support(poisoned,[day,day],['A','B'],calendar),expected)
        # Future calendar knowledge never licenses consuming future prices.
        np.testing.assert_array_equal(h.signal_support(px.loc[:day],[day],['A'],calendar[:301]),[True])
    def test_missing_identity_off_calendar_and_empty(self):
        px,calendar=price_fixture();weekend=calendar[300]+pd.Timedelta(days=1)
        while weekend in calendar:weekend+=pd.Timedelta(days=1)
        np.testing.assert_array_equal(h.signal_support(px,[calendar[300],weekend],['MISSING','A'],calendar),[False,False])
        got=h.signal_support(px,np.array([],dtype='datetime64[D]'),[],calendar);self.assertEqual(got.shape,(0,));self.assertEqual(got.dtype,np.dtype(bool))
    def test_bad_axes_masks_and_calendar_rejected(self):
        px,calendar=price_fixture()
        cases=[([calendar[300]],['A','B'],calendar),(['NaT'],['A'],calendar),([calendar[300]],[' A'],calendar),
            ([calendar[300]],['A'],calendar[::-1]),([calendar[300]],['A'],calendar.insert(2,calendar[2]))]
        for dates,symbols,sessions in cases:
            with self.subTest(dates=dates),self.assertRaises((ValueError,TypeError)):h.signal_support(px,dates,symbols,sessions)
        with self.assertRaises(ValueError):h.signal_support(pd.concat([px,px.iloc[:1]]),[calendar[300]],['A'],calendar)


class TrainingPrefixTests(unittest.TestCase):
    def test_exact_mature_end_filter_and_whole_month_identity(self):
        train,meta=training_fixture();before=copy.deepcopy((train,meta))
        small,tm,mask=h.slice_training(train,meta,'2017-12-31')
        np.testing.assert_array_equal(mask,[True]*5+[False]);np.testing.assert_array_equal(small['offsets'],[0,2,5])
        self.assertEqual(tm['dates'],meta['dates'][:2]);self.assertEqual(tm['symbols'],meta['symbols'][:2]);self.assertEqual(tm['feature_names'],meta['feature_names'])
        for key in ('dates','ends','train_plain','train_good','train_eligible','common_good','target_log_excess'):
            np.testing.assert_array_equal(small[key],train[key][:5])
        np.testing.assert_array_equal(small['event_plain'],train['event_plain'])
        self.assertEqual(meta,before[1])
        for key in train:np.testing.assert_array_equal(train[key],before[0][key])
    def test_empty_prefix_and_future_target_poison_not_consumed(self):
        train,meta=training_fixture();train['target_log_excess']=train['target_log_excess'].astype(object);train['target_log_excess'][-1]='future label'
        small,tm,mask=h.slice_training(train,meta,'2017-12-31')
        self.assertEqual(small['target_log_excess'].tolist(),list(np.arange(5.)))
        empty,em,mask=h.slice_training(train,meta,'2017-10-01')
        self.assertEqual(empty['offsets'].tolist(),[0]);self.assertEqual(empty['train_plain'].shape,(0,10));self.assertEqual(em['dates'],[])
    def test_offset_and_flat_end_identity_corruption_is_rejected(self):
        for kind in ('offset','end','natcutoff'):
            train,meta=training_fixture();cutoff='2017-12-31'
            if kind=='offset':train['offsets'][1]+=1
            elif kind=='end':train['ends'][0]='2019-01-01'
            else:cutoff='NaT'
            with self.subTest(kind=kind),self.assertRaises((AssertionError,ValueError)):h.slice_training(train,meta,cutoff)


class SourcePhaseTests(unittest.TestCase):
    def test_contract_is_source_only_and_keeps_training_reference_fixed(self):
        c=h.contract();self.assertTrue(c['source_only']);self.assertEqual(c['strategy_trials'],0)
        for k in ('parameter_search','portfolio_returns','observer_writes','original_trading_pool_changed','original_strategy_changed'):self.assertIs(c[k],False)
        self.assertEqual(c['model_prefixes'],[0,1,53,88,122]);self.assertEqual(c['label_physical_cutoffs'],['2019-12-31','2023-12-31','2025-12-31'])
        self.assertIn('Extra labels never enter model fitting',c['training']);self.assertIn('ORIGINAL common_good',c['labels'])
        self.assertEqual(c['old_label_parity_atol'],1e-12);self.assertEqual(c['old_label_parity_rtol'],0.)
        self.assertEqual(len(h.code_files()),6);self.assertEqual(len(set(h.code_files())),6)
    def test_state_expects224_strategies1189_trials_and_does_not_count_source(self):
        s=state_fixture();h.state_check(s)
        s['source_experiments']=[dict(directory='another_source',strategy_trials=0)];h.state_check(s)
        for kind in ('completed','trials','latest','incumbent','target'):
            bad=copy.deepcopy(s)
            if kind=='completed':bad['completed'].pop(0)
            elif kind=='trials':bad['completed'][0]['trials']+=1
            elif kind=='latest':bad['completed'][-1]['directory']='other'
            elif kind=='incumbent':bad['best_training_selected_candidate']['directory']='other'
            else:bad['target_reached']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.state_check(bad)
    def test_explicit_plan_gate_before_read_or_model_loading(self):
        with patch.object(h,'sha',return_value='wrong'),patch.object(h,'read') as reader:
            with self.assertRaisesRegex(ValueError,'Explicit source PLAN'):h.gate('expected')
            reader.assert_not_called()
        with patch.object(h,'gate',side_effect=ValueError('not registered')),patch.object(h.previous,'local') as loader:
            with self.assertRaisesRegex(ValueError,'not registered'):h.prepare('expected')
            loader.assert_not_called()
    def test_gate_contract_runtime_and_queue_are_frozen(self):
        s=state_fixture();s['registered_queue']=[dict(directory=h.OUT.name,plan_sha256='expected')]
        plan=dict(runtime=h.ev.runtime(),contract=h.contract(),input_sha256={},prior_central_sha256='expected')
        def reader(path):
            if path==h.OUT/'PLAN.json':return plan
            if path==h.CENTRAL:return s
            if path==h.CONTEXT:return h.contract()
            raise AssertionError('Unexpected source read')
        with patch.object(h,'sha',return_value='expected'),patch.object(h,'read',side_effect=reader),patch.object(h.previous,'verify') as verify:
            self.assertEqual(h.gate('expected'),plan);verify.assert_called_once_with({})
            s['registered_queue']=[]
            with self.assertRaisesRegex(ValueError,'registration mismatch'):h.gate('expected')
            s['registered_queue']=[dict(directory=h.OUT.name,plan_sha256='expected')]
            plan['contract']=dict(plan['contract'],strategy_trials=1)
            with self.assertRaisesRegex(ValueError,'runtime/contract'):h.gate('expected')
    def test_writeonce_source_and_formal_tests_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'exists';out.mkdir();tests=root/'TESTS.json';tests.write_text('keep')
            with patch.object(h,'OUT',out),patch.object(h,'read') as reader:
                with self.assertRaisesRegex(ValueError,'write-once'):h.register()
                reader.assert_not_called()
            with patch.object(h,'TESTS',tests),patch.object(h.previous,'module') as loader:
                with self.assertRaisesRegex(ValueError,'receipt already'):h.test()
                loader.assert_not_called()
            self.assertEqual(tests.read_text(),'keep')
    def test_register_rejects_missing_tests_and_source_context_mismatch_before_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);central=root/'CENTRAL.json';central.write_text(json.dumps(state_fixture()))
            tests=dict(passed=True,synthetic_only=True,runtime=h.ev.runtime(),code_sha256={str(p):'digest' for p in h.code_files()})
            review=dict(passed=True,metadata_only=True,context_sha256='digest')
            failure=[None]
            def reader(path):
                if path==h.TESTS:
                    if failure[0]=='missing_tests':raise FileNotFoundError('missing tests')
                    return tests
                if path==h.REVIEW:return review
                if path==h.CONTEXT:return h.contract()
                raise AssertionError('Gate allowed early model/source read')
            cases=('missing_tests','failed_tests','source_scope','context_sha')
            with patch.object(h,'OUT',root/'new'),patch.object(h,'CENTRAL',central),patch.object(h,'read',side_effect=reader),patch.object(h,'sha',return_value='digest'):
                for kind in cases:
                    failure[0]=kind;tests['passed']=kind!='failed_tests';review['metadata_only']=kind!='source_scope';review['context_sha256']='wrong' if kind=='context_sha' else 'digest'
                    with self.subTest(kind=kind),self.assertRaises((ValueError,FileNotFoundError)):h.register()
                    self.assertFalse((root/'new').exists())
            self.assertEqual(json.loads(central.read_text()),state_fixture())
    def test_gate_rejects_runtime_and_changed_pins(self):
        state=state_fixture();state['registered_queue']=[dict(directory=h.OUT.name,plan_sha256='digest')]
        plan=dict(runtime=h.ev.runtime(),contract=h.contract(),input_sha256={'synthetic':'digest'},prior_central_sha256='digest')
        def reader(path):
            if path==h.OUT/'PLAN.json':return plan
            if path==h.CONTEXT:return h.contract()
            if path==h.CENTRAL:return state
            raise AssertionError('Unexpected source read')
        with patch.object(h,'sha',return_value='digest'),patch.object(h,'read',side_effect=reader),patch.object(h.previous,'verify',side_effect=ValueError('changed frozen pin')):
            with self.assertRaisesRegex(ValueError,'changed frozen pin'):h.gate('digest')
            plan['runtime']=dict(plan['runtime'],python='wrong')
            with self.assertRaisesRegex(ValueError,'runtime/contract'):h.gate('digest')
    def test_prepared_manifest_missing_member_blocks_closure_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            fake=Path(directory);done=dict(plan_sha256='expected',status='COMPLETE',source_only=True,artifacts_sha256={'FORECASTS.npz':'x'})
            with patch.object(h,'OUT',fake),patch.object(h,'gate',return_value={}),patch.object(h,'read',return_value=done),patch.object(h,'dump') as dump:
                with self.assertRaisesRegex(ValueError,'artifacts differ'):h.close('expected')
                dump.assert_not_called()
    def test_compare_and_swap_rejects_concurrent_central(self):
        with tempfile.TemporaryDirectory() as directory:
            central=Path(directory)/'CENTRAL.json';central.write_bytes(b'changed')
            with patch.object(h,'CENTRAL',central):
                with self.assertRaisesRegex(ValueError,'Concurrent central'):h.commit_state(b'old',{},'test')
            self.assertEqual(central.read_bytes(),b'changed');self.assertEqual(len(list(Path(directory).iterdir())),1)
    def test_source_preparation_no_order_or_return_calls_static(self):
        text=Path(h.__file__).read_text();tree=ast.parse(text)
        functions={node.name:ast.get_source_segment(text,node) for node in tree.body if isinstance(node,ast.FunctionDef)}
        prepare=functions['prepare']
        for forbidden in ('rank_and_delta(','.decisions(','.make_orders(','.replay(','.evaluate(','.select('):self.assertNotIn(forbidden,prepare)
        self.assertLess(prepare.index('gate(expected)'),prepare.index('study.load()'))
        self.assertIn('original_bytes==',prepare);self.assertIn('arrays_bytes==',prepare)
        self.assertIn('args=source206.model_inputs(train,meta,a)',prepare)
        self.assertIn('forecasts=model.forecasts(*args,cached_source=source,expansion_allowed=support)',prepare)
        self.assertLess(prepare.index('forecasts=model.forecasts'),prepare.index('labels=model.build_labels'))
        self.assertNotIn('model_inputs(labels',prepare)
        self.assertIn("{'FORECASTS.npz','LABELS.npz','META.json','SOURCE_AUDIT.json','INDEPENDENT.json'}",functions['close'])
        self.assertIn('len(protected)==21444',functions['close'])
        self.assertIn('previous.verify(protected)',functions['register']);self.assertIn('previous.verify(protected)',functions['close'])
        self.assertNotIn("['completed'].append",functions['close']);self.assertEqual(functions['close'].count("state.setdefault('source_experiments',[]).append"),1)


if __name__=='__main__':unittest.main()

"""Synthetic267 reader adapter/source binding tests; no actual evidence read."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import monthly_boundary_pair267_saved_reader as r


def training():
    row=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3,daily_return_sd=.02,sharpe_status='defined',sharpe_reason=None),early=dict(cagr=.4),recent=dict(cagr=.6))
    metrics={a:{p:copy.deepcopy(row) for p in r.PATHS[:2]} for a in r.ARMS}
    for path in r.PATHS[:2]:metrics[r.ARMS[2]][path]['training']['cagr']=.52
    orders={a:[dict(symbols=['A','B','C']) for _ in range(122)] for a in r.ARMS}
    for i in (1,54,55):orders[r.ARMS[2]][i]['symbols']=['A','B','D']
    return metrics,orders


class AdapterTests(unittest.TestCase):
    def test_candidate_and_reference_roles(self):
        m,o=training();result=r.saved_selection(m,o)
        self.assertTrue(result['admitted']);self.assertEqual(result['primary'],r.ARMS[2])
        self.assertEqual(set(result['guards']),set(r.ARMS[:2]))
        for path in r.PATHS[:2]:m[r.ARMS[1]][path]['training']['cagr']=100.
        self.assertEqual(r.saved_selection(m,o)['primary'],r.ARMS[0])
    def test_later_rows_never_accessed(self):
        class Prefix(list):
            def __getitem__(self,k):
                if type(k) is not int or k>=88:raise AssertionError('Future order')
                return super().__getitem__(k)
        m,o=training();expected=r.saved_selection(m,o)
        self.assertEqual(r.saved_selection(m,{a:Prefix(v) for a,v in o.items()}),expected)
    def test_no_global_or_input_mutation(self):
        m,o=training();before=copy.deepcopy((m,o));roles=r.inherited.ARMS
        r.saved_selection(m,o);self.assertEqual((m,o),before);self.assertIs(r.inherited.ARMS,roles)
    def test_missing_or_foreign_roles_fail(self):
        m,o=training();m.pop(r.ARMS[1])
        with self.assertRaises(ValueError):r.saved_selection(m,o)
    def test_undefined_sharpe_cannot_rescue(self):
        m,o=training();m[r.ARMS[2]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        self.assertFalse(r.saved_selection(m,o)['admitted'])
    def test_exact_export_path_map(self):
        class Store:
            def __init__(self):self.paths=[]
            def read(self,p):self.paths.append(p);return {'path':str(p)}
        ev=Store();view=r.AccountEvidence(ev);self.assertEqual(len(view.paths),36)
        for src,dest in view.paths.items():self.assertEqual(view.read(src),{'path':str(dest)})
        with self.assertRaises(ValueError):view.read(r.inherited.OUT/'results/other.json')
        self.assertEqual(len(ev.paths),36)


class RegistrationTests(unittest.TestCase):
    def test_same_hash_aliases_and_lookup_index(self):
        p=r.ROOT/'work/synthetic.py';h='a'*64
        plan={'input_sha256':{'work/synthetic.py':h,str(p):h}}
        ev=r.Evidence();self.assertEqual(ev.registered(plan,p),h)
        for _ in range(10):self.assertEqual(ev.registered(plan,p),h)
        self.assertEqual(len(ev.registered_indexes),1);ev.finish()
    def test_conflicting_alias_rejected(self):
        p=r.ROOT/'work/synthetic.py';plan={'input_sha256':{'work/synthetic.py':'a'*64,str(p):'b'*64}}
        with self.assertRaises(ValueError):r.Evidence().registered(plan,p)
    def test_map_mutation_detected_at_finish(self):
        p=r.ROOT/'work/synthetic.py';plan={'input_sha256':{str(p):'a'*64}}
        ev=r.Evidence();ev.registered(plan,p);plan['input_sha256'][str(p)]='b'*64
        with self.assertRaises(ValueError):ev.finish()
    def test_missing_pin_and_bad_sha_rejected(self):
        for data in ({},{'work/synthetic.py':'not-a-digest'}):
            with self.assertRaises(ValueError):r.Evidence().registered({'input_sha256':data},r.ROOT/'work/synthetic.py')
    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):json.loads('{"a":1,"a":2}',object_pairs_hook=r.strict_pairs)
    def test_changed_pinned_file_rejected(self):
        ev=r.Evidence();ev.pins['work/synthetic.py']='a'*64
        with patch.object(r,'digest',return_value='b'*64):
            with self.assertRaises(ValueError):ev.finish()
    def test_external_runtime_pin_indexed_without_reading(self):
        plan={'input_sha256':{'/private/tmp/external_runtime_example.py':'a'*64,'work/synthetic.py':'b'*64}}
        self.assertEqual(r.Evidence().registered(plan,r.ROOT/'work/synthetic.py'),'b'*64)


class TerminalTests(unittest.TestCase):
    def record(self):
        hashes={k:str(i)*64 for i,k in enumerate(('prepare','select','diagnose'),1)}
        names={'prepare':'prepared/COMPLETION.json','select':'selected/FROZEN.json','diagnose':'results/COMPLETION.json'}
        phases={k:dict(session_id=42,exit_code=0,chunk_id='actual',artifact_path=str(r.OUT/names[k]),artifact_sha256=h) for k,h in hashes.items()}
        return dict(plan_sha256='a'*64,phases=phases),hashes
    def test_all_three_real_terminal_fields(self):
        record,h=self.record();self.assertEqual(r.terminals_check(record,'a'*64,h),record)
    def test_wrong_exit_binding_or_extra_phase_rejected(self):
        for key,value in [('exit_code',1),('session_id',True),('chunk_id',''),('artifact_sha256','x'*64),('artifact_path',str(r.PRIOR/'prepared/COMPLETION.json'))]:
            record,h=self.record();record['phases']['prepare'][key]=value
            with self.assertRaises(ValueError):r.terminals_check(record,'a'*64,h)
        record,h=self.record();record['phases']['other']={}
        with self.assertRaises(ValueError):r.terminals_check(record,'a'*64,h)




# Defined below for import/discovery; a direct invocation is added after all cases.
class ModelEvidenceTests(unittest.TestCase):
    def fixture(self):
        import numpy as np
        import monthly_boundary_pair267_kernel as k
        from test_monthly_boundary_pair267_kernel import history
        bundle=k.fit_models(history(),2019)
        def encode(value):
            if isinstance(value,np.ndarray):
                return dict(__ndarray__=True,dtype=value.dtype.str,shape=list(value.shape),values_hex=[float(x).hex() for x in value.flat])
            if isinstance(value,dict):return {a:encode(b) for a,b in value.items()}
            if isinstance(value,(list,tuple)):return [encode(v) for v in value]
            return value
        return dict(schema='monthly_boundary_pair267_saved_models_v1',annual=[encode(bundle)],annual_years=[2019],producer_counts={}),dict(dates=['2016-08-17','2019-01-31'])
    def test_saved_weight_support_and_pair_evidence(self):
        record,meta=self.fixture();out=r.models_check(record,meta,2)
        self.assertEqual(out['producer_ridge_solves'],2);self.assertEqual(out['candidate_pair_rows'],18);self.assertEqual(out['control_pair_rows'],180)
    def test_changed_support_clock_pair_weights_fail(self):
        for field,value in [('common_months',19),('cutoff','2019-12-31'),('training_gap_filter',True),('status','cold')]:
            record,meta=self.fixture();record['annual'][0]['cohort'][field]=value
            with self.assertRaises(ValueError):r.models_check(record,meta,2)
        record,meta=self.fixture();record['annual'][0]['cohort']['arms'][r.ARMS[2]]['weights']['values_hex'][0]=0.0.hex()
        with self.assertRaises(ValueError):r.models_check(record,meta,2)
    def test_alternative_pair_rejected(self):
        record,meta=self.fixture();record['annual'][0]['cohort']['arms'][r.ARMS[2]]['pairs'][0]['original_ranks']=[4,5]
        with self.assertRaises(ValueError):r.models_check(record,meta,2)
    def test_future_year_and_false_solve_count_rejected(self):
        record,meta=self.fixture();record['annual_years']=[2020]
        with self.assertRaises(ValueError):r.models_check(record,meta,2)
        record,meta=self.fixture();record['annual'][0]['actual_solve_calls']=0
        with self.assertRaises(ValueError):r.models_check(record,meta,2)
    def test_array_dtype_shape_nan_and_hex_rejected(self):
        good=dict(__ndarray__=True,dtype='<f8',shape=[1],values_hex=[1.0.hex()])
        for field,value in [('dtype','<f4'),('shape',[2]),('values_hex',['nan']),('values_hex',['1.0'])]:
            altered=copy.deepcopy(good);altered[field]=value
            with self.assertRaises(ValueError):r.packed_array(altered,(1,),'test')
    def test_empty_and_initial_only_have_no_years(self):
        record=dict(schema='monthly_boundary_pair267_saved_models_v1',annual=[],annual_years=[],producer_counts={})
        for n in (0,1):self.assertEqual(r.models_check(record,dict(dates=['2016-08-17']),n)['producer_ridge_solves'],0)





class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_monthly_boundary_pair267_host as fixture
        h=fixture.h;args=fixture.fixture();package,wanted,main,src=fixture.run(args);n=args[1]['events']
        cached=[fixture.run(args,i,cached_source=src,cached_models=package)[2] for i in range(n+1)]
        prefixes=(0,1,15,25,n);physical=[fixture.run(args,i)[2] for i in prefixes[:-1]]
        alias=copy.deepcopy(main);alias.update(model_recomputed=False,reuses_main_verified_full=True);physical.append(alias)
        e=dict(schema='monthly_boundary_pair267_independent_prepare_v1',main=main,cached_prefixes=cached,physical_prefixes=physical,all123_order_prefixes=True,source_arrays_unchanged=True,source_training_unchanged=True,source_panel_unchanged=True,original_reference_unchanged=True)
        e['fit_execution_accounting']=h.execution_accounting(e)
        models=dict(schema='monthly_boundary_pair267_saved_models_v1',annual_years=main['source_audit']['consumed_years'],annual=[h.encode_models(package['annual'][str(y)]) for y in main['source_audit']['consumed_years']],producer_counts=main['producer_counts'])
        source=dict(join=args[1],main=main['source_audit'],physical=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in physical],cached=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in cached])
        cls.data=(e,models,wanted[4],source,args[1]['monthly']);cls.prefixes=prefixes
    def test_complete_synthetic_pipeline_saved_review(self):
        summary,work=r.preparation_check(*self.data,prefixes=self.prefixes)
        self.assertEqual(summary['producer_ridge_solves'],2)
        self.assertEqual(work['cached']['producer']['producer_ridge_solves'],0)
        self.assertEqual(work['physical_fresh']['executions'],4)
    def test_underreported_source_integrity_work_rejected(self):
        args=copy.deepcopy(self.data)
        args[0]['main']['source_integrity_work']['checks']-=1
        with self.assertRaises(ValueError):r.preparation_check(*args,prefixes=self.prefixes)
    def test_missing_prefix_or_double_count_rejected(self):
        for kind in ('prefix','alias','work'):
            args=copy.deepcopy(self.data)
            if kind=='prefix':args[0]['cached_prefixes'].pop()
            elif kind=='alias':args[0]['physical_prefixes'][-1]['reuses_main_verified_full']=False
            else:args[0]['fit_execution_accounting']['total']['independent']['independent_svd_solves']+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):r.preparation_check(*args,prefixes=self.prefixes)
    def test_model_proof_and_effect_gate_failure_rejected(self):
        for kind in ('svd','flag','tolerance'):
            args=copy.deepcopy(self.data);entry=args[0]['main']
            if kind=='svd':entry['model_proofs']['2019']['work']['independent_svd_solves']=0
            elif kind=='flag':entry['model_proofs']['2019']['independent_augmented_svd']=False
            else:entry['selection_proofs'][r.ARMS[2]]['numerical_tolerance']=1e-10
            with self.subTest(kind=kind),self.assertRaises(ValueError):r.preparation_check(*args,prefixes=self.prefixes)
    def test_raw_source_units_or_feature_parity_rejected(self):
        for key,value in [('canonical_label_price_source','execution_close'),('train_query_features_bitwise',False),('month_gap_filter',True)]:
            args=copy.deepcopy(self.data);args[0]['main']['source_audit'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r.preparation_check(*args,prefixes=self.prefixes)


if __name__=='__main__':unittest.main(verbosity=2)

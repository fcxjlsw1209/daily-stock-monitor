"""Synthetic-only saved273 evidence tests; never inspect actual273 artifacts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import monthly_matured_clock273_saved_reader as r
import monthly_matured_clock273_independent as o
import monthly_matured_clock273_kernel as k
import monthly_selector_scope268_kernel as selector_kernel
from test_monthly_matured_clock273_kernel import fixture
from test_monthly_matured_clock273_inputs import synthetic_fixture


def bundle(t,m,a):
    b=k.fit_models(t,m,a,control_validator=lambda v:o.verify_control_models(t,m,a,v))
    ranks,delta,audit=k.rank_and_delta(a,m['monthly'],b)
    fc={role+'_'+key:b[role][key].copy() for role in ('control','candidate') for key in ('raw','available','rank')};fc['delta']=delta
    ma={role:deepcopy(b[role]['audit']) for role in ('control','candidate')};ma['audit']=deepcopy(b['audit'])
    e=dict(control_gate=b['audit']['control_gate'],model=o.verify_models(t,m,a,b),delta=o.verify_delta(a,m['monthly'],b,ranks,delta,audit),producer_counts=b['audit']['counts'])
    return b,fc,ma,audit,e


def metrics(cagr=.3):
    return dict(training=dict(cagr=cagr,sharpe=1.2,max_drawdown=-.2),early=dict(cagr=.2),recent=dict(cagr=.25))


class Models(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t,cls.m,cls.a=fixture();cls.b,cls.fc,cls.ma,cls.da,cls.e=bundle(cls.t,cls.m,cls.a)

    def data(self):return deepcopy((self.t,self.m,self.a,self.fc,self.ma,self.da,self.e))

    def test_saved_equations_without_any_fit_or_host(self):
        with patch.object(np.linalg,'solve',side_effect=AssertionError('No solve')),patch.object(np.linalg,'svd',side_effect=AssertionError('No SVD')),patch.object(o,'ridge_fit',side_effect=AssertionError('No fit')):
            _,_,proof=r.saved_models(*self.data())
        self.assertTrue(proof['stored_prediction_equations_exact']);self.assertFalse(proof['coefficient_solutions_refitted'])
        modules=[]
        for node in ast.walk(ast.parse(Path(r.__file__).read_text())):
            if isinstance(node,ast.Import):modules.extend(v.name for v in node.names)
            elif isinstance(node,ast.ImportFrom):modules.append(node.module)
        self.assertFalse(any('host' in v or 'kernel' in v for v in modules))

    def test_saved_monthly_cutoff_reject(self):
        values=self.data();values[4]['candidate']['fits'][0]['cutoff']='2016-09-30'
        with self.assertRaises(ValueError):r.saved_models(*values)

    def test_future_row_membership_reject(self):
        values=self.data();values[4]['candidate']['fits'][0]['training_row_ids'].append(len(values[0]['dates'])-1)
        with self.assertRaises(ValueError):r.saved_models(*values)

    def test_original_annual_raw_reject(self):
        values=self.data();values[2]['raw_source_log'][1,0]+=.01
        with self.assertRaises(ValueError):r.saved_models(*values)

    def test_weights_hash_scaler_and_rhs_reject(self):
        for key in ('weights','mean','scale','rhs'):
            with self.subTest(key=key):
                values=self.data();values[4]['candidate']['fits'][0][key][0]+=.001
                with self.assertRaises(ValueError):r.saved_models(*values)
        values=self.data();values[4]['candidate']['fits'][0]['weights_sha256']='0'*64
        with self.assertRaises(ValueError):r.saved_models(*values)

    def test_stored_coefficient_prediction_mismatch_reject(self):
        values=self.data();values[4]['candidate']['fits'][0]['beta'][0]+=.01
        with self.assertRaises(ValueError):r.saved_models(*values)

    def test_raw_prediction_and_available_mask_reject(self):
        for key in ('candidate_raw','candidate_available','candidate_rank'):
            values=self.data()
            if 'available' in key:values[3][key][1,0]=False
            else:values[3][key][1,0]+=.01
            with self.assertRaises(ValueError):r.saved_models(*values)

    def test_outside_active_copy_and_signed_zero_reject(self):
        values=self.data();values[3]['candidate_raw'][2,0]=0.
        with self.assertRaises(ValueError):r.saved_models(*values)

    def test_exact_delta_audit_reject(self):
        values=self.data();values[3]['delta'][1,0]+=1e-14
        with self.assertRaises(ValueError):r.saved_models(*values)
        values=self.data();values[5]['active_indices'][4]=[0]
        with self.assertRaises(ValueError):r.saved_models(*values)

    def test_wrong_proof_flags_counts_or_tolerance_reject(self):
        for field in ('flags','work','svd_atol'):
            values=self.data();proof=values[6]['model']
            if field=='flags':proof[field]['canonical_ridge_exact']=False
            elif field=='work':proof[field]['canonical_solves']+=1
            else:proof[field]=1e-5
            with self.assertRaises(ValueError):r.saved_models(*values)

    def test_control_gate_is_required_not_passed_placeholder(self):
        values=self.data();values[6]['control_gate']={'passed':True}
        with self.assertRaises(ValueError):r.saved_models(*values)

    def test_label_preservation_digest_tamper_reject(self):
        values=self.data();values[4]['audit']['label_preservation']['target_sha256']['loss_utility']='0'*64
        with self.assertRaises(ValueError):r.saved_models(*values)

    def test_no_mutation(self):
        values=self.data();before={k:v.tobytes() for k,v in values[0].items()}
        r.saved_models(*values)
        self.assertEqual(before,{k:v.tobytes() for k,v in values[0].items()})

    def test_empty_and_partial_saved_models(self):
        for n in (0,1):
            t,m,a,_=k.physical_inputs(self.t,self.m,self.a,n);_,fc,ma,da,e=bundle(t,m,a)
            _,_,proof=r.saved_models(t,m,a,fc,ma,da,e)
            self.assertEqual(proof['annual_records'],0);self.assertEqual(proof['monthly_records'],0)


class SelectionAndAccounting(unittest.TestCase):
    def test_strict_first88_not_equal_or_epsilon_boundary(self):
        x={a:metrics() for a in r.ARMS};self.assertFalse(r.saved_selection(x)['admitted'])
        x[r.ARMS[1]]['training']['cagr']=.3+1e-12;self.assertFalse(r.saved_selection(x)['admitted'])
        x[r.ARMS[1]]['training']['cagr']=.30001;self.assertTrue(r.saved_selection(x)['admitted'])

    def test_all_risk_and_split_gates(self):
        for group,field,value in [('training','sharpe',1.149),('training','max_drawdown',-.231),('early','cagr',.174),('recent','cagr',.224)]:
            x={r.ARMS[0]:metrics(),r.ARMS[1]:metrics(.4)};x[r.ARMS[1]][group][field]=value
            self.assertFalse(r.saved_selection(x)['admitted'])

    def test_selection_forbids_extra_later_cost_and_bad_values(self):
        for change in ('later','bool','nan'):
            x={a:metrics() for a in r.ARMS}
            if change=='later':x[r.ARMS[1]]['later34']={'cagr':100.}
            else:x[r.ARMS[1]]['training']['cagr']=True if change=='bool' else float('nan')
            with self.assertRaises(ValueError):r.saved_selection(x)

    def test_terminal_requires_actual_success_and_artifact(self):
        hashes=dict(prepare='a'*64,select='b'*64,diagnose='c'*64)
        relatives=('prepared/COMPLETION.json','selected/FROZEN.json','results/COMPLETION.json')
        x=dict(plan_sha256='d'*64,phases={k:dict(session_id=99,exit_code=0,chunk_id='abc',artifact_path=str(r.OUT/rel),artifact_sha256=hashes[k],owner='/root') for k,rel in zip(hashes,relatives)})
        r.terminal_check(x,'d'*64,hashes)
        for key,value in [('exit_code',1),('owner','/other'),('artifact_sha256','e'*64),('session_id',True),('chunk_id','')]:
            bad=deepcopy(x);bad['phases']['prepare'][key]=value
            with self.assertRaises(ValueError):r.terminal_check(bad,'d'*64,hashes)

    @staticmethod
    def accounting_fixture():
        stats=dict(start='2016-08-17',end='2026-09-24',years=10.,final_equity=200000.,cagr=.1,sharpe=1.,max_drawdown=-.2,trading_sessions=2520,total_return=1.)
        result=dict(arms={});ind=dict(passed=True,checked_paths=6,checked_slices=30,checked_training_groups=6,independent_accounts=True,independent_of_production_model_and_engine=True,terminal_liquidation=False,paths={});data={};training={}
        for arm in r.ARMS:
            result['arms'][arm]={};ind['paths'][arm]={};training[arm]={k:dict(stats) for k in ('training','early','recent')}
            for path in r.PATHS:
                mt=dict(stats,fees_paid=0.,cost_per_side=.0005 if path=='next_open_cost' else 0.,dollar_turnover=0.,trade_rows=0)
                result['arms'][arm][path]=dict(metrics=mt,slices={s:dict(mt) for s in r.SLICES},accounting=dict(exported_fill_ledger_matches=True,terminal_liquidation=False))
                ind['paths'][arm][path]=dict(passed=True,monthly_executions=122,fees_paid=0.,dollar_turnover=0.,trade_rows=0,slices={s:dict(stats) for s in r.SLICES})
                for kind,value in [('HOLDINGS',[{'fee':0.} for _ in range(122)]),('TRADES',[]),('DAILY',[{'date':'2016-08-17','equity':100000.}]),('DAILY_POSITIONS',[{'date':'2016-08-17'}])]:data[r.OUT/'results'/(arm+'_'+path+'_'+kind+'.json')]=value
        ev=SimpleNamespace(read=lambda path:data[path]);return ev,result,ind,training,data

    def test_six_paths_thirty_slices_six_training_groups(self):
        ev,result,ind,training,_=self.accounting_fixture();proof,_=r.accounting(ev,result,ind,training)
        self.assertEqual((proof['paths'],proof['slices'],proof['training_groups'],proof['zero_streams']),(6,30,6,4))
        self.assertFalse(proof['portfolio_replay'])

    def test_zero_cost_header_and_each_fee_strict(self):
        for change in ('header','holding','independent'):
            ev,result,ind,training,data=self.accounting_fixture()
            if change=='header':result['arms'][r.ARMS[0]]['same_close_zero']['metrics']['cost_per_side']=1e-20
            elif change=='holding':data[r.OUT/'results'/(r.ARMS[0]+'_same_close_zero_HOLDINGS.json')][0]['fee']=1e-20
            else:ind['paths'][r.ARMS[0]]['same_close_zero']['fees_paid']=1e-20
            with self.assertRaises(ValueError):r.accounting(ev,result,ind,training)

    def test_missing_slice_bad_metrics_and_count_reject(self):
        for change in ('slice','metric','count'):
            ev,result,ind,training,_=self.accounting_fixture()
            if change=='slice':del result['arms'][r.ARMS[0]]['same_close_zero']['slices']['later34']
            elif change=='metric':result['arms'][r.ARMS[0]]['same_close_zero']['metrics']['cagr']+=.001
            else:ind['checked_paths']=9
            with self.assertRaises(ValueError):r.accounting(ev,result,ind,training)

    def test_frozen_fee_helpers_reused_without_new_tolerance(self):
        self.assertEqual(r.FIXED['work/monthly_monotone_tree266_saved_reader.py'],hashlib.sha256(Path(r.shared.__file__).read_bytes()).hexdigest())
        good=[dict(symbols=['A'],signal_date='2020-01-31',execution_date='2020-02-03',fee=1.)]
        tiny=deepcopy(good);tiny[0]['fee']=np.nextafter(1.,2.).item()
        self.assertTrue(r.shared.diagnostic_holdings_parity(tiny,good)['display_fee_only'])
        bad=deepcopy(good);bad[0]['symbols']=['B']
        with self.assertRaises(ValueError):r.shared.diagnostic_holdings_parity(bad,good)


class WholePrefixLedger(unittest.TestCase):
    def test_actual_control_full_cached_work_sums_without_model_fits(self):
        t,m,_,_=synthetic_fixture();n,w=122,49
        t['common_good'][:]=False
        a=dict(baseline=np.zeros((n,w)),eligible=np.zeros((n,w),bool),forecast_available=np.zeros((n,w),bool),raw_source_log=np.full((n,w),np.nan),rank_log_excess=np.zeros((n,w)),
            state_weak_narrow=np.zeros(n,bool),paired_month_ready=np.zeros(n,bool),model_available=np.zeros((n,w),bool),raw_local_linear=np.full((n,w),np.nan),raw_local_constant=np.full((n,w),np.nan),
            effects=np.zeros((n,w)),core=np.zeros((n,w)),core_weight=np.full(n,.52),state_calm10=np.zeros(n,bool))
        b,fc,ma,da,e=bundle(t,m,a);P=np.zeros((n,w));scores={arm:P.copy() for arm in r.ARMS};pres=deepcopy(scores)
        orders={arm:[dict(date=d,end=end,symbols=[]) for d,end in zip(m['monthly']['dates'],m['monthly']['ends'])] for arm in r.ARMS}
        def proofs(count):return {arm:r.selector_check(a,m['monthly'],P,P,orders[arm],count) for arm in r.ARMS}
        e.update(selector=proofs(122),early_selector=proofs(122),full122_reuses_main=True,annual_control_calls=[122,0,1,53,88],
            cached_prefixes=[dict(n=i,exact=True,selector=proofs(i)) for i in range(123)],fresh_prefixes=[])
        for count in (0,1,53,88):
            pt,pm,pa,scope=k.physical_inputs(t,m,a,count);bb,ff,mm,dd,ee=bundle(pt,pm,pa)
            e['fresh_prefixes'].append(dict(n=count,scope=scope,delta_audit=dd,selector=proofs(count),**ee))
        e['work_totals']=dict(prescore_calls=129,selector_calls=258,independent_selector_calls=258,make_orders_calls=258,producer_fits=0,independent_canonical_solves=0,independent_svd_solves=0)
        work=r.preparation_work(e,t,m,a,fc,da,scores,pres,orders)
        self.assertEqual(work['verified_saved_selector_work_summed']['selector_calls'],258)
        self.assertEqual(work['model_invocations'],5);self.assertEqual(work['reader_refits'],0)
        bad=deepcopy(e);bad['work_totals']['selector_calls']=256
        with self.assertRaises(ValueError):r.preparation_work(bad,t,m,a,fc,da,scores,pres,orders)
        bad=deepcopy(e);bad['fresh_prefixes'][-1]['producer_counts']['annual_ridge_fits']=1
        with self.assertRaises(ValueError):r.preparation_work(bad,t,m,a,fc,da,scores,pres,orders)


class BoundFiles(unittest.TestCase):
    def test_registration_dependency_files_and_wrong_pin_reject(self):
        files=list(r.dependency_files())
        self.assertEqual(files,[r.ROOT/name for name in r.FIXED])
        self.assertEqual({str(p.relative_to(r.ROOT)):r.shared.digest(p) for p in files},r.FIXED)
        with patch.object(r.shared,'digest',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'Pinned saved-reader dependency'):
                r.dependency_files()

    def test_phase_missing_extra_hash_and_traversal_reject(self):
        with tempfile.TemporaryDirectory() as td,patch.object(r.shared,'ROOT',Path(td).resolve()):
            folder=Path(td)/'prepared';folder.mkdir();body=folder/'A.json';body.write_text('{}')
            cp=dict(status='COMPLETE',plan_sha256='a'*64,artifacts_sha256={'A.json':r.shared.digest(body)})
            completion=folder/'COMPLETION.json';completion.write_text(json.dumps(cp));sha=r.shared.digest(completion)
            r.Evidence().phase(folder,sha,'a'*64,{'A.json'})
            (folder/'extra.json').write_text('{}')
            with self.assertRaises(ValueError):r.Evidence().phase(folder,sha,'a'*64,{'A.json'})
            (folder/'extra.json').unlink();body.write_text('{"changed":true}')
            with self.assertRaises(ValueError):r.Evidence().phase(folder,sha,'a'*64,{'A.json'})

    def test_explicit_digests_required_before_actual_access(self):
        args=SimpleNamespace(**{k+'_sha256':'invalid' for k in ('plan','prepared','frozen','results','terminals')})
        with patch.object(r.Evidence,'read',side_effect=AssertionError('No reads')):
            with self.assertRaises(ValueError):r.run(args)


if __name__=='__main__':unittest.main()

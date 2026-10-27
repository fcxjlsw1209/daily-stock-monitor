"""Saved272 verifier tests on synthetic arrays/receipts only."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import monthly_roa_increment272_saved_reader as r
import monthly_roa_increment272_runner as h
from test_monthly_roa_increment272_runner import prepared_fixture
from test_monthly_roa_increment272_factor import metrics,Poison


def rows():return {a:dict(same_close_zero={k:v for k,v in metrics(a,.6 if i==2 else .5).items() if k!='name'}) for i,a in enumerate(r.ARMS)}


class AdmissionTests(unittest.TestCase):
    def test_exact_independent_first88_selection(self):
        t=rows();self.assertEqual(r.selection(t),h.select_first88(t));self.assertTrue(r.selection(t)['admitted'])
        for field,limit in (('sharpe',.05),('max_drawdown',.03)):
            t=rows();base=t[r.ARMS[0]]['same_close_zero']['training'][field]
            t[r.ARMS[2]]['same_close_zero']['training'][field]=base-limit
            self.assertTrue(r.selection(t)['admitted'])
            t[r.ARMS[2]]['same_close_zero']['training'][field]-=1e-8
            self.assertFalse(r.selection(t)['admitted'])

    def test_strict_cagr_both_references_and_no_control_promotion(self):
        for role in r.ARMS[:2]:
            t=rows();t[role]['same_close_zero']['training']['cagr']=.6
            self.assertEqual(r.selection(t)['primary'],r.ARMS[0])
        t=rows();t[r.ARMS[2]]['same_close_zero']['training']['cagr']=.5+1e-12
        self.assertFalse(r.selection(t)['admitted'])

    def test_era_gates_and_missing_nan_bool(self):
        for segment in ('early','recent'):
            t=rows();base=t[r.ARMS[0]]['same_close_zero'][segment]['cagr'];t[r.ARMS[2]]['same_close_zero'][segment]['cagr']=base-.025
            self.assertTrue(r.selection(t)['admitted']);t[r.ARMS[2]]['same_close_zero'][segment]['cagr']-=1e-8;self.assertFalse(r.selection(t)['admitted'])
        for value in (None,float('nan'),float('inf'),True):
            t=rows();t[r.ARMS[2]]['same_close_zero']['training']['sharpe']=value;self.assertFalse(r.selection(t)['admitted'])

    def test_later_and_nextopen_never_consumed(self):
        t=rows();self.assertTrue(r.selection(t)['admitted'])  # existing rows contain Poison extras
        t[r.ARMS[0]]['next_open_zero']=Poison()
        with self.assertRaises(ValueError):r.selection(t)


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=prepared_fixture()

    def pieces(self):
        full,proof,source,args=self.fixture
        return deepcopy(full),deepcopy(proof),deepcopy(source),(None,deepcopy(args[1]),deepcopy(args[2]),None,deepcopy(args[4]),None)

    def check(self,full,proof,source,args):
        return r.preparation(proof,full[3],source,args[2],full[5],full[4],args[1]['eligible'],total=5,width=49,prefixes=(0,1,3,4,5),orders=full[0],rowwidth=sum(int(np.prod(v.shape[1:])) for v in args[1].values()))

    def test_exact_real_host_synthetic_fixture(self):
        full,proof,source,args=self.pieces();work=self.check(full,proof,source,args)
        self.assertEqual(work,proof['execution_accounting']);self.assertEqual(r.structured_digest(work),h.structured_digest(work))
        self.assertEqual(work['actual_prepare_entries'],11)

    def test_saved_math_uses_independent_full_values_once(self):
        full,p,_,args=self.pieces();fin=args[4]
        report=r.math_check(args[2],full[4],full[2],full[1],full[0],args[1],fin['catalog'],fin['records'],fin['sessions'],full[5],p['main'])
        self.assertEqual(report['saved_overlay_verifications'],1);self.assertEqual(report['saved_pre_score_verifications'],2);self.assertEqual(report['original_prefix_executions_repeated'],0)

    def test_saved_financial_value_mutation_rejected(self):
        full,p,_,args=self.pieces();fin=args[4];full[4]['candidate_hand'][0,1]+=.01
        with self.assertRaises(ValueError):r.math_check(args[2],full[4],full[2],full[1],full[0],args[1],fin['catalog'],fin['records'],fin['sessions'],full[5],p['main'])

    def test_saved_pre_score_mutation_rejected(self):
        full,p,_,args=self.pieces();fin=args[4];full[2][r.ARMS[2]][0,1]+=.01
        with self.assertRaises(ValueError):r.math_check(args[2],full[4],full[2],full[1],full[0],args[1],fin['catalog'],fin['records'],fin['sessions'],full[5],p['main'])

    def test_saved_order_and_display_scores_bound(self):
        for key in ('symbols','scores'):
            full,p,_,args=self.pieces();fin=args[4]
            full[0][r.ARMS[2]][0][key]=list(reversed(full[0][r.ARMS[2]][0][key]))
            with self.assertRaises(ValueError):r.math_check(args[2],full[4],full[2],full[1],full[0],args[1],fin['catalog'],fin['records'],fin['sessions'],full[5],p['main'])

    def test_balanced_cached_physical_incumbent_counter_forgery(self):
        for location,index in (('cached_prefixes',1),('physical_prefixes',1),('main',None)):
            full,p,s,args=self.pieces();e=p[location] if index is None else p[location][index]
            e['independent'][r.ARMS[0]]['selector']['work']['partial_events']=0
            p['execution_accounting']=h.execution_accounting(p)
            with self.assertRaises(ValueError):self.check(full,p,s,args)

    def test_balanced_overlay_counts_rejected(self):
        full,p,s,args=self.pieces();e=p['cached_prefixes'][1]
        for counts in (e['overlay_counts'],e['overlay_proof']['counts']):counts['ratio_evaluations']+=1
        p['execution_accounting']=h.execution_accounting(p)
        with self.assertRaises(ValueError):self.check(full,p,s,args)

    def test_balanced_pre_score_count_rejected(self):
        full,p,s,args=self.pieces();p['main']['independent'][r.ARMS[1]]['pre_score']['work']['eligible_delta_cells']+=1;p['execution_accounting']=h.execution_accounting(p)
        with self.assertRaises(ValueError):self.check(full,p,s,args)

    def test_alias_claim_source_flag_and_omitted_prefix_rejected(self):
        for mutation in ('alias','source','prefix'):
            full,p,s,args=self.pieces()
            if mutation=='alias':p['physical_prefixes'][-1]['reuses_main_verified_full']=False
            elif mutation=='source':p['main']['source_audit']['both_reference_gates_before_overlay']=False
            else:p['cached_prefixes'].pop(2)
            with self.assertRaises(ValueError):self.check(full,p,s,args)

    def test_zero_prefix_actual_empty_counts(self):
        full,p,s,args=self.pieces();e=p['cached_prefixes'][0]
        r.entry(e,0,49,args[2],full[3],full[5],full[4],args[1]['eligible'],cached=True,orders=full[0])
        self.assertEqual(e['overlay_counts']['catalog_headers_checked'],0);self.assertEqual(e['overlay_counts']['ratio_evaluations'],0)

    def test_scope_arrays_signedzero_nan_payload(self):
        a=np.array([-0.,np.nan]);r.array_exact(a,a.copy(),'copy')
        for position in (0,1):
            b=a.copy();b.view(np.uint64)[position]^=1
            with self.assertRaises(ValueError):r.array_exact(a,b,'bits')


class EvidenceTests(unittest.TestCase):
    def test_terminals_require_true_zero_and_bound_artifact(self):
        hashes=dict(prepare='a'*64,select='b'*64,diagnose='c'*64)
        record=dict(plan_sha256='p',phases={name:dict(session_id=1,exit_code=0,chunk_id='abc',artifact_path=str(r.OUT/path),artifact_sha256=hashes[name],owner='/root') for name,path in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]})
        r.terminals(record,'p',hashes)
        for bad in (True,1):
            d=deepcopy(record);d['phases']['prepare']['exit_code']=bad
            with self.assertRaises(ValueError):r.terminals(d,'p',hashes)
        d=deepcopy(record);d['phases']['prepare']['owner']='elsewhere'
        with self.assertRaises(ValueError):r.terminals(d,'p',hashes)

    def test_accounts_explicit36_paths(self):
        calls=[];proxy=r.Accounts(SimpleNamespace(read=lambda p:calls.append(p)))
        self.assertEqual(len(proxy.paths),36)
        for p,target in proxy.paths.items():proxy.read(p);self.assertTrue(str(target).startswith(str(r.OUT/'results')))
        self.assertEqual(len(calls),36)
        with self.assertRaises(ValueError):proxy.read(r.OUT/'PLAN.json')

    def test_diagnostic_open_groups_added_only_for_accounting(self):
        t=rows();result=dict(arms={a:dict(next_open_zero=dict(slices={k:dict(cagr=.4) for k in ('first88','early53','recent35')})) for a in r.ARMS})
        ind=dict(paths={a:{} for a in r.ARMS});before=deepcopy(t)
        def check(proxy,res,proof,training):
            self.assertEqual(set(training),set(r.old.ARMS))
            self.assertTrue(all(set(v)=={'same_close_zero','next_open_zero'} for v in training.values()))
            return dict(checked_training_groups=18),{}
        with patch.object(r.old,'accounting',side_effect=check):report,_=r.accounting(None,result,ind,t)
        self.assertEqual(report['checked_training_groups'],18);self.assertEqual(r.selection(t),r.selection(before))
        self.assertTrue(all(set(v)=={'same_close_zero'} for v in t.values()))


if __name__=='__main__':unittest.main()

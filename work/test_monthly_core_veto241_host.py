"""Synthetic lifecycle checks for prospectively registered core veto241."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import monthly_core_veto241_host as h

def training_fixture():
    reference=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))
    metric={a:{p:copy.deepcopy(reference) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    for path in metric[h.ARMS[1]]:metric[h.ARMS[1]][path]['training']['cagr']=.52
    orders={a:[dict(symbols=['A','B','C']) for i in range(122)] for a in h.ARMS}
    for i in (1,54,55):orders[h.ARMS[1]][i]['symbols']=['A','B','D']
    return metric,orders




def receipt_fixture():
    ind=SimpleNamespace(SELECTION_FLAGS=('passed','unchanged_core','initial_partial_preserved'))
    def check(n):
        return dict(passed=True,unchanged_core=True,initial_partial_preserved=True,checked_events=n,ordered_choices_checked=3*n,full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False,counts=dict(events=n,new_model_fits=0))
    counts=dict(events=122,new_model_fits=0)
    audit=dict(counts=counts,events=list(range(122)))
    e=dict(selection=check(122),physical_prefixes=[dict(prefix=n,selection=check(n),exact=True,counts=dict(events=n,new_model_fits=0)) for n in h.PREFIXES],all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,new_model_fits=0)
    c=dict(counts=copy.deepcopy(counts),new_model_fits=0,new_returns_computed=False)
    return e,audit,c,ind

class HostTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(len(h.ARMS),3);self.assertEqual(len(h.RESULT_FILES),38);self.assertEqual(len(h.PREP_FILES),7)
        self.assertEqual(h.PRIOR.name,'monthly_peer_rank240')
    def test_candidate_admission(self):
        r=h.select_first88(*training_fixture());self.assertTrue(r['admitted']);self.assertEqual(r['primary'],h.ARMS[1])
    def test_later_orders_irrelevant(self):
        m,o=training_fixture();r=h.select_first88(m,o)
        for a in o:
            for i in range(88,122):o[a][i]['symbols']=['X','Y','Z']
        self.assertEqual(r,h.select_first88(m,o))
    def test_no_episode_control_not_opportunistically_promoted(self):
        m,o=training_fixture();m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.9
        r=h.select_first88(m,o);self.assertFalse(r['admitted']);self.assertEqual(r['primary'],h.ARMS[0])
    def test_open_zero_comparison(self):
        m,o=training_fixture();m[h.ARMS[1]]['next_open_zero']['training']['cagr']=.49
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_gain_not_just_noninferiority(self):
        m,o=training_fixture();m[h.ARMS[1]]['same_close_zero']['training']['cagr']=.504
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_risk_and_split_guards(self):
        for split,key,value in [('training','sharpe',1.64),('training','max_drawdown',-.331),('early','cagr',.374),('recent','cagr',.574)]:
            m,o=training_fixture();m[h.ARMS[1]]['same_close_zero'][split][key]=value
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_both_early_and_recent_changes_required(self):
        m,o=training_fixture();o[h.ARMS[1]][1]['symbols']=['A','B','C']
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_candidate_vs_control_change_support_required(self):
        m,o=training_fixture();o[h.ARMS[2]]=copy.deepcopy(o[h.ARMS[1]])
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_exact_three_arms(self):
        m,o=training_fixture();m.pop(h.ARMS[2])
        with self.assertRaises(ValueError):h.select_first88(m,o)
    def test_initial_state(self):
        completed=[dict(directory='old',trials=5) for i in range(235)]+[dict(directory=h.PRIOR.name,trials=46)]
        s=dict(completed=completed,best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.initial_state(s)
        s['completed'][-1]['trials']+=1
        with self.assertRaises(ValueError):h.initial_state(s)
    def test_registration_recheck(self):
        state=dict(current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='fixed')])
        with patch.object(h,'initial_state'):
            h.registration_check(state,'fixed')
            for key,value in [('current','different'),('registered_queue',[]),('registered_queue',[dict(directory=h.OUT.name,plan_sha256='other')])]:
                x=copy.deepcopy(state);x[key]=value
                with self.assertRaises(ValueError):h.registration_check(x,'fixed')
    def test_receipt_passes(self):h.preparation_check(*receipt_fixture())
    def test_missing_duplicate_prefix(self):
        for ix in ([],[0,1,3,4],[0,1,2,3,3]):
            args=receipt_fixture();args[0]['physical_prefixes']=[args[0]['physical_prefixes'][i] for i in ix]
            with self.assertRaises(ValueError):h.preparation_check(*args)
    def test_counts_and_rows(self):
        for where,key,value in [(1,'events',[]),(2,'counts',dict(events=121,new_model_fits=0))]:
            args=receipt_fixture();args[where][key]=value
            with self.assertRaises(ValueError):h.preparation_check(*args)
    def test_independent_counts_bound(self):
        for nested in (False,True):
            args=receipt_fixture();x=args[0]['physical_prefixes'][2] if nested else args[0]
            x['selection']['counts']['unverified']=1
            with self.assertRaises(ValueError):h.preparation_check(*args)
    def test_false_selection_or_audit_flag(self):
        for nested in (False,True):
            for key in ('passed','unchanged_core','initial_partial_preserved','full_audit_supplied','full_scores_supplied','full_pre_scores_supplied'):
                args=receipt_fixture();x=args[0]['physical_prefixes'][2] if nested else args[0];x['selection'][key]=False
                with self.assertRaises(ValueError):h.preparation_check(*args)
    def test_exact_independent_scope(self):
        for key,value in [('numerical_tolerance',1e-6),('producer_modules_imported',True),('computes_returns',True),('ordered_choices_checked',365)]:
            args=receipt_fixture();args[0]['selection'][key]=value
            with self.assertRaises(ValueError):h.preparation_check(*args)
    def test_preservation_and_no_fits(self):
        for key,value in [('original_eligible_unchanged',False),('original_arrays_unchanged',False),('all123_order_prefixes',False),('new_model_fits',1)]:
            args=receipt_fixture();args[0][key]=value
            with self.assertRaises(ValueError):h.preparation_check(*args)
    def test_results_accounting_scope(self):
        p=dict(passed=True,monthly_executions=122,slices={k:{} for k in ('full122','first88','early53','recent35','later34')})
        x=dict(passed=True,checked_paths=9,checked_slices=45,independent_accounts=True,independent_of_production_model_and_engine=True,terminal_liquidation=False,paths={a:{k:copy.deepcopy(p) for k in ('same_close_zero','next_open_zero','next_open_cost')} for a in h.ARMS})
        h.outcome_check(x)
        x['paths'][h.ARMS[2]]['next_open_cost']['monthly_executions']=121
        with self.assertRaises(ValueError):h.outcome_check(x)

if __name__=='__main__':unittest.main()

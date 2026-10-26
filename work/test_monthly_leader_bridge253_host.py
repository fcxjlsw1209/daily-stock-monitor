"""Synthetic bridge lifecycle and prospective selection checks; no market replay."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import monthly_leader_bridge253_host as h

def fixture():
    r=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))
    m={a:{p:copy.deepcopy(r) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    for p in m[h.ARMS[1]]:m[h.ARMS[1]][p]['training']['cagr']=.52
    o={a:[dict(symbols=['A','B','C']) for _ in range(122)] for a in h.ARMS}
    for i in (1,54,55):o[h.ARMS[1]][i]['symbols']=['A','B','D']
    return m,o

def source(n):
    return dict(**{k:True for k in h.ref_ind.SOURCE_GATE_FLAGS},checked_events=n,episode_calls=0,new_target_ranks_computed=False,new_basket_selection_computed=False,C_source_atol=1e-12,C_source_rtol=0.,continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,producer_modules_imported=False,computes_returns=False,candidate_selection_computed=False)
def inc(n):
    return dict(**{k:True for k in h.ref_ind.INCUMBENT_GATE_FLAGS},checked_events=n,ordered_choices_checked=n,numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False,candidate_selection_computed=False)
def endpoints(n):
    return dict(passed=True,endpoint_checked=True,checked_events=n,ordered_choices_checked=2*n,numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False,candidate_selection_computed=False)
def ind():return SimpleNamespace(ENDPOINT_FLAGS=('passed','endpoint_checked'))
def gates(n,full):
    d=dict(incumbent=inc(n),endpoints=endpoints(n))
    if full:d['source']=source(n)
    return d

class HostTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(h.ARMS,('incumbent_local67','long_leader_local67','basket_core251_control'))
        self.assertEqual((len(h.PREP_FILES),len(h.RESULT_FILES),len(h.code_names)),(8,38,6))
        self.assertEqual(h.PRIOR.name,'monthly_peer_path252_v2')
    def test_candidate_selected_on_training_only(self):
        m,o=fixture();v=h.select_first88(m,o);self.assertTrue(v['admitted']);self.assertEqual(v['primary'],h.ARMS[1])
        for a in o:
            for i in range(88,122):o[a][i]['symbols']=['X','Y','Z']
        self.assertEqual(v,h.select_first88(m,o))
    def test_diagnostic_never_promoted(self):
        m,o=fixture()
        for p in m[h.ARMS[2]]:m[h.ARMS[2]][p]['training'].update(cagr=10.,sharpe=9.,max_drawdown=0.)
        self.assertEqual(h.select_first88(m,o)['primary'],h.ARMS[1])
        m[h.ARMS[1]]['same_close_zero']['training']['cagr']=.49
        self.assertEqual(h.select_first88(m,o)['primary'],h.ARMS[0])
    def test_close_and_open_gates(self):
        for p,v in [('same_close_zero',.504),('next_open_zero',.49)]:
            m,o=fixture();m[h.ARMS[1]][p]['training']['cagr']=v
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_risk_and_segments_both_timings(self):
        for p in ('same_close_zero','next_open_zero'):
            for g,k,v in [('training','sharpe',1.64),('training','max_drawdown',-.331),('early','cagr',.374),('recent','cagr',.574)]:
                m,o=fixture();m[h.ARMS[1]][p][g][k]=v
                self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_only_cagr_inherits_roundoff_tolerance(self):
        m,o=fixture();m[h.ARMS[1]]['same_close_zero']['training']['cagr']=.505-5e-13
        self.assertTrue(h.select_first88(m,o)['admitted'])
        m[h.ARMS[1]]['same_close_zero']['training']['cagr']=.505-2e-12
        self.assertFalse(h.select_first88(m,o)['admitted'])
        m,o=fixture();m[h.ARMS[1]]['same_close_zero']['training']['sharpe']=1.65-5e-13
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_changed_set_and_eras_required(self):
        for ids in ((54,55),(1,2,3),(53,54,55)):
            m,o=fixture();o[h.ARMS[1]]=copy.deepcopy(o[h.ARMS[0]])
            for i in ids:o[h.ARMS[1]][i]['symbols']=['A','B','D']
            self.assertFalse(h.select_first88(m,o)['admitted'])
        m,o=fixture()
        for i in (1,54,55):o[h.ARMS[1]][i]['symbols']=['C','B','A']
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_required_metrics(self):
        for a in h.ARMS:
            for v in (True,'0.7',float('nan'),float('inf')):
                m,o=fixture();m[a]['same_close_zero']['training']['cagr']=v
                with self.assertRaises(ValueError):h.select_first88(m,o)
        for extra in ('later34','next_open_cost'):
            m,o=fixture();m[h.ARMS[1]][extra]={}
            with self.assertRaises(ValueError):h.select_first88(m,o)
    def test_undefined_sharpe_treatment(self):
        for a in h.ARMS:
            m,o=fixture();m[a]['same_close_zero']['training'].update(sharpe=None,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd',daily_return_sd=0.)
            if a==h.ARMS[0]:
                with self.assertRaises(ValueError):h.select_first88(m,o)
            else:self.assertEqual(h.select_first88(m,o)['admitted'],a==h.ARMS[2])
    def test_counter_and_queue(self):
        s=dict(completed=[dict(directory='old',trials=5) for _ in range(247)]+[dict(directory=h.PRIOR.name,trials=22)])
        h.initial_state(s)
        s.update(current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='x')]);h.registration_check(s,'x')
        for k,v in [('current','bad'),('registered_queue',[])]:
            d=copy.deepcopy(s);d[k]=v
            with self.assertRaises(ValueError):h.registration_check(d,'x')
        s['completed'][-1]['trials']=21
        with self.assertRaises(ValueError):h.initial_state(s)
    def test_predecessor_closure_digest(self):
        s=dict(completed=[dict(completion=dict(path=str(h.PRIOR/'COMPLETION.json'),sha256='x'))])
        with patch.object(h,'sha',return_value='x'):
            h.predecessor_binding(s);s['completed'][-1]['completion']['sha256']='bad'
            with self.assertRaises(ValueError):h.predecessor_binding(s)
    def test_mandatory_three_gates(self):
        for full in (False,True):
            g=gates(5,full);h.pre_gate_check(g,5,ind(),full)
            for key in g:
                bad=copy.deepcopy(g);bad.pop(key)
                with self.assertRaises(ValueError):h.pre_gate_check(bad,5,ind(),full)
    def test_endpoint_gate_rejects_missing_scope(self):
        for k,v in [('endpoint_checked',False),('ordered_choices_checked',5),('checked_events',4),('numerical_tolerance',1e-12),('candidate_selection_computed',True),('computes_returns',True),('producer_modules_imported',True)]:
            g=gates(5,True);g['endpoints'][k]=v
            with self.assertRaises(ValueError):h.pre_gate_check(g,5,ind(),True)
    def test_callbacks_are_ordered(self):
        i=ind();i.verify_endpoints=lambda *a,**k:endpoints(3)
        with patch.object(h.ref_ind,'verify_source_features',return_value=source(3)),patch.object(h.ref_ind,'verify_incumbent',return_value=inc(3)):
            sg,ig,eg,g=h.gate_callbacks(i,{}, {}, {}, [], {}, [],3,True)
            with self.assertRaises(ValueError):ig([],[],[])
            with self.assertRaises(ValueError):eg({}, {}, {})
            sg([]);ig([],[],[]);eg({}, {}, {});h.pre_gate_check(g,3,i,True)
            with self.assertRaises(ValueError):sg([])
            with self.assertRaises(ValueError):ig([],[],[])
            with self.assertRaises(ValueError):eg({}, {}, {})
    def test_full_saved_review_required(self):
        paths=[h.OUT/'PLAN.json',h.CONTEXT,h.OUT/'prepared/COMPLETION.json',h.OUT/'selected/FROZEN.json',h.OUT/'selected/TRAIN_RESULTS.json',h.OUT/'results/COMPLETION.json']+[h.OUT/'prepared'/x for x in h.PREP_FILES]+[h.OUT/'results'/x for x in h.RESULT_FILES]
        r=dict(passed=True,unresolved_blockers=[],plan_sha256='x',context_sha256='x',prepared_completion_sha256='x',results_completion_sha256='x',input_sha256={str(p):'x' for p in paths})
        with patch.object(h,'sha',return_value='x'),patch.object(h.old,'verify'):
            with patch.object(h,'read',return_value=r):h.result_review_check('x')
            for p in paths:
                v=copy.deepcopy(r);v['input_sha256'].pop(str(p))
                with patch.object(h,'read',return_value=v),self.assertRaises(ValueError):h.result_review_check('x')
    def test_failure_receipt_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'PLAN.json').write_text('{}')
            with patch.object(h,'OUT',p),patch.object(h,'sha',return_value='x'),patch.object(h,'prepare',side_effect=ValueError('bad source')):
                with self.assertRaises(ValueError):h.prepare_recorded('x')
                first=(p/'PREPARE_FAILURE.json').read_bytes()
                with self.assertRaises(ValueError):h.prepare_recorded('x')
                self.assertEqual(first,(p/'PREPARE_FAILURE.json').read_bytes())
if __name__=='__main__':unittest.main()

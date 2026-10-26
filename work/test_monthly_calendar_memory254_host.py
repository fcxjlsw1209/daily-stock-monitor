"""Synthetic paired-memory lifecycle and prospective selection checks; no market replay."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import monthly_calendar_memory254_host as h

def fixture():
    r=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))
    m={a:{p:copy.deepcopy(r) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    for p in m[h.ARMS[1]]:m[h.ARMS[1]][p]['training']['cagr']=.52
    o={a:[dict(symbols=['A','B','C']) for _ in range(122)] for a in h.ARMS}
    for i in (1,54,55):o[h.ARMS[1]][i]['symbols']=['A','B','D']
    return m,o

def receipt(n,kind):
    return dict(passed=True,checked_events=n,producer_modules_imported=False,computes_returns=False,
        ordered_choices_checked=(3*n if kind=='selection' else n),verified=True,
        episode_calls=0,memory_effects_computed=False,candidate_selection_computed=False,full_source_audit_supplied=True,
        continuous_atol=2e-10,continuous_rtol=2e-10,exact_discrete=True,full_scores_supplied=True,full_pre_scores_supplied=True,
        numerical_tolerance=0.,full_effects_supplied=True,full_audit_supplied=True,source_data_sha256=h.data_digest({}))
def ind():
    return SimpleNamespace(SOURCE_FLAGS=('passed','verified'),INCUMBENT_FLAGS=('passed','verified'),SELECTION_FLAGS=('passed','verified'))
def gates(n,cached=False):
    source=receipt(n,'source') if not cached else dict(passed=True,checked_events=n,cached_source_reference=True,full_source_checked_events=122,no_new_source_rebuild_claim=True,full_source_receipt_sha256=h.digest_record(receipt(122,'source')),full_source_data_sha256=h.data_digest({}))
    return dict(source=source,incumbent=receipt(n,'incumbent'))

class HostTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(h.ARMS,('incumbent_local67','calendar_memory_local67','fixed21_memory_control'))
        self.assertEqual((len(h.PREP_FILES),len(h.RESULT_FILES),len(h.code_names)),(10,38,6))
        self.assertEqual(h.PRIOR.name,'monthly_leader_bridge253')
    def test_candidate_selected_on_training_only(self):
        m,o=fixture();v=h.select_first88(m,o);self.assertTrue(v['admitted']);self.assertEqual(v['primary'],h.ARMS[1])
        for a in o:
            for i in range(88,122):o[a][i]['symbols']=['X','Y','Z']
        self.assertEqual(v,h.select_first88(m,o))
    def test_diagnostic_never_promoted(self):
        m,o=fixture()
        for p in m[h.ARMS[2]]:m[h.ARMS[2]][p]['training'].update(cagr=10.,sharpe=9.,max_drawdown=0.)
        self.assertEqual(h.select_first88(m,o)['primary'],h.ARMS[0])
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
            else:self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_counter_and_queue(self):
        s=dict(completed=[dict(directory='old',trials=5) for _ in range(248)]+[dict(directory=h.PRIOR.name,trials=20)])
        h.initial_state(s)
        s.update(current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='x')]);h.registration_check(s,'x')
        for k,v in [('current','bad'),('registered_queue',[])]:
            d=copy.deepcopy(s);d[k]=v
            with self.assertRaises(ValueError):h.registration_check(d,'x')
        s['completed'][-1]['trials']=19
        with self.assertRaises(ValueError):h.initial_state(s)
    def test_predecessor_closure_digest(self):
        s=dict(completed=[dict(completion=dict(path=str(h.PRIOR/'COMPLETION.json'),sha256='x'))])
        with patch.object(h,'sha',return_value='x'):
            h.predecessor_binding(s);s['completed'][-1]['completion']['sha256']='bad'
            with self.assertRaises(ValueError):h.predecessor_binding(s)
    def test_required_two_gates(self):
        for cached in (False,True):
            g=gates(5,cached);h.pre_gate_check(g,5,ind(),cached=cached,full_source=receipt(122,'source'))
            for key in g:
                bad=copy.deepcopy(g);bad.pop(key)
                with self.assertRaises(ValueError):h.pre_gate_check(bad,5,ind(),cached=cached,full_source=receipt(122,'source'))
    def test_source_scope(self):
        for key,val in [('verified',False),('checked_events',4),('computes_returns',True),('producer_modules_imported',True)]:
            g=gates(5);g['source'][key]=val
            with self.assertRaises(ValueError):h.pre_gate_check(g,5,ind())
    def test_callbacks_order(self):
        i=ind();i.verify_source=lambda *a,**k:receipt(3,'source');i.verify_incumbent=lambda *a,**k:receipt(3,'incumbent')
        sg,ig,g=h.gate_callbacks(i,{}, {}, {}, [], {},{},3)
        with self.assertRaises(ValueError):ig([],[],[])
        sg({});ig([],[],[]);h.pre_gate_check(g,3,i)
        with self.assertRaises(ValueError):sg({})
        with self.assertRaises(ValueError):ig([],[],[])
    def test_source_cached_reference_not_rebuild(self):
        i=ind();i.verify_source=lambda *a,**k: self.fail('cached source must not claim independent recomputation')
        i.verify_incumbent=lambda *a,**k:receipt(3,'incumbent')
        sg,ig,g=h.gate_callbacks(i,{}, {},None,None,{},None,3,cached_source=receipt(122,'source'))
        sg({});ig([],[],[]);h.pre_gate_check(g,3,i,cached=True,full_source=receipt(122,'source'))
    def test_calendar_vs_fixed21_each_path_and_sharpe(self):
        for path in ('same_close_zero','next_open_zero'):
            m,o=fixture();m[h.ARMS[2]][path]['training']['cagr']=.53
            self.assertFalse(h.select_first88(m,o)['guards']['calendar_vs_fixed21'])
        m,o=fixture();m[h.ARMS[2]]['same_close_zero']['training']['sharpe']=1.7000000000001
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_comparator_open_undefined_not_new_admission_condition(self):
        m,o=fixture();m[h.ARMS[2]]['next_open_zero']['training'].update(sharpe=None,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd',daily_return_sd=0.)
        self.assertTrue(h.select_first88(m,o)['admitted'])
    def test_comparator_cagr_roundoff_only(self):
        m,o=fixture();m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.52+5e-13
        self.assertTrue(h.select_first88(m,o)['admitted'])
        m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.52+2e-12
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_kind_scope_rejections(self):
        for kind,changes in [('source',{'episode_calls':1,'memory_effects_computed':True,'candidate_selection_computed':True,'full_source_audit_supplied':False,'continuous_atol':1e-8,'exact_discrete':False}),('incumbent',{'ordered_choices_checked':0,'full_scores_supplied':False,'full_pre_scores_supplied':False,'numerical_tolerance':1e-12,'candidate_selection_computed':True}),('selection',{'ordered_choices_checked':1,'full_effects_supplied':False,'full_audit_supplied':False})]:
            for k,v in changes.items():
                d=receipt(3,kind);d[k]=v
                with self.assertRaises(ValueError):h.receipt_check(d,3,('passed','verified'),kind)
    def test_cached_source_binding(self):
        i=ind();i.verify_incumbent=lambda *a,**k:receipt(3,'incumbent')
        g=gates(3,True);wrong=receipt(122,'source');wrong['source_data_sha256']='wrong'
        with self.assertRaises(ValueError):h.pre_gate_check(g,3,i,cached=True,full_source=wrong)
        sg,ig,g=h.gate_callbacks(i,{}, {},None,None,{},None,3,cached_source=wrong)
        with self.assertRaises(ValueError):sg({})
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

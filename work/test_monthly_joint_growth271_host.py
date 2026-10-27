"""Synthetic lifecycle guards, selection separation and exact zero-fee assertions."""
import copy
import unittest
from unittest.mock import patch
import monthly_joint_growth271_host as h


def fixtures():
    one={'training':{'cagr':.30,'sharpe':1.5,'max_drawdown':-.2},'early':{'cagr':.25},'recent':{'cagr':.35}}
    metrics={a:{path:copy.deepcopy(one) for path in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    orders={a:[{'symbols':['A','B','C']} for _ in range(122)] for a in h.ARMS}
    for a in h.ARMS[1:]:
        for path in metrics[a]:metrics[a][path]['training']['cagr']=.32
        for i in (2,30,70):orders[a][i]['symbols']=['A','B','D']
    return metrics,orders


class SelectionTests(unittest.TestCase):
    def test_exact_tie_prefers_member(self):
        m,o=fixtures();r=h.select_first88(m,o);self.assertEqual(r['primary'],'member_log_growth');self.assertFalse(r['later34_used']);self.assertFalse(r['cost_used'])
    def test_higher_training_return_wins(self):
        m,o=fixtures();m['joint_log_growth']['same_close_zero']['training']['cagr']+=1e-14
        self.assertEqual(h.select_first88(m,o)['primary'],'joint_log_growth')
    def test_exact_return_tie_higher_sharpe_wins(self):
        m,o=fixtures();m['joint_log_growth']['same_close_zero']['training']['sharpe']+=.01
        self.assertEqual(h.select_first88(m,o)['primary'],'joint_log_growth')
    def test_later34_not_read(self):
        m,o=fixtures();expected=h.select_first88(m,o)
        for a in h.ARMS:o[a][88:]=[object() for _ in range(34)]
        self.assertEqual(h.select_first88(m,o),expected)
    def test_open_zero_guard_matters(self):
        m,o=fixtures()
        for a in h.ARMS[1:]:m[a]['next_open_zero']['training']['cagr']=.29
        self.assertEqual(h.select_first88(m,o)['primary'],'incumbent_local67')
    def test_training_return_minimum(self):
        m,o=fixtures()
        for a in h.ARMS[1:]:m[a]['same_close_zero']['training']['cagr']=.304
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_risk_guard(self):
        for key,value in [('sharpe',1.44),('max_drawdown',-.231)]:
            m,o=fixtures()
            for a in h.ARMS[1:]:m[a]['same_close_zero']['training'][key]=value
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_subperiod_guard(self):
        m,o=fixtures()
        for a in h.ARMS[1:]:m[a]['next_open_zero']['early']['cagr']=.224
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_support_requires_three_and_both_subperiods(self):
        m,o=fixtures()
        for a in h.ARMS[1:]:o[a][70]['symbols']=['A','B','C'];o[a][35]['symbols']=['A','B','D']
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_nonfinite_required_metric_fails(self):
        m,o=fixtures();m[h.ARMS[1]]['same_close_zero']['training']['cagr']=float('nan')
        with self.assertRaises(ValueError):h.select_first88(m,o)
    def test_undefined_experimental_sharpe_blocks(self):
        m,o=fixtures()
        for a in h.ARMS[1:]:m[a]['same_close_zero']['training'].update(sharpe=None,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd',daily_return_sd=0.)
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_no_charged_path_in_selection(self):
        m,o=fixtures();m[h.ARMS[1]]['next_open_cost']=copy.deepcopy(m[h.ARMS[1]]['same_close_zero'])
        with self.assertRaises(ValueError):h.select_first88(m,o)
    def test_wrong_counter_fails_before_closure_read(self):
        with patch.object(h,'sha',side_effect=AssertionError('No file before counter check')):
            with self.assertRaises(ValueError):h.initial_state({'completed':[]})


class AccountingGuardTests(unittest.TestCase):
    def fixture(self):
        exports={};results={}
        for a in h.ARMS:
            exports[a]={};results[a]={}
            for p in ('same_close_zero','next_open_zero'):
                exports[a][p]={'TRADES':[{'cost':0.}],'HOLDINGS':[{'fee':0.}]}
                results[a][p]={'metrics':{'cost_per_side':0.,'fees_paid':0.},'slices':{'full122':{'cost_per_side':0.,'fees_paid':0.}}}
        return exports,results
    def test_exact_zero_paths(self):
        x,r=self.fixture();h.zero_cost_check(x,r)
    def test_nonzero_or_bool_fee_rejected(self):
        for bad in (1e-16,True,None):
            x,r=self.fixture();x[h.ARMS[0]]['same_close_zero']['TRADES'][0]['cost']=bad
            with self.assertRaises(ValueError):h.zero_cost_check(x,r)
    def test_zero_parameter_with_nonzero_total_rejected(self):
        x,r=self.fixture();r[h.ARMS[1]]['next_open_zero']['metrics']['fees_paid']=.01
        with self.assertRaises(ValueError):h.zero_cost_check(x,r)
    def test_structured_exact_checks_types_and_signed_zero(self):
        for a,b in [(1,True),(0.,-0.),({'a':1},{'a':1,'b':2})]:
            with self.assertRaises(ValueError):h.exact(a,b,'synthetic')
    def test_explicit_owned_phases_required(self):
        with patch.object(h,'read',return_value={'plan_sha256':'x','phases':{}}):
            with self.assertRaises(ValueError):h.phase_terminals_check('x')
    def test_incomplete_path_count_rejected(self):
        c={'passed':True,'checked_paths':8,'checked_slices':45,'checked_training_groups':18,'independent_accounts':True,'independent_of_production_model_and_engine':True,'terminal_liquidation':False,'paths':{}}
        with self.assertRaises(ValueError):h.outcome_check(c)



class LifecycleTests(unittest.TestCase):
    def test_unfrozen_diagnostics_never_reads_prices(self):
        with patch.object(h,'prepared',return_value=({},{})),patch.object(h,'frozen',side_effect=ValueError('no frozen selection')),patch.object(h.ev,'prices',side_effect=AssertionError('no outcome prices')):
            with self.assertRaisesRegex(ValueError,'no frozen'):h.diagnose('synthetic')
    def complete_outcome(self):
        proof={'passed':True,'monthly_executions':122,'slices':dict.fromkeys(('full122','first88','early53','recent35','later34'),{})}
        return dict(passed=True,checked_paths=9,checked_slices=45,checked_training_groups=18,independent_accounts=True,independent_of_production_model_and_engine=True,terminal_liquidation=False,paths={a:{p[0]:copy.deepcopy(proof) for p in h.ev.PATHS} for a in h.ARMS})
    def test_complete_nine_path_scope(self):h.outcome_check(self.complete_outcome())
    def test_missing_single_slice_path_or_month_rejected(self):
        for mode in ('slice','path','month'):
            proof=self.complete_outcome();path=proof['paths'][h.ARMS[1]]['next_open_zero']
            if mode=='slice':del path['slices']['later34']
            elif mode=='path':del proof['paths'][h.ARMS[1]]['next_open_zero']
            else:path['monthly_executions']=121
            with self.assertRaises(ValueError):h.outcome_check(proof)
    def test_independent_group_scope(self):
        c=self.complete_outcome();c['checked_training_groups']=17
        with self.assertRaises(ValueError):h.outcome_check(c)
    def test_drifted_code_never_registers(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as tmp:
            fake=Path(tmp)/'absent'
            with patch.object(h,'OUT',fake),patch.object(h,'context_check',side_effect=ValueError('code digest changed')),patch.object(h,'code_review',side_effect=AssertionError('no later review')):
                with self.assertRaisesRegex(ValueError,'digest'):h.register()
                self.assertFalse(fake.exists())
    def test_saved_decision_audit_mutation_rejected(self):
        import monthly_joint_growth271_kernel as k
        def prices(g):return [float(100*(g**(j//21))) for j in range(253)]
        events=[dict(date='2020-01-31',symbols=['A','B','C','D'],eligible=[True]*4,prices253=[prices(g) for g in (1.04,1.03,1.02,1.01)],incumbent_choices=[0,1,2])]
        choices={};audits={};orders={};meta={'dates':['2020-01-31'],'ends':['2020-02-28']}
        for a in h.ARMS:
            choices[a],audits[a]=k.select(events,a)
            orders[a]=[dict(date=meta['dates'][0],signal_date=meta['dates'][0],end=meta['ends'][0],symbols=[events[0]['symbols'][j] for j in choices[a][0]],scores=[3.,2.,1.])]
        h.verify_decisions(events,orders,audits,meta)
        changed=copy.deepcopy(orders);changed[h.ARMS[1]][0]['symbols']=['A','B','D']
        with self.assertRaises(ValueError):h.verify_decisions(events,changed,audits,meta)
        altered=copy.deepcopy(audits);altered[h.ARMS[2]]['counts']['required_cells']+=1
        with self.assertRaises(ValueError):h.verify_decisions(events,orders,altered,meta)

if __name__=='__main__':unittest.main(verbosity=2)

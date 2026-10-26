"""Synthetic-only orchestration tests; no historical arrays or price sources."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import monthly_peer_rank240_host as h


def score_fixture():
    base=np.array([[.4,.3,.2,.195,.18]])
    a=dict(baseline=base,core=np.zeros_like(base),core_weight=np.array([.5]),eligible=np.array([[True,True,True,True,True]]),effects=np.zeros_like(base),state_calm10=np.array([False]),state_weak_narrow=np.array([False]))
    # Keep inherited bounded support.
    a['core']=base-.5*np.array([[.2,.1,.0,-.1,-.2]])
    m=dict(dates=['2016-08-31'],ends=['2016-09-30'],symbols=[['A','B','C','D','QQQ']],etfs=['QQQ'])
    v=dict(mature=np.array([[True,True,True,True,False]]),supported=np.array([[True,True,True,True,False]]))
    for k in h.KEYS[2:]:v[k]=np.array([[.1,.2,.3,.4,np.nan]])
    real=h.old.local()
    study=SimpleNamespace(modified_pre_score=lambda spec,arrays,n:arrays['baseline'].copy(),configs=lambda:[{}]*5,adapter=real.adapter,ep=real.ep,rp=real.rp)
    return study,a,m,v


def training_fixture():
    reference=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))
    metric={a:{p:copy.deepcopy(reference) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    for path in metric[h.ARMS[1]]:metric[h.ARMS[1]][path]['training']['cagr']=.52
    orders={a:[dict(symbols=['A','B','C']) for i in range(122)] for a in h.ARMS}
    for i in (1,54,55):orders[h.ARMS[1]][i]['symbols']=['A','B','D']
    return metric,orders



def receipt_fixture():
    import monthly_peer_rank240_independent as ind
    values={k:np.ones((122,2),dtype=bool if k in h.KEYS[:2] else float) for k in h.KEYS}
    records=[dict(query_index=i,history_status='observed_full_window',initial_partial=False,structural_peer_support=True,deferred_gap_reasons=[],gap_reasons=[]) for i in range(122) for j in range(2)]
    source=dict(events=list(range(122)),records=records,gaps=[],counts=dict(mature_cells=244,supported_cells=244))
    def feature(n):
        counts=dict(events=n,stock_cells=2*n,mature_cells=2*n,supported_cells=2*n,endpoint_observations=4*n,new_model_fits=0)
        return dict(**{k:True for k in ind.FEATURE_FLAGS},counts=counts,numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False)
    def selection(n):
        return dict(**{k:True for k in ind.SELECTION_FLAGS},checked_events=n,ordered_choices_checked=3*n,full_scores_supplied=True,full_pre_scores_supplied=True,numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False)
    e=dict(feature=feature(122),selection=selection(122),physical_prefixes=[dict(prefix=n,feature=feature(n),selection=selection(n),exact=True) for n in h.PREFIXES],all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,source_inventory_preserved=True,new_model_fits=0)
    audit=dict(counts=copy.deepcopy(e['feature']['counts']));completion=dict(counts=copy.deepcopy(audit['counts']),new_model_fits=0,new_returns_computed=False)
    return e,values,audit,completion,source,ind

class HostTests(unittest.TestCase):
    def test_three_arm_nine_path_artifact_scope(self):
        self.assertEqual(len(h.ARMS),3);self.assertEqual(len(h.RESULT_FILES),38)
        self.assertEqual(len(h.PREP_FILES),6)
    def test_valid_training_candidate_and_control_guards(self):
        r=h.select_first88(*training_fixture());self.assertTrue(r['admitted']);self.assertEqual(r['primary'],h.ARMS[1])
    def test_later_months_cannot_change_selection(self):
        m,o=training_fixture();before=h.select_first88(m,o)
        for a in o:
            for i in range(88,122):o[a][i]['symbols']=['X','Y','Z']
        self.assertEqual(before,h.select_first88(m,o))
    def test_control_outperforms_peer_prevents_attribution(self):
        m,o=training_fixture();m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.53
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_next_open_zero_guard_matters(self):
        m,o=training_fixture();m[h.ARMS[1]]['next_open_zero']['training']['cagr']=.49
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_recent_only_changes_not_enough(self):
        m,o=training_fixture();o[h.ARMS[1]][1]['symbols']=['A','B','C']
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_sharpe_drawdown_early_guard(self):
        for field,value in [('sharpe',1.6),('max_drawdown',-.34)]:
            m,o=training_fixture();m[h.ARMS[1]]['same_close_zero']['training'][field]=value
            self.assertFalse(h.select_first88(m,o)['admitted'])
        m,o=training_fixture();m[h.ARMS[1]]['same_close_zero']['early']['cagr']=.37
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_all_na_targets_exact_incumbent_and_arrays_unchanged(self):
        s,a,m,v=score_fixture();v['supported'][:]=False
        for k in ('peer_median','peer_target','absolute_target'):v[k][:]=np.nan
        before=h.old.array_snapshot(a);orders,scores,pres=h.decisions(s,a,m,v,1)
        self.assertEqual(before,h.old.array_snapshot(a));self.assertEqual(orders[h.ARMS[0]],orders[h.ARMS[1]])
        for arm in h.ARMS:np.testing.assert_array_equal(pres[arm],a['baseline'])
    def test_zero_target_is_not_missing(self):
        s,a,m,v=score_fixture();v['peer_target'][0,:4]=0.
        o,sc,pr=h.decisions(s,a,m,v,1)
        support=(a['baseline']-a['core'])/(1-a['core_weight'][:,None])
        expected=a['baseline'].copy();delta=.05*(v['peer_target']-support);mask=np.isfinite(delta)&(delta!=0);expected[mask]+=delta[mask]
        np.testing.assert_array_equal(pr[h.ARMS[1]],expected);self.assertNotEqual(pr[h.ARMS[1]][0,0],pr[h.ARMS[0]][0,0])
        self.assertEqual(pr[h.ARMS[1]][0,4],pr[h.ARMS[0]][0,4])
    def test_original_eligibility_preserved(self):
        s,a,m,v=score_fixture();a['eligible'][0,0]=False
        before=a['eligible'].copy();o,sc,pr=h.decisions(s,a,m,v,1)
        np.testing.assert_array_equal(a['eligible'],before)
        self.assertTrue(all('A' not in o[arm][0]['symbols'] for arm in h.ARMS))
    def test_missing_mature_value_rejected(self):
        s,a,m,v=score_fixture();v['momentum'][0,1]=np.nan
        with self.assertRaises(ValueError):h.decisions(s,a,m,v,1)
    def test_out_of_bounds_target_rejected(self):
        s,a,m,v=score_fixture();v['peer_target'][0,1]=.50001
        with self.assertRaises(ValueError):h.decisions(s,a,m,v,1)
    def test_non_boolean_availability_rejected(self):
        s,a,m,v=score_fixture();v['mature']=v['mature'].astype(float)
        with self.assertRaises(ValueError):h.decisions(s,a,m,v,1)
    def test_whole_selection_arm_set_required(self):
        m,o=training_fixture();m.pop(h.ARMS[2])
        with self.assertRaises(ValueError):h.select_first88(m,o)


    def test_registration_queue_and_current_rechecked(self):
        state=dict(current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='fixed')])
        with patch.object(h,'initial_state'):
            h.registration_check(state,'fixed')
            for key,value in [('current','different'),('registered_queue',[]),('registered_queue',[dict(directory=h.OUT.name,plan_sha256='other')])]:
                wrong=copy.deepcopy(state);wrong[key]=value
                with self.assertRaises(ValueError):h.registration_check(wrong,'fixed')

    def test_prepared_receipt_valid(self):
        h.preparation_check(*receipt_fixture())

    def test_prepared_missing_duplicate_or_empty_prefix_rejected(self):
        for indices in ([],[0,1,3,4],[0,1,2,3,3]):
            args=receipt_fixture();e=args[0];e['physical_prefixes']=[e['physical_prefixes'][i] for i in indices]
            with self.assertRaises(ValueError):h.preparation_check(*args)

    def test_prepared_false_feature_or_selection_flag_rejected(self):
        for where in ('feature','selection'):
            for prefix in (False,True):
                args=receipt_fixture();e=args[0]['physical_prefixes'][2] if prefix else args[0];e[where]['passed']=False
                with self.assertRaises(ValueError):h.preparation_check(*args)

    def test_prepared_incomplete_count_rejected(self):
        for prefix in (False,True):
            args=receipt_fixture();e=args[0]['physical_prefixes'][2] if prefix else args[0];e['feature']['counts']['stock_cells']-=1
            with self.assertRaises(ValueError):h.preparation_check(*args)

    def test_prepared_preservation_and_no_fits_required(self):
        for key,value in [('original_eligible_unchanged',False),('source_inventory_preserved',False),('new_model_fits',1)]:
            args=receipt_fixture();args[0][key]=value
            with self.assertRaises(ValueError):h.preparation_check(*args)

    def test_prepared_independent_and_exact_scope_required(self):
        for key,value in [('numerical_tolerance',1e-6),('producer_modules_imported',True),('computes_returns',True)]:
            args=receipt_fixture();args[0]['feature'][key]=value
            with self.assertRaises(ValueError):h.preparation_check(*args)

if __name__=='__main__':unittest.main()

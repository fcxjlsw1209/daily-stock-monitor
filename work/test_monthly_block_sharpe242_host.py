"""Synthetic phase and selection guard regressions; no historical feature reads."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import monthly_block_sharpe242_host as h

def training_fixture():
    ref=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))
    m={a:{p:copy.deepcopy(ref) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    for path in m[h.ARMS[1]]:m[h.ARMS[1]][path]['training']['cagr']=.52
    o={a:[dict(symbols=['A','B','C']) for i in range(122)] for a in h.ARMS}
    for i in (1,54,55):o[h.ARMS[1]][i]['symbols']=['A','B','D']
    return m,o

def receipt_fixture():
    ind=SimpleNamespace(SELECTION_FLAGS=('passed','exact_checked'),FEATURE_FLAGS=('passed','math_checked'))
    def sc(n):return dict(passed=True,exact_checked=True,ordered_choices_checked=3*n,checked_events=n,numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False,counts=dict(events=n),full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True)
    def fc(n):return dict(passed=True,math_checked=True,events=n,numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False,counts=dict(events=n),full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True)
    e=dict(all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,new_model_fits=0,selection=sc(122),feature=fc(122),physical_prefixes=[dict(prefix=n,selection=sc(n),feature=fc(n),feature_counts=dict(events=n),decision_counts=dict(events=n),exact=True) for n in h.PREFIXES])
    return e,dict(new_model_fits=0,new_returns_computed=False,events=122),ind,dict(counts=dict(events=122),event_history=list(range(122))),dict(counts=dict(events=122),events=list(range(122)))

def listing_fixture():
    bound=dict(symbol='N',issuer_key='issuer-N',available_from='2025-01-02',not_before='2025-01-01')
    proof=dict(query_index=0,column_index=0,symbol='N',issuer_key='issuer-N',date='2025-02-28',new_price_start='2024-01-01',new_price_end='2024-12-30')
    review=dict(passed=True,strict_pit=False,listing_identity_metadata_verified=True,listing_bounds=[bound],listing_identity_cells=[proof])
    audit=dict(event_history=[dict(query_index=0,date='2025-02-28',price_start='2024-01-01',price_end='2024-12-30',cells=[dict(column_index=0,symbol='N',source_available=False)])])
    return review,audit

class HostTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(h.ARMS,('incumbent_local67','block21_sharpe','daily_sharpe_control'))
        self.assertEqual(len(h.RESULT_FILES),38);self.assertEqual(len(h.PREP_FILES),9)
        self.assertEqual(h.PRIOR.name,'monthly_core_veto241')
    def test_admit(self):self.assertTrue(h.select_first88(*training_fixture())['admitted'])
    def test_later_not_used(self):
        m,o=training_fixture();r=h.select_first88(m,o)
        for a in o:
            for i in range(88,122):o[a][i]['symbols']=['X','Y','Z']
        self.assertEqual(r,h.select_first88(m,o))
    def test_diagnostic_control_not_promoted(self):
        m,o=training_fixture();m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.9
        self.assertEqual(h.select_first88(m,o)['primary'],h.ARMS[0])
    def test_gain_and_open_guard(self):
        for path,value in [('same_close_zero',.504),('next_open_zero',.49)]:
            m,o=training_fixture();m[h.ARMS[1]][path]['training']['cagr']=value
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_risk_and_segments(self):
        for split,key,value in [('training','sharpe',1.64),('training','max_drawdown',-.331),('early','cagr',.374),('recent','cagr',.574)]:
            m,o=training_fixture();m[h.ARMS[1]]['same_close_zero'][split][key]=value
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_change_support_both_references(self):
        m,o=training_fixture();o[h.ARMS[2]]=copy.deepcopy(o[h.ARMS[1]])
        self.assertFalse(h.select_first88(m,o)['admitted'])
        m,o=training_fixture();o[h.ARMS[1]][1]['symbols']=['A','B','C']
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_state_counters(self):
        s=dict(completed=[dict(directory='old',trials=5) for i in range(236)]+[dict(directory=h.PRIOR.name,trials=44)],best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.initial_state(s);s['completed'][-1]['trials']+=1
        with self.assertRaises(ValueError):h.initial_state(s)
    def test_queue_binding(self):
        s=dict(current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='fixed')])
        with patch.object(h,'initial_state'):
            h.registration_check(s,'fixed');s['registered_queue']=[]
            with self.assertRaises(ValueError):h.registration_check(s,'fixed')
    def test_preparation(self):h.preparation_check(*receipt_fixture())
    def test_feature_flags_rejected(self):
        for key,val in [('math_checked',False),('events',121),('numerical_tolerance',1e-5),('producer_modules_imported',True)]:
            e,c,i,fa,da=receipt_fixture();e['feature'][key]=val
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,fa,da)
    def test_selection_flags_rejected(self):
        for key,val in [('exact_checked',False),('ordered_choices_checked',365),('computes_returns',True)]:
            e,c,i,fa,da=receipt_fixture();e['selection'][key]=val
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,fa,da)
    def test_prefix_missing_duplicate_or_false(self):
        for mutation in ('missing','duplicate','false','wrongscope'):
            e,c,i,fa,da=receipt_fixture()
            if mutation=='missing':e['physical_prefixes'].pop()
            if mutation=='duplicate':e['physical_prefixes'][-1]=e['physical_prefixes'][-2]
            if mutation=='false':e['physical_prefixes'][1]['exact']=False
            if mutation=='wrongscope':e['physical_prefixes'][2]['feature']['events']=122
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,fa,da)
    def test_listed_source_clock(self):
        r,a=listing_fixture()
        self.assertEqual(h.listing_bound_check(a,r)['prebound_unavailable_cells'],1)
        a['event_history'][0]['cells'][0]['source_available']=True
        with self.assertRaises(ValueError):h.listing_bound_check(a,r)
        a['event_history'][0]['price_start']='2025-01-01';r['listing_identity_cells'][0]['new_price_start']='2025-01-01'
        self.assertEqual(h.listing_bound_check(a,r)['prebound_unavailable_cells'],0)
    def test_bound_not_used_before_public_time(self):
        r,a=listing_fixture();r['listing_bounds'][0]['available_from']='2025-03-01'
        a['event_history'][0]['cells'][0]['source_available']=True
        self.assertEqual(h.listing_bound_check(a,r)['known_bound_cells_checked'],0)

    def test_bound_full_identity_clock_and_missing(self):
        for key,val in [('issuer_key','other'),('date','2025-02-27'),('column_index',1),('new_price_start','2023-01-01')]:
            r,a=listing_fixture();r['listing_identity_cells'][0][key]=val
            with self.assertRaises(ValueError):h.listing_bound_check(a,r)
        r,a=listing_fixture();a['event_history'][0]['cells']=[]
        with self.assertRaises(ValueError):h.listing_bound_check(a,r)

    def test_full_audit_bindings(self):
        for where in ('feature','decision','prefix'):
            e,c,i,fa,da=receipt_fixture()
            if where=='feature':fa['counts']['changed']=True
            if where=='decision':da['counts']['changed']=True
            if where=='prefix':e['physical_prefixes'][2]['feature_counts']['changed']=True
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,fa,da)
    def test_full_inputs_required(self):
        for key in ('full_scores_supplied','full_pre_scores_supplied','full_audit_supplied'):
            e,c,i,fa,da=receipt_fixture();e['selection'][key]=False
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,fa,da)

    def test_source_review_fail_closed(self):
        raw={str(i):'digest' for i in range(64)}
        source=dict(passed=True,strict_pit=False,listing_identity_metadata_verified=True,context_sha256='fixed',coverage_sha256='fixed',raw_source_hashes=raw)
        yes=('metadata_only','source_admission_metadata_only','raw_source_hashes_verified','raw_source_map_matches_original','metadata_axes_verified','calendar_clock_verified','source_window_matches219')
        no=('actual_support_computed','actual_new_features_computed','actual_new_orders_computed','actual_new_returns_computed','actual_model_fits_computed','npz_decoded','prices_loaded','loader_executed','observers_modified')
        source.update({k:True for k in yes});source.update({k:False for k in no})
        with patch.object(h,'sha',return_value='fixed'),patch.object(h,'read',return_value=dict(raw_source_hashes=raw)):
            h.source_review_check(source)
            for key in yes+no:
                bad=copy.deepcopy(source);bad[key]=not bad[key]
                with self.assertRaises(ValueError):h.source_review_check(bad)
            for key in ('context_sha256','coverage_sha256'):
                bad=copy.deepcopy(source);bad[key]='different'
                with self.assertRaises(ValueError):h.source_review_check(bad)

if __name__=='__main__':unittest.main()

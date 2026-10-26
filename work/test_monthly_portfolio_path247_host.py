"""Synthetic lifecycle, selection and complete-audit regressions, no source data."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import monthly_portfolio_path247_host as h


def training_fixture():
    ref=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))
    m={a:{p:copy.deepcopy(ref) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    for path in m[h.ARMS[1]]:m[h.ARMS[1]][path]['training']['cagr']=.52
    o={a:[dict(symbols=['A','B','C']) for i in range(122)] for a in h.ARMS}
    for i in (1,54,55):o[h.ARMS[1]][i]['symbols']=['A','B','D']
    return m,o


def receipt_fixture():
    ind=SimpleNamespace(SELECTION_FLAGS=('passed','exact_checked'),PREPARE_FLAGS=('passed','math_checked'))
    def counts(n):return dict(events=n,new_model_fits=0,predictive_model_fits=0)
    def sc(n):return dict(passed=True,exact_checked=True,ordered_choices_checked=4*n,checked_events=n,numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False,counts=counts(n),full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True)
    def fc(n):return dict(passed=True,math_checked=True,full_features_supplied=True,descriptive_path_returns_checked=True,ordered_choices_checked=4*n,checked_events=n,continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,exact_discrete=True,producer_modules_imported=False,computes_returns=False,counts=counts(n),full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True)
    e=dict(all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,new_model_fits=0,selection=sc(122),full=fc(122),physical_prefixes=[dict(prefix=n,selection=sc(n),full=fc(n),counts=counts(n),exact=True) for n in h.PREFIXES])
    completion=dict(new_model_fits=0,new_predictive_model_fits=0,new_returns_computed=False,events=122,descriptive_counts=counts(122))
    return e,completion,ind,dict(counts=counts(122),events=list(range(122)))


def source_fixture():
    raw={str(i):'digest' for i in range(64)}
    source=dict(passed=True,metadata_only=True,calendar_clock_verified=True,metadata_axes_verified=True,raw_source_hashes_verified=True,raw_source_map_matches_original=True,listing_identity_metadata_verified=True,source_window_recomputed_253=True,strict_pit=False,registered_strategy=False,actual_shortlist_computed=False,prices_loaded=False,npz_decoded=False,raw_source_count=64,raw_source_hashes=raw,calendar=dict(sha256='fixed',prices=253,return_intervals=252,windows=[{} for _ in range(122)]),metadata=dict(events=122,width=49,formal_monthends=121),draft_path='work/synthetic-draft.json',draft_sha256='fixed',source_scope=dict(VOX_used=False,sector_mapping_used=False,extra_sources=[]))
    binding=dict(raw_source_hashes=raw,input_sha256={source['draft_path']:'fixed'})
    return source,binding


def outcome_fixture():
    return dict(passed=True,checked_paths=12,checked_slices=60,independent_accounts=True,independent_of_production_model_and_engine=True,terminal_liquidation=False,paths={a:{p[0]:dict(passed=True,monthly_executions=122,slices={s:{} for s in ('full122','first88','early53','recent35','later34')}) for p in h.ev.PATHS} for a in h.ARMS})


class HostTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(h.ARMS,('incumbent_local67','portfolio_joint_path','portfolio_diagonal_control','portfolio_structure_control'))
        self.assertEqual(len(h.RESULT_FILES),50);self.assertEqual(len(h.PREP_FILES),8)
        self.assertEqual(len(h.code_names),8);self.assertEqual(h.PRIOR.name,'monthly_residual_core246')
        self.assertIn('FEATURES.json',h.PREP_FILES)

    def test_admission_requires_all_three_references(self):
        result=h.select_first88(*training_fixture())
        self.assertTrue(result['admitted']);self.assertEqual(result['primary'],h.ARMS[1])
        self.assertEqual(set(result['guards']),{h.ARMS[i] for i in (0,2,3)})

    def test_later34_holdings_not_used(self):
        m,o=training_fixture();expected=h.select_first88(m,o)
        for a in o:
            for i in range(88,122):o[a][i]['symbols']=['X','Y','Z']
        self.assertEqual(expected,h.select_first88(m,o))

    def test_controls_can_never_be_selected(self):
        for control in h.ARMS[2:]:
            m,o=training_fixture();m[control]['same_close_zero']['training']['cagr']=.9
            r=h.select_first88(m,o);self.assertFalse(r['admitted']);self.assertEqual(r['primary'],h.ARMS[0])

    def test_incumbent_gain_and_nextopen_guards(self):
        for path,value in [('same_close_zero',.504),('next_open_zero',.49)]:
            m,o=training_fixture();m[h.ARMS[1]][path]['training']['cagr']=value
            self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_each_control_checked_on_both_zero_paths(self):
        for a in h.ARMS[2:]:
            for path in ('same_close_zero','next_open_zero'):
                m,o=training_fixture();m[a][path]['training']['cagr']=.521
                self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_risk_and_segment_guards_on_both_paths(self):
        for path in ('same_close_zero','next_open_zero'):
            for split,key,value in [('training','sharpe',1.64),('training','max_drawdown',-.331),('early','cagr',.374),('recent','cagr',.574)]:
                m,o=training_fixture();m[h.ARMS[1]][path][split][key]=value
                self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_change_support_against_each_reference(self):
        for a in (h.ARMS[0],h.ARMS[2],h.ARMS[3]):
            m,o=training_fixture();o[a]=copy.deepcopy(o[h.ARMS[1]])
            self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_changes_need_three_months_and_both_segments(self):
        for months in ((54,55),(53,54,55),(1,2,3)):
            m,o=training_fixture();o[h.ARMS[1]]=copy.deepcopy(o[h.ARMS[0]])
            for i in months:o[h.ARMS[1]][i]['symbols']=['A','B','D']
            self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_reordering_is_not_changed_holdings(self):
        m,o=training_fixture()
        for i in (1,54,55):o[h.ARMS[1]][i]['symbols']=['C','B','A']
        self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_extra_or_missing_arms_rejected(self):
        for which in ('metrics','orders'):
            m,o=training_fixture();(m if which=='metrics' else o).pop(h.ARMS[3])
            with self.assertRaises(ValueError):h.select_first88(m,o)

    def test_predecessor_counter_scope_and_best_independence(self):
        s=dict(completed=[dict(directory='old',trials=5) for i in range(241)]+[dict(directory=h.PRIOR.name,trials=34)],best_training_selected_candidate=dict(directory='unrelated_new_best'))
        h.initial_state(s)
        for field,value in [('trials',35),('directory','wrong_predecessor')]:
            bad=copy.deepcopy(s);bad['completed'][-1][field]=value
            with self.assertRaises(ValueError):h.initial_state(bad)
        s['completed'].pop()
        with self.assertRaises(ValueError):h.initial_state(s)

    def test_predecessor_completion_bound_to_central(self):
        s=dict(completed=[dict(completion=dict(path=str(h.PRIOR/'COMPLETION.json'),sha256='fixed'))])
        with patch.object(h,'sha',return_value='fixed'):
            h.predecessor_binding(s)
            for key,value in [('path',str(h.PRIOR/'OTHER.json')),('sha256','wrong')]:
                bad=copy.deepcopy(s);bad['completed'][-1]['completion'][key]=value
                with self.assertRaises(ValueError):h.predecessor_binding(bad)

    def test_queue_binding(self):
        s=dict(current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='fixed')])
        with patch.object(h,'initial_state'):
            h.registration_check(s,'fixed')
            for field,value in [('current','wrong'),('registered_queue',[]),('registered_queue',[dict(directory=h.OUT.name,plan_sha256='wrong')])]:
                bad=copy.deepcopy(s);bad[field]=value
                with self.assertRaises(ValueError):h.registration_check(bad,'fixed')

    def test_complete_preparation(self):h.preparation_check(*receipt_fixture())

    def test_full_audit_is_independent_and_complete(self):
        for key,val in [('math_checked',False),('ordered_choices_checked',487),('checked_events',121),('producer_modules_imported',True),('computes_returns',True)]:
            e,c,i,a=receipt_fixture();e['full'][key]=val
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_descriptive_path_and_features_must_be_checked(self):
        for key in ('full_features_supplied','descriptive_path_returns_checked'):
            for value in (False,None):
                e,c,i,a=receipt_fixture();e['full'][key]=value
                with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_numerical_tolerances_and_discrete_gates_cannot_drift(self):
        for key,val in [('continuous_atol',1e-5),('continuous_rtol',1e-5),('rank_tolerance',2e-10),('exact_discrete',False)]:
            e,c,i,a=receipt_fixture();e['full'][key]=val
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_selection_scope_is_exact(self):
        for key,val in [('exact_checked',False),('ordered_choices_checked',487),('checked_events',121),('numerical_tolerance',2e-10),('producer_modules_imported',True),('computes_returns',True)]:
            e,c,i,a=receipt_fixture();e['selection'][key]=val
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_all_full_arrays_and_audit_required(self):
        for kind in ('full','selection'):
            for key in ('full_scores_supplied','full_pre_scores_supplied','full_audit_supplied'):
                e,c,i,a=receipt_fixture();e[kind][key]=False
                with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_all_physical_prefixes_required(self):
        for mutation in ('missing','duplicate','false','wrongscope'):
            e,c,i,a=receipt_fixture()
            if mutation=='missing':e['physical_prefixes'].pop()
            if mutation=='duplicate':e['physical_prefixes'][-1]=e['physical_prefixes'][-2]
            if mutation=='false':e['physical_prefixes'][1]['exact']=False
            if mutation=='wrongscope':e['physical_prefixes'][2]['full']['checked_events']=122
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_audit_count_bindings(self):
        for where in ('full','selection','audit','completion','prefix'):
            e,c,i,a=receipt_fixture()
            if where in ('full','selection'):e[where]['counts']['extra']=1
            if where=='audit':a['counts']['extra']=1
            if where=='completion':c['descriptive_counts']['extra']=1
            if where=='prefix':e['physical_prefixes'][2]['counts']['extra']=1
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_no_returns_or_fits_before_selection(self):
        for key,value in [('new_model_fits',1),('new_predictive_model_fits',1),('new_returns_computed',True),('events',121)]:
            e,c,i,a=receipt_fixture();c[key]=value
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_source_review_binds_metadata_and_clock(self):
        source,binding=source_fixture()
        with patch.object(h,'sha',return_value='fixed'),patch.object(h,'read',return_value=binding):
            h.source_review_check(source)
            for key in ('passed','metadata_only','calendar_clock_verified','metadata_axes_verified','raw_source_hashes_verified','raw_source_map_matches_original','listing_identity_metadata_verified','source_window_recomputed_253','strict_pit','registered_strategy','actual_shortlist_computed','prices_loaded','npz_decoded'):
                bad=copy.deepcopy(source);bad[key]=not bad[key]
                with self.assertRaises(ValueError):h.source_review_check(bad)

    def test_source_review_rejects_wrong_windows_axes_and_prices(self):
        source,binding=source_fixture()
        with patch.object(h,'sha',return_value='fixed'),patch.object(h,'read',return_value=binding):
            for section,key,value in [('calendar','sha256','wrong'),('calendar','prices',252),('calendar','return_intervals',251),('calendar','windows',[{}]*121),('metadata','events',121),('metadata','width',48),('metadata','formal_monthends',122),('source_scope','VOX_used',True),('source_scope','sector_mapping_used',True),('source_scope','extra_sources',['new'])]:
                bad=copy.deepcopy(source);bad[section][key]=value
                with self.assertRaises(ValueError):h.source_review_check(bad)
            for key,value in [('raw_source_count',63),('raw_source_hashes',{}),('draft_sha256','wrong')]:
                bad=copy.deepcopy(source);bad[key]=value
                with self.assertRaises(ValueError):h.source_review_check(bad)

    def test_independent_outcome_complete(self):h.outcome_check(outcome_fixture())

    def test_independent_outcome_rejects_scope_and_liquidation(self):
        for key,value in [('passed',False),('checked_paths',9),('checked_slices',45),('independent_accounts',False),('independent_of_production_model_and_engine',False),('terminal_liquidation',True)]:
            c=outcome_fixture();c[key]=value
            with self.assertRaises(ValueError):h.outcome_check(c)
        for key,value in [('passed',False),('monthly_executions',121),('slices',{})]:
            c=outcome_fixture();c['paths'][h.ARMS[1]]['same_close_zero'][key]=value
            with self.assertRaises(ValueError):h.outcome_check(c)
        c=outcome_fixture();c['paths'].pop(h.ARMS[3])
        with self.assertRaises(ValueError):h.outcome_check(c)


if __name__=='__main__':unittest.main()

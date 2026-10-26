"""Synthetic231 integration: selector economics, phase gates and immutable inputs."""
from pathlib import Path
import ast
import copy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_market_drift231_host as h
import monthly_market_drift231_model as model
import monthly_market_drift231_independent as oracle
from test_monthly_market_drift231_model import fixture as price_fixture


def snapshot(a):
    return {k:(np.asarray(v).dtype.str,np.asarray(v).shape,np.asarray(v).tobytes()) for k,v in a.items()}


def fixture():
    panel,meta,cal,small=price_fixture();n=len(meta['dates']);w=5
    panel['adj']['D']=panel['adj']['A']*np.exp(np.arange(len(panel['adj']))*.0002)
    panel['adj']['E']=panel['adj']['B']*np.exp(np.arange(len(panel['adj']))*.0003)
    meta['symbols']=[list('ABCDE') for _ in range(n)];meta['etfs']=[]
    calm=small['state_calm10'].copy();cw=np.where(calm,.5,.52)[:,None]
    support=np.tile(np.linspace(-.1,.1,w),(n,1));core=cw*np.tile(np.linspace(.45,.2,w),(n,1))
    zero=np.zeros((n,w));yes=np.ones((n,w),bool)
    a=dict(baseline=core+(1-cw)*support,core=core,core_weight=cw,eligible=yes.copy(),
        state_calm10=calm,state_weak_narrow=np.zeros(n,bool),pre_score=np.tile([.205,.204,.203,.202,.201],(n,1)),
        effects=np.tile([0.,0.,-.5,.5,0.],(n,1)),forecast_available=yes.copy(),extension_available=yes.copy(),
        old_ml4=zero.copy(),rank_log_excess=zero.copy(),model_available=yes.copy(),local_available=np.zeros((n,w),bool),
        raw_source_log=zero.copy(),raw_local_constant=zero.copy(),raw_local_linear=zero.copy())
    for name in ('simple_excess','log_excess','vol_scaled','loss_utility'):
        a['raw_'+name]=zero.copy();a.setdefault('rank_'+name,zero.copy())
    a['rank_local_constant']=zero.copy();a['rank_local_linear']=zero.copy();a['paired_month_ready']=np.ones(n,bool)
    assert len(a)==27
    return panel,meta,cal,a


def selection_fixture():
    metric={era:dict(cagr=.5,sharpe=1.5,max_drawdown=-.3) for era in ('training','early','recent')}
    scores={arm:{path:copy.deepcopy(metric) for path in ('same_close_zero','next_open_zero')} for arm in h.NAMES}
    orders={arm:[dict(symbols=['A','B','C']) for _ in range(122)] for arm in h.NAMES}
    for path in scores[h.NAMES[1]]:scores[h.NAMES[1]][path]['training']['cagr']+=.01
    for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['A','B','D']
    return scores,orders


class DecisionTests(unittest.TestCase):
    def test_real_frozen_study_and_independent_selector_exact(self):
        panel,m,c,a=fixture();built=model.build(panel,m,c,a);before=snapshot(a)
        orders,scores,choices=h.decisions(m,a,built,len(m['dates']))
        receipt=oracle.verify_selection(a,m,built,scores,choices)
        self.assertTrue(receipt['passed']);self.assertEqual(snapshot(a),before)
        self.assertEqual(set(orders),set(h.NAMES));self.assertEqual(len(orders[h.NAMES[0]]),4)
        for arm in h.NAMES[1:]:self.assertEqual(orders[arm][0],orders[h.NAMES[0]][0])

    def test_supported_full_eligible_can_replace_first_two(self):
        panel,m,c,a=fixture();panel['adj']['SPY']=64.;built=model.build(panel,m,c,a)
        orders,scores,choices=h.decisions(m,a,built,4)
        oracle.verify_selection(a,m,built,scores,choices)
        pre=h.local().modified_pre_score(h.local().configs()[4],a,4)
        changed=False
        for i in np.flatnonzero(built['arrays']['ready']):
            old_top2=set(np.argsort(-pre[i])[:2]);changed|=not old_top2.issubset(choices[h.NAMES[1]][i])
        self.assertTrue(changed,'Fixture must exercise a top-two replacement')

    def test_zero_effect_is_supported_replacement_not_old_episode(self):
        panel,m,c,a=fixture();panel['adj']['SPY']=64.
        for s in 'ABCDE':panel['adj'][s]=64.
        built=model.build(panel,m,c,a);o,s,q=h.decisions(m,a,built,4)
        oracle.verify_selection(a,m,built,s,q)
        self.assertTrue(built['arrays']['ready'][1:].all())
        self.assertTrue(np.all(built['arrays']['candidate_effect']==0))
        self.assertNotEqual(q[h.NAMES[0]][1],q[h.NAMES[1]][1])
        self.assertEqual(q[h.NAMES[1]],q[h.NAMES[2]])

    def test_whole_month_common_fallback(self):
        panel,m,c,a=fixture();panel['adj'].loc[:,'SPY']=64.
        pos=c.index(m['dates'][1])-100;panel['adj'].loc[pd.Timestamp(c[pos]),'E']=np.nan
        built=model.build(panel,m,c,a);o,s,q=h.decisions(m,a,built,4)
        self.assertFalse(built['arrays']['ready'][1])
        for arm in h.NAMES[1:]:
            self.assertEqual(o[arm][1],o[h.NAMES[0]][1]);np.testing.assert_array_equal(s[arm][1],s[h.NAMES[0]][1])

    def test_two_candidates_half_one_candidate_cash(self):
        panel,m,c,a=fixture();a['eligible'][1]=[True,True,False,False,False];a['eligible'][2]=[True,False,False,False,False]
        built=model.build(panel,m,c,a);o,s,q=h.decisions(m,a,built,4)
        oracle.verify_selection(a,m,built,s,q)
        for arm in h.NAMES:self.assertEqual(len(q[arm][1]),2);self.assertEqual(q[arm][2],[])

    def test_all_state_budget_and_original_support_clamp(self):
        panel,m,c,a=fixture();panel['adj']['SPY']=64.
        a['state_calm10'][:]=[False,False,True,True];a['state_weak_narrow'][:]=[False,True,False,True]
        a['core_weight']=np.where(a['state_calm10'],.50,.52)[:,None]
        a['baseline']=a['core']+(1-a['core_weight'])*np.tile([.49,-.49,.1,-.1,0.],(4,1))
        built=model.build(panel,m,c,a);o,s,q=h.decisions(m,a,built,4)
        oracle.verify_selection(a,m,built,s,q)

    def test_all_decision_prefixes_unchanged_input_bytes(self):
        args=fixture();panel,m,c,a=args;built=model.build(*args);before=snapshot(a)
        oo,ss,qq=h.decisions(m,a,built,4)
        for n in range(5):
            o,s,q=h.decisions(m,a,built,n)
            self.assertEqual(o,{k:v[:n] for k,v in oo.items()});self.assertEqual(q,{k:v[:n] for k,v in qq.items()})
            for arm in h.NAMES:np.testing.assert_array_equal(s[arm],ss[arm][:n])
        self.assertEqual(snapshot(a),before)

    def test_physical_price_metadata_and_array_prefixes(self):
        args=fixture();built=model.build(*args)
        for n in range(5):
            p,m,c,a=h.prefix_inputs(*args,n);small=model.build(p,m,c,a)
            self.assertEqual(len(m['dates']),n);self.assertEqual(len(a),27)
            self.assertTrue(all(len(v)==n for v in a.values()))
            if n:self.assertLessEqual(p['adj'].index.max(),pd.Timestamp(m['dates'][-1]))
            else:self.assertEqual(len(p['adj']),0)
            for key in model.ARRAY_KEYS:np.testing.assert_array_equal(small['arrays'][key],built['arrays'][key][:n])
            oracle.check_build(p,m,c,a,small)
            o,s,q=h.decisions(m,a,small,n);oracle.verify_selection(a,m,small,s,q)

    def test_future_values_not_consumed_in_prefix(self):
        args=fixture();panel,m,c,a=args;built=model.build(*args);expected=h.decisions(m,a,built,2)
        for key in a:
            if a[key].dtype.kind=='f':a[key][2:]=np.inf
        for key in built['arrays']:
            if built['arrays'][key].dtype.kind=='f':built['arrays'][key][2:]=np.inf
        actual=h.decisions(m,a,built,2)
        self.assertEqual(actual[0],expected[0]);self.assertEqual(actual[2],expected[2])
        for arm in h.NAMES:np.testing.assert_array_equal(actual[1][arm],expected[1][arm])

    def test_invalid_prefixes_rejected(self):
        p,m,c,a=fixture();built=model.build(p,m,c,a)
        for n in (True,-1,1.5,5):
            with self.subTest(n=n),self.assertRaises(ValueError):h.decisions(m,a,built,n)


class MarketParityTests(unittest.TestCase):
    def test_original_calm63_verified_on_official_calendar(self):
        panel,m,c,a=fixture()
        spy=panel['adj']['SPY'].reindex(pd.DatetimeIndex(c))
        vol=spy.pct_change(fill_method=None).rolling(63).std(ddof=1)*np.sqrt(252.)
        a['state_calm10']=np.asarray([bool(vol.loc[day]<.10) for day in m['dates']],bool)
        receipt=h.calm_parity(panel,m,c,a)
        self.assertIsInstance(receipt,dict)
        a['state_calm10'][1]=not a['state_calm10'][1]
        with self.assertRaises(ValueError):h.calm_parity(panel,m,c,a)

    def test_missing_current_market_not_silently_noncalm(self):
        panel,m,c,a=fixture();panel['adj'].loc[pd.Timestamp(m['dates'][1]),'SPY']=np.nan
        with self.assertRaises(ValueError):h.calm_parity(panel,m,c,a)


class SelectionGateTests(unittest.TestCase):
    def test_first88_only_later_extremes_do_not_change_selection(self):
        metrics,orders=selection_fixture();expected=h.select_first88(metrics,orders)
        self.assertTrue(expected['admitted']);self.assertFalse(expected['later34_used'])
        orders[h.NAMES[1]][88:]=[dict(symbols=['ZZZ']) for _ in range(34)]
        self.assertEqual(h.select_first88(metrics,orders),expected)

    def test_zero_cost_next_open_required_cost_path_not_consumed(self):
        metrics,orders=selection_fixture()
        for arm in h.NAMES:metrics[arm]['next_open_cost']={'poison':object()}
        self.assertTrue(h.select_first88(metrics,orders)['admitted'])
        metrics[h.NAMES[1]]['next_open_zero']['training']['cagr']=.49
        self.assertFalse(h.select_first88(metrics,orders)['admitted'])

    def test_control_never_promoted_and_support_both_eras_both_refs(self):
        metrics,orders=selection_fixture();orders[h.NAMES[2]]=copy.deepcopy(orders[h.NAMES[1]])
        self.assertEqual(h.select_first88(metrics,orders)['primary'],h.NAMES[0])
        metrics,orders=selection_fixture();orders[h.NAMES[1]][53]['symbols']=['A','B','C']
        orders[h.NAMES[1]][3]['symbols']=['A','B','D']
        self.assertFalse(h.select_first88(metrics,orders)['admitted'])

    def test_each_return_risk_and_era_guard_is_material(self):
        for path,era,field,value in [('same_close_zero','training','cagr',.504),('next_open_zero','training','cagr',.49),
            ('same_close_zero','training','sharpe',1.449),('next_open_zero','training','max_drawdown',-.331),
            ('same_close_zero','early','cagr',.474),('next_open_zero','recent','cagr',.474)]:
            m,o=selection_fixture();m[h.NAMES[1]][path][era][field]=value
            with self.subTest(path=path,era=era,field=field):self.assertFalse(h.select_first88(m,o)['admitted'])


class EvidenceTests(unittest.TestCase):
    def test_model22_flags16counts_required_and_exact(self):
        args=fixture();built=model.build(*args);receipt=oracle.check_build(*args,built)
        h.require_model_audit(receipt,built,4)
        self.assertEqual(len(h.MODEL_FLAGS),22);self.assertEqual(len(h.MODEL_COUNT_FIELDS),16)
        for key in h.MODEL_FLAGS:
            bad=copy.deepcopy(receipt);bad[key]=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_model_audit(bad,built,4)
        for key in h.MODEL_COUNT_FIELDS:
            bad=copy.deepcopy(receipt);bad['counts'][key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_model_audit(bad,built,4)
        for key,value in [('query_rows',5),('numerical_tolerance',1e-12),('producer_modules_imported',True),
            ('computes_returns',True),('new_model_fits',1),('predictive_model_fits',1),('new_ridge_fits',1),('new_tree_fits',1)]:
            bad=copy.deepcopy(receipt);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_model_audit(bad,built,4)

    def test_selection10_flags_exact_independent_scope(self):
        p,m,c,a=fixture();built=model.build(p,m,c,a);o,s,q=h.decisions(m,a,built,4)
        check=oracle.verify_selection(a,m,built,s,q);h.require_selection_audit(check,4)
        for key in h.SELECTION_FLAGS:
            bad=copy.deepcopy(check);bad[key]=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_selection_audit(bad,4)
        for key,value in [('ordered_choices_checked',11),('numerical_tolerance',1e-12),('producer_modules_imported',True),('computes_returns',True)]:
            bad=copy.deepcopy(check);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_selection_audit(bad,4)

    def test_result_manifest_exact38(self):
        files={k:'synthetic' for k in h.RESULT_FILES};self.assertEqual(len(files),38);h.require_result_manifest(files)
        for kind in ('missing','extra'):
            bad=dict(files)
            if kind=='missing':bad.pop(next(iter(bad)))
            else:bad['COMPLETION.json']='unexpected'
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.require_result_manifest(bad)

    def test_source_pins_conflicts_and_mutable_central_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'synthetic';file.write_bytes(b'first');pins={};h.bind(pins,file);prior=dict(pins)
            file.write_bytes(b'second')
            with self.assertRaises(ValueError):h.bind(pins,file)
            self.assertEqual(pins,prior)
        with self.assertRaises(ValueError):h.verify({str(h.CENTRAL):'wrong'})
        with self.assertRaises(ValueError):h.verify({'/tmp/unregistered.py':'wrong'})

    def test_contract_scope_frozen_source_and_zero_cost_target(self):
        c=h.contract();self.assertEqual(c['period'],['2016-08-17','2026-09-24'])
        self.assertEqual(c['arms'],list(h.NAMES));self.assertEqual(c['registered_trials'],3);self.assertEqual(c['execution_paths'],9)
        self.assertTrue(c['no_observer_writes']);self.assertEqual(list(h.MODEL_PREFIXES),[0,1,53,88,122])
        self.assertEqual(len(h.PREPARED_FILES),7)

    def test_no_new_fits_in_host_or_model(self):
        for path in (Path(h.__file__),Path(model.__file__)):
            tree=ast.parse(path.read_text())
            calls=[x.func.attr for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)]
            self.assertNotIn('fit',calls);self.assertNotIn('fit_transform',calls);self.assertNotIn('predict',calls)


class PhaseTests(unittest.TestCase):
    def test_source_gate_precedes_actual_loading_in_replay_phases(self):
        with patch.object(h,'gate',side_effect=ValueError('synthetic bad PLAN')),patch.object(h,'local') as loader,patch.object(h.ev,'prices') as prices:
            for name in ('prepare','select','diagnose','close'):
                with self.subTest(phase=name),self.assertRaisesRegex(ValueError,'synthetic bad PLAN'):
                    getattr(h,name)('wrong')
            loader.assert_not_called();prices.assert_not_called()

    def test_register_is_write_once_before_source_or_state_reads(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(h,'OUT',Path(directory)),patch.object(h,'read') as read:
            with self.assertRaises(ValueError):h.register()
            read.assert_not_called()

    def test_state_requires_current_predecessor_and_exact_trial_counts(self):
        rows=[dict(directory='prior',trials=5) for _ in range(226)]
        rows[0]['trials']+=65;rows[-1]['directory']=h.PRIOR.name
        state=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state)
        for kind in ('count','trial','latest','incumbent'):
            bad=copy.deepcopy(state)
            if kind=='count':bad['completed'].pop(0)
            if kind=='trial':bad['completed'][0]['trials']+=1
            if kind=='latest':bad['completed'][-1]['directory']='older'
            if kind=='incumbent':bad['best_training_selected_candidate']['directory']='other'
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.state_check(bad)


if __name__=='__main__':unittest.main(verbosity=2)

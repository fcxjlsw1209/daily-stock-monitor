"""Synthetic-only204 full pre-episode support integration and lifecycle gates."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import importlib.util
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
import monthly_aux_rank204_kernel as kernel
import monthly_aux_rank204_independent as independent

spec=importlib.util.spec_from_file_location('tested_aux204_host',Path(__file__).with_name('monthly_aux_rank204_host.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def fixture(rows=6,width=5):
    support=np.tile(np.linspace(-.32,.18,width)**3,(rows,1))
    if rows>2:support[2]=.125
    if rows>3:support[3]=np.where(np.arange(width)<width//2,-.125,.125)
    calm=np.zeros(rows,bool)
    if rows>1:calm[1]=True
    cw=np.where(calm,.5,.52)[:,None]
    a=dict(baseline=(1-cw)*support,core=np.zeros((rows,width)),core_weight=cw,
        state_calm10=calm,state_weak_narrow=np.zeros(rows,bool),eligible=np.ones((rows,width),bool),
        extension_available=np.ones((rows,width),bool),forecast_available=np.ones((rows,width),bool),
        rank_log_excess=np.zeros((rows,width)),effects=np.zeros((rows,width)))
    a['eligible'][:,-1]=False
    m=dict(dates=['synthetic-%d'%i for i in range(rows)],symbols=[[chr(65+j) for j in range(width)] for _ in range(rows)])
    v,_=kernel.build_features(a)
    return a,m,v


class SyntheticStudy:
    def __init__(self):
        self.ep = SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        self.adapter = SimpleNamespace(score_and_choose=Mock(side_effect=self.choose))
        self.rp = SimpleNamespace(make_orders=self.orders)

    @staticmethod
    def configs():
        return [dict(id=i) for i in range(5)]

    @staticmethod
    def modified_pre_score(specification, a, n):
        assert specification == dict(id=4)
        return a['baseline'].copy() + .11  # Includes all already-frozen slots.

    @staticmethod
    def boundary(pre, a, m, n):
        choices = []
        for i in range(n):
            ids = sorted(np.flatnonzero(a['eligible'][i]), key=lambda j: (-pre[i, j], j))
            choices.append([m['symbols'][i][j] for j in ids[:3]] if len(ids) >= 2 else [])
        return choices, pre[:n].copy()

    def choices(self, specification, a, m, n, *, policy):
        return self.boundary(self.modified_pre_score(specification, a, n), a, m, n)

    @staticmethod
    def orders(choices, scores, meta):
        return [dict(date=meta['dates'][i], symbols=symbols, scores=scores[i].tolist()) for i, symbols in enumerate(choices)]

    def choose(self, base, weight, extra, a, m, n, *, pre_score, effects, episode_policy, boundary_selector):
        assert base == 'incumbent66' and weight == 0.
        assert extra.shape == a['baseline'].shape and np.isnan(extra).all()
        assert effects is a['effects'] and boundary_selector is self.ep.select_boundary
        return boundary_selector(pre_score, a, m, n)


def selection_fixture():
    metrics = {n: {p: {e: dict(cagr=.5, sharpe=1.5, max_drawdown=-.3) for e in ('training', 'early', 'recent')}
                  for p in ('same_close_zero', 'next_open_cost')} for n in h.NAMES}
    orders = {n: [dict(symbols=['A', 'B', 'C']) for _ in range(122)] for n in h.NAMES}
    for path in metrics[h.NAMES[1]]: metrics[h.NAMES[1]][path]['training']['cagr'] += .01
    for i in (1, 2, 53): orders[h.NAMES[1]][i]['symbols'] = ['A', 'B', 'D']
    return metrics, orders


class SelectionAndBindingTests(unittest.TestCase):
    def test_later34_and_extra_metrics_never_choose_arm(self):
        metrics, orders = selection_fixture(); expected = h.select_first88(metrics, orders)
        self.assertEqual(expected['primary'], h.NAMES[1])
        for name in h.NAMES:
            metrics[name]['full122'] = dict(cagr=1000. if name != h.NAMES[1] else -1.)
            for i in range(88, 122): orders[name][i]['symbols'] = [name]
        self.assertEqual(h.select_first88(metrics, orders), expected)

    def test_affine_control_cannot_promote_and_must_be_beaten(self):
        for path in ('same_close_zero', 'next_open_cost'):
            metrics, orders = selection_fixture(); metrics[h.NAMES[2]][path]['training']['cagr'] = 2.
            result = h.select_first88(metrics, orders)
            self.assertEqual(result['primary'], h.NAMES[0]); self.assertFalse(result['admitted'])

    def test_exact_gain_era_risk_and_changed_set_guards(self):
        for path, era, key, value in [('same_close_zero', 'training', 'cagr', .504),
                ('next_open_cost', 'training', 'cagr', .49), ('same_close_zero', 'early', 'cagr', .474),
                ('same_close_zero', 'recent', 'cagr', .474), ('same_close_zero', 'training', 'sharpe', 1.449),
                ('same_close_zero', 'training', 'max_drawdown', -.331)]:
            metrics, orders = selection_fixture(); metrics[h.NAMES[1]][path][era][key] = value
            self.assertFalse(h.select_first88(metrics, orders)['admitted'])
        metrics, orders = selection_fixture()
        for i in (1, 2, 53): orders[h.NAMES[1]][i]['symbols'] = ['C', 'B', 'A']
        for i in (90, 100, 110): orders[h.NAMES[1]][i]['symbols'] = ['A', 'B', 'D']
        self.assertEqual(h.select_first88(metrics, orders)['changed_months'], [])

    def test_candidate_must_change_holdings_vs_control_in_both_eras(self):
        metrics, orders = selection_fixture()
        orders[h.NAMES[2]] = copy.deepcopy(orders[h.NAMES[1]])
        decision = h.select_first88(metrics, orders)
        self.assertFalse(decision['admitted'])
        self.assertEqual(decision['changed_months_vs_control'], [])
        for i in (1,2,3): orders[h.NAMES[2]][i]['symbols'] = ['A','E','F']
        self.assertFalse(h.select_first88(metrics, orders)['guards']['changed_support_vs_control'])
        orders[h.NAMES[2]][53]['symbols'] = ['A','E','F']
        self.assertTrue(h.select_first88(metrics, orders)['admitted'])


    def test_inherited_pin_never_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fake-source'; path.write_text('synthetic')
            pins = {}; h.bind(pins, path); before = dict(pins)
            h.bind(pins, path, h.sha(path)); self.assertEqual(pins, before)
            path.write_text('changed')
            with self.assertRaises(ValueError): h.bind(pins, path)
            self.assertEqual(pins, before)

    def test_rejected_plan_cannot_load_source_or_compute_features(self):
        with patch.object(h, 'gate', side_effect=ValueError('unregistered')), patch.object(h, 'local') as loader:
            with self.assertRaisesRegex(ValueError, 'unregistered'):
                h.prepare('bad-plan-sha')
        loader.assert_not_called()




class IntegrationTests(unittest.TestCase):
    def test_full_support_delta_no_eligibility_filter_and_calm_active(self):
        a,m,v=fixture();r,d=h.scale_and_delta(v,6)
        self.assertTrue(a['state_calm10'][1]);self.assertFalse(a['eligible'][1,-1])
        self.assertNotEqual(d[h.NAMES[1]][1,-1],0.)
        self.assertGreater(np.max(np.abs(d[h.NAMES[1]])),.005)
        for arm,key in h.RAW.items():
            np.testing.assert_array_equal(d[arm],v[key+'_delta'])
            np.testing.assert_array_equal(r[arm],v[key])
        self.assertTrue(np.all(d[h.NAMES[1]][0]==0.))

    def test_constant_whole_row_noop_and_two_level_bitidentity(self):
        a,m,v=fixture();r,d=h.scale_and_delta(v,6)
        for arm in h.RAW:
            self.assertTrue(np.all(d[arm][2]==0.));self.assertTrue(np.all(r[arm][2]==0.))
        self.assertEqual(v['candidate'][3].tobytes(),v['control'][3].tobytes())
        self.assertEqual(v['candidate_delta'][3].tobytes(),v['control_delta'][3].tobytes())
        self.assertFalse(np.array_equal(v['candidate'][1],v['control'][1]))

    def test_corrupt_feature_values_and_delta_rejected(self):
        for kind in ('nan','dtype','shape','extra','delta','bound','initial','constant'):
            _,_,v=fixture()
            if kind=='nan':v['weight'][1,0]=np.nan
            if kind=='dtype':v['support']=v['support'].astype(np.float32)
            if kind=='shape':v['control']=v['control'][:,:-1]
            if kind=='extra':v['extra']=v['candidate']
            if kind=='delta':v['candidate_delta'][1,0]+=.001
            if kind=='bound':v['control'][1,0]=.6
            if kind=='initial':v['control_delta'][0,0]=.001
            if kind=='constant':v['candidate_delta'][2,0]=.001
            with self.subTest(kind=kind),self.assertRaises((ValueError,AssertionError)):h.scale_and_delta(v,6)

    def test_zero_prefix_future_values_not_evaluated(self):
        a,m,v=fixture()
        for value in v.values():value[1:]=np.nan
        for n in (0,1):h.scale_and_delta(v,n)
        with self.assertRaises(ValueError):h.scale_and_delta(v,2)
        a['state_calm10']=a['state_calm10'].astype(object);a['state_calm10'][1:]=object()
        a['baseline']=a['baseline'].astype(object);a['baseline'][1:]=object()
        for n in (0,1):kernel.build_features(h.cache_prefix(a,n))
        with self.assertRaises(ValueError):kernel.build_features(h.cache_prefix(a,2))

    def test_nonzero_delta_applies_to_zero_pre_and_zero_delta_keeps_negative_zero(self):
        a,m,v=fixture();study=SyntheticStudy()
        pre=np.zeros(a['baseline'].shape);pre[0]=-0.;pre[2]=-0.
        study.modified_pre_score=Mock(side_effect=lambda *args:pre.copy())
        with patch.object(h,'read',return_value={}):_,_,d=h.decisions(study,a,m,v,6)
        for arm,call in zip(h.NAMES[1:],study.adapter.score_and_choose.call_args_list):
            actual=call.kwargs['pre_score'];mask=d[arm]!=0.
            np.testing.assert_array_equal(actual[mask],d[arm][mask])
            self.assertTrue(np.signbit(actual[0]).all());self.assertTrue(np.signbit(actual[2]).all())

    def test_real_frozen_path_adapter_keeps_original_support_and_returned_choices(self):
        a,m,v=fixture();saved=copy.deepcopy(a);study=SyntheticStudy()
        path=h.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py'
        study.adapter=h.module('synthetic_204_real_adapter',path)
        calls=[]
        def boundary(pre,eligible,symbols,effects,original,policy):
            calls.append((pre.copy(),eligible.copy(),effects.copy(),copy.deepcopy(original)))
            choices=[]
            for row,good in zip(pre,eligible):
                ids=list(np.flatnonzero(good));choices.append([ids[-1],ids[0]] if len(ids)>=2 else [])
            return [[ss[j] for j in ids] for ss,ids in zip(symbols,choices)],pre.copy()
        study.ep.select_boundary=boundary
        policy=dict(budget=.05,ceiling=5,gap=.02)
        with patch.object(h,'read',return_value=policy):orders,r,d=h.decisions(study,a,m,v,6)
        self.assertEqual(len(calls),2)
        base=study.modified_pre_score(dict(id=4),a,6)
        for arm,(pre,good,effects,original) in zip(h.NAMES[1:],calls):
            expected=base.copy();mask=d[arm]!=0;expected[mask]+=d[arm][mask]
            np.testing.assert_array_equal(pre,expected)
            for key in ('baseline','core','state_calm10','state_weak_narrow'):np.testing.assert_array_equal(original[key],a[key])
            np.testing.assert_array_equal((original['baseline']-original['core'])/(1-original['core_weight']),
                                         (a['baseline']-a['core'])/(1-a['core_weight']))
            np.testing.assert_array_equal(good,a['eligible']);np.testing.assert_array_equal(effects,a['effects'])
            self.assertEqual(orders[arm][1]['symbols'],['D','A'])
        for key in a:np.testing.assert_array_equal(a[key],saved[key])


    def test_actual_kernel_oracle_host_interfaces_and_every_prefix(self):
        self.assertEqual(h.FEATURE_FLAGS,independent.FEATURE_FLAGS)
        self.assertEqual(h.DELTA_FLAGS,independent.DELTA_FLAGS)
        a,m,v=fixture();v,audit=kernel.build_features(a)
        independent.verify_features(a,v,audit)
        with patch.object(h,'read',return_value={}):orders,transforms,deltas=h.decisions(SyntheticStudy(),a,m,v,len(m['dates']))
        h.require_delta_audit(independent.verify_deltas(v,a,deltas),deltas)
        saved={k:(x.dtype.str,x.shape,x.tobytes()) for k,x in a.items()}
        for n in range(len(m['dates'])+1):
            aa=h.cache_prefix(a,n);small,sa=kernel.build_features(aa)
            self.assertEqual(sa['event_history'],audit['event_history'][:n])
            for key in v:np.testing.assert_array_equal(small[key],v[key][:n])
            with patch.object(h,'read',return_value={}):oo,tt,dd=h.decisions(SyntheticStudy(),a,m,v,n)
            self.assertEqual(oo,{k:vv[:n] for k,vv in orders.items()})
            h.require_delta_audit(independent.verify_deltas(small,aa,dd),dd)
            for k in tt:
                np.testing.assert_array_equal(tt[k],transforms[k][:n]);np.testing.assert_array_equal(dd[k],deltas[k][:n])
        self.assertEqual(saved,{k:(x.dtype.str,x.shape,x.tobytes()) for k,x in a.items()})

    def test_full122x49_receipts_and_required_flags(self):
        a,_,v=fixture(122,49);v,aa=kernel.build_features(a)
        feature=independent.verify_features(a,v,aa);_,d=h.scale_and_delta(v,122)
        delta=independent.verify_deltas(v,a,d)
        h.require_feature_audit(feature,v);h.require_delta_audit(delta,d)
        counts=h.estimate_counts(feature)
        base=dict(new_model_fits=0,predictive_model_fits=0,new_predictive_model_fits=0,
            model_fit_accounting=h.MODEL_FIT_ACCOUNTING,original_array_count=27,**counts)
        audit=dict(base,feature=feature,delta=delta,prefixes=123,feature_prefixes=list(range(123)),
            feature_prefix_sources_truncated=True,original_cache_exact=True,original_orders_exact=True,
            original_core_metadata_and_support_preserved=True,single_original_episode=True,original_arrays_unchanged=True)
        receipt=dict(base,new_returns_computed=False,artifacts_sha256={k:'synthetic' for k in h.PREPARED_FILES})
        h.verify_prepared_evidence(receipt,v,d,audit)
        for key in ('original_cache_exact','single_original_episode','original_arrays_unchanged','feature_prefix_sources_truncated'):
            bad=copy.deepcopy(audit);bad[key]=False
            with self.assertRaises(ValueError):h.verify_prepared_evidence(receipt,v,d,bad)
        for key in h.FEATURE_FLAGS:
            bad=copy.deepcopy(feature);bad.pop(key)
            with self.assertRaises(ValueError):h.require_feature_audit(bad,v)
        for key in h.DELTA_FLAGS:
            bad=copy.deepcopy(delta);bad.pop(key)
            with self.assertRaises(ValueError):h.require_delta_audit(bad,d)
        for key in ('new_predictive_model_fits',)+h.ESTIMATE_FIELDS:
            bad=copy.deepcopy(receipt);bad[key]=None
            with self.assertRaises(ValueError):h.verify_prepared_evidence(bad,v,d,audit)

    def test_physical_prefix_includes_all27_arrays_and_empty(self):
        a,_,_=fixture()
        for i in range(27-len(a)):a['unconsumed_'+str(i)]=np.full((6,2),object(),dtype=object)
        for n in (0,1,6):
            small=h.cache_prefix(a,n);self.assertEqual(set(small),set(a))
            self.assertTrue(all(x.shape[0]==n for x in small.values()))
            self.assertTrue(all(not np.shares_memory(small[k],a[k]) for k in a))
        a['bad']=np.zeros(5)
        with self.assertRaises(ValueError):h.cache_prefix(a,0)


class SourceAndLifecycleTests(unittest.TestCase):
    def test_actual_registered_cache_metadata_without_loading_npz(self):
        with patch.object(np,'load',side_effect=AssertionError('No actual arrays allowed')):
            binding=h.cache_source_binding()
        self.assertEqual(binding['original_array_count'],27)
        self.assertFalse(binding['fresh_raw_price_reconstruction'])
        self.assertIn(str(h.LOCAL/'DATA.npz'),binding['input_sha256'])
        self.assertIn(str(h.LOCAL/'ROW_IDENTITIES.npz'),binding['input_sha256'])
        self.assertNotIn(str(h.CENTRAL),binding['input_sha256'])

    def test_predecessor_metadata_and_central_counter_contract(self):
        state=h.read(h.CENTRAL);binding=h.predecessor_binding(state)
        self.assertEqual(binding['directory'],'monthly_volume_pair203')
        rows=[dict(trials=1,directory='old') for _ in range(201)]
        rows[0]['trials']=920;rows[-1]['directory']=h.PRIOR.name
        synthetic=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(synthetic);synthetic['completed'][-1]['directory']='wrong'
        with self.assertRaises(ValueError):h.state_check(synthetic)
        text=Path(h.__file__).read_text();self.assertIn("len(state['completed']) == 202",text);self.assertIn("== 1123",text)

    def test_contract_all123_cached_prefixes_and_no_source_rebuild(self):
        c=h.contract()
        self.assertEqual(c['arms'],list(h.NAMES));self.assertEqual(c['max_absolute_delta'],.5)
        self.assertEqual(c['verification']['feature_prefixes'],list(range(123)))
        self.assertEqual(len(h.PREPARED_FILES),7);self.assertEqual(c['execution_paths'],9)
        self.assertIn(h.COVERAGE,h.code_files())
        text=Path(h.__file__).read_text()
        self.assertNotIn('panel_load(',text);self.assertNotIn('SOURCE_REVIEW',text)
        self.assertIn('kernel.build_features(prefix_arrays)',text)

    def test_review_rejects_stale_code_and_mutable_central_pin(self):
        with patch.object(h,'read',return_value={'passed':False}):
            with self.assertRaises(ValueError):h.code_review_binding()
        review=dict(passed=True,unresolved_blockers=[],code_sha256={str(p.relative_to(h.ROOT)):h.sha(p) for p in h.code_files()[:6]},
            input_sha256={str(h.CENTRAL):'fake'},actual_features_computed=False,actual_portfolio_returns_computed=False,
            production_files_modified=False,strategy_registered_by_reviewer=False)
        with patch.object(h,'read',return_value=review),patch.object(h,'sha',return_value='fake'):
            review['code_sha256']={str(p.relative_to(h.ROOT)):'fake' for p in h.code_files()[:6]}
            with self.assertRaisesRegex(ValueError,'Mutable central'):h.code_review_binding()

    def test_count_gate_rejects_bad_types_and_row_accounting(self):
        a,_,_=fixture();_,audit=kernel.build_features(a);h.estimate_counts(audit)
        for key,value in [('rank_rows',5),('constant_rows',True),('two_level_rows',7),
                          ('active_forecast_cells',31),('checked_cells',29)]:
            bad=copy.deepcopy(audit);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):h.estimate_counts(bad)

    def test_outcomes_require_all_three_arms_nine_paths_fortyfive_slices(self):
        slices={'full122':{},'first88':{},'early53':{},'recent35':{},'later34':{}}
        check=dict(passed=True,independent_accounts=True,independent_of_production_model_and_engine=True,
            checked_paths=9,checked_slices=45,terminal_liquidation=False,
            accounting_source_sha256=h.sha(h.ROOT/'work/ptr169_results_independent.py'),
            paths={arm:{p[0]:dict(passed=True,monthly_executions=122,slices=copy.deepcopy(slices)) for p in h.ev.PATHS} for arm in h.NAMES})
        h.require_outcome_audit(check)
        for mutation in ('arm','path','slice','terminal'):
            bad=copy.deepcopy(check)
            if mutation=='arm':bad['paths'].pop(h.NAMES[2])
            if mutation=='path':bad['paths'][h.NAMES[1]].pop(h.ev.PATHS[2][0])
            if mutation=='slice':bad['paths'][h.NAMES[1]][h.ev.PATHS[0][0]]['slices'].pop('later34')
            if mutation=='terminal':bad['terminal_liquidation']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):h.require_outcome_audit(bad)
    def test_exact38_result_manifest_missing_extra_and_completion_rejected(self):
        manifest={name:'synthetic-digest' for name in h.RESULT_FILES}
        self.assertEqual(len(manifest),38)
        self.assertNotIn('COMPLETION.json',manifest)
        h.require_result_manifest(manifest)
        for missing in h.RESULT_FILES:
            bad=dict(manifest);bad.pop(missing)
            with self.subTest(missing=missing),self.assertRaises(ValueError):h.require_result_manifest(bad)
        for extra in ('COMPLETION.json','EXTRA.json'):
            bad=dict(manifest);bad[extra]='synthetic-digest'
            with self.subTest(extra=extra),self.assertRaises(ValueError):h.require_result_manifest(bad)
        for bad in (None,list(manifest),{}):
            with self.assertRaises(ValueError):h.require_result_manifest(bad)

    def test_write_once_and_protection_before_publishing_and_central_cas(self):
        code=Path(h.__file__).read_text();tree=ast.parse(code)
        parts={x.name:ast.get_source_segment(code,x) for x in tree.body if isinstance(x,ast.FunctionDef)}
        register=parts['register'];self.assertLess(register.index('code_review_binding()'),register.index('OUT.mkdir()'))
        self.assertLess(register.index('len(protected) == 21444'),register.index('OUT.mkdir()'))
        self.assertIn('Code changed since tests',register)
        self.assertLess(register.index('Mutable central cannot enter registered input pins'),register.index('OUT.mkdir()'))
        closer=parts['close']
        self.assertLess(closer.index('require_result_manifest('),closer.index('require_outcome_audit('))
        self.assertLess(closer.index('require_result_manifest('),closer.index("results = read("))
        self.assertEqual(parts['commit_state'].count('CENTRAL.read_bytes() == before'),2)
        self.assertIn("not(OUT/'prepared').exists()",parts['prepare'].replace(' ',''))
        self.assertIn("not (OUT / 'selected').exists()",parts['select'])
        self.assertIn('Selection escaped first88 cutoff',parts['select'])


if __name__=='__main__':unittest.main()


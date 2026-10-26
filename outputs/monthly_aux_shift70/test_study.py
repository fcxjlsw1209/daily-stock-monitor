"""Synthetic effect-slot, selector, source identity and frozen protocol tests."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('aux_shift70_study_tests',P/'study.py')
s=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(s)
POLICY=dict(s.POLICY_FIELDS)


def fixture():
    shape=(4,6);cw=np.array([.52,.5,.52,.5])[:,None]
    support=np.tile([.1,-.1,.05,0.,-.05,.1],(4,1))
    core=cw*np.tile([.45,.4,.3,.29,.28,.27],(4,1))
    available=np.tile([True,True,True,True,False,False],(4,1))
    rank=np.tile([.4,0.,-.3,.2,np.nan,np.nan],(4,1))
    a=dict(baseline=core+(1-cw)*support,core=core,core_weight=cw,
        eligible=np.ones(shape,bool),state_calm10=np.array([False,True,False,True]),
        state_weak_narrow=np.array([False,False,True,True]),
        pre_score=np.tile([.30,.25,.20,.19,.181,.1805],(4,1)),
        effects=np.tile([0.,0.,-.10,.05,.15,.50],(4,1)),
        forecast_available=np.ones(shape,bool),extension_available=np.ones(shape,bool),
        old_ml4=np.zeros(shape),rank_log_excess=np.tile([-.4,-.3,-.2,.1,.3,.5],(4,1)),
        bond_available=available.copy(),bond_effect=rank.copy(),
        ulcer_available=available.copy(),ulcer_rank=rank.copy())
    return a,dict(symbols=[list('ABCDEF') for _ in range(4)],
        dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31'],
        ends=['2016-08-31','2016-09-30','2016-10-31','2016-11-30'])


class StudyTests(unittest.TestCase):
    def setUp(self):self.a,self.m=fixture()


    def choose(self,ident,n=4,**kw):return s.choices(s.configs()[ident],self.a,self.m,n,policy=POLICY,**kw)


    def rows(self):
        def metric(v):return dict(cagr=v,sharpe=1.6,max_drawdown=-.3)
        return [dict(spec=z,training=metric(.5 if z['id']==2 else .49),early=metric(.4),recent=metric(.6)) for z in s.configs()]


    def test_fixed_five_grid_and_exact_source(self):
        self.assertEqual(s.design()['grid'],s.configs());self.assertEqual(len(s.configs()),5)
        self.assertEqual(s.configs()[3],dict(id=3,base='incumbent_scope67',mode='ulcer_shift'))
        self.assertEqual(s.configs()[4],dict(id=4,base='incumbent_scope67',mode='bond_shift'))
        self.assertEqual(s.SOURCE_PRIMARY_SPEC,s.upstream.configs()[3])
        self.assertEqual([s.diagnostic_name(x) for x in s.configs()[3:]],['ulcer_shift','bond_shift'])


    def test_three_controls_bypass_new_features_and_adapters(self):
        del self.a['bond_effect'];del self.a['bond_available'];del self.a['ulcer_rank'];del self.a['ulcer_available']
        with patch.object(s,'score_delta',side_effect=AssertionError('controls bypass')):
            for ident,source in ((0,0),(1,1),(2,3)):
                expected=s.upstream.choices(s.upstream.configs()[source],self.a,self.m,4,policy=POLICY)
                actual=self.choose(ident)
                self.assertEqual(actual[0],expected[0]);np.testing.assert_array_equal(actual[1],expected[1])
                self.assertEqual(s.rp.make_orders(*actual,self.m),s.rp.make_orders(*expected,self.m))


    def keys(self,ident):
        return ('ulcer_rank','ulcer_available') if ident==3 else ('bond_effect','bond_available')

    def test_exact_shift_formula_original_support_and_actual_pre(self):
        support=s.original_support(self.a,4)
        source=s.upstream.modified_pre_score(s.SOURCE_PRIMARY_SPEC,self.a,4)
        for ident in (3,4):
            key,mask=self.keys(ident);factor=self.a[key];available=self.a[mask]
            active=available&(factor!=0.);want=np.zeros(factor.shape)
            want[active]=.05*(np.clip(support[active]+factor[active],-.5,.5)-support[active])
            np.testing.assert_array_equal(s.score_delta(s.configs()[ident],self.a,4),want)
            expected=source.copy();changed=want!=0.;expected[changed]+=want[changed]
            np.testing.assert_array_equal(s.modified_pre_score(s.configs()[ident],self.a,4),expected)
        wrong=(source-self.a['core'])/(1-self.a['core_weight'])
        self.assertTrue(np.any(wrong!=support))

    def test_independent_masks_never_intersect_or_use_other_arm(self):
        for ident in (3,4):
            a=copy.deepcopy(self.a);other=4 if ident==3 else 3
            key,mask=self.keys(other);a[key][:]=np.nan;a[mask][:]=False
            got=s.choices(s.configs()[ident],a,self.m,4,policy=POLICY);expected=self.choose(ident)
            self.assertEqual(got[0],expected[0]);np.testing.assert_array_equal(got[1],expected[1])
            own_key,own_mask=self.keys(ident)
            a[own_key][0,-1]=.25;a[own_mask][0,-1]=True
            self.assertNotEqual(s.score_delta(s.configs()[ident],a,4)[0,-1],0.)

    def test_all_zero_signed_zero_and_all_missing_exact_source(self):
        source=self.choose(2)
        for ident in (3,4):
            key,mask=self.keys(ident)
            for value,valid in ((0.,True),(-0.,True),(np.nan,False)):
                self.a[key][:]=value;self.a[mask][:]=valid
                got=self.choose(ident)
                self.assertEqual(got[0],source[0]);self.assertEqual(got[1].tobytes(),source[1].tobytes())
                self.assertEqual(s.rp.make_orders(*got,self.m),s.rp.make_orders(*source,self.m))
                self.assertEqual(s.score_delta(s.configs()[ident],self.a,4).tobytes(),np.zeros((4,6)).tobytes())

    def test_clipping_boundaries_and_signs(self):
        self.a['core_weight'][:]=.5;self.a['core'][:]=0.
        support=np.tile([-.5,-.49,0.,0.,.49,.5],(4,1));self.a['baseline']=.5*support
        for ident in (3,4):
            key,mask=self.keys(ident);self.a[mask][:]=True
            for sign in (1.,-1.):
                self.a[key][:]=sign*.5
                delta=s.score_delta(s.configs()[ident],self.a,4)
                np.testing.assert_array_equal(delta,.05*(np.clip(support+sign*.5,-.5,.5)-support))
                self.assertTrue((np.abs(delta)<=.025).all());self.assertTrue((sign*delta>=0.).all())
                self.assertEqual(delta[0,-1 if sign>0 else 0],0.)

    def test_zero_identity_with_tolerated_support_roundoff_no_isclose_gate(self):
        self.a['core_weight'][:]=.5;self.a['core'][:]=0.
        support=np.tile([.5+5e-13,-.5-5e-13,.5,-.5,0.,0.],(4,1));self.a['baseline']=.5*support
        for ident in (3,4):
            key,mask=self.keys(ident);self.a[key][:]=0.;self.a[key][:,1::2]=-0.;self.a[mask][:]=True
            np.testing.assert_array_equal(s.score_delta(s.configs()[ident],self.a,4),0.)
            self.a[key][:,4:]=[1e-20,-1e-20]
            delta=s.score_delta(s.configs()[ident],self.a,4)
            np.testing.assert_array_equal(delta[:,4:],.05*self.a[key][:,4:])
            self.assertTrue((delta[:,4]!=0.).all())
        self.a['baseline'][0,0]=.5*(.5+2e-12)
        with self.assertRaises(ValueError):s.score_delta(s.configs()[3],self.a,4)

    def test_missing_and_zero_cells_only_guarantee_pre_identity(self):
        source=s.upstream.modified_pre_score(s.SOURCE_PRIMARY_SPEC,self.a,4)
        for ident in (3,4):
            key,mask=self.keys(ident);inactive=~self.a[mask]|(self.a[key]==0.)
            np.testing.assert_array_equal(s.score_delta(s.configs()[ident],self.a,4)[inactive],0.)
            np.testing.assert_array_equal(s.modified_pre_score(s.configs()[ident],self.a,4)[inactive],source[inactive])

    def test_nonzero_ineligible_factor_does_not_change_universe(self):
        for ident in (3,4):
            before=s.score_delta(s.configs()[ident],self.a,4)
            self.a['eligible'][:,0]=False
            np.testing.assert_array_equal(s.score_delta(s.configs()[ident],self.a,4),before)
            self.assertTrue(all(0 not in row for row in self.choose(ident)[0]))

    def test_both_modes_use_modified_pre_but_original_selector_payload(self):
        for ident in (3,4):
            captured={}
            def capture(pre,eligible,symbols,effects,original,policy):
                captured.update(pre=pre.copy(),eligible=eligible.copy(),effects=effects.copy(),original=original,policy=policy)
                return s.ep.select_boundary(pre,eligible,symbols,effects,original,policy)
            got=self.choose(ident,boundary_selector=capture)
            pre=s.modified_pre_score(s.configs()[ident],self.a,4)
            np.testing.assert_array_equal(captured['pre'],pre)
            np.testing.assert_array_equal(captured['effects'],self.a['effects'])
            np.testing.assert_array_equal(captured['eligible'],self.a['eligible'])
            for key in ('baseline','core','state_calm10','state_weak_narrow'):
                np.testing.assert_array_equal(captured['original'][key],self.a[key])
            np.testing.assert_array_equal(captured['original']['core_weight'],np.broadcast_to(self.a['core_weight'],pre.shape))
            self.assertEqual(captured['policy'],POLICY)
            for i,row in enumerate(pre):
                ranked=sorted(np.flatnonzero(self.a['eligible'][i]),key=lambda j:(-row[j],self.m['symbols'][i][j]))
                self.assertTrue(set(ranked[:2]).issubset(got[0][i]))
                allowed={j for j in ranked[2:5] if row[ranked[2]]-row[j]<=.02}
                self.assertTrue(set(got[0][i])-set(ranked[:2])<=allowed)
                np.testing.assert_array_equal(got[1][i,ranked[:2]],pre[i,ranked[:2]])


    def test_both_modes_may_change_original_top2_before_boundary(self):
        self.a['state_calm10'][:]=False;self.a['state_weak_narrow'][:]=False
        self.a['pre_score'][:]=[.200,.199,.198,.197,.196,.195]
        self.a['bond_effect'][:]=[-.5,-.5,.5,.5,.5,.5];self.a['bond_available'][:]=True
        self.a['ulcer_rank'][:]=self.a['bond_effect'];self.a['ulcer_available'][:]=True
        for ident in (3,4):
            pre=s.modified_pre_score(s.configs()[ident],self.a,4)
            anchors=sorted(range(6),key=lambda j:(-pre[0,j],self.m['symbols'][0][j]))[:2]
            self.assertNotEqual(set(anchors),{0,1})
            self.assertTrue(set(anchors)<=set(self.choose(ident)[0][0]))


    def test_aux_support_reserve_includes_scope_true_mask_and_old_episode(self):
        weights=s.support_budget(self.a,4)
        cw=np.broadcast_to(self.a['core_weight'],self.a['baseline'].shape)
        old=.025*self.a['state_calm10'][:,None]+.05*self.a['state_weak_narrow'][:,None]
        active=(self.a['state_calm10']&~self.a['state_weak_narrow'])[:,None]&self.a['eligible']&self.a['extension_available']&self.a['forecast_available']&np.isfinite(self.a['rank_log_excess'])
        want=1-cw-old-np.minimum(.05,np.maximum(0.,.10-old))-.01*active
        np.testing.assert_array_equal(weights['remaining_original_support'],want)
        after=want-.05*self.a['bond_available'];self.assertGreaterEqual(after.min(),.33-1e-15)
        self.assertTrue((np.abs(s.score_delta(s.configs()[3],self.a,4))<=.025).all())
        self.a['extension_available'][1,0]=False;self.a['forecast_available'][1,1]=False
        self.a['eligible'][1,2]=False;self.a['rank_log_excess'][1,3]=np.nan
        np.testing.assert_array_equal(s.support_budget(self.a,4)['scope_extension'][1,:4],0.)


    def test_inputs_unchanged_no_global_post_rerank(self):
        before=copy.deepcopy(self.a)
        for ident in range(5):self.choose(ident)
        for key in before:np.testing.assert_array_equal(self.a[key],before[key])
        self.a['baseline']=self.a['pre_score'].copy();self.a['core']=self.a['pre_score'].copy()
        self.a['state_calm10'][:]=False;self.a['state_weak_narrow'][:]=False
        self.a['effects'][:]=[0.,0.,-.5,-.5,-.5,0.]
        self.a['bond_effect'][:]=np.nan;self.a['bond_available'][:]=False
        self.a['ulcer_rank'][:]=np.nan;self.a['ulcer_available'][:]=False
        for ident in (3,4):
            chosen,score=self.choose(ident)
            self.assertTrue((score[:,5]>score[:,2]).all());self.assertEqual(chosen,[[0,1,2]]*4)


    def test_eligibility_cash_twohalf_and_no_extra_signal_requirement(self):
        for count in (0,1,2):
            self.a['eligible'][:]=False;self.a['eligible'][:,:count]=True
            for ident in (3,4):
                chosen,score=self.choose(ident)
                self.assertEqual([len(row) for row in chosen],[count if count==2 else 0]*4)
                np.testing.assert_array_equal(score,s.modified_pre_score(s.configs()[ident],self.a,4)[:4])


    def test_all_prefixes_and_future_poison_independent(self):
        for ident in range(5):
            full=self.choose(ident)
            for n in range(5):
                got=self.choose(ident,n=n)
                self.assertEqual(got[0],full[0][:n]);np.testing.assert_array_equal(got[1],full[1][:n])
                poison=copy.deepcopy(self.a)
                for value in poison.values():value[n:]=False if value.dtype.kind=='b' else np.nan
                meta=copy.deepcopy(self.m);meta['symbols'][n:]=[[None]]*(4-n)
                other=s.choices(s.configs()[ident],poison,meta,n,policy=POLICY)
                self.assertEqual(other[0],got[0]);np.testing.assert_array_equal(other[1],got[1])


    def test_bad_specs_prefixes_masks_and_funding_fail(self):
        with self.assertRaises(ValueError):s.choices(dict(s.configs()[3],mode='other'),self.a,self.m,4,policy=POLICY)
        for ident in (3,4):
            for n in (-1,5,True,1.5):
                with self.assertRaises(ValueError):self.choose(ident,n=n)
            key,mask=self.keys(ident)
            for field,value in ((key,np.inf),(key,.51),(mask,False)):
                a=copy.deepcopy(self.a);a[field][1,0]=value
                with self.assertRaises(ValueError):s.choices(s.configs()[ident],a,self.m,4,policy=POLICY)
            a=copy.deepcopy(self.a);a[mask]=a[mask].astype(int)
            with self.assertRaises(ValueError):s.choices(s.configs()[ident],a,self.m,4,policy=POLICY)
            with patch.object(s,'support_budget',return_value={'remaining_original_support':np.full((4,6),.049)}):
                with self.assertRaises(ValueError):self.choose(ident)

    def test_nonfinite_original_inputs_rejected_even_for_zero_factor(self):
        for ident in (3,4):
            key,mask=self.keys(ident)
            for field in ('baseline','core'):
                for value in (np.nan,np.inf,-np.inf):
                    a=copy.deepcopy(self.a);a[key][:]=0.;a[mask][:]=True;a[field][0,0]=value
                    with self.assertRaises(ValueError):s.score_delta(s.configs()[ident],a,4)

    def test_audit_records_zero_missing_saturation_and_no_normalization(self):
        self.a['core_weight'][:]=.5;self.a['core'][:]=0.
        self.a['baseline'][:]=[.25,-.25,0.,0.,0.,0.]
        for ident in (3,4):
            key,mask=self.keys(ident);self.a[key][:]=[.5,-.5,0.,-.25,np.nan,np.nan]
            audit=s.placement_audit(s.configs()[ident],self.a,4)
            self.assertEqual(audit['upper_saturation_cells'],4);self.assertEqual(audit['lower_saturation_cells'],4)
            self.assertEqual(audit['zero_cells'],4);self.assertEqual(audit['missing_cells'],8)
            self.assertEqual(audit['modified_cells'],4);self.assertFalse(audit['normalization_changed'])
            self.assertLessEqual(audit['max_abs_delta'],.025)

    def test_synthetic_frozen_feature_contracts_to_actual_selectors(self):
        index=pd.bdate_range('2014-01-01','2017-01-31');t=np.arange(len(index)-1)
        x=.007*np.sin(t/11)+.003*np.cos(t/5);b=.004*np.cos(t/13)+.002*np.sin(t/7)+.0001
        spy=100*np.exp(np.r_[0.,np.cumsum(x)]);bond=80*np.exp(np.r_[0.,np.cumsum(b)])
        adj=pd.DataFrame(dict(A=spy,B=bond,C=spy*bond,D=spy/np.sqrt(bond),
            E=np.ones(len(index)),F=np.full(len(index),np.nan),SPY=spy,TLT=bond),index=index)
        class Calendar:
            def sessions_in_range(self,start,end):return pd.bdate_range(start,end)
        for study,prefix in ((s.ulcer,'ulcer_'),(s.bond,'bond_')):
            arrays,_=study.features.build_features(adj,self.m,Calendar())
            self.a.update({prefix+key:value.copy() for key,value in arrays.items()})
        s.validate_cached_features(self.a,(4,6))
        np.testing.assert_array_equal(self.a['bond_effect'][:,0],0.)
        for ident in (3,4):
            self.assertTrue(all(len(row)==3 for row in self.choose(ident)[0]))
            np.testing.assert_array_equal(s.score_delta(s.configs()[ident],self.a,4)[:,5],0.)
        # Complete flat Ulcer rawQ0 retains its existing mapped rank, not forced0.
        self.assertTrue(self.a['ulcer_available'][:,4].all())
        bad=copy.deepcopy(self.a);bad['ulcer_rank'][0,0]=np.nan
        with self.assertRaises(ValueError):s.validate_cached_features(bad,(4,6))
        bad=copy.deepcopy(self.a);bad['bond_raw'][0,0]=.1
        with self.assertRaises(AssertionError):s.validate_cached_features(bad,(4,6))

    def test_shared_array_dtype_values_shape_and_nan_exact(self):
        original={'x':np.array([[0.,np.nan],[1.,2.]]),'b':np.array([True,False])}
        s.assert_original_arrays(original,copy.deepcopy(original))
        for changed in ({'x':original['x'].astype(np.float32),'b':original['b']},
                {'x':original['x']+1,'b':original['b']}, {'x':original['x'][0],'b':original['b']},
                {'x':np.nan_to_num(original['x']),'b':original['b']}, {'x':original['x']}):
            with self.assertRaises(AssertionError):s.assert_original_arrays(original,changed)


    def test_raw64_hash_parity_and_missing_conflicts_rejected(self):
        loader=str(s.O/'monthly_stability_monitor/inputs.py')
        raw={str(i):'h' for i in range(63)};raw[loader]=s.sha(loader)
        manifests=[dict(raw,extra='ok') for _ in range(3)]
        self.assertEqual(s.validate_raw_hashes(raw,manifests),raw)
        for bad in (dict(raw,extra='bad'),{k:v for k,v in raw.items() if k!=loader},dict(raw,**{loader:'bad'})):
            with self.assertRaises(AssertionError):s.validate_raw_hashes(bad,manifests)
        for i in range(3):
            bad=copy.deepcopy(manifests);bad[i]['0']='wrong'
            with self.assertRaises(AssertionError):s.validate_raw_hashes(raw,bad)
            del bad[i]['0']
            with self.assertRaises(AssertionError):s.validate_raw_hashes(raw,bad)


    def bundle_context(self,*,metadata=False,policy=False,price=False,shared=False,collision=False,calendar=False,raw=False):
        stack=ExitStack();self.addCleanup(stack.close)
        original={key:value for key,value in self.a.items() if not key.startswith(('ulcer_','bond_'))}
        bs=[dict(copy.deepcopy(original),shared_extra=np.array([1.])) for _ in range(2)]
        for data,prefix in zip(bs,('ulcer_','bond_')):data.update({key:value.copy() for key,value in self.a.items() if key.startswith(prefix)})
        index=pd.bdate_range('2020-01-01',periods=3);px=pd.DataFrame(np.arange(18).reshape(3,6)+1.,index=index,columns=list('ABCDEF'))
        op=px*.99;cm=copy.deepcopy(self.m);cp=px.copy()
        if metadata:cm['symbols'][0][0]='WRONG'
        if price:cp.iloc[0,0]+=1.
        if shared:bs[1]['core'][0,0]+=1.
        if collision:bs[1]['shared_extra'][0]=2.
        stack.enter_context(patch.object(s.upstream,'load',return_value=(self.m,original,px,op)))
        stack.enter_context(patch.object(s.ulcer,'load',return_value=(self.m,bs[0],px,op)))
        stack.enter_context(patch.object(s.bond,'load',return_value=(cm,bs[1],cp,op)))
        protection={str(i):'p' for i in range(21436)}
        def read(path):
            path=Path(path)
            if path.name=='INCUMBENT_POLICY.json':return dict(POLICY,budget=.1) if policy and path.parent==s.bond.P else POLICY
            if path.name=='PROTECTED.json':return protection
            if path.name=='PLAN.json':return {'raw_source_hashes':{'raw':'different' if raw and path.parent==s.bond.P else 'hash'},'calendar':{'version':'bad' if calendar and path.parent==s.bond.P else 'test'}}
            if path.name=='FEATURE_AUDIT.json':return {'calendar':{'version':'test'}}
            if path.name=='INPUT_HASHES.json':return {'raw':'hash'}
            raise AssertionError('Unexpected file read: '+str(path))
        stack.enter_context(patch.object(s,'read',side_effect=read))
        stack.enter_context(patch.object(s,'validate_raw_hashes',return_value={'raw':'hash'}))
        for study in s.components():
            stack.enter_context(patch.object(study,'validate_features'))
            stack.enter_context(patch.object(study.features,'calendar_provenance',return_value={'version':'test'}))
        return stack


    def test_component_bundles_match_meta_policy_prices_source_and_cross_branch_arrays(self):
        with self.bundle_context():
            m,a,px,op,policy,bundles,raw,calendar=s.source_bundles()
            self.assertEqual(m,self.m);self.assertEqual(policy,POLICY);self.assertEqual(raw,{'raw':'hash'})
            self.assertEqual(len(bundles),2)
        for mode in ('metadata','policy','price','shared','collision','calendar','raw'):
            with self.subTest(mode=mode),self.bundle_context(**{mode:True}):
                with self.assertRaises(AssertionError):s.source_bundles()


    def test_component_completion_checks_no_performance_values(self):
        good=dict(original_controls_exact=True,independent_fixed_unit_replay=True,training_freeze_parity=True,
            fixed_configs=5,protected_files=21436,protected_changes=[],observer_accounts_changed=False)
        s.validate_component_completion(good)
        for key in good:
            bad=dict(good);del bad[key]
            with self.assertRaises(AssertionError):s.validate_component_completion(bad)


    def test_old_controls_not_promoted_both_diagnostics_frozen(self):
        rows=self.rows();rows[0]['training']['cagr']=10.;rows[1]['training']['cagr']=20.
        chosen=s.select(rows)
        self.assertEqual(chosen['primary']['spec']['id'],2)
        self.assertEqual(len(chosen),6)
        for row in rows[3:]:
            self.assertEqual(chosen[s.diagnostic_name(row['spec'])]['spec'],row['spec'])
            self.assertFalse(row['admitted'])


    def test_training_guards_tie_tolerance_ignore_later_fields(self):
        rows=self.rows();rows[3]['training']['cagr']=.5+5e-13
        self.assertEqual(s.select(rows)['primary']['spec']['id'],2)
        rows[3]['training']['cagr']=.5+2e-12
        self.assertEqual(s.select(rows)['primary']['spec']['id'],3)
        for section,key,value in (('training','cagr',.49),('training','sharpe',1.54),
                ('training','max_drawdown',-.331),('early','cagr',.374),('recent','cagr',.574)):
            rows=self.rows();rows[3]['training']['cagr']=.99;rows[3][section][key]=value
            rows[3]['later_cagr']=1000.
            self.assertEqual(s.select(rows)['primary']['spec']['id'],2)


    def test_freeze_and_protection_phase_boundaries(self):
        source=(P/'study.py').read_text();tree=ast.parse(source)
        functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
        protection_calls=[]
        for name,node in functions.items():
            for item in ast.walk(node):
                if isinstance(item,ast.Call) and isinstance(item.func,ast.Name) and item.func.id=='assert_protected':
                    protection_calls.append(name)
        self.assertEqual(protection_calls,['evaluate'])
        train=ast.get_source_segment(source,functions['train'])
        for text in ('rp.paths(px,m,TRAIN)','choices(spec,a,m,TRAIN)','both_diagnostics_frozen=True','trials=5',
                     "assert not (P/'FROZEN.json').exists()"):
            self.assertIn(text,train)
        evaluate=ast.get_source_segment(source,functions['evaluate'])
        self.assertIn("frozen['selected'].items()",evaluate)
        self.assertNotIn('select(',evaluate)


    def test_dependencies_pin_both_caches_without_outcome_inputs(self):
        files=set(s.implementations());sources=set(s.source_files())
        for path in (P/'study.py',P/'test_study.py',Path(s.upstream.__file__),Path(s.adapter.__file__),Path(s.ep.__file__)):
            self.assertIn(path,files)
        for study in s.components():
            for path in (Path(study.__file__),Path(study.features.__file__),study.P/'test_study.py',study.P/'test_features.py',*study.features.REFERENCE_PATHS):
                self.assertIn(path,files)
            for name in ('DESIGN.json','DESIGN.sha256','PLAN.json','FROZEN.json','FROZEN.sha256','DATA.npz','META.json',
                    'INCUMBENT_POLICY.json','INPUT_HASHES.json','CACHE_HASHES.json','FEATURE_AUDIT.json','PLACEMENT_AUDIT.json',
                    'VALIDATION.json','original65_ORDERS.json','incumbent66_ORDERS.json','incumbent_scope67_ORDERS.json'):
                self.assertIn(study.P/name,sources)
        self.assertFalse({'RESULTS.json','TRAIN_TRIALS.json','TRAIN_RETURNS.npz'}&{p.name for p in sources})
        self.assertNotIn(P/'features.py',files)

    def test_no_feature_rebuild_and_provenance_preserved_each_phase(self):
        source=(P/'study.py').read_text();tree=ast.parse(source)
        fs={x.name:ast.get_source_segment(source,x) for x in tree.body if isinstance(x,ast.FunctionDef)}
        self.assertNotIn('build_features(',source)
        for name in ('initialize','prepare'):self.assertIn('verify_source()',fs[name])
        self.assertIn("old.assert_hashes(plan['raw_source_hashes'])",fs['registered'])
        self.assertIn("study.features.calendar_provenance()==plan['calendar']",fs['registered'])
        self.assertIn('validate_cached_features(a,shape)',fs['prepare'])
        self.assertIn('no_component_performance_filter=True',fs['prepare'])
        self.assertIn("('next_open_zero',0.),('next_open_cost',.0005)",fs['evaluate'])
        self.assertNotIn('zero_shrink',source)


if __name__=='__main__':unittest.main()

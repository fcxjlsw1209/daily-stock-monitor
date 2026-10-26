"""Synthetic cached-rank composites; no actual input, fitting or return reads."""
import ast
import copy
import inspect
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

P=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('boundary_preserve70_study_tests',P/'study.py')
s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
parent=s.module('preserve70_parent_fixture',s.local.P/'test_study.py')
POLICY=dict(s.POLICY_FIELDS)


def fixture():
    a,meta=parent.fixture()
    ranks=([.5,-.5,-.5,.4,.5,.5],[-.5,.5,.35,-.5,.1,-.5],[.5,.5,-.4,0.,.35,.5])
    for (_,rank,available),values in zip(s.FACTOR_SOURCES.values(),ranks):
        a[rank]=np.tile(values,(4,1))
        a[available]=np.ones((4,6),bool)
    a['leadlag_available'][:,[0,1,3,5]]=False
    a['rank_leadlag'][~a['leadlag_available']]=np.nan
    return a,meta


class StudyTests(unittest.TestCase):
    def setUp(self):self.a,self.m=fixture()

    def choose(self,ident=4,n=4,*,a=None,meta=None,**kw):
        return s.choices(s.configs()[ident],self.a if a is None else a,
            self.m if meta is None else meta,n,policy=POLICY,**kw)

    def support(self,a=None,n=4):
        a=self.a if a is None else a
        cw=np.broadcast_to(a['core_weight'],a['core'].shape)[:n]
        return (a['baseline'][:n]-a['core'][:n])/(1-cw)

    def keys(self,ident):return tuple(s.FACTOR_SOURCES.values())[ident-4][1:]

    def reference(self,ident=4,a=None,n=4,meta=None):
        a=self.a if a is None else a;meta=self.m if meta is None else meta
        key,available=self.keys(ident)
        pre=s.local.modified_pre_score(s.LOCAL_SPEC,a,n)[:n]
        chosen,scores=s.local.choices(s.LOCAL_SPEC,a,meta,n,policy=POLICY)
        chosen=copy.deepcopy(chosen);scores=scores.copy();groups=[];anchors=[]
        support=self.support(a,n)
        for i,names in enumerate(meta['symbols'][:n]):
            order=sorted(np.flatnonzero(a['eligible'][i]),key=lambda j:(-pre[i,j],names[j]))
            anchors.append(order[:2])
            group=[] if len(order)<3 else [j for j in order[2:5] if pre[i,order[2]]-pre[i,j]<=.02]
            groups.append(group);changed=False
            for j in group:
                factor=a[key][i,j]
                if a[available][i,j] and factor!=0.:
                    bounded=min(.5,max(-.5,support[i,j]+factor))
                    delta=.025*(bounded-support[i,j])
                    if delta!=0.:
                        scores[i,j]+=delta;changed=True
            if changed:
                winner=min(group,key=lambda j:(-scores[i,j],names[j]))
                chosen[i]=sorted(order[:2]+[winner],key=lambda j:(-scores[i,j],names[j]))
        return (chosen,scores),anchors,groups

    def test_seven_fixed_configs_dependencies_boundary_and_no_source_outcomes(self):
        design,digest=s.design()
        self.assertEqual(digest,'ae0a1a4c159c4da6bf4c078eb365fc8d0eb37268777c998d62c9571796e131ed')
        self.assertEqual(design['configs'],s.configs());self.assertEqual(len(s.configs()),7)
        self.assertEqual([x['mode'] for x in s.configs()[4:]],
            ['ordinal25_preserve','cotail25_preserve','leadlag25_preserve'])
        self.assertEqual(s.LOCAL_SPEC,s.local.configs()[4]);self.assertEqual(s.TRAIN,88)
        self.assertEqual(s.PROTECTED_COUNT,21444);self.assertEqual(str(s.BOUNDARY),design['boundary'])
        files=set(s.source_files());implementation=set(s.implementations())
        for path in (s.P/'placement.py',s.P/'test_placement.py',s.placement.SOURCE_PLACEMENT_PATH,
                s.placement.SOURCE_PLACEMENT_PATH.with_name('test_placement.py'),
                Path(s.placement.path_adapter.__file__)):
            self.assertIn(path,files);self.assertIn(path,implementation)
        for folder,_,_ in s.FACTOR_SOURCES.values():
            for name in s.FACTOR_FILES:self.assertIn(folder/name,files)
            for name in ('FROZEN.json','RESULTS.json','TRAIN_TRIALS.json'):
                self.assertNotIn(folder/name,files)
        self.assertNotIn('panel_load',inspect.getsource(s.prepare))
        self.assertIn('trials=7',inspect.getsource(s.train))
        self.assertIn('TESTS.txt',inspect.getsource(s.train))

    def test_four_controls_bypass_all_new_ranks_masks_and_match_orders(self):
        for key in s.FACTOR_KEYS:del self.a[key]
        with patch.object(s,'feature_rank',side_effect=AssertionError('control read new ranks')):
            for ident in range(4):
                old=s.local.choices(s.source_spec(s.configs()[ident]),self.a,self.m,4,policy=POLICY)
                got=self.choose(ident)
                self.assertEqual(got[0],old[0]);self.assertEqual(got[1].tobytes(),old[1].tobytes())
                self.assertEqual(s.rp.make_orders(*got,self.m),s.rp.make_orders(*old,self.m))

    def test_each_arm_copies_only_own_rank_and_mask_without_reranking(self):
        for ident in (4,5,6):
            a=copy.deepcopy(self.a);rank,available=self.keys(ident)
            for key in s.FACTOR_KEYS:
                if key not in (rank,available):a[key]='must not inspect'
            got=s.feature_rank(s.configs()[ident],a,4)
            self.assertEqual(got.tobytes(),self.a[rank].tobytes())
            expected=self.reference(ident)[0];actual=self.choose(ident,a=a)
            self.assertEqual(actual[0],expected[0]);np.testing.assert_array_equal(actual[1],expected[1])
        # A constant already-ranked .2 is used as .2, never reranked to 1/(2n).
        self.a['rank_ordinal'][:]=.2
        np.testing.assert_array_equal(s.feature_rank(s.configs()[4],self.a,4),.2)

    def test_composite_scalar_oracle_only_changes_original_cohort(self):
        old=self.choose(3)
        for ident in (4,5,6):
            got=self.choose(ident);want,anchors,groups=self.reference(ident)
            self.assertEqual(got[0],want[0]);np.testing.assert_array_equal(got[1],want[1])
            for i,group in enumerate(groups):
                self.assertTrue(set(anchors[i]).issubset(got[0][i]))
                for j in range(6):
                    if j not in group:self.assertEqual(got[1][i,j],old[1][i,j])
            self.assertLessEqual(np.max(np.abs(got[1]-old[1])),.0125+1e-16)

    def test_clip_both_directions_and_nonbinding_composite_is_point025F(self):
        self.a['state_calm10'][:]=False;self.a['state_weak_narrow'][:]=False
        self.a['pre_score'][:]=[.30,.25,.20,.195,.19,.18]
        support=np.tile([.1,-.1,.49,-.49,0.,0.],(4,1))
        self.a['baseline']=self.a['core']+(1-self.a['core_weight'])*support
        self.a['rank_ordinal'][:]=[.5,-.5,.5,-.5,.2,.5]
        old=self.choose(3);got=self.choose(4)
        want=self.reference(4)[0]
        np.testing.assert_array_equal(got[1],want[1])
        np.testing.assert_allclose(got[1][:,2]-old[1][:,2],.025*(.5-self.support()[:,2]),atol=1e-17,rtol=0)
        np.testing.assert_allclose(got[1][:,3]-old[1][:,3],.025*(-.5-self.support()[:,3]),atol=1e-17,rtol=0)
        np.testing.assert_allclose(got[1][:,4]-old[1][:,4],.025*.2,atol=1e-17,rtol=0)

    def test_original_local_pre_payload_and_episode_exactly_once(self):
        for ident in (4,5,6):
            calls=[]
            def selector(pre,eligible,symbols,effects,original,policy):
                calls.append((pre.copy(),copy.deepcopy(original),effects.copy(),eligible.copy()))
                return s.ep.select_boundary(pre,eligible,symbols,effects,original,policy)
            self.choose(ident,boundary_selector=selector)
            self.assertEqual(len(calls),1)
            pre,original,effects,eligible=calls[0]
            np.testing.assert_array_equal(pre,s.local.modified_pre_score(s.LOCAL_SPEC,self.a,4))
            self.assertFalse(np.array_equal(pre,self.a['pre_score']))
            for key in ('baseline','core','state_calm10','state_weak_narrow'):
                np.testing.assert_array_equal(original[key],self.a[key])
            np.testing.assert_array_equal(original['core_weight'],np.broadcast_to(self.a['core_weight'],self.a['core'].shape))
            np.testing.assert_array_equal(effects,self.a['effects']);np.testing.assert_array_equal(eligible,self.a['eligible'])

    def test_budget_replaces_original_support_and_preserves_existing_weights(self):
        for ident in (4,5,6):
            factor=s.feature_rank(s.configs()[ident],self.a,4)
            before=s.adapter.budget_weights('incumbent66',0.,factor,self.a,4)
            after=s.adapter.budget_weights('incumbent66',.025,factor,self.a,4)
            for key in ('core','prior_residual','prior_ml','episode'):
                np.testing.assert_array_equal(before[key],after[key])
            hand=.025*np.isfinite(factor)
            np.testing.assert_array_equal(after['hand'],hand)
            np.testing.assert_allclose(after['original_support'],before['original_support']-hand,rtol=0,atol=1e-16)
            np.testing.assert_allclose(sum(after.values()),1.,rtol=0,atol=1e-15)
            self.assertTrue((after['original_support']>=.355-1e-12).all())

    def test_zero_signedzero_and_missing_exact_identity_with_support_roundoff(self):
        for ident in (4,5,6):
            rank,available=self.keys(ident)
            for value in (0.,-0.,np.nan):
                a=copy.deepcopy(self.a)
                # Explicit zero must bypass clipping of tolerated support roundoff.
                support=np.tile([.5+3e-13,-.5-3e-13,.5+3e-13,-.5-3e-13,.2,0.],(4,1))
                a['baseline']=a['core']+(1-a['core_weight'])*support
                a[rank][:]=value;a[available][:]=np.isfinite(value)
                old=self.choose(3,a=a);got=self.choose(ident,a=a)
                self.assertEqual(got[0],old[0]);self.assertEqual(got[1].tobytes(),old[1].tobytes())

    def test_partial_missing_is_cellwise_and_small_nonzero_is_not_discarded(self):
        self.a['state_calm10'][:]=False;self.a['state_weak_narrow'][:]=False
        self.a['pre_score'][:]=[.04,.03,0.,-.01,-.015,-.03]
        self.a['baseline']=self.a['core'].copy();self.a['effects'][:]=0.
        for ident in (4,5,6):
            a=copy.deepcopy(self.a);rank,available=self.keys(ident)
            a[rank][:]=np.nan;a[available][:]=False
            a[rank][:,2]=1e-14;a[available][:,2]=True
            old=self.choose(3,a=a);got=self.choose(ident,a=a)
            self.assertTrue((got[1][:,2]>old[1][:,2]).all())
            np.testing.assert_array_equal(got[1][:,3:],old[1][:,3:])
            np.testing.assert_allclose(got[1][:,2]-old[1][:,2],.025e-14,rtol=0,atol=0)

    def test_exact_boundary_ties_anchors_and_no_global_post_rerank(self):
        self.a['state_calm10'][:]=False;self.a['state_weak_narrow'][:]=False
        self.a['baseline']=self.a['core'].copy()
        self.a['pre_score'][:]=[.04,.03,0.,-.02,-np.nextafter(.02,np.inf),-.021]
        self.a['effects'][:]=[0.,0.,-.5,.5,.5,.5]
        for _,rank,available in s.FACTOR_SOURCES.values():
            self.a[rank][:]=0.;self.a[available][:]=True
        for ident in (4,5,6):self.assertEqual(self.choose(ident)[0],[[0,1,3]]*4)
        self.a['pre_score'][:,3:5]=-.01;self.m['symbols']=[list('ABCEDF') for _ in range(4)]
        for ident in (4,5,6):self.assertEqual(self.choose(ident)[0],[[0,1,4]]*4)
        self.a['pre_score'][:]=[.20,.19,.18,.179,.178,.1779]
        self.a['effects'][:]=[0.,0.,-.5,-.5,-.5,0.]
        for _,rank,_ in s.FACTOR_SOURCES.values():self.a[rank][:]=[-.5,-.5,-.5,-.5,-.5,.5]
        for ident in (4,5,6):
            chosen,scores=self.choose(ident)
            self.assertEqual(chosen,[[0,1,2]]*4)
            self.assertTrue((scores[:,5]>scores[:,2]).all())

    def test_prefix_no_future_score_factor_metadata_access(self):
        for ident in range(7):
            full=self.choose(ident)
            for n in range(5):
                a,meta=copy.deepcopy(self.a),copy.deepcopy(self.m)
                for value in a.values():value[n:]=False if value.dtype.kind=='b' else np.nan
                meta['symbols'][n:]=[[None]]*(4-n)
                got=self.choose(ident,n,a=a,meta=meta)
                self.assertEqual(got[0],full[0][:n]);np.testing.assert_array_equal(got[1],full[1][:n])
                if ident>=4:
                    rank=s.feature_rank(s.configs()[ident],a,n)
                    self.assertTrue(np.isnan(rank[n:]).all())
                    factor,_=self.keys(ident)
                    a[factor]=a[factor].astype(object);a[factor][n:]='unread future'
                    again=self.choose(ident,n,a=a,meta=meta)
                    np.testing.assert_array_equal(again[1],got[1])

    def test_contract_rejects_mask_rank_mismatch_dtype_shape_inf_and_bounds(self):
        for ident in (4,5,6):
            rank,available=self.keys(ident)
            for value in (self.a[available].astype(float),self.a[available].astype(object),self.a[available][:,:-1]):
                a=copy.deepcopy(self.a);a[available]=value
                with self.assertRaises(ValueError):self.choose(ident,a=a)
            for value in (np.inf,-np.inf,.50000001,-.50000001,np.nan):
                a=copy.deepcopy(self.a);a[rank][0,2]=value;a[available][0,2]=True
                with self.assertRaises(ValueError):self.choose(ident,a=a)
            a=copy.deepcopy(self.a);a[available][0,2]=False
            with self.assertRaises(ValueError):self.choose(ident,a=a)
            a=copy.deepcopy(self.a);a[rank]=a[rank][:,:-1]
            with self.assertRaises(ValueError):self.choose(ident,a=a)
            for n in (True,-1,5,1.5):
                with self.assertRaises(ValueError):self.choose(ident,n)
        with self.assertRaises(ValueError):s.choices(dict(s.configs()[4],mode='new'),self.a,self.m,4,policy=POLICY)

    def test_no_mutation_and_original_twohalf_cash_rules(self):
        a,meta=copy.deepcopy(self.a),copy.deepcopy(self.m)
        for ident in range(7):self.choose(ident)
        for key in a:np.testing.assert_array_equal(self.a[key],a[key])
        self.assertEqual(self.m,meta)
        for count in (0,1,2):
            self.a['eligible'][:]=False;self.a['eligible'][:,:count]=True
            old=self.choose(3)
            for ident in (4,5,6):
                got=self.choose(ident);self.assertEqual(got[0],old[0]);self.assertEqual(got[1].tobytes(),old[1].tobytes())
                self.assertEqual([len(row) for row in got[0]],[2 if count==2 else 0]*4)

    def test_metadata_and_policy_changes_fail_before_selection(self):
        for symbols in ([list('ABCDE')]*4,[list('AABCDE')]*4,[[None,'B','C','D','E','F']]*4):
            meta=copy.deepcopy(self.m);meta['symbols']=symbols
            with self.assertRaises(ValueError):self.choose(meta=meta)
        for key,value in (('budget',.10),('ceiling',6),('gap',.03)):
            with self.assertRaises(AssertionError):
                s.choices(s.configs()[4],self.a,self.m,4,policy=dict(POLICY,**{key:value}))

    def test_first88_gates_default_ties_and_all_three_diagnostics_frozen(self):
        def metric(cagr):return dict(cagr=cagr,sharpe=1.7,max_drawdown=-.3)
        rows=[dict(spec=spec,training=metric(.5),early=metric(.4),recent=metric(.6),
                   later=dict(cagr=999.)) for spec in s.configs()]
        self.assertEqual(s.select(copy.deepcopy(rows))['primary']['spec']['id'],3)
        rows[4]['training']['cagr']=.50+5e-13
        self.assertEqual(s.select(copy.deepcopy(rows))['primary']['spec']['id'],3)
        rows[4]['training']['cagr']=.51;rows[5]['training']['cagr']=.51+5e-13
        self.assertEqual(s.select(copy.deepcopy(rows))['primary']['spec']['id'],4)
        rows[5]['training']['cagr']=.5
        for section,key,value in (('training','cagr',.49),('training','sharpe',1.64),
                ('training','max_drawdown',-.331),('early','cagr',.374),('recent','cagr',.574)):
            bad=copy.deepcopy(rows);bad[4][section][key]=value
            self.assertFalse(s.admitted(bad[4],bad[3]))
            selected=s.select(bad)
            self.assertEqual(selected['primary']['spec']['id'],3)
            self.assertEqual(len(selected),8)
            for arm in bad[4:]:self.assertEqual(selected[arm['spec']['mode']],arm)
        at_boundary=copy.deepcopy(rows[4])
        at_boundary.update(training=dict(cagr=.5-1e-12,sharpe=1.7-.05,max_drawdown=-.3-.03),
            early=metric(.4-.025),recent=metric(.6-.025))
        self.assertTrue(s.admitted(at_boundary,rows[3]))
        rows[0]['training']['cagr']=999.;rows[4]['later']=dict(cagr=-1e20)
        self.assertEqual(s.select(copy.deepcopy(rows))['primary']['spec']['id'],4)
        for row in rows[4:]:row['training']['cagr']=.49;row['later']=dict(cagr=1e20)
        self.assertEqual(s.select(copy.deepcopy(rows))['primary']['spec']['id'],3)

    def test_training_requires_independent_pass_bound_to_all_six_prepared_files(self):
        bound={name:'digest-'+name for name in s.ACTUAL_BOUND_FILES}
        good=dict(passed=True,hashes=bound)
        with patch.object(s,'read',return_value=good),patch.object(s,'sha',side_effect=lambda path:'digest-'+path.name):
            self.assertIs(s.verify_actual_review(),good)
        cases=[{},dict(passed=False,hashes=bound),dict(passed=1,hashes=bound),dict(passed=True,hashes={})]
        for name in s.ACTUAL_BOUND_FILES:
            wrong=dict(bound);wrong[name]='stale';cases.append(dict(passed=True,hashes=wrong))
        for bad in cases:
            with patch.object(s,'read',return_value=bad),patch.object(s,'sha',side_effect=lambda path:'digest-'+path.name):
                with self.assertRaises(AssertionError):s.verify_actual_review()
        tree=ast.parse(inspect.getsource(s.train))
        names=[node.value.func.id for node in tree.body[0].body
               if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)]
        self.assertEqual(names[:2],['registered','verify_actual_review'])
        self.assertIn("'ACTUAL_FEATURE_REVIEW.json'",inspect.getsource(s.train))
        self.assertIn('((36,60),(60,76),(76,88))',inspect.getsource(s.train))

    def test_all27_inherited_dtype_shape_and_nan_payload_exact(self):
        source={str(i):np.array([[float(i),np.nan]]) for i in range(27)}
        s.verify_inherited_arrays(copy.deepcopy(source),source)
        for changed in ('dtype','shape','value','nan_payload'):
            current=copy.deepcopy(source)
            if changed=='dtype':current['0']=current['0'].astype(np.float32)
            elif changed=='shape':current['0']=current['0'].reshape(2,1)
            elif changed=='value':current['0'][0,0]=1.
            else:current['0'].view(np.uint64)[0,1]=0x7ff8000000000001
            with self.assertRaises(AssertionError):s.verify_inherited_arrays(current,source)
        with self.assertRaises(AssertionError):s.verify_inherited_arrays(source,{k:v for k,v in source.items() if k!='0'})

    def test_cache_extraction_preserves_all_ranks_bytes_and_independent_masks(self):
        class Poison:
            def __array__(self,*args,**kw):raise AssertionError('source raw factor was read')
        source={str(i):np.array([[float(i),np.nan]]) for i in range(26)}
        source['baseline']=np.zeros((4,6));meta={'dates':['x'],'symbols':[['A']]}
        caches={};metas={label:copy.deepcopy(meta) for label in s.FACTOR_SOURCES}
        for label,(_,rank,available) in s.FACTOR_SOURCES.items():
            caches[label]=dict(copy.deepcopy(source),**{rank:self.a[rank].copy(),
                available:self.a[available].copy(),'raw_unused':Poison()})
        caches['cotail']['rank_cotail'][0,2]=np.nan;caches['cotail']['cotail_available'][0,2]=False
        # Include a non-default NaN payload: copied cached bytes must be retained.
        caches['cotail']['rank_cotail'].view(np.uint64)[0,2]=0x7ff8000000000001
        got=s.matched_features(caches,metas,source,meta)
        self.assertEqual(set(got),set(s.FACTOR_KEYS));self.assertEqual(len(dict(source,**got)),33)
        for label,(_,rank,available) in s.FACTOR_SOURCES.items():
            for key in (rank,available):
                self.assertEqual(got[key].dtype,caches[label][key].dtype)
                self.assertEqual(got[key].tobytes(),caches[label][key].tobytes())
                self.assertFalse(np.shares_memory(got[key],caches[label][key]))
        self.assertTrue(got['ordinal_available'][0,2]);self.assertFalse(got['cotail_available'][0,2])
        for label,(_,rank,available) in s.FACTOR_SOURCES.items():
            old=caches[label][available];caches[label][available]=old.astype(float)
            with self.assertRaises(ValueError):s.matched_features(caches,metas,source,meta)
            caches[label][available]=old;metas[label]={'wrong':'meta'}
            with self.assertRaises(AssertionError):s.matched_features(caches,metas,source,meta)
            metas[label]=copy.deepcopy(meta)
            caches[label]['baseline'][0,0]=1.
            with self.assertRaises(AssertionError):s.matched_features(caches,metas,source,meta)
            caches[label]['baseline'][0,0]=0.
            old=caches[label][rank];caches[label][rank]=old.astype(object)
            with self.assertRaises(ValueError):s.matched_features(caches,metas,source,meta)
            caches[label][rank]=old

    def test_source_audit_requires_passed_six_bindings_and_identical64raw(self):
        folder=Path('/synthetic/source');raw={str(i):'raw-'+str(i) for i in range(64)}
        plan=dict(design_sha256='digest-DESIGN.json',hashes={},source_hashes={},raw_source_hashes=raw)
        audit=dict(passed=True,hashes={name:'digest-'+name for name in s.ACTUAL_BOUND_FILES})
        def reader(path):
            if path.name=='PLAN.json':return plan
            if path.name=='ACTUAL_FEATURE_REVIEW.json':return audit
            if path.name in ('INPUT_HASHES.json','CACHE_HASHES.json'):return {}
            raise AssertionError('Unexpected read '+str(path))
        with patch.object(s,'read',side_effect=reader),patch.object(s,'sha',side_effect=lambda path:'digest-'+path.name),patch.object(Path,'read_text',return_value='digest-DESIGN.json'),patch.object(s.old,'assert_hashes'):
            binding=s.verify_factor_source(folder,raw)
            self.assertEqual(binding['hashes'],audit['hashes']);self.assertEqual(binding['raw_source_hashes'],raw)
            self.assertEqual(binding['directory'],str(folder))
            self.assertEqual(binding['actual_feature_review_sha256'],'digest-ACTUAL_FEATURE_REVIEW.json')
            for bad in (False,1,None):
                audit['passed']=bad
                with self.assertRaises(AssertionError):s.verify_factor_source(folder,raw)
            audit['passed']=True
            for name in s.ACTUAL_BOUND_FILES:
                audit['hashes'][name]='stale'
                with self.assertRaises(AssertionError):s.verify_factor_source(folder,raw)
                audit['hashes'][name]='digest-'+name
            with self.assertRaises(AssertionError):s.verify_factor_source(folder,dict(raw,extra='new'))
            wrong=dict(raw);wrong['0']='different'
            with self.assertRaises(AssertionError):s.verify_factor_source(folder,wrong)


if __name__=='__main__':
    unittest.main()

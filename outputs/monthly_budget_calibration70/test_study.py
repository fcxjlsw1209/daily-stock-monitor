"""Synthetic global guidance-score and audited-stage integration; no real data."""
import ast
import copy
import importlib.util
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('budget_calibration70_study_tests',P/'study.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
parent=s.module('budget70_parent_fixture',s.local.P/'test_study.py')
POLICY=dict(s.POLICY_FIELDS)


def fixture():
    a,m=parent.fixture()
    # Deliberately favor names outside the old top2/boundary candidate group.
    x=np.array([[-.45,-.2,-.1,0.],[-.4,-.1,0.,.1],[-.2,0.,.1,.2],
                [.1,.1,.2,.3],[.3,.2,.3,.4],[.5,.3,.4,.5]])
    a['budget_event_X']=np.tile(x,(4,1,1))
    a['budget_event_weights']=np.tile([.60,.20,.10,.10],(4,1))
    a['budget_event_eligible']=a['eligible'].copy()
    return a,m


def training_fixture(m):
    tm=dict(dates=['2014-01-31','2014-02-28'],ends=['2014-02-28','2014-03-31'],
            symbols=[list('ABC'),list('DEFG')])
    arrays=dict(budget_train_X=np.zeros((7,4)),budget_event_X=np.zeros((4,6,4)),
        budget_train_dates=np.array(['2014-01-31']*3+['2014-02-28']*4,dtype='datetime64[D]'),
        budget_train_ends=np.array(['2014-02-28']*3+['2014-03-31']*4,dtype='datetime64[D]'),
        budget_train_symbols=np.array(list('ABCDEFG')),budget_train_offsets=np.array([0,3,7]),
        budget_train_eligible=np.array([True,True,False,True,True,True,True]),
        budget_event_eligible=np.ones((4,6),bool),budget_train_target=np.array([-.25,.25,np.nan,np.nan,np.nan,np.nan,np.nan]),
        budget_train_label_valid=np.array([True,True,False,False,False,False,False]),
        budget_event_dates=np.array(m['dates'],dtype='datetime64[D]'))
    return arrays,tm


class StudyTests(unittest.TestCase):
    def setUp(self):self.a,self.m=fixture()

    def choose(self,ident=4,n=4,a=None,m=None,**kw):
        return s.choices(s.configs()[ident],self.a if a is None else a,self.m if m is None else m,n,policy=POLICY,**kw)

    def test_fixed_design_six_configs_dependency_and_new_stage_contract(self):
        fixed,digest=s.design()
        self.assertEqual(digest,'995fb42776fb0b0dece659997b2b4f0ecd642066093f3e95a95b405cb51bcf35')
        self.assertEqual(fixed['configs'],s.configs());self.assertEqual(len(s.configs()),6)
        self.assertEqual([s.diagnostic_name(z) for z in s.configs()[4:]],['guidance4_fixed','annualcalibrated4f'])
        self.assertEqual(s.TRAIN,88);self.assertEqual(s.PROTECTED_COUNT,21444)
        np.testing.assert_array_equal(s.PRIOR,[.65,.15,.10,.10])
        files=set(s.source_files());implementations=set(s.implementations())
        for f in (s.TRAIN_META,s.TRAIN_CACHE,s.SECTOR_SOURCE,s.MAPPING_SOURCE):self.assertIn(f,files)
        for f in (Path(s.arch.__file__),Path(s.ordinal.__file__),s.ordinal._FILE,s.MAPPING_SOURCE,*s.PRICE_SOURCE_FILES):
            self.assertIn(f,implementations)
        for name in ('features.py','test_features.py','model.py','test_model.py','study.py','test_study.py','report.py'):
            self.assertIn(s.P/name,implementations)
        self.assertIn('FIT_HASHES.json',inspect.getsource(s.train))
        self.assertIn('MODEL_REVIEW.json',inspect.getsource(s.train))
        self.assertIn('TESTS.txt',inspect.getsource(s.train))
        self.assertEqual(s.LOCAL_SPEC,s.local.configs()[4])

    def test_four_control_bypasses_all_new_inputs_and_exact_orders(self):
        for key in list(self.a):
            if key.startswith('budget_'):del self.a[key]
        with patch.object(s,'score',side_effect=AssertionError('control must bypass new score')):
            for ident in range(4):
                want=s.local.choices(s.source_spec(s.configs()[ident]),self.a,self.m,4,policy=POLICY)
                got=self.choose(ident)
                self.assertEqual(got[0],want[0]);self.assertEqual(got[1].tobytes(),want[1].tobytes())
                self.assertEqual(s.rp.make_orders(*got,self.m),s.rp.make_orders(*want,self.m))

    def test_exact_four_factor_coefficients_and_no_postscore_rank(self):
        for ident,weights in ((4,s.PRIOR),(5,self.a['budget_event_weights'][0])):
            got=self.choose(ident)
            want=np.array([[sum(float(v)*float(w) for v,w in zip(row,weights)) for row in month]
                           for month in self.a['budget_event_X']])
            np.testing.assert_allclose(got[1],want,rtol=0,atol=1e-16)
            self.assertEqual(got[0],[[5,4,3]]*4)
            self.assertFalse(np.array_equal(got[1],np.tile(np.arange(1,7)/6-.5,(4,1))))

    def test_global_all_three_seats_unrelated_to_original_episode_or_state(self):
        for ident in (4,5):
            original=self.choose(ident)
            a=copy.deepcopy(self.a)
            for key in a:
                if key not in ('baseline','eligible','budget_event_X','budget_event_weights','budget_event_eligible'):
                    a[key]='must not inspect'
            a['baseline']=np.full((4,6),np.nan) # only inherited shape is consulted
            with patch.object(s.local,'choices',side_effect=AssertionError('new candidates bypass old selection')):
                got=self.choose(ident,a=a,boundary_selector=lambda *args:(_ for _ in ()).throw(AssertionError('episode called')))
            self.assertEqual(got[0],original[0]);np.testing.assert_array_equal(got[1],original[1])
            self.assertNotIn(0,got[0][0]);self.assertNotIn(1,got[0][0])

    def test_fixed_does_not_read_weights_and_coldstart_exact_fixed_parity(self):
        a=copy.deepcopy(self.a);del a['budget_event_weights']
        fixed=self.choose(4,a=a)
        a['budget_event_weights']=np.tile(s.PRIOR,(4,1))
        annual=self.choose(5,a=a)
        self.assertEqual(fixed[0],annual[0]);self.assertEqual(fixed[1].tobytes(),annual[1].tobytes())
        np.testing.assert_array_equal(s.weight_rows(a['budget_event_weights'],self.m['dates'],4),a['budget_event_weights'])

    def test_bounded_weight_transfer_and_score_difference(self):
        fixed=self.choose(4)[1];annual=self.choose(5)[1]
        self.assertLessEqual(np.max(np.abs(annual-fixed)),.05+1e-15)
        x=np.zeros((4,6,4));x[:]=[-.5,.5,.5,.5]
        a=copy.deepcopy(self.a);a['budget_event_X']=x
        np.testing.assert_allclose(self.choose(5,a=a)[1]-self.choose(4,a=a)[1],.05,rtol=0,atol=1e-16)

    def test_zero_features_mean_zero_newscore_not_local_fallback(self):
        self.a['budget_event_X'][:]=0.
        self.m['symbols']=[list('FEDCBA') for _ in range(4)]
        for ident in (4,5):
            got=self.choose(ident)
            np.testing.assert_array_equal(got[1],0.)
            self.assertEqual(got[0],[[5,4,3]]*4)
        # Missing raw factors are neutralized by the frozen snapshot before cache;
        # a missing prepared cell is an error, never a trading-time fallback.
        self.a['budget_event_X'][0,1,2]=np.nan
        for ident in (4,5):
            with self.assertRaises(ValueError):self.choose(ident)

    def test_original_eligibility_only_and_two_half_or_cash(self):
        self.a['budget_event_eligible']='unused future-label surrogate'
        for count in (0,1,2,3,6):
            self.a['eligible'][:]=False;self.a['eligible'][:,:count]=True
            for ident in (4,5):
                got=self.choose(ident)
                self.assertEqual(got[0],[list(range(count-1,max(-1,count-4),-1)) if count>=2 else []]*4)
        self.a['eligible']=self.a['eligible'].astype(float)
        with self.assertRaises(ValueError):self.choose()

    def test_prefix_future_poison_for_features_weights_metadata_and_controls(self):
        for ident in range(6):
            full=self.choose(ident)
            for n in range(5):
                a,m=copy.deepcopy(self.a),copy.deepcopy(self.m)
                for key,value in a.items():
                    value[n:]=False if value.dtype.kind=='b' else np.nan
                m['symbols'][n:]=[[None]]*(4-n);m['dates'][n:]=['not a date']*(4-n)
                if ident>=4:
                    a['budget_event_X']=a['budget_event_X'].astype(object)
                    a['budget_event_X'][n:]='unread future'
                    a['budget_event_weights']=a['budget_event_weights'].astype(object)
                    a['budget_event_weights'][n:]='unread future'
                got=self.choose(ident,n,a=a,m=m)
                self.assertEqual(got[0],full[0][:n]);np.testing.assert_array_equal(got[1],full[1][:n])

    def test_invalid_weights_fail_without_clipping_normalization_or_fallback(self):
        for weights in ([.59,.21,.10,.10],[.65,.25,.10,0.],[.65,.15,.1,.09],
                        [.65,.15,.21,-.01],[np.nan,.15,.10,.10],[np.inf,.15,.10,.10]):
            a=copy.deepcopy(self.a);a['budget_event_weights'][:]=weights
            with self.assertRaises(ValueError):self.choose(5,a=a)
        a=copy.deepcopy(self.a);a['budget_event_weights'][1]=s.PRIOR
        with self.assertRaises(ValueError):self.choose(5,a=a) # same-year drift
        a=copy.deepcopy(self.a);a['budget_event_weights']=a['budget_event_weights'][:,:3]
        with self.assertRaises(ValueError):self.choose(5,a=a)

    def test_invalid_feature_shapes_bounds_config_identity_and_prefix(self):
        for value in (np.nan,np.inf,.50001,-.50001):
            a=copy.deepcopy(self.a);a['budget_event_X'][0,0,0]=value
            with self.assertRaises(ValueError):self.choose(a=a)
        a=copy.deepcopy(self.a);a['budget_event_X']=a['budget_event_X'][:,:,:3]
        with self.assertRaises(ValueError):self.choose(a=a)
        for n in (-1,5,True,1.2):
            with self.assertRaises(ValueError):self.choose(n=n)
        for ss in (list('AABCDE'),list('ABCDE'),[None,'B','C','D','E','F']):
            m=copy.deepcopy(self.m);m['symbols'][0]=ss
            with self.assertRaises(ValueError):self.choose(m=m)
        with self.assertRaises(ValueError):s.choices(dict(s.configs()[4],mode='new'),self.a,self.m,4)

    def test_no_mutation(self):
        before=copy.deepcopy(self.a);m=copy.deepcopy(self.m)
        for ident in range(6):self.choose(ident)
        for key in before:np.testing.assert_array_equal(before[key],self.a[key])
        self.assertEqual(m,self.m)

    def test_eleven_feature_identity_schema_and_no_future_label_filter(self):
        a,tm=training_fixture(self.m)
        result=s.validate_features(a,self.m,tm,self.a['eligible'])
        self.assertEqual(result['training_rows'],7);self.assertEqual(result['train_label_valid_rows'],2)
        self.assertTrue(a['budget_train_eligible'][3:].all())
        self.assertTrue((~a['budget_train_label_valid'][3:]).all())
        for key in s.FEATURE_KEYS:
            bad=copy.deepcopy(a);del bad[key]
            with self.assertRaises(ValueError):s.validate_features(bad,self.m,tm)
        for key,value in (('budget_train_eligible',np.ones(7,float)),('budget_train_X',np.zeros((7,3))),
                          ('budget_event_X',np.zeros((4,6,3))),('budget_train_symbols',np.arange(7))):
            bad=copy.deepcopy(a);bad[key]=value
            with self.assertRaises(ValueError):s.validate_features(bad,self.m,tm)
        for key in ('budget_train_dates','budget_train_ends','budget_event_dates'):
            bad=copy.deepcopy(a);bad[key][0]=np.datetime64('NaT')
            with self.assertRaises(ValueError):s.validate_features(bad,self.m,tm)
        for key in ('budget_train_offsets','budget_train_symbols','budget_train_dates','budget_train_ends'):
            bad=copy.deepcopy(a)
            if key=='budget_train_symbols':bad[key][0]='X'
            elif key in ('budget_train_dates','budget_train_ends'):bad[key][0]+=np.timedelta64(1,'D')
            else:bad[key][1]+=1
            with self.assertRaises(AssertionError):s.validate_features(bad,self.m,tm)
        bad=copy.deepcopy(a);bad['budget_event_eligible'][0,0]=False
        with self.assertRaises(AssertionError):s.validate_features(bad,self.m,tm,self.a['eligible'])
        self.assertEqual(len(dict({str(i):np.zeros(1) for i in range(27)},**a)),38)

    def test_label_partial_month_nonfinite_ineligible_and_not_centered_rejected(self):
        original,tm=training_fixture(self.m)
        cases=[]
        a=copy.deepcopy(original);a['budget_train_target'][0]=np.inf;cases.append(a)
        a=copy.deepcopy(original);a['budget_train_label_valid'][0]=False;cases.append(a)
        a=copy.deepcopy(original);a['budget_train_target'][0]=np.nan;a['budget_train_label_valid'][0]=False;cases.append(a)
        a=copy.deepcopy(original);a['budget_train_target'][2]=0.;a['budget_train_label_valid'][2]=True;cases.append(a)
        a=copy.deepcopy(original);a['budget_train_target'][:2]=[.3,.3];cases.append(a)
        a=copy.deepcopy(original);a['budget_train_target'][:2]=[-.6,.6];cases.append(a)
        a=copy.deepcopy(original);a['budget_train_X'][6,1]=np.nan;cases.append(a) # no silently removed row
        for a in cases:
            with self.assertRaises(ValueError):s.validate_features(a,self.m,tm)
        a=copy.deepcopy(original);a['budget_train_target'][:2]=0.
        s.validate_features(a,self.m,tm) # all-tied targets valid0

    def test_inherited27_arrays_byte_exact_including_nan_payload(self):
        source={str(i):np.array([[float(i),np.nan]]) for i in range(27)}
        s.verify_inherited_arrays(copy.deepcopy(source),source)
        for change in ('dtype','shape','value','nan_payload'):
            current=copy.deepcopy(source)
            if change=='dtype':current['0']=current['0'].astype(np.float32)
            elif change=='shape':current['0']=current['0'].reshape(2,1)
            elif change=='value':current['0'][0,0]=1.
            else:current['0'].view(np.uint64)[0,1]=0x7ff8000000000001
            with self.assertRaises(AssertionError):s.verify_inherited_arrays(current,source)

    def test_six_feature_and_seven_model_review_bindings_are_mandatory(self):
        for fn,names in ((s.verify_actual_review,s.ACTUAL_BOUND_FILES),(s.verify_model_review,s.MODEL_BOUND_FILES)):
            good=dict(passed=True,hashes={n:'digest-'+n for n in names})
            with patch.object(s,'read',return_value=good),patch.object(s,'sha',side_effect=lambda p:'digest-'+p.name):
                self.assertIs(fn(),good)
            cases=[{},dict(passed=1,hashes=good['hashes']),dict(passed=False,hashes=good['hashes']),dict(passed=True,hashes={})]
            for name in names:
                bad=copy.deepcopy(good);bad['hashes'][name]='stale';cases.append(bad)
            for bad in cases:
                with patch.object(s,'read',return_value=bad),patch.object(s,'sha',side_effect=lambda p:'digest-'+p.name):
                    with self.assertRaises(AssertionError):fn()
        self.assertEqual(len(s.ACTUAL_BOUND_FILES),6);self.assertEqual(len(s.MODEL_BOUND_FILES),7)

    def test_prepare_fit_train_stage_order_and_immutable_feature_bindings(self):
        def calls(fn):
            tree=ast.parse(inspect.getsource(fn))
            return [n.value.func.id for n in tree.body[0].body if isinstance(n,ast.Expr)
                    and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)]
        self.assertEqual(calls(s.fit)[:2],['registered','verify_actual_review'])
        self.assertEqual(calls(s.train)[:3],['registered','verify_actual_review','verify_model_review'])
        source=inspect.getsource(s.prepare)
        self.assertNotIn('annual_weights(',source)
        self.assertIn('for spec in configs()[:4]',source)
        self.assertNotIn('configs()[4:]',source)
        self.assertNotIn("z['good']",source);self.assertNotIn("z['Y']",source)
        self.assertIn("(P/'META.json').write_bytes((local.P/'META.json').read_bytes())",source)
        source=inspect.getsource(s.fit)
        self.assertNotIn("dump('CACHE_HASHES.json'",source)
        self.assertNotIn("dump('INPUT_HASHES.json'",source)
        self.assertNotIn('choices(',source)
        self.assertIn("dump('FIT_HASHES.json'",source)
        self.assertIn('((36,60),(60,76),(76,88))',inspect.getsource(s.train))

    def test_fit_pure_wiring_separate_weights_cache_and_no_new_choices(self):
        arrays,tm=training_fixture(self.m);a=dict(self.a,**arrays)
        weights=np.tile(s.PRIOR,(4,1));out=[]
        with tempfile.TemporaryDirectory() as temp,patch.object(s,'P',Path(temp)),\
                patch.object(s,'registered'),patch.object(s,'verify_actual_review'),\
                patch.object(s,'load',return_value=(self.m,a,None,None)),\
                patch.object(s.model,'annual_weights',return_value=(weights,dict(annual=[]))) as fit,\
                patch.object(s.np,'savez_compressed') as save,patch.object(s,'dump',side_effect=lambda n,v:out.append((n,v))),\
                patch.object(s,'sha',side_effect=lambda p:'digest-'+p.name),\
                patch.object(s,'choices',side_effect=AssertionError('fit cannot choose new portfolio')):
            s.fit()
        self.assertEqual(fit.call_count,1)
        for got,key in zip(fit.call_args.args,('train_X','train_target','train_dates','train_ends','train_eligible','train_label_valid','event_dates')):
            self.assertIs(got,a['budget_'+key])
        self.assertEqual(set(save.call_args.kwargs),{'event_weights'})
        self.assertEqual([n for n,v in out],['MODEL_AUDIT.json','FIT_HASHES.json'])
        self.assertFalse(out[0][1]['driver_validation']['new_candidate_choices'])
        self.assertEqual(set(out[0][1]['driver_validation']['feature_input_bindings']),set(s.ACTUAL_BOUND_FILES))

    def test_first88_gates_default3_ties_both_diagnostics_and_no_later_selection(self):
        def metric(cagr):return dict(cagr=cagr,sharpe=1.7,max_drawdown=-.3)
        rows=[dict(spec=spec,training=metric(.5),early=metric(.4),recent=metric(.6),later=dict(cagr=999.)) for spec in s.configs()]
        self.assertEqual(s.select(copy.deepcopy(rows))['primary']['spec']['id'],3)
        rows[4]['training']['cagr']=.5+5e-13
        self.assertEqual(s.select(copy.deepcopy(rows))['primary']['spec']['id'],3)
        rows[4]['training']['cagr']=.51;rows[5]['training']['cagr']=.51+5e-13
        self.assertEqual(s.select(copy.deepcopy(rows))['primary']['spec']['id'],4)
        rows[5]['training']['cagr']=.5
        for section,key,value in (('training','cagr',.49),('training','sharpe',1.64),('training','max_drawdown',-.331),('early','cagr',.374),('recent','cagr',.574)):
            bad=copy.deepcopy(rows);bad[4][section][key]=value
            selected=s.select(bad);self.assertEqual(selected['primary']['spec']['id'],3)
            self.assertEqual(len(selected),7)
            for z in bad[4:]:self.assertEqual(selected[s.diagnostic_name(z['spec'])],z)
        rows[0]['training']['cagr']=99.;rows[4]['later']['cagr']=-1e30
        self.assertEqual(s.select(copy.deepcopy(rows))['primary']['spec']['id'],4)


if __name__=='__main__':unittest.main()

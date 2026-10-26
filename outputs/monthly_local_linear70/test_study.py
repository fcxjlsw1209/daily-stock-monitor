"""Synthetic effect-slot, selector, source identity and frozen protocol tests."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('local_linear70_study_tests',P/'study.py')
s=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(s)
POLICY=dict(s.POLICY_FIELDS)


def fixture():
    shape=(4,6);cw=np.array([.52,.5,.52,.5])[:,None]
    support=np.tile([.1,-.1,.05,0.,-.05,.1],(4,1))
    core=cw*np.tile([.45,.4,.3,.29,.28,.27],(4,1))
    a=dict(baseline=core+(1-cw)*support,core=core,core_weight=cw,
        eligible=np.ones(shape,bool),state_calm10=np.array([False,True,False,True]),
        state_weak_narrow=np.array([False,False,True,True]),
        pre_score=np.tile([.30,.25,.20,.19,.181,.1805],(4,1)),
        effects=np.tile([0.,0.,-.10,.05,.15,.50],(4,1)),
        forecast_available=np.ones(shape,bool),extension_available=np.ones(shape,bool),
        old_ml4=np.zeros(shape),rank_log_excess=np.tile([-.4,-.3,-.2,.1,.3,.5],(4,1)),
        model_available=np.ones(shape,bool),local_available=np.ones(shape,bool),
        raw_source_log=np.tile([-.06,-.05,-.04,.02,.04,.06],(4,1)),
        raw_local_constant=np.tile([5.,3.,1.,-2.,-3.,-4.],(4,1)),
        raw_local_linear=np.tile([-4.,-3.,-2.,1.,3.,5.],(4,1)))
    return a,dict(symbols=[list('ABCDEF') for _ in range(4)],
        dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31'],
        ends=['2016-08-31','2016-09-30','2016-10-31','2016-11-30'])


class StudyTests(unittest.TestCase):
    def setUp(self):self.a,self.m=fixture()


    def choose(self,ident,n=4,**kw):return s.choices(s.configs()[ident],self.a,self.m,n,policy=POLICY,**kw)


    def rows(self):
        def metric(v):return dict(cagr=v,sharpe=1.6,max_drawdown=-.3)
        return [dict(spec=z,training=metric(.5 if z['id']==2 else .49),early=metric(.4),recent=metric(.6)) for z in s.configs()]


    def ranks(self,n=4):
        return s.ranked_forecasts({key:self.a['raw_'+key] for key in s.ARMS},self.a['model_available'],self.a,n)

    def test_fixed_five_configurations_and_internal_source_not_candidate(self):
        self.assertEqual(s.design()['configs'],s.configs());self.assertEqual(len(s.configs()),5)
        self.assertEqual([x['mode'] for x in s.configs()[3:]],['local_constant','local_linear'])
        self.assertEqual(s.SOURCE_SCOPE_SPEC,s.scope.configs()[3])
        self.assertEqual([s.diagnostic_name(x) for x in s.configs()[3:]],list(s.ARMS))

    def test_three_controls_bypass_new_forecasts_and_exact_orders(self):
        for key in ('raw_local_constant','raw_local_linear','model_available'):del self.a[key]
        with patch.object(s,'ranked_forecasts',side_effect=AssertionError('control bypass')):
            for ident,source in ((0,0),(1,1),(2,3)):
                want=s.scope.choices(s.scope.configs()[source],self.a,self.m,4,policy=POLICY)
                got=self.choose(ident);self.assertEqual(got[0],want[0]);np.testing.assert_array_equal(got[1],want[1])
                self.assertEqual(s.rp.make_orders(*got,self.m),s.rp.make_orders(*want,self.m))

    def test_exact_two_original_slot_coefficients_and_no_new_budget(self):
        ranks,ready=self.ranks();self.assertTrue(ready.all())
        original=s.scope.modified_pre_score(s.SOURCE_SCOPE_SPEC,self.a,4)
        for ident in (3,4):
            key=s.configs()[ident]['mode'];coef=.25*np.array([0.,.01,.05,.05])[:,None]
            want=coef*(ranks[key]-self.a['rank_log_excess'])
            np.testing.assert_array_equal(s.score_delta(s.configs()[ident],self.a,4),want)
            expected=original.copy();expected[want!=0.]+=want[want!=0.]
            np.testing.assert_array_equal(s.modified_pre_score(s.configs()[ident],self.a,4),expected)
            self.assertLessEqual(np.abs(want).max(),.0125)
        for spec in s.configs()[:3]:np.testing.assert_array_equal(s.score_delta(spec,self.a,4),0.)

    def test_slot_masks_source_availability_eligibility_and_extension(self):
        self.a['forecast_available'][2,1]=False;self.a['eligible'][3,2]=False
        self.a['extension_available'][1,3]=False
        ranks,ready=self.ranks();self.assertTrue(ready.all())
        for ident in (3,4):
            delta=s.score_delta(s.configs()[ident],self.a,4)
            for row,col in ((2,1),(3,2),(1,3)):self.assertEqual(delta[row,col],0.)
            self.assertTrue(np.any(delta[2]!=0.))

    def test_missing_one_U_forecast_falls_back_both_entire_months(self):
        for key in s.ARMS:
            a=copy.deepcopy(self.a);a['raw_'+key][1,4]=np.nan
            ranks,ready=s.ranked_forecasts({k:a['raw_'+k] for k in s.ARMS},a['model_available'],a,4)
            self.assertEqual(ready.tolist(),[True,False,True,True])
            for k in s.ARMS:np.testing.assert_array_equal(ranks[k][1],a['rank_log_excess'][1])
            for spec in s.configs()[3:]:np.testing.assert_array_equal(s.score_delta(spec,a,4)[1],0.)
        self.a['model_available'][2,1]=False
        for ident in (3,4):np.testing.assert_array_equal(s.score_delta(s.configs()[ident],self.a,4)[2],0.)

    def test_no_slot_state_or_extension_filter_on_ranking_population(self):
        self.a['raw_local_constant'][0,0]=np.nan  # no slot this month, still a required query.
        self.a['extension_available'][1,0]=False
        self.a['raw_local_linear'][1,0]=np.nan  # no extension for this stock, still in U.
        _,ready=self.ranks();self.assertEqual(ready.tolist(),[False,False,True,True])
        for ident in (3,4):np.testing.assert_array_equal(s.score_delta(s.configs()[ident],self.a,4)[:2],0.)

    def test_missing_outside_U_does_not_shrink_U_or_disable_month(self):
        self.a['eligible'][:,0]=False;self.a['forecast_available'][:,1]=False
        self.a['model_available'][:,:2]=False
        for key in s.ARMS:self.a['raw_'+key][:,:2]=np.nan
        ranks,ready=self.ranks();self.assertTrue(ready.all())
        for key in s.ARMS:
            np.testing.assert_array_equal(ranks[key][:,:2],self.a['rank_log_excess'][:,:2])
            for i in range(4):
                want=s.ep.old.prior.ranks(pd.Series(self.a['raw_'+key][i,2:]))
                np.testing.assert_array_equal(ranks[key][i,2:],want)

    def test_whole_missing_exact_scope_not_old_model_or_cash(self):
        expected=self.choose(2);self.assertFalse(np.array_equal(expected[1],self.choose(1)[1]))
        self.a['model_available'][:]=False
        for key in s.ARMS:self.a['raw_'+key][:]=np.nan
        for ident in (3,4):
            got=self.choose(ident);self.assertEqual(got[0],expected[0]);self.assertEqual(got[1].tobytes(),expected[1].tobytes())
            self.assertEqual(s.rp.make_orders(*got,self.m),s.rp.make_orders(*expected,self.m))

    def test_tie_rank_constant_raw_not_zero_and_empty_U_fallback(self):
        for key in s.ARMS:self.a['raw_'+key][:]=1.
        ranks,ready=self.ranks()
        want=s.ep.old.prior.ranks(pd.Series(np.ones(6)))
        for key in s.ARMS:np.testing.assert_array_equal(ranks[key],np.tile(want,(4,1)))
        self.a['forecast_available'][2]=False
        ranks,ready=self.ranks();self.assertFalse(ready[2])
        for key in s.ARMS:np.testing.assert_array_equal(ranks[key][2],self.a['rank_log_excess'][2])

    def test_equal_forecasts_equal_ranks_scores_orders(self):
        self.a['raw_local_linear']=self.a['raw_local_constant'].copy()
        ranks,_=self.ranks();np.testing.assert_array_equal(ranks['local_constant'],ranks['local_linear'])
        left,right=self.choose(3),self.choose(4)
        self.assertEqual(left[0],right[0]);np.testing.assert_array_equal(left[1],right[1])
        self.assertEqual(s.rp.make_orders(*left,self.m),s.rp.make_orders(*right,self.m))

    def test_new_rank_equal_original_exact_scope_without_own_history_gate(self):
        ranks,_=self.ranks();self.a['rank_log_excess']=ranks['local_constant'].copy()
        self.a['raw_local_linear']=self.a['raw_local_constant'].copy()
        expected=self.choose(2)
        for ident in (3,4):
            got=self.choose(ident);self.assertEqual(got[0],expected[0]);np.testing.assert_array_equal(got[1],expected[1])
        self.a['local_available'][:]=False
        for ident in (3,4):np.testing.assert_array_equal(self.choose(ident)[1],expected[1])

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


    def test_new_pre_can_change_top2_before_true_boundary(self):
        self.a['state_weak_narrow'][:]=True;self.a['state_calm10'][:]=False
        self.a['pre_score'][:]=[.20,.1999,.1998,.1997,.1996,.1995]
        self.a['old_ml4']=self.a['rank_log_excess'].copy() # source weak replacement has no delta.
        for key in s.ARMS:self.a['raw_'+key][:]=[-9.,-8.,7.,8.,9.,10.]
        for ident in (3,4):
            pre=s.modified_pre_score(s.configs()[ident],self.a,4)
            anchors=sorted(range(6),key=lambda j:(-pre[0,j],self.m['symbols'][0][j]))[:2]
            self.assertNotEqual(set(anchors),{0,1});self.assertTrue(set(anchors)<=set(self.choose(ident)[0][0]))

    def test_inputs_unchanged_no_global_post_rerank(self):
        before=copy.deepcopy(self.a)
        for ident in range(5):self.choose(ident)
        for key in before:np.testing.assert_array_equal(self.a[key],before[key])
        self.a['baseline']=self.a['pre_score'].copy();self.a['core']=self.a['pre_score'].copy()
        self.a['state_calm10'][:]=False;self.a['state_weak_narrow'][:]=False
        self.a['effects'][:]=[0.,0.,-.5,-.5,-.5,0.]
        self.a['model_available'][:]=False
        for key in s.ARMS:self.a['raw_'+key][:]=np.nan
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


    def test_forecast_schema_numerical_failures_do_not_become_missing(self):
        available=np.array([True,True,False])
        good=dict(raw={key:np.array([0.,.1,np.nan]) for key in ('source_log',*s.ARMS)},
                  available=available.copy(),local_available=np.array([True,False,False]))
        s.validate_forecasts(good,3)
        for key in ('source_log',*s.ARMS):
            for value in (np.nan,np.inf,-np.inf):
                bad=copy.deepcopy(good);bad['raw'][key][0]=value
                with self.assertRaises(FloatingPointError):s.validate_forecasts(bad,3)
        bad=copy.deepcopy(good);bad['local_available'][2]=True
        with self.assertRaises(ValueError):s.validate_forecasts(bad,3)
        bad=copy.deepcopy(good);bad['raw']['local_constant'][2]=0.
        with self.assertRaises(FloatingPointError):s.validate_forecasts(bad,3)
        bad=copy.deepcopy(good);bad['available']=available.astype(int)
        with self.assertRaises(ValueError):s.validate_forecasts(bad,3)
        bad=copy.deepcopy(good);bad['raw']['local_linear'][1]+=.001
        with self.assertRaises(AssertionError):s.validate_forecasts(bad,3)

    def test_exact_source_raw_reuses_cached_rank_and_score_bytes(self):
        # Source raw replay permits tiny differences from archived raw; exact
        # local==replayed pooled means zero correction, never rerank either.
        self.a['raw_log_excess']=self.a['raw_source_log']+1e-13
        self.a['rank_log_excess'][0,2]=-0.
        for key in s.ARMS:self.a['raw_'+key]=self.a['raw_source_log'].copy()
        before=self.a['rank_log_excess'].tobytes();expected=self.choose(2)
        with patch.object(s.ep.old.prior,'ranks',side_effect=AssertionError('no rerank')):
            ranks,ready=self.ranks();self.assertTrue(ready.all())
            for key in s.ARMS:self.assertEqual(ranks[key].tobytes(),before)
            for ident in (3,4):
                got=self.choose(ident);self.assertEqual(got[0],expected[0])
                self.assertEqual(got[1].tobytes(),expected[1].tobytes())
                self.assertEqual(s.rp.make_orders(*got,self.m),s.rp.make_orders(*expected,self.m))

    def test_one_equal_arm_reuses_cache_other_arm_still_ranked(self):
        self.a['raw_local_constant']=self.a['raw_source_log'].copy()
        ranks,ready=self.ranks();self.assertTrue(ready.all())
        self.assertEqual(ranks['local_constant'].tobytes(),self.a['rank_log_excess'].tobytes())
        self.assertFalse(np.array_equal(ranks['local_linear'],self.a['rank_log_excess']))
        # own-history availability is audit only, never a query rank gate.
        self.a['local_available'][:]=False
        again,_=self.ranks()
        for key in s.ARMS:np.testing.assert_array_equal(again[key],ranks[key])

    def test_ranking_schema_and_nonfinite_original_fail(self):
        for n in (True,-1,5,1.5):
            with self.assertRaises(ValueError):self.choose(3,n=n)
        with self.assertRaises(ValueError):s.choices(dict(s.configs()[3],mode='unknown'),self.a,self.m,4,policy=POLICY)
        for key in s.ARMS:
            a=copy.deepcopy(self.a);a['raw_'+key][0,0]=np.inf
            with self.assertRaises(FloatingPointError):s.choices(s.configs()[3],a,self.m,4,policy=POLICY)
        a=copy.deepcopy(self.a);a['rank_log_excess'][0,0]=np.nan
        with self.assertRaises(ValueError):s.choices(s.configs()[3],a,self.m,4,policy=POLICY)
        a=copy.deepcopy(self.a);a['model_available']=a['model_available'].astype(int)
        with self.assertRaises(ValueError):s.choices(s.configs()[3],a,self.m,4,policy=POLICY)

    def test_source_raw_parity_ranker_preserves_original_population(self):
        self.a['eligible'][0,0]=False;raw=self.a['raw_local_constant'].copy();raw[0,0]=np.nan
        got=s.source_ranked_forecast(raw,self.a)
        self.assertEqual(got[0,0],self.a['rank_log_excess'][0,0])
        for i in range(4):
            use=self.a['eligible'][i]&self.a['forecast_available'][i]
            want=s.ep.old.prior.ranks(pd.Series(raw[i]).where(use))
            np.testing.assert_array_equal(got[i,use],want[use])
        raw[1,0]=np.nan
        with self.assertRaises(ValueError):s.source_ranked_forecast(raw,self.a)

    def test_identity_exact_past_pool_source_and_offsets(self):
        import pandas as pd
        tm=dict(dates=['2018-01-31','2018-02-28'],ends=['2018-02-28','2018-03-29'],symbols=[['A','B'],['ETF','A']])
        monthly=dict(dates=['2020-01-31'],symbols=[['A','ETF']])
        meta=dict(training=tm,monthly=monthly)
        a=dict(dates=np.repeat(np.array(tm['dates'],dtype='datetime64[D]'),2),
            ends=np.repeat(np.array(tm['ends'],dtype='datetime64[D]'),2),
            event_dates=np.array(['2020-01-31']*2,dtype='datetime64[D]'),offsets=np.array([0,2,4]),
            train_plain=np.arange(40).reshape(4,10),event_plain=np.ones((2,10)),
            train_eligible=np.array([True,True,False,True]),train_good=np.array([False,True,False,True]),
            eligible=np.array([[True,True]]))
        original=copy.deepcopy(a)
        audit=s.validate_identity(meta,a,original,copy.deepcopy(meta))
        self.assertEqual(audit['eligible_rows'],3)
        self.assertTrue(audit['explicit_symbols_offsets_dates_ends'])
        for key in s.target.CACHED_KEYS:
            bad=copy.deepcopy(a)
            bad[key].flat[0]=not bool(bad[key].flat[0]) if bad[key].dtype.kind=='b' else bad[key].flat[0]+1
            with self.subTest(key=key),self.assertRaises((AssertionError,ValueError)):
                s.validate_identity(meta,bad,original,meta)
        wrong=copy.deepcopy(meta);wrong['training']['symbols'][0]=['B','A']
        with self.assertRaises(AssertionError):s.validate_identity(wrong,a,original,meta)


    def test_panel_parity_fixed_scale_missing_and_opening(self):
        index=pd.bdate_range('2020-01-01',periods=5)
        adj=pd.DataFrame({'A':[1.,2.,np.nan,3.,4.],'B':[np.nan]*5},index=index)
        close=adj*3
        panel=dict(adj=adj,close=close,open=close*.99)
        px=adj*7
        op=px*.99
        audit=s.verify_panel_parity(panel,px,op)
        self.assertEqual(audit['scales'],{'A':7.})
        self.assertEqual(audit['all_missing_symbols'],['B'])
        bad=px.copy();bad.iloc[-1,0]*=1.01
        with self.assertRaises(AssertionError): s.verify_panel_parity(panel,bad,op)
        bad=px.ffill()
        with self.assertRaises(AssertionError): s.verify_panel_parity(panel,bad,op)
        with self.assertRaises(AssertionError): s.verify_panel_parity(panel,px,op*1.01)
        for value in (0.,-1.):
            bad_panel={key:frame.copy() for key,frame in panel.items()}
            bad_panel['adj'].iloc[0,0]=value
            with self.assertRaises(AssertionError): s.verify_panel_parity(bad_panel,px,op)
        bad_panel={key:frame.copy() for key,frame in panel.items()}
        bad_panel['adj'].iloc[0,0]=0.
        bad=px.copy();bad.iloc[0,0]=0.
        with self.assertRaises(ValueError): s.verify_panel_parity(bad_panel,bad,op)


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


    def test_synthetic_source_model_local_rank_selector_full_contract(self):
        days=pd.date_range('2013-01-31',periods=22,freq=pd.offsets.BMonthEnd()).to_numpy(dtype='datetime64[D]')
        ends=pd.date_range('2013-02-28',periods=22,freq=pd.offsets.BMonthEnd()).to_numpy(dtype='datetime64[D]')
        dates=np.repeat(days,6);end=np.repeat(ends,6);symbols=np.tile(list('ABCDEG'),22)
        rng=np.random.default_rng(1729);X=rng.uniform(-.5,.5,size=(132,10));eventX=rng.uniform(-.5,.5,size=(24,10))
        targets={key:rng.normal(scale=.02,size=132) for key in s.target.TARGETS}
        common=np.ones(132,bool)
        events=np.repeat(np.array(self.m['dates'],dtype='datetime64[D]'),6);event_good=np.ones(24,bool)
        event_symbols=np.array([symbol for pool in self.m['symbols'] for symbol in pool])
        before=[value.copy() for value in (X,eventX,common,*targets.values())]
        forecast=s.model.annual_forecasts(dates,end,symbols,X,targets,common,events,event_symbols,eventX,event_good)
        s.validate_forecasts(forecast,24);self.assertTrue(forecast['available'].all())
        self.assertFalse(forecast['local_available'].reshape(4,6)[:,5].any())
        self.assertTrue(forecast['local_available'].reshape(4,6)[:,:5].all())
        reference=s.target.model.annual_forecasts(dates,end,X,targets,common,events,eventX,event_good)
        np.testing.assert_allclose(forecast['raw']['source_log'],reference['log_excess'],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(forecast['available'],reference['available'])
        for old,new in zip(before,(X,eventX,common,*targets.values())):np.testing.assert_array_equal(old,new)
        self.a['raw_source_log']=forecast['raw']['source_log'].reshape(4,6)
        self.a['rank_log_excess']=s.source_ranked_forecast(self.a['raw_source_log'],self.a)
        self.a['model_available']=forecast['available'].reshape(4,6)
        self.a['local_available']=forecast['local_available'].reshape(4,6)
        for key in s.ARMS:
            self.a['raw_'+key]=forecast['raw'][key].reshape(4,6)
            np.testing.assert_array_equal(self.a['raw_'+key][:,5],self.a['raw_source_log'][:,5])
        ranks,ready=self.ranks();self.assertTrue(ready.all())
        # F has no same-security training history but remains a full ranked U member.
        for key in s.ARMS:
            for i in range(4):
                expected=s.ep.old.prior.ranks(pd.Series(self.a['raw_'+key][i]))
                np.testing.assert_array_equal(ranks[key][i],expected)
        for ident in (3,4):
            full=self.choose(ident)
            self.assertTrue(all(len(row)==3 for row in full[0]))
            self.assertLessEqual(np.abs(s.score_delta(s.configs()[ident],self.a,4)).max(),.0125)
            for n in range(5):
                prefix=self.choose(ident,n=n)
                self.assertEqual(prefix[0],full[0][:n]);np.testing.assert_array_equal(prefix[1],full[1][:n])

    def test_source_hash_verification_and_internal_source_model_independent(self):
        payload={'FROZEN.json':dict(hashes={'s':'h'},selected={'primary':{'spec':dict(s.SOURCE_SCOPE_SPEC)}}),
                 'INPUT_HASHES.json':{'i':'h'},'CACHE_HASHES.json':{'c':'h'}}
        calls=[]
        with patch.object(s.scope,'registered') as registered,patch.object(s.scope,'verify_source') as verify, \
                patch.object(s,'sha',return_value='hash'),patch.object(Path,'read_text',return_value='hash\n'), \
                patch.object(s,'read',side_effect=lambda f:payload[Path(f).name]), \
                patch.object(s.old,'assert_hashes',side_effect=lambda x,root=None:calls.append((x,root))):
            s.verify_source()
        registered.assert_called_once_with();verify.assert_called_once_with()
        self.assertEqual(len(calls),5)
        self.assertEqual(calls[0],({'s':'h'},s.scope.P))
        self.assertEqual(calls[-1],({'c':'h'},s.target.P))

    def test_raw64_original_source_manifest_requires_all_exact(self):
        loader=str(s.O/'monthly_stability_monitor/inputs.py')
        raw={str(i):'hash' for i in range(63)};raw[loader]=s.sha(loader)
        with patch.object(s,'read',return_value=dict(raw)):
            self.assertEqual(s.raw_source_hashes(dict(source_hashes=raw)),raw)
            for bad in (dict(raw,extra='x'),{k:v for k,v in raw.items() if k!=loader},dict(raw,**{loader:'bad'})):
                with self.assertRaises(AssertionError):s.raw_source_hashes(dict(source_hashes=bad))
        with patch.object(s,'read',return_value={}):
            with self.assertRaises(AssertionError):s.raw_source_hashes(dict(source_hashes=raw))

    def test_dependency_pins_include_models_numeric_calendar_and_identity(self):
        files=set(s.implementations());sources=set(s.source_files())
        for path in (P/'study.py',P/'test_study.py',P/'model.py',P/'test_model.py',
                     Path(s.adapter.__file__),Path(s.ep.__file__),Path(s.scope.__file__),Path(s.calendar_labels.__file__),s.calendar_labels.CALENDAR_PATH,*s.model.dependency_files()):
            self.assertIn(path,files)
        for path in (s.target.P/'TRAIN_INPUTS.npz',s.target.old.P/'META.json',s.target.old.P/'DATA.npz',s.arch.P/'META.json',
                     s.scope.P/'primary_ORDERS.json',s.scope.P/'original65_ORDERS.json',s.scope.P/'incumbent66_ORDERS.json'):
            self.assertIn(path,sources)
        self.assertFalse({'RESULTS.json','TRAIN_TRIALS.json','TRAIN_RETURNS.npz'}&{p.name for p in sources})

    def test_prepare_independent_source_availability_and_no_own_history_gate(self):
        text=(P/'study.py').read_text();tree=ast.parse(text)
        fs={x.name:ast.get_source_segment(text,x) for x in tree.body if isinstance(x,ast.FunctionDef)}
        prepare=fs['prepare']
        self.assertIn("event_good=a['eligible'].reshape(-1)&np.isfinite(cached['event_plain']).all(axis=1)",prepare)
        self.assertIn("np.testing.assert_array_equal(source_available,a['forecast_available'])",prepare)
        self.assertIn("np.testing.assert_allclose(source_raw,a['raw_log_excess'],rtol=0.,atol=PARITY_ATOL,equal_nan=True)",prepare)
        self.assertIn("cached['train_plain'],source_targets,",prepare)
        self.assertIn("cached['common_good'],cached['event_dates'],event_symbols",prepare)
        self.assertIn("source_ranked_forecast(source_raw,a),a['rank_log_excess']",prepare)
        self.assertIn("assert cached['train_plain'].shape==(8820,10)",prepare)
        self.assertNotIn('local_available',fs['ranked_forecasts']);self.assertNotIn('local_available',fs['score_delta'])
        self.assertIn("('local_linear','local_constant')",fs['evaluate'])
        self.assertIn("('next_open_zero',0.),('next_open_cost',.0005)",fs['evaluate'])
        self.assertIn("old.assert_hashes(plan['raw_source_hashes'])",fs['registered'])
        self.assertIn("calendar_provenance()==plan['calendar']",fs['registered'])
        self.assertNotIn('monthly_monthly_update70',text)
        self.assertNotIn('monthly_beta_target70',text)
        self.assertNotIn('build_targets(',prepare)


if __name__=='__main__':unittest.main()

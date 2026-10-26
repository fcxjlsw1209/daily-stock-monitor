"""Invented complete-history, covariance, boundary and causal-prefix tests."""
import copy
import math
import unittest
import numpy as np
import pandas as pd
import monthly_joint_error226_independent as oracle


def fixture():
    ds=pd.date_range('2016-08-31','2019-02-28',freq='ME');names=['A','B','C','D','E']
    tm=dict(dates=[str(d.date()) for d in ds],ends=[str((d+pd.offsets.MonthEnd(1)).date()) for d in ds],symbols=[names.copy() for _ in ds])
    qm=dict(dates=['2016-08-17']+tm['dates'],ends=['2016-08-31']+tm['ends'],symbols=[names.copy()]+copy.deepcopy(tm['symbols']),etfs=['E'])
    nt=len(ds)*5;n=len(qm['dates'])
    train=dict(offsets=np.arange(len(ds)+1,dtype=int)*5,dates=np.repeat(np.array(tm['dates'],dtype='datetime64[D]'),5),
        ends=np.repeat(np.array(tm['ends'],dtype='datetime64[D]'),5),train_plain=np.zeros((nt,10)),common_good=np.ones(nt,bool))
    for k in oracle.TARGETS:train['target_'+k]=np.zeros(nt)
    values=[]
    for i in range(n):
        t=float((i%7)-3)/8.;values.extend([t,t,-2*t,t,-t])
    source=dict(raw_local_linear=np.array(values),available=np.ones(n*5,bool),local_available=np.zeros(n*5,bool),
        raw_source_log=np.full(n*5,np.nan),raw_candidate=np.full(n*5,123.))
    return [train,tm,qm,source,np.ones((n,5),bool),np.tile([.4,.3,.2,.19,.185],(n,1)),np.ones((n,5),bool)]


def physical_prefix(args,n):
    train,tm,qm,source,use,pre,eligible=copy.deepcopy(args);w=use.shape[1]
    for k in ('dates','ends','symbols'):qm[k]=qm[k][:n]
    source={k:v[:n*w] for k,v in source.items()};use,pre,eligible=use[:n],pre[:n],eligible[:n]
    cutoff='%04d-12-31'%(int(qm['dates'][-1][:4])-1) if n else None
    keep=[i for i,e in enumerate(tm['ends']) if cutoff is not None and e<=cutoff]
    ids=np.array([j for i in keep for j in range(int(train['offsets'][i]),int(train['offsets'][i+1]))],dtype=int)
    train={k:v[ids] for k,v in train.items() if k!='offsets'}
    for k in ('dates','ends','symbols'):tm[k]=[tm[k][i] for i in keep]
    train['offsets']=np.r_[0,np.cumsum(np.array([len(ss) for ss in tm['symbols']],dtype=int))]
    return [train,tm,qm,source,use,pre,eligible]


def original(pre,eligible):
    n,w=pre.shape;zero=np.zeros((n,w));yes=np.ones((n,w),bool)
    return dict(baseline=zero.copy(),core=zero.copy(),core_weight=np.full((n,1),.52),eligible=eligible.copy(),
        state_weak_narrow=np.zeros(n,bool),state_calm10=np.zeros(n,bool),pre_score=pre.copy(),effects=np.tile([0.,0.,-.5,.5,0.],(n,1)),
        forecast_available=yes.copy(),rank_log_excess=zero.copy(),old_ml4=zero.copy(),extension_available=np.zeros((n,w),bool),
        model_available=yes.copy(),raw_local_constant=zero.copy(),raw_local_linear=zero.copy(),raw_source_log=zero.copy())


def current(out):return out['audit']['event_history'][-1]
def mature(args):return [i for i,e in enumerate(args[2]['ends']) if e<='2018-12-31' and args[2]['dates'][i]!='2016-08-17']


class ArithmeticTests(unittest.TestCase):
    def test_opposite_errors_cancel_joint_but_not_diagonal(self):
        z=np.array([[t,t,-2*t,t,-t] for t in (-3.,-1.,1.,3.)])
        s=oracle.centered_gram(z)
        self.assertEqual(s['candidate'][0],0.)
        self.assertGreater(s['control'][0],s['control'][1])
        self.assertGreater(s['candidate'][1],s['candidate'][2])
        self.assertTrue(np.allclose(s['gram'],np.cov(z,rowvar=False,ddof=1)))
        self.assertGreaterEqual(s['minimum_eigenvalue'],-s['diagnostic_tolerance'])

    def test_perfect_common_errors_and_affine_column_mean_invariance(self):
        a=np.tile(np.array([-3.,-1.,1.,3.])[:,None],(1,5));x=oracle.centered_gram(a)
        for j in range(3):self.assertAlmostEqual(x['candidate'][j],3*x['control'][j])
        b=a+np.array([4.,-2.,10.,16.,-24.]);y=oracle.centered_gram(b)
        for k in ('candidate','control','variances'):self.assertEqual(x[k],y[k])
        self.assertEqual(oracle.negative_risk_ranks(x['candidate']),[0.,0.,0.])

    def test_orthogonal_columns_full_and_diagonal_match(self):
        a=np.array([[1,1,1,-1],[1,-1,-1,-1],[-1,1,-1,1],[-1,-1,1,1]],float)
        s=oracle.centered_gram(a[:,:3]);self.assertAlmostEqual(s['candidate'][0],s['control'][0])

    def test_zero_variance_is_supported_not_unavailable(self):
        s=oracle.centered_gram(np.ones((18,5)))
        self.assertEqual(s['candidate'],[0.,0.,0.]);self.assertEqual(s['control'],[0.,0.,0.])
        self.assertEqual(oracle.negative_risk_ranks([0.,1.,1.]),[.5,-.25,-.25])
        self.assertEqual(oracle.negative_risk_ranks([1.,0.]),[-.5,.5])

    def test_hard_arithmetic_overflow_and_underflow(self):
        for v in (1e308,1e-200):
            with self.assertRaises(FloatingPointError):oracle.centered_gram(np.array([[v,1.,1.],[-v,-1.,-1.]]))
        with self.assertRaises(ValueError):oracle.negative_risk_ranks([-1.,0.])
        tiny=float(np.nextafter(0.,1.))
        with self.assertRaises(FloatingPointError):oracle.mean(tiny,2,'tiny mean')
        with self.assertRaises(FloatingPointError):oracle.centered_gram([[math.inf,1.,1.],[0.,0.,0.]])


class CohortTests(unittest.TestCase):
    def test_exact_complete_join_counter_accounting_and_gram(self):
        args=fixture();out=oracle.independent_build(*args);e=current(out)
        self.assertEqual(e['H'],[args[2]['dates'][i] for i in mature(args)])
        self.assertEqual(e['status'],'ready');self.assertEqual(e['counts']['complete_history_months'],28)
        self.assertEqual(e['counts']['parameter_estimates'],16);self.assertEqual(e['counts']['error_observations'],140)
        self.assertEqual(e['J'],[2,3,4]);self.assertEqual(e['B'],[0,1,2,3,4])
        self.assertEqual(e['trios'][0]['candidate_risk'],0.)
        receipt=oracle.check_build(*args,out)
        self.assertTrue(all(receipt[k] for k in oracle.MODEL_FLAGS));self.assertEqual(receipt['numerical_tolerance'],0.)
        for k in oracle.COUNT_FIELDS:self.assertEqual(out['audit']['counts'][k],sum(x['counts'][k] for x in out['audit']['event_history']))
        self.assertEqual(receipt['source_model_refits'],0);self.assertEqual(receipt['new_ridge_fits'],0)

    def test_no_current_source_availability_requirement_or_local_history_filter(self):
        args=fixture();i=len(args[2]['dates'])-1
        args[3]['available'][i*5:]=False;args[3]['raw_local_linear'][i*5:]=np.nan
        out=oracle.independent_build(*args)
        self.assertEqual(current(out)['status'],'ready');self.assertEqual(current(out)['counts']['cached_forecast_rows'],0)

    def test_initial_and_early_prior_year_cold_fallback(self):
        args=fixture();out=oracle.independent_build(*args)
        for i,d in enumerate(args[2]['dates']):
            e=out['audit']['event_history'][i]
            if d=='2016-08-17':self.assertEqual(e['status'],'initial_partial');self.assertEqual(e['history'],[])
            elif int(d[:4])<=2018:self.assertEqual(e['status'],'insufficient_history')
            if not out['arrays']['ready'][i]:
                self.assertTrue(np.isnan(out['arrays']['candidate_risk'][i]).all())
                self.assertFalse(np.any(out['arrays']['candidate_effect'][i]))

    def test_17_18_month_boundary_shared_all_boundary_ids(self):
        for kept in (17,18):
            args=fixture()
            for i in mature(args)[kept:]:args[4][i,4]=False
            out=oracle.independent_build(*args);e=current(out)
            self.assertEqual(len(e['H']),kept);self.assertEqual(e['status'],'ready' if kept==18 else 'insufficient_history')
            self.assertEqual(e['counts']['unsupported_source_rows'],28-kept)
            self.assertEqual(e['counts']['error_observations'],kept*5)
            # Cannot retain easier per-trio samples when E is missing.
            self.assertEqual(e['counts']['complete_history_months'],kept)

    def test_missing_identity_drops_whole_month_not_whole_query(self):
        args=fixture();h=mature(args)[0];args[2]['symbols'][h][4]='OLD'
        out=oracle.independent_build(*args);e=current(out)
        self.assertEqual(e['status'],'ready');self.assertEqual(len(e['H']),27)
        self.assertEqual(e['history'][0]['rows'][4]['reason'],'missing_security')
        self.assertIsNone(e['history'][0]['rows'][4]['source_row_id'])

    def test_missing_consumed_training_triple_is_integrity_error(self):
        args=fixture();args[1]['symbols'][0][2]='WRONG'
        with self.assertRaisesRegex(ValueError,'Missing mature exact training identity'):oracle.independent_build(*args)

    def test_common_good_all_four_targets_and_plain10_gate(self):
        args=fixture();args[0]['common_good'][4]=False;args[0]['target_loss_utility'][5+4]=np.nan;args[0]['train_plain'][10+4,9]=np.nan
        e=current(oracle.independent_build(*args))
        self.assertEqual(len(e['H']),25);self.assertEqual(e['counts']['invalid_label_rows'],3)
        self.assertEqual([h['rows'][4]['reason'] for h in e['history'][:3]],['common_good_false','nonfinite_training','nonfinite_training'])

    def test_no_target_recentering_and_unchanged_source_bytes(self):
        args=fixture();before=copy.deepcopy(args)
        for i in range(len(args[1]['dates'])):args[0]['target_log_excess'][5*i:5*i+5]=[2.,-3.,1.,4.,-1.]
        out=oracle.independent_build(*args);row=current(out)['history'][0]['rows'][0]
        self.assertEqual(row['target'],2.);self.assertEqual(row['error'],row['prediction']-2.)
        np.testing.assert_array_equal(args[3]['raw_local_linear'],before[3]['raw_local_linear'])

    def test_prior_december_maturity_and_exclusive_five_year_start(self):
        args=fixture();out=oracle.independent_build(*args)
        for e in out['audit']['event_history']:
            for h in e['history']:
                self.assertGreater(h['date'],e['lookback_exclusive_start']);self.assertLessEqual(h['end'],e['cutoff'])
                self.assertNotEqual(h['date'],'2016-08-17')
        # Numeric future labels are never consumed despite their structural metadata.
        for i,e in enumerate(args[1]['ends']):
            if e>'2018-12-31':
                args[0]['train_plain'][i*5:(i+1)*5]=np.inf
                for k in oracle.TARGETS:args[0]['target_'+k][i*5:(i+1)*5]=np.inf
        changed=oracle.independent_build(*args)
        oracle.record_check(changed['audit'],out['audit'])

    def test_gap_inclusive_top5_no_fifth_tie_expansion(self):
        args=fixture();args[5][-1]=[.4,.3,0.,-.02,-np.nextafter(.02,math.inf)]
        e=current(oracle.independent_build(*args));self.assertEqual(e['J'],[2,3])
        args[5][-1]=[.4,.3,.2,.1,.1]
        self.assertEqual(current(oracle.independent_build(*args))['status'],'insufficient_candidates')
        args[6][-1,:]=False;args[6][-1,:2]=True
        e=current(oracle.independent_build(*args));self.assertEqual(e['eligible_order'],[0,1]);self.assertEqual(e['history'],[])

    def test_sixth_place_exact_tie_is_not_expanded(self):
        args=fixture();train,tm,qm,source,use,pre,eligible=args
        for meta in (tm,qm):
            for row in meta['symbols']:row.append('F')
        nt=len(tm['dates']);n=len(qm['dates'])
        for key,value in list(train.items()):
            if key=='offsets':continue
            if key=='train_plain':
                v=value.reshape(nt,5,10);train[key]=np.concatenate([v,v[:,-1:,:]],axis=1).reshape(nt*6,10)
            else:
                v=value.reshape(nt,5);train[key]=np.concatenate([v,v[:,-1:]],axis=1).ravel()
        train['offsets']=np.arange(nt+1)*6
        for key,value in source.items():
            v=value.reshape(n,5);source[key]=np.concatenate([v,v[:,-1:]],axis=1).ravel()
        for index in (4,5,6):args[index]=np.concatenate([args[index],args[index][:,-1:]],axis=1)
        e=current(oracle.independent_build(*args));self.assertEqual(e['eligible_order'],[0,1,2,3,4,5])
        self.assertEqual(e['J'],[2,3,4]);self.assertNotIn(5,e['B'])

    def test_finite_prescore_gap_overflow_is_not_missing_history(self):
        args=fixture();args[5][-1]=[1e308,1e308,1e308,-1e308,-1e308]
        with self.assertRaisesRegex(FloatingPointError,'boundary score gap'):oracle.independent_build(*args)

    def test_symbol_ties_and_invalid_cache_rejected(self):
        args=fixture();args[5][-1]=0.;args[2]['symbols'][-1]=['E','D','C','B','A']
        self.assertEqual(current(oracle.independent_build(*args))['eligible_order'],[4,3,2,1,0])
        for index in (4,6):
            bad=fixture();bad[index]=bad[index].astype(int)
            with self.assertRaises(ValueError):oracle.independent_build(*bad)
        bad=fixture();bad[5][0,0]=np.nan
        with self.assertRaises(ValueError):oracle.independent_build(*bad)
        bad=fixture();bad[3]['raw_local_linear'][0]=np.nan
        with self.assertRaises(ValueError):oracle.independent_build(*bad)

    def test_all_physical_prefixes_and_mutation_detectors(self):
        args=fixture();full=oracle.independent_build(*args)
        for n in range(len(args[2]['dates'])+1):
            small=physical_prefix(args,n);out=oracle.independent_build(*small)
            oracle.record_check(out['audit']['event_history'],full['audit']['event_history'][:n])
            for k in oracle.ARRAY_KEYS:np.testing.assert_array_equal(out['arrays'][k],full['arrays'][k][:n])
        for part,key in [('arrays','candidate_effect'),('audit','parameter_estimates')]:
            mutated=copy.deepcopy(full)
            if part=='arrays':mutated[part][key][-1,2]+=.01
            else:mutated[part][key]+=1
            with self.assertRaises(ValueError):oracle.check_build(*args,mutated)


class SelectionTests(unittest.TestCase):
    def test_single_episode_replaces_effect_and_locks_members_not_order(self):
        args=fixture();built=oracle.independent_build(*args);a=original(args[5],args[6]);scores,choices=oracle.selection_oracle(a,args[2],built)
        self.assertEqual(choices[oracle.ARMS[0]][-1],[0,1,3])
        self.assertEqual(choices[oracle.ARMS[1]][-1],[0,1,2]);self.assertEqual(choices[oracle.ARMS[2]][-1],[0,1,3])
        proof=oracle.verify_selection(a,args[2],built,scores,choices);self.assertTrue(all(proof[k] for k in oracle.SELECTION_FLAGS))
        bad=copy.deepcopy(scores);bad[oracle.ARMS[1]][-1,2]+=.001
        with self.assertRaises(ValueError):oracle.verify_selection(a,args[2],built,bad,choices)

    def test_top_two_members_locked_but_final_score_order_can_change(self):
        args=fixture();args[5][:]=[.05,.04,.035,.03,.025]
        built=oracle.independent_build(*args);a=original(args[5],args[6])
        scores,choices=oracle.selection_oracle(a,args[2],built)
        self.assertEqual(choices[oracle.ARMS[1]][-1],[2,0,1])
        self.assertEqual(set(choices[oracle.ARMS[1]][-1]),{0,1,2})
        oracle.verify_selection(a,args[2],built,scores,choices)

    def test_full_support_zero_effect_is_pre_choice_not_incumbent_fallback(self):
        args=fixture();args[3]['raw_local_linear'][:]=0.;built=oracle.independent_build(*args)
        a=original(args[5],args[6]);scores,choices=oracle.selection_oracle(a,args[2],built)
        self.assertTrue(built['arrays']['ready'][-1]);self.assertEqual(choices[oracle.ARMS[0]][-1],[0,1,3])
        for arm in oracle.ARMS[1:]:
            self.assertEqual(choices[arm][-1],[0,1,2]);np.testing.assert_array_equal(scores[arm][-1],args[5][-1])

    def test_cold_initial_domain_fallback_exact_incumbent(self):
        args=fixture();args[6][-1]=[True,True,False,False,False];built=oracle.independent_build(*args)
        a=original(args[5],args[6]);scores,choices=oracle.selection_oracle(a,args[2],built)
        for i,ready in enumerate(built['arrays']['ready']):
            if not ready:
                for arm in oracle.ARMS[1:]:
                    self.assertEqual(choices[arm][i],choices[oracle.ARMS[0]][i]);self.assertEqual(scores[arm][i].tobytes(),scores[oracle.ARMS[0]][i].tobytes())
        self.assertEqual(choices[oracle.ARMS[1]][-1],[0,1])

    def test_support_clipping_original_baseline_not_modified_pre_and_state_budget(self):
        args=fixture();built=oracle.independent_build(*args);a=original(args[5],args[6])
        a['baseline'][:]=.48*.49;a['state_calm10'][-1]=True;a['state_weak_narrow'][-1]=True
        scores,choices=oracle.selection_oracle(a,args[2],built)
        budget=float(np.minimum(.05,np.maximum(0.,.10-(.025*a['state_calm10']+.05*a['state_weak_narrow'])))[-1])
        self.assertEqual(scores[oracle.ARMS[1]][-1,2],args[5][-1,2]+budget*(.5-.49))
        self.assertEqual(set(choices[oracle.ARMS[1]][-1]) & {0,1},{0,1})


class BilateralTests(unittest.TestCase):
    def test_unique_bilateral_fixtures_all_prefixes_full_audit(self):
        import monthly_joint_error226_model as producer
        from test_monthly_joint_error226_model import fixture as other_fixture
        for args in (fixture(),other_fixture()):
            n=len(args[2]['dates'])
            for end in range(n+1):
                small=physical_prefix(args,end)
                built=producer.build(*small)
                oracle.check_build(*small,built)


if __name__=='__main__':unittest.main()

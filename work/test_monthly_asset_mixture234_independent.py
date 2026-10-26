"""Synthetic-only common-x mixture, source funding and original selection tests."""
import copy
import importlib.util
import inspect
import math
import unittest
from unittest.mock import patch
import numpy as np
import monthly_asset_mixture234_independent as audit


def fixture():
    support=np.array([
        [-.4,-.2,0.,.2,.4],[-.4,-.2,0.,.2,.4],
        [.125]*5,[-.2,-.2,-.2,.2,.2],[-.4,-.2,0.,.2,.4],
        [-.4,-.2,0.,.2,.4],[-.25,0.,0.,.25,.25]])
    calm=np.array([False,True,True,True,False,False,False])
    weak=np.array([False,False,False,True,False,True,False])
    n,w=support.shape;cw=np.where(calm,.5,.52)[:,None]
    a=dict(baseline=(1-cw)*support,core=np.zeros((n,w)),core_weight=cw,
        state_calm10=calm,state_weak_narrow=weak,eligible=np.ones((n,w),bool),
        extension_available=np.ones((n,w),bool),forecast_available=np.ones((n,w),bool),
        rank_log_excess=np.zeros((n,w)),unconsumed=np.full((n,w),'poison',object))
    a['extension_available'][1,0]=False;a['forecast_available'][1,1]=False
    a['eligible'][1,2]=False;a['rank_log_excess'][1,3]=np.nan
    a['eligible'][2,0]=False
    names=[['S0','S1','S2','E0','E1'] for _ in range(n)]
    names[4]=['S'+str(j) for j in range(w)];names[5]=['E'+str(j) for j in range(w)]
    return a,dict(symbols=names,etfs=['E'+str(j) for j in range(w)]+['ABSENT'])


def physical_prefix(args,n):
    a,m=copy.deepcopy(args)
    return {k:v[:n].copy() for k,v in a.items()},dict(symbols=m['symbols'][:n],etfs=m['etfs'].copy())


def deltas(v):return {arm:v[key].copy() for arm,key in audit.DELTA_KEYS.items()}


def selection_fixture():
    dates=['2016-08-17','2016-08-31','2016-09-30'];n,w=3,5;shape=(n,w)
    pre=np.tile([.24,.18,.10,.095,-.08],(n,1));z=np.zeros(shape);f=np.zeros(shape,bool)
    a=dict(baseline=pre.copy(),core=np.zeros(shape),core_weight=np.full((n,1),.5),
        eligible=np.ones(shape,bool),effects=np.tile([0.,0.,-.4,.4,.5],(n,1)),
        state_calm10=np.ones(n,bool),state_weak_narrow=np.zeros(n,bool),pre_score=pre.copy(),
        rank_resid=z.copy(),old_ml4=z.copy(),rank_log_excess=z.copy(),extension_available=f.copy(),
        forecast_available=f.copy(),local_available=f.copy(),model_available=f.copy(),
        raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())
    m=dict(dates=dates,ends=['2016-08-31','2016-09-30','2016-10-31'],
        symbols=[['S0','S1','S2','E0','E1'] for _ in dates],etfs=['E0','E1'])
    return a,m


def order_rows(scores,choices,m):
    return {arm:[dict(date=m['dates'][i],end=m['ends'][i],signal_date=m['dates'][i],
        symbols=[m['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in audit.ARMS}


def original_production():
    p=audit.ROOT/'outputs/monthly_local_linear70/study.py'
    spec=importlib.util.spec_from_file_location('mixture234_test_original_local',p)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


class ScalarTests(unittest.TestCase):
    def test_hand_count_mixture_and_direct_pooled(self):
        q,p,c=audit.row_targets([-.4,-.2,0.,.2,.4],[0,1,2],[3,4])
        self.assertEqual(c['stock_less'],[0,1,2,3,3]);self.assertEqual(c['stock_equal'],[1,1,1,0,0])
        self.assertEqual(c['etf_less'],[0,0,0,0,1]);self.assertEqual(c['etf_equal'],[0,0,0,1,1])
        self.assertEqual(q,[.5*(.5/3)-.5,.5*(1.5/3)-.5,.5*(2.5/3)-.5,.5+.5*.25-.5,.5+.5*.75-.5])
        self.assertEqual(p,[(j+.5)/5-.5 for j in range(5)])
        self.assertNotEqual(q,p)

    def test_common_value_across_groups_same_target_and_zero_is_not_missing(self):
        q,p,c=audit.row_targets([-.25,0.,0.,.25],[0,1],[2,3])
        self.assertEqual(q[1],q[2]);self.assertEqual(p[1],p[2]);self.assertEqual(q[1],0.)
        self.assertEqual(c['pooled_equal'][1:3],[2,2]);self.assertEqual(c['stock_equal'][1],1)

    def test_constant_subgroups_are_step_cdfs_not_neutral(self):
        q,p,c=audit.row_targets([-.2]*3+[.2]*2,[0,1,2],[3,4])
        self.assertEqual(q,[-.25]*3+[.25]*2)
        self.assertEqual(p,[(1.5/5)-.5]*3+[(4./5)-.5]*2)
        self.assertTrue(c['stock_less']);self.assertNotEqual(q,p)

    def test_whole_constant_and_singleton_both_zero_without_cdf(self):
        for s,stocks,funds in (([.125]*5,[0,1,2],[3,4]),([.2],[0],[]),([-.0,0.],[0],[1])):
            q,p,c=audit.row_targets(s,stocks,funds)
            self.assertEqual(q,[0.]*len(s));self.assertEqual(q,p);self.assertTrue(all(not x for x in c.values()))

    def test_one_group_exact_copy_and_group_swap_symmetry(self):
        s=[-.4,-.1,0.,.3,.4]
        q,p,_=audit.row_targets(s,list(range(5)),[]);self.assertEqual(np.array(q).tobytes(),np.array(p).tobytes())
        q2,p2,_=audit.row_targets(s,[],list(range(5)));self.assertEqual(q2,q);self.assertEqual(p2,p)
        a=audit.row_targets(s,[0,1,2],[3,4]);b=audit.row_targets(s,[3,4],[0,1,2])
        self.assertEqual(a[:2],b[:2])

    def test_common_monotonicity_group_balance_math_not_float_identity(self):
        for s,st,et in (([-.2,-.1,0.,.1,.2],[0,1,2],[3,4]),([-.3,0.,.1,.4],[0,2],[1,3])):
            q,p,c=audit.row_targets(s,st,et)
            self.assertEqual(sorted(q),q);self.assertEqual(sorted(p),p)
            for j in range(len(s)):
                fs=(c['stock_less'][j]+.5*c['stock_equal'][j])/len(st)
                fe=(c['etf_less'][j]+.5*c['etf_equal'][j])/len(et)
                self.assertAlmostEqual(q[j]-p[j],(.5-len(st)/len(s))*(fs-fe),places=15)
                self.assertLessEqual(abs(q[j]),.5);self.assertLessEqual(abs(p[j]),.5)

    def test_exact_near_ties_and_scale_monotonic_invariance(self):
        s=[0.,np.nextafter(0.,1.),.1,.2]
        q,p,c=audit.row_targets(s,[0,2],[1,3]);self.assertEqual(c['pooled_equal'],[1]*4)
        x=[-.25,0.,.125,.25];a=audit.row_targets(x,[0,2],[1,3]);b=audit.row_targets([2*y+1 for y in x],[0,2],[1,3])
        self.assertEqual(a,b)


class FeatureTests(unittest.TestCase):
    def test_full_counts_receipts_and_no_input_mutation(self):
        a,m=fixture();before=copy.deepcopy(a);v,au=audit.independent_features(a,m)
        r=audit.verify_features(a,m,v,au);dr=audit.verify_deltas(v,a,m,deltas(v))
        self.assertTrue(all(r[k] is True for k in audit.FEATURE_FLAGS));self.assertTrue(all(dr[k] is True for k in audit.DELTA_FLAGS))
        self.assertEqual(r['counts'],{k:au[k] for k in audit.COUNT_FIELDS})
        self.assertEqual((au['both_group_rows'],au['single_group_rows'],au['constant_rows']),(5,2,1))
        self.assertEqual((au['checked_cells'],au['group_cdf_evaluations'],au['pooled_cdf_evaluations']),(35,40,30))
        self.assertEqual(au['stock_cells']+au['etf_cells'],35)
        for k in a:audit.exact_array(a[k],before[k],k)
        self.assertFalse(r['producer_modules_imported']);self.assertFalse(r['computes_returns'])

    def test_initial_and_whole_constant_noop_even_varying_weight(self):
        a,m=fixture();v,au=audit.independent_features(a,m)
        self.assertGreater(np.ptp(v['weight'][2]),0.)
        for key in ('candidate_delta','control_delta'):
            np.testing.assert_array_equal(v[key][0],0.);np.testing.assert_array_equal(v[key][2],0.)
        self.assertGreater(np.ptp(v['candidate'][0]),0.)
        for key in ('candidate','control'):np.testing.assert_array_equal(v[key][2],0.)
        self.assertEqual(au['event_history'][2]['branch'],'constant_row')
        self.assertTrue(np.any(v['candidate_delta'][3]!=0.))

    def test_full_domain_and_inherited_funding_mask_only(self):
        a,m=fixture();v,_=audit.independent_features(a,m)
        b=copy.deepcopy(a);b['eligible'][1]=False;w,_=audit.independent_features(b,m)
        for key in ('candidate','control','support'):audit.exact_array(v[key],w[key],key)
        self.assertNotEqual(v['weight'][1,4],w['weight'][1,4])
        self.assertTrue(np.any(w['candidate_delta'][1]!=0.))
        b=copy.deepcopy(a);b['eligible'][4]=False;w,_=audit.independent_features(b,m)
        for key in audit.VALUE_KEYS:audit.exact_array(v[key][4],w[key][4],key)

    def test_active_definition_exact_and_quarter_ml_not_deducted(self):
        a,m=fixture();v,_=audit.independent_features(a,m)
        self.assertEqual(v['weight'][1].tolist(),[.5-.025]*4+[.5-(.025+.01)])
        self.assertEqual(v['weight'][3,0],.5-(.025+.05))
        self.assertEqual(v['weight'][5,0],.48-.05)
        for key in ('extension_available','forecast_available','eligible'):
            b=copy.deepcopy(a);b[key][1,4]=False;z,_=audit.independent_features(b,m)
            self.assertEqual(z['weight'][1,4],.5-.025)
        for x in (math.nan,math.inf,-math.inf):
            b=copy.deepcopy(a);b['rank_log_excess'][1,4]=x;z,_=audit.independent_features(b,m)
            self.assertEqual(z['weight'][1,4],.5-.025)

    def test_explicit_identity_and_column_permutation(self):
        a,m=fixture();v,_=audit.independent_features(a,m)
        order=[4,2,0,3,1];aa={k:(x[:,order] if x.ndim==2 and x.shape[1]==5 else x.copy()) for k,x in a.items()}
        mm=dict(symbols=[[row[j] for j in order] for row in m['symbols']],etfs=m['etfs'])
        w,_=audit.independent_features(aa,mm)
        for k in v:audit.exact_array(w[k],v[k][:,order],k)
        mm=copy.deepcopy(m);mm['symbols'][0][0]='ETF_LOOKING';w,_=audit.independent_features(a,mm)
        for k in v:audit.exact_array(w[k],v[k],k)
        mm=copy.deepcopy(m);mm['etfs'].append('S0');w,_=audit.independent_features(a,mm)
        self.assertFalse(np.array_equal(w['candidate'][0],v['candidate'][0]))

    def test_both_groups_influence_every_common_x(self):
        a,m=fixture();v,_=audit.independent_features(a,m)
        b=copy.deepcopy(a);b['baseline'][1,3]=.5*(-.45)
        w,_=audit.independent_features(b,m)
        self.assertNotEqual(v['candidate'][1,0],w['candidate'][1,0])

    def test_all_physical_and_logical_prefixes_and_future_poison(self):
        a,m=fixture();v,au=audit.independent_features(a,m)
        for n in range(len(a['baseline'])+1):
            x,y=audit.independent_features(a,m,limit=n);p,q=physical_prefix((a,m),n);xx,yy=audit.independent_features(p,q)
            audit.record_check(y,yy)
            for k in v:audit.exact_array(x[k],v[k][:n],k);audit.exact_array(x[k],xx[k],k)
        b=copy.deepcopy(a);b['baseline']=b['baseline'].astype(object);b['baseline'][3:]=object()
        mm=copy.deepcopy(m);mm['symbols'][3:]=['bad']*4
        x,y=audit.independent_features(b,mm,limit=3)
        for k in v:audit.exact_array(x[k],v[k][:3],k)
        with self.assertRaises(ValueError):audit.independent_features(b,mm)

    def test_invalid_shapes_boolean_numeric_and_bounds_reject(self):
        for mode in ('cw','core','state','eligible','str','bool','nan','bound','prefix','symbols','etfs'):
            a,m=fixture()
            if mode=='cw':a['core_weight'][0,0]=.5
            if mode=='core':a['core']=a['core'][:,:-1]
            if mode=='state':a['state_calm10']=a['state_calm10'].astype(int)
            if mode=='eligible':a['eligible']=a['eligible'].astype(int)
            if mode in ('str','bool'):
                a['baseline']=a['baseline'].astype(object);a['baseline'][0,0]='0' if mode=='str' else True
            if mode=='nan':a['baseline'][0,0]=np.nan
            if mode=='bound':a['baseline'][0,0]=.48*.5001
            if mode=='symbols':m['symbols'][0][0]=m['symbols'][0][1]
            if mode=='etfs':m['etfs'].append(m['etfs'][0])
            with self.assertRaises(ValueError):audit.independent_features(a,m,limit=True if mode=='prefix' else None)

    def test_value_audit_and_delta_poison_rejected(self):
        a,m=fixture();v,au=audit.independent_features(a,m)
        for mode in ('raw','weight','delta','count','cdf','group','branch'):
            vv,aa=copy.deepcopy((v,au))
            if mode in ('raw','weight','delta'):
                key={'raw':'candidate','weight':'weight','delta':'candidate_delta'}[mode]
                vv[key][1,0]=np.nextafter(vv[key][1,0],math.inf)
            elif mode=='count':aa['pooled_cdf_evaluations']+=1
            elif mode=='cdf':aa['event_history'][1]['stock_less'][0]+=1
            elif mode=='group':aa['event_history'][1]['etf_indices']=[0,1]
            else:aa['event_history'][1]['branch']='single_group_copy'
            with self.assertRaises(ValueError):audit.verify_features(a,m,vv,aa)
        dd=deltas(v);dd[audit.ARMS[1]][1,0]+=1e-15
        with self.assertRaises(ValueError):audit.verify_deltas(v,a,m,dd)


class SelectionTests(unittest.TestCase):
    def test_complete_frozen_production_selector_parity(self):
        a,m=selection_fixture();v,_=audit.independent_features(a,m);s,c=audit.selection_oracle(a,m,v)
        source=original_production();pre=source.modified_pre_score(source.configs()[4],a,len(m['dates']))
        for arm in audit.ARMS:
            p=pre.copy()
            if arm in audit.DELTA_KEYS:
                d=v[audit.DELTA_KEYS[arm]];mask=d!=0.;p[mask]+=d[mask]
            cc,ss=source.adapter.score_and_choose('incumbent66',0.,np.full(p.shape,np.nan),a,m,len(p),
                pre_score=p,effects=a['effects'],episode_policy=source.POLICY_FIELDS,boundary_selector=source.ep.select_boundary)
            audit.exact_array(s[arm],ss,arm);self.assertEqual(c[arm],cc)
        receipt=audit.verify_selection(a,m,v,order_rows(s,c,m),scores=s)
        self.assertTrue(all(receipt[k] is True for k in audit.SELECTION_FLAGS));self.assertTrue(receipt['full_scores_supplied'])

    def test_selection_initial_constants_cash_two_and_all_prefixes(self):
        a,m=selection_fixture();v,_=audit.independent_features(a,m);s,c=audit.selection_oracle(a,m,v)
        for arm in audit.ARMS[1:]:audit.exact_array(s[arm][0],s[audit.ARMS[0]][0],arm)
        for n in range(4):
            ss,cc=audit.selection_oracle(a,m,v,n=n)
            for arm in audit.ARMS:audit.exact_array(ss[arm],s[arm][:n],arm);self.assertEqual(cc[arm],c[arm][:n])
        for count in (1,2):
            b=copy.deepcopy(a);b['eligible'][:]=False;b['eligible'][:,:count]=True;w,_=audit.independent_features(b,m)
            _,cc=audit.selection_oracle(b,m,w)
            self.assertTrue(all(len(row)==(2 if count==2 else 0) for rows in cc.values() for row in rows))
        b=copy.deepcopy(a);b['baseline'][:]=.0625;w,_=audit.independent_features(b,m);ss,cc=audit.selection_oracle(b,m,w)
        for arm in audit.ARMS[1:]:audit.exact_array(ss[arm],ss[audit.ARMS[0]],arm);self.assertEqual(cc[arm],cc[audit.ARMS[0]])

    def test_original_support_anchor_and_budget_grouping(self):
        a,m=selection_fixture();a['state_weak_narrow'][:]=True;v,_=audit.independent_features(a,m)
        s,c=audit.selection_oracle(a,m,v);x=audit.frozen_source_oracle().old.original_helper().original_prefix(a,m,3)
        budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
        for arm in audit.ARMS[1:]:
            pre=x['local']+v[audit.DELTA_KEYS[arm]];expected=pre.copy()
            for i,names in enumerate(m['symbols']):
                order=sorted(range(5),key=lambda j:(-pre[i,j],names[j]))
                for j in [j for j in order[2:5] if pre[i,order[2]]-pre[i,j]<=.02]:
                    expected[i,j]+=budget[i]*(np.clip(x['support'][i,j]+x['effects'][i,j],-.5,.5)-x['support'][i,j])
            audit.exact_array(s[arm],expected,arm)

    def test_wrong_full_score_or_order_or_source_rejected(self):
        a,m=selection_fixture();v,_=audit.independent_features(a,m);s,c=audit.selection_oracle(a,m,v);orders=order_rows(s,c,m)
        bad=copy.deepcopy(s);bad[audit.ARMS[1]][1,4]+=1e-12
        with self.assertRaises(ValueError):audit.verify_selection(a,m,v,orders,scores=bad)
        bad=copy.deepcopy(orders);bad[audit.ARMS[1]][1]['scores'][0]+=1e-12
        with self.assertRaises(ValueError):audit.verify_selection(a,m,v,bad,scores=s)
        bad=copy.deepcopy(v);bad['support'][1,0]+=1e-12
        with self.assertRaises(ValueError):audit.selection_oracle(a,m,bad)

    def test_no_producer_import_or_unpinned_original_helper(self):
        self.assertNotIn('import monthly_asset_mixture234_kernel',inspect.getsource(audit))
        with patch.object(audit,'FROZEN_HELPER_SHA','bad'),self.assertRaises(ValueError):audit.accounting()
        a,m=selection_fixture();v,_=audit.independent_features(a,m)
        with patch.object(audit,'SOURCE_SHA','bad'),self.assertRaises(ValueError):audit.selection_oracle(a,m,v)


class OutcomeTests(unittest.TestCase):
    def test_independent_nine_paths_45_slices(self):
        import test_monthly_upside_retention181_independent as helper
        with patch.object(helper,'audit',audit):args=helper.account_fixture()
        receipt=audit.audit_outcomes(*args);self.assertEqual((receipt['checked_paths'],receipt['checked_slices']),(9,45))
        bad=copy.deepcopy(args);bad[4][audit.ARMS[1]]['same_close_zero']['metrics']['cagr']+=.01
        with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*bad)


class BilateralTests(unittest.TestCase):
    def test_both_fixtures_all_prefixes_full_values_and_audit(self):
        import monthly_asset_mixture234_kernel as producer
        from test_monthly_asset_mixture234_kernel import fixture as other_fixture
        for args in (fixture(),other_fixture()):
            a,m=args;full=producer.build_features(a,m);audit.verify_features(a,m,*full)
            for n in range(len(a['baseline'])+1):
                logical=producer.build_features(a,m,limit=n);audit.verify_features(a,m,*logical,limit=n)
                pa,pm=physical_prefix(args,n);out=producer.build_features(pa,pm);audit.verify_features(pa,pm,*out)
                audit.record_check(out[1],logical[1])
                for k in audit.VALUE_KEYS:audit.exact_array(out[0][k],full[0][k][:n],k)


if __name__=='__main__':unittest.main(verbosity=2)

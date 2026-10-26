"""Independent synthetic separate-history axis and exact226 arithmetic checks."""
import copy
import math
import unittest
import numpy as np
import pandas as pd
import monthly_separate_history230_independent as oracle


def fixture():
    ds=pd.date_range('2014-01-31','2019-02-28',freq='ME')
    names=['A','B','C','D','E'];dates=[str(x.date()) for x in ds]
    ends=[str((x+pd.offsets.MonthEnd(1)).date()) for x in ds]
    tm=dict(dates=dates,ends=ends,symbols=[names.copy() for _ in ds])
    pos=next(i for i,d in enumerate(dates) if d>'2016-08-17')
    hd=dates[:pos]+['2016-08-17']+dates[pos:];he=ends[:pos]+['2016-08-31']+ends[pos:]
    hm=dict(dates=hd,ends=he,symbols=[names.copy() if i%2==0 else names[::-1] for i in range(len(hd))])
    qm=dict(dates=['2016-08-17','2016-08-31','2017-01-31','2018-01-31','2019-01-31'],
        ends=['2016-08-31','2016-09-30','2017-02-28','2018-02-28','2019-02-28'],symbols=[names.copy() for _ in range(5)],etfs=['E'])
    n=len(ds)*5
    train=dict(offsets=np.arange(len(ds)+1,dtype=int)*5,dates=np.repeat(np.array(dates,dtype='datetime64[D]'),5),
        ends=np.repeat(np.array(ends,dtype='datetime64[D]'),5),train_plain=np.zeros((n,10)),common_good=np.ones(n,bool))
    for k in oracle.TARGETS:train['target_'+k]=np.zeros(n)
    values=[]
    for i,row in enumerate(hm['symbols']):
        t=float((i%7)-3)/8.;v=dict(zip(names,[t,t,-2*t,t,-t]));values.extend(v[s] for s in row)
    source=dict(raw_local_linear=np.array(values),available=np.ones(len(values),bool),ignored='not consumed')
    pre=np.tile([.4,.3,.2,.19,.185],(5,1));eligible=np.ones((5,5),bool)
    return [train,tm,qm,hm,source,np.ones((len(hd),5),bool),pre,eligible],np.ones((5,5),bool)


def run(f):
    args,current=f
    return oracle.independent_build(*args,current_available=current)


def last(out):return out['audit']['event_history'][-1]


def physical_prefix(f,n):
    a,current=copy.deepcopy(f);w=current.shape[1]
    for k in ('dates','ends','symbols'):a[2][k]=a[2][k][:n]
    a[6]=a[6][:n];a[7]=a[7][:n];current=current[:n]
    limit=a[2]['dates'][-1] if n else ''
    h=sum(d<=limit for d in a[3]['dates'])
    for k in ('dates','ends','symbols'):a[3][k]=a[3][k][:h]
    a[4]={k:(v[:h*w].copy() if isinstance(v,np.ndarray) else v) for k,v in a[4].items()};a[5]=a[5][:h]
    cutoff='%04d-12-31'%(int(limit[:4])-1) if n else ''
    keep=np.array([e<=cutoff for e in a[1]['ends']],bool)
    rows=np.repeat(keep,5)
    a[0]={k:v[rows].copy() for k,v in a[0].items() if k!='offsets'}
    for k in ('dates','ends','symbols'):a[1][k]=[v for v,u in zip(a[1][k],keep) if u]
    a[0]['offsets']=np.arange(int(keep.sum())+1,dtype=int)*5
    return a,current


class ArithmeticTests(unittest.TestCase):
    def test_opposite_errors_cancel_joint_not_diagonal(self):
        a=np.array([[t,t,-2*t,t,-t] for t in (-3.,-1.,1.,3.)]);s=oracle.centered_gram(a)
        self.assertEqual(s['candidate'][0],0.);self.assertGreater(s['control'][0],s['control'][1])
        self.assertTrue(np.allclose(s['gram'],np.cov(a,rowvar=False,ddof=1)))
    def test_constant_and_perfectly_common_errors(self):
        z=oracle.centered_gram(np.ones((18,5)));self.assertEqual(z['candidate'],[0.,0.,0.])
        self.assertEqual(oracle.negative_risk_ranks(z['candidate']),[0.,0.,0.])
        a=np.tile(np.array([-3.,-1.,1.,3.])[:,None],(1,5));b=oracle.centered_gram(a)
        for j in range(3):self.assertAlmostEqual(b['candidate'][j],3*b['control'][j])
        c=oracle.centered_gram(a+np.array([4.,-2.,10.,16.,-24.]));self.assertEqual(b['candidate'],c['candidate'])
    def test_ties_and_nonfinite_overflow_underflow_hardfail(self):
        self.assertEqual(oracle.negative_risk_ranks([0.,1.,1.]),[.5,-.25,-.25])
        for v in (1e308,1e-200):
            with self.subTest(v=v),self.assertRaises(FloatingPointError):oracle.centered_gram([[v,1.,1.],[-v,-1.,-1.]])
        with self.assertRaises(FloatingPointError):oracle.mean(float(np.nextafter(0.,1.)),2,'mean')
        with self.assertRaises(ValueError):oracle.negative_risk_ranks([-1.,0.])


class SeparateAxisTests(unittest.TestCase):
    def test_earlier_history_supports_current2016_without_changing_currentaxis(self):
        f=fixture();out=run(f);events=out['audit']['event_history']
        self.assertEqual(out['audit']['query_months'],5);self.assertEqual(out['audit']['history_months'],63)
        self.assertTrue(out['audit']['separate_history_axis']);self.assertEqual(events[0]['status'],'initial_partial')
        self.assertEqual(events[1]['status'],'ready');self.assertEqual(len(events[1]['H']),23)
        self.assertEqual(events[1]['latest_label_end'],'2015-12-31');self.assertEqual(events[1]['trios'][0]['candidate_risk'],0.)
        # Same current identities with only late history are correctly cold.
        a,c=copy.deepcopy(f);start=a[3]['dates'].index('2016-08-17')
        for k in ('dates','ends','symbols'):a[3][k]=a[3][k][start:]
        for k in ('raw_local_linear','available'):a[4][k]=a[4][k][start*5:]
        a[5]=a[5][start:];late=run((a,c));self.assertEqual(late['audit']['event_history'][1]['status'],'insufficient_history')
    def test_current_availability_counts_only_not_boundaryfilter(self):
        f=fixture();f[1][:]=False;out=run(f)
        self.assertEqual(out['audit']['counts']['cached_forecast_rows'],0)
        self.assertTrue(out['arrays']['ready'][1:].all())
        self.assertEqual(last(out)['B'],[0,1,2,3,4])
        f[1][-1,2]=True;again=run(f);self.assertEqual(again['audit']['counts']['cached_forecast_rows'],1)
        for k in oracle.ARRAY_KEYS:np.testing.assert_array_equal(out['arrays'][k],again['arrays'][k])
    def test_exact_identity_join_uses_history_columns_and_flatoffsets(self):
        f=fixture();out=run(f);row=last(out)['history'][1]['rows'][0]
        self.assertEqual(row['symbol'],'A');self.assertEqual(row['source_row_id'],9);self.assertEqual(row['training_row_id'],5)
        self.assertEqual(row['prediction'],f[0][4]['raw_local_linear'][9])
    def test_partial_any_history_position_skipped_and_truncation_allowed(self):
        f=fixture();out=run(f);e=last(out)
        self.assertEqual(e['counts']['skipped_initial_months'],1)
        self.assertNotIn('2016-08-17',e['H']);self.assertTrue(all(h['date']!='2016-08-17' for h in e['history']))
        a,c=copy.deepcopy(f);i=a[3]['dates'].index('2016-08-17')+1
        for k in ('dates','ends','symbols'):a[3][k]=a[3][k][:i]
        for k in ('raw_local_linear','available'):a[4][k]=a[4][k][:i*5]
        a[5]=a[5][:i];self.assertEqual(last(run((a,c)))['counts']['skipped_initial_months'],1)
    def test_partial_followed_by_non_augustformal_rejected(self):
        a,c=fixture();i=a[3]['dates'].index('2016-08-31')
        for k in ('dates','ends','symbols'):del a[3][k][i]
        for k in ('raw_local_linear','available'):a[4][k]=np.delete(a[4][k],slice(i*5,(i+1)*5))
        a[5]=np.delete(a[5],i,axis=0)
        with self.assertRaisesRegex(ValueError,'partial'):run((a,c))
    def test_17_18_complete_common_months_boundary(self):
        for kept in (17,18):
            a,c=fixture();a[5][kept:,0]=False;out=run((a,c));e=out['audit']['event_history'][1]
            self.assertEqual(len(e['H']),kept);self.assertEqual(e['status'],'ready' if kept==18 else 'insufficient_history')
            self.assertEqual(e['counts']['error_observations'],kept*5)
    def test_missing_id_drops_entire_month_but_otherhistory_survives(self):
        a,c=fixture();a[3]['symbols'][0][4]='OLD';e=last(run((a,c)))
        self.assertEqual(e['status'],'ready');self.assertEqual(e['history'][0]['rows'][4]['reason'],'missing_security')
        self.assertIsNone(e['history'][0]['rows'][4]['source_row_id']);self.assertFalse(e['history'][0]['complete'])
    def test_missing_mature_training_triple_harderror(self):
        a,c=fixture();a[1]['symbols'][0][2]='WRONG'
        with self.assertRaisesRegex(ValueError,'Missing mature exact training identity'):run((a,c))
    def test_commonfourtargets_and_features_shared_gate(self):
        a,c=fixture();a[0]['common_good'][0]=False;a[0]['target_loss_utility'][5]=np.nan;a[0]['train_plain'][10,9]=np.inf
        e=last(run((a,c)));self.assertEqual(e['counts']['invalid_label_rows'],3)
        self.assertEqual([h['rows'][0]['reason'] for h in e['history'][:3]],['common_good_false','nonfinite_training','nonfinite_training'])
    def test_source_available_and_history_use_each_block_common_month(self):
        a,c=fixture();a[5][0,0]=False;a[4]['available'][5]=False;a[4]['raw_local_linear'][5]=np.nan
        e=last(run((a,c)));self.assertEqual(e['counts']['unsupported_source_rows'],2)
        self.assertFalse(e['history'][0]['complete']);self.assertFalse(e['history'][1]['complete'])
    def test_no_recentering_no_source_mutation(self):
        a,c=fixture();before=copy.deepcopy(a[4]);a[0]['target_log_excess'][:5]=[2.,-3.,1.,4.,-1.]
        e=last(run((a,c)));row=e['history'][0]['rows'][0];self.assertEqual(row['target'],2.);self.assertEqual(row['error'],row['prediction']-2.)
        for k in ('raw_local_linear','available'):self.assertEqual(a[4][k].tobytes(),before[k].tobytes())
    def test_allphysical_currentprefixes_keep_independent_historyprefix(self):
        f=fixture();full=run(f)
        for n in range(len(f[1])+1):
            small=run(physical_prefix(f,n))
            for k in oracle.ARRAY_KEYS:np.testing.assert_array_equal(small['arrays'][k],full['arrays'][k][:n])
            self.assertEqual(small['audit']['event_history'],full['audit']['event_history'][:n])
            self.assertEqual(small['audit']['counts'],{k:sum(e['counts'][k] for e in full['audit']['event_history'][:n]) for k in oracle.COUNT_FIELDS})
    def test_future_values_and_maturity_cutoffs_do_not_affect_earlier_rows(self):
        f=physical_prefix(fixture(),3);baseline=run(f);a,c=copy.deepcopy(fixture())
        for k in ('dates','ends','symbols'):a[2][k]=a[2][k][:3]
        a[6]=a[6][:3];a[7]=a[7][:3];c=c[:3]
        future=np.asarray(a[0]['ends'])>np.datetime64('2016-12-31')
        for k in oracle.TARGETS:a[0]['target_'+k][future]=np.inf
        a[0]['train_plain'][future]=np.inf
        for i,e in enumerate(a[3]['ends']):
            if e>'2016-12-31':a[4]['raw_local_linear'][5*i:5*i+5]=12345.
        changed=run((a,c));self.assertEqual(changed['audit']['event_history'],baseline['audit']['event_history'])
        for e in changed['audit']['event_history']:
            for hist in e['history']:
                self.assertGreater(hist['date'],e['lookback_exclusive_start']);self.assertLessEqual(hist['end'],e['cutoff'])
    def test_current_vs_history_shape_and_mask_corruption_rejected(self):
        for kind in ('current_dtype','current_shape','history_shape','source_mask','query_width','history_width'):
            a,c=fixture()
            if kind=='current_dtype':c=c.astype(int)
            elif kind=='current_shape':c=c[:-1]
            elif kind=='history_shape':a[5]=a[5][:-1]
            elif kind=='source_mask':a[4]['raw_local_linear'][0]=np.nan
            elif kind=='query_width':a[2]['symbols'][0].pop()
            else:a[3]['symbols'][0].pop()
            with self.subTest(kind=kind),self.assertRaises(ValueError):run((a,c))
    def test_gap_top2_and_maxfifth_rules_unchanged(self):
        a,c=fixture();a[6][-1]=[.4,.3,0.,-.02,-np.nextafter(.02,math.inf)]
        e=last(run((a,c)));self.assertEqual(e['J'],[2,3]);self.assertEqual(e['ordered_top2'],[0,1])
        a[6][-1]=[.4,.3,.2,.1,.1];self.assertEqual(last(run((a,c)))['status'],'insufficient_candidates')
    def test_sameaxes_exact_frozen226_parity(self):
        import monthly_joint_error226_independent as old
        from test_monthly_joint_error226_independent import fixture as old_fixture
        oldargs=old_fixture();tr,tm,qm,src,use,pre,el=oldargs
        out=oracle.independent_build(tr,tm,qm,copy.deepcopy(qm),src,use,pre,el,current_available=src['available'].reshape(use.shape))
        previous=old.independent_build(*oldargs)
        for k in oracle.ARRAY_KEYS:np.testing.assert_array_equal(out['arrays'][k],previous['arrays'][k])
        adjusted=copy.deepcopy(out['audit']);adjusted.pop('history_months');adjusted.pop('separate_history_axis');adjusted['arms']=list(old.ARMS)
        self.assertEqual(adjusted,previous['audit'])
    def test_checker_rejects_tampered_risk_effect_mask_audit_and_currentcounts(self):
        args,current=fixture();out=run((args,current));receipt=oracle.check_build(*args,out,current_available=current)
        self.assertTrue(all(receipt[k] for k in oracle.MODEL_FLAGS));self.assertEqual(receipt['numerical_tolerance'],0.)
        for kind in ('risk','effect','mask','count','history'):
            bad=copy.deepcopy(out)
            if kind=='risk':bad['arrays']['candidate_risk'][-1,2]=1.
            elif kind=='effect':bad['arrays']['candidate_effect'][-1,2]=-.4
            elif kind=='mask':bad['arrays']['ready'][-1]=False
            elif kind=='count':bad['audit']['counts']['cached_forecast_rows']+=1
            else:bad['audit']['history_months']-=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):oracle.check_build(*args,bad,current_available=current)
    def test_changed_historical_prediction_cannot_reuse_old_risk_receipt(self):
        a,c=fixture();out=run((a,c));a[4]['raw_local_linear'][0]+=.2
        with self.assertRaises(ValueError):oracle.check_build(*a,out,current_available=c)


class SelectionTests(unittest.TestCase):
    def test_original_episode_and_supported_effect_replacement(self):
        from test_monthly_joint_error226_independent import original
        args,current=fixture();out=run((args,current));orig=original(args[6],args[7])
        scores,choices=oracle.selection_oracle(orig,args[2],out)
        proof=oracle.verify_selection(orig,args[2],out,scores,choices)
        self.assertTrue(all(proof[k] for k in oracle.SELECTION_FLAGS));self.assertEqual(proof['ordered_choices_checked'],15)
        for arm in oracle.ARMS:
            self.assertTrue(all(set(row).issuperset({0,1}) for row in choices[arm]))
        bad=copy.deepcopy(scores);bad[oracle.ARMS[1]][1,2]+=.001
        with self.assertRaises(ValueError):oracle.verify_selection(orig,args[2],out,bad,choices)


class BilateralTests(unittest.TestCase):
    def test_both_fixtures_allphysicalprefixes_fullarray_audit(self):
        import monthly_separate_history230_model as producer
        from test_monthly_separate_history230_model import fixture as other,physical_prefix as other_prefix
        for value,prefixer in ((fixture(),physical_prefix),(other(),other_prefix)):
            for n in range(len(value[1])+1):
                args,current=prefixer(value,n);built=producer.build(*args,current_available=current)
                oracle.check_build(*args,built,current_available=current)

if __name__=='__main__':unittest.main()

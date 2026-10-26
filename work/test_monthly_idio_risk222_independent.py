"""Synthetic-only independent projection-risk and half-core verification."""
import copy
import hashlib
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_idio_risk222_independent as independent


def fixture(count=4,width=6):
    index=pd.bdate_range('2014-01-01',periods=1100)
    x=np.arange(len(index),dtype=float)
    market=np.exp(.00025*x+.04*np.sin(x/29.))
    a=np.exp(.0003*x+.08*np.sin(x/37.))
    frame=pd.DataFrame({'A':a,'B':3000.-x,'FUND':2.+x/1000.+.2*np.sin(x/71.),
        'SPY':market,'D':np.full(len(index),8.),'E':a.copy()},index=index)
    names=['A','B','FUND','SPY','D','E'][:width]
    days=['2016-08-17','2016-08-31','2016-09-30','2016-10-31'][:count]
    return {'adj':frame},dict(dates=days,symbols=[names[:] for _ in days],etfs=['FUND','SPY']),index.strftime('%Y-%m-%d').tolist()


def examples():
    formation=np.ones(253);formation[-1]=2.
    stock=np.ones(64);stock[2]=2.
    market=np.ones(64);market[1]=2.
    return formation,stock,market


def original_fixture(rows=4,width=6,shape='column'):
    calm=np.array([i%2==0 for i in range(rows)],bool)
    weights=np.where(calm,.5,.52)
    rank=(np.arange(width)+1)/max(width,1)-.5
    core=np.array([np.roll(rank,i) for i in range(rows)])*weights[:,None] if rows else np.empty((0,width))
    if shape=='matrix':weights=np.broadcast_to(weights[:,None],(rows,width)).copy()
    elif shape=='column':weights=weights[:,None]
    return dict(core=core,core_weight=weights,state_calm10=calm,eligible=np.zeros((rows,width),bool))


def targets(values,original,full=None):
    result={};n,width=values['available'].shape
    calm=np.asarray(original['state_calm10'][:n],bool)
    for arm,key in independent.RAW.items():
        v=np.full((n if full is None else full,width),np.nan)
        v[:n]=values[key];v[:n][calm]=np.nan
        if n:v[0]=np.nan
        result[arm]=v
    return result


def physical_prefix(args,n):
    p,m,c=args;mm=copy.deepcopy(m)
    for key in ('dates','ends','symbols'):
        if key in mm:mm[key]=mm[key][:n]
    last=pd.Timestamp.min if not n else pd.Timestamp(m['dates'][n-1])
    return {'adj':p['adj'].loc[:last].copy()},mm,c


def scalar_expected(stock,market):
    sr=[float(b)/float(a)-1. for a,b in zip(stock,stock[1:])]
    mr=[float(b)/float(a)-1. for a,b in zip(market,market[1:])]
    sm,mm=math.fsum(sr)/63.,math.fsum(mr)/63.
    y,x=[r-sm for r in sr],[r-mm for r in mr]
    sy,sx=math.fsum(a*a for a in y),math.fsum(a*a for a in x)
    cross=math.fsum(a*b for a,b in zip(x,y));beta=cross/sx
    e=[a-beta*b for a,b in zip(y,x)]
    rss=math.fsum(a*a for a in e)
    return sm,mm,sy,sx,cross,beta,e,rss


class ScalarTests(unittest.TestCase):
    def test_exact_handworked_projection_and_denominator62(self):
        f,s,m=examples();a=independent.measure_oracle(f,s,m)
        sm,mm,sy,sx,cross,beta,e,rss=scalar_expected(s,m)
        self.assertTrue(a['source_available']);self.assertTrue(a['available'])
        self.assertEqual(a['momentum'],1.)
        for key,value in [('stock_mean',sm),('market_mean',mm),('stock_centered_sum_squares',sy),
                         ('market_centered_sum_squares',sx),('cross_product_sum',cross),('beta',beta),('residual_sum_squares',rss)]:
            self.assertEqual(a[key],value,key)
        self.assertEqual(a['total_risk'],math.sqrt(sy/62.))
        self.assertEqual(a['idiosyncratic_risk'],math.sqrt(rss/62.))
        self.assertEqual(a['raw_candidate'],1./math.sqrt(rss/62.))
        self.assertEqual(a['raw_control'],1./math.sqrt(sy/62.))
        self.assertNotEqual(a['idiosyncratic_risk'],math.sqrt(rss/61.))
        self.assertNotEqual(a['idiosyncratic_risk'],math.sqrt(rss/63.))
        self.assertEqual(a['descriptive_beta_fit'],1)

    def test_market_self_and_scaled_price_copy_exact_zero_residual(self):
        f,_,m=examples()
        for s in (m,m*8.,m*.125):
            a=independent.measure_oracle(f,s,m)
            self.assertTrue(a['source_available']);self.assertFalse(a['available'])
            self.assertEqual(a['beta'],1.)
            self.assertEqual(a['residual_sum_squares'],0.)
            self.assertEqual(a['idiosyncratic_risk'],0.)
            self.assertEqual(a['reason'],'zero_idiosyncratic_risk')
            self.assertIsNone(a['raw_control']);self.assertEqual(a['descriptive_beta_fit'],1)

    def test_constant_market_no_fit_and_constant_stock_fit_zero(self):
        f,s,m=examples()
        a=independent.measure_oracle(f,s,np.ones(64))
        self.assertEqual(a['reason'],'zero_market_variance');self.assertEqual(a['descriptive_beta_fit'],0)
        self.assertTrue(a['source_available']);self.assertGreater(a['total_risk'],0.)
        self.assertEqual(a['market_centered_sum_squares'],0.)
        for key in ('beta','residual_sum_squares','idiosyncratic_risk','raw_candidate','raw_control'):self.assertIsNone(a[key])
        a=independent.measure_oracle(f,np.ones(64),m)
        self.assertEqual(a['reason'],'zero_total_risk');self.assertEqual(a['descriptive_beta_fit'],1)
        for key in ('beta','total_risk','idiosyncratic_risk','residual_sum_squares'):self.assertEqual(a[key],0.)
        a=independent.measure_oracle(f,np.ones(64),np.ones(64))
        self.assertEqual(a['reason'],'zero_market_variance');self.assertEqual(a['descriptive_beta_fit'],0)

    def test_signed_zero_momentum_and_negative_beta_are_valid(self):
        f,s,m=examples()
        for end in (.5,1.,2.):
            f[-1]=end;a=independent.measure_oracle(f,s,m)
            self.assertTrue(a['available']);self.assertEqual(a['momentum'],end-1.)
            self.assertLess(a['beta'],0.)
            self.assertEqual(a['raw_candidate'],a['momentum']/a['idiosyncratic_risk'])
            self.assertEqual(a['raw_control'],a['momentum']/a['total_risk'])
            if end==1.:self.assertEqual(a['raw_candidate'],0.);self.assertFalse(np.signbit(a['raw_candidate']))

    def test_near_collinear_nonzero_risk_is_not_floored(self):
        f,_,m=examples();s=m.copy();s[-1]=np.nextafter(1.,math.inf)
        a=independent.measure_oracle(f,s,m)
        self.assertTrue(a['available']);self.assertGreater(a['idiosyncratic_risk'],0.)
        self.assertLess(a['idiosyncratic_risk'],1e-12)
        self.assertGreater(a['raw_candidate'],1e12)

    def test_residual_rss_has_no_second_mean_removal(self):
        p,m,c=fixture(1);stock=p['adj']['A'].to_numpy()[-64:];market=p['adj']['SPY'].to_numpy()[-64:]
        f,_,_=examples();a=independent.measure_oracle(f,stock,market)
        sm,mm,sy,sx,cross,beta,e,rss=scalar_expected(stock,market)
        self.assertNotEqual(math.fsum(e),0.)
        self.assertEqual(a['residual_sum_squares'],rss)
        self.assertEqual(a['idiosyncratic_risk'],math.sqrt(rss/62.))
        self.assertEqual(a['raw_candidate'],1./math.sqrt(rss/62.))

    def test_all_three_source_windows_full_positive_numeric_required(self):
        for k,label in enumerate(('formation','stock_risk','market_risk')):
            for position in (0,1,42,63 if k else 126,-1):
                for bad in (0.,-1.,np.nan,np.inf):
                    args=[x.copy() for x in examples()];args[k][position]=bad
                    a=independent.measure_oracle(*args)
                    self.assertEqual(a['reason'],'invalid_'+label+'_price')
                    self.assertFalse(a['source_available']);self.assertEqual(a['descriptive_beta_fit'],0)
                    self.assertTrue(all(a[key] is None for key in independent.MEASURE_KEYS))
            for bad in (True,None,'2',2+0j):
                args=[x.astype(object) for x in examples()];args[k][20]=bad
                self.assertEqual(independent.measure_oracle(*args)['reason'],'nonnumeric_'+label+'_price')
            args=list(examples());args[k]=args[k][:-1]
            with self.assertRaises(ValueError):independent.measure_oracle(*args)
        args=list(examples());args[0][0]=np.nan;args[2]=args[2][:-1]
        with self.assertRaises(ValueError):independent.measure_oracle(*args)

    def test_units_input_immutability_and_valid_source_overflow(self):
        args=examples();copy_args=[x.copy() for x in args];base=independent.measure_oracle(*args)
        for scales in ((2.**-500,2.**200,2.**-200),(2.**500,2.**-200,2.**200)):
            a=independent.measure_oracle(*[x*scale for x,scale in zip(args,scales)])
            for key in ('momentum','beta','total_risk','idiosyncratic_risk','raw_candidate','raw_control'):
                self.assertEqual(a[key],base[key])
        for x,y in zip(args,copy_args):np.testing.assert_array_equal(x,y)
        for k in range(3):
            bad=[x.copy() for x in args];bad[k][0]=5e-324;bad[k][1 if k else -1]=1e308
            with self.assertRaises(FloatingPointError):independent.measure_oracle(*bad)
        bad=[x.copy() for x in args];bad[1][1]=1e308
        with self.assertRaises(FloatingPointError):independent.measure_oracle(*bad)
        with self.assertRaises(FloatingPointError):independent.square_sum([5e-324])
        with self.assertRaises(FloatingPointError):independent.risk_from_sum(5e-324)

    def test_pair_counter_original_pct_ties_singletons_and_empty(self):
        for row in ([3.,1.,1.,5.],[0.,0.,0.],[-1.],[-1.,0.,1.]):
            x=np.array([row]);mask=np.ones_like(x,bool)
            np.testing.assert_array_equal(independent.original_pct_ranks(x,mask)[0],pd.Series(row).rank(pct=True).to_numpy()-.5)
        np.testing.assert_array_equal(independent.original_pct_ranks(np.array([[2.]]),np.array([[True]])),[[.5]])
        np.testing.assert_array_equal(independent.original_pct_ranks(np.array([[np.nan]]),np.array([[False]])),[[np.nan]])


class BuildTests(unittest.TestCase):
    def test_dual_window_dates_support_and_truthful_beta_counts(self):
        p,m,c=fixture();before=p['adj'].copy();v,a=independent.independent_features(p,m,c)
        r=independent.verify_features(p,m,c,v,a)
        self.assertEqual(a['source_available_by_event'],[6]*4);self.assertEqual(a['available_by_event'],[4]*4)
        self.assertEqual(a['descriptive_beta_fits'],24)
        for key,expected in (('source_available_by_event',[6]*4),
                             ('available_by_event',[4]*4),('descriptive_beta_fits_by_event',[6]*4)):
            self.assertEqual(r[key],expected)
            self.assertEqual(r[key],a[key])
            self.assertTrue(all(type(value) is int for value in r[key]))
        self.assertEqual((a['formation_price_observations'],a['risk_price_observations'],a['valid_price_observations'],a['simple_returns']),
                         (253*24,128*24,381*24,126*24))
        self.assertEqual((a['normalized_momentum_scores'],a['ranked_observations']),(32,32))
        for event in a['event_history']:
            t=c.index(event['date'])
            self.assertEqual((event['formation_price_start'],event['formation_price_end']),(c[t-294],c[t-42]))
            self.assertEqual((event['risk_price_start'],event['first_risk_return_date'],event['risk_price_end']),
                             (c[t-63],c[t-62],c[t]))
            self.assertEqual(event['cells'][3]['reason'],'zero_idiosyncratic_risk')
            self.assertEqual(event['cells'][4]['reason'],'zero_total_risk')
        self.assertTrue(np.all(v['beta'][:,3]==1.));self.assertTrue(np.all(v['idiosyncratic_risk'][:,3]==0.))
        self.assertTrue(all(r[k] for k in independent.FEATURE_FLAGS));self.assertEqual(r['numerical_tolerance'],0.)
        pd.testing.assert_frame_equal(p['adj'],before)

    def test_invalid_source_window_or_market_removes_both_domains(self):
        for offset,symbol,expected in ((-200,'A',5),(-1,'A',5),(-1,'SPY',0),(-200,'SPY',5)):
            p,m,c=fixture(1);t=c.index(m['dates'][0]);p['adj'].loc[c[t+offset],symbol]=np.nan
            v,a=independent.independent_features(p,m,c)
            self.assertEqual(a['source_available_cells'],expected)
            independent.verify_features(p,m,c,v,a)
        p,m,c=fixture(1);p['adj']=p['adj'].drop(columns='SPY');v,a=independent.independent_features(p,m,c)
        self.assertEqual(a['reason_counts'],{'missing_market_symbol':5,'missing_symbol':1})
        self.assertTrue(np.isnan(v['momentum']).all())

    def test_ranks_common_before_eligibility_and_zero_market_mask(self):
        p,m,c=fixture(1);p['adj']['E']=p['adj']['A'];v,a=independent.independent_features(p,m,c)
        for raw,key in (('raw_candidate','candidate'),('raw_control','control')):
            good=v['available'][0]
            np.testing.assert_array_equal(v[key][0,good],pd.Series(v[raw][0,good]).rank(pct=True).to_numpy()-.5)
            self.assertEqual(v[key][0,0],v[key][0,5])
        p['adj']['SPY']=9.;v,a=independent.independent_features(p,m,c)
        self.assertEqual(a['source_available_cells'],6);self.assertEqual(a['available_cells'],0)
        self.assertEqual(a['descriptive_beta_fits'],0)
        self.assertTrue(np.isfinite(v['total_risk']).all());self.assertTrue(np.isnan(v['beta']).all())
        independent.verify_features(p,m,c,v,a)

    def test_logical_and_physical_prefixes_ignore_after_signal_only(self):
        args=fixture();p,m,c=args;v,a=independent.independent_features(p,m,c)
        for n in range(len(m['dates'])+1):
            stop=pd.Timestamp.min if not n else pd.Timestamp(m['dates'][n-1])
            q={'adj':p['adj'].astype(object).copy()};q['adj'].loc[q['adj'].index>stop,:]='future poison'
            vv,aa=independent.independent_features(q,m,c,n)
            independent.verify_features(q,m,c,vv,aa,n)
            for key in v:np.testing.assert_array_equal(vv[key],v[key][:n])
            self.assertEqual(aa['event_history'],a['event_history'][:n])
            qq,mm,cc=physical_prefix(args,n);vvv,aaa=independent.independent_features(qq,mm,cc)
            for key in vvv:
                if n:np.testing.assert_array_equal(vvv[key],v[key][:n])
                else:self.assertEqual(vvv[key].shape,(0,0))
            self.assertEqual(aaa['event_history'],a['event_history'][:n])

    def test_recent42_changes_risk_but_not_numerator_and_gap_not_invented(self):
        p,m,c=fixture(1);v,a=independent.independent_features(p,m,c);t=c.index(m['dates'][0])
        q=copy.deepcopy(p);q['adj'].loc[c[t],'A']*=1.02
        vv,aa=independent.independent_features(q,m,c)
        self.assertEqual(vv['momentum'][0,0],v['momentum'][0,0])
        self.assertNotEqual(vv['total_risk'][0,0],v['total_risk'][0,0])
        self.assertNotEqual(vv['idiosyncratic_risk'][0,0],v['idiosyncratic_risk'][0,0])
        q=copy.deepcopy(p);q['adj']=q['adj'].drop(pd.Timestamp(c[t-10]));_,a=independent.independent_features(q,m,c)
        self.assertEqual(a['reason_counts'],{'missing_risk_session':6})
        q=copy.deepcopy(p);q['adj']=q['adj'].drop(pd.Timestamp(c[t-50]));_,a=independent.independent_features(q,m,c)
        self.assertEqual(a['reason_counts'],{'missing_formation_session':6})

    def test_metadata_validation_and_insufficient_history(self):
        p,m,c=fixture()
        for limit in (-1,5,True,1.5):
            with self.assertRaises(ValueError):independent.independent_features(p,m,c,limit)
        bad=copy.deepcopy(m);bad['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError):independent.independent_features(p,bad,c)
        with self.assertRaises(ValueError):independent.independent_features(p,m,c[:c.index(m['dates'][-1])+1])
        bad=dict(m);bad.pop('etfs')
        with self.assertRaises(ValueError):independent.independent_features(p,bad,c)
        bad=copy.deepcopy(m);bad['symbols'][1][1]=bad['symbols'][1][0]
        with self.assertRaises(ValueError):independent.independent_features(p,bad,c)
        with self.assertRaises(ValueError):independent.independent_features(p,m,c+[c[-1]])
        _,a=independent.independent_features(p,m,[x for x in c if x>='2016-01-01'])
        self.assertEqual(a['reason_counts'],{'insufficient_history':24})

    def test_exact_values_masks_count_types_and_audit_tampering_rejected(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for kind in ('momentum','beta','idiosyncratic_risk','total_risk','raw_candidate','raw_control','candidate','control',
                     'mask','date','count','type','beta_fit','fit_series','rss'):
            vv,aa=copy.deepcopy(v),copy.deepcopy(a)
            if kind in independent.VALUE_KEYS and kind!='available':vv[kind][0,0]=np.nextafter(vv[kind][0,0],math.inf)
            elif kind=='mask':vv['available'][0,0]=False
            elif kind=='date':aa['event_history'][0]['risk_price_end']=aa['event_history'][0]['risk_price_start']
            elif kind=='count':aa['descriptive_beta_fits']+=1
            elif kind=='type':aa['simple_returns']=float(aa['simple_returns'])
            elif kind=='beta_fit':aa['event_history'][0]['cells'][0]['descriptive_beta_fit']=0
            elif kind=='fit_series':aa['descriptive_beta_fits_by_event'][0]-=1
            else:aa['event_history'][0]['cells'][0]['residual_sum_squares']*=2.
            with self.assertRaises(ValueError,msg=kind):independent.verify_features(p,m,c,vv,aa)
        for key in ('beta','idiosyncratic_risk'):
            vv=copy.deepcopy(v);vv[key][0,0]=np.nan
            with self.assertRaises(ValueError):independent.feature_arrays(vv)


class IntegrationTests(unittest.TestCase):
    def test_half_core_calm_initial_missing_and_ineligible_cells(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c);o=original_fixture();before=copy.deepcopy(o)
        ts=targets(v,o);d=independent.independent_deltas(v,o,ts);r=independent.verify_deltas(v,o,ts,d)
        self.assertEqual(r['max_absolute_delta'],.26)
        for arm in independent.RAW:
            np.testing.assert_array_equal(d[arm][o['state_calm10']],0.)
            np.testing.assert_array_equal(d[arm][0],0.)
            np.testing.assert_array_equal(d[arm][:,3:5],0.)
            mask=np.isfinite(ts[arm]);cw=np.broadcast_to(o['core_weight'],o['core'].shape)
            np.testing.assert_array_equal(d[arm][mask],.5*(cw[mask]*ts[arm][mask]-o['core'][mask]))
            self.assertTrue(np.any(d[arm]!=0))
        for key in o:np.testing.assert_array_equal(o[key],before[key])

    def test_all_weight_shapes_prefix_poison_and_no_second_rank(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c);ref=None
        for shape in ('vector','column','matrix'):
            o=original_fixture(shape=shape);d=independent.independent_deltas(v,o,targets(v,o))
            if ref is not None:
                for arm in d:np.testing.assert_array_equal(d[arm],ref[arm])
            ref=d
            for n in range(5):
                vv={k:x[:n] for k,x in v.items()};oo=copy.deepcopy(o)
                for key in ('core','core_weight','state_calm10'):
                    oo[key]=oo[key].astype(object);oo[key][n:]='future unread'
                ts=targets(vv,oo,4);dd=independent.independent_deltas(vv,oo,ts)
                independent.verify_deltas(vv,oo,ts,dd)
                for arm in dd:np.testing.assert_array_equal(dd[arm],ref[arm][:n])
        o=original_fixture();ts=targets(v,o)
        for arm in ts:
            bad=copy.deepcopy(ts);bad[arm][2]=0.
            with self.assertRaises(ValueError):independent.independent_deltas(v,o,bad)
        d=independent.independent_deltas(v,o,ts);d[independent.ARMS[1]][1,1]+=1e-15
        with self.assertRaises(ValueError):independent.verify_deltas(v,o,ts,d)

    def test_invalid_core_blend_and_nonempty_suffix_are_rejected(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c);o=original_fixture();ts=targets(v,o)
        for key,value in [('core',np.inf),('core_weight',np.nan),('core',True),('core_weight','0.52')]:
            bad=copy.deepcopy(o);bad[key]=bad[key].astype(object);bad[key][1,0]=value
            with self.assertRaises(ValueError):independent.independent_deltas(v,bad,ts)
        for blend in (0.,1.,True):
            with self.assertRaises(ValueError):independent.independent_deltas(v,o,ts,blend)
        bad=copy.deepcopy(o);bad['state_calm10']=np.ones(4,int)
        with self.assertRaises(ValueError):independent.independent_deltas(v,bad,ts)
        vv={k:x[:1] for k,x in v.items()};tt=targets(vv,o,4);tt[independent.ARMS[1]][2,0]=0.
        with self.assertRaises(ValueError):independent.independent_deltas(vv,o,tt)

    def test_accounting_is_frozen_independent_and_flags_are_explicit(self):
        self.assertEqual(hashlib.sha256(independent.FROZEN_HELPER.read_bytes()).hexdigest(),independent.FROZEN_HELPER_SHA)
        self.assertTrue(callable(independent.accounting().audit_path))
        self.assertNotIn('import monthly_idio_risk222_kernel',inspect.getsource(independent))
        self.assertEqual((len(independent.FEATURE_FLAGS),len(independent.DELTA_FLAGS),len(independent.COUNT_FIELDS)),(20,9,11))
        with patch.object(independent,'FROZEN_HELPER_SHA','0'*64):
            with self.assertRaises(ValueError):independent.accounting()

    def test_unique_blackbox_both_fixtures_all_logical_and_physical_prefixes(self):
        import monthly_idio_risk222_kernel as producer
        import test_monthly_idio_risk222_kernel as producer_tests
        for args in (fixture(),producer_tests.fixture()):
            p,m,c=args;full,full_audit=producer.build_features(p,m,c)
            for n in range(len(m['dates'])+1):
                pp,mm,cc=physical_prefix(args,n)
                v,a=producer.build_features(pp,mm,cc);independent.verify_features(pp,mm,cc,v,a)
                for key in v:
                    if n:np.testing.assert_array_equal(v[key],full[key][:n])
                    else:self.assertEqual(v[key].shape,(0,0))
                self.assertEqual(a['event_history'],full_audit['event_history'][:n])
                vv,aa=producer.build_features(pp,m,c,limit=n)
                independent.verify_features(pp,m,c,vv,aa,limit=n)
                o=original_fixture(len(m['dates']),len(m['symbols'][0]))
                ts=targets(vv,o,len(m['dates']))
                independent.verify_deltas(vv,o,ts,independent.independent_deltas(vv,o,ts))
        p,m,c=fixture(1)
        for key in ('A','SPY'):
            for bad in (0.,np.nan,'bad'):
                q={'adj':p['adj'].astype(object).copy()};q['adj'].loc[m['dates'][0],key]=bad
                v,a=producer.build_features(q,m,c);independent.verify_features(q,m,c,v,a)


if __name__=='__main__':unittest.main(verbosity=2)

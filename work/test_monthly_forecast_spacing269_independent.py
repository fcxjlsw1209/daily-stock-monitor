"""Synthetic independent checks; no saved market data or returns."""
import copy
import unittest
import numpy as np
import pandas as pd
import monthly_forecast_spacing269_kernel as producer
import monthly_forecast_spacing269_independent as independent


def case(n=4,w=7):
    rng=np.random.RandomState(269)
    pre=rng.normal(size=(n,w)).astype('float64'); pool=rng.normal(size=(n,w)).astype('float64');local=rng.normal(size=(n,w)).astype('float64')
    use=np.ones((n,w),bool);ready=np.ones(n,bool);weak=np.ones(n,bool)
    rank=lambda x:np.asarray([pd.Series(row).rank(pct=True,method='average').values-.5 for row in x],dtype='float64').reshape(n,w)
    dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31'][:n]
    return [pre,pool,rank(pool),local,rank(local),use,ready,weak,dates]


def produced(args,mode='mixed_affine'):
    return producer.transform(*args,mode)


class VerificationTests(unittest.TestCase):
    def verify(self,args=None,mode='mixed_affine'):
        args=case() if args is None else args
        output,audit=produced(args,mode)
        return independent.verify_transform(*args,mode,output,audit)

    def reject(self,mutate,mode='mixed_affine'):
        args=case();output,audit=produced(args,mode);mutate(args,output,audit)
        with self.assertRaises((ValueError,TypeError,IndexError)):independent.verify_transform(*args,mode,output,audit)

    def test_all_arms(self):
        for mode in independent.ARMS:self.assertTrue(self.verify(mode=mode)['passed'])
    def test_empty_prefix(self):
        a=case(0);self.assertEqual(self.verify(a)['work']['event_rows'],0)
    def test_one_partial(self):
        self.assertEqual(self.verify(case(1))['work']['active_rows'],0)
    def test_scalar_manual_spacing(self):
        a=case(1,3);a[8]=['2016-08-31'];a[0][:]=0.;a[1][:]=[-1.,0.,3.];a[3][:]=[-1.,0.,3.]
        for k in [2,4]:a[k][:]=pd.Series([-1.,0.,3.]).rank(pct=True).values-.5
        p,audit=produced(a)
        lo=a[2][0,0];hi=a[2][0,2];target=((0.-(-1.))/(3.-(-1.)))*(hi-lo)+lo
        expected=.0375*(target-a[2][0,1])+.0125*(target-a[4][0,1])
        self.assertEqual(p[0,1].tobytes(),np.float64(expected).tobytes())
        self.assertEqual(p[0,0],0.);self.assertEqual(p[0,2],0.);self.assertTrue(self.verify(a)['passed'])
    def test_ties(self):
        a=case();a[1][:]=[1.,1.,2.,2.,3.,4.,5.];a[2][:]=pd.Series(a[1][0]).rank(pct=True).values-.5;self.verify(a)
    def test_constant(self):
        a=case();a[1][:]=1.;a[2][:]=pd.Series(a[1][0]).rank(pct=True).values-.5
        r=self.verify(a,'pooled_affine');self.assertEqual(r['work']['affine_evaluations'],0)
    def test_two_levels_identity(self):
        a=case();a[1][:]=[1.,1.,1.,1.,2.,2.,2.];a[2][:]=pd.Series(a[1][0]).rank(pct=True).values-.5
        p,_=produced(a,'pooled_affine');self.assertEqual(p.tobytes(),a[0].tobytes());self.verify(a,'pooled_affine')
    def test_whole_row_fallback(self):
        a=case();a[6][:]=False;a[1][:]=np.inf;a[2][:]=np.nan;a[3][:]=-np.inf;a[4][:]=np.nan
        self.assertEqual(self.verify(a)['work']['active_rows'],0)
    def test_nonweak(self):
        a=case();a[7][:]=False;a[1][:]=np.inf;self.verify(a)
    def test_pool_only_does_not_read_local(self):
        a=case();a[3][:]=np.inf;a[4][:]=np.nan;self.verify(a,'pooled_affine')
    def test_outside_use_preserved(self):
        a=case();a[5][:,-1]=False;a[0][:,-1]=-0.;a[1][:,-1]=np.nan;a[3][:,-1]=np.inf
        for raw,r in [(1,2),(3,4)]:
            for i in range(len(a[0])):a[r][i,:-1]=pd.Series(a[raw][i,:-1]).rank(pct=True).values-.5
        self.verify(a)
    def test_single_use(self):
        a=case();a[5][:]=False;a[5][:,0]=True;a[2][:,0]=.5;a[4][:,0]=.5;self.verify(a)
    def test_empty_use(self):
        a=case();a[5][:]=False;a[1][:]=np.nan;self.verify(a)
    def test_output_corruption(self):self.reject(lambda a,p,d:p.__setitem__((1,0),p[1,0]+.01))
    def test_nonconsumed_signed_zero_corruption(self):
        a=case();a[0][0,0]=-0.;p,d=produced(a);p[0,0]=0.
        with self.assertRaises(ValueError):independent.verify_transform(*a,'mixed_affine',p,d)
    def test_mutated_count(self):self.reject(lambda a,p,d:d['counts'].__setitem__('active_cells',0))
    def test_bool_count_rejected(self):self.reject(lambda a,p,d:d['counts'].__setitem__('transform_calls',True))
    def test_missing_audit_field(self):self.reject(lambda a,p,d:d.pop('no_new_budget'))
    def test_extra_audit_field(self):self.reject(lambda a,p,d:d.__setitem__('extra',True))
    def test_mutated_budget_delta(self):self.reject(lambda a,p,d:d['events'][1]['delta'].__setitem__(0,.1))
    def test_mutated_rank_endpoint(self):self.reject(lambda a,p,d:d['events'][1]['pooled'].__setitem__('rank_maximum',.49))
    def test_wrong_branch(self):self.reject(lambda a,p,d:d['events'][1].__setitem__('branch','not_ready'))
    def test_wrong_raw_forecast_audit(self):self.reject(lambda a,p,d:d['events'][1]['local']['raw'].__setitem__(0,999.))
    def test_wrong_map_level_count(self):self.reject(lambda a,p,d:d['events'][1]['pooled'].__setitem__('distinct_levels',2))
    def test_rank_source_mutation(self):self.reject(lambda a,p,d:a[2].__setitem__((1,0),.01))
    def test_raw_source_mutation(self):self.reject(lambda a,p,d:a[1].__setitem__((1,0),float('inf')))
    def test_pre_source_mutation(self):self.reject(lambda a,p,d:a[0].__setitem__((1,0),float('nan')))
    def test_nonfinite_pre_without_delta(self):
        a=case();a[1][:]=1.;a[2][:]=pd.Series(a[1][0]).rank(pct=True).values-.5;p,d=produced(a,'pooled_affine');a[0][1,0]=float('nan')
        with self.assertRaises(ValueError):independent.verify_transform(*a,'pooled_affine',p,d)
    def test_partial_date_at_wrong_row(self):
        self.reject(lambda a,p,d:a[8].__setitem__(1,'2016-08-17'))
    def test_shape_failure(self):
        a=case();p,d=produced(a);a[1]=a[1][:,:-1]
        with self.assertRaises(ValueError):independent.verify_transform(*a,'mixed_affine',p,d)
    def test_mode_failure(self):
        a=case();p,d=produced(a)
        with self.assertRaises(ValueError):independent.verify_transform(*a,'guessed',p,d)
    def test_all_prefixes_match(self):
        a=case();whole,_=produced(a)
        for n in range(5):
            x=[v[:n].copy() if isinstance(v,np.ndarray) else v[:n] for v in a];p,d=produced(x)
            self.assertEqual(p.tobytes(),whole[:n].tobytes());self.verify(x)
    def test_scale_shift_invariance(self):
        a=case();a[1][:]=[-2.,-1.,0.,1.,3.,5.,9.];a[2][:]=pd.Series(a[1][0]).rank(pct=True).values-.5
        p,_=produced(a,'pooled_affine');a[1]=a[1]*2.+4.;q,_=produced(a,'pooled_affine')
        self.assertEqual(p.tobytes(),q.tobytes());self.verify(a,'pooled_affine')
    def test_seeded_sparse_consumption(self):
        rng=np.random.RandomState(404)
        for _ in range(8):
            a=case();a[5]=rng.rand(*a[5].shape)>.4;a[6]=rng.rand(4)>.3;a[7]=rng.rand(4)>.3
            for raw,r in [(1,2),(3,4)]:
                for i in range(4):
                    use=a[5][i];a[r][i,use]=pd.Series(a[raw][i,use]).rank(pct=True).values-.5
            self.verify(a)

if __name__=='__main__':unittest.main(verbosity=2)

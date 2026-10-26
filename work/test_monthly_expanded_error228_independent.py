"""Synthetic independent risk-only adapter tests; no historical inputs."""
import copy
import importlib
import unittest
import numpy as np
import monthly_expanded_error228_independent as audit


def fixture():
    tm={'dates':['2019-01-31','2019-02-28'],'ends':['2019-02-28','2019-03-29'],
        'symbols':[['A','B','C'],['C','A','B']]}
    qm={'dates':['2020-01-31','2020-02-28','2020-03-31'],
        'ends':['2020-02-28','2020-03-31','2020-04-30'],
        'symbols':[['A','B','C'],['C','A','B'],['B','C','A']],'etfs':['C']}
    good=np.array([1,1,0,1,1,0],dtype=bool)
    train={'offsets':np.array([0,3,6],dtype=np.int64),
        'dates':np.repeat(np.array(tm['dates'],dtype='datetime64[D]'),3),
        'ends':np.repeat(np.array(tm['ends'],dtype='datetime64[D]'),3),
        'train_plain':np.arange(60,dtype=float).reshape(6,10)/200,
        'common_good':good.copy(),'train_eligible':good.copy(),
        'vol63':np.linspace(.1,.2,6),'untouched_aux':np.arange(6,dtype=np.int16)}
    for i,k in enumerate(audit.TARGETS):
        train['target_'+k]=np.where(good,np.arange(6,dtype=float)/20+i/10,np.nan)
    olda=np.array([1,1,0,1,1,0,1,1,0],bool);newa=olda.copy();newa[[2,5]]=True
    old={'available':olda.copy(),'local_available':olda.copy()}
    for i,k in enumerate((*audit.RAW,'candidate','control')):
        old['raw_'+k]=np.where(olda,np.arange(9,dtype=float)/100+i/10,np.nan)
    old['raw_local_linear'][0]=-0.
    new={'available':newa.copy(),'local_available':newa.copy(),'expansion_allowed':np.ones(9,bool)}
    for i,k in enumerate(audit.RAW):
        new['raw_'+k]=old['raw_'+k].copy();new['raw_'+k][[2,5]]=[.13+i,.17+i]
    added=np.array([0,0,1,0,0,0],bool)
    labels={'common_good':good|added,'added':added,'vol63':np.linspace(.2,.3,6),
        'dates':train['dates'].copy(),'ends':train['ends'].copy(),'offsets':train['offsets'].copy()}
    for i,k in enumerate(audit.TARGETS):
        labels['target_'+k]=train['target_'+k].copy();labels['target_'+k][2]=.33+i/10
    use=np.zeros((3,3),bool);use[:,0]=True
    return (train,tm,qm,old,new,labels,use),{'training':copy.deepcopy(tm),'monthly':copy.deepcopy(qm)}


def prefix(args,meta,n):
    args=copy.deepcopy(args);meta=copy.deepcopy(meta)
    train,tm,qm,old,new,labels,use=args;q=n*use.shape[1]
    for k in ('dates','ends','symbols'):qm[k]=qm[k][:n]
    for source in (old,new):
        for k in source:source[k]=source[k][:q].copy()
    use=use[:n].copy()
    if n==0:
        tm={k:v[:0] for k,v in tm.items()}
        train={k:v[:0].copy() for k,v in train.items()};train['offsets']=np.array([0],np.int64)
        labels={k:v[:0].copy() for k,v in labels.items()};labels['offsets']=np.array([0],np.int64)
    meta={'training':copy.deepcopy(tm),'monthly':copy.deepcopy(qm)}
    return (train,tm,qm,old,new,labels,use),meta


class OracleTests(unittest.TestCase):
    def setUp(self):self.args,self.meta=fixture()
    def result(self):return audit.rebuild(*self.args,source_meta227=self.meta)
    def check(self,result=None):
        return audit.check_adapter(*self.args,self.result() if result is None else result,source_meta227=self.meta)
    def test_expands_only_new_available_and_keeps_old_excluded(self):
        out=self.result();self.assertEqual(out['historical_use'].tolist(),[[True,False,True],[True,False,True],[True,False,False]])
        self.assertEqual(out['audit']['counts'],dict(training_rows=6,query_rows=9,original_forecast_rows=6,
            expanded_forecast_rows=8,added_forecast_rows=2,original_label_rows=4,added_label_rows=1,
            extended_label_rows=5,original_history_rows=3,expanded_history_rows=5))
        self.assertTrue(self.check(out)['passed'])
    def test_training_only_five_fields_change(self):
        out=self.result()
        for k in self.args[0]:
            if k not in ['common_good']+['target_'+v for v in audit.TARGETS]:audit.exact(out['train'][k],self.args[0][k],k)
        self.assertEqual(out['train']['target_log_excess'][2],self.args[5]['target_log_excess'][2])
    def test_signed_zero_and_nan_bytes_preserved(self):
        out=self.result();self.assertTrue(np.signbit(out['source']['raw_local_linear'][0]))
        self.check(out)
        self.args[4]['raw_local_linear'][0]=0.
        with self.assertRaises(ValueError):self.result()
    def test_no_prediction_or_target_arithmetic(self):
        out=self.result();check=self.check(out)
        for k in ('computes_predictions','computes_labels','computes_risk_estimates','computes_returns','producer_modules_imported'):
            self.assertIs(check[k],False)
        self.assertEqual(check['new_model_fits'],0)
    def test_mutation_scope_and_deep_detachment(self):
        before=copy.deepcopy(self.args);out=self.result();self.check(out);audit.exact(self.args,before,'inputs')
        for value in audit.arrays_in(out):
            self.assertTrue(all(not np.shares_memory(value,x) for x in audit.arrays_in(self.args)))
    def test_aliasing_labels_is_rejected(self):
        out=self.result();out['train']['target_log_excess']=self.args[5]['target_log_excess']
        with self.assertRaises(ValueError):self.check(out)
    def test_aliasing_original_aux_is_rejected(self):
        out=self.result();out['train']['untouched_aux']=self.args[0]['untouched_aux']
        with self.assertRaises(ValueError):self.check(out)
    def test_train_eligibility_must_remain_unchanged(self):
        out=self.result();out['train']['train_eligible'][2]=True
        with self.assertRaises(ValueError):self.check(out)
    def test_old_available_may_not_disappear(self):
        self.args[4]['available'][0]=False
        with self.assertRaises(ValueError):self.result()
    def test_added_forecast_requires_expansion_mask(self):
        self.args[4]['expansion_allowed'][2]=False
        with self.assertRaises(ValueError):self.result()
    def test_original_use_not_availability_replacement(self):
        out=self.result();out['historical_use']=self.args[4]['available'].reshape(3,3).copy()
        with self.assertRaises(ValueError):self.check(out)
    def test_original_use_must_lie_in_old_source(self):
        self.args[6][0,2]=True
        with self.assertRaises(ValueError):self.result()
    def test_original_eligible_bad_label_not_revived(self):
        self.args[0]['train_eligible'][2]=True
        with self.assertRaises(ValueError):self.result()
    def test_original_target_offmask_change_rejected(self):
        self.args[5]['target_simple_excess'][5]=0.
        with self.assertRaises(ValueError):self.result()
    def test_nonfinite_added_target_or_plain_rejected(self):
        for kind in ('target','plain'):
            self.args,self.meta=fixture()
            if kind=='target':self.args[5]['target_log_excess'][2]=np.nan
            else:self.args[0]['train_plain'][2,3]=np.nan
            with self.assertRaises(ValueError):self.result()
    def test_nonfinite_source_and_local_mask(self):
        for kind in ('raw','local'):
            self.args,self.meta=fixture()
            if kind=='raw':self.args[4]['raw_source_log'][2]=np.inf
            else:self.args[4]['local_available'][8]=True
            with self.assertRaises(ValueError):self.result()
    def test_identity_metadata_and_offsets_rejected(self):
        for kind in ('meta','end','offset','duplicate'):
            self.args,self.meta=fixture()
            if kind=='meta':self.meta['monthly']['symbols'][0][0]='OTHER'
            elif kind=='end':self.args[5]['ends'][0]=np.datetime64('2019-03-01')
            elif kind=='offset':self.args[0]['offsets'][1]=2
            else:self.args[1]['symbols'][0][1]='A';self.meta['training']=copy.deepcopy(self.args[1])
            with self.assertRaises(ValueError):self.result()
    def test_exact_date_format_and_initial_partial(self):
        self.args[2]['dates'][0]='2020-1-31';self.meta['monthly']=copy.deepcopy(self.args[2])
        with self.assertRaises(ValueError):self.result()
        self.args,self.meta=fixture();self.args[2]['dates'][0]='2016-08-17';self.args[2]['ends'][0]='2016-08-31'
        self.meta['monthly']=copy.deepcopy(self.args[2]);self.check()
    def test_dtype_and_shape_not_just_equal_values(self):
        out=self.result();out['train']['untouched_aux']=out['train']['untouched_aux'].astype(np.int64)
        with self.assertRaises(ValueError):self.check(out)
        self.args[5]['offsets']=self.args[5]['offsets'].astype(np.int32)
        with self.assertRaises(ValueError):self.result()
    def test_schema_rejects_unexpected_old_or_expanded_fields(self):
        for index in (3,4,5):
            self.args,self.meta=fixture();self.args[index]['extra']=np.zeros(1)
            with self.assertRaises(ValueError):self.result()
    def test_all_physical_prefixes(self):
        full=self.result()
        for n in range(4):
            args,meta=prefix(self.args,self.meta,n);out=audit.rebuild(*args,source_meta227=meta)
            self.assertTrue(audit.check_adapter(*args,out,source_meta227=meta)['passed'])
            audit.exact(out['source']['raw_local_linear'],full['source']['raw_local_linear'][:n*3],'prefix')
            audit.exact(out['historical_use'],full['historical_use'][:n],'prefix use')
    def test_no_expansion_exact_identity(self):
        old,new,labels=self.args[3:6]
        new['available']=old['available'].copy();new['local_available']=old['local_available'].copy()
        for k in audit.RAW:new['raw_'+k]=old['raw_'+k].copy()
        labels['common_good']=self.args[0]['common_good'].copy();labels['added'][:]=False
        for k in audit.TARGETS:labels['target_'+k]=self.args[0]['target_'+k].copy()
        out=self.result();self.check(out)
        audit.exact(out['train'],self.args[0],'all training');audit.exact(out['historical_use'],self.args[6],'all U')
    def test_unused_expanded_volatility_not_injected(self):
        self.args[5]['vol63'][:]=999.
        out=self.result();audit.exact(out['train']['vol63'],self.args[0]['vol63'],'old vol')
        self.check(out)
    def test_receipt_rejects_every_missing_or_false_flag(self):
        out=self.result();good=self.check(out)
        for flag in audit.FLAGS:
            for v in (None,False):
                bad=copy.deepcopy(good)
                if v is None:bad.pop(flag)
                else:bad[flag]=v
                with self.assertRaises(ValueError):audit.require_receipt(bad,out['audit'])
    def test_receipt_counts_and_semantics_checked(self):
        out=self.result();good=self.check(out)
        bad=copy.deepcopy(good);bad['counts']['added_forecast_rows']+=1
        with self.assertRaises(ValueError):audit.require_receipt(bad,out['audit'])
        for k,v in (('historical_use_rule','expanded_available'),('train_output_role','Ridge_training'),('source_forecasts_refitted',True)):
            bad=copy.deepcopy(out['audit']);bad[k]=v
            with self.assertRaises(ValueError):audit.require_receipt(good,bad)
    def test_receipt_scope_not_fabricated(self):
        out=self.result();good=self.check(out)
        for k,v in (('new_model_fits',1),('computes_labels',True),('mutation_scope','producer inputs unchanged')):
            bad=copy.deepcopy(good);bad[k]=v
            with self.assertRaises(ValueError):audit.require_receipt(bad,out['audit'])


class BilateralTests(unittest.TestCase):
    def test_both_independent_fixtures_and_all_prefixes(self):
        producer=importlib.import_module('monthly_expanded_error228_adapter')
        examples=importlib.import_module('test_monthly_expanded_error228_adapter')
        for make in (fixture,examples.fixture):
            args,meta=make()
            n=args[6].shape[0]
            for count in range(n+1):
                # Use independent physical slicing for both fixtures.
                pa,pm=prefix(args,meta,count)
                before=copy.deepcopy((pa,pm))
                out=producer.adapt(*pa,source_meta227=pm)
                audit.exact((pa,pm),before,'Producer inputs unchanged')
                self.assertTrue(audit.check_adapter(*pa,out,source_meta227=pm)['passed'])


if __name__=='__main__':unittest.main()

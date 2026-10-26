"""Fabricated saved-artifact tests only; no prices, model imports or actual209 reads."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import relative_path209_close_review as r


def fixture(month_count=20):
    axis=np.arange('2012-12-01','2016-02-01',dtype='datetime64[D]');axis=axis[np.is_busday(axis)]
    months=np.unique(axis.astype('datetime64[M]'));ends={m:axis[axis.astype('datetime64[M]')==m][-1] for m in months}
    starts=months[1:month_count+1]
    dates=np.repeat([ends[m] for m in starts],2).astype('datetime64[D]');maturity=np.repeat([ends[m+1] for m in starts],2).astype('datetime64[D]')
    n=len(dates);labels=dict(dates=dates,ends=maturity,symbols=np.tile(np.array(['A','B']),month_count),candidate_target=np.tile([.1,.4],month_count),control_target=np.tile([.2,.3],month_count),base_good=np.ones(n,bool),good=np.ones(n,bool))
    audits=[];checks=[];obs=prices=0
    for day in np.unique(dates):
        ids=np.flatnonzero(dates==day);end=maturity[ids[0]];window=axis[(axis>=day)&(axis<=end)];L=len(window)-1
        audits.append(dict(signal_date=str(day),label_end=str(end),base_rows=2,endpoint_rows=2,good_rows=2,status='complete',complete_path_rows=2,source_dates=window.astype(str).tolist(),target_dates=window[1:].astype(str).tolist(),target_session_count=L))
        for i in ids:
            checks.append(dict(row_index=int(i),signal_date=str(day),label_end=str(end),symbol=str(labels['symbols'][i]),base_good=True,path_complete=True,target_session_count=L,status='complete',missing_official_sessions=[],invalid_stock_dates=[],invalid_spy_dates=[]))
            obs+=L;prices+=2*(L+1)
    audit=dict(targets=list(r.TARGETS),rows=n,base_rows=n,good_rows=n,months=audits,clipped=False,centered=False,relative_target_definitions=dict(candidate='mean_d max(0,1-R_d)',control='max(0,1-R_end)'),maturity='next official calendar month-end close',calendar_completeness_certified_by_this_function=False,path_checks=checks,source_path_rows=n,source_price_observations=prices,target_session_observations=obs,missing_path_rows=0)
    for k in ('exact_binary_float_rational','rounded_targets_once','complete_positive_finite_paths','target_bounds_independent','inherited_common_good_retained','no_endpoint_shortening'):audit[k]=True
    return labels,audit,axis


def forecast_fixture():
    mask=np.array([[False]*4,[True]*4,[True,False,True,True]])
    a=np.array([[np.nan]*4,[.1,.1,.4,.3],[.2,np.nan,.2,.2]])
    b=np.array([[np.nan]*4,[.1,.2,.3,.4],[.4,np.nan,.1,.4]])
    values=dict(candidate_prediction=a,control_prediction=b,common_available=mask)
    transforms={};changed={}
    for arm,short,pred in zip(r.ARMS[1:],('candidate','control'),(a,b)):
        ranks=np.zeros(mask.shape)
        for i in range(len(mask)):
            v=-pred[i,mask[i]]
            if len(v)>1:ranks[i,mask[i]]=[(np.count_nonzero(v<x)+(np.count_nonzero(v==x)-1)/2)/(len(v)-1)-.5 for x in v]
        values[short+'_rank']=ranks;transforms['transform_'+arm]=ranks.copy();transforms['delta_'+arm]=.01*ranks
        changed[short]=int(np.count_nonzero(ranks))
    rank=dict(common_counts=mask.sum(axis=1).tolist(),max_absolute_rank=.5,query_evaluation_ends_not_used_for_labels=True,non_monthend_fallback_indices=[0])
    delta=dict(**{k:True for k in r.DELTA_FLAGS},dose=.01,max_absolute_correction=.005,changed_cells=changed)
    return values,transforms,rank,delta


def model_fixture():
    labels,_,_=fixture();ids=np.arange(len(labels['dates']));values=dict(common_available=np.array([[False,False],[True,True]]));meta=dict(dates=['2015-01-02','2015-01-30'])
    row=dict(year=2015,cutoff='2014-12-31',lookback_exclusive_start='2009-12-31',training_indices=ids.tolist(),training_rows=len(ids),training_months=20,latest_label_end=str(labels['ends'].max()),training_indices_sha256=r.digest(ids),training_dates_sha256=r.digest(labels['dates']),training_symbols_sha256=r.digest(labels['symbols']),status='fit',model_fits=2,prediction_rows=2,shared_rows_weights_scalers=True,weight_sum=float(len(ids)),target_sha256={k:r.digest(labels[k]) for k in r.TARGETS},weighted_target_means={k:float(labels[k].mean()) for k in r.TARGETS})
    for key,func in (('target_positive_counts',lambda y:int(np.count_nonzero(y>0))),('target_sums',lambda y:float(y.sum())),('target_means',lambda y:float(y.mean())),('target_maxima',lambda y:float(y.max()))):row[key]={k:func(labels[k]) for k in r.TARGETS}
    audit=dict(alpha=10.,half_life_months=12.,lookback_years=5,minimum_mature_months=18,training_only_standardization=True,calibrated_probabilities=False,target_clipping=False,target_centering=False,local_model_fits=0,shared_training_mask_sha256=r.digest(labels['good']),annual=[row],new_pooled_model_fits=2)
    return labels,values,audit,meta


def receipt_fixture():
    meta=dict(dates=['2016-08-17']+['2017-01-31']*121);counts=dict(source_path_rows=40,source_price_observations=1800,target_session_observations=860,missing_path_rows=0)
    prefix=[dict(prefix=n,producer_model_fits=0 if n<2 else 2,independent_model_fits=0 if n<2 else 2,training_rows=0 if n==0 else 40,training_months=0 if n==0 else 20,training_label_end_cutoff=None if n==0 else ('2015-12-31' if n==1 else '2016-12-31')) for n in (0,1,53,88,122)]
    shared=dict(original_array_count=27,model_fit_accounting=r.MODEL_FIT_ACCOUNTING,new_model_fits=2,new_predictive_model_fits=2,new_pooled_model_fits=2,independent_diagnostic_model_fits=2,prefix_model_fit_counts=prefix,prefix_diagnostic_model_fits=6,prefix_independent_diagnostic_model_fits=6,**counts)
    check=dict(**{k:True for k in r.BUILD_FLAGS},producer_modules_imported=False,computes_portfolio_returns=False,max_forecast_error=1e-14,new_pooled_model_fits=2,independent_model_fits=2,local_model_fits=0,events=122,symbols_per_event=49,label_arrays_checked=7,forecast_arrays_checked=5,annual_records_checked=11,**counts)
    v=dict(**shared,model=check,prefixes=123,feature_prefixes=[0,1,53,88,122],prefix_prices_truncated=True,prefix_training_inputs_truncated=True,original_orders_exact=True,original_arrays_unchanged=True,original_training_arrays_unchanged=True,source_cohort_exact=True,selection=dict(passed=True,scores_and_choices_checked=True,one_original_episode_checked=True,initial_partial_fallback_checked=True,ordered_choices_checked=366))
    prep=dict(**copy.deepcopy(shared),new_returns_computed=False);close=copy.deepcopy(shared)
    return v,prep,close,dict(new_pooled_model_fits=2),counts,meta


class ReaderTests(unittest.TestCase):
    def test_final_pins_and_no_strategy_imports(self):
        r.final_pins();self.assertEqual((len(r.PREPARED_FILES),len(r.RESULT_FILES)),(13,38))
        import ast
        tree=ast.parse(Path(r.__file__).read_text());imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(x and ('model' in x or 'kernel' in x or 'independent' in x) for x in imports))
        with patch.object(r,'TESTS_SHA','PENDING'):
            with self.assertRaises(ValueError):r.final_pins()

    def test_label_full_paths_and_independent_target_bounds(self):
        labels,audit,axis=fixture();counts=r.label_history(labels,audit,axis)
        self.assertEqual(counts['source_path_rows'],40)
        self.assertTrue(np.any(labels['candidate_target']>labels['control_target']))
        self.assertTrue(np.any(labels['candidate_target']<labels['control_target']))

    def test_label_missing_interior_and_min_two(self):
        labels,audit,axis=fixture(2);labels['candidate_target'][0]=labels['control_target'][0]=np.nan;labels['good'][:2]=False
        row=audit['months'][0];L=row['target_session_count'];row.update(endpoint_rows=1,complete_path_rows=1,good_rows=0,status='fewer_than_two_shared_rows')
        audit['good_rows']=2;audit['source_path_rows']=3;audit['source_price_observations']-=2*(L+1);audit['target_session_observations']-=L;audit['missing_path_rows']=1
        audit['path_checks'][0].update(path_complete=False,status='invalid_stock_price',invalid_stock_dates=[row['target_dates'][3]])
        r.label_history(labels,audit,axis)
        labels['good'][1]=True
        with self.assertRaises(ValueError):r.label_history(labels,audit,axis)

    def test_label_rejects_corrupt_counts_clock_and_status(self):
        for kind in ('counter','clock','status','bounds'):
            labels,audit,axis=fixture(2)
            if kind=='counter':audit['source_price_observations']-=2
            elif kind=='clock':audit['months'][0]['target_dates']=audit['months'][0]['source_dates']
            elif kind=='status':audit['path_checks'][0]['status']='invalid_stock_price'
            else:labels['candidate_target'][0]=1.01
            with self.subTest(kind=kind),self.assertRaises(ValueError):r.label_history(labels,audit,axis)

    def test_calendar_requires_following_session_and_complete_month(self):
        labels,audit,axis=fixture(2);axis=axis[axis<=labels['ends'].max()]
        with self.assertRaises(ValueError):r.label_history(labels,audit,axis)

    def test_annual_cohort_statistics_and_fitcount(self):
        args=model_fixture();self.assertEqual(r.model_history(*args),2)
        for key,value in (('cutoff','2015-12-31'),('training_months',19),('model_fits',0)):
            x=copy.deepcopy(args);x[2]['annual'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r.model_history(*x)

    def test_annual_target_and_mask_corruption(self):
        args=model_fixture();args[2]['annual'][0]['target_sums']['candidate_target']+=.1
        with self.assertRaises(ValueError):r.model_history(*args)
        args=model_fixture();args[2]['shared_training_mask_sha256']='0'*64
        with self.assertRaises(ValueError):r.model_history(*args)

    def test_negative_ranks_ties_initial_and_saved_deltas(self):
        args=forecast_fixture();got=r.array_evidence(*args);self.assertEqual(got['available_cells'],7)
        self.assertTrue((args[0]['candidate_rank'][2]==0).all())
        self.assertEqual(args[0]['candidate_rank'][1,0],args[0]['candidate_rank'][1,1])

    def test_delta_and_rank_and_commonmask_corruption_rejected(self):
        for kind in ('sign','rank','mask','initial'):
            args=list(forecast_fixture())
            if kind=='sign':args[1]['delta_'+r.ARMS[1]]*=-1
            elif kind=='rank':args[0]['candidate_rank'][1,0]=np.nextafter(args[0]['candidate_rank'][1,0],1.)
            elif kind=='mask':args[0]['control_prediction'][1,0]=np.nan
            else:args[0]['common_available'][0,0]=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):r.array_evidence(*args)

    def test_actual_fit_and_prefix_accounting_separated(self):
        args=receipt_fixture();out=r.fit_receipts(*args);self.assertEqual(out['main_paired_annual_fits'],2);self.assertEqual(out['prefix_diagnostic_fits'],6)

    def test_all_independent_flags_required(self):
        for flag in r.BUILD_FLAGS:
            args=receipt_fixture();args[0]['model'][flag]=False
            with self.subTest(flag=flag),self.assertRaises(ValueError):r.fit_receipts(*args)

    def test_prefix_physical_truncation_and_cross_receipt_counts(self):
        for kind in ('price','training','cutoff','count','definition','prefixes'):
            args=receipt_fixture()
            if kind=='price':args[0]['prefix_prices_truncated']=False
            elif kind=='training':args[0]['prefix_training_inputs_truncated']=False
            elif kind=='cutoff':args[0]['prefix_model_fit_counts'][2]['training_label_end_cutoff']='2017-12-31'
            elif kind=='count':args[2]['source_path_rows']+=1
            elif kind=='definition':args[2]['model_fit_accounting']='zero fits'
            else:args[0]['prefixes']=122
            with self.subTest(kind=kind),self.assertRaises(ValueError):r.fit_receipts(*args)

    def test_manifest_omissions_extras_and_corruption(self):
        h,_=r.helpers()
        with tempfile.TemporaryDirectory() as tmp:
            h.ROOT=Path(tmp).resolve();folder=h.ROOT/'saved';folder.mkdir()
            for name in r.PREPARED_FILES:(folder/name).write_text('{}')
            mapping={name:r.sha(folder/name) for name in r.PREPARED_FILES}
            receipt=dict(status='COMPLETE',plan_sha256='a'*64,artifacts_sha256=mapping)
            def check(record):
                path=folder/'COMPLETION.json';path.write_text(json.dumps(record));return h.Evidence().manifest(folder,r.sha(path),'a'*64,r.PREPARED_FILES)
            check(receipt)
            for name in r.PREPARED_FILES:
                bad=copy.deepcopy(receipt);del bad['artifacts_sha256'][name]
                with self.subTest(name=name),self.assertRaises(ValueError):check(bad)
            bad=copy.deepcopy(receipt);bad['artifacts_sha256']['extra.json']='b'*64
            with self.assertRaises(ValueError):check(bad)
            bad=copy.deepcopy(receipt);bad['status']='RUNNING'
            with self.assertRaises(ValueError):check(bad)
            (folder/'LABEL_AUDIT.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):check(receipt)

    def test_exact_array_dtype_bytes_and_nan_payload(self):
        a=np.array([1.,np.nan]);r.exact_array(a,a.copy(),'good')
        with self.assertRaises(ValueError):r.exact_array(a,a.astype(np.float32),'dtype')
        b=a.copy();b.view(np.uint64)[1]+=np.uint64(1)
        with self.assertRaises(ValueError):r.exact_array(a,b,'nanpayload')

    def test_control191_exact_mapping_and_scope(self):
        labels,_,_=fixture(2);values,_,_,_=forecast_fixture()
        orders={r.ARMS[2]:[dict(symbols=['A'])]}
        check=dict(passed=True,control_orders_exact=True,arrays_dtype_shape_bytes_exact=True,incidence191_arrays_loaded=False,prior191_returns_read=False,prior_plan_sha256=r.CONTROL_PLAN_SHA,prior_prepared_completion_sha256=r.CONTROL_PREPARED_SHA,label_keys_checked=['dates','ends','symbols','control_target','base_good','good'],forecast_keys_checked=['control_prediction','control_rank','common_available'])
        class Evidence:
            def verify(self,*args):pass
            def manifest(self,*args):return dict(artifacts_sha256={'ORDERS.json':'fabricated'})
            def load(self,*args):return {'relative_severity_ridge':orders[r.ARMS[2]]}
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'CONTROL',Path(tmp)):
            dest=Path(tmp)/'prepared';dest.mkdir()
            old_labels={k:labels[k] for k in ('dates','ends','symbols','base_good','good')};old_labels['candidate_target']=labels['control_target'].copy()
            old_values=dict(candidate_prediction=values['control_prediction'],candidate_rank=values['control_rank'],common_available=values['common_available'])
            np.savez(dest/'LABELS.npz',**old_labels);np.savez(dest/'FEATURES.npz',**old_values)
            result=r.control_evidence(Evidence(),labels,values,orders,check);self.assertEqual(result['label_arrays_exact'],6)
            old_labels['candidate_target'][0]=np.nextafter(old_labels['candidate_target'][0],1.)
            np.savez(dest/'LABELS.npz',**old_labels)
            with self.assertRaises(ValueError):r.control_evidence(Evidence(),labels,values,orders,check)
            check['prior191_returns_read']=True
            with self.assertRaises(ValueError):r.control_evidence(Evidence(),labels,values,orders,check)

    def test_selection_only_first88_both_references(self):
        metric=dict(cagr=.5,sharpe=1.5,max_drawdown=-.3)
        metrics={a:{p:{k:copy.deepcopy(metric) for k in ('training','early','recent')} for p in ('same_close_zero','next_open_cost')} for a in r.ARMS}
        orders={a:[dict(symbols=['A','B','C']) for _ in range(122)] for a in r.ARMS}
        for p in metrics[r.ARMS[1]]:metrics[r.ARMS[1]][p]['training']['cagr']=.51
        for i in (2,54,55):orders[r.ARMS[1]][i]['symbols']=['A','B','D']
        self.assertTrue(r.selection(metrics,orders)['admitted'])
        original=r.selection(metrics,orders)
        for i in range(88,122):orders[r.ARMS[1]][i]['symbols']=['X','Y','Z']
        self.assertEqual(original,r.selection(metrics,orders))
        metrics[r.ARMS[2]]['same_close_zero']['training']['cagr']=.52
        self.assertFalse(r.selection(metrics,orders)['admitted'])

    def test_wealth_telescope_first_fee_drawdown_and_nullable_sharpe(self):
        _,p=r.helpers();rows=[dict(date='2020-01-31',equity=99950.),dict(date='2020-02-28',equity=110000.),dict(date='2020-03-31',equity=121000.)]
        f=p.monthly_factors(rows,['2020-01-31','2020-02-28'],['2020-02-28','2020-03-31'])
        self.assertEqual(f,[1.1,1.1]);self.assertAlmostEqual(math.prod(f),1.21)
        m=p.measure(rows,100000.);self.assertAlmostEqual(m['max_drawdown'],-.0005)
        flat=[dict(date=x['date'],equity=100000.) for x in rows];self.assertIsNone(p.measure(flat,100000.)['sharpe'])
        self.assertIn('未定义',p.format_metrics(dict(cagr=0.,sharpe=None,max_drawdown=0.)))

    def test_cagr_attribution_conserves_difference(self):
        _,p=r.helpers();x=p.relative_contributions([1.1,.9,1.2],[1.,1.,1.],.2,3.)
        self.assertAlmostEqual(sum(x['cagr_contributions_pp'])/100,x['cagr_difference'])
        self.assertAlmostEqual(x['cagr_difference'],1.2*(1.1*.9*1.2)**(1/3)-1-.2)
        self.assertEqual(p.relative_contributions([1.,1.],[1.,1.],.2,2.)['cagr_difference'],0.)

    def test_fee_tolerance_only_cost_and_no_other_field(self):
        h,_=r.helpers();a=[dict(signal_date=str(i),execution_date=str(i),fee=1.,symbols=['A']) for i in range(122)];b=copy.deepcopy(a);b[0]['fee']+=5e-11
        self.assertEqual(len(h.compare_holdings(a,b,'next_open_cost')),1)
        with self.assertRaises(ValueError):h.compare_holdings(a,b,'next_open_zero')
        b[0]['fee']=1.+2e-10
        with self.assertRaises(ValueError):h.compare_holdings(a,b,'next_open_cost')
        b=copy.deepcopy(a);b[0]['symbols']=['B']
        with self.assertRaises(ValueError):h.compare_holdings(a,b,'next_open_cost')

    def test_zero_fee_evidence_uses_full_metrics_and_saved_fills(self):
        result=dict(metrics=dict(fees_paid=0.),slices={'later34':dict(cagr=.1)})
        trades=[dict(cost=0.)];holdings=[dict(fee=0.)]
        self.assertTrue(r.zero_fee_evidence(result,trades,holdings)['all_saved_fees_exact_zero'])
        for kind in ('metrics','fill','period'):
            x,t,h=copy.deepcopy((result,trades,holdings))
            if kind=='metrics':x['metrics']['fees_paid']=1e-20
            elif kind=='fill':t[0]['cost']=1e-20
            else:h[0]['fee']=1e-20
            with self.subTest(kind=kind),self.assertRaises(ValueError):r.zero_fee_evidence(x,t,h)

    def test_metric_record_nullable_and_missing_rejected(self):
        h,_=r.helpers();h.metric_evidence(dict(sharpe=None,cagr=0.),dict(sharpe=None,cagr=0.),'flat')
        with self.assertRaises(ValueError):h.metric_evidence(dict(sharpe=None),dict(sharpe=0.),'bad')
        with self.assertRaises(ValueError):h.metric_evidence(dict(cagr=0.),{},'missing')

    def test_scope_pin_gate_precedes_actual_reads(self):
        with patch.object(r,'TESTS_SHA','PENDING'),patch.object(r,'helpers',side_effect=AssertionError('No actual read')):
            with self.assertRaises(ValueError):r.review('a'*64)


if __name__=='__main__':unittest.main(verbosity=2)

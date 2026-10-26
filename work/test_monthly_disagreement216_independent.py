"""Invented cached predictions and mature labels only; never actual216 errors or returns."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_disagreement216_independent as oracle


def fixture(end='2019-02-28'):
    months=pd.date_range('2016-08-31',end,freq='ME');names=['ONE','FUND','TWO'];width=3
    tm=dict(dates=[str(d.date()) for d in months],
        ends=[str((d+pd.offsets.MonthEnd(1)).date()) for d in months],
        symbols=[names.copy() for _ in months])
    qm=dict(dates=['2016-08-17']+tm['dates'],ends=['2016-08-31']+tm['ends'],
        symbols=[names.copy()]+copy.deepcopy(tm['symbols']),etfs=['FUND'])
    train=dict(offsets=np.arange(len(months)+1,dtype=int)*width,
        dates=np.repeat(np.array(tm['dates'],dtype='datetime64[D]'),width),
        ends=np.repeat(np.array(tm['ends'],dtype='datetime64[D]'),width),
        train_plain=np.zeros((len(months)*width,10)),common_good=np.ones(len(months)*width,bool))
    for key in oracle.TARGETS:train['target_'+key]=np.zeros(len(months)*width)
    p=np.array([[(i%9)/16.,-(i%7)/16.,(i%5-2)/16.] for i in range(len(qm['dates']))]).reshape(-1)
    source=dict(raw_local_linear=p,available=np.ones(len(p),bool),local_available=np.zeros(len(p),bool),
        raw_source_log=np.zeros(len(p)),raw_local_constant=np.full(len(p),100.),
        raw_candidate=np.full(len(p),200.),raw_control=np.full(len(p),300.))
    return [train,tm,qm,source,np.ones((len(qm['dates']),width),bool)]


def prefix(inputs, n, physical=True):
    train, tm, qm, source, eligible = copy.deepcopy(inputs)
    width = eligible.shape[1]
    for key in ('dates', 'ends', 'symbols'): qm[key] = qm[key][:n]
    source = {k: v[:n*width] for k, v in source.items()}
    eligible = eligible[:n]
    if physical:
        cutoff = '%04d-12-31' % (int(qm['dates'][-1][:4])-1) if n else None
        months = [i for i, end in enumerate(tm['ends']) if cutoff is not None and end <= cutoff]
        flat = [j for i in months for j in range(int(train['offsets'][i]), int(train['offsets'][i+1]))]
        for key in ('dates', 'ends', 'symbols'): tm[key] = [tm[key][i] for i in months]
        for key in train:
            if key != 'offsets': train[key] = train[key][flat]
        train['offsets'] = np.r_[0, np.cumsum([len(ss) for ss in tm['symbols']], dtype=int)]
    return train, tm, qm, source, eligible


def latest(built):
    return built['audit']['annual'][-1]


def group(built,symbol='ONE'):
    return next(g for g in latest(built)['groups'] if g['symbol']==symbol)


def rank_fixture():
    source = np.array([[0., .1, .2, .3], [.2, .1, 0., -.1],
                       [0., .2, .1, .3], [0., .1, .2, .3]])
    shape = source.shape
    arrays = dict(baseline=np.zeros(shape), eligible=np.ones(shape, bool),
        forecast_available=np.ones(shape, bool), model_available=np.ones(shape, bool),
        local_available=np.zeros(shape, bool), raw_source_log=source.copy(),
        raw_local_constant=source+.02, raw_local_linear=source[:, ::-1].copy(),
        rank_log_excess=np.vstack([oracle.centered_rank(x) for x in source]),
        state_calm10=np.array([True, True, False, True]),
        state_weak_narrow=np.array([False, False, True, False]),
        extension_available=np.ones(shape, bool))
    forecast = dict(raw=dict(candidate=np.array([[.1, .1, .2, .2], [.3, .1, .1, .2],
        [.1, .2, .1, .2], [.2, .3, .1, 0.]]).ravel(),
        control=np.array([[0., .3, .2, .1], [.2, .3, .1, 0.],
                          [.3, .2, .1, 0.], [0., .2, .1, .3]]).ravel()),
        available=np.ones(source.size, bool))
    return arrays, forecast


class Tests(unittest.TestCase):
    def test_signed_groups_zero_disagreement_and_fixed_shrinkage(self):
        total,mu,c=oracle.statistics([-4.,-2.,2.,4.],[1.,1.,0.,-1.])
        self.assertEqual(total,0.);self.assertEqual(mu,0.)
        self.assertEqual(c['positive'],dict(rows=2,error_sum=-6.,mean=-.3,penalty=0.,empty_group_copy=False))
        self.assertEqual(c['nonpositive'],dict(rows=2,error_sum=6.,mean=.3,penalty=.3,empty_group_copy=False))
        total,mu,c=oracle.statistics([1.,2.,5.],[1.,0.,-1.])
        self.assertEqual(mu,8./3.)
        self.assertEqual(c['positive']['mean'],math.fsum([1.,18.*mu])/19.)
        self.assertEqual(c['nonpositive']['mean'],math.fsum([7.,18.*mu])/20.)
        self.assertLessEqual(min(1.,mu),c['positive']['mean'])
        self.assertLessEqual(c['positive']['mean'],max(1.,mu))

    def test_empty_conditional_group_exact_prior_copy(self):
        for ds,empty in (([1.]*18,'nonpositive'),([0.]*18,'positive'),([-1.]*18,'positive')):
            _,mu,c=oracle.statistics([1./7.]*18,ds)
            self.assertEqual(c[empty]['rows'],0);self.assertTrue(c[empty]['empty_group_copy'])
            self.assertEqual(c[empty]['error_sum'],0.)
            self.assertEqual(np.float64(c[empty]['mean']).tobytes(),np.float64(mu).tobytes())
        with self.assertRaises(ValueError):oracle.statistics([1.],[])
        with self.assertRaises(ValueError):oracle.statistics([],[])

    def test_counted_three_parameters_and_current_sign_selects_frozen_groups(self):
        args=fixture();built=oracle.reconstruct(*args)
        self.assertEqual([r['year'] for r in built['audit']['annual']],[2016,2017,2018,2019])
        self.assertEqual([r['supported_groups'] for r in built['audit']['annual']],[0,0,0,3])
        self.assertEqual(latest(built)['parameter_estimates'],9)
        proof=oracle.check_forecasts(*args,built)
        self.assertTrue(all(proof[k] for k in oracle.MODEL_FLAGS))
        self.assertEqual(proof['new_model_fits'],9);self.assertEqual(proof['new_ridge_fits'],0)
        self.assertEqual(group(built)['own_months'],28)
        for g in latest(built)['groups']:
            for detail in g['query_details']:
                j=detail['source_row_id'];p=args[3]['raw_local_linear'][j];base=args[3]['raw_source_log'][j]
                kind='positive' if p-base>0. else 'nonpositive'
                self.assertEqual(detail['disagreement_group'],kind)
                self.assertEqual(built['raw']['candidate'][j],oracle.penalized(p,g['conditional'][kind]['penalty']))
                self.assertEqual(built['raw']['control'][j],oracle.penalized(p,g['unconditional_penalty']))

    def test_partial_excluded_exact_same_august_formal_identity(self):
        args=fixture();b=oracle.reconstruct(*args);g=group(b)
        self.assertEqual(g['skipped_initial_rows'],1)
        self.assertEqual(g['history'][0]['source_row_id'],3)
        self.assertEqual(g['history'][0]['training_row_id'],0)
        self.assertEqual(g['history'][0]['date'],'2016-08-31')
        self.assertEqual(latest(b)['skipped_initial_rows'],3)
        self.assertTrue(all(h['date']!='2016-08-17' for r in b['audit']['annual'] for g in r['groups'] for h in g['history']))

    def test_exact18_month_threshold_is_per_security(self):
        for count in (17,18):
            args=fixture();train,tm,_,_,_=args
            mature=[i for i,e in enumerate(tm['ends']) if e<='2018-12-31']
            for i in mature[count:]:train['common_good'][3*i]=False
            b=oracle.reconstruct(*args);g=group(b)
            self.assertEqual(g['own_months'],count)
            self.assertEqual(g['status'],'calibrated' if count==18 else 'cold_start')
            self.assertEqual(group(b,'FUND')['status'],'calibrated')
            if count==17:
                self.assertIsNone(g['unconditional_mean']);self.assertIsNone(g['error_sum']);self.assertTrue(all(c['mean'] is None for c in g['conditional'].values()))
                for j in g['query_row_ids']:self.assertEqual(b['raw']['candidate'][j],args[3]['raw_local_linear'][j])

    def test_signed_errors_group_by_p_minus_b_not_error_sign(self):
        args=fixture();train,tm,qm,src,use=args
        pattern=[-9.]*20+[1.,2.,3.,4.,5.,6.,7.,8.]
        for i,value in enumerate(pattern):
            src['raw_local_linear'][(i+1)*3]=value
            src['raw_source_log'][(i+1)*3]=value-(1. if i%2 else -1.)
        b=oracle.reconstruct(*args);g=group(b)
        expected=math.fsum(pattern)/28.
        self.assertEqual(g['unconditional_mean'],expected)
        self.assertEqual(g['conditional']['positive']['rows'],14)
        self.assertEqual(g['conditional']['positive']['error_sum'],math.fsum(pattern[1::2]))
        self.assertEqual(g['conditional']['nonpositive']['error_sum'],math.fsum(pattern[::2]))
        self.assertEqual(g['unconditional_penalty'],0.)
        self.assertTrue(all(h['disagreement_group']==('positive' if i%2 else 'nonpositive') for i,h in enumerate(g['history'])))

    def test_current_disagreement_switch_not_current_target_or_current_U(self):
        args=fixture();b=oracle.reconstruct(*args);g=group(b,'TWO')
        j=g['query_row_ids'][-1];args[3]['raw_source_log'][j]=args[3]['raw_local_linear'][j]-1.
        args[4][j//3,j%3]=False
        changed=oracle.reconstruct(*args);cg=group(changed,'TWO')
        self.assertEqual(cg['conditional'],g['conditional'])
        self.assertEqual(cg['query_details'][-1]['disagreement_group'],'positive')
        self.assertEqual(changed['raw']['control'][j],b['raw']['control'][j])
        self.assertEqual(changed['raw']['candidate'][j],oracle.penalized(args[3]['raw_local_linear'][j],g['conditional']['positive']['penalty']))

    def test_negative_predictions_and_zero_penalty_copy_signed_zero(self):
        self.assertEqual(oracle.penalized(-.2,.1),-.2-.1)
        self.assertEqual(np.float64(oracle.penalized(-0.,0.)).tobytes(),np.float64(-0.).tobytes())
        args=fixture();args[3]['raw_local_linear'][:]=-.25;args[3]['raw_local_linear'][-1]=-0.
        b=oracle.reconstruct(*args)
        self.assertEqual(b['raw']['candidate'].tobytes(),args[3]['raw_local_linear'].tobytes())
        self.assertEqual(b['raw']['control'].tobytes(),args[3]['raw_local_linear'].tobytes())
        self.assertEqual(b['audit']['new_model_fits'],9) # zero-clamped parameters are still estimated.
        bad=copy.deepcopy(b);bad['raw']['candidate'][-1]=0.
        with self.assertRaises(ValueError):oracle.check_forecasts(*args,bad)

    def test_original_U_not_local_history_or_current_eligible_filter(self):
        args=fixture();args[4][1:,1:]=False
        b=oracle.reconstruct(*args)
        self.assertEqual(group(b)['own_months'],28)
        self.assertEqual(group(b,'FUND')['own_months'],0)
        self.assertTrue(b['available'].all())
        self.assertEqual(group(b)['status'],'calibrated') # one surviving ID/month is enough.
        args[3]['local_available']=np.full(args[4].size,'IGNORED',object)
        args[3]['raw_control']=np.full(args[4].size,'IGNORED',object)
        got=oracle.reconstruct(*args);oracle.record_check(got['audit'],b['audit'])
        args[4][-1]=False
        got=oracle.reconstruct(*args);self.assertEqual(group(got)['unconditional_penalty'],group(b)['unconditional_penalty'])
        self.assertTrue(got['available'][-3:].all())

    def test_same_security_not_row_position_and_no_cross_security_pool(self):
        args=fixture();b=oracle.reconstruct(*args)
        args[3]['raw_local_linear'][1::3]=1e6
        got=oracle.reconstruct(*args)
        self.assertEqual(group(got),group(b))
        # Reordering each formal query, with its own values/mask, preserves identities.
        args=fixture();expected=oracle.reconstruct(*args);permutation=[2,0,1]
        for i in range(len(args[2]['dates'])):
            args[2]['symbols'][i]=[args[2]['symbols'][i][k] for k in permutation]
            args[4][i]=args[4][i,permutation]
            for k in args[3]:args[3][k][3*i:3*i+3]=args[3][k][3*i:3*i+3][permutation]
        got=oracle.reconstruct(*args)
        for k in ('candidate','control'):
            np.testing.assert_array_equal(got['raw'][k].reshape(-1,3),expected['raw'][k].reshape(-1,3)[:,permutation])

    def test_fiveyear_exclusive_start_and_maturity_end(self):
        b=oracle.reconstruct(*fixture('2022-01-31'));g=group(b)
        self.assertEqual(latest(b)['lookback_exclusive_start'],'2016-12-31')
        self.assertEqual(g['history'][0]['date'],'2017-01-31')
        self.assertEqual(g['history'][-1]['date'],'2021-11-30')
        self.assertEqual(g['latest_label_end'],'2021-12-31')
        self.assertEqual(g['own_months'],59)

    def test_exact_identity_missing_mature_label_and_offsets_fail(self):
        for change in ('symbol','end','offset'):
            args=fixture()
            if change=='symbol':args[1]['symbols'][0][0]='OTHER'
            elif change=='end':
                args[1]['ends'][0]='2016-09-29';args[0]['ends'][:3]=np.datetime64('2016-09-29')
            else:args[0]['offsets'][1]+=1
            with self.assertRaises(ValueError):oracle.reconstruct(*args)

    def test_invalid_common_labels_excluded_with_counts_not_filled(self):
        args=fixture();t=args[0]
        t['common_good'][0]=False;t['target_simple_excess'][1]=np.nan;t['train_plain'][3,0]=np.inf
        b=oracle.reconstruct(*args)
        self.assertEqual(latest(b)['invalid_label_rows'],3)
        self.assertEqual(group(b)['own_months'],26)
        self.assertEqual(group(b,'FUND')['own_months'],27)
        t['train_plain']=t['train_plain'].astype(object);t['train_plain'][0]='NOT CONSUMED'
        oracle.reconstruct(*args)
        t['train_plain'][4,0]='BAD CONSUMED'
        with self.assertRaises(ValueError):oracle.reconstruct(*args)

    def test_future_numeric_labels_and_empty_prefix_not_read(self):
        args=fixture();b=oracle.reconstruct(*args);t=args[0]
        future=t['ends']>np.datetime64('2018-12-31')
        for key in ('common_good','train_plain')+tuple('target_'+k for k in oracle.TARGETS):
            t[key]=t[key].astype(object);t[key][future]='FUTURE'
        got=oracle.reconstruct(*args);oracle.record_check(got['audit'],b['audit'])
        empty=prefix(args,0,physical=False)
        for key in ('common_good','train_plain')+tuple('target_'+k for k in oracle.TARGETS):empty[0][key][:]='EMPTY'
        self.assertEqual(oracle.reconstruct(*empty)['audit']['counts']['parameter_estimates'],0)

    def test_unavailable_current_nan_and_historical_no_join(self):
        args=fixture();args[3]['available'][3]=False;args[3]['raw_local_linear'][3]=np.nan;args[3]['raw_source_log'][3]=np.nan
        args[1]['symbols'][0][0]='NO_HISTORY_JOIN'
        b=oracle.reconstruct(*args)
        for k in ('candidate','control'):self.assertTrue(np.isnan(b['raw'][k][3]))
        args[3]['raw_local_linear'][3]=np.inf
        with self.assertRaises(ValueError):oracle.reconstruct(*args)

    def test_nonboolean_numbers_and_bad_metadata_rejected(self):
        for place in ('target','source','pooled','mask','identity','double_month'):
            args=fixture()
            if place=='target':
                args[0]['target_log_excess']=args[0]['target_log_excess'].astype(object);args[0]['target_log_excess'][0]=True
            elif place=='source':
                args[3]['raw_local_linear']=args[3]['raw_local_linear'].astype(object);args[3]['raw_local_linear'][0]='0.2'
            elif place=='pooled':
                args[3]['raw_source_log']=args[3]['raw_source_log'].astype(object);args[3]['raw_source_log'][0]=True
            elif place=='mask':args[4]=args[4].astype(int)
            elif place=='identity':args[2]['symbols'][1][1]=args[2]['symbols'][1][0]
            else:args[2]['dates'][2]='2016-09-01';args[2]['dates'][3]='2016-09-30'
            with self.assertRaises(ValueError):oracle.reconstruct(*args)

    def test_finite_arithmetic_overflow_hardfails(self):
        args=fixture();args[3]['raw_local_linear'][3]=1e308;args[0]['target_log_excess'][0]=-1e308
        with self.assertRaises(FloatingPointError):oracle.reconstruct(*args)
        with self.assertRaises(FloatingPointError):oracle.statistics([1e308,1e308],[0.,0.])
        with self.assertRaises(FloatingPointError):oracle.penalized(-1e308,1e308)
        with self.assertRaises(ValueError):oracle.penalized(1.,-1.)

    def test_artifact_raw_mask_audit_and_counts_corruption_rejected(self):
        args=fixture();b=oracle.reconstruct(*args)
        for key in ('candidate','control'):
            bad=copy.deepcopy(b);bad['raw'][key][0]=np.nextafter(bad['raw'][key][0],np.inf)
            with self.assertRaises(ValueError):oracle.check_forecasts(*args,bad)
        bad=copy.deepcopy(b);bad['available'][0]=False
        with self.assertRaises(ValueError):oracle.check_forecasts(*args,bad)
        for key in oracle.COUNT_FIELDS:
            bad=copy.deepcopy(b);bad['audit']['counts'][key]+=1
            with self.assertRaises(ValueError):oracle.check_forecasts(*args,bad)
        bad=copy.deepcopy(b);bad['audit']['annual'][-1]['groups'][0]['history'][0]['training_row_id']+=1
        with self.assertRaises(ValueError):oracle.check_forecasts(*args,bad)

    def test_pooled_source_availability_and_disagreement_overflow(self):
        for changed in ('missing','nan','infinite','bad_mask'):
            args=fixture()
            if changed=='missing':del args[3]['raw_source_log']
            elif changed=='nan':args[3]['raw_source_log'][0]=np.nan
            elif changed=='infinite':args[3]['raw_source_log'][0]=np.inf
            else:args[3]['available'][0]=False
            with self.assertRaises(ValueError):oracle.reconstruct(*args)
        for row in (0,3,-1):
            args=fixture();args[3]['raw_local_linear'][row]=1e308;args[3]['raw_source_log'][row]=-1e308
            with self.assertRaises(FloatingPointError):oracle.reconstruct(*args)
        with self.assertRaises(FloatingPointError):oracle.statistics([1e307]*18,[1.]*18)

    def test_conditional_stats_and_query_audit_mutation_rejected(self):
        args=fixture();expected=oracle.reconstruct(*args)
        for mutation in ('unconditional','groupmean','grouprows','emptycopy','querysign','querypenalty','historicalsign','fitcount'):
            bad=copy.deepcopy(expected);g=group(bad)
            if mutation=='unconditional':g['unconditional_mean']+=1.
            elif mutation=='groupmean':g['conditional']['positive']['mean']+=1.
            elif mutation=='grouprows':g['conditional']['positive']['rows']+=1
            elif mutation=='emptycopy':g['conditional']['positive']['empty_group_copy']=True
            elif mutation=='querysign':g['query_details'][0]['disagreement_group']='MUTATED'
            elif mutation=='querypenalty':g['query_details'][0]['conditional_penalty']+=1.
            elif mutation=='historicalsign':g['history'][0]['disagreement']+=1.
            else:bad['audit']['new_model_fits']=0
            with self.assertRaises(ValueError):oracle.check_forecasts(*args,bad)

    def test_inputs_and_original_targets_unchanged(self):
        args=fixture();before=copy.deepcopy(args);b=oracle.reconstruct(*args);oracle.check_forecasts(*args,b)
        b['raw']['candidate'][:]=999.
        for pos in (0,3):
            for key in args[pos]:np.testing.assert_array_equal(args[pos][key],before[pos][key])
        self.assertEqual(args[1:3],before[1:3]);np.testing.assert_array_equal(args[4],before[4])

    def test_rank_slot_fullU_shared_readiness_initialzero(self):
        arrays,forecast=rank_fixture();rr,dd,ready=oracle.rank_oracle(arrays,forecast)
        proof=oracle.verify_rank_slot(arrays,forecast,rr,dd)
        self.assertTrue(all(proof[k] for k in oracle.RANK_FLAGS))
        for arm in oracle.NEW_RAW:
            np.testing.assert_array_equal(dd[arm][0],0.);self.assertLessEqual(abs(dd[arm]).max(),.0125)
        forecast['raw']['candidate'][4]=np.nan;forecast['available'][4]=False
        rr,dd,ready=oracle.rank_oracle(arrays,forecast)
        for arm in oracle.NEW_RAW:
            self.assertFalse(ready[arm][1]);np.testing.assert_array_equal(dd[arm][1],0.)
        bad=copy.deepcopy(dd);bad[oracle.ARMS[1]][0,0]=1e-16
        with self.assertRaises(ValueError):oracle.verify_rank_slot(arrays,forecast,rr,bad)

    def test_accounting_is_pinned_and_independent(self):
        self.assertTrue(callable(oracle.accounting().metrics))
        with patch.object(oracle,'ACCOUNTING_SHA','0'*64):
            with self.assertRaises(ValueError):oracle.accounting()

    def test_both_fixtures_all_prefixes_complete_audit_and_host_interfaces(self):
        import monthly_disagreement216_model as production
        import test_monthly_disagreement216_model as producer_test
        import monthly_disagreement216_host as host
        for args in (fixture(),producer_test.fixture()):
            whole=oracle.reconstruct(*args);nrows,width=args[4].shape
            for n in range(nrows+1):
                for small in (prefix(args,n),host.prefix_inputs(*args,n)):
                    b=production.build_forecasts(*small)
                    proof=oracle.check_forecasts(*small,b)
                    host.require_model_audit(proof,b,expected_rows=n*width,
                        expected_years=len({d[:4] for d in small[2]['dates']}))
                    for key in ('candidate','control'):
                        np.testing.assert_array_equal(b['raw'][key],whole['raw'][key][:n*width])
        import test_monthly_disagreement216_host as host_test
        arrays,meta,forecast=host_test.fixture()
        ranks,deltas=host.rank_and_delta(host_test.SyntheticStudy(),arrays,forecast,len(meta['dates']))
        oracle.verify_rank_slot(arrays,forecast,ranks,deltas)


if __name__=='__main__':unittest.main(verbosity=2)

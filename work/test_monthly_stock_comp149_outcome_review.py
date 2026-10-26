"""Generated-price accounting and tamper tests; never open real market outputs."""
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import hashlib
import io
import json
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import monthly_stock_comp149_outcome_review as m


def order(day,end,names):
    return dict(date=str(pd.Timestamp(day).date()),signal_date=str(pd.Timestamp(day).date()),end=str(pd.Timestamp(end).date()),symbols=list(names),scores=[.2]*len(names))


def synthetic_outputs():
    days=pd.bdate_range('2016-01-04',periods=245);x=np.arange(len(days))
    close=pd.DataFrame({s:base*np.exp(rate*x+.007*np.sin(x/freq)) for s,base,rate,freq in [('A',10,.001,4),('B',20,.0007,7),('C',30,.0013,6)]},index=days)
    opening=close*np.array([.999,1.001,1.0002])
    meta=dict(dates=[str(d.date()) for d in days[:-1:2]],ends=[str(d.date()) for d in days[2::2]],symbols=[['A','B','C']+['S'+str(i) for i in range(3,32)]+['ETF'+str(i) for i in range(17)] for _ in range(122)],etfs=['ETF'+str(i) for i in range(17)])
    orders={name:[order(day,end,['A','C'] if k and i%(k+2)==0 else ['A','B']) for i,(day,end) in enumerate(zip(meta['dates'],meta['ends']))] for k,name in enumerate(m.NAMES)}
    data={};results={};navs={};opens={}
    bounds=dict(early2016_2020=(meta['dates'][0],meta['ends'][52]),original2021_2026=(meta['dates'][53],meta['ends'][-1]),training2016_2023=(meta['dates'][0],meta['ends'][87]),later34=(meta['dates'][88],meta['ends'][-1]))
    for name in m.NAMES:
        results[name]={}
        for execution,cost,key in m.PATHS:
            met,nav,trades,hold=m.replay(close,opening,orders[name],days,execution,cost);prefix=name+'_'+key
            results[name][key]=dict(metrics=met,slices={k:m.h.slice_stat(days,nav,*span) for k,span in bounds.items()})
            data[prefix+'_DAILY.json']=[dict(date=str(d.date()),equity=float(v)) for d,v in zip(days,nav)]
            data[prefix+'_TRADES.json']=trades;data[prefix+'_HOLDINGS.json']=hold
            if key=='same_close':navs[name]=nav
            if key=='next_open_cost':opens[name]=nav
    rows=[dict(name=name,**m.h.measures(days,navs[name],meta)) for name in m.NAMES]
    opened=[dict(name=name,**m.h.measures(days,opens[name],meta)) for name in m.NAMES]
    selection=m.select(rows,opened)
    cells=[]
    for i,(day,pool) in enumerate(zip(meta['dates'],meta['symbols'])):
        for j,symbol in enumerate(pool):
            good=j<2
            cells.append(dict(signal_index=i,pool_index=j,signal_date=day,symbol=symbol,available=good,reason='available' if good else 'etf_not_applicable' if j>=32 else 'no_known_event',method=m.METHOD,
                current_event_id='synthetic' if good else None,current_available_from='2016-01-01' if good else None,current_period_end='2015-12-31' if good else None,prior_period_end='2014-12-31' if good else None,current_days=365 if good else None,prior_days=365 if good else None,currency_key='same-report-dollar:synthetic' if good else None,
                measurement=m.MEASUREMENT,earnings_bounded=.2 if good else None,
                cash_flow_bounded=.25 if good else None,comp_adjusted_cash_flow_bounded=-.4 if good else None,
                original_cash_quality_available=good,
                stock_compensation_event_id='synthetic' if good else None))
    feature=dict(original_arrays_unchanged=True,original_months=122,original_width=49,new_model_fits=0,returns_or_nav_computed=False,original_auxiliary_states_and_core_coefficients_preserved=True,common_available_cells=244,original_cash_quality_available_cells=244,reasons=dict(Counter(c['reason'] for c in cells)),arithmetic_method='exact_fraction_scaled_cashflow_minus_signed_sbc_over_average_assets_then_bounded_float',annual_duration_normalization=False,comparison_flow_values_used=False,latest_event_before_validity_filter=True,same_current_release_for_all_cells=True,cells=cells)
    feature.update(measurement=m.MEASUREMENT,stock_compensation_definition='current_annual_reported_stock_compensation_addback_v1',
        earnings_and_cashflow_controls_identical_to146_on_common_mask=True,common_mask_subset_of146=True,
        same_selected_current_event_for_all_fields=True,signed_stock_compensation_preserved=True,
        stock_compensation_raw_flags_do_not_substitute_new_admission=True,
        adjusted_cashflow_is_research_measure_not_reported_GAAP_cashflow=True)
    data.update({'RESULTS.json':results,'TRAIN_TRIALS.json':rows,'TRAIN_OPEN_TRIALS.json':opened,'FROZEN.json':dict(selection=selection),'WALKFORWARD.json':m.walkforward(days,navs,opens,meta),'FEATURE_AUDIT.json':feature,'VALIDATION.json':dict(selection=selection,new_model_fits=0,no_observer_writes=True,no_intramonth_trades=True,independent_outcome_review_pending=True,pristine_oos=False)})
    return close,opening,orders,meta,data


class SelectionTests(unittest.TestCase):
    def groups(self):
        rows=[dict(name=n,training=dict(cagr=.5,sharpe=2.,max_drawdown=-.2),early=dict(cagr=.4),recent=dict(cagr=.6)) for n in m.NAMES]
        return rows,deepcopy(rows)
    def winner(self):
        rows,op=self.groups();rows[3]['training']['cagr']=.6;op[3]['training']['cagr']=.6
        return rows,op
    def test_tie_retains_incumbent(self):self.assertEqual(m.select(*self.groups())['primary'],m.NAMES[0])
    def test_primary_only_can_be_promoted(self):
        rows,op=self.groups();rows[1]['training']['cagr']=9;op[1]['training']['cagr']=9
        self.assertFalse(m.select(rows,op)['admitted'])
    def test_strict_win_all_controls_both_paths(self):self.assertTrue(m.select(*self.winner())['admitted'])
    def test_open_incumbent_noninferiority_allowed(self):
        rows,op=self.winner();op[0]['training']['cagr']=.6
        self.assertTrue(m.select(rows,op)['admitted'])
    def test_open_control_ties_rejected_individually(self):
        for i in (1,2):
            rows,op=self.winner();op[i]['training']['cagr']=.6
            self.assertFalse(m.select(rows,op)['admitted'])
    def test_closed_all_ties_rejected_individually(self):
        for i in range(3):
            rows,op=self.winner();rows[i]['training']['cagr']=.6
            self.assertFalse(m.select(rows,op)['admitted'])
    def test_each_guard_on_each_path_and_reference(self):
        for which in (0,1):
            for i in range(3):
                for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:
                    pair=self.winner();pair[which][i][section][key]=pair[which][3][section][key]+tol+.001
                    self.assertFalse(m.select(*pair)['admitted'])
    def test_boundaries_inclusive(self):
        rows,op=self.winner()
        for group in (rows,op):
            for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:group[3][section][key]=group[0][section][key]-tol
        self.assertTrue(m.select(rows,op)['admitted'])
    def test_tiny_strict_advantage_rejected(self):
        rows,op=self.winner();rows[3]['training']['cagr']=.5+5e-13
        self.assertFalse(m.select(rows,op)['admitted'])
    def test_nonfinite_boolean_or_missing_fails(self):
        for value in (None,True,'2',float('nan'),float('inf')):
            rows,op=self.winner();op[3]['training']['sharpe']=value
            self.assertFalse(m.select(rows,op)['admitted'])
        rows,op=self.winner();del rows[3]['early']['cagr'];self.assertFalse(m.select(rows,op)['admitted'])
    def test_later34_irrelevant(self):
        rows,op=self.groups();rows[3]['later34']={'cagr':1000}
        self.assertFalse(m.select(rows,op)['admitted'])
    def test_exact_arm_order_required(self):
        rows,op=self.winner();rows[1],rows[2]=rows[2],rows[1]
        with self.assertRaises(ValueError):m.select(rows,op)


class AccountingTests(unittest.TestCase):
    def small(self):
        d=pd.bdate_range('2016-01-04',periods=4)
        p=pd.DataFrame(dict(A=[10.,20.,30.,40.],B=[20.,20.,20.,20.],C=[10.,10.,10.,10.]),index=d)
        o=p.copy();o.iloc[1]=[10.,10.,10.]
        return d,p,o
    def test_fixed_shares(self):
        d,p,o=self.small();_,nav,trades,_=m.replay(p,o,[order(d[0],d[-1],['A','B'])],d)
        np.testing.assert_allclose(nav,[100000,150000,200000,250000]);self.assertEqual(len(trades),2)
    def test_next_open(self):
        d,p,o=self.small();_,nav,trades,_=m.replay(p,o,[order(d[0],d[-1],['A','B'])],d,'open',0)
        self.assertEqual(nav[0],100000);self.assertEqual(nav[1],200000);self.assertEqual(trades[0]['date'],str(d[1].date()))
    def test_fees_entry_exit(self):
        d,p,o=self.small();met,nav,_,_=m.replay(p,o,[order(d[0],d[2],['B','C']),order(d[2],d[-1],[])],d,'close',.0005)
        final=100000/1.0005*(1-.0005)
        self.assertAlmostEqual(nav[-1],final);self.assertAlmostEqual(met['fees_paid'],100000-final)
    def test_cash_and_bad_cost(self):
        d,p,o=self.small();_,nav,tr,_=m.replay(p,o,[order(d[0],d[-1],[])],d)
        np.testing.assert_array_equal(nav,np.full(4,100000.));self.assertEqual(tr,[])
        with self.assertRaises(ValueError):m.replay(p,o,[],d,'close',.01)


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=synthetic_outputs()
    def setUp(self):self.px,self.op,self.orders,self.meta,self.data=deepcopy(self.fixture)
    def check(self):return m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__)
    def test_twelve_paths(self):
        out=self.check();self.assertTrue(out['passed']);self.assertEqual(out['execution_paths_verified'],12);self.assertEqual(out['total_monthly_orders_verified'],488);self.assertEqual(len(out['independent_monthly_comparison']),122)
    def test_nav_tamper(self):
        self.data[m.NAMES[3]+'_next_open_cost_DAILY.json'][4]['equity']+=10
        with self.assertRaises(AssertionError):self.check()
    def test_trade_cost_tamper(self):
        self.data[m.NAMES[0]+'_next_open_cost_TRADES.json'][0]['cost']+=1
        with self.assertRaises(AssertionError):self.check()
    def test_holdings_units_tamper(self):
        rows=self.data[m.NAMES[1]+'_same_close_HOLDINGS.json'];rows[0]['symbols']=[]
        with self.assertRaises((AssertionError,ValueError)):self.check()
    def test_wrong_execution_date(self):
        self.data[m.NAMES[0]+'_next_open_zero_TRADES.json'][0]['date']=self.meta['dates'][0]
        with self.assertRaises(AssertionError):self.check()
    def test_single_name_order_rejected(self):
        self.orders[m.NAMES[0]][0].update(symbols=['A'],scores=[.2])
        with self.assertRaises(ValueError):m.validate_orders(self.px,self.op,self.orders,self.meta)
    def test_feature_mask_disagreement(self):
        self.data['FEATURE_AUDIT.json']['cells'][3]['earnings_bounded']=.2
        with self.assertRaises(ValueError):m.validate_cells(self.data['FEATURE_AUDIT.json']['cells'],self.meta)
    def test_future_availability_rejected(self):
        self.data['FEATURE_AUDIT.json']['cells'][0]['current_available_from']='2017-01-01'
        with self.assertRaises(ValueError):m.validate_cells(self.data['FEATURE_AUDIT.json']['cells'],self.meta)
    def test_etf_cannot_receive_financials(self):
        cell=deepcopy(self.data['FEATURE_AUDIT.json']['cells'][0]);cell.update(pool_index=32,symbol='ETF0')
        self.data['FEATURE_AUDIT.json']['cells'][32]=cell
        with self.assertRaises(ValueError):m.validate_cells(self.data['FEATURE_AUDIT.json']['cells'],self.meta)
    def test_exact_matrix_match_and_tamper(self):
        feature=self.data['FEATURE_AUDIT.json'];arrays={k:np.full((122,49),np.nan) for k in m.HANDS};arrays['common_available']=np.zeros((122,49),bool)
        for c in feature['cells']:
            i,j=c['signal_index'],c['pool_index'];arrays['common_available'][i,j]=c['available']
            if c['available']:
                for k in m.HANDS:arrays[k][i,j]=c[k]
        self.assertEqual(m.verify_feature_matrices(arrays,feature,self.meta),4)
        arrays[m.HANDS[2]][0,0]+=.0001
        with self.assertRaises(AssertionError):m.verify_feature_matrices(arrays,feature,self.meta)
    def test_frozen_choice_tamper(self):
        self.data['FROZEN.json']['selection']['primary']='forged'
        with self.assertRaises(AssertionError):self.check()
    def test_training_measure_tamper(self):
        self.data['TRAIN_OPEN_TRIALS.json'][3]['training']['sharpe']+=.01
        with self.assertRaises(ValueError):self.check()
    def test_walkforward_measure_tamper(self):
        self.data['WALKFORWARD.json']['folds'][0]['selection']['admitted']=not self.data['WALKFORWARD.json']['folds'][0]['selection']['admitted']
        with self.assertRaises(ValueError):self.check()
    def test_new_fit_or_comparison_flow_use_rejected(self):
        for key,val in (('new_model_fits',1),('comparison_flow_values_used',True)):
            f=deepcopy(self.data['FEATURE_AUDIT.json']);f[key]=val
            with self.assertRaises(ValueError):m.feature_checks(f,self.data['VALIDATION.json'],self.meta)
    def test_no_new_producer_imported(self):
        self.assertNotIn('monthly_stock_comp149_runner',sys.modules);self.assertNotIn('monthly_stock_comp149_factor',sys.modules)
    def test_wrong_measurement_rejected(self):
        self.data['FEATURE_AUDIT.json']['cells'][0]['measurement']='cash_quality'
        with self.assertRaises(ValueError):m.validate_cells(self.data['FEATURE_AUDIT.json']['cells'],self.meta)
    def test_stock_comp_cannot_exceed_bounded_limit(self):
        self.data['FEATURE_AUDIT.json']['cells'][0]['comp_adjusted_cash_flow_bounded']=1.1
        with self.assertRaises(ValueError):m.validate_cells(self.data['FEATURE_AUDIT.json']['cells'],self.meta)
    def test_old_asset_growth_hand_is_rejected(self):
        self.data['FEATURE_AUDIT.json']['cells'][0]['low_asset_growth_bounded']=.2
        with self.assertRaises(ValueError):m.validate_cells(self.data['FEATURE_AUDIT.json']['cells'],self.meta)

    def test_new_mask_cannot_broaden_original(self):
        self.data['FEATURE_AUDIT.json']['cells'][0]['original_cash_quality_available']=False
        with self.assertRaises(ValueError):self.check()

    def test_stock_comp_same_selected_event(self):
        self.data['FEATURE_AUDIT.json']['cells'][0]['stock_compensation_event_id']='older'
        with self.assertRaises(ValueError):self.check()

    def test_unavailable_has_no_fake_admitted_event(self):
        self.data['FEATURE_AUDIT.json']['cells'][3]['stock_compensation_event_id']='fake'
        with self.assertRaises(ValueError):self.check()

    def test_signed_correction_is_not_positive_only(self):
        cell=self.data['FEATURE_AUDIT.json']['cells'][0]
        for value in (-1., -.9, 0., .9, 1.):
            cell['comp_adjusted_cash_flow_bounded']=value
            m.validate_cells(self.data['FEATURE_AUDIT.json']['cells'],self.meta)

    def test_original_count_cannot_be_relabelled(self):
        self.data['FEATURE_AUDIT.json']['original_cash_quality_available_cells']+=1
        with self.assertRaises(ValueError):self.check()

    def test_source_claims_required(self):
        for key in ('common_mask_subset_of146','signed_stock_compensation_preserved',
                    'same_selected_current_event_for_all_fields',
                    'adjusted_cashflow_is_research_measure_not_reported_GAAP_cashflow'):
            feature=deepcopy(self.data['FEATURE_AUDIT.json']);feature[key]=False
            with self.assertRaises(ValueError):m.feature_checks(feature,self.data['VALIDATION.json'],self.meta)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.folder=self.root/m.STUDY
        self.folder.mkdir(parents=True)
        self.reader=m.Reader(self.root)
        self.audit_time=datetime(2026,10,20,tzinfo=timezone.utc)
        self.plan_sha='a'*64

    def marker(self,**changes):
        data=dict(plan_sha256=self.plan_sha,started_utc='2026-10-19T00:00:00+00:00')
        data.update(changes)
        (self.folder/'DIAGNOSE_STARTED.json').write_text(json.dumps(data))

    def test_diagnostic_must_exist(self):
        with self.assertRaises(ValueError):
            m.verify_diagnostic_start(self.reader,self.folder,self.plan_sha,self.audit_time)

    def test_utc_marker_prior_to_audit(self):
        self.marker()
        m.verify_diagnostic_start(self.reader,self.folder,self.plan_sha,self.audit_time)

    def test_marker_plan_and_time_tamper(self):
        for change in (dict(plan_sha256='b'*64),
                       dict(started_utc='2999-01-01T00:00:00Z'),
                       dict(started_utc='2026-10-19T00:00:00'),
                       dict(extra=True)):
            self.marker(**change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                m.verify_diagnostic_start(m.Reader(self.root),self.folder,self.plan_sha,self.audit_time)

    def test_reader_rejects_mid_audit_mutation(self):
        self.marker();path=self.folder/'DIAGNOSE_STARTED.json'
        self.reader.sha(path);self.marker(plan_sha256='b'*64)
        with self.assertRaises(ValueError):self.reader.sha(path)

    def test_exact_sources_named(self):
        paths=m.source_files(self.root)
        self.assertEqual(paths[0],self.root/'work/monthly_stock_comp149_outcome_review.py')
        self.assertIn(self.root/'work/monthly_stock_comp149_review_contract.py',paths)
        self.assertFalse(any('149_runner' in str(p) or '149_factor' in str(p) for p in paths))

    def test_plan_digest_required_before_data_read(self):
        for value in ('',None,'A'*64,'a'*63,'x'*64):
            with self.assertRaises(ValueError):m.bindings(self.reader,value)

    def chain(self):
        """Synthetic I/O chain; source semantics are tested by shared contract tests."""
        def put(path,value):
            p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(value));return p
        def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
        source=put('synthetic_review.py',{})
        inputs={k:put('input/'+k+'.json',{}) for k in m.INPUTS}
        inputs['prices']=put(m.PRICE,{})
        inputs['original_orders']=put('outputs/monthly_local_linear70/local_linear_ORDERS.json',[])
        self.plan=dict(round=149,status='REGISTERED',arms=list(m.NAMES),primary_challenger=m.NAMES[3],
            output_directory=str(self.folder),dose=.01,
            runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),
            inputs={k:dict(path=str(p),sha256=sha(p)) for k,p in inputs.items()},
            hashes={str(p):sha(p) for p in [source,*inputs.values()]},
            raw_source_hashes={str(inputs['prices']):sha(inputs['prices'])},
            independent_prepare_source_hashes={str(source):sha(source)},
            actual_feature_calculations_before_registration=0,
            actual_returns_before_registration=0,actual_model_fits_before_registration=0)
        put(m.STUDY+'/PLAN.json',self.plan)
        self.plan_sha=sha(self.folder/'PLAN.json')
        for name in m.CACHE_FILES:put(m.STUDY+'/'+name,{'synthetic':name})
        put(m.STUDY+'/PRETRAIN_ORDERS.json',{name:[] for name in m.NAMES})
        put(m.STUDY+'/CACHE.json',{name:sha(self.folder/name) for name in m.CACHE_FILES})
        flags=dict.fromkeys((*m.PREPARE_CHECKS,'source_admission_bound',
            'source_union_preservation_verified','raw_prices_independently_verified',
            'calendar_overlay_independently_verified','all_signal_prefixes_verified',
            'exact_integer_closed_form_arithmetic','stock_comp_source_admission_verified',
            'common_mask_subset_of146_verified'),True)
        self.prep=dict(passed=True,stage='prepare',**flags,orders_checked=488,
            source_cells_checked=5978,stock_cells_checked=3904,new_model_fits=0,
            monthly_label_models_fit=0,producer_modules_imported=False,returns_or_nav_computed=False,
            plan_sha256=self.plan_sha,cache_sha256=sha(self.folder/'CACHE.json'),
            orders_sha256=sha(self.folder/'PRETRAIN_ORDERS.json'),
            source_hashes={str(source):sha(source)},
            artifact_hashes={str(self.folder/name):sha(self.folder/name)
                for name in (*m.CACHE_FILES,'PLAN.json','CACHE.json')})
        put(m.STUDY+'/INDEPENDENT_PREPARE.json',self.prep)
        for name in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','WALKFORWARD.json'):
            put(m.STUDY+'/'+name,{})
        put(m.STUDY+'/FROZEN.json',dict(
            hashes={name:sha(self.folder/name) for name in m.FROZEN_FILES},
            pristine_oos=False,all_diagnostic_orders_frozen=True))
        self.marker()
        self.put=put
        for method,result in (('source_files',[source]),('verify_proposal',None),('verify_admission',None)):
            mock=patch.object(m,method,return_value=result);mock.start();self.addCleanup(mock.stop)

    def test_complete_io_chain(self):
        self.chain()
        self.assertEqual(m.bindings(m.Reader(self.root),self.plan_sha,self.audit_time),self.folder)

    def test_missing_new_prepare_assurance_rejected(self):
        self.chain();del self.prep['stock_comp_source_admission_verified']
        self.put(m.STUDY+'/INDEPENDENT_PREPARE.json',self.prep)
        with self.assertRaisesRegex(ValueError,'Independent prepare'):
            m.bindings(m.Reader(self.root),self.plan_sha,self.audit_time)

    def test_common_mask_assurance_rejected(self):
        self.chain();self.prep['common_mask_subset_of146_verified']=False
        self.put(m.STUDY+'/INDEPENDENT_PREPARE.json',self.prep)
        with self.assertRaisesRegex(ValueError,'Independent prepare'):
            m.bindings(m.Reader(self.root),self.plan_sha,self.audit_time)

    def test_cache_changed_rejected(self):
        self.chain();self.put(m.STUDY+'/SCORES.npz',{'changed':True})
        with self.assertRaises(ValueError):m.bindings(m.Reader(self.root),self.plan_sha,self.audit_time)

    def test_prepare_missing_artifact_rejected(self):
        self.chain();del self.prep['artifact_hashes'][str(self.folder/'PRICE_PARITY.json')]
        self.put(m.STUDY+'/INDEPENDENT_PREPARE.json',self.prep)
        with self.assertRaisesRegex(ValueError,'Complete prepare artifacts'):
            m.bindings(m.Reader(self.root),self.plan_sha,self.audit_time)

    def test_frozen_missing_input_rejected(self):
        self.chain();path=self.folder/'FROZEN.json'
        data=json.loads(path.read_text());del data['hashes']['TRAIN_OPEN_TRIALS.json']
        self.put(m.STUDY+'/FROZEN.json',data)
        with self.assertRaisesRegex(ValueError,'Exact FROZEN'):
            m.bindings(m.Reader(self.root),self.plan_sha,self.audit_time)


if __name__=='__main__':
    write_review='--write-review' in sys.argv
    if write_review:sys.argv.remove('--write-review')
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    log=stream.getvalue();print(log)
    if write_review:
        if not result.wasSuccessful():raise SystemExit('No receipt for failing review')
        directory=m.ROOT/m.STUDY;directory.mkdir(parents=True,exist_ok=True)
        logpath=directory/'OUTCOME_REVIEW_TESTS.log'
        with logpath.open('x') as f:f.write(log)
        receipt=m.synthetic_receipt(result)
        receipt['test_log']=dict(path=str(logpath),sha256=hashlib.sha256(logpath.read_bytes()).hexdigest())
        dest=m.ROOT/'work/monthly_stock_comp149_outcome_prereview.json'
        with dest.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


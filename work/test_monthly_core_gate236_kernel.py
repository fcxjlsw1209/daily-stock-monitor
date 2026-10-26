"""Synthetic-only236 full replacement, inherited warm semantics and clocks."""
import copy
import json
import unittest

import numpy as np
import pandas as pd

import monthly_core_gate236_kernel as k


def endpoints(meta, calendar, i):
    t = calendar.index(meta['dates'][i])
    return [pd.Timestamp(calendar[t - offset]) for offset in (294, 252, 42, 0)]


def fixture():
    days = pd.bdate_range('2015-01-01', '2016-11-10')
    frame = pd.DataFrame(100., index=days, columns=['KEEP', 'ADD', 'DROP', 'COLD'])
    meta = dict(dates=['2016-08-17', '2016-08-31', '2016-09-30', '2016-10-31'],
        symbols=[['KEEP', 'ADD', 'DROP', 'COLD'] for _ in range(4)], etfs=['ADD'])
    arrays = dict(eligible=np.array([[True, False, True, False]] * 4),
                  state_calm10=np.array([False, False, True, False]))
    calendar = [str(d.date()) for d in days]
    for i in (1, 3):
        newstart, oldstart, end, current = endpoints(meta, calendar, i)
        frame.loc[newstart, ['KEEP','ADD','DROP']] = [80., 80., 120.]
        frame.loc[oldstart, ['KEEP','ADD','DROP']] = [100., 120., 80.]
        frame.loc[end, ['KEEP','ADD','DROP']] = [120., 100., 100.]
    frame.at[pd.Timestamp('2016-05-02'), 'COLD'] = np.nan
    return {'adj':frame}, meta, calendar, arrays


def run(data, limit=None):
    return k.build_features(*data, limit=limit)


def physical(data, n):
    panel, meta, calendar, arrays = data
    mm = copy.deepcopy(meta);mm['dates'] = mm['dates'][:n];mm['symbols'] = mm['symbols'][:n]
    aa = {key:value[:n].copy() for key,value in arrays.items()}
    pp = {'adj':panel['adj'].loc[:pd.Timestamp(meta['dates'][n-1])].copy() if n else panel['adj'].iloc[:0].copy()}
    cc = calendar[:calendar.index(meta['dates'][n-1])+2] if n else []
    return pp, mm, cc, aa


class GateTests(unittest.TestCase):
    def assert_values(self, left, right):
        self.assertEqual(set(left),set(right))
        for key in left:np.testing.assert_array_equal(left[key],right[key],err_msg=key)

    def test_full_replacement_adds_and_removes_not_nested(self):
        values,audit = run(fixture())
        np.testing.assert_array_equal(values['candidate_eligible'], [[1,0,1,0],[1,1,0,0],[1,0,1,0],[1,1,0,0]])
        np.testing.assert_array_equal(values['required'], [[0]*4,[1]*4,[0]*4,[1]*4])
        np.testing.assert_array_equal(values['warm'], [[0]*4,[1,1,1,0],[0]*4,[1,1,1,0]])
        self.assertTrue(values['candidate_eligible'][1,1] and not values['original_eligible'][1,1])
        self.assertFalse(values['candidate_eligible'][1,2])
        self.assertEqual(audit['event_history'][1]['added_indices'], [1])
        self.assertEqual(audit['event_history'][1]['rejected_indices'], [2])

    def test_named_partial_and_calm_bypass_numeric_evidence(self):
        values,audit = run(fixture())
        for i in (0,2):
            np.testing.assert_array_equal(values['candidate_eligible'][i],values['original_eligible'][i])
            self.assertFalse(values['required'][i].any());self.assertFalse(values['warm'][i].any())
            self.assertTrue(np.isnan(values['momentum210'][i]).all())
            self.assertTrue(np.isnan(values['momentum252'][i]).all())
            self.assertEqual(audit['event_history'][i]['source_positions'], [])
            self.assertEqual(audit['event_history'][i]['source_dates'], [])
        self.assertEqual([x['branch'] for x in audit['event_history']],['initial_partial','core_horizon_gate','calm_bypass','core_horizon_gate'])

    def test_named_partial_only_not_arbitrary_first_regular_row(self):
        data=fixture();panel,meta,calendar,arrays=data
        meta['dates']=meta['dates'][1:2];meta['symbols']=meta['symbols'][1:2]
        aa={key:value[1:2].copy() for key,value in arrays.items()}
        values,audit=run((panel,meta,calendar,aa))
        np.testing.assert_array_equal(values['candidate_eligible'][0],[1,1,0,0])
        self.assertEqual(audit['active_events'],1)

    def test_calm_and_partial_need_no_numeric_price_or_history(self):
        data=fixture();data[3]['state_calm10'][:]=True
        data[0]['adj']=data[0]['adj'].iloc[:0,:0]
        values,audit=run(data)
        np.testing.assert_array_equal(values['candidate_eligible'],data[3]['eligible'])
        self.assertEqual(audit['warm_observations'],0)
        data=physical(fixture(),1);data[2][:]=['2016-08-17','2016-08-18'];data[0]['adj']=data[0]['adj'].iloc[:0]
        self.assertEqual(run(data)[1]['active_events'],0)

    def test_strict_zero_horizons_and_exact_near_zero(self):
        for delta,expected in ((0.,False),(np.spacing(100.),True),(-np.spacing(100.),False)):
            data=fixture();newstart,oldstart,end,_=endpoints(data[1],data[2],1)
            data[0]['adj'].at[newstart,'KEEP']=100.
            data[0]['adj'].at[oldstart,'KEEP']=80.
            data[0]['adj'].at[end,'KEEP']=100.+delta
            values,_=run(data,2)
            self.assertEqual(values['candidate_eligible'][1,0],expected)
        data=fixture();_,oldstart,end,_=endpoints(data[1],data[2],1)
        data[0]['adj'].at[oldstart,'ADD']=data[0]['adj'].at[end,'ADD']
        values,_=run(data,2)
        self.assertEqual(values['momentum210'][1,1],0.)
        self.assertTrue(values['candidate_eligible'][1,1])

    def test_source_momentum_matches_binary64_order(self):
        values,_=run(fixture())
        np.testing.assert_array_equal(values['momentum210'][1,:3],[120./100.-1.,100./120.-1.,100./80.-1.])
        np.testing.assert_array_equal(values['momentum252'][1,:3],[120./80.-1.,100./80.-1.,100./120.-1.])
        self.assertTrue(np.isnan(values['momentum210'][:,3]).all())
        self.assertTrue(np.isnan(values['momentum252'][:,3]).all())

    def test_exact_fifteen_counts_and_official_dates(self):
        data=fixture();_,audit=run(data)
        expected=dict(events=4,symbols_per_event=4,checked_cells=16,original_eligible_cells=8,
            calm_events=1,active_events=2,required_cells=8,warm_cells=6,warm_observations=2360,
            source_price_observations=24,added_cells=2,rejected_cells=2,candidate_eligible_cells=8,
            new_model_fits=0,predictive_model_fits=0)
        self.assertEqual({key:audit[key] for key in k.COUNT_FIELDS},expected)
        for i in (1,3):
            event=audit['event_history'][i];t=data[2].index(data[1]['dates'][i])
            self.assertEqual(event['source_positions'],[t-294,t-252,t-42,t])
            self.assertEqual(event['source_dates'],[data[2][j] for j in event['source_positions']])
            self.assertEqual(event['required_indices'],[0,1,2,3])
            self.assertEqual(event['warm_indices'],[0,1,2])
        json.dumps(audit,allow_nan=False)

    def test_etfs_and_ineligible_sources_still_required(self):
        data=fixture();first,first_audit=run(data)
        data[1]['etfs']=['KEEP','DROP'];second,second_audit=run(data)
        self.assert_values(first,second);self.assertEqual(first_audit,second_audit)
        data[0]['adj']=data[0]['adj'].drop(columns='COLD')
        with self.assertRaisesRegex(ValueError,'Missing required security'):run(data)

    def test_power_two_unit_rescaling_preserves_results(self):
        data=fixture();first,audit=run(data)
        for symbol,scale in zip(data[0]['adj'].columns,(2.,.25,16.,8.)):data[0]['adj'][symbol]*=scale
        second,aa=run(data);self.assert_values(first,second);self.assertEqual(audit,aa)


class WarmTests(unittest.TestCase):
    assert_values=GateTests.assert_values

    def test_interior_notna_does_not_mean_finite_positive_or_numeric(self):
        for value in (np.inf,-np.inf,-100.,0.,True,'interior',1+2j):
            data=fixture();original,audit=run(data)
            data[0]['adj']=data[0]['adj'].astype(object)
            data[0]['adj'].at[pd.Timestamp('2016-05-03'),'KEEP']=value
            actual,aa=run(data);self.assert_values(original,actual);self.assertEqual(audit,aa)

    def test_any_missing_value_is_warm_failure_not_silent_new_filter(self):
        for bad in (np.nan,None,pd.NA):
            data=fixture();data[0]['adj']=data[0]['adj'].astype(object)
            data[0]['adj'].at[pd.Timestamp('2016-05-03'),'ADD']=bad
            values,audit=run(data,2)
            self.assertFalse(values['warm'][1,1]);self.assertFalse(values['candidate_eligible'][1,1])
            self.assertTrue(values['required'][1,1]);self.assertTrue(np.isnan(values['momentum210'][1,1]))
            self.assertEqual(audit['source_price_observations'],8)

    def test_missing_anywhere_bypasses_invalid_current_and_lag_numerics(self):
        data=fixture();data[0]['adj']=data[0]['adj'].astype(object)
        newstart,oldstart,end,current=endpoints(data[1],data[2],1)
        data[0]['adj'].at[pd.Timestamp('2016-05-03'),'ADD']=np.nan
        for day,bad in zip((newstart,oldstart,end,current),('bad',False,-np.inf,np.inf)):
            data[0]['adj'].at[day,'ADD']=bad
        values,_=run(data,2)
        self.assertFalse(values['warm'][1,1]);self.assertTrue(np.isnan(values['momentum252'][1,1]))

    def test_current_missing_is_legitimate_nonwarm(self):
        data=fixture();current=endpoints(data[1],data[2],1)[3]
        data[0]['adj'].at[current,'ADD']=np.nan
        self.assertFalse(run(data,2)[0]['warm'][1,1])

    def test_finite_nonpositive_current_is_nonwarm(self):
        for value in (0.,-0.,-5.):
            data=fixture();current=endpoints(data[1],data[2],1)[3]
            data[0]['adj'].at[current,'ADD']=value
            values,_=run(data,2)
            self.assertFalse(values['warm'][1,1]);self.assertFalse(values['candidate_eligible'][1,1])
            self.assertTrue(np.isnan(values['momentum210'][1,1]))

    def test_allnotna_invalid_current_aborts(self):
        for value in (np.inf,-np.inf,True,np.bool_(False),'100',1+0j,10**400):
            data=fixture();data[0]['adj']=data[0]['adj'].astype(object)
            current=endpoints(data[1],data[2],1)[3];data[0]['adj'].at[current,'ADD']=value
            with self.subTest(value=str(value)[:25]):
                with self.assertRaises(ValueError):run(data,2)

    def test_warmtrue_invalid_lag_endpoint_aborts(self):
        for endpoint in (0,1,2):
            for value in (0.,-1.,np.inf,-np.inf,True,np.bool_(False),'100',1+0j,10**400):
                data=fixture();data[0]['adj']=data[0]['adj'].astype(object)
                day=endpoints(data[1],data[2],1)[endpoint];data[0]['adj'].at[day,'ADD']=value
                with self.subTest(endpoint=endpoint,value=str(value)[:25]):
                    with self.assertRaises(ValueError):run(data,2)

    def test_lag_nan_is_nonwarm_and_no_finite_positive_claim(self):
        for endpoint in (0,1,2):
            data=fixture();day=endpoints(data[1],data[2],1)[endpoint]
            data[0]['adj'].at[day,'ADD']=np.nan
            values,_=run(data,2)
            self.assertFalse(values['warm'][1,1]);self.assertTrue(np.isnan(values['momentum252'][1,1]))

    def test_original_eligibility_parity_both_signs_and_warm_failure(self):
        for symbol,old in (('KEEP',False),('ADD',True),('COLD',True)):
            data=fixture();j=data[1]['symbols'][1].index(symbol);data[3]['eligible'][1,j]=old
            with self.subTest(symbol=symbol):
                with self.assertRaisesRegex(ValueError,'Original eligible/'):run(data,2)
        data=fixture();current=endpoints(data[1],data[2],1)[3]
        data[0]['adj'].at[current,'KEEP']=0.
        with self.assertRaisesRegex(ValueError,'eligible/warm'):run(data,2)

    def test_derived_nonfinite_ratio_aborts_for_either_horizon(self):
        for endpoint in (0,1):
            data=fixture();dates=endpoints(data[1],data[2],1)
            data[0]['adj'].at[dates[endpoint],'ADD']=np.nextafter(0.,1.)
            data[0]['adj'].at[dates[2],'ADD']=np.finfo(float).max
            with self.assertRaises(FloatingPointError):run(data,2)

    def test_missing_or_extra_original_window_row_fails_without_reindex(self):
        for mode in ('interior','current','start','extra'):
            data=fixture();newstart,_,_,current=endpoints(data[1],data[2],1)
            if mode=='extra':
                extra=pd.DataFrame(100.,index=[pd.Timestamp('2016-05-07')],columns=data[0]['adj'].columns)
                data[0]['adj']=pd.concat([data[0]['adj'],extra]).sort_index()
            else:
                day={'interior':pd.Timestamp('2016-05-03'),'current':current,'start':newstart}[mode]
                data[0]['adj']=data[0]['adj'].drop(index=day)
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError,'all295 official'):run(data,2)

    def test_insufficient_history_fails_even_if_all_E_false(self):
        data=fixture();data[2][:]=data[2][-250:];data[3]['eligible'][:]=False
        with self.assertRaisesRegex(ValueError,'insufficient official history'):run(data,2)


class PrefixTests(unittest.TestCase):
    assert_values=GateTests.assert_values

    def test_every_logical_and_physical_prefix_including_empty(self):
        data=fixture();values,audit=run(data)
        for n in range(5):
            vv,aa=run(data,n);pp,pa=run(physical(data,n))
            self.assert_values({key:value[:n] for key,value in values.items()},vv)
            self.assert_values(vv,pp);self.assertEqual(aa,pa)
            self.assertEqual(aa['event_history'],audit['event_history'][:n]);self.assertEqual(aa['symbols_per_event'],4)

    def test_future_price_state_and_metadata_poison_not_consumed(self):
        data=fixture();expected,audit=run(data,2);query=pd.Timestamp(data[1]['dates'][1])
        data[0]['adj']=data[0]['adj'].astype(object);data[0]['adj'].loc[data[0]['adj'].index>query]='future'
        for key in data[3]:data[3][key]=data[3][key].astype(object);data[3][key][2:]='future'
        data[1]['dates'][2:]=['notdate','future'];data[1]['symbols'][2:]=[None,['bad']]
        actual,aa=run(data,2);self.assert_values(expected,actual);self.assertEqual(audit,aa)
        with self.assertRaises(ValueError):run(data)

    def test_before_warm_window_values_do_not_change_result(self):
        data=fixture();expected,audit=run(data,2);start=endpoints(data[1],data[2],1)[0]
        data[0]['adj']=data[0]['adj'].astype(object);data[0]['adj'].loc[data[0]['adj'].index<start]='before'
        actual,aa=run(data,2);self.assert_values(expected,actual);self.assertEqual(audit,aa)

    def test_nonmutation_and_detached_output(self):
        data=fixture();frame=data[0]['adj'].copy(deep=True);meta=copy.deepcopy(data[1]);arrays={k:v.copy() for k,v in data[3].items()}
        values,audit=run(data);pd.testing.assert_frame_equal(frame,data[0]['adj']);self.assertEqual(meta,data[1])
        values['original_eligible'][0]=False;values['candidate_eligible'][1]=False;audit['event_history'][0]['symbols'][0]='mutated'
        self.assertEqual(meta,data[1])
        for key in arrays:np.testing.assert_array_equal(arrays[key],data[3][key])

    def test_shapes_boolean_types_and_limits_rejected(self):
        for key,value in (('eligible',np.ones((4,4),int)),('eligible',np.ones(4,bool)),('eligible',np.empty((4,0),bool)),('state_calm10',np.ones((4,1),bool)),('state_calm10',np.ones(3,bool)),('state_calm10',np.ones(4,int))):
            data=fixture();data[3][key]=value
            with self.subTest(key=key,shape=value.shape):
                with self.assertRaises(ValueError):run(data)
        for limit in (-1,5,True,1.,'1'):
            with self.assertRaises(ValueError):run(fixture(),limit)

    def test_calendar_duplicates_order_canonical_and_monthend_witness(self):
        for mode in ('duplicate','reverse','weekend','timestamp','noncanonical','empty'):
            data=fixture()
            if mode=='duplicate':data[2][1]=data[2][0]
            elif mode=='reverse':data[2].reverse()
            elif mode=='weekend':data[2][1]='2015-01-03'
            elif mode=='timestamp':data[2][1]=pd.Timestamp(data[2][1])
            elif mode=='noncanonical':data[2][0]='2015-1-1'
            else:data[2].clear()
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):run(data)
        data=physical(fixture(),2);data[2].pop()
        with self.assertRaisesRegex(ValueError,'month-end'):run(data)

    def test_query_axes_symbols_and_dates_validation(self):
        for mode in ('date_count','symbol_count','width','duplicate','blank','event_order','not_monthend','noncanonical'):
            data=fixture();meta=data[1]
            if mode=='date_count':meta['dates'].pop()
            elif mode=='symbol_count':meta['symbols'].pop()
            elif mode=='width':meta['symbols'][1].pop()
            elif mode=='duplicate':meta['symbols'][1][1]='KEEP'
            elif mode=='blank':meta['symbols'][1][1]=' '
            elif mode=='event_order':meta['dates'][2]=meta['dates'][1]
            elif mode=='not_monthend':meta['dates'][1]='2016-08-30'
            else:meta['dates'][0]='2016-8-17'
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):run(data)

    def test_panel_axes_schema_and_columns_validation(self):
        for mode in ('extra','notframe','notdatetime','duplicate_date','reverse','timezone','time','duplicate_column','blank_column'):
            data=fixture();frame=data[0]['adj']
            if mode=='extra':data[0]['sectors']=frame
            elif mode=='notframe':data[0]['adj']=frame.to_numpy()
            elif mode=='notdatetime':frame.index=frame.index.astype(str)
            elif mode=='duplicate_date':frame.index=pd.DatetimeIndex([frame.index[0]]+frame.index[:-1].tolist())
            elif mode=='reverse':data[0]['adj']=frame.iloc[::-1]
            elif mode=='timezone':frame.index=frame.index.tz_localize('UTC')
            elif mode=='time':frame.index=frame.index+pd.Timedelta(hours=1)
            elif mode=='duplicate_column':frame.columns=['KEEP','KEEP','DROP','COLD']
            else:frame.columns=['KEEP','ADD','DROP',' ']
            with self.subTest(mode=mode):
                with self.assertRaises((ValueError,TypeError)):run(data)


if __name__=='__main__':
    unittest.main(verbosity=2)

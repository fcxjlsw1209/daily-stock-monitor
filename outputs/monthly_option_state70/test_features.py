"""Synthetic parsing, timing, coverage, and exact option-state contracts."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('option122_features_test',P/'features.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.sessions=pd.to_datetime(['2024-07-03','2024-07-05','2024-07-08','2024-07-09','2024-07-10'])
        self.events=self.sessions[1:4]
        self.left=pd.Series([20.,10.,15.,999.,999.],index=self.sessions)
        self.right=pd.Series([15.,15.,15.,1.,1.],index=self.sessions)
    def build(self,left=None,right=None,events=None,sessions=None):
        return f.build_features(self.left if left is None else left,self.right if right is None else right,
            self.events if events is None else events,self.sessions if sessions is None else sessions)
    def test_exact_previous_session_holiday_weekend_and_equality(self):
        a,b=self.build()
        self.assertEqual(set(a),set(f.ARRAY_KEYS))
        self.assertEqual(a['option_source_dates'].tolist(),['2024-07-03','2024-07-05','2024-07-08'])
        self.assertEqual(a['option_inverted'].tolist(),[True,False,False])
        self.assertEqual(a['option_available'].tolist(),[True,True,True])
        np.testing.assert_array_equal(a['option_ratio'],[20./15.,10./15.,1.])
        self.assertEqual(b['inverted_count'],1);self.assertEqual(b['available_count'],3)
        json.dumps(b,allow_nan=False)
    def test_equality_and_one_ulp_are_not_rounded(self):
        left=self.left.copy();right=self.right.copy()
        left.iloc[:3]=[15.,np.nextafter(15.,np.inf),np.nextafter(15.,-np.inf)]
        self.assertEqual(self.build(left,right)[0]['option_inverted'].tolist(),[False,True,False])
    def test_missing_either_source_no_stale_fill(self):
        for which in ('left','right'):
            source=getattr(self,which).drop(self.sessions[1])
            a,b=self.build(**{which:source})
            self.assertFalse(a['option_available'][1]);self.assertFalse(a['option_inverted'][1])
            self.assertTrue(np.isnan(a['option_ratio'][1]))
            self.assertEqual(a['option_source_dates'][1],'2024-07-05')
            self.assertEqual(b['rows'][1]['status'],'missing_source_row')
            self.assertEqual(b['rows'][1]['missing_symbols'],['VIX' if which=='left' else 'VIX3M'])
    def test_both_missing_and_empty_sources(self):
        a,b=self.build(self.left.iloc[:0],self.right.iloc[:0])
        self.assertFalse(a['option_available'].any());self.assertTrue(np.isnan(a['option_ratio']).all())
        self.assertEqual(b['rows'][0]['missing_symbols'],['VIX','VIX3M'])
        json.dumps(b,allow_nan=False)
    def test_bad_consumed_values_fail(self):
        for value in (0.,-1.,np.inf,-np.inf,np.nan,True,'garbage','',None):
            left=self.left.astype(object);left.iloc[0]=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.build(left)
    def test_bad_present_close_fails_even_when_other_missing(self):
        left=self.left.copy();left.iloc[0]=0.
        with self.assertRaises(ValueError):self.build(left,self.right.drop(self.sessions[0]))
    def test_same_day_and_future_values_are_ignored(self):
        original=self.build(events=self.events[:1])
        for value in (np.inf,np.nan,-1.,'garbage'):
            left=self.left.astype(object);right=self.right.astype(object)
            left.iloc[1:]=value;right.iloc[1:]=value
            a,b=self.build(left,right,events=self.events[:1])
            for key in a:np.testing.assert_array_equal(a[key],original[0][key])
            self.assertEqual(b,original[1])
    def test_prefix_output_exact(self):
        full,audit=self.build()
        for n in range(4):
            a,b=self.build(events=self.events[:n])
            for key in a:self.assertEqual(a[key].tobytes(),full[key][:n].tobytes())
            self.assertEqual(b['rows'],audit['rows'][:n])
    def test_inputs_unchanged(self):
        left,right=self.left.copy(),self.right.copy();events=self.events.copy();sessions=self.sessions.copy()
        self.build();pd.testing.assert_series_equal(left,self.left);pd.testing.assert_series_equal(right,self.right)
        self.assertTrue(events.equals(self.events));self.assertTrue(sessions.equals(self.sessions))
    def test_event_must_be_official_with_predecessor(self):
        for dates in (['2024-07-04'],['2024-07-03'],['2024-07-11']):
            with self.assertRaises(ValueError):self.build(events=dates)
    def test_invalid_date_axes(self):
        axes=[self.sessions[::-1],self.sessions[[0,0,1]],self.sessions.tz_localize('UTC'),
              self.sessions+pd.Timedelta(hours=1),pd.DatetimeIndex([pd.NaT])]
        for axis in axes:
            with self.subTest(axis=str(axis)),self.assertRaises(ValueError):self.build(sessions=axis)
            with self.subTest(axis=str(axis)),self.assertRaises(ValueError):self.build(events=axis)
            with self.subTest(axis=str(axis)),self.assertRaises(ValueError):self.build(left=pd.Series(np.ones(len(axis)),index=axis))
    def test_series_and_calendar_required(self):
        with self.assertRaises(ValueError):self.build(left=self.left.to_frame())
        with self.assertRaises(ValueError):self.build(sessions=[])
    def test_ratio_overflow_fails(self):
        left=self.left.copy();right=self.right.copy();left.iloc[0]=1e308;right.iloc[0]=1e-308
        with self.assertRaises(FloatingPointError):self.build(left,right)


class ParsingTests(unittest.TestCase):
    def text(self,rows):return 'DATE,OPEN,HIGH,LOW,CLOSE\n'+'\n'.join(rows)+'\n'
    def test_exact_schema_date_and_close_only(self):
        result=f.parse_history(self.text(['07/03/2024,nan,unused,bad,15.50','07/05/2024,1,2,0,garbage']),'VIX')
        self.assertEqual(result.name,'VIX');self.assertEqual(result.tolist(),['15.50','garbage'])
        self.assertEqual(result.index.tolist(),list(pd.to_datetime(['2024-07-03','2024-07-05'])))
    def test_schema_and_row_width_fail(self):
        for text in ('Date,OPEN,HIGH,LOW,CLOSE\n','DATE,CLOSE\n','DATE,OPEN,HIGH,LOW,CLOSE,EXTRA\n',
                     self.text(['07/03/2024,1,2,0']),self.text(['07/03/2024,1,2,0,1,extra'])):
            with self.subTest(text=text),self.assertRaises(ValueError):f.parse_history(text,'VIX')
    def test_dates_are_exact_canonical_and_valid(self):
        for date in ('7/03/2024','07/3/2024','2024-07-03','02/30/2024','07/03/2024 ','07/03/2024T16:15:00',''):
            with self.subTest(date=date),self.assertRaises(ValueError):f.parse_history(self.text([date+',1,2,0,1']),'VIX')
    def test_duplicate_and_unsorted_dates_fail(self):
        for dates in (['07/03/2024','07/03/2024'],['07/05/2024','07/03/2024']):
            with self.assertRaises(ValueError):f.parse_history(self.text([x+',1,2,0,1' for x in dates]),'VIX')
    def test_empty_schema_valid_but_invalid_inputs_fail(self):
        self.assertEqual(len(f.parse_history(self.text([]),'VIX')),0)
        for text,symbol in ((None,'VIX'),('',None),(self.text([]),'')):
            with self.assertRaises(ValueError):f.parse_history(text,symbol)
    def test_parsed_future_garbage_cannot_change_earlier_event(self):
        source=f.parse_history(self.text(['07/03/2024,1,2,0,20','07/05/2024,1,2,0,garbage']),'VIX')
        sessions=pd.to_datetime(['2024-07-03','2024-07-05'])
        right=pd.Series([15.,np.inf],index=sessions)
        a,_=f.build_features(source,right,sessions[1:],sessions)
        self.assertEqual(a['option_inverted'].tolist(),[True])


if __name__=='__main__':unittest.main()

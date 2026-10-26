"""Synthetic full-grid tests only; never opens source or market data."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
import unittest

import numpy as np

import monthly_annual_lag158_model as model


def fixture():
    # Synthetic weekdays and constant20:00 closes, not an exchange-calendar claim.
    start = date(2013, 1, 1)
    days = [(start+timedelta(days=i)).isoformat() for i in range(6200)
            if (start+timedelta(days=i)).weekday() < 5]
    closes = {day:model.timestamp_ns(day+'T20:00:00+00:00') for day in days}
    signals = [days[1000+i*20] for i in range(122)]
    names = ['KO','WMT']+['S%d'%i for i in range(30)]+['ETF%d'%i for i in range(17)]
    meta = dict(dates=signals, symbols=[names[:] for _ in signals], etfs=names[-17:])
    rows, events = [], {}
    source = np.zeros((122,49),bool); source[:,:2] = True
    eligible = np.ones((122,49),bool)
    for i, day in enumerate(signals):
        pos = days.index(day); cutoff = days[pos-42]
        # Independent same-year input pairs for every synthetic month; no freshness policy is added.
        yr = int(cutoff[:4])-1
        for j, symbol in enumerate(names):
            ids = []
            if j < 2:
                for old, report in ((False,date(yr,1,1)), (True,date(yr-1,1,1))):
                    key = '%s-%s-%s' % (i,j,old)
                    # KO current10/prior30 => change20; WMT current20/prior25 => change5.
                    lag = ((30 if old else 10) if j == 0 else (25 if old else 20))
                    events[key] = dict(event_id=key, cik=model.ISSUERS[symbol], form='10-K',
                        accessionNumber='%010d-%02d-%06d'%(j+1,(yr-int(old))%100,i*2+int(old)),
                        primaryDocument='annual.htm',reportDate=report.isoformat(),
                        filingDate=(report+timedelta(days=lag)).isoformat())
                    ids.append(key)
            row = dict(signal_index=i,candidate_index=j,symbol=symbol,signal_date=day,
                original_decision_cutoff=day+'T20:00:00+00:00', information_cutoff_date=cutoff,
                information_cutoff=cutoff+'T20:00:00+00:00', information_cutoff_ns=closes[cutoff],
                cik=model.ISSUERS.get(symbol),is_etf=j>=32,tier2=j<2,tier3=False,
                selected_event_ids=ids,known_amendment_event_ids=[],
                reasons=[model.MISSING_CONTEXT] if j<2 else ['NO_SOURCE'])
            rows.append(dict(original_row=row,body_pair_context_verified=j<2,
                body_context_source=symbol if j<2 else None,remaining_row_reasons=[] if j<2 else ['NO_SOURCE'],
                conditional_annual_pair_admitted=j<2,old_admission_flags_changed=False))
    return dict(meta=meta,rows=rows,event_metadata=events,source_mask=source,
                eligible=eligible,calendar=dict(sessions=days,closes=closes))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=fixture()

    def setUp(self):
        self.data=deepcopy(self.base)

    def build(self,n=1):
        return model.build(**self.data,n=n)

    def event(self,which=0,candidate=0):
        key=self.data['rows'][candidate]['original_row']['selected_event_ids'][which]
        return self.data['event_metadata'][key]

    def unavailable(self,reason):
        got=self.build(); self.assertFalse(got['arrays']['common_available'][0,0])
        self.assertTrue(np.isnan(got['arrays']['negative_current_lag'][0,0]))
        self.assertEqual(got['arrays']['current_lag_rank'][0,0],0.)
        self.assertIn(reason,got['audit']['cells'][0]['reasons'])

    def test_full_grid_values_and_fixed_dose(self):
        got=self.build(None); a=got['arrays']
        self.assertEqual(a['common_available'].shape,(122,49))
        np.testing.assert_array_equal(a['negative_current_lag'][:,:2],np.tile([-10.,-20.],(122,1)))
        np.testing.assert_array_equal(a['negative_lag_change'][:,:2],np.tile([20.,5.],(122,1)))
        np.testing.assert_array_equal(a['current_lag_rank'][:,:2],np.tile([.5,-.5],(122,1)))
        self.assertEqual(got['audit']['usable_rows'],244)
        self.assertTrue(np.isnan(a['negative_current_lag'][:,2:]).all())
        self.assertTrue((a['current_lag_rank'][:,2:]==0).all())
        self.assertEqual(model.DOSE,.01);self.assertEqual(model.MAX_ABSOLUTE_CORRECTION,.005)
        self.assertLessEqual(abs(model.DOSE*a['current_lag_rank']).max(),.005)

    def test_date_sort_not_input_order_or_filing_order(self):
        expected=self.build()['arrays']
        self.data['rows'][0]['original_row']['selected_event_ids'].reverse()
        actual=self.build()['arrays']
        for k in model.EXTRA:np.testing.assert_array_equal(actual[k],expected[k])

    def test_current_period_not_latest_filing(self):
        older=self.event(1);older['filingDate']=self.event()['filingDate']
        got=self.build();self.assertEqual(got['arrays']['negative_current_lag'][0,0],-10.)
        self.assertGreater(got['arrays']['negative_lag_change'][0,0],300.)

    def test_equal_periods_block(self):
        self.event(1)['reportDate']=self.event()['reportDate']
        self.event(1)['filingDate']=self.event()['filingDate']
        self.unavailable('SELECTED_ORIGINAL_PERIOD_OR_ACCESSION_COLLISION')

    def test_duplicate_ids_block(self):
        ids=self.data['rows'][0]['original_row']['selected_event_ids'];ids[1]=ids[0]
        self.unavailable('EXACT_TWO_SELECTED_ORIGINALS_REQUIRED')

    def test_extra_selected_original_blocks_no_older_fallback(self):
        self.data['rows'][0]['original_row']['selected_event_ids'].append('extra')
        self.unavailable('EXACT_TWO_SELECTED_ORIGINALS_REQUIRED')

    def test_invalid_latest_never_scans_older_catalog(self):
        self.data['event_metadata']['older-clean']=deepcopy(self.event(1))
        self.event()['reportDate']='not-a-date'
        self.unavailable('SELECTED_ORIGINAL_DATE_INVALID')

    def test_strict_invalid_dates(self):
        for bad in ('2020-02-30','2020-2-01','2020-01-01T00:00:00Z',None,True):
            with self.subTest(bad=bad):
                self.data=deepcopy(self.base);self.event()['filingDate']=bad
                self.unavailable('SELECTED_ORIGINAL_DATE_INVALID')

    def test_negative_lag_is_not_zero(self):
        self.event()['filingDate']='2013-01-01'
        self.unavailable('NEGATIVE_CALENDAR_LAG')

    def test_true_zero_day_lag_is_valid(self):
        self.event()['filingDate']=self.event()['reportDate']
        got=self.build()['arrays']
        self.assertTrue(got['common_available'][0,0]);self.assertEqual(got['negative_current_lag'][0,0],0.)

    def test_leap_year_natural_days(self):
        self.event()['reportDate']='2016-02-28';self.event()['filingDate']='2016-03-01'
        got=self.build()['arrays'];self.assertEqual(got['negative_current_lag'][0,0],-2.)

    def test_wrong_row_cik(self):
        self.data['rows'][0]['original_row']['cik']=model.ISSUERS['WMT']
        self.unavailable('ROW_ISSUER_MISMATCH')

    def test_wrong_event_identity(self):
        self.event()['cik']=model.ISSUERS['WMT']
        self.unavailable('SELECTED_ORIGINAL_IDENTITY_OR_FORM_MISMATCH')

    def test_context_source_must_identify_this_issuer(self):
        for source in (None,'WMT'):
            with self.subTest(source=source):
                self.data=deepcopy(self.base);self.data['rows'][0]['body_context_source']=source
                self.unavailable('BODY_CONTEXT_ISSUER_MISMATCH')

    def test_original_etf_identity_alignment(self):
        self.data['meta']['etfs'].append('KO')
        with self.assertRaisesRegex(ValueError,'ETF identity alignment'):self.build()

    def test_missing_original_or_version_metadata_blocks(self):
        for key,value,reason in [('event_id','wrong','SELECTED_ORIGINAL_NOT_FOUND'),
                ('accessionNumber','bad','SELECTED_ORIGINAL_VERSION_METADATA_INVALID'),
                ('primaryDocument','','SELECTED_ORIGINAL_VERSION_METADATA_INVALID')]:
            with self.subTest(key=key):
                self.data=deepcopy(self.base);self.event()[key]=value
                self.unavailable(reason)

    def test_missing_context_or_tier2_never_admitted(self):
        self.data['rows'][0]['body_pair_context_verified']=False
        self.unavailable('SOURCE_OR_ANNUAL_CONTEXT_NOT_VERIFIED')
        self.data=deepcopy(self.base);self.data['rows'][0]['original_row']['tier2']=False
        self.unavailable('SOURCE_OR_ANNUAL_CONTEXT_NOT_VERIFIED')

    def test_transition_and_amendment_forms_block(self):
        for form in ('10-KT','10-K/A','10-Q'):
            with self.subTest(form=form):
                self.data=deepcopy(self.base);self.event()['form']=form
                self.unavailable('SELECTED_ORIGINAL_IDENTITY_OR_FORM_MISMATCH')

    def test_source_blocker_not_overridden_by_true_mask(self):
        self.data['rows'][0]['remaining_row_reasons']=['TIMING_BLOCKER']
        self.unavailable('SOURCE_BLOCKER:TIMING_BLOCKER')

    def test_original_blocker_cannot_be_silently_removed(self):
        self.data['rows'][0]['original_row']['reasons'].append('KNOWN_BAD_LATEST')
        self.unavailable('SOURCE_BLOCKER:KNOWN_BAD_LATEST')

    def test_known_amendment_blocks_even_without_reason(self):
        self.data['rows'][0]['original_row']['known_amendment_event_ids']=['amended']
        self.unavailable('KNOWN_SELECTED_PERIOD_AMENDMENT')

    def test_source_mask_cannot_expand_or_shrink_overlay(self):
        for j in (0,2):
            self.data=deepcopy(self.base);self.data['source_mask'][0,j]=not self.data['source_mask'][0,j]
            with self.assertRaisesRegex(ValueError,'Source mask'):self.build()

    def test_original_eligible_only_narrows(self):
        self.data['eligible'][0,0]=False
        self.unavailable('ORIGINAL_INELIGIBLE')
        got=self.build()['arrays'];self.assertTrue(got['common_available'][0,1])
        self.assertEqual(got['current_lag_rank'][0,1],0.)

    def test_ties_are_zero_independently_for_each_hand(self):
        other=self.event(candidate=1);other['filingDate']=self.event()['filingDate']
        got=self.build()['arrays'];np.testing.assert_array_equal(got['current_lag_rank'],0.)
        np.testing.assert_array_equal(got['lag_change_rank'][0,:2],[.5,-.5])

    def test_all_raw_and_rank_ties_zero(self):
        for index in (0,1):
            other=self.event(index,candidate=1);other['filingDate']=self.event(index)['filingDate']
        got=self.build()['arrays']
        for key in model.RANK_NAMES:np.testing.assert_array_equal(got[key],0.)

    def test_initial_partial_is_nan_and_zero(self):
        day=model.INITIAL_PARTIAL;calendar=self.data['calendar'];pos=calendar['sessions'].index(day)
        cutoff=calendar['sessions'][pos-42];self.data['meta']['dates'][0]=day
        for overlay in self.data['rows'][:49]:
            r=overlay['original_row'];r.update(signal_date=day,original_decision_cutoff=day+'T20:00:00Z',
                information_cutoff_date=cutoff,information_cutoff=cutoff+'T20:00:00Z',information_cutoff_ns=calendar['closes'][cutoff])
        self.unavailable('INITIAL_PARTIAL_EXACT_FALLBACK')
        self.assertEqual(self.build()['audit']['usable_rows'],0)

    def test_wrong_cutoff_date_or_ns_or_original_close(self):
        for key,value in [('information_cutoff_date','2013-01-01'),('information_cutoff_ns',1),
                          ('information_cutoff','2013-01-01T20:00:00Z'),('original_decision_cutoff','bad')]:
            with self.subTest(key=key):
                self.data=deepcopy(self.base);self.data['rows'][0]['original_row'][key]=value
                self.unavailable('ORIGINAL_CUTOFF_MISMATCH')

    def test_second_strict_session_boundary(self):
        row=self.data['rows'][0]['original_row'];days=self.data['calendar']['sessions']
        cut=days.index(row['information_cutoff_date'])
        self.event()['filingDate']=days[cut-2]
        self.assertTrue(self.build()['arrays']['common_available'][0,0])
        self.event()['filingDate']=days[cut-1]
        self.unavailable('SELECTED_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42')

    def test_filing_after_cutoff_blocks(self):
        self.event()['filingDate']=self.data['meta']['dates'][0]
        self.unavailable('SELECTED_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42')

    def test_grid_axis_mismatch_raises(self):
        for key,value in [('signal_index',True),('candidate_index',1),('symbol','MSFT'),('signal_date','2016-01-01')]:
            with self.subTest(key=key):
                self.data=deepcopy(self.base);self.data['rows'][0]['original_row'][key]=value
                with self.assertRaisesRegex(ValueError,'alignment'):self.build()

    def test_mask_shape_and_boolean_dtype_required(self):
        for key in ('source_mask','eligible'):
            self.data=deepcopy(self.base);self.data[key]=self.data[key].astype(int)
            with self.assertRaisesRegex(ValueError,'boolean'):self.build()

    def test_future_rows_metadata_events_and_masks_are_not_read(self):
        expected=self.build()
        self.data['rows'][49:]=[object()]*(5978-49)
        self.data['meta']['dates'][1:]=[object()]*121
        self.data['meta']['symbols'][1:]=[object()]*121
        self.data['eligible'][1:]=False;self.data['source_mask'][1:]=False
        for key in list(self.data['event_metadata']):
            if not key.startswith('0-'):self.data['event_metadata'][key]=object()
        actual=self.build()
        for key in model.EXTRA:np.testing.assert_array_equal(actual['arrays'][key],expected['arrays'][key])
        self.assertEqual(actual['audit'],expected['audit'])

    def test_all123_prefixes_match(self):
        expected=self.build(None)['arrays']
        for n in range(123):
            got=self.build(n)['arrays']
            for key in model.EXTRA:np.testing.assert_array_equal(got[key],expected[key][:n])

    def test_inputs_immutable(self):
        before=deepcopy(self.data);self.build(None)
        for key in ('source_mask','eligible'):np.testing.assert_array_equal(self.data[key],before[key])
        for key in ('meta','rows','event_metadata','calendar'):self.assertEqual(self.data[key],before[key])

    def test_invalid_prefix(self):
        for n in (-1,123,True,1.0):
            with self.assertRaises(ValueError):self.build(n)


class RankTests(unittest.TestCase):
    def test_exact_average_ties(self):
        raw=[1,1,3,4];expected=[float(-Fraction(1,3)),float(-Fraction(1,3)),float(Fraction(1,6)),.5]
        np.testing.assert_allclose(model.centered_rank(raw),expected,rtol=0,atol=1e-16)

    def test_empty_singleton_all_equal(self):
        for values in (np.array([],float),[8],[7,7]):
            np.testing.assert_array_equal(model.centered_rank(values),np.zeros(len(values)))

    def test_reject_nonfinite_and_non_numeric(self):
        for values in ([1,np.nan],[np.inf],[True],['1'],[[1]]):
            with self.assertRaises(ValueError):model.centered_rank(values)


if __name__=='__main__':
    unittest.main(verbosity=2)

"""Synthetic tests only: no actual source values, cache arrays or returns."""
import copy
import unittest
from datetime import date, timedelta
from fractions import Fraction

import numpy as np
import monthly_financial_coverage154_factor as f


def event(event_id='x', release='2020-02-03', available='2020-02-04', symbol='PLTR', duration=None, year=2019):
    source = dict(path='synthetic.html', sha256='a'*64)
    end = date(year, 12, 31)
    if duration is None:duration=(end-date(year,1,1)).days+1
    start = end - timedelta(days=duration-1)
    period = dict(period_start=start.isoformat(), period_end=end.isoformat(), days_inclusive=duration)
    fields = {}
    for name, value in zip(f.FIELDS, ('10', '120', '80')):
        fields[name] = dict(source=source.copy(), available_from=available,
            currency_literal='$', currency_iso=None, value_decimal=value, scale='1000000',
            as_of=(start-timedelta(days=1)).isoformat() if name == 'prior_total_assets' else end.isoformat(),
            period=period.copy() if name == 'annual_net_income' else None)
    return dict(event_id=event_id, symbol=symbol, source=source,
        release_date=release, available_from=available, fields=fields,
        identity=dict(issuer_id='synthetic-issuer', security_id=symbol, identity_valid_from=available,
                      identity_valid_through=(end+timedelta(days=550)).isoformat()),
        ni_assets_admission=dict(admitted=True, reasons=[], definition=f.DEFINITION))


def catalog(e):
    return dict({k:e[k] for k in ('event_id','symbol','release_date','available_from')},
                clock_review=dict(verified=True, first_pinned_session=e['available_from']))


def fixture():
    names=['AAPL','AMZN','INTC','KO','MU','PLTR','PYPL','TMO']+['S'+str(i) for i in range(24)]+['E'+str(i) for i in range(17)]
    meta=dict(dates=['2020-02-04','2020-03-03','2020-04-03'], symbols=[names.copy() for _ in range(3)], etfs=names[-17:])
    original=np.full((3,49),np.nan);original[:,0]=[-0.,.1,.2]
    original.view(np.uint64)[:,8]=np.uint64(0x7ff8000000000011)
    b1=original.copy();b1[:,1]=[-.2,0.,.3]
    b2=b1.copy();b2[:,2]=[.2,-.3,.4]
    b3=b2.copy();b3[:,3]=[-.4,.3,0.]
    b4=b3.copy();b4[:,4]=[.1,-.2,-0.]
    e=event();data=dict(schema_version=1,method=f.METHOD,events=[e],latest_event_catalog=[catalog(e)])
    return meta,[original,b1,b2,b3,b4],data


class FieldTests(unittest.TestCase):
    def test_three_issuers_and_normal_leap_calendar_years(self):
        for symbol in ('PLTR','PYPL','TMO'):
            for year in (2019,2020):
                with self.subTest(symbol=symbol,year=year):
                    signal=str(year+1)+'-02-04'
                    r=f.evaluate(event(symbol=symbol,year=year,release=str(year+1)+'-02-03',available=signal),signal)
                    self.assertTrue(r['available'])
                    self.assertEqual(r['earnings_bounded'],f.bounded(Fraction(1,10)))

    def test_52_53_week_or_other_duration_not_substituted(self):
        for days in (330,363,364,366,370,371,372,400):
            with self.subTest(days=days):
                self.assertEqual(f.evaluate(event(duration=days),'2020-02-04')['reason'],'invalid_annual_period')

    def test_inclusive_duration_must_match_actual_dates(self):
        e=event();e['fields']['annual_net_income']['period']['days_inclusive']=371
        self.assertEqual(f.evaluate(e,'2020-02-04')['reason'],'invalid_annual_period')

    def test_equal_length_shifted_year_is_not_calendar_year(self):
        e=event()
        e['fields']['annual_net_income']['period'].update(
            period_start='2018-12-31',period_end='2019-12-30')
        self.assertEqual(f.evaluate(e,'2020-02-04')['reason'],'invalid_annual_period')

    def test_positive_zero_and_negative_income_retained(self):
        for v in ('10','0','-10'):
            e=event();e['fields']['annual_net_income']['value_decimal']=v
            r=f.evaluate(e,'2020-02-04')
            self.assertTrue(r['available']);self.assertEqual(r['earnings_bounded'],f.bounded(Fraction(v)/100))

    def test_explicit_scales_normalize_exactly(self):
        e=event();e['fields']['annual_net_income'].update(value_decimal='10000',scale='1000')
        self.assertEqual(f.evaluate(e,'2020-02-04'),f.evaluate(event(),'2020-02-04'))

    def test_large_numbers_do_not_round_before_division(self):
        e=event();e['fields']['annual_net_income']['value_decimal']='0.000000000000000000001'
        e['fields']['current_total_assets']['value_decimal']='100000000000000000000.1'
        e['fields']['prior_total_assets']['value_decimal']='100000000000000000000.3'
        ratio=Fraction('0.000000000000000000001')/((Fraction('100000000000000000000.1')+Fraction('100000000000000000000.3'))/2)
        self.assertEqual(f.evaluate(e,'2020-02-04')['earnings_bounded'],f.bounded(ratio))

    def test_bad_values_dates_currency_and_source_are_unavailable(self):
        changes=[('prior_total_assets','as_of','2018-12-28'),
            ('current_total_assets','as_of','2019-12-27'),
            ('annual_net_income','as_of','2019-12-27'),
            ('current_total_assets','value_decimal','0'),('prior_total_assets','value_decimal','-1'),
            ('annual_net_income','value_decimal','NaN'),('annual_net_income','value_decimal',True),
            ('annual_net_income','scale','0'),('annual_net_income','scale','-1'),
            ('annual_net_income','available_from','2020-02-03'),
            ('annual_net_income','currency_literal','EUR'),('annual_net_income','currency_iso','USD'),
            ('annual_net_income','source',dict(path='other.html',sha256='a'*64)),
            ('annual_net_income','source',dict(path='synthetic.html',sha256='b'*64))]
        for name,key,value in changes:
            with self.subTest(name=name,key=key,value=value):
                e=event();e['fields'][name][key]=value
                self.assertFalse(f.evaluate(e,'2020-02-04')['available'])

    def test_null_field_and_extra_field_not_zero_filled(self):
        for name in f.FIELDS:
            e=event();e['fields'][name]=None
            self.assertFalse(f.evaluate(e,'2020-02-04')['available'])
        e=event();e['fields']['cash_flow']={}
        self.assertFalse(f.evaluate(e,'2020-02-04')['available'])

    def test_source_gate_cannot_be_inferred_from_complete_values(self):
        for gate in (None,dict(admitted=True,reasons=[],definition='other'),
                     dict(admitted=False,reasons=['hold'],definition=f.DEFINITION),
                     dict(admitted=True,reasons=['hold'],definition=f.DEFINITION)):
            e=event();e['ni_assets_admission']=gate
            self.assertFalse(f.evaluate(e,'2020-02-04')['available'])

    def test_identity_and_release_time(self):
        for key,value in [('issuer_id',''),('security_id','KO'),('identity_valid_from','2020-02-03'),
                          ('identity_valid_through','2025-01-01')]:
            e=event();e['identity'][key]=value
            self.assertFalse(f.evaluate(e,'2020-02-04')['available'])
        self.assertFalse(f.evaluate(event(),'2020-02-03')['available'])
        e=event(release='2019-12-01');self.assertFalse(f.evaluate(e,'2020-02-04')['available'])

    def test_staleness_boundary_550_days(self):
        e=event();end=date.fromisoformat(e['fields']['annual_net_income']['as_of'])
        self.assertTrue(f.evaluate(e,(end+timedelta(days=550)).isoformat())['available'])
        self.assertEqual(f.evaluate(e,(end+timedelta(days=551)).isoformat())['reason'],'stale_latest_event')


class LatestTests(unittest.TestCase):
    def test_latest_unprocessed_blocks_old(self):
        a,b=event(),event('y','2020-03-02','2020-03-03')
        self.assertEqual(f.latest([catalog(a),catalog(b)],{'x':a},'PLTR','2020-03-03')['reason'],
                         'latest_annual_event_unprocessed')

    def test_latest_held_or_null_blocks_old(self):
        for null in (False,True):
            a,b=event(),event('y','2020-03-02','2020-03-03')
            if null:b['fields']['annual_net_income']=None
            else:b['ni_assets_admission'].update(admitted=False,reasons=['accounting_hold'])
            self.assertFalse(f.latest([catalog(a),catalog(b)],{'x':a,'y':b},'PLTR','2020-03-03')['available'])

    def test_latest_unverified_clock_blocks_old_only_after_release_day(self):
        a,b=event(),event('y','2020-03-02','2020-03-03');r=catalog(b)
        r.update(available_from=None,clock_review=dict(verified=False,first_pinned_session=None))
        self.assertEqual(f.latest([catalog(a),r],{'x':a},'PLTR','2020-03-02')['event_id'],'x')
        self.assertEqual(f.latest([catalog(a),r],{'x':a},'PLTR','2020-03-03')['reason'],
                         'latest_annual_event_clock_unverified')

    def test_future_event_does_not_change_past(self):
        a,b=event(),event('y','2020-03-02','2020-03-03')
        self.assertEqual(f.latest([catalog(a),catalog(b)],{'x':a,'y':b},'PLTR','2020-03-02'),
                         f.latest([catalog(a)],{'x':a},'PLTR','2020-03-02'))

    def test_duplicate_release_is_ambiguous(self):
        a,b=event(),event('y')
        self.assertEqual(f.latest([catalog(a),catalog(b)],{'x':a,'y':b},'PLTR','2020-02-04')['reason'],
                         'ambiguous_latest_annual_event')

    def test_catalog_clock_source_release_mismatch_hard_fails(self):
        for key,value in [('release_date','2020-02-02'),('available_from','2020-02-05'),
                          ('symbol','PYPL')]:
            e=event();r=catalog(e);r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                f.latest([r],{'x':e},r['symbol'],'2020-03-03')


class PreservationTests(unittest.TestCase):
    def test_all_three_new_issuers_share_the_same_definition(self):
        meta,arrays,data=fixture()
        for symbol in ('PYPL','TMO'):
            e=event(symbol,symbol=symbol)
            data['events'].append(e);data['latest_event_catalog'].append(catalog(e))
        result=f.build(meta,*arrays,data)
        for j in (5,6,7):
            self.assertTrue(result['arrays']['expanded_available'][:,j].all())
            np.testing.assert_array_equal(result['arrays']['earnings_expanded'][:,j],
                                          np.full(3,f.bounded(Fraction(1,10))))
        self.assertEqual(result['audit']['added_cells'],9)

    def test_five_controls_and_all_nonnew_cells_byte_exact(self):
        meta,arrays,data=fixture();before=[a.tobytes() for a in arrays];before_data=copy.deepcopy(data)
        r=f.build(meta,*arrays,data);out=r['arrays']
        for name,a in zip(('original','batch1','batch2','batch3','batch4'),arrays):
            self.assertEqual(out['earnings_'+name].tobytes(),a.tobytes())
        for j,s in enumerate(meta['symbols'][0]):
            if s not in f.NEW_SYMBOLS:self.assertEqual(out['earnings_expanded'][:,j].tobytes(),arrays[4][:,j].tobytes())
        self.assertEqual(len(out),12)
        for name in ('original','batch1','batch2','batch3','batch4','expanded'):
            np.testing.assert_array_equal(out[name+'_available'],np.isfinite(out['earnings_'+name]))
        self.assertEqual(before,[a.tobytes() for a in arrays]);self.assertEqual(data,before_data)
        self.assertEqual(r['audit']['added_cells'],3);self.assertTrue(r['audit']['batch4_input_unchanged'])

    def test_each_prior_may_change_only_its_own_new_issuers(self):
        for step,columns in ((1,(0,2,3,4,5)),(2,(0,1,3,4,5)),(3,(0,1,2,4,5)),(4,(0,1,2,3,5))):
            for col in columns:
                meta,arrays,data=fixture();arrays[step][0,col]=.9
                with self.subTest(step=step,col=col),self.assertRaises(ValueError):f.build(meta,*arrays,data)

    def test_changed_nan_payload_or_signed_zero_is_rejected(self):
        for step in (1,2,3,4):
            for col,value in ((8,0x7ff8000000000022),(0,0)):
                meta,arrays,data=fixture();arrays[step].view(np.uint64)[0,col]=value
                with self.subTest(step=step,col=col),self.assertRaises(ValueError):f.build(meta,*arrays,data)

    def test_cumulative_old_source_events_are_not_reevaluated(self):
        meta,arrays,data=fixture();expected=f.build(meta,*arrays,data)['arrays']
        for symbol in ('AMZN','CSCO','INTC','KLAC','KO','LRCX','MU','PEP'):
            e=event(symbol,symbol=symbol);e['fields']={}
            data['events'].append(e);data['latest_event_catalog'].append(catalog(e))
        actual=f.build(meta,*arrays,data)['arrays']
        for key in expected:self.assertEqual(actual[key].tobytes(),expected[key].tobytes())

    def test_all_prefixes_and_future_invalid_latest(self):
        meta,arrays,data=fixture();future=event('future','2020-03-02','2020-03-03')
        future['ni_assets_admission'].update(admitted=False,reasons=['accounting_hold'])
        data['events'].append(future);data['latest_event_catalog'].append(catalog(future))
        full=f.build(meta,*arrays,data)['arrays']
        self.assertTrue(full['expanded_available'][0,5]);self.assertFalse(full['expanded_available'][1:,5].any())
        for n in range(4):
            part=f.build(dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n]),*[a[:n] for a in arrays],data)['arrays']
            for key in full:self.assertEqual(part[key].tobytes(),full[key][:n].tobytes())

    def test_missing_new_source_leaves_full_prior_bytes(self):
        meta,arrays,data=fixture();data.update(events=[],latest_event_catalog=[])
        r=f.build(meta,*arrays,data)
        self.assertEqual(r['arrays']['earnings_expanded'].tobytes(),arrays[4].tobytes())
        self.assertEqual(r['audit']['added_cells'],0)

    def test_held_null_new_source_retained_and_unavailable(self):
        meta,arrays,data=fixture();e=data['events'][0]
        e['fields']=dict.fromkeys(f.FIELDS);e['ni_assets_admission'].update(admitted=False,reasons=['missing_fields'])
        before=copy.deepcopy(data);r=f.build(meta,*arrays,data)
        self.assertEqual(r['arrays']['earnings_expanded'].tobytes(),arrays[4].tobytes())
        self.assertEqual(data,before);self.assertEqual(r['audit']['added_cells'],0)

    def test_zero_is_valid_new_coverage(self):
        meta,arrays,data=fixture();data['events'][0]['fields']['annual_net_income']['value_decimal']='0'
        r=f.build(meta,*arrays,data)['arrays']
        self.assertTrue(r['expanded_available'][:,5].all());self.assertTrue((r['earnings_expanded'][:,5]==0).all())

    def test_old_issuer_reasons_and_other_universe_members(self):
        meta,arrays,data=fixture();rows=f.build(meta,*arrays,data)['audit']['cells'][:49]
        for j,name in enumerate(('original','batch1','batch2','batch3','batch4')):
            self.assertEqual(rows[j]['reason'],'preserved_'+name+'_coverage')
        self.assertEqual(rows[8]['reason'],'no_new_source_coverage');self.assertEqual(rows[-1]['reason'],'etf_not_applicable')

    def test_duplicates_unknown_issuer_and_wrong_schema_fail(self):
        for kind in ('event','catalog','issuer','method'):
            meta,arrays,data=fixture()
            if kind=='event':data['events'].append(data['events'][0])
            elif kind=='catalog':data['latest_event_catalog'].append(data['latest_event_catalog'][0])
            elif kind=='issuer':data['events'][0]['symbol']='UNKNOWN'
            else:data['method']='wrong'
            with self.subTest(kind=kind),self.assertRaises(ValueError):f.build(meta,*arrays,data)

    def test_each_control_requires_float64_shape_and_no_infinity(self):
        for step in range(5):
            for kind in ('dtype','shape','infinity'):
                meta,arrays,data=fixture()
                if kind=='dtype':arrays[step]=arrays[step].astype(np.float32)
                elif kind=='shape':arrays[step]=arrays[step][:,:48]
                else:arrays[step][0,0]=np.inf
                with self.subTest(step=step,kind=kind),self.assertRaises(ValueError):f.build(meta,*arrays,data)

    def test_universe_and_dates_require_original_structure(self):
        for kind in ('duplicate','etf','date'):
            meta,arrays,data=fixture()
            if kind=='duplicate':meta['symbols'][0][1]=meta['symbols'][0][0]
            elif kind=='etf':meta['etfs']=meta['etfs'][:-1]
            else:meta['dates']=meta['dates'][::-1]
            with self.subTest(kind=kind),self.assertRaises(ValueError):f.build(meta,*arrays,data)


if __name__=='__main__':
    unittest.main(verbosity=2)

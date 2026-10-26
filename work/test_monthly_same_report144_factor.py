"""Synthetic-only same-report timing, exact arithmetic and missingness tests."""
from copy import deepcopy
from datetime import date, timedelta
from decimal import localcontext
import json
import unittest
import numpy as np
import pandas as pd
import monthly_same_report144_factor as f


def amount(value, scale='1'):
    return dict(value=value, scale=scale, currency_literal='$')


def period(start, end, gp, assets='1000'):
    return dict(period_start=start, period_end=end,
                days_inclusive=(date.fromisoformat(end)-date.fromisoformat(start)).days+1,
                gross_profit=amount(gp), assets=amount(assets))


def event(identifier='current', symbol='S0'):
    return dict(event_id=identifier,symbol=symbol,issuer_id='issuer_'+symbol,security_id=symbol,
                method=f.METHOD,release_date='2026-01-30',available_from='2026-02-02',
                identity_valid_from='2026-02-02',identity_valid_through=str(date(2025,12,31)+timedelta(days=550)),
                admitted=True,rejection_codes=[],currency_key='same-report-dollar:'+identifier,
                same_report=dict(current=period('2025-01-01','2025-12-31','36500'),
                                 comparison=period('2024-01-01','2024-12-31','36600')))


def fixture(dates=None):
    names=['S'+str(i) for i in range(32)]+['E'+str(i) for i in range(17)]
    dates=dates or ['2026-03-02','2026-04-01']
    return (dict(dates=dates,symbols=[names[:] for _ in dates],etfs=names[32:]),
            dict(schema_version=1,method=f.METHOD,events=[event()]))


class ReportTests(unittest.TestCase):
    def evaluate(self,record=None,signal='2026-03-02'):
        return f.evaluate_report(event() if record is None else record,signal)
    def test_exact_duration_equality(self):
        got=self.evaluate();self.assertTrue(got['available']);self.assertEqual((got['current_days'],got['prior_days']),(365,366))
        self.assertEqual(got['gross_profit_growth_sign'],0.);self.assertEqual(got['gross_profitability_improvement_sign'],0.)
    def test_both_columns_have_current_report_availability(self):
        got=self.evaluate();self.assertEqual(got['current_event_id'],got['prior_event_id'])
        self.assertEqual(got['current_available_from'],got['prior_available_from']);self.assertEqual(got['prior_available_from'],'2026-02-02')
        self.assertEqual(got['method'],f.METHOD)
    def test_growth_and_efficiency_distinct(self):
        record=event();cur=record['same_report']['current'];cur['gross_profit']=amount('73000');cur['assets']=amount('3000')
        got=self.evaluate(record);self.assertEqual((got[f.EXTRA[0]],got[f.EXTRA[1]]),(.5,-.5))
        cur['gross_profit']=amount('18250');cur['assets']=amount('250');got=self.evaluate(record)
        self.assertEqual((got[f.EXTRA[0]],got[f.EXTRA[1]]),(-.5,.5))
    def test_negative_and_zero_profit_valid(self):
        for value in ['-18250','0','36500']:
            record=event();record['same_report']['comparison']['gross_profit']=amount('-36600')
            record['same_report']['current']['gross_profit']=amount(value)
            self.assertEqual(self.evaluate(record)[f.EXTRA[1]],.5)
    def test_independent_scales_normalized(self):
        record=event();cur,prev=record['same_report'].values()
        cur['gross_profit']=amount('.0365','1000000');prev['gross_profit']=amount('36.6','1000')
        cur['assets']=amount('1','1000');prev['assets']=amount('.001','1000000')
        got=self.evaluate(record);self.assertEqual((got[f.EXTRA[0]],got[f.EXTRA[1]]),(0.,0.))
    def test_arbitrary_precision_crossproducts(self):
        record=event();record['same_report']['current']=period('2025-01-01','2025-12-31','1.'+'0'*70+'1')
        record['same_report']['comparison']=period('2024-01-02','2024-12-31','1')
        with localcontext() as ctx:
            ctx.prec=3;got=self.evaluate(record);self.assertEqual(ctx.prec,3)
        self.assertEqual((got[f.EXTRA[0]],got[f.EXTRA[1]]),(.5,.5))
    def test_huge_scales_exact(self):
        record=event()
        for col in record['same_report'].values():
            for field in ['assets','gross_profit']:col[field]['scale']='1e1000'
        self.assertEqual(self.evaluate(record)[f.EXTRA[1]],0.)
    def test_nonpositive_assets_block_shared_mask(self):
        for col in ['current','comparison']:
            for value in ['0','-1','-0']:
                record=event();record['same_report'][col]['assets']['value']=value;got=self.evaluate(record)
                self.assertEqual(got['reason'],col+'_nonpositive_assets');self.assertIsNone(got[f.EXTRA[0]])
    def test_bad_amounts_scales_and_currencies_rejected(self):
        for field in ['assets','gross_profit']:
            for value in ['NaN','sNaN','Infinity',None,True,123.,' 1']:
                record=event();record['same_report']['current'][field]['value']=value
                self.assertEqual(self.evaluate(record)['reason'],'current_invalid_amount_or_scale')
        for value in ['0','-1','NaN',1,None]:
            record=event();record['same_report']['comparison']['assets']['scale']=value
            self.assertEqual(self.evaluate(record)['reason'],'comparison_invalid_amount_or_scale')
        record=event();record['same_report']['current']['gross_profit']['currency_literal']='USD'
        self.assertFalse(self.evaluate(record)['available'])
    def test_no_cross_report_currency_alias(self):
        record=event();record['currency_key']='USD';self.assertEqual(self.evaluate(record)['reason'],'invalid_same_report_currency_key')
    def test_duration_bounds(self):
        for dc,dp in [(330,330),(400,400),(364,371),(371,364)]:
            end=date(2025,12,31);start=end-timedelta(days=dc-1);pe=start-timedelta(days=1);ps=pe-timedelta(days=dp-1)
            record=event();record['same_report']=dict(current=period(str(start),str(end),str(dc)),comparison=period(str(ps),str(pe),str(dp)))
            got=self.evaluate(record);self.assertTrue(got['available']);self.assertEqual(got[f.EXTRA[1]],0.)
        for dc in [329,401]:
            record=event();record['same_report']['current']=period(str(date(2025,12,31)-timedelta(days=dc-1)),'2025-12-31','1')
            self.assertEqual(self.evaluate(record)['reason'],'current_nonannual_duration')
    def test_declared_duration_must_match(self):
        for value in [True,366,None,'365']:
            record=event();record['same_report']['current']['days_inclusive']=value
            self.assertEqual(self.evaluate(record)['reason'],'current_duration_mismatch')
    def test_gap_and_overlap_rejected(self):
        for start in ['2025-01-02','2024-12-31']:
            record=event();record['same_report']['current']=period(start,'2025-12-31','1')
            self.assertEqual(self.evaluate(record)['reason'],'noncontiguous_annual_periods')
    def test_stale_boundary(self):
        end=date(2025,12,31)
        self.assertTrue(self.evaluate(signal=str(end+timedelta(days=550)))['available'])
        self.assertEqual(self.evaluate(signal=str(end+timedelta(days=551)))['reason'],'stale_current_report')
    def test_no_use_before_current_availability(self):
        self.assertEqual(self.evaluate(signal='2026-01-30')['reason'],'report_not_yet_available')
    def test_release_and_period_chronology(self):
        record=event();record['release_date']=record['available_from'];self.assertEqual(self.evaluate(record)['reason'],'availability_not_after_release')
        record=event();record['release_date']='2025-12-30';self.assertEqual(self.evaluate(record)['reason'],'release_before_period_end')
    def test_admission_and_method_explicit(self):
        for value in [False,None,1,'true']:
            record=event();record['admitted']=value;self.assertFalse(self.evaluate(record)['available'])
        record=event();record['method']='original_annual';self.assertEqual(self.evaluate(record)['reason'],'wrong_comparison_method')
    def test_identity_interval_inclusive(self):
        record=event();self.assertTrue(self.evaluate(record,'2026-02-02')['available'])
        record['identity_valid_from']='2026-03-02'
        self.assertEqual(self.evaluate(record)['reason'],'identity_start_not_current_availability')
        record=event();record['identity_valid_through']='2035-01-01'
        self.assertEqual(self.evaluate(record)['reason'],'identity_end_not_causal_expiry')
        for field in ['issuer_id','security_id','identity_valid_from','identity_valid_through']:
            record=event();del record[field];self.assertFalse(self.evaluate(record)['available'])
    def test_missing_columns_and_bad_period_neutral(self):
        record=event();del record['same_report']['comparison'];self.assertFalse(self.evaluate(record)['available'])
        record=event();record['same_report']['comparison']=None;self.assertFalse(self.evaluate(record)['available'])


class BuilderTests(unittest.TestCase):
    def test_zero_length_prefix_and_every_factor_prefix(self):
        meta,data=fixture([str(d.date()) for d in pd.date_range('2016-08-31',periods=122,freq='BME')])
        full=f.build_features(meta,data)
        for n in range(123):
            short=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
            prefix=f.build_features(short,data)
            for key in f.EXTRA:np.testing.assert_array_equal(prefix['arrays'][key],full['arrays'][key][:n])
            self.assertEqual(prefix['audit']['cells'],full['audit']['cells'][:n*49])
    def test_wrong_security_mapping_is_not_a_new_issuer(self):
        meta,data=fixture();data['events'][0]['security_id']='OTHER'
        self.assertEqual(f.build_features(meta,data)['audit']['cells'][0]['reason'],'security_symbol_mismatch')
    def test_axes_shared_masks_and_nonmutation(self):
        meta,data=fixture();other=event('other','S1');other['same_report']['current']['gross_profit']=amount('73000');data['events'].append(other)
        meta['symbols'][1][0],meta['symbols'][1][1]=meta['symbols'][1][1],meta['symbols'][1][0]
        saved=deepcopy((meta,data));got=f.build_features(meta,data)
        np.testing.assert_array_equal(got['arrays'][f.EXTRA[0]][:,:2],[[0.,.5],[.5,0.]])
        for key in f.EXTRA[:-1]:np.testing.assert_array_equal(np.isfinite(got['arrays'][key]),got['arrays'][f.EXTRA[-1]])
        self.assertEqual((meta,data),saved);self.assertEqual(got['audit']['common_available_cells'],4)
        self.assertEqual(len(got['audit']['cells']),98);json.dumps(got['audit'],allow_nan=False)
    def test_full_shape_without_any_source(self):
        meta,data=fixture([str(d.date()) for d in pd.date_range('2016-08-31',periods=122,freq='BME')]);data['events']=[]
        got=f.build_features(meta,data);self.assertEqual(got['arrays'][f.EXTRA[0]].shape,(122,49));self.assertEqual(len(got['audit']['cells']),5978)
        self.assertFalse(got['arrays'][f.EXTRA[-1]].any())
    def test_etf_source_never_applied(self):
        meta,data=fixture();data['events'].append(event('etf','E0'));got=f.build_features(meta,data)
        self.assertTrue(np.isnan(got['arrays'][f.EXTRA[0]][:,32:]).all())
    def test_latest_rejected_blocks_older_valid(self):
        meta,data=fixture();data['events'].append(dict(event_id='rejected',symbol='S0',available_from='2026-02-20',admitted=False))
        got=f.build_features(meta,data);self.assertFalse(got['arrays'][f.EXTRA[-1]][:,0].any())
        self.assertEqual(got['audit']['cells'][0]['current_event_id'],'rejected');self.assertEqual(got['audit']['cells'][0]['reason'],'current_not_admitted')
    def test_latest_expired_identity_does_not_fallback(self):
        meta,data=fixture();later=event('expired');later['available_from']='2026-02-20';later['identity_valid_through']='2026-02-21';data['events'].append(later)
        self.assertEqual(f.build_features(meta,data)['audit']['cells'][0]['reason'],'identity_start_not_current_availability')
    def test_future_poison_no_effect_on_prefix(self):
        meta,data=fixture(['2026-03-02']);before=f.build_features(meta,data)
        future=dict(event_id='future',symbol='S0',available_from='2026-03-20',admitted=False,same_report={'poison':'future'});data['events'].append(future)
        after=f.build_features(meta,data);self.assertEqual(before['audit'],after['audit'])
        for key in f.EXTRA:np.testing.assert_array_equal(before['arrays'][key],after['arrays'][key])
        meta['dates'].append('2026-04-01');meta['symbols'].append(meta['symbols'][0][:]);later=f.build_features(meta,data)
        self.assertFalse(later['arrays'][f.EXTRA[-1]][1,0])
    def test_same_date_competing_reports_ambiguous(self):
        meta,data=fixture();data['events'].append(event('competing'))
        self.assertEqual(f.build_features(meta,data)['audit']['cells'][0]['reason'],'current_ambiguous_latest_event')
    def test_duplicates_and_conflicts(self):
        meta,data=fixture();data['events'].append(deepcopy(data['events'][0]))
        self.assertEqual(f.build_features(meta,data)['audit']['common_available_cells'],2)
        data['events'][1]['same_report']['current']['assets']['value']='2000'
        self.assertEqual(f.build_features(meta,data)['audit']['cells'][0]['reason'],'conflicting_event_id')
    def test_event_order_invariant(self):
        meta,data=fixture();data['events'].append(dict(event_id='old',symbol='S0',available_from='2025-02-01',admitted=False))
        before=f.build_features(meta,data);data['events'].reverse();self.assertEqual(before['audit'],f.build_features(meta,data)['audit'])
    def test_missing_prior_release_is_not_required(self):
        meta,data=fixture();self.assertEqual(len(data['events']),1);self.assertEqual(f.build_features(meta,data)['audit']['common_available_cells'],2)
    def test_bad_chronology_fails_explicitly(self):
        for value in [None,'2026-02-01T00:00Z','2026-02-30']:
            meta,data=fixture();data['events'][0]['available_from']=value
            with self.assertRaises(ValueError):f.build_features(meta,data)
    def test_bad_schema_and_axes(self):
        for value in [True,'1',2]:
            meta,data=fixture();data['schema_version']=value
            with self.assertRaises(ValueError):f.build_features(meta,data)
        for dates in [['2026-04-01','2026-03-02'],['2026-03-02','2026-03-02'],['2026-03-02T00:00Z']]:
            meta,data=fixture(dates)
            with self.assertRaises(ValueError):f.build_features(meta,data)
        for mode in ['width','duplicate','etf']:
            meta,data=fixture()
            if mode=='width':meta['symbols'][0].pop()
            if mode=='duplicate':meta['symbols'][0][0]=meta['symbols'][0][1]
            if mode=='etf':meta['etfs'].pop()
            with self.assertRaises(ValueError):f.build_features(meta,data)

if __name__=='__main__':unittest.main()

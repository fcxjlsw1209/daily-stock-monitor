"""Synthetic-only independent oracle tests; no saved sources, arrays or prices."""
from copy import deepcopy
from datetime import date, timedelta
from fractions import Fraction
import json
import unittest
from unittest.mock import patch

import numpy as np
import monthly_roa_increment272_independent as q
import monthly_roa_increment272_factor as producer


class Poison:
    def __getitem__(self, key): raise AssertionError('Unconsumed body')
    def __iter__(self): raise AssertionError('Unconsumed body')
    def __len__(self): raise AssertionError('Unconsumed body')


def record(event):
    hashes = {p:q.digest(event['fields'][p.split('.')[0]][p.split('.')[1]]) for p in q.FIELDS}
    return dict(event_id=event['event_id'], increment_admitted=True, event_sha256=q.digest(event),
        fields={p:deepcopy(event['fields'][p.split('.')[0]][p.split('.')[1]]) for p in q.FIELDS},
        saved_scope_decision=dict(schema='roa_field_scope270_assessment_v2',
            definition='current_annual_net_income_average_assets_field_scope_v1', admitted=True, reasons=[],
            source_only=True, strict_pit=False, ratios_computed=False, chain_admitted=False,
            event_sha256=q.digest(event), required_cell_sha256=hashes))


def synthetic(dates=None):
    dates = ['2016-08-17', '2016-10-31'] if dates is None else dates
    current = dict(period_start='2015-01-01', period_end='2015-12-31', days_inclusive=365)
    comparison = dict(period_start=None, period_end='2014-12-31', days_inclusive=None)
    events = []
    for key,symbol,ni in zip(q.EVENT_IDS,('AAPL','AMD','MSFT'),('5','-7','0')):
        src=dict(path='synthetic/'+symbol+'.html',sha256='a'*64,
            receipt=dict(path='synthetic/'+symbol+'.meta.json',sha256='b'*64),url='https://example.invalid/'+symbol)
        e=dict(event_id=key,symbol=symbol,source=src,release_date='2016-01-04',available_from='2016-01-05',
            periods=dict(current=deepcopy(current),comparison=deepcopy(comparison)),
            identity=dict(issuer_id=symbol+':Synthetic',security_id=symbol,identity_valid_from='2016-01-05',
                identity_valid_through=(date(2015,12,31)+timedelta(days=550)).isoformat()),
            fields=dict(current={},comparison={}))
        for p,value in zip(q.FIELDS,(ni,'100','200')):
            side,name=p.split('.');period=deepcopy(current if side=='current' else comparison)
            e['fields'][side][name]=dict(value_decimal=value,scale='1000000',currency_literal='$',currency_iso=None,
                source=deepcopy(src),available_from=e['available_from'],period=period,
                locator=dict(period_end=period['period_end'],ni_scope='indirect_method_consolidated_start'))
        events.append(e)
    accepted=[record(e) for e in events]
    catalog=deepcopy(events)
    for n in range(41):catalog.append(dict(event_id='SYN:'+str(n),symbol='OTHER',available_from='2020-01-01',
        fields=Poison(),periods=Poison(),identity=Poison()))
    hand=np.full((len(dates),5),np.nan);hand[:,3]=-0.0
    hand.view(np.uint64)[:,4]=0x7ff80000000000ab
    start=date(2014,1,1)
    return dict(meta=dict(dates=dates,symbols=[['AAPL','AMD','MSFT','OTHER','ETF'] for _ in dates],etfs=['ETF']),
        baseline_hand=hand,baseline_mask=np.isfinite(hand),full_catalog=catalog,accepted_records=accepted,
        sessions=[(start+timedelta(days=i)).isoformat() for i in range(2000)])


def rebind(d, j):d['accepted_records'][j]=record(d['full_catalog'][j])


def verify(d, output):return q.verify_overlay(**d,actual_hand=output[0],actual_mask=output[1],audit=output[2])


def literal_pre(p,h,e):
    out=p.copy();active=e & np.isfinite(h) & (h!=0.)
    delta=np.zeros(p.shape);delta[active]=.01*h[active];out[active]+=delta[active]
    return out


class OverlayOracleTests(unittest.TestCase):
    def check(self,d):
        before=(d['baseline_hand'].tobytes(),d['baseline_mask'].tobytes())
        out=producer.overlay_prefix(**d)
        r=verify(d,out)
        self.assertTrue(r['passed']);self.assertEqual(r['counts'],out[2]['counts'])
        self.assertEqual(before,(d['baseline_hand'].tobytes(),d['baseline_mask'].tobytes()))
        return out,r
    def test_literal_signed_and_zero_values(self):
        d=synthetic();(h,m,a),r=self.check(d)
        self.assertEqual(h[0,:3].tolist(),[float(Fraction(1,31)),float(Fraction(-7,157)),0.])
        self.assertTrue(m[:,:3].all());self.assertEqual(a['counts']['zero_fills'],2)
        self.assertEqual(r['events'],2);self.assertEqual(r['width'],5)
        self.assertFalse(r['source_files_authenticated']);self.assertFalse(r['computes_returns'])
    def test_independent_does_not_call_producer_or_assessor(self):
        d=synthetic();out=producer.overlay_prefix(**d)
        with patch.object(producer,'overlay_prefix',side_effect=AssertionError('producer reuse')):
            with patch.object(producer.source,'assess',side_effect=AssertionError('source assess')):
                self.assertTrue(verify(d,out)['passed'])
    def test_different_scales_decimal_and_single_round(self):
        d=synthetic();e=d['full_catalog'][1]
        e['fields']['current']['net_income'].update(value_decimal='-.7e1',scale='10')
        e['fields']['current']['total_assets']['scale']='100'
        e['fields']['comparison']['total_assets']['scale']='.5'
        rebind(d,1);(h,_,_),_=self.check(d)
        self.assertEqual(h[0,1],float(Fraction(-140,10240)))
    def test_nan_payload_signedzero_unchanged(self):
        d=synthetic();(h,m,_),_=self.check(d)
        self.assertEqual(h[:,3:].tobytes(),d['baseline_hand'][:,3:].tobytes())
        self.assertEqual(m[:,3:].tobytes(),d['baseline_mask'][:,3:].tobytes())
    def test_all_physical_prefixes_and_empty_poison(self):
        d=synthetic(['2016-08-17','2016-10-31','2017-01-31']);full,_=self.check(d)
        for n in range(4):
            p=dict(d,meta=dict(d['meta'],dates=d['meta']['dates'][:n],symbols=d['meta']['symbols'][:n]),
                baseline_hand=d['baseline_hand'][:n].copy(),baseline_mask=d['baseline_mask'][:n].copy())
            if n==0:p.update(full_catalog=Poison(),accepted_records=Poison(),sessions=Poison())
            out,_=self.check(p)
            self.assertEqual(out[0].tobytes(),full[0][:n].tobytes());self.assertEqual(out[1].tobytes(),full[1][:n].tobytes())
    def test_future_nonselected_held_and_ambiguous_bodies(self):
        for mode in ('future','newest_held','tie'):
            d=synthetic(['2016-08-17']);e=d['full_catalog'][0]
            if mode=='future':e['available_from']='2017-01-01'
            else:d['full_catalog'][3].update(symbol='AAPL',available_from='2016-01-05' if mode=='tie' else '2016-07-01')
            e['fields']=Poison();d['accepted_records'][0]['fields']=Poison()
            (h,m,a),_=self.check(d);self.assertFalse(m[0,0])
            self.assertEqual(a['cells'][0]['reason'],{'future':'no_known_event','newest_held':'latest_not_increment','tie':'ambiguous_latest'}[mode])
    def test_day550_and551_then_stale_body_not_read(self):
        for age in (550,551):
            d=synthetic([(date(2015,12,31)+timedelta(days=age)).isoformat()])
            if age==551:
                for j in range(3):d['full_catalog'][j]['fields']=Poison()
            (_,m,a),_=self.check(d)
            self.assertEqual(bool(m[0,0]),age==550)
    def test_live_baseline_conflict_before_financial_body(self):
        d=synthetic();d['baseline_hand'][0,0]=.1;d['baseline_mask'][0,0]=True
        d['full_catalog'][0]['fields']=Poison()
        with self.assertRaisesRegex(ValueError,'Deduplication'):
            verify(d,(d['baseline_hand'],d['baseline_mask'],{}))
    def test_bad_clock_identity_and_duration_before_body(self):
        changes=(lambda e:e.update(available_from='2016-01-06'),lambda e:e['identity'].update(security_id='OTHER'),
            lambda e:e['periods']['current'].update(days_inclusive=364),
            lambda e:e['periods']['comparison'].update(period_end='2014-12-30'))
        for change in changes:
            d=synthetic();change(d['full_catalog'][0]);d['full_catalog'][0]['fields']=Poison()
            with self.assertRaises(ValueError):verify(d,(d['baseline_hand'],d['baseline_mask'],{}))
    def test_original_output_byte_or_mask_tamper(self):
        d=synthetic();out=producer.overlay_prefix(**d)
        for kind in ('newvalue','oldzero','nanpayload','mask'):
            bad=(out[0].copy(),out[1].copy(),deepcopy(out[2]))
            if kind=='newvalue':bad[0][0,0]=np.nextafter(bad[0][0,0],np.inf)
            elif kind=='oldzero':bad[0][0,3]=0.
            elif kind=='nanpayload':bad[0].view(np.uint64)[0,4]=0x7ff80000000000ac
            else:bad[1][0,0]=False
            with self.subTest(kind=kind),self.assertRaises(ValueError):verify(d,bad)
    def test_audit_counts_hash_reason_and_claim_tamper(self):
        d=synthetic();out=producer.overlay_prefix(**d)
        changes=(lambda a:a['counts'].update(required_cell_checks=1),lambda a:a['cells'][0].update(reason='baseline_preserved'),
            lambda a:a.update(source_files_authenticated=True),lambda a:a['cells'][0].update(ratio_fraction=[1,1]),
            lambda a:a['cells'][0]['required_cell_sha256'].update({'current.net_income':'0'*64}))
        for change in changes:
            bad=(out[0],out[1],deepcopy(out[2]));change(bad[2])
            with self.assertRaises(ValueError):verify(d,bad)
    def test_audit_json_key_order_not_material(self):
        d=synthetic();out=producer.overlay_prefix(**d)
        ordered=json.loads(json.dumps(out[2],sort_keys=True))
        self.assertTrue(verify(d,(out[0],out[1],ordered))['passed'])
    def test_selected_binding_invalid_amount_scope_and_currency(self):
        changes=(lambda e:e['fields']['current']['total_assets'].update(value_decimal='0'),
            lambda e:e['fields']['current']['net_income'].update(scale='-1'),
            lambda e:e['fields']['current']['net_income']['locator'].update(ni_scope='continuing_operations'),
            lambda e:e['fields']['comparison']['total_assets'].update(currency_iso='USD'))
        for change in changes:
            d=synthetic();change(d['full_catalog'][0]);rebind(d,0)
            with self.assertRaises(ValueError):verify(d,(d['baseline_hand'],d['baseline_mask'],{}))
    def test_bad_shapes_counts_ids_and_decimal_forms(self):
        changes=(lambda d:d['full_catalog'].pop(),lambda d:d['accepted_records'].reverse(),
            lambda d:d['full_catalog'][3].update(event_id=q.EVENT_IDS[0]),lambda d:d['meta']['dates'].reverse(),
            lambda d:d.update(baseline_hand=d['baseline_hand'].astype(np.float32)),
            lambda d:d['baseline_mask'].__setitem__((0,0),True))
        for change in changes:
            d=synthetic();change(d)
            with self.assertRaises(ValueError):verify(d,(d['baseline_hand'],d['baseline_mask'],{}))
        for value in ('+1','NaN','Infinity',' 1','1/2',True):
            with self.assertRaises(ValueError):q.decimal_ratio(value)
    def test_availability_independent_E_and_no_factor_clipping(self):
        d=synthetic();d['meta']['eligible']=Poison();e=d['full_catalog'][0]
        e['fields']['current']['net_income']['value_decimal']='1500';rebind(d,0)
        (h,m,_),_=self.check(d)
        self.assertEqual(h[0,0],float(Fraction(10,11)));self.assertTrue(m[0,0])


class PreScoreOracleTests(unittest.TestCase):
    def test_literal_operation_and_work(self):
        p=np.array([[.1,-0.,.3,.4,.5]],dtype=np.float64)
        h=np.array([[.5,-0.,np.nan,-.25,.25]])
        e=np.array([[True,True,True,True,False]])
        out=literal_pre(p,h,e);r=q.verify_pre_score(p,h,e,out)
        self.assertEqual(r['work'],dict(pre_score_verifications=1,cells_compared=5,eligible_delta_cells=2,hand_bound_cells=4,new_model_fits=0))
        self.assertEqual(out[0,1].tobytes(),p[0,1].tobytes())
    def test_double_correction_wrong_dose_and_wrong_E_rejected(self):
        p=np.array([[.1,.2]]);h=np.array([[.25,.4]]);e=np.array([[True,False]])
        out=literal_pre(p,h,e)
        for bad in (literal_pre(out,h,e),p+.02*np.nan_to_num(h),literal_pre(p,h,np.ones_like(e))):
            with self.assertRaises(ValueError):q.verify_pre_score(p,h,e,bad)
    def test_hard_adapter_bound_even_Efalse(self):
        for value in (np.nextafter(.5,np.inf),-.5000000000000001,np.inf):
            p=np.array([[.1]]);h=np.array([[value]]);e=np.array([[False]])
            with self.assertRaises(ValueError):q.verify_pre_score(p,h,e,p.copy())
    def test_zero_missing_and_empty_prefix(self):
        for n in (0,2):
            p=np.zeros((n,3));p[:]=-0.;h=np.full((n,3),np.nan);h[:,0]=0.;h[:,1]=-0.;e=np.ones((n,3),bool)
            r=q.verify_pre_score(p,h,e,p.copy());self.assertEqual(r['work']['eligible_delta_cells'],0)
    def test_invalid_matrix_or_nonfinite_pre(self):
        p=np.zeros((1,2));h=np.zeros_like(p);e=np.ones((1,2),bool)
        for args in ((p.astype(np.float32),h,e,p),(p,h,e.astype(int),p),(p,h,e,p.astype(np.float32)),(p+np.nan,h,e,p)):
            with self.assertRaises(ValueError):q.verify_pre_score(*args)
    def test_exact_nextafter_rounding_and_all_input_bytes_unchanged(self):
        p=np.array([[np.nextafter(.1,np.inf),-.0,1e10]])
        h=np.array([[np.nextafter(.25,np.inf),-.5,np.nextafter(0.,1.)]])
        e=np.array([[True,True,True]]);before=[v.tobytes() for v in (p,h,e)]
        out=literal_pre(p,h,e);self.assertTrue(q.verify_pre_score(p,h,e,out)['passed'])
        self.assertEqual(before,[v.tobytes() for v in (p,h,e)])


if __name__=='__main__':unittest.main(verbosity=2)

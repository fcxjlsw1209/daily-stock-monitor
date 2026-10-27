"""Pure synthetic financial values/metadata; no actual caches or price reads."""
from copy import deepcopy
from datetime import date, timedelta
from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np
import monthly_roa_increment272_factor as f
from test_roa_field_scope270_v2 import fixture, review


class Poison:
    def __iter__(self): raise AssertionError('Unconsumed financial body')
    def __getitem__(self,key): raise AssertionError('Unconsumed financial body')
    def __len__(self): raise AssertionError('Unconsumed financial body')


def record(event):
    sessions=calendar()
    decision=f.source.assess(event,sessions,review(event))
    assert decision['admitted'],decision['reasons']
    return dict(event_id=event['event_id'],increment_admitted=True,
        event_sha256=f.source.canonical_sha256(event),
        fields={s+'.'+k:deepcopy(event['fields'][s][k]) for s,k in f.source.REQUIRED},
        saved_scope_decision=decision)


def calendar():
    start=date(2014,1,1)
    return [(start+timedelta(days=i)).isoformat() for i in range(5000)]


def data(dates=None):
    events=[]
    for kind,key in zip(('apple_shape','amd_shape','microsoft_shape'),f.EVENT_IDS):
        event,_=fixture(kind);event['event_id']=key;events.append(event)
    catalog=deepcopy(events)
    for i in range(41):
        catalog.append(dict(event_id='OTHER:'+str(i),symbol='OTHER',available_from='2020-01-01',
            fields=Poison(),periods=Poison(),identity=Poison()))
    meta=dict(dates=dates or ['2016-08-17','2016-10-31'],
        symbols=[['AAPL','AMD','MSFT','OTHER','ETF'] for _ in (dates or ['2016-08-17','2016-10-31'])],etfs=['ETF'])
    hand=np.full((len(meta['dates']),5),np.nan,dtype=np.float64);hand[:,3]=-0.0
    mask=np.isfinite(hand)
    return dict(meta=meta,baseline_hand=hand,baseline_mask=mask,full_catalog=catalog,
        accepted_records=[record(e) for e in events],sessions=calendar())


def rebind(d,index): d['accepted_records'][index]=record(d['full_catalog'][index])


def metrics(name,cagr=.5,sharpe=1.0,drawdown=-.2,early=.4,recent=.6):
    return dict(name=name,training=dict(cagr=cagr,sharpe=sharpe,max_drawdown=drawdown),
        early=dict(cagr=early),recent=dict(cagr=recent),later=Poison(),next_open=Poison())


class OverlayTests(unittest.TestCase):
    def test_exact_signed_rational_and_counts(self):
        d=data();h,m,a=f.overlay_prefix(**d)
        self.assertTrue(m[0,1]);self.assertLess(h[0,1],0)
        row=a['cells'][1];self.assertEqual(row['ratio_fraction'],[-7,120])
        self.assertEqual(row['bounded_fraction'],[-7,127]);self.assertEqual(h[0,1],float(Fraction(-7,127)))
        self.assertEqual(tuple(a),f.AUDIT_KEYS);self.assertEqual(tuple(a['counts']),f.COUNT_FIELDS)
        self.assertTrue(all(tuple(r)==f.CELL_KEYS for r in a['cells']))
        c=a['counts'];self.assertEqual(c['required_cell_checks'],3*c['filled_cells'])
        self.assertEqual(c['ratio_evaluations'],c['filled_cells']);self.assertEqual(c['bounded_evaluations'],c['filled_cells'])
        self.assertEqual(c['catalog_headers_checked'],44);self.assertEqual(c['new_model_fits'],0)
        self.assertFalse(a['source_files_authenticated']);self.assertFalse(a['computes_returns'])
    def test_different_positive_scales_exact_and_zero_available(self):
        d=data();e=d['full_catalog'][1]
        e['fields']['current']['net_income']['value_decimal']='0';rebind(d,1)
        h,m,a=f.overlay_prefix(**d);self.assertTrue(m[0,1]);self.assertEqual(h[0,1],0.)
        self.assertEqual(a['counts']['zero_fills'],2)
        e['fields']['current']['net_income']['value_decimal']='-7'
        e['fields']['current']['net_income']['scale']='10'
        e['fields']['current']['net_income']['locator']['scale']='10'
        e['fields']['comparison']['total_assets']['scale']='100'
        e['fields']['comparison']['total_assets']['locator']['scale']='100';rebind(d,1)
        h,_,_=f.overlay_prefix(**d)
        x=Fraction(-70)/((Fraction(120000000)+12000)/2)
        self.assertEqual(h[0,1],float(x/(1+abs(x))))
    def test_untouched_nan_payload_signedzero_and_input_bytes(self):
        d=data();d['baseline_hand'].view(np.uint64)[:,4]=0x7ff80000000000ab
        before=(d['baseline_hand'].tobytes(),d['baseline_mask'].tobytes())
        h,m,_=f.overlay_prefix(**d)
        self.assertEqual(h[:,4].view(np.uint64).tolist(),[0x7ff80000000000ab]*2)
        self.assertEqual(h[:,3].tobytes(),d['baseline_hand'][:,3].tobytes())
        self.assertEqual(before,(d['baseline_hand'].tobytes(),d['baseline_mask'].tobytes()))
    def test_finite_baseline_live_increment_conflict_precedes_body(self):
        d=data();d['baseline_hand'][0,1]=.1;d['baseline_mask'][0,1]=True
        d['full_catalog'][1]['fields']=Poison()
        with self.assertRaisesRegex(ValueError,'Deduplication conflict'):f.overlay_prefix(**d)
    def test_full_catalog_and_exact_three_required(self):
        for change in (lambda d:d['full_catalog'].pop(),lambda d:d['accepted_records'].pop(),
                       lambda d:d['accepted_records'].reverse(),lambda d:d['full_catalog'][3].update(event_id=f.EVENT_IDS[0])):
            d=data();change(d)
            with self.assertRaises(ValueError):f.overlay_prefix(**d)
    def test_future_and_nonselected_financial_bodies_never_read(self):
        d=data(['2016-08-17'])
        d['full_catalog'][0]['fields']=Poison();d['accepted_records'][0]['fields']=Poison()
        h,m,_=f.overlay_prefix(**d);self.assertFalse(m[0,0]);self.assertTrue(m[0,1])
        d['full_catalog'][3].update(symbol='AMD',available_from='2016-05-01')
        d['full_catalog'][1]['fields']=Poison();d['accepted_records'][1]['fields']=Poison()
        h,m,a=f.overlay_prefix(**d);self.assertFalse(m[0,1]);self.assertEqual(a['cells'][1]['reason'],'latest_not_increment')
    def test_newest_held_blocks_older_and_uses_available_not_release(self):
        d=data(['2016-10-31'])
        d['full_catalog'][3].update(symbol='AAPL',available_from='2016-10-27',release_date='2016-01-01')
        d['full_catalog'][0]['fields']=Poison();d['accepted_records'][0]['fields']=Poison()
        _,m,a=f.overlay_prefix(**d);self.assertFalse(m[0,0]);self.assertEqual(a['cells'][0]['event_id'],'OTHER:0')
    def test_ambiguous_latest_does_not_read_either_body(self):
        d=data(['2016-10-31']);d['full_catalog'][3].update(symbol='AAPL',available_from='2016-10-26')
        d['full_catalog'][0]['fields']=Poison();d['accepted_records'][0]['fields']=Poison()
        _,m,a=f.overlay_prefix(**d);self.assertFalse(m[0,0]);self.assertEqual(a['cells'][0]['reason'],'ambiguous_latest')
    def test_day550_available_day551_stale_without_body(self):
        end=date(2015,12,26)
        d=data([(end+timedelta(days=550)).isoformat()]);_,m,_=f.overlay_prefix(**d);self.assertTrue(m[0,1])
        d=data([(end+timedelta(days=551)).isoformat()]);d['full_catalog'][1]['fields']=Poison()
        _,m,a=f.overlay_prefix(**d);self.assertFalse(m[0,1]);self.assertEqual(a['cells'][1]['reason'],'stale_latest')
    def test_empty_prefix_no_catalog_body_and_partial_not_overridden(self):
        d=data();d.update(meta=dict(dates=[],symbols=[],etfs=['ETF']),baseline_hand=np.empty((0,5)),
            baseline_mask=np.empty((0,5),bool),full_catalog=Poison(),accepted_records=Poison(),sessions=Poison())
        h,m,a=f.overlay_prefix(**d);self.assertEqual(h.shape,(0,5));self.assertEqual(a['counts']['event_bindings'],0)
        d=data(['2016-08-17']);_,m,_=f.overlay_prefix(**d);self.assertTrue(m[0,1])
    def test_each_physical_prefix_exact_to_full(self):
        d=data(['2016-08-17','2016-10-31','2017-01-31'])
        full=f.overlay_prefix(**d)
        for n in range(4):
            p=dict(d,meta=dict(d['meta'],dates=d['meta']['dates'][:n],symbols=d['meta']['symbols'][:n]),
                baseline_hand=d['baseline_hand'][:n].copy(),baseline_mask=d['baseline_mask'][:n].copy())
            h,m,a=f.overlay_prefix(**p)
            self.assertEqual(h.tobytes(),full[0][:n].tobytes());self.assertEqual(m.tobytes(),full[1][:n].tobytes())
            self.assertEqual(a['cells'],full[2]['cells'][:n*5])
    def test_invalid_original_mask_and_axis_rejected(self):
        changes=(lambda d:d['baseline_mask'].__setitem__((0,0),True),lambda d:d['baseline_hand'].__setitem__((0,0),np.inf),
            lambda d:d.update(baseline_hand=d['baseline_hand'].astype(np.float32)),lambda d:d['meta']['dates'].reverse(),
            lambda d:d['meta']['symbols'][0].__setitem__(0,'AMD'),lambda d:d['sessions'].remove(d['meta']['dates'][0]))
        for change in changes:
            d=data();change(d)
            with self.assertRaises(ValueError):f.overlay_prefix(**d)
    def test_bad_calendar_identity_and_annual_metadata_fail_before_body(self):
        changes=(lambda e:e.update(available_from='2016-01-21'),lambda e:e['identity'].update(security_id='OTHER'),
            lambda e:e['identity'].update(identity_valid_through='2099-01-01'),
            lambda e:e['periods']['current'].update(days_inclusive=12),
            lambda e:e['periods']['comparison'].update(period_end='2014-12-26'))
        for change in changes:
            d=data();change(d['full_catalog'][1]);d['full_catalog'][1]['fields']=Poison()
            with self.assertRaises(ValueError):f.overlay_prefix(**d)
    def test_event_cell_or_admission_tamper_rejected(self):
        changes=(lambda d:d['accepted_records'][1].update(event_sha256='0'*64),
            lambda d:d['accepted_records'][1].update(increment_admitted=False),
            lambda d:d['accepted_records'][1]['fields']['current.net_income'].update(value_decimal='8'),
            lambda d:d['accepted_records'][1]['saved_scope_decision'].update(reasons=['unknown']))
        for change in changes:
            d=data();change(d)
            with self.assertRaises(ValueError):f.overlay_prefix(**d)
    def test_selected_required_fields_invalid_after_rebinding_rejected(self):
        changes=(lambda c:c.update(value_decimal='0'),lambda c:c.update(scale='-1'),
            lambda c:c.update(currency_literal='EUR'),lambda c:c.update(as_of='2000-01-01'))
        for change in changes:
            d=data();e=d['full_catalog'][1];change(e['fields']['current']['total_assets'])
            # Bind mutated source bytes explicitly; no assess used to force invalid fields through.
            r=d['accepted_records'][1];r['event_sha256']=f.source.canonical_sha256(e)
            r['saved_scope_decision']['event_sha256']=r['event_sha256']
            r['fields']['current.total_assets']=deepcopy(e['fields']['current']['total_assets'])
            r['saved_scope_decision']['required_cell_sha256']['current.total_assets']=f.source.canonical_sha256(r['fields']['current.total_assets'])
            with self.assertRaises(ValueError):f.overlay_prefix(**d)
    def test_no_clipping_above_adapter_limit(self):
        d=data();d['full_catalog'][1]['fields']['current']['net_income']['value_decimal']='1200';rebind(d,1)
        h,_,_=f.overlay_prefix(**d)
        self.assertEqual(h[0,1],float(Fraction(10,11)));self.assertGreater(h[0,1],.5)
        # The runner retains original adapter's hard .5 bound, not a missing-data conversion.
    def test_source_availability_has_no_eligibility_input(self):
        d=data();d['meta']['eligible']=Poison()
        _,m,a=f.overlay_prefix(**d);self.assertTrue(m[0,1]);self.assertTrue(a['availability_independent_of_E'])


class AdmissionTests(unittest.TestCase):
    def rows(self):return [metrics(f.ARMS[0]),metrics(f.ARMS[1]),metrics(f.ARMS[2],cagr=.6)]
    def test_both_references_only_candidate_promotable(self):
        rows=self.rows();result=f.select_first88(rows)
        self.assertTrue(result['admitted']);self.assertEqual(result['primary'],f.ARMS[2])
        rows[1]['training']['cagr']=.7
        self.assertEqual(f.select_first88(rows)['primary'],f.ARMS[0])
    def test_exact_guard_thresholds_and_nextafter(self):
        fields=(('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025))
        for section,key,t in fields:
            rows=self.rows();edge=rows[0][section][key]-t;rows[2][section][key]=edge
            self.assertTrue(f.select_first88(rows)['admitted'])
            rows[2][section][key]=np.nextafter(edge,-np.inf)
            self.assertFalse(f.select_first88(rows)['admitted'])
    def test_cagr_strict_one_e_minus12_and_tie(self):
        rows=self.rows();edge=rows[0]['training']['cagr']+1e-12
        for value in (rows[0]['training']['cagr'],edge):
            rows[2]['training']['cagr']=value;self.assertFalse(f.select_first88(rows)['admitted'])
        rows[2]['training']['cagr']=np.nextafter(edge,np.inf);self.assertTrue(f.select_first88(rows)['admitted'])
    def test_nonfinite_bool_missing_metrics_reject_guard(self):
        for value in (None,True,np.bool_(True),np.nan,np.inf,'1'):
            rows=self.rows();rows[2]['training']['sharpe']=value
            self.assertFalse(f.select_first88(rows)['admitted'])
        rows=self.rows();del rows[2]['recent']['cagr'];self.assertFalse(f.select_first88(rows)['admitted'])
    def test_later_and_cost_objects_never_consumed(self):
        rows=self.rows()
        for r in rows:r['full']=Poison();r['cost']=Poison()
        result=f.select_first88(rows);self.assertTrue(result['admitted'])
        for key in ('later_used','next_open_used','cost_diagnostics_used','control_promotable'):self.assertFalse(result[key])
    def test_execution_window_names_strict(self):
        for kwargs in (dict(execution='open'),dict(cost_per_side=.0005),dict(cost_per_side=False),dict(training_months=122)):
            with self.assertRaises(ValueError):f.select_first88(self.rows(),**kwargs)
        rows=self.rows();rows.reverse()
        with self.assertRaises(ValueError):f.select_first88(rows)
    def test_uses_literal_145_guard_twice(self):
        with patch.object(f,'guard',wraps=f.guard) as gate:
            f.select_first88(self.rows())
        self.assertEqual(gate.call_count,2)
        self.assertTrue(all(c.kwargs=={'strict':True} for c in gate.call_args_list))


if __name__=='__main__':unittest.main(verbosity=2)

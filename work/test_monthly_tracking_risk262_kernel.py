"""Synthetic262 arithmetic, source, as-of, funding and prefix tests. No actual I/O."""
from copy import deepcopy
from fractions import Fraction
import hashlib,importlib.util,json,math,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_tracking_risk262_kernel as k
import test_monthly_downside_basket258_kernel as base


def make_review(meta,calendar):
    ss=pd.DatetimeIndex(calendar);etfs=set(meta['etfs']);clocks=[];rows=[]
    for q,(d,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        t=calendar.index(d);partial=d==k.INITIAL_PARTIAL;clocks.append(k._clock(q,pd.Timestamp(d),names,t,partial,ss,etfs))
        for j,s in enumerate(names):
            if s in etfs:continue
            b=next((b for b in k.BOUNDS if b['symbol']==s),None)
            sector=None if s=='SPCX' else 'XLY'
            rows.append(dict(query_index=q,column_index=j,date=d,symbol=s,issuer_key=b['issuer_key'] if b else 'issuer-'+s,original_sector=sector,sector=sector,mapping_origin='query_date_sector_panel',mapping_repair='unresolved' if sector is None else 'unchanged',mapping_evidence=None))
    return dict(schema=k.SOURCE_SCHEMA,passed=True,metadata_only=True,source_admitted=False,strict_pit=False,clock_events=clocks,mapping_records=rows,listing_bounds=deepcopy(k.BOUNDS),listing_policy252_v2={'policy':deepcopy(k.LATE)},early_listing_policy={'policy':deepcopy(k.EARLY)},factor_listing_bounds=[])


def fixture():
    s,a,m,p,c,_=base.fixture();x=np.arange(len(p['adj']),dtype=float)
    p['adj']['XLY']=90.+.035*x+.5*np.sin(x/13.)
    return s,a,m,p,c,make_review(m,c)


def warm_fixture():
    s,a,m,_,_,_=fixture();names=['SPCX','B','C','D','E','F'];m=dict(dates=['2026-06-30','2026-07-31','2026-08-31'],ends=['2026-07-31','2026-08-31','2026-09-30'],symbols=[names.copy() for _ in range(3)],etfs=['F'])
    ds=pd.bdate_range('2024-01-01','2026-10-02');x=np.arange(len(ds),dtype=float)
    p={'adj':pd.DataFrame({v:100.+.1*x+.6*np.sin(x/(j+3.)) for j,v in enumerate(names+['XLY'])},index=ds)}
    p['adj'].loc[:'2026-06-11','SPCX']=np.nan;c=ds.strftime('%Y-%m-%d').tolist()
    return s,a,m,p,c,make_review(m,c)


def physical(args,n):
    s,a,m,p,c,r=args;aa={key:np.asarray(v)[:n].copy() for key,v in a.items()};mm={key:deepcopy(m[key][:n]) for key in ('dates','ends','symbols')};mm['etfs']=m['etfs'].copy()
    last=mm['dates'][-1] if n else None;cc=c[:c.index(last)+2] if n else [];pp={'adj':p['adj'].loc[:last].copy() if n else p['adj'].iloc[:0].copy()}
    return s,aa,mm,pp,cc,deepcopy(r)


def run(args,n=None,**kw):
    s,a,m,p,c,r=args;return k.prepare(s,a,m,p,c,r,len(m['dates']) if n is None else n,**kw)


def prices():
    return [100.+.11*i+.9*math.sin(i/3.) for i in range(65)],[90.+.08*i+.4*math.cos(i/7.) for i in range(65)]


class Math(unittest.TestCase):
    def test_exact_operation_order_and_evidence(self):
        a,b=prices();z=k.measure(a,b);self.assertEqual(tuple(z),k.STAT_KEYS)
        mi=a[22]/a[0]-1.;mb=b[22]/b[0]-1.;self.assertEqual(z['relative_momentum'],(1.+mi)/(1.+mb)-1.)
        self.assertEqual(z['own_returns'],[a[j]/a[j-1]-1. for j in range(2,65)])
        for key in ('total_risk','tracking_risk'):
            v=z['own_returns'] if key=='total_risk' else z['tracking_returns'];r=z[key];mu=math.fsum(v)/63.
            self.assertEqual(r['mean'],mu);self.assertEqual(r['square_sum'],math.fsum((x-mu)*(x-mu) for x in v));self.assertEqual(r['sd'],math.sqrt(r['square_sum']/62.))
        json.dumps(z,allow_nan=False)
    def test_constant_nonzero_branch_exact_positive_zero(self):
        z=k.risk63([.1]*63);self.assertTrue(z['constant']);self.assertEqual(z['mean'],.1)
        for key in ('square_sum','variance','sd'):self.assertEqual(z[key].hex(),'0x0.0p+0')
        self.assertEqual(z['centered'],[0.]*63)
    def test_zero_total_or_tracking_common_unavailable(self):
        a,b=prices()
        for aa,bb in (([100.]*65,b),(a,a)):
            z=k.measure(aa,bb);self.assertFalse(z['supported']);self.assertIsNone(z['total_ratio']);self.assertIsNone(z['tracking_ratio'])
    def test_zero_sector_exact_same_risk(self):
        a,_=prices();z=k.measure(a,[80.]*65);k._exact(z['total_risk'],z['tracking_risk'],'zero sector');self.assertEqual(z['total_ratio'],z['tracking_ratio'])
    def test_binary_rescaling_and_negative_relative(self):
        a,b=prices();z=k.measure(a,b);res=k.measure([2*x for x in a],[.5*x for x in b]);k._exact(z,res,'exact powers2')
        a[0]=1000.;self.assertLess(k.measure(a,b)['relative_momentum'],0.)
    def test_true_zero_relative_is_valid(self):
        a,b=prices();a[22]=a[0];b[22]=b[0];z=k.measure(a,b);self.assertTrue(z['supported']);self.assertEqual(z['total_ratio'],0.)
    def test_one_ulp_constant_perturbation_not_epsilon(self):
        v=[.1]*63;v[-1]=math.nextafter(.1,math.inf);z=k.risk63(v);self.assertFalse(z['constant']);self.assertGreater(z['sd'],0.)
    def test_underflow_and_overflow_fail(self):
        for f,args in ((k._square,(1e-200,'x')),(k._div,(math.ulp(0.),62.,'x')),(k._sum,([1e308,1e308],'x'))):
            with self.subTest(f=f.__name__),self.assertRaises(ArithmeticError):f(*args)
        with self.assertRaises(ArithmeticError):k.risk63([0.]*62+[1e-200])
    def test_price_type_shape_and_positive(self):
        a,b=prices()
        for bad in (True,np.bool_(True),0.,-1.,None,math.nan,math.inf,'1',Fraction(1,10**1000)):
            aa=a.copy();aa[3]=bad
            with self.subTest(bad=type(bad).__name__),self.assertRaises((ValueError,ArithmeticError)):k.measure(aa,b)
        with self.assertRaises(ValueError):k.measure(a[:-1],b)
    def test_rank_original_sequence_ties_n0_n1(self):
        d=[0.,1.,1.,3.,4.,5.]
        for v in d:self.assertEqual(k.rank(v,d),(float(2*sum(x<v for x in d)+sum(x==v for x in d)+1)/2.)/6.-.5)
        self.assertEqual(k.rank(0.,[0.]),.5)
        with self.assertRaises(ValueError):k.rank(0.,[])


class Source(unittest.TestCase):
    def test_all_stocks_before_E_and_65_not253(self):
        args=fixture();args[1]['eligible'][:]=False;f=k.build_features(args[3],args[2],args[4],args[1],args[5],3)
        self.assertEqual([len(x['cells']) for x in f],[5]*3);self.assertEqual(len(f[1]['required_dates']),65)
        self.assertEqual(f[1]['required_dates'][22],f[1]['numerator_dates'][1]);self.assertEqual(len(f[1]['cells'][0]['statistics']['own_returns']),63)
        self.assertIsNone(f[1]['total_target']);self.assertEqual(f[0]['required_dates'],[])
    def test_nonrequired_price_poison_is_not_read(self):
        args=fixture();t=args[4].index(args[2]['dates'][1]);d=args[4][t-200];args[3]['adj'].at[pd.Timestamp(d),'A']='POISON'
        k.build_features(args[3],args[2],args[4],args[1],args[5],2)
    def test_full_window_missing_row_column_nan_hardfail(self):
        for mode in ('row','column','value'):
            args=fixture();day=args[5]['clock_events'][1]['risk_dates'][10]
            if mode=='row':args[3]['adj'].drop(index=pd.Timestamp(day),inplace=True)
            elif mode=='column':args[3]['adj'].drop(columns=['A'],inplace=True)
            else:args[3]['adj'].at[pd.Timestamp(day),'A']=math.nan
            with self.subTest(mode=mode),self.assertRaises(ValueError):run(args,2)
    def test_invalid_in_Efalse_or_zero_stock_not_masked(self):
        for bad in (0.,-1.,math.inf,True,'bad'):
            args=fixture();args[1]['eligible'][:]=False;args[3]['adj']['A']=100.;d=args[5]['clock_events'][1]['risk_dates'][10];args[3]['adj'].at[pd.Timestamp(d),'B']=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):run(args,2)
    def test_mature_missing_mapping_hardfail(self):
        args=fixture();r=args[5]['mapping_records'][5];r.update(original_sector=None,sector=None,mapping_repair='unresolved')
        with self.assertRaises(ValueError):run(args,2)
    def test_warmup_new_early_policy_allslots_scanned(self):
        out=run(warm_fixture());f=out[0];c=f[0]['cells'][0]
        self.assertEqual(c['history_bound']['not_before'],'2026-06-08');self.assertEqual(c['path_history_bound']['available_from'],'2026-06-16');self.assertEqual(c['path_history_bound']['not_before'],'2026-06-12')
        self.assertEqual(c['history_status'],'official_pre_inception');self.assertEqual(len(c['own_observations']),65);self.assertEqual(c['sector_observations'],[]);self.assertIsNone(c['statistics']);self.assertIsNone(f[0]['tracking_target'][0])
    def test_warmup_prepositive_postmissing_and_badtype_fail(self):
        for day,value in (('2026-06-10',1.),('2026-06-12',math.nan),('2026-06-10',-1.),('2026-06-10',True)):
            args=warm_fixture();args[3]['adj'].at[pd.Timestamp(day),'SPCX']=value
            with self.subTest(day=day,value=value),self.assertRaises(ValueError):run(args,1)
    def test_policy_not_visible_before_june16_and_wrongissuer(self):
        r=dict(symbol='SPCX',issuer_key=k.EARLY['path_bound_override']['issuer_key'],date='2026-06-15');b,p=k._bounds(r,False);self.assertEqual(p['not_before'],'2026-06-08')
        r['date']='2026-06-16';self.assertEqual(k._bounds(r,False)[1]['not_before'],'2026-06-12')
        args=warm_fixture();args[5]['mapping_records'][0]['issuer_key']='old-SPCX-fund'
        with self.assertRaises(ValueError):run(args,1)
    def test_warmup_known_sector_is_still_scanned(self):
        args=warm_fixture();r=args[5]['mapping_records'][0];r.update(original_sector='XLY',sector='XLY',mapping_repair='unchanged');d=args[5]['clock_events'][0]['required_dates'][2];args[3]['adj'].at[pd.Timestamp(d),'XLY']=math.nan
        with self.assertRaises(ValueError):run(args,1)
    def test_duplicate_issuer_mapping_clock_tamper(self):
        args=fixture();args[5]['mapping_records'][1]['issuer_key']=args[5]['mapping_records'][0]['issuer_key']
        with self.assertRaises(ValueError):run(args,1)
        args=fixture();args[5]['clock_events'][1]['required_dates'][0]='2015-01-01'
        with self.assertRaises(ValueError):run(args,2)
    def test_factor_inception_is_not_source_admission(self):
        args=fixture();args[5]['factor_listing_bounds']=[dict(symbol='XLY',inception_date='2016-01-01',available_from=None,evidence_clock_certified=False)]
        with self.assertRaises(ValueError):run(args,2)
    def test_partial_no_numeric_read(self):
        args=fixture();args[3]['adj'].iloc[:,:]=math.nan;f=run(args,1)[0];self.assertTrue(f[0]['initial_partial']);self.assertTrue(all(c['statistics'] is None for c in f[0]['cells']))


class Integration(unittest.TestCase):
    def test_all_prefix_cached_physical_exact_and_json(self):
        args=fixture();full=run(args);json.dumps(full[0],allow_nan=False);json.dumps(full[4],allow_nan=False)
        for n in range(4):
            left=run(args,n);right=run(physical(args,n));cached=k.decisions(args[0],args[1],args[2],full[0],n)
            k._exact(left[0],full[0][:n],'feature prefix');k._exact(left[0],right[0],'physical feature')
            for out in (right[1:],cached):
                k._exact(left[1],out[0],'orders');k._exact(left[4],out[3],'audit')
                for arm in k.ARMS:
                    self.assertEqual(left[2][arm].tobytes(),out[1][arm].tobytes());self.assertEqual(left[3][arm].tobytes(),out[2][arm].tobytes())
    def test_gate_order_before_stats_ranks_or_candidate_selector(self):
        args=fixture();s=args[0];events=[];orig=s.ep.select_boundary
        def select(*a,**kw):events.append('episode');return orig(*a,**kw)
        def ig(*a):self.assertEqual(events,['episode']);events.append('inc');return {'passed':True,'scope':'inc'}
        def sg(fs):self.assertEqual(events,['episode','inc']);self.assertTrue(all(f['total_target'] is None for f in fs));events.append('source');return {'passed':True,'scope':'source'}
        with patch.object(s.ep,'select_boundary',select):out=run(args,incumbent_validator=ig,source_validator=sg)
        self.assertEqual(events,['episode','inc','source','episode','episode']);self.assertEqual(out[4]['source_gate'],{'passed':True,'scope':'source'})
    def test_failed_inc_gate_zero_new_stats(self):
        args=fixture()
        with patch.object(k,'measure',side_effect=AssertionError('new stats before gate')):
            with self.assertRaises(ValueError):run(args,incumbent_validator=lambda *a:{'passed':False})
    def test_failed_source_gate_no_rank(self):
        with patch.object(k,'rank',side_effect=AssertionError('rank before gate')):
            with self.assertRaises(ValueError):run(fixture(),source_validator=lambda *a:{'passed':False})
    def test_gate_original_mutation_stops_before_candidates(self):
        args=fixture();calls=[];orig=args[0].ep.select_boundary
        def sel(*a,**kw):calls.append(1);return orig(*a,**kw)
        def sg(fs):args[1]['effects'][0,0]=.123;return {'passed':True}
        with patch.object(args[0].ep,'select_boundary',sel),self.assertRaises(ValueError):run(args,source_validator=sg)
        self.assertEqual(len(calls),1)
    def test_original_arrays_meta_panel_not_mutated(self):
        args=fixture();before=k.old._snapshot(args[1]);mm=deepcopy(args[2]);p=args[3]['adj'].copy();run(args)
        self.assertEqual(before,k.old._snapshot(args[1]));self.assertEqual(mm,args[2]);pd.testing.assert_frame_equal(p,args[3]['adj'])
    def test_finite_zero_rank_funds_support_and_partial_copy(self):
        args=fixture();a=args[1];a['baseline']=a['core']+(1.-a['core_weight'])*.2;out=run(args)
        for f in out[0][1:]:self.assertEqual(sum(x is not None for x in f['total_target']),5)
        for arm in k.ARMS[1:]:self.assertEqual(out[3][arm][0].tobytes(),out[3][k.ARMS[0]][0].tobytes())
        for i,f in enumerate(out[0]):
            for arm,key in zip(k.ARMS[1:],('total_target','tracking_target')):
                for j,x in enumerate(f[key]):
                    expected=0. if x is None else .05*(x-.2)
                    if x is not None:self.assertAlmostEqual(out[3][arm][i,j]-out[3][k.ARMS[0]][i,j],expected)
    def test_zero_common_domain_keeps_inc_and_singleton_rank(self):
        args=fixture();args[3]['adj'].loc[:,list('ABCDE')]=100.;out=run(args)
        self.assertTrue(all(x is None for x in out[0][1]['total_target']))
        for arm in k.ARMS:self.assertEqual(out[1][arm],out[1][k.ARMS[0]])
        args=fixture();args[3]['adj'].loc[:,list('BCDE')]=100.;out=run(args);self.assertEqual(out[0][1]['total_target'][0],.5)
    def test_cached_ulp_stat_rank_source_tamper(self):
        args=fixture();features=run(args)[0]
        for kind in ('stat','rank','source'):
            f=deepcopy(features)
            if kind=='stat':obj=f[1]['cells'][0]['statistics'];obj['relative_momentum']=math.nextafter(obj['relative_momentum'],math.inf)
            elif kind=='rank':f[1]['total_target'][0]=math.nextafter(f[1]['total_target'][0],math.inf)
            else:f[1]['cells'][0]['sector_observations'][4]['value']*=1.01
            with self.subTest(kind=kind),self.assertRaises(ValueError):k.decisions(args[0],args[1],args[2],f,3)
    def test_future_objects_not_consumed(self):
        args=fixture();full=run(args)[0]
        for n in (0,1,2):
            a=deepcopy(args[1]);m=deepcopy(args[2]);f=deepcopy(full)
            for field in ('dates','ends','symbols'):m[field][n:]=[base.Poison() for _ in m[field][n:]]
            f[n:]=[base.Poison() for _ in f[n:]]
            for key in a:
                if a[key].dtype.kind=='f':a[key][n:]=math.inf
            k.decisions(args[0],a,m,f,n)
    def test_source_and_cached_actual_measure_counts(self):
        args=fixture();orig=k.measure
        for n in (0,1,2,3):
            with patch.object(k,'measure',wraps=orig) as spy:out=run(args,n);self.assertEqual(spy.call_count,out[4]['counts']['mature_cells'])
            with patch.object(k,'measure',wraps=orig) as spy:k.decisions(args[0],args[1],args[2],out[0],n);self.assertEqual(spy.call_count,out[4]['counts']['mature_cells'])
    def test_expected_incumbent_and_invalid_n(self):
        args=fixture();out=run(args)
        bad=deepcopy(out[1][k.ARMS[0]]);bad[0]['symbols']=['BAD']
        with self.assertRaises(ValueError):run(args,expected_incumbent_orders=bad)
        for n in (-1,True,4):
            with self.assertRaises((ValueError,TypeError)):run(args,n)

class ExtraEdges(unittest.TestCase):
    def test_default_n_and_noncanonical_n(self):
        args=fixture();out=k.prepare(*args);cached=k.decisions(args[0],args[1],args[2],out[0]);k._exact(cached[0],out[1],'default n')
        for n in (True,np.int64(2),1.0):
            with self.assertRaises(ValueError):k.prepare(*args,n)
    def test_tiny_real_conversion_cannot_be_constant_zero(self):
        with self.assertRaises(ValueError):k.risk63([Fraction(1,10**1000)]*63)
    def test_complete49_stock_inventory_and_no_peer_requirement(self):
        s,a,m,p,c,_=fixture();names=['S'+str(i) for i in range(49)]
        for key,v in a.items():
            if v.ndim==2 and v.shape[1]==6:a[key]=np.tile(v[:,:1],(1,49))
        m['symbols']=[names.copy() for _ in m['dates']];m['etfs']=names[32:]
        for j,name in enumerate(names):p['adj'][name]=p['adj']['A']*(1.+.00001*j*np.arange(len(p['adj'])))
        r=make_review(m,c);out=run((s,a,m,p,c,r));self.assertEqual(out[4]['counts']['stock_cells'],96);self.assertEqual(out[4]['counts']['own_price_observations'],2*32*65)
        self.assertEqual(out[4]['counts']['supported_cells'],64);self.assertTrue(all(x is None for x in out[0][1]['total_target'][32:]))
    def test_unofficial_consumed_axis_fails(self):
        args=fixture();p=args[3]['adj'];extra=p.iloc[[0]].copy();extra.index=pd.DatetimeIndex(['2016-08-20']);args[3]['adj']=pd.concat([p,extra]).sort_index()
        with self.assertRaises(ValueError):run(args)
    def test_full_budget_calm_extension_and_weak(self):
        args=fixture();a=args[1];a['forecast_available'][:]=True;a['extension_available'][:]=True;a['model_available'][:]=True;a['local_available'][:]=True;a['state_weak_narrow'][1]=True
        out=run(args);self.assertEqual(out[4]['counts']['original_episode_calls'],3);self.assertEqual(out[4]['events'][2]['core_weight'],.5)
    def test_early_and_source_gate_retained_exact(self):
        args=fixture();out=run(args,incumbent_validator=lambda *a:{'passed':True,'details':[1]},source_validator=lambda *a:{'passed':True,'kind':'fresh'})
        self.assertEqual(out[4]['incumbent_gate'],{'passed':True,'details':[1]});self.assertEqual(out[4]['source_gate'],{'passed':True,'kind':'fresh'})
    def test_fake_vox_loader_hash_copy_and_mismatch(self):
        import tracking_risk262_sources as src
        raw=json.dumps([{'date':'2015-01-02T00:00:00Z','adjClose':12.}]).encode()
        class FakePath:
            def read_bytes(self):return raw
        panel={'adj':pd.DataFrame({'XLY':[10.,11.]},index=pd.DatetimeIndex(['2015-01-02','2015-01-05']))};before=panel['adj'].copy()
        review=dict(schema=k.SOURCE_SCHEMA,source_admitted=False,metadata_only=True,vox={'parser_sha256':src.PARSER_SHA},vox_prices_path='synthetic-only',vox_prices_sha256=hashlib.sha256(raw).hexdigest())
        with patch.object(src,'_path',return_value=FakePath()):out=src.attach_vox(panel,review,['2015-01-02','2015-01-05'])
        self.assertEqual(out['adj'].at[pd.Timestamp('2015-01-02'),'VOX'],12.);self.assertTrue(pd.isna(out['adj'].at[pd.Timestamp('2015-01-05'),'VOX']));pd.testing.assert_frame_equal(before,panel['adj'])
        review['vox_prices_sha256']='0'*64
        with patch.object(src,'_path',return_value=FakePath()),self.assertRaises(ValueError):src.attach_vox(panel,review,['2015-01-02'])
    def test_fake_vox_parser_rejects_nonofficial_or_invalid(self):
        import tracking_risk262_sources as src
        for value in (0.,-1.,True,math.inf):
            raw=json.dumps([{'date':'2015-01-02T00:00:00Z','adjClose':value}]).encode()
            class FakePath:
                def read_bytes(self):return raw
            panel={'adj':pd.DataFrame({'XLY':[1.]},index=pd.DatetimeIndex(['2015-01-02']))}
            review=dict(schema=k.SOURCE_SCHEMA,source_admitted=False,metadata_only=True,vox={'parser_sha256':src.PARSER_SHA},vox_prices_path='synthetic-only',vox_prices_sha256=hashlib.sha256(raw).hexdigest())
            with patch.object(src,'_path',return_value=FakePath()),self.assertRaises(ValueError):src.attach_vox(panel,review,['2015-01-02'])


if __name__=='__main__':unittest.main(verbosity=2)

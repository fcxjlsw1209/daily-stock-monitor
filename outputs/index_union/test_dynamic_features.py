import unittest,importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from dynamic_features import features
P=Path(__file__).resolve().parent.parent

def module(folder,file='strategy.py'):
    spec=importlib.util.spec_from_file_location('union_test_'+folder+'_'+file.replace('.','_'),P/folder/file)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def fixture():
    rng=np.random.default_rng(4321);days=pd.bdate_range('2010-01-01',periods=400);cols=['A','B','C','D','E','F']
    close=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.001,.015,(400,8)),axis=0)),index=days,columns=cols+['SPY','QQQ'])
    close.SPY=100*np.exp(np.arange(400)*.0007+.02*np.sin(np.arange(400)*.4))
    adj=close[cols].copy();volume=close*0+2e6;sector=pd.DataFrame({'XLK':close.SPY*.9},index=days)
    data=SimpleNamespace(close=close,opening=close*.999,volume=volume,dividends=close*0,sector={s:'Technology' for s in cols},sector_prices=sector)
    raw={'adj_close':adj,'high':adj*1.02,'low':adj*.98,'sector_adjusted':sector}
    args=dict(adj=adj,close=adj,opening=data.opening[cols],high=raw['high'],low=raw['low'],volume=volume[cols],
      sector60=pd.DataFrame({s:sector.XLK.pct_change(60,fill_method=None) for s in cols}),
      market_close=close.SPY,market_dividends=data.dividends.SPY,eligible=pd.DataFrame(True,index=days,columns=cols))
    return data,raw,args

class DynamicFactorTests(unittest.TestCase):
    def test_all_six_match_original_when_universe_unchanged(self):
        data,raw,args=fixture();got=features(**args)
        for name,folder in [('baseline','daily_stock_monitor'),('pressure','pressure_shadow_monitor'),('resilience','resilience_shadow_monitor')]:
            expected=module(folder).features(data,raw)
            for key in ['score','ma5','atr']:pd.testing.assert_frame_equal(got[name][key],expected[key])
            pd.testing.assert_frame_equal(got[name]['mask'],expected['masks']['broad'])
        for name,folder,key in [('accel','accel_shadow_monitor','accel'),('high52','high52_shadow_monitor','high52'),('accel40','accel40_shadow_monitor','accel')]:
            expected=module(folder,'frontier_features.py').all_features(data,raw)[key]
            for field in ['score','ma5','atr']:pd.testing.assert_frame_equal(got[name][field],expected[field])
            pd.testing.assert_frame_equal(got[name]['mask'],expected['masks']['trend'])
    def test_future_membership_and_prices_do_not_change_past(self):
        _,_,args=fixture();full=features(**args);cut=350
        short=features(**{k:v.iloc[:cut] for k,v in args.items()})
        for name in full:
            for field in full[name]:pd.testing.assert_frame_equal(full[name][field].iloc[:cut],short[name][field])
    def test_nonmembers_excluded_before_ranking(self):
        _,_,args=fixture();args['eligible'].loc[:,:]=False;args['eligible']['A']=True
        got=features(**args)
        for result in got.values():
            self.assertTrue(result['score'].drop(columns='A').isna().all().all())
            self.assertFalse(result['mask'].drop(columns='A').any().any())
            scores=result['score']['A'].dropna();self.assertGreater(len(scores),20)
            np.testing.assert_allclose(scores,1.)
    def test_membership_is_not_price_history(self):
        _,_,args=fixture();args['eligible'].loc[:,:]=False;args['eligible'].iloc[-1]=True
        got=features(**args)
        for result in got.values():
            self.assertTrue(result['score'].iloc[-1].notna().all())
            self.assertTrue(result['score'].iloc[:-1].isna().all().all())
    def test_missing_membership_rejected(self):
        _,_,args=fixture();args['eligible']=args['eligible'].astype(object);args['eligible'].iloc[-1,0]=None
        with self.assertRaises(ValueError):features(**args)
    def test_misaligned_inputs_rejected(self):
        _,_,args=fixture();args['sector60']=args['sector60'].iloc[1:]
        with self.assertRaises(ValueError):features(**args)
if __name__=='__main__':unittest.main()

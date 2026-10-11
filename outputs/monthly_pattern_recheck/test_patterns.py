import unittest
import numpy as np
import pandas as pd
import patterns as p
class PatternTests(unittest.TestCase):
 def prices(self):
  rng=np.random.default_rng(27);idx=pd.bdate_range('2018-01-01',periods=600);cols=['SPY','A','B'];r=rng.normal(.0005,.015,(600,3));px=pd.DataFrame(100*np.exp(np.cumsum(r,axis=0)),index=idx,columns=cols)
  raw=dict(close=px.copy(),high=px*1.02,low=px*.98,opening=px*.999,volume=pd.DataFrame(rng.integers(1000,10000,(600,3)),index=idx,columns=cols))
  return px,raw
 def test_shapes_use_no_future_prices(self):
  px,raw=self.prices();full=p.shape_only(px,raw);prefix=p.shape_only(px.iloc[:400],{k:v.iloc[:400] for k,v in raw.items()})
  for k in full:np.testing.assert_allclose(full[k].iloc[:400],prefix[k],rtol=1e-12,atol=1e-12,equal_nan=True)
 def test_ohlc_adjustment_units_consistent(self):
  px,raw=self.prices();z=p.shape_only(px,raw);scaled={k:(v*4 if k!='volume' else v) for k,v in raw.items()};zz=p.shape_only(px,scaled)
  for k in z:np.testing.assert_allclose(z[k],zz[k],rtol=1e-12,atol=1e-12,equal_nan=True)
 @unittest.skipUnless((p.P/'FROZEN.json').exists(),'Train first')
 def test_patterns_share_twenty_percent_budget(self):
  for r in p.read(p.P/'FROZEN.json')['leaders'].values():
   spec=r['rule']['spec'];A,b,_=p.e.limits(spec);ids=[i for i,v in enumerate(spec) if v[2] in ['shape','accel','ml']]
   for w in r['rule']['weights'].values():
    self.assertLessEqual(sum(w[i] for i in ids),.2000001);self.assertLessEqual(np.max(np.einsum('ij,j->i',A,w,optimize=False)-b),1e-7)
if __name__=='__main__':unittest.main()

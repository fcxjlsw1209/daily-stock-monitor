import unittest
from extend import *
import optimize as opt

class ExtensionTests(unittest.TestCase):
 def test_event_dates_and_baseline_features(self):
  m,d,L,x,px,op=load();om,od,ol,ox,_,_=ref.load()
  self.assertEqual(m['dates'][0],'2016-08-17');self.assertEqual(m['dates'][53:],om['dates']);self.assertEqual(len(m['dates']),122)
  for k in ['core','support','eligible','high','weak']:np.testing.assert_allclose(d[k][53:],od[k],rtol=0,atol=1e-12)
  for ss in m['symbols']:self.assertEqual(len(set(ss)),49)
 def test_future_cap_observation_ignored(self):
  z=pd.Series([100.,1e15],index=pd.to_datetime(['2016-07-29','2016-08-31']));v=cap.snapshot_cap(z,pd.Timestamp('2016-08-16'),None)
  self.assertEqual(v['observed_cap'],100.);self.assertEqual(v['observation_date'],'2016-07-29')
 def test_rank_timestamps(self):
  for row in read(P/'EARLY_RANKS.json'):
   self.assertLess(row['asof'],row['date'])
   for r in row['top40']:self.assertLessEqual(r.get('observation_date',r.get('observation')),row['asof'])
 def test_training_label_maturity(self):
  for row in read(P/'FORECAST_AUDIT.json'):
   self.assertLessEqual(row['latest_label_end'],row['cutoff']);self.assertLess(row['cutoff'],str(row['year'])+'-01-01');self.assertGreater(row['rows'],1000)
 def test_insufficient_candidates_and_no_future_price_filter(self):
  sc=np.array([[.2,.1,.0]]);self.assertEqual(rp.choose(sc,np.array([[True,False,False]]),[['A','B','C']]),[[]])
  choices=rp.choose(sc,np.array([[True,True,True]]),[['A','B','C']]);self.assertEqual(choices,[[0,1,2]])
  with self.assertRaises(AssertionError):rp.fast_returns(choices,[np.array([[1.,1.,1.],[np.nan,1.,1.]])],[np.array([False,True,True])])
 def test_future_prices_do_not_change_features(self):
  px,raw,members,_,_,_=prices();day='2019-06-28';m=read(P/'INPUTS.json');i=m['dates'].index(day);one=dict(dates=[day],symbols=[m['symbols'][i]])
  # Build from a finite prefix plus10future sessions, then perturb only future.
  end=px.index[px.index.get_loc(pd.Timestamp(day))+10];px=px.loc[:end].copy();raw={k:v.loc[:end].astype(float).copy() for k,v in raw.items()}
  a=features(px,raw,one,members);px.loc[px.index>day]*=3.7
  for k,v in raw.items():v.loc[v.index>day]*=2.3
  b=features(px,raw,one,members)
  for aa,bb in zip(a,b):
   for k in aa:
    if isinstance(aa[k],dict):np.testing.assert_allclose(aa[k]['score'],bb[k]['score'],rtol=0,atol=0)
    else:np.testing.assert_allclose(aa[k],bb[k],rtol=0,atol=0)
 def test_frozen_and_protected(self):
  if not (P/'FROZEN.json').exists():self.skipTest('Before selection freeze')
  self.assertEqual(sha(P/'FROZEN.json'),(P/'FROZEN.sha256').read_text().strip())
  for f,h in read(P/'PROTECTED_BEFORE.json').items():self.assertEqual(sha(f),h,f)

if __name__=='__main__':unittest.main()

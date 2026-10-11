import unittest
import pandas as pd
from consolidate_security_aliases import consolidate,PRICE_FIELDS
class AliasPanelTests(unittest.TestCase):
 def setUp(self):
  dates=pd.date_range('2020-01-01',periods=4)
  self.data={k:pd.DataFrame({'OLD':[10.,11.,12.,13.],'NEW':[10.,11.,12.,13.]},index=dates) for k in PRICE_FIELDS}
  self.data['raw_per_signal'][:]=1.
  self.data.update(eligible=pd.DataFrame({'OLD':[True,True,False,False],'NEW':[False,False,True,True]},index=dates),sectors=pd.DataFrame('Technology',index=dates,columns=['OLD','NEW']),sector60=pd.DataFrame(0.,index=dates,columns=['OLD','NEW']),issuers={'OLD':'old','NEW':'new'})
  self.rows=[dict(symbol=s,security_id='EQ',issuer_id='ISS',effective_from=a,effective_to=b,retrospective_identity_link_verified=True) for s,a,b in [('OLD','2000-01-01','2020-01-03'),('NEW','2020-01-03','9999-12-31')]]
  self.obs=[dict(symbol='OLD',usable_from='2019-01-01',known_at='2019-01-01T00:00:00Z',sector='Technology')]
 def test_no_duplicate_and_no_rename_gap(self):
  d,log=consolidate(self.data,self.rows,self.obs);self.assertEqual(list(d['close']),['EQ']);self.assertTrue(d['eligible']['EQ'].all());self.assertTrue(d['sectors']['EQ'].eq('Technology').all());self.assertEqual(d['issuers'],{'EQ':'ISS'})
 def test_price_conflict_blocks(self):
  self.data['close'].loc['2020-01-02','NEW']=100
  with self.assertRaisesRegex(ValueError,'conflict'):consolidate(self.data,self.rows,self.obs)
 def test_reused_old_ticker_outside_epoch_is_not_merged(self):
  self.data['close']['OLD']=[float('nan'),float('nan'),1000.,1000.]
  d,log=consolidate(self.data,self.rows,self.obs)
  self.assertEqual(log[0]['price_provider_symbol'],'NEW');self.assertEqual(d['close']['EQ'].iloc[-1],13.)
 def test_not_merge_unreviewed(self):
  self.rows[0]['retrospective_identity_link_verified']=False
  with self.assertRaises(ValueError):consolidate(self.data,self.rows,self.obs)
if __name__=='__main__':unittest.main()

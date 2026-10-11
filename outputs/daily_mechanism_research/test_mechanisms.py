import unittest,copy
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('daily_mechanism',Path(__file__).with_name('run.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


class MechanismTests(unittest.TestCase):
 def test_conditional_features_never_read_future_prices(self):
  dates=pd.bdate_range('2017-01-01',periods=400);cols=['A','B','C'];rng=np.random.RandomState(5)
  a=pd.DataFrame(100*np.cumprod(1+rng.normal(.001,.02,(400,3)),axis=0),index=dates,columns=cols)
  d=dict(adj=a,close=a,opening=a*.999,high=a*1.02,low=a*.98)
  x=a.rank(axis=1,pct=True);f=dict(valid=a.notna(),groups=[x]*4,X=[x]*8,event=a.lt(a.shift()),y=d['opening'].shift(-6)/d['opening'].shift(-1)-1-.001)
  before=r.condition_features(d,f);d2=copy.deepcopy(d)
  for k in d2:d2[k].iloc[350:]*=2
  f2=copy.deepcopy(f);f2['y']=d2['opening'].shift(-6)/d2['opening'].shift(-1)-1-.001
  after=r.condition_features(d2,f2)
  for i in [1,2]:pd.testing.assert_frame_equal(before['groups'][i].iloc[:350],after['groups'][i].iloc[:350])
 def test_no_replacement_keeps_other_exits(self):
  dates=pd.bdate_range('2020-01-01',periods=15);cols=['A','B','C']
  score=pd.DataFrame([[.6,.8,.9]]*15,index=dates,columns=cols);close=score*0+100;atr=score*0+10;mask=score.gt(.75)
  state=dict(equity=100000,positions={'A':dict(entry_date=str(dates[0].date()),entry_price=100,atr=10),'B':dict(entry_date=str(dates[0].date()),entry_price=100,atr=10)})
  ids={s:s for s in cols};p=r.planner('no_rank_replacement')
  def call(i):return p('baseline',state,dates[i],dates,close,close,atr,mask,score,ids,cols,ids)
  self.assertFalse(call(3)['sells'])
  self.assertEqual({v['reason'] for v in call(10)['sells']},{'holding_limit'})


if __name__=='__main__':unittest.main()

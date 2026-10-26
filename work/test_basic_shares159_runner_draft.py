"""Only synthetic arrays/events and frozen pure adapter/episode code."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock,patch
from types import SimpleNamespace as NS
from datetime import date,timedelta
import importlib.util
import unittest
import numpy as np
import pandas as pd
import basic_shares159_runner_draft as r

ROOT=Path(__file__).resolve().parents[1]
ADAPTER=r.base.module('shares159_synthetic_adapter',ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
EPISODE=r.base.module('shares159_synthetic_episode',ROOT/'outputs/monthly_daily_episodes/episodes.py')
POLICY=dict(budget=.05,ceiling=5,gap=.02)


def fixture(n=4):
 names=list(r.ISSUERS)+['S'+str(i)for i in range(28)]+['ETF'+str(i)for i in range(17)]
 dates=[(date(2017,1,31)+timedelta(days=31*i)).isoformat()for i in range(n)]
 meta=dict(dates=dates,ends=dates,symbols=[names[:]for _ in dates],etfs=names[-17:],train_months=88)
 shape=(n,49);baseline=np.tile(np.linspace(.2,.1,49),(n,1));mag=np.zeros(shape);rank=np.zeros(shape);mask=np.zeros(shape,bool)
 mag[:,:4]=[-.7,-.1,.1,.7];rank[:,:4]=[-1,-1/3,1/3,1];mask[:,:4]=True
 arrays=dict(baseline=baseline,core=.5*baseline,core_weight=np.full(n,.5),eligible=np.ones(shape,bool),effects=np.zeros(shape),state_calm10=np.ones(n,bool),state_weak_narrow=np.zeros(n,bool),basic_share_magnitude=mag,basic_share_rank=rank,common_available=mask)
 return arrays,meta

class SyntheticBackend:
 adapter=ADAPTER
 def __init__(self):self.calls=[]
 def pre(self,a,n):
  p=np.zeros_like(a['baseline']);p[:n]=a['baseline'][:n];return p
 def selector(self,scores,good,symbols,effects,original,policy):
  self.calls.append(deepcopy(dict(scores=scores,good=good,symbols=symbols,effects=effects,original=original,policy=policy)))
  return EPISODE.select_boundary(scores,good,symbols,effects,original,policy)
 def baseline(self,a,m,n,p):return self.adapter.score_and_choose('incumbent66',0.,np.zeros_like(a['baseline']),a,m,n,pre_score=self.pre(a,n),effects=a['effects'],episode_policy=p,boundary_selector=self.selector)
 @staticmethod
 def make_orders(choices,scores,meta):return [dict(date=meta['dates'][i],symbols=[meta['symbols'][i][j]for j in choice],scores=[float(scores[i,j])for j in choice])for i,choice in enumerate(choices)]

class Scoring(unittest.TestCase):
 def test_real_adapter_episode_once_and_full_one_percent(self):
  a,m=fixture();b=SyntheticBackend();chosen,scores,_=r.choices_all(a,m,4,b,POLICY);self.assertEqual(len(b.calls),3)
  for idx,key in enumerate(r.HANDS,1):
   np.testing.assert_array_equal(b.calls[idx]['scores'],b.pre(a,4)+.01*a[key]);np.testing.assert_array_equal(b.calls[idx]['original']['baseline'],a['baseline']);np.testing.assert_array_equal(b.calls[idx]['effects'],a['effects'])
   expected=EPISODE.select_boundary(b.calls[idx]['scores'],a['eligible'],m['symbols'],a['effects'],b.calls[idx]['original'],POLICY)
   self.assertEqual(chosen[r.NAMES[idx]],expected[0]);np.testing.assert_array_equal(scores[r.NAMES[idx]],expected[1])
 def test_old_adapter_rejects_unscaled_hand_but_draft_dummy_does_not(self):
  a,m=fixture();b=SyntheticBackend()
  with self.assertRaisesRegex(ValueError,'bounded ranks'):ADAPTER.score_and_choose('incumbent66',0.,a['basic_share_rank'],a,m,4,pre_score=b.pre(a,4),effects=a['effects'],episode_policy=POLICY,boundary_selector=b.selector)
  r.choices_all(a,m,4,b,POLICY)
 def test_delta_zero_exact_and_no_extra_selector(self):
  a,m=fixture();a[r.HANDS[0]][:]=0;a[r.HANDS[1]][:]=0;b=SyntheticBackend();c,s,_=r.choices_all(a,m,4,b,POLICY);self.assertEqual(len(b.calls),1)
  for name in r.NAMES[1:]:self.assertEqual(c[name],c[r.NAMES[0]]);self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes())
 def test_initial_partial_neutral(self):
  a,m=fixture();m['dates'][0]=r.INITIAL_PARTIAL
  with self.assertRaises(ValueError):r.choices_all(a,m,4,SyntheticBackend(),POLICY)
  a['common_available'][0]=False
  for key in r.HANDS:a[key][0]=0
  c,s,_=r.choices_all(a,m,4,SyntheticBackend(),POLICY)
  for name in r.NAMES[1:]:self.assertEqual(c[name][0],c[r.NAMES[0]][0]);self.assertEqual(s[name][0].tobytes(),s[r.NAMES[0]][0].tobytes())
 def test_eligibility_population_and_etfs_rejected(self):
  for kind in('eligible','etf','bounds','uncentered'):
   a,m=fixture()
   if kind=='eligible':a['eligible'][0,0]=False
   elif kind=='etf':m['etfs'].append('AAPL')
   elif kind=='bounds':a[r.HANDS[1]][0,0]=-1.001
   else:a[r.HANDS[0]][0,0]=0
   with self.subTest(kind=kind),self.assertRaises(ValueError):r.validated_hands(a,m,4)
 def test_full122_and123prefixes_no_mutation(self):
  a,m=fixture(122);before={k:v.tobytes()for k,v in a.items()};orders,scores,audit=r.panels(a,m,SyntheticBackend(),POLICY)
  self.assertEqual(sum(map(len,orders.values())),366);self.assertEqual(len(audit),122);self.assertEqual(before,{k:v.tobytes()for k,v in a.items()})
 def test_future_hands_poison_ignored(self):
  a,m=fixture();expected=r.choices_all(a,m,1,SyntheticBackend(),POLICY)[1]
  for key in r.HANDS:a[key]=a[key].astype(object);a[key][1:]='future'
  actual=r.choices_all(a,m,1,SyntheticBackend(),POLICY)[1]
  for name in r.NAMES:self.assertEqual(actual[name].tobytes(),expected[name].tobytes())
 def test_changed_policy_rejected(self):
  a,m=fixture()
  with self.assertRaises(ValueError):r.choices_all(a,m,4,SyntheticBackend(),dict(POLICY,budget=.06))
 def test_original_cash_and_two_stock_rules(self):
  for count in (1,2):
   a,m=fixture();a['eligible'][:]=False;a['eligible'][:,:count]=True;a['common_available'][:]=False
   for key in r.HANDS:a[key][:]=0
   choices,_,_=r.choices_all(a,m,4,SyntheticBackend(),POLICY)
   for name in r.NAMES:self.assertTrue(all(len(row)==(0 if count==1 else 2)for row in choices[name]))

class Features(unittest.TestCase):
 def events(self):
  return [dict(symbol=s,event_id=s+':2016',period_end='2016-12-31',prior_period_end='2015-12-31',known_at='2017-01-01T21:00:00Z',usable_at='2017-01-01T21:00:00Z',conditional_source_ready=True,same_report_annual_pair=True,same_share_basis=True,unresolved_reasons=[],current_shares=str(v),prior_shares='100')for s,v in zip(r.ISSUERS,(90,95,105,110))]
 def test_exact_raw_audit_shared_population(self):
  a,m=fixture();f=r.build_features(m,self.events(),a['eligible'],[d+'T21:00:00Z'for d in m['dates']]);self.assertEqual(f['audit']['cells'][0]['raw_exact'],{'numerator':'1','denominator':'10'});self.assertTrue(np.all(f['arrays']['common_available'][:,:4]));self.assertTrue(np.all(f['arrays'][r.HANDS[0]][:,4:]==0))
 def test_original_ineligible_removed_before_transform(self):
  a,m=fixture();a['eligible'][:,0]=False;f=r.build_features(m,self.events(),a['eligible'],[d+'T21:00:00Z'for d in m['dates']]);self.assertFalse(f['arrays']['common_available'][:,0].any());self.assertAlmostEqual(f['arrays'][r.HANDS[0]][0].sum(),0.)
 def test_all_equal_and_one_source_neutral(self):
  a,m=fixture();events=self.events()
  for e in events:e['current_shares']='100'
  for es in(events,events[:1]):
   f=r.build_features(m,es,a['eligible'],[d+'T21:00:00Z'for d in m['dates']]);self.assertTrue(all(not f['arrays'][key].any()for key in r.HANDS))
 def test_bad_latest_does_not_use_old_clean(self):
  a,m=fixture();events=self.events();new=dict(events[0],period_end='2017-01-01',event_id='bad',current_shares='0');f=r.build_features(m,events+[new],a['eligible'],[d+'T21:00:00Z'for d in m['dates']]);self.assertFalse(f['arrays']['common_available'][:,0].any())
 def source_rows(self,m):
  return [dict(signal_index=i,candidate_index=j,symbol=s,conditional_source_ready=s in r.ISSUERS,selected_event_id=s+':2016'if s in r.ISSUERS else None)for i,names in enumerate(m['symbols'])for j,s in enumerate(names)]
 def test_final_row_gate_precedes_centering_and_preserves_blocked_event(self):
  a,m=fixture();a['eligible'][:]=False;a['eligible'][:,:2]=True;rows=self.source_rows(m)
  for i in range(len(m['dates'])):rows[i*r.WIDTH+1].update(conditional_source_ready=False,selected_event_id='AMD:latest-blocked')
  before=a['eligible'].tobytes();saved=deepcopy(rows)
  f=r.build_features(m,self.events()[:2],a['eligible'],[d+'T21:00:00Z'for d in m['dates']],source_rows=rows)
  self.assertTrue(f['arrays']['common_available'][:,0].all());self.assertFalse(f['arrays']['common_available'][:,1:].any())
  for key in r.HANDS:self.assertTrue(np.all(f['arrays'][key]==0))
  for i in range(len(m['dates'])):
   blocked=f['audit']['cells'][i*r.WIDTH+1];self.assertEqual(blocked['event_id'],'AMD:latest-blocked');self.assertFalse(blocked['available']);self.assertIsNone(blocked['raw_exact']);self.assertTrue(blocked['original_eligible']);self.assertEqual(blocked['reason'],'source_row_not_conditionally_ready')
  self.assertEqual(a['eligible'].tobytes(),before);self.assertEqual(rows,saved)
 def test_row_gate_does_not_supply_initial_or_ineligible_event(self):
  a,m=fixture();m['dates'][0]=r.INITIAL_PARTIAL;a['eligible'][1,0]=False;rows=self.source_rows(m)
  for source in rows:source['conditional_source_ready']=False
  f=r.build_features(m,self.events(),a['eligible'],[d+'T21:00:00Z'for d in m['dates']],source_rows=rows)
  self.assertTrue(all(c['event_id']is None for c in f['audit']['cells'][:r.WIDTH]));self.assertIsNone(f['audit']['cells'][r.WIDTH]['event_id'])
  self.assertTrue(all(not f['arrays'][key].any()for key in r.EXTRA))
 def test_source_row_full_axis_prefix_equivalence_and_no_future_access(self):
  a,m=fixture();rows=self.source_rows(m);cutoffs=[d+'T21:00:00Z'for d in m['dates']];full=r.build_features(m,self.events(),a['eligible'],cutoffs,source_rows=rows)
  for n in range(len(m['dates'])+1):
   prefix=r.build_features(m,self.events(),a['eligible'],cutoffs,n=n,source_rows=rows)
   for key in r.EXTRA:self.assertEqual(prefix['arrays'][key].tobytes(),full['arrays'][key][:n].tobytes())
   self.assertEqual(prefix['audit']['cells'],full['audit']['cells'][:n*r.WIDTH])
  rows[r.WIDTH:]=[None]*(len(rows)-r.WIDTH)
  prefix=r.build_features(m,self.events(),a['eligible'],cutoffs,n=1,source_rows=rows)
  self.assertEqual(prefix['audit']['cells'],full['audit']['cells'][:r.WIDTH])
 def test_source_row_axis_and_gate_are_explicit(self):
  a,m=fixture();cutoffs=[d+'T21:00:00Z'for d in m['dates']]
  for fault in('short','order','gate','event'):
   rows=self.source_rows(m)
   if fault=='short':rows.pop()
   elif fault=='order':rows[0]['symbol']='AMD'
   elif fault=='gate':rows[0]['conditional_source_ready']=1
   else:del rows[0]['selected_event_id']
   with self.subTest(fault=fault),self.assertRaises(ValueError):r.build_features(m,self.events(),a['eligible'],cutoffs,source_rows=rows)

class Selection(unittest.TestCase):
 def rows(self):return [dict(name=n,training=dict(cagr=.2,sharpe=1.,max_drawdown=-.2),early=dict(cagr=.2),recent=dict(cagr=.2))for n in r.NAMES]
 def select(self,rows,opened=None,changed=None):return r.select_first88(rows,opened or self.rows(),{'passed':True},changed or {n:{'passed':True}for n in r.NAMES[1:]})
 def test_fifty_basis_point_gate_and_cost_guard(self):
  rows=self.rows();rows[1]['training']['cagr']=.2049;self.assertEqual(self.select(rows)['primary'],r.NAMES[0]);rows[1]['training']['cagr']=.205;self.assertEqual(self.select(rows)['primary'],r.NAMES[1]);opened=self.rows();opened[1]['training']['cagr']=.199;self.assertEqual(self.select(rows,opened)['primary'],r.NAMES[0])
 def test_two_arms_can_both_pass_fixed_tie_order(self):
  rows=self.rows();rows[1]['training']['cagr']=rows[2]['training']['cagr']=.21;self.assertEqual(self.select(rows)['primary'],r.NAMES[1]);rows[2]['training']['cagr']=.211;self.assertEqual(self.select(rows)['primary'],r.NAMES[2])
 def test_source_support_failure_stops_promotion(self):
  rows=self.rows();rows[1]['training']['cagr']=.3;res=r.select_first88(rows,self.rows(),{'passed':False},{n:{'passed':True}for n in r.NAMES[1:]});self.assertFalse(res['admitted'])
 def test_later34_not_present_or_used(self):
  rows=self.rows();rows[1]['training']['cagr']=.21;rows[2]['later34']={'cagr':999};self.assertEqual(self.select(rows)['primary'],r.NAMES[1]);self.assertFalse(self.select(rows)['later34_used_for_selection'])
 def test_changed_held_sets_not_order(self):
  orders={n:[{'symbols':['AAPL','MSFT','S']}for _ in range(122)]for n in r.NAMES};orders[r.NAMES[1]][0]['symbols']=['MSFT','S','AAPL'];s=r.changed_support(orders,[],r.NAMES[1]);self.assertEqual(s['months'],[])
 def test_changed_event_and_era_support(self):
  orders={n:[{'symbols':['AAPL','MSFT','S']}for _ in range(122)]for n in r.NAMES};cells=[]
  for i in(0,1,53):
   orders[r.NAMES[1]][i]['symbols']=['NVDA','MSFT','S'];cells.extend([dict(signal_index=i,available=True,event_id=s+':'+str(i),symbol=s)for s in('AAPL','NVDA')])
  support=r.changed_support(orders,cells,r.NAMES[1]);self.assertTrue(support['passed']);self.assertEqual(len(support['months']),3)
 def test_repeated_source_months_not_distinct_events(self):
  cells=[dict(signal_index=i,available=True,event_id=s,symbol=s)for i in range(88)for s in('AAPL','MSFT','NVDA')];self.assertEqual(r.source_support(cells)['distinct_events'],3);self.assertFalse(r.source_support(cells)['passed'])
 def test_real_entry_disabled_before_any_load(self):
  with patch.object(r.Backend,'load',side_effect=AssertionError('real load')):
   with self.assertRaisesRegex(RuntimeError,'Draft159'):r.Runner('unused',plan_sha='a'*64)

class AccountingWiring(unittest.TestCase):
 def backend(self):
  a,m=fixture(122);orders={name:[{'date':d,'symbols':['AAPL','MSFT']}for d in m['dates']]for name in r.NAMES}
  eq=pd.Series(np.linspace(100.,120.,len(pd.date_range(m['dates'][0],m['ends'][-1]))),index=pd.date_range(m['dates'][0],m['ends'][-1]))
  metrics={'training':{'cagr':.1},'early':{'cagr':.1},'recent':{'cagr':.1}}
  rp=NS(paths=Mock(return_value=(None,None,None,None)),fast_returns=Mock(return_value=np.zeros(10)),stats_rr=Mock(return_value={'cagr':.1}))
  replay=Mock(side_effect=lambda *a,**kw:({'annual_returns':[{'year':2016},{'year':2026}]},eq.copy(),[],[]))
  b=object.__new__(r.Backend);b.local=NS(rp=rp,old=NS(session=NS(measure=Mock(return_value=metrics))),ex=NS(e=NS(engine=NS(replay=replay))),arch=NS(old=NS(prior=NS(old=NS(previous=NS(cuts=Mock(return_value={})))))))
  return b,m,orders,rp,replay
 def test_training_is_single_first88_without_fold_selection(self):
  b,m,orders,rp,replay=self.backend();rows,opened=b.training(orders,m,None,None)
  self.assertEqual(rp.paths.call_count,1);self.assertEqual(rp.paths.call_args.args[2],88);self.assertEqual(replay.call_count,3)
  self.assertEqual([x['name']for x in rows],list(r.NAMES));self.assertEqual([x['name']for x in opened],list(r.NAMES))
  for call in replay.call_args_list:self.assertEqual(len(call.args[2]),88);self.assertEqual(call.kwargs['cost'],.0005);self.assertEqual(call.kwargs['execution'],'open')
 def test_diagnostics_exact_nine_paths_and_27_artifacts(self):
  b,m,orders,_,replay=self.backend();written=[];result=b.diagnostics(orders,m,None,None,lambda name,value:written.append(name))
  self.assertEqual(replay.call_count,9);self.assertEqual(len(written),27);self.assertEqual(set(result),set(r.NAMES))
  for name in r.NAMES:self.assertEqual(set(result[name]),{'same_close','next_open_zero','next_open_cost'})
  self.assertTrue(all(len(call.args[2])==122 for call in replay.call_args_list))

if __name__=='__main__':unittest.main()

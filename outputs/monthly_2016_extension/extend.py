"""Isolated, auditable monthly extension and finite training-only refinement."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,itertools,argparse
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_industry_refinement'));import refine as ref
sys.path.insert(0,str(O/'monthly_causal_ridge'));import ridge_study as ridge
sys.path.insert(0,str(O/'monthly_top35_extension'));import run as cap
t,e,rp=ref.t,ref.e,ref.rp
read,sha=ref.read,ref.sha

def dump(name,z):
 (P/name).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def initialize():
 assert not (P/'PLAN.json').exists()
 dump('PLAN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),dates=['2016-08-17','2026-09-24'],initialization='Initial three selected at Aug17 close, then actual monthends. Initial midmonth is an explicit convention, not recovered author holdings.',
 pool='Primary early reconstructed listed-issuer Top32+same assumed theme17. 474 cached histories, historical listing evidence, cap observation no later than previous trading day. Add CHL using Apr2016 known shares/5; PTR uses Apr2016 known total shares/100 times ADR price as H-price proxy. Shell multiplier2.1 is upper-bound scenario, not exact cap. Run foreign-cap sensitivities. From2020-12-31 preserve archived69 pools exactly.',
 uncertainties='Not certified full US market census or author universe. Retrospective cap archive, multiclass/ADR proxies, current coarse sector mapping, prior historical membership for Ridge training. Missing prelaunch ETF and insufficient295history ineligible, never filter on future returns. Frozen2021+ prices/features remain unchanged.',
 baseline='Extend exact77.6221 anchor,80.8349 industry and82.1979 annual-primary specifications. Early Ridge annually retrained on previous5years weekly mature21session excess-return targets, same102features/alpha10/half-life12. Preserve2021+ original forecasts.',
 search='Finite non-ML single-family candidates from previous refinement grid plus small recent-Ridge adjustments; exclude unavailable tree forecasts. Add bounded long-core weight .55/.60/.65 and skip-two-month 252-vs294 blend25/50%, support scale proportionately. No per-ticker/date rules or author holding labels.',
 selection='Return-priority subject to trainSharpe>=frozen industry-.10 and maxDD>=industry-.05. Training2016-08-17..2023-11-30. Freeze highest CAGR primary and robust control before evaluation Dec2023..Sep2026. Also yearly training return diagnostics. No post-holdout grid expansion. Reused history, not pristine OOS.',
 execution='Long-only max3 equal,2half,<2cash; fixed monthly adjusted-total-return units. Zero-cost same-close primary. Next-open0cost and5bp/side controls, independent replay. Do not sum monthly and daily strategy returns.',
 protection='No observer/source/prior-research/automation mutation. No live orders. Author2016 chart statistics reference only; no fabricated pre2021 author trades.'))
 folders=['monthly_industry_monitor','monthly_regime_monitor','monthly_buffer5_monitor','daily_stock_monitor','pressure_shadow_monitor','resilience_shadow_monitor','accel_shadow_monitor','high52_shadow_monitor','accel40_shadow_monitor','monthly_industry_refinement','monthly_guidance_tweaks']
 dump('PROTECTED_BEFORE.json',{str(f):sha(f) for folder in folders for f in (O/folder).rglob('*') if f.is_file() and f.suffix!='.pyc' and f.name!='run.lock'})
 dump('FOREIGN_EVIDENCE.json',dict(CHL=dict(shares=20475482897,ordinary_per_ADS=5,known='2016-04-26',source='https://www.sec.gov/Archives/edgar/data/1117795/000119312516555034/d151371d20f.htm'),PTR=dict(shares=183020977818,ordinary_per_ADS=100,known='2016-04-28',source='https://www.sec.gov/Archives/edgar/data/1108329/000119312516561899/d264935d20f.htm',caveat='All shares priced at H-share equivalent; A-share premium not certified. Factor2 sensitivity.'),SHEL=dict(multiplier=2.1,caveat='Carried research upper-bound approximation; multiplier1/2.1 sensitivity, no claim precise total shares.',source='https://www.shell.com/investors/information-for-shareholders/share-information.html')))

def prices():
 px,raw,members,periods,sources,pools=e.load_prices();joins=[]
 for f in (P/'prices').glob('*.csv'):
  assert sha(f)==read(f.with_suffix('.source.json'))['sha256'];sources[str(f)]=sha(f);s=f.stem;z=pd.read_csv(f,index_col=0,parse_dates=True).reindex(px.index)
  if s not in px:
   px[s]=np.nan
   for v in raw.values():v[s]=np.nan
  overlap=px[s].notna()&z['Adj Close'].notna();first=px.index[overlap][0] if overlap.any() else px.index[-1]+pd.Timedelta(days=1);ratio=float(px.at[first,s]/z.at[first,'Adj Close']) if overlap.any() else 1.;gap=px[s].isna()&(px.index<first)
  joins.append(dict(symbol=s,join=str(first.date()),ratio=ratio,overlap_max_ratio_error=float(((px.loc[overlap,s]/z.loc[overlap,'Adj Close'])/ratio-1).abs().max()) if overlap.any() else None))
  px.loc[gap,s]=z.loc[gap,'Adj Close']*ratio
  for k,c in [('close','Close'),('high','High'),('low','Low'),('volume','Volume'),('opening','Open')]:raw[k].loc[gap,s]=z.loc[gap,c]
 for s,cache in [('CHL','CHL'),('PTR','PTRCY')]:
  f=O/'dynamic_top100/additional_sources/tiingo'/f'{cache}_prices.json';sources[str(f)]=sha(f);z=pd.DataFrame(read(f));z.index=pd.to_datetime(z.date,utc=True).dt.tz_localize(None);assert z.index.is_unique;z=z.reindex(px.index)
  # Pre2021 US-listed portion only. Do not treat later OTC rows as NYSE.
  ok=z.index<pd.Timestamp('2021-01-01');px[s]=z.adjClose.where(ok)
  for k,c in [('close','close'),('high','high'),('low','low'),('volume','volume'),('opening','open')]:raw[k][s]=z[c].where(ok)
  raw['sector60'][s]=np.nan
 dump('PRICE_JOINS.json',joins)
 return px,raw,members,periods,sources,pools

def early_pools(px,raw,dates,shell_multiplier=2.1,ptr_multiplier=1.,include_china=True):
 records=read(O/'monthly_top35_extension/CAP_COLLECTION.json')['records'];ev=read(O/'us_listed_top32_yahoo/LISTING_EVIDENCE.json');unique={}
 for r in records:
  if r['url'] in unique and r['symbol'] not in {'GOOGL','BRK-B'}:continue
  unique[r['url']]=r
 caps={r['symbol']:pd.read_csv(r['data_file'],index_col=0,parse_dates=True).market_cap_usd for r in unique.values()};out={};audit=[]
 for date in dates:
  asof=px.index[px.index.get_loc(date)-1];rank=[]
  for s,z in caps.items():
   active,conflict=cap.listing_status(s,str(asof.date()),ev)
   if not active or (s=='LTM' and '2020-06-22'<=str(asof.date())<'2024-07-25'):continue
   a=e.actual(s);v=cap.snapshot_cap(z,asof,raw['close'][a] if a in raw['close'] else None)
   if v is None:continue
   if s=='SHEL':v['market_cap_usd']*=shell_multiplier
   rank.append(dict(symbol=s,**v))
  if include_china:
   for s,shares,ratio,k in [('CHL',20475482897,5,1),('PTR',183020977818,100,ptr_multiplier)]:
    v=raw['close'].at[asof,s]
    assert np.isfinite(v) and v>0
    rank.append(dict(symbol=s,market_cap_usd=float(v*shares/ratio*k),observation=str(asof.date()),age_days=0,source_kind='2016_known_share_carry_ADR_price_proxy'))
  rank.sort(key=lambda r:(-r['market_cap_usd'],r['symbol']));out[str(date.date())]=[r['symbol'] for r in rank[:32]]
  assert len(set(out[str(date.date())]))==32
  audit.append(dict(date=str(date.date()),asof=str(asof.date()),top40=[dict(r,rank=i+1) for i,r in enumerate(rank[:40])]))
 return out,audit

def prepare():
 px,raw,members,periods,sources,pools=prices();oldm,oldd,oldlayers,oldx,oldpx,oldop=ref.load()
 dates=[pd.Timestamp('2016-08-17')]+px.loc['2016-08-18':'2020-11-30'].index.to_series().groupby(px.loc['2016-08-18':'2020-11-30'].index.to_period('M')).last().tolist();assert len(dates)==53
 pp,audit=early_pools(px,raw,dates);dump('EARLY_RANKS.json',audit);pp.update(pools['32']);dump('POOLS.json',pp)
 ds=sorted(pp);ss=[sorted(set(map(e.actual,pp[day]))|members['theme17'][1]) for day in ds];assert all(len(z)==49 for z in ss)
 m=dict(dates=ds,ends=ds[1:]+['2026-09-24'],symbols=ss,etfs=sorted(members['theme17'][1]),early_months=53,train_months=88)
 dump('INPUTS.json',m)
 D,X,layers=features(px,raw,m,members)
 # Exact parity on all69 old periods, including ranks and eligibility.
 for key in ['core','support','eligible','high','weak']:np.testing.assert_allclose(D[key][53:],oldd[key],rtol=0,atol=1e-12)
 for key in oldx:np.testing.assert_allclose(X[key][53:],oldx[key],rtol=0,atol=1e-12)
 for key in layers:np.testing.assert_allclose(layers[key]['score'][53:],oldlayers[key]['score'],rtol=0,atol=1e-12)
 np.savez_compressed(P/'DATA.npz',**D);np.savez_compressed(P/'FEATURES.npz',**X);np.savez_compressed(P/'LAYERS.npz',**{k:v['score'] for k,v in layers.items()})
 op=raw['opening']*px/raw['close'];np.savez_compressed(P/'PRICES.npz',close=px.to_numpy(),opening=op.to_numpy(),dates=px.index.strftime('%Y-%m-%d').to_numpy(dtype='U10'),symbols=np.array(px.columns,dtype="U64"))
 for f in [O/'monthly_top35_extension/CAP_COLLECTION.json',O/'us_listed_top32_yahoo/LISTING_EVIDENCE.json',Path(ref.__file__),Path(t.__file__),Path(ridge.__file__),Path(rp.__file__),Path(e.engine.__file__)]:sources[str(f)]=sha(f)
 for r in read(O/'monthly_top35_extension/CAP_COLLECTION.json')['records']:sources[r['data_file']]=sha(r['data_file'])
 dump('SOURCE_HASHES.json',sources);dump('PREPARE_VALIDATION.json',dict(exact69_feature_parity=True,early_event_count=53,total=122,missing_warmup=int((~D['warm'][:53]).sum()),missing_names=sorted({s for i in range(53) for s,g in zip(ss[i],D['warm'][i]) if not g})))
 print('Prepared122 periods; exact69 feature parity verified',flush=True)

def features(px,raw,m,members):
 ss,ds=m['symbols'],m['dates'];N=len(ds);rank=t.ranks
 feat,states,warm=t.anchor.feature_panel(px,raw);r=px.pct_change(fill_method=None);v=r.rolling(63).std();long=px.shift(42)/px.shift(294)-1
 down=np.sqrt(r.clip(upper=0).pow(2).rolling(63).mean());beta=r.rolling(126).cov(r.SPY).div(r.SPY.rolling(126).var(),axis=0).shift(1);res=r-beta.mul(r.SPY,axis=0)
 xf={'core_half':long/np.sqrt(v.replace(0,np.nan)),'core_aligned':long/r.rolling(252).std().shift(42).replace(0,np.nan),'core_downside':long/down.replace(0,np.nan),'vol126':r.rolling(126).std(),'upside126':np.sqrt(r.clip(lower=0).pow(2).rolling(126).mean()),'residual126':res.rolling(126).std(),'upside_balance':np.sqrt(r.clip(lower=0).pow(2).rolling(126).mean())-np.sqrt(r.clip(upper=0).pow(2).rolling(126).mean()),'core252':feat['risk252']}
 shapes=t.patterns.shape_only(px,raw);dv=raw['volume']*r.clip(upper=0).abs();shapes['pressure_easing']=-(dv.rolling(5).mean()/raw['volume'].rolling(5).mean()-dv.rolling(21).mean()/raw['volume'].rolling(21).mean())
 sectors=pd.read_csv(O/'index_union/alias_resolved_inputs/sectors.csv.gz',index_col=0,parse_dates=True,low_memory=False)
 cor={b:(r.rolling(63).corr(r[b]),r.rolling(252).corr(r[b])) for b in t.strict.SECTOR.values()};mom={n:px.shift(42)/px.shift(n)-1 for n in [126,252]}
 Q={k:np.zeros((N,49)) for k in list(xf)+['sector126','sector252','stability','tail']};sh={k:np.zeros((N,49)) for k in shapes};D={k:np.zeros((N,49)) for k in ['core','support','eligible','warm']}
 for i,(day,names) in enumerate(zip(ds,ss)):
  z=t.anchor.snapshot(day,names,feat,states,warm);D['core'][i]=z['comp'].momentum;D['support'][i]=z['score']-z['comp'].momentum;D['eligible'][i]=z['eligible'];D['warm'][i]=warm.loc[day,names]
  for k,vv in xf.items():Q[k][i]=rank(vv.loc[day,names].replace([np.inf,-np.inf],np.nan))
  for k,vv in shapes.items():sh[k][i]=rank(vv.loc[day,names])
  bm=t.strict.benchmark_map(day,names,sectors,members['theme17'][1])
  for n in [126,252]:Q[f'sector{n}'][i]=rank(t.strict.relative_row(mom[n],day,names,bm))
  Q['stability'][i]=rank([-abs(cor[b][0].at[pd.Timestamp(day),s]-cor[b][1].at[pd.Timestamp(day),s]) if b is not None else np.nan for s,b in bm.items()]);Q['tail'][i]=sh['market_tail_residual252'][i]
 D['eligible']=D['eligible'].astype(bool);D['warm']=D['warm'].astype(bool);D['high']=states.loc[ds,'high_vol'].to_numpy();D['weak']=~states.loc[ds,'spy_above200'].to_numpy()
 L={'anti_chase':(-sh['rebound21_atr']-sh['positive_gap5_vol'])/2,'volume_recovery':(sh['shakeout_recovery5']+sh['dry_consolidation']-sh['failed_rebounds5'])/3,'pressure_easing':sh['pressure_easing']}
 L['shape_mix']=.3*L['anti_chase']+.4*L['volume_recovery']+.3*L['pressure_easing']
 return D,Q,{k:dict(score=v,active=np.ones(N)) for k,v in L.items()}

def load():
 m=read(P/'INPUTS.json');D=dict(np.load(P/'DATA.npz'));X=dict(np.load(P/'FEATURES.npz'));L={k:dict(score=v,active=np.ones(len(m['dates']))) for k,v in np.load(P/'LAYERS.npz').items()};z=np.load(P/'PRICES.npz');idx=pd.to_datetime(z['dates']);px=pd.DataFrame(z['close'],index=idx,columns=z['symbols']);op=pd.DataFrame(z['opening'],index=idx,columns=z['symbols'])
 if (P/'PREDICTIONS.npz').exists():
  for k,v in np.load(P/'PREDICTIONS.npz').items():L[k]=dict(score=v,active=np.ones(len(m['dates'])))
 return m,D,L,X,px,op

def forecast():
 m,D,L,x,_,_=load();px,raw,members,periods,sources,pools=prices();names=read(O/'monthly_adaptive_core/INPUTS.json')['packs']['top35_percentile']['names']
 f,_,states=e.previous.panels(px,raw);core,side=e.features(px,raw);clean={k:f[k] if k in f else core[k] if k in core else side[k] for k in names if k not in ['stock_sector_skip42_126','stock_sector_skip42_252','highvol_cmf63','trend_cmf63']}
 relatives={f'stock_sector_skip42_{n}':px.shift(42)/px.shift(n)-1 for n in [126,252]};sectors=pd.read_csv(O/'index_union/alias_resolved_inputs/sectors.csv.gz',index_col=0,parse_dates=True,low_memory=False)
 weekly=px.loc['2011-01-01':'2019-12-31'].index.to_series().groupby(px.loc['2011-01-01':'2019-12-31'].index.to_period('W-FRI')).last().tolist();days=sorted(set(weekly+pd.to_datetime(m['dates'][:53]).tolist()));style=ridge.old.market_styles(px,days);warm=px.notna().rolling(295).sum().eq(295);mom=px.shift(42)/px.shift(252)-1
 def row(day,ss):
  bm=t.strict.benchmark_map(day,ss,sectors,members['theme17'][1]);tab=pd.DataFrame({k:v.loc[day,ss] for k,v in clean.items()})
  for k,v in relatives.items():tab[k]=t.strict.relative_row(v,day,ss,bm)
  q=e.engine.normalize(tab,'percentile');q['highvol_cmf63']=q.cmf63*states.at[day,'high_market_vol'];q['trend_cmf63']=q.cmf63*states.at[day,'spy_trend'];a=q[names].to_numpy();st=style.loc[day];sc=np.array([float(st[k])*2-1 for k in ['growth','smallcap','defensive','bonds','spy_above200','highvol']]);return np.column_stack([a,a*sc[0],a*sc[-1],np.tile(sc,(len(ss),1))])
 arrays=[];targets=[];origins=[];ends=[];dates=[];oldmember=members['theme17'][0]
 for day in weekly:
  ss=sorted(oldmember.columns[oldmember.loc[day]]);end=px.index[px.index.get_loc(day)+21];yy=((px.loc[end,ss]/px.loc[day,ss]-1)-(px.at[end,'SPY']/px.at[day,'SPY']-1)).clip(-1,1);good=(warm.loc[day,ss]&mom.loc[day,ss].gt(0)&yy.notna()).to_numpy();arrays.append(row(day,ss)[good]);targets.append(yy.to_numpy()[good]);dates.append(str(day.date()));ends.append(str(end.date()))
 TX=np.stack([row(pd.Timestamp(day),ss) for day,ss in zip(m['dates'][:53],m['symbols'][:53])]);pred=np.zeros((122,49));fits=[]
 for year in range(2016,2021):
  cutoff=str(px.loc[:f'{year-1}-12-31'].index[-1].date());ii=ridge.matured_rows(dates,ends,cutoff);xx=np.vstack([arrays[i] for i in ii]);yy=np.concatenate([targets[i] for i in ii]);age=np.concatenate([np.repeat((pd.Timestamp(cutoff)-pd.Timestamp(dates[i])).days/30.4375,len(arrays[i])) for i in ii]);fit=ridge.fit_weighted_ridge(xx,yy,ridge.recency_weights(age,12),10.)
  ids=[i for i,day in enumerate(m['dates'][:53]) if pd.Timestamp(m['ends'][i]).year==year];pp=np.einsum('ijk,k->ij',TX[ids],fit['coef'],optimize=False)+fit['intercept']
  for j,i in enumerate(ids):pred[i]=t.ranks(pp[j],D['eligible'][i])
  fits.append(dict(year=year,cutoff=cutoff,training_start=dates[ii[0]],latest_label_end=max(ends[i] for i in ii),rows=len(xx),alpha=10,half_life_months=12,coef=fit['coef'].tolist(),intercept=float(fit['intercept']),training_pool='inherited pre2021 historical proxy; no author labels'))
  print('Early Ridge',year,'rows',len(xx),'mature label',fits[-1]['latest_label_end'],flush=True)
 pred[53:]=np.load(O/'monthly_guidance_tweaks/PREDICTIONS.npz')['ridge_10_12'];np.savez_compressed(P/'PREDICTIONS.npz',ridge_10_12=pred);dump('FORECAST_AUDIT.json',fits)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','prepare','forecast']);globals()[ap.parse_args().phase]()

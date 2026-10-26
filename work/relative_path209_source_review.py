"""Source-only complete official-month coverage; no209 labels/fits/orders/NAV."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def require(ok,msg):
    if not ok:raise ValueError(msg)
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prior=ROOT/'outputs/monthly_relative_severity191'
plan=load(prior/'PLAN.json')
require(sha(prior/'PLAN.json')=='573e04e327544782a091ea3d9651809d4a6e5c2e55dd61adf33ea8618f99f97d','191 PLAN changed')
require(sha(prior/'prepared/COMPLETION.json')=='3a2ca3c727d76ef708c1527a5b0d677e4988ff746575865dfe24266ea14bb069','191 prepared changed')
h=module('path209_frozen_source_host',ROOT/'work/monthly_relative_severity191_host.py')
h.verify(plan['input_sha256']);h.verify(plan['raw_source_hashes'])
for name,digest in load(prior/'prepared/COMPLETION.json')['artifacts_sha256'].items():require(sha(prior/'prepared'/name)==digest,'191prepared drift '+name)
study=h.local();h.original_binding(study,plan);meta,arrays,px,op=study.load()
independent=module('path209_frozen_source_independent',ROOT/'work/monthly_relative_severity191_independent.py')
cached,trainmeta,sessions,identity,calendar,parity=h.source_inputs(study,independent,plan,meta,arrays,px,op)
with np.load(prior/'prepared/LABELS.npz',allow_pickle=False) as z:
    # Targets deliberately notread: coverage only.
    old={k:z[k].copy() for k in ('dates','ends','symbols','base_good','good')}
dates=np.asarray(old['dates'],dtype='datetime64[D]');ends=np.asarray(old['ends'],dtype='datetime64[D]')
symbols=old['symbols'];base=cached['common_good'] & cached['train_eligible'] & np.isfinite(cached['train_plain']).all(axis=1)
np.testing.assert_array_equal(base,old['base_good'])
require(len(dates)==len(ends)==len(symbols)==8820,'Training axis')
axis=np.asarray(sessions,dtype='datetime64[D]'); price_axis=px.index.to_numpy(dtype='datetime64[D]')
monthlast={str(d)[:7]:d for d in axis};complete=np.zeros(len(dates),bool);records=[]
for day in np.unique(dates):
    ids=np.flatnonzero(dates==day);es=np.unique(ends[ids]);require(len(es)==1,'Mixedends')
    end=es[0];nextmonth=str(day.astype('datetime64[M]')+1)
    require(monthlast[str(day)[:7]]==day and monthlast[nextmonth]==end,'Official month anchors')
    position=np.searchsorted(axis,end);require(position+1<len(axis) and axis[position+1].astype('datetime64[M]')>end.astype('datetime64[M]'),'Full calendarmonth')
    wanted=axis[(axis>=day)&(axis<=end)];require(len(wanted)>=2,'Path too short')
    require(np.isin(wanted,price_axis).all(),'Missingofficialpriceaxis')
    market=px.loc[pd.DatetimeIndex(wanted),'SPY'].to_numpy(float)
    market_ok=bool(np.isfinite(market).all() and (market>0).all())
    issues=[]
    for i in ids:
        if symbols[i] not in px.columns:issues.append({'row':int(i),'symbol':str(symbols[i]),'reason':'missing_symbol'});continue
        path=px.loc[pd.DatetimeIndex(wanted),symbols[i]].to_numpy(float)
        complete[i]=market_ok and np.isfinite(path).all() and (path>0).all()
        if base[i] and not complete[i]:issues.append({'row':int(i),'symbol':str(symbols[i]),'reason':'base_row_invalid_path'})
    records.append(dict(signal=str(day),end=str(end),official_path_dates=[str(d) for d in wanted],rows=len(ids),base_rows=int(base[ids].sum()),complete_source_rows=int(complete[ids].sum()),issues=issues))
good=base&complete
for day in np.unique(dates):
    ids=dates==day
    if good[ids].sum()<2:good[ids]=False
np.testing.assert_array_equal(good,old['good'])
coverage=dict(training_rows=len(dates),training_months=len(np.unique(dates)),candidate191_good_rows=int(old['good'].sum()),path_complete_rows=int(complete.sum()),base_rows=int(base.sum()),path_complete_base_rows=int((base&complete).sum()),same_good_as191=True,path_complete_mask_sha256=hashlib.sha256(np.ascontiguousarray(complete).tobytes()).hexdigest(),final_good_mask_sha256=hashlib.sha256(np.ascontiguousarray(good).tobytes()).hexdigest(),records=records)
ctx=ROOT/'work/monthly_relative_path209_context.json'
pins=dict(plan['input_sha256']);pins.update(plan['raw_source_hashes'])
for p in [Path(__file__),ctx,ROOT/'work/after208_path_shortfall_coverage.json',prior/'PLAN.json',prior/'prepared/COMPLETION.json',prior/'prepared/LABELS.npz']:
    pins[str(p)]=sha(p)
require(not str(ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json') in pins,'Mutablecentralpin')
out=ROOT/'work/relative_path209_SOURCE_REVIEW.json';require(not out.exists(),'Writeonce')
report=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,source_only=True,context_sha256=sha(ctx),source_sha256=pins,raw_source_hashes=plan['raw_source_hashes'],training_path_coverage=coverage,source_identity=identity,source_parity=parity,calendar_sha256=sha(ROOT/'outputs/monthly_paired_target143/CALENDAR.json'),actual_features_computed=False,actual_labels_computed=False,actual_model_fits_computed=False,actual_orders_computed=False,actual_returns_computed=False,strict_point_in_time=False,limitations=['Revalidates inherited stored price/cache/calendar identity; does not certify historical vendor vintages or marketcap pool.','Reads191prepared metadata/masks only; no209labels or models. Original cachedfeature/rank checks are reused-source validation.'])
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':True,'output':str(out),'sha256':sha(out),'coverage':{k:v for k,v in coverage.items() if k!='records'}},indent=2))

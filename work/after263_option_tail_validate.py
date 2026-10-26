"""Offline source structure/coverage only; no strategy features, ranks or returns."""
import csv,hashlib,json,math
from datetime import datetime,timezone
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/after263_option_tail_source'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    dest=BASE/'STRUCTURAL_VALIDATION.json'
    if dest.exists():raise ValueError('Write-once source validation')
    receipt=BASE/'RECEIPTS.json'; records=json.loads(receipt.read_text())['records']
    pins={str(receipt.relative_to(ROOT)):sha(receipt)}
    for item in records:
        path=BASE/item['file']
        if item.get('status')!=200 or not item.get('saved') or sha(path)!=item['sha256']:
            raise ValueError('Source receipt/digest mismatch: '+item['file'])
        pins[str(path.relative_to(ROOT))]=item['sha256']
    raw=BASE/'SKEW_History.csv'
    if sha(raw)!='0f97cd721a6367ad3137cf8cf89bdeb8607b2f5a9636103de450183d0d1a350c':raise ValueError('Explicit source version mismatch')
    with raw.open(newline='') as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=['DATE','SKEW']:raise ValueError('Unexpected exact CSV header')
        rows=list(reader)
    days=[]
    for row in rows:
        if set(row)!= {'DATE','SKEW'} or any(v is None for v in row.values()):raise ValueError('Malformed CSV row')
        day=datetime.strptime(row['DATE'],'%m/%d/%Y').date().isoformat()
        value=Decimal(row['SKEW']);converted=float(value)
        if not value.is_finite() or value<=0 or not math.isfinite(converted) or converted<=0:raise ValueError('Nonpositive/nonfinite/underflow SKEW observation')
        days.append(day)
    if days!=sorted(set(days)):raise ValueError('Duplicate or unsorted source dates')
    cp=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
    mp=ROOT/'outputs/monthly_local_linear70/META.json'
    if sha(mp)!='3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4':raise ValueError('Original query metadata version')
    cal=json.loads(cp.read_text())['sessions'];meta=json.loads(mp.read_text())
    if len(meta['dates'])!=122 or len(cal)!=len(set(cal)) or cal!=sorted(cal):raise ValueError('Calendar/query axes')
    pins.update({str(p.relative_to(ROOT)):sha(p) for p in [cp,mp,Path(__file__)]})
    data_dates=set(days);events=[];union=set()
    for q,query in enumerate(meta['dates']):
        pos=cal.index(query)
        if pos<252:raise ValueError('Insufficient official calendar past')
        required=cal[pos-252:pos]
        missing=sorted(set(required)-data_dates);union.update(required)
        events.append(dict(query_index=q,query=query,initial_partial=q==0,required_from=required[0],required_through=required[-1],required_sessions=252,missing_dates=missing))
    missing=sorted(union-data_dates)
    scope=[d for d in cal if '2016-08-17'<=d<='2026-09-24']
    result=dict(schema='after263_skew_structural_source_validation_v1',created_utc=datetime.now(timezone.utc).isoformat(),input_sha256=pins,source_rows=len(days),source_first_date=days[0],source_last_date=days[-1],csv_header=['DATE','SKEW'],source_values_all_positive_finite=True,source_dates_unique_ordered=True,required_query_count=len(events),query_events=events,required_unique_sessions=len(union),required_union_from=min(union),required_union_through=max(union),missing_required_dates=missing,required_windows_complete=not missing,target_interval_missing_sessions=sorted(set(scope)-data_dates),source_admitted=False,methodology_revision_history_verified=False,original_publication_vintages_verified=False,strict_pit=False,features_computed=False,ranks_computed=False,orders_computed=False,returns_computed=False,new_strategy_trials=0,observers_modified=False,limitation='Current downloaded source vintage only. Complete positive CSV/window coverage does not prove historical methodology or publication vintage. Previous-day252 windows are feasibility diagnostics, not a selected strategy definition.')
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['source_rows','source_first_date','source_last_date','required_query_count','required_windows_complete','target_interval_missing_sessions','source_admitted','new_strategy_trials']}),flush=True)
    print('sha256',sha(dest),flush=True)
if __name__=='__main__':main()

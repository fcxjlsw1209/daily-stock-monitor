"""Read-only XLS extraction to reference evidence; no PIT certification."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'work/research_pdf_deps'))
import xlrd
P=Path(__file__).resolve().parent/'additional_sources/ft_rankings'
rows=[];reports=[]
for year in [2014,2015]:
 f=P/f'{year}_reference.xls';sh=xlrd.open_workbook(f).sheet_by_index(0);headers=sh.row_values(4);part=[]
 assert sh.cell_value(1,0)==f'Market values and prices at 31 March {year}'
 for i in range(5,sh.nrows):
  r=sh.row_values(i)
  if not isinstance(r[0],(int,float)) or not 1<=r[0]<=500:continue
  get=lambda col:r[headers.index(col)]
  part.append(dict(reference_date=f'{year}-03-31',global_rank=int(r[0]),company_name_as_printed=get('Company'),country=get('Country'),sector_as_printed=get('Sector'),market_cap_usd=float(get('Market value $m'))*1e6,source_file=f.name,source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),us_exchange_eligibility_verified=False,daily_membership_usable=False))
 assert len(part)==500 and {r['global_rank'] for r in part}==set(range(1,501))
 assert all(part[i]['market_cap_usd']>=part[i+1]['market_cap_usd'] for i in range(499))
 rows.extend(part);reports.append(dict(year=year,rows=500,rank_complete=True,cap_descending=True,reference_date=f'{year}-03-31'))
(P/'xls_reference_rankings.json').write_text(json.dumps(rows,indent=2));(P/'xls_validation.json').write_text(json.dumps(reports,indent=2));print(reports)

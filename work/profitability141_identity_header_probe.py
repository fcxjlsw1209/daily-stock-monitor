from pathlib import Path
import json,hashlib,re,html
R=Path.cwd();a=json.loads((R/'work/monthly_profitability141_local_annual_feasibility.json').read_text())
for x in a['annual_sources']:
 raw=(R/x['raw_path']).read_bytes();s=raw.decode('utf8',errors='replace');s=re.sub(r'<script\b.*?</script>|<style\b.*?</style>',' ',s,flags=re.S|re.I);s=html.unescape(re.sub('<[^>]+>',' ',s));s=re.sub(r'\s+',' ',s)
 pat=r'.{0,45}(?:NASDAQ|Nasdaq).{0,35}'+x['symbol']+r'.{0,30}'
 hits=re.findall(pat,s)
 print(json.dumps({'symbol':x['symbol'],'year':x['fiscal_year'],'date':x['release_date'],'raw_matches':hashlib.sha256(raw).hexdigest()==x['raw_sha256'],'ticker_literal':hits[:1]}))
p=R/'outputs/monthly_earnings_repair/batch_health_energy_foreign/raw/ABBV/ef2a6dec614e3638b479.bin';b=p.read_bytes();s=html.unescape(re.sub('<[^>]+>',' ',b.decode('utf8',errors='replace')));s=re.sub(r'\s+',' ',s)
print('ABBV',hashlib.sha256(b).hexdigest(),re.findall(r'.{0,80}(?:NYSE: ABBV|NYSE: ABT|January 2, 2013|Jan. 2, 2013|independent|stock symbol).{0,170}',s)[:8])

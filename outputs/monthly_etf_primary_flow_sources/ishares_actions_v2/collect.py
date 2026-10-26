"""Source-only collector wrapper; existing source files remain immutable."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
import hashlib,importlib.util,json,sys
ROOT=Path('/Users/shiweili/code/monthly-stock-research')
p=ROOT/'work/etf_flow_ishares_actions_collect.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='6508d46b221f83b6f67950d36d4bf6906161edebcf81c411d53ecb042e23bb6c'
spec=importlib.util.spec_from_file_location('frozen_source_collector',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.OUT=Path(__file__).resolve().parent
label,url=sys.argv[1:]
host=urlparse(url).hostname
provider=host.removeprefix('www.')
day=datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
for stop in (ROOT/'outputs/monthly_etf_primary_flow_sources').rglob('STOP_*_'+day+'.json'):
 r=json.loads(stop.read_text())
 if str(r.get('host','')).removeprefix('www.')==provider:raise RuntimeError('Provider stopped for local day: '+str(stop))
print(json.dumps(m.collect(label,url)))

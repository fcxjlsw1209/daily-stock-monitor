"""Read-only original-loader axis diagnostic after248 failed before candidates."""
from pathlib import Path
import hashlib,inspect,json
import monthly_ownsector_gate238_host as old
root=Path(__file__).resolve().parents[1]
read=old.read
study=old.local()
meta=read(old.LOCAL/'META.json')
panel,info=study.arch.old.panel_load(meta['etfs'])
raw=study.raw_source_hashes(info)
plan=read(root/'outputs/monthly_calendar_clock248/PLAN.json')
assert raw==plan['raw_source_hashes']
old.verify(raw)
source=read(root/'work/calendar_clock248_SOURCE_REVIEW.json')
manifest=read(root/source['panel_manifest_path'])
result=dict(status='ORIGINAL_LOADER_AXIS_DIAGNOSTIC',raw_sources=64,raw_sources_exact=True,
 claimed_origin=source['panel_first_session'],manifest_start=manifest.get('start'),
 actual_origin=str(panel['adj'].index[0].date()),actual_end=str(panel['adj'].index[-1].date()),actual_rows=len(panel['adj']),
 loader_path=inspect.getsourcefile(study.arch.old.panel_load),
 original_features_recomputed=False,candidate_features_computed=False,orders_computed=False,returns_computed=False,
 protected_files_modified=False)
print(json.dumps(result,ensure_ascii=False),flush=True)
p=root/'work/calendar_clock248_ORIGIN_DIAGNOSTIC.json'
assert not p.exists()
p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(output=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())),flush=True)

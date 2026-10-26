"""Render frozen Apple cash-flow source pages for independent visual checking."""
from pathlib import Path
import json
import cashflow_source_preflight_v1 as source

root = Path(__file__).resolve().parents[1]
base = source.OUT
plan_path = base + '/PLAN.json'
plan = source.read(root, plan_path)
assert plan['stage'] == 'frozen_cashflow_source_preflight_plan'
source.safe.verify_hashes(root, plan['source_hashes'])
source.safe.verify_hashes(root, plan['source_code_hashes'])
_, pdfium = source.configure_pdf(plan)
out = root / base / 'visual_review'
out.mkdir(exist_ok=True)
records = []
for event in plan['events']:
    if event['symbol'] != 'AAPL':
        continue
    raw = source.safe.local(root, event['source']['path']).read_bytes()
    assert source.safe.digest(raw) == event['source']['sha256']
    pages = set()
    for side in source.SIDES:
        for field in source.FIELDS:
            for loc in event['locator_inventory']['candidates'][side][field]:
                pages.add(loc['page_1based'])
    doc = pdfium.PdfDocument(raw)
    # Preserve all statement pages if no locator exists, for gap diagnosis.
    for page_number in range(1, len(doc)+1):
        path = out / ('AAPL_FY%s_page%s.png' % (event['fiscal_year'], page_number))
        if path.exists():
            raise FileExistsError(path)
        page = doc[page_number-1]
        bitmap = page.render(scale=1.7)
        bitmap.to_pil().save(path)
        bitmap.close()
        page.close()
        records.append(dict(event_id=event['event_id'], fiscal_year=event['fiscal_year'],
                            page_1based=page_number, source=event['source'],
                            image=str(path.relative_to(root)), sha256=source.safe.digest(path.read_bytes())))
    doc.close()
receipt = dict(plan_sha256=source.sha(root, plan_path), renderer_sha256=source.sha(root, 'work/render_cashflow_source_preflight_v1.py'),
               purpose='Visual source review only; no admission or financial calculations', pages=records)
source.publish(root, base+'/VISUAL_PAGES.json', receipt)
print(json.dumps(dict(pages=len(records), images=[r['image'] for r in records])))

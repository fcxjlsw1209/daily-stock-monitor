import json
from pathlib import Path
import cashflow_pdf_repair_v2 as m

plan = m.v1.read(m.ROOT, m.OUT + "/PLAN.json", "126e5e6afad0ac50013c04bda9813ca3d698b6d2ff6de4991243c4edf32272d0")
data = m.v1.read(m.ROOT, m.OUT + "/RAW_FACTS.json", "57eb60d0cada5f2ac521ec2c75db9992f52fbd2ec433fb70de1da2b79d68335c")
specs = {e["event_id"]: e for e in plan["events"]}
_, pdfium = m.v1.configure_pdf(plan)
for event in data["events"]:
    fields = {g["field"] for g in event["gaps"]}
    for field in sorted(fields):
        loc = specs[event["event_id"]]["locator_inventory"]["candidates"]["current"][field][0]
        raw = m.v1.safe.local(m.ROOT, event["source"]["path"]).read_bytes()
        doc = pdfium.PdfDocument(raw)
        page = doc[loc["page_1based"] - 1]
        tp = page.get_textpage()
        try:
            for row in m.physical_rows(tp):
                if m.row_text(row).lower().startswith(m.compact(loc["row_label"])):
                    tail = row[len(m.compact(loc["row_label"])):]
                    print(json.dumps({
                        "event_id": event["event_id"], "field": field,
                        "row": m.row_text(row),
                        "groups": [{"text": m.row_text(g), "bbox": m.bbox(g)} for g in m.groups(tail)],
                        "chars": tail,
                    }))
        finally:
            tp.close()
            page.close()
            doc.close()

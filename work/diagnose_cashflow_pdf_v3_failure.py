"""Debug a failed source build; never publish accepted raw facts or bypass its gate."""
import json
import cashflow_pdf_repair_v3 as m

plan = m.v1.read(m.ROOT, m.OUT + "/PLAN.json", "a9c63e4ba60e448843d7462bfd3533cd3f729b87383b30efc7a8553421021f1f")
m.v1.safe.verify_hashes(m.ROOT, plan["source_hashes"])
m.v1.safe.verify_hashes(m.ROOT, plan["source_code_hashes"])
_, pdfium = m.v1.configure_pdf(plan)
for spec in plan["events"]:
    raw = m.v1.safe.local(m.ROOT, spec["source"]["path"]).read_bytes()
    texts = m.v1.pdf_texts(raw, plan)
    doc = pdfium.PdfDocument(raw)
    for field in m.v1.FIELDS:
        loc = spec["locator_inventory"]["candidates"]["current"][field][0]
        try:
            m.extract_cell(raw, texts, loc, plan)
        except ValueError as exc:
            page = doc[loc["page_1based"] - 1]
            tp = page.get_textpage()
            try:
                matches = []
                for row in m.physical_rows(tp):
                    if m.row_text(row).lower().startswith(m.compact(loc["row_label"])):
                        tail = row[len(m.compact(loc["row_label"])):]
                        matches.append({
                            "row": m.row_text(row),
                            "groups": [m.row_text(g) for g in m.numeric_groups(tail)],
                            "breaks": [{"left": a["text"], "right": b["text"],
                                        "index_delta": b["index"] - a["index"],
                                        "gap": b["bbox"][0] - a["bbox"][2]}
                                       for a, b in zip(tail, tail[1:])
                                       if b["index"] != a["index"] + 1
                                       or b["bbox"][0] < a["bbox"][2]],
                        })
                print(json.dumps({"event": spec["event_id"], "field": field,
                                  "failure": str(exc), "matches": matches,
                                  "diagnostic_only": True}))
            finally:
                tp.close()
                page.close()
    doc.close()

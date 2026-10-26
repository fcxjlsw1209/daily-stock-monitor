"""Bounded source-layout diagnostic for the two193 Apple PP&E parser gaps."""
import io
import json
import capex193_msft_aapl as m

_, events, pins, runtime = m.gate(m.SCOPE_SHA)
reader, pdfium = m.configure_pdf193(runtime)
out = m.OUT / 'diagnostics'
out.mkdir(parents=True, exist_ok=True)
for event in events:
    if event['symbol'] != 'AAPL' or event['fiscal_year'] not in (2016, 2018):
        continue
    raw = m.local(event['source']['path']).read_bytes()
    page_index = event['fields']['current']['operating_cash_flow']['locator']['page_1based'] - 1
    text = reader(io.BytesIO(raw)).pages[page_index].extract_text(extraction_mode='layout')
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if m.v1.pdf_row_match(line, m.PDF_LABEL):
            print(json.dumps(dict(event=event['event_id'], line_1based=i + 1, lines=lines[i-1:i+2])))
    doc = pdfium.PdfDocument(raw); page = doc[page_index]; tp = page.get_textpage()
    for row in m.pdf4.physical_rows(tp):
        if m.pdf4.row_text(row).lower().startswith(m.pdf4.compact(m.PDF_LABEL)):
            print(json.dumps(dict(event=event['event_id'], pdfium_row=m.pdf4.row_text(row))))
    path = out / ('AAPL_FY' + str(event['fiscal_year']) + '_cashflow.png')
    if path.exists():
        raise ValueError('Diagnostic render already exists')
    bitmap = page.render(scale=1.6)
    with path.open('xb') as f:
        bitmap.to_pil().save(f, format='PNG')
    bitmap.close(); tp.close(); page.close(); doc.close()
m.verify(pins)

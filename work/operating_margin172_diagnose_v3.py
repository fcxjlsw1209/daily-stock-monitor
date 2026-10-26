"""Read-only diagnosis of the frozen v3 HTML parser; no financial ratios."""
import json
import operating_margin172_v3_common as common
import operating_margin172_independent_v3 as audit

plan = common.gate('d9034467d0116008fe27091257bc70abb0244476d0cdd4b3731f6c269895367e')
sources = common.read(plan['source_manifests']['html'])
for issuer in ('MSFT', 'ABBV'):
    for report in sources[issuer]:
        raw = audit.bound_read(report['source']['raw'], plan['input_sha256'])
        try:
            audit.parse_html_statement(raw, issuer, report['fiscal_year'])
        except Exception as error:
            trace = error.__traceback__
            while trace and trace.tb_frame.f_code.co_name != 'parse_html_statement':
                trace = trace.tb_next
            row = dict(issuer=issuer, year=report['fiscal_year'], error=str(error))
            if trace:
                loc = trace.tb_frame.f_locals
                row.update({key: loc.get(key) for key in ('ti', 'ri', 'measure', 'role', 'year', 'first')})
                row['selected_cells'] = [(ci, literal, cols) for ci, _, cols, literal in loc.get('cells', [])]
                ri, physical, occupied, grid, first = [loc.get(key) for key in ('ri', 'physical', 'occupied', 'grid', 'first')]
                if ri is not None and physical is not None:
                    row['row'] = [dict(ci=ci, text=audit.node_text(cell), columns=occupied[ri, ci],
                        headers=[audit._headers(grid, first, col) for col in occupied[ri, ci]])
                        for ci, cell in enumerate(physical[ri])]
            print(json.dumps(row))

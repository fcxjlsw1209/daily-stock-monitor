"""Bounded offline review of five original Boeing earnings-release PDFs."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import os

BUNDLE = Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
sys.path.append(str(BUNDLE))
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'work/after158_ba_source_plan.json'
PLAN_SHA = '4b7067b5a0e60aece954ba6617d4aa050375d323c6d13fd3899eabdadba7f422'
OUT = 'work/after158_ba_independent_review.json'
SELF = 'work/after158_ba_independent_review.py'
VISUAL_PAGES = {2014:[4,9],2015:[4,9],2016:[4,9],2017:[3,6,7,9,15],2018:[5,8,15]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    assert sha(ROOT/PLAN) == PLAN_SHA
    plan = json.loads((ROOT/PLAN).read_text())
    pins = {PLAN:PLAN_SHA, **plan['source_hashes']}
    for name, digest in pins.items():
        assert sha(ROOT/name) == digest, name
    archive_path = next(name for name in pins if name.endswith('/0bfa47080dc9bf32a7da.bin'))
    archive = json.loads((ROOT/archive_path).read_text())['GetFinancialReportListResult']
    pages, links = {}, {}
    for target in plan['targets']:
        paths = target['paths']; raw = (ROOT/paths['raw']).read_bytes()
        receipt = json.loads((ROOT/paths['receipt']).read_text())
        binding = json.loads((ROOT/paths['text_binding']).read_text())
        assert raw.startswith(b'%PDF-') and receipt['status'] == 200
        assert receipt['url'] == receipt['final_url'] == target['url']
        assert receipt['sha256'] == sha(ROOT/paths['raw']) and receipt['bytes'] == len(raw)
        assert receipt['source_representation'] == 'original_http_bytes'
        assert binding['source_sha256'] == receipt['sha256'] and binding['text_sha256'] == sha(ROOT/paths['text'])
        found = [(r,d) for r in archive for d in r.get('Documents',[]) if d.get('DocumentPath') == target['url']]
        assert len(found) == 1, (target['fiscal_year'], len(found))
        report, document = found[0]
        links[target['fiscal_year']] = {k:report.get(k) for k in ('ReportId','ReportTitle','ReportYear','ReportDate')}
        links[target['fiscal_year']]['document'] = {k:document.get(k) for k in
            ('DocumentId','DocumentTitle','DocumentCategory','DocumentPath')}
        pages[target['fiscal_year']] = [page.extract_text() for page in PdfReader(io.BytesIO(raw)).pages]
    for name, digest in pins.items():
        assert sha(ROOT/name) == digest, name
    return plan, pins, pages, links


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


def row_values(text, label, count=4):
    pattern = re.escape(label) + ''.join(r'\s+(\$?[0-9][0-9,]*)' for _ in range(count))
    match = re.search(pattern, normalized(text))
    assert match, label
    return [value.lstrip('$') for value in match.groups()]


def review(plan, pins, pages, links):
    targets = {t['fiscal_year']:t for t in plan['targets']}
    assertions = {
        2014:('2015-01-28',['90,762','86,623']),
        2015:('2016-01-27',['96,114','90,762']),
        2016:('2017-01-25',['94,571','96,114']),
        2017:('2018-01-31',['93,392','94,571']),
        2018:('2019-01-30',['101,127','94,005'])}
    evidence=[]
    def quote(year, page, text, interpretation):
        page_text=normalized(pages[year][page-1]); needle=normalized(text)
        assert needle in page_text, (year,page,needle)
        entry=dict(fiscal_year=year,physical_pdf_page=page,raw=targets[year]['paths']['raw'],
            raw_sha256=pins[targets[year]['paths']['raw']],quote=needle,
            normalized_character_offset=page_text.index(needle),interpretation=interpretation)
        evidence.append(entry)
        return len(evidence)-1
    reports=[]
    for year,(published,expected) in assertions.items():
        page=8 if year==2018 else 9
        text=pages[year][page-1]
        assert 'Consolidated Statements of Operations' in text and '(Unaudited)' in text
        assert 'Twelve months ended December 31' in normalized(text)
        assert 'Dollars in millions, except per share data' in normalized(text)
        date_match=re.search(r'CHICAGO,\s+(January)\s+(\d{1,2}),\s+(\d{4})',normalized(pages[year][0]))
        assert date_match and f'{date_match[3]}-01-{int(date_match[2]):02}'==published
        total=row_values(text,'Total revenues')
        assert total[:2]==expected
        product=row_values(text,'Sales of products'); service=row_values(text,'Sales of services')
        reports.append(dict(fiscal_year=year,role=targets[year]['role'],raw=targets[year]['paths']['raw'],
            raw_sha256=pins[targets[year]['paths']['raw']],original_publication_dateline=published,
            dateline_physical_page=1,exact_publication_instant_certified=False,
            retrieved_at=targets[year]['retrieved_at'],archive_reference=links[year],
            statement_title='The Boeing Company and Subsidiaries - Consolidated Statements of Operations (Unaudited)',
            statement_physical_page=page,annual_current_column_year=year,annual_comparative_column_year=year-1,
            annual_header='Twelve months ended December 31',current_period_end=f'{year}-12-31',
            period_start_explicitly_printed=False,calendar_start_derivation=f'{year}-01-01',
            period_start_derivation_status='calendar inference only; not a quoted source field',
            unit_literal='Dollars in millions, except per share data',iso_currency_certified=False,
            sales_of_products_current_comparative=product[:2],sales_of_services_current_comparative=service[:2],
            total_revenues_current_comparative=total[:2],quarterly_columns_excluded=total[2:],
            accounting_basis='post-2018-adoption comparative basis' if year==2018 else 'original pre-2018-adoption reported basis'))
    for year in (2014,2015,2016):
        quote(year,4,f'During the first quarter of {year}, certain programs were realigned between Boeing Military Aircraft and Global Services & Support.',
            'Explicit intra-defense-business program realignment; do not interpret segment reallocations as consolidated total-revenue rebasing.')
    quote(2014,9,'Net loss on disposal of discontinued operations, net of taxes of',
        'The formal statement separates disposal-of-discontinued-operations below continuing-operation earnings. This is not a revenue reclassification bridge. The selected2014 revenue is also reproduced in the2015 original.')
    quote(2017,3,'Prior year results have been adjusted to reflect the realignment of the services business.',
        'A segment-table footnote.2017 introduces Global Services presentation; the original consolidated products/services comparative rows still reproduce2016.')
    quote(2017,6,'Additional exhibits are included on pages 15-18 with restated 2017 and 2016 results adjusted for the adoption of ASC 606 and ASC 715.',
        'The release explicitly separates original statements from supplemental adoption-adjusted comparative exhibits.')
    quote(2017,7,'Adoption of ASC 606 and ASC 715',
        'The section separately changes core/non-GAAP pension adjustments; these core measures are not the selected total-revenue measure.')
    quote(2017,15,'The restated amounts below reflect the impact of the adoption of ASC 606 and ASC 715.',
        'Joint adoption-adjusted table; source does not split every revenue effect between the two standards.')
    quote(2018,8,'In the first quarter of 2018, we adopted the following Accounting Standards Updates (ASU)',
        'Formal2018 statement explicitly uses adopted standards, including Topics606 and715; old2017 cannot be spliced directly into it.')
    quote(2018,5,'Beginning in 2019, all revenues and costs associated with military derivative aircraft will be reported in the Defense, Space & Security segment.',
        'Announced next-period segment transfer. It is distinct from the2018 revenue-recognition basis transition.')
    quote(2018,15,'The restated amounts below reflect the change in accounting for military derivative aircraft as well as the realignment of certain programs between Global Services and Defense, Space & Security.',
        'The supplemental2019-organization view retains the2018/2017 adopted-basis consolidated totals while changing segment assignments.')
    by_year={r['fiscal_year']:r for r in reports}
    edges=[]
    for earlier,later in ((2014,2015),(2015,2016),(2016,2017)):
        a,b=by_year[earlier],by_year[later]
        equal={field:a[field][0]==b[field][1] for field in ('total_revenues_current_comparative',
            'sales_of_products_current_comparative','sales_of_services_current_comparative')}
        assert all(equal.values())
        edges.append(dict(earlier_fiscal_year=earlier,later_fiscal_year=later,
            equality_checks=equal,scope='consolidated original-reported pre-adoption total revenue',
            comparable_on_named_original_basis=True,source_admission_changed=False,
            reasoning='Same consolidated issuer, annual heading, unit and revenue row definition; both product/service components are reproduced. Disclosed program/service reorganizations are segment presentation changes, not a substitute for this consolidated check. The2018-adoption supplemental basis is kept separate.',
            known_adoption_caveat=(later==2017),admitted_to_strategy=False))
    restated=row_values(pages[2017][14],'Total revenues',6)
    assert restated==['94,005','24,770','24,223','23,051','21,961','93,496']
    future_view=row_values(pages[2018][14],'Total revenues',6)
    assert future_view[0]==by_year[2018]['total_revenues_current_comparative'][0]
    assert future_view[-1]==by_year[2018]['total_revenues_current_comparative'][1]
    inputs=dict(pins)
    for name in (SELF,'AGENTS.md','LOCAL_SETUP.md','run_local.sh','local_run.py','local.sandbox.sb'):
        inputs[name]=sha(ROOT/name)
    parser_sources={}
    for name,module in sorted(sys.modules.items()):
        if name=='pypdf' or name.startswith('pypdf.'):
            path=getattr(module,'__file__',None)
            if path and Path(path).is_file():parser_sources[str(Path(path).resolve())]=sha(path)
    inputs.update(parser_sources)
    render=Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm').resolve()
    inputs[str(render)]=sha(render)
    source_pages={str(year):[dict(physical_page=i+1,extracted_text_sha256=hashlib.sha256(text.encode()).hexdigest())
                            for i,text in enumerate(texts)] for year,texts in pages.items()}
    for name,digest in inputs.items(): assert sha(ROOT/name)==digest,name
    return dict(schema='after158_ba_independent_original_statement_review_v1',passed=True,
        created_utc=datetime.now(timezone.utc).isoformat(),symbol='BA',source_plan_sha256=PLAN_SHA,
        source_admission_changed=False,strategy_admitted=False,original_basis_three_edges_supported=True,
        input_hashes=inputs,reports=reports,accounting_edges=edges,evidence=evidence,
        boundary_2017_to_2018=dict(direct_original_reported_edge_comparable=False,
            original2017_total_revenue='93,392',adopted_basis2017_comparative='94,005',
            original2016_total_revenue='94,571',supplemental_adopted_basis2016='93,496',
            documented_basis_break=True,adoption_disclosed_by='2018-01-31',
            adoption_effective_period='first quarter2018',formal_adopted_statement_publication='2019-01-30',
            all_changes_exclusively_attributed_to_ASC606=False,
            reason='The jointly labeled ASC606/715 supplement and adopted2018 formal statement establish a genuine consolidated-revenue basis boundary.715 concerns pension presentation, but this bounded source set does not provide a standard-by-standard bridge for every revenue change.',
            later_comparatives_backfilled=False,latest_invalid_fallback_to2017_permitted=False),
        other_changes=dict(segment_realignments_not_total_revenue_basis_changes=[2014,2015,2016,2017],
            future2019_military_derivative_transfer='Segment organization view; identical adopted-basis consolidated totals in2018 original and supplemental tables. No2019 original reviewed.',
            program_charges='747/KC-46 program charges and earnings estimates appear in the releases. They are not automatically revenue restatements; this review verifies the named reported total, not economic organic-growth comparability.'),
        review_method=dict(independent_pdf_parser='pypdf fresh extraction from raw bytes, not producer output',
            parser_sources=parser_sources,all_pdf_pages_extracted=sum(map(len,pages.values())),
            extracted_page_hashes=source_pages,complete_visual_pages=VISUAL_PAGES,
            render_tool=str(render),render_options='pdftoppm -f P -l P -png -singlefile;120dpi formal tables,100dpi context',
            producer_numeric_extraction_used=False,source_receipt_text_and_exact_official_archive_links_verified=True),
        limits=['All five releases are unaudited annual earnings releases, not independently audited full10-K accounting notes.',
            'Positive edges apply only to the explicitly named original pre-adoption consolidated total-revenue basis. Equality alone was not treated as sufficient accounting evidence.',
            'The Jan2018 release already discloses alternate restated2017/2016 figures; they must remain tagged and cannot silently overwrite earlier originals.',
            'No proof of unchanged organic business perimeter, absence of all acquisitions, program-estimate effects, or absence of all undisclosed corrections is claimed.',
            'Literal dollars in millions verified; ISO USD and a within-issuer currency-use policy are not separately admitted.',
            'Publication datelines are dates, not verified first-publication instants; archive ReportDate mixes period dates and publication dates.',
            'Actual retrieval occurred in2026. Contemporaneous byte vintage and absence of later replacement are not certified.',
            'No source-to-signal timing map, source eligibility change, growth ratio, factor, score, order, return or strategy admission is produced.'],
        activity=dict(network_requests=0,new_source_downloads=0,growth_ratios=0,new_factors=0,
                      scores=0,orders=0,returns=0,new_strategy_trials=0,observer_writes=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--page', help='fiscal-year:physical-page, one based')
    parser.add_argument('--index', action='store_true')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    plan, pins, pages, links = load()
    if args.write:
        result=review(plan,pins,pages,links)
        fd=os.open(ROOT/OUT,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'w') as stream:
            json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps(dict(passed=True,reports=5,original_basis_edges=3,adoption_boundary=True,
                             source_admission_changed=False,output=OUT),sort_keys=True))
    elif args.page:
        year, page = map(int,args.page.split(':'))
        print(pages[year][page-1])
    else:
        pattern = re.compile(r'Consolidated Statements of Operations|ASC 606|ASC 715|Topic 606|Topic 715|realign|retrospect|reclassif', re.I)
        print(json.dumps(dict(plan_sha256=PLAN_SHA, verified_files=len(pins), documents={year:dict(
            page_count=len(texts), archive_link=links[year], relevant_pages=[i+1 for i,text in enumerate(texts)
            if pattern.search(text)]) for year,texts in pages.items()}),indent=2))


if __name__ == '__main__': main()

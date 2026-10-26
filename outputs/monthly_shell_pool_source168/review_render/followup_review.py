"""Offline independent review of saved Shell follow-up source evidence."""
from pathlib import Path
from datetime import datetime, timezone
from html import unescape
import hashlib, json, re

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/monthly_shell_pool_source168'
FOLLOWUP = OUT / 'external_followup'
OLD_REVIEW_SHA = 'fc103480d047ae5b8a577a86c7e733164ddd99e68418d1d0df8869d79c3fc2af'
RATIO_SHA = 'd77ca72d27ce928ec805a2c35dfb83a392025d480e78d0aab24b6a868e4a8aad'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def plain(text):
    return ' '.join(unescape(re.sub('<[^>]+>', ' ', text)).split())

def nodes(node):
    if isinstance(node, dict):
        yield node
        for child in node.get('children', []):
            yield from nodes(child)

def main():
    assert sha(OUT / 'INDEPENDENT_REVIEW.json') == OLD_REVIEW_SHA
    sources = {}
    for name in ['ratio_model', 'historical_page', 'historical_model', 'euroland_lookup']:
        planpath = FOLLOWUP / (name + '.plan.json')
        receiptpath = FOLLOWUP / (name + '.receipt.json')
        plan = json.loads(planpath.read_text())
        receipt = json.loads(receiptpath.read_text())
        body = ROOT / receipt['path']
        assert body.is_relative_to(ROOT) and body == FOLLOWUP / (name + '.body')
        assert plan['name'] == receipt['name'] == name
        assert plan['url'] == receipt['url'] == receipt['final_url']
        assert receipt['status'] == 200 and sha(body) == receipt['sha256']
        assert len(body.read_bytes()) == receipt['bytes']
        assert plan['registered_utc'] <= receipt['requested_utc'] <= receipt['completed_utc']
        sources[name] = dict(plan_sha256=sha(planpath), receipt_sha256=sha(receiptpath),
                             **receipt)
    assert sources['ratio_model']['sha256'] == RATIO_SHA
    original_path = OUT / 'external/issuer_2016q2_ads_ratio.html'
    original_receipt_path = OUT / 'external/issuer_2016q2_ads_ratio.receipt.json'
    original_receipt = json.loads(original_receipt_path.read_text())
    assert sha(original_path) == original_receipt['sha256']
    original = original_path.read_text()
    ratio = json.loads((FOLLOWUP / 'ratio_model.body').read_text())
    historical_html = (FOLLOWUP / 'historical_page.body').read_text()
    historical = json.loads((FOLLOWUP / 'historical_model.body').read_text())
    euroland = (FOLLOWUP / 'euroland_lookup.body').read_text()
    ratio_nodes = list(nodes(ratio))
    explicit = []
    headings = []
    current_section = None
    for node in ratio_nodes:
        text = node.get('model', {}).get('text', '')
        if re.search('<h2[ >]', text):
            current_section = plain(text)
            headings.append(dict(section=current_section, node_id=node.get('id')))
        if 'Each ADS represents two ordinary shares' in plain(text):
            sentences = re.findall(r'Each ADS represents two ordinary shares.*?(?:RDS\.B\.)', plain(text))
            assert len(sentences) == 1
            explicit.append(dict(section=current_section, node_id=node['id'],
                                 statement=sentences[0]))
    assert len(explicit) == 2
    assert 'two A Shares in the case of RDS.A or two B Shares in the case of RDS.B' in explicit[0]['statement']
    assert 'two ordinary A Shares in the case of RDS.A or two ordinary B Shares in the case of RDS.B' in explicit[1]['statement']
    visible = ' '.join(plain(n.get('model', {}).get('text', '')) for n in ratio_nodes)
    assert all(date in visible for date in ['Jul 28, 2016', 'August 18, 2016', 'Sep 5, 2016'])
    metadata_date = ratio['model']['date']
    assert metadata_date == '2016-07-28T13:13:00.000Z'
    embedded = []
    for node in nodes(historical):
        for link in node.get('model', {}).get('links', []):
            if link.get('value') == sources['euroland_lookup']['url']:
                embedded.append(dict(node_id=node.get('id'), link_name=link.get('name'), url=link['value']))
    assert len(embedded) == 1 and embedded[0]['link_name'] == 'src'
    preloads = dict(
        ratio_html_to_model=sources['ratio_model']['url'] in re.findall(r'<link[^>]*rel="preload"[^>]*href="([^"]+)"', original),
        historical_html_to_model=sources['historical_model']['url'] in re.findall(r'<link[^>]*rel="preload"[^>]*href="([^"]+)"', historical_html))
    assert all(preloads.values())
    assert all(token in euroland for token in ['28/01/2022', '31/01/2022',
        '/tools/splookup/?companycode=uk-rdsa&v=old&lang=en-gb',
        '/tools/splookup/?companycode=uk-rdsa&v=new&lang=en-gb'])
    out = dict(
        schema='shell_source168_followup_independent_review_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), reviewer_source_sha256=sha(Path(__file__)),
        original_independent_review_unchanged_sha256=OLD_REVIEW_SHA,
        scope='Independent read-only review of four saved follow-up bodies and receipts; separate from the frozen twenty-PDF audit.',
        sources=sources, original_ratio_html=dict(path=str(original_path.relative_to(ROOT)),
            sha256=sha(original_path), receipt_sha256=sha(original_receipt_path)),
        receipt_and_body_checks_passed=True, source_chain_preloads=preloads,
        ratio=dict(verdict='ACCEPT_EXPLICIT_ISSUER_STATEMENT_WITH_HISTORICAL_VINTAGE_LIMITS',
            A_ADS_ordinary_shares=2, A_ADS_symbol='RDS.A', B_ADS_ordinary_shares=2, B_ADS_symbol='RDS.B',
            direct_statements=explicit, page_metadata_date=metadata_date, page_sections=headings,
            dated_sections=[dict(section='Dividend announcement', stated_date='2016-07-28'),
                dict(section=explicit[1]['section'], stated_date='2016-08-18'),
                dict(section='GBP & EUR announcement', stated_date='2016-09-05')],
            explanation='The first explicit ratio is within the July 28 dividend-announcement section; the second is repeated in the August 18 reference-price section. It is not inferred from dividend or price arithmetic.',
            timing_limits=['This is one currently retrieved page model aggregating at least July 28, August 18 and September 5 announcements.',
                'The page metadata timestamp is not evidence that later sections or every present wording were publicly available on July 28.',
                'The exact archived bytes and first-publication time of each ratio sentence have not been independently verified.',
                'This source supports the stated 2016 convention; it does not alone certify uninterrupted ratio validity for every date from 2015 through 2020 or the 2022 reorganisation.'],
            historic_every_day_ratio_certified=False, exact_original_publication_timestamp_certified=False),
        official_historical_link=dict(verdict='OFFICIAL_EMBED_CHAIN_CONFIRMED_QUOTES_NOT_VERIFIED',
            embedded_source=embedded[0],
            chain=['Saved official Shell historical-price HTML', 'Its explicit model.json preload',
                   'Model WebComponent src pointing to Euroland', 'Matching successful Euroland receipt and body'],
            until_tab_label='Share structure until 28/01/2022', from_tab_label='Share structure from 31/01/2022',
            old_lookup_relative_url='/tools/splookup/?companycode=uk-rdsa&v=old&lang=en-gb',
            new_lookup_relative_url='/tools/splookup/?companycode=uk-rdsa&v=new&lang=en-gb',
            limitations=['Official embedding establishes provenance of the link, not completeness or accuracy of every underlying quote.',
                'This review did not request or inspect the old/new query payloads, any quote row, class-specific identifiers, currency units, or adjustment convention.',
                'The historical_page plan cites a higher-level Shell URL as discovery; that higher-level response is not among these four bodies and its outgoing link was not independently checked.']),
        remaining=['Verify original-vintage or section-specific publication evidence if exact information-availability timestamps are required.',
            'Confirm date coverage and any ratio/security-identity changes outside the two explicitly dated 2016 sections.',
            'Separately validate A/B securities, ADS versus ordinary quote units, currency, adjusted versus unadjusted data and quote availability.',
            'Reconcile issued shares with intended outstanding basis and recover or replace the old vendor series through a separately registered data rule.'],
        verdict='PASS_FOLLOWUP_SOURCE_REVIEW_ONLY', network_requests=0, price_rows_inspected=0,
        prices_computed=False, market_caps_computed=False, ranks_computed=False, returns_computed=False,
        original_audit_modified=False, pool_modified=False)
    destination = OUT / 'FOLLOWUP_REVIEW.json'
    assert not destination.exists(), 'write-once review output already exists'
    destination.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + '\n')
    assert sha(OUT / 'INDEPENDENT_REVIEW.json') == OLD_REVIEW_SHA
    print(json.dumps(dict(verdict=out['verdict'], explicit_statements=len(explicit),
                          page_sections=headings, review_sha256=sha(destination)), sort_keys=True))

if __name__ == '__main__':
    main()

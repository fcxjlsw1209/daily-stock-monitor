"""Derive searchable source text offline; no fiscal-period or strategy admission."""
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import after157_annual_body_collect as collector

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'work/after157_annual_body_scope.json'
SCOPE_SHA = 'f49f335509ae9e769750a5a521f1897795aa38b907b430ecadac38cb85cc3709'


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1
        elif not self.skip and tag in ('p', 'div', 'tr', 'td', 'th', 'br', 'li', 'h1', 'h2', 'h3'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.skip:
            self.skip -= 1
        elif not self.skip and tag in ('p', 'div', 'tr', 'td', 'th', 'li'):
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_text(raw, suffix):
    if suffix == '.docx':
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            member = archive.getinfo('word/document.xml')
            if member.file_size > 64 * 1024 * 1024:
                raise ValueError('Unexpected DOCX expansion')
            root = ET.fromstring(archive.read(member))
        paragraphs = []
        for p in root.findall('.//w:p', ns):
            text = ''.join(p.itertext())
            # Retain symbol/font metadata when a checkbox is not stored as text.
            symbols = [dict(e.attrib) for e in p.findall('.//w:sym', ns)]
            if symbols:
                text += ' [DOCX symbols: ' + json.dumps(symbols, sort_keys=True) + ']'
            paragraphs.append(text)
        return '\n'.join(paragraphs), 'DOCX word/document.xml paragraphs; symbols retained'
    parser = TextParser()
    parser.feed(raw.decode('utf-8-sig', errors='strict'))
    return ''.join(parser.parts), 'HTML text in document order; scripts/styles omitted'


def main():
    scope, out, helper_sha = collector.validate_scope(ROOT, SCOPE, SCOPE_SHA)
    summary_raw = (out / 'COLLECTION.json').read_bytes()
    summary = json.loads(summary_raw)
    if summary['scope_sha256'] != SCOPE_SHA or summary['helper_sha256'] != helper_sha:
        raise ValueError('Collection summary scope mismatch')
    records = []
    rendered = {}
    for target in scope['targets']:
        binding = collector.base_binding(target, SCOPE_SHA, helper_sha)
        receipt = collector.verify_existing(out, target, binding)
        if receipt is None:
            raise ValueError('Unfinished collection')
        rpath = out / target['receipt_file']
        expected = [r for r in summary['targets'] if r['id'] == target['id']]
        if len(expected) != 1 or expected[0]['receipt_sha256'] != digest(rpath.read_bytes()):
            raise ValueError('Receipt differs from collection summary')
        record = dict(id=target['id'], role=target['role'], outcome=receipt['outcome'],
                      receipt_sha256=digest(rpath.read_bytes()), http_status=receipt['http_status'])
        if receipt['outcome'] == 'success':
            raw = (out / target['body_file']).read_bytes()
            text, method = source_text(raw, Path(target['body_file']).suffix)
            lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines()]
            text = '\n'.join(line for line in lines if line) + '\n'
            name = target['id'] + '.txt'
            rendered[name] = text.encode('utf-8')
            record.update(raw_sha256=digest(raw), text_file=name,
                          text_sha256=digest(rendered[name]), lines=len(text.splitlines()),
                          extraction_method=method)
        records.append(record)
    destination = out / 'derived_text'
    destination.mkdir(exist_ok=False)
    for name, raw in rendered.items():
        collector.write_once(destination / name, raw)
    manifest = dict(scope_sha256=SCOPE_SHA, collection_sha256=digest(summary_raw),
                    extractor_sha256=digest(Path(__file__).read_bytes()), records=records,
                    created_utc=collector.utc(), admission=False, factor_calculations=0,
                    note='Search aid only. Raw originals control checkbox/layout, version, identity and fiscal-period adjudication.')
    collector.write_json(destination / 'MANIFEST.json', manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == '__main__':
    main()

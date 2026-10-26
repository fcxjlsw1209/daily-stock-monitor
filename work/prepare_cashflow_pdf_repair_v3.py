"""Create a separate source-repair revision; do not edit frozen v2 files."""
from pathlib import Path
import hashlib
import json

root = Path.cwd()
old = root / "work/cashflow_pdf_repair_v2.py"
assert hashlib.sha256(old.read_bytes()).hexdigest() == "4e7f5c0868c080847c8a290e17df4cc68d91d556aa26e28332737b86253869b5"
text = old.read_text()
text = text.replace("cashflow_pdf_repair_v2", "cashflow_pdf_repair_v3")
text = text.replace("Isolated Apple PDF source repair.", "Isolated Apple PDF source repair, revision3 for font-dependent numeral spacing.")
needle = "require = v1.require"
pins = {}
for name in ("PLAN.json", "RAW_FACTS.json", "AUDIT.json", "MANIFEST.json", "TESTS.json"):
    path = "outputs/monthly_cashflow_pdf_repair_v2/" + name
    pins[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
for path in ("work/cashflow_pdf_repair_v2.py", "work/test_cashflow_pdf_repair_v2.py",
             "work/cashflow_pdf_repair_v2_review.json"):
    pins[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
assert text.count(needle) == 1
text = text.replace(needle, "V2_PINS = " + repr(pins) + "\n" + needle)
text = text.replace(" v1.safe.verify_hashes(root,V1_PINS)\n", " v1.safe.verify_hashes(root,V1_PINS)\n v1.safe.verify_hashes(root,V2_PINS)\n", 1)
text = text.replace("source_hashes.update(V1_PINS)", "source_hashes.update(V1_PINS);source_hashes.update(V2_PINS)")
anchor = "\ndef numeric_tail(chars):"
new = '''
NUMERIC_GAP_FONT_HEIGHT_RATIO = 0.75


def numeric_groups(chars):
 """Respect original character adjacency and font-scaled spacing.

 A narrow numeral can have more than 2pt of side-bearing before a comma
 or next digit. Joining requires consecutive original PDFium indices as
 well as a bounded geometric gap. Whitespace, separate text runs and
 separated columns cannot be joined merely to match pypdf values.
 """
 out=[]
 for c in chars:
  join=False
  if out:
   prev=out[-1][-1]
   height=max(prev['bbox'][3]-prev['bbox'][1],c['bbox'][3]-c['bbox'][1])
   gap=c['bbox'][0]-prev['bbox'][2]
   join=(c['index']==prev['index']+1 and
         0 <= gap <= NUMERIC_GAP_FONT_HEIGHT_RATIO*height)
  if not join:out.append([])
  out[-1].append(c)
 return out

'''
assert text.count(anchor) == 1
text = text.replace(anchor, "\n" + new + anchor)
assert text.count(" for group in groups(chars):") == 1
text = text.replace(" for group in groups(chars):", " for group in numeric_groups(chars):")
text = text.replace("token_gap_pt=TOKEN_GAP_PT)", "token_gap_pt=TOKEN_GAP_PT,\n  numeric_gap_font_height_ratio=NUMERIC_GAP_FONT_HEIGHT_RATIO,\n  numeric_requires_consecutive_source_character_indices=True)")
text = text.replace("  row='Exact formal row label", "  numeric_tokenization='Consecutive original PDFium character indices and nonnegative gap <=0.75 times adjacent maximum glyph height; original whitespace and column geometry preserved.',\n  row='Exact formal row label")
needle = " count=sum(e['fields'][s][f] is not None"
assert text.count(needle) == 1
preserve = '''
 previous=v1.read(root,'outputs/monthly_cashflow_pdf_repair_v2/RAW_FACTS.json',V2_PINS['outputs/monthly_cashflow_pdf_repair_v2/RAW_FACTS.json'])
 old={e['event_id']:e for e in previous['events']}
 preserved=0
 for event in rows:
  require(event['event_id'] in old,'Original v2 event missing')
  for side in v1.SIDES:
   for field in v1.FIELDS:
    prior=old[event['event_id']]['fields'][side][field]
    if prior is None:continue
    current=event['fields'][side][field]
    require(current is not None,'V3 cannot lose a recovered V2 value')
    for key in ('displayed_amount','value_decimal','scale','currency_literal','currency_iso','period','available_from','source'):
     require(current[key]==prior[key],'V3 changed recovered V2 field: '+key)
    preserved+=1
 require(preserved==previous['counts']['extracted_cells']==50,'Exact50 recovered V2 values required')
'''
text = text.replace(needle, preserve + needle)
text = text.replace("  v1_frozen_artifacts_unchanged=True,", "  v1_frozen_artifacts_unchanged=True,v2_frozen_artifacts_unchanged=True,v2_values_preserved=preserved,")
test = (root / "work/test_cashflow_pdf_repair_v2.py").read_text().replace("cashflow_pdf_repair_v2", "cashflow_pdf_repair_v3")
anchor = "\nif __name__=='__main__':"
extra = '''
class FontSpacingTests(unittest.TestCase):
 def test_narrow_digit_side_bearing_does_not_split_numeral(self):
  p=page(('11','222'))
  indexes=[i for i,(ch,b) in enumerate(p.chars) if b[1]==400 and b[0]>=380 and b[0]<400]
  first=indexes[0]
  ch,b=p.chars[first]
  p.chars[first]=(ch,(b[0],b[1],b[0]+2.4,b[3]))
  self.assertEqual(geometry(p,('11','222'))['selected_cell']['text'],'11')
 def test_real_whitespace_prevents_numeric_join(self):
  p=page(('1 1','222'))
  with self.assertRaises(ValueError):geometry(p,('11','222'))
 def test_large_gap_is_not_removed_to_match_expected_value(self):
  p=page(('11','222'))
  for i,(ch,b) in enumerate(p.chars):
   if b[1]==400 and b[0]==385:p.chars[i]=(ch,(399,b[1],403,b[3]))
  with self.assertRaises(ValueError):geometry(p,('11','222'))
 def test_source_index_discontinuity_is_not_removed(self):
  chars=[{'text':'1','index':10,'bbox':[0,0,3,8]},
         {'text':'2','index':12,'bbox':[4,0,7,8]}]
  self.assertEqual(len(m.numeric_groups(chars)),2)
 def test_font_scaled_gap_still_keeps_financial_columns_separate(self):
  result=geometry(page(('11','11')),('11','11'))
  self.assertEqual(len(result['pdfium_all_row_cells']),2)

'''
assert test.count(anchor) == 1
test = test.replace(anchor, "\n" + extra + anchor)
paths = {"work/cashflow_pdf_repair_v3.py": text, "work/test_cashflow_pdf_repair_v3.py": test}
for path, content in paths.items():
    with (root / path).open("x") as out:
        out.write(content)
print(json.dumps({"created": list(paths), "original_v2_unchanged": hashlib.sha256(old.read_bytes()).hexdigest() == pins["work/cashflow_pdf_repair_v2.py"]}))

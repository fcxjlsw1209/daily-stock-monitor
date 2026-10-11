import unittest
from collect_class_shares import extract
class ClassSharesTest(unittest.TestCase):
 def test_economic_conversion_requires_rights(self):
  html=b'''<html><body><context id="a">us-gaap:CommonClassAMember2020-01-01</context><context id="b">us-gaap:CommonClassBMember2020-01-01</context><ix:nonfraction name="dei:EntityCommonStockSharesOutstanding" contextref="a" scale="0">2</ix:nonfraction><ix:nonfraction name="dei:EntityCommonStockSharesOutstanding" contextref="b" scale="0">100</ix:nonfraction></body></html>'''
  self.assertEqual(extract(html,'0001326801')['val'],102)
  self.assertIsNone(extract(html,'0001067983'))
  self.assertEqual(extract(html.replace(b'</body>',b'1/1,500</body>'),'0001067983')['val'],3100)
if __name__=='__main__':unittest.main()

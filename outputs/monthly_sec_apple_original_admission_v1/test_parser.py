"""Synthetic Apple publication and existing strict statement integration tests."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('apple_test_parser',P/'parser.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
URL='https://www.apple.com/newsroom/2016/10/apple-reports-fourth-quarter-results/'

def html(day='2016-10-25',headline='Apple Reports Fourth Quarter Results',body='The fiscal year ended September 27, 2014.'):
 from datetime import date
 printed=date.fromisoformat(day).strftime('%B %d, %Y').replace(' 0',' ')
 article={'@type':'NewsArticle','mainEntityOfPage':URL,'headline':headline,'datePublished':day+'Z','dateModified':'2099-01-01','author':{'@id':'https://www.apple.com/#organization'},'publisher':{'@id':'https://www.apple.com/#organization'}}
 return ('<h1>'+headline+'</h1><span class="category-eyebrow__date">'+printed+'</span><p>'+body+'</p><script type="application/ld+json">'+json.dumps(article)+'</script>').encode()

def page():
 return '''Apple Inc.
UNAUDITED CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except number of shares which are reflected in thousands and per share amounts)
Three Months Ended Twelve Months Ended
September 24, September 26, September 24, September 26,
2016 2015 2016 2015
Net sales $ 11,002 $ 10,001 $ 44,008 $ 40,004
Cost of sales 5,000 4,000 20,000 16,000
'''

def payload():return dict(text=page(),chars=[],bbox=[0,0,612,792],version='0.11.9',page_count=3)

class ParserTests(unittest.TestCase):
 def test_publication_visible_and_structured(self):
  x=p.publication(html(),URL);self.assertEqual(x['release_date'],'2016-10-25');self.assertEqual(x['fiscal_quarter'],4)
 def test_mismatch_publication_rejected(self):
  with self.assertRaises(ValueError):p.publication(html().replace(b'2016-10-25Z',b'2016-10-24Z'),URL)
 def test_duplicate_date_rejected(self):
  with self.assertRaises(ValueError):p.publication(html()+b'<span class="category-eyebrow__date">October 25, 2016</span>',URL)
 def test_duplicate_article_rejected(self):
  with self.assertRaises(ValueError):p.publication(html()+html(),URL)
 def test_wrong_issuer_or_url(self):
  for raw,url in [(html(headline='Other Reports Fourth Quarter Results'),URL),(html(),URL+'x')]:
   with self.subTest(url=url),self.assertRaises(ValueError):p.publication(raw,url)
 def test_wrong_publisher_rejected(self):
  with self.assertRaises(ValueError):p.publication(html().replace(b'www.apple.com/#organization',b'evil.example/#organization'),URL)
 def test_future_boundary_rejected(self):
  with self.assertRaises(ValueError):p.publication(html(body='fiscal year ended September 30, 2017'),URL)
 def test_non_fiscal_date_not_boundary(self):
  self.assertEqual(p.publication(html(body='The quarter ended September 24, 2016.'),URL)['year_ends'],[])
 def test_explicit_fiscal_boundary(self):
  self.assertEqual(p.publication(html(),URL)['year_ends'][0]['end'],'2014-09-27')
 def test_footer_boundary_not_consumed(self):
  self.assertEqual(len(p.publication(html()+b'<footer>fiscal year ended September 30, 2017</footer>',URL)['year_ends']),1)
 def test_annual_financial_and_both_boundaries(self):
  x=p.financial(payload());self.assertEqual(x['statement']['current_usd'],44008000000);self.assertEqual(len(x['year_ends']),2)
 def test_quarter_amount_not_substituted(self):
  x=payload();x['text']=x['text'].replace('11,002','99,999');self.assertEqual(p.financial(x)['statement']['current_usd'],44008000000)
 def test_missing_amount_rejected(self):
  x=payload();x['text']=x['text'].replace('$ 40,004','');
  with self.assertRaises(ValueError):p.financial(x)
 def test_unit_and_nonconsolidated_rejected(self):
  for old,new in [('In millions','In billions'),('CONSOLIDATED','SEGMENT')]:
   x=payload();x['text']=x['text'].replace(old,new)
   with self.subTest(new=new),self.assertRaises(ValueError):p.financial(x)
 def test_revenue_negative_rejected(self):
  x=payload();x['text']=x['text'].replace('$ 44,008','$ -44,008')
  with self.assertRaises(ValueError):p.financial(x)
 def test_header_year_direction_rejected(self):
  x=payload();x['text']=x['text'].replace('2016 2015 2016 2015','2015 2016 2015 2016')
  with self.assertRaises(ValueError):p.financial(x)

if __name__=='__main__':unittest.main()

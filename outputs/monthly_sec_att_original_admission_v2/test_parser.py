import importlib.util,unittest,calendar
from copy import deepcopy
from pathlib import Path
s=importlib.util.spec_from_file_location('att_test_parser',Path(__file__).with_name('parser.py'));p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
IDENTITY='<div class="col-sm-10 col-sm-offset-1"><p>DALLAS, Dec. 16, 2015 /PRNewswire/ -- release.</p><p>About AT&T AT&T Inc. (NYSE: T ) company.</p><p>SOURCE AT&T Inc.</p></div>'
def payload(q=2,year=2020,explicit=False):
 month=q*3;day=calendar.monthrange(year,month)[1];mm=calendar.month_name[month];pub=f'{calendar.month_name[month+1] if q<4 else "January"} 28, {year if q<4 else year+1}';dy=['Three Months Ended']+([]if q==1 else[{2:'Six Months Ended',3:'Nine Months Ended',4:'Twelve Months Ended'}[q]])
 header=' '.join(d+' '+mm+' '+str(day)+','for d in dy)if explicit else ('First','Second','Third','Fourth')[q-1]+' Quarter'+(' '+{2:'Six-Month Period',3:'Nine-Month Period',4:'Year Ended'}[q]if q>1 else'')
 text='AT&T INC. FINANCIAL DATA\nConsolidated Statements of Income\nDollars in millions except per share amounts\nUnaudited '+header+'\n'+' '.join([str(year),str(year-1)]*(2 if q>1 else 1))+' As Adjusted\nOperating Revenues\nTotal Operating Revenues 100 90'+(' 300 260'if q>1 else'')
 words=[]
 def w(t,x,y,wide=None):words.append(dict(text=t,x0=x,x1=x+(wide if wide is not None else len(t)*4),top=y,bottom=y+8))
 for t,x in [('Total',50),('Operating',74),('Revenues',114)]:w(t,x,200)
 for i in range(2 if q>1 else 1):
  base=240+180*i
  w(str(year),base,145);w(str(year-1),base+50,145);w('Percent',base+100,140)
  w(str(100 if i==0 else 300),base+4,200);w(str(90 if i==0 else 260),base+54,200);w('1.1',base+99,200);w('%',base+116,200)
 cover=dict(page=1,text=f'Q{q} {year} AT&T EARNINGS Investor Briefing No. 300 | {pub.upper()}',words=[],width=612,height=792)
 income=dict(page=2,text=text,words=words,width=612,height=792)
 balance=dict(page=3,text=f'AT&T INC. FINANCIAL DATA Consolidated Balance Sheets Dollars in millions Unaudited {mm} {day}, Dec. 31, {year} {year-1} Assets Cash',words=[],width=612,height=792)
 md=f'D:{year if q<4 else year+1}{month+1 if q<4 else 1:02d}28'
 return dict(engine='pdfplumber',version='0.11.9',metadata=dict(CreationDate=md,ModDate=md),pages=[cover,income,balance])
class ParserTests(unittest.TestCase):
 def test_nominal_four_quarters(self):
  for q in range(1,5):self.assertEqual(p.parse_pdf(payload(q),'2022-01-01')['statement']['periods'][-1]['months_printed'],q*3)
 def test_explicit_four_quarters(self):
  for q in range(1,5):self.assertEqual(p.parse_pdf(payload(q,explicit=True),'2022-01-01')['statement']['periods'][-1]['months_printed'],q*3)
 def test_calendar_leap_and_adjusted(self):
  a=p.financial(payload()['pages'])['periods'][-1];self.assertEqual(a['current']['duration_days'],182);self.assertEqual(a['prior']['duration_days'],181);self.assertTrue(a['as_adjusted_label_present']);self.assertFalse(a['prior']['end_directly_printed'])
 def test_fragment_join(self):
  x=payload();ws=x['pages'][1]['words'];at=next(i for i,w in enumerate(ws)if w['text']=='300');w=ws.pop(at);ws +=[dict(w,text='3',x1=w['x0']+4),dict(w,text='00',x0=w['x0']+4)];self.assertEqual(p.financial(x['pages'])['periods'][-1]['current']['revenue_usd'],300000000)
 def test_percent_rejected_as_money(self):
  x=payload();next(w for w in x['pages'][1]['words']if w['text']=='300')['text']='300%'
  with self.assertRaises(p.ParseError):p.financial(x['pages'])
 def test_crossing_token(self):
  x=payload();next(w for w in x['pages'][1]['words']if w['text']=='300')['x1']=480
  with self.assertRaisesRegex(p.ParseError,'ambiguous'):p.financial(x['pages'])
 def test_conflicting_explicit_date(self):
  x=payload(explicit=True);x['pages'][1]['text']=x['pages'][1]['text'].replace('June 30','March 31')
  with self.assertRaises(p.ParseError):p.financial(x['pages'])
 def test_wrong_year_order(self):
  x=payload();next(w for w in x['pages'][1]['words']if w['text']=='2020')['text']='2019'
  with self.assertRaises(p.ParseError):p.financial(x['pages'])
 def test_wrong_nominal_duration(self):
  x=payload();x['pages'][1]['text']=x['pages'][1]['text'].replace('Six-Month','Nine-Month')
  with self.assertRaises(p.ParseError):p.financial(x['pages'])
 def test_duplicate_income(self):
  x=payload();x['pages'].append(deepcopy(x['pages'][1]))
  with self.assertRaises(p.ParseError):p.financial(x['pages'])
 def test_duplicate_balance(self):
  x=payload();x['pages'].append(deepcopy(x['pages'][2]))
  with self.assertRaises(p.ParseError):p.financial(x['pages'])
 def test_units(self):
  x=payload();x['pages'][1]['text']=x['pages'][1]['text'].replace('Dollars','Euros')
  with self.assertRaises(p.ParseError):p.financial(x['pages'])
 def test_infinite_page(self):
  x=payload();x['pages'][1]['width']=float('inf')
  with self.assertRaises(p.ParseError):p.financial(x['pages'])
 def test_cover_and_metadata_guards(self):
  x=payload();x['metadata']['ModDate']='D:20210101'
  with self.assertRaisesRegex(p.ParseError,'version_after'):p.parse_pdf(x,'2020-07-29')
  x=payload();x['pages'][0]['text']+=' UPDATED'
  with self.assertRaisesRegex(p.ParseError,'dated_original_cover'):p.parse_pdf(x,'2020-07-29')
 def test_dated_revision_not_guidance(self):
  x=payload();x['pages'][1]['text']+=' UPDATED 2020 OUTLOOK';p.metadata_guard(x['metadata'],x['pages'],'2020-07-29');x['pages'][1]['text']+=' Updated 2/3/2020'
  with self.assertRaisesRegex(p.ParseError,'revised'):p.metadata_guard(x['metadata'],x['pages'],'2020-07-29')
 def test_cover_statement_conflict(self):
  x=payload();x['pages'][0]['text']=x['pages'][0]['text'].replace('Q2 2020','Q3 2020')
  with self.assertRaisesRegex(p.ParseError,'cover_statement_period_conflict'):p.parse_pdf(x,'2020-07-29')
 def test_identity_container(self):self.assertEqual(p.identity(IDENTITY)['evidence_date'],'2015-12-16')
 def test_hidden_or_wrong_identity(self):
  for raw in [IDENTITY.replace('(NYSE: T )','(NYSE: X )'),IDENTITY.replace('<div class','<div hidden class'),IDENTITY+IDENTITY]:
   with self.assertRaises(p.ParseError):p.identity(raw)
class ViewportTests(unittest.TestCase):
 def word(self,**values):return dict(dict(text='1',x0=565.99,x1=571.02,top=1019.65,bottom=1028.7,doctop=15000,upright=True,direction='ltr'),**values)
 def test_off_page_preserves_raw_box(self):
  x=payload();w=self.word();x['pages'][1]['words'].append(w);before=deepcopy(x);out=p.financial(x['pages']);self.assertEqual(x,before);self.assertEqual(out['excluded_off_page_words'],[dict(page=2,word=w)]);self.assertEqual(out['periods'][-1]['current']['revenue_usd'],300000000)
 def test_clean_has_no_extra_field(self):self.assertNotIn('excluded_off_page_words',p.financial(payload()['pages']))
 def test_all_wholly_off_page_sides(self):
  for w in [self.word(),self.word(x0=-10,x1=0,top=50,bottom=60),self.word(x0=612,x1=620,top=50,bottom=60),self.word(top=-10,bottom=0)]:
   kept,dropped=p.visible_words([w],612,792,2);self.assertEqual(kept,[]);self.assertEqual(dropped,[dict(page=2,word=w)])
 def test_partial_crossing_rejected(self):
  for w in [self.word(top=780,bottom=800),self.word(x0=-1,x1=2,top=20,bottom=30),self.word(x0=610,x1=613,top=20,bottom=30)]:
   with self.assertRaisesRegex(p.ParseError,'word_geometry'):p.visible_words([w],612,792,2)
 def test_nonfinite_or_inverted_off_page_rejected(self):
  for w in [self.word(top=float('inf')),self.word(x0=600,x1=590),self.word(top=1028,bottom=1019)]:
   with self.assertRaisesRegex(p.ParseError,'word_geometry'):p.visible_words([w],612,792,2)

if __name__=='__main__':unittest.main()

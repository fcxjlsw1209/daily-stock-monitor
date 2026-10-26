import collect as c,re
from urllib.parse import urlencode
u='https://www.novonordisk.com/news-and-media/news-and-ir-materials.html';p,_=c.fetch(u,'NVO');s=c.BeautifulSoup(p.read_bytes(),'html.parser')
news=s.find('input',attrs={'ref':'newsPath'})['value'];print(news,flush=True)
params=dict(searchtext='profit',start='2011-01-01',end='2016-02-29',newspath=news,limit=100,currentresults=0,function='search',language='all',type='all',disablesearchfromdb='false')
u='https://www.novonordisk.com/bin/nncorp/news-search?'+urlencode(params)
p,m=c.fetch(u,'NVO','Public official archive API for historical release discovery');d=c.load(p,{});print(str(d)[:2000],flush=True);c.dump(c.HERE/'NVO_early_archive_api.json',dict(path=c.rel(p),url=u,sha256=m['sha256']))

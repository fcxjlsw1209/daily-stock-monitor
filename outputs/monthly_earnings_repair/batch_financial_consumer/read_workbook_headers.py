"""Read-only original XLSX header extraction, retaining exact source cell locations."""
import sys,json,datetime
import openpyxl
w=openpyxl.load_workbook(sys.argv[1],read_only=True,data_only=True)
out=[]
for s in w:
 rows=[];cells=[]
 for row in s.iter_rows(min_row=1,max_row=14):
  vals=[]
  for c in row:
   if c.value is None or c.value=='':continue
   value=c.value.strftime('%B %d, %Y')if isinstance(c.value,(datetime.date,datetime.datetime))else str(c.value)
   value=value.replace('_x000D_',' ').replace('\n',' ');vals.append(value);cells.append(dict(cell=c.coordinate,value=value))
  if vals:rows.append(' '.join(vals))
 out.append(dict(sheet=s.title,text=' '.join(rows),cells=cells))
print(json.dumps(out,ensure_ascii=False))

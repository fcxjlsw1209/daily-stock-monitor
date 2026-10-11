"""Digitize supplied raster. No strategy search, trading, or account mutations."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from PIL import Image
P=Path(__file__).resolve().parent; O=P.parent
IMG=P/'SOURCE.jpg'
START=pd.Timestamp('2016-08-17'); END=pd.Timestamp('2026-09-24')

def read_prices():
 sources=[O/'index_union/alias_resolved_inputs/adj.csv.gz',O/'daily_rebuild_research/price_scaled_membership.csv.gz',O/'four_factor_rebound/ETF_SOURCES.json']
 px=pd.read_csv(sources[0],index_col=0,parse_dates=True)
 for r in json.loads(sources[2].read_text()):
  f=Path(r['file']);assert sha(f)==r['sha256'];z=pd.read_csv(f,index_col=0,parse_dates=True);px[r['symbol']]=z['Adj Close'].reindex(px.index);sources.append(f)
 member=pd.read_csv(sources[1],index_col=0,parse_dates=True)
 return px,member,sources

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def masks(image):
 a=np.asarray(image).astype(float);r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
 return {'purple':(r-g>14)&(b-g>22)&(b-r>-10)&(b-r<60)&(r>75)&(g<215),
 'gold':(r-b>32)&(g-b>18)&(r-g>0)&(r-g<80)&(r>95)&(b<180),
 'blue':(b-r>40)&(g-r>18)&(b-g>12)&(g<220)}

def panel_mask(mask,panel):
 out=mask.copy(); yy,xx=np.indices(out.shape)
 limits={'wealth':(140,475),'relative':(548,717),'dd':(800,932)}
 lo,hi=limits[panel];out&=(xx>=110)&(xx<=1154)&(yy>=lo)&(yy<=hi)
 if panel=='wealth':out&=~((xx<485)&(yy<207))
 if panel=='relative':out&=~((xx<665)&(yy<588))
 return out

def column_points(mask):
 rows=[]
 for x in range(110,1155):
  ys=np.flatnonzero(mask[:,x])
  if len(ys):rows.append((x,float(np.median(ys)),int(ys.min()),int(ys.max()),len(ys)))
 return pd.DataFrame(rows,columns=['x','y','ymin','ymax','n'])

def axis_time(dates,kind):
 return ((dates-START).days/(END-START).days).to_numpy() if kind=='calendar' else np.linspace(0,1,len(dates))

def fit_axis(q,points,kind):
 z=points[(points.x>=220)&(points.x<=1150)].copy();u=axis_time(q.index,kind);logq=np.log(q.to_numpy()/q.iloc[0])
 # y = y0 - k log(QQQ), x = x0 + width * t; leave odd columns as validation.
 train=(z.x.to_numpy()%2==0);x=z.x.to_numpy();y=z.y.to_numpy()
 best=None
 for step,span in [(0.5,4),(0.1,0.6),(0.02,0.12)]:
  cx,cw=(110,1044) if best is None else (best['x0'],best['width'])
  for x0 in np.arange(cx-span,cx+span+step/2,step):
   for w in np.arange(cw-span,cw+span+step/2,step):
    v=np.interp((x-x0)/w,u,logq);X=np.column_stack([np.ones(len(x)),-v]);b=np.linalg.lstsq(X[train],y[train],rcond=None)[0]
    residual=(X*b).sum(axis=1)-y;assert np.isfinite(residual).all();err=float(np.mean(residual[train]**2))
    if best is None or err<best['mse']:best=dict(kind=kind,x0=float(x0),width=float(w),y0=float(b[0]),log_scale=float(b[1]),mse=err,validation_rmse=float(np.sqrt(np.mean(residual[~train]**2))))
 return best

def main():
 image=Image.open(IMG).convert('RGB');ma=masks(image);px,member,sources=read_prices();q=px.loc[START:END,'QQQ']
 allpoints={}
 for c in ma:
  for panel in ['wealth','relative','dd']:
   mask=panel_mask(ma[c],panel);np.save(P/f'{c}_{panel}_mask.npy',mask);v=column_points(mask);v.to_csv(P/f'{c}_{panel}_pixels.csv',index=False);allpoints[c,panel]=v
 fits=[fit_axis(q,allpoints['gold','wealth'],kind) for kind in ['calendar','trading']];fit=min(fits,key=lambda a:a['validation_rmse']);u=axis_time(q.index,fit['kind']);dx=fit['x0']+fit['width']*u
 # Relative chart gridline locations read from screenshot, regressed on exact labels.
 ratio_levels=np.array([1,3,10,30,100,300]);ratio_y=np.array([713,684.5,653,624.5,592.5,564]);b=np.linalg.lstsq(np.column_stack([np.ones(6),-np.log(ratio_levels)]),ratio_y,rcond=None)[0]
 # Drawdown gridline locations (0,-.1,-.2,-.3,-.4); independent of purple curve.
 dd_levels=np.array([0,-.1,-.2,-.3,-.4]);dd_y=np.array([801,838,875,912,949]);db=np.linalg.lstsq(np.column_stack([np.ones(5),-dd_levels]),dd_y,rcond=None)[0]
 calibration=dict(wealth=fit,axis_candidates=fits,relative=dict(y0=float(b[0]),log_scale=float(b[1]),grid_y=ratio_y.tolist()),dd=dict(y0=float(db[0]),scale=float(db[1])),dates=[str(START.date()),str(END.date())],image_sha256=sha(IMG),qqq_wealth=float(q.iloc[-1]/q.iloc[0]),source='Axis calibrated on gold QQQ only. Relative and drawdown scale from labeled gridlines. No author purple-driven axis fitting.',notes=['Pixel columns represent several trading sessions and many visible vertical strokes, not exact daily NAV observations.','Color occlusion produces missing pixels; they remain missing.','Training/validation refer only to image calibration, not strategy out-of-sample data.'])
 # QQQ daily DD predictions checked against visible gold pixel envelopes.
 dd=q/q.cummax()-1;pred_dd=db[0]-db[1]*dd.to_numpy();mask=np.load(P/'gold_dd_mask.npy');errs=[]
 for x,y in zip(dx,pred_dd):
  if not 220<x<1150:continue
  xc=int(round(x));ys,xx=np.where(mask[:,max(0,xc-1):xc+2]);xx=xx+max(0,xc-1)
  if len(ys):errs.append(float(np.min(np.sqrt((xx-x)**2+(ys-y)**2))))
 calibration['qqq_dd_nearest_visible_pixel_median']=float(np.median(errs));calibration['qqq_dd_nearest_visible_pixel_p90']=float(np.quantile(errs,.9))
 (P/'CALIBRATION.json').write_text(json.dumps(calibration,indent=2)+'\n')
 rows=[]
 for panel in ['wealth','relative']:
  for row in allpoints['purple',panel].itertuples():
   t=(row.x-fit['x0'])/fit['width']; day=START+pd.Timedelta(days=float(t*(END-START).days)) if fit['kind']=='calendar' else q.index[min(len(q)-1,max(0,round(t*(len(q)-1))))]
   sc=calibration['wealth' if panel=='wealth' else 'relative'];logvalue=(sc['y0']-row.y)/sc['log_scale'];logqqq=np.interp(t,u,np.log(q.to_numpy()/q.iloc[0]));lw=logvalue+(logqqq if panel=='relative' else 0)
   rows.append(dict(panel=panel,x=row.x,date=str(day),log_wealth=lw,wealth=float(np.exp(lw)),y=row.y,ymin=row.ymin,ymax=row.ymax,stroke_pixels=row.n,log_half_band=(row.ymax-row.ymin+2)/2/sc['log_scale']))
 pd.DataFrame(rows).to_csv(P/'DIGITIZED_WEALTH.csv',index=False)
 (P/'SOURCE_HASHES.json').write_text(json.dumps({str(p):sha(p) for p in sources+[IMG]},indent=2)+'\n')
 # Standalone diagnostic, original remains untouched.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,1,figsize=(16,11),gridspec_kw={'height_ratios':[2,1,1]})
 regions=[(140,475),(548,725),(794,952)]
 for ax,panel,(lo,hi) in zip(axes,['wealth','relative','dd'],regions):
  ax.imshow(np.asarray(image)[lo:hi,100:1165],extent=(100,1165,hi,lo));v=allpoints['purple',panel];ax.scatter(v.x,v.y,s=3,c='#ff4081',alpha=.5,label='Detected purple pixels (column median)');ax.set_xlim(100,1165);ax.set_ylim(hi,lo);ax.set_ylabel(panel)
 axes[0].plot(dx,fit['y0']-fit['log_scale']*np.log(q/q.iloc[0]),c='lime',lw=.8,label='Known QQQ price, calibrated');axes[2].plot(dx,pred_dd,c='lime',lw=.7,label='Known QQQ drawdown');axes[0].legend(fontsize=8);axes[2].legend(fontsize=8);fig.suptitle('Author screenshot digitization: image pixels, not recovered trades');fig.tight_layout();fig.savefig(P/'EXTRACTION_QA.png',dpi=130);plt.close(fig)
 print(json.dumps(calibration,indent=2))
if __name__=='__main__':main()

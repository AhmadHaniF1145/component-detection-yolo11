"""
Auto-rapihin label: kecilkan tiap kotak supaya pas ke objek (GrabCut). Hanya MENYUSUT (tak pernah membesar).
Kotak yang tidak yakin di-segmen -> dibiarkan apa adanya. Backup otomatis dibuat.

  python tools/tighten_boxes.py --images labeling/images --labels labeling/labels
  (opsional) --prefix v3   (proses 1 video saja)
"""
import argparse, os, glob, shutil, cv2, numpy as np
def tighten(img, bpx, pad=0.05):
    x1,y1,x2,y2=bpx; H,W=img.shape[:2]; bw,bh=x2-x1,y2-y1
    if bw<10 or bh<10: return bpx
    mx,my=int(bw*0.18),int(bh*0.18)
    cx1,cy1=max(0,int(x1-mx)),max(0,int(y1-my)); cx2,cy2=min(W,int(x2+mx)),min(H,int(y2+my))
    crop=img[cy1:cy2,cx1:cx2]; ch,cw=crop.shape[:2]
    if cw<8 or ch<8: return bpx
    sc=min(1.0,170.0/max(cw,ch)); sm=cv2.resize(crop,(max(2,int(cw*sc)),max(2,int(ch*sc)))); sh,sw=sm.shape[:2]
    rx1,ry1=int((x1-cx1)*sc),int((y1-cy1)*sc); rx2,ry2=int((x2-cx1)*sc),int((y2-cy1)*sc)
    rx1=max(1,rx1);ry1=max(1,ry1);rx2=min(sw-2,rx2);ry2=min(sh-2,ry2)
    if rx2-rx1<4 or ry2-ry1<4: return bpx
    try:
        mask=np.zeros((sh,sw),np.uint8); bgd=np.zeros((1,65),np.float64); fgd=np.zeros((1,65),np.float64)
        cv2.grabCut(sm,mask,(rx1,ry1,rx2-rx1,ry2-ry1),bgd,fgd,3,cv2.GC_INIT_WITH_RECT)
    except Exception: return bpx
    fg=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)).astype(np.uint8)
    ys,xs=np.where(fg>0)
    if len(xs)<0.04*sw*sh: return bpx                       # fg kekecilan -> tak yakin
    nx1,ny1,nx2,ny2=xs.min(),ys.min(),xs.max(),ys.max()
    if (nx2-nx1)>0.95*sw and (ny2-ny1)>0.95*sh: return bpx  # tak ke-segmen -> biarkan
    fx1=cx1+nx1/sc; fy1=cy1+ny1/sc; fx2=cx1+nx2/sc; fy2=cy1+ny2/sc
    pw=(fx2-fx1)*pad; ph=(fy2-fy1)*pad; fx1-=pw;fy1-=ph;fx2+=pw;fy2+=ph
    fx1=max(fx1,x1);fy1=max(fy1,y1);fx2=min(fx2,x2);fy2=min(fy2,y2)   # hanya menyusut
    if fx2-fx1<4 or fy2-fy1<4: return bpx
    return (fx1,fy1,fx2,fy2)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images",required=True); ap.add_argument("--labels",required=True)
    ap.add_argument("--out",default=None); ap.add_argument("--prefix",default="")
    a=ap.parse_args(); out=a.out or a.labels
    if out==a.labels:
        bk=a.labels.rstrip("/\\")+"_backup_tighten"
        if not os.path.isdir(bk): shutil.copytree(a.labels,bk); print("backup ->",bk)
    os.makedirs(out,exist_ok=True)
    labs=sorted(glob.glob(os.path.join(a.labels,(a.prefix or "")+"*.txt")))
    nimg=0; nb=0; shr=[]
    for lp in labs:
        base=os.path.splitext(os.path.basename(lp))[0]
        if base=="classes": continue
        ip=None
        for e in (".jpg",".jpeg",".png",".bmp"):
            if os.path.exists(os.path.join(a.images,base+e)): ip=os.path.join(a.images,base+e); break
        if ip is None: continue
        img=cv2.imread(ip); H,W=img.shape[:2]; out_lines=[]
        for l in open(lp):
            p=l.split()
            if len(p)!=5: continue
            c=int(p[0]); cx,cy,w,h=map(float,p[1:])
            x1,y1,x2,y2=(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H
            nx1,ny1,nx2,ny2=tighten(img,(x1,y1,x2,y2))
            a0=(x2-x1)*(y2-y1); a1=(nx2-nx1)*(ny2-ny1); shr.append(a1/max(1,a0))
            ncx=((nx1+nx2)/2)/W; ncy=((ny1+ny2)/2)/H; nw=(nx2-nx1)/W; nh=(ny2-ny1)/H
            out_lines.append(f"{c} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n"); nb+=1
        open(os.path.join(out,base+".txt"),"w").writelines(out_lines); nimg+=1
    print(f"diproses: {nimg} gambar, {nb} kotak | rata2 luas jadi {np.mean(shr)*100:.0f}% dari semula")
if __name__=="__main__": main()

"""
Gandakan dataset dengan ROTASI 90/180/270 + flip. Gambar DAN label diputar otomatis (tepat,
tidak perlu re-label hasil putaran). Jalankan pada folder TRAIN saja (setelah split) biar val tetap asli.

  python tools/03_augment.py --images dataset_yolo/train/images --labels dataset_yolo/train/labels --rot 90 180 270
  (opsional) tambah  --fliph  --flipv

Catatan: ini menambah variasi orientasi saja (objek & latar sama). Untuk lompatan akurasi,
lebih ampuh menambah FOTO BARU yang benar-benar beda latar/sudut/cahaya.
"""
import argparse, os, glob, cv2
ROT={"r90":cv2.ROTATE_90_CLOCKWISE,"r180":cv2.ROTATE_180,"r270":cv2.ROTATE_90_COUNTERCLOCKWISE}
def tx(c,cx,cy,w,h,mode):
    if mode=="r90":  return c,1-cy,cx,  h,w
    if mode=="r180": return c,1-cx,1-cy,w,h
    if mode=="r270": return c,cy,  1-cx,h,w
    if mode=="flh":  return c,1-cx,cy,  w,h
    if mode=="flv":  return c,cx,  1-cy,w,h
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images",required=True); ap.add_argument("--labels",required=True)
    ap.add_argument("--rot",nargs="*",type=int,default=[90,180,270],choices=[90,180,270])
    ap.add_argument("--fliph",action="store_true"); ap.add_argument("--flipv",action="store_true")
    a=ap.parse_args()
    modes=["r"+str(r) for r in a.rot]+(["flh"] if a.fliph else [])+(["flv"] if a.flipv else [])
    SUF=("_r90","_r180","_r270","_flh","_flv")
    imgs=[f for e in (".jpg",".jpeg",".png",".bmp") for f in glob.glob(os.path.join(a.images,"*"+e))]
    added=0; src=0
    for ip in sorted(imgs):
        base,ext=os.path.splitext(os.path.basename(ip))
        if any(s in base for s in SUF): continue          # jangan augmentasi hasil augmentasi
        lp=os.path.join(a.labels,base+".txt")
        if not os.path.exists(lp): continue
        src+=1; img=cv2.imread(ip)
        rows=[l.split() for l in open(lp) if l.strip()]
        for mode in modes:
            ob=base+"_"+mode
            op=os.path.join(a.images,ob+ext)
            if os.path.exists(op): continue
            aug=cv2.rotate(img,ROT[mode]) if mode in ROT else cv2.flip(img,1 if mode=="flh" else 0)
            out=[]
            for r in rows:
                if len(r)!=5: continue
                c=int(r[0]); cx,cy,w,h=map(float,r[1:])
                c2,cx2,cy2,w2,h2=tx(c,cx,cy,w,h,mode)
                out.append(f"{c2} {cx2:.6f} {cy2:.6f} {w2:.6f} {h2:.6f}\n")
            cv2.imwrite(op,aug)
            open(os.path.join(a.labels,ob+".txt"),"w").writelines(out)
            added+=1
    print(f"sumber: {src} gambar | ditambah: {added} gambar augmentasi (mode: {modes})")
    print(f"total sekarang di {a.images}: {len(glob.glob(os.path.join(a.images,'*'+'.jpg')))} gambar")
if __name__=="__main__": main()

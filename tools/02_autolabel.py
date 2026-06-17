"""
Auto-label gambar memakai model YOLO terlatih (model-assisted labeling).
Output = file .txt format YOLO untuk DIVALIDASI di tools/label_web. Alur iteratif:
  label awal -> train -> auto-label gambar baru -> validasi web -> train lagi.

Contoh:
  python tools/02_autolabel.py --model runs/laptop_yolo11s/weights/best.pt \
      --images labeling/images --out labeling/labels --conf 0.25
"""
import argparse, os, glob
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--images",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--conf",type=float,default=0.25)
    ap.add_argument("--overwrite",action="store_true",help="timpa label yang sudah ada")
    a=ap.parse_args()
    from ultralytics import YOLO
    os.makedirs(a.out,exist_ok=True)
    model=YOLO(a.model)
    imgs=[f for e in (".jpg",".jpeg",".png",".bmp") for f in glob.glob(os.path.join(a.images,"*"+e))]
    n=0; boxes=0
    for ip in sorted(imgs):
        base=os.path.splitext(os.path.basename(ip))[0]; lp=os.path.join(a.out,base+".txt")
        if os.path.exists(lp) and not a.overwrite: continue
        r=model(ip,verbose=False,conf=a.conf)[0]
        lines=[]
        for b in r.boxes:
            c=int(b.cls.item()); x,y,w,h=b.xywhn[0].tolist()
            lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"); boxes+=1
        open(lp,"w").writelines(lines); n+=1
    print(f"auto-label selesai: {n} gambar, {boxes} box (conf>={a.conf}). REVIEW di tools/label_web!")
if __name__=="__main__": main()

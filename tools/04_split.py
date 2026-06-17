"""
Split labeling/ (images+labels) -> dataset_yolo/{train,val} + tulis data.yaml.
Folder output DIBERSIHKAN dulu tiap run (biar tidak tercampur sisa augmentasi lama).
Stratified per-prefix (v1..v4). Ganti --seed untuk 'roll' komposisi.

  python tools/04_split.py --src labeling --out dataset_yolo --train_pct 0.8 --seed 42
"""
import argparse, os, glob, random, shutil
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src",default="labeling"); ap.add_argument("--out",default="dataset_yolo")
    ap.add_argument("--train_pct",type=float,default=0.8); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    root=os.path.abspath("."); src=os.path.join(root,a.src); out=os.path.join(root,a.out)
    classes=[l.strip() for l in open(os.path.join(src,"classes.txt")) if l.strip()]
    # bersihkan output lama (hindari sisa augmentasi tercampur)
    if os.path.isdir(out): shutil.rmtree(out, ignore_errors=True)
    imgs=[f for e in (".jpg",".jpeg",".png",".bmp") for f in glob.glob(os.path.join(src,"images","*"+e))]
    # JANGAN ikutkan hasil augmentasi yg mungkin ada di src (augment dilakukan setelah split)
    imgs=[i for i in imgs if not any(s in os.path.basename(i) for s in ("_r90","_r180","_r270","_flh","_flv"))]
    pairs=[(i,os.path.join(src,"labels",os.path.splitext(os.path.basename(i))[0]+".txt")) for i in imgs]
    pairs=[(i,l) for i,l in pairs if os.path.exists(l)]
    random.seed(a.seed); groups={}
    for p in pairs: groups.setdefault(os.path.basename(p[0]).split("_")[0],[]).append(p)
    train,val=[],[]
    for g in groups.values():
        random.shuffle(g); k=int(round(len(g)*a.train_pct)); train+=g[:k]; val+=g[k:]
    for split,items in (("train",train),("val",val)):
        for sub in ("images","labels"): os.makedirs(os.path.join(out,split,sub),exist_ok=True)
        for img,lab in items:
            shutil.copy(img,os.path.join(out,split,"images",os.path.basename(img)))
            shutil.copy(lab,os.path.join(out,split,"labels",os.path.basename(lab)))
    yaml=(f"# dibuat 04_split.py (seed={a.seed})\n"
          f'path: "{out.replace(os.sep,"/")}"\ntrain: train/images\nval: val/images\n'
          f"nc: {len(classes)}\nnames: {classes}\n")
    open(os.path.join(root,"data.yaml"),"w").write(yaml)
    print(f"TRAIN={len(train)} VAL={len(val)} | kelas={classes}")
    print(f"output BERSIH -> {out}  |  data.yaml diperbarui")
if __name__=="__main__": main()

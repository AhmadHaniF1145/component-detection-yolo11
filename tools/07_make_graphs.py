"""Buat grafik perbandingan dari logs/*.csv -> assets/ (training + FPS laptop vs Pi). Robust thd skema CSV lama/baru."""
import os, csv, glob, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
L="logs"; A="assets"; os.makedirs(A,exist_ok=True); os.makedirs(f"{L}/graphs",exist_ok=True)
CY="#06b6d4"; AM="#f59e0b"; GR="#10b981"; INK="#0f172a"
def rd(p): return list(csv.DictReader(open(p))) if os.path.exists(p) else []
def fnum(x):
    try: return float(x)
    except: return None
# 1) TRAINING
tr=[r for r in rd(f"{L}/training_runs.csv") if fnum(r.get("mAP50"))]
if tr:
    n=[r["name"] for r in tr]
    plt.figure(figsize=(max(6,len(n)*1.6),4))
    import numpy as np; x=np.arange(len(n)); w=0.38
    plt.bar(x-w/2,[float(r["mAP50"]) for r in tr],w,label="mAP50",color=CY)
    plt.bar(x+w/2,[float(r["mAP50_95"]) for r in tr],w,label="mAP50-95",color=AM)
    plt.xticks(x,n,rotation=15,ha="right"); plt.ylim(0,1); plt.ylabel("mAP"); plt.title("Training Accuracy per Model"); plt.legend()
    plt.tight_layout(); [plt.savefig(f"{d}/training_map.png",dpi=130) for d in (A,f"{L}/graphs")]; plt.close()
    plt.figure(figsize=(max(6,len(n)*1.6),4))
    v=[float(r["train_time_min"]) for r in tr]; plt.bar(n,v,color=GR)
    for i,t in enumerate(v): plt.text(i,t,f"{t:.1f}m",ha="center",va="bottom")
    plt.ylabel("minutes"); plt.title("Training Time per Model"); plt.xticks(rotation=15,ha="right")
    plt.tight_layout(); [plt.savefig(f"{d}/training_time.png",dpi=130) for d in (A,f"{L}/graphs")]; plt.close()
    print("ok: training_map, training_time")
# 2) FPS per run (infer + loop)
rows=[]
for p in sorted(glob.glob(f"{L}/fps_*.csv")):
    r=rd(p)
    if not r: continue
    k0=r[0]
    ic="infer_fps" if "infer_fps" in k0 else ("fps" if "fps" in k0 else None)
    lc="loop_fps" if "loop_fps" in k0 else None
    if not ic: continue
    inf=[fnum(x[ic]) for x in r if fnum(x.get(ic))]
    loo=[fnum(x[lc]) for x in r if lc and fnum(x.get(lc))]
    tag=os.path.basename(p)[4:-4]
    rows.append((tag, st.mean(inf) if inf else 0, st.mean(loo) if loo else 0))
if rows:
    rows.sort(key=lambda r:("pi" not in r[0], r[0]))   # Pi dulu biar kebandingan
    tags=[r[0] for r in rows]; inf=[r[1] for r in rows]; loo=[r[2] for r in rows]
    import numpy as np; x=np.arange(len(tags)); w=0.4
    plt.figure(figsize=(max(8,len(tags)*1.3),4.5))
    cols=[AM if "pi" in t else CY for t in tags]
    plt.bar(x-w/2,inf,w,label="Infer FPS",color=cols)
    plt.bar(x+w/2,loo,w,label="Loop FPS",color=["#fcd34d" if "pi" in t else "#67e8f9" for t in tags])
    for i,t in enumerate(inf): plt.text(i-w/2,t,f"{t:.0f}",ha="center",va="bottom",fontsize=8)
    plt.xticks(x,tags,rotation=30,ha="right"); plt.ylabel("FPS"); plt.title("FPS Comparison — Laptop (cyan) vs Raspberry Pi (amber)"); plt.legend()
    plt.tight_layout(); [plt.savefig(f"{d}/fps_compare.png",dpi=130) for d in (A,f"{L}/graphs")]; plt.close()
    # timeline
    plt.figure(figsize=(10,4.5))
    for p in sorted(glob.glob(f"{L}/fps_*.csv")):
        r=rd(p); 
        if not r: continue
        k0=r[0]; ic="infer_fps" if "infer_fps" in k0 else ("fps" if "fps" in k0 else None)
        if not ic: continue
        y=[fnum(x[ic]) for x in r if fnum(x.get(ic))]
        plt.plot(range(len(y)),y,lw=1.1,label=os.path.basename(p)[4:-4])
    plt.xlabel("frame"); plt.ylabel("Infer FPS"); plt.title("FPS over time (per run)"); plt.legend(fontsize=7,ncol=2)
    plt.tight_layout(); [plt.savefig(f"{d}/fps_timeline.png",dpi=130) for d in (A,f"{L}/graphs")]; plt.close()
    print("ok: fps_compare, fps_timeline")
    print("RINGKAS FPS:"); [print(f"  {t:22s} infer {a:5.1f} | loop {b:5.1f}") for t,a,b in rows]

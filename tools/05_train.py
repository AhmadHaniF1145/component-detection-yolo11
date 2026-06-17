"""
Training YOLO + CATAT hasil ke logs/training_runs.csv (waktu, device, epoch, mAP, dll).
AUGMENTASI ON (putar/flip/zoom) supaya model tahan terhadap orientasi & kondisi berbeda
-> TIDAK perlu menduplikasi/memutar dataset manual.

  python tools/05_train.py --model yolo11n.pt --name laptop_yolo11n_v2 --epochs 120 --device 0
  python tools/05_train.py --model yolo11s.pt --name laptop_yolo11s_v2 --epochs 120 --device 0
"""
import argparse, os, csv, time, datetime, platform
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="yolo11n.pt")
    ap.add_argument("--data",default="data.yaml")
    ap.add_argument("--name",required=True)
    ap.add_argument("--epochs",type=int,default=120)
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--batch",type=int,default=16)
    ap.add_argument("--device",default="0")
    # --- augmentasi (default cocok utk komponen top-down yg bisa di sembarang sudut) ---
    ap.add_argument("--degrees",type=float,default=180.0,help="rotasi acak +-derajat (180=putaran penuh)")
    ap.add_argument("--fliplr",type=float,default=0.5,help="peluang flip horizontal")
    ap.add_argument("--flipud",type=float,default=0.5,help="peluang flip vertikal")
    ap.add_argument("--scale",type=float,default=0.5,help="variasi zoom")
    ap.add_argument("--mosaic",type=float,default=1.0)
    a=ap.parse_args()
    import torch
    from ultralytics import YOLO
    dev_name=torch.cuda.get_device_name(0) if (a.device!="cpu" and torch.cuda.is_available()) else "CPU"
    os.makedirs("logs",exist_ok=True)
    t0=time.time()
    model=YOLO(a.model)
    model.train(data=a.data,epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,device=a.device,
                project="runs",name=a.name,exist_ok=True,verbose=True,
                degrees=a.degrees,fliplr=a.fliplr,flipud=a.flipud,scale=a.scale,mosaic=a.mosaic)
    train_s=time.time()-t0
    save_dir=str(getattr(model.trainer,"save_dir","runs/"+a.name))   # path sebenarnya
    weights=os.path.join(save_dir,"weights","best.pt")
    m=model.val(data=a.data,device=a.device,verbose=False)
    row={"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),
         "name":a.name,"base_model":a.model,"device":dev_name,"os":platform.system(),
         "epochs":a.epochs,"imgsz":a.imgsz,"batch":a.batch,
         
         "train_time_s":round(train_s,1),"train_time_min":round(train_s/60,2),
         "sec_per_epoch":round(train_s/max(1,a.epochs),2),
         "mAP50":round(float(m.box.map50),4),"mAP50_95":round(float(m.box.map),4),
         "precision":round(float(m.box.mp),4),"recall":round(float(m.box.mr),4),
         "weights":weights}
    csvp="logs/training_runs.csv"; new=not os.path.exists(csvp)
    with open(csvp,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if new: w.writeheader()
        w.writerow(row)
    print("\n=== TERCATAT di logs/training_runs.csv ===")
    for k,v in row.items(): print(f"  {k}: {v}")
    print(f"\nMODEL: {weights}")
if __name__=="__main__": main()

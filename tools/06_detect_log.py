"""
Deteksi + OSD proporsional + LOG ke CSV + SIMPAN hasil (video/gambar). Untuk benchmark laptop vs Pi.
OSD: Model | Infer FPS | Loop FPS | Frame | tanggal-jam, + daftar komponen & jumlahnya. Teks skala mengikuti resolusi.
Log per-frame -> logs/fps_<tag>.csv ; ringkasan -> logs/runs_summary.csv ; video -> logs/rec_<tag>.mp4 (full-res).

  python tools/06_detect_log.py --model runs/detect/runs/laptop_yolo11s/weights/best.pt --source test.mp4 --tag laptop_11s_video --device 0 --save
  python tools/06_detect_log.py --model best_ncnn_model --source picamera0 --tag pi_11n_cam --device cpu --imgsz 640 --save
"""
import argparse, os, sys, glob, time, csv, datetime
import cv2, numpy as np
IMG=(".jpg",".jpeg",".png",".bmp"); VID=(".mp4",".avi",".mov",".mkv")
COL=[(0,0,255),(0,165,255),(255,255,0),(0,255,0),(255,0,255),(255,0,0),(0,255,255),(180,105,255),(98,118,150),(170,170,0)]
def open_cam(idx,res):
    for be in (cv2.CAP_DSHOW,cv2.CAP_MSMF,cv2.CAP_ANY):
        c=cv2.VideoCapture(idx,be)
        if res: c.set(3,res[0]); c.set(4,res[1])
        if c.isOpened():
            ok,_=c.read()
            if ok: return c
        c.release()
    return None
def fit_scale(txt,F,want_w,th,start):     # cari fontScale agar lebar teks <= want_w
    s=start
    while s>0.3 and cv2.getTextSize(txt,F,s,th)[0][0]>want_w: s-=0.05
    return s
def model_label(path):
    p=path.replace("\\","/").rstrip("/")
    if p.endswith("weights/best.pt") or p.endswith("weights/last.pt"):
        return os.path.basename(os.path.dirname(os.path.dirname(p)))
    return os.path.splitext(os.path.basename(p))[0]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True); ap.add_argument("--source",required=True)
    ap.add_argument("--tag",required=True); ap.add_argument("--device",default="0")
    ap.add_argument("--conf",type=float,default=0.4); ap.add_argument("--resolution",default=None)
    ap.add_argument("--imgsz",type=int,default=640); ap.add_argument("--model-name",default=None,dest="mname")
    ap.add_argument("--save",action="store_true"); ap.add_argument("--record",action="store_true")
    ap.add_argument("--no-show",action="store_true"); ap.add_argument("--win",type=int,default=1280)
    a=ap.parse_args(); do_save=a.save or a.record
    from ultralytics import YOLO
    model=YOLO(a.model); names=model.names; nc=len(names)
    mname=a.mname or model_label(a.model)
    res=tuple(map(int,a.resolution.lower().split("x"))) if a.resolution else None
    st="image" if a.source.lower().endswith(IMG) else "video" if a.source.lower().endswith(VID) \
       else "stream" if a.source.startswith(("http","rtsp")) else "usb" if a.source.startswith("usb") else "picamera" if a.source.startswith("picamera") \
       else "folder" if os.path.isdir(a.source) else "?"
    cap=None; picam=None; imgs=[]; save_fps=20.0
    if st=="image": imgs=[a.source]
    elif st=="folder": imgs=sorted(f for f in glob.glob(os.path.join(a.source,"*")) if f.lower().endswith(IMG))
    elif st=="video": cap=cv2.VideoCapture(a.source); save_fps=cap.get(cv2.CAP_PROP_FPS) or 20.0
    elif st=="stream": cap=cv2.VideoCapture(a.source)
    elif st=="usb":
        cap=open_cam(int(a.source[3:] or 0),res)
        if cap is None: sys.exit("Webcam gagal dibuka (coba index lain / cek izin kamera).")
    elif st=="picamera":
        from picamera2 import Picamera2
        picam=Picamera2(); picam.configure(picam.create_video_configuration(main={"format":"RGB888","size":res or (640,480)})); picam.start()
    else: sys.exit("source tidak dikenal")
    os.makedirs("logs",exist_ok=True)
    fcsv=open(f"logs/fps_{a.tag}.csv","w",newline=""); fw=csv.writer(fcsv)
    fw.writerow(["frame","iso_time","infer_fps","loop_fps","infer_ms","n_obj"]+[names[i] for i in range(nc)])
    WIN=f"Deteksi [{a.tag}] - q keluar"; F=cv2.FONT_HERSHEY_SIMPLEX
    vwriter=None; vw_size=None; save_dir=None
    fcount=0; ifps_hist=[]; lfps_hist=[]; obj_hist=[]; loop_prev=time.perf_counter()
    while True:
        if st in ("image","folder"):
            if fcount>=len(imgs): break
            frame=cv2.imread(imgs[fcount])
            if frame is None: fcount+=1; continue
        elif st=="picamera": frame=cv2.cvtColor(picam.capture_array(),cv2.COLOR_RGB2BGR)
        else:
            ok,frame=cap.read()
            if not ok: break
        if res and st not in ("usb","picamera"): frame=cv2.resize(frame,res)
        t0=time.perf_counter()
        r=model(frame,verbose=False,conf=a.conf,imgsz=a.imgsz,device=a.device)[0]
        infer_ms=(time.perf_counter()-t0)*1000; infer_fps=1000.0/infer_ms if infer_ms>0 else 0
        H,Wd=frame.shape[:2]; per=[0]*nc
        fs=max(0.45,Wd/1600.0); th=max(1,int(round(fs*2))); lh=int(30*fs); pad=int(10*fs)
        for b in r.boxes:
            c=int(b.cls.item()); per[c]+=1; x1,y1,x2,y2=map(int,b.xyxy[0].tolist()); col=COL[c%len(COL)]
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,max(1,th))
            cv2.putText(frame,f"{names[c]} {int(b.conf.item()*100)}%",(x1,max(y1-int(6*fs),int(14*fs))),F,0.6*fs,col,th)
        n=int(sum(per))
        now=time.perf_counter(); loop_ms=(now-loop_prev)*1000; loop_prev=now; loop_fps=1000.0/loop_ms if loop_ms>0 else 0
        ifps_hist.append(infer_fps); lfps_hist.append(loop_fps); obj_hist.append(n)
        ifps_hist=ifps_hist[-90:]; lfps_hist=lfps_hist[-90:]
        # ---- OSD bar atas ----
        barh=lh+2*pad; ov=frame.copy(); cv2.rectangle(ov,(0,0),(Wd,barh),(15,23,42),-1); frame=cv2.addWeighted(ov,0.6,frame,0.4,0)
        ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (tw,_),_=cv2.getTextSize(ts,F,fs,th); cv2.putText(frame,ts,(Wd-tw-pad,int(barh*0.68)),F,fs,(255,255,255),th)
        left=f"Model:{mname}  Infer:{infer_fps:.0f} FPS  Loop:{loop_fps:.0f} FPS  Frame:{fcount}"
        ls=fit_scale(left,F,Wd-tw-3*pad,th,fs); cv2.putText(frame,left,(pad,int(barh*0.68)),F,ls,(0,255,255),th)
        # ---- panel daftar komponen ----
        det=[(names[i],per[i]) for i in range(nc) if per[i]>0]
        head=f"Objek terdeteksi: {n}"; rows=[head]+[f"- {nm}: {ct}" for nm,ct in det]
        pw=max(cv2.getTextSize(t,F,fs,th)[0][0] for t in rows)+2*pad
        ph=len(rows)*lh+pad; py=barh+pad
        ov=frame.copy(); cv2.rectangle(ov,(pad//2,py),(pad//2+pw,py+ph),(15,23,42),-1); frame=cv2.addWeighted(ov,0.55,frame,0.45,0)
        for i,t in enumerate(rows):
            color=(0,255,0) if i==0 else (226,232,240)
            cv2.putText(frame,t,(pad,py+int(lh*(i+0.8))),F,fs*(0.95 if i else 1.0),color,th)
        fw.writerow([fcount,datetime.datetime.now().isoformat(timespec="milliseconds"),round(infer_fps,2),round(loop_fps,2),round(infer_ms,2),n]+per)
        if do_save:
            if st in ("image","folder"):
                if save_dir is None: save_dir=f"logs/pred_{a.tag}"; os.makedirs(save_dir,exist_ok=True)
                cv2.imwrite(os.path.join(save_dir,os.path.basename(imgs[fcount])),frame)
            else:
                if vwriter is None:
                    vw_size=(Wd,H); vwriter=cv2.VideoWriter(f"logs/rec_{a.tag}.mp4",cv2.VideoWriter_fourcc(*"mp4v"),save_fps,vw_size)
                vwriter.write(frame if (frame.shape[1],frame.shape[0])==vw_size else cv2.resize(frame,vw_size))
        if not a.no_show:
            if fcount==0:
                cv2.namedWindow(WIN,cv2.WINDOW_NORMAL); scd=min(1.0,float(a.win)/max(Wd,H)); cv2.resizeWindow(WIN,int(Wd*scd),int(H*scd))
            cv2.imshow(WIN,frame)
            if (cv2.waitKey(0 if st in ("image","folder") else 1)&0xFF) in (ord("q"),27): break
        fcount+=1
    fcsv.close()
    if cap is not None: cap.release()
    if picam is not None: picam.stop()
    if vwriter is not None: vwriter.release()
    cv2.destroyAllWindows()
    summ={"tag":a.tag,"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"model":mname,"model_path":a.model,
          "device":a.device,"source":st,"imgsz":a.imgsz,"frames":fcount,
          "avg_infer_fps":round(float(np.mean(ifps_hist)) if ifps_hist else 0,2),
          "avg_loop_fps":round(float(np.mean(lfps_hist)) if lfps_hist else 0,2),
          "avg_obj":round(float(np.mean(obj_hist)) if obj_hist else 0,2)}
    sp="logs/runs_summary.csv"; new=not os.path.exists(sp)
    with open(sp,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summ))
        if new: w.writeheader()
        w.writerow(summ)
    print(f"\n[{a.tag}] frames={fcount} infer={summ['avg_infer_fps']}fps loop={summ['avg_loop_fps']}fps | logs/fps_{a.tag}.csv + rec_{a.tag}.mp4")
if __name__=="__main__": main()

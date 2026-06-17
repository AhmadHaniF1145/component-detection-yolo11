"""Export best.pt -> NCNN (ringan & cepat di Raspberry Pi).  python tools/pi/export_ncnn.py --model best.pt"""
import argparse
from ultralytics import YOLO
ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--imgsz",type=int,default=640)
a=ap.parse_args(); YOLO(a.model).export(format="ncnn",imgsz=a.imgsz)
print("Selesai -> folder *_ncnn_model. Pakai di 06_detect_log.py: --model best_ncnn_model")

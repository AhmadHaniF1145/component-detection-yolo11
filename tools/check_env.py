"""Cek environment laptop: paket sudah terinstall belum + versi + GPU. JALANKAN dulu biar gak install ulang."""
import importlib, sys
print("Python:", sys.version.split()[0])
def chk(mod,pip=None):
    try:
        m=importlib.import_module(mod); v=getattr(m,"__version__","?")
        print(f"  [OK] {mod:14s} {v}")
    except Exception:
        print(f"  [-- ] {mod:14s} BELUM ADA  -> pip install {pip or mod}")
for m,pip in [("ultralytics",None),("cv2","opencv-python"),("torch",None),("torchvision",None),("matplotlib",None),("numpy",None)]:
    chk(m,pip)
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0), "| CUDA", torch.version.cuda)
    else: print("  -> GPU TIDAK terdeteksi. install: pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124")
except Exception as e: print("torch belum ada:", e)

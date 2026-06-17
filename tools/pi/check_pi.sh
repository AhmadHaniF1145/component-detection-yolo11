#!/usr/bin/env bash
# Cek Raspberry Pi: paket sudah ada belum + ruang & memori. Biar gak install ulang (hemat memori).
echo "== Sistem =="; uname -m; cat /etc/os-release 2>/dev/null | grep PRETTY_NAME
echo; echo "== Disk & RAM =="; df -h / | tail -1; free -h | grep Mem
echo; echo "== Python paket =="
for m in ultralytics cv2 numpy picamera2 ncnn; do
  python3 - <<PY 2>/dev/null && continue
import importlib,sys
mod="$m"
try:
    x=importlib.import_module(mod); print(f"  [OK] {mod} {getattr(x,'__version__','?')}")
except Exception:
    sys.exit(1)
PY
  echo "  [-- ] $m BELUM ADA"
done
echo; echo "Kalau ada yang [--], install hanya yang itu. Jangan reinstall yang [OK]."
echo "ultralytics: pip install ultralytics --no-cache-dir ; picamera2: sudo apt install -y python3-picamera2 ; ncnn ikut ultralytics export."

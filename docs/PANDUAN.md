# PANDUAN Lengkap — Deteksi Komponen YOLO11 (Laptop + Raspberry Pi)

Proyek deteksi objek **8 kelas** komponen (`bldc_motor`, `stm32f4`, `mpu6500`, `ir_sensor`, `elco_capacitor`, `resistor`, `pin_header`, `xt30_female`) dengan **YOLO11**, dibandingkan antara **laptop (RTX 3060)** dan **Raspberry Pi**. Semua hasil (waktu training, FPS, metrik) **tercatat ke CSV** dan dibuat grafik untuk PPT. Repo ini dirancang agar bisa di-clone & dijalankan ulang oleh siapa pun.

> Urutan kerja: **Cek install → Ambil data → Ekstrak frame → Auto-label → Validasi (web) → Split → Train 2 model (log) → Test + OSD + log FPS → Grafik → Kirim ke Pi → Upload GitHub → PPT.**

---

## 0. Struktur repo

```
04-Vision Detection/
├─ README.md                  ringkas + link cara pakai
├─ requirements-laptop.txt / requirements-pi.txt
├─ data.yaml                  (dibuat oleh tools/04_split.py)
├─ labeling/                  images/  labels/  classes.txt  validated.json   ← sumber dataset
├─ dataset_yolo/              train/  val/   (hasil split, dipakai training)
├─ tools/
│  ├─ 01_extract_frames.py    video → frame
│  ├─ 02_autolabel.py         auto-label pakai model (model-assisted)
│  ├─ label_web/              EDITOR/VALIDATOR LABEL berbasis web (server.py + index.html)
│  ├─ 04_split.py             split train/val + tulis data.yaml
│  ├─ 05_train.py             training + catat waktu/metrik → logs/training_runs.csv
│  ├─ 06_detect_log.py        deteksi + OSD + catat FPS → logs/fps_*.csv
│  ├─ 07_make_graphs.py       CSV → grafik PNG
│  ├─ check_env.py            cek paket laptop
│  └─ pi/  check_pi.sh, export_ncnn.py
├─ logs/                      semua CSV + logs/graphs/*.png
└─ docs/                      PANDUAN.md (file ini) + ilustrasi
```

---

## 1. Instalasi & CEK (biar tidak install ulang)

### Laptop (Windows + Miniconda + RTX 3060)
```bat
conda create -n yolo11-env python=3.12 -y && conda activate yolo11-env
pip install -r requirements-laptop.txt
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python tools\check_env.py      REM cek: semua [OK]? CUDA available True?
```
`check_env.py` menandai paket `[OK]` / `[-- ]`. **Install hanya yang `[-- ]`.**

### Raspberry Pi
```bash
bash tools/pi/check_pi.sh       # cek paket + sisa disk/RAM dulu
# install HANYA yang [--]:
pip install ultralytics --no-cache-dir
sudo apt install -y python3-picamera2
```
`check_pi.sh` juga menampilkan sisa memori/disk supaya Pi tidak penuh.

---

## 2. TAHAP 1 — Pengambilan Data (protokol)

Direkam pakai **HP**, objek diletakkan di permukaan, **kamera dari ATAS (top-down)**.

| Parameter | Nilai | Catatan |
|-----------|-------|---------|
| Background | 4 macam: **lantai, mousepad, meja, kain** | variasi tekstur/warna agar model tidak menghafal latar |
| Jarak kamera ke objek | **30–50 cm** | objek terlihat jelas, tidak terpotong |
| Posisi kamera | **di atas objek**, menghadap ke bawah | objek tergeletak di permukaan |
| Sudut (angle) | **±90° (tegak lurus / top-down)**, digerakkan ±15–30° untuk variasi | variasi sudut menambah ketahanan model |
| Pencahayaan | **terang & redup** (kecerahan video di-adjust gelap–terang) | melatih model di berbagai kondisi cahaya |
| Mode | **Video** dan **Foto** | video: di-stop lalu di-screenshot / diekstrak per-frame; foto: langsung |
| Resolusi | 1080p+ (HP) | nanti di-resize saat ekstraksi |

**Dua cara ambil:**
- **Video** → jalankan `01_extract_frames.py` untuk ambil gambar per-frame (otomatis, banyak), atau stop video lalu screenshot manual.
- **Foto** → langsung dipakai sebagai gambar dataset (taruh di `labeling/images/`).

> Ilustrasi setup (kamera di atas, komponen di permukaan, jarak 30–50 cm, cahaya terang/redup) ada di `docs/` dan dimasukkan ke PPT.

---

## 3. TAHAP 2 — Ekstraksi Frame

```bat
python tools\01_extract_frames.py --video data\video\klip.mp4 --out labeling\images --fps 5 --long 1280 --prefix v5
```
`--fps` = berapa gambar per detik diambil, `--long` = sisi terpanjang (px), `--prefix` = penanda sumber (v1..vN). Foto tinggal di-copy ke `labeling\images`.

---

## 4. TAHAP 3 — Auto-Labeling

Auto-label memberi **kotak awal** supaya validasi cepat (tidak menggambar dari nol). Setelah punya model awal:
```bat
python tools\02_autolabel.py --model runs\laptop_yolo11s\weights\best.pt --images labeling\images --out labeling\labels --conf 0.25
```
Untuk awal (belum ada model), label awal sudah tersedia dari hasil anotasi (CVAT). **Semua label tetap WAJIB divalidasi** di tahap berikut.

---

## 5. TAHAP 4 — Validasi Label via Web (penting)

Editor label sendiri, jalan lokal, **tanpa install** (Python bawaan):
```bat
python tools\label_web\server.py --images labeling\images --labels labeling\labels --classes labeling\classes.txt
```
Buka browser → **http://localhost:8000**. Cara pakai:

| Aksi | Cara |
|------|------|
| Pindah gambar | tombol **A/D** atau klik daftar kiri |
| Gambar kotak baru | **tarik** di area kosong (pakai kelas aktif) |
| Pilih kotak | **klik** di dalam kotak |
| Geser kotak | tarik bagian tengah kotak |
| Resize | tarik **sudut** kotak |
| Ganti kelas kotak | tekan **1–8** atau klik palet warna |
| Hapus kotak | **Del** atau tombol 🗑 |
| Simpan + validasi | tombol **S** (otomatis lanjut gambar berikut) |

Progres `x/200 tervalidasi` terlihat di kiri (tersimpan di `labeling/validated.json`). Edit otomatis tersimpan saat pindah gambar. Validasi **satu per satu** sampai semua benar.

> Alternatif: CVAT / Label Studio / X-AnyLabeling (export YOLO 1.1). Jaga urutan `classes.txt` tetap sama.

---

## 6. TAHAP 5 — Split Train/Val + data.yaml

```bat
python tools\04_split.py --src labeling --out dataset_yolo --train_pct 0.8 --seed 42
```
Membuat `dataset_yolo/train` & `val` (stratified per sumber video) dan menulis `data.yaml` (8 kelas). Ganti `--seed` untuk **menukar komposisi** train/val (cross-validation manual → bandingkan hasil).

---

## 7. TAHAP 6 — Pemilihan YOLO: versi & alasan

**Pakai YOLO11 (Ultralytics).** Alasan: generasi terbaru saat ini, akurasi & kecepatan terbaik di kelasnya, anchor-free, **mudah di-export** (NCNN/ONNX/TensorRT), dokumentasi & komunitas besar, training 1 perintah.

| Target | Model | Alasan |
|--------|-------|--------|
| **Laptop (RTX 3060)** | **YOLO11s** | GPU kuat → pilih akurasi lebih tinggi, tetap real-time |
| **Raspberry Pi (CPU ARM)** | **YOLO11n** | paling ringan → FPS layak di CPU; di-export **NCNN** agar makin cepat |

Eksperimen: **dua model (11n & 11s) dilatih di laptop**, lalu **dijalankan di laptop & Pi** → 4 kombinasi (model × device) untuk dibandingkan.

---

## 8. TAHAP 7 — Training 2 Model (otomatis tercatat)

```bat
python tools\05_train.py --model yolo11n.pt --name laptop_yolo11n --epochs 60 --imgsz 640 --batch 16 --device 0
python tools\05_train.py --model yolo11s.pt --name laptop_yolo11s --epochs 60 --imgsz 640 --batch 16 --device 0
```
Tiap training otomatis menambah 1 baris ke **`logs/training_runs.csv`**:

`timestamp, name, base_model, device, os, epochs, imgsz, batch, train_time_s, train_time_min, sec_per_epoch, mAP50, mAP50_95, precision, recall, weights`

→ ini bukti **berapa lama training sekian epoch** + akurasi tiap model. Bobot terbaik: `runs/<name>/weights/best.pt`.

(Opsional, kalau Pi sanggup, latih juga di Pi untuk bandingan waktu — biasanya tidak disarankan, cukup latih di laptop.)

---

## 9. TAHAP 8 — Test/Deteksi + OSD + Log FPS

OSD overlay menampilkan **FPS** dan **jumlah objek per-kelas**; FPS tiap frame dicatat ke CSV.

**Laptop — 2 model × (video & webcam):**
```bat
python tools\06_detect_log.py --model runs\laptop_yolo11s\weights\best.pt --source test.mp4 --tag laptop_11s_video --device 0
python tools\06_detect_log.py --model runs\laptop_yolo11n\weights\best.pt --source test.mp4 --tag laptop_11n_video --device 0
python tools\06_detect_log.py --model runs\laptop_yolo11s\weights\best.pt --source usb0   --tag laptop_11s_webcam --device 0
python tools\06_detect_log.py --model runs\laptop_yolo11n\weights\best.pt --source usb0   --tag laptop_11n_webcam --device 0
```
Tiap run menghasilkan **`logs/fps_<tag>.csv`** (per-frame: fps, infer_ms, jumlah objek, jumlah per-kelas) + 1 baris ringkasan ke **`logs/runs_summary.csv`** (avg/min/max FPS). `--record` untuk merekam videonya. Tekan `q` untuk berhenti.

> Webcam tidak kebuka? Script sudah otomatis coba beberapa backend (DSHOW→MSMF→default). Kalau tetap, ganti index `--source usb1`.

---

## 10. TAHAP 9 — Kirim ke Raspberry Pi & Run

**Yang DIKIRIM ke Pi (hemat memori — jangan kirim seluruh dataset):**
- `runs/laptop_yolo11n/weights/best.pt` (dan `laptop_yolo11s` kalau mau dibandingkan)
- folder `tools/` (khususnya `06_detect_log.py`, `pi/`)
- `requirements-pi.txt`

**Di Pi:**
```bash
bash tools/pi/check_pi.sh                              # pastikan paket ada
python3 tools/pi/export_ncnn.py --model best.pt        # best.pt → best_ncnn_model (lebih cepat)
# jalankan + log (PiCamera atau webcam USB):
python3 tools/06_detect_log.py --model best_ncnn_model --source picamera0 --tag pi_11n_cam --device cpu --imgsz 640
python3 tools/06_detect_log.py --model best_ncnn_model --source test.mp4   --tag pi_11n_video --device cpu --imgsz 640
```
CSV `logs/fps_pi_*.csv` dikirim balik ke laptop untuk digabung di grafik.

---

## 11. TAHAP 10 — Grafik Perbandingan

Setelah CSV laptop + Pi terkumpul di `logs/`:
```bat
python tools\07_make_graphs.py
```
Menghasilkan `logs/graphs/`: `training_time.png`, `training_map.png`, `fps_compare.png` (laptop vs Pi, 11n vs 11s), `fps_timeline.png`. → langsung dipakai di PPT.

---

## 12. Yang perlu DIULANG / DIHAPUS (rapikan sebelum GitHub)

**Diulang (kalau perlu):** tambah data (cahaya/sudut/background lain) → ekstrak → validasi → split → train ulang.

**Boleh DIHAPUS (sisa eksperimen lama 8-kelas, supaya repo bersih):**
- `dataset/` (versi lama 8-kelas: images_all, labels_all, annotated_preview, train, validation)
- `scripts/` (lama, digantikan `tools/`)
- `runs/detect/train-2` (model 8-kelas lama) — kalau tak dipakai
- `2026-06-08-PANDUAN-Training-YOLO11.md`, `2026-06-10-CARA-PAKAI.md`, PPTX lama — digantikan `docs/PANDUAN.md`
- `video.mp4`, `yolo26n.pt` jika tidak dipakai

> Hapus manual setelah yakin. Yang DIPERTAHANKAN: `labeling/`, `dataset_yolo/`, `tools/`, `logs/`, `docs/`, `data.yaml`, `requirements-*.txt`, `runs/laptop_yolo11n|s`.

---

## 13. Upload GitHub

```bat
git init
git add .
git commit -m "Deteksi komponen YOLO11 — dataset, tools, logs, panduan"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```
`.gitignore` sudah mengabaikan cache & rekaman video besar. Cantumkan **link GitHub** ini di slide PPT.

---

## 14. Rencana Pengembangan (untuk adik kelas)

1. **Rapikan kualitas label** semua 8 kelas (terutama objek tipis: resistor, xt30_female) lewat validasi web — label rapi = mAP naik.
2. **Perbanyak & variasikan data**: lebih banyak background, sudut, jarak, dan kondisi cahaya; tambah objek "pengganggu".
3. **Uji generalisasi** pada data benar-benar baru (bukan dari video yang sama) untuk angka yang jujur.
4. **Optimasi Pi**: bandingkan NCNN vs ONNX vs TensorRT (di Jetson), turunkan `imgsz`, quantization INT8.
5. **Aplikasi nyata**: hitung & log stok komponen otomatis, alarm bila komponen tertentu terdeteksi, integrasi ke web dashboard.
6. **Auto-label cerdas**: pakai model terbaik untuk pre-label data baru (`02_autolabel.py`) → validasi → re-train (active learning).
7. **Robotik**: gabung dengan lengan/konveyor untuk sortir komponen otomatis.

---

## 🍓 Raspberry Pi — Full Test Matrix & Transfer (detailed)

On the Pi we test **2 models** (`.pt` nano and the **NCNN** export of the same model) on **video (`test3.mp4`)** and a **live camera/stream**, exactly like the kick-detection setup. The live detection window is shown on the Pi too (over its desktop / VNC / Raspberry Pi Connect) — same view as the laptop, only the **FPS and results differ**.

### A. Send Laptop → Pi  (only these — keep it light)
```bat
REM from the repo folder on the laptop:
scp -r models\best_yolo11n.pt models\best_yolo11n_ncnn_model tools test\test3.mp4 requirements-pi.txt ahmadhanif@192.168.110.51:~/component-detection/
```

### B. On the Raspberry Pi  (ALWAYS --device cpu)
```bash
cd ~/component-detection
bash tools/pi/check_pi.sh            # install only what is missing (pip install ultralytics ; sudo apt install -y python3-picamera2)

# --- Model 1: plain .pt nano ---
python3 tools/06_detect_log.py --model best_yolo11n.pt          --source test3.mp4  --tag pi_n_video    --device cpu --imgsz 640 --save
python3 tools/06_detect_log.py --model best_yolo11n.pt          --source picamera0  --tag pi_n_cam      --device cpu --imgsz 640 --save

# --- Model 2: NCNN (faster on ARM) ---
python3 tools/06_detect_log.py --model best_yolo11n_ncnn_model  --source test3.mp4  --tag pi_ncnn_video --device cpu --imgsz 640 --save
python3 tools/06_detect_log.py --model best_yolo11n_ncnn_model  --source picamera0  --tag pi_ncnn_cam   --device cpu --imgsz 640 --save

# --- Live stream from the laptop (IP-cam), like kick-detection ---
# laptop: run any IP-webcam (phone "IP Webcam" app, or OBS virtual cam over MJPEG), get its URL, then on Pi:
python3 tools/06_detect_log.py --model best_yolo11n_ncnn_model  --source http://<LAPTOP_OR_PHONE_IP>:8080/video --tag pi_ncnn_stream --device cpu --imgsz 640 --save
python3 tools/06_detect_log.py --model best_yolo11n_ncnn_model  --source http://10.124.28.185:8080/video --tag pi_ncnn_stream --device cpu --imgsz 320 --save
python3 tools/06_detect_log.py --model best_yolo11n.pt  --source http://10.124.28.185:8080/video --tag pi_n_stream --device cpu --imgsz 320 --save
# USB webcam on the Pi instead: --source usb0
```

### C. Results produced on the Pi  (names & location)
Each run writes to `logs/` on the Pi:
- **`logs/fps_<tag>.csv`** — per-frame Infer/Loop FPS + per-class counts
- **`logs/rec_<tag>.mp4`** — recorded video with the OSD overlay

Tags above → `pi_n_video`, `pi_n_cam`, `pi_ncnn_video`, `pi_ncnn_cam`, `pi_ncnn_stream`.

### D. Send Pi → Laptop  (bring results back)
```bash
# run on the Pi (push to laptop), OR use scp from the laptop to pull:
scp ~/component-detection/logs/fps_pi_*.csv   ahmadhanif@<LAPTOP_IP>:.../component-detection-yolo11/logs_pi/
scp ~/component-detection/logs/rec_pi_*.mp4   ahmadhanif@<LAPTOP_IP>:.../component-detection-yolo11/hasil_pi/
```
Put the CSVs in **`logs_pi/`** and the videos in **`hasil_pi/`**. Then on the laptop:
```bat
copy logs_pi\fps_pi_*.csv logs\        REM gabungkan agar ikut ke grafik
python tools\07_make_graphs.py         REM -> assets/fps_compare.png + fps_timeline.png (laptop vs Pi, n vs s vs NCNN)
```
The comparison graphs (laptop vs Pi, `.pt` vs NCNN) are then ready for the slides.

---

## ▶ Ready-to-paste — my setup (Pi `192.168.110.51`, user `ahmadhanif`)

**① LAPTOP — send to Pi** (Command Prompt / PowerShell):
```bat
cd "C:\Users\Ahmad Hanif A.K\Documents\Raspberry-Pi-Computer-Vision\04-Vision Detection\component-detection-yolo11"
ssh ahmadhanif@192.168.110.51 "mkdir -p ~/component-detection"
scp -r models\best_yolo11n.pt models\best_yolo11n_ncnn_model tools test\test3.mp4 requirements-pi.txt ahmadhanif@192.168.110.51:~/component-detection/
```

**② RASPBERRY PI — run** (2 models × video + camera; always `--device cpu`):
```bash
cd ~/component-detection
bash tools/pi/check_pi.sh
pip install ultralytics --no-cache-dir
sudo apt install -y python3-picamera2

python3 tools/06_detect_log.py --model best_yolo11n.pt         --source test3.mp4 --tag pi_n_video    --device cpu --imgsz 640 --save
python3 tools/06_detect_log.py --model best_yolo11n_ncnn_model --source test3.mp4 --tag pi_ncnn_video --device cpu --imgsz 640 --save
python3 tools/06_detect_log.py --model best_yolo11n.pt         --source picamera0 --tag pi_n_cam      --device cpu --imgsz 640 --save
python3 tools/06_detect_log.py --model best_yolo11n_ncnn_model --source picamera0 --tag pi_ncnn_cam   --device cpu --imgsz 640 --save
# USB webcam: --source usb0   |   laptop IP-cam stream: --source http://<LAPTOP_IP>:8080/video
```

**③ LAPTOP — bring results back** (pull from Pi, then plot):
```bat
cd "C:\Users\Ahmad Hanif A.K\Documents\Raspberry-Pi-Computer-Vision\04-Vision Detection\component-detection-yolo11"
scp ahmadhanif@192.168.110.51:~/component-detection/logs/fps_pi_*.csv logs_pi\
scp ahmadhanif@192.168.110.51:~/component-detection/logs/rec_pi_*.mp4 hasil_pi\

scp ahmadhanif@10.124.28.15:~/component-detection/logs/fps_pi_*.csv logs_pi\
scp ahmadhanif@10.124.28.15:~/component-detection/logs/rec_pi_*.mp4 hasil_pi\

copy logs_pi\fps_pi_*.csv logs\
python tools\07_make_graphs.py
```

---

## ♻️ Reuse existing `yolo-env` on the Pi (skip re-install)
The Pi already has a virtualenv from the kick-detection project that contains `ultralytics` + `ncnn`. Reuse it instead of installing again:
```bash
source ~/04-Vision-Detection/yolo-env/bin/activate   # or ~/04-Vision-Detection1/yolo-env/bin/activate
cd ~/component-detection
bash tools/pi/check_pi.sh        # now checks the ENV -> ultralytics & ncnn should be [OK]
# then run tools/06_detect_log.py directly (no pip/apt needed)
```
`check_pi.sh` showed `[--]` earlier only because it was run in the system Python, not inside the env. Re-source the env in every new terminal.

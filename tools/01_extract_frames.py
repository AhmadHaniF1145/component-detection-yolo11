"""
Ekstrak frame dari video menjadi gambar (untuk menambah data baru).
Pakai OpenCV (tanpa ffmpeg). Auto-rotate mengikuti metadata diabaikan -> jika hasil miring,
rekam video landscape atau tambahkan --rotate.

Contoh:
    python 01_extract_frames.py --video ../data/video/clip.mp4 --out ../dataset/images_all --fps 5 --long 1280 --prefix v5
"""
import argparse, os, cv2
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--fps",type=float,default=5.0,help="frame per detik yang diambil")
    ap.add_argument("--long",type=int,default=1280,help="sisi terpanjang hasil (px)")
    ap.add_argument("--prefix",default="vid"); ap.add_argument("--rotate",choices=["0","90","180","270"],default="0")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    cap=cv2.VideoCapture(a.video)
    src_fps=cap.get(cv2.CAP_PROP_FPS) or 30
    step=max(1,round(src_fps/a.fps)); rot={"90":cv2.ROTATE_90_CLOCKWISE,"180":cv2.ROTATE_180,"270":cv2.ROTATE_90_COUNTERCLOCKWISE}
    i=0; saved=0
    while True:
        ok,fr=cap.read()
        if not ok: break
        if i%step==0:
            if a.rotate!="0": fr=cv2.rotate(fr,rot[a.rotate])
            h,w=fr.shape[:2]; s=a.long/max(h,w)
            if s<1: fr=cv2.resize(fr,(int(w*s),int(h*s)))
            saved+=1; cv2.imwrite(os.path.join(a.out,f"{a.prefix}_{saved:04d}.jpg"),fr,[cv2.IMWRITE_JPEG_QUALITY,90])
        i+=1
    cap.release(); print(f"Tersimpan {saved} frame ke {a.out}")
if __name__=="__main__": main()

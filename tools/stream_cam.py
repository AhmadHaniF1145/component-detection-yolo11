"""
Siarkan webcam LAPTOP sebagai MJPEG stream (jalankan DI LAPTOP). Pi membacanya realtime via URL.
  (laptop)  python tools/stream_cam.py --cam 0 --res 1280x720 --port 8080
  cari IP laptop: Windows -> ipconfig (IPv4 Address)
  (raspi)   python3 tools/06_detect_log.py --model best_yolo11n_ncnn_model --source http://<LAPTOP_IP>:8080/video --tag pi_ncnn_stream --device cpu --imgsz 640 --save
"""
import argparse, time, cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
def open_cam(idx,w,h):
    for be in (cv2.CAP_DSHOW,cv2.CAP_MSMF,cv2.CAP_ANY):
        c=cv2.VideoCapture(idx,be)
        if w: c.set(3,w); c.set(4,h)
        if c.isOpened():
            ok,_=c.read()
            if ok: return c
        c.release()
    return None
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cam",type=int,default=0); ap.add_argument("--port",type=int,default=8080)
    ap.add_argument("--res",default="1280x720"); ap.add_argument("--q",type=int,default=80)
    a=ap.parse_args(); w,h=map(int,a.res.lower().split("x"))
    cap=open_cam(a.cam,w,h)
    if cap is None: raise SystemExit("Webcam gagal dibuka (coba --cam 1).")
    class H(BaseHTTPRequestHandler):
        def log_message(self,*x): pass
        def do_GET(self):
            if self.path!="/video":
                self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
                self.wfile.write(b"<h3>MJPEG: <a href='/video'>/video</a></h3>"); return
            self.send_response(200)
            self.send_header("Content-Type","multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            while True:
                ok,frame=cap.read()
                if not ok: break
                ok,jpg=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,a.q])
                if not ok: continue
                try:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "+str(len(jpg)).encode()+b"\r\n\r\n"+jpg.tobytes()+b"\r\n")
                except (BrokenPipeError,ConnectionResetError): break
                time.sleep(0.005)
    srv=ThreadingHTTPServer(("0.0.0.0",a.port),H)
    print(f"Streaming webcam {a.cam} @ {a.res}")
    print(f"URL: http://<LAPTOP_IP>:{a.port}/video   (cari IP laptop: ipconfig -> IPv4)")
    print("Ctrl+C untuk berhenti.")
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
if __name__=="__main__": main()

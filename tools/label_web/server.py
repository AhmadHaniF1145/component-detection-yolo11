"""
Editor & Validator Label YOLO berbasis web (stdlib Python, TANPA install apa pun).
Lihat tiap gambar, gambar/geser/resize kotak, pilih kelas, validasi satu per satu, simpan ke .txt.

Jalankan:
  python tools/label_web/server.py --images labeling/images --labels labeling/labels --classes labeling/classes.txt
Lalu buka browser ke http://localhost:8000

Pintasan: A/D=prev/next, S=simpan+validasi, 1..9=pilih kelas, Del=hapus, G=salin ke grup, C=salin sebelumnya.
"""
import argparse, os, json, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    IMAGES = os.path.abspath(a.images)
    LABELS = os.path.abspath(a.labels)
    os.makedirs(LABELS, exist_ok=True)
    STATE = os.path.join(os.path.dirname(LABELS), "validated.json")
    CLASSES = [l.strip() for l in open(a.classes, encoding="utf-8") if l.strip()]
    HERE = os.path.dirname(os.path.abspath(__file__))

    def imglist():
        return sorted(f for f in os.listdir(IMAGES) if f.lower().endswith(EXTS))

    def load_state():
        try:
            return set(json.load(open(STATE)).get("validated", []))
        except Exception:
            return set()

    def save_state(s):
        json.dump({"validated": sorted(s)}, open(STATE, "w"))

    def clean_lines(raw):
        good = []
        for ln in raw.replace("\r", "").split("\n"):
            p = ln.split()
            if len(p) != 5:
                continue
            try:
                int(float(p[0])); [float(x) for x in p[1:]]
            except ValueError:
                continue
            good.append(" ".join(p))
        return ("\n".join(good) + "\n") if good else ""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, ctype, body):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self):
            n = int(self.headers.get("Content-Length", "0"))
            buf = b""
            while len(buf) < n:
                chunk = self.rfile.read(n - len(buf))
                if not chunk:
                    break
                buf += chunk
            return buf.decode("utf-8")

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            p = u.path
            if p in ("/", "/index.html"):
                return self._send(200, "text/html; charset=utf-8",
                                  open(os.path.join(HERE, "index.html"), "rb").read())
            if p == "/api/config":
                return self._send(200, "application/json", json.dumps({
                    "classes": CLASSES, "images": imglist(),
                    "validated": sorted(load_state())}))
            if p == "/api/label":
                name = q.get("name", [""])[0]
                lp = os.path.join(LABELS, os.path.splitext(os.path.basename(name))[0] + ".txt")
                return self._send(200, "text/plain",
                                  open(lp, encoding="utf-8").read() if os.path.exists(lp) else "")
            if p.startswith("/img/"):
                fp = os.path.join(IMAGES, os.path.basename(urllib.parse.unquote(p[5:])))
                if os.path.exists(fp):
                    ct = "image/png" if fp.lower().endswith(".png") else "image/jpeg"
                    return self._send(200, ct, open(fp, "rb").read())
                return self._send(404, "text/plain", "not found")
            return self._send(404, "text/plain", "not found")

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            body = self._read_body()
            if u.path == "/api/save":
                name = q.get("name", [""])[0]
                base = os.path.splitext(os.path.basename(name))[0]
                try:
                    raw = "\n".join(json.loads(body).get("lines", []))
                except Exception:
                    raw = body
                open(os.path.join(LABELS, base + ".txt"), "w", encoding="utf-8").write(clean_lines(raw))
                st = load_state()
                img = base + (os.path.splitext(name)[1] or ".jpg")
                if q.get("validated", ["0"])[0] == "1":
                    st.add(img)
                else:
                    st.discard(img)
                save_state(st)
                return self._send(200, "application/json", json.dumps({"ok": True, "validated": len(st)}))
            if u.path == "/api/propagate":
                frm = q.get("from", [""])[0]
                base = os.path.splitext(os.path.basename(frm))[0]
                prefix = base.split("_")[0]
                sp = os.path.join(LABELS, base + ".txt")
                src_txt = open(sp, encoding="utf-8").read() if os.path.exists(sp) else ""
                st = load_state()
                cnt = 0
                for f in imglist():
                    b2 = os.path.splitext(f)[0]
                    if b2.split("_")[0] == prefix and b2 != base:
                        open(os.path.join(LABELS, b2 + ".txt"), "w", encoding="utf-8").write(src_txt)
                        st.discard(f)
                        cnt += 1
                save_state(st)
                return self._send(200, "application/json", json.dumps({"ok": True, "count": cnt, "prefix": prefix}))
            return self._send(404, "text/plain", "not found")

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print("== Label Validator ==")
    print("Images :", IMAGES)
    print("Labels :", LABELS)
    print("Classes:", CLASSES)
    print("\nBuka browser:  http://localhost:%d\n(Ctrl+C untuk berhenti)" % a.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nberhenti.")

if __name__ == "__main__":
    main()

"""CPB - Compress PDF to a target size (100 / 200 / 300 / 500 KB or custom).

How it works
  1. "Keep text" pass (pypdf + Pillow): re-compress and shrink the images inside the PDF,
     keep the text as real text. Tried first.
  2. "Image" fallback (poppler + Pillow): if step 1 cannot reach the size and the visitor
     allows it, every page is saved as a JPEG image and a new PDF is built from them.

Files are only kept in memory / a temporary folder while a request runs, then deleted.
"""
import io
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import defaultdict, deque

from flask import Flask, Response, abort, jsonify, request, send_from_directory
from PIL import Image
from pypdf import PdfReader, PdfWriter
from werkzeug.middleware.proxy_fix import ProxyFix

log = logging.getLogger("cpb")
logging.basicConfig(level=logging.INFO)

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", "30"))
MAX_PAGES = int(os.environ.get("MAX_PAGES", "80"))
MAX_JOBS = int(os.environ.get("MAX_JOBS", "2"))
JOB_SECONDS = int(os.environ.get("JOB_SECONDS", "110"))
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "30"))  # requests per IP per 10 minutes
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "")  # only needed if the web page is hosted elsewhere

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

JOB_SLOTS = threading.BoundedSemaphore(MAX_JOBS)
HITS = defaultdict(deque)


class UserError(Exception):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.message = message
        self.status = status


# ----------------------------------------------------------------------------
# Step 1: keep the text, shrink the images
# ----------------------------------------------------------------------------
# (longest image side in px, JPEG quality), best quality first
KEEP_LADDER = [(2200, 82), (1700, 72), (1350, 62), (1050, 52), (800, 42), (620, 34), (480, 26), (360, 20), (260, 16)]


def jpeg_bytes(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def open_reader(src):
    try:
        reader = PdfReader(io.BytesIO(src))
        if reader.is_encrypted and not reader.decrypt(""):
            raise UserError("This PDF is password protected. Remove the password first, then try again.")
        return reader
    except UserError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.info("unreadable pdf: %s", exc)
        raise UserError("Could not read this PDF. It may be damaged or in an unsupported format.")


def shrink_image(img, ref, maxdim, quality, gray):
    obj = ref.get_object()
    if "/SMask" in obj or "/Mask" in obj or obj.get("/ImageMask"):
        return
    orig_len = len(getattr(obj, "_data", b"") or b"")
    pil = img.image
    if pil.mode in ("1", "RGBA", "LA", "I", "I;16", "F"):
        return
    if pil.mode in ("P", "CMYK"):
        pil = pil.convert("RGB")
    if gray and pil.mode != "L":
        pil = pil.convert("L")
    if max(pil.size) > maxdim:
        pil = pil.copy()
        pil.thumbnail((maxdim, maxdim), Image.LANCZOS)
    trial = jpeg_bytes(pil, quality)
    if orig_len and len(trial) >= orig_len * 0.97:
        return  # no real gain, keep the original image
    img.replace(pil, quality=quality)


def keep_text_pass(src, level, gray, quality=None):
    maxdim, ladder_quality = KEEP_LADDER[level]
    quality = quality or ladder_quality
    reader = open_reader(src)
    writer = PdfWriter(clone_from=reader)
    done = set()
    for page in writer.pages:
        try:
            images = list(page.images)
        except Exception:  # noqa: BLE001
            images = []
        for img in images:
            ref = img.indirect_reference
            if ref is None or ref.idnum in done:
                continue
            done.add(ref.idnum)
            try:
                shrink_image(img, ref, maxdim, quality, gray)
            except Exception as exc:  # noqa: BLE001
                log.info("skipped one image: %s", exc)
        try:
            page.compress_content_streams()
        except Exception:  # noqa: BLE001
            pass
    try:
        writer.compress_identical_objects(remove_identicals=True, remove_orphans=True)
    except Exception:  # noqa: BLE001
        pass
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise UserError("This file took too long to process. Try a smaller file or a larger target size.", 504)


def compress_keep_text(src, target, gray, deadline):
    lo, hi = 0, len(KEEP_LADDER) - 1
    fit, fit_level, smallest = None, -1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        check_deadline(deadline)
        out = keep_text_pass(src, mid, gray)
        if smallest is None or len(out) < len(smallest):
            smallest = out
        if len(out) <= target:
            fit, fit_level = out, mid
            hi = mid - 1
        else:
            lo = mid + 1
    if fit is not None:
        # use the room left under the target: raise the JPEG quality while it still fits
        base_q = KEEP_LADDER[fit_level][1]
        for bump in (6, 12, 18):
            check_deadline(deadline)
            out = keep_text_pass(src, fit_level, gray, quality=min(92, base_q + bump))
            if len(out) <= target:
                fit = out
            else:
                break
    return fit, smallest


# ----------------------------------------------------------------------------
# Step 2: every page becomes a JPEG image
# ----------------------------------------------------------------------------
RENDER_MAX = 1800  # longest side in px for the first render
# (scale relative to the first render, JPEG quality), best quality first
RASTER_LADDER = [(1.0, 80), (0.85, 74), (0.7, 66), (0.58, 58), (0.48, 50), (0.4, 44),
                 (0.32, 38), (0.26, 32), (0.2, 28), (0.15, 24), (0.1, 22)]


def fit_page(png_path, budget, gray):
    im = Image.open(png_path)
    im.load()
    im = im.convert("L" if gray else "RGB")
    cache = {}

    def at(idx, q):
        scale = RASTER_LADDER[idx][0]
        if idx not in cache:
            if scale >= 0.999:
                cache[idx] = im
            else:
                cache[idx] = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
        return jpeg_bytes(cache[idx], q), cache[idx].size

    lo, hi = 0, len(RASTER_LADDER) - 1
    found, found_bytes, found_size = -1, None, None
    while lo <= hi:
        mid = (lo + hi) // 2
        data, size = at(mid, RASTER_LADDER[mid][1])
        if len(data) <= budget:
            found, found_bytes, found_size = mid, data, size
            hi = mid - 1
        else:
            lo = mid + 1
    if found < 0:
        last = len(RASTER_LADDER) - 1
        data, size = at(last, RASTER_LADDER[last][1])
        return data, size, True
    qlo = RASTER_LADDER[found][1]
    qhi = min(95, qlo + 20)
    best, best_size = found_bytes, found_size
    for _ in range(3):
        qm = (qlo + qhi) // 2
        data, size = at(found, qm)
        if len(data) <= budget:
            best, best_size, qlo = data, size, qm
        else:
            qhi = qm
    return best, best_size, False


def build_pdf(pages):
    n = len(pages)
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join("%d 0 R" % (3 + 3 * i) for i in range(n))
    objs.append(("<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, n)).encode())
    for i, p in enumerate(pages):
        pid = 3 + 3 * i
        w, h = p["wpt"], p["hpt"]
        objs.append(("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] /Resources << /XObject << /Im0 %d 0 R >> >> /Contents %d 0 R >>"
                     % (w, h, pid + 2, pid + 1)).encode())
        content = ("q %.2f 0 0 %.2f 0 0 cm /Im0 Do Q" % (w, h)).encode()
        objs.append(("<< /Length %d >>\nstream\n" % len(content)).encode() + content + b"\nendstream")
        cs = "/DeviceGray" if p["gray"] else "/DeviceRGB"
        head = ("<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace %s /BitsPerComponent 8 "
                "/Filter /DCTDecode /Length %d >>\nstream\n" % (p["wpx"], p["hpx"], cs, len(p["jpeg"]))).encode()
        objs.append(head + p["jpeg"] + b"\nendstream")
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for k, body in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(("%d 0 obj\n" % k).encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(("xref\n0 %d\n" % (len(objs) + 1)).encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(("%010d 00000 n \n" % off).encode())
    out.write(("trailer\n<< /Size %d /Root 1 0 R /Producer (CPB) >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)).encode())
    return out.getvalue()


def page_sizes(reader):
    sizes = []
    for page in reader.pages:
        try:
            box = page.cropbox
            w, h = float(box.width), float(box.height)
            unit = float(page.get("/UserUnit", 1) or 1)
            w, h = w * unit, h * unit
            if (page.rotation or 0) % 360 in (90, 270):
                w, h = h, w
        except Exception:  # noqa: BLE001
            w, h = 595.0, 842.0
        sizes.append((w, h))
    return sizes


def compress_raster(src, target, gray, deadline, workdir):
    reader = open_reader(src)
    n = len(reader.pages)
    if n > MAX_PAGES:
        raise UserError("This PDF has %d pages. The limit is %d pages." % (n, MAX_PAGES), 413)
    sizes = page_sizes(reader)
    in_path = os.path.join(workdir, "in.pdf")
    with open(in_path, "wb") as fh:
        fh.write(src)
    try:
        subprocess.run(["pdftoppm", "-scale-to", str(RENDER_MAX), "-png", in_path, os.path.join(workdir, "pg")],
                       check=True, timeout=max(10, int(deadline - time.monotonic())),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise UserError("This file took too long to process. Try a smaller file or a larger target size.", 504)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise UserError("Could not read this PDF. It may be damaged or in an unsupported format.")
    pngs = sorted(f for f in os.listdir(workdir) if f.startswith("pg") and f.endswith(".png"))
    if len(pngs) != n:
        raise UserError("Could not read all pages of this PDF.")

    shrink, best = 1.0, None
    for _attempt in range(3):
        left = max(500.0, target * 0.96 * shrink - 1500 - n * 500)
        parts, too_big = [], False
        for i, name in enumerate(pngs):
            check_deadline(deadline)
            budget = max(250.0, left / (n - i))
            data, (wpx, hpx), tb = fit_page(os.path.join(workdir, name), budget, gray)
            left -= len(data)
            too_big = too_big or tb
            parts.append({"jpeg": data, "wpx": wpx, "hpx": hpx, "gray": gray, "wpt": sizes[i][0], "hpt": sizes[i][1]})
        out = build_pdf(parts)
        if best is None or len(out) < len(best):
            best = out
        if len(out) <= target:
            return out, True
        if too_big:
            break
        shrink *= (target / len(out)) * 0.95
    return best, False


def compress_pdf(src, target, gray, allow_images):
    deadline = time.monotonic() + JOB_SECONDS
    fit, smallest = compress_keep_text(src, target, gray, deadline)
    if fit is not None:
        return fit, "text", True
    if allow_images and shutil.which("pdftoppm"):  # the image fallback needs poppler; skip it when missing
        workdir = tempfile.mkdtemp(prefix="cpb_")
        try:
            data, met = compress_raster(src, target, gray, deadline, workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        if met or len(data) < len(smallest):
            return data, "image", met
    return smallest, "text", False


# ----------------------------------------------------------------------------
# Web
# ----------------------------------------------------------------------------
PAGES = {"index.html", "privacy.html", "terms.html", "about.html", "contact.html", "ads.txt", "robots.txt", "sitemap.xml"}


def rate_ok(ip):
    now = time.time()
    q = HITS[ip]
    while q and now - q[0] > 600:
        q.popleft()
    if len(HITS) > 5000:
        HITS.clear()
    if len(q) >= RATE_LIMIT:
        return False
    q.append(now)
    return True


@app.after_request
def add_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    if ALLOWED_ORIGIN:
        resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    resp.headers["Access-Control-Expose-Headers"] = "X-Original-Size, X-Output-Size, X-Mode, X-Target-Met"
    return resp


def fail(message, status):
    return jsonify({"error": message}), status


@app.errorhandler(413)
def too_large(_e):
    return fail("This file is too large. The limit is %d MB." % MAX_MB, 413)


@app.get("/health")
def health():
    return "ok"


@app.get("/")
def home():
    return send_from_directory(HERE, "index.html")


@app.get("/<name>")
def page(name):
    if name in PAGES and os.path.exists(os.path.join(HERE, name)):
        return send_from_directory(HERE, name)
    abort(404)


@app.post("/api/compress")
def api_compress():
    if not rate_ok(request.remote_addr or "?"):
        return fail("Too many requests. Please wait a few minutes and try again.", 429)
    upload = request.files.get("file")
    if upload is None:
        return fail("No file received.", 400)
    try:
        kb = float(request.form.get("target_kb", "100"))
    except ValueError:
        kb = 100.0
    kb = min(max(kb, 20.0), 20000.0)
    target = int(kb * 1024)
    gray = request.form.get("gray") == "1"
    allow_images = request.form.get("allow_images", "1") != "0"

    src = upload.read()
    if b"%PDF" not in src[:1024]:
        return fail("This does not look like a PDF file.", 400)

    def reply(data, mode, met):
        resp = Response(data, mimetype="application/pdf")
        resp.headers["X-Original-Size"] = str(len(src))
        resp.headers["X-Output-Size"] = str(len(data))
        resp.headers["X-Mode"] = mode
        resp.headers["X-Target-Met"] = "1" if met else "0"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    if len(src) <= target:
        return reply(src, "original", True)

    if not JOB_SLOTS.acquire(timeout=45):
        return fail("The server is busy right now. Please try again in a minute.", 503)
    try:
        data, mode, met = compress_pdf(src, target, gray, allow_images)
        return reply(data, mode, met)
    except UserError as err:
        return fail(err.message, err.status)
    except Exception:  # noqa: BLE001
        log.exception("compress failed")
        return fail("Something went wrong while compressing this PDF. It may be damaged or unusual.", 500)
    finally:
        JOB_SLOTS.release()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
  

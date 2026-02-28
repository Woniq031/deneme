import io
import zipfile
from pathlib import Path
import tempfile
import subprocess

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import imageio_ffmpeg
import tifffile as tiff


# =========================
# Helpers
# =========================
def make_zip(files: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            z.writestr(name, data)
    return buf.getvalue()


def convert_avi_to_mp4(avi_path: str, mp4_path: str) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i", avi_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        mp4_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def read_image_any(uploaded_file) -> np.ndarray:
    """PNG/JPG/TIF okur. TIF çok-kanallı olabilir."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getbuffer()
    if name.endswith((".tif", ".tiff")):
        arr = tiff.imread(io.BytesIO(data))
        return arr
    file_bytes = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    return img


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# =========================
# IMAGE MODE: segment from RED, measure GREEN
# =========================
def segment_cells_from_red(red_gray: np.ndarray, min_area: int = 50, peak_thresh: float = 0.35) -> np.ndarray:
    r = to_uint8(red_gray)
    blur = cv2.GaussianBlur(r, (0, 0), 1.2)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    _, peaks = cv2.threshold(dist_norm, float(peak_thresh), 1.0, cv2.THRESH_BINARY)
    peaks = (peaks * 255).astype(np.uint8)

    num, markers = cv2.connectedComponents(peaks)
    markers = markers + 1
    markers[bw == 0] = 0

    color = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    ws = cv2.watershed(color, markers)

    labels = ws.copy()
    labels[labels < 2] = 0
    labels = labels - 1  # hücreler 1..N

    out = np.zeros_like(labels, dtype=np.int32)
    new_id = 1
    for cid in np.unique(labels):
        if cid == 0:
            continue
        mask = (labels == cid)
        area = int(mask.sum())
        if area < int(min_area):
            continue
        out[mask] = new_id
        new_id += 1

    return out


def measure_green_intensity(cell_labels: np.ndarray, green_gray: np.ndarray) -> pd.DataFrame:
    g = green_gray
    rows = []
    max_id = int(cell_labels.max())
    for cid in range(1, max_id + 1):
        mask = (cell_labels == cid)
        if not mask.any():
            continue
        vals = g[mask].astype(np.float32)
        rows.append({
            "cell_id": cid,
            "area_px": int(mask.sum()),
            "green_mean": float(vals.mean()),
            "green_median": float(np.median(vals)),
            "green_sum": float(vals.sum()),
            "green_max": float(vals.max()),
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values("green_mean", ascending=False)


def draw_cell_overlay(gray: np.ndarray, labels: np.ndarray, put_ids: bool = True) -> np.ndarray:
    base = to_uint8(gray)
    out = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    mask = (labels > 0).astype(np.uint8) * 255
    edges = cv2.Canny(mask, 50, 150)
    out[edges > 0] = (0, 255, 255)

    if put_ids:
        for cid in range(1, int(labels.max()) + 1):
            ys, xs = np.where(labels == cid)
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(out, str(cid), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return out


# =========================
# VIDEO MODE: (existing pipeline)
# =========================
def auto_calibrate_hsv(video_path: str, sample_frames: int = 8,
                       s_min: int = 30, v_min: int = 30,
                       h_margin: int = 5) -> tuple[list[int], list[int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        frame_count = 200

    idxs = np.linspace(0, max(0, frame_count - 1), sample_frames).astype(int)
    target_set = set(int(i) for i in idxs)

    hs, ss, vs = [], [], []
    cur = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cur in target_set:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            H = hsv[:, :, 0]
            S = hsv[:, :, 1]
            V = hsv[:, :, 2]

            b, g, r = cv2.split(frame)
            rg = (r.astype(np.int16) + g.astype(np.int16)) // 2

            mask = (S >= s_min) & (V >= v_min) & (rg >= np.percentile(rg, 70))
            if mask.any():
                hs.append(H[mask].astype(np.int16))
                ss.append(S[mask].astype(np.int16))
                vs.append(V[mask].astype(np.int16))
        cur += 1
    cap.release()

    if len(hs) == 0:
        return [10, 30, 30], [55, 255, 255]

    H_all = np.concatenate(hs)
    S_all = np.concatenate(ss)
    V_all = np.concatenate(vs)

    h_lo = int(np.percentile(H_all, 5)) - h_margin
    h_hi = int(np.percentile(H_all, 95)) + h_margin
    h_lo = max(0, h_lo)
    h_hi = min(179, h_hi)

    s_lo = max(0, int(np.percentile(S_all, 10)) - 10)
    v_lo = max(0, int(np.percentile(V_all, 10)) - 10)

    return [h_lo, s_lo, v_lo], [h_hi, 255, 255]


def process_video(video_path: str, out_dir: str, params: dict, progress_cb=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_video = out_dir / "tracked_preview.avi"
    output_cell_csv = out_dir / "tracks_with_protein.csv"
    output_spot_csv = out_dir / "spot_tracks.csv"

    BLUR_SIGMA = params["BLUR_SIGMA"]
    CLAHE_CLIP = params["CLAHE_CLIP"]
    CLAHE_TILE = tuple(params["CLAHE_TILE"])
    THRESH_BIAS = params["THRESH_BIAS"]
    MIN_AREA = params["MIN_AREA"]

    MAX_DIST = params["MAX_DIST"]
    AREA_WEIGHT = params["AREA_WEIGHT"]
    MAX_MISSED = params["MAX_MISSED"]
    NEW_TRACK_MIN_AREA = params["NEW_TRACK_MIN_AREA"]
    MAX_NEW_TRACKS_PER_FRAME = params["MAX_NEW_TRACKS_PER_FRAME"]
    CONFIRM_HITS = params["CONFIRM_HITS"]
    DUP_RADIUS = params["DUP_RADIUS"]

    HSV_LOWER = np.array(params["HSV_LOWER"], dtype=np.uint8)
    HSV_UPPER = np.array(params["HSV_UPPER"], dtype=np.uint8)
    SPOT_MIN_AREA = params["SPOT_MIN_AREA"]
    SPOT_MAX_AREA = params["SPOT_MAX_AREA"]
    SPOT_ASSIGN_MAX_DIST = params["SPOT_ASSIGN_MAX_DIST"]

    SPOT_MAX_DIST = params["SPOT_MAX_DIST"]
    SPOT_AREA_WEIGHT = params["SPOT_AREA_WEIGHT"]
    SPOT_MAX_MISSED = params["SPOT_MAX_MISSED"]

    # ========================================================
    # GÜNCELLENEN GÖRÜNTÜ İŞLEME FONKSİYONLARI BURADA BAŞLIYOR
    # ========================================================
    def segment_nuclei(frame_bgr: np.ndarray):
        B = frame_bgr[:, :, 0].astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        B = clahe.apply(B)
        blurred = cv2.GaussianBlur(B, (5, 5), 0)

        # Otsu
        t, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Gürültü temizleme
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k3, iterations=2)

        # Watershed ile hücreleri ayırma
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, surf_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        
        surf_fg = np.uint8(surf_fg)
        unknown = cv2.subtract(opening, surf_fg)

        num, markers = cv2.connectedComponents(surf_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        frame_rgb = cv2.cvtColor(B, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(frame_rgb, markers)

        dets = []
        for m_id in np.unique(markers):
            if m_id <= 1: 
                continue
            
            mask = np.uint8(markers == m_id)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: 
                continue
            
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area < MIN_AREA: 
                continue
            
            M = cv2.moments(c)
            if M["m00"] == 0: 
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(c)
            
            dets.append({
                "cx": cx, "cy": cy, "area": int(area), 
                "bbox": (x, y, w, h), "cc_id": int(m_id),
                "mask": mask # PROTEIN ATAMASI İÇİN MASKEYİ SAKLIYORUZ
            })
        return dets

    def compute_yellow_mask_hsv(frame_bgr: np.ndarray) -> np.ndarray:
        # Top-Hat filtresi (Arka planı sil, küçük/parlak noktaları vurgula)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kernel_top = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_top)
        
        # DÜZELTME: Tophat sonucunu siyah-beyaz maskeye çeviriyoruz (gökyüzü yıldız hatasını çözen yer)
        _, tophat_mask = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
        
        # Orijinal HSV maskesi
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        
        # İkisinin kesişimi
        return cv2.bitwise_and(mask_hsv, tophat_mask)

    def extract_spots(yellow_mask: np.ndarray):
        num, _spot_labels, stats, centroids = cv2.connectedComponentsWithStats(yellow_mask, connectivity=8)
        spots = []
        for sid in range(1, num):
            area = int(stats[sid, cv2.CC_STAT_AREA])
            if area < SPOT_MIN_AREA or area > SPOT_MAX_AREA:
                continue
            cx, cy = centroids[sid]
            spots.append({"x": float(cx), "y": float(cy), "area": area})
        return spots

    def assign_spots_to_nearest_nucleus(spots: list, nuclei: list, max_dist: float):
        cell_spot_count = {}
        cell_spot_area = {}

        for s in spots:
            sx, sy = int(s["x"]), int(s["y"])
            assigned_id = 0
            
            # 1. ÖNCELİK: Maske İçinde mi? (Kesin Atama)
            for n in nuclei:
                x, y, w, h = n["bbox"]
                if x <= sx <= x + w and y <= sy <= y + h:
                    try:
                        if n["mask"][sy, sx] > 0:
                            assigned_id = n["cc_id"]
                            break
                    except IndexError:
                        pass
            
            # 2. YEDEK (Fallback): Eğer maske dışındaysa eski mesafe formülünü kullan
            if assigned_id == 0 and len(nuclei) > 0:
                nuc_xy = np.array([(d["cx"], d["cy"], d["cc_id"]) for d in nuclei])
                dx = nuc_xy[:, 0] - sx
                dy = nuc_xy[:, 1] - sy
                dist2 = dx * dx + dy * dy
                k = int(np.argmin(dist2))
                dist = float(np.sqrt(dist2[k]))
                if dist <= max_dist:
                    assigned_id = int(nuc_xy[k, 2])

            s["cell_cc_id"] = assigned_id

            if assigned_id != 0:
                cell_spot_count[assigned_id] = cell_spot_count.get(assigned_id, 0) + 1
                cell_spot_area[assigned_id] = cell_spot_area.get(assigned_id, 0) + int(s["area"])

        return spots, cell_spot_count, cell_spot_area
    # ========================================================
    # GÜNCELLENEN GÖRÜNTÜ İŞLEME FONKSİYONLARI BURADA BİTİYOR
    # ========================================================

    def hungarian_assign_tracks(tracks: dict, dets_xy_area: list, max_dist: float, area_weight: float):
        track_ids = list(tracks.keys())
        if len(track_ids) == 0 or len(dets_xy_area) == 0:
            return [-1] * len(dets_xy_area)

        cost = np.full((len(track_ids), len(dets_xy_area)), 1e6, dtype=np.float32)

        for i, tid in enumerate(track_ids):
            tx, ty = tracks[tid]["x"], tracks[tid]["y"]
            ta = max(int(tracks[tid]["area"]), 1)
            for j, d in enumerate(dets_xy_area):
                dx = d["x"] - tx
                dy = d["y"] - ty
                dist = float((dx * dx + dy * dy) ** 0.5)
                if dist <= max_dist:
                    area_ratio = abs(int(d["area"]) - ta) / ta
                    cost[i, j] = dist + area_weight * area_ratio

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned = [-1] * len(dets_xy_area)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1e5:
                assigned[c] = track_ids[r]
        return assigned

    def near_existing(tracks: dict, x: float, y: float, radius: float) -> bool:
        for tr in tracks.values():
            dx = x - tr["x"]
            dy = y - tr["y"]
            if (dx * dx + dy * dy) ** 0.5 <= radius:
                return True
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (W, H))
    if not out.isOpened():
        raise RuntimeError("VideoWriter açılamadı (AVI/MJPG).")

    # cell tracks
    cell_tracks = {}
    next_cell_id = 1
    cell_rows = []

    # spot tracks
    spot_tracks = {}
    next_spot_id = 1
    spot_rows = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        nuclei = segment_nuclei(frame)

        yellow_mask = compute_yellow_mask_hsv(frame)
        spots = extract_spots(yellow_mask)
        spots, cell_spot_count, cell_spot_area = assign_spots_to_nearest_nucleus(spots, nuclei, SPOT_ASSIGN_MAX_DIST)

        # --- cell tracking ---
        dets_cells = [{"x": d["cx"], "y": d["cy"], "area": d["area"], "bbox": d["bbox"], "cc_id": d["cc_id"]} for d in nuclei]
        assigned = hungarian_assign_tracks(cell_tracks, dets_cells, MAX_DIST, AREA_WEIGHT)

        for tid in list(cell_tracks.keys()):
            cell_tracks[tid]["matched"] = False

        new_tracks_opened = 0
        for i, d in enumerate(dets_cells):
            tid = assigned[i]
            if tid == -1:
                if d["area"] < NEW_TRACK_MIN_AREA:
                    continue
                if new_tracks_opened >= MAX_NEW_TRACKS_PER_FRAME:
                    continue
                if near_existing(cell_tracks, d["x"], d["y"], DUP_RADIUS):
                    continue

                tid = next_cell_id
                next_cell_id += 1
                cell_tracks[tid] = {"x": d["x"], "y": d["y"], "area": d["area"], "missed": 0, "hits": 1, "confirmed": False, "matched": True}
                new_tracks_opened += 1
            else:
                tr = cell_tracks[tid]
                tr["x"], tr["y"], tr["area"] = d["x"], d["y"], d["area"]
                tr["missed"] = 0
                tr["matched"] = True
                tr["hits"] = tr.get("hits", 0) + 1
                if (not tr.get("confirmed", False)) and tr["hits"] >= CONFIRM_HITS:
                    tr["confirmed"] = True

            cc_id = int(d["cc_id"])
            spot_n = int(cell_spot_count.get(cc_id, 0))
            spot_area_sum = int(cell_spot_area.get(cc_id, 0))

            if cell_tracks[tid].get("confirmed", False):
                cell_rows.append({
                    "frame": frame_idx,
                    "track_id": tid,
                    "x": d["x"],
                    "y": d["y"],
                    "nucleus_area": int(d["area"]),
                    "spot_count": spot_n,
                    "spot_area_sum": spot_area_sum,
                })

                x, y, w, h = d["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(frame, f"ID {tid} S{spot_n}", (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for tid in list(cell_tracks.keys()):
            if not cell_tracks[tid].get("matched", False):
                cell_tracks[tid]["missed"] += 1
            if cell_tracks[tid]["missed"] > MAX_MISSED:
                del cell_tracks[tid]

        # --- spot tracking ---
        dets_spots = [{"x": s["x"], "y": s["y"], "area": s["area"]} for s in spots]
        assigned_s = hungarian_assign_tracks(spot_tracks, dets_spots, SPOT_MAX_DIST, SPOT_AREA_WEIGHT)

        for sid in list(spot_tracks.keys()):
            spot_tracks[sid]["matched"] = False

        for j, s in enumerate(dets_spots):
            sid = assigned_s[j]
            if sid == -1:
                sid = next_spot_id
                next_spot_id += 1
                spot_tracks[sid] = {"x": s["x"], "y": s["y"], "area": s["area"], "missed": 0, "matched": True}
            else:
                tr = spot_tracks[sid]
                tr["x"], tr["y"], tr["area"] = s["x"], s["y"], s["area"]
                tr["missed"] = 0
                tr["matched"] = True

            spot_rows.append({
                "frame": frame_idx,
                "spot_id": sid,
                "x": float(s["x"]),
                "y": float(s["y"]),
                "spot_area": int(s["area"]),
            })

        for sid in list(spot_tracks.keys()):
            if not spot_tracks[sid].get("matched", False):
                spot_tracks[sid]["missed"] += 1
            if spot_tracks[sid]["missed"] > SPOT_MAX_MISSED:
                del spot_tracks[sid]

        for s in spots:
            cv2.circle(frame, (int(round(s["x"])), int(round(s["y"]))), 2, (0, 255, 255), 1)

        out.write(frame)
        frame_idx += 1
        if progress_cb and total_frames > 0 and frame_idx % 2 == 0:
            progress_cb(min(frame_idx / total_frames, 1.0), frame_idx, total_frames)

    cap.release()
    out.release()

    cell_df = pd.DataFrame(cell_rows)
    spot_df = pd.DataFrame(spot_rows)
    cell_df.to_csv(output_cell_csv, index=False)
    spot_df.to_csv(output_spot_csv, index=False)

    return {
        "video_out": str(output_video),
        "cell_csv": str(output_cell_csv),
        "spot_csv": str(output_spot_csv),
        "cell_df": cell_df,
        "spot_df": spot_df,
    }


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Cell/Embryo Analyzer", layout="wide")
st.title("Cell / Embryo Analyzer")

st.warning("Lütfen yüz/kimlik gibi hassas içerik içeren dosyalar yüklemeyin. Büyük dosyalar (200MB+) sorun çıkarabilir.")

mode = st.radio("Mod seç", ["Video", "Embryo Image (Red + Green)"])

# -------------------------------------------------------------------
# IMAGE MODE UI
# -------------------------------------------------------------------
if mode == "Embryo Image (Red + Green)":
    st.subheader("Embryo Image Modu: Kırmızıdan hücre ID → Yeşilden intensity ölç")

    red_up = st.file_uploader("Red channel (ID için) - TIF/PNG", type=["tif", "tiff", "png", "jpg", "jpeg"], key="red")
    green_up = st.file_uploader("Green channel (intensity için) - TIF/PNG", type=["tif", "tiff", "png", "jpg", "jpeg"], key="green")

    c1, c2, c3 = st.columns(3)
    min_area = c1.number_input("Min hücre alanı (px)", value=50, step=10,
                               help="Çok küçük gürültüleri atar. Hücre kaçırıyorsa düşür.")
    peak_thr = c2.slider("Ayırma hassasiyeti (peak)", 0.20, 0.60, 0.35,
                         help="Hücreler çok birleşiyorsa biraz düşür (0.30). Fazla parçalıysa artır (0.40-0.50).")
    put_ids = c3.checkbox("ID yazdır", value=True)

    run = st.button("Çalıştır", disabled=(red_up is None or green_up is None))

    if run and red_up is not None and green_up is not None:
        red_img = read_image_any(red_up)
        green_img = read_image_any(green_up)

        if red_img.ndim == 3:
            red_gray = cv2.cvtColor(to_uint8(red_img), cv2.COLOR_BGR2GRAY)
        else:
            red_gray = red_img

        if green_img.ndim == 3:
            green_gray = cv2.cvtColor(to_uint8(green_img), cv2.COLOR_BGR2GRAY)
        else:
            green_gray = green_img

        labels = segment_cells_from_red(red_gray, min_area=int(min_area), peak_thresh=float(peak_thr))
        df = measure_green_intensity(labels, green_gray)

        red_overlay = draw_cell_overlay(red_gray, labels, put_ids=put_ids)
        green_overlay = draw_cell_overlay(green_gray, labels, put_ids=False)

        st.success(f"Bitti ✅ Bulunan hücre sayısı: {int(labels.max())}")

        colA, colB = st.columns(2)
        with colA:
            st.write("Red overlay (ID + sınırlar)")
            st.image(red_overlay, channels="BGR", use_container_width=True)
        with colB:
            st.write("Green overlay (sınırlar)")
            st.image(green_overlay, channels="BGR", use_container_width=True)

        st.subheader("Hücre başına Green intensity tablosu")
        st.dataframe(df, use_container_width=True)

        _, red_png = cv2.imencode(".png", red_overlay)
        _, green_png = cv2.imencode(".png", green_overlay)
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        zip_bytes = make_zip([
            ("cell_intensity.csv", csv_bytes),
            ("red_overlay_ids.png", red_png.tobytes()),
            ("green_overlay.png", green_png.tobytes()),
        ])

        st.download_button("Tüm çıktıları indir (ZIP)", data=zip_bytes,
                           file_name="embryo_image_outputs.zip", mime="application/zip")
        st.download_button("CSV indir (cell_intensity.csv)", data=csv_bytes,
                           file_name="cell_intensity.csv", mime="text/csv")

# -------------------------------------------------------------------
# VIDEO MODE UI
# -------------------------------------------------------------------
else:
    st.subheader("Video Modu: Hücre takibi + protein spot")

    uploaded = st.file_uploader("Videoyu yükle", type=["mp4", "avi", "mov", "mkv"], key="video")

    if uploaded is not None and getattr(uploaded, "size", 0) > 200 * 1024 * 1024:
        st.error("Video çok büyük (200MB üstü). Daha küçük bir video yükleyin.")
        st.stop()

    default_out = str((Path.cwd() / "outputs").resolve())
    out_dir = st.text_input("Çıktı klasörü (yol)", value=default_out)

    col1, col2, col3 = st.columns(3)
    auto_hsv = col1.checkbox("Sarı rengi otomatik ayarla (önerilir)", value=True)
    sample_frames = col2.slider("HSV örnek kare sayısı", 3, 20, 8)
    make_mp4 = col3.checkbox("Önizlemeyi MP4'e çevir ve sayfada oynat", value=True)

    with st.expander("Ayarlar (açıklamalı)"):
        cA, cB = st.columns(2)
        MIN_AREA = cA.number_input(
            "Minimum çekirdek alanı (piksel)", value=180, step=10,
            help="Çekirdek kaçırıyorsa düşür (140-180). Gürültü fazlaysa yükselt (200-300)."
        )
        THRESH_BIAS = cB.number_input(
            "Eşik düzeltme (THRESH_BIAS)", value=0, step=1,
            help="Az çekirdek çıkıyorsa negatif (-5). Gürültü fazlaysa pozitif (+5)."
        )
        SPOT_ASSIGN_MAX_DIST = cA.number_input(
            "Spot→çekirdek max mesafe (piksel)", value=60.0, step=5.0,
            help="Protein çekirdeğin çevresinde ise artır (60-90). Yanlış atama varsa düşür (35-60)."
        )
        SPOT_MAX_DIST = cB.number_input(

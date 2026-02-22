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


# =========================
# HSV AUTO-CALIBRATION
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


# =========================
# ZIP helper
# =========================
def make_zip(files: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            z.writestr(name, data)
    return buf.getvalue()


# =========================
# AVI -> MP4 conversion (H.264)
# =========================
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


# =========================
# PIPELINE (GUI'siz) + PROGRESS
# =========================
def process_video(video_path: str, out_dir: str, params: dict, progress_cb=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_video = out_dir / "tracked_preview.avi"
    output_cell_csv = out_dir / "tracks_with_protein.csv"
    output_spot_csv = out_dir / "spot_tracks.csv"

    # --- params ---
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

    DRAW_BBOX = True
    DRAW_SPOTS = True

    def segment_nuclei(frame_bgr: np.ndarray):
        B = frame_bgr[:, :, 0].astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        B = clahe.apply(B)
        B = cv2.GaussianBlur(B, (0, 0), BLUR_SIGMA)

        t, _ = cv2.threshold(B, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = int(np.clip(t + THRESH_BIAS, 0, 255))
        mask = (B > t).astype(np.uint8) * 255

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)

        num, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        dets = []
        for cc_id in range(1, num):
            area = int(stats[cc_id, cv2.CC_STAT_AREA])
            if area < MIN_AREA:
                continue
            cx, cy = centroids[cc_id]
            x = int(stats[cc_id, cv2.CC_STAT_LEFT])
            y = int(stats[cc_id, cv2.CC_STAT_TOP])
            w = int(stats[cc_id, cv2.CC_STAT_WIDTH])
            h = int(stats[cc_id, cv2.CC_STAT_HEIGHT])
            dets.append({"cx": float(cx), "cy": float(cy), "area": area, "bbox": (x, y, w, h), "cc_id": int(cc_id)})
        return dets

    def compute_yellow_mask_hsv(frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

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
        if len(nuclei) == 0:
            for s in spots:
                s["cell_cc_id"] = 0
            return spots, {}, {}

        nuc_xy = np.array([(d["cx"], d["cy"], d["cc_id"]) for d in nuclei], dtype=np.float32)
        cell_spot_count = {}
        cell_spot_area = {}

        for s in spots:
            sx, sy = s["x"], s["y"]
            dx = nuc_xy[:, 0] - sx
            dy = nuc_xy[:, 1] - sy
            dist2 = dx*dx + dy*dy
            k = int(np.argmin(dist2))
            dist = float(np.sqrt(dist2[k]))
            cc_id = int(nuc_xy[k, 2]) if dist <= max_dist else 0
            s["cell_cc_id"] = cc_id

            if cc_id != 0:
                cell_spot_count[cc_id] = cell_spot_count.get(cc_id, 0) + 1
                cell_spot_area[cc_id] = cell_spot_area.get(cc_id, 0) + int(s["area"])

        return spots, cell_spot_count, cell_spot_area

    def hungarian_assign_cells(cell_tracks: dict, dets: list, max_dist: float):
        track_ids = list(cell_tracks.keys())
        if len(track_ids) == 0 or len(dets) == 0:
            return [-1] * len(dets)

        cost = np.full((len(track_ids), len(dets)), 1e6, dtype=np.float32)
        for i, tid in enumerate(track_ids):
            tx, ty = cell_tracks[tid]["cx"], cell_tracks[tid]["cy"]
            ta = max(int(cell_tracks[tid]["area"]), 1)
            for j, d in enumerate(dets):
                dx = d["cx"] - tx
                dy = d["cy"] - ty
                dist = float((dx*dx + dy*dy) ** 0.5)
                if dist <= max_dist:
                    area_ratio = abs(int(d["area"]) - ta) / ta
                    cost[i, j] = dist + AREA_WEIGHT * area_ratio

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned = [-1] * len(dets)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1e5:
                assigned[c] = track_ids[r]
        return assigned

    def near_existing_track(cell_tracks: dict, d: dict, radius: float) -> bool:
        for tr in cell_tracks.values():
            dx = d["cx"] - tr["cx"]
            dy = d["cy"] - tr["cy"]
            if (dx*dx + dy*dy) ** 0.5 <= radius:
                return True
        return False

    def hungarian_assign_spots(spot_tracks: dict, spots: list):
        track_ids = list(spot_tracks.keys())
        if len(track_ids) == 0 or len(spots) == 0:
            return [-1] * len(spots)

        cost = np.full((len(track_ids), len(spots)), 1e6, dtype=np.float32)
        for i, sid in enumerate(track_ids):
            tx, ty = spot_tracks[sid]["x"], spot_tracks[sid]["y"]
            ta = max(int(spot_tracks[sid]["area"]), 1)
            for j, s in enumerate(spots):
                dx = s["x"] - tx
                dy = s["y"] - ty
                dist = float((dx*dx + dy*dy) ** 0.5)
                if dist <= SPOT_MAX_DIST:
                    area_ratio = abs(int(s["area"]) - ta) / ta
                    cost[i, j] = dist + SPOT_AREA_WEIGHT * area_ratio

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned = [-1] * len(spots)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1e5:
                assigned[c] = track_ids[r]
        return assigned

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

    cell_tracks = {}
    next_cell_id = 1
    cell_rows = []

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
        spots, cell_spot_count, cell_spot_area = assign_spots_to_nearest_nucleus(
            spots, nuclei, max_dist=SPOT_ASSIGN_MAX_DIST
        )

        assigned_cells = hungarian_assign_cells(cell_tracks, nuclei, MAX_DIST)
        for tid in list(cell_tracks.keys()):
            cell_tracks[tid]["matched"] = False

        new_tracks_opened = 0
        for i, d in enumerate(nuclei):
            tid = assigned_cells[i]
            if tid == -1:
                if d["area"] < NEW_TRACK_MIN_AREA:
                    continue
                if new_tracks_opened >= MAX_NEW_TRACKS_PER_FRAME:
                    continue
                if near_existing_track(cell_tracks, d, DUP_RADIUS):
                    continue

                tid = next_cell_id
                next_cell_id += 1
                cell_tracks[tid] = {
                    "cx": d["cx"], "cy": d["cy"], "area": d["area"],
                    "missed": 0, "matched": True,
                    "hits": 1, "confirmed": False
                }
                new_tracks_opened += 1
            else:
                tr = cell_tracks[tid]
                tr["cx"], tr["cy"], tr["area"] = d["cx"], d["cy"], d["area"]
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
                    "x": d["cx"],
                    "y": d["cy"],
                    "nucleus_area": d["area"],
                    "spot_count": spot_n,
                    "spot_area_sum": spot_area_sum,
                })

                x, y, w, h = d["bbox"]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.putText(frame, f"ID {tid} S{spot_n}", (x, max(0, y-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for tid in list(cell_tracks.keys()):
            if not cell_tracks[tid].get("matched", False):
                cell_tracks[tid]["missed"] += 1
            if cell_tracks[tid]["missed"] > MAX_MISSED:
                del cell_tracks[tid]

        for sid in list(spot_tracks.keys()):
            spot_tracks[sid]["matched"] = False

        assigned_spots = hungarian_assign_spots(spot_tracks, spots)
        for j, s in enumerate(spots):
            sid = assigned_spots[j]
            if sid == -1:
                sid = next_spot_id
                next_spot_id += 1
                spot_tracks[sid] = {"x": s["x"], "y": s["y"], "area": s["area"], "missed": 0, "matched": True}
            else:
                spot_tracks[sid]["x"] = s["x"]
                spot_tracks[sid]["y"] = s["y"]
                spot_tracks[sid]["area"] = s["area"]
                spot_tracks[sid]["missed"] = 0
                spot_tracks[sid]["matched"] = True

            spot_rows.append({
                "frame": frame_idx,
                "spot_id": sid,
                "x": s["x"],
                "y": s["y"],
                "spot_area": int(s["area"]),
            })

        for sid in list(spot_tracks.keys()):
            if not spot_tracks[sid].get("matched", False):
                spot_tracks[sid]["missed"] += 1
            if spot_tracks[sid]["missed"] > SPOT_MAX_MISSED:
                del spot_tracks[sid]

        if DRAW_SPOTS:
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
st.set_page_config(page_title="Hücre + Protein Spot Takibi", layout="wide")
st.title("Hücre + Protein Spot Takibi")

uploaded = st.file_uploader("Videoyu yükle", type=["mp4", "avi", "mov", "mkv"])

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
        "Spot tracking max hareket (piksel)", value=15.0, step=1.0,
        help="Spot hızlıysa artır (18-25). Karışıyorsa azalt (12-16)."
    )

btn = st.button("Başlat", disabled=(uploaded is None))

if btn and uploaded is not None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_path = tmp / uploaded.name
        in_path.write_bytes(uploaded.getbuffer())

        hsv_lower = [10, 30, 30]
        hsv_upper = [55, 255, 255]
        if auto_hsv:
            hsv_lower, hsv_upper = auto_calibrate_hsv(str(in_path), sample_frames=sample_frames)

        params = dict(
            BLUR_SIGMA=1.2,
            CLAHE_CLIP=2.0,
            CLAHE_TILE=[8, 8],
            THRESH_BIAS=int(THRESH_BIAS),
            MIN_AREA=int(MIN_AREA),

            MAX_DIST=20,
            AREA_WEIGHT=18.0,
            MAX_MISSED=2,
            NEW_TRACK_MIN_AREA=300,
            MAX_NEW_TRACKS_PER_FRAME=6,
            CONFIRM_HITS=4,
            DUP_RADIUS=20,

            HSV_LOWER=hsv_lower,
            HSV_UPPER=hsv_upper,
            SPOT_MIN_AREA=1,
            SPOT_MAX_AREA=200,
            SPOT_ASSIGN_MAX_DIST=float(SPOT_ASSIGN_MAX_DIST),

            SPOT_MAX_DIST=float(SPOT_MAX_DIST),
            SPOT_AREA_WEIGHT=2.0,
            SPOT_MAX_MISSED=2,
        )

        progress = st.progress(0)
        status = st.empty()

        def cb(p, fi, total):
            progress.progress(int(p * 100))
            status.write(f"İşleniyor: {fi}/{total}")

        st.info("İşleniyor...")
        res = process_video(str(in_path), out_dir, params, progress_cb=cb)

    st.success("Bitti ✅")
    st.code(f"HSV LOWER={hsv_lower}\nHSV UPPER={hsv_upper}")

    # ---- VIDEO PREVIEW (MP4) ----
    mp4_bytes = None
    mp4_path = None
    if make_mp4:
        try:
            avi_path = res["video_out"]
            mp4_path = str(Path(avi_path).with_suffix(".mp4"))
            convert_avi_to_mp4(avi_path, mp4_path)
            mp4_bytes = Path(mp4_path).read_bytes()
            st.subheader("Önizleme (Sayfada Oynat)")
            st.video(mp4_bytes)
        except Exception as e:
            st.warning("MP4 dönüştürme başarısız oldu. AVI dosyasını indirip VLC ile açabilirsin.")
            st.caption(str(e))

    # ---- TABLES ----
    st.subheader("Sonuçlar (Tablo)")

    tab1, tab2, tab3 = st.tabs(["Hücre verisi", "Spot verisi", "Hızlı özet"])

    with tab1:
        st.write("tracks_with_protein.csv (ilk 200 satır)")
        st.dataframe(res["cell_df"].head(200), use_container_width=True)

    with tab2:
        st.write("spot_tracks.csv (ilk 200 satır)")
        st.dataframe(res["spot_df"].head(200), use_container_width=True)

    with tab3:
        spot_df = res["spot_df"].sort_values(["spot_id", "frame"])
        g = spot_df.groupby("spot_id", sort=False)
        summary = g.agg(
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            n_frames=("frame", "count"),
            area_first=("spot_area", "first"),
            area_last=("spot_area", "last"),
            area_max=("spot_area", "max"),
        ).reset_index()
        summary["area_delta"] = summary["area_last"] - summary["area_first"]
        summary = summary.sort_values(["area_delta", "area_max"], ascending=False)

        st.write("En çok büyüyen 20 spot")
        st.dataframe(summary.head(20), use_container_width=True)

    # ---- DOWNLOADS ----
    st.subheader("İndir")

    video_bytes = Path(res["video_out"]).read_bytes()
    cell_csv_bytes = Path(res["cell_csv"]).read_bytes()
    spot_csv_bytes = Path(res["spot_csv"]).read_bytes()

    zip_list = [
        ("tracked_preview.avi", video_bytes),
        ("tracks_with_protein.csv", cell_csv_bytes),
        ("spot_tracks.csv", spot_csv_bytes),
    ]
    if mp4_bytes is not None:
        zip_list.insert(0, ("tracked_preview.mp4", mp4_bytes))

    zip_bytes = make_zip(zip_list)

    st.download_button("Tüm çıktıları indir (ZIP)", data=zip_bytes,
                       file_name="cell_protein_outputs.zip", mime="application/zip")

    if mp4_bytes is not None:
        st.download_button("Önizleme videosu (MP4) indir", data=mp4_bytes,
                           file_name="tracked_preview.mp4", mime="video/mp4")

    st.download_button("Önizleme videosu (AVI) indir", data=video_bytes,
                       file_name="tracked_preview.avi", mime="video/x-msvideo")
    st.download_button("Hücre CSV indir", data=cell_csv_bytes,
                       file_name="tracks_with_protein.csv", mime="text/csv")
    st.download_button("Spot CSV indir", data=spot_csv_bytes,
                       file_name="spot_tracks.csv", mime="text/csv")

st.caption("Not: OpenCV pencere açamadığı için önizleme video dosya olarak üretilir.")
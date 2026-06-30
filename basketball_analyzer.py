"""
basketball_analyzer.py
=======================
AI Basketball Motion Analyzer — v2.4
Developer: Ryan Su | Midland School, Class of 2027

Changes in v2.4 (over v2.3):
  [FIX] Reset now removes ONLY the ID(s) visible in the current frame,
        not linked_ids (which always holds the most-recently-clicked ID).

        Before (v2.3 bug):
          _reset_ids() Case B → deletes self.linked_ids
          → if user scrubs back to old segment and resets, the NEW (correct)
            ID stored in linked_ids gets deleted instead of the old one.

        After (v2.4):
          _get_frame_history_ids() returns the intersection of
          history_ids ∩ ids-visible-in-current-frame.
          _reset_ids() removes only those frame-local IDs.
          linked_ids is updated to remove the same IDs so it stays consistent,
          but it is NOT the primary target of deletion.
"""
# New main root script for basketball motion analysis

import os
import pickle
import csv
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_FILE = "final_cache.pkl"
ANGLE_FILE = "id_angle_cache.pkl"

SKELETON_LIMBS = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

LIMB_COLORS_BGR = {
    (0,1):(200,200,200),(0,2):(200,200,200),
    (1,3):(200,200,200),(2,4):(200,200,200),
    (5,6):(0,220,255),(5,7):(0,220,255),(7,9):(0,220,255),
    (6,8):(0,220,255),(8,10):(0,220,255),
    (5,11):(80,200,255),(6,12):(80,200,255),
    (11,12):(80,255,120),(11,13):(80,255,120),(13,15):(80,255,120),
    (12,14):(80,255,120),(14,16):(80,255,120),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL & CACHE
# ═══════════════════════════════════════════════════════════════════════════════

model = YOLO('yolov8n-pose.pt')


def _valid_point(pt):
    return len(pt) >= 2 and (pt[0] > 0 or pt[1] > 0)


def calculate_joint_angle(p1, mid, p2):
    if not (_valid_point(p1) and _valid_point(mid) and _valid_point(p2)):
        return np.nan
    p1  = np.array(p1,  dtype=np.float32)
    mid = np.array(mid, dtype=np.float32)
    p2  = np.array(p2,  dtype=np.float32)
    v1, v2 = p1 - mid, p2 - mid
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-4 or n2 < 1e-4:
        return np.nan
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def load_or_build_cache(video_path,
                        cache_file=CACHE_FILE,
                        angle_file=ANGLE_FILE):
    cap = cv2.VideoCapture(video_path)

    if os.path.exists(cache_file) and os.path.exists(angle_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        with open(angle_file, 'rb') as f:
            id_angle_data = pickle.load(f)
        cap.release()
        total_frames = len(cache)
        print(f"✅ Loaded cache ({total_frames} frames)")
        return cache, id_angle_data, total_frames

    print("⏳ Pre-analyzing video (runs once, then cached)…")
    cache = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res  = model.track(frame, persist=True,
                           tracker="bytetrack.yaml", verbose=False)
        data = {"ids": [], "boxes": [], "kps": []}
        if len(res) > 0 and res[0].boxes.id is not None:
            data["ids"]   = res[0].boxes.id.cpu().numpy().astype(int).tolist()
            data["boxes"] = res[0].boxes.xyxy.cpu().numpy().tolist()
            data["kps"]   = res[0].keypoints.xy.cpu().numpy().tolist()
        cache.append(data)
    cap.release()

    total_frames = len(cache)
    print(f"⏳ Pre-computing joint angles for {total_frames} frames…")

    id_angle_data = {}
    for f_idx in range(total_frames):
        data = cache[f_idx]
        for i, pid in enumerate(data.get("ids", [])):
            if pid not in id_angle_data:
                id_angle_data[pid] = np.full((total_frames, 4), np.nan)
            kp = np.array(data["kps"][i])
            if len(kp) < 17:
                continue
            id_angle_data[pid][f_idx] = [
                calculate_joint_angle(kp[5],  kp[7],  kp[9]),
                calculate_joint_angle(kp[6],  kp[8],  kp[10]),
                calculate_joint_angle(kp[11], kp[13], kp[15]),
                calculate_joint_angle(kp[12], kp[14], kp[16]),
            ]

    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    with open(angle_file, 'wb') as f:
        pickle.dump(id_angle_data, f)
    print("✅ Pre-analysis finished and cached")
    return cache, id_angle_data, total_frames


def delete_cache():
    """Remove cache files so next launch always re-processes."""
    for path in (CACHE_FILE, ANGLE_FILE):
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"🗑  Deleted cache: {path}")
        except Exception as e:
            print(f"⚠  Could not delete {path}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SKELETON DRAWING HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def draw_skeleton_on_frame(frame, kp_array, angles):
    kp = kp_array

    def valid(idx):
        return kp[idx, 0] > 0 or kp[idx, 1] > 0

    for (a, b), color in LIMB_COLORS_BGR.items():
        if valid(a) and valid(b):
            cv2.line(frame,
                     (int(kp[a,0]), int(kp[a,1])),
                     (int(kp[b,0]), int(kp[b,1])),
                     color, 1, cv2.LINE_AA)

    for idx in range(17):
        if valid(idx):
            pt = (int(kp[idx,0]), int(kp[idx,1]))
            cv2.circle(frame, pt, 3, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 3, (40,40,40),    1,  cv2.LINE_AA)

    le, re, lk, rk = angles[0], angles[1], angles[2], angles[3]
    for kp_idx, val, tag in [(7,le,"LE"),(8,re,"RE"),(13,lk,"LK"),(14,rk,"RK")]:
        if not np.isnan(val) and valid(kp_idx):
            x, y  = int(kp[kp_idx,0]), int(kp[kp_idx,1])
            off_y = -14 if kp_idx in (7, 8) else 22
            pos   = (x - 18, y + off_y)
            text  = f"{tag}:{int(round(val))}"
            cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 2)
            cv2.putText(frame, text, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,240,255), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MAIN APPLICATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class BasketballAnalyzerApp:
    """
    v2.4 — light milk-gray theme.

    Player identity model
    ─────────────────────
    history_ids : set  — all tracker IDs ever confirmed as "this player"
                         (treated as ONE logical identity across ID changes).
    linked_ids  : set  — the most-recently clicked ID(s); used to initialise
                         playback direction only, NOT as the deletion target.

    Reset logic (v2.4 fix)
    ──────────────────────
    "Which ID(s) to delete?" = history_ids ∩ ids-visible-in-current-frame
                               = _get_frame_history_ids()

    This means:
      • Scrub to old segment (old ID visible) → Reset deletes OLD ID only.
      • Scrub to new segment (new ID visible) → Reset deletes NEW ID only.
      • Regardless of what linked_ids currently holds.

    Case A (_was_lost True)  : frame has NO history ID → keep all, clear
                               linked_ids so user can re-click a new ID.
    Case B (_was_lost False) : frame has ≥1 history ID → delete only those
                               frame-local IDs from history_ids.
    """

    # ── Palette ───────────────────────────────────────────────────────────────
    BG         = "#F8F9FA"
    SURFACE    = "#FFFFFF"
    BORDER     = "#DEE2E6"
    TEXT       = "#212529"
    TEXT_DIM   = "#6C757D"
    BTN_TEXT   = "#343A40"
    ACCENT     = "#1971C2"
    ACCENT_LT  = "#D0EBFF"
    ACCENT2    = "#C92A2A"
    ACCENT2_LT = "#FFE3E3"
    SUCCESS    = "#2F9E44"
    BTN_SURF   = "#E9ECEF"
    BTN_HOVER  = "#CED4DA"
    STATUS_BG  = "#F1F3F5"

    CHART_COLORS = ["#1971C2", "#F76707", "#0CA678", "#C92A2A"]
    CHART_LABELS = ["Left Elbow", "Right Elbow", "Left Knee", "Right Knee"]
    CSV_FIELDS = [
        "frame", "time_sec", "player_id", "shot_id",
        "shot_start_frame", "shot_end_frame",
        "analysis_start_frame", "analysis_end_frame",
        "left_elbow_angle", "right_elbow_angle",
        "left_knee_angle", "right_knee_angle",
        "shoulder_y_px", "hip_y_px",
        "left_wrist_y_px", "right_wrist_y_px", "knee_y_px",
        "left_ankle_y_px", "right_ankle_y_px",
        "left_wrist_height_norm", "right_wrist_height_norm",
        "left_ankle_height_norm", "right_ankle_height_norm",
        "wrist_height_norm", "knee_height_norm",
        "avg_elbow_angle", "avg_knee_angle",
        "shooting_hand", "power_leg",
        "pose_quality", "lower_body_quality",
        "view_orientation", "data_confidence",
        "is_knee_lowest_frame", "is_elbow_lowest_frame",
        "is_release_frame", "is_elbow_max_extension_frame",
        "is_knee_max_extension_frame",
        "is_observed_knee_max_extension_frame",
        "knee_max_extension_confirmed", "is_max_extension_frame",
        "event_label"
    ]
    SUMMARY_FIELDS = [
        "shot_id", "player_id", "frame_number", "time_sec",
        "shooting_hand", "power_leg",
        "shooting_elbow_angle_deg", "power_knee_angle_deg",
        "hip_height_norm", "event_label"
    ]

    def __init__(self, root, video_path, cache, id_angle_data, total_frames):
        self.root          = root
        self.video_path    = video_path
        self.cache         = cache
        self.id_angle_data = id_angle_data
        self.total_frames  = total_frames

        self.current_frame    = 0
        self.paused           = True
        self.players_at_frame = []
        self._after_id        = None

        self.linked_ids  = set()
        self.history_ids = set()
        self._was_lost   = False

        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._compute_display_size(self.video_w, self.video_h)

        self._build_ui()
        self._build_chart()
        self._render_frame()

    # ── Adaptive sizing ───────────────────────────────────────────────────────

    def _compute_display_size(self, vid_w, vid_h, panel_w=700, panel_h=560):
        if vid_h == 0 or vid_w == 0:
            self.display_w, self.display_h = panel_w, panel_h
            return
        ratio = vid_w / vid_h
        if vid_h >= vid_w:
            self.display_h = panel_h
            self.display_w = int(panel_h * ratio)
        else:
            self.display_w = panel_w
            self.display_h = int(panel_w / ratio)
        self.display_w = max(self.display_w, 200)
        self.display_h = max(self.display_h, 150)

    # ── Key helper: which history IDs are visible RIGHT NOW? ─────────────────

    def _get_frame_history_ids(self):
        """
        Return the subset of history_ids that appear in the current frame.

        This is the correct answer to "which ID is the user looking at now?"
        regardless of what linked_ids holds.

        Examples
        ────────
        history_ids = {1, 5}   (old ID=1 from minutes ago, new ID=5)
        current frame shows ID 1 (user scrubbed back)
          → returns {1}     ← only delete ID 1 on Reset
        current frame shows ID 5
          → returns {5}     ← only delete ID 5 on Reset
        current frame shows neither (gap / true lost)
          → returns set()  ← _was_lost will be True, Case A applies
        """
        safe = min(self.current_frame, len(self.cache) - 1)
        frame_ids = set(self.cache[safe].get("ids", []))
        return self.history_ids & frame_ids   # intersection

    # ── Tracking-lost check ───────────────────────────────────────────────────

    def _is_tracking_lost(self):
        """Lost = current frame has NONE of the history_ids."""
        if not self.history_ids:
            return False
        return len(self._get_frame_history_ids()) == 0

    def _update_lost_status(self):
        self._was_lost = self._is_tracking_lost()
        if not self.history_ids:
            self.title_hint.config(
                text="Click a player in the video to start tracking",
                fg=self.TEXT_DIM)
        elif self._was_lost:
            self.title_hint.config(
                text="⚠  Tracking lost — click Reset, then re-click the player",
                fg=self.ACCENT2)
        else:
            self.title_hint.config(
                text=f"Tracking player  ·  IDs: {sorted(self.history_ids)}",
                fg=self.SUCCESS)

    # ── Graph / highlight ─────────────────────────────────────────────────────

    def update_full_graph(self, ids_to_show=None):
        if ids_to_show is None:
            ids_to_show = self.history_ids

        x      = np.arange(self.total_frames)
        merged = np.full((self.total_frames, 4), np.nan)

        for pid in ids_to_show:
            if pid not in self.id_angle_data:
                continue
            d    = self.id_angle_data[pid]
            mask = np.isnan(merged) & ~np.isnan(d)
            merged[mask] = d[mask]

        for i, line in enumerate(self.chart_lines):
            line.set_data(x, merged[:, i])

        self.graph_status_lbl.config(
            text=f"Player IDs: {sorted(ids_to_show)}" if ids_to_show else "")
        self.canvas.draw_idle()

    def update_highlight(self):
        if not self.history_ids:
            self.highlight_dot.set_data([], [])
            self.canvas.draw_idle()
            return

        val = np.nan
        for pid in list(self.linked_ids) + list(self.history_ids):
            if pid in self.id_angle_data:
                cf = min(self.current_frame, self.id_angle_data[pid].shape[0]-1)
                v  = self.id_angle_data[pid][cf, 1]
                if not np.isnan(v):
                    val = v
                    break

        self.highlight_dot.set_data([self.current_frame], [val])
        self.canvas.draw_idle()

        pid_show = None
        for pid in list(self.linked_ids) + list(self.history_ids):
            if pid in self.id_angle_data:
                cf = min(self.current_frame, self.id_angle_data[pid].shape[0]-1)
                if not np.all(np.isnan(self.id_angle_data[pid][cf])):
                    pid_show = pid
                    break

        if pid_show is not None:
            cf = min(self.current_frame, self.id_angle_data[pid_show].shape[0]-1)
            a  = self.id_angle_data[pid_show][cf]
            def _f(v): return f"{int(round(v))}°" if not np.isnan(v) else "--"
            self.angle_label.config(
                text=(f"ID {pid_show}  ·  "
                      f"L-Elbow: {_f(a[0])}   R-Elbow: {_f(a[1])}   "
                      f"L-Knee: {_f(a[2])}   R-Knee: {_f(a[3])}"),
                fg=self.ACCENT)

    # ── Data export helpers ───────────────────────────────────────────────────

    @staticmethod
    def _valid_kp(kp, idx):
        return len(kp) > idx and (kp[idx, 0] > 0 or kp[idx, 1] > 0)

    @staticmethod
    def _mean_y(kp, idxs):
        ys = [float(kp[i, 1]) for i in idxs
              if len(kp) > i and (kp[i, 0] > 0 or kp[i, 1] > 0)]
        return float(np.mean(ys)) if ys else np.nan

    @staticmethod
    def _dist(kp, a, b):
        if len(kp) <= max(a, b):
            return np.nan
        if not ((kp[a, 0] > 0 or kp[a, 1] > 0) and
                (kp[b, 0] > 0 or kp[b, 1] > 0)):
            return np.nan
        return float(np.linalg.norm(kp[a] - kp[b]))

    @staticmethod
    def _nan_extreme(values, fn):
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        return float(fn(arr)) if len(arr) else np.nan

    @staticmethod
    def _mean(values):
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        return float(np.mean(arr)) if len(arr) else np.nan

    @staticmethod
    def _has_landmarks(kp, idxs):
        return all(len(kp) > i and (kp[i, 0] > 0 or kp[i, 1] > 0)
                   for i in idxs)

    def _has_visible_landmarks(self, kp, idxs, edge_margin=2):
        for i in idxs:
            if not self._valid_kp(kp, i):
                return False
            x, y = float(kp[i, 0]), float(kp[i, 1])
            if x <= edge_margin or y <= edge_margin:
                return False
            if self.video_w > 0 and x >= self.video_w - edge_margin:
                return False
            if self.video_h > 0 and y >= self.video_h - edge_margin:
                return False
        return True

    def _sanitize_angles(self, kp, angles):
        clean = np.array(angles, dtype=float, copy=True)
        required = [
            (0, [5, 7, 9], False),
            (1, [6, 8, 10], False),
            (2, [11, 13], True),
            (3, [12, 14], True),
        ]
        for angle_idx, kp_idxs, check_edges in required:
            has_points = (
                self._has_visible_landmarks(kp, kp_idxs)
                if check_edges else self._has_landmarks(kp, kp_idxs)
            )
            if not has_points:
                clean[angle_idx] = np.nan
        return clean

    def _quality_markers(self, kp, pose_quality, lower_body_quality):
        shoulder_y = self._mean_y(kp, [5, 6])
        hip_y = self._mean_y(kp, [11, 12])
        torso = abs(hip_y - shoulder_y) if (
            not np.isnan(hip_y) and not np.isnan(shoulder_y)) else np.nan
        shoulder_w = self._dist(kp, 5, 6)
        hip_w = self._dist(kp, 11, 12)

        width_ratio = np.nan
        if not np.isnan(torso) and torso >= 20:
            widths = [v for v in (shoulder_w, hip_w) if not np.isnan(v)]
            if widths:
                width_ratio = max(widths) / torso

        if np.isnan(width_ratio):
            orientation = "unknown"
        elif width_ratio >= 0.55:
            orientation = "front"
        elif width_ratio <= 0.30:
            orientation = "side"
        else:
            orientation = "three_quarter"

        core_idxs = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        visible = sum(1 for idx in core_idxs if self._valid_kp(kp, idx))
        completeness = visible / len(core_idxs)
        edge_cut = any(
            self._valid_kp(kp, idx) and (
                kp[idx, 0] <= 2 or kp[idx, 1] <= 2 or
                (self.video_w > 0 and kp[idx, 0] >= self.video_w - 2) or
                (self.video_h > 0 and kp[idx, 1] >= self.video_h - 2)
            )
            for idx in core_idxs
        )

        score = 0.0
        score += 0.35 if pose_quality else 0.0
        score += 0.35 if lower_body_quality else 0.0
        score += 0.20 * completeness
        score += 0.10 if orientation in ("side", "three_quarter") else 0.0
        if edge_cut:
            score -= 0.20
        score = max(0.0, min(1.0, score))

        if score >= 0.75:
            confidence = "high"
        elif score >= 0.50:
            confidence = "medium"
        else:
            confidence = "low"
        return orientation, confidence

    @staticmethod
    def _merge_segments(segments, gap=10):
        if not segments:
            return []
        segments = sorted(segments)
        merged = [list(segments[0])]
        for start, end in segments[1:]:
            if start <= merged[-1][1] + gap:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [tuple(seg) for seg in merged]

    @staticmethod
    def _fmt_csv_value(value):
        if value is None:
            return ""
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        try:
            if np.isnan(value):
                return ""
        except TypeError:
            pass
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return value

    @staticmethod
    def _fmt_time(seconds):
        seconds = max(0.0, float(seconds))
        mins = int(seconds // 60)
        secs = seconds - mins * 60
        return f"{mins}:{secs:04.1f}"

    def _default_export_path(self):
        export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "exports")
        os.makedirs(export_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(self.video_path))[0]
        id_part = "-".join(str(pid) for pid in sorted(self.history_ids))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base}_ids-{id_part}_shot_motion_data_{stamp}.csv"
        return os.path.join(export_dir, filename)

    @staticmethod
    def _summary_export_path(csv_path):
        directory = os.path.dirname(csv_path)
        filename = os.path.basename(csv_path)
        summary_name = filename.replace("_shot_motion_data_", "_shot_summary_")
        if summary_name == filename:
            root, ext = os.path.splitext(filename)
            summary_name = f"{root}_summary{ext}"
        return os.path.join(directory, summary_name)

    @staticmethod
    def _hip_height_norm(row):
        shoulder_y = row.get("shoulder_y_px", np.nan)
        hip_y = row.get("hip_y_px", np.nan)
        knee_y = row.get("knee_y_px", np.nan)
        ankle_values = [
            row.get("left_ankle_y_px", np.nan),
            row.get("right_ankle_y_px", np.nan),
        ]
        visible_ankles = [v for v in ankle_values if not np.isnan(v)]

        if np.isnan(shoulder_y) or np.isnan(hip_y):
            return np.nan
        if visible_ankles:
            lower_body_y = max(visible_ankles)
        elif not np.isnan(knee_y):
            lower_body_y = knee_y
        else:
            return np.nan

        player_height_proxy = lower_body_y - shoulder_y
        if player_height_proxy < 20:
            return np.nan
        return (hip_y - shoulder_y) / player_height_proxy

    def _build_summary_rows(self, rows):
        summary_rows = []
        event_specs = [
            ("is_knee_lowest_frame", "knee_lowest"),
            ("is_elbow_lowest_frame", "elbow_lowest"),
            ("is_release_frame", "release_proxy"),
            ("is_elbow_max_extension_frame", "elbow_max_extension"),
            ("is_knee_max_extension_frame", "knee_max_extension"),
            ("is_observed_knee_max_extension_frame",
             "observed_knee_max_extension_unconfirmed"),
        ]
        shot_ids = sorted({r["shot_id"] for r in rows if r["shot_id"] != ""})

        for shot_id in shot_ids:
            shot_rows = [r for r in rows if r["shot_id"] == shot_id]
            for flag, label in event_specs:
                matches = [r for r in shot_rows if r.get(flag)]
                if not matches:
                    continue
                if label == "observed_knee_max_extension_unconfirmed":
                    matches = [
                        r for r in matches
                        if not bool(r.get("knee_max_extension_confirmed"))
                    ]
                    if not matches:
                        continue

                for row in matches:
                    shooting_hand = row.get("shooting_hand", "")
                    power_leg = row.get("power_leg", "")
                    if shooting_hand == "left":
                        elbow_angle = row.get("left_elbow_angle", np.nan)
                    elif shooting_hand == "right":
                        elbow_angle = row.get("right_elbow_angle", np.nan)
                    else:
                        elbow_angle = row.get("avg_elbow_angle", np.nan)

                    if power_leg == "left":
                        knee_angle = row.get("left_knee_angle", np.nan)
                    elif power_leg == "right":
                        knee_angle = row.get("right_knee_angle", np.nan)
                    else:
                        knee_angle = row.get("avg_knee_angle", np.nan)

                    summary_rows.append({
                        "shot_id": shot_id,
                        "player_id": row.get("player_id"),
                        "frame_number": row.get("frame"),
                        "time_sec": row.get("time_sec"),
                        "shooting_hand": shooting_hand,
                        "power_leg": power_leg,
                        "shooting_elbow_angle_deg": elbow_angle,
                        "power_knee_angle_deg": knee_angle,
                        "hip_height_norm": self._hip_height_norm(row),
                        "event_label": f"shot_{shot_id}_{label}",
                    })

        summary_rows.sort(key=lambda r: (r["shot_id"], r["frame_number"]))
        return summary_rows

    def _get_player_sample_at_frame(self, frame_idx):
        data = self.cache[frame_idx]
        ids = data.get("ids", [])
        kps = data.get("kps", [])

        for i, pid in enumerate(ids):
            if pid not in self.history_ids or i >= len(kps):
                continue
            kp = np.array(kps[i])
            if len(kp) < 17:
                continue
            angles = np.full(4, np.nan)
            if pid in self.id_angle_data:
                safe = min(frame_idx, self.id_angle_data[pid].shape[0] - 1)
                angles = self.id_angle_data[pid][safe]
            angles = self._sanitize_angles(kp, angles)
            return pid, kp, angles
        return None, None, None

    def _build_export_rows(self):
        rows = []
        for frame_idx in range(self.total_frames):
            pid, kp, angles = self._get_player_sample_at_frame(frame_idx)
            if pid is None:
                continue

            shoulder_y = self._mean_y(kp, [5, 6])
            hip_y = self._mean_y(kp, [11, 12])
            left_wrist_y = float(kp[9, 1]) if self._valid_kp(kp, 9) else np.nan
            right_wrist_y = float(kp[10, 1]) if self._valid_kp(kp, 10) else np.nan
            wrist_y = self._nan_extreme([left_wrist_y, right_wrist_y], np.min)
            knee_y = self._mean_y(kp, [13, 14])
            left_ankle_y = float(kp[15, 1]) if self._valid_kp(kp, 15) else np.nan
            right_ankle_y = float(kp[16, 1]) if self._valid_kp(kp, 16) else np.nan
            torso = abs(hip_y - shoulder_y) if (
                not np.isnan(hip_y) and not np.isnan(shoulder_y)) else np.nan
            if np.isnan(torso) or torso < 20:
                torso = np.nan

            def _height_norm(y):
                if np.isnan(y) or np.isnan(shoulder_y) or np.isnan(torso):
                    return np.nan
                return (shoulder_y - y) / torso

            avg_elbow = self._mean([angles[0], angles[1]])
            avg_knee = self._mean([angles[2], angles[3]])
            pose_quality = int(self._has_landmarks(kp, [5, 6, 7, 8, 9, 10]))
            lower_body_quality = int(self._has_visible_landmarks(
                kp, [5, 6, 11, 12, 13, 14]))
            view_orientation, data_confidence = self._quality_markers(
                kp, pose_quality, lower_body_quality)

            rows.append({
                "frame": frame_idx,
                "time_sec": frame_idx / self.fps,
                "player_id": pid,
                "shot_id": "",
                "shot_start_frame": "",
                "shot_end_frame": "",
                "analysis_start_frame": "",
                "analysis_end_frame": "",
                "left_elbow_angle": angles[0],
                "right_elbow_angle": angles[1],
                "left_knee_angle": angles[2],
                "right_knee_angle": angles[3],
                "shoulder_y_px": shoulder_y,
                "hip_y_px": hip_y,
                "left_wrist_y_px": left_wrist_y,
                "right_wrist_y_px": right_wrist_y,
                "knee_y_px": knee_y,
                "left_ankle_y_px": left_ankle_y,
                "right_ankle_y_px": right_ankle_y,
                "left_wrist_height_norm": _height_norm(left_wrist_y),
                "right_wrist_height_norm": _height_norm(right_wrist_y),
                "left_ankle_height_norm": _height_norm(left_ankle_y),
                "right_ankle_height_norm": _height_norm(right_ankle_y),
                "wrist_height_norm": _height_norm(wrist_y),
                "knee_height_norm": _height_norm(knee_y),
                "avg_elbow_angle": avg_elbow,
                "avg_knee_angle": avg_knee,
                "shooting_hand": "",
                "power_leg": "",
                "pose_quality": pose_quality,
                "lower_body_quality": lower_body_quality,
                "is_knee_lowest_frame": False,
                "is_elbow_lowest_frame": False,
                "is_release_frame": False,
                "is_elbow_max_extension_frame": False,
                "is_knee_max_extension_frame": False,
                "is_observed_knee_max_extension_frame": False,
                "knee_max_extension_confirmed": "",
                "is_max_extension_frame": False,
                "view_orientation": view_orientation,
                "data_confidence": data_confidence,
                "event_label": ""
            })

        shots = self._detect_shot_segments(rows)
        self._mark_shot_events(rows, shots)
        return rows

    def _detect_shot_segments(self, rows):
        if not rows:
            return []

        wrist = np.array([r["wrist_height_norm"] for r in rows], dtype=float)
        left_wrist = np.array([r["left_wrist_height_norm"] for r in rows],
                              dtype=float)
        right_wrist = np.array([r["right_wrist_height_norm"] for r in rows],
                               dtype=float)
        elbow = np.array([r["avg_elbow_angle"] for r in rows], dtype=float)
        upper_ok = np.array([r["pose_quality"] for r in rows], dtype=bool)
        valid = (
            ~np.isnan(wrist) & ~np.isnan(left_wrist) &
            ~np.isnan(right_wrist) & ~np.isnan(elbow) & upper_ok
        )
        if not np.any(valid):
            return []

        two_hands_up = (left_wrist > 0.25) & (right_wrist > 0.25)
        high_hand = valid & two_hands_up & (wrist > 0.35) & (elbow > 120)
        high_idxs = np.where(high_hand)[0]
        if len(high_idxs) == 0:
            return []

        raw_segments = []
        start = high_idxs[0]
        prev = high_idxs[0]
        high_gap = max(4, int(0.35 * self.fps))
        for idx in high_idxs[1:]:
            if idx <= prev + high_gap:
                prev = idx
                continue
            raw_segments.append((start, prev))
            start = idx
            prev = idx
        raw_segments.append((start, prev))

        min_len = 1
        pad_before = max(8, int(0.45 * self.fps))
        pad_after = max(6, int(0.35 * self.fps))
        padded = []
        for start, end in raw_segments:
            if end - start + 1 < min_len:
                continue
            seg_start = max(0, start - pad_before)
            seg_end = min(len(rows) - 1, end + pad_after)
            segment = rows[seg_start:seg_end + 1]
            seg_knee = np.array([r["avg_knee_angle"] for r in segment],
                                dtype=float)
            seg_lower = np.array([r["lower_body_quality"] for r in segment],
                                 dtype=bool)
            valid_knee = seg_knee[~np.isnan(seg_knee) & seg_lower]
            knee_range = np.nan
            if len(valid_knee) >= 5:
                knee_range = np.nanpercentile(valid_knee, 90) - np.nanpercentile(
                    valid_knee, 10)
                if knee_range < 24:
                    continue
            padded.append((seg_start, seg_end))
        return self._merge_segments(padded, gap=max(8, int(0.35 * self.fps)))

    def _mark_shot_events(self, rows, shots):
        if not rows:
            return

        for shot_id, (start, end) in enumerate(shots, start=1):
            segment = rows[start:end + 1]
            for row in segment:
                row["shot_id"] = shot_id
                row["shot_start_frame"] = rows[start]["frame"]
                row["shot_end_frame"] = rows[end]["frame"]

            all_wrist = np.array([r["wrist_height_norm"] for r in rows], dtype=float)
            all_left_wrist = np.array([r["left_wrist_height_norm"] for r in rows],
                                      dtype=float)
            all_right_wrist = np.array([r["right_wrist_height_norm"] for r in rows],
                                       dtype=float)
            all_left_elbow = np.array([r["left_elbow_angle"] for r in rows],
                                      dtype=float)
            all_right_elbow = np.array([r["right_elbow_angle"] for r in rows],
                                       dtype=float)
            all_elbow = np.array([r["avg_elbow_angle"] for r in rows], dtype=float)
            all_left_knee = np.array([r["left_knee_angle"] for r in rows],
                                     dtype=float)
            all_right_knee = np.array([r["right_knee_angle"] for r in rows],
                                      dtype=float)
            all_knee = np.array([r["avg_knee_angle"] for r in rows], dtype=float)
            all_hip_y = np.array([r["hip_y_px"] for r in rows], dtype=float)
            all_knee_height = np.array([r["knee_height_norm"] for r in rows],
                                       dtype=float)
            all_left_ankle = np.array([r["left_ankle_height_norm"] for r in rows],
                                      dtype=float)
            all_right_ankle = np.array([r["right_ankle_height_norm"] for r in rows],
                                       dtype=float)
            all_lower_ok = np.array([r["lower_body_quality"] for r in rows],
                                    dtype=bool)
            all_pose_ok = np.array([r["pose_quality"] for r in rows], dtype=bool)

            knee_low_idx = None
            elbow_low_idx = None
            release_idx = None
            elbow_max_idx = None
            knee_max_idx = None
            observed_knee_max_idx = None
            knee_max_confirmed = False

            valid_release = np.where(
                ~np.isnan(all_wrist) & ~np.isnan(all_left_wrist) &
                ~np.isnan(all_right_wrist) & ~np.isnan(all_elbow) &
                all_pose_ok & (all_left_wrist > 0.25) &
                (all_right_wrist > 0.25) & (all_wrist > 0.35) &
                (all_elbow > 120)
            )[0]
            valid_release = valid_release[
                (valid_release >= start) & (valid_release <= end)
            ]
            if len(valid_release) == 0:
                continue

            shooting_hand = ""

            hand_search = np.arange(start, end + 1)
            hand_search = hand_search[
                all_pose_ok[hand_search] &
                (~np.isnan(all_left_elbow[hand_search]) |
                 ~np.isnan(all_right_elbow[hand_search]))
            ]

            def _elbow_extension_range(values, candidates):
                valid_values = values[candidates]
                valid_values = valid_values[~np.isnan(valid_values)]
                if len(valid_values) < 4:
                    return np.nan
                return (
                    np.nanpercentile(valid_values, 90) -
                    np.nanpercentile(valid_values, 10)
                )

            left_range = _elbow_extension_range(all_left_elbow, hand_search)
            right_range = _elbow_extension_range(all_right_elbow, hand_search)
            if not np.isnan(left_range) and not np.isnan(right_range):
                range_diff = left_range - right_range
                if abs(range_diff) >= 8:
                    shooting_hand = "left" if range_diff > 0 else "right"

            if not shooting_hand:
                left_elbow_peak = (
                    np.nanmax(all_left_elbow[valid_release])
                    if np.any(~np.isnan(all_left_elbow[valid_release]))
                    else np.nan
                )
                right_elbow_peak = (
                    np.nanmax(all_right_elbow[valid_release])
                    if np.any(~np.isnan(all_right_elbow[valid_release]))
                    else np.nan
                )
                if not np.isnan(left_elbow_peak) and not np.isnan(right_elbow_peak):
                    peak_diff = left_elbow_peak - right_elbow_peak
                    if abs(peak_diff) >= 5:
                        shooting_hand = "left" if peak_diff > 0 else "right"

            if not shooting_hand:
                left_score = np.nanmean(all_left_wrist[valid_release])
                right_score = np.nanmean(all_right_wrist[valid_release])
                if not np.isnan(left_score) or not np.isnan(right_score):
                    if np.isnan(right_score) or (
                        not np.isnan(left_score) and left_score >= right_score
                    ):
                        shooting_hand = "left"
                    else:
                        shooting_hand = "right"
            shot_elbow = (
                all_left_elbow if shooting_hand == "left" else all_right_elbow
                if shooting_hand == "right" else all_elbow
            )

            valid_shot_elbow = valid_release[~np.isnan(shot_elbow[valid_release])]
            if len(valid_shot_elbow) > 0:
                peak_elbow = np.nanmax(shot_elbow[valid_shot_elbow])
                peak_wrist = np.nanmax(all_wrist[valid_shot_elbow])
                mature = valid_shot_elbow[
                    (shot_elbow[valid_shot_elbow] >= max(120, peak_elbow - 6)) &
                    (all_wrist[valid_shot_elbow] >= peak_wrist * 0.85)
                ]
                if len(mature) > 0:
                    release_idx = int(mature[0])
                else:
                    score = all_wrist[valid_shot_elbow] + (
                        shot_elbow[valid_shot_elbow] / 180.0)
                    release_idx = int(valid_shot_elbow[np.nanargmax(score)])

            if release_idx is None:
                continue

            pre_load_pad = max(6, int(0.25 * self.fps))
            analysis_start = max(0, start - pre_load_pad)
            analysis_end = min(len(rows) - 1,
                               release_idx + max(8, int(0.35 * self.fps)))
            analysis_idxs = np.arange(analysis_start, analysis_end + 1)

            for row in rows[analysis_start:analysis_end + 1]:
                if row["shot_id"] in ("", shot_id):
                    row["analysis_start_frame"] = rows[analysis_start]["frame"]
                    row["analysis_end_frame"] = rows[analysis_end]["frame"]

            def _before(idx, candidates):
                if idx is None:
                    return candidates
                before = candidates[candidates < idx]
                return before if len(before) > 0 else candidates

            def _after(idx, candidates):
                if idx is None:
                    return candidates
                after = candidates[candidates > idx]
                return after if len(after) > 0 else candidates

            def _takeoff_index(candidates):
                if len(candidates) < 5:
                    return None
                first_n = max(3, int(round(len(candidates) * 0.4)))
                early_height = all_knee_height[candidates[:first_n]]
                baseline = np.nanmedian(early_height)
                if np.isnan(baseline):
                    return None
                lift_threshold = baseline + 0.18
                lift_candidates = candidates[
                    (all_knee_height[candidates] > lift_threshold) &
                    ~np.isnan(all_wrist[candidates]) & (all_wrist[candidates] > 0.25)
                ]
                return int(lift_candidates[0]) if len(lift_candidates) else None

            base_knee_search = np.where(
                ~np.isnan(all_knee_height) & all_lower_ok &
                (np.arange(len(rows)) >= analysis_start) &
                (np.arange(len(rows)) <= analysis_end)
            )[0]
            base_knee_search = base_knee_search[base_knee_search < release_idx]
            takeoff_idx = _takeoff_index(base_knee_search)
            if takeoff_idx is not None:
                before_takeoff = base_knee_search[base_knee_search < takeoff_idx]
                if len(before_takeoff) > 0:
                    base_knee_search = before_takeoff

            left_knee_search = base_knee_search[
                ~np.isnan(all_left_knee[base_knee_search])]
            right_knee_search = base_knee_search[
                ~np.isnan(all_right_knee[base_knee_search])]
            power_leg = ""
            if len(left_knee_search) > 0 or len(right_knee_search) > 0:
                left_min = (
                    np.nanmin(all_left_knee[left_knee_search])
                    if len(left_knee_search) > 0 else np.nan
                )
                right_min = (
                    np.nanmin(all_right_knee[right_knee_search])
                    if len(right_knee_search) > 0 else np.nan
                )
                left_support = (
                    np.nanmedian(all_left_ankle[base_knee_search])
                    if np.any(~np.isnan(all_left_ankle[base_knee_search]))
                    else np.nan
                )
                right_support = (
                    np.nanmedian(all_right_ankle[base_knee_search])
                    if np.any(~np.isnan(all_right_ankle[base_knee_search]))
                    else np.nan
                )
                support_margin = 0.18
                if not np.isnan(left_support) and not np.isnan(right_support):
                    if left_support < right_support - support_margin:
                        power_leg = "right"
                    elif right_support < left_support - support_margin:
                        power_leg = "left"

                if not power_leg and shooting_hand:
                    power_leg = "right" if shooting_hand == "left" else "left"

                if not power_leg and not np.isnan(left_min) and not np.isnan(right_min):
                    knee_diff = left_min - right_min
                    if abs(knee_diff) >= 8:
                        # A much deeper bend often marks the free/tuck leg.
                        power_leg = "left" if knee_diff > 0 else "right"

                if not power_leg:
                    if np.isnan(right_min) or (
                        not np.isnan(left_min) and left_min <= right_min
                    ):
                        power_leg = "left"
                    else:
                        power_leg = "right"
            power_knee = (
                all_left_knee if power_leg == "left" else all_right_knee
                if power_leg == "right" else all_knee
            )
            power_ankle = (
                all_left_ankle if power_leg == "left" else all_right_ankle
                if power_leg == "right" else np.full(len(rows), np.nan)
            )

            landing_idx = None
            landing_search = base_knee_search[base_knee_search >= start]
            for idx in landing_search:
                if idx - analysis_start < 4 or np.isnan(power_ankle[idx]):
                    continue
                prev = power_ankle[idx - 4:idx]
                prev = prev[~np.isnan(prev)]
                if len(prev) < 3:
                    continue
                lower_than_prev = np.sum(prev > power_ankle[idx] + 0.10)
                if lower_than_prev >= 3:
                    landing_idx = int(idx)
                    break
            if landing_idx is not None:
                landed = base_knee_search[base_knee_search >= landing_idx]
                if len(landed) > 0:
                    base_knee_search = landed

            hip_valid = base_knee_search[~np.isnan(all_hip_y[base_knee_search])]
            if len(hip_valid) >= 3:
                hip_low_idx = int(hip_valid[np.nanargmax(all_hip_y[hip_valid])])
                hip_radius = max(2, int(round(0.10 * self.fps)))
                near_lowest_hip = hip_valid[
                    (hip_valid >= hip_low_idx - hip_radius) &
                    (hip_valid <= hip_low_idx + hip_radius)
                ]
                if len(near_lowest_hip) > 0:
                    base_knee_search = near_lowest_hip

            valid_knee = base_knee_search[~np.isnan(power_knee[base_knee_search])]
            if len(valid_knee) > 0:
                search = valid_knee
                if release_idx is not None:
                    before_release = search[search < release_idx]
                    if len(before_release) > 0:
                        search = before_release

                if takeoff_idx is not None:
                    before_takeoff = search[search < takeoff_idx]
                    if len(before_takeoff) > 0:
                        search = before_takeoff

                knee_low_idx = int(search[np.nanargmin(power_knee[search])])

            valid_elbow = np.where(
                ~np.isnan(shot_elbow) & all_pose_ok &
                (np.arange(len(rows)) >= analysis_start) &
                (np.arange(len(rows)) <= analysis_end)
            )[0]
            if len(valid_elbow) > 0:
                search = _before(release_idx, valid_elbow)
                search = _after(knee_low_idx, search)
                elbow_low_idx = int(search[np.nanargmin(shot_elbow[search])])

                search = _after(release_idx, valid_elbow)
                near_release = search[
                    search <= release_idx + max(2, int(round(0.15 * self.fps)))
                ]
                if len(near_release) > 0:
                    search = near_release
                elbow_max_idx = int(search[np.nanargmax(shot_elbow[search])])

            valid_knee_ext = np.where(
                ~np.isnan(power_knee) & all_lower_ok &
                (np.arange(len(rows)) > release_idx) &
                (np.arange(len(rows)) <= analysis_end)
            )[0]
            if len(valid_knee_ext) > 0:
                observed_knee_max_idx = int(
                    valid_knee_ext[np.nanargmax(power_knee[valid_knee_ext])])
                last_valid = int(valid_knee_ext[-1])
                enough_after_release = len(valid_knee_ext) >= max(
                    3, int(round(0.12 * self.fps)))
                reaches_window_end = last_valid >= analysis_end - 1
                knee_max_confirmed = bool(enough_after_release and reaches_window_end)
                if knee_max_confirmed:
                    knee_max_idx = observed_knee_max_idx

            for row in rows[analysis_start:analysis_end + 1]:
                row["shooting_hand"] = shooting_hand
                row["power_leg"] = power_leg

            for local_idx, label, key in [
                (knee_low_idx, "knee_lowest", "is_knee_lowest_frame"),
                (elbow_low_idx, "elbow_lowest", "is_elbow_lowest_frame"),
                (release_idx, "release_proxy", "is_release_frame"),
                (elbow_max_idx, "elbow_max_extension",
                 "is_elbow_max_extension_frame"),
                (knee_max_idx, "knee_max_extension",
                 "is_knee_max_extension_frame"),
            ]:
                if local_idx is None:
                    continue
                row = rows[local_idx]
                row[key] = True
                if label in ("elbow_max_extension", "knee_max_extension"):
                    row["is_max_extension_frame"] = True
                tag = f"shot_{shot_id}_{label}"
                row["event_label"] = (
                    tag if not row["event_label"]
                    else row["event_label"] + "|" + tag
                )

            if observed_knee_max_idx is not None:
                row = rows[observed_knee_max_idx]
                row["is_observed_knee_max_extension_frame"] = True
                row["knee_max_extension_confirmed"] = int(knee_max_confirmed)
                if not knee_max_confirmed:
                    tag = f"shot_{shot_id}_observed_knee_max_extension_unconfirmed"
                    row["event_label"] = (
                        tag if not row["event_label"]
                        else row["event_label"] + "|" + tag
                    )

    def _export_csv(self):
        if not self.history_ids:
            self._update_status("Select a player before exporting CSV.")
            return

        self.paused = True
        self.play_btn.config(text="▶  Play")
        self._update_status("Building export rows...")
        self.root.update_idletasks()

        rows = self._build_export_rows()
        if not rows:
            self._update_status("No tracked player data found to export.")
            return

        path = self._default_export_path()

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    field: self._fmt_csv_value(row.get(field))
                    for field in self.CSV_FIELDS
                })

        summary_rows = self._build_summary_rows(rows)
        summary_path = self._summary_export_path(path)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_FIELDS)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({
                    field: self._fmt_csv_value(row.get(field))
                    for field in self.SUMMARY_FIELDS
                })

        shot_ids = sorted({r["shot_id"] for r in rows if r["shot_id"] != ""})
        self._update_status(
            f"Exported {len(rows)} frames, {len(shot_ids)} shots and "
            f"{len(summary_rows)} summary rows.")
        print(f"[EXPORT] CSV saved: {path}")
        print(f"[EXPORT] Summary CSV saved: {summary_path}")
        print(f"[EXPORT] Detected shot segments: {len(shot_ids)}")

    # ── Frame rendering ───────────────────────────────────────────────────────

    def _render_frame(self, update_chart=True, seek=True):
        safe_frame = min(self.current_frame, len(self.cache)-1)
        if seek:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, safe_frame)
        ret, frame = self.cap.read()
        if not ret:
            return

        out  = frame.copy()
        data = self.cache[safe_frame]
        self.players_at_frame.clear()

        ids   = data.get("ids",   [])
        boxes = data.get("boxes", [])
        kps   = data.get("kps",   [])

        for i, pid in enumerate(ids):
            if i >= len(boxes) or i >= len(kps):
                continue
            x1, y1, x2, y2 = map(int, boxes[i])
            self.players_at_frame.append((pid, (x1, y1, x2, y2)))

            kp_arr     = np.array(kps[i])
            is_tracked = (pid in self.history_ids)
            box_color  = (60, 60, 220) if is_tracked else (30, 180, 180)
            box_thick  = 3             if is_tracked else 1
            cv2.rectangle(out, (x1,y1), (x2,y2), box_color, box_thick)

            lbl_color = (60, 60, 220) if is_tracked else (30, 180, 180)
            cv2.putText(out, f"ID:{pid}", (x1, max(y1-8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,0,0), 3)
            cv2.putText(out, f"ID:{pid}", (x1, max(y1-8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, lbl_color, 1)

            if len(kp_arr) >= 17:
                angles = [np.nan]*4
                if pid in self.id_angle_data:
                    cf = min(safe_frame, self.id_angle_data[pid].shape[0]-1)
                    angles = self.id_angle_data[pid][cf]
                draw_skeleton_on_frame(out, kp_arr, angles)

        if not self.history_ids:
            cx = max(30, out.shape[1]//2 - 200)
            cv2.putText(out, "CLICK PLAYER TO LINK", (cx, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 4)
            cv2.putText(out, "CLICK PLAYER TO LINK", (cx, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,230,230), 2)

        rgb   = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        pil   = pil.resize((self.display_w, self.display_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self.video_label.config(image=photo,
                                width=self.display_w, height=self.display_h)
        self.video_label.image = photo

        self.timeline_var.set(self.current_frame)
        total_frame_idx = max(0, self.total_frames - 1)
        self.frame_label.config(
            text=(f"{self.current_frame} / {total_frame_idx}  ·  "
                  f"{self._fmt_time(self.current_frame / self.fps)} / "
                  f"{self._fmt_time(total_frame_idx / self.fps)}"))
        if update_chart:
            self.update_highlight()
        self._update_lost_status()

    # ── UI builder ────────────────────────────────────────────────────────────

    def _build_ui(self):
        r = self.root
        r.title("AI Basketball Motion Analyzer  ·  Ryan Su  |  Midland School '27")
        r.configure(bg=self.BG)
        r.resizable(True, True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Horizontal.TScale",
                        background=self.BG, troughcolor=self.BORDER,
                        sliderlength=14, sliderrelief="flat")

        title_bar = tk.Frame(r, bg=self.SURFACE,
                             highlightbackground=self.BORDER,
                             highlightthickness=1, height=46)
        title_bar.pack(fill="x", side="top")
        tk.Label(title_bar, text="🏀  AI Basketball Motion Analyzer",
                 bg=self.SURFACE, fg=self.ACCENT,
                 font=("Menlo", 14, "bold")).pack(side="left", padx=16, pady=10)
        self.title_hint = tk.Label(
            title_bar,
            text="Click a player in the video to start tracking",
            bg=self.SURFACE, fg=self.TEXT_DIM, font=("Menlo", 9))
        self.title_hint.pack(side="right", padx=16)

        content = tk.Frame(r, bg=self.BG)
        content.pack(fill="both", expand=True, padx=8, pady=(6,4))

        # LEFT
        left = tk.Frame(content, bg=self.SURFACE,
                        highlightbackground=self.BORDER, highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0,4))

        tk.Label(left, text="VIDEO", bg=self.SURFACE, fg=self.TEXT_DIM,
                 font=("Menlo", 8, "bold")).pack(anchor="w", padx=10, pady=(6,2))

        self.video_label = tk.Label(left, bg="#1A1A2E",
                                    width=self.display_w, height=self.display_h,
                                    cursor="crosshair")
        self.video_label.pack(padx=8, pady=(0,4))
        self.video_label.bind("<Button-1>", self._on_video_click)

        slider_row = tk.Frame(left, bg=self.SURFACE)
        slider_row.pack(fill="x", padx=8, pady=(0,2))
        tk.Label(slider_row, text="⏱", bg=self.SURFACE, fg=self.TEXT_DIM,
                 font=("Menlo", 9)).pack(side="left")
        self.timeline_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(slider_row, from_=0, to=self.total_frames-1,
                                orient="horizontal", variable=self.timeline_var,
                                command=self._on_slider_move)
        self.slider.pack(side="left", fill="x", expand=True, padx=(4,0))

        ctrl = tk.Frame(left, bg=self.BG, pady=4)
        ctrl.pack(fill="x", padx=8)

        self.play_btn = tk.Button(
            ctrl, text="▶  Play",
            bg=self.ACCENT, fg=self.BTN_TEXT,
            activebackground="#1560AE", activeforeground=self.BTN_TEXT,
            font=("Menlo", 10, "bold"), relief="flat",
            padx=16, pady=6, cursor="hand2",
            command=self._toggle_play)
        self.play_btn.pack(side="left", padx=(0,6))

        self.reset_btn = tk.Button(
            ctrl, text="↺  Reset",
            bg=self.BTN_SURF, fg=self.BTN_TEXT,
            activebackground=self.BTN_HOVER, activeforeground=self.BTN_TEXT,
            font=("Menlo", 10), relief="flat",
            padx=14, pady=6, cursor="hand2",
            command=self._reset_ids)
        self.reset_btn.pack(side="left")

        self.export_btn = tk.Button(
            ctrl, text="Export CSV",
            bg=self.BTN_SURF, fg=self.BTN_TEXT,
            activebackground=self.BTN_HOVER, activeforeground=self.BTN_TEXT,
            font=("Menlo", 10), relief="flat",
            padx=14, pady=6, cursor="hand2",
            command=self._export_csv)
        self.export_btn.pack(side="left", padx=(6,0))

        self.frame_label = tk.Label(ctrl, text=f"0 / {self.total_frames-1}",
                                    bg=self.BG, fg=self.TEXT_DIM,
                                    font=("Menlo", 9))
        self.frame_label.pack(side="right")

        tk.Label(left, text="Space = Play/Pause   ·   R = Reset   ·   Q = Quit",
                 bg=self.SURFACE, fg=self.TEXT_DIM,
                 font=("Menlo", 7)).pack(pady=(2,6))

        # RIGHT
        right = tk.Frame(content, bg=self.SURFACE,
                         highlightbackground=self.BORDER, highlightthickness=1)
        right.pack(side="left", fill="both", expand=True, padx=(4,0))

        hdr = tk.Frame(right, bg=self.SURFACE)
        hdr.pack(fill="x", padx=10, pady=(6,0))
        tk.Label(hdr, text="JOINT ANGLE TIMELINE", bg=self.SURFACE,
                 fg=self.TEXT_DIM, font=("Menlo", 8, "bold")).pack(side="left")
        self.graph_status_lbl = tk.Label(hdr, text="", bg=self.SURFACE,
                                         fg=self.SUCCESS, font=("Menlo", 8))
        self.graph_status_lbl.pack(side="right")

        self.chart_frame = tk.Frame(right, bg=self.SURFACE)
        self.chart_frame.pack(fill="both", expand=True, padx=6, pady=4)

        self.angle_label = tk.Label(
            right,
            text="No player linked — click a player in the video to begin",
            bg=self.SURFACE, fg=self.TEXT_DIM,
            font=("Menlo", 8), wraplength=400)
        self.angle_label.pack(pady=(0,6))

        status_bar = tk.Frame(r, bg=self.STATUS_BG,
                              highlightbackground=self.BORDER,
                              highlightthickness=1, height=24)
        status_bar.pack(fill="x", side="bottom")
        self.status_label = tk.Label(status_bar, text="Ready",
                                     bg=self.STATUS_BG, fg=self.TEXT_DIM,
                                     font=("Menlo", 8))
        self.status_label.pack(side="left", padx=10)

        r.bind("<space>", lambda e: self._toggle_play())
        r.bind("r",       lambda e: self._reset_ids())
        r.bind("R",       lambda e: self._reset_ids())
        r.bind("q",       lambda e: self._quit())
        r.bind("Q",       lambda e: self._quit())
        r.protocol("WM_DELETE_WINDOW", self._quit)

    def _build_chart(self):
        fig_bg   = "#FFFFFF"
        ax_bg    = "#FAFAFA"
        grid_clr = "#DEE2E6"
        txt_clr  = "#495057"

        self.fig, self.ax = plt.subplots(figsize=(7, 4.5))
        self.fig.patch.set_facecolor(fig_bg)
        self.ax.set_facecolor(ax_bg)

        lw     = 0.9
        styles = ["-", "-", "--", "--"]

        self.chart_lines = [
            self.ax.plot([], [], color=c, lw=lw, linestyle=ls, label=lbl)[0]
            for c, ls, lbl in zip(self.CHART_COLORS, styles, self.CHART_LABELS)
        ]
        self.highlight_dot, = self.ax.plot(
            [], [], 'o', ms=7,
            markerfacecolor=self.ACCENT2,
            markeredgecolor="white", markeredgewidth=1.0, zorder=5)

        self.ax.set_xlabel("Frame", fontsize=9, color=txt_clr)
        self.ax.set_ylabel("Angle (°)", fontsize=9, color=txt_clr)
        self.ax.set_ylim(0, 180)
        self.ax.set_xlim(0, self.total_frames)
        self.ax.tick_params(colors=txt_clr, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(grid_clr)
        self.ax.grid(True, alpha=0.6, color=grid_clr, linewidth=0.5)
        self.ax.legend(loc="upper right", fontsize=8,
                       facecolor=fig_bg, edgecolor=grid_clr,
                       labelcolor=txt_clr, framealpha=0.9)
        self.ax.set_title(
            "Solid = Elbow (upper limb)   ·   Dashed = Knee (lower limb)",
            fontsize=7.5, color=self.TEXT_DIM, pad=4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_video_click(self, event):
        vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sx = vid_w / max(self.display_w, 1)
        sy = vid_h / max(self.display_h, 1)
        cx, cy = event.x * sx, event.y * sy

        for pid, (x1, y1, x2, y2) in self.players_at_frame:
            if x1 < cx < x2 and y1 < cy < y2:
                if self._was_lost and self.history_ids:
                    self.linked_ids = {pid}
                    self.history_ids.add(pid)
                else:
                    self.linked_ids = {pid}
                    self.history_ids = {pid}
                self._was_lost = False
                self.paused    = False
                self.play_btn.config(text="⏸  Pause")
                self.update_full_graph()
                self._schedule_playback()
                self._update_status(
                    f"Linked ID {pid}  ·  Player identity: "
                    f"{sorted(self.history_ids)}")
                print(f"[CLICK] ID={pid} | history={sorted(self.history_ids)}")
                break

    def _on_slider_move(self, value):
        new_frame = min(int(float(value)), len(self.cache)-1)
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            self._render_frame()

    def _toggle_play(self):
        self.paused = not self.paused
        if self.paused:
            self.play_btn.config(text="▶  Play")
        else:
            self.play_btn.config(text="⏸  Pause")
            self._schedule_playback()

    def _schedule_playback(self):
        if self.paused:
            return
        safe_max = len(self.cache) - 1
        if self.current_frame < safe_max:
            self.current_frame += 1
            chart_every = max(1, int(round(self.fps / 6)))
            update_chart = (self.current_frame % chart_every == 0)
            self._render_frame(update_chart=update_chart, seek=False)
            delay_ms = max(1, int(1000 / self.fps))
            self._after_id = self.root.after(delay_ms, self._schedule_playback)
        else:
            self.paused = True
            self.play_btn.config(text="▶  Play")
            self._update_status("End of video")

    def _reset_ids(self):
        """
        v2.4 Reset — frame-local deletion.

        Key change from v2.3
        ────────────────────
        target_ids = _get_frame_history_ids()
            = history_ids ∩ {ids visible in current frame}

        Case A (_was_lost): target_ids is empty (no history ID in this frame).
            → keep everything, just clear linked_ids for re-linking.

        Case B (not lost):  target_ids contains the ID(s) the user is
            looking at right now — delete only those from history_ids.
            linked_ids is also cleaned up for consistency, but it is
            NOT the primary deletion target.
        """
        target_ids = self._get_frame_history_ids()

        if self._was_lost:
            # Case A: no history ID visible → true lost, preserve all data
            lost = sorted(self.linked_ids)
            self.linked_ids.clear()
            self.paused = True
            self.play_btn.config(text="▶  Play")
            self._update_status(
                f"Tracking lost ({lost}).  "
                f"History kept: {sorted(self.history_ids)}.  "
                f"Click player to resume.")
            self.title_hint.config(
                text="Re-click the player to continue tracking →",
                fg=self.TEXT_DIM)
            print(f"[RESET-LOST] linked={lost} cleared; "
                  f"history={sorted(self.history_ids)} preserved")

        else:
            # Case B: delete only the IDs visible in this specific frame
            if not target_ids:
                # Shouldn't normally reach here, but guard anyway
                self._update_status("Nothing to reset at this position.")
                return

            removed = sorted(target_ids)
            self.history_ids -= target_ids          # ← frame-local deletion
            self.linked_ids  -= target_ids          # keep linked_ids consistent

            self.paused = True
            self.play_btn.config(text="▶  Play")
            self.update_full_graph()

            remaining = sorted(self.history_ids)
            if remaining:
                self.angle_label.config(
                    text=f"Removed ID(s) {removed}.  "
                         f"Remaining IDs: {remaining}",
                    fg=self.ACCENT2)
                self._update_status(
                    f"Removed ID(s) {removed} at frame {self.current_frame}.  "
                    f"Kept: {remaining}")
            else:
                self.angle_label.config(
                    text="No player linked — click a player to begin",
                    fg=self.TEXT_DIM)
                self._update_status("All IDs cleared.")

            print(f"[RESET-FRAME] removed={removed}; "
                  f"history={sorted(self.history_ids)}")

        self._render_frame()

    def _update_status(self, msg):
        self.status_label.config(text=msg)

    def _quit(self):
        if self._after_id:
            self.root.after_cancel(self._after_id)
        self.cap.release()
        plt.close(self.fig)
        delete_cache()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    picker = tk.Tk()
    picker.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Basketball Video",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")]
    )
    picker.destroy()

    if not video_path:
        print("No video selected — exiting.")
        exit()

    cache, id_angle_data, total_frames = load_or_build_cache(video_path)

    root = tk.Tk()
    app  = BasketballAnalyzerApp(root, video_path,
                                 cache, id_angle_data, total_frames)
    root.mainloop()

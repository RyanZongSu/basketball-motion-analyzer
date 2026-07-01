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
# New main root script for basketball analysis

import os
import pickle
import csv
import hashlib
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def _cache_paths_for_video(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    digest = hashlib.md5(os.path.abspath(video_path).encode("utf-8")).hexdigest()[:10]
    cache_file = f"{base}_{digest}_{CACHE_FILE}"
    angle_file = f"{base}_{digest}_{ANGLE_FILE}"
    return cache_file, angle_file


def load_or_build_cache(video_path, cache_file=None, angle_file=None):
    if cache_file is None or angle_file is None:
        cache_file, angle_file = _cache_paths_for_video(video_path)
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

    CHART_COLORS = ["#1971C2", "#0CA678"]
    CHART_LABELS = ["Shooting Elbow", "Power Knee"]
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

    def __init__(self, root, video_path=None, cache=None, id_angle_data=None,
                 total_frames=0):
        self.root          = root
        self.video_path    = video_path
        self.cache         = cache or []
        self.id_angle_data = id_angle_data or {}
        self.total_frames  = total_frames

        self.current_frame    = 0
        self.paused           = True
        self.players_at_frame = []
        self._after_id        = None

        self.linked_ids  = set()
        self.history_ids = set()
        self._was_lost   = False
        self.id_analysis_rows = {}
        self._analysis_cache_key = None
        self._analysis_rows_cache = None
        self._analysis_rows_by_frame = {}
        self._current_chart_shot_id = None
        self._selected_shot_id = None
        self._manual_shot_sides = {}
        self._pending_shot_sides = {}
        self._manual_recalc_version = 0
        self.release_line = None
        self.playhead_line = None
        self.ax_time = None
        self.no_data_text = None
        self.highlight_dots = []
        self.event_markers = []
        self.chart_tabs_frame = None

        self.cap = cv2.VideoCapture(video_path) if video_path else None
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30.0
        self.fps = self.fps or 30.0

        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 0
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else 0
        self._compute_display_size(self.video_w, self.video_h)
        self._last_video_label_size = None
        if self.video_path:
            self._precompute_all_id_analysis()

        self._build_ui()
        self._build_chart()
        self._render_frame()

    # ── Adaptive sizing ───────────────────────────────────────────────────────

    def _compute_display_size(self, vid_w, vid_h, panel_w=700, panel_h=560):
        panel_w = max(160, int(panel_w))
        panel_h = max(120, int(panel_h))
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
            hint = (
                "Click Open Video to choose a video"
                if not self.video_path
                else "Click a player in the video to start tracking"
            )
            self.title_hint.config(text=hint, fg=self.TEXT_DIM)
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

        x = np.arange(self.total_frames)
        shooting_elbow = np.full(self.total_frames, np.nan)
        power_knee = np.full(self.total_frames, np.nan)

        rows = self._get_analysis_rows(ids_to_show)
        shot_rows = [r for r in rows if r.get("shot_id") != ""]
        available_shots = sorted({r["shot_id"] for r in shot_rows})
        if self._selected_shot_id in available_shots:
            chart_shot_id = self._selected_shot_id
        else:
            chart_shot_id = available_shots[0] if available_shots else None
            self._selected_shot_id = chart_shot_id
        self._current_chart_shot_id = chart_shot_id
        for row in rows:
            if row.get("shot_id") != chart_shot_id:
                continue
            if not row.get("shooting_hand") and not row.get("power_leg"):
                continue
            frame = row.get("frame")
            if frame is None or frame < 0 or frame >= self.total_frames:
                continue
            shooting_hand = row.get("shooting_hand", "")
            power_leg = row.get("power_leg", "")
            if shooting_hand == "left":
                shooting_elbow[frame] = row.get("left_elbow_angle", np.nan)
            elif shooting_hand == "right":
                shooting_elbow[frame] = row.get("right_elbow_angle", np.nan)
            if power_leg == "left":
                power_knee[frame] = row.get("left_knee_angle", np.nan)
            elif power_leg == "right":
                power_knee[frame] = row.get("right_knee_angle", np.nan)

        self.chart_lines[0].set_data(x, shooting_elbow)
        self.chart_lines[1].set_data(x, power_knee)
        for line in (self.release_line, self.playhead_line):
            if line is not None:
                line.set_visible(False)
        for dot in self.highlight_dots:
            dot.set_data([], [])
        for marker, text in self.event_markers:
            marker.set_data([], [])
            text.set_visible(False)

        if shot_rows:
            self._set_chart_data_mode()
            first_shot_rows = [
                r for r in shot_rows if r.get("shot_id") == chart_shot_id
            ]
            shot_start = min(r["frame"] for r in first_shot_rows)
            shot_end = max(r["frame"] for r in first_shot_rows)
            pad = max(3, int(round(0.15 * self.fps)))
            x0 = max(0, shot_start - pad)
            x1 = min(self.total_frames - 1, shot_end + pad)
            self.ax.set_xlim(x0, x1)
            if self.ax_time is not None:
                self.ax_time.set_xlim(x0 / self.fps, x1 / self.fps)
            release_rows = [
                r for r in first_shot_rows if r.get("is_release_frame")
            ]
            if release_rows and self.release_line is not None:
                release_x = release_rows[0]["frame"]
                self.release_line.set_xdata([release_x, release_x])
                self.release_line.set_visible(True)
            self._draw_event_markers(first_shot_rows)
            visible_values = np.concatenate([
                shooting_elbow[~np.isnan(shooting_elbow)],
                power_knee[~np.isnan(power_knee)]
            ])
            if len(visible_values):
                ymax = min(205, max(185, float(np.nanmax(visible_values)) + 24))
                self.ax.set_ylim(0, ymax)
            self.graph_status_lbl.config(
                text=(f"IDs: {sorted(ids_to_show)}  ·  "
                      f"Shot {chart_shot_id}: frames {shot_start}-{shot_end}"))
        else:
            self._current_chart_shot_id = None
            self.ax.set_ylim(0, 190)
            self.ax.set_xlim(0, 1)
            if self.ax_time is not None:
                self.ax_time.set_xlim(0, 1 / self.fps)
            self.graph_status_lbl.config(
                text="")
            self._set_chart_blank()

        self._refresh_chart_tabs(available_shots)
        self.canvas.draw_idle()

    def _set_chart_blank(self, message="No Data Available"):
        if self.no_data_text is not None:
            self.no_data_text.set_text(message)
            self.no_data_text.set_visible(True)
        self.ax.title.set_visible(False)
        self.ax.set_axis_off()
        if self.ax_time is not None:
            self.ax_time.set_axis_off()
        legend = self.ax.get_legend()
        if legend is not None:
            legend.set_visible(False)

    def _set_chart_data_mode(self):
        self.ax.set_axis_on()
        self.ax.title.set_visible(True)
        if self.ax_time is not None:
            self.ax_time.set_axis_on()
        if self.no_data_text is not None:
            self.no_data_text.set_visible(False)
        legend = self.ax.get_legend()
        if legend is not None:
            legend.set_visible(True)

    def _refresh_chart_tabs(self, shot_ids=None):
        if self.chart_tabs_frame is None:
            return
        for child in self.chart_tabs_frame.winfo_children():
            child.destroy()
        if not shot_ids:
            return
        for shot_id in shot_ids:
            active = shot_id == self._current_chart_shot_id
            bg = self.ACCENT_LT if active else self.SURFACE
            fg = self.ACCENT if active else self.TEXT_DIM
            underline = "  " if active else ""
            tab = tk.Label(
                self.chart_tabs_frame,
                text=f"{underline}Shot {shot_id}{underline}",
                bg=bg, fg=fg, font=("Menlo", 12, "bold" if active else "normal"),
                padx=12, pady=5, cursor="hand2",
                highlightbackground=self.ACCENT if active else self.BORDER,
                highlightthickness=1)
            tab.pack(side="left", padx=(0,4), pady=(0,2))
            tab.bind("<Button-1>", lambda _e, sid=shot_id: self._select_shot(sid))

    def update_highlight(self):
        if not self.history_ids:
            self.update_full_graph(set())
            if self.playhead_line is not None:
                self.playhead_line.set_visible(False)
            for dot in self.highlight_dots:
                dot.set_data([], [])
            self.canvas.draw_idle()
            return

        row = self._get_analysis_row_for_frame(
            self.current_frame, preferred_ids=self._get_frame_history_ids())
        elbow_val = np.nan
        knee_val = np.nan
        show_marker = bool(
            row and row.get("shot_id") == self._current_chart_shot_id
        )
        if row:
            shooting_hand = row.get("shooting_hand", "")
            if shooting_hand == "left":
                elbow_val = row.get("left_elbow_angle", np.nan)
            elif shooting_hand == "right":
                elbow_val = row.get("right_elbow_angle", np.nan)
            power_leg = row.get("power_leg", "")
            if power_leg == "left":
                knee_val = row.get("left_knee_angle", np.nan)
            elif power_leg == "right":
                knee_val = row.get("right_knee_angle", np.nan)

        current_x = self.current_frame
        if self.playhead_line is not None:
            self.playhead_line.set_xdata([current_x, current_x])
            self.playhead_line.set_visible(show_marker)
        values = [elbow_val, knee_val]
        for dot, y_val in zip(self.highlight_dots, values):
            if show_marker and not np.isnan(y_val):
                dot.set_data([current_x], [y_val])
            else:
                dot.set_data([], [])
        self.canvas.draw_idle()

        if row:
            pid_show = row.get("player_id")
            shooting_hand = row.get("shooting_hand", "")
            power_leg = row.get("power_leg", "")
            def _f(v): return f"{int(round(v))}°" if not np.isnan(v) else "--"
            self.angle_label.config(
                text=(f"ID {pid_show}  ·  "
                      f"{shooting_hand.title() or 'Shooting'} Elbow: "
                      f"{_f(elbow_val)}   "
                      f"{power_leg.title() or 'Power'} Knee: {_f(knee_val)}"),
                fg=self.ACCENT)

    def _ensure_event_markers(self, count):
        while len(self.event_markers) < count:
            marker, = self.ax.plot(
                [], [], "o", ms=5, markerfacecolor="#FFFFFF",
                markeredgecolor="#343A40", markeredgewidth=1.0, zorder=6)
            text = self.ax.text(
                0, 0, "", fontsize=9, color="#343A40",
                ha="center", va="bottom", visible=False,
                bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF",
                          ec="#CED4DA", alpha=0.92))
            self.event_markers.append((marker, text))

    def _draw_event_markers(self, shot_rows):
        event_specs = [
            ("is_knee_lowest_frame", "Knee Min", "knee"),
            ("is_elbow_lowest_frame", "Elbow Min", "elbow"),
            ("is_elbow_max_extension_frame", "Elbow Max", "elbow"),
            ("is_knee_max_extension_frame", "Knee Max", "knee"),
        ]
        points = []
        for flag, label, series in event_specs:
            matches = [r for r in shot_rows if r.get(flag)]
            if not matches:
                continue
            row = matches[0]
            shooting_hand = row.get("shooting_hand", "")
            power_leg = row.get("power_leg", "")
            if series == "elbow":
                y_val = (
                    row.get("left_elbow_angle", np.nan)
                    if shooting_hand == "left" else
                    row.get("right_elbow_angle", np.nan)
                    if shooting_hand == "right" else np.nan
                )
            else:
                y_val = (
                    row.get("left_knee_angle", np.nan)
                    if power_leg == "left" else
                    row.get("right_knee_angle", np.nan)
                    if power_leg == "right" else np.nan
                )
            if np.isnan(y_val):
                continue
            points.append((row["frame"], y_val, label))

        self._ensure_event_markers(len(points))
        for idx, (marker, text) in enumerate(self.event_markers):
            if idx >= len(points):
                marker.set_data([], [])
                text.set_visible(False)
                continue
            x_val, y_val, label = points[idx]
            marker.set_data([x_val], [y_val])
            y_top = self.ax.get_ylim()[1]
            text.set_position((x_val, min(y_top - 10, y_val + 8)))
            text.set_text(label)
            text.set_visible(True)

    def _invalidate_analysis_cache(self):
        self._analysis_cache_key = None
        self._analysis_rows_cache = None
        self._analysis_rows_by_frame = {}

    def _get_analysis_rows(self, ids_to_show=None):
        if ids_to_show is None:
            ids_to_show = self.history_ids
        cache_key = (tuple(sorted(ids_to_show)), self._manual_recalc_version)
        if self._analysis_cache_key == cache_key and self._analysis_rows_cache is not None:
            return self._analysis_rows_cache

        rows = []
        for pid in sorted(ids_to_show):
            rows.extend(self.id_analysis_rows.get(pid, []))
        rows = self._renumber_merged_rows(rows) if rows else []

        self._analysis_cache_key = cache_key
        self._analysis_rows_cache = rows
        self._analysis_rows_by_frame = {}
        for row in rows:
            if row.get("shot_id") == "":
                continue
            self._analysis_rows_by_frame.setdefault(row["frame"], []).append(row)
        return rows

    def _get_analysis_row_for_frame(self, frame_idx, preferred_ids=None):
        self._get_analysis_rows()
        matches = self._analysis_rows_by_frame.get(frame_idx, [])
        if not matches:
            return None
        if preferred_ids:
            for row in matches:
                if row.get("player_id") in preferred_ids:
                    return row
        return matches[0]

    def _refresh_summary_table(self):
        if not hasattr(self, "summary_container"):
            return
        for child in self.summary_container.winfo_children():
            child.destroy()
        if not self.history_ids:
            self._update_summary_scroll_region()
            return

        rows = self._get_analysis_rows()
        summary_rows = self._build_summary_rows(rows)
        shot_ids = sorted({r["shot_id"] for r in summary_rows})
        if self._selected_shot_id not in shot_ids:
            self._selected_shot_id = shot_ids[0] if shot_ids else None
        label_map = {
            "knee_lowest": "Knee Min",
            "elbow_lowest": "Elbow Min",
            "release_proxy": "Release",
            "elbow_max_extension": "Elbow Max",
            "knee_max_extension": "Knee Max",
            "observed_knee_max_extension_unconfirmed": "Knee Max*",
        }

        def _fmt_angle(value):
            try:
                if np.isnan(value):
                    return ""
            except TypeError:
                pass
            return f"{float(value):.1f}"

        for shot_id in shot_ids:
            shot_summary = [r for r in summary_rows if r["shot_id"] == shot_id]
            if not shot_summary:
                continue
            shot_data = [r for r in rows if r.get("shot_id") == shot_id]
            shot_start = min((r["frame"] for r in shot_data), default="")
            shot_end = max((r["frame"] for r in shot_data), default="")
            hand = shot_summary[0].get("shooting_hand", "")
            leg = shot_summary[0].get("power_leg", "")
            pending = self._pending_shot_sides.get(shot_id, {})
            display_hand = pending.get("shooting_hand", hand)
            display_leg = pending.get("power_leg", leg)
            pending_note = "  Pending" if pending else ""
            selected = shot_id == self._selected_shot_id
            border = self.ACCENT if selected else self.BORDER
            card_bg = self.ACCENT_LT if selected else "#FFFFFF"

            card = tk.Frame(
                self.summary_container, bg=card_bg,
                highlightbackground=border, highlightthickness=2 if selected else 1)
            card.pack(fill="x", pady=(0,5))
            card.bind("<Button-1>", lambda _e, sid=shot_id: self._select_shot(sid))

            card_body = tk.Frame(card, bg=card_bg)
            card_body.pack(fill="x", padx=7, pady=6)
            card_body.bind("<Button-1>", lambda _e, sid=shot_id: self._select_shot(sid))

            side_panel = tk.Frame(card_body, bg=card_bg)
            side_panel.pack(side="left", fill="y", padx=(0,6))
            info_lines = [
                f"Shot #{shot_id}",
                f"Frames: {shot_start}-{shot_end}",
                f"Hand: {display_hand or '--'}",
                f"Leg: {display_leg or '--'}",
            ]
            for line in info_lines:
                tk.Label(
                    side_panel, text=line, bg=card_bg, fg=self.TEXT,
                    font=("Menlo", 11, "bold"), width=18,
                    padx=4, pady=2, anchor="w"
                ).pack(anchor="w", fill="x")

            change_box = tk.Frame(side_panel, bg=card_bg)
            change_box.pack(anchor="w", pady=(5,2))
            hand_toggle = tk.Button(
                change_box, text="Change🤚", bg=self.BTN_SURF, fg=self.BTN_TEXT,
                activebackground=self.BTN_HOVER, activeforeground=self.BTN_TEXT,
                font=("Menlo", 10), relief="flat", padx=6, pady=2,
                cursor="hand2",
                command=lambda sid=shot_id: self._toggle_pending_shot_side(sid, "hand"))
            hand_toggle.pack(side="left", padx=(0,4))
            leg_toggle = tk.Button(
                change_box, text="Change🦶", bg=self.BTN_SURF, fg=self.BTN_TEXT,
                activebackground=self.BTN_HOVER, activeforeground=self.BTN_TEXT,
                font=("Menlo", 10), relief="flat", padx=6, pady=2,
                cursor="hand2",
                command=lambda sid=shot_id: self._toggle_pending_shot_side(sid, "leg"))
            leg_toggle.pack(side="left")

            apply_box = tk.Frame(side_panel, bg=card_bg)
            apply_box.pack(anchor="w", pady=(2,0))
            apply_toggle = tk.Button(
                apply_box, text="Apply", bg=self.BTN_SURF,
                fg=self.SUCCESS if pending else self.TEXT_DIM,
                activebackground=self.BTN_HOVER, activeforeground=self.BTN_TEXT,
                font=("Menlo", 10, "bold"), relief="flat", padx=8, pady=2,
                cursor="hand2",
                command=lambda sid=shot_id: self._apply_pending_shot_sides(sid))
            apply_toggle.pack(side="left")
            tk.Label(
                side_panel, text="pending" if pending else " ",
                bg=card_bg, fg=self.ACCENT2,
                font=("Menlo", 11, "bold"), padx=4, pady=2
            ).pack(anchor="w")

            table = tk.Frame(card_body, bg="#FFFFFF")
            table.pack(side="left", fill="x", expand=True)
            cols = ("Event", "Frame", "Time", "Elbow", "Knee", "Hip")
            widths = [16, 8, 8, 8, 8, 8]
            for col_idx, (col, width) in enumerate(zip(cols, widths)):
                tk.Label(
                    table, text=col, bg="#F0F0F0", fg=self.TEXT,
                    font=("Menlo", 11, "bold"), width=width,
                    padx=3, pady=3, anchor="center",
                    highlightbackground=self.BORDER, highlightthickness=1
                ).grid(row=0, column=col_idx, sticky="nsew")
                table.grid_columnconfigure(col_idx, weight=1)

            for row_idx, row in enumerate(shot_summary, start=1):
                label = row.get("event_label", "")
                event_key = label.split("_", 2)[2] if label.count("_") >= 2 else label
                event_name = label_map.get(event_key, event_key)
                elbow_display = ""
                knee_display = ""
                if event_key in (
                    "elbow_lowest", "release_proxy", "elbow_max_extension"
                ):
                    elbow_display = _fmt_angle(
                        row.get("shooting_elbow_angle_deg", np.nan))
                if event_key in (
                    "knee_lowest", "release_proxy", "knee_max_extension",
                    "observed_knee_max_extension_unconfirmed"
                ):
                    knee_display = _fmt_angle(
                        row.get("power_knee_angle_deg", np.nan))
                values = (
                    event_name,
                    row.get("frame_number", ""),
                    f"{float(row.get('time_sec', 0.0)):.2f}",
                    elbow_display,
                    knee_display,
                    _fmt_angle(row.get("hip_height_norm", np.nan)),
                )
                for col_idx, (value, width) in enumerate(zip(values, widths)):
                    bg = "#E6F0FF" if col_idx == 0 else "#FFFFFF"
                    weight = "bold" if col_idx == 0 else "normal"
                    tk.Label(
                        table, text=value, bg=bg, fg=self.TEXT,
                        font=("Menlo", 11, weight), width=width,
                        padx=3, pady=2, anchor="center",
                        highlightbackground=self.BORDER, highlightthickness=1
                    ).grid(row=row_idx, column=col_idx, sticky="nsew")
        self._update_summary_scroll_region()

    def _update_summary_scroll_region(self):
        if not hasattr(self, "summary_canvas"):
            return
        self.summary_container.update_idletasks()
        bbox = self.summary_canvas.bbox("all")
        if not bbox:
            self.summary_canvas.configure(height=40, scrollregion=(0, 0, 0, 0))
            if hasattr(self, "summary_scroll"):
                self.summary_scroll.grid_remove()
            if hasattr(self, "summary_hscroll"):
                self.summary_hscroll.grid_remove()
            return
        content_w = bbox[2] - bbox[0]
        content_h = bbox[3] - bbox[1]
        max_h = max(120, self.summary_canvas.master.winfo_height() - 4)
        target_h = min(content_h + 4, max_h)
        self.summary_canvas.configure(height=target_h, scrollregion=bbox)
        view_w = max(1, self.summary_canvas.winfo_width())
        if hasattr(self, "summary_hscroll"):
            if content_w > view_w + 2:
                if not self.summary_hscroll.winfo_ismapped():
                    self.summary_hscroll.grid(row=1, column=0, sticky="ew")
            else:
                self.summary_hscroll.grid_remove()
        if hasattr(self, "summary_scroll"):
            if content_h > max_h:
                if not self.summary_scroll.winfo_ismapped():
                    self.summary_scroll.grid(row=0, column=1, sticky="ns")
            else:
                self.summary_scroll.grid_remove()

    @staticmethod
    def _opposite_side(side):
        return "right" if side == "left" else "left"

    def _select_shot(self, shot_id):
        self._selected_shot_id = shot_id
        self.update_full_graph()
        self._refresh_summary_table()
        self.update_highlight()

    def _toggle_pending_shot_side(self, shot_id, side_kind):
        rows = self._get_analysis_rows()
        shot_rows = [r for r in rows if r.get("shot_id") == shot_id]
        if not shot_rows:
            return
        current_hand = shot_rows[0].get("shooting_hand", "")
        current_leg = shot_rows[0].get("power_leg", "")
        pending = dict(self._pending_shot_sides.get(shot_id, {}))
        if side_kind == "hand" and current_hand in ("left", "right"):
            base = pending.get("shooting_hand", current_hand)
            pending["shooting_hand"] = self._opposite_side(base)
        elif side_kind == "leg" and current_leg in ("left", "right"):
            base = pending.get("power_leg", current_leg)
            pending["power_leg"] = self._opposite_side(base)
        else:
            return
        self._pending_shot_sides[shot_id] = pending
        self._selected_shot_id = shot_id
        self._refresh_summary_table()

    def _apply_pending_shot_sides(self, shot_id):
        rows = self._get_analysis_rows()
        shot_rows = [r for r in rows if r.get("shot_id") == shot_id]
        if not shot_rows:
            return
        pending = dict(self._pending_shot_sides.get(shot_id, {}))
        current_hand = shot_rows[0].get("shooting_hand", "")
        current_leg = shot_rows[0].get("power_leg", "")
        next_hand = pending.get("shooting_hand", current_hand)
        next_leg = pending.get("power_leg", current_leg)
        auto_hand, auto_leg = self._get_auto_shot_sides(shot_id)
        if next_hand == auto_hand and next_leg == auto_leg:
            self._manual_shot_sides.pop(shot_id, None)
        else:
            self._manual_shot_sides[shot_id] = {
                "shooting_hand": next_hand,
                "power_leg": next_leg,
            }
        self._pending_shot_sides.pop(shot_id, None)
        self._manual_recalc_version += 1
        self._invalidate_analysis_cache()
        self._selected_shot_id = shot_id
        self.update_full_graph()
        self._refresh_summary_table()
        self.update_highlight()

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

    def _get_id_sample_at_frame(self, target_id, frame_idx):
        data = self.cache[frame_idx]
        ids = data.get("ids", [])
        kps = data.get("kps", [])

        for i, pid in enumerate(ids):
            if pid != target_id or i >= len(kps):
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

    def _precompute_all_id_analysis(self):
        self.id_analysis_rows = {}
        all_ids = sorted(self.id_angle_data.keys())
        if not all_ids:
            return
        print(f"⏳ Pre-computing shot windows for {len(all_ids)} IDs…")
        for pid in all_ids:
            rows = self._build_rows_for_id(pid)
            self.id_analysis_rows[pid] = rows
            shot_count = len({r["shot_id"] for r in rows if r["shot_id"] != ""})
            if shot_count:
                print(f"  ID {pid}: {shot_count} shot window(s)")
        print("✅ Shot-window pre-computation finished")

    def _renumber_merged_rows(self, rows, apply_manual_overrides=True):
        merged = [dict(row) for row in rows]
        shot_first_frames = {}
        for row in merged:
            sid = row.get("shot_id")
            if sid == "":
                continue
            key = (row.get("player_id"), sid)
            frame = row.get("frame", self.total_frames)
            if key not in shot_first_frames or frame < shot_first_frames[key]:
                shot_first_frames[key] = frame
        shot_keys = sorted(shot_first_frames, key=lambda key: shot_first_frames[key])
        shot_map = {key: idx for idx, key in enumerate(shot_keys, start=1)}

        for row in merged:
            sid = row.get("shot_id")
            if sid == "":
                continue
            old_shot_id = sid
            new_shot_id = shot_map[(row.get("player_id"), old_shot_id)]
            row["shot_id"] = new_shot_id
            if apply_manual_overrides:
                override = self._manual_shot_sides.get(new_shot_id, {})
                if override.get("shooting_hand"):
                    row["shooting_hand"] = override["shooting_hand"]
                if override.get("power_leg"):
                    row["power_leg"] = override["power_leg"]
            label = row.get("event_label", "")
            if label:
                row["event_label"] = label.replace(
                    f"shot_{old_shot_id}_", f"shot_{new_shot_id}_")
        if apply_manual_overrides:
            for shot_id, override in self._manual_shot_sides.items():
                if override:
                    self._recompute_overridden_shot_events(merged, shot_id)
        merged.sort(key=lambda r: (r["frame"], r["player_id"]))
        return merged

    def _get_auto_shot_sides(self, shot_id):
        rows = []
        for pid in sorted(self.history_ids):
            rows.extend(self.id_analysis_rows.get(pid, []))
        base_rows = self._renumber_merged_rows(rows, apply_manual_overrides=False)
        shot_rows = [r for r in base_rows if r.get("shot_id") == shot_id]
        if not shot_rows:
            return "", ""
        return (
            shot_rows[0].get("shooting_hand", ""),
            shot_rows[0].get("power_leg", ""),
        )

    def _recompute_overridden_shot_events(self, rows, shot_id):
        shot_rows = [r for r in rows if r.get("shot_id") == shot_id]
        if not shot_rows:
            return

        event_flags = [
            "is_knee_lowest_frame", "is_elbow_lowest_frame",
            "is_elbow_max_extension_frame", "is_knee_max_extension_frame",
            "is_observed_knee_max_extension_frame", "is_max_extension_frame",
        ]
        shot_prefix = f"shot_{shot_id}_"
        analysis_frames = [
            frame
            for row in shot_rows
            for frame in (
                row.get("analysis_start_frame"),
                row.get("analysis_end_frame"),
            )
            if frame != ""
        ]
        if analysis_frames:
            analysis_start = int(min(analysis_frames))
            analysis_end = int(max(analysis_frames))
        else:
            analysis_start = min(r["frame"] for r in shot_rows)
            analysis_end = max(r["frame"] for r in shot_rows)

        for row in rows:
            if not (analysis_start <= row["frame"] <= analysis_end):
                continue
            labels = [
                label for label in str(row.get("event_label", "")).split("|")
                if label
            ]
            if row.get("shot_id") not in ("", shot_id) and not any(
                label.startswith(shot_prefix) for label in labels
            ):
                continue
            for flag in event_flags:
                row[flag] = False
            row["knee_max_extension_confirmed"] = ""
            row["event_label"] = "|".join(
                label for label in labels
                if not label.startswith(shot_prefix)
            )

        shooting_hand = shot_rows[0].get("shooting_hand", "")
        power_leg = shot_rows[0].get("power_leg", "")
        release_rows = [r for r in shot_rows if r.get("is_release_frame")]
        release_row = release_rows[0] if release_rows else None
        if release_row is not None:
            release_row["event_label"] = self._append_event_label(
                release_row.get("event_label", ""), f"{shot_prefix}release_proxy")

        release_frame = (
            release_row["frame"] if release_row is not None
            else max(r["frame"] for r in shot_rows)
        )
        window_rows = [
            r for r in rows
            if analysis_start <= r["frame"] <= analysis_end and
            (r.get("shot_id") in ("", shot_id))
        ]

        def _side_angle(row, side, joint):
            if joint == "elbow":
                return row.get(f"{side}_elbow_angle", np.nan)
            return row.get(f"{side}_knee_angle", np.nan)

        def _mark(row, flag, label, max_ext=False):
            if row is None:
                return
            row[flag] = True
            if max_ext:
                row["is_max_extension_frame"] = True
            row["event_label"] = self._append_event_label(
                row.get("event_label", ""), f"{shot_prefix}{label}")

        knee_candidates = [
            r for r in window_rows
            if r["frame"] < release_frame and power_leg in ("left", "right") and
            not np.isnan(_side_angle(r, power_leg, "knee"))
        ]
        knee_low_row = (
            min(knee_candidates, key=lambda r: _side_angle(r, power_leg, "knee"))
            if knee_candidates else None
        )
        _mark(knee_low_row, "is_knee_lowest_frame", "knee_lowest")

        elbow_candidates = [
            r for r in window_rows
            if shooting_hand in ("left", "right") and
            not np.isnan(_side_angle(r, shooting_hand, "elbow")) and
            r["frame"] < release_frame
        ]
        if knee_low_row is not None:
            after_knee = [r for r in elbow_candidates if r["frame"] > knee_low_row["frame"]]
            if after_knee:
                elbow_candidates = after_knee
        elbow_low_row = (
            min(elbow_candidates, key=lambda r: _side_angle(r, shooting_hand, "elbow"))
            if elbow_candidates else None
        )
        _mark(elbow_low_row, "is_elbow_lowest_frame", "elbow_lowest")

        elbow_ext_candidates = [
            r for r in window_rows
            if shooting_hand in ("left", "right") and
            not np.isnan(_side_angle(r, shooting_hand, "elbow")) and
            r["frame"] > release_frame
        ]
        near = [
            r for r in elbow_ext_candidates
            if r["frame"] <= release_frame + max(2, int(round(0.15 * self.fps)))
        ]
        if near:
            elbow_ext_candidates = near
        elbow_max_row = (
            max(elbow_ext_candidates,
                key=lambda r: _side_angle(r, shooting_hand, "elbow"))
            if elbow_ext_candidates else None
        )
        _mark(elbow_max_row, "is_elbow_max_extension_frame",
              "elbow_max_extension", max_ext=True)

        knee_ext_candidates = [
            r for r in window_rows
            if power_leg in ("left", "right") and
            not np.isnan(_side_angle(r, power_leg, "knee")) and
            r["frame"] > release_frame
        ]
        observed_knee_max_row = (
            max(knee_ext_candidates, key=lambda r: _side_angle(r, power_leg, "knee"))
            if knee_ext_candidates else None
        )
        knee_max_confirmed = False
        if observed_knee_max_row is not None:
            last_frame = max(r["frame"] for r in knee_ext_candidates)
            enough_after_release = len(knee_ext_candidates) >= max(
                3, int(round(0.12 * self.fps)))
            reaches_window_end = last_frame >= analysis_end - 1
            knee_max_confirmed = bool(enough_after_release and reaches_window_end)
            observed_knee_max_row["is_observed_knee_max_extension_frame"] = True
            observed_knee_max_row["knee_max_extension_confirmed"] = int(
                knee_max_confirmed)
            if not knee_max_confirmed:
                observed_knee_max_row["event_label"] = self._append_event_label(
                    observed_knee_max_row.get("event_label", ""),
                    f"{shot_prefix}observed_knee_max_extension_unconfirmed")
        knee_max_row = (
            observed_knee_max_row if knee_max_confirmed else None
        )
        _mark(knee_max_row, "is_knee_max_extension_frame",
              "knee_max_extension", max_ext=True)

    @staticmethod
    def _append_event_label(current, label):
        labels = [part for part in str(current).split("|") if part]
        if label not in labels:
            labels.append(label)
        return "|".join(labels)

    def _build_rows_for_id(self, target_id):
        rows = []
        for frame_idx in range(self.total_frames):
            pid, kp, angles = self._get_id_sample_at_frame(target_id, frame_idx)
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

    def _build_export_rows(self):
        rows = []
        for pid in sorted(self.history_ids):
            rows.extend(self.id_analysis_rows.get(pid, []))
        return self._renumber_merged_rows(rows)

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

            label_start = min(start, analysis_start)
            label_end = max(end, analysis_end)
            for row in rows[label_start:label_end + 1]:
                if row["shot_id"] in ("", shot_id):
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
        if not self.video_path:
            self._update_status("Open a video before exporting CSV.")
            return
        if not self.history_ids:
            self._update_status("Select a player before exporting CSV.")
            return

        self.paused = True
        self.play_btn.config(text="Play")
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
        if not self.video_path or self.cap is None or not self.cache:
            self._render_empty_video()
            if update_chart:
                self.update_full_graph(set())
            self._update_lost_status()
            return

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
        current_history_ids = self._get_frame_history_ids()
        shot_row = (
            self._get_analysis_row_for_frame(
                safe_frame, preferred_ids=current_history_ids)
            if self.history_ids else None
        )
        shot_player_id = shot_row.get("player_id") if shot_row else None

        for i, pid in enumerate(ids):
            if i >= len(boxes) or i >= len(kps):
                continue
            x1, y1, x2, y2 = map(int, boxes[i])
            self.players_at_frame.append((pid, (x1, y1, x2, y2)))

            kp_arr     = np.array(kps[i])
            is_tracked = (pid in self.history_ids)
            in_shot_window = is_tracked and shot_player_id == pid
            box_color  = (0, 95, 215) if is_tracked else (30, 180, 180)
            box_thick  = 2 if is_tracked else 1
            if in_shot_window:
                box_thick = 3
            cv2.rectangle(out, (x1,y1), (x2,y2), box_color, box_thick)
            if in_shot_window:
                tag_h = 28
                tag_y1 = max(0, y1 - tag_h)
                tag_w = 86
                cv2.rectangle(out, (x1, tag_y1), (x1 + tag_w, y1),
                              box_color, -1)
                cv2.rectangle(out, (x1, tag_y1), (x1 + tag_w, y1),
                              box_color, 2)
                cv2.putText(out, "SHOT", (x1 + 11, max(tag_y1 + 21, 21)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 2,
                            cv2.LINE_AA)

            lbl_color = (0, 95, 215) if is_tracked else (30, 180, 180)
            id_text = f"ID:{pid}"
            id_x = min(max(x1 + 6, x2 - 72), out.shape[1] - 76)
            id_y = min(max(y1 + 22, 22), out.shape[0] - 8)
            cv2.putText(out, id_text, (id_x, id_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,0,0), 3)
            cv2.putText(out, id_text, (id_x, id_y),
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
        self.video_label.config(image=photo)
        self.video_label.image = photo
        self._update_video_scroll_region()

        self.timeline_var.set(self.current_frame)
        total_frame_idx = max(0, self.total_frames - 1)
        self.frame_label.config(
            text=(f"{self.current_frame} / {total_frame_idx}  ·  "
                  f"{self._fmt_time(self.current_frame / self.fps)} / "
                  f"{self._fmt_time(total_frame_idx / self.fps)}"))
        if update_chart:
            self.update_highlight()
        self._update_lost_status()

    def _render_empty_video(self, message=None):
        message = message or "Click Open Video to choose a basketball video"
        out = np.full((max(self.display_h, 240), max(self.display_w, 320), 3),
                      (233, 236, 239), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(message, font, scale, thickness)[0]
        x = max(16, (out.shape[1] - text_size[0]) // 2)
        y = max(40, out.shape[0] // 2)
        cv2.putText(out, message, (x, y), font, scale, (73,80,87),
                    thickness, cv2.LINE_AA)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = pil.resize((self.display_w, self.display_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self.video_label.config(image=photo)
        self.video_label.image = photo
        self._update_video_scroll_region()
        self.timeline_var.set(0)
        self.frame_label.config(text="No video loaded")

    # ── UI builder ────────────────────────────────────────────────────────────

    def _build_ui(self):
        r = self.root
        r.title("AI Basketball Motion Analyzer")
        r.configure(bg=self.BG)
        r.resizable(True, True)
        try:
            screen_w = r.winfo_screenwidth()
            screen_h = r.winfo_screenheight()
            win_w = int(screen_w * 0.85)
            win_h = int(screen_h * 0.85)
        except Exception:
            win_w, win_h = 1400, 900
        r.geometry(f"{win_w}x{win_h}")

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
                 font=("Menlo", 16, "bold")).pack(side="left", padx=16, pady=10)
        self.title_hint = tk.Label(
            title_bar,
            text="Click a player in the video to start tracking",
            bg=self.SURFACE, fg=self.TEXT_DIM, font=("Menlo", 11))
        self.title_hint.pack(side="right", padx=16)

        body = tk.Frame(r, bg=self.BG)
        body.pack(fill="both", expand=True)
        self.main_canvas = tk.Canvas(body, bg=self.BG, highlightthickness=0)
        self.main_scroll = ttk.Scrollbar(
            body, orient="vertical", command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=self.main_scroll.set)
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.main_scroll.pack(side="right", fill="y")
        content_holder = tk.Frame(self.main_canvas, bg=self.BG)
        self.main_canvas_window = self.main_canvas.create_window(
            (0, 0), window=content_holder, anchor="nw")
        content_holder.bind("<Configure>", self._on_main_content_configure)
        self.main_canvas.bind("<Configure>", self._on_main_canvas_configure)
        self.main_canvas.bind("<Enter>", self._bind_main_scroll)
        self.main_canvas.bind("<Leave>", self._unbind_main_scroll)

        content = tk.PanedWindow(
            content_holder, orient="horizontal", bg=self.BG, sashwidth=6,
            sashrelief="flat", bd=0)
        content.pack(fill="both", expand=True, padx=8, pady=(6,4))

        # LEFT
        left = tk.Frame(content, bg=self.SURFACE,
                        highlightbackground=self.BORDER, highlightthickness=1)
        content.add(left, minsize=360)

        left_panes = tk.PanedWindow(
            left, orient="vertical", bg="#CED4DA", sashwidth=10,
            sashrelief="raised", bd=0)
        left_panes.pack(fill="both", expand=True)

        video_panel = tk.Frame(left_panes, bg=self.SURFACE)
        self.video_panel = video_panel
        left_panes.add(video_panel, minsize=130)

        tk.Label(video_panel, text="VIDEO", bg=self.SURFACE, fg=self.TEXT_DIM,
                 font=("Menlo", 10, "bold")).pack(anchor="w", padx=10, pady=(6,2))

        video_scroll_frame = tk.Frame(video_panel, bg=self.SURFACE)
        video_scroll_frame.pack(padx=8, pady=(0,4), fill="both", expand=True)
        self.video_canvas = tk.Canvas(
            video_scroll_frame, bg="#E9ECEF", highlightthickness=0)
        self.video_hscroll = ttk.Scrollbar(
            video_scroll_frame, orient="horizontal",
            command=self.video_canvas.xview)
        self.video_vscroll = ttk.Scrollbar(
            video_scroll_frame, orient="vertical",
            command=self.video_canvas.yview)
        self.video_canvas.configure(
            xscrollcommand=self.video_hscroll.set,
            yscrollcommand=self.video_vscroll.set)
        self.video_label = tk.Label(self.video_canvas, bg="#E9ECEF",
                                    cursor="crosshair")
        self.video_canvas_window = self.video_canvas.create_window(
            (0, 0), window=self.video_label, anchor="nw")
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        video_scroll_frame.grid_rowconfigure(0, weight=1)
        video_scroll_frame.grid_columnconfigure(0, weight=1)
        self.video_label.bind("<Button-1>", self._on_video_click)
        self.video_canvas.bind("<Configure>", self._on_video_canvas_configure)

        slider_row = tk.Frame(video_panel, bg=self.SURFACE)
        slider_row.pack(fill="x", padx=8, pady=(0,2))
        tk.Label(slider_row, text="⏱", bg=self.SURFACE, fg=self.TEXT_DIM,
                 font=("Menlo", 11)).pack(side="left")
        self.timeline_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(slider_row, from_=0,
                                to=max(0, self.total_frames-1),
                                orient="horizontal", variable=self.timeline_var,
                                command=self._on_slider_move)
        self.slider.pack(side="left", fill="x", expand=True, padx=(4,0))

        ctrl = tk.Frame(video_panel, bg=self.BG, pady=4)
        ctrl.pack(fill="x", padx=8)

        self.open_btn = tk.Button(
            ctrl, text="Open Video",
            bg=self.BTN_SURF, fg=self.BTN_TEXT,
            activebackground=self.BTN_HOVER, activeforeground=self.BTN_TEXT,
            font=("Menlo", 12), relief="flat", width=10,
            padx=4, pady=6, cursor="hand2",
            command=self._open_video)
        self.open_btn.grid(row=0, column=0, padx=(0,4), pady=2, sticky="ew")

        self.play_btn = tk.Button(
            ctrl, text="Play",
            bg=self.ACCENT, fg=self.BTN_TEXT,
            activebackground="#1560AE", activeforeground=self.BTN_TEXT,
            font=("Menlo", 12), relief="flat", width=10,
            padx=4, pady=6, cursor="hand2",
            command=self._toggle_play)
        self.play_btn.grid(row=0, column=1, padx=4, pady=2, sticky="ew")

        self.reset_btn = tk.Button(
            ctrl, text="↺  Reset",
            bg=self.BTN_SURF, fg=self.BTN_TEXT,
            activebackground=self.BTN_HOVER, activeforeground=self.BTN_TEXT,
            font=("Menlo", 12), relief="flat",
            width=10, padx=4, pady=6, cursor="hand2",
            command=self._reset_ids)
        self.reset_btn.grid(row=0, column=2, padx=4, pady=2, sticky="ew")

        self.export_btn = tk.Button(
            ctrl, text="Export CSV",
            bg=self.BTN_SURF, fg=self.BTN_TEXT,
            activebackground=self.BTN_HOVER, activeforeground=self.BTN_TEXT,
            font=("Menlo", 12), relief="flat",
            width=10, padx=4, pady=6, cursor="hand2",
            command=self._export_csv)
        self.export_btn.grid(row=0, column=3, padx=(4,0), pady=2, sticky="ew")

        self.frame_label = tk.Label(ctrl, text="No video loaded",
                                    bg=self.BG, fg=self.TEXT_DIM,
                                    font=("Menlo", 11))
        self.frame_label.grid(row=1, column=0, columnspan=4, sticky="e", pady=(2,0))
        for col in range(4):
            ctrl.grid_columnconfigure(col, weight=1, uniform="video_buttons")

        tk.Label(video_panel, text="Space = Play/Pause   ·   R = Reset   ·   Q = Quit",
                 bg=self.SURFACE, fg=self.TEXT_DIM,
                 font=("Menlo", 9)).pack(pady=(2,6))

        summary_panel = tk.Frame(left_panes, bg=self.SURFACE)
        left_panes.add(summary_panel, minsize=90)

        summary_hdr = tk.Frame(summary_panel, bg=self.SURFACE)
        summary_hdr.pack(fill="x", padx=10, pady=(0,2))
        tk.Label(summary_hdr, text="SHOT KEYFRAME SUMMARY",
                 bg=self.SURFACE, fg=self.TEXT_DIM,
                 font=("Menlo", 12, "bold")).pack(side="left")

        summary_scroll_frame = tk.Frame(summary_panel, bg=self.SURFACE)
        summary_scroll_frame.pack(fill="both", expand=True, padx=8, pady=(0,6))
        self.summary_canvas = tk.Canvas(
            summary_scroll_frame, bg=self.SURFACE, height=40,
            highlightthickness=0)
        self.summary_container = tk.Frame(self.summary_canvas, bg=self.SURFACE)
        self.summary_scroll = ttk.Scrollbar(
            summary_scroll_frame, orient="vertical",
            command=self.summary_canvas.yview)
        self.summary_hscroll = ttk.Scrollbar(
            summary_scroll_frame, orient="horizontal",
            command=self.summary_canvas.xview)
        self.summary_canvas.configure(
            yscrollcommand=self.summary_scroll.set,
            xscrollcommand=self.summary_hscroll.set)
        self.summary_canvas.create_window(
            (0, 0), window=self.summary_container, anchor="nw")
        self.summary_container.bind(
            "<Configure>",
            lambda _e: self.summary_canvas.configure(
                scrollregion=self.summary_canvas.bbox("all")))
        self.summary_canvas.grid(row=0, column=0, sticky="nsew")
        summary_scroll_frame.grid_rowconfigure(0, weight=1)
        summary_scroll_frame.grid_columnconfigure(0, weight=1)
        self.summary_canvas.bind("<Enter>", self._bind_summary_scroll)
        self.summary_container.bind("<Enter>", self._bind_summary_scroll)
        self.summary_canvas.bind("<Leave>", self._unbind_summary_scroll)
        self.summary_container.bind("<Leave>", self._unbind_summary_scroll)
        self._refresh_summary_table()

        # RIGHT
        right = tk.Frame(content, bg=self.SURFACE,
                         highlightbackground=self.BORDER, highlightthickness=1)
        content.add(right, minsize=360)

        hdr = tk.Frame(right, bg=self.SURFACE)
        hdr.pack(fill="x", padx=10, pady=(6,0))
        tk.Label(hdr, text="JOINT ANGLE TIMELINE", bg=self.SURFACE,
                 fg=self.TEXT_DIM, font=("Menlo", 12, "bold")).pack(side="left")
        self.graph_status_lbl = tk.Label(hdr, text="", bg=self.SURFACE,
                                         fg=self.SUCCESS, font=("Menlo", 12))
        self.graph_status_lbl.pack(side="right")

        self.chart_frame = tk.Frame(right, bg=self.SURFACE)
        self.chart_frame.pack(fill="both", expand=True, padx=6, pady=4)

        self.chart_tabs_frame = tk.Frame(right, bg=self.SURFACE)
        self.chart_tabs_frame.pack(fill="x", padx=8, pady=(0,4))

        self.angle_label = tk.Label(
            right,
            text="No player linked — click a player in the video to begin",
            bg=self.SURFACE, fg=self.TEXT_DIM,
            font=("Menlo", 12), wraplength=560)
        self.angle_label.pack(pady=(0,6))

        status_bar = tk.Frame(r, bg=self.STATUS_BG,
                              highlightbackground=self.BORDER,
                              highlightthickness=1, height=24)
        status_bar.pack(fill="x", side="bottom")
        self.status_label = tk.Label(status_bar, text="Ready",
                                     bg=self.STATUS_BG, fg=self.TEXT_DIM,
                                     font=("Menlo", 10))
        self.status_label.pack(side="left", padx=10)
        tk.Label(status_bar, text="Developed by Ryan Su",
                 bg=self.STATUS_BG, fg=self.TEXT_DIM,
                 font=("Menlo", 10)).pack(side="right", padx=10)

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
        styles = ["-", "--"]

        self.chart_lines = [
            self.ax.plot([], [], color=c, lw=lw, linestyle=ls, label=lbl)[0]
            for c, ls, lbl in zip(self.CHART_COLORS, styles, self.CHART_LABELS)
        ]
        self.release_line = self.ax.axvline(
            0, color="#D9480F", lw=1.8, linestyle="-",
            alpha=1.0, label="Release", visible=False, zorder=3)
        self.playhead_line = self.ax.axvline(
            0, color="#343A40", lw=1.0, linestyle=":",
            alpha=0.9, visible=False, zorder=4)
        self.highlight_dots = [
            self.ax.plot([], [], 'o', ms=6,
                         markerfacecolor=color,
                         markeredgecolor="white",
                         markeredgewidth=1.0, zorder=5)[0]
            for color in self.CHART_COLORS
        ]

        self.ax.set_xlabel("Frame", fontsize=11, color=txt_clr)
        self.ax.set_ylabel("Angle (°)", fontsize=11, color=txt_clr)
        self.ax.set_ylim(0, 190)
        self.ax.set_xlim(0, 1)
        self.ax.tick_params(colors=txt_clr, labelsize=10)
        self.ax_time = self.ax.twiny()
        self.ax_time.set_xlim(0, 1 / self.fps)
        self.ax_time.set_xlabel("Time (sec)", fontsize=10, color=txt_clr)
        self.ax_time.tick_params(colors=txt_clr, labelsize=9)
        self.ax_time.spines["top"].set_edgecolor(grid_clr)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(grid_clr)
        self.ax.grid(True, alpha=0.6, color=grid_clr, linewidth=0.5)
        release_proxy = Line2D([0], [0], color="#D9480F", lw=1.8,
                               linestyle="-", label="Release")
        handles = self.chart_lines + [release_proxy]
        self.ax.legend(handles=handles, loc="upper right", fontsize=10,
                       facecolor=fig_bg, edgecolor=grid_clr,
                       labelcolor=txt_clr, framealpha=0.9)
        self.ax.set_title(
            "Shooting elbow and power knee inside detected shot windows",
            fontsize=10, color=self.TEXT_DIM, pad=6)
        self.no_data_text = self.ax.text(
            0.5, 0.5, "No Data Available",
            transform=self.ax.transAxes, ha="center", va="center",
            fontsize=13, color=self.TEXT_DIM)
        self._set_chart_blank()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_video_click(self, event):
        if not self.video_path:
            return
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
                self._invalidate_analysis_cache()
                self._was_lost = False
                self.paused    = False
                self.play_btn.config(text="Pause")
                self.update_full_graph()
                self._refresh_summary_table()
                self._schedule_playback()
                self._update_status(
                    f"Linked ID {pid}  ·  Player identity: "
                    f"{sorted(self.history_ids)}")
                print(f"[CLICK] ID={pid} | history={sorted(self.history_ids)}")
                break

    def _on_content_resize(self, event):
        if event.width <= 0 or event.height <= 0:
            return
        if not hasattr(self, "video_panel"):
            return
        panel_w = max(160, self.video_panel.winfo_width() - 18)
        fixed_h = 118
        panel_h = max(110, self.video_panel.winfo_height() - fixed_h)
        old_size = (self.display_w, self.display_h)
        self._compute_display_size(
            self.video_w, self.video_h, panel_w=panel_w, panel_h=panel_h)
        if (self.display_w, self.display_h) != old_size:
            self._render_frame(update_chart=False)
        self._update_video_scroll_region()
        self._update_summary_scroll_region()

    def _on_video_canvas_configure(self, _event=None):
        self._update_video_scroll_region()

    def _update_video_scroll_region(self):
        if not hasattr(self, "video_canvas"):
            return
        self.video_label.update_idletasks()
        bbox = self.video_canvas.bbox("all")
        if not bbox:
            return
        self.video_canvas.configure(scrollregion=bbox)
        content_w = bbox[2] - bbox[0]
        content_h = bbox[3] - bbox[1]
        view_w = max(1, self.video_canvas.winfo_width())
        view_h = max(1, self.video_canvas.winfo_height())
        if content_w > view_w + 2:
            if not self.video_hscroll.winfo_ismapped():
                self.video_hscroll.grid(row=1, column=0, sticky="ew")
        else:
            self.video_hscroll.grid_remove()
        if content_h > view_h + 2:
            if not self.video_vscroll.winfo_ismapped():
                self.video_vscroll.grid(row=0, column=1, sticky="ns")
        else:
            self.video_vscroll.grid_remove()

    def _on_main_content_configure(self, _event=None):
        if hasattr(self, "main_canvas"):
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_main_canvas_configure(self, event):
        if not hasattr(self, "main_canvas_window"):
            return
        self.main_canvas.itemconfigure(
            self.main_canvas_window, width=event.width, height=max(520, event.height))
        self._on_content_resize(event)

    def _bind_main_scroll(self, _event=None):
        if hasattr(self, "main_canvas"):
            self.main_canvas.focus_set()
            self.main_canvas.bind_all("<MouseWheel>", self._on_main_mousewheel)

    def _unbind_main_scroll(self, _event=None):
        if hasattr(self, "main_canvas"):
            self.main_canvas.unbind_all("<MouseWheel>")

    def _on_main_mousewheel(self, event):
        if not hasattr(self, "main_canvas"):
            return
        if event.delta == 0:
            return
        step = -1 if event.delta > 0 else 1
        self.main_canvas.yview_scroll(step, "units")

    def _bind_summary_scroll(self, _event=None):
        if hasattr(self, "summary_canvas"):
            self.summary_canvas.focus_set()
            self.summary_canvas.bind_all("<MouseWheel>", self._on_summary_mousewheel)

    def _unbind_summary_scroll(self, _event=None):
        if hasattr(self, "summary_canvas"):
            self.summary_canvas.unbind_all("<MouseWheel>")

    def _on_summary_mousewheel(self, event):
        if not hasattr(self, "summary_canvas"):
            return
        delta = event.delta
        if delta == 0:
            return
        step = -1 if delta > 0 else 1
        self.summary_canvas.yview_scroll(step, "units")

    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Select Basketball Video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")]
        )
        if path:
            self._load_video(path)

    def _reset_analysis_state(self):
        self.current_frame = 0
        self.paused = True
        self.players_at_frame = []
        self.linked_ids.clear()
        self.history_ids.clear()
        self._was_lost = False
        self.id_analysis_rows = {}
        self._manual_shot_sides = {}
        self._pending_shot_sides = {}
        self._selected_shot_id = None
        self._current_chart_shot_id = None
        self._invalidate_analysis_cache()

    def _load_video(self, video_path):
        self.paused = True
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self.cap is not None:
            self.cap.release()

        self._reset_analysis_state()
        self.video_path = video_path
        self.play_btn.config(text="Play")
        self._update_status("Loading and pre-processing video...")
        self._render_empty_video("Pre-processing video... please wait")
        self.update_full_graph(set())
        self._refresh_summary_table()
        self.root.update_idletasks()

        self.cache, self.id_angle_data, self.total_frames = load_or_build_cache(
            video_path)
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._compute_display_size(self.video_w, self.video_h)
        self.slider.configure(to=max(0, self.total_frames - 1))
        self._precompute_all_id_analysis()
        self.update_full_graph(set())
        self._refresh_summary_table()
        self._render_frame()
        self._update_status(f"Loaded {os.path.basename(video_path)}")

    def _on_slider_move(self, value):
        if not self.video_path or not self.cache:
            return
        new_frame = min(int(float(value)), len(self.cache)-1)
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            self._render_frame()

    def _toggle_play(self):
        if not self.video_path or not self.cache:
            self._update_status("Open a video before playback.")
            return
        self.paused = not self.paused
        if self.paused:
            self.play_btn.config(text="Play")
            self.update_highlight()
        else:
            self.play_btn.config(text="Pause")
            self._schedule_playback()

    def _schedule_playback(self):
        if self.paused or not self.video_path or not self.cache:
            return
        safe_max = len(self.cache) - 1
        if self.current_frame < safe_max:
            self.current_frame += 1
            self._render_frame(update_chart=True, seek=False)
            delay_ms = max(1, int(1000 / self.fps))
            self._after_id = self.root.after(delay_ms, self._schedule_playback)
        else:
            self.paused = True
            self.play_btn.config(text="Play")
            self._update_status("End of video")

    def _reset_ids(self):
        if not self.video_path:
            self._update_status("Open a video before resetting IDs.")
            return
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
            self._invalidate_analysis_cache()
            self.paused = True
            self.play_btn.config(text="Play")
            self._refresh_summary_table()
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
            self._invalidate_analysis_cache()

            self.paused = True
            self.play_btn.config(text="Play")
            self.update_full_graph()
            self._refresh_summary_table()

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
        if self.cap is not None:
            self.cap.release()
        plt.close(self.fig)
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = BasketballAnalyzerApp(root)
    root.mainloop()

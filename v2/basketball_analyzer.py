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

import os
import pickle

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


def calculate_joint_angle(p1, mid, p2):
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

        vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._compute_display_size(vid_w, vid_h)

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

    # ── Frame rendering ───────────────────────────────────────────────────────

    def _render_frame(self):
        safe_frame = min(self.current_frame, len(self.cache)-1)
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
        self.frame_label.config(
            text=f"{self.current_frame} / {self.total_frames-1}")
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
                self.linked_ids.add(pid)
                self.history_ids.add(pid)
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
            self._render_frame()
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

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

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import filedialog, ttk

import analysis_engine
import data_manager
from chart_controller import ChartController
from summary_controller import SummaryController
from video_controller import VideoController


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
    CSV_FIELDS = data_manager.CSV_FIELDS
    SUMMARY_FIELDS = data_manager.SUMMARY_FIELDS

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
        self._playback_tick   = 0

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
        self._rendering_frame = False
        self._pending_render_args = None
        self._scheduled_render_id = None
        self._resize_after_id = None
        self._video_canvas_resize_after_id = None
        self._slider_after_id = None
        self._updating_slider = False
        self._loading_video = False
        self._load_generation = 0
        self._closing = False

        self.cap = cv2.VideoCapture(video_path) if video_path else None
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30.0
        self.fps = self.fps or 30.0

        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 0
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else 0
        self.chart_controller = ChartController(self)
        self.summary_controller = SummaryController(self)
        self.video_controller = VideoController(self)
        self._compute_display_size(self.video_w, self.video_h)
        self._last_video_label_size = None
        if self.video_path:
            self._precompute_all_id_analysis()

        self._build_ui()
        self._build_chart()
        self._render_frame()

    # ── Adaptive sizing ───────────────────────────────────────────────────────

    def _compute_display_size(self, vid_w, vid_h, panel_w=700, panel_h=560):
        self.video_controller.compute_display_size(
            vid_w, vid_h, panel_w=panel_w, panel_h=panel_h)

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
        return data_manager.frame_history_ids(
            self.history_ids, self.cache, self.current_frame)

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
        self.chart_controller.update_full_graph(ids_to_show)

    def _set_chart_blank(self, message="No Data Available"):
        self.chart_controller.set_chart_blank(message)

    def _set_chart_data_mode(self):
        self.chart_controller.set_chart_data_mode()

    def _refresh_chart_tabs(self, shot_ids=None):
        self.chart_controller.refresh_chart_tabs(shot_ids)

    def update_highlight(self):
        self.chart_controller.update_highlight()

    def _ensure_event_markers(self, count):
        self.chart_controller.ensure_event_markers(count)

    def _draw_event_markers(self, shot_rows):
        self.chart_controller.draw_event_markers(shot_rows)

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
        self.summary_controller.refresh_summary_table()

    def _bind_tree_click(self, widget, callback, exclude=()):
        self.summary_controller.bind_tree_click(widget, callback, exclude)

    def _update_summary_scroll_region(self):
        self.summary_controller.update_summary_scroll_region()

    @staticmethod
    def _opposite_side(side):
        return SummaryController.opposite_side(side)

    def _select_shot(self, shot_id):
        self.summary_controller.select_shot(shot_id)

    def _toggle_pending_shot_side(self, shot_id, side_kind):
        self.summary_controller.toggle_pending_shot_side(shot_id, side_kind)

    def _apply_pending_shot_sides(self, shot_id):
        self.summary_controller.apply_pending_shot_sides(shot_id)

    # ── Data export helpers ───────────────────────────────────────────────────

    @staticmethod
    def _valid_kp(kp, idx):
        return analysis_engine.valid_kp(kp, idx)

    @staticmethod
    def _mean_y(kp, idxs):
        return analysis_engine.mean_y(kp, idxs)

    @staticmethod
    def _dist(kp, a, b):
        return analysis_engine.dist(kp, a, b)

    @staticmethod
    def _nan_extreme(values, fn):
        return analysis_engine.nan_extreme(values, fn)

    @staticmethod
    def _mean(values):
        return analysis_engine.mean(values)

    @staticmethod
    def _has_landmarks(kp, idxs):
        return analysis_engine.has_landmarks(kp, idxs)

    def _has_visible_landmarks(self, kp, idxs, edge_margin=2):
        return analysis_engine.has_visible_landmarks(
            kp, idxs, self.video_w, self.video_h, edge_margin)

    def _sanitize_angles(self, kp, angles):
        return analysis_engine.sanitize_angles(
            kp, angles, self.video_w, self.video_h)

    def _quality_markers(self, kp, pose_quality, lower_body_quality):
        return analysis_engine.quality_markers(
            kp, pose_quality, lower_body_quality, self.video_w, self.video_h)

    @staticmethod
    def _merge_segments(segments, gap=10):
        return analysis_engine.merge_segments(segments, gap)

    @staticmethod
    def _fmt_csv_value(value):
        return data_manager.fmt_csv_value(value)

    @staticmethod
    def _fmt_time(seconds):
        seconds = max(0.0, float(seconds))
        mins = int(seconds // 60)
        secs = seconds - mins * 60
        return f"{mins}:{secs:04.1f}"

    def _default_export_path(self):
        return data_manager.default_export_path(
            self.video_path, self.history_ids)

    @staticmethod
    def _summary_export_path(csv_path):
        return data_manager.summary_export_path(csv_path)

    @staticmethod
    def _hip_height_norm(row):
        return data_manager.hip_height_norm(row)

    def _build_summary_rows(self, rows):
        return data_manager.build_summary_rows(rows)

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
        return analysis_engine.get_id_sample_at_frame(
            target_id, frame_idx, self.cache, self.id_angle_data,
            self.video_w, self.video_h)

    def _precompute_all_id_analysis(self):
        self.id_analysis_rows = self._build_all_id_analysis()

    def _build_all_id_analysis(self):
        return analysis_engine.build_all_id_analysis(
            self.cache, self.id_angle_data, self.total_frames, self.fps,
            self.video_w, self.video_h)

    def _renumber_merged_rows(self, rows, apply_manual_overrides=True):
        return analysis_engine.renumber_merged_rows(
            rows, self.total_frames, self._manual_shot_sides,
            apply_manual_overrides=apply_manual_overrides, fps=self.fps)

    def _get_auto_shot_sides(self, shot_id):
        return analysis_engine.get_auto_shot_sides(
            shot_id, self.history_ids, self.id_analysis_rows,
            self.total_frames, self.fps)

    def _recompute_overridden_shot_events(self, rows, shot_id):
        return analysis_engine.recompute_overridden_shot_events(
            rows, shot_id, self.fps)

    @staticmethod
    def _append_event_label(current, label):
        return analysis_engine.append_event_label(current, label)

    def _build_rows_for_id(self, target_id):
        return analysis_engine.build_rows_for_id(
            target_id, self.cache, self.id_angle_data, self.total_frames,
            self.fps, self.video_w, self.video_h)

    def _build_export_rows(self):
        return analysis_engine.build_export_rows(
            self.history_ids, self.id_analysis_rows, self.total_frames,
            self._manual_shot_sides, self.fps)

    def _detect_shot_segments(self, rows):
        return analysis_engine.detect_shot_segments(rows, self.fps)

    def _mark_shot_events(self, rows, shots):
        return analysis_engine.mark_shot_events(rows, shots, self.fps)

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
        summary_path, summary_rows = data_manager.export_motion_data(path, rows)

        shot_ids = sorted({r["shot_id"] for r in rows if r["shot_id"] != ""})
        self._update_status(
            f"Exported {len(rows)} frames, {len(shot_ids)} shots and "
            f"{len(summary_rows)} summary rows.")
        print(f"[EXPORT] CSV saved: {path}")
        print(f"[EXPORT] Summary CSV saved: {summary_path}")
        print(f"[EXPORT] Detected shot segments: {len(shot_ids)}")

    # ── Frame rendering ───────────────────────────────────────────────────────

    def _request_render_frame(self, delay_ms=20, update_chart=True, seek=True):
        self.video_controller.request_render_frame(
            delay_ms=delay_ms, update_chart=update_chart, seek=seek)

    def _run_scheduled_render(self, update_chart=True, seek=True):
        self.video_controller.run_scheduled_render(
            update_chart=update_chart, seek=seek)

    def _cancel_playback_timer(self):
        self.video_controller.cancel_playback_timer()

    def _render_frame(self, update_chart=True, seek=True):
        self.video_controller.render_frame(update_chart=update_chart, seek=seek)

    def _render_frame_impl(self, update_chart=True, seek=True):
        self.video_controller.render_frame_impl(
            update_chart=update_chart, seek=seek)

    def _render_empty_video(self, message=None):
        self.video_controller.render_empty_video(message)

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
        self.content_panes = content
        content.pack(fill="both", expand=True, padx=8, pady=(6,4))

        # LEFT
        left = tk.Frame(content, bg=self.SURFACE,
                        highlightbackground=self.BORDER, highlightthickness=1)
        content.add(left, minsize=360)

        left_panes = tk.PanedWindow(
            left, orient="vertical", bg="#CED4DA", sashwidth=10,
            sashrelief="raised", bd=0)
        self.left_panes = left_panes
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
        self.video_label.bind("<ButtonRelease-1>", self._on_video_click)
        self.video_canvas.bind("<ButtonRelease-1>", self._on_video_click)
        self.video_canvas.bind("<Configure>", self._on_video_canvas_configure)
        self.video_canvas.bind("<Enter>", self._bind_video_scroll)
        self.video_label.bind("<Enter>", self._bind_video_scroll)
        self.video_canvas.bind("<Leave>", self._unbind_video_scroll)
        self.video_label.bind("<Leave>", self._unbind_video_scroll)

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

        tk.Label(video_panel, text="Space = Play/Pause   ·   ←/→ = 1 frame   ·   ↑/↓ = 20 frames",
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
        r.bind("<Left>",  lambda e: self._step_frames(-1))
        r.bind("<Right>", lambda e: self._step_frames(1))
        r.bind("<Up>",    lambda e: self._step_frames(-20))
        r.bind("<Down>",  lambda e: self._step_frames(20))
        r.bind("r",       lambda e: self._reset_ids())
        r.bind("R",       lambda e: self._reset_ids())
        r.bind("q",       lambda e: self._quit())
        r.bind("Q",       lambda e: self._quit())
        r.protocol("WM_DELETE_WINDOW", self._quit)

    def _build_chart(self):
        self.chart_controller.build_chart()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_video_click(self, event):
        if not self.video_path or self._loading_video:
            return
        if self.cap is None:
            return
        vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if event.widget == self.video_canvas:
            bbox = self.video_canvas.bbox(self.video_canvas_window)
            if not bbox:
                return
            click_x = self.video_canvas.canvasx(event.x) - bbox[0]
            click_y = self.video_canvas.canvasy(event.y) - bbox[1]
        else:
            click_x, click_y = event.x, event.y
        if click_x < 0 or click_y < 0:
            return
        if click_x > self.display_w or click_y > self.display_h:
            return
        sx = vid_w / max(self.display_w, 1)
        sy = vid_h / max(self.display_h, 1)
        cx, cy = click_x * sx, click_y * sy

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
                self._cancel_playback_timer()
                self.update_full_graph()
                self._refresh_summary_table()
                self._schedule_playback()
                self._update_status(
                    f"Linked ID {pid}  ·  Player identity: "
                    f"{sorted(self.history_ids)}")
                print(f"[CLICK] ID={pid} | history={sorted(self.history_ids)}")
                break
        return "break"

    def _on_content_resize(self, event):
        if event.width <= 0 or event.height <= 0:
            return
        if not hasattr(self, "video_panel"):
            return
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(
            80, lambda: self._handle_content_resize(event.width, event.height))

    def _handle_content_resize(self, _width, _height):
        self._resize_after_id = None
        if self._loading_video:
            return
        panel_w = max(160, self.video_canvas.winfo_width())
        panel_h = max(110, self.video_canvas.winfo_height())
        old_size = (self.display_w, self.display_h)
        self._compute_display_size(
            self.video_w, self.video_h, panel_w=panel_w, panel_h=panel_h)
        if (self.display_w, self.display_h) != old_size:
            self._request_render_frame(delay_ms=120, update_chart=False)
        self._update_video_scroll_region()
        self._update_summary_scroll_region()

    def _on_video_canvas_configure(self, _event=None):
        if self._loading_video:
            return
        if self._video_canvas_resize_after_id is not None:
            self.root.after_cancel(self._video_canvas_resize_after_id)
        self._video_canvas_resize_after_id = self.root.after(
            80, self._handle_video_canvas_resize)
        self._update_video_scroll_region()

    def _handle_video_canvas_resize(self):
        self._video_canvas_resize_after_id = None
        if self._loading_video or not hasattr(self, "video_canvas"):
            return
        view_w = max(160, self.video_canvas.winfo_width())
        view_h = max(110, self.video_canvas.winfo_height())
        old_size = (self.display_w, self.display_h)
        self._compute_display_size(
            self.video_w, self.video_h, panel_w=view_w, panel_h=view_h)
        if (self.display_w, self.display_h) != old_size:
            self._request_render_frame(delay_ms=1, update_chart=False)
        self._update_video_scroll_region()

    def _update_video_scroll_region(self):
        self.video_controller.update_video_scroll_region()

    def _on_main_content_configure(self, _event=None):
        if hasattr(self, "main_canvas"):
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_main_canvas_configure(self, event):
        if not hasattr(self, "main_canvas_window"):
            return
        self.main_canvas.itemconfigure(
            self.main_canvas_window, width=event.width, height=max(520, event.height))
        if not self._loading_video:
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
        delta_y, _delta_x = self._scroll_deltas(event)
        if delta_y == 0:
            return
        step = -1 if delta_y > 0 else 1
        self.main_canvas.yview_scroll(step, "units")

    def _bind_video_scroll(self, _event=None):
        if hasattr(self, "video_canvas"):
            self.video_canvas.focus_set()
            self.video_canvas.bind_all("<MouseWheel>", self._on_video_mousewheel)
            self.video_canvas.bind_all("<Shift-MouseWheel>", self._on_video_mousewheel)

    def _unbind_video_scroll(self, _event=None):
        if hasattr(self, "video_canvas"):
            self.video_canvas.unbind_all("<MouseWheel>")
            self.video_canvas.unbind_all("<Shift-MouseWheel>")

    def _on_video_mousewheel(self, event):
        if not hasattr(self, "video_canvas"):
            return "break"
        delta_y, delta_x = self._scroll_deltas(event)
        self._scroll_canvas(self.video_canvas, delta_y, delta_x)
        return "break"

    def _bind_summary_scroll(self, _event=None):
        if hasattr(self, "summary_canvas"):
            self.summary_canvas.focus_set()
            self.summary_canvas.bind_all("<MouseWheel>", self._on_summary_mousewheel)
            self.summary_canvas.bind_all(
                "<Shift-MouseWheel>", self._on_summary_mousewheel)

    def _unbind_summary_scroll(self, _event=None):
        if hasattr(self, "summary_canvas"):
            self.summary_canvas.unbind_all("<MouseWheel>")
            self.summary_canvas.unbind_all("<Shift-MouseWheel>")

    def _on_summary_mousewheel(self, event):
        if not hasattr(self, "summary_canvas"):
            return "break"
        delta_y, delta_x = self._scroll_deltas(event)
        self._scroll_canvas(self.summary_canvas, delta_y, delta_x)
        return "break"

    def _scroll_deltas(self, event):
        delta_y = getattr(event, "delta", 0)
        delta_x = getattr(event, "deltaX", 0)
        if getattr(event, "state", 0) & 0x0001 and delta_x == 0:
            delta_x, delta_y = delta_y, 0
        return delta_y, delta_x

    def _scroll_canvas(self, canvas, delta_y, delta_x):
        if delta_x:
            step_x = -1 if delta_x > 0 else 1
            canvas.xview_scroll(step_x, "units")
        if delta_y:
            step_y = -1 if delta_y > 0 else 1
            canvas.yview_scroll(step_y, "units")

    def _open_video(self):
        if self._loading_video:
            self._update_status("Video is still loading. Please wait.")
            return
        path = filedialog.askopenfilename(
            title="Select Basketball Video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")]
        )
        if path:
            self._load_video(path)

    def _reset_analysis_state(self):
        self.current_frame = 0
        self.paused = True
        self._playback_tick = 0
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
        self.video_controller.load_video(video_path)

    def _load_video_worker(self, generation, video_path):
        self.video_controller.load_video_worker(generation, video_path)

    def _finish_load_video(self, generation, video_path, cache, id_angle_data,
                           total_frames, fps, video_w, video_h,
                           id_analysis_rows):
        self.video_controller.finish_load_video(
            generation, video_path, cache, id_angle_data, total_frames, fps,
            video_w, video_h, id_analysis_rows)

    def _fail_load_video(self, generation, video_path, exc, tb):
        self.video_controller.fail_load_video(generation, video_path, exc, tb)

    def _on_slider_move(self, value):
        self.video_controller.on_slider_move(value)

    def _render_slider_frame(self):
        self.video_controller.render_slider_frame()

    def _toggle_play(self):
        self.video_controller.toggle_play()

    def _schedule_playback(self):
        self.video_controller.schedule_playback()

    def _step_frames(self, delta):
        self.video_controller.step_frames(delta)
        return "break"

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
            self._cancel_playback_timer()
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
            self._cancel_playback_timer()
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
        self._closing = True
        self._cancel_playback_timer()
        for after_id in (
            self._scheduled_render_id,
            self._resize_after_id,
            self._video_canvas_resize_after_id,
            self._slider_after_id,
        ):
            if after_id:
                self.root.after_cancel(after_id)
        if self.cap is not None:
            self.cap.release()
        self.chart_controller.close()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = BasketballAnalyzerApp(root)
    root.mainloop()

import tkinter as tk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ChartController:
    def __init__(self, app):
        self.app = app

    def build_chart(self):
        app = self.app
        fig_bg = "#FFFFFF"
        ax_bg = "#FAFAFA"
        grid_clr = "#DEE2E6"
        txt_clr = "#495057"

        app.fig, app.ax = plt.subplots(figsize=(7, 4.5))
        app.fig.patch.set_facecolor(fig_bg)
        app.ax.set_facecolor(ax_bg)

        lw = 0.9
        styles = ["-", "--"]

        app.chart_lines = [
            app.ax.plot([], [], color=c, lw=lw, linestyle=ls, label=lbl)[0]
            for c, ls, lbl in zip(app.CHART_COLORS, styles, app.CHART_LABELS)
        ]
        app.release_line = app.ax.axvline(
            0, color="#D9480F", lw=1.8, linestyle="-",
            alpha=1.0, label="Release", visible=False, zorder=3)
        app.playhead_line = app.ax.axvline(
            0, color="#343A40", lw=1.0, linestyle=":",
            alpha=0.9, visible=False, zorder=4)
        app.highlight_dots = [
            app.ax.plot([], [], "o", ms=6,
                        markerfacecolor=color,
                        markeredgecolor="white",
                        markeredgewidth=1.0, zorder=5)[0]
            for color in app.CHART_COLORS
        ]

        app.ax.set_xlabel("Frame", fontsize=11, color=txt_clr)
        app.ax.set_ylabel("Angle (°)", fontsize=11, color=txt_clr)
        app.ax.set_ylim(0, 190)
        app.ax.set_xlim(0, 1)
        app.ax.tick_params(colors=txt_clr, labelsize=10)
        app.ax_time = app.ax.twiny()
        app.ax_time.set_xlim(0, 1 / app.fps)
        app.ax_time.set_xlabel("Time (sec)", fontsize=10, color=txt_clr)
        app.ax_time.tick_params(colors=txt_clr, labelsize=9)
        app.ax_time.spines["top"].set_edgecolor(grid_clr)
        for spine in app.ax.spines.values():
            spine.set_edgecolor(grid_clr)
        app.ax.grid(True, alpha=0.6, color=grid_clr, linewidth=0.5)
        release_proxy = Line2D([0], [0], color="#D9480F", lw=1.8,
                               linestyle="-", label="Release")
        handles = app.chart_lines + [release_proxy]
        app.ax.legend(handles=handles, loc="upper right", fontsize=10,
                      facecolor=fig_bg, edgecolor=grid_clr,
                      labelcolor=txt_clr, framealpha=0.9)
        app.ax.set_title(
            "Shooting elbow and power knee inside detected shot windows",
            fontsize=10, color=app.TEXT_DIM, pad=6)
        app.no_data_text = app.ax.text(
            0.5, 0.5, "No Data Available",
            transform=app.ax.transAxes, ha="center", va="center",
            fontsize=13, color=app.TEXT_DIM)
        self.set_chart_blank()

        app.canvas = FigureCanvasTkAgg(app.fig, master=app.chart_frame)
        app.canvas.draw()
        app.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_full_graph(self, ids_to_show=None):
        app = self.app
        if ids_to_show is None:
            ids_to_show = app.history_ids

        x = np.arange(app.total_frames)
        shooting_elbow = np.full(app.total_frames, np.nan)
        power_knee = np.full(app.total_frames, np.nan)

        rows = app._get_analysis_rows(ids_to_show)
        shot_rows = [r for r in rows if r.get("shot_id") != ""]
        available_shots = sorted({r["shot_id"] for r in shot_rows})
        if app._selected_shot_id in available_shots:
            chart_shot_id = app._selected_shot_id
        else:
            chart_shot_id = available_shots[0] if available_shots else None
            app._selected_shot_id = chart_shot_id
        app._current_chart_shot_id = chart_shot_id
        for row in rows:
            if row.get("shot_id") != chart_shot_id:
                continue
            if not row.get("shooting_hand") and not row.get("power_leg"):
                continue
            frame = row.get("frame")
            if frame is None or frame < 0 or frame >= app.total_frames:
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

        app.chart_lines[0].set_data(x, shooting_elbow)
        app.chart_lines[1].set_data(x, power_knee)
        for line in (app.release_line, app.playhead_line):
            if line is not None:
                line.set_visible(False)
        for dot in app.highlight_dots:
            dot.set_data([], [])
        for marker, text in app.event_markers:
            marker.set_data([], [])
            text.set_visible(False)

        if shot_rows:
            self.set_chart_data_mode()
            first_shot_rows = [
                r for r in shot_rows if r.get("shot_id") == chart_shot_id
            ]
            shot_start = min(r["frame"] for r in first_shot_rows)
            shot_end = max(r["frame"] for r in first_shot_rows)
            pad = max(3, int(round(0.15 * app.fps)))
            x0 = max(0, shot_start - pad)
            x1 = min(app.total_frames - 1, shot_end + pad)
            app.ax.set_xlim(x0, x1)
            if app.ax_time is not None:
                app.ax_time.set_xlim(x0 / app.fps, x1 / app.fps)
            release_rows = [
                r for r in first_shot_rows if r.get("is_release_frame")
            ]
            if release_rows and app.release_line is not None:
                release_x = release_rows[0]["frame"]
                app.release_line.set_xdata([release_x, release_x])
                app.release_line.set_visible(True)
            self.draw_event_markers(first_shot_rows)
            visible_values = np.concatenate([
                shooting_elbow[~np.isnan(shooting_elbow)],
                power_knee[~np.isnan(power_knee)]
            ])
            if len(visible_values):
                ymax = min(205, max(185, float(np.nanmax(visible_values)) + 24))
                app.ax.set_ylim(0, ymax)
            app.graph_status_lbl.config(
                text=(f"IDs: {sorted(ids_to_show)}  ·  "
                      f"Shot {chart_shot_id}: frames {shot_start}-{shot_end}"))
        else:
            app._current_chart_shot_id = None
            app.ax.set_ylim(0, 190)
            app.ax.set_xlim(0, 1)
            if app.ax_time is not None:
                app.ax_time.set_xlim(0, 1 / app.fps)
            app.graph_status_lbl.config(text="")
            self.set_chart_blank()

        self.refresh_chart_tabs(available_shots)
        app.canvas.draw_idle()

    def set_chart_blank(self, message="No Data Available"):
        app = self.app
        if app.no_data_text is not None:
            app.no_data_text.set_text(message)
            app.no_data_text.set_visible(True)
        app.ax.title.set_visible(False)
        app.ax.set_axis_off()
        if app.ax_time is not None:
            app.ax_time.set_axis_off()
        legend = app.ax.get_legend()
        if legend is not None:
            legend.set_visible(False)

    def set_chart_data_mode(self):
        app = self.app
        app.ax.set_axis_on()
        app.ax.title.set_visible(True)
        if app.ax_time is not None:
            app.ax_time.set_axis_on()
        if app.no_data_text is not None:
            app.no_data_text.set_visible(False)
        legend = app.ax.get_legend()
        if legend is not None:
            legend.set_visible(True)

    def refresh_chart_tabs(self, shot_ids=None):
        app = self.app
        if app.chart_tabs_frame is None:
            return
        for child in app.chart_tabs_frame.winfo_children():
            child.destroy()
        if not shot_ids:
            return
        for shot_id in shot_ids:
            active = shot_id == app._current_chart_shot_id
            bg = app.ACCENT_LT if active else app.SURFACE
            fg = app.ACCENT if active else app.TEXT_DIM
            underline = "  " if active else ""
            tab = tk.Label(
                app.chart_tabs_frame,
                text=f"{underline}Shot {shot_id}{underline}",
                bg=bg, fg=fg,
                font=("Menlo", 12, "bold" if active else "normal"),
                padx=12, pady=5, cursor="hand2",
                highlightbackground=app.ACCENT if active else app.BORDER,
                highlightthickness=1)
            tab.pack(side="left", padx=(0, 4), pady=(0, 2))
            tab.bind(
                "<ButtonRelease-1>",
                lambda _e, sid=shot_id: app._select_shot(sid))

    def update_highlight(self):
        app = self.app
        if not app.history_ids:
            self.update_full_graph(set())
            if app.playhead_line is not None:
                app.playhead_line.set_visible(False)
            for dot in app.highlight_dots:
                dot.set_data([], [])
            app.canvas.draw_idle()
            return

        row = app._get_analysis_row_for_frame(
            app.current_frame, preferred_ids=app._get_frame_history_ids())
        elbow_val = np.nan
        knee_val = np.nan
        show_marker = bool(
            row and row.get("shot_id") == app._current_chart_shot_id
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

        current_x = app.current_frame
        if app.playhead_line is not None:
            app.playhead_line.set_xdata([current_x, current_x])
            app.playhead_line.set_visible(show_marker)
        values = [elbow_val, knee_val]
        for dot, y_val in zip(app.highlight_dots, values):
            if show_marker and not np.isnan(y_val):
                dot.set_data([current_x], [y_val])
            else:
                dot.set_data([], [])
        app.canvas.draw_idle()

        if row:
            pid_show = row.get("player_id")
            shooting_hand = row.get("shooting_hand", "")
            power_leg = row.get("power_leg", "")

            def _f(v):
                return f"{int(round(v))}°" if not np.isnan(v) else "--"

            app.angle_label.config(
                text=(f"ID {pid_show}  ·  "
                      f"{shooting_hand.title() or 'Shooting'} Elbow: "
                      f"{_f(elbow_val)}   "
                      f"{power_leg.title() or 'Power'} Knee: {_f(knee_val)}"),
                fg=app.ACCENT)

    def ensure_event_markers(self, count):
        app = self.app
        while len(app.event_markers) < count:
            marker, = app.ax.plot(
                [], [], "o", ms=5, markerfacecolor="#FFFFFF",
                markeredgecolor="#343A40", markeredgewidth=1.0, zorder=6)
            text = app.ax.text(
                0, 0, "", fontsize=9, color="#343A40",
                ha="center", va="bottom", visible=False,
                bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF",
                          ec="#CED4DA", alpha=0.92))
            app.event_markers.append((marker, text))

    def draw_event_markers(self, shot_rows):
        app = self.app
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

        self.ensure_event_markers(len(points))
        for idx, (marker, text) in enumerate(app.event_markers):
            if idx >= len(points):
                marker.set_data([], [])
                text.set_visible(False)
                continue
            x_val, y_val, label = points[idx]
            marker.set_data([x_val], [y_val])
            y_top = app.ax.get_ylim()[1]
            text.set_position((x_val, min(y_top - 10, y_val + 8)))
            text.set_text(label)
            text.set_visible(True)

    def close(self):
        if getattr(self.app, "fig", None) is not None:
            plt.close(self.app.fig)

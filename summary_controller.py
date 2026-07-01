import tkinter as tk

import numpy as np

import data_manager


class SummaryController:
    def __init__(self, app):
        self.app = app

    def refresh_summary_table(self):
        app = self.app
        if not hasattr(app, "summary_container"):
            return
        for child in app.summary_container.winfo_children():
            child.destroy()
        if not app.history_ids:
            self.update_summary_scroll_region()
            return

        rows = app._get_analysis_rows()
        summary_rows = data_manager.build_summary_rows(rows)
        shot_ids = sorted({r["shot_id"] for r in summary_rows})
        if app._selected_shot_id not in shot_ids:
            app._selected_shot_id = shot_ids[0] if shot_ids else None
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
            pending = app._pending_shot_sides.get(shot_id, {})
            display_hand = pending.get("shooting_hand", hand)
            display_leg = pending.get("power_leg", leg)
            selected = shot_id == app._selected_shot_id
            border = app.ACCENT if selected else app.BORDER
            card_bg = app.ACCENT_LT if selected else "#FFFFFF"

            card = tk.Frame(
                app.summary_container, bg=card_bg,
                highlightbackground=border, highlightthickness=2 if selected else 1)
            card.pack(fill="x", pady=(0, 5))

            card_body = tk.Frame(card, bg=card_bg)
            card_body.pack(fill="x", padx=7, pady=6)

            side_panel = tk.Frame(card_body, bg=card_bg)
            side_panel.pack(side="left", fill="y", padx=(0, 6))
            info_lines = [
                f"Shot #{shot_id}",
                f"Frames: {shot_start}-{shot_end}",
                f"Hand: {display_hand or '--'}",
                f"Leg: {display_leg or '--'}",
            ]
            for line in info_lines:
                tk.Label(
                    side_panel, text=line, bg=card_bg, fg=app.TEXT,
                    font=("Menlo", 11, "bold"), width=18,
                    padx=4, pady=2, anchor="w"
                ).pack(anchor="w", fill="x")

            change_box = tk.Frame(side_panel, bg=card_bg)
            change_box.pack(anchor="w", pady=(5, 2))
            hand_toggle = tk.Button(
                change_box, text="Change🤚", bg=app.BTN_SURF, fg=app.BTN_TEXT,
                activebackground=app.BTN_HOVER, activeforeground=app.BTN_TEXT,
                font=("Menlo", 10), relief="flat", padx=6, pady=2,
                cursor="hand2",
                command=lambda sid=shot_id: self.toggle_pending_shot_side(sid, "hand"))
            hand_toggle.pack(side="left", padx=(0, 4))
            leg_toggle = tk.Button(
                change_box, text="Change🦶", bg=app.BTN_SURF, fg=app.BTN_TEXT,
                activebackground=app.BTN_HOVER, activeforeground=app.BTN_TEXT,
                font=("Menlo", 10), relief="flat", padx=6, pady=2,
                cursor="hand2",
                command=lambda sid=shot_id: self.toggle_pending_shot_side(sid, "leg"))
            leg_toggle.pack(side="left")

            apply_box = tk.Frame(side_panel, bg=card_bg)
            apply_box.pack(anchor="w", pady=(2, 0))
            apply_toggle = tk.Button(
                apply_box, text="Apply", bg=app.BTN_SURF,
                fg=app.SUCCESS if pending else app.TEXT_DIM,
                activebackground=app.BTN_HOVER, activeforeground=app.BTN_TEXT,
                font=("Menlo", 10, "bold"), relief="flat", padx=8, pady=2,
                cursor="hand2",
                command=lambda sid=shot_id: self.apply_pending_shot_sides(sid))
            apply_toggle.pack(side="left")
            tk.Label(
                side_panel, text="pending" if pending else " ",
                bg=card_bg, fg=app.ACCENT2,
                font=("Menlo", 11, "bold"), padx=4, pady=2
            ).pack(anchor="w")

            table = tk.Frame(card_body, bg="#FFFFFF")
            table.pack(side="left", fill="x", expand=True)
            cols = ("Event", "Frame", "Time", "Elbow", "Knee", "Hip")
            widths = [16, 8, 8, 8, 8, 8]
            for col_idx, (col, width) in enumerate(zip(cols, widths)):
                tk.Label(
                    table, text=col, bg="#F0F0F0", fg=app.TEXT,
                    font=("Menlo", 11, "bold"), width=width,
                    padx=3, pady=3, anchor="center",
                    highlightbackground=app.BORDER, highlightthickness=1
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
                        table, text=value, bg=bg, fg=app.TEXT,
                        font=("Menlo", 11, weight), width=width,
                        padx=3, pady=2, anchor="center",
                        highlightbackground=app.BORDER, highlightthickness=1
                    ).grid(row=row_idx, column=col_idx, sticky="nsew")
            self.bind_tree_click(
                card,
                lambda _e, sid=shot_id: self.select_shot(sid),
                exclude=(tk.Button,))
        self.update_summary_scroll_region()

    def bind_tree_click(self, widget, callback, exclude=()):
        if isinstance(widget, exclude):
            return
        widget.bind("<ButtonRelease-1>", callback, add="+")
        widget.configure(cursor="hand2")
        for child in widget.winfo_children():
            self.bind_tree_click(child, callback, exclude=exclude)

    def update_summary_scroll_region(self):
        app = self.app
        if not hasattr(app, "summary_canvas"):
            return
        bbox = app.summary_canvas.bbox("all")
        if not bbox:
            app.summary_canvas.configure(height=40, scrollregion=(0, 0, 0, 0))
            if hasattr(app, "summary_scroll"):
                app.summary_scroll.grid_remove()
            if hasattr(app, "summary_hscroll"):
                app.summary_hscroll.grid_remove()
            return
        content_w = bbox[2] - bbox[0]
        content_h = bbox[3] - bbox[1]
        max_h = max(120, app.summary_canvas.master.winfo_height() - 4)
        target_h = min(content_h + 4, max_h)
        app.summary_canvas.configure(height=target_h, scrollregion=bbox)
        view_w = max(1, app.summary_canvas.winfo_width())
        if hasattr(app, "summary_hscroll"):
            if content_w > view_w + 2:
                if not app.summary_hscroll.winfo_ismapped():
                    app.summary_hscroll.grid(row=1, column=0, sticky="ew")
            else:
                app.summary_hscroll.grid_remove()
        if hasattr(app, "summary_scroll"):
            if content_h > max_h:
                if not app.summary_scroll.winfo_ismapped():
                    app.summary_scroll.grid(row=0, column=1, sticky="ns")
            else:
                app.summary_scroll.grid_remove()

    @staticmethod
    def opposite_side(side):
        return "right" if side == "left" else "left"

    def select_shot(self, shot_id):
        app = self.app
        if app._selected_shot_id == shot_id:
            return
        app._selected_shot_id = shot_id
        app.update_full_graph()
        self.refresh_summary_table()
        app.update_highlight()

    def toggle_pending_shot_side(self, shot_id, side_kind):
        app = self.app
        rows = app._get_analysis_rows()
        shot_rows = [r for r in rows if r.get("shot_id") == shot_id]
        if not shot_rows:
            return
        current_hand = shot_rows[0].get("shooting_hand", "")
        current_leg = shot_rows[0].get("power_leg", "")
        pending = dict(app._pending_shot_sides.get(shot_id, {}))
        if side_kind == "hand" and current_hand in ("left", "right"):
            base = pending.get("shooting_hand", current_hand)
            pending["shooting_hand"] = self.opposite_side(base)
        elif side_kind == "leg" and current_leg in ("left", "right"):
            base = pending.get("power_leg", current_leg)
            pending["power_leg"] = self.opposite_side(base)
        else:
            return
        app._pending_shot_sides[shot_id] = pending
        app._selected_shot_id = shot_id
        self.refresh_summary_table()

    def apply_pending_shot_sides(self, shot_id):
        app = self.app
        rows = app._get_analysis_rows()
        shot_rows = [r for r in rows if r.get("shot_id") == shot_id]
        if not shot_rows:
            return
        pending = dict(app._pending_shot_sides.get(shot_id, {}))
        current_hand = shot_rows[0].get("shooting_hand", "")
        current_leg = shot_rows[0].get("power_leg", "")
        next_hand = pending.get("shooting_hand", current_hand)
        next_leg = pending.get("power_leg", current_leg)
        auto_hand, auto_leg = app._get_auto_shot_sides(shot_id)
        if next_hand == auto_hand and next_leg == auto_leg:
            app._manual_shot_sides.pop(shot_id, None)
        else:
            app._manual_shot_sides[shot_id] = {
                "shooting_hand": next_hand,
                "power_leg": next_leg,
            }
        app._pending_shot_sides.pop(shot_id, None)
        app._manual_recalc_version += 1
        app._invalidate_analysis_cache()
        app._selected_shot_id = shot_id
        app.update_full_graph()
        self.refresh_summary_table()
        app.update_highlight()

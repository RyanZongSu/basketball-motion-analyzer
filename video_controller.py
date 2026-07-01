import os
import threading
import traceback

import cv2
import numpy as np
from PIL import Image, ImageTk

from utils import draw_skeleton_on_frame, load_or_build_cache


class VideoController:
    def __init__(self, app):
        self.app = app

    def compute_display_size(self, vid_w, vid_h, panel_w=700, panel_h=560):
        app = self.app
        panel_w = max(160, int(panel_w))
        panel_h = max(120, int(panel_h))
        if vid_h == 0 or vid_w == 0:
            app.display_w, app.display_h = panel_w, panel_h
            return
        ratio = vid_w / vid_h
        if vid_h >= vid_w:
            app.display_h = panel_h
            app.display_w = int(panel_h * ratio)
        else:
            app.display_w = panel_w
            app.display_h = int(panel_w / ratio)
        app.display_w = max(app.display_w, 200)
        app.display_h = max(app.display_h, 150)

    def request_render_frame(self, delay_ms=20, update_chart=True, seek=True):
        app = self.app
        if app._scheduled_render_id is not None:
            app.root.after_cancel(app._scheduled_render_id)
        app._scheduled_render_id = app.root.after(
            delay_ms,
            lambda: self.run_scheduled_render(update_chart, seek))

    def run_scheduled_render(self, update_chart=True, seek=True):
        app = self.app
        app._scheduled_render_id = None
        self.render_frame(update_chart=update_chart, seek=seek)

    def cancel_playback_timer(self):
        app = self.app
        if app._after_id:
            app.root.after_cancel(app._after_id)
            app._after_id = None

    def render_frame(self, update_chart=True, seek=True):
        app = self.app
        if app._rendering_frame:
            pending = app._pending_render_args
            if pending is None:
                app._pending_render_args = (update_chart, seek)
            else:
                app._pending_render_args = (pending[0] or update_chart, seek)
            return

        app._rendering_frame = True
        try:
            self.render_frame_impl(update_chart=update_chart, seek=seek)
        finally:
            app._rendering_frame = False
            pending = app._pending_render_args
            app._pending_render_args = None
            if pending is not None:
                self.request_render_frame(
                    delay_ms=1, update_chart=pending[0], seek=pending[1])

    def render_frame_impl(self, update_chart=True, seek=True):
        app = self.app
        if app._loading_video:
            return
        if not app.video_path or app.cap is None or not app.cache:
            self.render_empty_video()
            if update_chart:
                app.update_full_graph(set())
            app._update_lost_status()
            return

        safe_frame = min(app.current_frame, len(app.cache)-1)
        if seek:
            app.cap.set(cv2.CAP_PROP_POS_FRAMES, safe_frame)
        ret, frame = app.cap.read()
        if not ret:
            return

        out = frame.copy()
        data = app.cache[safe_frame]
        app.players_at_frame.clear()

        ids = data.get("ids", [])
        boxes = data.get("boxes", [])
        kps = data.get("kps", [])
        current_history_ids = app._get_frame_history_ids()
        shot_row = (
            app._get_analysis_row_for_frame(
                safe_frame, preferred_ids=current_history_ids)
            if app.history_ids else None
        )
        shot_player_id = shot_row.get("player_id") if shot_row else None

        for i, pid in enumerate(ids):
            if i >= len(boxes) or i >= len(kps):
                continue
            x1, y1, x2, y2 = map(int, boxes[i])
            app.players_at_frame.append((pid, (x1, y1, x2, y2)))

            kp_arr = np.array(kps[i])
            is_tracked = (pid in app.history_ids)
            in_shot_window = is_tracked and shot_player_id == pid
            box_color = (0, 95, 215) if is_tracked else (30, 180, 180)
            box_thick = 2 if is_tracked else 1
            if in_shot_window:
                box_thick = 3
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, box_thick)
            if in_shot_window:
                tag_h = 28
                tag_y1 = max(0, y1 - tag_h)
                tag_w = 86
                cv2.rectangle(out, (x1, tag_y1), (x1 + tag_w, y1),
                              box_color, -1)
                cv2.rectangle(out, (x1, tag_y1), (x1 + tag_w, y1),
                              box_color, 2)
                cv2.putText(out, "SHOT", (x1 + 11, max(tag_y1 + 21, 21)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255),
                            2, cv2.LINE_AA)

            lbl_color = (0, 95, 215) if is_tracked else (30, 180, 180)
            id_text = f"ID:{pid}"
            id_x = min(max(x1 + 6, x2 - 72), out.shape[1] - 76)
            id_y = min(max(y1 + 22, 22), out.shape[0] - 8)
            cv2.putText(out, id_text, (id_x, id_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
            cv2.putText(out, id_text, (id_x, id_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, lbl_color, 1)

            if len(kp_arr) >= 17:
                angles = [np.nan]*4
                if pid in app.id_angle_data:
                    cf = min(safe_frame, app.id_angle_data[pid].shape[0]-1)
                    angles = app.id_angle_data[pid][cf]
                draw_skeleton_on_frame(out, kp_arr, angles)

        if not app.history_ids:
            cx = max(30, out.shape[1]//2 - 200)
            cv2.putText(out, "CLICK PLAYER TO LINK", (cx, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4)
            cv2.putText(out, "CLICK PLAYER TO LINK", (cx, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 230), 2)

        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = pil.resize((app.display_w, app.display_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        app.video_label.config(image=photo)
        app.video_label.image = photo
        self.update_video_scroll_region()

        app._updating_slider = True
        try:
            app.timeline_var.set(app.current_frame)
        finally:
            app._updating_slider = False
        total_frame_idx = max(0, app.total_frames - 1)
        app.frame_label.config(
            text=(f"{app.current_frame} / {total_frame_idx}  ·  "
                  f"{app._fmt_time(app.current_frame / app.fps)} / "
                  f"{app._fmt_time(total_frame_idx / app.fps)}"))
        if update_chart:
            app.update_highlight()
        app._update_lost_status()

    def render_empty_video(self, message=None):
        app = self.app
        message = message or "Click Open Video to choose a basketball video"
        out = np.full((max(app.display_h, 240), max(app.display_w, 320), 3),
                      (233, 236, 239), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(message, font, scale, thickness)[0]
        x = max(16, (out.shape[1] - text_size[0]) // 2)
        y = max(40, out.shape[0] // 2)
        cv2.putText(out, message, (x, y), font, scale, (73, 80, 87),
                    thickness, cv2.LINE_AA)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = pil.resize((app.display_w, app.display_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        app.video_label.config(image=photo)
        app.video_label.image = photo
        self.update_video_scroll_region()
        app._updating_slider = True
        try:
            app.timeline_var.set(0)
        finally:
            app._updating_slider = False
        app.frame_label.config(text="No video loaded")

    def load_video(self, video_path):
        app = self.app
        app.paused = True
        self.cancel_playback_timer()
        if app.cap is not None:
            app.cap.release()
            app.cap = None

        app._reset_analysis_state()
        app.video_path = video_path
        app.cache = []
        app.id_angle_data = {}
        app.total_frames = 0
        app._load_generation += 1
        generation = app._load_generation
        app._loading_video = True
        app.play_btn.config(text="Play")
        app.open_btn.config(state="disabled")
        app.play_btn.config(state="disabled")
        app.reset_btn.config(state="disabled")
        app.export_btn.config(state="disabled")
        app._update_status("Processing video...")
        self.render_empty_video("Processing video...")
        app.update_full_graph(set())
        app._refresh_summary_table()
        worker = threading.Thread(
            target=self.load_video_worker,
            args=(generation, video_path),
            daemon=True)
        worker.start()

    def load_video_worker(self, generation, video_path):
        app = self.app
        try:
            cache, id_angle_data, total_frames = load_or_build_cache(video_path)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            app.cache = cache
            app.id_angle_data = id_angle_data
            app.total_frames = total_frames
            app.fps = fps or 30.0
            app.video_w = video_w
            app.video_h = video_h
            id_analysis_rows = app._build_all_id_analysis()
            if not app._closing:
                app.root.after(
                    0,
                    lambda: self.finish_load_video(
                        generation, video_path, cache, id_angle_data,
                        total_frames, fps, video_w, video_h, id_analysis_rows))
        except Exception as exc:
            tb = traceback.format_exc()
            err = str(exc)
            if not app._closing:
                app.root.after(
                    0,
                    lambda: self.fail_load_video(generation, video_path, err, tb))

    def finish_load_video(self, generation, video_path, cache, id_angle_data,
                          total_frames, fps, video_w, video_h,
                          id_analysis_rows):
        app = self.app
        if app._closing or generation != app._load_generation:
            return
        app.cache = cache
        app.id_angle_data = id_angle_data
        app.total_frames = total_frames
        app.cap = cv2.VideoCapture(video_path)
        app.fps = fps or 30.0
        app.video_w = video_w
        app.video_h = video_h
        self.compute_display_size(app.video_w, app.video_h)
        app.slider.configure(to=max(0, app.total_frames - 1))
        app.id_analysis_rows = id_analysis_rows
        app._loading_video = False
        app.update_full_graph(set())
        app._refresh_summary_table()
        self.render_frame()
        app._update_status(f"Loaded {os.path.basename(video_path)}")
        app.open_btn.config(state="normal")
        app.play_btn.config(state="normal")
        app.reset_btn.config(state="normal")
        app.export_btn.config(state="normal")

    def fail_load_video(self, generation, video_path, exc, tb):
        app = self.app
        if app._closing or generation != app._load_generation:
            return
        app._loading_video = False
        app.video_path = None
        app.cache = []
        app.id_angle_data = {}
        app.total_frames = 0
        app.open_btn.config(state="normal")
        app.play_btn.config(state="normal")
        app.reset_btn.config(state="normal")
        app.export_btn.config(state="normal")
        self.render_empty_video("Could not load video")
        app._update_status(f"Could not load {os.path.basename(video_path)}: {exc}")
        print(tb)

    def on_slider_move(self, value):
        app = self.app
        if app._updating_slider:
            return
        if not app.video_path or not app.cache:
            return
        new_frame = min(int(float(value)), len(app.cache)-1)
        if new_frame != app.current_frame:
            app.current_frame = new_frame
            if app._slider_after_id is not None:
                app.root.after_cancel(app._slider_after_id)
            app._slider_after_id = app.root.after(35, self.render_slider_frame)

    def render_slider_frame(self):
        app = self.app
        app._slider_after_id = None
        self.render_frame()

    def toggle_play(self):
        app = self.app
        if not app.video_path or not app.cache:
            app._update_status("Open a video before playback.")
            return
        app.paused = not app.paused
        if app.paused:
            self.cancel_playback_timer()
            app.play_btn.config(text="Play")
            app.update_highlight()
        else:
            self.cancel_playback_timer()
            app.play_btn.config(text="Pause")
            self.schedule_playback()

    def schedule_playback(self):
        app = self.app
        app._after_id = None
        if app.paused or not app.video_path or not app.cache:
            return
        safe_max = len(app.cache) - 1
        if app.current_frame < safe_max:
            app.current_frame += 1
            app._playback_tick += 1
            update_chart = app._playback_tick % 6 == 0
            self.render_frame(update_chart=update_chart, seek=False)
            delay_ms = max(33, int(1000 / app.fps))
            app._after_id = app.root.after(delay_ms, self.schedule_playback)
        else:
            app.paused = True
            app.play_btn.config(text="Play")
            app._update_status("End of video")

    def step_frames(self, delta):
        app = self.app
        if not app.video_path or not app.cache:
            app._update_status("Open a video before stepping frames.")
            return
        self.cancel_playback_timer()
        app.paused = True
        app.play_btn.config(text="Play")
        safe_max = len(app.cache) - 1
        old_frame = app.current_frame
        app.current_frame = max(0, min(safe_max, app.current_frame + int(delta)))
        if app.current_frame == old_frame:
            app.update_highlight()
            return
        self.render_frame(update_chart=True, seek=True)

    def update_video_scroll_region(self):
        app = self.app
        if not hasattr(app, "video_canvas"):
            return
        bbox = app.video_canvas.bbox("all")
        if not bbox:
            return
        app.video_canvas.configure(scrollregion=bbox)
        content_w = bbox[2] - bbox[0]
        content_h = bbox[3] - bbox[1]
        view_w = max(1, app.video_canvas.winfo_width())
        view_h = max(1, app.video_canvas.winfo_height())
        if content_w > view_w + 2:
            if not app.video_hscroll.winfo_ismapped():
                app.video_hscroll.grid(row=1, column=0, sticky="ew")
        else:
            app.video_hscroll.grid_remove()
        if content_h > view_h + 2:
            if not app.video_vscroll.winfo_ismapped():
                app.video_vscroll.grid(row=0, column=1, sticky="ns")
        else:
            app.video_vscroll.grid_remove()

import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

import data_manager


class DataManagerTest(unittest.TestCase):
    def test_fmt_csv_value_handles_empty_numeric_and_bool_values(self):
        self.assertEqual(data_manager.fmt_csv_value(None), "")
        self.assertEqual(data_manager.fmt_csv_value(np.nan), "")
        self.assertEqual(data_manager.fmt_csv_value(True), 1)
        self.assertEqual(data_manager.fmt_csv_value(np.bool_(False)), 0)
        self.assertEqual(data_manager.fmt_csv_value(12.34567), "12.3457")
        self.assertEqual(data_manager.fmt_csv_value("left"), "left")

    def test_default_and_summary_export_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp)
            now = datetime(2026, 1, 2, 3, 4, 5)
            path = data_manager.default_export_path(
                "/videos/my shot.mp4", {7, 2}, export_dir=export_dir, now=now)

            self.assertEqual(
                path,
                str(export_dir /
                    "my shot_ids-2-7_shot_motion_data_20260102_030405.csv"))
            self.assertEqual(
                data_manager.summary_export_path(path),
                str(export_dir /
                    "my shot_ids-2-7_shot_summary_20260102_030405.csv"))
            self.assertEqual(
                data_manager.summary_export_path(str(export_dir / "plain.csv")),
                str(export_dir / "plain_summary.csv"))

    def test_build_summary_rows_selects_event_angles_and_hip_height(self):
        rows = [
            {
                "shot_id": 1,
                "player_id": 9,
                "frame": 10,
                "time_sec": 0.4,
                "shooting_hand": "right",
                "power_leg": "left",
                "right_elbow_angle": 155.0,
                "left_knee_angle": 88.0,
                "avg_elbow_angle": 140.0,
                "avg_knee_angle": 95.0,
                "shoulder_y_px": 100.0,
                "hip_y_px": 150.0,
                "knee_y_px": 190.0,
                "left_ankle_y_px": 220.0,
                "right_ankle_y_px": np.nan,
                "is_release_frame": True,
            },
            {
                "shot_id": 1,
                "player_id": 9,
                "frame": 12,
                "time_sec": 0.48,
                "shooting_hand": "right",
                "power_leg": "left",
                "right_elbow_angle": 170.0,
                "left_knee_angle": 120.0,
                "avg_elbow_angle": 160.0,
                "avg_knee_angle": 110.0,
                "shoulder_y_px": 100.0,
                "hip_y_px": 150.0,
                "knee_y_px": 190.0,
                "left_ankle_y_px": 220.0,
                "right_ankle_y_px": np.nan,
                "is_observed_knee_max_extension_frame": True,
                "knee_max_extension_confirmed": "",
            },
        ]

        summary_rows = data_manager.build_summary_rows(rows)

        self.assertEqual(
            [row["event_label"] for row in summary_rows],
            [
                "shot_1_release_proxy",
                "shot_1_observed_knee_max_extension_unconfirmed",
            ])
        self.assertEqual(summary_rows[0]["shooting_elbow_angle_deg"], 155.0)
        self.assertEqual(summary_rows[0]["power_knee_angle_deg"], 88.0)
        self.assertEqual(summary_rows[0]["hip_height_norm"], 50.0 / 120.0)

    def test_export_motion_data_writes_main_and_summary_csv(self):
        rows = [{
            "frame": 5,
            "time_sec": 0.25,
            "player_id": 3,
            "shot_id": 1,
            "right_elbow_angle": 123.45678,
            "left_knee_angle": np.nan,
            "avg_elbow_angle": 120.0,
            "avg_knee_angle": 90.0,
            "shooting_hand": "right",
            "power_leg": "left",
            "shoulder_y_px": 100.0,
            "hip_y_px": 150.0,
            "knee_y_px": 180.0,
            "left_ankle_y_px": 220.0,
            "right_ankle_y_px": np.nan,
            "is_release_frame": True,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip_ids-3_shot_motion_data_20260102_030405.csv"

            summary_path, summary_rows = data_manager.export_motion_data(path, rows)

            with open(path, newline="") as f:
                main = list(csv.DictReader(f))
            with open(summary_path, newline="") as f:
                summary = list(csv.DictReader(f))

        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(main[0]["right_elbow_angle"], "123.4568")
        self.assertEqual(main[0]["left_knee_angle"], "")
        self.assertEqual(summary[0]["event_label"], "shot_1_release_proxy")

    def test_frame_history_ids_uses_current_frame_intersection(self):
        cache = [
            {"ids": [1, 2]},
            {"ids": [5]},
            {"ids": []},
        ]

        self.assertEqual(data_manager.frame_history_ids({1, 5}, cache, 0), {1})
        self.assertEqual(data_manager.frame_history_ids({1, 5}, cache, 1), {5})
        self.assertEqual(data_manager.frame_history_ids({1, 5}, cache, 2), set())
        self.assertEqual(data_manager.frame_history_ids({1, 5}, cache, 99), set())
        self.assertEqual(data_manager.frame_history_ids({1, 5}, [], 0), set())


if __name__ == "__main__":
    unittest.main()

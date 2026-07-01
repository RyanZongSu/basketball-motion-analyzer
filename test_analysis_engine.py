import unittest

import numpy as np

import analysis_engine


def _kp():
    points = np.zeros((17, 2), dtype=float)
    for idx in range(17):
        points[idx] = [50 + idx * 3, 80 + idx * 4]
    return points


def _row(frame, wrist=0.0, elbow=90.0, knee=120.0, pose=True, lower=True):
    return {
        "frame": frame,
        "player_id": 1,
        "shot_id": "",
        "shot_start_frame": "",
        "shot_end_frame": "",
        "analysis_start_frame": "",
        "analysis_end_frame": "",
        "left_wrist_height_norm": wrist,
        "right_wrist_height_norm": wrist,
        "wrist_height_norm": wrist,
        "left_elbow_angle": elbow,
        "right_elbow_angle": elbow - 3,
        "avg_elbow_angle": elbow,
        "left_knee_angle": knee,
        "right_knee_angle": knee + 4,
        "avg_knee_angle": knee,
        "hip_y_px": 200 + frame,
        "knee_height_norm": 0.2 + frame * 0.02,
        "left_ankle_height_norm": 0.0,
        "right_ankle_height_norm": 0.1,
        "pose_quality": int(pose),
        "lower_body_quality": int(lower),
        "shooting_hand": "",
        "power_leg": "",
        "is_knee_lowest_frame": False,
        "is_elbow_lowest_frame": False,
        "is_release_frame": False,
        "is_elbow_max_extension_frame": False,
        "is_knee_max_extension_frame": False,
        "is_observed_knee_max_extension_frame": False,
        "knee_max_extension_confirmed": "",
        "is_max_extension_frame": False,
        "event_label": "",
    }


class AnalysisEngineTest(unittest.TestCase):
    def test_sanitize_angles_removes_edge_cut_lower_body_angles(self):
        kp = _kp()
        kp[13] = [1, 100]
        angles = np.array([10.0, 20.0, 30.0, 40.0])

        clean = analysis_engine.sanitize_angles(kp, angles, video_w=300, video_h=300)

        self.assertEqual(clean[0], 10.0)
        self.assertEqual(clean[1], 20.0)
        self.assertTrue(np.isnan(clean[2]))
        self.assertEqual(clean[3], 40.0)

    def test_build_rows_for_id_reads_cache_and_marks_quality(self):
        kp = _kp()
        cache = [{"ids": [7], "kps": [kp.tolist()]}]
        id_angle_data = {7: np.array([[100.0, 110.0, 120.0, 130.0]])}

        rows = analysis_engine.build_rows_for_id(
            7, cache, id_angle_data, total_frames=1, fps=25.0,
            video_w=400, video_h=400)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["player_id"], 7)
        self.assertEqual(rows[0]["time_sec"], 0.0)
        self.assertEqual(rows[0]["avg_elbow_angle"], 105.0)
        self.assertEqual(rows[0]["pose_quality"], 1)

    def test_detect_and_mark_shot_events_sets_release_and_window_fields(self):
        rows = []
        for frame in range(24):
            wrist = 0.5 if 8 <= frame <= 12 else 0.1
            elbow = 132.0 if 8 <= frame <= 12 else 90.0
            knee = 90.0 + (frame % 8) * 8
            rows.append(_row(frame, wrist=wrist, elbow=elbow, knee=knee))

        shots = analysis_engine.detect_shot_segments(rows, fps=10.0)
        analysis_engine.mark_shot_events(rows, shots, fps=10.0)

        self.assertTrue(shots)
        self.assertTrue(any(row["is_release_frame"] for row in rows))
        shot_rows = [row for row in rows if row["shot_id"] == 1]
        self.assertTrue(shot_rows)
        self.assertTrue(all(row["shot_start_frame"] != "" for row in shot_rows))

    def test_build_export_rows_renumbers_by_first_frame(self):
        id_analysis_rows = {
            2: [{"frame": 20, "player_id": 2, "shot_id": 1, "event_label": "shot_1_release_proxy"}],
            5: [{"frame": 10, "player_id": 5, "shot_id": 1, "event_label": "shot_1_release_proxy"}],
        }

        rows = analysis_engine.build_export_rows(
            {2, 5}, id_analysis_rows, total_frames=30, manual_shot_sides={},
            fps=30.0)

        self.assertEqual([row["shot_id"] for row in rows], [1, 2])
        self.assertEqual([row["player_id"] for row in rows], [5, 2])
        self.assertEqual(rows[0]["event_label"], "shot_1_release_proxy")
        self.assertEqual(rows[1]["event_label"], "shot_2_release_proxy")


if __name__ == "__main__":
    unittest.main()

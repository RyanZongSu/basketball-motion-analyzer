import csv
import os
from datetime import datetime

import numpy as np


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


def fmt_csv_value(value):
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


def default_export_path(video_path, history_ids, export_dir=None, now=None):
    if export_dir is None:
        export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "exports")
    os.makedirs(export_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]
    id_part = "-".join(str(pid) for pid in sorted(history_ids))
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    filename = f"{base}_ids-{id_part}_shot_motion_data_{stamp}.csv"
    return os.path.join(export_dir, filename)


def summary_export_path(csv_path):
    directory = os.path.dirname(csv_path)
    filename = os.path.basename(csv_path)
    summary_name = filename.replace("_shot_motion_data_", "_shot_summary_")
    if summary_name == filename:
        root, ext = os.path.splitext(filename)
        summary_name = f"{root}_summary{ext}"
    return os.path.join(directory, summary_name)


def hip_height_norm(row):
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


def build_summary_rows(rows):
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
                    "hip_height_norm": hip_height_norm(row),
                    "event_label": f"shot_{shot_id}_{label}",
                })

    summary_rows.sort(key=lambda r: (r["shot_id"], r["frame_number"]))
    return summary_rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: fmt_csv_value(row.get(field))
                for field in fields
            })


def export_motion_data(path, rows):
    write_csv(path, rows, CSV_FIELDS)
    summary_rows = build_summary_rows(rows)
    summary_path = summary_export_path(path)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    return summary_path, summary_rows


def frame_history_ids(history_ids, cache, current_frame):
    if not cache:
        return set()
    safe = min(current_frame, len(cache) - 1)
    frame_ids = set(cache[safe].get("ids", []))
    return set(history_ids) & frame_ids

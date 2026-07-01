import numpy as np


def valid_kp(kp, idx):
    return len(kp) > idx and (kp[idx, 0] > 0 or kp[idx, 1] > 0)


def mean_y(kp, idxs):
    ys = [float(kp[i, 1]) for i in idxs
          if len(kp) > i and (kp[i, 0] > 0 or kp[i, 1] > 0)]
    return float(np.mean(ys)) if ys else np.nan


def dist(kp, a, b):
    if len(kp) <= max(a, b):
        return np.nan
    if not ((kp[a, 0] > 0 or kp[a, 1] > 0) and
            (kp[b, 0] > 0 or kp[b, 1] > 0)):
        return np.nan
    return float(np.linalg.norm(kp[a] - kp[b]))


def nan_extreme(values, fn):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(fn(arr)) if len(arr) else np.nan


def mean(values):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.mean(arr)) if len(arr) else np.nan


def has_landmarks(kp, idxs):
    return all(len(kp) > i and (kp[i, 0] > 0 or kp[i, 1] > 0)
               for i in idxs)


def has_visible_landmarks(kp, idxs, video_w=0, video_h=0, edge_margin=2):
    for i in idxs:
        if not valid_kp(kp, i):
            return False
        x, y = float(kp[i, 0]), float(kp[i, 1])
        if x <= edge_margin or y <= edge_margin:
            return False
        if video_w > 0 and x >= video_w - edge_margin:
            return False
        if video_h > 0 and y >= video_h - edge_margin:
            return False
    return True


def sanitize_angles(kp, angles, video_w=0, video_h=0):
    clean = np.array(angles, dtype=float, copy=True)
    required = [
        (0, [5, 7, 9], False),
        (1, [6, 8, 10], False),
        (2, [11, 13], True),
        (3, [12, 14], True),
    ]
    for angle_idx, kp_idxs, check_edges in required:
        has_points = (
            has_visible_landmarks(kp, kp_idxs, video_w, video_h)
            if check_edges else has_landmarks(kp, kp_idxs)
        )
        if not has_points:
            clean[angle_idx] = np.nan
    return clean


def quality_markers(kp, pose_quality, lower_body_quality, video_w=0, video_h=0):
    shoulder_y = mean_y(kp, [5, 6])
    hip_y = mean_y(kp, [11, 12])
    torso = abs(hip_y - shoulder_y) if (
        not np.isnan(hip_y) and not np.isnan(shoulder_y)) else np.nan
    shoulder_w = dist(kp, 5, 6)
    hip_w = dist(kp, 11, 12)

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
    visible = sum(1 for idx in core_idxs if valid_kp(kp, idx))
    completeness = visible / len(core_idxs)
    edge_cut = any(
        valid_kp(kp, idx) and (
            kp[idx, 0] <= 2 or kp[idx, 1] <= 2 or
            (video_w > 0 and kp[idx, 0] >= video_w - 2) or
            (video_h > 0 and kp[idx, 1] >= video_h - 2)
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


def merge_segments(segments, gap=10):
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


def get_id_sample_at_frame(target_id, frame_idx, cache, id_angle_data,
                           video_w=0, video_h=0):
    data = cache[frame_idx]
    ids = data.get("ids", [])
    kps = data.get("kps", [])

    for i, pid in enumerate(ids):
        if pid != target_id or i >= len(kps):
            continue
        kp = np.array(kps[i])
        if len(kp) < 17:
            continue
        angles = np.full(4, np.nan)
        if pid in id_angle_data:
            safe = min(frame_idx, id_angle_data[pid].shape[0] - 1)
            angles = id_angle_data[pid][safe]
        angles = sanitize_angles(kp, angles, video_w, video_h)
        return pid, kp, angles
    return None, None, None


def build_rows_for_id(target_id, cache, id_angle_data, total_frames, fps,
                      video_w=0, video_h=0):
    rows = []
    for frame_idx in range(total_frames):
        pid, kp, angles = get_id_sample_at_frame(
            target_id, frame_idx, cache, id_angle_data, video_w, video_h)
        if pid is None:
            continue

        shoulder_y = mean_y(kp, [5, 6])
        hip_y = mean_y(kp, [11, 12])
        left_wrist_y = float(kp[9, 1]) if valid_kp(kp, 9) else np.nan
        right_wrist_y = float(kp[10, 1]) if valid_kp(kp, 10) else np.nan
        wrist_y = nan_extreme([left_wrist_y, right_wrist_y], np.min)
        knee_y = mean_y(kp, [13, 14])
        left_ankle_y = float(kp[15, 1]) if valid_kp(kp, 15) else np.nan
        right_ankle_y = float(kp[16, 1]) if valid_kp(kp, 16) else np.nan
        torso = abs(hip_y - shoulder_y) if (
            not np.isnan(hip_y) and not np.isnan(shoulder_y)) else np.nan
        if np.isnan(torso) or torso < 20:
            torso = np.nan

        def _height_norm(y):
            if np.isnan(y) or np.isnan(shoulder_y) or np.isnan(torso):
                return np.nan
            return (shoulder_y - y) / torso

        avg_elbow = mean([angles[0], angles[1]])
        avg_knee = mean([angles[2], angles[3]])
        pose_quality = int(has_landmarks(kp, [5, 6, 7, 8, 9, 10]))
        lower_body_quality = int(has_visible_landmarks(
            kp, [5, 6, 11, 12, 13, 14], video_w, video_h))
        view_orientation, data_confidence = quality_markers(
            kp, pose_quality, lower_body_quality, video_w, video_h)

        rows.append({
            "frame": frame_idx,
            "time_sec": frame_idx / fps,
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

    shots = detect_shot_segments(rows, fps)
    mark_shot_events(rows, shots, fps)
    return rows


def build_all_id_analysis(cache, id_angle_data, total_frames, fps,
                          video_w=0, video_h=0):
    id_analysis_rows = {}
    all_ids = sorted(id_angle_data.keys())
    if not all_ids:
        return id_analysis_rows
    print(f"⏳ Pre-computing shot windows for {len(all_ids)} IDs…")
    for pid in all_ids:
        rows = build_rows_for_id(
            pid, cache, id_angle_data, total_frames, fps, video_w, video_h)
        id_analysis_rows[pid] = rows
        shot_count = len({r["shot_id"] for r in rows if r["shot_id"] != ""})
        if shot_count:
            print(f"  ID {pid}: {shot_count} shot window(s)")
    print("✅ Shot-window pre-computation finished")
    return id_analysis_rows


def append_event_label(current, label):
    labels = [part for part in str(current).split("|") if part]
    if label not in labels:
        labels.append(label)
    return "|".join(labels)


def renumber_merged_rows(rows, total_frames, manual_shot_sides=None,
                         apply_manual_overrides=True, fps=30.0):
    manual_shot_sides = manual_shot_sides or {}
    merged = [dict(row) for row in rows]
    shot_first_frames = {}
    for row in merged:
        sid = row.get("shot_id")
        if sid == "":
            continue
        key = (row.get("player_id"), sid)
        frame = row.get("frame", total_frames)
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
            override = manual_shot_sides.get(new_shot_id, {})
            if override.get("shooting_hand"):
                row["shooting_hand"] = override["shooting_hand"]
            if override.get("power_leg"):
                row["power_leg"] = override["power_leg"]
        label = row.get("event_label", "")
        if label:
            row["event_label"] = label.replace(
                f"shot_{old_shot_id}_", f"shot_{new_shot_id}_")
    if apply_manual_overrides:
        for shot_id, override in manual_shot_sides.items():
            if override:
                recompute_overridden_shot_events(merged, shot_id, fps)
    merged.sort(key=lambda r: (r["frame"], r["player_id"]))
    return merged


def build_export_rows(history_ids, id_analysis_rows, total_frames,
                      manual_shot_sides=None, fps=30.0):
    rows = []
    for pid in sorted(history_ids):
        rows.extend(id_analysis_rows.get(pid, []))
    return renumber_merged_rows(
        rows, total_frames, manual_shot_sides, fps=fps)


def get_auto_shot_sides(shot_id, history_ids, id_analysis_rows, total_frames,
                        fps=30.0):
    rows = []
    for pid in sorted(history_ids):
        rows.extend(id_analysis_rows.get(pid, []))
    base_rows = renumber_merged_rows(
        rows, total_frames, apply_manual_overrides=False, fps=fps)
    shot_rows = [r for r in base_rows if r.get("shot_id") == shot_id]
    if not shot_rows:
        return "", ""
    return (
        shot_rows[0].get("shooting_hand", ""),
        shot_rows[0].get("power_leg", ""),
    )


def recompute_overridden_shot_events(rows, shot_id, fps):
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
        release_row["event_label"] = append_event_label(
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
        row["event_label"] = append_event_label(
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
        if r["frame"] <= release_frame + max(2, int(round(0.15 * fps)))
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
            3, int(round(0.12 * fps)))
        reaches_window_end = last_frame >= analysis_end - 1
        knee_max_confirmed = bool(enough_after_release and reaches_window_end)
        observed_knee_max_row["is_observed_knee_max_extension_frame"] = True
        observed_knee_max_row["knee_max_extension_confirmed"] = int(
            knee_max_confirmed)
        if not knee_max_confirmed:
            observed_knee_max_row["event_label"] = append_event_label(
                observed_knee_max_row.get("event_label", ""),
                f"{shot_prefix}observed_knee_max_extension_unconfirmed")
    knee_max_row = (
        observed_knee_max_row if knee_max_confirmed else None
    )
    _mark(knee_max_row, "is_knee_max_extension_frame",
          "knee_max_extension", max_ext=True)


def detect_shot_segments(rows, fps):
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
    high_gap = max(4, int(0.35 * fps))
    for idx in high_idxs[1:]:
        if idx <= prev + high_gap:
            prev = idx
            continue
        raw_segments.append((start, prev))
        start = idx
        prev = idx
    raw_segments.append((start, prev))

    min_len = 1
    pad_before = max(8, int(0.45 * fps))
    pad_after = max(6, int(0.35 * fps))
    def _has_body_lift(segment):
        hip_y = np.array([r["hip_y_px"] for r in segment], dtype=float)
        knee_y = np.array([r["knee_y_px"] for r in segment], dtype=float)
        shoulder_y = np.array([r["shoulder_y_px"] for r in segment],
                              dtype=float)
        seg_wrist = np.array([r["wrist_height_norm"] for r in segment],
                             dtype=float)
        seg_elbow = np.array([r["avg_elbow_angle"] for r in segment],
                             dtype=float)
        seg_lower = np.array([r["lower_body_quality"] for r in segment],
                             dtype=bool)

        valid_body = (
            ~np.isnan(hip_y) & ~np.isnan(knee_y) & ~np.isnan(shoulder_y) &
            seg_lower
        )
        release_like = np.where(
            valid_body & ~np.isnan(seg_wrist) & ~np.isnan(seg_elbow) &
            (seg_wrist > 0.35) & (seg_elbow > 120)
        )[0]
        if len(release_like) == 0:
            return False

        release_start = int(release_like[0])
        pre = np.where(valid_body & (np.arange(len(segment)) < release_start))[0]
        post = np.where(valid_body & (np.arange(len(segment)) >= release_start))[0]
        if len(pre) < 3 or len(post) < 3:
            return False

        hip_low_idx = int(pre[np.nanargmax(hip_y[pre])])
        after_low = post[post > hip_low_idx]
        if len(after_low) < 3:
            return False

        hip_high_idx = int(after_low[np.nanargmin(hip_y[after_low])])
        torso = abs(hip_y[hip_low_idx] - shoulder_y[hip_low_idx])
        if np.isnan(torso) or torso < 20:
            return False

        hip_lift = hip_y[hip_low_idx] - hip_y[hip_high_idx]
        knee_lift = knee_y[hip_low_idx] - knee_y[hip_high_idx]
        return bool(
            hip_lift >= max(35, 0.30 * torso) and
            knee_lift >= max(25, 0.20 * torso)
        )

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
        if len(valid_knee) >= 5:
            knee_range = (
                np.nanpercentile(valid_knee, 90) -
                np.nanpercentile(valid_knee, 10)
            )
            if knee_range < 24 and not _has_body_lift(segment):
                continue
        padded.append((seg_start, seg_end))
    return merge_segments(padded, gap=max(8, int(0.35 * fps)))


def mark_shot_events(rows, shots, fps):
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
        all_shoulder_y = np.array([r.get("shoulder_y_px", np.nan)
                                   for r in rows], dtype=float)
        all_knee_y = np.array([r.get("knee_y_px", np.nan) for r in rows],
                              dtype=float)
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

        def _body_load_low_index(first_release_like):
            body_candidates = np.where(
                ~np.isnan(all_hip_y) & ~np.isnan(all_shoulder_y) &
                ~np.isnan(all_knee_y) & all_lower_ok &
                (np.arange(len(rows)) >= start) &
                (np.arange(len(rows)) < first_release_like)
            )[0]
            if len(body_candidates) < 3:
                return None

            hip_low = int(body_candidates[np.nanargmax(
                all_hip_y[body_candidates])])
            after_low = np.where(
                ~np.isnan(all_hip_y) & ~np.isnan(all_knee_y) & all_lower_ok &
                (np.arange(len(rows)) > hip_low) &
                (np.arange(len(rows)) <= end)
            )[0]
            if len(after_low) < 3:
                return None

            hip_high = int(after_low[np.nanargmin(all_hip_y[after_low])])
            torso = abs(all_hip_y[hip_low] - all_shoulder_y[hip_low])
            if np.isnan(torso) or torso < 20:
                return None

            hip_lift = all_hip_y[hip_low] - all_hip_y[hip_high]
            knee_lift = all_knee_y[hip_low] - all_knee_y[hip_high]
            if (
                hip_lift >= max(35, 0.30 * torso) and
                knee_lift >= max(25, 0.20 * torso)
            ):
                return hip_low
            return None

        def _rapid_extension_release(body_low_idx, candidates):
            if body_low_idx is None:
                return None
            elbow_search = np.where(
                ~np.isnan(shot_elbow) & all_pose_ok &
                (np.arange(len(rows)) > body_low_idx) &
                (np.arange(len(rows)) <= candidates[-1])
            )[0]
            if len(elbow_search) < 4:
                return None

            pre_release = elbow_search[elbow_search < candidates[0]]
            if len(pre_release) < 2:
                return None
            elbow_low = int(pre_release[np.nanargmin(shot_elbow[pre_release])])
            max_rise_frames = max(6, int(round(0.45 * fps)))
            rapid_candidates = candidates[
                (candidates > elbow_low) &
                (candidates <= elbow_low + max_rise_frames)
            ]
            rapid_candidates = rapid_candidates[
                ~np.isnan(shot_elbow[rapid_candidates])
            ]
            if len(rapid_candidates) == 0:
                return None

            peak_elbow = np.nanmax(shot_elbow[rapid_candidates])
            elbow_gain = peak_elbow - shot_elbow[elbow_low]
            if np.isnan(elbow_gain) or elbow_gain < 35:
                return None
            peak_wrist = np.nanmax(all_wrist[rapid_candidates])
            mature = rapid_candidates[
                (shot_elbow[rapid_candidates] >= max(120, peak_elbow - 6)) &
                (all_wrist[rapid_candidates] >= peak_wrist * 0.85)
            ]
            return int(mature[0]) if len(mature) else None

        body_low_idx = _body_load_low_index(int(valid_release[0]))
        rapid_release_idx = _rapid_extension_release(
            body_low_idx, valid_release)

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
        if rapid_release_idx is not None:
            release_idx = rapid_release_idx

        if release_idx is None:
            continue

        pre_load_pad = max(6, int(0.25 * fps))
        analysis_start = max(0, start - pre_load_pad)
        analysis_end = min(len(rows) - 1,
                           release_idx + max(8, int(0.35 * fps)))

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
            hip_radius = max(2, int(round(0.10 * fps)))
            near_lowest_hip = hip_valid[
                (hip_valid >= hip_low_idx - hip_radius) &
                (hip_valid <= hip_low_idx + hip_radius)
            ]
            if len(near_lowest_hip) > 0:
                base_knee_search = near_lowest_hip

        valid_knee = base_knee_search[~np.isnan(power_knee[base_knee_search])]
        if len(valid_knee) > 0:
            search = valid_knee
            before_release = search[search < release_idx]
            if len(before_release) > 0:
                search = before_release

            if takeoff_idx is not None:
                before_takeoff = search[search < takeoff_idx]
                if len(before_takeoff) > 0:
                    search = before_takeoff

            knee_low_idx = int(search[np.nanargmin(power_knee[search])])
        if body_low_idx is not None and body_low_idx < release_idx:
            knee_low_idx = body_low_idx

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
                search <= release_idx + max(2, int(round(0.15 * fps)))
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
                3, int(round(0.12 * fps)))
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

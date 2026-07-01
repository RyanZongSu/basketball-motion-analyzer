import os
import pickle
import hashlib

import cv2
import numpy as np


CACHE_FILE = "final_cache.pkl"
ANGLE_FILE = "id_angle_cache.pkl"
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "yolov8n-pose.pt")

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


model = None


def _get_model():
    global model
    if model is None:
        from ultralytics import YOLO
        model = YOLO(MODEL_FILE)
    return model


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
    model = _get_model()
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

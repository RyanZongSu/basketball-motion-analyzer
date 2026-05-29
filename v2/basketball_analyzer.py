from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pickle
import os

# ============================
# Model Initialization
# ============================
model = YOLO('yolov8n-pose.pt')

# ============================
# Joint Angle Calculation (100% Accurate)
# ============================
def calculate_joint_angle(p1, mid, p2):
    p1 = np.array(p1, dtype=np.float32)
    mid = np.array(mid, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    vec1 = p1 - mid
    vec2 = p2 - mid

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-4 or norm2 < 1e-4:
        return np.nan

    dot = np.dot(vec1, vec2)
    cos_ang = dot / (norm1 * norm2)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_ang))
    return angle

# ============================
# Video File Selection
# ============================
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="Select Video",
    filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv")]
)
if not video_path:
    exit()

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ============================
# Cache Loading / Preprocessing
# ============================
cache_file = "final_cache.pkl"
angle_cache_file = "id_angle_cache.pkl"

if os.path.exists(cache_file) and os.path.exists(angle_cache_file):
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    with open(angle_cache_file, 'rb') as f:
        id_angle_data = pickle.load(f)
    print("✅ Loaded cache")
else:
    print("⏳ Pre-analyzing video...")
    cache = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        data = {"ids": [], "boxes": [], "kps": []}
        if len(res) > 0 and res[0].boxes.id is not None:
            data["ids"] = res[0].boxes.id.cpu().numpy().astype(int).tolist()
            data["boxes"] = res[0].boxes.xyxy.cpu().numpy().tolist()
            data["kps"] = res[0].keypoints.xy.cpu().numpy().tolist()
        cache.append(data)
        frame_id += 1

    print("⏳ Precompute joint angles...")
    id_angle_data = {}
    for f_idx in range(total_frames):
        data = cache[f_idx]
        ids = data.get("ids", [])
        kps_list = data.get("kps", [])

        for i, pid in enumerate(ids):
            if pid not in id_angle_data:
                id_angle_data[pid] = np.full((total_frames, 4), np.nan)

            kp = np.array(kps_list[i])
            if len(kp) < 17:
                continue

            left_elbow  = calculate_joint_angle(kp[5],  kp[7],  kp[9])
            right_elbow = calculate_joint_angle(kp[6],  kp[8],  kp[10])
            left_knee   = calculate_joint_angle(kp[11], kp[13], kp[15])
            right_knee  = calculate_joint_angle(kp[12], kp[14], kp[16])

            id_angle_data[pid][f_idx] = [left_elbow, right_elbow, left_knee, right_knee]

    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    with open(angle_cache_file, 'wb') as f:
        pickle.dump(id_angle_data, f)
    print("✅ Pre-analysis finished")

cap.release()
cap = cv2.VideoCapture(video_path)

# ============================
# Runtime State Variables
# ============================
current_frame = 0
linked_ids = set()
paused = True
players_at_current = []

# ============================
# Single Clean Plot (Optimized Style)
# ============================
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))

# Professional color scheme
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
labels = ['Left Elbow', 'Right Elbow', 'Left Knee', 'Right Knee']

# Thinner lines for clarity
lines = [ax.plot([], [], c, lw=1.2, label=l)[0] for c, l in zip(colors, labels)]
highlight, = ax.plot([], [], 'ko', ms=8, markerfacecolor='red')

ax.set_xlabel("Frame", fontsize=12)
ax.set_ylabel("Angle (°)", fontsize=12)
ax.set_ylim(0, 180)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, total_frames)

# ============================
# OpenCV UI Setup
# ============================
cv2.namedWindow("Basketball AI Tracker", cv2.WINDOW_NORMAL)

def set_frame(val):
    global current_frame
    current_frame = val

cv2.createTrackbar("Timeline", "Basketball AI Tracker", 0, total_frames - 1, set_frame)

def mouse_click(event, x, y, flags, param):
    global paused
    if event == cv2.EVENT_LBUTTONDOWN:
        for pid, box in players_at_current:
            x1, y1, x2, y2 = box
            if x1 < x < x2 and y1 < y < y2:
                linked_ids.add(pid)
                paused = False
                print(f"Linked ID: {pid} | All: {sorted(linked_ids)}")
                update_full_graph()

cv2.setMouseCallback("Basketball AI Tracker", mouse_click)

# ============================
# Graph Update Functions
# ============================
def update_full_graph():
    x = np.arange(total_frames)
    merged = np.full((total_frames, 4), np.nan)

    for pid in linked_ids:
        if pid not in id_angle_data:
            continue
        d = id_angle_data[pid]
        mask = np.isnan(merged) & ~np.isnan(d)
        merged[mask] = d[mask]

    for i in range(4):
        lines[i].set_data(x, merged[:, i])
    fig.canvas.draw()
    fig.canvas.flush_events()

def update_highlight():
    if not linked_ids:
        highlight.set_data([], [])
        return
    val = np.nan
    for pid in linked_ids:
        if pid in id_angle_data:
            v = id_angle_data[pid][current_frame, 1]
            if not np.isnan(v):
                val = v
                break
    highlight.set_data([current_frame], [val])
    fig.canvas.draw()
    fig.canvas.flush_events()

# ============================
# Main Playback Loop
# ============================
while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        break

    out = frame.copy()
    cv2.setTrackbarPos("Timeline", "Basketball AI Tracker", current_frame)
    data = cache[current_frame]
    players_at_current.clear()

    ids = data.get("ids", [])
    boxes = data.get("boxes", [])
    kps = data.get("kps", [])

    for i, pid in enumerate(ids):
        x1, y1, x2, y2 = map(int, boxes[i])
        players_at_current.append((pid, (x1, y1, x2, y2)))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(out, f"ID:{pid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    if not linked_ids:
        cv2.putText(out, "CLICK PLAYERS TO LINK", (50,60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,255), 3)

    for i, pid in enumerate(ids):
        if pid in linked_ids:
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 4)
            kp = np.array(kps[i])
            if len(kp) >= 17 and pid in id_angle_data:
                ang = id_angle_data[pid][current_frame]
                le, re, lk, rk = ang[0], ang[1], ang[2], ang[3]

                if not np.isnan(le):
                    cv2.putText(out, f"LE:{int(round(le))}", (int(kp[7,0]), int(kp[7,1])-20), 0, 0.6, (0,255,255), 2)
                if not np.isnan(re):
                    cv2.putText(out, f"RE:{int(round(re))}", (int(kp[8,0]), int(kp[8,1])-20), 0, 0.6, (0,255,255), 2)
                if not np.isnan(lk):
                    cv2.putText(out, f"LK:{int(round(lk))}", (int(kp[13,0]), int(kp[13,1])+20), 0, 0.6, (0,255,255), 2)
                if not np.isnan(rk):
                    cv2.putText(out, f"RK:{int(round(rk))}", (int(kp[14,0]), int(kp[14,1])+20), 0, 0.6, (0,255,255), 2)
            break

    update_highlight()
    cv2.imshow("Basketball AI Tracker", out)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('r'):
        linked_ids.clear()
        update_full_graph()
        print("Reset linked IDs")

    if not paused and linked_ids:
        current_frame += 1

# ============================
# Clean Exit (Single Figure Only)
# ============================
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()
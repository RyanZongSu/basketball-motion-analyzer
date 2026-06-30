# basketball_pose.py
# Ryan Su - AI Basketball Motion Analyzer v1.0

from ultralytics import YOLO
import cv2
import numpy as np

# ============================
# 步骤1：加载YOLO姿态模型
# ============================
model = YOLO('yolov8n-pose.pt')

# ============================
# 核心分析函数
# ============================
def calculate_angle(point_a, point_b, point_c):
    a = np.array(point_a)
    b = np.array(point_b)
    c = np.array(point_c)
    
    ba = a - b
    bc = c - b
    
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def analyze_shooting_form(keypoints, frame):
    RIGHT_SHOULDER = 6
    RIGHT_ELBOW    = 8
    RIGHT_WRIST    = 10
    RIGHT_HIP      = 12
    RIGHT_KNEE     = 14
    RIGHT_ANKLE    = 16
    
    shoulder = keypoints[RIGHT_SHOULDER]
    elbow    = keypoints[RIGHT_ELBOW]
    wrist    = keypoints[RIGHT_WRIST]
    hip      = keypoints[RIGHT_HIP]
    knee     = keypoints[RIGHT_KNEE]
    ankle    = keypoints[RIGHT_ANKLE]
    
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    knee_angle  = calculate_angle(hip, knee, ankle)
    
    feedback = []
    
    if elbow_angle < 80:
        feedback.append("⚠️  Elbow too bent - extend your arm more")
    elif elbow_angle > 100:
        feedback.append("⚠️  Elbow too straight - bend more at set point")
    else:
        feedback.append("✅ Elbow angle: PERFECT")
    
    if knee_angle < 110:
        feedback.append("⚠️  Knee too bent - you may be squatting too deep")
    elif knee_angle > 150:
        feedback.append("⚠️  Knees barely bent - use your legs more!")
    else:
        feedback.append("✅ Knee angle: PERFECT")
    
    y_offset = 50
    for msg in feedback:
        color = (0, 255, 0) if "✅" in msg else (0, 0, 255)
        cv2.putText(frame, msg, (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 35
    
    cv2.putText(frame, f"Elbow: {elbow_angle:.1f}°", (20, y_offset + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Knee: {knee_angle:.1f}°", (20, y_offset + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# ============================
# 读取视频
# ============================
video_path = 'videos/basketball_shot.mov'
cap = cv2.VideoCapture(video_path)

# ============================
# 逐帧处理（终极防崩溃版本）
# ============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    annotated_frame = frame.copy()

    # ==============================================
    # ✅✅✅ 终极修复：判断是否检测到“人”
    # ==============================================
    if len(results) > 0:
        res = results[0]
        if res.keypoints is not None:
            kp = res.keypoints.xy  # shape: (1, 17, 2)
            
            # 最关键的判断：是否有人
            if kp.shape[0] > 0:
                keypoints = kp[0].numpy()
                annotated_frame = res.plot()
                analyze_shooting_form(keypoints, annotated_frame)

    # 显示画面
    cv2.imshow('Basketball Motion Analysis', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()

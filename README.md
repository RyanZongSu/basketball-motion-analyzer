# 🏀 Basketball Motion Analyzer

**Developer:** Ryan Su | Midland School, Class of 2027  
**Tech Stack:** Python · YOLOv8 · OpenCV · NumPy

---

## 🎯 Project Overview

A tool that analyzes basketball shooting form in real-time, 
comparing player movements against professional benchmarks to provide 
actionable feedback.

**Motivation:** As Midland School's basketball MIP, I noticed that most 
players — including myself — develop subtle technical flaws without realizing it. 
This tool makes professional-level coaching accessible to everyone.

---

## 🔧 Features

- ✅ Real-time skeletal tracking (17 keypoints via YOLOv8-Pose)
- ✅ Automatic joint angle calculation (elbow, knee, wrist)
- ✅ Comparison with professional NBA shooting standards  
- ✅ Visual feedback overlay on video
- 🔄 Coming soon: Shot arc trajectory analysis

---

## 📸 Demo

<img width="1800" height="1126" alt="image" src="https://github.com/user-attachments/assets/c838f3db-7c94-45ff-9b8f-9ca52f6bd28d" />


---

## 🚀 How to Run

```bash
git clone https://github.com/RyanZongSu/basketball-motion-analyzer
cd basketball-motion-analyzer
pip install -r requirements.txt
python basketball_pose.py --video your_video.mp4
```

---

## 📊 Results

| Joint | Professional Standard | Typical Amateur | Improvement Tip |
|-------|----------------------|-----------------|-----------------|
| Elbow | 85-95° at set point | Often <75° | Extend follow-through |
| Knee  | 120-140° at jump | Often >150° | Bend knees more |

---

## 🔮 Future Work

- [ ] Multi-player tracking
- [ ] Web app interface for easy access

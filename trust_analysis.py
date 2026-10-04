"""How far can the camera numbers be trusted?

Model for each angle reading:  y = x + b_view + e
  x       the true joint angle
  b_view  an offset caused by the camera angle
  e       frame-to-frame noise

This script estimates the size of e from how much each angle curve jitters
from one frame to the next, and compares it with how much the angle changes
from shot to shot for one player. It then draws a bar chart.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MOTION = glob.glob("/mnt/user-data/uploads/Rylee_10shots*motion*.csv")[0]
SUMMARY = glob.glob("/mnt/user-data/uploads/Rylee_10shots*summary*.csv")[0]


def frame_noise(series):
    """Robust noise estimate from second differences.

    For independent noise with standard deviation s, the second difference
    y[t+1] - 2*y[t] + y[t-1] has variance 6*s^2. The median absolute value
    (times 1.4826) ignores the occasional big jump.
    """
    x = np.asarray(series, float)
    x = x[~np.isnan(x)]
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    return 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)


motion = pd.read_csv(MOTION)
summary = pd.read_csv(SUMMARY)
shots = motion[motion.shot_id.notna()]

# 1) Frame noise, one value per shot, for the power knee and the shooting elbow
knee_noise, elbow_noise = [], []
for _, g in shots.groupby("shot_id"):
    leg = g.power_leg.dropna().iloc[0]
    hand = g.shooting_hand.dropna().iloc[0]
    knee_noise.append(frame_noise(g[f"{leg}_knee_angle"]))
    elbow_noise.append(frame_noise(g[f"{hand}_elbow_angle"]))
sigma_knee, sigma_elbow = np.median(knee_noise), np.median(elbow_noise)

# 2) Per-shot values at the key moments
summary["key"] = summary.event_label.str.replace(r"shot_\d+_", "", regex=True)
knee = summary.pivot_table(index="shot_id", columns="key", values="power_knee_angle_deg")
elbow = summary.pivot_table(index="shot_id", columns="key", values="shooting_elbow_angle_deg")
metrics = {
    "Knee at\nits lowest": (knee["knee_lowest"].dropna(), sigma_knee),
    "Knee at\nrelease": (knee["release_proxy"].dropna(), sigma_knee),
    "Elbow at\nrelease": (elbow["release_proxy"].dropna(), sigma_elbow),
}

# 3) Compare spread across shots with noise: Var(Y) = Var(X) + Var(e)
rows = []
for name, (v, s) in metrics.items():
    sd = v.std(ddof=1)
    share = min(s**2 / sd**2, 1.0)
    rows.append((name, sd, s, share))
    print(f"{name.replace(chr(10), ' '):18s} spread SD {sd:5.1f}  noise {s:5.1f}  noise share of variance {share:4.0%}")

# 4) Bar chart
fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(rows))
ax.bar(x - 0.18, [r[1] for r in rows], 0.36, color="#1F6B8A", label="Spread across Rylee's shots (SD)")
ax.bar(x + 0.18, [r[2] for r in rows], 0.36, color="#C9A227", label="Frame-to-frame noise (estimated)")
for i, r in enumerate(rows):
    ax.text(i, max(r[1], r[2]) + 1.2, f"{r[3]:.0%} of spread\nis noise", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in rows], fontsize=9)
ax.set_ylabel("degrees")
ax.set_ylim(0, 33)
ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.set_title("Which readings can tell Rylee's shots apart?", fontsize=10)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig("noise_vs_spread.png", dpi=170)

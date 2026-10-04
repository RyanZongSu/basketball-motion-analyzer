"""Spread of key shooting metrics across one player's shots.

Reads the app's shot summary CSV and draws one dot per shot for each metric,
with the mean and a +/-1 standard deviation band. The goal is to see how much
a player varies from shot to shot BEFORE comparing to anyone else.
"""
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Path to the summary CSV exported by the app (first argument, or auto-detect)
path = sys.argv[1] if len(sys.argv) > 1 else glob.glob(
    "/mnt/user-data/uploads/Rylee_10shots*summary*.csv")[0]
s = pd.read_csv(path)

# Turn "shot_3_release_proxy" into just "release_proxy"
s["key"] = s["event_label"].str.replace(r"shot_\d+_", "", regex=True)
t = s.pivot_table(index="shot_id", columns="key", values="time_sec")
elbow = s.pivot_table(index="shot_id", columns="key",
                      values="shooting_elbow_angle_deg")
knee = s.pivot_table(index="shot_id", columns="key",
                     values="power_knee_angle_deg")

metrics = {
    "Knee at its lowest (deg)": knee["knee_lowest"],
    "Knee at release (deg)": knee["release_proxy"],
    "Time: lowest knee to release (s)": t["release_proxy"] - t["knee_lowest"],
    "Elbow at release (deg)": elbow["release_proxy"],
    "Elbow at its lowest (deg)  [unreliable]": elbow["elbow_lowest"],
}

fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.2))
for ax, (name, vals) in zip(axes, metrics.items()):
    v = vals.dropna()
    mean, sd = v.mean(), v.std()
    ax.axhspan(mean - sd, mean + sd, color="#cfe3ff", alpha=0.7)
    ax.axhline(mean, color="#2b6cb0", lw=1.5)
    color = "#c05621" if "unreliable" in name else "#1a365d"
    ax.scatter(np.zeros(len(v)) + np.random.uniform(-0.08, 0.08, len(v)),
               v.values, color=color, s=40, zorder=3)
    for sid, y in v.items():
        ax.annotate(str(int(sid)), (0.1, y), fontsize=7, color="gray")
    ax.set_title(name, fontsize=9)
    ax.set_xticks([])
    ax.set_xlim(-0.4, 0.4)
    ax.text(0, ax.get_ylim()[0], f"mean {mean:.2f}, SD {sd:.2f}",
            ha="center", va="bottom", fontsize=8)
fig.suptitle("Rylee, 10 shots: how much does one player vary? "
             "(band = mean +/- 1 SD, numbers = shot id)", fontsize=11)
plt.tight_layout()
plt.savefig("rylee_shot_spread.png", dpi=160)

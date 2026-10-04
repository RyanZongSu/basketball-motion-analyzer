"""Compare shot metrics for three groups of shots.

Groups: Rylee (side view), Ricky set shots (side view), Ricky set shots
(three-quarter view). One dot per shot, a bar for the mean, and thin whiskers
for +/- 1 standard deviation. Shot 7 of Ricky's set shots is left out because
its "lowest knee" frame was found too close to the release to be trusted.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R = pd.read_csv("/tmp/R.csv")            # Rylee, built from the app's summary CSV
S = pd.read_csv("/tmp/S.csv")            # Ricky set shots
S = S[S.shot_id != 7.0]                  # drop the shot with a bad knee-low frame

groups = {
    "Rylee\n(side)": R,
    "Ricky set\n(side)": S[S.view == "side"],
    "Ricky set\n(3/4 view)": S[S.view == "three_quarter"],
}
colors = ["#1a365d", "#2f855a", "#c05621"]
metrics = [("knee_low", "Knee at its lowest (deg)"),
           ("load_rel", "Time: lowest knee to release (s)"),
           ("knee_rel", "Knee at release (deg)")]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
rng = np.random.default_rng(1)
for ax, (col, title) in zip(axes, metrics):
    for i, (name, d) in enumerate(groups.items()):
        v = d[col].dropna().values
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(v)), v, color=colors[i], s=42, zorder=3)
        if len(v) > 1:
            m, sd = v.mean(), v.std(ddof=1)
            ax.hlines(m, i - 0.25, i + 0.25, color=colors[i], lw=2.5)
            ax.vlines(i, m - sd, m + sd, color=colors[i], lw=1.2, alpha=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(groups), fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Same app, different players and cameras: which gaps are bigger than the wobble?",
             fontsize=11)
plt.tight_layout()
plt.savefig("group_compare.png", dpi=160)

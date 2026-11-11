"""Quick visualization of a run_log.csv produced by train.py.

Usage: python plot_results.py [run_log.csv]
Produces: results.png (intrinsic reward curve + achievement-probe count over time)
"""
import sys
import csv
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "run_log.csv"

steps, reward, achievements = [], [], []
with open(path) as f:
    for row in csv.DictReader(f):
        steps.append(int(row["steps"]))
        reward.append(float(row["mean_intrinsic_reward"]))
        achievements.append(int(row["new_achievements_unlocked"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(steps, reward)
ax1.set_xlabel("env steps")
ax1.set_ylabel("mean intrinsic reward (forward-model surprise)")
ax1.set_title("Curiosity signal over training\n(should trend down as the world model improves)")

ax2.plot(steps, achievements)
ax2.set_xlabel("env steps")
ax2.set_ylabel("# distinct achievements probed (never fed to agent)")
ax2.set_title("Emergent milestones discovered\npurely from curiosity-driven behavior")

fig.tight_layout()
fig.savefig("results.png", dpi=150)
print("Saved results.png")

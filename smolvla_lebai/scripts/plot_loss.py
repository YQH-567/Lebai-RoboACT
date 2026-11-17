#!/usr/bin/env python3
"""Plot the logged 1000-step SmolVLA loss without changing training artifacts."""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
log = (ROOT / "outputs/train/smolvla_lebai_1000.log").read_text(errors="replace")
points = [(int(step[:-1]) * 1000 if step.endswith("K") else int(step), float(loss))
          for step, loss in re.findall(r"step:(\d+K?)\s+smpl:.*?\bloss:([0-9.]+)", log)]
if len(points) < 2:
    raise RuntimeError("No loss points found")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(*zip(*points), linewidth=1.5)
ax.set(xlabel="Optimizer step", ylabel="Training loss", title="SmolVLA Lebai 1000-step fine-tune")
ax.grid(alpha=0.3)
fig.tight_layout()
target = ROOT / "outputs/train/smolvla_lebai_1000_loss.png"
fig.savefig(target, dpi=160)
print(target, "points", len(points), "first", points[0], "last", points[-1])

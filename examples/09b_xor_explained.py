"""examples/09b_xor_explained.py

Visual explanation of why LinUCB cannot learn the XOR reward pattern
and why NeuralLinUCB can.

Run:
    python examples/09b_xor_explained.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)
N = 600
x0 = np.random.uniform(0, 1, N)   # tiredness
x1 = np.random.uniform(0, 1, N)   # coldness

diagonal = (x0 > 0.5) == (x1 > 0.5)
colors   = ['#2196F3' if d else '#FF9800' for d in diagonal]
BLUE, ORANGE = '#2196F3', '#FF9800'

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Why LinUCB cannot learn XOR — and why NeuralLinUCB can",
    fontsize=14, fontweight="bold", y=1.02,
)

# ── Panel 1: The four quadrants ───────────────────────────────────────────────
ax = axes[0]

# Shade quadrants
ax.add_patch(patches.Rectangle((0.0, 0.5), 0.5, 0.5, color=ORANGE, alpha=0.12))  # top-left:  arm_b
ax.add_patch(patches.Rectangle((0.5, 0.5), 0.5, 0.5, color=BLUE,   alpha=0.12))  # top-right: arm_a
ax.add_patch(patches.Rectangle((0.0, 0.0), 0.5, 0.5, color=BLUE,   alpha=0.12))  # bot-left:  arm_a
ax.add_patch(patches.Rectangle((0.5, 0.0), 0.5, 0.5, color=ORANGE, alpha=0.12))  # bot-right: arm_b

ax.scatter(x0, x1, c=colors, alpha=0.55, s=22, edgecolors="none")
ax.axvline(0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.4)
ax.axhline(0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.4)

ax.text(0.25, 0.75, "arm_b", ha="center", va="center",
        fontsize=13, fontweight="bold", color=ORANGE)
ax.text(0.75, 0.75, "arm_a", ha="center", va="center",
        fontsize=13, fontweight="bold", color=BLUE)
ax.text(0.25, 0.25, "arm_a", ha="center", va="center",
        fontsize=13, fontweight="bold", color=BLUE)
ax.text(0.75, 0.25, "arm_b", ha="center", va="center",
        fontsize=13, fontweight="bold", color=ORANGE)

ax.set_xlabel("x0 — tired  (0 = awake, 1 = exhausted)", fontsize=10)
ax.set_ylabel("x1 — cold   (0 = warm,  1 = freezing)",  fontsize=10)
ax.set_title("The XOR reward pattern\narm_a = diagonal, arm_b = off-diagonal", fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE,   label="arm_a (espresso)"),
                   Patch(color=ORANGE, label="arm_b (hot chocolate)")],
          loc="upper center", fontsize=9, bbox_to_anchor=(0.5, -0.14), ncol=2)


# ── Panel 2: Every straight line fails ────────────────────────────────────────
ax = axes[1]

ax.add_patch(patches.Rectangle((0.0, 0.5), 0.5, 0.5, color=ORANGE, alpha=0.12))
ax.add_patch(patches.Rectangle((0.5, 0.5), 0.5, 0.5, color=BLUE,   alpha=0.12))
ax.add_patch(patches.Rectangle((0.0, 0.0), 0.5, 0.5, color=BLUE,   alpha=0.12))
ax.add_patch(patches.Rectangle((0.5, 0.0), 0.5, 0.5, color=ORANGE, alpha=0.12))

ax.scatter(x0, x1, c=colors, alpha=0.45, s=22, edgecolors="none")
ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.25)
ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.25)

x_line = np.linspace(0, 1, 200)

# Attempt 1: horizontal
ax.axhline(0.5, color="#E53935", linewidth=2.2, label="x₁ = 0.5  ✗")
ax.text(0.02, 0.53, "✗ both colours above AND below", fontsize=8, color="#E53935")

# Attempt 2: vertical
ax.axvline(0.5, color="#B71C1C", linewidth=2.2, linestyle="dashdot", label="x₀ = 0.5  ✗")
ax.text(0.52, 0.04, "✗ both colours\nleft AND right", fontsize=8, color="#B71C1C")

# Attempt 3: diagonal x1 = x0
ax.plot(x_line, x_line, color="#880E4F", linewidth=2.2,
        linestyle=(0, (5, 2)), label="x₁ = x₀  ✗")
ax.text(0.62, 0.56, "✗ mixes both\ncolours", fontsize=8, color="#880E4F")

# Attempt 4: anti-diagonal x1 = 1 - x0
ax.plot(x_line, 1 - x_line, color="#4A148C", linewidth=2.2,
        linestyle=(0, (3, 1, 1, 1)), label="x₁ = 1−x₀  ✗")
ax.text(0.03, 0.40, "✗ same problem", fontsize=8, color="#4A148C")

ax.set_xlabel("x0 — tired", fontsize=10)
ax.set_ylabel("x1 — cold",  fontsize=10)
ax.set_title("Every straight line leaves both\nclasses on both sides", fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper center", fontsize=8.5, bbox_to_anchor=(0.5, -0.14), ncol=2)


# ── Panel 3: What NeuralLinUCB does ───────────────────────────────────────────
ax = axes[2]

ax.add_patch(patches.Rectangle((0.0, 0.5), 0.5, 0.5, color=ORANGE, alpha=0.12))
ax.add_patch(patches.Rectangle((0.5, 0.5), 0.5, 0.5, color=BLUE,   alpha=0.12))
ax.add_patch(patches.Rectangle((0.0, 0.0), 0.5, 0.5, color=BLUE,   alpha=0.12))
ax.add_patch(patches.Rectangle((0.5, 0.0), 0.5, 0.5, color=ORANGE, alpha=0.12))

ax.scatter(x0, x1, c=colors, alpha=0.45, s=22, edgecolors="none")

# The two correct boundaries — NeuralLinUCB learns to use BOTH
ax.axvline(0.5, color="#1B5E20", linewidth=2.8, label="Boundary 1: x₀ = 0.5")
ax.axhline(0.5, color="#1B5E20", linewidth=2.8, linestyle="dashed", label="Boundary 2: x₁ = 0.5")

ax.text(0.25, 0.75, "arm_b", ha="center", va="center",
        fontsize=13, fontweight="bold", color=ORANGE, alpha=0.7)
ax.text(0.75, 0.75, "arm_a", ha="center", va="center",
        fontsize=13, fontweight="bold", color=BLUE, alpha=0.7)
ax.text(0.25, 0.25, "arm_a", ha="center", va="center",
        fontsize=13, fontweight="bold", color=BLUE, alpha=0.7)
ax.text(0.75, 0.25, "arm_b", ha="center", va="center",
        fontsize=13, fontweight="bold", color=ORANGE, alpha=0.7)

ax.text(0.5, -0.08,
        "LinUCB gets ONE line.\nNeuralLinUCB's hidden layer computes BOTH — XOR solved.",
        ha="center", va="top", fontsize=9.5, color="#1B5E20",
        transform=ax.transData, clip_on=False)

ax.set_xlabel("x0 — tired", fontsize=10)
ax.set_ylabel("x1 — cold",  fontsize=10)
ax.set_title("NeuralLinUCB learns TWO boundaries\none per hidden neuron", fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper center", fontsize=9, bbox_to_anchor=(0.5, -0.14), ncol=2)


plt.tight_layout(rect=[0, 0.04, 1, 1])
out = os.path.join(SCRIPT_DIR, "09b_xor_explained.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()

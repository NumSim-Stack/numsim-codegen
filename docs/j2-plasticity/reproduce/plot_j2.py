import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("/tmp/sc/j2_data.csv")
SY, H, G = 250.0, 5000.0, 80000.0
modes = ["uniaxial", "shear", "equibiaxial"]
labels = {"uniaxial": "Uniaxial  (σ₁₁ vs ε₁₁)",
          "shear": "Simple shear  (σ₁₂ vs γ)",
          "equibiaxial": "Equibiaxial  (σ₁₁ vs ε₁₁)"}
colors = {"uniaxial": "#2563eb", "shear": "#e11d48", "equibiaxial": "#059669"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

# Panel 1 — physical component stress vs drive strain
for m in modes:
    d = df[df["mode"] == m].sort_values("drive")
    ax1.plot(d["drive"] * 100, d["sig"], color=colors[m], lw=2, label=labels[m])
ax1.set_xlabel("drive strain  [%]")
ax1.set_ylabel("stress component  [MPa]")
ax1.set_title("Stress–strain per loading mode")
ax1.legend(frameon=False, fontsize=9)
ax1.grid(alpha=0.25)

# Panel 2 — von Mises equivalent stress: the J2 signature (common yield level)
for m in modes:
    d = df[df["mode"] == m].sort_values("drive")
    ax2.plot(d["drive"] * 100, d["vm"], color=colors[m], lw=2, label=m)
vm_yield = (1.5 ** 0.5) * SY
ax2.axhline(vm_yield, ls="--", color="#6b7280", lw=1)
ax2.text(0.02, vm_yield + 4, f"first yield  √(3/2)·σᵧ ≈ {vm_yield:.0f} MPa",
         fontsize=8, color="#374151")
ax2.set_xlabel("drive strain  [%]")
ax2.set_ylabel("von Mises equivalent stress  [MPa]")
ax2.set_title("von Mises stress — all modes yield at the same level")
ax2.legend(frameon=False, fontsize=9)
ax2.grid(alpha=0.25)

fig.suptitle(
    "Generated J2 radial-return material  (G=80 GPa, σᵧ=250 MPa, H=5 GPa)  "
    "— codegen → numsim-materials → backward_euler",
    fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("/tmp/sc/j2_stress_strain.png", dpi=130)
print("saved")

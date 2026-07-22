import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("/tmp/sc/nl_data.csv")
info = {
    "linear": ("Linear      σy = σy0 + H·Δγ",              "#2563eb"),
    "voce":   ("Voce         σy = σy0 + Q(1 − e^{-bΔγ})",  "#e11d48"),
    "swift":  ("Swift        σy = C(ε0 + Δγ)^k",            "#059669"),
}
fig, ax = plt.subplots(figsize=(8.2, 5.2))
for law, (label, c) in info.items():
    d = df[df["law"] == law].sort_values("eps")
    ax.plot(d["eps"] * 100, d["vm"], color=c, lw=2.2, label=label)

vm_y = (1.5 ** 0.5) * 250
ax.axhline(vm_y, ls="--", color="#9ca3af", lw=1)
ax.text(1.05, vm_y - 26, f"initial yield ≈ {vm_y:.0f} MPa", fontsize=8, color="#4b5563")
ax.annotate("shared elastic branch\n+ same initial yield", xy=(0.2, vm_y),
            xytext=(0.55, 150), fontsize=8, color="#374151",
            arrowprops=dict(arrowstyle="->", color="#9ca3af"))
ax.annotate("saturates (slope → 0)", xy=(1.85, 660), xytext=(1.0, 610),
            fontsize=8, color="#e11d48",
            arrowprops=dict(arrowstyle="->", color="#e11d48"))

ax.set_xlabel("uniaxial strain  [%]")
ax.set_ylabel("von Mises (hardening) stress  [MPa]")
ax.set_title("Generated J2 material — nonlinear isotropic hardening laws\n"
             "uniaxial, codegen → numsim-materials → backward_euler  (G=80 GPa)",
             fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.grid(alpha=0.25)
ax.set_xlim(left=0)
fig.tight_layout()
fig.savefig("/tmp/sc/j2_nonlinear_hardening.png", dpi=130)
print("saved")

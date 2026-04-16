"""
Monte-Carlo simulation illustrating how audit filtering recovers the
true persona effect-size estimator sigma_E under rewrite-noise corruption.

Corruption replaces genuine persona-dependent tiers with random base-rate
draws, diluting the systematic signal.  The audit filter (with known
sensitivity/specificity) removes most corrupted observations, recovering
sigma_E close to its true value at the cost of lower retention.

Produces: output/fig__simulation__rewrite_noise_attenuation__v1.{pdf,png}
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── style constants (match project figure_generator conventions) ──────────
_GRAYS = [
    "#1a1a1a", "#4d4d4d", "#808080", "#b3b3b3",
    "#333333", "#666666", "#999999", "#cccccc",
]
DPI = 300
OUTPUT_DIR = Path("output")

# ── simulation parameters ────────────────────────────────────────────────
rng = np.random.default_rng(0)
n, K, reps = 2_000, 36, 200          # 200 reps for smooth curves
base = np.array([0.78, 0.10, 0.12])  # tier 0 / 1 / 2 base rates
c1, c2, beta_sd = -0.8, 1.2, 0.18   # ordinal thresholds + true effect SD
sens, spec = 0.85, 0.98              # audit sensitivity / specificity
qs = np.linspace(0, 0.30, 31)        # corruption-rate sweep


def sigma_e(T: np.ndarray) -> float:
    """SD of persona-level mean tier (effect-size proxy)."""
    return float(np.nanmean(T, axis=0).std(ddof=1))


# ── run simulation ───────────────────────────────────────────────────────
full_mean: list[float] = []
full_lo: list[float] = []
full_hi: list[float] = []
hf_mean: list[float] = []
hf_lo: list[float] = []
hf_hi: list[float] = []
retain_mean: list[float] = []

for q in qs:
    f_vals, h_vals, r_vals = [], [], []
    for _ in range(reps):
        alpha = rng.normal(0, 1.0, size=(n, 1))
        beta = rng.normal(0, beta_sd, size=(1, K))
        Z = alpha + beta + rng.normal(0, 0.6, size=(n, K))
        T = (Z > c1).astype(int) + (Z > c2).astype(int)

        corrupted = rng.random((n, K)) < q
        T_obs = np.where(corrupted, rng.choice(3, size=(n, K), p=base), T)

        pass_audit = np.where(
            corrupted,
            rng.random((n, K)) > sens,
            rng.random((n, K)) < spec,
        )
        T_hf = np.where(pass_audit, T_obs, np.nan)

        f_vals.append(sigma_e(T_obs))
        h_vals.append(sigma_e(T_hf))
        r_vals.append(float(pass_audit.mean()))

    f_arr = np.array(f_vals)
    h_arr = np.array(h_vals)

    full_mean.append(float(f_arr.mean()))
    full_lo.append(float(np.percentile(f_arr, 2.5)))
    full_hi.append(float(np.percentile(f_arr, 97.5)))
    hf_mean.append(float(h_arr.mean()))
    hf_lo.append(float(np.percentile(h_arr, 2.5)))
    hf_hi.append(float(np.percentile(h_arr, 97.5)))
    retain_mean.append(float(np.mean(r_vals)))

true_sigma = full_mean[0]

# ── figure ────────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(10, 4), gridspec_kw={"wspace": 0.35},
)

# ── panel (a): sigma_E vs corruption rate ─────────────────────────────────

# shaded band: region of signal loss (true -> unfiltered)
ax_a.fill_between(
    qs, full_mean, true_sigma,
    color=_GRAYS[7], alpha=0.40, label="Signal dilution",
)

# 95% simulation bands
ax_a.fill_between(qs, full_lo, full_hi, color=_GRAYS[5], alpha=0.18)
ax_a.fill_between(qs, hf_lo, hf_hi, color=_GRAYS[3], alpha=0.18)

# mean curves
ax_a.plot(
    qs, full_mean,
    color=_GRAYS[0], linewidth=1.8, linestyle="-",
    marker="o", markersize=3, markevery=5,
    label="Unfiltered",
)
ax_a.plot(
    qs, hf_mean,
    color=_GRAYS[4], linewidth=1.8, linestyle="--",
    marker="s", markersize=3, markevery=5,
    label="Audit-filtered",
)
ax_a.axhline(
    true_sigma, color=_GRAYS[2], linewidth=1.0,
    linestyle=":", zorder=0,
    label=r"True $\sigma_E$ ($q = 0$)",
)

ax_a.set_xlabel("Corruption rate  $q$", fontsize=11)
ax_a.set_ylabel(r"$\sigma_E$  (SD of persona-level mean tier)", fontsize=11)
ax_a.set_xlim(qs[0], qs[-1])
ax_a.set_ylim(
    min(full_lo) * 0.96,
    true_sigma * 1.04,
)
ax_a.legend(fontsize=8.5, loc="lower left", framealpha=0.92, edgecolor="#cccccc")
ax_a.tick_params(labelsize=9)
ax_a.set_title("(a)  Signal dilution under corruption", fontsize=11, fontweight="bold")

# ── panel (b): retention rate vs corruption rate ──────────────────────────
ax_b.plot(
    qs, retain_mean,
    color=_GRAYS[0], linewidth=1.8, linestyle="-",
    marker="D", markersize=3, markevery=5,
)
ax_b.axhline(
    spec, color=_GRAYS[2], linewidth=1.0,
    linestyle=":", label=f"Specificity = {spec:.2f}",
)
ax_b.set_xlabel("Corruption rate  $q$", fontsize=11)
ax_b.set_ylabel("Retention rate (fraction passing audit)", fontsize=11)
ax_b.set_xlim(qs[0], qs[-1])
ax_b.set_ylim(0.70, 1.01)
ax_b.legend(fontsize=8.5, loc="lower left", framealpha=0.92, edgecolor="#cccccc")
ax_b.tick_params(labelsize=9)
ax_b.set_title("(b)  Audit retention", fontsize=11, fontweight="bold")

# ── save ──────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
stem = "fig__simulation__rewrite_noise_attenuation__v1"

for fmt in ("pdf", "png"):
    fig.savefig(
        OUTPUT_DIR / f"{stem}.{fmt}",
        format=fmt,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

plt.close(fig)
print(f"Saved  {OUTPUT_DIR / stem}.{{pdf,png}}")
print(f"  true sigma_E = {true_sigma:.4f}")
print(f"  sigma_E at q=0.30 (unfiltered)  = {full_mean[-1]:.4f}")
print(f"  sigma_E at q=0.30 (audit-filt)  = {hf_mean[-1]:.4f}")
print(f"  retention at q=0.30 = {retain_mean[-1]:.3f}")

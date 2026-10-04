"""Run a fair mechanistic-ABM versus independent-RWR comparison."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.sankey import Sankey

from light_response import compute_experimental_response, normalize_sequence_response
from random_walk_resetting import RWRConfig, run_random_walk_resetting


ROOT = Path(__file__).resolve().parent


def _load_abm_module():
    """Load the existing hyphenated ABM script without renaming it."""
    path = ROOT / "merged-abm-full-V.py"
    spec = importlib.util.spec_from_file_location("merged_abm_full_v", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ABM module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_fluorescence_variants(path: Path, normalization_mode: str) -> List[Dict[str, Any]]:
    """Load fluorescence CSV and calculate D_exp, D_tilde, and unclipped values.

    Blank fluorescence cells are rejected because no experimental value may be
    invented. ``batch_max`` is accepted only as an explicitly selected
    exploratory normalization mode.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("fluorescence CSV contains no variants")
    for row in rows:
        if not row.get("F_dark", "").strip() or not row.get("F_light", "").strip():
            raise ValueError(f"Missing fluorescence for variant {row.get('variant', '<unnamed>')}")
        row["D_exp"] = compute_experimental_response(float(row["F_dark"]), float(row["F_light"]))
    refs = [r for r in rows if r.get("is_reference", "").strip().lower() == "true"]
    positives = [r for r in rows if r.get("is_positive_control", "").strip().lower() == "true"]
    if len(refs) != 1:
        raise ValueError("exactly one is_reference=true row is required")
    d_ref = float(refs[0]["D_exp"])
    if normalization_mode == "fixed_positive":
        if len(positives) != 1:
            raise ValueError("fixed_positive mode requires exactly one positive control")
        d_positive = float(positives[0]["D_exp"])
    elif normalization_mode == "batch_max":
        d_positive = max(float(r["D_exp"]) for r in rows)
    else:
        raise ValueError("normalization_mode must be fixed_positive or batch_max")
    for row in rows:
        clipped, raw = normalize_sequence_response(
            float(row["D_exp"]), d_ref, d_positive, return_unclipped=True
        )
        row["D_tilde"] = clipped
        row["D_tilde_unclipped"] = raw
        row["speed_um_s"] = float(row["speed_um_s"]) if row.get("speed_um_s", "").strip() else None
    return rows


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def _write_rows(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _comparison_row(model: str, variant: str, light_condition: str, metrics: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    row = {
        "model": model,
        "variant": variant,
        "light_condition": light_condition,
        "D_exp": extra.get("D_exp"),
        "D_tilde": extra.get("D_tilde"),
        "D_tilde_unclipped": extra.get("D_tilde_unclipped"),
        "I_light": extra.get("I_light"),
        "response_factor_f": extra.get("response_factor_f"),
        "anisotropy_mode": extra.get("anisotropy_mode"),
        "reset_rate": extra.get("reset_rate", 0.0),
        "speed": extra.get("speed"),
        "n_agents": metrics["n_agents"] if "n_agents" in metrics else metrics["N"],
        "hit_rate": metrics["hit_rate"],
        "conditional_MFPT": metrics["conditional_MFPT"],
        "penetration_efficiency": metrics["penetration_efficiency"],
        "penetration_rate": metrics["penetration_rate"],
        "target_arrival_rate": metrics["target_arrival_rate"],
        "mean_x_displacement": metrics["mean_x_displacement"],
        "directionality": metrics["directionality"],
        "trap_rate": metrics["trap_rate"],
        "seed": metrics.get("seed", extra.get("seed")),
    }
    return row


def _plot_results(
    rows: List[Dict[str, Any]],
    fig_dir: Path,
    abm_module: Any,
    config: Dict[str, Any],
    spatial_diagnostics: Dict[str, Any],
    peptide_grid_results: List[Dict[str, Any]],
) -> None:
    """Generate five publication-style comparison figures (PNG and vector PDF)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "blue": "#2F5597", "teal": "#2A9D8F", "orange": "#E76F51",
        "gold": "#E9C46A", "ink": "#263238", "muted": "#6B7280", "grid": "#D8DEE9",
    }
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10.5, "axes.titlesize": 13,
        "axes.labelsize": 11, "axes.titleweight": "bold", "axes.edgecolor": "#AAB2BF",
        "axes.linewidth": 0.8, "xtick.color": colors["ink"], "ytick.color": colors["ink"],
        "text.color": colors["ink"], "figure.facecolor": "white", "axes.facecolor": "#FBFCFE",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
    })

    def polish(ax, *, grid_axis: str = "y"):
        ax.grid(axis=grid_axis, color=colors["grid"], linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def save(fig, filename: str):
        fig.savefig(fig_dir / f"{filename}.png", dpi=300)
        fig.savefig(fig_dir / f"{filename}.pdf")
        plt.close(fig)

    def short_label(row: Dict[str, Any]) -> str:
        if row["model"] == "mechanistic_ABM":
            return "Dark" if row["variant"] == "dark" else str(row["variant"])
        return "RW  ·  r=0" if float(row["reset_rate"]) == 0.0 else f"RWR  ·  r={float(row['reset_rate']):g}"

    base = abm_module.HyperParams()
    for key, value in config.get("abm_overrides", {}).items():
        setattr(base, key, value)

    d_grid = np.linspace(0.0, 1.0, 101)
    gate = float(config.get("light_gate_for_mapping_figure", 1.0))
    intensity = float(config.get("I_light", 1.0))
    u_eff = d_grid * intensity * gate
    c_inf = base.C_INF_DARK + (base.C_INF_LIT - base.C_INF_DARK) * u_eff
    cw = c_inf**base.C_HILL_N / (base.C_HALF**base.C_HILL_N + c_inf**base.C_HILL_N)
    lam = base.LAMBDA_MIN + (base.LAMBDA_MAX - base.LAMBDA_MIN) * cw
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.15), gridspec_kw={"wspace": 0.28})
    ax1.plot(d_grid, u_eff, color=colors["blue"], lw=2.5, label=r"Effective input $u_{eff}$")
    ax1.plot(d_grid, c_inf, color=colors["teal"], lw=2.5, label=r"Steady state $c_{\infty}$")
    ax1.fill_between(d_grid, u_eff, alpha=0.08, color=colors["blue"])
    ax1.set(xlabel=r"Normalized sequence response $\tilde{D}$", ylabel="Normalized level", xlim=(0, 1))
    ax1.legend(frameon=False, loc="center left"); polish(ax1)
    ax2.plot(d_grid, lam, color=colors["orange"], lw=2.8)
    ax2.fill_between(d_grid, lam, np.nanmin(lam), color=colors["orange"], alpha=0.10)
    ax2.set(xlabel=r"Normalized sequence response $\tilde{D}$", ylabel=r"Tumble rate $\lambda$ (s$^{-1}$)", xlim=(0, 1))
    polish(ax2)
    fig.suptitle("Upstream response mapping", x=0.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.91, r"$u_{eff}=\tilde{D}I_{light}u(t)$; downstream mapping is unchanged", color=colors["muted"], fontsize=9.5)
    fig.subplots_adjust(top=0.78)
    save(fig, "figure1_light_response_mapping")

    for filename, key, ylabel in (
        ("figure2_abm_vs_rwr_mfpt.png", "conditional_MFPT", "Conditional MFPT (s)"),
        ("figure3_abm_vs_rwr_penetration_efficiency.png", "penetration_efficiency", r"Penetration efficiency (s$^{-1}$)"),
    ):
        values = np.array([float(r[key]) if r[key] is not None else np.nan for r in rows])
        labels = [short_label(r) for r in rows]
        y = np.arange(len(rows))[::-1]
        point_colors = [colors["blue"] if r["model"] == "mechanistic_ABM" else colors["orange"] for r in rows]
        fig, ax = plt.subplots(figsize=(8.4, 5.5))
        ax.axhspan(3.5, 9.5, color=colors["blue"], alpha=0.035)
        ax.axhspan(-0.5, 3.5, color=colors["orange"], alpha=0.045)
        for i, value in enumerate(values):
            yy = y[i]
            if np.isfinite(value):
                ax.hlines(yy, 0.0, value, color=point_colors[i], lw=2.2, alpha=0.32)
                ax.scatter(value, yy, s=62, color=point_colors[i], edgecolor="white", linewidth=0.8, zorder=3)
                fmt = f"{value:.2f}" if key == "conditional_MFPT" else f"{value:.4f}"
                ax.annotate(fmt, (value, yy), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
            else:
                ax.text(0.0, yy, "No hit", va="center", color=colors["muted"], fontsize=9)
        ax.set_yticks(y, labels); ax.set_xlabel(ylabel); ax.set_ylabel("")
        ax.set_title("First-passage comparison" if key == "conditional_MFPT" else "Transport efficiency comparison", loc="left", pad=14)
        ax.text(0.995, 1.02, "Mechanistic ABM  ●     Random-walk baseline  ●", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.8, color=colors["muted"])
        polish(ax, grid_axis="x"); ax.set_xlim(left=0.0)
        fig.tight_layout(); save(fig, filename.removesuffix(".png"))

    rwr = sorted((r for r in rows if r["model"] in {"RWR", "ordinary_random_walk"}), key=lambda r: float(r["reset_rate"]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True, gridspec_kw={"hspace": 0.12})
    rates = np.array([float(r["reset_rate"]) for r in rwr])
    mfpt = np.array([float(r["conditional_MFPT"]) if r["conditional_MFPT"] is not None else np.nan for r in rwr])
    hit = np.array([float(r["hit_rate"]) for r in rwr])
    ax1.plot(rates, mfpt, "o-", color=colors["blue"], lw=2.3, ms=6)
    ax1.set_ylabel("Conditional MFPT (s)"); polish(ax1)
    ax2.plot(rates, hit, "s-", color=colors["orange"], lw=2.3, ms=5.5)
    ax2.fill_between(rates, 0, hit, color=colors["orange"], alpha=0.08)
    ax2.set(xlabel=r"Reset rate $r$ (s$^{-1}$)", ylabel="Hit rate", ylim=(0, max(0.18, 1.12 * float(np.max(hit)))))
    polish(ax2)
    fig.suptitle("RWR reset-rate sensitivity", x=0.10, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.10, 0.925, "MFPT is conditional on successful arrivals; hit rate must be read jointly", color=colors["muted"], fontsize=9)
    fig.subplots_adjust(top=0.86); save(fig, "figure4_rwr_reset_rate_sensitivity")

    abm_rows = [r for r in rows if r["model"] == "mechanistic_ABM" and r["variant"] != "dark"]
    d_values = np.array([float(r["D_tilde"]) for r in abm_rows])
    mfpt_values = np.array([float(r["conditional_MFPT"]) for r in abm_rows])
    hit_values = np.array([float(r["hit_rate"]) for r in abm_rows])
    labels = [str(r["variant"]) for r in abm_rows]
    cmap = plt.get_cmap("viridis")
    point_colors = cmap(0.12 + 0.78 * d_values)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5), gridspec_kw={"wspace": 0.28})
    ax1.plot(d_values, mfpt_values, color="#AAB2BF", lw=1.4, zorder=1)
    ax1.scatter(d_values, mfpt_values, s=90, c=point_colors, edgecolor="white", linewidth=1.0, zorder=2)
    ax1.set(xlabel=r"Sequence response $\tilde{D}$", ylabel="Conditional MFPT (s)", xlim=(-0.04, 1.04))
    polish(ax1)
    ax2.plot(d_values, hit_values, color="#AAB2BF", lw=1.4, zorder=1)
    ax2.scatter(d_values, hit_values, s=90, c=point_colors, edgecolor="white", linewidth=1.0, zorder=2)
    hit_pad = max(0.01, 0.25 * (float(np.max(hit_values)) - float(np.min(hit_values)) + 0.01))
    ax2.set(xlabel=r"Sequence response $\tilde{D}$", ylabel="Target hit rate", xlim=(-0.04, 1.04),
            ylim=(max(0.0, float(np.min(hit_values)) - hit_pad), min(1.01, float(np.max(hit_values)) + hit_pad)))
    polish(ax2)
    for ax, yy in ((ax1, mfpt_values), (ax2, hit_values)):
        for x_value, y_value, label in zip(d_values, yy, labels):
            offset = (5, 7) if label != "Positive-Control" else (-5, 7)
            ax.annotate(label, (x_value, y_value), xytext=offset, textcoords="offset points",
                        ha="left" if offset[0] > 0 else "right", fontsize=8.5)
    fig.suptitle("Predicted response of linker variants", x=0.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.91, "Illustrative response amplitudes; replace with measured fluorescence ratios", color=colors["muted"], fontsize=9.2)
    fig.subplots_adjust(top=0.79); save(fig, "figure5_abm_variant_predictions")

    def relative_scores(data: np.ndarray, lower_is_better: set[int]) -> np.ndarray:
        """Column-wise min-max scores used only for heatmap colour."""
        scores = np.full_like(data, np.nan, dtype=float)
        for column in range(data.shape[1]):
            values = data[:, column]
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            low, high = float(np.min(values[finite])), float(np.max(values[finite]))
            scaled = np.full(values.shape, np.nan)
            scaled[finite] = 0.5 if np.isclose(low, high) else (values[finite] - low) / (high - low)
            if column in lower_is_better:
                scaled[finite] = 1.0 - scaled[finite]
            scores[:, column] = scaled
        return scores

    # Figure 6: all-model multi-metric heatmap. Colour is relative within each
    # column; annotations retain the raw values to prevent scale ambiguity.
    heat_keys = ["hit_rate", "conditional_MFPT", "penetration_efficiency", "penetration_rate", "mean_x_displacement", "directionality"]
    heat_labels = ["Hit rate ↑", "MFPT (s) ↓", "Efficiency ↑", "Penetration ↑", "Δx (µm) ↑", "Directionality ↑"]
    raw = np.array([[float(row[key]) if row[key] is not None else np.nan for key in heat_keys] for row in rows])
    score = relative_scores(raw, {1})
    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    image = ax.imshow(score, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(heat_labels)), heat_labels)
    ax.set_yticks(np.arange(len(rows)), [short_label(row) for row in rows])
    ax.tick_params(axis="x", rotation=25)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            value = raw[i, j]
            label = "—" if not np.isfinite(value) else (f"{value:.2f}" if j in {1, 4} else f"{value:.3f}")
            text_color = "white" if np.isfinite(score[i, j]) and score[i, j] > 0.62 else colors["ink"]
            ax.text(j, i, label, ha="center", va="center", fontsize=8.4, color=text_color,
                    fontweight="bold" if np.isfinite(score[i, j]) and score[i, j] > 0.82 else "normal")
    abm_count = sum(row["model"] == "mechanistic_ABM" for row in rows)
    ax.axhline(abm_count - 0.5, color="white", lw=3.0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Column-wise relative score\n(higher colour = more favourable)", fontsize=9)
    ax.set_title("Multi-metric performance map", loc="left", pad=30)
    ax.text(0.0, 1.025, "Cell labels are raw values; arrows show the preferred direction",
            transform=ax.transAxes, color=colors["muted"], fontsize=9.2)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(); save(fig, "figure6_model_metric_heatmap")

    # Figure 7: RWR-only heatmap, making the joint MFPT/hit-rate trade-off easy
    # to scan without implying that conditional MFPT alone is sufficient.
    rwr_keys = ["conditional_MFPT", "hit_rate", "penetration_efficiency", "penetration_rate"]
    rwr_labels = ["MFPT (s) ↓", "Hit rate ↑", "Efficiency ↑", "Penetration ↑"]
    rwr_raw = np.array([[float(row[key]) if row[key] is not None else np.nan for key in rwr_keys] for row in rwr])
    rwr_score = relative_scores(rwr_raw, {0})
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    image = ax.imshow(rwr_score, cmap="mako" if "mako" in plt.colormaps() else "PuBuGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(rwr_labels)), rwr_labels)
    ax.set_yticks(np.arange(len(rwr)), [short_label(row) for row in rwr])
    for i in range(rwr_raw.shape[0]):
        for j in range(rwr_raw.shape[1]):
            value = rwr_raw[i, j]
            label = "—" if not np.isfinite(value) else (f"{value:.2f}" if j == 0 else f"{value:.3f}")
            text_color = "white" if np.isfinite(rwr_score[i, j]) and rwr_score[i, j] > 0.55 else colors["ink"]
            ax.text(j, i, label, ha="center", va="center", fontsize=9.2, color=text_color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Relative desirability", fontsize=9)
    ax.set_title("RWR sensitivity heatmap", loc="left", pad=28)
    ax.text(0.0, 1.035, "Read MFPT together with hit and penetration rates",
            transform=ax.transAxes, color=colors["muted"], fontsize=9.2)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(); save(fig, "figure7_rwr_sensitivity_heatmap")

    # Figure 8: reference-PDF-style XY occupancy comparison with a shared scale.
    if {"dark", "strong"}.issubset(spatial_diagnostics):
        dark_diag = spatial_diagnostics["dark"]
        strong_diag = spatial_diagnostics["strong"]
        dark_occ = np.log1p(np.asarray(dark_diag["occupancy_xy"], dtype=float))
        strong_occ = np.log1p(np.asarray(strong_diag["occupancy_xy"], dtype=float))
        vmax = max(float(np.max(dark_occ)), float(np.max(strong_occ)), 1e-12)
        extent = [base.X_MIN, base.X_MAX, base.Y_MIN, base.Y_MAX]
        fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.3), sharex=True, sharey=True,
                                 gridspec_kw={"hspace": 0.10})
        panels = [
            (strong_occ, f"Blue light · {spatial_diagnostics['strong_label']}", colors["blue"]),
            (dark_occ, "Dark condition", colors["ink"]),
        ]
        last_image = None
        for ax, (occupancy, label, label_color) in zip(axes, panels):
            last_image = ax.imshow(occupancy, origin="lower", aspect="auto", extent=extent,
                                   cmap="viridis", vmin=0.0, vmax=vmax, interpolation="bilinear")
            ax.axvline(base.REG_A_END, color="white", lw=1.2, ls="--", alpha=0.9)
            ax.axvline(base.REG_B_END, color="white", lw=1.2, ls="--", alpha=0.9)
            ax.axvline(base.X_TARGET, color=colors["gold"], lw=1.8, ls=":", alpha=0.95)
            ax.text(0.015, 0.90, label, transform=ax.transAxes, color="white", fontsize=11,
                    fontweight="bold", bbox={"boxstyle": "round,pad=0.28", "facecolor": label_color,
                                             "edgecolor": "none", "alpha": 0.86})
            ax.set_ylabel("y (µm)")
            for spine in ax.spines.values(): spine.set_visible(False)
        axes[-1].set_xlabel("x (µm)")
        axes[-1].text(base.REG_A_END, base.Y_MIN, " A/B interface ", color="white", ha="center", va="bottom", fontsize=8)
        axes[-1].text(base.REG_B_END, base.Y_MIN, " B/C interface ", color="white", ha="center", va="bottom", fontsize=8)
        axes[-1].text(base.X_TARGET, base.Y_MIN, " target ", color=colors["gold"], ha="center", va="bottom", fontsize=8)
        cbar = fig.colorbar(last_image, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("log(1 + accumulated occupancy)")
        fig.suptitle("Spatial occupancy across the mucus domain", x=0.08, ha="left",
                     fontsize=15, fontweight="bold")
        fig.text(0.08, 0.925, "Shared colour scale: brighter regions contain more accumulated agent visits",
                 color=colors["muted"], fontsize=9.2)
        fig.subplots_adjust(top=0.86, right=0.90)
        save(fig, "figure8_xy_occupancy_comparison")

    # Figure 9: contrast-focused ABM comparison relative to CheZ-LOV. Every
    # cell is oriented so positive values mean improvement over the reference.
    abm_all = [row for row in rows if row["model"] == "mechanistic_ABM"]
    reference = next((row for row in abm_all if row["variant"] == "CheZ-LOV"), None)
    compared = [row for row in abm_all if row["variant"] not in {"CheZ-LOV", "dark"}]
    if reference is not None and compared:
        ref_mfpt = float(reference["conditional_MFPT"])
        ref_eff = float(reference["penetration_efficiency"])
        ref_dx = float(reference["mean_x_displacement"])
        ref_dir = float(reference["directionality"])
        delta = np.array([
            [
                100.0 * (ref_mfpt - float(row["conditional_MFPT"])) / ref_mfpt,
                100.0 * (float(row["penetration_efficiency"]) - ref_eff) / ref_eff,
                100.0 * (float(row["hit_rate"]) - float(reference["hit_rate"])),
                100.0 * (float(row["penetration_rate"]) - float(reference["penetration_rate"])),
                100.0 * (float(row["mean_x_displacement"]) - ref_dx) / abs(ref_dx),
                100.0 * (float(row["directionality"]) - ref_dir) / max(abs(ref_dir), 1e-12),
            ]
            for row in compared
        ])
        delta_labels = ["MFPT improvement (%)", "Efficiency gain (%)", "Hit-rate gain (pp)",
                        "Penetration gain (pp)", "Δx change (%)", "Directionality change (%)"]
        limit = max(1.0, float(np.nanmax(np.abs(delta))))
        fig, ax = plt.subplots(figsize=(10.6, 4.3))
        image = ax.imshow(delta, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(np.arange(len(delta_labels)), delta_labels, rotation=24, ha="right")
        ax.set_yticks(np.arange(len(compared)), [str(row["variant"]) for row in compared])
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                value = delta[i, j]
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(value) > 0.55 * limit else colors["ink"], fontweight="bold")
        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("Improvement relative to CheZ-LOV")
        ax.set_title("Sequence-level contrast relative to CheZ-LOV", loc="left", pad=28)
        ax.text(0.0, 1.035, "Blue = improvement; red = deterioration; values preserve signs and units",
                transform=ax.transAxes, color=colors["muted"], fontsize=9.2)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.tight_layout(); save(fig, "figure9_abm_relative_contrast_heatmap")

    # Figure 10: RWR degradation relative to ordinary random walk. Positive
    # values consistently mean worse performance, making reset effects obvious.
    rw_reference = next((row for row in rwr if float(row["reset_rate"]) == 0.0), None)
    reset_rows = [row for row in rwr if float(row["reset_rate"]) > 0.0]
    if rw_reference is not None and reset_rows:
        ref_mfpt = float(rw_reference["conditional_MFPT"])
        ref_eff = float(rw_reference["penetration_efficiency"])
        ref_dx = float(rw_reference["mean_x_displacement"])
        loss = np.array([
            [
                100.0 * (float(row["conditional_MFPT"]) - ref_mfpt) / ref_mfpt,
                100.0 * (float(rw_reference["hit_rate"]) - float(row["hit_rate"])),
                100.0 * (ref_eff - float(row["penetration_efficiency"])) / ref_eff,
                100.0 * (float(rw_reference["penetration_rate"]) - float(row["penetration_rate"])),
                100.0 * (ref_dx - float(row["mean_x_displacement"])) / max(abs(ref_dx), 1e-12),
            ]
            for row in reset_rows
        ])
        loss_labels = ["MFPT increase (%)", "Hit-rate loss (pp)", "Efficiency loss (%)",
                       "Penetration loss (pp)", "Δx loss (%)"]
        vmax = max(1.0, float(np.nanmax(loss)))
        fig, ax = plt.subplots(figsize=(9.3, 3.9))
        image = ax.imshow(loss, cmap="OrRd", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(loss_labels)), loss_labels, rotation=22, ha="right")
        ax.set_yticks(np.arange(len(reset_rows)), [f"r = {float(row['reset_rate']):g} s⁻¹" for row in reset_rows])
        for i in range(loss.shape[0]):
            for j in range(loss.shape[1]):
                value = loss[i, j]
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9,
                        color="white" if value > 0.58 * vmax else colors["ink"], fontweight="bold")
        cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label("Performance loss relative to r=0")
        ax.set_title("Reset-induced performance loss", loc="left", pad=28)
        ax.text(0.0, 1.04, "Darker red indicates a larger deterioration from ordinary random walk",
                transform=ax.transAxes, color=colors["muted"], fontsize=9.2)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.tight_layout(); save(fig, "figure10_rwr_relative_loss_heatmap")

    if {"dark", "strong"}.issubset(spatial_diagnostics):
        dark_diag = spatial_diagnostics["dark"]
        blue_diag = spatial_diagnostics["strong"]
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # Figure 11: two spatial distributions plus their direct difference.
        # Normalization removes the trivial effect of unequal total residence time.
        dark_raw = np.asarray(dark_diag["occupancy_xy"], dtype=float)
        blue_raw = np.asarray(blue_diag["occupancy_xy"], dtype=float)
        dark_density = dark_raw / max(float(dark_raw.sum()), 1e-12) * 1e4
        blue_density = blue_raw / max(float(blue_raw.sum()), 1e-12) * 1e4
        density_max = max(float(np.max(dark_density)), float(np.max(blue_density)), 1e-12)
        density_diff = blue_density - dark_density
        diff_limit = max(float(np.max(np.abs(density_diff))), 1e-12)
        extent = [base.X_MIN, base.X_MAX, base.Y_MIN, base.Y_MAX]
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), sharex=True, sharey=True,
                                 gridspec_kw={"wspace": 0.08})
        maps = [
            (blue_density, "蓝光条件", "viridis", 0.0, density_max),
            (dark_density, "暗态条件", "viridis", 0.0, density_max),
            (density_diff, "蓝光 − 暗态", "RdBu_r", -diff_limit, diff_limit),
        ]
        images = []
        for ax, (data, title, cmap_name, vmin, vmax) in zip(axes, maps):
            image = ax.imshow(data, origin="lower", aspect="auto", extent=extent,
                              cmap=cmap_name, vmin=vmin, vmax=vmax, interpolation="bilinear")
            images.append(image)
            ax.axvline(base.REG_A_END, color="white", lw=1.2, ls="--")
            ax.axvline(base.REG_B_END, color="white", lw=1.2, ls="--")
            ax.axvline(base.X_TARGET, color=colors["gold"], lw=1.8, ls=":")
            ax.set_title(title, pad=10)
            ax.set_xlabel("x（µm）")
            for spine in ax.spines.values(): spine.set_visible(False)
        axes[0].set_ylabel("y（µm）")
        cbar_density = fig.colorbar(images[0], ax=axes[:2], fraction=0.025, pad=0.015)
        cbar_density.set_label("每万次访问中的空间占比")
        cbar_diff = fig.colorbar(images[2], ax=axes[2], fraction=0.05, pad=0.025)
        cbar_diff.set_label("占据密度差值")
        fig.suptitle("蓝光诱导的细菌空间再分布", x=0.06, ha="left", fontsize=17, fontweight="bold")
        fig.text(0.06, 0.91, "红色表示蓝光下访问增加，蓝色表示访问减少；三图使用一致的空间网格",
                 color=colors["muted"], fontsize=10)
        fig.subplots_adjust(top=0.82, right=0.92)
        save(fig, "figure11_blue_dark_spatial_difference")

        # Figure 12: cumulative first-passage probability as a time-stage heatmap.
        time_grid = np.linspace(0.0, base.T_S, 180)
        stage_keys = ["first_passage_interface_s", "first_passage_target_s", "first_passage_exit_s"]
        stage_labels = ["到达 A/B 界面", "到达目标区", "完全穿透"]

        def cumulative_stage_matrix(diag: Dict[str, Any]) -> np.ndarray:
            result = []
            for key in stage_keys:
                passage = np.asarray(diag[key], dtype=float)
                result.append([float(np.mean(np.isfinite(passage) & (passage <= t))) for t in time_grid])
            return np.asarray(result)

        blue_cdf = cumulative_stage_matrix(blue_diag)
        dark_cdf = cumulative_stage_matrix(dark_diag)
        fig, axes = plt.subplots(2, 1, figsize=(10.4, 5.9), sharex=True,
                                 gridspec_kw={"hspace": 0.16})
        for ax, matrix, title in ((axes[0], blue_cdf, "蓝光条件"), (axes[1], dark_cdf, "暗态条件")):
            image = ax.imshow(matrix, origin="upper", aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0,
                              extent=[0.0, base.T_S, len(stage_labels) - 0.5, -0.5], interpolation="nearest")
            ax.set_yticks(np.arange(len(stage_labels)), stage_labels)
            ax.set_title(title, loc="left", pad=7, color=colors["blue"] if title == "蓝光条件" else colors["ink"])
            for i, final_value in enumerate(matrix[:, -1]):
                ax.text(base.T_S * 0.955, i, f"最终 {final_value:.0%}", ha="right", va="center",
                        color="white" if final_value > 0.55 else colors["ink"], fontsize=9, fontweight="bold")
            for spine in ax.spines.values(): spine.set_visible(False)
        axes[-1].set_xlabel("时间（s）")
        cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
        cbar.set_label("截至该时刻的累计到达概率")
        fig.suptitle("蓝光对细菌首达过程的影响", x=0.08, ha="left", fontsize=17, fontweight="bold")
        fig.text(0.08, 0.925, "颜色越早变深表示到达越快；最终颜色越深表示到达比例越高",
                 color=colors["muted"], fontsize=10)
        fig.subplots_adjust(top=0.84, right=0.91)
        save(fig, "figure12_first_passage_stage_heatmap")

        # Figure 13: two outcome Sankey diagrams. Loss categories are mutually
        # exclusive, so ribbon widths sum exactly to the initial population.
        def outcome_fractions(diag: Dict[str, Any]) -> List[float]:
            interface_n = int(np.isfinite(diag["first_passage_interface_s"]).sum())
            target_n = int(np.isfinite(diag["first_passage_target_s"]).sum())
            exit_n = int(np.isfinite(diag["first_passage_exit_s"]).sum())
            total = len(diag["first_passage_interface_s"])
            return [
                max(total - interface_n, 0) / total,
                max(interface_n - target_n, 0) / total,
                max(target_n - exit_n, 0) / total,
                exit_n / total,
            ]

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.1), gridspec_kw={"wspace": 0.20})
        for ax, diag, title, facecolor in (
            (axes[0], blue_diag, "蓝光条件", colors["blue"]),
            (axes[1], dark_diag, "暗态条件", "#667085"),
        ):
            losses = outcome_fractions(diag)
            outcome_names = ["未到界面", "界面后未达目标", "目标后未穿透", "成功穿透"]
            labels = ["初始细菌\n100%"] + [f"{name}\n{value:.1%}" for name, value in zip(outcome_names, losses)]
            sankey = Sankey(ax=ax, scale=0.72, offset=0.18, head_angle=115, shoulder=0.02,
                            format="", unit=None, gap=0.35)
            sankey.add(
                flows=[1.0] + [-value for value in losses],
                labels=labels,
                orientations=[0, -1, -1, 1, 0],
                pathlengths=[0.25, 0.35, 0.65, 0.65, 0.35],
                facecolor=facecolor,
                alpha=0.78,
                edgecolor="white",
                linewidth=1.0,
            )
            diagrams = sankey.finish()
            for text in diagrams[0].texts:
                text.set_fontsize(8.5)
            ax.set_title(title, fontsize=14, color=facecolor, pad=12)
            ax.set_axis_off()
        fig.suptitle("细菌迁移结局桑基图", x=0.06, ha="left", fontsize=17, fontweight="bold")
        fig.text(0.06, 0.91, "流带宽度表示细菌比例，展示迁移过程中发生损失的阶段",
                 color=colors["muted"], fontsize=10)
        fig.subplots_adjust(top=0.82)
        save(fig, "figure13_blue_dark_migration_sankey")

        # Figure 14: preserve absolute accumulated occupancy so the combined
        # speed/trapping/directional effects are not removed by normalization.
        x_edges = np.asarray(blue_diag["x_edges_xy"], dtype=float)
        x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
        blue_x = np.asarray(blue_diag["occupancy_xy"], dtype=float).sum(axis=0)
        dark_x = np.asarray(dark_diag["occupancy_xy"], dtype=float).sum(axis=0)
        kernel = np.ones(5, dtype=float) / 5.0
        blue_x_s = np.convolve(blue_x, kernel, mode="same")
        dark_x_s = np.convolve(dark_x, kernel, mode="same")
        delta_x = blue_x_s - dark_x_s

        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(11.0, 6.6), sharex=True,
            gridspec_kw={"height_ratios": [2.15, 1.0], "hspace": 0.08},
        )
        ax_top.plot(x_centres, blue_x_s, color=colors["blue"], lw=2.8,
                    label="蓝光：加速、减困陷、定向迁移")
        ax_top.fill_between(x_centres, 0, blue_x_s, color=colors["blue"], alpha=0.18)
        ax_top.plot(x_centres, dark_x_s, color="#667085", lw=2.5,
                    label="暗态：减速、加困陷、增加翻滚")
        ax_top.fill_between(x_centres, 0, dark_x_s, color="#667085", alpha=0.13)
        ax_top.set_ylabel("累计空间访问次数")
        ax_top.legend(frameon=False, loc="upper left", fontsize=10)
        polish(ax_top)

        positive = np.clip(delta_x, 0.0, None)
        negative = np.clip(delta_x, None, 0.0)
        ax_bottom.fill_between(x_centres, 0, positive, color="#D84A4A", alpha=0.82,
                               label="蓝光更多")
        ax_bottom.fill_between(x_centres, 0, negative, color="#3A78B4", alpha=0.82,
                               label="暗态更多")
        ax_bottom.axhline(0.0, color=colors["ink"], lw=0.9)
        ax_bottom.set(xlabel="迁移位置 x（µm）", ylabel="蓝光 − 暗态")
        ax_bottom.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
        polish(ax_bottom)
        for ax in (ax_top, ax_bottom):
            ax.axvline(base.REG_A_END, color="#98A2B3", ls="--", lw=1.0)
            ax.axvline(base.REG_B_END, color="#98A2B3", ls="--", lw=1.0)
            ax.axvline(base.X_TARGET, color=colors["gold"], ls=":", lw=1.8)
        fig.suptitle("蓝光与暗态的综合迁移效应", x=0.075, ha="left",
                     fontsize=17, fontweight="bold")
        fig.text(0.075, 0.925, "保留绝对累计量；上图比较总体迁移分布，下图直接显示差值",
                 color=colors["muted"], fontsize=10)
        fig.subplots_adjust(top=0.84)
        save(fig, "figure14_blue_dark_absolute_migration_contrast")

    # Figure 15: CheZ-LOV ABM versus ordinary RW and all RWR rates on shared
    # first-passage metrics. Values are normalized to the no-reset RW baseline.
    chez = next((r for r in rows if r["model"] == "mechanistic_ABM" and r["variant"] == "CheZ-LOV"), None)
    rw0 = next((r for r in rows if r["model"] == "ordinary_random_walk"), None)
    if chez is not None and rw0 is not None:
        baselines = np.array([
            float(rw0["hit_rate"]),
            1.0 / float(rw0["conditional_MFPT"]),
            float(rw0["penetration_efficiency"]),
        ])
        comparison_rows = [rw0] + sorted(
            [r for r in rows if r["model"] == "RWR"],
            key=lambda r: float(r["reset_rate"]),
        ) + [chez]
        names = ["RW\nr=0"] + [f"RWR\nr={float(r['reset_rate']):g}" for r in comparison_rows[1:-1]] + ["CheZ-LOV\nABM"]
        matrix = []
        for row in comparison_rows:
            values = np.array([
                float(row["hit_rate"]),
                1.0 / float(row["conditional_MFPT"]),
                float(row["penetration_efficiency"]),
            ])
            matrix.append(values / baselines)
        matrix = np.asarray(matrix)
        metric_labels = ["命中率", "到达速度\n（1/MFPT）", "穿透效率"]
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.0), gridspec_kw={"wspace": 0.28})
        bar_colors = ["#98A2B3"] + [colors["orange"]] * (len(comparison_rows) - 2) + [colors["blue"]]
        for j, (ax, metric) in enumerate(zip(axes, metric_labels)):
            bars = ax.bar(np.arange(len(names)), matrix[:, j], color=bar_colors, width=0.68)
            ax.axhline(1.0, color=colors["ink"], ls="--", lw=1.0)
            for bar, value in zip(bars, matrix[:, j]):
                ax.text(bar.get_x() + bar.get_width()/2, value + 0.08,
                        f"{value:.2f}×", ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold" if value > 1.0 else "normal")
            ax.set_xticks(np.arange(len(names)), names)
            ax.set_title(metric)
            ax.set_ylim(0, max(1.35, float(matrix[:, j].max()) * 1.18))
            polish(ax)
        axes[0].set_ylabel("相对于普通随机游走（r=0）的倍数")
        fig.suptitle("CheZ-LOV 与 RWR 的首达性能对比", x=0.06, ha="left",
                     fontsize=17, fontweight="bold")
        fig.text(0.06, 0.91, "蓝色为 CheZ-LOV 机制 ABM，橙色为不同重置率 RWR；虚线表示无重置随机游走基准",
                 color=colors["muted"], fontsize=9.6)
        fig.subplots_adjust(top=0.78, bottom=0.18)
        save(fig, "figure15_chezlov_vs_rwr_comparison")

    if peptide_grid_results:
        peptide_names = [item["variant"] for item in peptide_grid_results]
        peptide_f = np.array([float(item["f"]) for item in peptide_grid_results])
        peptide_v = np.array([float(item["speed"]) for item in peptide_grid_results])
        peptide_metrics = [item["metrics"] for item in peptide_grid_results]
        peptide_mfpt = np.array([float(m["conditional_MFPT"]) for m in peptide_metrics])
        peptide_eff = np.array([float(m["penetration_efficiency"]) for m in peptide_metrics])
        peptide_hit = np.array([float(m["hit_rate"]) for m in peptide_metrics])
        palette = plt.cm.Blues(np.linspace(0.38, 0.92, len(peptide_names)))

        # Figure 16: ranked lollipop chart. MFPT is the axis; its exact inverse
        # is retained as a label instead of duplicating the same information.
        order = np.argsort(peptide_mfpt)
        ranked_names = np.asarray(peptide_names)[order]
        ranked_mfpt = peptide_mfpt[order]
        ranked_eff = peptide_eff[order]
        ranked_hit = peptide_hit[order]
        ranked_f = peptide_f[order]
        ypos = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(10.5, 5.4))
        norm = plt.Normalize(0.0, 1.0)
        cmap = plt.cm.Blues
        for y, value in zip(ypos, ranked_mfpt):
            ax.hlines(y, 0.0, value, color="#A9B8CF", lw=4.0, alpha=0.72)
        points = ax.scatter(
            ranked_mfpt, ypos,
            s=115 + 240 * ranked_hit,
            c=ranked_f, cmap=cmap, norm=norm,
            edgecolor="white", linewidth=1.6, zorder=3,
        )
        for y, mfpt_value, efficiency, hit in zip(ypos, ranked_mfpt, ranked_eff, ranked_hit):
            ax.text(mfpt_value + 1.4, y,
                    f"{mfpt_value:.1f} s   |   1/MFPT={efficiency:.3f}   |   命中率={hit:.0%}",
                    va="center", fontsize=9.2, color=colors["ink"])
        ax.set_yticks(ypos, ranked_names)
        ax.invert_yaxis()
        ax.set_xlabel("条件平均首次穿透时间 MFPT（s，越低越快）")
        ax.set_xlim(0.0, float(ranked_mfpt.max()) * 1.48)
        polish(ax, grid_axis="x")
        cbar = fig.colorbar(points, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("归一化蓝光响应 f")
        fig.suptitle("不同肽的首次穿透性能排名", x=0.10, ha="left", fontsize=17, fontweight="bold")
        fig.text(0.10, 0.91, "颜色表示蓝光响应 f，圆点大小表示命中率；1/MFPT 在标签中同步给出",
                 color=colors["muted"], fontsize=9.7)
        fig.subplots_adjust(top=0.80, left=0.18, right=0.92, bottom=0.14)
        save(fig, "figure16_peptide_mfpt_efficiency")

        # Figure 17: same-scale trajectory small multiples for the grid simulation.
        fig, axes = plt.subplots(1, len(peptide_names), figsize=(15.5, 5.2), sharex=True, sharey=True,
                                 gridspec_kw={"wspace": 0.08})
        axes = np.atleast_1d(axes)
        for ax, item, colour in zip(axes, peptide_grid_results, palette):
            metrics = item["metrics"]
            tx = np.asarray(metrics["trajectory_x"])
            ty = np.asarray(metrics["trajectory_y"])
            for j in range(tx.shape[0]):
                ax.plot(tx[j], ty[j], color=colour, lw=0.8, alpha=0.62)
                ax.scatter(tx[j, 0], ty[j, 0], s=9, color=colors["ink"], zorder=3)
            ax.axvline(base.X_TARGET, color=colors["gold"], ls="--", lw=1.8)
            ax.set_title(f"{item['variant']}\nf={item['f']:.2f}, v={item['speed']:.1f}", fontsize=10)
            ax.set_xlim(base.X_MIN, base.X_MAX)
            ax.set_ylim(base.Y_MIN, base.Y_MAX)
            ax.grid(color=colors["grid"], lw=0.55, alpha=0.7)
            for spine in ax.spines.values(): spine.set_visible(False)
        axes[0].set_ylabel("网格 y（µm）")
        for ax in axes: ax.set_xlabel("x（µm）")
        fig.suptitle("不同肽的二维各向异性网格穿透轨迹", x=0.055, ha="left", fontsize=17, fontweight="bold")
        fig.text(0.055, 0.91, "每组显示相同数量的代表轨迹；黄色虚线为首次穿透目标边界",
                 color=colors["muted"], fontsize=9.7)
        fig.subplots_adjust(top=0.77, bottom=0.13)
        save(fig, "figure17_peptide_grid_trajectories")

        # Figure 18: true x-y spatial occupancy heatmaps with one shared scale.
        occupancy_maps = [np.asarray(m["spatial_occupancy"], dtype=float) for m in peptide_metrics]
        occupancy_log = [np.log1p(values) for values in occupancy_maps]
        occupancy_vmax = max(float(values.max()) for values in occupancy_log)
        extent = [base.X_MIN, base.X_MAX, base.Y_MIN, base.Y_MAX]
        fig, axes = plt.subplots(1, len(peptide_names), figsize=(15.8, 5.2), sharex=True, sharey=True,
                                 gridspec_kw={"wspace": 0.07})
        axes = np.atleast_1d(axes)
        last_im = None
        for ax, name, response, mfpt, hit, values in zip(
            axes, peptide_names, peptide_f, peptide_mfpt, peptide_hit, occupancy_log
        ):
            last_im = ax.imshow(values, origin="lower", aspect="auto", extent=extent,
                                cmap="magma", vmin=0.0, vmax=occupancy_vmax,
                                interpolation="bilinear")
            ax.axvline(base.X_TARGET, color="#54D2FF", ls="--", lw=1.8)
            ax.scatter(config.get("reset_position_um", [30.0, -400.0])[0],
                       config.get("reset_position_um", [30.0, -400.0])[1],
                       s=22, color="white", edgecolor=colors["ink"], linewidth=0.6, zorder=3)
            ax.set_title(f"{name}\nf={response:.2f}  MFPT={mfpt:.1f}s\n命中率={hit:.0%}", fontsize=9.4)
            for spine in ax.spines.values(): spine.set_visible(False)
        axes[0].set_ylabel("y（µm）")
        cbar = fig.colorbar(last_im, ax=axes, orientation="horizontal", fraction=0.055,
                            pad=0.22, aspect=48)
        cbar.set_label("累计空间访问密度  log(1+次数)")
        fig.supxlabel("迁移方向 x（µm）", y=0.20)
        fig.suptitle("不同肽的二维网格穿透空间热力图", x=0.055, ha="left", fontsize=17, fontweight="bold")
        fig.text(0.055, 0.91, "所有面板共用同一色标；亮色表示频繁经过或停留，青色虚线为目标边界",
                 color=colors["muted"], fontsize=9.7)
        fig.subplots_adjust(top=0.75, bottom=0.27, right=0.97)
        save(fig, "figure18_peptide_spatial_occupancy_heatmap")

        # Figure 19: response-performance association, with speed encoded by size.
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), gridspec_kw={"wspace": 0.30})
        sizes = 95 + 170 * (peptide_v - peptide_v.min()) / max(float(np.ptp(peptide_v)), 1e-12)
        for ax, yy, ylabel, better in (
            (axes[0], peptide_mfpt, "MFPT（s）", "越低越快"),
            (axes[1], peptide_eff, r"1/MFPT（s$^{-1}$）", "越高越好"),
        ):
            ax.scatter(peptide_f, yy, s=sizes, c=peptide_f, cmap="Blues", vmin=0, vmax=1,
                       edgecolor="white", linewidth=1.2, zorder=3)
            ax.plot(peptide_f, yy, color=colors["blue"], lw=1.2, alpha=0.45)
            for xx, value, name in zip(peptide_f, yy, peptide_names):
                ax.annotate(name, (xx, value), xytext=(5, 6), textcoords="offset points", fontsize=8.5)
            ax.set(xlabel="归一化蓝光响应 f", ylabel=ylabel, title=better, xlim=(-0.04, 1.04))
            polish(ax)
        fig.suptitle("肽蓝光响应与穿透性能的关系", x=0.07, ha="left", fontsize=17, fontweight="bold")
        fig.text(0.07, 0.91, "每个点代表一种肽；点大小表示游动速度 v",
                 color=colors["muted"], fontsize=9.7)
        fig.subplots_adjust(top=0.78, bottom=0.16)
        save(fig, "figure19_response_vs_penetration_performance")

        # Figure 20: individual first-passage distributions, retaining misses in annotations.
        valid_sets = []
        for metrics in peptide_metrics:
            values = np.array([np.nan if t is None else float(t) for t in metrics["first_passage_times"]])
            valid_sets.append(values[np.isfinite(values)])
        fig, ax = plt.subplots(figsize=(10.6, 5.2))
        parts = ax.violinplot(valid_sets, positions=np.arange(1, len(valid_sets)+1),
                              showmedians=True, showextrema=False, widths=0.78)
        for body, colour in zip(parts["bodies"], palette):
            body.set_facecolor(colour); body.set_edgecolor("white"); body.set_alpha(0.82)
        parts["cmedians"].set_color(colors["ink"])
        for i, (values, hit) in enumerate(zip(valid_sets, peptide_hit), start=1):
            jitter = np.linspace(-0.11, 0.11, len(values)) if len(values) else np.array([])
            ax.scatter(i + jitter, values, s=9, color=colors["ink"], alpha=0.32)
            ax.text(i, base.T_S * 0.97, f"命中 {hit:.0%}", ha="center", va="top", fontsize=8.5)
        ax.set_xticks(np.arange(1, len(peptide_names)+1), peptide_names, rotation=15, ha="right")
        ax.set_ylabel("个体首次穿透时间（s）")
        ax.set_ylim(0, base.T_S)
        polish(ax)
        fig.suptitle("不同肽的个体首次穿透时间分布", x=0.08, ha="left", fontsize=17, fontweight="bold")
        fig.text(0.08, 0.91, "形状宽度表示到达时间密度；点为成功穿透个体，未命中比例单独标注",
                 color=colors["muted"], fontsize=9.7)
        fig.subplots_adjust(top=0.79, bottom=0.18)
        save(fig, "figure20_peptide_first_passage_distribution")


    # Figure 21: f-v response surface; every grid point is an actual RWR run.
    surface_f = np.asarray(config.get("surface_f_values", [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0]), dtype=float)
    surface_v = np.asarray(config.get("surface_speed_values_um_s", [12, 14, 16, 18, 20, 22, 24, 26, 28]), dtype=float)
    surface_z = np.full((len(surface_v), len(surface_f)), np.nan)
    surface_rows = []
    reset_position = tuple(config.get("reset_position_um", [30.0, -400.0]))
    reset_rate = float(config.get("peptide_grid_reset_rate_per_s", 0.02))
    for iv, speed_value in enumerate(surface_v):
        for jf, response_value in enumerate(surface_f):
            surface_cfg = RWRConfig(n_agents=base.N, duration_s=base.T_S, dt_s=base.DT_S,
                speed_um_s=float(speed_value), response_factor_f=float(response_value),
                anisotropy_mode="axis_weighted", reset_rate_per_s=reset_rate,
                reset_mode=config.get("reset_mode", "position"), reset_position_um=reset_position,
                x_min_um=base.X_MIN, x_max_um=base.X_MAX, y_min_um=base.Y_MIN, y_max_um=base.Y_MAX,
                target_x_um=base.X_TARGET, periodic_y=base.USE_Y_PERIODIC,
                reflect_left=base.USE_REFLECT_LEFT_WALL, trajectory_sample_count=0,
                collect_spatial_occupancy=False, seed=base.SEED)
            metrics = run_random_walk_resetting(surface_cfg)
            surface_z[iv, jf] = float(metrics["penetration_efficiency"])
            surface_rows.append({"response_factor_f": float(response_value), "speed_um_s": float(speed_value),
                "reset_rate": reset_rate, "hit_rate": metrics["hit_rate"],
                "conditional_MFPT": metrics["conditional_MFPT"],
                "penetration_efficiency": metrics["penetration_efficiency"], "seed": base.SEED})
    _write_rows(fig_dir.parent / "fv_response_surface.csv", surface_rows,
        ["response_factor_f", "speed_um_s", "reset_rate", "hit_rate", "conditional_MFPT", "penetration_efficiency", "seed"])
    grid_f, grid_v = np.meshgrid(surface_f, surface_v)
    fig = plt.figure(figsize=(11.8, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(grid_f, grid_v, np.ma.masked_invalid(surface_z), cmap="turbo",
        edgecolor="white", linewidth=0.25, antialiased=True, alpha=0.94)
    ax.contourf(grid_f, grid_v, np.nan_to_num(surface_z), zdir="z", offset=0.0, levels=14, cmap="turbo", alpha=0.48)
    if peptide_grid_results:
        point_f = np.array([float(item["f"]) for item in peptide_grid_results])
        point_v = np.array([float(item["speed"]) for item in peptide_grid_results])
        point_z = np.array([float(item["metrics"]["penetration_efficiency"]) for item in peptide_grid_results])
        ax.scatter(point_f, point_v, point_z, s=62, color="#102A43", edgecolor="white", linewidth=1.2, depthshade=False)
        for item, xx, yy, zz in zip(peptide_grid_results, point_f, point_v, point_z):
            ax.text(xx, yy, zz + 0.006, item["variant"], fontsize=8, color="#102A43")
    ax.set_xlabel("归一化蓝光响应 f", labelpad=10)
    ax.set_ylabel("游动速度 v（µm/s）", labelpad=10)
    ax.set_zlabel(r"穿透效率 1/MFPT（s$^{-1}$）", labelpad=9)
    ax.set_zlim(bottom=0.0)
    ax.view_init(elev=27, azim=-132)
    ax.grid(False)
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.08, label=r"穿透效率 1/MFPT（s$^{-1}$）")
    fig.suptitle("蓝光响应—游动速度—穿透效率三维响应曲面", x=0.06, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.06, 0.92, "每个网格点均由二维各向异性 RWR 实际计算；深色标记为肽构建体", color=colors["muted"], fontsize=9.7)
    fig.subplots_adjust(top=0.84, left=0.02, right=0.92, bottom=0.05)
    save(fig, "figure21_fv_penetration_efficiency_surface")

    # Figure 22: top-down contour view of the same f-v response surface.
    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    valid_z = np.ma.masked_invalid(surface_z)
    filled = ax.contourf(grid_f, grid_v, valid_z, levels=18, cmap="turbo", extend="both")
    contour = ax.contour(grid_f, grid_v, valid_z, levels=9, colors="white", linewidths=0.75, alpha=0.72)
    ax.clabel(contour, inline=True, fontsize=7.5, fmt="%.3f")
    if peptide_grid_results:
        point_f = np.array([float(item["f"]) for item in peptide_grid_results])
        point_v = np.array([float(item["speed"]) for item in peptide_grid_results])
        ax.scatter(point_f, point_v, s=92, color="#102A43", edgecolor="white", linewidth=1.7, zorder=5)
        for index, (item, xx, yy) in enumerate(zip(peptide_grid_results, point_f, point_v)):
            offset_y = 10 if index % 2 == 0 else -16
            ax.annotate(item["variant"], (xx, yy), xytext=(7, offset_y), textcoords="offset points",
                        fontsize=8.5, fontweight="bold", color="#102A43",
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.82),
                        zorder=6)
    ax.set_xlabel("归一化蓝光响应 f")
    ax.set_ylabel("游动速度 v（µm/s）")
    ax.set_xlim(float(surface_f.min()), float(surface_f.max()))
    ax.set_ylim(float(surface_v.min()), float(surface_v.max()))
    polish(ax)
    colorbar = fig.colorbar(filled, ax=ax, pad=0.02)
    colorbar.set_label(r"穿透效率 1/MFPT（s$^{-1}$）")
    fig.suptitle("蓝光响应与速度的穿透效率等高线图", x=0.08, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.08, 0.91, "颜色越暖表示首次穿透越快；标记点为5种肽构建体在参数空间中的位置",
             color=colors["muted"], fontsize=9.7)
    fig.subplots_adjust(top=0.84, left=0.10, right=0.91, bottom=0.12)
    save(fig, "figure22_fv_penetration_efficiency_contour")
def run_model_comparison(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run ABM dark/blue variants plus ordinary walk and three-or-more RWR rates."""
    abm = _load_abm_module()
    variants = config.get("variants", [])
    fluorescence_path = config.get("fluorescence_csv")
    if fluorescence_path:
        variants = load_fluorescence_variants(ROOT / fluorescence_path, config.get("normalization_mode", "fixed_positive"))
    if not variants:
        raise ValueError("No variants configured")
    reset_rates = [float(v) for v in config.get("reset_rates_per_s", [])]
    if 0.0 not in reset_rates or len([v for v in reset_rates if v > 0.0]) < 3:
        raise ValueError("reset_rates_per_s must contain 0 and at least three positive rates")

    base = abm.HyperParams()
    for key, value in config.get("abm_overrides", {}).items():
        if not hasattr(base, key):
            raise ValueError(f"Unknown ABM override: {key}")
        setattr(base, key, value)
    base.SAVE_FRAMES = False; base.MAKE_VIDEO = False; base.SHOW_PLOTS = False
    rows: List[Dict[str, Any]] = []
    peptide_grid_rows: List[Dict[str, Any]] = []
    peptide_grid_results: List[Dict[str, Any]] = []
    raw_first_passage: List[Dict[str, Any]] = []
    spatial_diagnostics: Dict[str, Any] = {}

    def record_first_passage(model: str, variant: str, metrics: Dict[str, Any]) -> None:
        for agent_id, passage_time in enumerate(metrics["first_passage_times"]):
            raw_first_passage.append({
                "model": model,
                "variant": variant,
                "agent_id": agent_id,
                "first_passage_time_s": passage_time,
                "hit": passage_time is not None,
            })

    dark = abm._hp_copy(base)
    dark.LIGHT_MODE = "off"; dark.CHEY_MODE = "off"; dark.SEQUENCE_RESPONSE_D_TILDE = 0.0
    dark_metrics = abm.simulate(dark, make_plots=False, save_outputs=False, return_diagnostics=True)
    spatial_diagnostics["dark"] = dark_metrics["_diagnostics"]
    rows.append(_comparison_row("mechanistic_ABM", "dark", "dark", dark_metrics, D_tilde=0.0, I_light=0.0, speed=dark.V0_A, seed=dark.SEED))
    record_first_passage("mechanistic_ABM", "dark", dark_metrics)

    strongest_variant = max(variants, key=lambda item: float(item["D_tilde"]))
    for variant in variants:
        hp = abm._hp_copy(base)
        hp.LIGHT_MODE = "blue"; hp.CHEY_MODE = "continuous"
        hp.SEQUENCE_RESPONSE_D_TILDE = float(variant["D_tilde"])
        hp.LIGHT_INTENSITY = float(config.get("I_light", 1.0))
        if variant.get("speed_um_s") is not None:
            hp.V0_A = hp.V0_B = hp.V0_C = float(variant["speed_um_s"])
        capture_spatial = variant is strongest_variant
        metrics = abm.simulate(
            hp,
            make_plots=False,
            save_outputs=False,
            return_diagnostics=capture_spatial,
        )
        if capture_spatial:
            spatial_diagnostics["strong"] = metrics["_diagnostics"]
            spatial_diagnostics["strong_label"] = str(variant["variant"])
        rows.append(_comparison_row(
            "mechanistic_ABM", str(variant["variant"]), "blue", metrics,
            D_exp=variant.get("D_exp"), D_tilde=variant["D_tilde"],
            D_tilde_unclipped=variant.get("D_tilde_unclipped", variant["D_tilde"]),
            I_light=hp.LIGHT_INTENSITY, speed=hp.V0_A, seed=hp.SEED,
        ))
        record_first_passage("mechanistic_ABM", str(variant["variant"]), metrics)

    reset_position = tuple(config.get("reset_position_um", [0.5 * (20.0 + base.REG_A_END - 20.0), 0.5 * (base.Y_MIN + base.Y_MAX)]))
    for rate in reset_rates:
        rwr_cfg = RWRConfig(
            n_agents=base.N, duration_s=base.T_S, dt_s=base.DT_S,
            speed_um_s=float(config.get("rwr_speed_um_s", base.V0_A)), reset_rate_per_s=rate,
            reset_mode=config.get("reset_mode", "position"), reset_position_um=reset_position,
            x_min_um=base.X_MIN, x_max_um=base.X_MAX, y_min_um=base.Y_MIN, y_max_um=base.Y_MAX,
            target_x_um=base.X_TARGET, periodic_y=base.USE_Y_PERIODIC,
            reflect_left=base.USE_REFLECT_LEFT_WALL, seed=base.SEED,
        )
        metrics = run_random_walk_resetting(rwr_cfg)
        rows.append(_comparison_row(metrics["model"], f"r={rate:g}", "not_applicable", metrics, reset_rate=rate, speed=rwr_cfg.speed_um_s, seed=rwr_cfg.seed))
        record_first_passage(metrics["model"], f"r={rate:g}", metrics)

    # Teacher-requested Step 2: one two-dimensional anisotropic grid RWR run
    # per peptide.  Only f and v vary by construct; geometry, reset process,
    # duration, target, population size, and seed are held fixed.
    peptide_reset_rate = float(config.get("peptide_grid_reset_rate_per_s", 0.02))
    for variant in variants:
        peptide_f = float(variant["D_tilde"])
        peptide_speed = (
            float(variant["speed_um_s"])
            if variant.get("speed_um_s") is not None
            else float(config.get("rwr_speed_um_s", base.V0_A))
        )
        grid_cfg = RWRConfig(
            n_agents=base.N,
            duration_s=base.T_S,
            dt_s=base.DT_S,
            speed_um_s=peptide_speed,
            response_factor_f=peptide_f,
            anisotropy_mode="axis_weighted",
            reset_rate_per_s=peptide_reset_rate,
            reset_mode=config.get("reset_mode", "position"),
            reset_position_um=reset_position,
            x_min_um=base.X_MIN,
            x_max_um=base.X_MAX,
            y_min_um=base.Y_MIN,
            y_max_um=base.Y_MAX,
            target_x_um=base.X_TARGET,
            periodic_y=base.USE_Y_PERIODIC,
            reflect_left=base.USE_REFLECT_LEFT_WALL,
            trajectory_sample_count=int(config.get("peptide_grid_trajectory_count", 8)),
            seed=base.SEED,
        )
        grid_metrics = run_random_walk_resetting(grid_cfg)
        peptide_grid_results.append({
            "variant": str(variant["variant"]),
            "f": peptide_f,
            "speed": peptide_speed,
            "metrics": grid_metrics,
        })
        peptide_grid_rows.append(_comparison_row(
            "anisotropic_grid_RWR",
            str(variant["variant"]),
            "blue",
            grid_metrics,
            D_exp=variant.get("D_exp"),
            D_tilde=peptide_f,
            D_tilde_unclipped=variant.get("D_tilde_unclipped", peptide_f),
            I_light=float(config.get("I_light", 1.0)),
            response_factor_f=peptide_f,
            anisotropy_mode="axis_weighted",
            reset_rate=peptide_reset_rate,
            speed=peptide_speed,
            seed=grid_cfg.seed,
        ))

    output_dir = ROOT / config.get("output_dir", "results")
    fields = [
        "model", "variant", "light_condition", "D_exp", "D_tilde", "D_tilde_unclipped", "I_light",
        "reset_rate", "speed", "n_agents", "hit_rate", "conditional_MFPT", "penetration_efficiency",
        "penetration_rate", "target_arrival_rate", "mean_x_displacement", "directionality", "trap_rate", "seed",
    ]
    _write_rows(output_dir / "model_comparison.csv", rows, fields)
    _write_rows(
        output_dir / "peptide_grid_penetration.csv",
        peptide_grid_rows,
        [
            "model", "variant", "D_exp", "D_tilde", "response_factor_f",
            "speed", "reset_rate", "anisotropy_mode", "n_agents", "hit_rate",
            "conditional_MFPT", "penetration_efficiency", "penetration_rate",
            "target_arrival_rate", "mean_x_displacement", "directionality", "seed",
        ],
    )
    _write_rows(
        output_dir / "raw_first_passage_times.csv",
        raw_first_passage,
        ["model", "variant", "agent_id", "first_passage_time_s", "hit"],
    )
    (output_dir / "comparison_config_snapshot.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_results(rows, output_dir / "figures", abm, config, spatial_diagnostics, peptide_grid_results)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "comparison_config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = run_model_comparison(config)
    print(f"Saved {len(rows)} comparison rows to {ROOT / config.get('output_dir', 'results')}")


if __name__ == "__main__":
    main()

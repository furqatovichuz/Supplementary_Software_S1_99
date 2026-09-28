"""Regenerate Figures 2-17 from saved results (run run_scenarios.py and run_refinement.py first).
Output: results/figures/Figure_XX.png"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grainsim import load_params  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
COL = {"no_ventilation": "#c0392b", "ventilation": "#1e6f43"}
LAB = {"no_ventilation": "No ventilation", "ventilation": "Continuous ventilation"}
TITLE = {"no_ventilation": "No ventilation", "ventilation": "Continuous ventilation"}
plt.rcParams.update({"figure.dpi": 200, "savefig.bbox": "tight"})


def read_ts(name):
    with open(RES / "scenarios" / f"{name}_timeseries.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def field_map(ax, X, Y, F, xlabel, ylabel, title, base_note=False):
    extent_x = np.concatenate([[0], (X[1:] + X[:-1]) / 2, [2 * X[-1] - (X[-1] + X[-2]) / 2]])
    extent_y = np.concatenate([[0], (Y[1:] + Y[:-1]) / 2, [2 * Y[-1] - (Y[-1] + Y[-2]) / 2]])
    m = ax.pcolormesh(extent_x, extent_y, F.T, cmap="inferno", shading="flat")
    if F.max() - F.min() > 1e-6:
        lev = np.linspace(F.min(), F.max(), 6)[1:-1]
        cs = ax.contour(X, Y, F.T, levels=lev, colors="white", linewidths=0.6, alpha=0.7)
        ax.clabel(cs, fmt="%.2f", fontsize=7)
    cb = plt.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("T (°C)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    if base_note:
        ax.text(0.1, 0.25, "adiabatic base / fan inlet", fontsize=7, color="0.2")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    p = load_params()
    fld = {n: np.load(RES / "scenarios" / f"{n}_fields.npz") for n in LAB}
    ts = {n: read_ts(n) for n in LAB}
    num = 2
    for name in ("no_ventilation", "ventilation"):
        F = fld[name]
        x, y, z = F["x"], F["y"], F["z"]
        kz, jy = len(z) // 2, len(y) // 2
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        field_map(ax, x, y, F["T_day30"][:, :, kz], "x (m)", "y (m)",
                  f"{TITLE[name]}, day 30 (mid-height, z = Lz/2)")
        fig.savefig(FIG / f"Figure_{num:02d}.png"); plt.close(fig); num += 1
        for day in p["snapshot_days"]:
            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            field_map(ax, x, z, F[f"T_day{day}"][:, jy, :], "x (m)", "z (m)",
                      f"{TITLE[name]}, day {day} (vertical mid-section, y = Ly/2)", base_note=True)
            ax.set_aspect("auto")
            fig.savefig(FIG / f"Figure_{num:02d}.png"); plt.close(fig); num += 1

    series = [("T_mean", "Mean temperature (°C)", "Domain-mean temperature"),
              ("M_mean", "Mean grain moisture (wet basis)", "Domain-mean grain moisture"),
              ("W_mean", "Mean humidity ratio (kg kg$^{-1}$)", "Domain-mean interstitial-air humidity ratio"),
              ("N_mean", "Mean insect density (m$^{-3}$)", "Domain-mean insect density")]
    for key, ylab, title in series:  # Figures 10-13
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for n in LAB:
            ax.plot(ts[n]["t_days"], ts[n][key], color=COL[n], label=LAB[n], lw=1.6)
        for d in p["snapshot_days"]:
            ax.axvline(d, color="0.5", ls=":", lw=0.7)
        ax.set_xlabel("Time (days)"); ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        fig.savefig(FIG / f"Figure_{num:02d}.png"); plt.close(fig); num += 1

    tab = RES / "refinement" / "table3.csv"
    if tab.exists():  # Figures 14-15
        with open(tab) as f:
            rows = list(csv.DictReader(f))
        for ref, title in (("300 -> 150 s", "temporal refinement (300→150 s)"),
                           ("13x13x21 -> 17x17x27", "spatial refinement (13×13×21→17×17×27)")):
            r = [row for row in rows if row["Refinement"] == ref]
            labels = [row["Reported change"].split(",")[0].replace("Maximum temperature-metric change", "Max T-metric")
                      .replace("Mean grain-moisture change", "Mean M").replace("Mean humidity-ratio change", "Mean W")
                      .replace("Mean insect-density change", "Mean N") for row in r]
            a = [float(row["No ventilation"]) for row in r]
            b = [float(row["Continuous ventilation"]) for row in r]
            xx = np.arange(len(r))
            fig, ax = plt.subplots(figsize=(5.6, 3.8))
            ax.bar(xx - 0.2, a, 0.4, color=COL["no_ventilation"], label=LAB["no_ventilation"])
            ax.bar(xx + 0.2, b, 0.4, color=COL["ventilation"], label=LAB["ventilation"])
            ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels(labels)
            ax.set_ylabel("Absolute change at day 30"); ax.set_title(f"Day-30 change under {title}", fontsize=10)
            ax.legend(fontsize=8)
            fig.savefig(FIG / f"Figure_{num:02d}.png"); plt.close(fig); num += 1
    else:
        print("refinement results missing -> Figures 14-15 skipped")
        num += 2

    # Figure 16: temperature curves annotated with qualitative literature claims
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for n in LAB:
        ax.plot(ts[n]["t_days"], ts[n]["T_mean"], color=COL[n], label=f"This work: {LAB[n]}", lw=1.6)
    ax.axhline(p["T_amb"], color="0.4", ls="--", lw=0.8, label="Fan-air temperature")
    ax.annotate("Cloud & Morey [17]: cooling front passes\nthrough the bulk over several days",
                xy=(5, ts["ventilation"]["T_mean"][np.searchsorted(ts["ventilation"]["t_days"], 5)]),
                xytext=(9, 19.5), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.annotate("Jia et al. [19]: front advances inward\nfrom the cooled/ventilated boundary",
                xy=(1.5, 20), xytext=(9, 22.3), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Mean temperature (°C)")
    ax.set_title("Qualitative comparison: temperature (no digitised literature data)", fontsize=9)
    ax.legend(fontsize=7, loc="center right")
    fig.savefig(FIG / f"Figure_{num:02d}.png"); plt.close(fig); num += 1

    # Figure 17: insect curves annotated with Thorpe et al. claim
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for n in LAB:
        ax.plot(ts[n]["t_days"], ts[n]["N_mean"], color=COL[n], label=f"This work: {LAB[n]}", lw=1.6)
    ax.annotate("Thorpe et al. [13]: cold-air aeration\nseverely curtails insect growth",
                xy=(20, ts["ventilation"]["N_mean"][np.searchsorted(ts["ventilation"]["t_days"], 20)]),
                xytext=(12, 150), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Mean insect density (m$^{-3}$)")
    ax.set_title("Qualitative comparison: insect density (no digitised literature data)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(FIG / f"Figure_{num:02d}.png"); plt.close(fig)
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()

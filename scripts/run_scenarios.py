"""Run the two 30-day publication scenarios (Section 3, Table 2).

Outputs (results/scenarios/):
  <case>_timeseries.csv   hourly domain mean/min/max of T, M, W, N
  <case>_fields.npz       3-D fields at days 10, 20, 30 + cell-centre coordinates
  table2.csv              day-30 comparison (Table 2)
  summary.json            all day-30 statistics, fan pressure, run settings
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grainsim import Simulation, load_params  # noqa: E402

CASES = {"no_ventilation": False, "ventilation": True}


def run_case(p, name, ventilated, outdir):
    print(f"[{name}] grid {p['Nx']}x{p['Ny']}x{p['Nz']}, dt={p['dt']} s, {p['t_final_days']} days")
    sim = Simulation(p, ventilated=ventilated)
    t0 = time.time()
    ts, snaps = sim.run(snapshot_days=p["snapshot_days"])
    keys = list(ts.keys())
    with open(outdir / f"{name}_timeseries.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(ts["t_days"])):
            w.writerow([f"{ts[k][i]:.10g}" for k in keys])
    arrays = {"x": sim.g.xc, "y": sim.g.yc, "z": sim.g.zc}
    for day, flds in snaps.items():
        for k, v in flds.items():
            arrays[f"{k}_day{day}"] = v
    np.savez_compressed(outdir / f"{name}_fields.npz", **arrays)
    final = sim.stats()
    final["p_fan_Pa"] = sim.p_fan
    final["runtime_s"] = time.time() - t0
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "scenarios"))
    args = ap.parse_args()
    p = load_params(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {name: run_case(p, name, v, out) for name, v in CASES.items()}
    a, b = summary["no_ventilation"], summary["ventilation"]
    rows = [
        ("Mean temperature (degC)", "T_mean", "{:.2f}", "diff"),
        ("Minimum temperature (degC)", "T_min", "{:.2f}", "diff"),
        ("Maximum temperature (degC)", "T_max", "{:.2f}", "diff"),
        ("Mean grain moisture (wet basis)", "M_mean", "{:.5f}", "diff"),
        ("Mean air humidity ratio (kg/kg)", "W_mean", "{:.5f}", "diff"),
        ("Mean insect density (m^-3)", "N_mean", "{:.2f}", "pct"),
    ]
    with open(out / "table2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Quantity", "No ventilation", "Continuous ventilation", "Difference / change"])
        print("\nTable 2. Day-30 numerical outputs")
        for label, key, fmt, kind in rows:
            d = (b[key] - a[key]) if kind == "diff" else 100 * (b[key] - a[key]) / a[key]
            dtxt = (fmt.format(d) if kind == "diff" else f"{d:.2f}%")
            w.writerow([label, fmt.format(a[key]), fmt.format(b[key]), dtxt])
            print(f"  {label:34s} {fmt.format(a[key]):>10s} {fmt.format(b[key]):>10s} {dtxt:>10s}")
    summary["settings"] = {k: p[k] for k in ("Nx", "Ny", "Nz", "dt", "t_final_days", "v_in")}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFan pressure (ventilated): {b['p_fan_Pa']:.2f} Pa "
          f"({b['p_fan_Pa'] - p['p_atm']:.2f} Pa gauge)")
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()

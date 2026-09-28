"""30-day temporal and spatial refinement (Section 3.4, Table 3).

Temporal: dt = 600, 300, 150 s on 13x13x21.   Spatial: 9x9x15, 13x13x21, 17x17x27 at dt = 300 s.
Runs in parallel (one process per run).  Output: results/refinement/{runs.json, table3.csv}.  Use --table-only to rebuild the table from runs.json.
"""
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grainsim import Simulation, load_params  # noqa: E402

RUNS = [
    # (label, Nx, Ny, Nz, dt)
    ("dt600", 13, 13, 21, 600.0),
    ("dt300", 13, 13, 21, 300.0),
    ("dt150", 13, 13, 21, 150.0),
    ("g9", 9, 9, 15, 300.0),
    ("g17", 17, 17, 27, 300.0),
]


def _one(args):
    label, nx, ny, nz, dt, vent, cfg = args
    p = load_params(cfg, Nx=nx, Ny=ny, Nz=nz, dt=dt)
    s = Simulation(p, ventilated=vent)
    s.run(verbose=False)
    st = s.stats()
    print(f"  done {label} {'vent' if vent else 'novent'}: T_mean={st['T_mean']:.4f} "
          f"T_max={st['T_max']:.4f} N={st['N_mean']:.3f}", flush=True)
    return label, vent, st


def main():
    out = ROOT / "results" / "refinement"
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--table-only" in sys.argv:  # rebuild Table 3 from saved runs.json
        with open(out / "runs.json") as f:
            runs = json.load(f)
    else:
        cfg = args[0] if args else None
        jobs = [(l, nx, ny, nz, dt, v, cfg) for (l, nx, ny, nz, dt) in RUNS for v in (False, True)]
        with ProcessPoolExecutor() as ex:
            res = list(ex.map(_one, jobs))
        runs = {("vent" if v else "novent"): {} for v in (False, True)}
        for label, v, st in res:
            runs["vent" if v else "novent"][label] = st

    def delta(case, a, b):
        A, B = runs[case][a], runs[case][b]
        return {
            # largest change among the reported temperatures (domain mean, minimum, maximum)
            "Maximum temperature-metric change, degC": max(abs(A[k] - B[k]) for k in ("T_mean", "T_min", "T_max")),
            "Mean grain-moisture change": abs(A["M_mean"] - B["M_mean"]),
            "Mean humidity-ratio change, kg/kg": abs(A["W_mean"] - B["W_mean"]),
            "Mean insect-density change, m^-3": abs(A["N_mean"] - B["N_mean"]),
        }

    out.mkdir(parents=True, exist_ok=True)
    table = []
    for ref, (a, b) in (("300 -> 150 s", ("dt300", "dt150")),
                        ("600 -> 300 s", ("dt600", "dt300")),
                        ("13x13x21 -> 17x17x27", ("dt300", "g17")),
                        ("9x9x15 -> 13x13x21", ("g9", "dt300"))):
        dn, dv = delta("novent", a, b), delta("vent", a, b)
        for k in dn:
            table.append([ref, k, f"{dn[k]:.3g}", f"{dv[k]:.3g}"])
    with open(out / "table3.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Refinement", "Reported change", "No ventilation", "Continuous ventilation"])
        w.writerows(table)
    with open(out / "runs.json", "w") as f:
        json.dump(runs, f, indent=2)
    print("\nTable 3. Absolute changes in day-30 outputs")
    for r in table:
        print(f"  {r[0]:22s} {r[1]:42s} {r[2]:>10s} {r[3]:>10s}")


if __name__ == "__main__":
    main()

"""Section 2.6 verification checks -> results/verification/verification.json"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grainsim import load_params  # noqa: E402
from grainsim.verification import run_all  # noqa: E402


def main():
    p = load_params(sys.argv[1] if len(sys.argv) > 1 else None)
    res = run_all(p)
    out = ROOT / "results" / "verification"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "verification.json", "w") as f:
        json.dump(res, f, indent=2)

    ic, cb, an, dc = (res[k] for k in ("initial_conditions", "closed_box_conservation",
                                        "analytical_heat_diffusion", "darcy_pressure_drop"))
    print("1) Initial-condition recovery: max |error| =", ic["max_abs_error"], "PASS" if ic["passed"] else "FAIL")
    print(f"2) Closed box ({cb['hours']} h): water rel. error = {cb['water_rel_error']:.3g}, "
          f"energy rel. error = {cb['energy_rel_error']:.3g} (tol {cb['tolerance']:g})",
          "PASS" if cb["passed"] else "FAIL")
    print("3) Analytical cosine-mode heat diffusion:")
    for n, e in zip(an["cells"], an["L2_errors_degC"]):
        print(f"     {n:3d} cells: L2 error = {e:.3e} degC")
    print("     observed order, log2(e_coarse/e_fine):      ", [round(x, 2) for x in an["observed_order_log2"]])
    print("     observed order, log(e ratio)/log(h ratio):  ", [round(x, 2) for x in an["observed_order_h_ratio"]],
          "PASS" if an["passed"] else "FAIL")
    print(f"4) Darcy: p_fan = {dc['p_fan_Pa']:.2f} Pa ({dc['gauge_Pa']:.2f} Pa gauge), "
          f"velocity rel. error = {dc['velocity_rel_error']:.2e}", "PASS" if dc["passed"] else "FAIL")


if __name__ == "__main__":
    main()

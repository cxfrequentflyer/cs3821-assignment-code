"""Check the simulator against exact answers, starting with the proposal's Example 1.

Run from the project root: python3 validate.py

Every line should say "ok": the simulated mean is within 4 standard errors of
the exact value. Make sure to check this still works after modifying any part of the simulator core.
"""
import math
from fractions import Fraction

from ecmpsim import ECMP, DChoice, Unit, exact, run, theory

TRIALS, SEED = 20_000, 3821


def mean_ci(values):
    mean = float(values.mean())
    return mean, 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))


def show(label, proposal, target, values, pct):
    mean, ci = mean_ci(values)
    ok = "ok" if abs(mean - float(target)) <= 4 * ci / 1.96 + 1e-12 else "CHECK"
    fmt = (lambda x: f"{100 * x:6.2f}%") if pct else (lambda x: f"{x:7.4f}")
    frac = f"({target})" if isinstance(target, Fraction) else ""
    print(f"  {label:<34}{proposal:>9}   {fmt(float(target))} {frac:<15} "
          f"{fmt(mean)} ± {fmt(ci).strip():<8} {ok}")


def main():
    print("Proposal, Example 1: n = 4 paths, m = 4 unit flows "
          f"({TRIALS:,} simulated trials, ± is a 95% CI)\n")
    print(f"  {'quantity':<34}{'proposal':>9}   {'exact':<24}  simulated")
    df = run(4, 4, Unit(), [ECMP(), DChoice(2), DChoice(2, replace=False)],
             trials=TRIALS, seed=SEED)
    by = {p: g for p, g in df.groupby("policy")}
    one = {"max_load": Fraction(17, 8), "idle_frac": Fraction(81, 256)}
    two = exact.profile_stats(exact.dchoice_exact(4, 4, 2, replace=True, exact=True))
    two_d = exact.profile_stats(exact.dchoice_exact(4, 4, 2, replace=False, exact=True))
    rows = [("paths in use, ECMP", "68.4%", one, "ecmp"),
            ("paths in use, 2 choices", "80.8%", two, "d2-bytes"),
            ("paths in use, 2 distinct choices", "-", two_d, "d2-bytes-distinct")]
    for label, prop, st, pol in rows:
        show(label, prop, 1 - st["idle_frac"], 1 - by[pol].idle_frac, pct=True)
    rows = [("busiest path, ECMP", "2.13", one, "ecmp"),
            ("busiest path, 2 choices", "1.71", two, "d2-bytes"),
            ("busiest path, 2 distinct choices", "-", two_d, "d2-bytes-distinct")]
    for label, prop, st, pol in rows:
        show(label, prop, st["max_load"], by[pol].max_load, pct=False)

    print("\nLarger groups (exact or fluid-limit targets)\n")
    big = run(1000, 1000, Unit(), [DChoice(2)], trials=50, seed=SEED)
    show("n=m=1000, 2 choices, in use", "76.1%", math.tanh(1), 1 - big.idle_frac, pct=True)
    print("    (target is the n -> infinity fluid limit tanh(1), so a tiny finite-n offset is expected)")
    for n, m in [(16, 256), (64, 64), (8, 512)]:
        sim = run(n, m, Unit(), [ECMP()], trials=5000, seed=SEED)
        show(f"n={n}, m={m}, ECMP, busiest path", "-", exact.expected_max_load(m, n),
             sim.max_load, pct=False)
    fluid = 1 - theory.fluid_idle_frac(1.0, 2)
    print(f"\n  fluid limit check: 1 - idle at t = 1, d = 2 is {fluid:.6f}; tanh(1) = {math.tanh(1):.6f}")


if __name__ == "__main__":
    main()

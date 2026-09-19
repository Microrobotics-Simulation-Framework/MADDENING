"""Compare the committed coupling sweep baseline against a re-run at HEAD.

    cd <worktree>
    PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python benchmarks/bench_coupling_sweep.py \
        --json /tmp/fast_head.json
    python compare_sweep.py benchmarks/results/coupling_sweep_cpu.json /tmp/fast_head.json
"""
import json, sys

old = {(r["fixture"], r["label"]): r for r in json.load(open(sys.argv[1]))["rows"]}
new = {(r["fixture"], r["label"]): r for r in json.load(open(sys.argv[2]))["rows"]}
common = sorted(set(old) & set(new))
print(f"{len(common)} rows compared\n")


def it(r):
    return r["iterations_mean"]


def cv(r):
    return r["converged_fraction"]


# --- claim 1: Gauss-Seidel / Jacobi iteration ratio -------------------------
print("== 'Gauss-Seidel needs 1.7-1.9x fewer iterations than Jacobi' ==")
print("   (guide 'Start here' row 1 and contradiction 1/2 tables; gs/none/l2 vs jac/none/l2)")
doc = {"chain-2": 1.69, "chain-5": 1.77, "chain-20": 1.72, "chain-50": 1.72}
print(f"{'fixture':16s} {'gs old':>7s} {'gs new':>7s} {'jac old':>8s} {'jac new':>8s} "
      f"{'ratio old':>9s} {'ratio new':>9s} {'guide':>7s}")
fixtures = sorted({f for f, _ in common})
for f in fixtures:
    kg, kj = (f, "gs/none/l2"), (f, "jac/none/l2")
    if kg not in new or kj not in new:
        continue
    ro = it(old[kg]) and it(old[kj]) / it(old[kg])
    rn = it(new[kj]) / it(new[kg])
    print(f"{f:16s} {it(old[kg]):7.2f} {it(new[kg]):7.2f} {it(old[kj]):8.2f} "
          f"{it(new[kj]):8.2f} {ro:9.2f} {rn:9.2f} {doc.get(f, ''):>7}")

# --- claim 2: interface norm removes -5%..31% of gs/none iterations ---------
print("\n== \"the interface norm removes -5% to 31% of the iterations\" (gs/none) ==")
print(f"{'fixture':16s} {'l2 old':>7s} {'if old':>7s} {'cut old':>8s} | "
      f"{'l2 new':>7s} {'if new':>7s} {'cut new':>8s}")
cuts_o, cuts_n = [], []
for f in fixtures:
    a, b = (f, "gs/none/l2"), (f, "gs/none/interface")
    if a not in new or b not in new:
        continue
    co = 100 * (1 - it(old[b]) / it(old[a]))
    cn = 100 * (1 - it(new[b]) / it(new[a]))
    cuts_o.append(co)
    cuts_n.append(cn)
    print(f"{f:16s} {it(old[a]):7.2f} {it(old[b]):7.2f} {co:7.1f}% | "
          f"{it(new[a]):7.2f} {it(new[b]):7.2f} {cn:7.1f}%")
print(f"  range recorded: {min(cuts_o):.0f}% to {max(cuts_o):.0f}%   "
      f"range now: {min(cuts_n):.0f}% to {max(cuts_n):.0f}%   guide says -5% to 31%")

# --- claim 3: slow-drift jacobi fixed omega ---------------------------------
print("\n== 'fixed point that barely moves -> jacobi / fixed w=0.8 / l2' ==")
print("   guide: slow-drift jacobi L2  6.14 -> 4.30 (w=0.5) -> 3.46 (w=0.8), best config")
for f in ("slow-drift", "expensive-pair"):
    for norm in ("l2", "interface"):
        for lab in ("jac/none/", "jac/fixed0.5/", "jac/fixed0.8/"):
            k = (f, lab + norm)
            if k in new:
                print(f"   {f:15s} {lab+norm:24s} old {it(old[k]):6.2f} it "
                      f"cv {cv(old[k]):4.0%}   new {it(new[k]):6.2f} it cv {cv(new[k]):4.0%}")

# --- claim 4: 'not one Gauss-Seidel row beats its unrelaxed counterpart' ----
print("\n== 'not one Gauss-Seidel row beats its unrelaxed counterpart on iterations' ==")
viol = []
for f in fixtures:
    for norm in ("l2", "interface"):
        base = (f, "gs/none/" + norm)
        if base not in new:
            continue
        for w in ("0.5", "0.8"):
            k = (f, f"gs/fixed{w}/{norm}")
            if k in new and it(new[k]) < it(new[base]):
                viol.append((f, norm, w, it(new[base]), it(new[k])))
print("   violations at HEAD:", viol or "none — claim still holds")

# --- claim 5: rows that stopped converging ---------------------------------
print("\n== rows that converged in the recorded file and no longer do ==")
lost = [(f, l, cv(old[(f, l)]), cv(new[(f, l)]))
        for (f, l) in common if cv(old[(f, l)]) > 0.99 and cv(new[(f, l)]) < 0.99]
for f, l, a, b in lost:
    print(f"   {f:16s} {l:26s} {a:5.0%} -> {b:5.0%}")
print(f"   {len(lost)} of {len(common)} rows")

# --- claim 6: aitken cannot exit in fewer than four -------------------------
print("\n== 'Aitken cannot exit in fewer than four [iterations]' ==")
bad = [(f, l, it(new[(f, l)])) for (f, l) in common
       if "aitken" in l and it(new[(f, l)]) < 3.995]
print("   aitken rows below 4.0 at HEAD:", bad or "none — claim holds")

# --- overall drift ----------------------------------------------------------
print("\n== overall ==")
import statistics
d = [it(new[k]) / it(old[k]) for k in common if it(old[k]) > 0]
print(f"   mean iterations ratio new/old: {statistics.mean(d):.3f}  "
      f"median {statistics.median(d):.3f}  max {max(d):.2f}")
worse = sum(1 for x in d if x > 1.001)
better = sum(1 for x in d if x < 0.999)
print(f"   rows with more iterations: {worse}, fewer: {better}, "
      f"unchanged: {len(d)-worse-better}")

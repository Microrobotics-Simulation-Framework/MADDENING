"""R1 DECISIVE #2: NET wall-clock of the MG-preconditioned solve vs hybrid-Jacobi.

kappa said MG conditions ~200-1600x better. But each V-cycle costs several
matvec-equivalents. This measures what actually matters: iterations to 1e-8 AND
wall-clock AND M^-1/matvec cost ratio, matrix-free, everything jitted, across
contrast. If net wall-clock speedup <= ~1x, the V-cycle cost ate the win.

Uses e2e.bench (hybrid vs mg-pullback), full unmasked A_wave.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")
from e2e import bench


def main():
    configs = [
        (2, 5, 2, "smooth"),   # 2D 64^2, N=4096
        (3, 4, 2, "smooth"),   # 3D 32^3, N=32768
        (2, 5, 2, "jump"),
    ]
    for (dim, nl, nc, kind) in configs:
        side = nc * 2 ** nl
        print(f"\n===== {side}^{dim} N={(nc*2**nl)**dim} kind={kind} — NET speedup vs contrast =====")
        print(f"{'contrast':>9} | {'hybrid iters/wall':>20} | {'mg iters/wall':>18} | "
              f"{'net wall x':>10} | {'Minv/matvec':>11} | {'agree':>9}")
        for ct in (1.0, 10.0, 1e2, 1e3):
            try:
                p, out = bench(dim, nl, nc, kind, ct, reps=3)
                hk, hr, ht = out["hybrid"]
                mk, mr, mt = out["mg-pullback"]
                netx = ht / mt if mt > 0 else float("nan")
                mvr = out["t_minv"] / out["t_matvec"] if out["t_matvec"] > 0 else float("nan")
                print(f"{ct:>9.0e} | {hk:>6d} it / {ht*1e3:>8.1f} ms | "
                      f"{mk:>5d} it / {mt*1e3:>7.1f} ms | {netx:>9.2f}x | "
                      f"{mvr:>10.1f}x | {out['agree']:>9.1e}")
            except Exception as ex:
                print(f"{ct:>9.0e} | ERROR: {repr(ex)[:90]}")


if __name__ == "__main__":
    main()

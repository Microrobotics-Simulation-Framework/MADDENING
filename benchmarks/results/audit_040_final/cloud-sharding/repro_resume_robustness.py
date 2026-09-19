"""Resume: trajectory equivalence, truncation, corruption, partial writes."""
import os, shutil, sys, tempfile, threading
import numpy as np
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.simulation.checkpoint import (
    save_state_with_manifest, load_state_with_manifest)
from maddening.cloud.resume import download_and_load_state


class Relax(SimulationNode):
    def __init__(self, name="relax", n=4, rate=0.5, timestep=0.1):
        super().__init__(name=name, timestep=timestep, rate=float(rate), n=int(n))
    def halo_width(self): return {}
    def state_fields(self): return ["x"]
    def initial_state(self):
        return {"x": jnp.arange(int(self.params["n"]), dtype=jnp.float32)}
    def boundary_input_spec(self): return {}
    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": state["x"] + p["rate"] * dt * (1.0 - state["x"])}

def build():
    gm = GraphManager(); gm.add_node(Relax()); gm.compile(); return gm

tmp = tempfile.mkdtemp(prefix="audit_resume2_")
try:
    # ---- 1. trajectory equivalence -------------------------------------
    print("== 1. resumed trajectory vs uninterrupted trajectory ==")
    a = build()
    for _ in range(10):
        a.step()
    uninterrupted = np.asarray(a.get_node_state("relax")["x"]).copy()

    b = build()
    for _ in range(4):
        b.step()
    npz, man = save_state_with_manifest(b, os.path.join(tmp, "snap.npz"))
    c = build()
    load_state_with_manifest(c, npz)
    for _ in range(6):
        c.step()
    resumed = np.asarray(c.get_node_state("relax")["x"])
    print(f"   uninterrupted (10 steps): {uninterrupted}")
    print(f"   4 + resume + 6 steps    : {resumed}")
    d = float(np.max(np.abs(uninterrupted - resumed)))
    print(f"   max abs diff = {d:.3e}  {'OK' if d < 1e-6 else '*** TRAJECTORY DIVERGES ***'}")

    # ---- 2. truncated .npz ---------------------------------------------
    print("\n== 2. truncated checkpoint (simulating an interrupted upload) ==")
    trunc = os.path.join(tmp, "trunc.npz")
    shutil.copyfile(npz, trunc)
    shutil.copyfile(str(man), trunc + ".manifest.json")
    size = os.path.getsize(trunc)
    with open(trunc, "r+b") as f:
        f.truncate(size // 2)
    for skip in (False, True):
        g = build()
        try:
            load_state_with_manifest(g, trunc, skip_integrity_check=skip)
            print(f"   skip_integrity_check={skip}: LOADED (no error)")
        except Exception as exc:
            print(f"   skip_integrity_check={skip}: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:95]}")

    # ---- 3. single-bit corruption (same length) ------------------------
    print("\n== 3. bit-flipped checkpoint, same length ==")
    corrupt = os.path.join(tmp, "corrupt.npz")
    shutil.copyfile(npz, corrupt)
    shutil.copyfile(str(man), corrupt + ".manifest.json")
    with open(corrupt, "r+b") as f:
        f.seek(size - 8); b0 = f.read(1)
        f.seek(size - 8); f.write(bytes([b0[0] ^ 0x01]))
    for skip in (False, True):
        g = build()
        try:
            load_state_with_manifest(g, corrupt, skip_integrity_check=skip)
            print(f"   skip_integrity_check={skip}: LOADED, x = "
                  f"{np.asarray(g.get_node_state('relax')['x'])}")
        except Exception as exc:
            print(f"   skip_integrity_check={skip}: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:95]}")

    # ---- 4. npz written, manifest missing (crash between the two) ------
    print("\n== 4. .npz present, manifest never written (crash mid-save) ==")
    nom = os.path.join(tmp, "nomanifest.npz")
    shutil.copyfile(npz, nom)
    for skip in (False, True):
        g = build()
        try:
            m = load_state_with_manifest(g, nom, skip_integrity_check=skip)
            print(f"   skip_integrity_check={skip}: LOADED, manifest={m}")
        except Exception as exc:
            print(f"   skip_integrity_check={skip}: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:95]}")

    # ---- 5. concurrent resumes through the transport -------------------
    print("\n== 5. 8 concurrent download_and_load_state() from one file:// URL ==")
    url = "file://" + str(npz)
    errs, oks = [], []
    def worker():
        try:
            g = build()
            download_and_load_state(g, url)
            oks.append(np.asarray(g.get_node_state("relax")["x"]).copy())
        except Exception as exc:
            errs.append(f"{type(exc).__name__}: {exc}")
    ts = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    print(f"   {len(oks)} succeeded, {len(errs)} failed")
    for e in errs[:3]:
        print(f"     {e[:110]}")
    if oks:
        agree = all(np.array_equal(oks[0], o) for o in oks)
        print(f"   all results identical: {agree}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

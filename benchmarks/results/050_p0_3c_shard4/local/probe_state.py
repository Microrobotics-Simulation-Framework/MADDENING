"""Scratch probe (not committed): the process state a test file leaves behind."""
import gc
import logging
import sys
import threading
import warnings


def test_state():
    loggers = {n: len(l.handlers) for n, l in logging.Logger.manager.loggerDict.items()
               if isinstance(l, logging.Logger) and l.handlers}
    root = logging.getLogger()
    lbm = sys.modules.get("maddening.nodes.lbm") or sys.modules.get("maddening.nodes")
    print("\nPROBE threads", threading.active_count(), [t.name for t in threading.enumerate()][:15])
    print("PROBE root handlers", len(root.handlers), "loggers with handlers", loggers)
    print("PROBE warnings.filters", len(warnings.filters), warnings.filters[:3])
    print("PROBE showwarning", warnings.showwarning)
    print("PROBE gc counts", gc.get_count(), "objects", len(gc.get_objects()), "gc enabled", gc.isenabled())
    import jax
    print("PROBE jax", {k: getattr(jax.config, k) for k in ("jax_enable_x64", "jax_disable_jit", "jax_log_compiles",
                                                           "jax_debug_nans", "jax_debug_infs", "jax_explain_cache_misses")})
    print("PROBE sys.settrace", sys.gettrace(), "profile", sys.getprofile())


def test_census():
    from collections import Counter
    gc.collect()
    objs = gc.get_objects()
    c = Counter(str(getattr(type(o), "__module__", "?")) + "." + str(getattr(type(o), "__qualname__", "?")) for o in objs)
    print("\nPROBE census", c.most_common(14))
    big = []
    for name, mod in list(sys.modules.items()):
        if not mod or not (name.startswith("tests") or name.startswith("maddening") or name.startswith("hypothesis")):
            continue
        for k, v in list(vars(mod).items()):
            try:
                n = len(v) if isinstance(v, (list, dict, set, tuple)) else (v.cache_info().currsize if hasattr(v, "cache_info") else 0)
            except Exception:
                n = 0
            if n > 50:
                big.append((n, f"{name}.{k}", type(v).__name__))
    print("PROBE big module globals", sorted(big, reverse=True)[:12])

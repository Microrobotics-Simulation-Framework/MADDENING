"""Run a command (pytest) and sample it every few seconds.

usage: monitor.py LOG INTERVAL -- cmd...

Writes one JSON line per sample to LOG: the parent's thread count, CPU
times, RSS, its busiest threads (by CPU time gained since the last sample),
and its descendants (pid, name, CPU, threads, RSS).  Ends with a summary.
"""
import json, subprocess, sys, time
import psutil

log_path, interval = sys.argv[1], float(sys.argv[2])
cmd = sys.argv[sys.argv.index("--") + 1:]
t0 = time.time()
proc = subprocess.Popen(cmd)
p = psutil.Process(proc.pid)
prev_threads = {}
with open(log_path, "w") as log:
    while proc.poll() is None:
        try:
            with p.oneshot():
                ct = p.cpu_times()
                threads = p.threads()
                rss = p.memory_info().rss
                nthreads = p.num_threads()
            gained = []
            for th in threads:
                tot = th.user_time + th.system_time
                gained.append((tot - prev_threads.get(th.id, 0.0), th.id, tot))
            prev_threads = {th.id: th.user_time + th.system_time for th in threads}
            gained.sort(reverse=True)
            kids = []
            for c in p.children(recursive=True):
                try:
                    with c.oneshot():
                        cc = c.cpu_times()
                        kids.append({"pid": c.pid, "cmd": " ".join(c.cmdline()[1:4])[-120:],
                                     "cpu": round(cc.user + cc.system, 1),
                                     "threads": c.num_threads(),
                                     "rss_mb": round(c.memory_info().rss / 2**20)})
                except psutil.Error:
                    pass
            rec = {"t": round(time.time() - t0, 1), "threads": nthreads,
                   "user": round(ct.user, 1), "sys": round(ct.system, 1),
                   "child_user": round(ct.children_user, 1), "child_sys": round(ct.children_system, 1),
                   "rss_mb": round(rss / 2**20),
                   "busy_threads": [(round(g, 2), tid) for g, tid, _ in gained[:3]],
                   "kids": kids}
            log.write(json.dumps(rec) + "\n"); log.flush()
        except psutil.Error:
            pass
        time.sleep(interval)
    log.write(json.dumps({"t": round(time.time() - t0, 1), "exit": proc.returncode}) + "\n")
sys.exit(proc.returncode)

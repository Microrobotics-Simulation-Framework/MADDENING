"""Summarise the slow-lane run on ci/slow-shard4 (scratch, not committed).

usage: new_run.py RUN_ID WORKTREE ARTIFACT_DIR JOBS_JSON
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

run, wt, art, jobs_json = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
sys.path.insert(0, str(wt))
from tests import _sharding  # noqa: E402

table = json.loads((wt / "tests/slow_lane_weights.json").read_text())
weights = table["seconds"]
ref, exp = table["runner_speed"]["reference_seconds"], table["runner_speed"]["exponent"]
dealt = _sharding.deal(weights, 10)
pred = defaultdict(float)
for p, s in weights.items():
    pred[dealt[p]] += s


def t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


step = {}
for job in json.load(open(jobs_json))["jobs"]:
    m = re.match(r"slow \(3.12, ([\d.]+), (\d+)\)", job["name"])
    if not m:
        continue
    st = {s["name"]: s for s in job["steps"]}.get("Run full suite (slow lane)")
    if st and st.get("completedAt") and st.get("startedAt"):
        step[(m.group(1), int(m.group(2)))] = ((t(st["completedAt"]) - t(st["startedAt"])).total_seconds() / 60,
                                               job["conclusion"])
SEED = "tests/cloud/multigpu/test_run_pod_seeded_faults.py"
print(f"{'lane':7} {'sh':>2} {'step':>6} {'tests':>6} {'pred':>6} {'coll':>5} {'xspd':>5} {'expect':>6} {'meas/exp':>8}  seeded  cpu")
for p in sorted(art.glob("*/*.xml")):
    m = re.search(r"jax([\d.]+)-shard(\d+)of10", p.parent.name)
    jax, shard = m.group(1), int(m.group(2))
    root = ET.parse(p).getroot()
    props = {q.get("name"): q.get("value") for q in root.iter("property") if q.get("name", "").startswith("runner_")}
    total = 0.0
    seeded = 0.0
    for case in root.iter("testcase"):
        total += float(case.get("time") or 0)
        if case.get("file") == SEED:
            seeded += float(case.get("time") or 0)
    coll = float(props.get("runner_collect_seconds", "nan"))
    speed = (coll / ref) ** exp
    pr = pred[shard] / 60
    sm, concl = step.get((jax, shard), (float("nan"), "?"))
    print(f"{jax:7} {shard:2d} {sm:6.1f} {total / 60:6.1f} {pr:6.1f} {coll:5.1f} {speed:5.2f} {pr * speed:6.1f} "
          f"{total / 60 / (pr * speed):8.2f}  {seeded / 60:6.1f}  {props.get('runner_cpu', '?')} {concl}")

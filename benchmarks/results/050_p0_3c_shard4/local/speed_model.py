import sys, glob, re, math, statistics as st, xml.etree.ElementTree as ET
from collections import defaultdict
base = sys.argv[1]; runs = sys.argv[2:]
T = {}  # (runner, file) -> seconds
for run in runs:
    for p in glob.glob(f'{base}/{run}/slow-durations-py3.12-jax*-shard*of10/*.xml'):
        jax = re.search(r'jax([\d.]+)-', p).group(1); shard = int(re.search(r'shard(\d+)of10', p).group(1))
        r = (run, jax, shard)
        acc = defaultdict(float)
        for tc in ET.parse(p).getroot().iter('testcase'):
            acc[tc.get('file')] += float(tc.get('time') or 0)
        for f, s in acc.items():
            if s > 20: T[(r, f)] = s
runners = sorted({r for r, f in T}); files = sorted({f for r, f in T})
# files in >=3 runners only
cnt = defaultdict(int)
for r, f in T: cnt[f] += 1
files = [f for f in files if cnt[f] >= 3]
lw = {f: 0.0 for f in files}; ls = {r: 0.0 for r in runners}
SEED = 'tests/cloud/multigpu/test_run_pod_seeded_faults.py'
for it in range(50):
    for f in files:
        v = [math.log(T[(r, f)]) - ls[r] for r in runners if (r, f) in T]
        lw[f] = st.median(v)
    for r in runners:
        v = [math.log(T[(r, f)]) - lw[f] for f in files if (r, f) in T and f != SEED]
        ls[r] = st.median(v) if v else 0.0
    m = st.median(ls.values())
    for r in runners: ls[r] -= m
    for f in files: lw[f] += m
print('runner speed factor (time multiplier vs median runner), files with >20 s, excluding seeded_faults from the fit:')
for r in runners:
    n = sum(1 for f in files if (r, f) in T)
    seed = T.get((r, SEED))
    extra = ''
    if seed: extra = '   seeded: %.1f min, residual after runner factor %.2f' % (seed/60, math.exp(math.log(seed) - lw[SEED] - ls[r]))
    print('  %s jax %s shard %2d  x%.2f  (n=%2d)%s' % (r[0], r[1], r[2], math.exp(ls[r]), n, extra))
print('seeded base (median-runner) %.1f min' % (math.exp(lw[SEED])/60))
import json; json.dump({"|".join(map(str,r)): math.exp(ls[r]) for r in runners}, open("runner_factors.json","w"))

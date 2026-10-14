import json, re
from datetime import datetime
regions = json.load(open('regions.json'))
def ts(line): return datetime.fromisoformat(line[:26] + '+00:00')
res = {}
for key, v in regions.items():
    start = coll = None
    with open(f"logs/job_{v['job']}.log", errors='replace') as f:
        for line in f:
            line = line.lstrip('﻿')
            if start is None and 'test session starts' in line: start = ts(line)
            elif start is not None and 'collected ' in line and ' items' in line:
                coll = ts(line); break
    res[key] = dict(v, collect_s=(coll - start).total_seconds() if start and coll else None)
json.dump(res, open('collect_speed.json', 'w'), indent=1)

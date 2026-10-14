"""Copy a downloaded slow-lane run, adding each report's collection time from its job log.

The three runs ``tests/slow_lane_weights.json`` was first scaled from
(38182068116, 38206585198, 38332673699) ran before ``tests/_sharding.py``
wrote ``runner_collect_seconds`` into the JUnit report.  Their collection
times were read from the job logs instead (pytest's "test session starts"
line to its "collected N items" line; ``local/collect_speed.py``) into
``ci_collect_seconds.json``, keyed ``run|jax|shard``.  On the 20 jobs of run
38504812929, which has the property, the log reading and the property agree
to under 2 ms.

    python backfill_collect_seconds.py RUN_ID DOWNLOADED_DIR OUT_DIR

writes OUT_DIR/<artifact>/<report>.xml with the property added, ready for
``scripts/slow_lane_weights.py --run RUN_ID=OUT_DIR``.
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROPERTY = "runner_collect_seconds"


def main(run: str, source: Path, out: Path) -> int:
    seconds = json.loads((Path(__file__).parent / "ci_collect_seconds.json").read_text(encoding="utf-8"))
    for report in sorted(source.glob("*/*.xml")):
        m = re.search(r"jax([\d.]+)-shard(\d+)of\d+$", report.parent.name)
        entry = seconds.get(f"{run}|{m.group(1)}|{m.group(2)}") if m else None
        tree = ET.parse(report)
        suite = next(tree.getroot().iter("testsuite"))
        if entry and entry.get("collect_s") and suite.find(f"properties/property[@name='{PROPERTY}']") is None:
            props = suite.find("properties")
            if props is None:
                props = ET.Element("properties")
                suite.insert(0, props)
            ET.SubElement(props, "property", name=PROPERTY, value=f"{entry['collect_s']:.2f}")
        else:
            print(f"{report.parent.name}: left as it is", file=sys.stderr)
        target = out / report.parent.name / report.name
        target.parent.mkdir(parents=True, exist_ok=True)
        tree.write(target, encoding="utf-8", xml_declaration=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])))

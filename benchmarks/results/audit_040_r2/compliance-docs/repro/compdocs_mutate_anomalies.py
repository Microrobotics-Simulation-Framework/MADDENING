"""Mutation-test scripts/check_anomalies.py: can it fail?"""
import copy, subprocess, sys, yaml, os
SRC = "docs/validation/known_anomalies.yaml"
OUT = os.environ["D"] + "/anomut"
base = yaml.safe_load(open(SRC))

def mut(name, fn):
    d = copy.deepcopy(base)
    fn(d)
    p = f"{OUT}/{name}.yaml"
    yaml.safe_dump(d, open(p, "w"), sort_keys=False, allow_unicode=True)
    r = subprocess.run([sys.executable, "scripts/check_anomalies.py", p,
                        "--prefix", "MADD-ANO-", "--repo-root", os.getcwd()],
                       capture_output=True, text=True)
    verdict = "CAUGHT" if r.returncode else "MISSED"
    msg = (r.stderr.strip() or r.stdout.strip()).splitlines()
    print(f"{verdict:7} {name}: {msg[0] if msg else ''}")

mut("M1_empty_registry",        lambda d: d.__setitem__("anomalies", []))
mut("M2_bogus_component",       lambda d: d["anomalies"][1]["affected_components"].append("maddening.nodes.heat.NoSuchThing"))
mut("M3_bogus_test_id",         lambda d: d["anomalies"][3]["verification"].append("tests/cloud/multigpu/test_sharded_params.py::test_that_does_not_exist"))
mut("M4_bogus_test_file",       lambda d: d["anomalies"][3]["verification"].append("tests/nowhere/test_absent.py::test_x"))
mut("M5_duplicate_id",          lambda d: d["anomalies"][2].__setitem__("anomaly_id", "MADD-ANO-002"))
mut("M6_resolved_no_version",   lambda d: d["anomalies"][3].pop("resolution_version"))
mut("M7_resolved_no_verif",     lambda d: d["anomalies"][3].pop("verification"))
mut("M8_bogus_versions",        lambda d: d["anomalies"][1].__setitem__("affected_versions", "banana"))
mut("M9_empty_rationale",       lambda d: d["anomalies"][1].__setitem__("safety_relevance_rationale", ""))
mut("M10_rationale_is_title",   lambda d: d["anomalies"][1].__setitem__("safety_relevance_rationale", d["anomalies"][1]["title"]))
mut("M11_bogus_status",         lambda d: d["anomalies"][1].__setitem__("resolution_status", "mostly_fine"))
mut("M12_bogus_severity",       lambda d: d["anomalies"][1].__setitem__("severity", "medium"))
mut("M13_version_drift",        lambda d: d.__setitem__("maddening_version", "9.9.9"))
mut("M14_no_components",        lambda d: d["anomalies"][1].__setitem__("affected_components", []))
mut("M15_wrong_prefix",         lambda d: d["anomalies"][1].__setitem__("anomaly_id", "XXX-ANO-002"))
mut("M16_safety_relevant_no_rationale", lambda d: (d["anomalies"][1].__setitem__("safety_relevance","safety_relevant"), d["anomalies"][1].pop("safety_relevance_rationale")))

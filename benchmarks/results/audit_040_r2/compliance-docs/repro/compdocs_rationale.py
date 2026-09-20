import yaml
d = yaml.safe_load(open("docs/validation/known_anomalies.yaml"))
print(f"{'id':<14}{'words':>6}{'title words reused':>20}  verdict")
for a in d["anomalies"]:
    r = " ".join(str(a["safety_relevance_rationale"]).split())
    title = set(w.lower().strip(".,`'\"") for w in a["title"].split())
    rw = [w.lower().strip(".,`'\"") for w in r.split()]
    ov = sum(1 for w in rw if w in title) / max(len(rw), 1)
    verdict = "THIN" if (len(rw) < 30 or ov > 0.5) else "substantive"
    print(f"{a['anomaly_id']:<14}{len(rw):>6}{ov:>19.0%}  {verdict}")

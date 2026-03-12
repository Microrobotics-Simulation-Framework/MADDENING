"""
Anomaly registry validation (Section 9.7 / Section 16).

Validates a ``known_anomalies.yaml`` file against the MADDENING anomaly schema.
Usable both as a library function and via the CLI
(``python -m maddening.compliance check-anomalies``).
"""

from __future__ import annotations

from typing import Optional

import yaml


_VALID_SEVERITIES = {"critical", "major", "minor", "enhancement"}
_VALID_SAFETY_RELEVANCES = {
    "safety_relevant", "not_safety_relevant", "context_dependent",
}
_REQUIRED_ANOMALY_FIELDS = (
    "anomaly_id", "title", "description",
    "severity", "safety_relevance", "safety_relevance_rationale",
)


def validate_anomaly_registry(
    path: str,
    *,
    prefix: str = "",
) -> list[str]:
    """Validate a known_anomalies.yaml file against the anomaly schema.

    Parameters
    ----------
    path : str
        Path to the YAML file to validate.
    prefix : str, optional
        Expected anomaly ID prefix (e.g., ``"MIME-ANO-"``).  If provided,
        all anomaly IDs must start with this prefix.  If empty, any
        prefix is accepted.

    Returns
    -------
    list[str]
        List of validation errors.  Empty list means the file is valid.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    errors: list[str] = []

    # Top-level structure
    if not isinstance(data, dict):
        return ["File must contain a YAML mapping"]

    for required in ("schema_version", "generated_date"):
        if required not in data:
            errors.append(f"Missing top-level field: {required}")

    if "anomalies" not in data:
        errors.append("Missing top-level field: anomalies (must be a list, may be empty)")
        return errors

    if not isinstance(data["anomalies"], list):
        errors.append("'anomalies' must be a list")
        return errors

    # Anomaly entries
    ids_seen: set[str] = set()
    for a in data.get("anomalies", []):
        if not isinstance(a, dict):
            errors.append(f"Anomaly entry is not a mapping: {a!r}")
            continue

        aid = a.get("anomaly_id", "<missing>")

        # Uniqueness
        if aid in ids_seen:
            errors.append(f"Duplicate anomaly_id: {aid}")
        ids_seen.add(aid)

        # Prefix enforcement
        if prefix and not str(aid).startswith(prefix):
            errors.append(f"{aid}: does not match required prefix '{prefix}'")

        # Required fields
        for fld in _REQUIRED_ANOMALY_FIELDS:
            if not a.get(fld):
                errors.append(f"{aid}: missing required field '{fld}'")

        # Valid enums
        sev = a.get("severity")
        if sev is not None and sev not in _VALID_SEVERITIES:
            errors.append(f"{aid}: invalid severity '{sev}'")

        sr = a.get("safety_relevance")
        if sr is not None and sr not in _VALID_SAFETY_RELEVANCES:
            errors.append(f"{aid}: invalid safety_relevance '{sr}'")

    return errors

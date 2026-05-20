---
orphan: false
---

# Surviving spot preemption

```{versionadded} v0.2
The preempt-snapshot hook, the `RESUME_FROM_URL` entry-point, and
the sidecar manifest landed in v0.2 #8.  See
{mod}`maddening.core.simulation.checkpoint` and
{func}`maddening.cloud.entrypoint.make_preempt_snapshot_hook`.
```

Spot VMs are cheap and disposable — until the cloud provider yanks
yours at 30 seconds' notice and your simulation state vapourises.
v0.2 wires up three things so that doesn't happen:

1. **Snapshot on preemption** — a {class}`~maddening.cloud.session.CloudSession`
   callback writes `state.npz` + a sidecar manifest the moment the
   preemption monitor fires.
2. **Resume from URL** — the cloud entry-point reads `RESUME_FROM_URL`
   and pulls the state in before the FastAPI server starts.
3. **Integrity manifest** — every snapshot ships a sidecar with a
   SHA-256 hash and schema version so a corrupted resume fails
   loudly instead of silently mangling state.

## The 30-second tour

```python
from maddening.cloud.session import CloudSession
from maddening.cloud.entrypoint import make_preempt_snapshot_hook

# Wire the hook
hook = make_preempt_snapshot_hook(
    server,                                # has .gm (GraphManager)
    snapshot_path="/mnt/snapshots/sim.npz",
    extra_meta={"commit": "abc123", "cluster": "runpod-spot-7"},
)
session = CloudSession(on_preempted=hook)
session.launch(cfg)

# ... time passes, spot gets reclaimed ...
# hook(info) fires automatically; sim.npz + sim.npz.manifest.json land on disk
```

The orchestrator's responsibility from that point is to upload both
files to durable storage (S3, GCS, a Selkies volume) and relaunch the
VM with:

```bash
RESUME_FROM_URL="https://my-bucket.s3.amazonaws.com/sim.npz" \
    python -m maddening.cloud.entrypoint
```

The entry-point downloads the .npz **and** the `.manifest.json`,
verifies the hash + schema version, then loads the state before
binding the HTTP port.

## The manifest schema

```json
{
  "schema_version": 1,
  "sha256": "41949865eaecffb496dc45c62ff400b01e11f51958c27599d96e12d6de80ca59",
  "size_bytes": 810,
  "extra": {
    "session_id": "...",
    "stage_at_snapshot": "preempted",
    "commit": "abc123",
    "cluster": "runpod-spot-7"
  }
}
```

* `schema_version` — bumps when the on-disk `.npz` key layout
  changes.  Readers refuse mismatched versions instead of silently
  producing wrong state.
* `sha256` — full hash of the `.npz` body.  Tampering or partial
  download → `CheckpointIntegrityError`.
* `extra` — caller-supplied dict.  The snapshot hook auto-populates
  `session_id` and `stage_at_snapshot`; merge anything else via the
  `extra_meta=` argument.

```{warning}
The snapshot is **not** an automatic upload to cloud storage — that's
the orchestrator's job.  See "What's still on you, the orchestrator"
below for the upload step and why MADDENING doesn't do it for you.
```

## Supported URL schemes

`RESUME_FROM_URL` and the underlying
{func}`maddening.core.simulation.checkpoint.download_and_load_state`
accept:

| Scheme | Behaviour |
|---|---|
| `file:///path/to/snap.npz` | Local file copy.  Useful for local testing and shared-filesystem clusters. |
| `http://…/snap.npz` | HTTP GET via the stdlib `urllib`.  No auth headers (yet); use a presigned URL if you need them. |
| `https://…/snap.npz` | Same as `http://` over TLS. |
| Bare path (`/path/to/snap.npz`) | Treated as `file://`. |
| `s3://`, `gs://`, `azure://` | **Not yet wired** — call out to your orchestrator's CLI (`aws s3 cp`, `gsutil cp`) and present a presigned `https://` URL instead. |

## What's still on you, the orchestrator

The MADDENING layer deliberately stops at "write the local file" and
"read a URL".  That gives you room to choose:

* **Where to push the snapshot** — S3, GCS, Azure Blob, Selkies
  volume, NFS, a raw HTTP server.  The hook writes locally; you
  upload.  Typical pattern: set `MADDENING_SNAPSHOT_DIR` to a
  bind-mounted volume that survives the VM, then have the
  orchestrator pick the latest file from there.
* **How to discover the latest snapshot** — by filename
  convention, by reading the manifest's `extra.session_id`, by
  listing the bucket sorted by mtime — your call.
* **What presigned URL to hand to the next VM** — RunPod, AWS, GCP
  all support short-lived URLs; pass that as `RESUME_FROM_URL` on
  the relaunch.

When MADDENING grows native `s3://` / `gs://` support (planned for
v0.3, blocked on the cloud-storage abstraction in #8 follow-up),
this whole layer collapses to one `RESUME_FROM_URL` and the
orchestrator's CLI calls go away.

## The full preempt-resume contract

1. Hook fires on `CloudSession._on_preemption_signal()` (called by
   the SkyPilot preemption monitor thread).
2. Hook calls
   {func}`~maddening.core.simulation.checkpoint.save_state_with_manifest`
   with the configured snapshot path + extra meta.
3. Hook returns; the `CloudSession` continues into teardown.
4. **(orchestrator)** picks up the local snapshot + manifest, uploads.
5. **(orchestrator)** relaunches the VM with `RESUME_FROM_URL=...`.
6. New VM's entrypoint reads `RESUME_FROM_URL` and calls
   {func}`~maddening.cloud.entrypoint.resume_from_url` →
   {func}`~maddening.core.simulation.checkpoint.download_and_load_state`.
7. `download_and_load_state` fetches the `.npz` + `.manifest.json`
   into a per-call temp dir (so concurrent resumes don't collide),
   then calls `load_state_with_manifest` which verifies the hash
   and schema version, then restores the state.
8. FastAPI server binds the port.  The new VM picks up where the
   old one left off.

If anything in steps 6-7 fails, the entry-point **logs and
continues with the in-memory (fresh) state** — a failed resume
should not block a healthy server from starting.  Lab convention:
have your orchestrator notify you if `RESUME_FROM_URL` was set but
the manifest didn't apply.

## Disabling the integrity check

For one-off loads of pre-v0.2 checkpoints that don't have a manifest:

```python
from maddening.core.simulation.checkpoint import (
    download_and_load_state,
)
download_and_load_state(
    gm, url, skip_integrity_check=True,
)
```

The `entrypoint.resume_from_url` helper passes the flag through.
**Do not use `skip_integrity_check=True` in production** — the whole
point of the manifest is to catch the silent-corruption case.

## Static-data: what gets restored, what doesn't

Following the v0.2 #3 contract, {attr}`static_data
<maddening.core.node.SimulationNode.static_data>` is **not** in the
`.npz`.  After a resume:

* {meth}`~maddening.core.node.SimulationNode.initial_state` outputs
  (state, meta) → restored from the snapshot.
* `static_data` (meshes, lookup tables) → rebuilt from `self.params`
  during your code's graph reconstruction.  See {ref}`design-static-data`
  in DESIGN.md.

If you rebuild the graph in code identical to the pre-preemption
process and call `load_state`, both pieces match.  If you change
the graph topology, `load_state` raises a `ValueError` listing the
nodes/fields that don't match — better than silently broadcasting
garbage.

## Test coverage and what's deferred

The file:// path is fully unit-covered in
`tests/cloud/test_preempt_checkpoint.py` — every codepath above
runs against a `_FakeCloudSession` + local tempfile.  What's
*not* yet covered:

* End-to-end RunPod spot preemption (requires real credentials).
* `s3://` / `gs://` / `azure://` URL schemes (orchestrator's
  problem until the cloud-storage abstraction lands).
* Multi-snapshot lifecycle (last-N retention, garbage collection).

For the first two, the trade-off is: until you wire them, your
orchestrator does the upload step explicitly with a CLI call.  The
MADDENING contract is "write local file → orchestrator handles
transport → entrypoint reads URL"; everything else is a
nice-to-have for v0.3.

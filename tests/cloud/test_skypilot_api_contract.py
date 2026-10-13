"""``_skypilot.py``'s calls match the SkyPilot that is actually installed.

``src/maddening/cloud/_skypilot.py`` was byte-identical from v0.1.0 to
v0.4.0 and was written against the pre-0.7 SkyPilot API, while the declared
floor has always been ``skypilot>=0.11``.  Every call in it was wrong:

* ``sky.launch(..., detach_run=True)`` — ``detach_run`` was removed with the
  client-server split, so this raised ``TypeError``;
* ``sky.launch`` / ``sky.status`` / ``sky.down`` return a ``RequestId``, a
  ``str`` subclass, so ``sky.status(...)[0]`` was a *character* and
  ``.get("handle", {})`` on it raised ``AttributeError``;
* ``getattr(sky, config.cloud.upper())`` never matched — the classes are
  ``sky.RunPod``, not ``sky.RUNPOD`` — so every launch fell back to GCP.

Nothing caught any of it, because the only test of the module substituted a
fake ``sky`` mirroring the stale signatures: the suite proved the code
consistent with itself.  These tests compare the module against the real
library with :func:`inspect.signature`, and hold the fake to the same
standard, so a fake cannot drift again and the *next* SkyPilot API break
fails here instead of on a live provider.  They need no cloud account.

What they do **not** establish: that a teardown releases the VM or that a
preemption callback fires.  A signature-correct port is not a verified port.
See MADD-ANO-016.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

sky = pytest.importorskip(
    "sky",
    reason="SkyPilot is an optional extra (maddening[cloud]); without it "
           "there is no installed signature to compare against",
)

from tests.cloud.test_skypilot_ports import _Recorder  # noqa: E402

MODULE = (
    Path(__file__).resolve().parents[2]
    / "src" / "maddening" / "cloud" / "_skypilot.py"
)

#: ``sky`` functions that return a ``RequestId`` the caller must resolve.
_REQUEST_RETURNING = ("launch", "status", "down")

#: The two ways a ``RequestId`` is resolved.
_RESOLVERS = ("get", "stream_and_get")

_PLACEHOLDER = object()


def _tree() -> ast.Module:
    return ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))


def _sky_calls(node: ast.AST) -> list[tuple[str, ast.Call]]:
    """Every ``sky.<name>(...)`` call under *node*, as ``(name, call)``."""
    found = []
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "sky"):
            found.append((sub.func.attr, sub))
    return found


def _enclosing_functions() -> list[ast.FunctionDef]:
    return [n for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef)]


# ---------------------------------------------------------------------------
# The module calls the library that is installed
# ---------------------------------------------------------------------------

class TestTheModuleMatchesTheInstalledSkyPilot:
    def test_the_module_actually_calls_sky(self):
        """Without this the rest of the class could pass on an empty list."""
        names = {name for name, _ in _sky_calls(_tree())}
        assert {"launch", "status", "down"} <= names, sorted(names)

    def test_every_attribute_the_module_uses_exists_on_the_installed_sky(self):
        missing = sorted(
            {name for name, _ in _sky_calls(_tree()) if not hasattr(sky, name)}
        )
        assert not missing, (
            f"_skypilot.py calls sky.{{{','.join(missing)}}}, which the "
            f"installed SkyPilot {sky.__version__} does not provide"
        )

    def test_every_call_binds_against_the_installed_signature(self):
        """The check that ``detach_run`` would have failed.

        Each ``sky.<name>(...)`` is bound against
        ``inspect.signature(getattr(sky, name))``, so a parameter that was
        renamed or removed upstream fails here rather than at launch time.
        """
        failures = []
        for name, call in _sky_calls(_tree()):
            target = getattr(sky, name)
            try:
                signature = inspect.signature(target)
            except (TypeError, ValueError) as exc:      # pragma: no cover
                failures.append(f"sky.{name}: uninspectable ({exc})")
                continue
            positional = [_PLACEHOLDER] * len(call.args)
            keywords = {
                kw.arg: _PLACEHOLDER for kw in call.keywords
                if kw.arg is not None
            }
            try:
                signature.bind_partial(*positional, **keywords)
            except TypeError as exc:
                failures.append(
                    f"line {call.lineno}: sky.{name}("
                    f"{len(positional)} positional, "
                    f"{sorted(keywords)}) does not fit {name}{signature}: "
                    f"{exc}"
                )
        assert not failures, (
            f"_skypilot.py does not match SkyPilot {sky.__version__}:\n  "
            + "\n  ".join(failures)
        )

    def test_sky_launch_no_longer_takes_detach_run(self):
        """The named defect, pinned from both ends.

        If a future SkyPilot reintroduced ``detach_run`` the first assertion
        would go quiet; the second keeps the module honest regardless.  The
        second reads the parse tree rather than the text, because the
        module's own history note names the parameter in prose.
        """
        assert "detach_run" not in inspect.signature(sky.launch).parameters
        passed = [
            f"line {call.lineno}: sky.{name}"
            for name, call in _sky_calls(_tree())
            for kw in call.keywords
            if kw.arg == "detach_run"
        ]
        assert not passed, "\n  ".join(passed)


# ---------------------------------------------------------------------------
# RequestIds are resolved, never used directly
# ---------------------------------------------------------------------------

class TestEveryRequestIdIsResolved:
    def test_a_request_id_is_a_str_subclass(self):
        """Why the defect was silent rather than loud.

        ``sky.status(...)[0]`` indexed a character instead of raising, and
        the ``AttributeError`` only surfaced on ``.get``, inside a
        background thread.  If ``RequestId`` ever stops being a ``str``,
        this comment stops being the explanation.
        """
        from sky.server.common import RequestId

        assert issubclass(RequestId, str)

    def test_the_request_returning_calls_still_return_a_request_id(self):
        """If upstream made these return their payload directly, resolving
        them through ``sky.get`` would become the wrong thing to do."""
        for name in _REQUEST_RETURNING:
            annotation = inspect.signature(getattr(sky, name)).return_annotation
            assert "RequestId" in str(annotation), (
                f"sky.{name} returns {annotation!r}, not a RequestId; "
                f"_skypilot.py resolves it with sky.get and would now be "
                f"resolving something that is already the answer"
            )

    def test_no_request_returning_call_is_used_without_being_resolved(self):
        """``sky.launch(...)``/``status(...)``/``down(...)`` must be handed
        to ``sky.get`` or ``sky.stream_and_get`` — directly, or through a
        local name — before anything is read off it."""
        unresolved = []
        for function in _enclosing_functions():
            resolved_args: set[str] = set()
            resolved_calls: set[int] = set()
            for name, call in _sky_calls(function):
                if name not in _RESOLVERS:
                    continue
                for arg in list(call.args) + [k.value for k in call.keywords]:
                    if isinstance(arg, ast.Name):
                        resolved_args.add(arg.id)
                    elif isinstance(arg, ast.Call):
                        resolved_calls.add(id(arg))

            assigned_from: dict[int, str] = {}
            for sub in ast.walk(function):
                if (isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Call)
                        and len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Name)):
                    assigned_from[id(sub.value)] = sub.targets[0].id

            for name, call in _sky_calls(function):
                if name not in _REQUEST_RETURNING:
                    continue
                if id(call) in resolved_calls:
                    continue
                bound = assigned_from.get(id(call))
                if bound is not None and bound in resolved_args:
                    continue
                unresolved.append(
                    f"{function.name} line {call.lineno}: sky.{name}(...) is "
                    f"neither passed to sky.get/stream_and_get nor bound to a "
                    f"name that is"
                )
        assert not unresolved, "\n  ".join(unresolved)

    def test_the_module_never_indexes_a_call_result_directly(self):
        """``sky.status(cluster_names=[...])[0]`` is the exact shape of the
        defect: legal on a ``str``, and wrong."""
        offenders = []
        for sub in ast.walk(_tree()):
            if isinstance(sub, (ast.Subscript, ast.Attribute)):
                inner = sub.value
                if (isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and isinstance(inner.func.value, ast.Name)
                        and inner.func.value.id == "sky"
                        and inner.func.attr in _REQUEST_RETURNING):
                    offenders.append(f"line {sub.lineno}: sky.{inner.func.attr}(...)")
        assert not offenders, (
            "a RequestId is a str, so indexing or attribute access on one "
            "silently does the wrong thing:\n  " + "\n  ".join(offenders)
        )


# ---------------------------------------------------------------------------
# The fake cannot drift away from the real library again
# ---------------------------------------------------------------------------

class TestTheFakeCannotDrift:
    """``tests/cloud/test_skypilot_ports.py`` substitutes ``_Recorder`` for
    ``sky``.  The previous fake declared ``launch(self, task, cluster_name,
    detach_run=True)`` and returned a list of dicts — the pre-0.7 API — so
    every test of the module passed against a library that no longer
    existed.  A fake is only evidence while it is the shape of the real
    thing."""

    def test_the_fake_declares_no_parameter_the_real_sky_lacks(self):
        fake = _Recorder()
        offenders = []
        for name in _REQUEST_RETURNING + _RESOLVERS:
            real = inspect.signature(getattr(sky, name)).parameters
            for param in inspect.signature(getattr(fake, name)).parameters.values():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                if param.name not in real:
                    offenders.append(
                        f"_Recorder.{name} declares {param.name!r}, which "
                        f"sky.{name} does not have"
                    )
        assert not offenders, "\n  ".join(offenders)

    def test_the_fake_accepts_every_call_the_module_makes(self):
        """Bound against the fake as well as the real library, so a call the
        module makes can never be exercised only against the double."""
        fake = _Recorder()
        failures = []
        for name, call in _sky_calls(_tree()):
            target = getattr(fake, name, None)
            if target is None:
                failures.append(f"_Recorder has no {name!r}")
                continue
            keywords = {
                kw.arg: _PLACEHOLDER for kw in call.keywords if kw.arg is not None
            }
            try:
                inspect.signature(target).bind_partial(
                    *([_PLACEHOLDER] * len(call.args)), **keywords
                )
            except TypeError as exc:
                failures.append(f"_Recorder.{name}: {exc}")
        assert not failures, "\n  ".join(failures)

    def test_the_fakes_request_id_is_a_str_subclass_like_the_real_one(self):
        """Otherwise the fake cannot reproduce the failure mode at all."""
        from tests.cloud.test_skypilot_ports import _RequestId

        assert issubclass(_RequestId, str)

    def test_the_fake_refuses_to_resolve_something_it_never_issued(self):
        """The fake's ``get`` is the thing that makes an unresolved
        RequestId visible in the port's own tests."""
        fake = _Recorder()
        with pytest.raises(AssertionError, match="not a RequestId"):
            fake.get("some-string")


# ---------------------------------------------------------------------------
# The module's own claims about SkyPilot
# ---------------------------------------------------------------------------

class TestTheClaimsInTheSource:
    def test_the_upper_case_cloud_lookup_could_not_have_worked(self):
        """The pre-port code did ``getattr(sky, config.cloud.upper())``.

        SkyPilot spells an acronym cloud in upper case (``sky.AWS``) and
        everything else in camel case, so the lookup missed for exactly the
        two providers MADDENING's ``[cloud]`` extra installs -- RunPod and
        Lambda -- and every such launch took the ``or sky.GCP()`` fallback.
        For ``aws``/``gcp`` it hit, but returned the *class* where the other
        branch returned an *instance*.
        """
        for provider in ("runpod", "lambda"):
            assert not hasattr(sky, provider.upper()), (
                f"sky.{provider.upper()} exists after all; the comment in "
                f"launch_vm explaining why the old lookup always missed for "
                f"this provider is no longer accurate"
            )
            assert hasattr(sky, provider.title()) or hasattr(sky, "RunPod")
        assert hasattr(sky, "RunPod") and hasattr(sky, "Lambda")
        # and the resolver the port uses does find them
        from maddening.cloud.launcher import _resolve_sky_cloud_class

        assert _resolve_sky_cloud_class(sky, "runpod") is sky.RunPod
        assert _resolve_sky_cloud_class(sky, "lambda") is sky.Lambda

    def test_cluster_status_has_no_preempted_member(self):
        """``monitor_preemption`` keeps ``PREEMPTED`` in its trigger set and
        says in a comment that SkyPilot does not produce it.  If that stops
        being true the comment is wrong and STOPPED may no longer be the
        signal."""
        from sky.utils.status_lib import ClusterStatus

        assert "PREEMPTED" not in {s.name for s in ClusterStatus}
        assert "STOPPED" in {s.name for s in ClusterStatus}

    def test_a_cluster_record_answers_to_dict_style_get(self):
        """``check_status`` and ``launch_vm`` read the status records with
        ``.get(...)``, as ``launcher.py`` does.  SkyPilot calls that
        backwards compatibility and has a TODO to remove it in 0.13."""
        from sky.schemas.api.responses import StatusResponse

        assert callable(getattr(StatusResponse, "get", None))
        assert "status" in StatusResponse.model_fields
        assert "handle" in StatusResponse.model_fields

    def test_the_declared_floor_is_at_or_below_the_installed_version(self):
        """A contract test taken against a version below the declared floor
        would be measuring the wrong library."""
        pyproject = (
            Path(__file__).resolve().parents[2] / "pyproject.toml"
        ).read_text(encoding="utf-8")
        floors = {
            tuple(int(p) for p in m.split("."))
            for m in re.findall(r'skypilot\[[^\]]*\]>=([0-9.]+)', pyproject)
        }
        assert floors, "no skypilot floor found in pyproject.toml"
        installed = tuple(
            int(p) for p in sky.__version__.split(".")[:2] if p.isdigit()
        )
        assert installed >= max(floors)[:len(installed)], (
            f"installed skypilot {sky.__version__} is below the declared "
            f"floor {max(floors)}"
        )

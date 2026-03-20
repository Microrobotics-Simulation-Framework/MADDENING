"""CloudGroup — orchestrate multiple inter-communicating cloud jobs.

Provisions multiple VMs, runs a coordinator on rank-0 for rendezvous,
distributes the ZMQ topology, and manages the group lifecycle.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

from maddening.cloud.launcher import (
    CloudJob,
    CloudLauncher,
    CostPolicy,
    JobConfig,
    JobPhase,
    LaunchError,
)

logger = logging.getLogger(__name__)


class GroupFailureMode(Enum):
    """How to respond when a worker fails."""
    TEARDOWN_ALL = "teardown_all"
    ISOLATE = "isolate"


@dataclass(frozen=True)
class GroupConfig:
    """Configuration for a cloud job group."""
    failure_mode: GroupFailureMode = GroupFailureMode.TEARDOWN_ALL
    rendezvous_timeout: float = 300.0
    heartbeat_interval: float = 10.0
    heartbeat_timeout: float = 30.0
    coordinator_port: int = 5580


@dataclass(frozen=True)
class SubgraphSpec:
    """One unit of work to be placed on one VM."""
    subgraph_id: str
    job_config: JobConfig
    zmq_ports: dict[str, int] = field(default_factory=dict)


class CloudGroup:
    """Orchestrates a group of inter-communicating cloud jobs.

    Provisions VMs with a coordinator on rank-0 for rendezvous:
    1. Rank-0 is provisioned first
    2. Coordinator process started on rank-0
    3. Remaining workers provisioned in parallel with COORDINATOR_ADDR injected
    4. All workers register with coordinator
    5. Coordinator broadcasts ZMQ topology
    6. Workers start simulation
    """

    def __init__(
        self,
        specs: list[SubgraphSpec],
        edges: list[dict],
        group_config: GroupConfig = GroupConfig(),
        credentials_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if not specs:
            raise ValueError("CloudGroup requires at least one SubgraphSpec")
        self._specs = list(specs)
        self._edges = edges
        self._config = group_config
        self._credentials_path = credentials_path
        self._jobs: dict[str, CloudJob] = {}
        self._launcher = CloudLauncher(credentials_path=credentials_path)
        self._rank0_id = specs[0].subgraph_id

    @classmethod
    def from_cluster_names(
        cls,
        names: dict[str, str],
        group_config: GroupConfig = GroupConfig(),
        credentials_path: Optional[Union[str, Path]] = None,
    ) -> "CloudGroup":
        """Reconnect to an existing group of clusters."""
        group = cls.__new__(cls)
        group._specs = []
        group._edges = []
        group._config = group_config
        group._credentials_path = credentials_path
        group._launcher = CloudLauncher(credentials_path=credentials_path)
        group._rank0_id = next(iter(names))
        group._jobs = {
            sid: CloudJob.from_cluster_name(name, credentials_path=credentials_path)
            for sid, name in names.items()
        }
        return group

    @property
    def jobs(self) -> dict[str, CloudJob]:
        return dict(self._jobs)

    def provision_all(self) -> dict[str, CloudJob]:
        """Provision all VMs.

        Rank-0 is provisioned first (to get the coordinator IP), then
        remaining workers are provisioned with ``COORDINATOR_ADDR``
        injected into their environment.

        Returns ``{subgraph_id: CloudJob}`` when all VMs are UP.
        Does NOT start execution — workers are waiting at rendezvous.
        """
        if not self._specs:
            raise LaunchError("No specs to provision")

        # Phase 1: Provision rank-0
        rank0_spec = self._specs[0]
        logger.info("Provisioning rank-0: %s", rank0_spec.subgraph_id)

        rank0_config = self._inject_rank0_env(rank0_spec)
        rank0_job = self._launcher.launch(rank0_config)
        self._jobs[rank0_spec.subgraph_id] = rank0_job

        coordinator_addr = f"{rank0_job.vm_ip}:{self._config.coordinator_port}"
        logger.info("Rank-0 UP at %s, coordinator at %s",
                     rank0_job.vm_ip, coordinator_addr)

        # Phase 2: Provision remaining workers with coordinator address
        for spec in self._specs[1:]:
            logger.info("Provisioning worker: %s", spec.subgraph_id)
            worker_config = self._inject_worker_env(spec, coordinator_addr)
            worker_job = self._launcher.launch(worker_config)
            self._jobs[spec.subgraph_id] = worker_job
            logger.info("Worker %s UP at %s",
                         spec.subgraph_id, worker_job.vm_ip)

        return dict(self._jobs)

    def start(self) -> None:
        """Trigger topology broadcast from coordinator.

        After this, all workers receive their topology and begin
        MADDENING graph execution.
        """
        rank0_job = self._jobs.get(self._rank0_id)
        if rank0_job is None:
            raise LaunchError("Rank-0 not provisioned")

        # The coordinator should already be running on rank-0.
        # Send a "start" signal via SSH.
        rank0_job.ssh_run(
            f"curl -s http://localhost:{self._config.coordinator_port}/start || true",
            check=False,
        )

    def status(self) -> dict[str, dict]:
        """Non-blocking status check for all jobs."""
        return {sid: job.status() for sid, job in self._jobs.items()}

    def cost_so_far(self) -> float:
        """Sum of all jobs' estimated cost."""
        return sum(job.cost_so_far() for job in self._jobs.values())

    def teardown_all(self) -> None:
        """Tear down all VMs in the group."""
        for sid, job in self._jobs.items():
            logger.info("Tearing down %s (%s)", sid, job.cluster_name)
            try:
                job.teardown()
            except Exception:
                logger.warning("Failed to teardown %s", sid, exc_info=True)

    def teardown_one(self, subgraph_id: str) -> None:
        """Tear down a single job.  Only valid in ISOLATE mode."""
        if self._config.failure_mode != GroupFailureMode.ISOLATE:
            raise ValueError("teardown_one only valid in ISOLATE mode")
        job = self._jobs.get(subgraph_id)
        if job is None:
            raise KeyError(f"No job for subgraph '{subgraph_id}'")
        job.teardown()

    # -- Internal helpers --------------------------------------------------

    def _inject_rank0_env(self, spec: SubgraphSpec) -> JobConfig:
        """Build a JobConfig for rank-0 with coordinator setup."""
        config = spec.job_config._with_envs({
            "SUBGRAPH_ID": spec.subgraph_id,
            "COORDINATOR_PORT": str(self._config.coordinator_port),
            "IS_RANK0": "1",
            "EXPECTED_WORKERS": json.dumps(
                [s.subgraph_id for s in self._specs]
            ),
            "INTER_JOB_EDGES": json.dumps(self._edges),
        })
        # Ensure coordinator port is exposed through NAT
        coord_port = self._config.coordinator_port
        if coord_port not in config.ports:
            object.__setattr__(
                config, "ports", list(config.ports) + [coord_port],
            )
        return config

    def _inject_worker_env(
        self, spec: SubgraphSpec, coordinator_addr: str,
    ) -> JobConfig:
        """Build a JobConfig for a worker with coordinator address."""
        return spec.job_config._with_envs({
            "SUBGRAPH_ID": spec.subgraph_id,
            "COORDINATOR_ADDR": coordinator_addr,
            "IS_RANK0": "0",
        })

# alpha6.py
"""Cloud bin-packing scheduler: classical heuristics + quantum/quantum-inspired paths.

This module is both a command-line demo and a FastAPI service. It packs jobs
(generated from fixed workload profiles) onto servers, minimizing the number of
active servers subject to per-server hard limits:
- 2 vCPUs
- 4 GB ECC RAM
- 40 GB local NVMe (non-shared, non-networked)
- 10 Gbps network bandwidth

What it provides (evolution of alpha5.py):
- Classical bin-packing heuristics (first-fit / best-fit / worst-fit decreasing)
  and a concurrent per-server makespan simulation.
- A quantum path (QAOA) that formulates the problem as a Qiskit
  QuadraticProgram with binary assignment variables x_{j}_{s} and server-active
  variables y_s, then solves it on a simulated quantum circuit. The QAOA run is
  now *warm-started*: its initial state and mixer are biased toward the
  classical best-fit-decreasing solution instead of a blind uniform
  superposition, which is known to speed convergence (see Egger et al.,
  "Warm-starting quantum optimization"). Per-iteration objective values are
  recorded so a convergence curve can be plotted.
- A dependency-free simulated-annealing solver over the *same* penalty-QUBO
  objective QAOA targets. It now acts as an automatic fallback whenever Qiskit
  is unavailable, the instance exceeds the qubit budget, or the real QAOA run
  times out — so a quantum-inspired result is always produced instead of a
  hard failure.
- An exact MILP baseline (scipy.optimize.milp / HiGHS) solved directly on the
  original assignment+capacity formulation (no QUBO/Qiskit dependency), giving
  a provably-optimal server count to measure the heuristics' optimality gap
  against.
- An event-driven online scheduler with Poisson / scheduled arrivals.
- A FastAPI app (`app`) exposing /schedule, /schedule_quantum,
  /schedule_quantum_async (+ polling), /schedule_milp and /schedule_compare.
  Requests are now validated (positive server counts, known profile types,
  non-negative counts) instead of failing unpredictably downstream.

Running `python alpha6.py` launches the interactive CLI demo; `--demo-online`
runs the non-interactive online-arrival demo. Qiskit is optional: without it
the quantum path degrades gracefully to the simulated-annealing fallback
rather than failing outright.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys
import heapq
import math
import random
import logging
from dataclasses import dataclass, replace
from statistics import mean

try:
    from qiskit_optimization import QuadraticProgram
except Exception:
    QuadraticProgram = None


# ---------------------------------------------------------------------------
# Logging (replaces the scattered print()-based diagnostics from alpha5).
# CLI report output (the tables in main() / online_scheduler_with_arrivals)
# stays on print() since that's user-facing program output, not a diagnostic.
# ---------------------------------------------------------------------------
logger = logging.getLogger("alpha6")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _log_memory_error(context: str, exc: Exception, extra: Optional[Dict[str, object]] = None) -> None:
    """Append MemoryError diagnostics to qp_memory_error.log (best-effort).

    Writes timestamp, context, exception, traceback, and a few runtime hints.
    """
    logger.error(f"MemoryError during {context}: {exc}")
    try:
        import datetime, traceback, os, json
        fn = os.path.join(os.getcwd(), "qp_memory_error.log")
        info = {
            "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "context": context,
            "exception": f"{type(exc).__name__}: {str(exc)}",
            "rust_backtrace": os.environ.get("RUST_BACKTRACE", ""),
        }
        if extra:
            info["extra"] = extra
        with open(fn, "a", encoding="utf-8") as f:
            f.write("=== MEMORY ERROR ===\n")
            f.write(json.dumps(info) + "\n")
            f.write("traceback:\n")
            f.write(traceback.format_exc())
            f.write("\n\n")
    except Exception:
        # Best-effort only; never raise from logging
        pass


# Module-level resource constants (single source of truth)
CORES_PER_SERVER: int = 2
RAM_PER_SERVER: float = 4.0
NVME_PER_SERVER: float = 40.0
BANDWIDTH_PER_SERVER: float = 10.0

# Named magic number constants
MIN_JOB_DURATION: float = 1e-6
FLOAT_EPSILON: float = 1e-9
HIGH_STRESS_THRESHOLD: float = 80.0
MEDIUM_STRESS_THRESHOLD: float = 50.0
COST_PER_SERVER_PER_HOUR: float = 0.50

# Quantum / quantum-inspired solver tuning
QAOA_MAX_QUBITS: int = 20            # statevector-sim budget (2**20 states)
QAOA_WEB_TIMEOUT_SEC: float = 60.0   # wall-clock cap for the web-facing QAOA attempt
SA_PENALTY_WEIGHT: float = 10.0      # constraint-violation penalty weight in the SA QUBO cost
SA_DEFAULT_ITERATIONS: int = 4000    # simulated-annealing sweep count
MILP_TIME_LIMIT_SEC: float = 20.0    # HiGHS wall-clock cap
MILP_MAX_VARS: int = 400             # skip the exact solver above this many binary vars


JOB_PROFILES: Dict[str, Dict[str, float]] = {
    # (cores, ram_gb, nvme_gb, duration_seconds)
    "web_api": {"cores": 1, "ram": 1.0, "nvme": 1.0, "bandwidth": 0.5, "duration": 5.0},
    "batch": {"cores": 2, "ram": 2.0, "nvme": 5.0, "bandwidth": 0.1, "duration": 300.0},
    "memory": {"cores": 1, "ram": 3.0, "nvme": 5.0, "bandwidth": 0.2, "duration": 60.0},
    "io_heavy": {"cores": 1, "ram": 1.0, "nvme": 30.0, "bandwidth": 2.0, "duration": 180.0},
}


@dataclass
class Job:
    """A single schedulable unit of work.

    Kept dict-compatible (`.get`, `[...]`, `.copy()`) on purpose: the packing
    engine, QP builders and online scheduler all pre-date this dataclass and
    are written against dict-style access. The shim lets every existing call
    site keep working unchanged while giving new code (the SA/MILP solvers,
    warm-start helpers) real attributes and type checking.
    """

    id: int
    type: str
    cores: int
    ram: float = 0.0
    nvme: float = 0.0
    bandwidth: float = 0.0
    duration: float = 10.0
    arrival_time: float = 0.0
    deadline: Optional[float] = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def copy(self) -> "Job":
        return replace(self)


def _resolve_server_limits(
    cores_per_server: Optional[int] = None,
    ram_per_server: Optional[float] = None,
    nvme_per_server: Optional[float] = None,
    bandwidth_per_server: Optional[float] = None,
) -> Tuple[int, float, float, float]:
    """Resolve per-server resource limits from overrides or module defaults."""
    return (
        cores_per_server if cores_per_server is not None else CORES_PER_SERVER,
        ram_per_server if ram_per_server is not None else RAM_PER_SERVER,
        nvme_per_server if nvme_per_server is not None else NVME_PER_SERVER,
        bandwidth_per_server if bandwidth_per_server is not None else BANDWIDTH_PER_SERVER,
    )


def _job_sort_key(job: Job) -> Tuple[float, float, float, float]:
    """Shared sort key for packing heuristics."""
    return (
        job.get('cores', 0),
        job.get('ram', 0.0),
        job.get('nvme', 0.0),
        job.get('bandwidth', 0.0),
    )


def _create_server_usage_record(job: Job) -> Dict[str, Any]:
    """Create a new server usage record seeded with a single job."""
    return {
        'cores': job.get('cores', 0),
        'ram': job.get('ram', 0.0),
        'nvme': job.get('nvme', 0.0),
        'bandwidth': job.get('bandwidth', 0.0),
        'jobs': [job['id']],
    }


def _job_fits_within_limits(
    job: Job,
    cores_cap: int,
    ram_cap: float,
    nvme_cap: float,
    bandwidth_cap: float,
) -> bool:
    """Check if a job can ever fit on an empty server."""
    return (
        job.get('cores', 0) <= cores_cap and
        job.get('ram', 0.0) <= ram_cap and
        job.get('nvme', 0.0) <= nvme_cap and
        job.get('bandwidth', 0.0) <= bandwidth_cap
    )


def _can_place_on_usage(
    usage: Dict[str, Any],
    job: Job,
    cores_cap: int,
    ram_cap: float,
    nvme_cap: float,
    bandwidth_cap: float,
) -> bool:
    """Check whether a job fits on a partially used server."""
    return (
        usage['cores'] + job.get('cores', 0) <= cores_cap and
        usage['ram'] + job.get('ram', 0.0) <= ram_cap and
        usage['nvme'] + job.get('nvme', 0.0) <= nvme_cap and
        usage['bandwidth'] + job.get('bandwidth', 0.0) <= bandwidth_cap
    )


def _remaining_capacity_tuple(
    usage: Dict[str, Any],
    job: Job,
    cores_cap: int,
    ram_cap: float,
    nvme_cap: float,
    bandwidth_cap: float,
) -> Tuple[float, float, float, float]:
    """Compute remaining capacity after placing a job on a server."""
    return (
        cores_cap - (usage['cores'] + job.get('cores', 0)),
        ram_cap - (usage['ram'] + job.get('ram', 0.0)),
        nvme_cap - (usage['nvme'] + job.get('nvme', 0.0)),
        bandwidth_cap - (usage['bandwidth'] + job.get('bandwidth', 0.0)),
    )


def _apply_job_to_usage(usage: Dict[str, Any], job: Job) -> None:
    """Mutate a server usage record by assigning a job to it."""
    usage['cores'] += job.get('cores', 0)
    usage['ram'] += job.get('ram', 0.0)
    usage['nvme'] += job.get('nvme', 0.0)
    usage['bandwidth'] += job.get('bandwidth', 0.0)
    usage['jobs'].append(job['id'])


def prompt_workloads() -> Dict[str, int]:
    """Prompt the user to pick workload types and number of jobs for each.

    Returns:
        Dict mapping profile type names to desired job counts (all >= 0).

    Raises:
        No exceptions raised; invalid input triggers retry loop.
    """
    print("Select workload types to include and how many jobs of each to generate.")
    print("Available types:")
    for k, v in JOB_PROFILES.items():
        print(f" - {k}: cores={v['cores']}, ram={v['ram']}GB, nvme={v['nvme']}GB")
    choices: Dict[str, int] = {}
    for k in JOB_PROFILES.keys():
        while True:
            try:
                val = input(f"Number of jobs of type '{k}' (enter 0 to skip): ").strip()
                if val == "":
                    count = 0
                else:
                    count = int(val)
                if count < 0:
                    raise ValueError()
                choices[k] = count
                break
            except ValueError:
                print("Please enter a non-negative integer.")
    return choices


def generate_jobs(profile_counts: Dict[str, int]) -> List[Job]:
    """Generate Job records from selected profiles.

    Args:
        profile_counts: Dict mapping profile type (str) to desired count (int).
    Returns:
        List of Job records. Each job ID is unique, starting from 0.

    Raises:
        ValueError: If any count is negative or profile type is invalid.

    Example:
        >>> jobs = generate_jobs({'web_api': 2, 'batch': 1})
        >>> len(jobs)  # 3 jobs total
        3
    """
    # Input validation
    if not isinstance(profile_counts, dict):
        raise ValueError("profile_counts must be a dictionary")

    for ptype, count in profile_counts.items():
        if not isinstance(count, int):
            raise ValueError(f"Count for profile '{ptype}' must be int, got {type(count).__name__}")
        if count < 0:
            raise ValueError(f"Count for profile '{ptype}' must be non-negative, got {count}")

    jobs: List[Job] = []
    jid = 0
    for ptype, count in profile_counts.items():
        profile = JOB_PROFILES.get(ptype)
        if profile is None:
            raise ValueError(f"Unknown profile type '{ptype}'. Valid types: {list(JOB_PROFILES.keys())}")
        for _ in range(count):
            jobs.append(Job(
                id=jid,
                type=ptype,
                cores=int(profile["cores"]),
                ram=float(profile["ram"]),
                nvme=float(profile["nvme"]),
                bandwidth=float(profile.get("bandwidth", 0.0)),
                duration=float(profile.get("duration", 10.0)),
            ))
            jid += 1
    return jobs


# -----------------
# Packing & simulation helpers (module-level)
# -----------------
def simulate_servers(
    server_usage_list: List[Dict[str, Any]],
    jobs_list: List[Job],
    verbose: bool = True,
    cores_per_server: Optional[int] = None,
    ram_per_server: Optional[float] = None,
    nvme_per_server: Optional[float] = None,
    bandwidth_per_server: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Simulate each server running its assigned jobs with resource constraints.

    Returns (per_server_records, makespan).
    """
    cores_cap, ram_cap, nvme_cap, bandwidth_cap = _resolve_server_limits(
        cores_per_server=cores_per_server,
        ram_per_server=ram_per_server,
        nvme_per_server=nvme_per_server,
        bandwidth_per_server=bandwidth_per_server,
    )
    job_map = {j['id']: j for j in jobs_list}
    per_server_records: List[Dict[str, Any]] = []
    overall_makespan = 0.0

    for su in server_usage_list:
        rec: Dict[str, Any] = {'jobs': [], 'completion': 0.0}
        pending_jobs = [job_map[jid].copy() for jid in su.get('jobs', [])]
        t = 0.0
        running_heap: List[Tuple[float, Any, Job]] = []
        used_cores = 0
        used_ram = 0.0
        used_nvme = 0.0
        used_bw = 0.0

        while pending_jobs or running_heap:
            started = False
            i = 0
            while i < len(pending_jobs):
                job = pending_jobs[i]
                if (used_cores + job.get('cores', 0) <= cores_cap and
                    used_ram + job.get('ram', 0.0) <= ram_cap and
                    used_nvme + job.get('nvme', 0.0) <= nvme_cap and
                    used_bw + job.get('bandwidth', 0.0) <= bandwidth_cap):
                    start = t
                    dur = max(job.get('duration', 0.0), MIN_JOB_DURATION)
                    end = start + dur
                    # tie-break on job id so equal end-times never compare dicts
                    heapq.heappush(running_heap, (end, job['id'], job))
                    used_cores += job.get('cores', 0)
                    used_ram += job.get('ram', 0.0)
                    used_nvme += job.get('nvme', 0.0)
                    used_bw += job.get('bandwidth', 0.0)
                    rec['jobs'].append({'id': job['id'], 'start': start, 'end': end})
                    pending_jobs.pop(i)
                    started = True
                else:
                    i += 1

            if not started:
                if running_heap:
                    next_end = running_heap[0][0]
                    if next_end <= t + FLOAT_EPSILON:
                        t = next_end + MIN_JOB_DURATION
                    else:
                        t = next_end
                    while running_heap and running_heap[0][0] <= t:
                        _, _, finished_job = heapq.heappop(running_heap)
                        used_cores = max(0,   used_cores - finished_job.get('cores', 0))
                        used_ram   = max(0.0, used_ram   - finished_job.get('ram', 0.0))
                        used_nvme  = max(0.0, used_nvme  - finished_job.get('nvme', 0.0))
                        used_bw    = max(0.0, used_bw    - finished_job.get('bandwidth', 0.0))
                else:
                    # No job could be started and no jobs are running: infeasible
                    if pending_jobs:
                        if verbose:
                            for job in pending_jobs:
                                logger.warning(f"Job {job['id']} could not be placed on any server (infeasible).")
                    break

        if rec['jobs']:
            rec['completion'] = max(j['end'] for j in rec['jobs'])
        else:
            rec['completion'] = 0.0
        overall_makespan = max(overall_makespan, rec['completion'])
        per_server_records.append(rec)

    return per_server_records, overall_makespan


def _pack_jobs(
    jobs_list: List[Job],
    score_fn: Callable[[Dict[str, Any], Job, int, float, float, float], Any],
    prefer_high_score: bool = False,
    cores_per_server: Optional[int] = None,
    ram_per_server: Optional[float] = None,
    nvme_per_server: Optional[float] = None,
    bandwidth_per_server: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Shared packing engine for the bin-packing heuristics."""
    cores_cap, ram_cap, nvme_cap, bandwidth_cap = _resolve_server_limits(
        cores_per_server=cores_per_server,
        ram_per_server=ram_per_server,
        nvme_per_server=nvme_per_server,
        bandwidth_per_server=bandwidth_per_server,
    )
    jobs_sorted = sorted(jobs_list, key=_job_sort_key, reverse=True)
    servers_local: List[Dict[str, Any]] = []

    for job in jobs_sorted:
        if not _job_fits_within_limits(job, cores_cap, ram_cap, nvme_cap, bandwidth_cap):
            logger.warning(f"Job {job['id']} ({job.get('type','?')}) exceeds server capacity — skipped.")
            continue

        chosen_idx = None
        chosen_score = None
        for i, su in enumerate(servers_local):
            if not _can_place_on_usage(su, job, cores_cap, ram_cap, nvme_cap, bandwidth_cap):
                continue
            score = score_fn(su, job, cores_cap, ram_cap, nvme_cap, bandwidth_cap)
            if chosen_score is None:
                chosen_idx = i
                chosen_score = score
                continue
            if prefer_high_score:
                if score > chosen_score:
                    chosen_idx = i
                    chosen_score = score
            elif score < chosen_score:
                chosen_idx = i
                chosen_score = score

        if chosen_idx is None:
            servers_local.append(_create_server_usage_record(job))
        else:
            _apply_job_to_usage(servers_local[chosen_idx], job)

    return servers_local


def _normalized_remaining_score(
    usage: Dict[str, Any],
    job: Job,
    cores_cap: int,
    ram_cap: float,
    nvme_cap: float,
    bandwidth_cap: float,
) -> float:
    """Sum of normalized remaining capacity across all dimensions after a placement.

    Lower means a tighter fit (best-fit takes the minimum); higher means a looser
    fit (worst-fit takes the maximum). Normalizing per dimension lets every
    resource contribute equally instead of being dominated by raw magnitude.
    """
    rem = _remaining_capacity_tuple(usage, job, cores_cap, ram_cap, nvme_cap, bandwidth_cap)
    return (rem[0] / max(1, cores_cap) +
            rem[1] / max(1.0, ram_cap) +
            rem[2] / max(1.0, nvme_cap) +
            rem[3] / max(1.0, bandwidth_cap))


def pack_first_fit_decreasing(
    jobs_list: List[Job],
    cores_per_server: Optional[int] = None,
    ram_per_server: Optional[float] = None,
    nvme_per_server: Optional[float] = None,
    bandwidth_per_server: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """First-Fit Decreasing bin-packing algorithm with dynamic server creation."""
    return _pack_jobs(
        jobs_list,
        score_fn=lambda su, job, *_: 0,
        prefer_high_score=False,
        cores_per_server=cores_per_server,
        ram_per_server=ram_per_server,
        nvme_per_server=nvme_per_server,
        bandwidth_per_server=bandwidth_per_server,
    )


def pack_best_fit_decreasing(
    jobs_list: List[Job],
    cores_per_server: Optional[int] = None,
    ram_per_server: Optional[float] = None,
    nvme_per_server: Optional[float] = None,
    bandwidth_per_server: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Best-Fit Decreasing bin-packing algorithm with dynamic server creation."""
    return _pack_jobs(
        jobs_list,
        score_fn=_normalized_remaining_score,
        prefer_high_score=False,
        cores_per_server=cores_per_server,
        ram_per_server=ram_per_server,
        nvme_per_server=nvme_per_server,
        bandwidth_per_server=bandwidth_per_server,
    )


def pack_worst_fit(
    jobs_list: List[Job],
    cores_per_server: Optional[int] = None,
    ram_per_server: Optional[float] = None,
    nvme_per_server: Optional[float] = None,
    bandwidth_per_server: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Worst-Fit bin-packing algorithm with dynamic server creation."""
    return _pack_jobs(
        jobs_list,
        score_fn=_normalized_remaining_score,
        prefer_high_score=True,
        cores_per_server=cores_per_server,
        ram_per_server=ram_per_server,
        nvme_per_server=nvme_per_server,
        bandwidth_per_server=bandwidth_per_server,
    )


def build_cloud_quadratic_program(servers: int, jobs: List[Job]) -> Tuple[Optional[QuadraticProgram], List[str]]:
    """Construct a Qiskit QuadraticProgram for the bin-packing problem.

    Each server has:
    - 2 vCPUs (cores)
    - 4 GB ECC RAM
    - 40 GB local NVMe storage
    - 10 Gbps bandwidth

    Args:
        servers: Number of identical servers available for bin packing.
        jobs: List of Job records.

    Returns:
        Tuple of (QuadraticProgram, list of variable names).

    Raises:
        RuntimeError: If qiskit-optimization is not installed.
    """
    if QuadraticProgram is None:
        raise RuntimeError("Qiskit Optimization not available (install qiskit-optimization).")

    qp = QuadraticProgram(name="cloud_binpacking")

    # Create server active variables y_s
    for s in range(servers):
        qp.binary_var(name=f"y_{s}")

    # Create assignment variables x_j_s only for feasible pairs (all pairs allowed here)
    for job in jobs:
        j = job["id"]
        for s in range(servers):
            qp.binary_var(name=f"x_{j}_{s}")

    # Constraint: each job assigned to exactly one server
    for job in jobs:
        j = job["id"]
        linear = {f"x_{j}_{s}": 1 for s in range(servers)}
        qp.linear_constraint(linear=linear, sense="==", rhs=1.0, name=f"assign_{j}")

    # Per-server resource constraints (hard): cores, ram, nvme
    # Use module-level constants defined at top of file

    for s in range(servers):
        # cores
        linear = {f"x_{job['id']}_{s}": job['cores'] for job in jobs}
        linear[f"y_{s}"] = -CORES_PER_SERVER
        qp.linear_constraint(linear=linear, sense="<=", rhs=0.0, name=f"cores_cap_{s}")

        # ram (GB)
        linear = {f"x_{job['id']}_{s}": job['ram'] for job in jobs}
        linear[f"y_{s}"] = -RAM_PER_SERVER
        qp.linear_constraint(linear=linear, sense="<=", rhs=0.0, name=f"ram_cap_{s}")

        # local NVMe (GB)
        linear = {f"x_{job['id']}_{s}": job['nvme'] for job in jobs}
        linear[f"y_{s}"] = -NVME_PER_SERVER
        qp.linear_constraint(linear=linear, sense="<=", rhs=0.0, name=f"nvme_cap_{s}")

        # network bandwidth (Gbps)
        linear = {f"x_{job['id']}_{s}": job.get('bandwidth', 0.0) for job in jobs}
        linear[f"y_{s}"] = -BANDWIDTH_PER_SERVER
        qp.linear_constraint(linear=linear, sense="<=", rhs=0.0, name=f"bandwidth_cap_{s}")

    # Objective: minimize number of active servers
    linear_obj = {f"y_{s}": 1 for s in range(servers)}
    qp.minimize(linear=linear_obj)

    var_names = [v.name for v in qp.variables]
    return qp, var_names


def build_cores_only_quadratic_program(servers: int, jobs: List[Job]) -> Tuple[Optional[QuadraticProgram], List[str]]:
    """Reduced QP for *real* QAOA: assignment + cores capacity only.

    The full build_cloud_quadratic_program() also adds RAM / NVMe / bandwidth
    capacity constraints. Converting those inequalities to a QUBO introduces many
    binary slack variables — the qubit count explodes to 40+ even for a 2-job /
    2-server instance (a 64 TiB statevector), so genuine QAOA simulation is
    impossible on a laptop. This reduced model keeps only the integer-coefficient
    cores capacity, which keeps the QUBO small enough (~10-18 qubits) to actually
    statevector-simulate on small inputs.

    Trade-off: the quantum placement does NOT enforce RAM/NVMe/bandwidth in the
    optimization itself. _decode_binary_assignment() re-checks every resource
    limit when turning a bitstring into a schedule, so a job that violates
    RAM/NVMe/bandwidth is reported as unplaced rather than silently scheduled.
    """
    if QuadraticProgram is None:
        raise RuntimeError("Qiskit Optimization not available (install qiskit-optimization).")

    qp = QuadraticProgram(name="cloud_binpacking_cores_only")

    for s in range(servers):
        qp.binary_var(name=f"y_{s}")
    for job in jobs:
        for s in range(servers):
            qp.binary_var(name=f"x_{job['id']}_{s}")

    # Each job assigned to exactly one server.
    for job in jobs:
        j = job["id"]
        qp.linear_constraint(
            linear={f"x_{j}_{s}": 1 for s in range(servers)},
            sense="==", rhs=1.0, name=f"assign_{j}",
        )

    # Cores capacity per server (integer coefficients -> QUBO-convertible).
    for s in range(servers):
        linear = {f"x_{job['id']}_{s}": job['cores'] for job in jobs}
        linear[f"y_{s}"] = -CORES_PER_SERVER
        qp.linear_constraint(linear=linear, sense="<=", rhs=0.0, name=f"cores_cap_{s}")

    qp.minimize(linear={f"y_{s}": 1 for s in range(servers)})
    return qp, [v.name for v in qp.variables]


def _decode_binary_assignment(
    bits: Dict[str, int],
    jobs: List[Job],
    servers: int,
    cores_cap: int,
    ram_cap: float,
    nvme_cap: float,
    bandwidth_cap: float,
) -> Tuple[List[Dict[str, Any]], int, List[int]]:
    """Turn a raw {var_name: 0/1} assignment into a feasible server placement.

    Shared by the QAOA decoder, the simulated-annealing solver and the MILP
    solver so all three report schedules the same way. A placement that would
    violate a resource cap (which a heuristic/quantum solver can still produce
    even after constraint penalties, since build_cores_only_quadratic_program
    only optimizes cores) is rejected and the job is reported as unplaced,
    rather than silently scheduled or silently dropped from a timeline.
    """
    server_usage: List[Dict[str, Any]] = [
        {"cores": 0, "ram": 0.0, "nvme": 0.0, "bandwidth": 0.0, "jobs": []}
        for _ in range(servers)
    ]
    unplaced: List[int] = []
    for job in jobs:
        jid = job["id"]
        placed = False
        for s in range(servers):
            if bits.get(f"x_{jid}_{s}", 0) != 1:
                continue
            su = server_usage[s]
            if _can_place_on_usage(su, job, cores_cap, ram_cap, nvme_cap, bandwidth_cap):
                _apply_job_to_usage(su, job)
                placed = True
                break
        if not placed:
            unplaced.append(jid)
    active = sum(1 for su in server_usage if su["jobs"])
    return server_usage, active, unplaced


def _classical_warm_bits(
    jobs: List[Job],
    servers: int,
    cores_cap: int,
    ram_cap: float,
    nvme_cap: float,
    bandwidth_cap: float,
) -> Dict[str, int]:
    """Best-fit-decreasing placement, expressed as {var_name: 0/1} bits.

    Used to seed both warm-start QAOA (initial state / mixer bias) and the
    simulated-annealing solver's starting point with a good known solution
    instead of a blank one. Jobs that best-fit places on a server index
    >= servers (more bins than the quantum/SA instance allows) are simply
    left at 0 for every x_{j}_{*} bit — there's no valid bit to set for them.
    """
    su_list = pack_best_fit_decreasing(
        jobs, cores_per_server=cores_cap, ram_per_server=ram_cap,
        nvme_per_server=nvme_cap, bandwidth_per_server=bandwidth_cap,
    )
    bits: Dict[str, int] = {f"y_{s}": 0 for s in range(servers)}
    for job in jobs:
        for s in range(servers):
            bits[f"x_{job['id']}_{s}"] = 0
    for s_idx, su in enumerate(su_list):
        if s_idx >= servers or not su["jobs"]:
            continue
        bits[f"y_{s_idx}"] = 1
        for jid in su["jobs"]:
            bits[f"x_{jid}_{s_idx}"] = 1
    return bits


def solve_binpack_sa(
    jobs: List[Job],
    servers: int,
    cores_per_server: Optional[int] = None,
    ram_per_server: Optional[float] = None,
    nvme_per_server: Optional[float] = None,
    bandwidth_per_server: Optional[float] = None,
    iterations: int = SA_DEFAULT_ITERATIONS,
    seed: Optional[int] = None,
    warm_start: bool = True,
) -> Dict[str, Any]:
    """Simulated annealing over the same penalty-QUBO objective QAOA targets.

    A classical, dependency-free ("quantum-inspired") heuristic: minimize
    number of active servers plus quadratic penalties for assignment and
    capacity violations, using bit-flip Metropolis annealing. Serves two
    purposes: a fast local benchmark to compare against real QAOA, and the
    automatic fallback result when QAOA is unavailable or infeasible to run
    (see solve_quantum_binpack).

    Note: cost() is recomputed in full on every flip (O(jobs * servers) per
    step) rather than updated incrementally — simple and correct, and fast
    enough at the capstone-scale instance sizes this module targets. It is
    not intended for large-scale production bin-packing.
    """
    cores_cap, ram_cap, nvme_cap, bandwidth_cap = _resolve_server_limits(
        cores_per_server=cores_per_server, ram_per_server=ram_per_server,
        nvme_per_server=nvme_per_server, bandwidth_per_server=bandwidth_per_server,
    )
    rng = random.Random(seed)

    var_names = [f"y_{s}" for s in range(servers)]
    var_names += [f"x_{job['id']}_{s}" for job in jobs for s in range(servers)]

    if warm_start:
        bits = _classical_warm_bits(jobs, servers, cores_cap, ram_cap, nvme_cap, bandwidth_cap)
    else:
        bits = {name: 0 for name in var_names}

    resource_caps = (("cores", cores_cap), ("ram", ram_cap), ("nvme", nvme_cap), ("bandwidth", bandwidth_cap))

    def cost(b: Dict[str, int]) -> float:
        total = sum(b[f"y_{s}"] for s in range(servers))
        penalty = 0.0
        for job in jobs:
            assigned = sum(b[f"x_{job['id']}_{s}"] for s in range(servers))
            penalty += (assigned - 1) ** 2
        for s in range(servers):
            for res_name, cap in resource_caps:
                used = sum(b[f"x_{job['id']}_{s}"] * job.get(res_name, 0.0) for job in jobs)
                over = used - cap * b[f"y_{s}"]
                if over > 0:
                    penalty += (over / max(cap, 1.0)) ** 2
        return total + SA_PENALTY_WEIGHT * penalty

    if not var_names:
        return {
            "server_usage": [], "objective": 0.0, "active_servers": 0, "unplaced": [],
            "status": "sa_heuristic", "warm_start_used": warm_start, "convergence": [], "qubo_cost": 0.0,
        }

    current_cost = cost(bits)
    best_bits = dict(bits)
    best_cost = current_cost
    convergence: List[float] = [best_cost]
    record_every = max(1, iterations // 300)

    t0, t_min = 2.0, 1e-3
    for it in range(iterations):
        temp = t0 * (t_min / t0) ** (it / max(1, iterations - 1))
        flip = rng.choice(var_names)
        bits[flip] ^= 1
        new_cost = cost(bits)
        delta = new_cost - current_cost
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_bits = dict(bits)
        else:
            bits[flip] ^= 1
        if it % record_every == 0:
            convergence.append(best_cost)

    server_usage, active, unplaced = _decode_binary_assignment(
        best_bits, jobs, servers, cores_cap, ram_cap, nvme_cap, bandwidth_cap
    )
    return {
        "server_usage": server_usage,
        "objective": float(sum(best_bits[f"y_{s}"] for s in range(servers))),
        "active_servers": active,
        "unplaced": unplaced,
        "status": "sa_heuristic",
        "warm_start_used": warm_start,
        "convergence": convergence,
        "qubo_cost": best_cost,
    }


def solve_binpack_milp_exact(
    jobs: List[Job],
    servers: int,
    cores_per_server: Optional[int] = None,
    ram_per_server: Optional[float] = None,
    nvme_per_server: Optional[float] = None,
    bandwidth_per_server: Optional[float] = None,
    time_limit_sec: float = MILP_TIME_LIMIT_SEC,
) -> Optional[Dict[str, Any]]:
    """Exact bin-packing optimum via mixed-integer linear programming.

    Builds the same assignment + per-resource-capacity formulation as
    build_cloud_quadratic_program(), but solves it directly with HiGHS
    (scipy.optimize.milp) instead of going through a QUBO/QAOA pipeline. This
    gives a provably optimal number-of-servers baseline to measure the
    optimality gap of the classical heuristics and the quantum/quantum-inspired
    solvers against, independent of Qiskit availability.

    Returns None if scipy is unavailable or HiGHS fails to find a feasible
    solution within time_limit_sec.
    """
    try:
        import numpy as np
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError as exc:
        logger.warning(f"[MILP] scipy unavailable: {exc}")
        return None

    cores_cap, ram_cap, nvme_cap, bandwidth_cap = _resolve_server_limits(
        cores_per_server=cores_per_server, ram_per_server=ram_per_server,
        nvme_per_server=nvme_per_server, bandwidth_per_server=bandwidth_per_server,
    )

    n_jobs = len(jobs)
    n_y = servers
    n_vars = n_y + n_jobs * servers

    if n_vars == n_y:
        # No jobs: zero servers needed, nothing to solve.
        return {"server_usage": [], "objective": 0.0, "active_servers": 0, "unplaced": [], "status": "optimal"}

    def y_idx(s: int) -> int:
        return s

    def x_idx(j: int, s: int) -> int:
        return n_y + j * servers + s

    c = np.zeros(n_vars)
    for s in range(servers):
        c[y_idx(s)] = 1.0

    constraints = []
    for j in range(n_jobs):
        row = np.zeros(n_vars)
        for s in range(servers):
            row[x_idx(j, s)] = 1.0
        constraints.append(LinearConstraint(row, lb=1.0, ub=1.0))

    resource_caps = (('cores', cores_cap), ('ram', ram_cap), ('nvme', nvme_cap), ('bandwidth', bandwidth_cap))
    for s in range(servers):
        for res_name, cap in resource_caps:
            row = np.zeros(n_vars)
            row[y_idx(s)] = -cap
            for j, job in enumerate(jobs):
                row[x_idx(j, s)] = job.get(res_name, 0.0)
            constraints.append(LinearConstraint(row, lb=-np.inf, ub=0.0))

    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars)

    try:
        result = milp(
            c, constraints=constraints, integrality=integrality, bounds=bounds,
            options={"time_limit": time_limit_sec, "disp": False},
        )
    except Exception as exc:
        logger.warning(f"[MILP] solver error: {type(exc).__name__}: {exc}")
        return None

    if not result.success or result.x is None:
        logger.warning(f"[MILP] no feasible solution found: {result.message}")
        return None

    bits: Dict[str, int] = {}
    for s in range(servers):
        bits[f"y_{s}"] = int(round(result.x[y_idx(s)]))
    for j, job in enumerate(jobs):
        for s in range(servers):
            bits[f"x_{job['id']}_{s}"] = int(round(result.x[x_idx(j, s)]))

    server_usage, active, unplaced = _decode_binary_assignment(
        bits, jobs, servers, cores_cap, ram_cap, nvme_cap, bandwidth_cap
    )
    return {
        "server_usage": server_usage,
        "objective": float(result.fun),
        "active_servers": active,
        "unplaced": unplaced,
        "status": "optimal" if result.status == 0 else f"status_{result.status}",
    }


def solve_binpack_qaoa(
    qp,
    jobs: List[Job],
    servers: int,
    reps: int = 3,
    maxiter: int = 300,
    shots: int = 16384,
    warm_start: bool = True,
    track_convergence: bool = True,
) -> Optional[Dict[str, Any]]:
    """Solve bin-packing via QAOA on a local quantum circuit simulator.

    Pipeline:
      QuadraticProgram -> QuadraticProgramToQubo -> QAOA (Sampler + SPSA)
      -> decode bitstring -> server assignments

    Uses QasmSimulator via Qiskit Sampler primitive (classical simulation of
    quantum circuits — no real QPU required). SPSA optimizer scales to large
    numbers of beta/gamma parameters without extra circuit evaluations.

    When warm_start is True, the initial state and mixer are biased toward the
    classical best-fit-decreasing solution: each qubit i gets RY(theta_i) with
    theta_i=0/pi for a bit the classical solution fixes to 0/1, or pi/2 (plain
    superposition) for slack/unknown qubits, and the mixer is the standard
    "aligned" warm-start mixer RY(-theta_i) RZ(-2*beta) RY(theta_i) per qubit
    (Egger et al., "Warm-starting quantum optimization"). This biases the
    search toward a known-good region instead of the uniform superposition.

    Args:
        qp: QuadraticProgram from build_cores_only_quadratic_program().
        jobs: List of Job records.
        servers: Number of servers in the problem.
        reps: QAOA circuit depth p. Higher = better quality, slower simulation.
        maxiter: SPSA optimizer iterations. More = better convergence.
        shots: Measurement shots per circuit. Higher = less sampling noise.
        warm_start: Bias the initial state/mixer with the classical solution.
        track_convergence: Record best-so-far objective per iteration.

    Returns:
        Dict with keys: server_usage, objective, active_servers, unplaced,
        status, warm_start_used, convergence. Returns None if required
        packages are unavailable or simulation fails.
    """
    # -- graceful imports --
    try:
        from qiskit_optimization.converters import QuadraticProgramToQubo  # type: ignore
    except ImportError:
        logger.info("[QAOA] qiskit-optimization not available. Install: pip install qiskit-optimization")
        return None
    try:
        from qiskit_algorithms import QAOA as _QAOA                        # type: ignore
        from qiskit_algorithms.optimizers import SPSA as _SPSA             # type: ignore
    except ImportError:
        logger.info("[QAOA] qiskit-algorithms not available. Install: pip install qiskit-algorithms")
        return None
    # qiskit <= 1.x shipped the V1 reference `Sampler`; it was REMOVED in qiskit 2.0.
    # Fall back to the V2 `StatevectorSampler`, which qiskit-algorithms' QAOA accepts.
    try:
        from qiskit.primitives import Sampler as _Sampler                  # type: ignore (qiskit < 2.0)
    except ImportError:
        try:
            from qiskit.primitives import StatevectorSampler as _Sampler   # type: ignore (qiskit >= 2.0)
        except ImportError:
            logger.info("[QAOA] No usable Sampler primitive in qiskit.primitives.")
            return None
    try:
        from qiskit_optimization.algorithms import MinimumEigenOptimizer as _MEO  # type: ignore
    except ImportError:
        logger.info("[QAOA] MinimumEigenOptimizer not available.")
        return None

    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    # Step 1: Convert QP -> QUBO
    logger.info("[QAOA] Converting QuadraticProgram to QUBO...")
    try:
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
    except Exception as exc:
        logger.warning(f"[QAOA] QUBO conversion failed: {type(exc).__name__}: {exc}")
        return None

    var_names_qubo = [v.name for v in qubo.variables]
    n_qubits = len(var_names_qubo)
    logger.info(f"[QAOA] QUBO size: {n_qubits} qubits (binary variables after constraint absorption)")
    logger.info(f"[QAOA] Circuit depth: p={reps} layers | Optimizer: SPSA maxiter={maxiter} | "
                f"Shots: {shots} | warm_start={warm_start}")
    logger.info("[QAOA] This is a classical simulation of a quantum circuit (no QPU needed).")

    # Step 2: Warm-start initial state + mixer (or standard uniform/X-mixer QAOA).
    cores_cap, ram_cap, nvme_cap, bandwidth_cap = _resolve_server_limits()
    init_state = None
    mixer = None
    warm_start_used = False
    if warm_start:
        try:
            warm_bits = _classical_warm_bits(jobs, servers, cores_cap, ram_cap, nvme_cap, bandwidth_cap)
            thetas = [
                (math.pi if warm_bits[name] == 1 else 0.0) if name in warm_bits else (math.pi / 2)
                for name in var_names_qubo
            ]
            init_state = QuantumCircuit(n_qubits)
            for i, theta in enumerate(thetas):
                init_state.ry(theta, i)
            beta = Parameter("beta")
            mixer = QuantumCircuit(n_qubits)
            for i, theta in enumerate(thetas):
                mixer.ry(-theta, i)
                mixer.rz(-2 * beta, i)
                mixer.ry(theta, i)
            warm_start_used = True
            logger.info("[QAOA] Warm-start initial state + mixer built from the classical best-fit solution.")
        except Exception as exc:
            logger.warning(f"[QAOA] Warm-start setup failed, falling back to standard QAOA: {exc}")
            init_state, mixer, warm_start_used = None, None, False

    # Step 3: Linear ramp initialization for beta/gamma angles.
    initial_point = np.zeros(2 * reps)
    for k in range(reps):
        initial_point[k] = (k + 1) / (reps + 1) * 0.5            # gamma_k: ramps up
        initial_point[reps + k] = (1 - (k + 1) / (reps + 1)) * 0.5  # beta_k: ramps down

    # Step 4: Convergence tracking via QAOA's per-iteration callback.
    convergence: List[float] = []
    best_seen = [float("inf")]

    def _callback(eval_count, params, value, metadata):
        best_seen[0] = min(best_seen[0], float(value))
        convergence.append(best_seen[0])

    # Step 5: Build QAOA circuit with Sampler primitive and SPSA optimizer.
    try:
        sampler = _Sampler()
        try:
            sampler.set_options(shots=shots)
        except Exception:
            pass
    except Exception:
        sampler = _Sampler()

    spsa = _SPSA(maxiter=maxiter)
    qaoa = _QAOA(
        sampler=sampler, optimizer=spsa, reps=reps, initial_point=initial_point,
        initial_state=init_state, mixer=mixer,
        callback=_callback if track_convergence else None,
    )

    # Step 6: Solve. MinimumEigenOptimizer wraps QAOA and handles the
    # Ising Hamiltonian encoding internally (Z-operator substitution for
    # each binary variable).
    logger.info("[QAOA] Running quantum circuit simulation...")
    logger.info(f"[QAOA] Expected time scales with 2^{n_qubits} internally — be patient for large problems.")
    try:
        optimizer_alg = _MEO(qaoa)
        result = optimizer_alg.solve(qubo)
    except MemoryError as mem_exc:
        logger.warning(f"[QAOA] MemoryError: problem too large for local simulation ({n_qubits} qubits).")
        _log_memory_error("QAOA solve", mem_exc, extra={"n_qubits": n_qubits, "reps": reps, "shots": shots})
        return None
    except Exception as exc:
        logger.warning(f"[QAOA] Simulation failed: {type(exc).__name__}: {exc}")
        return None

    # Step 7: Decode bitstring -> server assignments (feasibility-checked).
    assignment = {name: int(round(float(val))) for name, val in zip(var_names_qubo, result.x)}
    server_usage, active_servers, unplaced = _decode_binary_assignment(
        assignment, jobs, servers, cores_cap, ram_cap, nvme_cap, bandwidth_cap
    )

    status_name = result.status.name if hasattr(result, 'status') else 'unknown'
    logger.info(f"[QAOA] Done. Active servers: {active_servers} | Objective: {result.fval:.4f} | "
                f"Status: {status_name}")
    if unplaced:
        logger.info(f"[QAOA] {len(unplaced)} job(s) unplaced (increase reps or maxiter for better coverage): "
                    f"{unplaced}")

    return {
        'server_usage': server_usage,
        'objective': result.fval,
        'active_servers': active_servers,
        'unplaced': unplaced,
        'status': status_name,
        'warm_start_used': warm_start_used,
        'convergence': convergence,
    }


def _sa_fallback(jobs: List[Job], servers: int, reason: str) -> Dict[str, Any]:
    """Run the simulated-annealing solver and tag the result as a QAOA fallback."""
    logger.info(f"[quantum] falling back to simulated annealing: {reason}")
    result = solve_binpack_sa(jobs, servers)
    result["solver"] = "sa_fallback"
    result["fallback_reason"] = reason
    return result


def solve_quantum_binpack(
    jobs: List[Job],
    servers: int,
    reps: int = 3,
    maxiter: int = 300,
    shots: int = 16384,
    timeout_sec: Optional[float] = None,
    warm_start: bool = True,
    track_convergence: bool = True,
) -> Dict[str, Any]:
    """Solve bin-packing with real QAOA when feasible, else simulated annealing.

    Single shared entry point used by both the CLI (main()) and the web API
    (solve_binpack_qaoa_web), so they apply one fallback policy instead of two
    slightly different ones. Never returns a hard failure: builds the reduced
    cores-only QUBO (see build_cores_only_quadratic_program) and falls back to
    solve_binpack_sa over the identical QUBO objective whenever:
      - Qiskit is unavailable,
      - the instance exceeds QAOA_MAX_QUBITS qubits, or
      - the optional wall-clock timeout is hit.
    """
    try:
        qp, _ = build_cores_only_quadratic_program(servers, jobs)
    except RuntimeError as exc:
        return _sa_fallback(jobs, servers, reason=f"qiskit-optimization not available: {exc}")

    try:
        from qiskit_optimization.converters import QuadraticProgramToQubo  # type: ignore
        n_qubits = QuadraticProgramToQubo().convert(qp).get_num_vars()
    except Exception as exc:
        return _sa_fallback(jobs, servers, reason=f"could not encode the problem for QAOA: {exc}")

    if n_qubits > QAOA_MAX_QUBITS:
        return _sa_fallback(
            jobs, servers,
            reason=(f"{n_qubits} qubits (2^{n_qubits} states) exceeds the "
                    f"{QAOA_MAX_QUBITS}-qubit simulation limit"),
        )

    def _run() -> Optional[Dict[str, Any]]:
        return solve_binpack_qaoa(
            qp, jobs, servers, reps=reps, maxiter=maxiter, shots=shots,
            warm_start=warm_start, track_convergence=track_convergence,
        )

    try:
        if timeout_sec is not None:
            finished, result = _run_with_timeout(_run, timeout_sec)
            if not finished:
                return _sa_fallback(
                    jobs, servers,
                    reason=(f"real QAOA exceeded the {timeout_sec:.0f}s wall-clock limit for this "
                            f"{n_qubits}-qubit problem"),
                )
        else:
            result = _run()
    except Exception as exc:
        return _sa_fallback(jobs, servers, reason=f"QAOA solver error: {type(exc).__name__}: {exc}")

    if result is None:
        return _sa_fallback(
            jobs, servers,
            reason="QAOA solver unavailable or failed (qiskit primitives/algorithms issue)",
        )

    result["solver"] = "qaoa"
    result["fallback_reason"] = None
    return result


def main() -> None:
    print("Cloud bin-packing QP builder")
    # prompt for servers
    try:
        raw = input("Number of available servers to pack into (e.g. 4): ").strip()
        servers = int(raw)
        if servers <= 0:
            raise ValueError("servers must be positive")
    except ValueError as exc:
        print(f"Invalid number of servers: {exc}; exiting.")
        sys.exit(1)

    profile_counts = prompt_workloads()
    jobs = generate_jobs(profile_counts)
    if not jobs:
        print("No jobs generated; exiting.")
        return

    print(f"Generated {len(jobs)} jobs across profiles: {profile_counts}")
    for j in jobs:
        print(f" Job {j.id}: type={j.type} cores={j.cores} ram={j.ram}GB nvme={j.nvme}GB")

    # Build the full QP purely for informational display (variable/constraint
    # counts). The QAOA thread below uses the reduced cores-only QP instead —
    # unlike alpha5, which fed this full QP straight into QAOA and could
    # explode past 40+ qubits even for tiny job counts. Qiskit being
    # unavailable here is no longer fatal to the whole demo: classical
    # heuristics and the simulated-annealing solver don't need it.
    try:
        qp_full, var_names = build_cloud_quadratic_program(servers, jobs)
        print("Full QuadraticProgram built (reference only; QAOA uses a reduced cores-only version):")
        print(f" Variables: {len(var_names)}")
        print(f" Constraints: {len(qp_full.linear_constraints) + len(qp_full.quadratic_constraints)}")
        print("Objective: minimize number of active servers (sum y_s)")
        print("Variable sample names:")
        print(var_names[:min(60, len(var_names))])
    except RuntimeError as e:
        print(e)
        print("Full QuadraticProgram display skipped (Qiskit not available) — continuing with classical/"
              "quantum-inspired solvers, which don't need it.")

    # ---- Run classical heuristics, quantum(QAOA/SA), and exact MILP concurrently ----
    import threading as _threading
    import time as _time

    _classical_out: Dict[str, Any] = {}   # filled by classical thread
    _quantum_out = [None]                 # filled by quantum thread
    _milp_out = [None]                    # filled by MILP thread

    def _run_classical():
        _methods = {
            'first_fit_decreasing': pack_first_fit_decreasing(jobs),
            'best_fit_decreasing':  pack_best_fit_decreasing(jobs),
            'worst_fit':            pack_worst_fit(jobs),
        }
        _results = {}
        for _name, _su_list in _methods.items():
            _per_rec, _makespan = simulate_servers(_su_list, jobs)
            _servers_used = len([s for s in _su_list if s.get('jobs')])
            _results[_name] = {
                'servers': _servers_used, 'makespan': _makespan,
                'per_server': _per_rec, 'placement': _su_list,
            }
        _baseline = _results['first_fit_decreasing']
        _best_name = min(_results, key=lambda n: (_results[n]['servers'], _results[n]['makespan']))
        _best = _results[_best_name]
        _classical_out.update(results=_results, baseline=_baseline,
                              best=_best, best_name=_best_name)

    def _run_quantum():
        _reps, _maxiter, _shots = 3, 300, 16384
        print(f"\n[Quantum] Starting solver (reps={_reps}, maxiter={_maxiter}, shots={_shots}, "
              f"warm_start=True)...")
        _t0 = _time.perf_counter()
        _res = solve_quantum_binpack(
            jobs, servers, reps=_reps, maxiter=_maxiter, shots=_shots,
            timeout_sec=None, warm_start=True, track_convergence=True,
        )
        _quantum_out[0] = (_res, _time.perf_counter() - _t0)

    def _run_milp():
        n_vars = servers * (len(jobs) + 1)
        if n_vars > MILP_MAX_VARS:
            _milp_out[0] = (None, 0.0, f"skipped: {n_vars} variables exceeds MILP_MAX_VARS={MILP_MAX_VARS}")
            return
        _t0 = _time.perf_counter()
        _res = solve_binpack_milp_exact(jobs, servers, time_limit_sec=MILP_TIME_LIMIT_SEC)
        _milp_out[0] = (_res, _time.perf_counter() - _t0, None)

    _threads = [
        _threading.Thread(target=_run_classical, name="classical-solver"),
        _threading.Thread(target=_run_quantum, name="quantum-solver"),
        _threading.Thread(target=_run_milp, name="milp-solver"),
    ]
    for _t in _threads:
        _t.start()
    for _t in _threads:
        _t.join()

    # ---- Display classical results ----
    baseline  = _classical_out['baseline']
    best      = _classical_out['best']
    best_name = _classical_out['best_name']

    saved       = baseline['servers'] - best['servers']
    cost_saving = max(0, saved) * COST_PER_SERVER_PER_HOUR

    print("\nBEFORE OPTIMIZATION (Classical):")
    print(f" - Servers used: {baseline['servers']}")
    print(f" - Estimated completion time: {baseline['makespan']:.2f} seconds")
    print("\nAFTER OPTIMIZATION (Classical):")
    print(f" - Method: {best_name}")
    print(f" - Servers used: {best['servers']} (saved {saved} servers)")
    print(f" - Estimated completion time: {best['makespan']:.2f} seconds")
    if baseline['makespan'] > 0:
        pct = (baseline['makespan'] - best['makespan']) / baseline['makespan'] * 100.0
        print(f" - Improvement: {pct:.1f}% faster")
    print(f" - Cost savings: €{cost_saving:.2f}/hour")
    print("\nDetailed timeline per server (best classical solution):")
    for idx, rec in enumerate(best['per_server']):
        job_ids = [j['id'] for j in rec['jobs']]
        print(f" - Server {idx}: Jobs {job_ids} -> completes at t={rec['completion']:.2f}s")
    print(f"\nTotal makespan (Classical): {best['makespan']:.2f} seconds")

    # ---- Display quantum / quantum-inspired results ----
    quantum_result, quantum_wall = _quantum_out[0] if _quantum_out[0] else (None, 0.0)

    if quantum_result is not None:
        q_su = quantum_result['server_usage']
        q_active = quantum_result['active_servers']
        q_per_rec, q_makespan = simulate_servers(
            [su for su in q_su if su['jobs']], jobs, verbose=False
        )
        solver_label = quantum_result.get('solver', 'qaoa').upper()
        print(f"\n{solver_label} SOLUTION:")
        if quantum_result.get('fallback_reason'):
            print(f" - Fallback reason  : {quantum_result['fallback_reason']}")
        print(f" - Active servers   : {q_active}")
        print(f" - Objective        : {quantum_result['objective']:.4f}")
        print(f" - Status           : {quantum_result['status']}")
        print(f" - Warm start       : {quantum_result.get('warm_start_used', False)}")
        conv = quantum_result.get('convergence') or []
        if conv:
            print(f" - Convergence      : {conv[0]:.4f} -> {conv[-1]:.4f} over {len(conv)} evaluations")
        print(f" - Makespan         : {q_makespan:.2f} seconds")
        print(f" - Wall-clock time  : {quantum_wall:.2f} seconds")
        if quantum_result.get('unplaced'):
            print(f" - Unplaced jobs    : {quantum_result['unplaced']} (increase reps/maxiter for QAOA)")
        print(f"\nDetailed timeline ({solver_label} solution):")
        for idx, rec in enumerate(q_per_rec):
            job_ids = [j['id'] for j in rec['jobs']]
            print(f" - Server {idx}: Jobs {job_ids} -> completes at t={rec['completion']:.2f}s")

        saved_vs_classical = best['servers'] - q_active
        cost_delta = saved_vs_classical * COST_PER_SERVER_PER_HOUR
        print(f"\n{solver_label} vs Classical best ({best_name}):")
        print(f" - Servers saved    : {saved_vs_classical}")
        print(f" - Cost delta       : €{cost_delta:.2f}/hour")
        if best['makespan'] > 0 and q_makespan > 0:
            speed_pct = (best['makespan'] - q_makespan) / best['makespan'] * 100.0
            print(f" - Makespan delta   : {speed_pct:.1f}% {'faster' if speed_pct >= 0 else 'slower'}")
    else:
        print("\nQUANTUM SOLUTION:")
        print(" - FAILED: solver was unavailable or failed to produce a solution.")

    # ---- Display exact MILP baseline ----
    milp_res, milp_wall, milp_skip_reason = _milp_out[0] if _milp_out[0] else (None, 0.0, "not run")
    print("\nEXACT MILP BASELINE:")
    if milp_skip_reason:
        print(f" - Skipped: {milp_skip_reason}")
    elif milp_res is None:
        print(" - FAILED: MILP solver did not return a feasible solution within the time limit.")
    else:
        m_per_rec, m_makespan = simulate_servers(
            [su for su in milp_res['server_usage'] if su['jobs']], jobs, verbose=False
        )
        print(f" - Active servers   : {milp_res['active_servers']} (provably optimal)")
        print(f" - Makespan         : {m_makespan:.2f} seconds")
        print(f" - Wall-clock time  : {milp_wall:.2f} seconds")
        gap_classical = best['servers'] - milp_res['active_servers']
        print(f" - Classical gap    : best classical uses {gap_classical} more server(s) than optimal"
              if gap_classical > 0 else " - Classical gap    : classical best matched the optimum")
        if quantum_result is not None:
            gap_quantum = quantum_result['active_servers'] - milp_res['active_servers']
            label = quantum_result.get('solver', 'quantum')
            print(f" - {label} gap".ljust(20) + f": uses {gap_quantum} more server(s) than optimal"
                  if gap_quantum > 0 else f" - {label} gap".ljust(20) + ": matched the optimum")



def generate_poisson_arrivals(rate_lambda: float, time_horizon: float):
    """
    Generate arrival times following a Poisson process (exponential inter-arrivals).
    rate_lambda: average arrivals per time unit
    time_horizon: generate arrivals up to this time
    """
    arrivals = []
    t = 0.0
    while t < time_horizon:
        inter = random.expovariate(rate_lambda) if rate_lambda > 0.0 else float('inf')
        t += inter
        if t < time_horizon:
            arrivals.append(t)
    return arrivals


def generate_jobs_with_arrivals(profile_counts: Dict[str, int], time_horizon: float = 3600.0) -> List[Job]:
    """Generate jobs with realistic arrival times over a time horizon.

    Uses a small built-in profile set for timing-enabled workloads.
    """
    JOB_PROFILES_WITH_TIMING = {
        "web_api": {
            "cores": 1, "ram": 1.0, "nvme": 1.0, "bandwidth": 0.5,
            "arrival_pattern": "poisson",
            "arrival_rate": 5.0 / 60.0,  # 5 per minute -> per second
            "duration_mean": 0.1,  # seconds
            "deadline_slack": 0.5
        },
        "batch": {
            "cores": 2, "ram": 2.0, "nvme": 5.0, "bandwidth": 0.1,
            "arrival_pattern": "scheduled",
            "arrival_times": [0.0, 3600.0, 7200.0],
            "duration_mean": 300.0,
            "deadline_slack": 1800.0
        },
    }

    jobs: List[Job] = []
    jid = 0
    for ptype, count in profile_counts.items():
        profile = JOB_PROFILES_WITH_TIMING.get(ptype)
        if profile is None:
            continue

        arrival_times = []
        if profile["arrival_pattern"] == "poisson":
            arrival_times = generate_poisson_arrivals(profile["arrival_rate"], time_horizon)[:count]
        elif profile["arrival_pattern"] == "scheduled":
            arrival_times = profile.get("arrival_times", [])[:count]
        else:
            arrival_times = [0.0] * count

        for ta in arrival_times:
            duration = random.expovariate(1.0 / max(1e-6, profile["duration_mean"]))
            jobs.append(Job(
                id=jid,
                type=ptype,
                cores=profile["cores"],
                ram=profile["ram"],
                nvme=profile["nvme"],
                arrival_time=ta,
                duration=duration,
                deadline=ta + profile["deadline_slack"],
                bandwidth=profile.get("bandwidth", 0.0),
            ))
            jid += 1

    return sorted(jobs, key=lambda j: j.arrival_time)


# Helper functions for online_scheduler_with_arrivals (extracted to module level)
def format_stress_line(name: str, used: float, cap: float, unit: str = '') -> str:
    """Format a resource utilization line with stress indicator.

    Args:
        name: Resource name (e.g. 'Cores', 'RAM').
        used: Current usage amount.
        cap: Capacity limit.
        unit: Unit string (e.g. 'GB', 'Gbps').

    Returns:
        Formatted string with percentage, status, and symbol.
    """
    pct = (used / cap) * 100.0 if cap > 0 else 0.0
    # choose symbol and descriptor per requested thresholds
    if pct > 100.0:
        sym = '❌'
        desc = 'OVERLOADED'
    elif abs(pct - 100.0) <= FLOAT_EPSILON:
        sym = '⚠⚠'
        desc = 'CRITICAL!'
    elif pct >= HIGH_STRESS_THRESHOLD:
        sym = '⚠⚠'
        desc = 'HIGH STRESS'
    elif pct >= MEDIUM_STRESS_THRESHOLD:
        sym = '⚠'
        desc = 'MEDIUM STRESS'
    else:
        sym = '✓'
        desc = 'OK'

    # friendly number formatting: integer-like values shown as ints, else one decimal for readability
    def fmt_num(v):
        if abs(v - round(v)) < FLOAT_EPSILON:
            return str(int(round(v)))
        return f"{v:.1f}"

    used_str = fmt_num(used)
    cap_str = fmt_num(cap)
    unit_sp = f"{unit}" if unit else ''
    return f"  {name}: {used_str}/{cap_str}{unit_sp} ({pct:.1f}% - {desc}) {sym}"


def format_server_snapshot(
    server_idx: int,
    server_st: Dict[str, Any],
    current_time: float,
    cores_per_server: Optional[int] = None,
    ram_per_server: Optional[float] = None,
    nvme_per_server: Optional[float] = None,
    bandwidth_per_server: Optional[float] = None,
) -> str:
    """Format a snapshot of server resource utilization.

    Args:
        server_idx: Server index (0-based).
        server_st: Server state dict with resource usage.
        current_time: Current simulation time.

    Returns:
        Multi-line string with resource usage and overall status.
    """
    cores_cap, ram_cap, nvme_cap, bandwidth_cap = _resolve_server_limits(
        cores_per_server=cores_per_server,
        ram_per_server=ram_per_server,
        nvme_per_server=nvme_per_server,
        bandwidth_per_server=bandwidth_per_server,
    )
    lines = [f"t={current_time:.2f}s: Server {server_idx} resources:"]
    lines.append(format_stress_line('Cores', server_st['cores'], cores_cap, ''))
    lines.append(format_stress_line('RAM', server_st['ram'], ram_cap, 'GB'))
    lines.append(format_stress_line('NVMe', server_st['nvme'], nvme_cap, 'GB'))
    lines.append(format_stress_line('Bandwidth', server_st['bandwidth'], bandwidth_cap, 'Gbps'))

    # determine overall status and concise reasons
    status = 'OK'
    reasons = []
    for name, used, cap in (('Cores', server_st['cores'], cores_cap),
                            ('RAM', server_st['ram'], ram_cap),
                            ('NVMe', server_st['nvme'], nvme_cap),
                            ('Bandwidth', server_st['bandwidth'], bandwidth_cap)):
        pct = (used / cap) * 100.0 if cap > 0 else 0.0
        if pct > 100.0:
            # worst-case: overloaded
            status = 'OVERLOADED'
            reasons.append(f"{name} exceeded")
        elif abs(pct - 100.0) <= FLOAT_EPSILON:
            if status not in ('OVERLOADED', 'CRITICAL'):
                status = 'CRITICAL'
            reasons.append(f"{name} maxed out")
        elif pct >= HIGH_STRESS_THRESHOLD:
            if status not in ('OVERLOADED', 'CRITICAL'):
                status = 'HIGH STRESS'
            reasons.append(f"{name} high")
        elif pct >= MEDIUM_STRESS_THRESHOLD:
            if status not in ('OVERLOADED', 'CRITICAL', 'HIGH STRESS'):
                status = 'MEDIUM STRESS'
            reasons.append(f"{name} medium")

    if reasons:
        lines.append(f"  Status: {status} ({'; '.join(reasons)})")
    else:
        lines.append(f"  Status: {status}")

    return "\n".join(lines)


def online_scheduler_with_arrivals(jobs: List[Job], servers: int, verbose: bool = True,
                                   cores_per_server: Optional[int] = None,
                                   ram_per_server: Optional[float] = None,
                                   nvme_per_server: Optional[float] = None,
                                   bandwidth_per_server: Optional[float] = None) -> Tuple[Dict[int, Dict], List[str], Dict[str, float]]:
    """Event-driven, time-based online scheduler with FIFO waiting queue.

    Behavior:
    - Jobs that arrive are appended to a FIFO waiting queue if no server can fit them immediately.
    - When jobs complete, resources are freed and the queue is scanned (FIFO) to assign any jobs that now fit.
    - The function prints a timeline of arrivals, assignments, completions and resource snapshots,
      and returns (assignments, timeline, stats) where stats includes makespan, peak_utilization,
      jobs_completed and average_wait_time.
    """
    # Use module-level heapq and mean and constants; use provided overrides if given
    _cores = cores_per_server if cores_per_server is not None else CORES_PER_SERVER
    _ram = ram_per_server if ram_per_server is not None else RAM_PER_SERVER
    _nvme = nvme_per_server if nvme_per_server is not None else NVME_PER_SERVER
    _bandwidth = bandwidth_per_server if bandwidth_per_server is not None else BANDWIDTH_PER_SERVER

    # Closure to check if job fits on server state using local constants
    def can_fit_on_server_state(st: Dict[str, Any], job: Job) -> bool:
        """Check if a job can fit on a server state without exceeding resource limits."""
        return (
            st['cores'] + job.get('cores', 0) <= _cores and
            st['ram'] + job.get('ram', 0.0) <= _ram and
            st['nvme'] + job.get('nvme', 0.0) <= _nvme and
            st['bandwidth'] + job.get('bandwidth', 0.0) <= _bandwidth
        )

    # Initialize server state
    server_state = [
        {
            'running': [],  # list of {'job': job, 'start': float, 'end': float}
            'cores': 0, 'ram': 0.0, 'nvme': 0.0, 'bandwidth': 0.0,
            'peak_cores_pct': 0.0,
        }
        for _ in range(servers)
    ]

    events = []  # min-heap of (time, type, job_id)
    for job in jobs:
        heapq.heappush(events, (job.get('arrival_time', 0.0), 'ARRIVAL', job['id']))

    # Fast lookup by id to avoid linear scans
    job_map = {j['id']: j for j in jobs}

    waiting_queue: List[Job] = []  # FIFO list of jobs
    assignments: Dict[int, Dict] = {}  # job_id -> {server, start, end, arrival}
    timeline: List[str] = []
    current_time = 0.0

    # Use module-level helper functions defined above
    # These closure wrappers adapt module-level functions to local context

    def snapshot_server(sidx: int) -> str:
        """Snapshot a server's resource state at current time using local constants."""
        return format_server_snapshot(
            sidx,
            server_state[sidx],
            current_time,
            cores_per_server=_cores,
            ram_per_server=_ram,
            nvme_per_server=_nvme,
            bandwidth_per_server=_bandwidth,
        )

    def assign_job_to_server(job: Job, sidx: int, start_time: float) -> None:
        """Assign a job to a server and record the assignment."""
        end_time = start_time + job.get('duration', 0.0)
        st = server_state[sidx]
        st['running'].append({'job': job, 'start': start_time, 'end': end_time})
        st['cores'] += job.get('cores', 0)
        st['ram'] += job.get('ram', 0.0)
        st['nvme'] += job.get('nvme', 0.0)
        st['bandwidth'] += job.get('bandwidth', 0.0)
        # update peak core utilization using local _cores constant
        cores_pct = (st['cores'] / _cores) * 100.0 if _cores > 0 else 0.0
        st['peak_cores_pct'] = max(st['peak_cores_pct'], cores_pct)
        assignments[job['id']] = {'server': sidx, 'start': start_time, 'end': end_time, 'arrival': job.get('arrival_time', 0.0)}
        heapq.heappush(events, (end_time, 'COMPLETION', job['id']))
        if verbose:
            timeline.append(f"t={start_time:.2f}s: Job {job['id']} assigned to Server {sidx} (duration={job.get('duration',0.0):.2f}s, ends at t={end_time:.2f}s)")
            snap = snapshot_server(sidx)
            for line in snap.split('\n'):
                timeline.append(line)

    def try_assign_from_queue(now: float) -> bool:
        """Attempt to assign jobs from waiting queue using Earliest-Deadline-First order."""
        waiting_queue.sort(key=lambda j: j.get('deadline', float('inf')) if j.get('deadline', None) is not None else float('inf'))
        i = 0
        assigned_any = False
        while i < len(waiting_queue):
            job = waiting_queue[i]
            deadline = job.get('deadline', None)
            deadline = deadline if deadline is not None else float('inf')
            if deadline < now:
                if verbose:
                    timeline.append(
                        f"t={now:.2f}s: Job {job['id']} MISSED DEADLINE "
                        f"(deadline={deadline:.2f})"
                    )
                waiting_queue.pop(i)
                continue
            placed = False
            for s in range(servers):
                if can_fit_on_server_state(server_state[s], job):
                    assign_job_to_server(job, s, now)
                    waiting_queue.pop(i)
                    placed = True
                    assigned_any = True
                    break
            if not placed:
                i += 1
        return assigned_any

    # Main event loop
    while events:
        time, etype, jid = heapq.heappop(events)
        current_time = time
        job = job_map.get(jid)
        # Clean up finished jobs from running lists (safety: remove any with end <= now)
        for s in range(servers):
            st = server_state[s]
            finished = [r for r in st['running'] if r['end'] <= current_time]
            for r in finished:
                # free resources (clamp to 0 to guard against floating-point drift)
                st['cores']     = max(0,   st['cores']     - r['job'].get('cores', 0))
                st['ram']       = max(0.0, st['ram']       - r['job'].get('ram', 0.0))
                st['nvme']      = max(0.0, st['nvme']      - r['job'].get('nvme', 0.0))
                st['bandwidth'] = max(0.0, st['bandwidth'] - r['job'].get('bandwidth', 0.0))
            st['running'] = [r for r in st['running'] if r['end'] > current_time]

        if etype == 'ARRIVAL':
            if verbose:
                timeline.append(f"t={current_time:.2f}s: Job {jid} arrived (duration={job.get('duration',0.0):.2f}s)")
            waiting_queue.append(job)
            # try to assign FIFO
            try_assign_from_queue(current_time)
            if waiting_queue and verbose:
                timeline.append(f"t={current_time:.2f}s: Waiting queue: {[j['id'] for j in waiting_queue]}")

        elif etype == 'COMPLETION':
            # Job may not have been assigned (defensive)
            rec = assignments.get(jid)
            if rec:
                sidx = rec['server']
                if verbose:
                    timeline.append(f"t={current_time:.2f}s: Job {jid} completed on Server {sidx}")
            else:
                if verbose:
                    timeline.append(f"t={current_time:.2f}s: Job {jid} completion event (no assignment recorded)")
            # After freeing resources, attempt to assign from queue
            try_assign_from_queue(current_time)
            if waiting_queue and verbose:
                timeline.append(f"t={current_time:.2f}s: Waiting queue: {[j['id'] for j in waiting_queue]}")

    # After all events processed, build statistics
    if assignments:
        makespan = max(rec['end'] for rec in assignments.values()) - min(rec['arrival'] for rec in assignments.values())
        jobs_completed = len(assignments)
        waits = [assignments[j]['start'] - assignments[j]['arrival'] for j in assignments]
        avg_wait = mean(waits) if waits else 0.0
    else:
        makespan = 0.0
        jobs_completed = 0
        avg_wait = 0.0

    # peak utilization across servers (cores percent)
    peak_util = max((st['peak_cores_pct'] for st in server_state), default=0.0)

    stats = {
        'makespan': makespan,
        'peak_utilization_pct': peak_util,
        'jobs_completed': jobs_completed,
        'average_wait_time': avg_wait,
    }

    # Print timeline (already collected) and final stats summary
    for line in timeline:
        print(line)

    if verbose:
        print("\nFINAL SUMMARY:")
        print(f" - Total completion time (makespan): {stats['makespan']:.2f} seconds")
        print(f" - Peak server utilization (cores): {stats['peak_utilization_pct']:.1f}%")
        print(f" - Jobs completed: {stats['jobs_completed']}")
        print(f" - Average wait time: {stats['average_wait_time']:.2f} seconds")

    # Also return a compact record for programmatic use
    return assignments, timeline, stats


try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    FastAPI = None
    CORSMiddleware = None
    BaseModel = None
    Field = None
    field_validator = None

# ---------------------------------------------------------------------------
# Quantum / quantum-inspired solver, web-facing wrapper
# ---------------------------------------------------------------------------
def _run_with_timeout(fn: Callable[[], Any], timeout_sec: float) -> Tuple[bool, Any]:
    """Run fn() in a daemon thread with a wall-clock cap.

    Returns (finished, result). On timeout returns (False, None); the worker
    thread is abandoned (Python can't kill a thread) and finishes in the
    background — kept cheap by using minimal QAOA iterations at the call site.
    """
    import threading
    box: Dict[str, Any] = {}

    def _target():
        try:
            box["result"] = fn()
        except Exception as exc:  # surfaced to caller below
            box["error"] = exc

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout_sec)
    if th.is_alive():
        return False, None
    if "error" in box:
        raise box["error"]
    return True, box.get("result")


def solve_binpack_qaoa_web(jobs: List[Job], num_servers: int) -> Dict[str, Any]:
    """Web-facing quantum/quantum-inspired solve, bounded to a short wall-clock budget.

    Thin wrapper around solve_quantum_binpack with web-appropriate parameters
    (minimal reps/maxiter/shots so a real QAOA attempt stays within
    QAOA_WEB_TIMEOUT_SEC, falling back to simulated annealing otherwise) that
    reshapes the result into the {assignments, active_servers, solver,
    timeline, stats} contract the FastAPI routes and companion frontend
    expect. Unlike alpha5's solve_binpack_qaoa_web, this never returns a hard
    "failed" result — the SA fallback always produces a usable schedule, so
    every request gets a real answer to compare against the classical column.
    """
    result = solve_quantum_binpack(
        jobs, num_servers, reps=1, maxiter=1, shots=512,
        timeout_sec=QAOA_WEB_TIMEOUT_SEC, warm_start=True, track_convergence=True,
    )

    server_usage = result["server_usage"]

    # Unified makespan: concurrent per-server simulation (the same model the
    # classical column uses), mirroring the CLI's display step.
    _per, makespan = simulate_servers(
        [su for su in server_usage if su["jobs"]], jobs, verbose=False
    )

    assignments: Dict[int, int] = {}
    for s, su in enumerate(server_usage):
        for jid in su["jobs"]:
            assignments[jid] = s

    active = result["active_servers"]
    peak_util = max(
        (min(su["cores"] / max(1, CORES_PER_SERVER) * 100.0, 100.0)
         for su in server_usage if su["jobs"]),
        default=0.0,
    )

    solver_label = result["solver"]
    timeline = [f"{solver_label}: {len(jobs)} job(s) packed onto {active} active server(s)"]
    if result.get("fallback_reason"):
        timeline.append(f"{solver_label}: fell back from real QAOA because {result['fallback_reason']}")
    if result.get("unplaced"):
        timeline.append(f"{solver_label}: {len(result['unplaced'])} job(s) unplaced: {result['unplaced']}")

    return {
        "assignments": {str(k): v for k, v in assignments.items()},
        "active_servers": active,
        "solver": solver_label,
        "fallback_reason": result.get("fallback_reason"),
        "timeline": timeline,
        "stats": {
            "makespan": makespan,
            "peak_utilization_pct": peak_util,
            "jobs_completed": len(jobs),
            "average_wait_time": 0.0,
            "active_servers": active,
        },
    }


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
if FastAPI is not None and CORSMiddleware is not None and BaseModel is not None and Field is not None:
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ServerConstants(BaseModel):
        cores: int = CORES_PER_SERVER
        ram: float = RAM_PER_SERVER
        nvme: float = NVME_PER_SERVER
        bandwidth: float = BANDWIDTH_PER_SERVER


    class WorkloadRequest(BaseModel):
        profile_counts: Dict[str, int]
        servers: int = 3
        server_constants: ServerConstants = Field(default_factory=ServerConstants)

        @field_validator("servers")
        @classmethod
        def _servers_positive(cls, v: int) -> int:
            if v <= 0:
                raise ValueError("servers must be a positive integer")
            return v

        @field_validator("profile_counts")
        @classmethod
        def _profile_counts_valid(cls, v: Dict[str, int]) -> Dict[str, int]:
            for ptype, count in v.items():
                if ptype not in JOB_PROFILES:
                    raise ValueError(f"unknown profile type '{ptype}'; valid types: {list(JOB_PROFILES.keys())}")
                if count < 0:
                    raise ValueError(f"count for profile '{ptype}' must be non-negative, got {count}")
            return v


    @app.post("/schedule")
    def schedule(req: WorkloadRequest):
        """Classical greedy online scheduler."""
        jobs = generate_jobs(req.profile_counts)
        assignments, timeline, stats = online_scheduler_with_arrivals(
            jobs, servers=req.servers, verbose=False,
            cores_per_server=req.server_constants.cores,
            ram_per_server=req.server_constants.ram,
            nvme_per_server=req.server_constants.nvme,
            bandwidth_per_server=req.server_constants.bandwidth
        )
        stats["active_servers"] = len(set(v["server"] for v in assignments.values())) if assignments else 0
        # Unify the makespan metric with the quantum column: evaluate the
        # classical placement with the same concurrent simulate_servers model.
        server_jobs: Dict[int, List[int]] = {}
        for jid, rec in assignments.items():
            server_jobs.setdefault(rec["server"], []).append(jid)
        if server_jobs:
            _per, makespan = simulate_servers(
                [{"jobs": v} for v in server_jobs.values()], jobs, verbose=False
            )
            stats["makespan"] = makespan
        return {"stats": stats, "timeline": timeline, "solver": "classical_greedy"}


    @app.post("/schedule_quantum")
    def schedule_quantum(req: WorkloadRequest):
        """Quantum/quantum-inspired bin-packing scheduler.

        Attempts real QAOA (warm-started, statevector-simulated); falls back to
        simulated annealing over the same QUBO objective if Qiskit is
        unavailable, the instance is too large, or the attempt times out. See
        solve_binpack_qaoa_web / solve_quantum_binpack.
        """
        jobs = generate_jobs(req.profile_counts)
        return solve_binpack_qaoa_web(jobs, num_servers=req.servers)


    @app.post("/schedule_compare")
    def schedule_compare(req: WorkloadRequest):
        """Run both classical and quantum/quantum-inspired solvers for comparison."""
        jobs = generate_jobs(req.profile_counts)

        classical_assignments, classical_timeline, classical_stats = online_scheduler_with_arrivals(
            jobs, servers=req.servers, verbose=False,
            cores_per_server=req.server_constants.cores,
            ram_per_server=req.server_constants.ram,
            nvme_per_server=req.server_constants.nvme,
            bandwidth_per_server=req.server_constants.bandwidth
        )
        classical_stats["active_servers"] = len(set(v["server"] for v in classical_assignments.values())) if classical_assignments else 0

        quantum_result = solve_binpack_qaoa_web(jobs, num_servers=req.servers)

        return {
            "classical": {
                "stats": classical_stats,
                "timeline": classical_timeline,
                "solver": "classical_greedy",
            },
            "quantum": quantum_result,
        }


    @app.post("/schedule_milp")
    def schedule_milp(req: WorkloadRequest):
        """Exact optimal server count via MILP (HiGHS) — independent of Qiskit.

        Gives a provably-optimal baseline to measure the classical/quantum
        columns' optimality gap against. Returns a structured error (not a
        500) if the instance is too large for the configured time budget.
        """
        jobs = generate_jobs(req.profile_counts)
        n_vars = req.servers * (len(jobs) + 1)
        if n_vars > MILP_MAX_VARS:
            return {
                "solver": "milp_exact",
                "error": f"instance too large for the exact solver within budget "
                         f"({n_vars} variables > {MILP_MAX_VARS})",
            }
        result = solve_binpack_milp_exact(
            jobs, req.servers, time_limit_sec=MILP_TIME_LIMIT_SEC,
            cores_per_server=req.server_constants.cores, ram_per_server=req.server_constants.ram,
            nvme_per_server=req.server_constants.nvme, bandwidth_per_server=req.server_constants.bandwidth,
        )
        if result is None:
            return {"solver": "milp_exact", "error": "no feasible solution found within the time limit"}
        _per, makespan = simulate_servers(
            [su for su in result["server_usage"] if su["jobs"]], jobs, verbose=False
        )
        return {
            "solver": "milp_exact",
            "active_servers": result["active_servers"],
            "stats": {"makespan": makespan, "active_servers": result["active_servers"]},
            "unplaced": result["unplaced"],
        }


    # ---- Pollable async QAOA endpoint --------------------------------------
    # Real QAOA can take up to QAOA_WEB_TIMEOUT_SEC; holding an HTTP request
    # open that long is wasteful for a browser client. These two routes let a
    # caller submit and poll instead of blocking, alongside the still-available
    # synchronous /schedule_quantum for simpler callers.
    import threading as _threading
    import uuid as _uuid

    _quantum_jobs: Dict[str, Dict[str, Any]] = {}
    _quantum_jobs_lock = _threading.Lock()

    @app.post("/schedule_quantum_async")
    def schedule_quantum_async(req: WorkloadRequest):
        """Submit a quantum/quantum-inspired run; poll it with /schedule_quantum_status/{job_id}."""
        jobs = generate_jobs(req.profile_counts)
        job_id = str(_uuid.uuid4())
        with _quantum_jobs_lock:
            _quantum_jobs[job_id] = {"status": "running", "result": None}

        def _worker():
            try:
                res = solve_binpack_qaoa_web(jobs, num_servers=req.servers)
                with _quantum_jobs_lock:
                    _quantum_jobs[job_id] = {"status": "done", "result": res}
            except Exception as exc:
                logger.warning(f"[async QAOA] job {job_id} failed: {exc}")
                with _quantum_jobs_lock:
                    _quantum_jobs[job_id] = {"status": "error", "result": {"error": str(exc)}}

        _threading.Thread(target=_worker, daemon=True).start()
        return {"job_id": job_id, "status": "running"}


    @app.get("/schedule_quantum_status/{job_id}")
    def schedule_quantum_status(job_id: str):
        """Poll the status/result of a job submitted via /schedule_quantum_async."""
        with _quantum_jobs_lock:
            entry = _quantum_jobs.get(job_id)
        if entry is None:
            return {"status": "not_found"}
        return entry
else:
    app = None


if __name__ == '__main__':
    import argparse

    # Best-effort: make the Windows console emit UTF-8 so the stress symbols
    # (checkmarks / warnings) print correctly instead of crashing on cp1252.
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo-online', action='store_true', help='Run non-interactive online-arrival demo')
    parser.add_argument('--seed', type=int, default=None,
                         help='Random seed for reproducible arrivals/annealing demos')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.demo_online:
        demo_profiles = {'web_api': 20, 'batch': 3}
        demo_jobs = generate_jobs_with_arrivals(demo_profiles, time_horizon=600.0)
        print(f"Running online scheduler demo with {len(demo_jobs)} jobs and 3 servers")
        assignments, timeline, stats = online_scheduler_with_arrivals(demo_jobs, servers=3)
        # Compact returned stats (detailed timeline and summary already printed by the scheduler)
        print('\nCompact stats:')
        print(f" makespan={stats['makespan']:.2f}s, peak_util={stats['peak_utilization_pct']:.1f}%, "
              f"jobs_completed={stats['jobs_completed']}, avg_wait={stats['average_wait_time']:.2f}s")
    else:
        # Default: run interactive prompt
        main()

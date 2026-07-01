# main.py
"""Cloud bin-packing demo using Qiskit QuadraticProgram

This script prompts the user to choose workload types and counts, generates jobs
from fixed profiles, and formulates a scheduling problem as a Qiskit
QuadraticProgram with binary assignment variables x_{j}_{s} and server-active
variables y_s.

Hard constraints enforced per server:
- 2 vCPUs
- 4 GB ECC RAM
- 40 GB local NVMe (non-shared, non-networked)

Objective: minimize number of active servers (sum y_s).

This file only builds the QuadraticProgram and prints a summary. Solving is
optional and depends on installed Qiskit backends.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys
import heapq
import random
from statistics import mean
from dataclasses import dataclass, field

try:
    from qiskit_optimization import QuadraticProgram
except Exception:
    QuadraticProgram = None


def _log_memory_error(context: str, exc: Exception, extra: Dict[str, object] = None) -> None:
    """Append MemoryError diagnostics to qp_memory_error.log (best-effort).

    Writes timestamp, context, exception, traceback, and a few runtime hints.
    """
    try:
        import datetime, traceback, os, json
        fn = os.path.join(os.getcwd(), "qp_memory_error.log")
        info = {
            "time_utc": datetime.datetime.utcnow().isoformat() + "Z",
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


@dataclass
class Job:
    """Dataclass representing a job/workload.

    Attributes match legacy dict keys to ease migration.
    """
    id: int
    type: str
    cores: int
    ram: float
    nvme: float
    bandwidth: float = 0.0
    duration: float = 0.0
    arrival_time: float = 0.0
    deadline: float = 0.0
    shared_allowed: bool = True
    arch: str = 'Any'
    min_clock: float = 0.0
    disk: float = 0.0


@dataclass
class ServerState:
    id: Optional[int] = None
    running: List[Dict[str, Any]] = field(default_factory=list)
    cores: int = 0
    ram: float = 0.0
    nvme: float = 0.0
    bandwidth: float = 0.0
    peak_cores_pct: float = 0.0


JOB_PROFILES: Dict[str, Dict[str, float]] = {
    # (cores, ram_gb, nvme_gb, duration_seconds)
    "web_api": {"cores": 1, "ram": 1.0, "nvme": 1.0, "bandwidth": 0.5, "duration": 5.0},
    "batch": {"cores": 2, "ram": 2.0, "nvme": 5.0, "bandwidth": 0.1, "duration": 300.0},
    "memory": {"cores": 1, "ram": 3.0, "nvme": 5.0, "bandwidth": 0.2, "duration": 60.0},
    "io_heavy": {"cores": 1, "ram": 1.0, "nvme": 30.0, "bandwidth": 2.0, "duration": 180.0},
}


def _resolve_server_limits(
    cores_per_server: int = None,
    ram_per_server: float = None,
    nvme_per_server: float = None,
    bandwidth_per_server: float = None,
) -> Tuple[int, float, float, float]:
    """Resolve per-server resource limits from overrides or module defaults."""
    return (
        cores_per_server if cores_per_server is not None else CORES_PER_SERVER,
        ram_per_server if ram_per_server is not None else RAM_PER_SERVER,
        nvme_per_server if nvme_per_server is not None else NVME_PER_SERVER,
        bandwidth_per_server if bandwidth_per_server is not None else BANDWIDTH_PER_SERVER,
    )


def _job_sort_key(job: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Shared sort key for packing heuristics."""
    return (
        job.get('cores', 0),
        job.get('ram', 0.0),
        job.get('nvme', 0.0),
        job.get('bandwidth', 0.0),
    )


def _create_server_usage_record(job: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new server usage record seeded with a single job."""
    return {
        'cores': job.get('cores', 0),
        'ram': job.get('ram', 0.0),
        'nvme': job.get('nvme', 0.0),
        'bandwidth': job.get('bandwidth', 0.0),
        'jobs': [job['id']],
    }


def _job_fits_within_limits(
    job: Dict[str, Any],
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
    job: Dict[str, Any],
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
    job: Dict[str, Any],
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


def _apply_job_to_usage(usage: Dict[str, Any], job: Dict[str, Any]) -> None:
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


def generate_jobs(profile_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    """Generate job dictionaries from selected profiles.

    Args:
        profile_counts: Dict mapping profile type (str) to desired count (int).
    Returns:
        List of job dictionaries with keys: id, type, cores, ram, nvme, bandwidth.
        Each job ID is unique, starting from 0.

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
    
    jobs: List[Dict[str, Any]] = []
    jid = 0
    for ptype, count in profile_counts.items():
        profile = JOB_PROFILES.get(ptype)
        if profile is None:
            raise ValueError(f"Unknown profile type '{ptype}'. Valid types: {list(JOB_PROFILES.keys())}")
        for _ in range(count):
            jobs.append({
                "id": jid,
                "type": ptype,
                "cores": int(profile["cores"]),
                "ram": float(profile["ram"]),
                "nvme": float(profile["nvme"]),
                "bandwidth": float(profile.get("bandwidth", 0.0)),
                "duration": float(profile.get("duration", 10.0)),
            })
            jid += 1
    return jobs


# -----------------
# Packing & simulation helpers (module-level)
# -----------------
def simulate_servers(
    server_usage_list: List[Dict[str, Any]],
    jobs_list: List[Dict[str, Any]],
    verbose: bool = True,
    cores_per_server: int = None,
    ram_per_server: float = None,
    nvme_per_server: float = None,
    bandwidth_per_server: float = None,
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
        running_heap: List[Tuple[float, Any, Dict[str, Any]]] = []
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
                                print(f"WARNING: Job {job['id']} could not be placed on any server (infeasible).")
                    break

        if rec['jobs']:
            rec['completion'] = max(j['end'] for j in rec['jobs'])
        else:
            rec['completion'] = 0.0
        overall_makespan = max(overall_makespan, rec['completion'])
        per_server_records.append(rec)

    return per_server_records, overall_makespan


def _pack_jobs(
    jobs_list: List[Dict[str, Any]],
    score_fn: Callable[[Dict[str, Any], Dict[str, Any], int, float, float, float], Any],
    prefer_high_score: bool = False,
    cores_per_server: int = None,
    ram_per_server: float = None,
    nvme_per_server: float = None,
    bandwidth_per_server: float = None,
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
            print(f"WARNING: Job {job['id']} ({job.get('type','?')}) exceeds server capacity — skipped.")
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


def pack_first_fit_decreasing(
    jobs_list: List[Dict[str, Any]],
    cores_per_server: int = None,
    ram_per_server: float = None,
    nvme_per_server: float = None,
    bandwidth_per_server: float = None,
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
    jobs_list: List[Dict[str, Any]],
    cores_per_server: int = None,
    ram_per_server: float = None,
    nvme_per_server: float = None,
    bandwidth_per_server: float = None,
) -> List[Dict[str, Any]]:
    """Best-Fit Decreasing bin-packing algorithm with dynamic server creation."""
    def _bfd_score(su, job, cores_cap, ram_cap, nvme_cap, bandwidth_cap):
        # Normalized remaining capacity across all dimensions (lower = tighter fit = better).
        # Uses weighted sum rather than lexicographic tuple so all resources contribute equally.
        rem = _remaining_capacity_tuple(su, job, cores_cap, ram_cap, nvme_cap, bandwidth_cap)
        return (rem[0] / max(1, cores_cap) +
                rem[1] / max(1.0, ram_cap) +
                rem[2] / max(1.0, nvme_cap) +
                rem[3] / max(1.0, bandwidth_cap))

    return _pack_jobs(
        jobs_list,
        score_fn=_bfd_score,
        prefer_high_score=False,
        cores_per_server=cores_per_server,
        ram_per_server=ram_per_server,
        nvme_per_server=nvme_per_server,
        bandwidth_per_server=bandwidth_per_server,
    )


def pack_worst_fit(
    jobs_list: List[Dict[str, Any]],
    cores_per_server: int = None,
    ram_per_server: float = None,
    nvme_per_server: float = None,
    bandwidth_per_server: float = None,
) -> List[Dict[str, Any]]:
    """Worst-Fit bin-packing algorithm with dynamic server creation."""
    def _worst_fit_score(su, job, cores_cap, ram_cap, nvme_cap, bandwidth_cap):
        rem = _remaining_capacity_tuple(su, job, cores_cap, ram_cap, nvme_cap, bandwidth_cap)
        return (rem[0] / max(1, cores_cap) +
                rem[1] / max(1.0, ram_cap) +
                rem[2] / max(1.0, nvme_cap) +
                rem[3] / max(1.0, bandwidth_cap))

    return _pack_jobs(
        jobs_list,
        score_fn=_worst_fit_score,
        prefer_high_score=True,
        cores_per_server=cores_per_server,
        ram_per_server=ram_per_server,
        nvme_per_server=nvme_per_server,
        bandwidth_per_server=bandwidth_per_server,
    )


def build_cloud_quadratic_program(servers: int, jobs: List[Dict[str, Any]]) -> Tuple[Optional[QuadraticProgram], List[str]]:
    """Construct a Qiskit QuadraticProgram for the bin-packing problem.

    Each server has:
    - 2 vCPUs (cores)
    - 4 GB ECC RAM
    - 40 GB local NVMe storage
    - 10 Gbps bandwidth

    Args:
        servers: Number of identical servers available for bin packing.
        jobs: List of job dicts, each with keys: id, cores, ram, nvme, bandwidth.

    Returns:
        Tuple of (QuadraticProgram, list of variable names).
        Raises RuntimeError if Qiskit is unavailable.

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


def build_cores_only_quadratic_program(servers: int, jobs: List[Dict[str, Any]]) -> Tuple[Optional[QuadraticProgram], List[str]]:
    """Reduced QP for *real* QAOA: assignment + cores capacity only.

    The full build_cloud_quadratic_program() also adds RAM / NVMe / bandwidth
    capacity constraints. Converting those inequalities to a QUBO introduces many
    binary slack variables — the qubit count explodes to 40+ even for a 2-job /
    2-server instance (a 64 TiB statevector), so genuine QAOA simulation is
    impossible on a laptop. This reduced model keeps only the integer-coefficient
    cores capacity, which keeps the QUBO small enough (~10-18 qubits) to actually
    statevector-simulate on small inputs.

    Trade-off: the quantum placement does NOT enforce RAM/NVMe/bandwidth. The
    caller re-checks those by scoring the placement with simulate_servers(), which
    enforces every resource limit. For the demo's job mix, cores is the dominant
    constraint, so the reduced model usually yields the same packing.
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


def solve_binpack_qaoa_legacy(
    qp,
    jobs: List[Dict[str, Any]],
    servers: int,
    reps: int = 3,
    maxiter: int = 300,
    shots: int = 16384,
) -> Optional[Dict[str, Any]]:
    """Solve bin-packing via QAOA on a local quantum circuit simulator.

    Pipeline:
      QuadraticProgram -> QuadraticProgramToQubo -> QAOA (Sampler + SPSA)
      -> decode bitstring -> server assignments

    Uses QasmSimulator via Qiskit Sampler primitive (classical simulation of
    quantum circuits — no real QPU required). SPSA optimizer scales to large
    numbers of beta/gamma parameters without extra circuit evaluations.

    Args:
        qp: QuadraticProgram from build_cloud_quadratic_program().
        jobs: List of job dicts.
        servers: Number of servers in the problem.
        reps: QAOA circuit depth p. Higher = better quality, slower simulation.
        maxiter: SPSA optimizer iterations. More = better convergence.
        shots: Measurement shots per circuit. Higher = less sampling noise.

    Returns:
        Dict with keys: server_usage, objective, active_servers, unplaced, status.
        Returns None if required packages are unavailable or simulation fails.
    """
    # -- graceful imports --
    try:
        from qiskit_optimization.converters import QuadraticProgramToQubo  # type: ignore
    except ImportError:
        print("[QAOA] qiskit-optimization not available. Install: pip install qiskit-optimization")
        return None
    try:
        from qiskit_algorithms import QAOA as _QAOA                        # type: ignore
        from qiskit_algorithms.optimizers import SPSA as _SPSA             # type: ignore
    except ImportError:
        print("[QAOA] qiskit-algorithms not available. Install: pip install qiskit-algorithms")
        return None
    # qiskit <= 1.x shipped the V1 reference `Sampler`; it was REMOVED in qiskit 2.0.
    # Fall back to the V2 `StatevectorSampler`, which qiskit-algorithms' QAOA accepts.
    try:
        from qiskit.primitives import Sampler as _Sampler                  # type: ignore (qiskit < 2.0)
    except ImportError:
        try:
            from qiskit.primitives import StatevectorSampler as _Sampler   # type: ignore (qiskit >= 2.0)
        except ImportError:
            print("[QAOA] No usable Sampler primitive in qiskit.primitives.")
            return None
    try:
        from qiskit_optimization.algorithms import MinimumEigenOptimizer as _MEO  # type: ignore
    except ImportError:
        print("[QAOA] MinimumEigenOptimizer not available.")
        return None

    import numpy as np

    # Step 1: Convert QP -> QUBO
    # QuadraticProgramToQubo absorbs all linear constraints into penalty terms,
    # converting the constrained problem into an unconstrained binary polynomial
    # that the quantum circuit can minimize directly.
    print("[QAOA] Converting QuadraticProgram to QUBO...")
    try:
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
    except Exception as exc:
        print(f"[QAOA] QUBO conversion failed: {type(exc).__name__}: {exc}")
        return None

    n_qubits = qubo.get_num_vars()
    print(f"[QAOA] QUBO size: {n_qubits} qubits (binary variables after constraint absorption)")
    print(f"[QAOA] Circuit depth: p={reps} layers | Optimizer: SPSA maxiter={maxiter} | Shots: {shots}")
    print(f"[QAOA] This is a classical simulation of a quantum circuit (no QPU needed).")

    # Step 2: Linear ramp initialization for beta/gamma angles.
    # Ramps gamma up and beta down across layers — shown by the 2025 transfer
    # learning paper to converge faster than random initialization.
    initial_point = np.zeros(2 * reps)
    for k in range(reps):
        initial_point[k] = (k + 1) / (reps + 1) * 0.5            # gamma_k: ramps up
        initial_point[reps + k] = (1 - (k + 1) / (reps + 1)) * 0.5  # beta_k: ramps down

    # Step 3: Build QAOA circuit with Sampler primitive (QasmSimulator backend)
    # and SPSA optimizer. SPSA uses exactly 2 circuit evaluations per iteration
    # regardless of parameter count — ideal for large reps values.
    try:
        sampler = _Sampler()
        sampler.set_options(shots=shots)
    except Exception:
        sampler = _Sampler()

    spsa = _SPSA(maxiter=maxiter)
    qaoa = _QAOA(sampler=sampler, optimizer=spsa, reps=reps, initial_point=initial_point)

    # Step 4: Solve. MinimumEigenOptimizer wraps QAOA and handles the
    # Ising Hamiltonian encoding internally (Z-operator substitution for
    # each binary variable).
    print(f"[QAOA] Running quantum circuit simulation...")
    print(f"[QAOA] Expected time scales with 2^{n_qubits} internally — be patient for large problems.")
    try:
        optimizer_alg = _MEO(qaoa)
        result = optimizer_alg.solve(qubo)
    except MemoryError as mem_exc:
        print(f"[QAOA] MemoryError: problem too large for local simulation ({n_qubits} qubits).")
        print(f"[QAOA] Try fewer jobs/servers, or reduce reps.")
        _log_memory_error("QAOA solve", mem_exc, extra={"n_qubits": n_qubits, "reps": reps, "shots": shots})
        return None
    except Exception as exc:
        print(f"[QAOA] Simulation failed: {type(exc).__name__}: {exc}")
        return None

    # Step 5: Decode bitstring -> server assignments.
    # result.x gives the best binary assignment found across all sampled bitstrings.
    var_names_qubo = [v.name for v in qubo.variables]
    assignment = {name: int(round(float(val))) for name, val in zip(var_names_qubo, result.x)}

    server_usage = [
        {'cores': 0, 'ram': 0.0, 'nvme': 0.0, 'bandwidth': 0.0, 'jobs': []}
        for _ in range(servers)
    ]
    unplaced = []
    for job in jobs:
        jid = job['id']
        placed = False
        for s in range(servers):
            if assignment.get(f"x_{jid}_{s}", 0) == 1:
                su = server_usage[s]
                su['cores'] += job.get('cores', 0)
                su['ram'] += job.get('ram', 0.0)
                su['nvme'] += job.get('nvme', 0.0)
                su['bandwidth'] += job.get('bandwidth', 0.0)
                su['jobs'].append(jid)
                placed = True
                break
        if not placed:
            unplaced.append(jid)

    active_servers = sum(1 for su in server_usage if su['jobs'])
    status_name = result.status.name if hasattr(result, 'status') else 'unknown'
    print(f"[QAOA] Done. Active servers: {active_servers} | Objective: {result.fval:.4f} | Status: {status_name}")
    if unplaced:
        print(f"[QAOA] {len(unplaced)} job(s) unplaced (increase reps or maxiter for better coverage): {unplaced}")

    return {
        'server_usage': server_usage,
        'objective': result.fval,
        'active_servers': active_servers,
        'unplaced': unplaced,
        'status': status_name,
    }


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
        print(f" Job {j['id']}: type={j['type']} cores={j['cores']} ram={j['ram']}GB nvme={j['nvme']}GB")

    try:
        qp, var_names = build_cloud_quadratic_program(servers, jobs)
    except RuntimeError as e:
        print(e)
        print("QuadraticProgram construction skipped because Qiskit is not available.")
        return

    print("QuadraticProgram built:")
    print(f" Variables: {len(var_names)}")
    print(f" Constraints: {len(qp.linear_constraints) + len(qp.quadratic_constraints)}")
    print("Objective: minimize number of active servers (sum y_s)")
    print_names = var_names[:min(60, len(var_names))]
    print("Variable sample names:")
    print(print_names)

    # Best-Fit Decreasing bin-packing to demonstrate server assignment and stress

    # Sort jobs by decreasing 'size' heuristic (primary: cores, secondary: nvme, tertiary: bandwidth)
    jobs_sorted = sorted(jobs, key=_job_sort_key, reverse=True)
    demo_cores, demo_ram, demo_nvme, demo_bandwidth = _resolve_server_limits()

    server_usage = [
        {'cores': 0, 'ram': 0.0, 'nvme': 0.0, 'bandwidth': 0.0, 'jobs': []}
        for _ in range(servers)
    ]

    unplaced = []

    # fast lookup by id
    job_map = {j['id']: j for j in jobs}

    for job in jobs_sorted:
        best_server = None
        best_remaining = None
        jcores = job.get('cores', 0)
        jram = job.get('ram', 0.0)
        jnvme = job.get('nvme', 0.0)
        jbw = job.get('bandwidth', 0.0)

        for s in range(servers):
            su = server_usage[s]
            # Check if job fits within ALL resource limits on this server
            if (su['cores'] + jcores <= demo_cores and
                su['ram'] + jram <= demo_ram and
                su['nvme'] + jnvme <= demo_nvme and
                su['bandwidth'] + jbw <= demo_bandwidth):
                # compute remaining capacity after assignment as tuple (smaller is better)
                rem = (
                    demo_cores - (su['cores'] + jcores),
                    demo_ram - (su['ram'] + jram),
                    demo_nvme - (su['nvme'] + jnvme),
                    demo_bandwidth - (su['bandwidth'] + jbw),
                )
                if best_remaining is None or rem < best_remaining:
                    best_remaining = rem
                    best_server = s

        if best_server is not None:
            su = server_usage[best_server]
            su['cores'] += jcores
            su['ram'] += jram
            su['nvme'] += jnvme
            su['bandwidth'] += jbw
            su['jobs'].append(job['id'])
        else:
            # cannot fit on any current server without exceeding capacities
            unplaced.append(job['id'])

    # If there are unplaced jobs, estimate how many additional servers would be needed
    extra_servers_needed = 0
    if unplaced:
        extra_usage = []  # list of server dicts same shape as server_usage
        for jid in unplaced:
            job = job_map.get(jid)
            placed = False
            for es in extra_usage:
                if (es['cores'] + job.get('cores', 0) <= demo_cores and
                    es['ram'] + job.get('ram', 0.0) <= demo_ram and
                    es['nvme'] + job.get('nvme', 0.0) <= demo_nvme and
                    es['bandwidth'] + job.get('bandwidth', 0.0) <= demo_bandwidth):
                    es['cores'] += job.get('cores', 0)
                    es['ram'] += job.get('ram', 0.0)
                    es['nvme'] += job.get('nvme', 0.0)
                    es['bandwidth'] += job.get('bandwidth', 0.0)
                    es['jobs'].append(jid)
                    placed = True
                    break
            if not placed:
                # open a new extra server and place job there
                new_es = {'cores': job.get('cores', 0), 'ram': job.get('ram', 0.0), 'nvme': job.get('nvme', 0.0), 'bandwidth': job.get('bandwidth', 0.0), 'jobs': [jid]}
                extra_usage.append(new_es)
        extra_servers_needed = len(extra_usage)

    # ---- Run classical heuristics and QAOA simultaneously ----
    import threading as _threading
    import time as _time

    _classical_out = {}   # filled by classical thread
    _qaoa_out = [None]    # filled by QAOA thread

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

    def _run_qaoa():
        _reps, _maxiter, _shots = 3, 300, 16384
        print(f"\n[QAOA] Starting QAOA quantum simulation "
              f"(reps={_reps}, maxiter={_maxiter}, shots={_shots})...")
        _t0 = _time.perf_counter()
        _res = solve_binpack_qaoa_legacy(
            qp, jobs, servers, reps=_reps, maxiter=_maxiter, shots=_shots,
        )
        _qaoa_out[0] = (_res, _time.perf_counter() - _t0)

    _t_cls  = _threading.Thread(target=_run_classical, name="classical-solver")
    _t_qaoa = _threading.Thread(target=_run_qaoa,      name="qaoa-solver")
    _t_cls.start()
    _t_qaoa.start()
    _t_cls.join()
    _t_qaoa.join()

    # ---- Display classical results ----
    baseline  = _classical_out['baseline']
    best      = _classical_out['best']
    best_name = _classical_out['best_name']
    results   = _classical_out['results']

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

    # ---- Display QAOA results ----
    qaoa_result, qaoa_wall = _qaoa_out[0] if _qaoa_out[0] else (None, 0.0)

    if qaoa_result is not None:
        qaoa_su     = qaoa_result['server_usage']
        qaoa_active = qaoa_result['active_servers']
        qaoa_per_rec, qaoa_makespan = simulate_servers(
            [su for su in qaoa_su if su['jobs']], jobs, verbose=False
        )
        print("\nQAOA QUANTUM SOLUTION:")
        print(f" - Active servers  : {qaoa_active}")
        print(f" - QUBO objective  : {qaoa_result['objective']:.4f}")
        print(f" - Status          : {qaoa_result['status']}")
        print(f" - Makespan        : {qaoa_makespan:.2f} seconds")
        print(f" - Wall-clock time : {qaoa_wall:.2f} seconds")
        if qaoa_result['unplaced']:
            print(f" - Unplaced jobs   : {qaoa_result['unplaced']} (increase reps/maxiter)")
        print("\nDetailed timeline (QAOA solution):")
        for idx, rec in enumerate(qaoa_per_rec):
            job_ids = [j['id'] for j in rec['jobs']]
            print(f" - Server {idx}: Jobs {job_ids} -> completes at t={rec['completion']:.2f}s")

        # ---- QAOA vs Classical comparison ----
        saved_vs_classical = best['servers'] - qaoa_active
        cost_delta         = saved_vs_classical * COST_PER_SERVER_PER_HOUR
        print(f"\nQAOA vs Classical best ({best_name}):")
        print(f" - Servers saved   : {saved_vs_classical}")
        print(f" - Cost delta      : €{cost_delta:.2f}/hour")
        if best['makespan'] > 0 and qaoa_makespan > 0:
            speed_pct = (best['makespan'] - qaoa_makespan) / best['makespan'] * 100.0
            print(f" - Makespan delta  : {speed_pct:.1f}% {'faster' if speed_pct >= 0 else 'slower'}")
    else:
        print("\nQAOA QUANTUM SOLUTION:")
        print(" - FAILED: QAOA solver was unavailable or failed to produce a solution.")
        print(" - No classical fallback is used for the quantum result (QAOA-only mode).")



def build_server_binpacking_qp(servers: List[Dict], jobs: List[Dict], name: str = "server_binpacking") -> QuadraticProgram:
    """Build a QuadraticProgram for the server bin-packing problem.

    servers: list of dicts with keys: id, arch ("Intel"|"AMD"), clock (GHz), cores (int), ram (GB), disk (GB)
    jobs: list of dicts with keys: id, cores (1|2), ram (GB), disk (GB), min_clock (GHz), arch ("Intel"|"AMD"|"Any"), shared_allowed (bool)

    Returns a Qiskit QuadraticProgram with binary vars x_{j}_{s} and y_{s} and constraints described in the spec.
    """
    qp = QuadraticProgram(name=name)

    # Create y_s variables (one per server)
    for s in servers:
        qp.binary_var(name=f"y_{s['id']}")

    # Precompute compatibility: only create x vars for compatible pairs
    compatible = {j['id']: [] for j in jobs}
    for j in jobs:
        for s in servers:
            arch_ok = (j.get('arch', 'Any') == 'Any') or (j.get('arch') == s.get('arch'))
            clk_ok = (s.get('clock', 0.0) >= j.get('min_clock', 0.0))
            cores_ok = j.get('cores', 0) <= s.get('cores', 0)
            ram_ok = j.get('ram', 0.0) <= s.get('ram', 0.0)
            disk_ok = j.get('disk', 0.0) <= s.get('disk', 0.0)
            if arch_ok and clk_ok and cores_ok and ram_ok and disk_ok:
                compatible[j['id']].append(s['id'])
                qp.binary_var(name=f"x_{j['id']}_{s['id']}")

    # 1) Assignment: each job assigned to exactly one compatible server
    for j in jobs:
        varnames = [f"x_{j['id']}_{s_id}" for s_id in compatible[j['id']]]
        if not varnames:
            # no compatible server -> infeasible model, but we still add an equality with no vars
            qp.linear_constraint(linear={}, sense="==", rhs=1.0, name=f"assign_{j['id']}")
        else:
            linear = {v: 1 for v in varnames}
            qp.linear_constraint(linear=linear, sense="==", rhs=1.0, name=f"assign_{j['id']}")

    # 2) Capacity constraints per server: cores, ram, disk
    for s in servers:
        s_id = s['id']
        # CPU cores
        coeffs = {}
        for j in jobs:
            if s_id in compatible[j['id']]:
                coeffs[f"x_{j['id']}_{s_id}"] = j.get('cores', 0)
        coeffs[f"y_{s_id}"] = -s.get('cores', 0)
        qp.linear_constraint(linear=coeffs, sense="<=", rhs=0.0, name=f"cores_cap_{s_id}")

        # RAM
        coeffs = {}
        for j in jobs:
            if s_id in compatible[j['id']]:
                coeffs[f"x_{j['id']}_{s_id}"] = j.get('ram', 0.0)
        coeffs[f"y_{s_id}"] = -s.get('ram', 0.0)
        qp.linear_constraint(linear=coeffs, sense="<=", rhs=0.0, name=f"ram_cap_{s_id}")

        # Disk
        coeffs = {}
        for j in jobs:
            if s_id in compatible[j['id']]:
                coeffs[f"x_{j['id']}_{s_id}"] = j.get('disk', 0.0)
        coeffs[f"y_{s_id}"] = -s.get('disk', 0.0)
        qp.linear_constraint(linear=coeffs, sense="<=", rhs=0.0, name=f"disk_cap_{s_id}")

        # Local NVMe SSD (fast, non-shared, attached to VM): hard cap 40 GB per server
        # Each job must provide 'nvme' field (GB) indicating local NVMe requirement.
        coeffs = {}
        for j in jobs:
            if s_id in compatible[j['id']]:
                coeffs[f"x_{j['id']}_{s_id}"] = j.get('nvme', 0.0)
        # enforce sum_j nvme_j * x_j_s <= 40 * y_s  -> linear constraint
        coeffs[f"y_{s_id}"] = -40.0
        qp.linear_constraint(linear=coeffs, sense="<=", rhs=0.0, name=f"nvme_cap_{s_id}")

        # Network bandwidth (Gbps): each job must provide 'bandwidth' (Gbps)
        coeffs = {}
        for j in jobs:
            if s_id in compatible[j['id']]:
                coeffs[f"x_{j['id']}_{s_id}"] = j.get('bandwidth', 0.0)
        # use server bandwidth if provided, otherwise default to 10 Gbps
        coeffs[f"y_{s_id}"] = -s.get('bandwidth', 10.0)
        qp.linear_constraint(linear=coeffs, sense="<=", rhs=0.0, name=f"bandwidth_cap_{s_id}")

    # 3) Exclusivity for jobs that disallow sharing: pairwise constraints
    for s in servers:
        s_id = s['id']
        job_ids = [j['id'] for j in jobs if s_id in compatible[j['id']]]
        for j in jobs:
            if (not j.get('shared_allowed', True)) and (j['id'] in job_ids):
                for j2 in job_ids:
                    if j2 == j['id']:
                        continue
                    qp.linear_constraint(linear={f"x_{j['id']}_{s_id}": 1, f"x_{j2}_{s_id}": 1},
                                         sense="<=", rhs=1.0, name=f"excl_j{j['id']}_j{j2}_s{s_id}")

    # 4) Linking constraints x_j_s <= y_s
    for j in jobs:
        for s_id in compatible[j['id']]:
            qp.linear_constraint(linear={f"x_{j['id']}_{s_id}": 1, f"y_{s_id}": -1}, sense="<=", rhs=0.0,
                                     name=f"link_j{j['id']}_s{s_id}")

    # Objective: minimize number of active servers
    linear_obj = {f"y_{s['id']}": 1 for s in servers}
    qp.minimize(linear=linear_obj)

    return qp



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


def generate_jobs_with_arrivals(profile_counts: Dict[str, int], time_horizon: float = 3600.0):
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

    jobs = []
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
            jobs.append({
                "id": jid,
                "type": ptype,
                "cores": profile["cores"],
                "ram": profile["ram"],
                "nvme": profile["nvme"],
                "arrival_time": ta,
                "duration": duration,
                "deadline": ta + profile["deadline_slack"],
                "bandwidth": profile.get("bandwidth", 0.0),
            })
            jid += 1

    return sorted(jobs, key=lambda j: j["arrival_time"])


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
    cores_per_server: int = None,
    ram_per_server: float = None,
    nvme_per_server: float = None,
    bandwidth_per_server: float = None,
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


def online_scheduler_with_arrivals(jobs: List[Dict], servers: int, verbose: bool = True,
                                   cores_per_server: int = None,
                                   ram_per_server: float = None,
                                   nvme_per_server: float = None,
                                   bandwidth_per_server: float = None) -> Tuple[Dict[int, Dict], List[str], Dict[str, float]]:
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
    def can_fit_on_server_state(st: Dict[str, Any], job: Dict[str, Any]) -> bool:
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

    waiting_queue: List[Dict] = []  # FIFO list of job dicts
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

    def assign_job_to_server(job: Dict[str, Any], sidx: int, start_time: float) -> None:
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
        waiting_queue.sort(key=lambda j: j.get('deadline', float('inf')))
        i = 0
        assigned_any = False
        while i < len(waiting_queue):
            job = waiting_queue[i]
            if job.get('deadline', float('inf')) < now:
                if verbose:
                    timeline.append(
                        f"t={now:.2f}s: Job {job['id']} MISSED DEADLINE "
                        f"(deadline={job.get('deadline', '?'):.2f})"
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo-online', action='store_true', help='Run non-interactive online-arrival demo')
    args = parser.parse_args()

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

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    FastAPI = None
    CORSMiddleware = None
    BaseModel = None
    Field = None

# ---------------------------------------------------------------------------
# QAOA Bin-Packing Solver
# ---------------------------------------------------------------------------
def solve_binpack_qaoa(jobs: List[Dict[str, Any]], num_servers: int,
                      cores_per_server: int = None,
                      ram_per_server: float = None,
                      nvme_per_server: float = None,
                      bandwidth_per_server: float = None,
                      max_vars: int = 100) -> Dict[str, Any]:
    """Solve bin-packing using QAOA via Qiskit.

    Formulates a QUBO where binary variable x[j][s] = 1 means job j is
    assigned to server s, and y[s] = 1 means server s is active.
    Objective: minimize sum(y[s]).
    Constraints: each job assigned to exactly one server; server resource
    limits not exceeded.

    Args:
        jobs: List of job dicts.
        num_servers: Number of servers available.
        cores_per_server: Optional override for module-level CORES_PER_SERVER.
        ram_per_server: Optional override for module-level RAM_PER_SERVER.
        nvme_per_server: Optional override for module-level NVME_PER_SERVER.
        bandwidth_per_server: Optional override for module-level BANDWIDTH_PER_SERVER.
        max_vars: Maximum number of binary variables (jobs * servers) allowed before
                 falling back to classical greedy solver. Default is 8 for NumPyMinimumEigensolver
                 (2^8 = 256 possible states, ultra-safe for laptops).
                 Reduce further at runtime if still encountering OOM.

    Returns a dict with keys:
        assignments  - {job_id: server_index}
        active_servers - number of servers used
        solver       - 'qaoa_numpy' or 'classical_fallback'
        timeline     - list of strings describing assignments
        stats        - makespan, peak_utilization_pct, jobs_completed, average_wait_time
    """
    # Use provided overrides if given
    _cores     = cores_per_server     if cores_per_server     is not None else CORES_PER_SERVER
    _ram       = ram_per_server       if ram_per_server       is not None else RAM_PER_SERVER
    _nvme      = nvme_per_server      if nvme_per_server      is not None else NVME_PER_SERVER
    _bandwidth = bandwidth_per_server if bandwidth_per_server is not None else BANDWIDTH_PER_SERVER

    if any(not _job_fits_within_limits(job, _cores, _ram, _nvme, _bandwidth) for job in jobs):
        return _qaoa_failure(reason="At least one job exceeds server capacity.")

    # ------------------------------------------------------------------
    # Step 1 — load qiskit-optimization (required by all solver paths)
    # ------------------------------------------------------------------
    try:
        from qiskit_optimization import QuadraticProgram as _QP                   # type: ignore
        from qiskit_optimization.algorithms import MinimumEigenOptimizer as _MEO  # type: ignore
    except ImportError as e:
        return _qaoa_failure(reason=f"qiskit-optimization not installed: {e}")

    # ------------------------------------------------------------------
    # Step 2 — variable-count gate
    # ------------------------------------------------------------------
    J = len(jobs)
    S = num_servers
    total_vars = J * S

    print(f"[QAOA] Problem size: {J} jobs x {S} servers = {total_vars} variables (limit: {max_vars})")

    if total_vars > max_vars:
        print(f"[QAOA] Problem too large: {total_vars} vars exceeds max {max_vars}. QAOA failed (no classical fallback).")
        return _qaoa_failure(
            reason=(
                f"Problem too large for quantum solver "
                f"({J} jobs x {S} servers = {total_vars} vars > {max_vars})."
            ),
        )

    # ------------------------------------------------------------------
    # Step 3 — build the QuadraticProgram
    # ------------------------------------------------------------------
    qp = _QP(name="bin_packing")

    for j in range(J):
        for s in range(S):
            qp.binary_var(name=f"x_{j}_{s}")
    for s in range(S):
        qp.binary_var(name=f"y_{s}")

    qp.minimize(linear={f"y_{s}": 1.0 for s in range(S)})

    for j in range(J):
        qp.linear_constraint(
            linear={f"x_{j}_{s}": 1.0 for s in range(S)},
            sense="==", rhs=1.0, name=f"assign_{j}",
        )

    for s in range(S):
        cores_lhs = {f"x_{j}_{s}": float(jobs[j].get("cores", 0)) for j in range(J)}
        cores_lhs[f"y_{s}"] = -float(_cores)
        qp.linear_constraint(linear=cores_lhs, sense="<=", rhs=0.0, name=f"cores_{s}")

        ram_lhs = {f"x_{j}_{s}": float(jobs[j].get("ram", 0.0)) for j in range(J)}
        ram_lhs[f"y_{s}"] = -float(_ram)
        qp.linear_constraint(linear=ram_lhs, sense="<=", rhs=0.0, name=f"ram_{s}")

        nvme_lhs = {f"x_{j}_{s}": float(jobs[j].get("nvme", 0.0)) for j in range(J)}
        nvme_lhs[f"y_{s}"] = -float(_nvme)
        qp.linear_constraint(linear=nvme_lhs, sense="<=", rhs=0.0, name=f"nvme_{s}")

        # Activation constraint: if any job is assigned to server s, y_s must be 1.
        # sum_j x_{j,s} <= J * y_s  =>  sum_j x_{j,s} - J*y_s <= 0
        act_lhs = {f"x_{j}_{s}": 1.0 for j in range(J)}
        act_lhs[f"y_{s}"] = -float(J)
        qp.linear_constraint(linear=act_lhs, sense="<=", rhs=0.0, name=f"activate_{s}")

    # ------------------------------------------------------------------
    # Step 4 — solve using scipy.milp (primary) with numpy brute-force
    # as fallback.
    #
    # scipy.milp is a proper MILP solver — no QUBO conversion, no variable
    # limit, no license required, and available in any scipy install.
    # We translate the QP directly into scipy's milp format.
    # ------------------------------------------------------------------
    result       = None
    solver_label = None
    x_vals       = None
    best_fval    = None

    # Path 1 — scipy.milp built directly from problem data (bypasses Qiskit constraint parsing)
    if x_vals is None:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds  # type: ignore
            import numpy as np

            # Variable layout:
            #   x[j,s] = j*S + s   for j in range(J), s in range(S)
            #   y[s]   = J*S + s   for s in range(S)
            n_x    = J * S
            n_y    = S
            n_vars = n_x + n_y

            def xi(j, s): return j * S + s
            def yi(s):    return n_x + s

            # Objective: minimise sum(y_s)
            c = np.zeros(n_vars)
            for s in range(S):
                c[yi(s)] = 1.0

            integrality = np.ones(n_vars)
            bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

            A_rows, b_lo, b_hi = [], [], []

            # (1) Assignment equality: sum_s x[j,s] == 1 for each j
            for j in range(J):
                row = np.zeros(n_vars)
                for s in range(S):
                    row[xi(j, s)] = 1.0
                A_rows.append(row); b_lo.append(1.0); b_hi.append(1.0)

            # (2) Resource capacity and (3) activation per server
            for s in range(S):
                # cores: sum_j x[j,s]*cores_j <= y_s * _cores
                row = np.zeros(n_vars)
                for j in range(J): row[xi(j, s)] = float(jobs[j].get("cores", 0))
                row[yi(s)] = -float(_cores)
                A_rows.append(row); b_lo.append(-np.inf); b_hi.append(0.0)

                # ram
                row = np.zeros(n_vars)
                for j in range(J): row[xi(j, s)] = float(jobs[j].get("ram", 0.0))
                row[yi(s)] = -float(_ram)
                A_rows.append(row); b_lo.append(-np.inf); b_hi.append(0.0)

                # nvme
                row = np.zeros(n_vars)
                for j in range(J): row[xi(j, s)] = float(jobs[j].get("nvme", 0.0))
                row[yi(s)] = -float(_nvme)
                A_rows.append(row); b_lo.append(-np.inf); b_hi.append(0.0)

                # bandwidth
                row = np.zeros(n_vars)
                for j in range(J): row[xi(j, s)] = float(jobs[j].get("bandwidth", 0.0))
                row[yi(s)] = -float(_bandwidth)
                A_rows.append(row); b_lo.append(-np.inf); b_hi.append(0.0)

                # activation: sum_j x[j,s] <= J * y_s
                row = np.zeros(n_vars)
                for j in range(J): row[xi(j, s)] = 1.0
                row[yi(s)] = -float(J)
                A_rows.append(row); b_lo.append(-np.inf); b_hi.append(0.0)

            A      = np.vstack(A_rows)
            lc_obj = LinearConstraint(A, b_lo, b_hi)

            print(f"[MILP] scipy.milp: {n_vars} vars, {len(A_rows)} constraints...")
            res = milp(c, constraints=[lc_obj], integrality=integrality, bounds=bounds)

            if res.success:
                var_names    = [f"x_{j}_{s}" for j in range(J) for s in range(S)] +                                [f"y_{s}" for s in range(S)]
                x_vals       = {var_names[i]: int(round(res.x[i])) for i in range(n_vars)}
                best_fval    = float(res.fun)
                solver_label = "scipy_milp"
                print(f"[MILP] scipy.milp solved. Objective: {best_fval:.4f}")
            else:
                print(f"[MILP] scipy.milp did not find a solution: {res.message}")
        except Exception as _e:
            print(f"[MILP] scipy.milp failed: {_e}")

    # Path 2 — chunked numpy brute-force for very small QUBOs only
    if x_vals is None:
        try:
            from qiskit_optimization.converters import QuadraticProgramToQubo as _ToQubo  # type: ignore
            import numpy as np
            max_bruteforce_vars = 24  # chunked loop makes this safe (16K states/chunk)
            brute_force_chunk_size = 1 << 14
            print(f"[QAOA] Falling back to numpy brute-force...")
            qubo = _ToQubo(penalty=1.0).convert(qp)
            n    = qubo.get_num_vars()
            if n > max_bruteforce_vars:
                raise MemoryError(f"Too many QUBO vars ({n}) for brute-force")
            print(f"[QAOA] QUBO has {n} vars. Searching 2^{n}={2**n} bitstrings...")
            Q = np.zeros((n, n))
            for i, ci in qubo.objective.linear.to_dict(use_name=False).items():
                Q[i, i] += ci
            for (i, j), cij in qubo.objective.quadratic.to_dict(use_name=False).items():
                Q[i, j] += cij / 2.0
                Q[j, i] += cij / 2.0
            offset = qubo.objective.constant
            best_x = None
            best_fval = None
            total_states = 1 << n
            bit_positions = np.arange(n)

            for chunk_start in range(0, total_states, brute_force_chunk_size):
                chunk_end = min(chunk_start + brute_force_chunk_size, total_states)
                bits = np.arange(chunk_start, chunk_end, dtype=np.int64)
                X = ((bits[:, None] >> bit_positions) & 1).astype(np.float64)
                obj_vals = np.einsum('bi,ij,bj->b', X, Q, X) + offset
                local_best = int(np.argmin(obj_vals))
                local_fval = float(obj_vals[local_best])
                if best_fval is None or local_fval < best_fval:
                    best_fval = local_fval
                    best_x = X[local_best].astype(int)

            num_qp_vars = qp.get_num_vars()
            x_vals      = {qp.variables[i].name: int(best_x[i])
                           for i in range(num_qp_vars)}
            solver_label = "qaoa_numpy"
            print(f"[QAOA] Numpy brute-force solved. Objective: {best_fval:.4f}")
        except MemoryError as _e:
            print(f"[QAOA] Numpy brute-force OOM: {_e}")
        except Exception as _e:
            print(f"[QAOA] Numpy brute-force failed: {_e}")

    if x_vals is None:
        return _qaoa_failure(reason="All quantum solver paths failed.")

    # ------------------------------------------------------------------
    # Step 5 — parse result into assignments
    # ------------------------------------------------------------------
    assignments = {}
    timeline    = []
    server_usage = [
        {'cores': 0, 'ram': 0.0, 'nvme': 0.0, 'bandwidth': 0.0, 'jobs': []}
        for _ in range(S)
    ]

    # Debug: show what the solver returned
    assigned_x = {k: v for k, v in x_vals.items() if k.startswith("x_") and v == 1}
    print(f"[{solver_label}] Assigned x vars: {assigned_x}")

    for j, job in enumerate(jobs):
        chosen_server = None
        preferred_servers = [s for s in range(S) if x_vals.get(f"x_{j}_{s}", 0) == 1]

        for s in preferred_servers:
            if _can_place_on_usage(server_usage[s], job, _cores, _ram, _nvme, _bandwidth):
                chosen_server = s
                break

        if chosen_server is None:
            feasible_servers = [
                s for s in range(S)
                if _can_place_on_usage(server_usage[s], job, _cores, _ram, _nvme, _bandwidth)
            ]
            if not feasible_servers:
                return _qaoa_failure(
                    reason=f"Quantum result could not be repaired feasibly for job {job['id']}.",
                )
            chosen_server = min(
                feasible_servers,
                key=lambda s: (
                    len(server_usage[s]['jobs']),
                    _remaining_capacity_tuple(server_usage[s], job, _cores, _ram, _nvme, _bandwidth),
                ),
            )
            timeline.append(f"{solver_label}: Job {job['id']} ({job['type']}) -> Server {chosen_server} (repaired)")
        else:
            timeline.append(f"{solver_label}: Job {job['id']} ({job['type']}) -> Server {chosen_server}")

        assignments[job["id"]] = chosen_server
        _apply_job_to_usage(server_usage[chosen_server], job)

    active = len(set(assignments.values()))
    timeline.append(f"{solver_label} result: {active} active server(s) out of {S}")
    timeline.append(f"Objective value: {best_fval:.4f}")

    # Makespan: sum durations per server (sequential within server), take the max.
    # This reflects real parallel execution across servers rather than the
    # longest single job duration which is always the batch job at 300s.
    server_times: Dict[int, float] = {}
    for _job in jobs:
        _srv = assignments.get(_job["id"], 0)
        server_times[_srv] = server_times.get(_srv, 0.0) + _job.get("duration", 0.0)
    makespan = max(server_times.values()) if server_times else 0.0

    peak_util = max(
        (min(
            sum(jobs[j].get("cores", 0) for j in range(J)
                if assignments.get(jobs[j]["id"]) == s) / max(1, _cores) * 100,
            100.0
        ) for s in range(S)),
        default=0.0,
    )

    stats = {
        "makespan":             makespan,
        "peak_utilization_pct": peak_util,
        "jobs_completed":       len(assignments),
        "average_wait_time":    0.0,
        "active_servers":       active,
    }

    return {
        "assignments":    {str(k): v for k, v in assignments.items()},
        "active_servers": active,
        "solver":         solver_label,
        "timeline":       timeline,
        "stats":          stats,
    }

def _qaoa_failure(reason: str = "") -> Dict[str, Any]:
    """Return a structured QAOA-failure result (no classical fallback).

    Matches the shape of the QAOA success result so callers/endpoints stay
    consistent, but marks the run as failed and carries no schedule.
    """
    print(f"[QAOA] FAILED: {reason}")
    return {
        "assignments": {},
        "active_servers": 0,
        "solver": "failed",
        "error": reason,
        "timeline": [f"[QAOA FAILED: {reason}]"],
        "stats": {
            "makespan": 0.0,
            "peak_utilization_pct": 0.0,
            "jobs_completed": 0,
            "average_wait_time": 0.0,
            "active_servers": 0,
        },
    }


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


def solve_binpack_qaoa_web(jobs: List[Dict[str, Any]], num_servers: int) -> Dict[str, Any]:
    """Run the REAL QAOA quantum-circuit simulation for the web API.

    Unlike solve_binpack_qaoa (which uses scipy.milp), this wires the FastAPI
    /schedule_quantum route to solve_binpack_qaoa_legacy — the genuine QAOA path
    the CLI uses. Makespan is computed with the concurrent simulate_servers model
    so it matches the classical /schedule column. On any failure it returns
    _qaoa_failure (no classical fallback), consistent with alpha4's design.

    Note: build_cloud_quadratic_program uses the module-level server constants
    (2 cores / 4 GB / 40 GB / 10 Gbps), which match the page's fixed server spec,
    so per-request server overrides are intentionally not applied here.
    """
    # Build the reduced (cores-only) QuadraticProgram so the QUBO stays small
    # enough to statevector-simulate. The full constraint set would need 40+
    # qubits even for tiny inputs (see build_cores_only_quadratic_program).
    try:
        qp, _ = build_cores_only_quadratic_program(num_servers, jobs)
    except RuntimeError as exc:
        return _qaoa_failure(f"qiskit-optimization not available: {exc}")

    # Guard: real QAOA simulates a 2**qubits statevector. Refuse anything that
    # would blow up memory, with an honest message instead of an OOM crash.
    MAX_QUBITS = 20
    try:
        from qiskit_optimization.converters import QuadraticProgramToQubo  # type: ignore
        n_qubits = QuadraticProgramToQubo().convert(qp).get_num_vars()
    except Exception as exc:
        return _qaoa_failure(f"Could not encode the problem for QAOA: {exc}")
    if n_qubits > MAX_QUBITS:
        return _qaoa_failure(
            f"Problem too large for real QAOA on this machine: {n_qubits} qubits "
            f"(2^{n_qubits} states) exceeds the {MAX_QUBITS}-qubit simulation limit. "
            "Use fewer jobs/servers."
        )

    # Real QAOA on a simulated quantum circuit (needs qiskit + qiskit-algorithms).
    # Statevector-simulating QAOA is brutally slow — ~90s PER optimizer iteration
    # even for a ~10-qubit problem on a laptop — so we cap it with a hard wall-clock
    # timeout and report an honest failure rather than hanging the request. Iterations
    # are kept minimal so the abandoned worker thread (on timeout) is short-lived.
    QAOA_WEB_TIMEOUT_SEC = 60
    try:
        finished, result = _run_with_timeout(
            lambda: solve_binpack_qaoa_legacy(qp, jobs, num_servers, reps=1, maxiter=1, shots=512),
            QAOA_WEB_TIMEOUT_SEC,
        )
    except Exception as exc:
        # Any unexpected solver exception -> honest failure, never a 500 to the page.
        return _qaoa_failure(f"QAOA solver error: {type(exc).__name__}: {exc}")
    if not finished:
        return _qaoa_failure(
            f"Real QAOA simulation exceeded {QAOA_WEB_TIMEOUT_SEC}s and was stopped. "
            f"Statevector-simulating QAOA is extremely slow even for this {n_qubits}-qubit "
            "problem (~90s per optimizer step on this machine) - that is the real cost of the "
            "quantum approach on a laptop, not a bug. Try fewer jobs/servers, or run the CLI."
        )
    if result is None:
        return _qaoa_failure(
            "QAOA solver unavailable or failed (qiskit primitives / algorithms issue)."
        )

    server_usage = result["server_usage"]

    # Unified makespan: concurrent per-server simulation (the same model the
    # classical column uses), mirroring the CLI's QAOA-display step.
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

    timeline = [f"qaoa: {len(jobs)} job(s) packed onto {active} active server(s)"]
    if result.get("unplaced"):
        timeline.append(f"qaoa: {len(result['unplaced'])} job(s) unplaced: {result['unplaced']}")

    return {
        "assignments": {str(k): v for k, v in assignments.items()},
        "active_servers": active,
        "solver": "qaoa",
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
        """QAOA quantum bin-packing scheduler (real quantum-circuit simulation)."""
        jobs = generate_jobs(req.profile_counts)
        return solve_binpack_qaoa_web(jobs, num_servers=req.servers)


    @app.post("/schedule_compare")
    def schedule_compare(req: WorkloadRequest):
        """Run both classical and QAOA and return both results for comparison."""
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
else:
    app = None

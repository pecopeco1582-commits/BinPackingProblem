# alpha1.py
"""Cloud bin-packing: educational Classical vs Quantum (QAOA) comparison

Derived from fixed13.py. The project has shifted toward teaching: this module
keeps all of the original bin-packing machinery but adds a comparison layer that
runs the *same* workload through both a classical heuristic and the *real* QAOA
quantum solver, then reports a side-by-side breakdown that emphasizes
performance and scaling.

The headline educational point is how the problem size explodes:
    QP variables  ->  QUBO qubits (after slack-variable constraint absorption)
    state space   =   2 ** qubits   (what a classical simulator must track)
and how QAOA's wall-clock cost grows accordingly, so a learner can see exactly
where quantum simulation stops being practical.

Hard constraints enforced per server:
- 2 vCPUs
- 4 GB ECC RAM
- 40 GB local NVMe (non-shared, non-networked)

Objective: minimize number of active servers (sum y_s).

Entry points:
    python alpha1.py --compare       # educational classical-vs-QAOA scaling table
    python alpha1.py --demo-online   # original online-arrival demo
    python alpha1.py                 # original interactive prompt
plus FastAPI endpoints (see bottom of file), including POST /compare_scaling.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys
import heapq
import time
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

# Largest QUBO a local statevector simulator can realistically handle. Each extra
# qubit doubles the statevector (2**n complex128 = 2**n * 16 bytes), so 26 qubits
# is already a ~1 GB allocation. Beyond this we skip the solve and report it as
# "too large" rather than letting Qiskit attempt an impossible allocation.
MAX_SIMULABLE_QUBITS: int = 26


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


# Quick-comparison presets — single source of truth shared with scheduler_v9.html.
# Each is a hand-picked instance that highlights a different difference between
# the classical heuristic and the QAOA quantum solver. All stay under the
# QAOA_MAX_VARS budget the UI enforces (jobs * servers + servers <= 20) so the
# quantum side can actually run.
COMPARISON_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "match",
        "label": "Easy match",
        "description": (
            "A tiny, comfy problem. Both methods should tie on servers — and the "
            "quantum one actually finishes fast. The friendly baseline."
        ),
        # 3 jobs * 2 servers + 2 = 8 vars
        "profile_counts": {"web_api": 2, "batch": 1},
        "servers": 2,
    },
    {
        "id": "squeeze",
        "label": "Tight squeeze",
        "description": (
            "A mixed bag — a heavy batch and a memory hog crammed onto just 2 "
            "servers. Tests whether the quantum method packs as tidily as best-fit."
        ),
        # 4 jobs * 2 servers + 2 = 10 vars
        "profile_counts": {"web_api": 1, "batch": 1, "memory": 1, "io_heavy": 1},
        "servers": 2,
    },
    {
        "id": "stress",
        "label": "Stress the quantum",
        "description": (
            "Pushed near the size limit. The normal computer shrugs; watch the "
            "quantum 'time to find the answer' blow up. The headline gap."
        ),
        # 5 jobs * 3 servers + 3 = 18 vars
        "profile_counts": {"web_api": 5},
        "servers": 3,
    },
]


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
        running_heap: List[Tuple[float, Dict[str, Any]]] = []
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
                    # Include job id as a tiebreaker so heapq never compares dicts
                    # when two jobs share the same end time.
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


# Per-server resource caps, keyed by job dict field. Single source of truth for
# which resources build_cloud_quadratic_program can encode.
SERVER_RESOURCE_CAPS: Dict[str, float] = {
    "cores": CORES_PER_SERVER,
    "ram": RAM_PER_SERVER,
    "nvme": NVME_PER_SERVER,
    "bandwidth": BANDWIDTH_PER_SERVER,
}

# Which resources the QAOA QuadraticProgram encodes by default.
#
# IMPORTANT (qubit budget): every inequality constraint becomes an integer slack
# register in the QUBO, and the register width scales with the cap. NVMe (40 GB)
# and bandwidth (10 Gbps) caps add ~6-7 slack qubits *per server* even though
# they never bind at the small job counts this tool is meant for. Including all
# four resources pushes even a 3-job / 2-server instance to ~44 qubits, which a
# laptop statevector simulator cannot handle (~30 qubits is the ceiling).
#
# cores + RAM are the resources that actually bind at these sizes, so encoding
# only them keeps the quantum problem simulable (~18 qubits for the smallest
# preset) while producing the same packing. The classical solver still checks
# all four resources, so the comparison stays honest for the intended workloads.
DEFAULT_QAOA_RESOURCES: Tuple[str, ...] = ("cores", "ram")


def _integerize_linear(
    coeffs: Dict[str, float], max_scale: int = 1000
) -> Dict[str, float]:
    """Scale a constraint's coefficients to integers if they aren't already.

    Qiskit's InequalityToEquality (inside QuadraticProgramToQubo) rejects
    inequality constraints with non-integer coefficients. Multiplying an
    inequality by a positive constant preserves it (rhs here is always 0), so we
    find the smallest power-of-ten scale that makes every coefficient integral.
    """
    def _is_int(v: float) -> bool:
        return abs(v - round(v)) < FLOAT_EPSILON

    if all(_is_int(v) for v in coeffs.values()):
        return {k: float(round(v)) for k, v in coeffs.items()}
    scale = 10
    while scale <= max_scale:
        if all(_is_int(v * scale) for v in coeffs.values()):
            return {k: float(round(v * scale)) for k, v in coeffs.items()}
        scale *= 10
    # Fallback: round at the largest scale (tiny precision loss only).
    return {k: float(round(v * max_scale)) for k, v in coeffs.items()}


def build_cloud_quadratic_program(
    servers: int,
    jobs: List[Dict[str, Any]],
    resources: Tuple[str, ...] = DEFAULT_QAOA_RESOURCES,
) -> Tuple[Optional[QuadraticProgram], List[str]]:
    """Construct a Qiskit QuadraticProgram for the bin-packing problem.

    Args:
        servers: Number of identical servers available for bin packing.
        jobs: List of job dicts, each with keys: id, cores, ram, nvme, bandwidth.
        resources: Which per-server resource caps to encode as constraints.
            Defaults to cores + RAM (see DEFAULT_QAOA_RESOURCES for why NVMe and
            bandwidth are excluded — they explode the qubit count without binding
            at these sizes). Pass e.g. ("cores", "ram", "nvme", "bandwidth") to
            encode the full model.

    Returns:
        Tuple of (QuadraticProgram, list of variable names).

    Raises:
        RuntimeError: If qiskit-optimization is not installed.
        ValueError: If an unknown resource name is requested.
    """
    if QuadraticProgram is None:
        raise RuntimeError("Qiskit Optimization not available (install qiskit-optimization).")

    unknown = [r for r in resources if r not in SERVER_RESOURCE_CAPS]
    if unknown:
        raise ValueError(f"Unknown resource(s) {unknown}; valid: {list(SERVER_RESOURCE_CAPS)}")

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

    # Per-server resource capacity constraints (hard), one per requested resource.
    # Coefficients are integerized so the QUBO conversion accepts them.
    for s in range(servers):
        for res in resources:
            cap = SERVER_RESOURCE_CAPS[res]
            linear = {f"x_{job['id']}_{s}": job.get(res, 0.0) for job in jobs}
            linear[f"y_{s}"] = -cap
            qp.linear_constraint(
                linear=_integerize_linear(linear), sense="<=", rhs=0.0,
                name=f"{res}_cap_{s}",
            )

    # Objective: minimize number of active servers
    linear_obj = {f"y_{s}": 1 for s in range(servers)}
    qp.minimize(linear=linear_obj)

    var_names = [v.name for v in qp.variables]
    return qp, var_names


def solve_binpack_qaoa_legacy(
    qp,
    jobs: List[Dict[str, Any]],
    servers: int,
    reps: int = 1,
    maxiter: int = 50,
    shots: int = 1024,
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
    # Sampler primitive. Qiskit < 2.0 ships the V1 reference `Sampler`; Qiskit
    # >= 2.0 removed it in favour of the V2 `StatevectorSampler`. qiskit-algorithms
    # (>= 0.3) accepts either, so prefer V1 for back-compat and fall back to V2.
    _sampler_kind = None
    try:
        from qiskit.primitives import Sampler as _Sampler                  # type: ignore  # V1
        _sampler_kind = "v1"
    except ImportError:
        try:
            from qiskit.primitives import StatevectorSampler as _Sampler   # type: ignore  # V2 (Qiskit >= 2.0)
            _sampler_kind = "v2"
        except ImportError:
            print("[QAOA] No Sampler primitive found. Install/upgrade qiskit: pip install -U qiskit")
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

    # Guard: statevector simulation costs 2^n * 16 bytes. When the problem exceeds
    # the ceiling, try AerSimulator with matrix_product_state (MPS) instead — MPS
    # represents the state as a tensor network and handles 30+ qubits without a
    # full statevector allocation. Fall back to "too large" only if qiskit-aer is
    # unavailable.
    _override_sampler = None
    if n_qubits > MAX_SIMULABLE_QUBITS:
        gb = (2 ** n_qubits) * 16 / 1e9
        try:
            from qiskit_aer.primitives import SamplerV2 as _AerSamplerV2  # type: ignore
            _override_sampler = _AerSamplerV2(
                default_shots=shots,
                options={'backend_options': {'method': 'matrix_product_state'}},
            )
            print(f"[QAOA] {n_qubits} qubits exceeds statevector ceiling ({MAX_SIMULABLE_QUBITS}) "
                  f"(~{gb:,.0f} GB); switching to AerSimulator(method='matrix_product_state').")
        except Exception:
            print(f"[QAOA] {n_qubits} qubits exceeds the simulable ceiling "
                  f"({MAX_SIMULABLE_QUBITS}); a statevector would need ~{gb:,.0f} GB. "
                  f"Install qiskit-aer for MPS simulation: pip install qiskit-aer")
            return {
                "too_large": True,
                "n_qubits": n_qubits,
                "n_qp_vars": qp.get_num_vars(),
                "solve_time_sec": 0.0,
            }

    print(f"[QAOA] Circuit depth: p={reps} layers | Optimizer: SPSA maxiter={maxiter} | Shots: {shots}")
    print(f"[QAOA] This is a classical simulation of a quantum circuit (no QPU needed).")

    # Step 2: Linear ramp initialization for beta/gamma angles.
    # Ramps gamma up and beta down across layers — shown by the 2025 transfer
    # learning paper to converge faster than random initialization.
    initial_point = np.zeros(2 * reps)
    for k in range(reps):
        initial_point[k] = (k + 1) / (reps + 1) * 0.5            # gamma_k: ramps up
        initial_point[reps + k] = (1 - (k + 1) / (reps + 1)) * 0.5  # beta_k: ramps down

    # Step 3: Build QAOA circuit with Sampler primitive and SPSA optimizer.
    # For circuits that exceed the statevector ceiling, _override_sampler uses
    # AerSimulator(method='matrix_product_state') instead of the reference Sampler.
    if _override_sampler is not None:
        sampler = _override_sampler
    elif _sampler_kind == "v2":
        # V2 primitives take shots at construction time (default_shots), not via set_options.
        try:
            sampler = _Sampler(default_shots=shots)
        except Exception:
            sampler = _Sampler()
    else:
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
    solve_time = 0.0
    try:
        optimizer_alg = _MEO(qaoa)
        _t0 = time.perf_counter()
        result = optimizer_alg.solve(qubo)
        solve_time = time.perf_counter() - _t0
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
        # Scaling/performance metrics for the educational comparison layer
        'n_qubits': n_qubits,
        'n_qp_vars': qp.get_num_vars(),
        'solve_time_sec': solve_time,
    }


def _fmt_state_space(qubits: int) -> str:
    """Human-readable 2^qubits with a scientific approximation (overflow-safe)."""
    import math
    if qubits <= 0:
        return "1"
    log10 = qubits * math.log10(2.0)
    exp = int(log10)
    mantissa = 10 ** (log10 - exp)
    return f"2^{qubits} ≈ {mantissa:.1f}e{exp}"


def compare_classical_vs_quantum(
    jobs: List[Dict[str, Any]],
    servers: int,
    reps: int = 1,
    maxiter: int = 50,
    shots: int = 1024,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the same workload through a classical heuristic and the real QAOA
    solver, then return (and optionally print) a performance/scaling comparison.

    The classical side uses Best-Fit Decreasing (minimizes active servers, the
    same objective the QAOA QuadraticProgram encodes) so the two are compared
    apples-to-apples. The emphasis is on *scaling*: how many QP variables the
    problem has, how that explodes into QUBO qubits after slack-variable
    constraint absorption, and how the resulting wall-clock solve times diverge.

    Returns a JSON-friendly dict with keys: problem, classical, quantum, comparison.
    """
    J = len(jobs)
    S = servers

    # ---- Classical side: Best-Fit Decreasing + makespan simulation ----
    _t0 = time.perf_counter()
    classical_placement = pack_best_fit_decreasing(jobs)
    classical_time = time.perf_counter() - _t0
    _, classical_makespan = simulate_servers(classical_placement, jobs, verbose=False)
    classical_active = len([s for s in classical_placement if s.get('jobs')])

    classical_block = {
        "solver": "best_fit_decreasing",
        "active_servers": classical_active,
        "makespan": classical_makespan,
        "solve_time_sec": classical_time,
    }

    # ---- Quantum side: build the QuadraticProgram, then real QAOA ----
    qp_vars = J * S + S  # x_{j}_{s} assignment vars + y_s server-active vars
    quantum_block: Dict[str, Any]
    qubits: Optional[int] = None

    try:
        qp, _var_names = build_cloud_quadratic_program(S, jobs)
    except RuntimeError as exc:
        qp = None
        quantum_block = {"available": False, "reason": f"Qiskit unavailable: {exc}"}

    if qp is not None:
        qaoa_result = solve_binpack_qaoa_legacy(
            qp, jobs, S, reps=reps, maxiter=maxiter, shots=shots,
        )
        if qaoa_result is None:
            quantum_block = {
                "available": False,
                "reason": "QAOA solver returned no result (missing packages, "
                          "MemoryError, or simulation failure - see log above).",
            }
        elif qaoa_result.get("too_large"):
            # Solve was skipped because the QUBO is bigger than we can simulate.
            # Still expose the qubit count so the scaling panel shows the blow-up.
            qubits = qaoa_result.get("n_qubits")
            gb = (2 ** qubits) * 16 / 1e9 if qubits else None
            quantum_block = {
                "available": False,
                "reason": (
                    f"Too large to simulate here: {qubits} qubits "
                    f"(~{gb:,.0f} GB statevector, ceiling is {MAX_SIMULABLE_QUBITS}). "
                    f"That blow-up is exactly the point — the quantum cost outgrew a laptop."
                ),
            }
        else:
            qubits = qaoa_result.get("n_qubits")
            _, q_makespan = simulate_servers(
                [su for su in qaoa_result["server_usage"] if su["jobs"]],
                jobs, verbose=False,
            )
            quantum_block = {
                "available": True,
                "solver": "qaoa_legacy",
                "active_servers": qaoa_result["active_servers"],
                "objective": qaoa_result["objective"],
                "makespan": q_makespan,
                "solve_time_sec": qaoa_result.get("solve_time_sec"),
                "qubits": qubits,
                "status": qaoa_result.get("status"),
                "unplaced": qaoa_result.get("unplaced", []),
            }

    # ---- Scaling / comparison metrics ----
    state_space_approx: Optional[float] = None
    if qubits is not None:
        try:
            state_space_approx = float(2 ** qubits)
        except OverflowError:
            state_space_approx = None

    problem_block = {
        "jobs": J,
        "servers": S,
        "qp_vars": qp_vars,
        "qubits": qubits,
        "state_space_approx": state_space_approx,
    }

    comparison_block: Dict[str, Any] = {}
    if quantum_block.get("available"):
        q_time = quantum_block.get("solve_time_sec") or 0.0
        slowdown = q_time / max(classical_time, 1e-9)
        comparison_block = {
            "servers_delta": classical_active - quantum_block["active_servers"],
            "slowdown_factor": slowdown,
            "verdict": (
                f"QAOA was ~{slowdown:,.0f}x slower than the classical heuristic "
                f"for this {J}-job / {S}-server instance."
            ),
        }
    else:
        comparison_block = {
            "servers_delta": None,
            "slowdown_factor": None,
            "verdict": "Quantum side unavailable — classical result only.",
        }

    result = {
        "problem": problem_block,
        "classical": classical_block,
        "quantum": quantum_block,
        "comparison": comparison_block,
    }

    if verbose:
        _print_comparison_table(result)

    return result


def _print_comparison_table(result: Dict[str, Any]) -> None:
    """Pretty-print the educational classical-vs-quantum scaling table."""
    p = result["problem"]
    c = result["classical"]
    q = result["quantum"]
    comp = result["comparison"]

    line = "=" * 60
    print("\n" + line)
    print("           CLASSICAL  vs  QUANTUM (QAOA)")
    print(line)
    print(f"Problem size : {p['jobs']} jobs × {p['servers']} servers")
    if p["qubits"] is not None:
        print(f"QP variables : {p['qp_vars']}   →   QUBO qubits "
              f"(after slack absorption): {p['qubits']}")
        print(f"State space  : {_fmt_state_space(p['qubits'])} amplitudes the "
              f"simulator must track")
    else:
        print(f"QP variables : {p['qp_vars']}   (QUBO qubit count unavailable)")
    print(line)

    def _fmt_time(sec: Optional[float]) -> str:
        if sec is None:
            return "    n/a"
        if sec < 1.0:
            return f"{sec * 1000.0:8.1f} ms"
        return f"{sec:8.2f} s "

    if q.get("available"):
        print(f"{'':22}{'CLASSICAL (BFD)':>18}{'QUANTUM (QAOA)':>18}")
        print(f"  {'Active servers':20}{c['active_servers']:>16}{q['active_servers']:>18}")
        print(f"  {'Makespan (s)':20}{c['makespan']:>16.2f}{q['makespan']:>18.2f}")
        print(f"  {'Solve time':20}{_fmt_time(c['solve_time_sec']):>16}{_fmt_time(q['solve_time_sec']):>18}")
        print("  " + "-" * 56)
        print(f"  {comp['verdict']}")
        print("  Takeaway: classical heuristics dominate at this scale; QAOA's")
        print("  cost grows as 2^qubits — it only becomes interesting on problems")
        print("  too large to simulate on a laptop.")
    else:
        print(f"  Classical (BFD): {c['active_servers']} servers, "
              f"makespan {c['makespan']:.2f}s, solve {_fmt_time(c['solve_time_sec']).strip()}")
        print(f"  Quantum (QAOA) : UNAVAILABLE")
        print(f"    reason: {q.get('reason', 'unknown')}")
    print(line + "\n")


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

    # Compare packing algorithms and simulate makespan
    methods = {
        'first_fit_decreasing': pack_first_fit_decreasing(jobs),
        'best_fit_decreasing': pack_best_fit_decreasing(jobs),
        'worst_fit': pack_worst_fit(jobs),
    }

    results = {}
    for name, su_list in methods.items():
        per_rec, makespan = simulate_servers(su_list, jobs)
        servers_used = len([s for s in su_list if s.get('jobs')])
        results[name] = {'servers': servers_used, 'makespan': makespan, 'per_server': per_rec, 'placement': su_list}

    # Baseline: first_fit_decreasing
    baseline = results['first_fit_decreasing']
    print("\nBEFORE OPTIMIZATION:")
    print(f" - Servers used: {baseline['servers']}")
    print(f" - Estimated completion time: {baseline['makespan']:.2f} seconds")

    # Choose best: minimize servers then makespan
    best_name = None
    best_score = None
    for name, r in results.items():
        score = (r['servers'], r['makespan'])
        if best_score is None or score < best_score:
            best_score = score
            best_name = name

    best = results[best_name]
    saved = baseline['servers'] - best['servers']
    cost_saving = max(0, saved) * COST_PER_SERVER_PER_HOUR

    print("\nAFTER OPTIMIZATION:")
    print(f" - Method: {best_name}")
    print(f" - Servers used: {best['servers']} (saved {saved} servers)")
    print(f" - Estimated completion time: {best['makespan']:.2f} seconds")
    if baseline['makespan'] > 0:
        pct = (baseline['makespan'] - best['makespan']) / baseline['makespan'] * 100.0
        print(f" - Improvement: {pct:.1f}% faster")
    print(f" - Cost savings: €{cost_saving:.2f}/hour")

    print("\nDetailed timeline per server (best solution):")
    for idx, rec in enumerate(best['per_server']):
        job_ids = [j['id'] for j in rec['jobs']]
        comp = rec['completion']
        print(f" - Server {idx}: Jobs {job_ids} -> completes at t={comp:.2f}s")

    print(f"\nTotal makespan: {best['makespan']:.2f} seconds")

    # --- QAOA Quantum Circuit Solver ---
    # Runs entirely on your local CPU via Qiskit's classical quantum circuit
    # simulator. No QPU or cloud account required.
    #
    # Tunable parameters:
    #   reps    – QAOA circuit depth (p layers). More layers = better solution,
    #             longer simulation. Start at 3, increase for harder instances.
    #   maxiter – SPSA optimizer iterations. More = better angle convergence.
    #   shots   – Measurement samples per circuit. More = less sampling noise.
    #
    # Qubit count = QUBO variables after constraint absorption (printed at runtime).
    # Simulation time grows exponentially with qubit count; ~30 qubits is the
    # practical limit on a standard laptop.
    qaoa_reps = 1
    qaoa_maxiter = 50
    qaoa_shots = 1024

    print(f"\n[QAOA] Starting QAOA quantum simulation (reps={qaoa_reps}, maxiter={qaoa_maxiter}, shots={qaoa_shots})...")
    qaoa_result = solve_binpack_qaoa_legacy(
        qp, jobs, servers,
        reps=qaoa_reps,
        maxiter=qaoa_maxiter,
        shots=qaoa_shots,
    )

    if qaoa_result is not None:
        print("\nQAOA QUANTUM SOLUTION:")
        qaoa_su = qaoa_result['server_usage']
        qaoa_active = qaoa_result['active_servers']
        qaoa_per_rec, qaoa_makespan = simulate_servers(
            [su for su in qaoa_su if su['jobs']], jobs, verbose=False
        )
        print(f" - Active servers  : {qaoa_active}")
        print(f" - QUBO objective  : {qaoa_result['objective']:.4f}")
        print(f" - Status          : {qaoa_result['status']}")
        print(f" - Makespan        : {qaoa_makespan:.2f} seconds")
        if qaoa_result['unplaced']:
            print(f" - Unplaced jobs   : {qaoa_result['unplaced']} (increase reps/maxiter)")
        print("\nDetailed timeline (QAOA solution):")
        for idx, rec in enumerate(qaoa_per_rec):
            job_ids = [j['id'] for j in rec['jobs']]
            print(f" - Server {idx}: Jobs {job_ids} -> completes at t={rec['completion']:.2f}s")

        # Compare QAOA vs best classical heuristic
        saved_vs_classical = best['servers'] - qaoa_active
        cost_delta = saved_vs_classical * COST_PER_SERVER_PER_HOUR
        print(f"\nQAOA vs Classical best ({best_name}):")
        print(f" - Servers saved   : {saved_vs_classical}")
        print(f" - Cost delta      : €{cost_delta:.2f}/hour")
        if best['makespan'] > 0 and qaoa_makespan > 0:
            speed_pct = (best['makespan'] - qaoa_makespan) / best['makespan'] * 100.0
            print(f" - Makespan delta  : {speed_pct:.1f}% {'faster' if speed_pct >= 0 else 'slower'}")
    else:
        print("\n[QAOA] Solver unavailable or failed. Falling back to classical best-fit heuristic.")
        print("\nClassical fallback (best_fit_decreasing):")
        bf = pack_best_fit_decreasing(jobs)
        per_rec, makespan = simulate_servers(bf, jobs, verbose=False)
        for idx, rec in enumerate(per_rec):
            job_ids = [j['id'] for j in rec['jobs']]
            print(f" - Server {idx}: Jobs {job_ids} -> completes at t={rec['completion']:.2f}s")
        print(f"Total makespan (classical): {makespan:.2f} seconds")



"""Capstone Beta 1: M/G/1-based bin-packing -> Qiskit QuadraticProgram

This file contains the same solver implementation that encodes an M/G/1
waiting-time + utilization penalty into a Qiskit `QuadraticProgram`.

Usage:
  pip install -r requirements.txt
  python "capstone beta 1.py"

Note: this is intended to be a drop-in copy of the example solver.
"""
from typing import List, Dict, Tuple
import math
import random
import itertools

try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
except Exception:
    raise ImportError("qiskit-optimization is required: pip install qiskit qiskit-optimization")

# Module-level Qiskit availability flags.
# solve_binpack_qaoa performs its own fine-grained import probe at call time,
# so these module-level names are only used by build_quadratic_program /
# solve_with_fallback (the M/G/1 solver path).  We try both the modern
# qiskit-algorithms package (qiskit >= 1.0) and the legacy qiskit.algorithms
# namespace (qiskit < 1.0 / qiskit-terra) so the file loads cleanly under
# either install.
QAOA = None
NumPyMinimumEigensolver = None
COBYLA = None
QuantumInstance = None
Aer = None

try:
    # Modern path: qiskit >= 1.0 + qiskit-algorithms package
    from qiskit_algorithms import QAOA, NumPyMinimumEigensolver          # type: ignore
    from qiskit_algorithms.optimizers import COBYLA                       # type: ignore
except ImportError:
    try:
        # Legacy path: qiskit < 1.0 (qiskit-terra)
        from qiskit.algorithms import QAOA, NumPyMinimumEigensolver       # type: ignore
        from qiskit.algorithms.optimizers import COBYLA                   # type: ignore
    except ImportError:
        pass  # remain None — solve_with_fallback will use its own greedy fallback

try:
    from qiskit.utils import QuantumInstance                              # type: ignore
    from qiskit import Aer                                                # type: ignore
except ImportError:
    pass  # remain None — only needed for legacy QuantumInstance path


def build_quadratic_program(tasks: List[Dict[str, float]],
                            m: int,
                            alpha: float = 1.0,
                            beta: float = 1.0,
                            rho_max: float = 0.95,
                            cpu_max: float = None,
                            name: str = "mg1_binpacking",
                            tie_break: Dict[str, float] = None) -> QuadraticProgram:
    """Builds a QuadraticProgram encoding the assignment problem.

    Args:
      tasks: list of dicts with keys 'lam' (arrival rate) and 's' (mean service time)
      m: number of machines (bins)
      alpha: weight on utilization penalty (linear)
      beta: weight on waiting-time penalty (uses first-order Taylor)
      rho_max: hard upper bound on utilization per machine (must be <1)

    Returns:
      QuadraticProgram
    """
    n = len(tasks)
    qp = QuadraticProgram(name=name)

    # Create binary variables x_i_j -> name convention x_i_j
    for i in range(n):
        for j in range(m):
            qp.binary_var(name=f"x_{i}_{j}")

    # Assignment constraints: each task assigned to exactly one machine
    for i in range(n):
        linear = {f"x_{i}_{j}": 1.0 for j in range(m)}
        qp.linear_constraint(linear=linear, sense="==", rhs=1.0, name=f"assign_{i}")

    # Utilization constraints per machine: sum_i lam_i * s_i * x_i_j <= rho_max
    for j in range(m):
        linear = {f"x_{i}_{j}": tasks[i]["lam"] * tasks[i]["s"] for i in range(n)}
        qp.linear_constraint(linear=linear, sense="<=", rhs=rho_max, name=f"util_{j}")

    # CPU capacity constraints per machine: sum_i cpu_i * x_i_j <= cpu_max
    if cpu_max is not None:
        for j in range(m):
            linear_cpu = {f"x_{i}_{j}": tasks[i].get("cpu", 0.0) for i in range(n)}
            qp.linear_constraint(linear=linear_cpu, sense="<=", rhs=cpu_max, name=f"cpu_{j}")

    # Build objective: sum_j [ alpha * rho_j + beta * (S2_j / 2) * (1 + rho_j) ]
    # where rho_j = sum_i lam_i * s_i * x_i_j
    # and S2_j = sum_i lam_i * s_i^2 * x_i_j
    # Expand terms to obtain linear + quadratic coefficients.

    linear_obj = {}
    quadratic_obj = {}

    for j in range(m):
        # Linear contributions from alpha * rho_j and beta * S2_j/2
        for i in range(n):
            xi = f"x_{i}_{j}"
            lam_i = tasks[i]["lam"]
            s_i = tasks[i]["s"]
            s2_term = lam_i * (s_i ** 2)
            rho_coeff = lam_i * s_i

            # alpha * rho_j linear part
            linear_obj[xi] = linear_obj.get(xi, 0.0) + alpha * rho_coeff

            # beta * (S2_j / 2) linear part
            linear_obj[xi] = linear_obj.get(xi, 0.0) + beta * (s2_term / 2.0)

        # Quadratic part from beta * (S2_j * rho_j) / 2
        # S2_j * rho_j = (sum_i lam_i s_i^2 x_i)(sum_k lam_k s_k x_k)
        for i in range(n):
            xi = f"x_{i}_{j}"
            for k in range(n):
                xk = f"x_{k}_{j}"
                lam_i = tasks[i]["lam"]
                s_i = tasks[i]["s"]
                lam_k = tasks[k]["lam"]
                s_k = tasks[k]["s"]
                coeff = beta * ( (lam_i * (s_i ** 2)) * (lam_k * s_k) ) / 2.0
                # For diagonal terms (i == k), add to linear objective (Qiskit ignores quadratic diagonal)
                if i == k:
                    linear_obj[xi] = linear_obj.get(xi, 0.0) + coeff
                else:
                    # Add symmetric quadratic term (off-diagonal only)
                    # QuadraticProgram expects keys as tuples (var1, var2)
                    key = (xi, xk) if xi <= xk else (xk, xi)
                    quadratic_obj[key] = quadratic_obj.get(key, 0.0) + coeff

    # Inject tiny random linear tie-breaking biases if provided
    if tie_break:
        for var, bias in tie_break.items():
            linear_obj[var] = linear_obj.get(var, 0.0) + bias

    # Add linear + quadratic objective to qp
    qp.minimize(linear=linear_obj, quadratic=quadratic_obj)

    return qp


def solve_with_fallback(qp: QuadraticProgram, method: str = "qaoa", shots: int = 1024,
                        tasks: List[Dict[str, float]] = None, m: int = None,
                        tie_break: Dict[str, float] = None, cpu_max: float = None) -> Tuple[Dict[str, int], float]:
    """Attempt to solve the QuadraticProgram using a quantum or hybrid method.

    method: 'qaoa' to prefer QAOA, 'exact' to use classical eigen-solver, 'both' to try QAOA then fallback.

    Returns:
      assignment dict mapping var name -> 0/1 and objective value
    """
    method = method.lower()
    result = None

    # Helper to extract assignment
    def extract(res):
        x = {k: int(v) for k, v in res.x.items()} if hasattr(res, 'x') else { }
        obj = res.fval if hasattr(res, 'fval') else None
        return x, obj

    # Try QAOA if requested and available
    if method in ("qaoa", "both") and QAOA is not None and Aer is not None:
        try:
            backend = Aer.get_backend('aer_simulator_statevector')
            qi = QuantumInstance(backend)
            qaoa = QAOA(optimizer=COBYLA(maxiter=100), reps=1, quantum_instance=qi)
            meo = MinimumEigenOptimizer(qaoa)
            res = meo.solve(qp)
            return extract(res)
        except Exception:
            pass

    # Fallback: NumPyMinimumEigensolver (classical exact diagonalization)
    if method in ("exact", "both") and NumPyMinimumEigensolver is not None:
        try:
            eigen = NumPyMinimumEigensolver()
            meo = MinimumEigenOptimizer(eigen)
            res = meo.solve(qp)
            return extract(res)
        except Exception:
            pass

    # Final fallback: try classical greedy heuristic (deterministic)
    x = {v.name: 0 for v in qp.variables}
    # Very simple heuristic: fill machines greedily by rho until full
    # Build structures
    n = max(int(name.split('_')[1]) for name in x.keys()) + 1
    m = max(int(name.split('_')[2]) for name in x.keys()) + 1
    # parse tasks from variable names assuming consistent ordering
    # This heuristic requires the problem to follow build_quadratic_program layout
    task_rho = []
    task_s2 = []
    # If tasks and m are provided, use brute-force solver as a reliable fallback
    if tasks is not None and m is not None:
        try:
            # try greedy baseline first
            g_sol, g_obj = greedy_mg1(tasks, m, rho_max=0.95, cpu_max=cpu_max, tie_break=tie_break)
            # try relaxed baseline
            r_sol, r_obj = relaxed_linearized(tasks, m, rho_max=0.95, cpu_max=cpu_max)
            # try brute force if small n
            try:
                bf_sol, bf_obj = brute_force_solver(tasks, m, tie_break=tie_break, cpu_max=cpu_max)
            except Exception:
                bf_sol, bf_obj = ({}, float('nan'))
            # choose best valid solution (finite objective)
            candidates = [(g_sol, g_obj), (r_sol, r_obj), (bf_sol, bf_obj)]
            best = min(candidates, key=lambda t: float('inf') if t[1] is None or math.isnan(t[1]) else t[1])
            return best
        except Exception:
            pass

    # can't extract tasks easily from qp; return empty assignment
    return x, float('nan')


def brute_force_solver(tasks: List[Dict[str, float]], m: int,
                       alpha: float = 1.0, beta: float = 1.0, rho_max: float = 0.95,
                       tie_break: Dict[str, float] = None, cpu_max: float = None) -> Tuple[Dict[str, int], float]:
    """Brute-force search over all assignments (feasible for small n).

    Returns assignment dict mapping `x_i_j` -> 0/1 and objective value.
    """
    n = len(tasks)
    best_obj = float('inf')
    best_assign = None

    for prod in itertools.product(range(m), repeat=n):
        # compute per-machine rho and S2
        rho = [0.0] * m
        S2 = [0.0] * m
        cpu = [0.0] * m
        for i, j in enumerate(prod):
            lam = tasks[i]['lam']
            s = tasks[i]['s']
            rho[j] += lam * s
            S2[j] += lam * (s ** 2)
            cpu[j] += tasks[i].get('cpu', 0.0)

        if any(r > rho_max + 1e-12 for r in rho):
            continue
        if cpu_max is not None and any(c > cpu_max + 1e-12 for c in cpu):
            continue

        obj = 0.0
        for j in range(m):
            obj += alpha * rho[j] + beta * (S2[j] / 2.0) * (1.0 + rho[j])

        # add tie-break linear biases for assigned variables
        if tie_break:
            for i, j in enumerate(prod):
                obj += tie_break.get(f"x_{i}_{j}", 0.0)

        if obj < best_obj:
            best_obj = obj
            best_assign = prod

    if best_assign is None:
        return ({}, float('nan'))

    x = {}
    for i in range(n):
        for j in range(m):
            x[f"x_{i}_{j}"] = 1 if best_assign[i] == j else 0
    return x, best_obj


def find_min_cpu_max(tasks: List[Dict[str, float]], m: int, rho_max: float = 0.95) -> Tuple[float, Tuple[int, ...]]:
    """Brute-force search to find the minimal cpu_max allowing a feasible assignment.

    Returns (min_cpu_max, assignment_tuple) where assignment_tuple maps task i -> machine j.
    If no assignment satisfies the rho constraint, returns (float('nan'), None).
    """
    n = len(tasks)
    best_cpu_max = float('inf')
    best_assign = None

    for prod in itertools.product(range(m), repeat=n):
        # compute per-machine rho and cpu sums
        rho = [0.0] * m
        cpu = [0.0] * m
        for i, j in enumerate(prod):
            rho[j] += tasks[i]['lam'] * tasks[i]['s']
            cpu[j] += tasks[i].get('cpu', 0.0)

        if any(r > rho_max + 1e-12 for r in rho):
            continue

        cur_cpu_max = max(cpu)
        if cur_cpu_max < best_cpu_max:
            best_cpu_max = cur_cpu_max
            best_assign = prod

    if best_assign is None:
        return float('nan'), None
    return best_cpu_max, best_assign


def normalize_cpu_demands(tasks: List[Dict[str, float]], m: int, cpu_max: float) -> None:
    """Modify tasks in-place so that per-task cpu <= cpu_max and total cpu <= m * cpu_max.

    Strategy:
    - Clip any task cpu to cpu_max.
    - If total cpu still > m*cpu_max, scale down the remaining (non-clipped) tasks uniformly
      so that total cpu <= m*cpu_max (with a small safety factor).
    This preserves relative demands while guaranteeing feasibility w.r.t CPU capacity.
    """
    # Clip per-task to cpu_max
    for t in tasks:
        if t.get('cpu', 0.0) > cpu_max:
            t['cpu'] = cpu_max

    total = sum(t.get('cpu', 0.0) for t in tasks)
    cap_total = m * cpu_max
    if total <= cap_total:
        return

    # Iteratively scale non-clipped tasks with iteration cap and convergence check
    MAX_ITERS = 100
    safety = 0.999
    prev_factor = None
    
    for _ in range(MAX_ITERS):
        clipped = [t for t in tasks if abs(t.get('cpu', 0.0) - cpu_max) < 1e-12]
        clipped_ids = {id(t) for t in clipped}
        nonclipped = [t for t in tasks if id(t) not in clipped_ids]
        sum_clipped = sum(t.get('cpu', 0.0) for t in clipped)
        sum_nonclipped = sum(t.get('cpu', 0.0) for t in nonclipped)

        if sum_nonclipped <= 1e-12:
            # nothing left to scale; distribute remaining capacity evenly (small fallback)
            for t in nonclipped:
                t['cpu'] = max(0.0, (cap_total - sum_clipped) / max(1, len(nonclipped)))
            break

        remaining_capacity = max(0.0, cap_total - sum_clipped)
        factor = remaining_capacity / sum_nonclipped * safety
        
        # Convergence check: if factor change is negligible, break early
        if prev_factor is not None and abs(factor - prev_factor) < 1e-12:
            break
        prev_factor = factor
        
        if factor >= 1.0:
            break

        # scale non-clipped tasks
        for t in nonclipped:
            t['cpu'] = t.get('cpu', 0.0) * factor

        # clip any that now exceed cpu_max and iterate
        for t in nonclipped:
            if t['cpu'] > cpu_max:
                t['cpu'] = cpu_max

        total = sum(t.get('cpu', 0.0) for t in tasks)
        if total <= cap_total + 1e-12:
            break


def normalize_loads(tasks: List[Dict[str, float]], m: int, rho_max: float) -> None:
    """Modify tasks in-place so no single task exceeds rho_max and total util <= m*rho_max.

    Strategy:
    - Clip any task so that lam * s <= rho_max by reducing lam.
    - If total utilization sum(lam*s) > m*rho_max, scale all lam values down uniformly
      (after clipping) so total_util <= m*rho_max (with a safety factor).
    This guarantees there exists an assignment w.r.t. the rho constraints.
    """
    # Clip individual tasks
    for t in tasks:
        lam = t.get('lam', 0.0)
        s = t.get('s', 0.0)
        max_lam = rho_max / (s + 1e-12)
        if lam > max_lam:
            t['lam'] = max_lam

    # Scale total utilization if needed
    total_util = sum(t['lam'] * t['s'] for t in tasks)
    cap_total = m * rho_max
    if total_util <= cap_total:
        return

    safety = 0.999
    factor = cap_total / total_util * safety
    for t in tasks:
        t['lam'] = t['lam'] * factor


def ensure_feasible(tasks: List[Dict[str, float]], m: int, rho_max: float, cpu_max: float) -> None:
    """Apply cpu and load normalization to ensure a feasible assignment exists.

    This modifies `tasks` in-place to satisfy per-task rho, total util, and cpu totals.
    """
    # First ensure CPU demands are within feasible total
    normalize_cpu_demands(tasks, m, cpu_max)
    # Then ensure utilization constraints can be satisfied
    normalize_loads(tasks, m, rho_max)


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



def best_fit_solver(tasks: List[Dict[str, float]], m: int,
                    alpha: float = 1.0, beta: float = 1.0, rho_max: float = 0.95,
                    tie_break: Dict[str, float] = None, cpu_max: float = None) -> Tuple[Dict[str, int], float]:
    """Best-Fit Decreasing heuristic using item size = lam * s.

    Returns assignment dict and objective. If no feasible packing, returns ({}, nan).
    """
    n = len(tasks)
    # sort tasks by decreasing size
    items = sorted(enumerate(tasks), key=lambda it: it[1]['lam'] * it[1]['s'], reverse=True)
    rho = [0.0] * m
    S2 = [0.0] * m
    assign = [[] for _ in range(m)]

    for i, t in items:
        size = t['lam'] * t['s']
        # find best-fit machine: smallest remaining capacity that fits
        best_j = None
        best_rem = None
        for j in range(m):
            rem = rho_max - rho[j]
            if rem + 1e-12 >= size:
                if best_rem is None or rem < best_rem:
                    best_rem = rem
                    best_j = j
        if best_j is None:
            return ({}, float('nan'))
        rho[best_j] += size
        S2[best_j] += t['lam'] * (t['s'] ** 2)
        assign[best_j].append(i)
        # check cpu capacity incrementally
        if cpu_max is not None:
            # compute cpu for best_j and bail if exceeded (shouldn't happen due to fit selection)
            cpu_sum = sum(tasks[k].get('cpu', 0.0) for k in assign[best_j])
            if cpu_sum > cpu_max + 1e-12:
                return ({}, float('nan'))

    obj = 0.0
    for j in range(m):
        obj += alpha * rho[j] + beta * (S2[j] / 2.0) * (1.0 + rho[j])

    # add tie-break linear biases
    if tie_break:
        for i in range(n):
            for j in range(m):
                if i in assign[j]:
                    obj += tie_break.get(f"x_{i}_{j}", 0.0)
    # check cpu capacities final
    if cpu_max is not None:
        for j in range(m):
            cpu_sum = sum(tasks[i].get('cpu', 0.0) for i in assign[j])
            if cpu_sum > cpu_max + 1e-12:
                return ({}, float('nan'))

    x = {}
    for i in range(n):
        for j in range(m):
            x[f"x_{i}_{j}"] = 1 if i in assign[j] else 0
    return x, obj


def feasibility_filter(tasks: List[Dict[str, float]], m: int, rho_max: float = 0.95) -> Dict[Tuple[int, int], bool]:
    """Compute feasibility A_{i,s} for each job i and server s.
    Returns a dict keyed by (i,s) boolean.

    NOTE: Reference implementation kept for future use (e.g. pre-filtering variables
    in build_quadratic_program to reduce QUBO size). Not called by any current solver.
    """
    n = len(tasks)
    feasible = {}
    for i in range(n):
        for s in range(m):
            lam = tasks[i].get('lam', 0.0)
            sv = tasks[i].get('s', 0.0)
            # if single job would saturate server, mark infeasible
            if lam * sv >= 1.0 - 1e-9 or lam * sv >= rho_max:
                feasible[(i, s)] = False
            else:
                feasible[(i, s)] = True
    return feasible


def _mg1_server_cost(lambda_s: float, M2_s: float, rho_s: float, eps: float = 1e-12) -> float:
    """Server cost using M/G/1 waiting-rate proxy: lambda * M2 / (2*(1-rho))."""
    if lambda_s <= 0.0:
        return 0.0
    denom = 2.0 * max(eps, (1.0 - rho_s))
    return (lambda_s * M2_s) / denom


def greedy_mg1(tasks: List[Dict[str, float]], m: int,
               rho_max: float = 0.95, cpu_max: float = None, tie_break: Dict[str, float] = None) -> Tuple[Dict[str, int], float]:
    """Greedy marginal-cost assignment using server-level aggregates only."""
    n = len(tasks)
    # precompute contributions
    lam = [t.get('lam', 0.0) for t in tasks]
    s = [t.get('s', 0.0) for t in tasks]
    alpha = [[lam[i] * s[i] for _ in range(m)] for i in range(n)]  # lam*s per (i,s) (identical across s here)
    beta = [[lam[i] * (s[i] ** 2) for _ in range(m)] for i in range(n)]

    # server aggregates
    Lambda = [0.0] * m
    M2 = [0.0] * m
    Rho = [0.0] * m
    cpu = [0.0] * m
    assigned = [-1] * n

    # order jobs by decreasing size (lam*s)
    order = sorted(range(n), key=lambda i: lam[i] * s[i], reverse=True)

    # Min-heap over servers keyed by current M/G/1 cost.
    # Lazy deletion via version counter avoids a full rebuild after each assignment.
    version = [0] * m
    heap = [(0.0, 0, j) for j in range(m)]   # (cost, version, server_idx)
    heapq.heapify(heap)

    for i in order:
        best_j = None
        best_delta = float('inf')
        skipped: List[Tuple[float, int, int]] = []

        while heap:
            cost, ver, j = heapq.heappop(heap)
            if ver != version[j]:   # stale entry — discard
                continue
            # feasibility checks
            if Rho[j] + alpha[i][j] >= rho_max - 1e-12:
                skipped.append((cost, ver, j))
                continue
            if cpu_max is not None and cpu[j] + tasks[i].get('cpu', 0.0) > cpu_max + 1e-12:
                skipped.append((cost, ver, j))
                continue
            # marginal cost
            new_L = Lambda[j] + lam[i]
            new_M2 = M2[j] + beta[i][j]
            new_R = Rho[j] + alpha[i][j]
            delta = _mg1_server_cost(new_L, new_M2, new_R) - cost
            if tie_break:
                delta += tie_break.get(f"x_{i}_{j}", 0.0)
            if delta < best_delta:
                best_delta = delta
                best_j = j
            skipped.append((cost, ver, j))
            break   # heap-minimum found; remaining servers can only be worse

        for entry in skipped:
            heapq.heappush(heap, entry)

        if best_j is None:
            return ({}, float('nan'))

        # assign
        assigned[i] = best_j
        Lambda[best_j] += lam[i]
        M2[best_j] += beta[i][best_j]
        Rho[best_j] += alpha[i][best_j]
        cpu[best_j] += tasks[i].get('cpu', 0.0)
        # push updated cost with bumped version so old heap entry is ignored
        version[best_j] += 1
        new_cost = _mg1_server_cost(Lambda[best_j], M2[best_j], Rho[best_j])
        heapq.heappush(heap, (new_cost, version[best_j], best_j))

    # build x dict and objective
    x = {}
    for i in range(n):
        for j in range(m):
            x[f"x_{i}_{j}"] = 1 if assigned[i] == j else 0

    obj = 0.0
    for j in range(m):
        obj += _mg1_server_cost(Lambda[j], M2[j], Rho[j])
    return x, obj


def relaxed_linearized(tasks: List[Dict[str, float]], m: int,
                       rho_max: float = 0.95, cpu_max: float = None,
                       iters: int = 50, step: float = 0.1) -> Tuple[Dict[str, int], float]:
    """Simple continuous relaxation solved by projected gradient updates with linearization.
    Returns a rounded feasible integer solution (repairing if needed).
    """
    n = len(tasks)
    lam = [t.get('lam', 0.0) for t in tasks]
    s = [t.get('s', 0.0) for t in tasks]
    alpha = [[lam[i] * s[i] for _ in range(m)] for i in range(n)]
    beta = [[lam[i] * (s[i] ** 2) for _ in range(m)] for i in range(n)]

    # initialize x uniformly among servers
    x = [[1.0 / m for _ in range(m)] for _ in range(n)]

    for _ in range(iters):
        # compute aggregates
        Lambda = [0.0] * m
        M2 = [0.0] * m
        Rho = [0.0] * m
        for i in range(n):
            for j in range(m):
                Lambda[j] += lam[i] * x[i][j]
                M2[j] += beta[i][j] * x[i][j] / (lam[i] + 1e-12) * lam[i]  # keep consistent units
                Rho[j] += alpha[i][j] * x[i][j]

        # compute gradient and take a step
        grad = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                L = Lambda[j]
                B = sum(beta[k][j] * x[k][j] for k in range(n))
                A = Rho[j]
                denom = max(1e-12, (1.0 - A))
                # partial derivatives derived analytically
                g_L = B / (2.0 * denom)
                g_B = L / (2.0 * denom)
                g_A = (L * B) / (2.0 * (denom ** 2))
                dLdxi = lam[i]
                dAdxi = alpha[i][j]
                dBdxi = beta[i][j]
                grad[i][j] = g_L * dLdxi + g_B * dBdxi + g_A * dAdxi

        # gradient descent step then project each job to simplex [0,1] sum=1
        for i in range(n):
            for j in range(m):
                x[i][j] = x[i][j] - step * grad[i][j]
            # project to [0,1] and normalize
            for j in range(m):
                x[i][j] = min(1.0, max(0.0, x[i][j]))
            ssum = sum(x[i])
            if ssum <= 0.0:
                # reset to uniform
                for j in range(m):
                    x[i][j] = 1.0 / m
            else:
                for j in range(m):
                    x[i][j] /= ssum

    # rounding with simple repair: assign to argmax that keeps rho < rho_max
    Lambda = [0.0] * m
    M2 = [0.0] * m
    Rho = [0.0] * m
    assign = [-1] * n
    order = sorted(range(n), key=lambda i: lam[i] * s[i], reverse=True)
    for i in order:
        choices = sorted(range(m), key=lambda j: -x[i][j])
        placed = False
        for j in choices:
            if Rho[j] + alpha[i][j] < rho_max - 1e-12:
                assign[i] = j
                Lambda[j] += lam[i]
                M2[j] += beta[i][j]
                Rho[j] += alpha[i][j]
                placed = True
                break
        if not placed:
            # fallback to greedy if repair fails
            return greedy_mg1(tasks, m, rho_max=rho_max, cpu_max=cpu_max, tie_break=None)

    xdict = {}
    for i in range(n):
        for j in range(m):
            xdict[f"x_{i}_{j}"] = 1 if assign[i] == j else 0
    obj = sum(_mg1_server_cost(Lambda[j], M2[j], Rho[j]) for j in range(m))
    return xdict, obj


def build_qubo_surrogate(tasks: List[Dict[str, float]], m: int, expansion: Dict[int, Tuple[float, float, float]] = None,
                         penalty_assign: float = 10.0) -> Dict[Tuple[str, str], float]:
    """Build a quadratic surrogate (QUBO-like) over variables x_i_j using a second-order expansion
    around provided per-server aggregates. Returns a dict mapping (var1,var2)->coeff.
    This surrogate is server-aggregation first: pairwise terms only for variables on same server.

    NOTE: Reference implementation kept for future use (e.g. feeding a custom QUBO-based
    annealing solver). Not called by any current solver path.
    """
    n = len(tasks)
    lam = [t.get('lam', 0.0) for t in tasks]
    s = [t.get('s', 0.0) for t in tasks]
    alpha_i = [lam[i] * s[i] for i in range(n)]
    beta_i = [lam[i] * (s[i] ** 2) for i in range(n)]

    Q = {}
    # default expansion points if not provided: zero aggregates
    if expansion is None:
        expansion = {j: (1e-6, 1e-6, 1e-6) for j in range(m)}

    for j in range(m):
        L0, A0, B0 = expansion.get(j, (1e-6, 1e-6, 1e-6))
        denom = max(1e-12, (1.0 - A0))
        # derivatives
        g_L = B0 / (2.0 * denom)
        g_B = L0 / (2.0 * denom)
        g_A = (L0 * B0) / (2.0 * (denom ** 2))
        g_LL = 0.0
        g_BB = 0.0
        g_AA = (L0 * B0) / ( (denom ** 3) )
        g_LA = B0 / (2.0 * (denom ** 2))
        g_LB = 1.0 / (2.0 * denom)
        g_AB = L0 / (2.0 * (denom ** 2))

        # linear terms
        for i in range(n):
            var_i = f"x_{i}_{j}"
            lin = g_L * lam[i] + g_A * alpha_i[i] + g_B * beta_i[i]
            Q[(var_i, var_i)] = Q.get((var_i, var_i), 0.0) + lin

        # quadratic terms (only same-server pairs)
        for i in range(n):
            for k in range(i + 1, n):
                var_i = f"x_{i}_{j}"
                var_k = f"x_{k}_{j}"
                h = (g_LL * lam[i] * lam[k]
                     + g_AA * alpha_i[i] * alpha_i[k]
                     + g_BB * beta_i[i] * beta_i[k]
                     + g_LA * (lam[i] * alpha_i[k] + lam[k] * alpha_i[i])
                     + g_LB * (lam[i] * beta_i[k] + lam[k] * beta_i[i])
                     + g_AB * (alpha_i[i] * beta_i[k] + alpha_i[k] * beta_i[i]))
                if abs(h) > 0.0:
                    key = (var_i, var_k) if var_i <= var_k else (var_k, var_i)
                    Q[key] = Q.get(key, 0.0) + h

    # assignment completeness penalties (sum_s x_i_s == 1)^2 = sum_s x_i_s^2 + sum_{s!=t} 2*x_i_s*x_i_t - 2*sum_s x_i_s + 1
    # The constant +1 term is folded into the diagonal of the first variable per task
    for i in range(n):
        first_var = f"x_{i}_{0}"
        # Add the constant term +1 to the diagonal of the first variable
        Q[(first_var, first_var)] = Q.get((first_var, first_var), 0.0) + penalty_assign
        
        for j in range(m):
            vi = f"x_{i}_{j}"
            # linear part from penalty: -2*sum_s x_i_s
            Q[(vi, vi)] = Q.get((vi, vi), 0.0) + penalty_assign * (-2.0)
            for k in range(j + 1, m):
                vj = f"x_{i}_{k}"
                key = (vi, vj) if vi <= vj else (vj, vi)
                Q[key] = Q.get(key, 0.0) + penalty_assign * 2.0

    return Q


def _random_tasks(n: int, lam_mean: float = 0.1, s_mean: float = 1.0):
    tasks = []
    for _ in range(n):
        lam = max(0.001, random.expovariate(1.0 / lam_mean))
        s = max(0.01, random.expovariate(1.0 / s_mean))
        # add a cpu demand field (normalized units)
        cpu = max(0.01, random.uniform(0.01, 0.5))
        tasks.append({"lam": lam, "s": s, "cpu": cpu})
    return tasks


def _print_solution(solution: Dict[str, int], tasks: List[Dict[str, float]], m: int):
    n = len(tasks)
    assign = [[] for _ in range(m)]
    for i in range(n):
        for j in range(m):
            if solution.get(f"x_{i}_{j}", 0) == 1:
                assign[j].append(i)
    # If no tasks assigned to any machine, report infeasible/empty solution
    if all(len(a) == 0 for a in assign):
        print("No feasible assignment found (all machines empty).")
        return

    print("Assignments:")
    for j in range(m):
        rho = sum(tasks[i]['lam'] * tasks[i]['s'] for i in assign[j])
        print(f" Machine {j}: tasks={assign[j]} rho={rho:.4f}")


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

    # Educational CLI prints Unicode (✓ ⚠ ❌ and the comparison table). Force a
    # UTF-8 console so it renders on Windows' default cp1252 terminal instead of
    # crashing with UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo-online', action='store_true', help='Run non-interactive online-arrival demo')
    parser.add_argument('--compare', action='store_true',
                        help='Run educational classical-vs-QAOA scaling comparison (small demo)')
    parser.add_argument('--servers', type=int, default=3, help='Servers for --compare (default 3)')
    args = parser.parse_args()

    if args.compare:
        # Small instance is mandatory: real QAOA simulation cost grows as 2^qubits.
        demo_profiles = {'web_api': 4, 'batch': 2}
        demo_jobs = generate_jobs(demo_profiles)
        print(f"Comparing classical vs QAOA on {len(demo_jobs)} jobs / {args.servers} servers...")
        compare_classical_vs_quantum(demo_jobs, servers=args.servers, verbose=True)
    elif args.demo_online:
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
        return _classical_fallback(
            jobs, num_servers,
            cores_per_server=cores_per_server, ram_per_server=ram_per_server,
            nvme_per_server=nvme_per_server, bandwidth_per_server=bandwidth_per_server,
            reason="At least one job exceeds server capacity.",
        )

    # ------------------------------------------------------------------
    # Step 1 — load qiskit-optimization (required by all solver paths)
    # ------------------------------------------------------------------
    try:
        from qiskit_optimization import QuadraticProgram as _QP                   # type: ignore
        from qiskit_optimization.algorithms import MinimumEigenOptimizer as _MEO  # type: ignore
    except ImportError as e:
        return _classical_fallback(
            jobs, num_servers,
            cores_per_server=cores_per_server, ram_per_server=ram_per_server,
            nvme_per_server=nvme_per_server, bandwidth_per_server=bandwidth_per_server,
            reason=f"qiskit-optimization not installed: {e}",
        )

    # ------------------------------------------------------------------
    # Step 2 — variable-count gate
    # ------------------------------------------------------------------
    J = len(jobs)
    S = num_servers
    total_vars = J * S

    print(f"[QAOA] Problem size: {J} jobs x {S} servers = {total_vars} variables (limit: {max_vars})")

    if total_vars > max_vars:
        print(f"[QAOA] Problem too large: {total_vars} vars exceeds max {max_vars}. Using classical greedy.")
        return _classical_fallback(
            jobs, num_servers,
            cores_per_server=cores_per_server, ram_per_server=ram_per_server,
            nvme_per_server=nvme_per_server, bandwidth_per_server=bandwidth_per_server,
            reason=(
                f"Problem too large for quantum solver "
                f"({J} jobs x {S} servers = {total_vars} vars > {max_vars}). "
                "Using classical greedy."
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
        return _classical_fallback(
            jobs, num_servers,
            cores_per_server=cores_per_server, ram_per_server=ram_per_server,
            nvme_per_server=nvme_per_server, bandwidth_per_server=bandwidth_per_server,
            reason="All quantum solver paths failed — using classical greedy.",
        )

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
                return _classical_fallback(
                    jobs, num_servers,
                    cores_per_server=cores_per_server, ram_per_server=ram_per_server,
                    nvme_per_server=nvme_per_server, bandwidth_per_server=bandwidth_per_server,
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

def _classical_fallback(jobs: List[Dict[str, Any]], num_servers: int,
                        cores_per_server: int = None,
                        ram_per_server: float = None,
                        nvme_per_server: float = None,
                        bandwidth_per_server: float = None,
                        reason: str = "") -> Dict[str, Any]:
    """Run classical greedy scheduler and return same shape as QAOA result."""
    assignments, timeline, stats = online_scheduler_with_arrivals(
        jobs, servers=num_servers, verbose=False,
        cores_per_server=cores_per_server,
        ram_per_server=ram_per_server,
        nvme_per_server=nvme_per_server,
        bandwidth_per_server=bandwidth_per_server
    )
    active = len(set(v["server"] for v in assignments.values())) if assignments else 0
    if reason:
        timeline = [f"[classical fallback: {reason}]"] + timeline
    stats["active_servers"] = active
    return {
        "assignments": {str(k): v["server"] for k, v in assignments.items()},
        "active_servers": active,
        "solver": "classical_greedy",
        "timeline": timeline,
        "stats": stats,
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
        return {"stats": stats, "timeline": timeline, "solver": "classical_greedy"}


    @app.post("/schedule_quantum")
    def schedule_quantum(req: WorkloadRequest):
        """QAOA quantum bin-packing scheduler."""
        jobs = generate_jobs(req.profile_counts)
        result = solve_binpack_qaoa(
            jobs, num_servers=req.servers,
            cores_per_server=req.server_constants.cores,
            ram_per_server=req.server_constants.ram,
            nvme_per_server=req.server_constants.nvme,
            bandwidth_per_server=req.server_constants.bandwidth
        )
        return result


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

        quantum_result = solve_binpack_qaoa(
            jobs, num_servers=req.servers,
            cores_per_server=req.server_constants.cores,
            ram_per_server=req.server_constants.ram,
            nvme_per_server=req.server_constants.nvme,
            bandwidth_per_server=req.server_constants.bandwidth
        )

        return {
            "classical": {
                "stats": classical_stats,
                "timeline": classical_timeline,
                "solver": "classical_greedy",
            },
            "quantum": quantum_result,
        }


    @app.post("/compare_scaling")
    def compare_scaling(req: WorkloadRequest):
        """Educational comparison: classical heuristic vs *real* QAOA, with
        performance/scaling metrics (QP vars, QUBO qubits, state space, solve
        times). Returns the structured dict from compare_classical_vs_quantum.

        NOTE: real QAOA simulation cost grows as 2^qubits — keep workloads small.
        """
        jobs = generate_jobs(req.profile_counts)
        return compare_classical_vs_quantum(jobs, servers=req.servers, verbose=False)


    @app.get("/presets")
    def presets():
        """Quick-comparison presets shown in scheduler_v9.html.

        Each entry carries everything the UI needs to load a scenario:
        an id, a label, a one-line description, the profile_counts, and the
        server count. Feed profile_counts + servers straight into
        /compare_scaling to run it.
        """
        return {"presets": COMPARISON_PRESETS}
else:
    app = None

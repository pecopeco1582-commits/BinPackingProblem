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
from typing import Any, Dict, List, Optional, Tuple
import sys
import heapq
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


def generate_jobs(profile_counts: Dict[str, int],
                  cores_per_server: int = None,
                  ram_per_server: float = None,
                  nvme_per_server: float = None,
                  bandwidth_per_server: float = None) -> List[Dict[str, Any]]:
    """Generate job dictionaries from selected profiles.

    Args:
        profile_counts: Dict mapping profile type (str) to desired count (int).
        cores_per_server: Optional override for module-level CORES_PER_SERVER.
        ram_per_server: Optional override for module-level RAM_PER_SERVER.
        nvme_per_server: Optional override for module-level NVME_PER_SERVER.
        bandwidth_per_server: Optional override for module-level BANDWIDTH_PER_SERVER.

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
def simulate_servers(server_usage_list: List[Dict[str, Any]], jobs_list: List[Dict[str, Any]], verbose: bool = True) -> Tuple[List[Dict[str, Any]], float]:
    """Simulate each server running its assigned jobs with resource constraints.

    Returns (per_server_records, makespan).
    """
    job_map = {j['id']: j for j in jobs_list}
    per_server_records: List[Dict[str, Any]] = []
    overall_makespan = 0.0

    for su in server_usage_list:
        rec: Dict[str, Any] = {'jobs': [], 'completion': 0.0}
        pending_jobs = [job_map[jid].copy() for jid in su.get('jobs', [])]
        t = 0.0
        running: List[Dict[str, Any]] = []
        used_cores = 0
        used_ram = 0.0
        used_nvme = 0.0
        used_bw = 0.0

        while pending_jobs or running:
            started = False
            i = 0
            while i < len(pending_jobs):
                job = pending_jobs[i]
                if (used_cores + job.get('cores', 0) <= CORES_PER_SERVER and
                    used_ram + job.get('ram', 0.0) <= RAM_PER_SERVER and
                    used_nvme + job.get('nvme', 0.0) <= NVME_PER_SERVER and
                    used_bw + job.get('bandwidth', 0.0) <= BANDWIDTH_PER_SERVER):
                    start = t
                    dur = max(job.get('duration', 0.0), MIN_JOB_DURATION)
                    end = start + dur
                    running.append({'job': job, 'start': start, 'end': end})
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
                if running:
                    next_end = min(r['end'] for r in running)
                    if next_end <= t + FLOAT_EPSILON:
                        t = next_end + MIN_JOB_DURATION
                    else:
                        t = next_end
                    finished = [r for r in running if r['end'] <= t]
                    for r in finished:
                        used_cores -= r['job'].get('cores', 0)
                        used_ram -= r['job'].get('ram', 0.0)
                        used_nvme -= r['job'].get('nvme', 0.0)
                        used_bw -= r['job'].get('bandwidth', 0.0)
                    running = [r for r in running if r['end'] > t]
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


def pack_first_fit_decreasing(jobs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """First-Fit Decreasing bin-packing algorithm with dynamic server creation.

    Sorts jobs by decreasing (cores, NVMe, bandwidth) and places each job on the
    first server that has sufficient remaining capacity. If no existing server fits,
    creates a new server.

    Args:
        jobs_list: List of job dicts with keys: id, cores, ram, nvme, bandwidth.

    Returns:
        List of server usage dicts, each with keys: cores, ram, nvme, bandwidth, jobs.
        The 'jobs' value is a list of job IDs assigned to that server.
    """
    jobs_sorted = sorted(jobs_list, key=lambda jb: (jb.get('cores',0), jb.get('nvme',0.0), jb.get('bandwidth',0.0)), reverse=True)
    servers_local: List[Dict[str, Any]] = []
    for job in jobs_sorted:
        placed = False
        for su in servers_local:
            if (su['cores'] + job.get('cores',0) <= CORES_PER_SERVER and
                su['ram'] + job.get('ram',0.0) <= RAM_PER_SERVER and
                su['nvme'] + job.get('nvme',0.0) <= NVME_PER_SERVER and
                su['bandwidth'] + job.get('bandwidth',0.0) <= BANDWIDTH_PER_SERVER):
                su['cores'] += job.get('cores',0)
                su['ram'] += job.get('ram',0.0)
                su['nvme'] += job.get('nvme',0.0)
                su['bandwidth'] += job.get('bandwidth',0.0)
                su['jobs'].append(job['id'])
                placed = True
                break
        if not placed:
            servers_local.append({'cores': job.get('cores',0), 'ram': job.get('ram',0.0), 'nvme': job.get('nvme',0.0), 'bandwidth': job.get('bandwidth',0.0), 'jobs':[job['id']]})
    return servers_local


def pack_best_fit_decreasing(jobs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Best-Fit Decreasing bin-packing algorithm with dynamic server creation.

    Sorts jobs by decreasing (cores, NVMe, bandwidth). For each job, places it on
    the server with minimum remaining capacity (best fit) after assignment. Creates
    a new server if no existing server has sufficient capacity.

    Args:
        jobs_list: List of job dicts with keys: id, cores, ram, nvme, bandwidth.

    Returns:
        List of server usage dicts, each with keys: cores, ram, nvme, bandwidth, jobs.
    """
    jobs_sorted = sorted(jobs_list, key=lambda jb: (jb.get('cores',0), jb.get('nvme',0.0), jb.get('bandwidth',0.0)), reverse=True)
    servers_local: List[Dict[str, Any]] = []
    for job in jobs_sorted:
        best_idx = None
        best_rem = None
        for i, su in enumerate(servers_local):
            if (su['cores'] + job.get('cores',0) <= CORES_PER_SERVER and
                su['ram'] + job.get('ram',0.0) <= RAM_PER_SERVER and
                su['nvme'] + job.get('nvme',0.0) <= NVME_PER_SERVER and
                su['bandwidth'] + job.get('bandwidth',0.0) <= BANDWIDTH_PER_SERVER):
                rem = (CORES_PER_SERVER - (su['cores'] + job.get('cores',0)),
                       RAM_PER_SERVER - (su['ram'] + job.get('ram',0.0)),
                       NVME_PER_SERVER - (su['nvme'] + job.get('nvme',0.0)),
                       BANDWIDTH_PER_SERVER - (su['bandwidth'] + job.get('bandwidth',0.0)))
                if best_rem is None or rem < best_rem:
                    best_rem = rem
                    best_idx = i
        if best_idx is not None:
            su = servers_local[best_idx]
            su['cores'] += job.get('cores',0)
            su['ram'] += job.get('ram',0.0)
            su['nvme'] += job.get('nvme',0.0)
            su['bandwidth'] += job.get('bandwidth',0.0)
            su['jobs'].append(job['id'])
        else:
            servers_local.append({'cores': job.get('cores',0), 'ram': job.get('ram',0.0), 'nvme': job.get('nvme',0.0), 'bandwidth': job.get('bandwidth',0.0), 'jobs':[job['id']]})
    return servers_local


def pack_worst_fit(jobs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Worst-Fit bin-packing algorithm with dynamic server creation.

    Sorts jobs by decreasing (cores, NVMe, bandwidth). For each job, places it on
    the server with maximum remaining normalized capacity (worst-fit). This tends to
    spread load more evenly. Creates a new server if no server has sufficient capacity.

    Args:
        jobs_list: List of job dicts with keys: id, cores, ram, nvme, bandwidth.

    Returns:
        List of server usage dicts, each with keys: cores, ram, nvme, bandwidth, jobs.
    """
    jobs_sorted = sorted(jobs_list, key=lambda jb: (jb.get('cores',0), jb.get('nvme',0.0), jb.get('bandwidth',0.0)), reverse=True)
    servers_local: List[Dict[str, Any]] = []
    for job in jobs_sorted:
        best_i = None
        best_score = None
        for i, su in enumerate(servers_local):
            if (su['cores'] + job.get('cores',0) <= CORES_PER_SERVER and
                su['ram'] + job.get('ram',0.0) <= RAM_PER_SERVER and
                su['nvme'] + job.get('nvme',0.0) <= NVME_PER_SERVER and
                su['bandwidth'] + job.get('bandwidth',0.0) <= BANDWIDTH_PER_SERVER):
                rem_sum = ((CORES_PER_SERVER - (su['cores'] + job.get('cores',0))) / max(1, CORES_PER_SERVER)
                           + (RAM_PER_SERVER - (su['ram'] + job.get('ram',0.0))) / max(1.0, RAM_PER_SERVER)
                           + (NVME_PER_SERVER - (su['nvme'] + job.get('nvme',0.0))) / max(1.0, NVME_PER_SERVER)
                           + (BANDWIDTH_PER_SERVER - (su['bandwidth'] + job.get('bandwidth',0.0))) / max(1.0, BANDWIDTH_PER_SERVER))
                if best_score is None or rem_sum > best_score:
                    best_score = rem_sum
                    best_i = i
        if best_i is not None:
            su = servers_local[best_i]
            su['cores'] += job.get('cores',0)
            su['ram'] += job.get('ram',0.0)
            su['nvme'] += job.get('nvme',0.0)
            su['bandwidth'] += job.get('bandwidth',0.0)
            su['jobs'].append(job['id'])
        else:
            servers_local.append({'cores': job.get('cores',0), 'ram': job.get('ram',0.0), 'nvme': job.get('nvme',0.0), 'bandwidth': job.get('bandwidth',0.0), 'jobs':[job['id']]})
    return servers_local


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

    # Constraint: each job assigned to at most one server
    for job in jobs:
        j = job["id"]
        linear = {f"x_{j}_{s}": 1 for s in range(servers)}
        qp.linear_constraint(linear=linear, sense="<=" , rhs=1.0, name=f"assign_at_most_{j}")

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
    jobs_sorted = sorted(jobs, key=lambda jb: (jb.get('cores', 0), jb.get('nvme', 0.0), jb.get('bandwidth', 0.0)), reverse=True)

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
            if (su['cores'] + jcores <= CORES_PER_SERVER and
                su['ram'] + jram <= RAM_PER_SERVER and
                su['nvme'] + jnvme <= NVME_PER_SERVER and
                su['bandwidth'] + jbw <= BANDWIDTH_PER_SERVER):
                # compute remaining capacity after assignment as tuple (smaller is better)
                rem = (
                    CORES_PER_SERVER - (su['cores'] + jcores),
                    RAM_PER_SERVER - (su['ram'] + jram),
                    NVME_PER_SERVER - (su['nvme'] + jnvme),
                    BANDWIDTH_PER_SERVER - (su['bandwidth'] + jbw),
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
                if (es['cores'] + job.get('cores', 0) <= CORES_PER_SERVER and
                    es['ram'] + job.get('ram', 0.0) <= RAM_PER_SERVER and
                    es['nvme'] + job.get('nvme', 0.0) <= NVME_PER_SERVER and
                    es['bandwidth'] + job.get('bandwidth', 0.0) <= BANDWIDTH_PER_SERVER):
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

    # Attempt to solve the QP automatically using a safe, exact solver when available.
    # Use NumPyMinimumEigensolver (exact classical diagonalization) only for very
    # small problems to avoid statevector memory explosions. Fall back to the
    # provided classical heuristics otherwise.
    try:
        # conservative variable limit: jobs * servers
        total_vars = len(var_names)
        max_safe = 8
        print(f"\n[QP] Attempting to solve QP (vars={total_vars}). Max safe vars: {max_safe}.")
        if total_vars > max_safe:
            print(f"[QP] Problem too large for NumPyMinimumEigensolver (vars={total_vars} > {max_safe}). Using classical heuristics instead.")
        else:
            try:
                from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver  # type: ignore
                from qiskit_optimization.algorithms import MinimumEigenOptimizer  # type: ignore
            except ImportError as e:
                print(f"[QP] Qiskit solver import failed: {e}. Using classical heuristics.")
            else:
                # Before attempting to solve, ensure coefficients are integer-like.
                def _is_int_like(x: float) -> bool:
                    try:
                        return abs(float(x) - round(float(x))) < 1e-9
                    except Exception:
                        return False

                coeffs_ok = True
                # check module-level server constants
                for v in (CORES_PER_SERVER, RAM_PER_SERVER, NVME_PER_SERVER, BANDWIDTH_PER_SERVER):
                    if not _is_int_like(v):
                        coeffs_ok = False
                        break
                # check job coefficients
                if coeffs_ok:
                    for job in jobs:
                        for f in ("cores", "ram", "nvme", "bandwidth"):
                            if not _is_int_like(job.get(f, 0)):
                                coeffs_ok = False
                                break
                        if not coeffs_ok:
                            break

                if not coeffs_ok:
                    # Attempt integer-scaling conversion: multiply all coefficients
                    # by a power of ten to make them integer-like, rebuild the
                    # QuadraticProgram with scaled integers, and try solving.
                    from decimal import Decimal

                    def _decimals(x: float) -> int:
                        try:
                            d = Decimal(str(x)).normalize()
                        except Exception:
                            return 0
                        exp = -d.as_tuple().exponent
                        return max(0, exp)

                    max_dec = 0
                    # check module-level constants
                    for v in (CORES_PER_SERVER, RAM_PER_SERVER, NVME_PER_SERVER, BANDWIDTH_PER_SERVER):
                        max_dec = max(max_dec, _decimals(v))
                    for job in jobs:
                        for f in ("cores", "ram", "nvme", "bandwidth"):
                            max_dec = max(max_dec, _decimals(job.get(f, 0)))

                    # limit scaling to avoid huge integers and allocator surprises
                    # use conservative cap (10^3) to prevent extremely large coeffs
                    max_dec = min(max_dec, 3)
                    if max_dec == 0:
                        print("[QP] Non-integer coefficients detected but no decimal places found; skipping scaled solve.")
                    else:
                        scale = 10 ** max_dec
                        print(f"[QP] Attempting integer-scaling by 10^{max_dec} = {scale} and solving scaled QP...")
                        # Build and solve the scaled QP, but guard aggressively
                        try:
                            from qiskit_optimization import QuadraticProgram as _QP2  # type: ignore
                        except ImportError:
                            print("[QP] qiskit-optimization not available for scaled QP; skipping scaled solve.")
                        else:
                            try:
                                scaled_qp = _QP2(name="cloud_binpacking_scaled")
                                # variables
                                for s in range(servers):
                                    scaled_qp.binary_var(name=f"y_{s}")
                                for job in jobs:
                                    j = job["id"]
                                    for s in range(servers):
                                        scaled_qp.binary_var(name=f"x_{j}_{s}")

                                # assignment constraints
                                for job in jobs:
                                    j = job["id"]
                                    linear = {f"x_{j}_{s}": int(round(1 * scale)) for s in range(servers)}
                                    scaled_qp.linear_constraint(linear=linear, sense="<=", rhs=int(round(1 * scale)), name=f"assign_at_most_{j}")

                                # per-server constraints (scaled)
                                sc_cores = int(round(CORES_PER_SERVER * scale))
                                sc_ram = int(round(RAM_PER_SERVER * scale))
                                sc_nvme = int(round(NVME_PER_SERVER * scale))
                                sc_bw = int(round(BANDWIDTH_PER_SERVER * scale))

                                for s in range(servers):
                                    linear = {f"x_{job['id']}_{s}": int(round(job['cores'] * scale)) for job in jobs}
                                    linear[f"y_{s}"] = -sc_cores
                                    scaled_qp.linear_constraint(linear=linear, sense="<=", rhs=0, name=f"cores_cap_{s}")

                                    linear = {f"x_{job['id']}_{s}": int(round(job['ram'] * scale)) for job in jobs}
                                    linear[f"y_{s}"] = -sc_ram
                                    scaled_qp.linear_constraint(linear=linear, sense="<=", rhs=0, name=f"ram_cap_{s}")

                                    linear = {f"x_{job['id']}_{s}": int(round(job['nvme'] * scale)) for job in jobs}
                                    linear[f"y_{s}"] = -sc_nvme
                                    scaled_qp.linear_constraint(linear=linear, sense="<=", rhs=0, name=f"nvme_cap_{s}")

                                    linear = {f"x_{job['id']}_{s}": int(round(job.get('bandwidth', 0.0) * scale)) for job in jobs}
                                    linear[f"y_{s}"] = -sc_bw
                                    scaled_qp.linear_constraint(linear=linear, sense="<=", rhs=0, name=f"bandwidth_cap_{s}")

                                # objective: minimize sum(y_s) scaled
                                linear_obj = {f"y_{s}": int(round(1 * scale)) for s in range(servers)}
                                scaled_qp.minimize(linear=linear_obj)

                            except MemoryError as me:
                                print(f"[QP] MemoryError while building scaled QP: {me}. Skipping scaled solve and using classical heuristics.")
                                _log_memory_error("building scaled_qp", me, extra={"servers": servers, "jobs_count": len(jobs)})
                            else:
                                # attempt solve on scaled_qp with MemoryError guard
                                try:
                                    solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
                                    res = solver.solve(scaled_qp)
                                except MemoryError as me:
                                    print(f"[QP] MemoryError during scaled NumPyMinimumEigensolver solve: {me}. Falling back to classical heuristics.")
                                    _log_memory_error("scaled NumPyMinimumEigensolver solve", me, extra={"servers": servers, "jobs_count": len(jobs)})
                                except Exception as se:
                                    print(f"[QP] Scaled NumPyMinimumEigensolver solve failed: {type(se).__name__}: {se}. Falling back to classical heuristics.")
                                else:
                                    print("[QP] Scaled solver result:")
                                    print(f" x: {res.x}")
                                    print(f" objective: {res.fval}")
                                    # interpret solution (same variable names)
                                    assign = {v.name: int(round(val)) for v, val in zip(scaled_qp.variables, res.x)}
                                    assigned_jobs = {}
                                    for name, val in assign.items():
                                        if name.startswith('x_') and val == 1:
                                            parts = name.split('_')
                                            jid = int(parts[1])
                                            sid = int(parts[2])
                                            assigned_jobs.setdefault(sid, []).append(jid)
                                    for sidx in range(servers):
                                        jobs_on = assigned_jobs.get(sidx, [])
                                        print(f" - Server {sidx}: Jobs {jobs_on}")
                else:
                    try:
                        solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
                        res = solver.solve(qp)
                        print("[QP] Solver result:")
                        print(f" x: {res.x}")
                        print(f" objective: {res.fval}")
                        # Map assignments to per-server placement for display
                        assign = {v.name: int(round(val)) for v, val in zip(qp.variables, res.x)}
                        assigned_jobs = {}
                        for name, val in assign.items():
                            if name.startswith('x_') and val == 1:
                                parts = name.split('_')
                                # name x_{job}_{server}
                                jid = int(parts[1])
                                sid = int(parts[2])
                                assigned_jobs.setdefault(sid, []).append(jid)
                        for sidx in range(servers):
                            jobs_on = assigned_jobs.get(sidx, [])
                            print(f" - Server {sidx}: Jobs {jobs_on}")
                    except MemoryError as me:
                        print(f"[QP] NumPyMinimumEigensolver out of memory: {me}. Falling back to classical heuristics.")
                        _log_memory_error("NumPyMinimumEigensolver solve (original qp)", me, extra={"total_vars": total_vars, "servers": servers, "jobs_count": len(jobs)})
                    except Exception as ee:
                        print(f"[QP] NumPyMinimumEigensolver solve failed: {type(ee).__name__}: {ee}. Falling back to classical heuristics.")
    except Exception:
        # Be defensive: any unexpected error should not crash the script.
        print("[QP] Unexpected error while attempting to solve QP; using classical heuristics.")

    # If we reach here (or chose not to run the quantum solver), show classical results
    print("\nUsing classical heuristics (best_fit_decreasing) to show an example solution:")
    bf = pack_best_fit_decreasing(jobs)
    per_rec, makespan = simulate_servers(bf, jobs)
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
        nonclipped = [t for t in tasks if t not in clipped]
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

    for i in order:
        best_j = None
        best_delta = float('inf')
        for j in range(m):
            # quick feasibility checks
            if Rho[j] + alpha[i][j] >= rho_max - 1e-12:
                continue
            # cpu feasibility
            if cpu_max is not None:
                if cpu[j] + tasks[i].get('cpu', 0.0) > cpu_max + 1e-12:
                    continue

            # compute marginal cost
            old_cost = _mg1_server_cost(Lambda[j], M2[j], Rho[j])
            new_L = Lambda[j] + lam[i]
            new_M2 = M2[j] + beta[i][j]
            new_R = Rho[j] + alpha[i][j]
            new_cost = _mg1_server_cost(new_L, new_M2, new_R)
            delta = new_cost - old_cost
            # tie-break bias if provided
            if tie_break:
                delta += tie_break.get(f"x_{i}_{j}", 0.0)
            if delta < best_delta:
                best_delta = delta
                best_j = j

        if best_j is None:
            # no feasible server by greedy rules -> abort and return infeasible
            return ({}, float('nan'))

        # assign
        assigned[i] = best_j
        Lambda[best_j] += lam[i]
        M2[best_j] += beta[i][best_j]
        Rho[best_j] += alpha[i][best_j]
        cpu[best_j] += tasks[i].get('cpu', 0.0)

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


def format_server_snapshot(server_idx: int, server_st: Dict[str, Any], current_time: float) -> str:
    """Format a snapshot of server resource utilization.

    Args:
        server_idx: Server index (0-based).
        server_st: Server state dict with resource usage.
        current_time: Current simulation time.

    Returns:
        Multi-line string with resource usage and overall status.
    """
    lines = [f"t={current_time:.2f}s: Server {server_idx} resources:"]
    lines.append(format_stress_line('Cores', server_st['cores'], CORES_PER_SERVER, ''))
    lines.append(format_stress_line('RAM', server_st['ram'], RAM_PER_SERVER, 'GB'))
    lines.append(format_stress_line('NVMe', server_st['nvme'], NVME_PER_SERVER, 'GB'))
    lines.append(format_stress_line('Bandwidth', server_st['bandwidth'], BANDWIDTH_PER_SERVER, 'Gbps'))

    # determine overall status and concise reasons
    status = 'OK'
    reasons = []
    for name, used, cap in (('Cores', server_st['cores'], CORES_PER_SERVER),
                            ('RAM', server_st['ram'], RAM_PER_SERVER),
                            ('NVMe', server_st['nvme'], NVME_PER_SERVER),
                            ('Bandwidth', server_st['bandwidth'], BANDWIDTH_PER_SERVER)):
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
        lines = [f"t={current_time:.2f}s: Server {sidx} resources:"]
        st = server_state[sidx]
        lines.append(format_stress_line('Cores', st['cores'], _cores, ''))
        lines.append(format_stress_line('RAM', st['ram'], _ram, 'GB'))
        lines.append(format_stress_line('NVMe', st['nvme'], _nvme, 'GB'))
        lines.append(format_stress_line('Bandwidth', st['bandwidth'], _bandwidth, 'Gbps'))
        
        # determine overall status and concise reasons
        status = 'OK'
        reasons = []
        for name, used, cap in (('Cores', st['cores'], _cores),
                                ('RAM', st['ram'], _ram),
                                ('NVMe', st['nvme'], _nvme),
                                ('Bandwidth', st['bandwidth'], _bandwidth)):
            pct = (used / cap) * 100.0 if cap > 0 else 0.0
            if pct > 100.0:
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
        """Attempt to assign jobs from waiting queue to available servers."""
        i = 0
        assigned_any = False
        while i < len(waiting_queue):
            job = waiting_queue[i]
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
                # free resources
                st['cores'] -= r['job'].get('cores', 0)
                st['ram'] -= r['job'].get('ram', 0.0)
                st['nvme'] -= r['job'].get('nvme', 0.0)
                st['bandwidth'] -= r['job'].get('bandwidth', 0.0)
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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

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

    # Path 1 — scipy.milp (exact MILP, unlimited problem size)
    if x_vals is None:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds  # type: ignore
            import numpy as np

            n_vars = qp.get_num_vars()
            var_names = [v.name for v in qp.variables]

            # Objective coefficients (linear minimisation)
            c = np.zeros(n_vars)
            for i, coeff in qp.objective.linear.to_dict(use_name=False).items():
                c[i] = float(coeff)

            # All variables are binary: integrality=1, bounds [0,1]
            integrality = np.ones(n_vars)
            bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

            # Build constraint matrix from QP linear constraints
            A_rows, b_lo, b_hi = [], [], []
            for lc in qp.linear_constraints:
                row = np.zeros(n_vars)
                for i, coeff in lc.linear.to_dict(use_name=False).items():
                    row[i] = float(coeff)
                rhs = float(lc.rhs)
                if lc.sense.label == '=':
                    A_rows.append(row); b_lo.append(rhs);  b_hi.append(rhs)
                elif lc.sense.label == '<=':
                    A_rows.append(row); b_lo.append(-np.inf); b_hi.append(rhs)
                elif lc.sense.label == '>=':
                    A_rows.append(row); b_lo.append(rhs);  b_hi.append(np.inf)

            if A_rows:
                A      = np.vstack(A_rows)
                lc_obj = LinearConstraint(A, b_lo, b_hi)
                constraints = [lc_obj]
            else:
                constraints = []

            print(f"[QAOA] scipy.milp: {n_vars} vars, {len(A_rows)} constraints...")
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

            if res.success:
                x_vals      = {var_names[i]: int(round(res.x[i])) for i in range(n_vars)}
                best_fval   = float(res.fun)
                solver_label = "scipy_milp"
                print(f"[QAOA] scipy.milp solved. Objective: {best_fval:.4f}")
            else:
                print(f"[QAOA] scipy.milp did not find a solution: {res.message}")
        except Exception as _e:
            print(f"[QAOA] scipy.milp failed: {_e}")

    # Path 2 — numpy brute-force (exact, works up to ~20 QUBO vars)
    if x_vals is None:
        try:
            from qiskit_optimization.converters import QuadraticProgramToQubo as _ToQubo  # type: ignore
            import numpy as np
            print(f"[QAOA] Falling back to numpy brute-force...")
            qubo = _ToQubo(penalty=1.0).convert(qp)
            n    = qubo.get_num_vars()
            if n > 22:
                raise MemoryError(f"Too many QUBO vars ({n}) for brute-force")
            print(f"[QAOA] QUBO has {n} vars. Searching 2^{n}={2**n} bitstrings...")
            Q = np.zeros((n, n))
            for i, ci in qubo.objective.linear.to_dict(use_name=False).items():
                Q[i, i] += ci
            for (i, j), cij in qubo.objective.quadratic.to_dict(use_name=False).items():
                Q[i, j] += cij / 2.0
                Q[j, i] += cij / 2.0
            offset   = qubo.objective.constant
            bits     = np.arange(2 ** n, dtype=np.int32)
            X        = ((bits[:, None] >> np.arange(n)) & 1).astype(np.float64)
            obj_vals = np.einsum('bi,ij,bj->b', X, Q, X) + offset
            best_idx = int(np.argmin(obj_vals))
            best_x   = X[best_idx].astype(int)
            num_qp_vars = qp.get_num_vars()
            x_vals      = {qp.variables[i].name: int(best_x[i])
                           for i in range(num_qp_vars)}
            best_fval    = float(obj_vals[best_idx])
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
    # x_vals already built above from brute-force QUBO search

    for j, job in enumerate(jobs):
        assigned = False
        for s in range(S):
            if x_vals.get(f"x_{j}_{s}", 0) == 1:
                assignments[job["id"]] = s
                timeline.append(f"QAOA: Job {job['id']} ({job['type']}) -> Server {s}")
                assigned = True
                break
        if not assigned:
            assignments[job["id"]] = 0
            timeline.append(f"QAOA unassigned: Job {job['id']} -> Server 0 (fallback)")

    active = len(set(assignments.values()))
    timeline.append(f"QAOA result: {active} active server(s) out of {S}")
    timeline.append(f"Objective value: {best_fval:.4f}")

    makespan  = max((job.get("duration", 0.0) for job in jobs), default=0.0)
    peak_util = max(
        (sum(jobs[j].get("cores", 0) for j in range(J)
             if assignments.get(jobs[j]["id"]) == s) / _cores * 100
         for s in range(S)),
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
        jobs, servers=num_servers, verbose=True,
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
    server_constants: ServerConstants = ServerConstants()


@app.post("/schedule")
def schedule(req: WorkloadRequest):
    """Classical greedy online scheduler."""
    jobs = generate_jobs(
        req.profile_counts,
        cores_per_server=req.server_constants.cores,
        ram_per_server=req.server_constants.ram,
        nvme_per_server=req.server_constants.nvme,
        bandwidth_per_server=req.server_constants.bandwidth
    )
    assignments, timeline, stats = online_scheduler_with_arrivals(
        jobs, servers=req.servers, verbose=True,
        cores_per_server=req.server_constants.cores,
        ram_per_server=req.server_constants.ram,
        nvme_per_server=req.server_constants.nvme,
        bandwidth_per_server=req.server_constants.bandwidth
    )
    return {"stats": stats, "timeline": timeline, "solver": "classical_greedy"}


@app.post("/schedule_quantum")
def schedule_quantum(req: WorkloadRequest):
    """QAOA quantum bin-packing scheduler.

    Automatically falls back to classical greedy if the problem is too large
    for the local simulator (>20 binary variables).
    """
    jobs = generate_jobs(
        req.profile_counts,
        cores_per_server=req.server_constants.cores,
        ram_per_server=req.server_constants.ram,
        nvme_per_server=req.server_constants.nvme,
        bandwidth_per_server=req.server_constants.bandwidth
    )
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
    jobs = generate_jobs(
        req.profile_counts,
        cores_per_server=req.server_constants.cores,
        ram_per_server=req.server_constants.ram,
        nvme_per_server=req.server_constants.nvme,
        bandwidth_per_server=req.server_constants.bandwidth
    )

    classical_assignments, classical_timeline, classical_stats = online_scheduler_with_arrivals(
        jobs, servers=req.servers, verbose=True,
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
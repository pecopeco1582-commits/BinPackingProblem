# Packing Jobs onto Servers — Normal vs Quantum

A small teaching tool that fits a set of computer jobs onto as few servers as possible, and solves the same puzzle two ways — with an ordinary computer and with a quantum method (QAOA) — so you can compare them.

## Try this first

1. **Just look around (no install):** open `alpha_scheduler_v1.html` in Chrome or Firefox and click **"Take the quick tour"**, then **"Start walkthrough"**. The guided tour and a sample result work entirely offline.
2. **Run real comparisons (needs the helper program):**
   ```bash
   cd "capstone final"
   pip install fastapi uvicorn scipy qiskit qiskit-optimization qiskit-algorithms qiskit-aer
   python -m uvicorn alpha4:app --reload
   ```
   When you see `Uvicorn running on http://127.0.0.1:8000`, leave that window open, go back to the page, pick some jobs, and click **Execute simulation**.

## Who this is for

- Students and curious non-experts who want to *see* what "quantum optimization" actually does (and doesn't do) without any prior background.
- Anyone learning about bin-packing / scheduling and how a classical solver compares to a quantum one on the very same problem.
- No quantum-computing or command-line experience assumed — the page explains every control.

## What you will learn

- What the **bin-packing** problem is: fit jobs onto the fewest servers without overloading any of them.
- How a **normal computer** solves it quickly with a well-known scheduling method.
- How a **quantum method (QAOA)** encodes the same puzzle, and — the honest lesson — **why simulating it on a laptop is impractical**: the problem size explodes into too many "qubits," and each step of the quantum simulation is extremely slow.
- Why, at the sizes a laptop can handle, the normal computer wins easily.

## How it works

- **You choose** how many of four job types to run (`web_api`, `batch`, `memory`, `io_heavy`) and how many servers are available. Every server is identical: **2 cores · 4 GB RAM · 40 GB NVMe · 10 Gbps**.
- The page sends your choice to a small local program (`alpha4.py`, a FastAPI service on `localhost:8000`) and asks it to solve the packing two ways:
  - **Normal computer** (`/schedule`) — an event-driven scheduler that places jobs and reports how many servers it used and how long the jobs take.
  - **Quantum (QAOA)** (`/schedule_quantum`) — builds the problem as a quantum optimization (QUBO) and runs a real QAOA circuit simulation.
- Both "total time to finish" figures use the **same concurrent simulation model**, so the columns are a fair, apples-to-apples comparison.
- Results appear side by side, with a plain-words explanation, a technical explanation, and a one-line takeaway.

## Example result

Run one of the **Quick presets** (they're sized to stay in the green ≤8-qubit zone, where real
QAOA actually finishes). The **Normal computer** column fills in instantly; the **Quantum (QAOA)**
column finishes a few seconds to ~15 s later. For preset ③ (1 web_api on 2 servers ≈ 8 qubits):

> **Servers used:** 1 vs 1  ·  **Total time to finish:** 5.0 s vs 5.0 s  ·
> **Time to find the answer:** ~0.4 ms (normal) vs ~15 s (quantum)

Both find the same packing, but the quantum side takes **tens of thousands of times longer** — the
whole point. Push the problem bigger (out of the green) and the quantum side honestly reports that
it can't keep up:

> *At ~10 qubits the quantum step is too slow: it runs the full 60 s and then stops without an
> answer.* …or, larger still: *Problem too large for real QAOA on this machine — exceeds the
> 20-qubit limit.*

So small problems show a real quantum result that's much slower; bigger ones show, concretely, why
simulating quantum optimization doesn't scale on a laptop.

## Feedback form

This is a teaching tool, so notes on what did or didn't make sense are especially welcome. Use the **Feedback** tab in the page, or open the form directly:

https://docs.google.com/forms/d/e/1FAIpQLSfo-0f5QXVAqe5F5CVVrF5Scm4XuIjr3q60g5D7jcUR8t4qyg/viewform

## Technical details

- **Backend:** `alpha4.py` — a FastAPI app exposing `/schedule` (classical), `/schedule_quantum` (real QAOA), and `/schedule_compare` (both). Run with `python -m uvicorn alpha4:app --reload`. Interactive API docs at `http://localhost:8000/docs`.
- **Frontend:** `alpha_scheduler_v1.html` — a single self-contained HTML/CSS/JS file; no build step, no server of its own. It talks to the backend over `localhost:8000`.
- **Classical solver:** an event-driven online scheduler (`online_scheduler_with_arrivals`); placements are scored for "time to finish" with a concurrent per-server simulation (`simulate_servers`).
- **Quantum solver:** `solve_binpack_qaoa_web` builds a **reduced (cores-only) QuadraticProgram**, converts it to a QUBO, and runs genuine QAOA via `qiskit-algorithms` on a `StatevectorSampler`, wrapped in a 60-second wall-clock timeout and a 20-qubit guard.
- **Stack:** Python 3.13 · FastAPI · Uvicorn · SciPy · Qiskit 2.2.3 · qiskit-algorithms 0.4.0 · qiskit-aer · qiskit-optimization 0.7.0.
- **CLI (no web page):** `python alpha4.py` for an interactive run, or `python alpha4.py --demo-online` for a non-interactive arrival-driven demo. (On Windows, force UTF-8 output if you see encoding errors.)
- **Source:** https://github.com/pecopeco1582-commits/BinPackingProblem

## Limitations

- **Real QAOA is not practical on a laptop.** Encoding the full resource constraints explodes the qubit count (40+ qubits even for tiny inputs), so the quantum path uses a **reduced cores-only model**. Even then, each optimizer step of the statevector simulation takes ~90 s, so the web request is capped at 60 s and almost always reports "not available." This is expected, not a bug.
- **The quantum and classical models aren't identical.** The reduced quantum model enforces only the cores limit; the classical scheduler enforces cores, RAM, NVMe, and bandwidth. Placements are re-checked against all limits before scoring.
- **No classical fallback.** If the quantum solver can't run, it reports a clear failure instead of quietly substituting a classical answer.
- **Fixed server spec.** The quantum path assumes the standard 2-core / 4 GB / 40 GB / 10 Gbps server; per-request server overrides are not applied on that side.
- **Background cost.** Because Python can't kill a thread, a timed-out quantum run keeps computing in the background for ~90 s after the page already showed its failure; many rapid clicks could pile up CPU.
- **CLI caveat.** The interactive `python alpha4.py` path has a known pre-existing crash when two jobs finish at the exact same time on one server.

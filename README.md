# BinPackingProblem

# Cloud Bin-Packing + Qiskit Optimization

## Overview
This project models a cloud scheduling / bin-packing problem and encodes it into a Qiskit `QuadraticProgram`.

It combines:
- Classical bin-packing heuristics
- Discrete-event simulation
- Optional quantum / hybrid optimization

Goal: minimize the number of active servers while respecting resource constraints.

---

## Core Features

### 1. Job Generation
Workloads are generated from predefined profiles:

- web_api — low latency, light resources  
- batch — long-running compute-heavy  
- memory — RAM-intensive  
- io_heavy — storage-heavy  

Each job includes:
- CPU cores
- RAM
- NVMe storage
- Bandwidth
- Duration

---

### 2. Server Constraints

Each server has fixed capacity:

- 2 vCPUs  
- 4 GB RAM  
- 40 GB NVMe  
- 10 Gbps bandwidth  

All assignments must satisfy these hard constraints.

---

### 3. Optimization Model (Qiskit)

Binary variables:
- x_{j,s} → job j assigned to server s
- y_s → server s is active

Constraints:
- Each job assigned to at most one server
- Resource limits per server
- Jobs can only be placed on active servers

Objective:
Minimize total active servers:
    sum(y_s)

---

### 4. Classical Algorithms

Implemented heuristics:

- First-Fit Decreasing (FFD)
- Best-Fit Decreasing (BFD)
- Worst-Fit

Used to:
- Generate feasible solutions
- Benchmark performance
- Provide fallback when Qiskit is unavailable

---

### 5. Simulation Engine

Simulates execution timeline per server:
- Tracks job start/end times
- Enforces real-time resource usage

Computes:
- Makespan (total completion time)
- Per-server utilization

---

### 6. Quantum / Hybrid Solving (Optional)

Attempts solving via:

- NumPyMinimumEigensolver (exact, small problems only)
- QAOA (if available)

Includes:
- Memory safety checks
- Automatic fallback to classical heuristics
- Integer scaling for floating-point stability

---

## File Structure

Single script containing:
- Data models (Job, ServerState)
- Job generation
- Packing algorithms
- Simulation logic
- Qiskit model construction
- Solver logic (quantum + fallback)

---

## Installation

Install required dependencies:

    pip install qiskit qiskit-optimization

Optional (for advanced solvers):

    pip install qiskit-algorithms

---

## Usage

Run the script:

    python main.py

Flow:
1. Enter number of servers
2. Select workload counts
3. Program will:
   - Generate jobs
   - Build optimization model
   - Run heuristics
   - Simulate execution
   - Attempt optimization

---

## Example Output

- Job list
- Model summary
- Before vs after optimization:
  - Servers used
  - Completion time
  - Cost savings
- Per-server execution timeline

---

## Advanced Module: M/G/1 Optimization

Includes a queueing-based formulation:

- Uses M/G/1 waiting-time approximation
- Objective includes:
  - Utilization (rho)
  - Service time variance

Supports:
- QAOA
- Exact solvers
- Greedy and brute-force fallback

---

## Design Principles

- Hard constraints are always enforced  
- Objective minimizes infrastructure cost  
- Hybrid approach (classical + quantum)  
- Fail-safe execution (graceful fallback)  

---

## Limitations

- Quantum solving only works for very small problems  
- Exponential scaling with number of variables  
- Classical heuristics dominate real workloads  

---

## Use Cases

- Cloud resource scheduling  
- VM/container placement  
- Edge computing allocation  
- Quantum optimization research  

---

## Summary

This project connects:
- Combinatorial optimization
- Cloud systems modeling
- Quantum computing

It shows how real infrastructure problems can be expressed as binary optimization models and solved using both classical and quantum approaches.

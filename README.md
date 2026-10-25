# macro-placement-floorplanning
Graph-theoretic macro placement: comparison of Random, Greedy, Simulated Annealing, and Reinforcement Learning.

# Graph-Theoretic Macro Placement and Floorplanning

A comparative study of Random placement, Greedy optimization, Simulated Annealing (SA), and Reinforcement Learning (RL).

## Objectives

- Model a circuit netlist as a graph.
- Generate valid macro placements subject to boundary and non-overlap constraints.
- Compare algorithms using half-perimeter wirelength (HPWL), estimated congestion, runtime, and feasibility.
- Investigate when RL justifies its training and computational cost compared with classical optimization.

## Technology Stack

- Python, NumPy, NetworkX
- Gymnasium, PyTorch, Stable-Baselines3 and sb3-contrib
- FastAPI and Uvicorn
- React, TypeScript and Vite
- Plotly or Recharts

## Team Responsibilities

- **Zoya:** Placement engine, constraints, objective integration and backend/API.
- **Harsh:** Frontend, floorplan visualization, controls and charts.
- **Shravani:** Random, Greedy and Simulated Annealing algorithms, classical experiments and comparisons.
- **Ishita:** RL environment integration, MaskablePPO training, checkpoints and evaluation.

## Research Principles

All algorithms must use the same placement constraints, benchmark instances, objective function and clearly defined evaluation budgets.

## Project Status

Initial repository setup.

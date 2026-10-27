# Macro Placement Benchmark Evaluation Report

## 1. Experimental Setup & Methodology

- **Engine Version**: 0.1.0
- **Random Baseline Seeds**: 42, 100, 2026
- **Origin & Coordinate System**: Grid cells with $(0, 0)$ top-left, half-open rectangles $[x, x+w) \times [y, y+h)$.
- **Objective Metric**: Half-Perimeter Wirelength (HPWL):
  $$\text{HPWL} = \sum_{net \in \mathcal{N}} w_{net} \cdot [(\max x_{center} - \min x_{center}) + (\max y_{center} - \min y_{center})]$$
  where macro centers are at $(x + w/2, y + h/2)$.
- **Statistical Conventions**:
  - **Success Rate**: Evaluated as $\frac{N_{successful}}{N_{total}}$ where $N_{total}$ includes all attempted runs.
  - **Sample Standard Deviation**: Calculated using Bessel's correction ($N - 1$ degrees of freedom) via `statistics.stdev`:
    $$s = \sqrt{\frac{\sum_{i=1}^N (x_i - \bar{x})^2}{N - 1}}$$
  - **Comparability Notice**: Raw HPWL is instance-dependent and non-comparable across different netlists due to varying grid sizes, macro counts, and net topologies.

---

## 2. Benchmark Instances

| Benchmark ID | Grid Size | Macro Count | Net Count | Total Macro Area | Grid Area | Density |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `toy_4` | $15 \times 15$ | 4 | 3 | 31 | 225 | 13.8% |
| `medium_8` | $30 \times 30$ | 8 | 6 | 211 | 900 | 23.4% |
| `dense_10` | $24 \times 24$ | 10 | 7 | 345 | 576 | 59.9% |

---

## 3. Comparative Summary Statistics

| Algorithm | Netlist ID | Total Runs | Success Rate | Mean HPWL | Median HPWL | Min HPWL | Max HPWL | Sample Std Dev | Mean Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `random` | `toy_4` | 3 | 100.0% | **61.00** | 62.50 | 53.00 | 67.50 | 7.37 | 5.41 ms |
| `random` | `medium_8` | 3 | 100.0% | **234.67** | 236.75 | 201.50 | 265.75 | 32.18 | 14.40 ms |
| `random` | `dense_10` | 3 | 100.0% | **250.42** | 245.00 | 241.50 | 264.75 | 12.54 | 15.83 ms |
| `greedy` | *All* | 0 | *Pending* | - | - | - | - | - | Awaiting runner |
| `sa` | *All* | 0 | *Pending* | - | - | - | - | - | Awaiting runner |
| `ppo` | *All* | 0 | *Pending* | - | - | - | - | - | Awaiting RL weights |

---

## 4. Per-Run Granular Data

| Algorithm | Netlist ID | Seed | Success | Legal? | Complete? | HPWL | Runtime (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `random` | `toy_4` | 42 | Yes | Yes | Yes | 67.50 | 0.005572 |
| `random` | `toy_4` | 100 | Yes | Yes | Yes | 62.50 | 0.005305 |
| `random` | `toy_4` | 2026 | Yes | Yes | Yes | 53.00 | 0.005361 |
| `random` | `medium_8` | 42 | Yes | Yes | Yes | 265.75 | 0.014309 |
| `random` | `medium_8` | 100 | Yes | Yes | Yes | 236.75 | 0.014425 |
| `random` | `medium_8` | 2026 | Yes | Yes | Yes | 201.50 | 0.014458 |
| `random` | `dense_10` | 42 | Yes | Yes | Yes | 264.75 | 0.015779 |
| `random` | `dense_10` | 100 | Yes | Yes | Yes | 245.00 | 0.015736 |
| `random` | `dense_10` | 2026 | Yes | Yes | Yes | 241.50 | 0.015961 |

---

## 5. Artifact Files

- `benchmarks/results/benchmark_raw_results.csv`: Complete raw per-run execution data.
- `benchmarks/results/benchmark_raw_results.json`: Structured machine-readable raw dataset.
- `benchmarks/results/benchmark_summary.csv`: Aggregated statistical performance table.
- `benchmarks/results/benchmark_summary.json`: JSON format statistical metrics.
- `benchmarks/results/hpwl_comparison_per_instance.svg`: High-resolution per-instance HPWL seed distribution chart.
- `benchmarks/results/runtime_comparison_per_instance.svg`: Mean execution time by instance chart.

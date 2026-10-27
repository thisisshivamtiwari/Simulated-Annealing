# Airline Crew Scheduling with Simulated Annealing and Genetic Algorithms

Solves the airline crew scheduling problem, modelled as a **set partitioning problem** (cover every flight leg exactly once at minimum cost), and compares three metaheuristics:

- **Simulated Annealing**: multi-restart SA with swap, add and remove moves (`simulated_annealing.py`)
- **Standard Binary Genetic Algorithm** (`standard_bga.py`)
- **Improved Binary Genetic Algorithm** (`improved_bga.py`)

Each algorithm runs 30 times (seeds 1–30) on each problem; the runner reports best, worst, mean, standard deviation, feasibility and run time.

## Data

Problems `sppnw41`, `sppnw42` and `sppnw43` from J. E. Beasley's [OR-Library](http://people.brunel.ac.uk/~mastjjb/jeb/info.html) set partitioning collection. The file format is described in `sppinfo.txt`.

## Run

Python 3, standard library only:

```bash
python3 main.py
```

Results are written to `results/` (`summary.txt` holds the combined table). Flowcharts of the three algorithms are in `images/`, and the full write-up is in `Report.pdf`.

## Files

| File | Purpose |
|---|---|
| `main.py` | Runs all three algorithms on all problems and saves the results |
| `utils.py` | Parsing, cost, coverage and constraint-violation helpers, and construction heuristics |
| `crew_scheduling.py` | Single-file version of the full pipeline |

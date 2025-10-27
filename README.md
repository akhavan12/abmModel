
# NetLogo → Python ABM (NumPy, GPU-ready) + Parameter Sweeps

This repo contains a minimal, vectorized agent-based epidemic model in Python and a sweep pipeline
that reproduces the 6×5 boxplot grid (six parameter groups × five metrics) like your figure.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the sweep (edit ranges in sweep.py)
python sweep.py  # writes results.csv

# make the figure
python plot_results.py  # writes figure.png
```

The simulation is in `abm/sim.py`. It's NumPy-based but structured so that you can pass
`xp=cupy` to `simulate(...)` to run on the GPU with CuPy (optional).

## Where to change parameters
- **Defaults:** in `sweep.py:DEFAULTS`
- **Ranges for sweeps:** in `sweep.py:RANGES`
- **Metrics + grid layout:** in `plot_results.py`

## Outputs
- `results.csv` with columns:
  `param_name, param_value, run, runtime_days, infected, reinfected, long_covid_cases, min_productivity`
- `figure.png` – the 6×5 boxplot grid (rows=metrics, cols=parameter groups).

## GPU note
To run on GPU:
```python
import cupy as cp
from abm.sim import simulate
out = simulate(xp=cp, ...)  # same args as NumPy
```
Keep the logic vectorized for best performance.

## Mapping from NetLogo
This is a faithful-but-simplified translation:
- Well-mixed contacts approximated by binomial draws against prevalence.
- Reinfection after `immunity_days`.
- Vaccination lowers per-contact infection probability.
- Long COVID flags reduce productivity toward `100*(1-severity)` and recover slowly.
- Infection produces a temporary productivity dip.

Adjust those mechanisms to match your NetLogo exactly if needed.

## Engines

- **NetLogo-style (default in this zip):** `abm/sim_netlogoish.py`
  - Explicit contacts on a static network with NetLogo-like tick order.
- **Fast vectorized (alternative):** `abm/sim.py`
  - Well-mixed prevalence approximation, GPU-ready with CuPy (`xp=cp`).

Switch engines by changing the import at the top of `sweep.py`.

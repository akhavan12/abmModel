import itertools
import numpy as np
import pandas as pd
from abm.sim_netlogoish import simulate_netlogoish as simulate

# Defaults mapped to NetLogo-ish names
DEFAULTS = dict(
    N=1000,
    max_days=365,
    covid_spread_chance_pct=5.0,   # percent
    initial_infected_agents=10,
    precaution_pct=0.0,            # percent
    avg_degree=20,                 # average connections (static M1)
    v_start_time=0,                # vaccination start day
    vaccination_pct=0.0,           # percent to vaccinate at start time
    infected_period=14,
    active_duration=7,
    symptomatic_start=2,
)

# Parameter ranges for the 6 columns in the figure
RANGES = {
    "covid_spread_chance_pct": [2, 5, 10, 20],  # percent
    "initial_infected_agents": [2, 5, 10, 20],
    "precaution_pct": [0, 30, 50, 80],
    "avg_degree": [10, 30, 50, 70],
    "v_start_time": [0, 30, 180, 360],
    "vaccination_pct": [0, 30, 50, 80],
}

METRICS = ["runtime_days","infected","reinfected","long_covid_cases","min_productivity"]

def run_sweep(n_runs=20, seed=0, out_csv="results.csv"):
    rows = []
    rs = np.random.RandomState(seed)
    for pname, values in RANGES.items():
        for val in values:
            for r in range(n_runs):
                params = DEFAULTS.copy()
                params[pname] = val
                params["seed"] = int(rs.randint(0, 2**31-1))
                out = simulate(**params)
                out.update(dict(param_name=pname, param_value=val, run=r))
                rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

if __name__ == "__main__":
    df = run_sweep(n_runs=8, seed=42, out_csv="results.csv")
    print("Wrote", len(df), "rows to results.csv")

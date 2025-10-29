import itertools
import numpy as np
import pandas as pd
from abm.sim_netlogoish import simulate_netlogoish

# Defaults matching your NetLogo screenshot
DEFAULTS = dict(
    N=10000,
    max_days=365,
    covid_spread_chance_pct=10.0,
    initial_infected_agents=5,
    precaution_pct=50.0,
    avg_degree=5,  # This is "Temporal Connections" in your plot but actually M1-Average-connections
    v_start_time=180,
    vaccination_pct=80.0,
    infected_period=10,
    active_duration=7,
    immune_period=21,
    incubation_period=4,
    symptomatic_duration_min=1,
    symptomatic_duration_mid=10,
    symptomatic_duration_max=60,
    symptomatic_duration_dev=8,
    asymptomatic_pct=40.0,
    effect_of_reinfection=3,
    super_immune_pct=4.0,
    # Long COVID parameters
    long_covid=True,
    long_covid_time_threshold=30,
    asymptomatic_lc_mult=0.50,
    lc_incidence_mult_female=1.20,
    lc_base_fast_prob=9.0,
    lc_base_persistent_prob=7.0,
    reinfection_new_onset_mult=0.70,
    lc_onset_base_pct=15.0,
    # Vaccination
    efficiency_pct=80.0,
    boosted_pct=30.0,
    vaccination_decay=True,
    vaccine_priority=True,
    # Demographics
    gender=True,
    male_population_pct=49.5,
    age_distribution=True,
    age_range=100,
    age_infection_scaling=True,
    # Health risk levels
    risk_level_2_pct=4.0,
    risk_level_3_pct=40.0,
    risk_level_4_pct=6.0,
)

# Parameter ranges EXACTLY matching your plot
RANGES = {
    "covid_spread_chance_pct": [2, 5, 10, 20],      # Matches your plot
    "initial_infected_agents": [2, 5, 10, 20],      # FIXED: was [10, 25, 50, 100]
    "precaution_pct": [0, 30, 50, 80],              # Matches your plot
    "avg_degree": [10, 30, 50, 70],                 # Matches your plot
    "v_start_time": [0, 30, 180, 360],              # Matches your plot
    "vaccination_pct": [0, 30, 50, 80],             # Matches your plot
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
                out = simulate_netlogoish(**params)
                out.update(dict(param_name=pname, param_value=val, run=r))
                rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

# if __name__ == "__main__":
#     df = run_sweep(n_runs=8, seed=42, out_csv="results.csv")
#     print("Wrote", len(df), "rows to results.csv")

if __name__ == "__main__":
    df = run_sweep(n_runs=50, seed=42, out_csv="results.csv")
    print("Wrote", len(df), "rows to results.csv")
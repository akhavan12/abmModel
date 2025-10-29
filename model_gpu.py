"""
Parameter sweep runner for GPU-accelerated simulation
Handles 100,000+ agents efficiently
"""

import numpy as np
import pandas as pd
from sim_gpu_jax import simulate_gpu
import time

# Defaults for 100K agents
DEFAULTS = dict(
    N=100000,  # 100K agents!
    max_days=365,
    covid_spread_chance_pct=10.0,
    initial_infected_agents=50,  # Scale up initial infections
    precaution_pct=50.0,
    avg_degree=5,
    v_start_time=180,
    vaccination_pct=80.0,
    infected_period=10,
    active_duration=7,
    immune_period=21,
    asymptomatic_pct=40.0,
    # Long COVID
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
    vaccination_decay=True,
    vaccine_priority=True,
    # Demographics
    gender=True,
    age_distribution=True,
    age_range=100,
    age_infection_scaling=True,
)

# Parameter ranges for exploration
# RANGES = {
#     "covid_spread_chance_pct": [2, 5, 10, 20],
#     "initial_infected_agents": [10, 25, 50, 100],  # Scaled for 100K
#     "precaution_pct": [0, 30, 50, 80],
#     "avg_degree": [10, 30, 50, 70],
#     "v_start_time": [0, 30, 180, 360],
#     "vaccination_pct": [0, 30, 50, 80],
# }

RANGES = {
    "covid_spread_chance_pct": [2, 5, 10, 20],
    "initial_infected_agents": [2, 5, 10, 20],   # match CPU
    "precaution_pct": [0, 30, 50, 80],
    "avg_degree": [10, 30, 50, 70],
    "v_start_time": [0, 30, 180, 360],
    "vaccination_pct": [0, 30, 50, 80],
}


METRICS = ["runtime_days", "infected", "reinfected", "long_covid_cases", "min_productivity"]

def run_sweep(n_runs=20, seed=0, out_csv="results_100k.csv", N=100000):
    """
    Run parameter sweep with GPU acceleration
    
    Args:
        n_runs: Number of replications per parameter value
        seed: Random seed for reproducibility
        out_csv: Output CSV file
        N: Number of agents (default 100,000)
    """
    rows = []
    rs = np.random.RandomState(seed)
    
    total_sims = sum(len(values) * n_runs for values in RANGES.values())
    print(f"Running {total_sims} simulations with {N:,} agents each...")
    print(f"This will take approximately {total_sims * 0.5:.1f} minutes on GPU")
    
    sim_count = 0
    start_time = time.time()
    
    for pname, values in RANGES.items():
        print(f"\n=== Testing parameter: {pname} ===")
        
        for val in values:
            print(f"  Value: {val}")
            
            for r in range(n_runs):
                params = DEFAULTS.copy()
                params["N"] = N
                params[pname] = val
                params["seed"] = int(rs.randint(0, 2**31-1))
                
                try:
                    out = simulate_gpu(**params)
                    out.update(dict(param_name=pname, param_value=val, run=r))
                    rows.append(out)
                    
                    sim_count += 1
                    elapsed = time.time() - start_time
                    rate = sim_count / elapsed if elapsed > 0 else 0
                    eta = (total_sims - sim_count) / rate if rate > 0 else 0
                    
                    if sim_count % 5 == 0:
                        print(f"    Progress: {sim_count}/{total_sims} ({sim_count/total_sims*100:.1f}%) "
                              f"- ETA: {eta/60:.1f} min")
                
                except Exception as e:
                    print(f"    ERROR in run {r}: {e}")
                    continue
    
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n=== Complete ===")
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"Average time per simulation: {elapsed/sim_count:.2f} seconds")
    print(f"Wrote {len(df)} rows to {out_csv}")
    
    return df

def run_single_test(N=100000):
    """Quick single run to test GPU setup"""
    print(f"Running single test with {N:,} agents...")
    
    result = simulate_gpu(
        N=N,
        max_days=180,
        covid_spread_chance_pct=10.0,
        initial_infected_agents=int(N * 0.0005),  # 0.05% initial infection
        seed=42
    )
    
    print("\n=== Single Test Results ===")
    for key, val in result.items():
        if isinstance(val, float):
            print(f"{key}: {val:.2f}")
        else:
            print(f"{key}: {val:,}")
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run COVID ABM parameter sweep on GPU')
    parser.add_argument('--test', action='store_true', help='Run single test instead of full sweep')
    parser.add_argument('--N', type=int, default=100000, help='Number of agents (default: 100,000)')
    parser.add_argument('--runs', type=int, default=8, help='Number of runs per parameter (default: 8)')
    parser.add_argument('--output', type=str, default='results_100k.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    if args.test:
        run_single_test(N=args.N)
    else:
        df = run_sweep(n_runs=args.runs, seed=42, out_csv=args.output, N=args.N)

import pandas as pd
from abm.sim_netlogoish import simulate_netlogoish

def run_netlogoish_demo(out_csv="results_nlish.csv"):
    rows = []
    for val in [0.02,0.05,0.10,0.20]:
        out = simulate_netlogoish(
            N=1500,
            max_days=365,
            covid_spread_chance_pct=val*100,
            initial_infected_agents=10,
            precaution_pct=30,
            avg_degree=20,
            vaccination=True,
            v_start_time=30,
            vaccination_pct=30,
            infected_period=14,
            active_duration=7,
            seed=123
        )
        out.update(dict(param_name="covid_spread_chance", param_value=val, run=0))
        rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

if __name__ == "__main__":
    df = run_netlogoish_demo()
    print("Wrote", len(df), "rows")

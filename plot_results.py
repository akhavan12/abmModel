import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METRICS = [
    ("runtime_days", "Run time"),
    ("infected", "Infected cases"),
    ("reinfected", "Reinfected cases"),
    ("long_covid_cases", "Long covid cases"),
    ("min_productivity", "Min productivity"),
]

ORDER = {
    "covid_spread_chance_pct": [2, 5, 10, 20],
    "initial_infected_agents": [2, 5, 10, 20],
    "precaution_pct": [0, 30, 50, 80],
    "avg_degree": [10, 30, 50, 70],
    "v_start_time": [0, 30, 180, 360],
    "vaccination_pct": [0, 30, 50, 80],
}

LABELS = {
    "covid_spread_chance_pct": "COVID Spread\nChance%",
    "initial_infected_agents": "Initial Infected\nAgents",
    "precaution_pct": "Precaution%",
    "avg_degree": "Average\nDegree",  # FIXED: was "Temporal\nConnections"
    "v_start_time": "Vaccination\nStart Time",
    "vaccination_pct": "Vaccination%",
}

def make_grid(df, out_png="figure.png"):
    cols = list(ORDER.keys())
    rows = METRICS
    nrows = len(rows)
    ncols = len(cols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 9), squeeze=False)

    for c, pname in enumerate(cols):
        values = ORDER[pname]
        for r, (mkey, mlabel) in enumerate(rows):
            ax = axes[r, c]
            data = []
            means = []
            for v in values:
                d = df[(df["param_name"]==pname) & (df["param_value"]==v)][mkey].values
                data.append(d)
                means.append(np.mean(d) if len(d)>0 else np.nan)
            ax.boxplot(data, labels=[str(v) for v in values], showfliers=False)
            ax.plot(range(1, len(values)+1), means, marker="^", color='red', markersize=5)
            if r == 0:
                ax.set_title(LABELS[pname], fontsize=10)
            if c == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return out_png

if __name__ == "__main__":
    df = pd.read_csv("results_10k.csv")
    out = make_grid(df, out_png="figure.png")
    print("Saved", out)



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# METRICS = [
#     ("runtime_days", "Run time"),
#     ("infected", "Infected cases"),
#     ("reinfected", "Reinfected cases"),
#     ("long_covid_cases", "Long covid cases"),
#     ("min_productivity", "Min productivity"),
# ]

# ORDER = {
#     "covid_spread_chance_pct": [2, 5, 10, 20],
#     "initial_infected_agents": [2, 5, 10, 20],
#     "precaution_pct": [0, 30, 50, 80],
#     "avg_degree": [10, 30, 50, 70],
#     "v_start_time": [0, 30, 180, 360],
#     "vaccination_pct": [0, 30, 50, 80],
# }

# LABELS = {
#     "covid_spread_chance_pct": "COVID Spread\nChance%",
#     "initial_infected_agents": "Initial Infected\nAgents",
#     "precaution_pct": "Precaution%",
#     "avg_degree": "Temporal\nConnections",
#     "v_start_time": "Vaccination\nStart Time",
#     "vaccination_pct": "Vaccination%",
# }

# def make_grid(df, out_png="figure.png"):
#     cols = list(ORDER.keys())
#     rows = METRICS
#     nrows = len(rows)
#     ncols = len(cols)

#     fig, axes = plt.subplots(nrows, ncols, figsize=(12, 9), squeeze=False)

#     for c, pname in enumerate(cols):
#         values = ORDER[pname]
#         for r, (mkey, mlabel) in enumerate(rows):
#             ax = axes[r, c]
#             data = []
#             means = []
#             for v in values:
#                 d = df[(df["param_name"]==pname) & (df["param_value"]==v)][mkey].values
#                 data.append(d)
#                 means.append(np.mean(d) if len(d)>0 else np.nan)
#             ax.boxplot(data, labels=[str(v) for v in values], showfliers=False)
#             ax.plot(range(1, len(values)+1), means, marker="^")
#             if r == 0:
#                 ax.set_title(LABELS[pname])
#             if c == 0:
#                 ax.set_ylabel(mlabel)
#     fig.tight_layout()
#     fig.savefig(out_png, dpi=200, bbox_inches="tight")
#     return out_png

# if __name__ == "__main__":
#     df = pd.read_csv("results.csv")
#     out = make_grid(df, out_png="figure.png")
#     print("Saved", out)

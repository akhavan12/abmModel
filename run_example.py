
from sweep import run_sweep
from plot_results import make_grid
import pandas as pd

df = run_sweep(n_runs=6, seed=123, out_csv="results.csv")
png = make_grid(df, out_png="figure.png")
print("Wrote results.csv and", png)

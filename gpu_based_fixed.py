"""
gpu_based_fixed.py
------------------
GPU-accelerated ABM scaffold (CuPy if available; NumPy fallback).

Aligned behaviors:
- precautions reduce transmission probability
- stop when infected == 0
- immune waning via daily hazard (half-life)
- vaccination rolls out daily after v_start_time until target coverage
- long-COVID incidence determined at recovery time

Public API: simulate_netlogoish(**kwargs)
"""

from __future__ import annotations
import numpy as np
from scipy import sparse

# Optional GPU backend
try:
    import cupy as cp
    xp = cp
    SPARSE_GPU = True
except Exception:
    xp = np
    SPARSE_GPU = False


def _rng(seed=None):
    if xp is np:
        return np.random.default_rng(seed)
    return cp.random.RandomState(seed) if seed is not None else cp.random


def _to_xp(arr):
    return xp.asarray(arr) if not isinstance(arr, xp.ndarray) else arr


def _as_int(x):
    # robust conversion for CuPy/NumPy scalars
    try:
        return int(x.item())
    except Exception:
        return int(x)


def create_network_csr(N: int, avg_degree: float, seed=None):
    """
    Undirected Erdos–Renyi G(N, p) as CSR. Scales well to large N.
    Expected avg_degree ≈ p*(N-1).
    """
    import numpy as _np
    from scipy import sparse as _sp

    p_edge = min(1.0, float(avg_degree) / max(1, N - 1))
    rng = _np.random.default_rng(seed)

    # Upper-triangular random matrix then symmetrize; binarize
    A_triu = _sp.random(
        N, N, density=p_edge, format="csr",
        random_state=rng, data_rvs=lambda k: _np.ones(k, dtype=_np.uint8)
    )
    A_triu = _sp.triu(A_triu, k=1)
    A = A_triu + A_triu.T
    A.data[:] = 1
    return A


def simulate_netlogoish(
    # Population & time
    N=50_000,
    max_days=365,
    seed=None,

    # Transmission & course
    covid_spread_chance_pct=10.0,   # base per-contact infection chance (%)
    initial_infected_agents=25,
    avg_degree=8,
    infected_period=10,             # days until recovery
    incubation_period=4,
    active_duration=7,              # symptomatic window length (used for productivity proxy)

    # Mitigations
    precaution_pct=50.0,            # reduces transmission probability
    v_start_time=180,               # day vaccination rollout begins
    vaccination_pct=80.0,           # target coverage (% of population)
    vaccination_decay=True,         # efficacy wanes over time (simple linear)
    efficiency_pct=80.0,            # vaccine efficacy (%)
    daily_vax_pct_of_remaining=0.02,# share of remaining unvaxxed vaccinated per day (2%)

    # Immunity
    immune_wane_half_life_days=180, # half-life for losing post-infection immunity (days)

    # Long COVID knobs
    long_covid=True,
    lc_onset_base_pct=15.0,         # base % at recovery that become LC
    asymptomatic_pct=40.0,          # % of infections that are asymptomatic
    asymptomatic_lc_mult=0.50,      # multiplier on LC risk if asymptomatic
    lc_incidence_mult_female=1.20,  # multiplier for female agents
    reinfection_new_onset_mult=0.70,# multiplier on LC risk for reinfections

    # Demographics (placeholders; used for LC multipliers)
    age_distribution=True,
    age_range=95,
    gender=True,
    male_population_pct=49.5,

    # Not used directly here but preserved for signature compatibility
    symptomatic_duration_min=1,
    symptomatic_duration_mid=10,
    symptomatic_duration_max=60,
    risk_level_2_pct=4.0,
    risk_level_3_pct=40.0,
    risk_level_4_pct=6.0,
    boosted_pct=30.0,
    vaccine_priority=True,
):
    rs = _rng(seed)

    # Graph (CPU), state (xp)
    adj = create_network_csr(N, avg_degree, seed=seed)
    indptr = adj.indptr
    indices = adj.indices

    infected      = xp.zeros(N, dtype=bool)
    immuned       = xp.zeros(N, dtype=bool)
    symptomatic   = xp.zeros(N, dtype=bool)
    super_immune  = xp.zeros(N, dtype=bool)  # hook for future rules
    vaccinated    = xp.zeros(N, dtype=bool)

    # infection course
    virus_timer          = xp.zeros(N, dtype=xp.int32)
    infection_start_tick = xp.zeros(N, dtype=xp.int32)
    symptomatic_start    = xp.zeros(N, dtype=xp.int32)
    symptomatic_duration = xp.zeros(N, dtype=xp.int32)
    num_infections       = xp.zeros(N, dtype=xp.int32)
    vacc_time            = xp.zeros(N, dtype=xp.int32)

    # flags for the current infection (per-agent)
    current_asymptomatic = xp.zeros(N, dtype=bool)

    # Long-COVID tracking (unique, persistent)
    lc_flag       = xp.zeros(N, dtype=bool)
    lc_start_day  = xp.zeros(N, dtype=xp.int32)

    # Demographics (CPU -> xp)
    if age_distribution:
        age_bins = [(0,5,5.7),(5,15,12.5),(15,25,13.0),(25,35,13.7),
                    (35,45,13.1),(45,55,12.3),(55,65,12.9),(65,75,10.1),
                    (75,85,4.9),(85,95,1.8)]
        weights = np.array([w for _,_,w in age_bins], dtype=float)
        weights /= weights.sum()
        bins = [(lo,hi) for lo,hi,_ in age_bins]
        cpu_rng = np.random.default_rng(seed)
        idx = cpu_rng.choice(len(bins), size=N, p=weights)
        ages_cpu = np.array([cpu_rng.integers(bins[j][0], bins[j][1]) for j in idx], dtype=np.int32)
    else:
        ages_cpu = np.random.default_rng(seed).integers(0, age_range, size=N, dtype=np.int32)

    if gender:
        # genders: 0 = male, 1 = female
        genders_cpu = (np.random.default_rng(seed).random(N) >= male_population_pct/100.0).astype(np.int32)
    else:
        genders_cpu = np.zeros(N, np.int32)

    ages    = _to_xp(ages_cpu)
    genders = _to_xp(genders_cpu)

    # Initialize infections
    init_idx_cpu = np.random.default_rng(seed).choice(N, size=min(N, initial_infected_agents), replace=False)
    init_idx = _to_xp(init_idx_cpu)
    infected[init_idx] = True
    infection_start_tick[init_idx] = 1
    virus_timer[init_idx] = 1
    symptomatic_start[init_idx] = incubation_period
    symptomatic_duration[init_idx] = active_duration
    num_infections[init_idx] = 1

    # draw asymptomatic for these initial infections
    if asymptomatic_pct > 0:
        a_draw = xp.random.random(init_idx.size) * 100.0
        current_asymptomatic[init_idx] = a_draw < asymptomatic_pct
    else:
        current_asymptomatic[init_idx] = False

    # Trackers
    min_productivity = 100.0
    total_infected_first = int(init_idx_cpu.size)  # unique first-time infections
    total_reinfected = 0
    long_covid_cases = 0  # unique LC cases

    # Precompute target vaccination count
    target_covered = int(N * (vaccination_pct / 100.0)) if vaccination_pct > 0 else 0

    # Daily immunity waning probability from half-life
    if immune_wane_half_life_days and immune_wane_half_life_days > 0:
        wane_p = 1.0 - (0.5 ** (1.0 / immune_wane_half_life_days))
    else:
        wane_p = 0.0

    base_lc = lc_onset_base_pct / 100.0

    for day in range(1, max_days + 1):

        # --- Vaccination rollout (ongoing after v_start_time) ---
        if day >= v_start_time and target_covered > 0:
            already = _as_int(vaccinated.sum())
            need = target_covered - already
            if need > 0:
                unvacc = xp.where(~vaccinated)[0]
                remaining = int(unvacc.size)
                if remaining > 0:
                    dose_today = max(1, int(min(need, remaining * daily_vax_pct_of_remaining)))
                    if hasattr(xp.random, "permutation"):
                        perm = xp.random.permutation(remaining)
                    else:
                        perm = np.random.permutation(remaining)
                    pick = unvacc[perm[:dose_today]]
                    vaccinated[pick] = True
                    vacc_time[pick] = 1

        # --- Transmission attempts ---
        # Effective spread after precautions
        eff_spread_pct = covid_spread_chance_pct * (1.0 - precaution_pct / 100.0)
        eff_spread_pct = max(0.0, min(100.0, eff_spread_pct))

        inf_idx = xp.where(infected)[0]
        for a in inf_idx.tolist():
            neigh = indices[indptr[int(a)]: indptr[int(a) + 1]]
            if neigh.size == 0:
                continue
            neigh_x = _to_xp(neigh)

            susceptible = (~infected[neigh_x]) & (~immuned[neigh_x]) & (~super_immune[neigh_x])
            neigh_susc = neigh_x[susceptible]
            if neigh_susc.size == 0:
                continue

            # Vaccine protection on susceptibles
            if day >= v_start_time and target_covered > 0:
                if vaccination_decay:
                    # linear decay by days since vaccination
                    real_eff = xp.maximum(0.0, (efficiency_pct - 0.11 * xp.asarray(vacc_time[neigh_susc])))
                else:
                    real_eff = xp.asarray(efficiency_pct)

                vacc_rand = xp.random.random(neigh_susc.size) * 100.0
                vacc_protected = vaccinated[neigh_susc] & (vacc_rand < real_eff)
            else:
                vacc_protected = xp.zeros(neigh_susc.size, dtype=bool)

            atk_candidates = neigh_susc[~vacc_protected]
            if atk_candidates.size == 0:
                continue

            draws = xp.random.random(atk_candidates.size) * 100.0
            newly = atk_candidates[draws < eff_spread_pct]

            if newly.size > 0:
                # first vs re-infections
                prev_inf  = num_infections[newly] > 0
                first_inf = newly[~prev_inf]

                total_reinfected += int(prev_inf.sum())
                total_infected_first += int(first_inf.size)

                infected[newly]              = True
                infection_start_tick[newly]  = day
                virus_timer[newly]           = 1
                symptomatic_start[newly]     = incubation_period
                symptomatic_duration[newly]  = active_duration
                num_infections[newly]       += 1

                # assign asymptomatic for the current infection
                if asymptomatic_pct > 0:
                    ad = xp.random.random(newly.size) * 100.0
                    current_asymptomatic[newly] = ad < asymptomatic_pct
                else:
                    current_asymptomatic[newly] = False

        # --- Timers & transitions ---
        infected_idx = xp.where(infected)[0]
        if infected_idx.size > 0:
            virus_timer[infected_idx] += 1
            symptomatic[infected_idx] = (virus_timer[infected_idx] >= symptomatic_start[infected_idx]) & \
                                        (virus_timer[infected_idx] < symptomatic_start[infected_idx] + symptomatic_duration[infected_idx])

            # Recover today
            done = infected_idx[virus_timer[infected_idx] > infected_period]
            if done.size > 0:
                infected[done] = False
                immuned[done] = True

                # ---- Long-COVID incidence at recovery ----
                if long_covid and base_lc > 0:
                    # Only agents who do not already have LC can get a new onset
                    cand = done[~lc_flag[done]]
                    if cand.size > 0:
                        # Start from base probability
                        p = xp.full(cand.size, base_lc)

                        # Female multiplier (genders: 1=female)
                        if gender:
                            female = (genders[cand] == 1)
                            p = xp.where(female, p * lc_incidence_mult_female, p)

                        # Asymptomatic multiplier (usually reduces risk)
                        p = xp.where(current_asymptomatic[cand], p * asymptomatic_lc_mult, p)

                        # Reinfection multiplier (often reduces risk of NEW LC)
                        reinf = (num_infections[cand] > 1)
                        p = xp.where(reinf, p * reinfection_new_onset_mult, p)

                        # Bound [0,1]
                        p = xp.clip(p, 0.0, 1.0)

                        # Draw new onsets
                        rd = xp.random.random(cand.size)
                        lc_new_mask = rd < p
                        if lc_new_mask.any():
                            lc_new = cand[lc_new_mask]
                            lc_flag[lc_new] = True
                            lc_start_day[lc_new] = day
                            long_covid_cases += int(lc_new.size)

                # clear current-infection flags for those recovered
                current_asymptomatic[done] = False

        # --- Immunity waning (hazard) ---
        if wane_p > 0.0:
            imm_idx = xp.where(immuned)[0]
            if imm_idx.size > 0:
                wdraw = xp.random.random(imm_idx.size)
                lose = imm_idx[wdraw < wane_p]
                if lose.size > 0:
                    immuned[lose] = False

        # --- Vaccine timer increments ---
        vacc_idx = xp.where(vaccinated)[0]
        if vacc_idx.size > 0:
            vacc_time[vacc_idx] += 1

        # --- Productivity proxy (symptomatic fraction) ---
        symptomatic_rate = float(symptomatic.sum()) / float(N)
        min_productivity = min(min_productivity, max(0.0, 100.0 - 50.0 * symptomatic_rate))

        # --- Stop when outbreak ends ---
        if infected.sum() == 0:
            break

    return {
        "runtime_days": int(day),
        "infected": int(total_infected_first),   # unique first-time infections
        "reinfected": int(total_reinfected),
        "long_covid_cases": int(long_covid_cases),
        "min_productivity": float(min_productivity),
        "backend": "cupy" if xp is not np else "numpy",
        "sparse_gpu": bool(SPARSE_GPU),
    }


if __name__ == "__main__":
    out = simulate_netlogoish(
        N=5000, max_days=180, avg_degree=6,
        initial_infected_agents=10, precaution_pct=30.0,
        vaccination_pct=70.0, v_start_time=60, seed=42
    )
    print(out)

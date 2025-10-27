
import numpy as np

class NLState:
    SUS = 0
    INF = 1
    IMM = 2

def simulate_netlogoish(
    N=1000,
    max_days=365,
    covid_spread_chance_pct=5.0,     # matches NetLogo's % slider
    initial_infected_agents=10,
    precaution_pct=0.0,              # % chance to skip symptomatic contacts
    avg_degree=20,                   # M1-average-connections
    vaccination=True,
    v_start_time=0,                  # day to vaccinate
    vaccination_pct=0.0,             # % of population to vaccinate at v_start_time
    infected_period=14,              # "Infected-period"
    active_duration=7,               # "Active-duration" (infectious window length)
    symptomatic_start=2,             # days after infection when symptoms begin (0 => asymptomatic)
    symptomatic_duration_mid=5,      # used to shape duration (simplified)
    seed=None,
):
    rng = np.random.default_rng(seed)

    # --- Agent state ---
    state = np.zeros(N, dtype=np.int8)   # 0 S, 1 I, 2 R
    virus_timer = np.zeros(N, dtype=np.int16)  # ticks since infection start
    infectious_start = np.ones(N, dtype=np.int16)  # NetLogo sets start=1
    transfer_active_duration = np.minimum(
        np.maximum(1, active_duration), np.maximum(1, infected_period - 1)
    ).astype(np.int16)
    infectious_end = infectious_start + transfer_active_duration
    symptomatic_start_arr = np.full(N, symptomatic_start, dtype=np.int16)
    vaccinated = np.zeros(N, dtype=bool)

    # seed infections (don’t filter by vaccination)
    chosen = rng.choice(N, size=min(N, initial_infected_agents), replace=False)
    state[chosen] = NLState.INF
    virus_timer[chosen] = 0

    # --- Network (static M1) ---
    # simple Erdos–Renyi with expected avg_degree
    p = min(1.0, avg_degree / max(1, N - 1))
    # adjacency as list of neighbors to avoid huge dense matrices
    neighbors = [set() for _ in range(N)]
    for i in range(N):
        # Add edges only j>i to avoid duplicates
        js = rng.choice(N - i - 1, size=max(0, int(p * (N - i - 1))), replace=False)
        for off in js:
            j = i + 1 + int(off)
            neighbors[i].add(j)
            neighbors[j].add(i)

    # --- Metrics ---
    total_infected = int(np.count_nonzero(state == NLState.INF))
    total_reinfected = 0
    long_covid_cases = 0    # placeholder: exact LC rules omitted here
    min_productivity = 100.0

    day = 0
    while day < max_days:
        # (1) set-temporal-links -> ignore (we keep static M1)

        # (2) do-long-covid-checks (skipped for parity scaffold)

        # (3) Vaccination event
        if vaccination and day == v_start_time and vaccination_pct > 0.0:
            k = int(N * vaccination_pct / 100.0)
            if k > 0:
                idx = rng.choice(N, size=k, replace=False)
                vaccinated[idx] = True

        # (4) transmission & acute state updates: NetLogo's 'spread-virus-days'
        inf_idx = np.flatnonzero(
            (state == NLState.INF)
            & (virus_timer >= infectious_start)
            & (virus_timer < infectious_end)
        )
        if inf_idx.size == 0:
            break

        new_inf = []
        for i in inf_idx:
            # if symptomatic and beyond start, optionally skip contacts by precaution
            if (symptomatic_start_arr[i] > 0) and (virus_timer[i] > symptomatic_start_arr[i]):
                if rng.random() * 100.0 < precaution_pct:
                    continue
            # try infect neighbors that are S and not vaccinated (or draw through efficacy)
            for j in neighbors[i]:
                if state[j] != NLState.SUS:
                    continue
                # vaccine gate: assume 60% skip chance if vaccinated
                if vaccinated[j] and (rng.random() < 0.60):
                    continue
                beta = covid_spread_chance_pct / 100.0
                if rng.random() < beta:
                    new_inf.append(j)

        if new_inf:
            new_inf = np.unique(np.asarray(new_inf, dtype=np.int32))
            # reinfection count (anyone not S would be reinfection, but we filtered S above)
            total_infected += int(new_inf.size)
            state[new_inf] = NLState.INF
            virus_timer[new_inf] = 0

        # Recovery/Immunity (end of day)
        # increase timers
        virus_timer[state == NLState.INF] += 1
        # move to immune at end of infected_period
        rec = np.flatnonzero((state == NLState.INF) & (virus_timer >= infected_period))
        if rec.size > 0:
            state[rec] = NLState.IMM
            virus_timer[rec] = 0

        # immunity waning → become susceptible again after infected_period (proxy for immunity_days)
        # (NetLogo uses separate immunity window; we mirror via reuse of virus_timer here)
        waned = np.flatnonzero((state == NLState.IMM) & (virus_timer >= infected_period))
        if waned.size > 0:
            state[waned] = NLState.SUS
            virus_timer[waned] = 0

        # productivity proxy
        current_prod = 100.0 - 10.0 * (np.count_nonzero(state == NLState.INF) / N)
        min_productivity = min(min_productivity, float(current_prod))

        day += 1

    out = dict(
        runtime_days=int(day),
        infected=int(total_infected),
        reinfected=int(total_reinfected),
        long_covid_cases=int(long_covid_cases),
        min_productivity=float(min_productivity),
    )
    return out

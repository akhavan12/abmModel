
import numpy as np

def simulate(
    N=1000,
    max_days=365,
    covid_spread_chance=0.05,     # base per-contact infection prob
    initial_infected_agents=10,
    precaution_pct=0.0,           # reduces spread chance linearly
    temporal_connections=20,      # random contacts per agent/day
    vaccination_start_time=0,     # day when vaccination starts
    vaccination_pct=0.0,          # % of population vaccinated at start day
    infection_duration_mean=7.0,
    infection_duration_sd=2.0,
    immunity_days=90,             # immunity after recovery
    long_covid_prob=0.10,         # probability an infection yields long-covid
    long_covid_severity_mean=0.3, # fraction productivity drop [0..1]
    prod_recovery_rate=0.002,     # per day recovery for productivity
    seed=None,
    xp=None,                      # pass cupy as xp to run on GPU; defaults to numpy
):
    """
    Vectorized agent-based model.
    States: 0=S, 1=I, 2=R (immune). Reinfection after immunity wanes.
    Long-covid modeled as persistent productivity penalty that slowly recovers.
    """
    if xp is None:
        xp = np

    rs = xp.random.RandomState(seed)

    # Agent arrays
    state = xp.zeros(N, dtype=xp.int8)
    days_in_state = xp.zeros(N, dtype=xp.int16)   # counts days in current state
    immunity_left = xp.zeros(N, dtype=xp.int16)   # days of immunity remaining
    infected_days_left = xp.zeros(N, dtype=xp.int16)

    vaccinated = xp.zeros(N, dtype=bool)
    prod = xp.ones(N, dtype=xp.float32) * 100.0

    long_covid = xp.zeros(N, dtype=bool)
    lc_severity = xp.zeros(N, dtype=xp.float32)   # 0..1 penalty
    lc_duration = xp.zeros(N, dtype=xp.int32)

    # seed infections
    if initial_infected_agents > 0:
        idx = rs.choice(N, size=min(N, initial_infected_agents), replace=False)
        state[idx] = 1
        # sample infection durations
        dur = xp.clip(rs.normal(infection_duration_mean, infection_duration_sd, size=idx.size), 2, 30).astype(xp.int16)
        infected_days_left[idx] = dur

    # metrics
    total_infected = int(np.count_nonzero(state == 1))
    total_reinfected = 0
    total_lc = 0
    min_prod = 100.0

    # precompute modifiers
    spread_modifier = 1.0 - (precaution_pct / 100.0)
    spread_modifier = max(0.0, spread_modifier)

    # runtime
    day = 0
    while day < max_days:
        # vaccination event
        if day == vaccination_start_time and vaccination_pct > 0.0:
            k = int(N * vaccination_pct / 100.0)
            if k > 0:
                v_idx = rs.choice(N, size=k, replace=False)
                vaccinated[v_idx] = True

        # Contacts: sample contacts for each agent (well-mixed approximation)
        # For speed, estimate exposures using binomial draws against prevalence
        I = xp.count_nonzero(state == 1)
        if I == 0:
            break

        prevalence = I / N
        # Each susceptible agent has 'temporal_connections' Bernoulli contacts;
        # approximating num infectious contacts as Binomial(connections, prevalence)
        susc = (state == 0)
        if xp.any(susc):
            infectious_contacts = rs.binomial(
                temporal_connections,
                prevalence,
                size=int(xp.count_nonzero(susc))
            )
            # per-contact prob -> prob of at least one transmission
            beta = covid_spread_chance * spread_modifier
            # vaccination cuts per-contact prob by 60% (simple assumption)
            s_idx = xp.flatnonzero(susc)
            vfac = xp.where(vaccinated[s_idx], 0.4, 1.0)  # vaccinated -> less likely
            # P(infection) = 1-(1 - beta*vfac)^{num_contacts}
            pinf = 1.0 - xp.power((1.0 - beta * vfac), infectious_contacts)
            new_inf_draw = rs.uniform(size=pinf.shape) < pinf
            new_infected_idx = s_idx[new_inf_draw]
        else:
            new_infected_idx = xp.array([], dtype=xp.int64)

        # progress infections
        inf_idx = xp.flatnonzero(state == 1)
        if inf_idx.size > 0:
            infected_days_left[inf_idx] -= 1
            # recovery
            rec_mask = infected_days_left[inf_idx] <= 0
            if xp.any(rec_mask):
                rec_idx = inf_idx[rec_mask]
                state[rec_idx] = 2
                immunity_left[rec_idx] = immunity_days
                days_in_state[rec_idx] = 0

        # progress immunity
        R_idx = xp.flatnonzero(state == 2)
        if R_idx.size > 0:
            immunity_left[R_idx] -= 1
            waned = R_idx[immunity_left[R_idx] <= 0]
            if waned.size > 0:
                state[waned] = 0
                days_in_state[waned] = 0
                immunity_left[waned] = 0

        # apply new infections (after recovery/waning to allow reinfection)
        if new_infected_idx.size > 0:
            # reinfection tracking: those not susceptible were reinfected
            reinf = xp.count_nonzero(state[new_infected_idx] != 0)
            total_reinfected += int(reinf)
            # set new infections
            state[new_infected_idx] = 1
            dur = xp.clip(
                rs.normal(infection_duration_mean, infection_duration_sd, size=new_infected_idx.size),
                2, 30
            ).astype(xp.int16)
            infected_days_left[new_infected_idx] = dur
            days_in_state[new_infected_idx] = 0
            total_infected += int(new_infected_idx.size)

            # long covid assignment for a fraction of new infections
            lc_draw = rs.uniform(size=new_infected_idx.size) < long_covid_prob
            if xp.any(lc_draw):
                lc_idx = new_infected_idx[lc_draw]
                # persistent flag latches
                long_covid[lc_idx] = True
                # if first time, assign severity
                new_sev = xp.clip(rs.normal(long_covid_severity_mean, 0.15, size=lc_idx.size), 0.05, 0.9)
                # keep max severity seen so far
                lc_severity[lc_idx] = xp.maximum(lc_severity[lc_idx], new_sev)

        # productivity dynamics:
        # active long-covid agents have productivity reduced; gradually recovers
        if xp.any(long_covid):
            lc_agents = xp.flatnonzero(long_covid)
            # target productivity = 100 * (1 - severity)
            target = 100.0 * (1.0 - lc_severity[lc_agents])
            # move current productivity a bit toward target (recovery)
            prod[lc_agents] = prod[lc_agents] + prod_recovery_rate * (target - prod[lc_agents])
            lc_duration[lc_agents] += 1

        # infected agents get temporary productivity dip (10%)
        if inf_idx.size > 0:
            prod[inf_idx] = np.minimum(prod[inf_idx], 90.0)

        # cap and floor
        prod = np.clip(prod, 50.0, 100.0)
        min_prod = float(min(min_prod, np.mean(prod)))

        day += 1

    # total long covid cases = agents ever flagged
    total_lc = int(np.count_nonzero(long_covid))
    out = {
        "runtime_days": int(day),
        "infected": int(total_infected),
        "reinfected": int(total_reinfected),
        "long_covid_cases": int(total_lc),
        "min_productivity": float(min_prod),
    }
    return out

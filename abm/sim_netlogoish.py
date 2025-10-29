import numpy as np
from scipy.stats import truncnorm

class AgentState:
    """Agent health states matching NetLogo"""
    SUSCEPTIBLE = 0
    INFECTED = 1
    IMMUNE = 2
    SUPER_IMMUNE = 3

class Agent:
    """Individual agent with all NetLogo turtle attributes"""
    def __init__(self, idx):
        self.idx = idx
        
        # Core state flags
        self.infected = False
        self.immuned = False
        self.symptomatic = False
        self.super_immune = False
        
        # Long COVID
        self.persistent_long_covid = False
        self.long_covid_severity = 0.0
        self.long_covid_duration = 0
        self.long_covid_recovery_group = -1  # 0=fast, 1=gradual, 2=persistent
        self.long_covid_weibull_k = 0.0
        self.long_covid_weibull_lambda = 0.0
        self.lc_pending = False
        self.lc_onset_day = 0
        
        # Infection tracking
        self.virus_check_timer = 0
        self.number_of_infection = 0
        self.infection_start_tick = 0
        self.initial_infections = 0
        
        # Infectious period
        self.infectious_start = 1
        self.infectious_end = 1
        self.transfer_active_duration = 0
        
        # Symptom timing
        self.symptomatic_start = 0
        self.symptomatic_duration = 0
        
        # Demographics
        self.age = 0
        self.gender = 0  # 0=male, 1=female
        self.health_risk_level = 1  # 1=baseline, 2=pregnancy, 3-4=higher risk
        
        # Age weights for infection probability
        self.covid_age_prob = 15.0
        self.us_age_prob = 13.0
        
        # Vaccination
        self.vaccinated = False
        self.vaccinated_time = 0
        
        # Network
        self.neighbors = set()

def simulate_netlogoish(
    N=1000,
    max_days=365,
    covid_spread_chance_pct=5.0,
    initial_infected_agents=10,
    precaution_pct=0.0,
    avg_degree=20,
    v_start_time=0,
    vaccination_pct=0.0,
    infected_period=14,
    active_duration=7,
    symptomatic_start=2,
    # Additional NetLogo parameters
    asymptomatic_pct=20.0,
    incubation_period=5,
    symptomatic_duration_min=3,
    symptomatic_duration_mid=7,
    symptomatic_duration_max=21,
    symptomatic_duration_dev=3,
    immune_period=180,
    effect_of_reinfection=0,
    # Vaccination parameters
    efficiency_pct=60.0,
    vaccination_decay=True,
    boosted_pct=0.0,
    vaccine_priority=False,
    # Long COVID parameters
    long_covid=True,
    lc_onset_base_pct=10.0,
    long_covid_time_threshold=28,
    lc_base_fast_prob=40.0,
    lc_base_persistent_prob=20.0,
    lc_incidence_mult_female=1.5,
    reinfection_new_onset_mult=0.5,
    asymptomatic_lc_mult=0.3,
    # Demographics
    age_distribution=True,
    age_range=95,
    gender=True,
    male_population_pct=49.0,
    age_infection_scaling=True,
    risk_level_2_pct=5.0,
    risk_level_3_pct=10.0,
    risk_level_4_pct=5.0,
    super_immune_pct=0.0,
    # Network (temporal links not implemented for performance)
    temporal_connections_pct=0.0,
    seed=None,
):
    """
    Faithful reimplementation of the NetLogo COVID ABM model.
    
    Returns dict with: runtime_days, infected, reinfected, long_covid_cases, min_productivity
    """
    rng = np.random.default_rng(seed)
    
    # Create agents
    agents = [Agent(i) for i in range(N)]
    
    # ========== SETUP NETWORK (M1 style) ==========
    # Simple Erdős-Rényi approximation of NetLogo's M1
    p = min(1.0, avg_degree / max(1, N - 1))
    for i in range(N):
        # Sample edges to higher-indexed nodes to avoid duplicates
        n_edges = rng.binomial(N - i - 1, p)
        if n_edges > 0:
            targets = rng.choice(np.arange(i + 1, N), size=min(n_edges, N - i - 1), replace=False)
            for j in targets:
                agents[i].neighbors.add(j)
                agents[j].neighbors.add(i)
    
    # ========== SETUP DEMOGRAPHICS ==========
    
    # Age distribution (US-like from NetLogo)
    age_bins = [(0, 5, 5.7), (5, 15, 12.5), (15, 25, 13.0), (25, 35, 13.7),
                (35, 45, 13.1), (45, 55, 12.3), (55, 65, 12.9), (65, 75, 10.1),
                (75, 85, 4.9), (85, 95, 1.8)]
    
    if age_distribution:
        # Weighted sampling by age bins
        bins = []
        weights = []
        for low, high, weight in age_bins:
            bins.append((low, high))
            weights.append(weight)
        weights = np.array(weights) / sum(weights)
        
        for agent in agents:
            bin_idx = rng.choice(len(bins), p=weights)
            low, high = bins[bin_idx]
            agent.age = rng.integers(low, high)
    else:
        for agent in agents:
            agent.age = rng.integers(0, age_range)
    
    # Gender
    if gender:
        male_prob = male_population_pct / 100.0
        for agent in agents:
            agent.gender = 0 if rng.random() < male_prob else 1
    else:
        for agent in agents:
            agent.gender = 0
    
    # Set age-based infection probabilities
    for agent in agents:
        _set_covid_age_prob(agent)
        _set_us_age_prob(agent)
    
    # Health risk levels (mutually exclusive)
    # Level 2: pregnancy (females 15-49)
    if gender:
        eligible_level2 = [a for a in agents if a.gender == 1 and 15 <= a.age <= 49]
    else:
        eligible_level2 = agents[:]
    
    n2 = min(int(risk_level_2_pct * N / 100), len(eligible_level2))
    if n2 > 0:
        selected = rng.choice(eligible_level2, size=n2, replace=False)
        for agent in selected:
            agent.health_risk_level = 2
    
    # Level 3
    eligible_level3 = [a for a in agents if a.health_risk_level == 1]
    n3 = min(int(risk_level_3_pct * N / 100), len(eligible_level3))
    if n3 > 0:
        selected = rng.choice(eligible_level3, size=n3, replace=False)
        for agent in selected:
            agent.health_risk_level = 3
    
    # Level 4
    eligible_level4 = [a for a in agents if a.health_risk_level == 1]
    n4 = min(int(risk_level_4_pct * N / 100), len(eligible_level4))
    if n4 > 0:
        selected = rng.choice(eligible_level4, size=n4, replace=False)
        for agent in selected:
            agent.health_risk_level = 4
    
    # Super-immune agents
    n_super = int(super_immune_pct * N / 100)
    if n_super > 0:
        selected = rng.choice(agents, size=n_super, replace=False)
        for agent in selected:
            agent.super_immune = True
    
    # ========== SEED INITIAL INFECTIONS ==========
    eligible_initial = [a for a in agents if not a.super_immune]
    n_initial = min(initial_infected_agents, len(eligible_initial))
    if n_initial > 0:
        initial_inf = rng.choice(eligible_initial, size=n_initial, replace=False)
        for agent in initial_inf:
            _become_infected_nonsymptomatic(agent, 0, infected_period, active_duration, rng)
            agent.virus_check_timer = 0
            agent.number_of_infection = 1
            agent.initial_infections = 1
    
    # ========== METRICS ==========
    total_infected = n_initial
    total_reinfected = 0
    long_covid_cases = 0
    min_productivity = 100.0
    
    # ========== MAIN LOOP ==========
    for day in range(max_days):
        
        # (1) Vaccination at start time
        if day == v_start_time and vaccination_pct > 0.0:
            _apply_vaccination(agents, vaccination_pct, vaccine_priority, rng)
        
        # (2) Long COVID progression
        if long_covid:
            _do_long_covid_checks(agents, rng)
        
        # (3) Transmission
        new_infections = _spread_virus_days(
            agents, day, v_start_time, precaution_pct, covid_spread_chance_pct,
            efficiency_pct, vaccination_decay, age_infection_scaling, rng
        )
        
        for agent in new_infections:
            if agent.number_of_infection > 0:
                total_reinfected += 1
            total_infected += 1
        
        # (4) Acute infection state updates
        _do_virus_checks_infected(
            agents, day, infected_period, active_duration, asymptomatic_pct,
            incubation_period, symptomatic_duration_min, symptomatic_duration_mid,
            symptomatic_duration_max, symptomatic_duration_dev, effect_of_reinfection,
            long_covid, lc_onset_base_pct, long_covid_time_threshold,
            lc_incidence_mult_female, reinfection_new_onset_mult, asymptomatic_lc_mult,
            lc_base_fast_prob, lc_base_persistent_prob, rng
        )
        
        # (5) Immunity state updates
        _do_virus_checks_immuned(agents, infected_period, immune_period)
        
        # (6) Vaccination time tracking and boosters
        _check_vaccination_time(agents, boosted_pct, rng)
        
        # (7) Process pending LC onsets
        if long_covid:
            new_lc = _process_pending_lc(agents, day)
            long_covid_cases += new_lc
        
        # (8) Update productivity metrics
        current_prod = _update_productivity(agents)
        min_productivity = min(min_productivity, current_prod)
        
        # (9) Check stopping condition (no active infections or immunity)
        any_infected = any(a.infected for a in agents)
        any_immuned = any(a.immuned for a in agents)
        if not any_infected and not any_immuned:
            break
    
    return {
        'runtime_days': day + 1,
        'infected': total_infected,
        'reinfected': total_reinfected,
        'long_covid_cases': long_covid_cases,
        'min_productivity': min_productivity,
    }


# ========== HELPER FUNCTIONS ==========

def _set_covid_age_prob(agent):
    """Set age-specific COVID contact risk weight"""
    a = agent.age
    if a < 10:
        agent.covid_age_prob = 2.3
    elif a < 20:
        agent.covid_age_prob = 5.1
    elif a < 30:
        agent.covid_age_prob = 15.5
    elif a < 40:
        agent.covid_age_prob = 16.9
    elif a < 50:
        agent.covid_age_prob = 16.4
    elif a < 60:
        agent.covid_age_prob = 16.4
    elif a < 70:
        agent.covid_age_prob = 11.9
    elif a < 80:
        agent.covid_age_prob = 7.0
    else:
        agent.covid_age_prob = 8.5

def _set_us_age_prob(agent):
    """Set baseline population age weight"""
    a = agent.age
    if a < 5:
        agent.us_age_prob = 5.7
    elif a < 15:
        agent.us_age_prob = 12.5
    elif a < 25:
        agent.us_age_prob = 13.0
    elif a < 35:
        agent.us_age_prob = 13.7
    elif a < 45:
        agent.us_age_prob = 13.1
    elif a < 55:
        agent.us_age_prob = 12.3
    elif a < 65:
        agent.us_age_prob = 12.9
    elif a < 75:
        agent.us_age_prob = 10.1
    elif a < 85:
        agent.us_age_prob = 4.9
    else:
        agent.us_age_prob = 1.8

def _become_infected_nonsymptomatic(agent, current_tick, infected_period, active_duration, rng):
    """Transition to infected (non-symptomatic) state"""
    agent.infectious_start = 1
    drawn_length = 1 + rng.integers(0, active_duration)
    max_length = max(1, infected_period - agent.infectious_start)
    agent.transfer_active_duration = min(drawn_length, max_length)
    agent.infectious_end = agent.infectious_start + agent.transfer_active_duration
    
    agent.infected = True
    agent.immuned = False
    agent.symptomatic = False
    agent.super_immune = False
    agent.infection_start_tick = current_tick
    
    # Preserve persistent LC if exists, otherwise clear LC params
    if not agent.persistent_long_covid:
        agent.long_covid_recovery_group = -1
        agent.long_covid_weibull_k = 0
        agent.long_covid_weibull_lambda = 0

def _become_infected_symptomatic(agent, current_tick, infected_period, active_duration, rng):
    """Transition to infected (symptomatic) state"""
    agent.infectious_start = 1
    drawn_length = 1 + rng.integers(0, active_duration)
    max_length = max(1, infected_period - agent.infectious_start)
    agent.transfer_active_duration = min(drawn_length, max_length)
    agent.infectious_end = agent.infectious_start + agent.transfer_active_duration
    
    agent.infected = True
    agent.immuned = False
    agent.symptomatic = True
    agent.super_immune = False
    agent.infection_start_tick = current_tick
    
    if not agent.persistent_long_covid:
        agent.long_covid_recovery_group = -1
        agent.long_covid_weibull_k = 0
        agent.long_covid_weibull_lambda = 0

def _become_susceptible(agent):
    """Transition to susceptible state"""
    agent.infected = False
    agent.immuned = False
    agent.symptomatic = False
    agent.super_immune = False
    # PRESERVE persistent LC status

def _become_immuned(agent):
    """Transition to immune (non-symptomatic) state"""
    agent.infected = False
    agent.immuned = True
    agent.symptomatic = False
    agent.super_immune = False
    # PRESERVE persistent LC status

def _become_immuned_symptomatic(agent):
    """Transition to immune (still symptomatic) state"""
    agent.infected = False
    agent.immuned = True
    agent.super_immune = False
    # Keep symptomatic = True
    # PRESERVE persistent LC status

def _apply_vaccination(agents, vaccination_pct, vaccine_priority, rng):
    """Apply vaccination to agents"""
    target_count = int(len(agents) * vaccination_pct / 100.0)
    vaccinated_count = sum(1 for a in agents if a.vaccinated)
    
    if vaccinated_count >= target_count:
        return
    
    unvaccinated = [a for a in agents if not a.vaccinated]
    
    if vaccine_priority:
        # Priority order: 65+, then risk 4/3/2/1 under 65
        priority_groups = [
            [a for a in unvaccinated if a.age >= 65],
            [a for a in unvaccinated if a.health_risk_level == 4 and a.age < 65],
            [a for a in unvaccinated if a.health_risk_level == 3 and a.age < 65],
            [a for a in unvaccinated if a.health_risk_level == 2 and a.age < 65],
            [a for a in unvaccinated if a.health_risk_level == 1 and a.age < 65],
        ]
        
        for group in priority_groups:
            for agent in group:
                if vaccinated_count >= target_count:
                    return
                agent.vaccinated = True
                agent.vaccinated_time = 1
                vaccinated_count += 1
    else:
        # Random vaccination
        n_to_vaccinate = min(target_count - vaccinated_count, len(unvaccinated))
        if n_to_vaccinate > 0:
            selected = rng.choice(unvaccinated, size=n_to_vaccinate, replace=False)
            for agent in selected:
                agent.vaccinated = True
                agent.vaccinated_time = 1

def _spread_virus_days(agents, current_day, v_start_time, precaution_pct,
                       covid_spread_chance_pct, efficiency_pct, vaccination_decay,
                       age_infection_scaling, rng):
    """Transmission process for day timestep"""
    new_infections = []
    
    for agent in agents:
        if not agent.infected:
            continue
        if agent.virus_check_timer < agent.infectious_start:
            continue
        if agent.virus_check_timer >= agent.infectious_end:
            continue
        
        # Precaution check for symptomatic agents
        if agent.symptomatic_start > 0 and agent.virus_check_timer > agent.symptomatic_start:
            if rng.random() * 100 < precaution_pct:
                continue
        
        # Try to infect neighbors
        for neighbor_idx in agent.neighbors:
            neighbor = agents[neighbor_idx]
            
            # Skip if not susceptible
            if neighbor.infected or neighbor.immuned or neighbor.super_immune:
                continue
            
            # Vaccine efficacy check
            if current_day >= v_start_time and neighbor.vaccinated:
                if vaccination_decay:
                    real_efficiency = max(0, min(100, efficiency_pct - 0.11 * neighbor.vaccinated_time))
                else:
                    real_efficiency = efficiency_pct
                
                if rng.random() * 100 < real_efficiency:
                    continue
            
            # Calculate infection probability
            infection_prob = covid_spread_chance_pct
            
            if age_infection_scaling:
                infection_prob *= neighbor.covid_age_prob / (neighbor.us_age_prob + 1e-9)
            
            infection_prob = max(0, min(100, infection_prob))
            
            # Attempt infection
            if rng.random() * 100 < infection_prob:
                _become_infected_nonsymptomatic(neighbor, current_day, 
                                               14, 7, rng)  # Use defaults for now
                neighbor.virus_check_timer = 0
                neighbor.number_of_infection += 1
                new_infections.append(neighbor)
    
    return new_infections

def _truncated_normal(mean, std, low, high, rng):
    """Sample from truncated normal distribution"""
    if std <= 0:
        return mean
    a = (low - mean) / std
    b = (high - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng)

def _do_virus_checks_infected(agents, current_day, infected_period, active_duration,
                               asymptomatic_pct, incubation_period,
                               symptomatic_duration_min, symptomatic_duration_mid,
                               symptomatic_duration_max, symptomatic_duration_dev,
                               effect_of_reinfection, long_covid, lc_onset_base_pct,
                               long_covid_time_threshold, lc_incidence_mult_female,
                               reinfection_new_onset_mult, asymptomatic_lc_mult,
                               lc_base_fast_prob, lc_base_persistent_prob, rng):
    """Update infected agents' disease progression"""
    
    for agent in agents:
        if not agent.infected:
            continue
        
        # First day of infection: set up disease course
        if agent.virus_check_timer == 0:
            agent.virus_check_timer = 1
            agent.transfer_active_duration = 1 + rng.integers(0, active_duration)
            
            # Determine if asymptomatic
            if rng.random() * 100 < asymptomatic_pct:
                agent.symptomatic_start = 0
            else:
                agent.symptomatic_start = 1 + rng.integers(0, incubation_period)
                while agent.symptomatic_start > agent.transfer_active_duration:
                    agent.symptomatic_start = 1 + rng.integers(0, incubation_period)
            
            # Set symptom duration
            if agent.symptomatic_start == 0:
                agent.symptomatic_duration = 0
            else:
                base_duration = _truncated_normal(
                    symptomatic_duration_mid, symptomatic_duration_dev,
                    symptomatic_duration_min, symptomatic_duration_max, rng
                )
                agent.symptomatic_duration = int(
                    effect_of_reinfection * agent.number_of_infection + base_duration
                )
                
                # If already has LC, reinfection worsens symptoms
                if agent.persistent_long_covid:
                    agent.symptomatic_duration = int(agent.symptomatic_duration * 1.5)
                    agent.long_covid_severity = min(90, agent.long_covid_severity + 10)
                    
                    # Group worsening
                    if agent.long_covid_recovery_group == 0:  # fast -> gradual
                        if rng.random() * 100 < 30:
                            agent.long_covid_recovery_group = 1
                            agent.long_covid_weibull_k = 1.2
                            agent.long_covid_weibull_lambda = 450
                    elif agent.long_covid_recovery_group == 1:  # gradual -> persistent
                        if rng.random() * 100 < 20:
                            agent.long_covid_recovery_group = 2
                            agent.long_covid_weibull_k = 0.5
                            agent.long_covid_weibull_lambda = 1200
        else:
            agent.virus_check_timer += 1
        
        # Update symptomatic status
        if agent.symptomatic_start > 0:
            symp_now = (agent.virus_check_timer >= agent.symptomatic_start and
                       agent.virus_check_timer < agent.symptomatic_start + agent.symptomatic_duration)
            agent.symptomatic = symp_now
        
        # Long COVID onset decisions
        if long_covid:
            # A) Asymptomatic path: decide at end of infection
            if agent.virus_check_timer > infected_period and agent.symptomatic_start == 0:
                p_onset = lc_onset_base_pct * _lc_incidence_multiplier(
                    agent, True, lc_incidence_mult_female, reinfection_new_onset_mult, asymptomatic_lc_mult
                )
                p_onset = max(0, min(100, p_onset))
                if rng.random() * 100 < p_onset:
                    agent.lc_pending = True
                    agent.lc_onset_day = agent.infection_start_tick + long_covid_time_threshold
                _become_immuned(agent)
                continue
            
            # B) Symptomatic: if duration > threshold, onset at threshold
            if (agent.symptomatic_start > 0 and 
                agent.symptomatic_duration > long_covid_time_threshold):
                if agent.virus_check_timer == agent.symptomatic_start + long_covid_time_threshold:
                    if not agent.persistent_long_covid:
                        agent.persistent_long_covid = True
                        agent.long_covid_duration = 0
                        _assign_long_covid_recovery_group(agent, lc_base_fast_prob, 
                                                         lc_base_persistent_prob, rng)
            
            # C) Symptomatic: if duration <= threshold, probabilistic onset at symptom end
            if (agent.symptomatic_start > 0 and 
                agent.symptomatic_duration <= long_covid_time_threshold):
                if agent.virus_check_timer == agent.symptomatic_start + agent.symptomatic_duration:
                    p_onset = lc_onset_base_pct * _lc_incidence_multiplier(
                        agent, False, lc_incidence_mult_female, reinfection_new_onset_mult, asymptomatic_lc_mult
                    )
                    p_onset = max(0, min(100, p_onset))
                    if rng.random() * 100 < p_onset:
                        agent.lc_pending = True
                        agent.lc_onset_day = agent.infection_start_tick + long_covid_time_threshold
        
        # State transitions
        if agent.symptomatic_start > 0:
            symptom_end = agent.symptomatic_start + agent.symptomatic_duration
            
            if symptom_end < infected_period:
                if agent.virus_check_timer == symptom_end:
                    _become_infected_nonsymptomatic(agent, current_day, infected_period, active_duration, rng)
                if agent.virus_check_timer > infected_period:
                    _become_immuned(agent)
            elif symptom_end == infected_period:
                if agent.virus_check_timer > infected_period:
                    _become_immuned(agent)
            else:  # symptom_end > infected_period
                if agent.virus_check_timer > infected_period:
                    _become_immuned_symptomatic(agent)
                if agent.virus_check_timer == symptom_end:
                    _become_immuned(agent)

def _do_virus_checks_immuned(agents, infected_period, immune_period):
    """Update immune agents' state"""
    for agent in agents:
        if not agent.immuned:
            continue
        
        # Update symptomatic status during immunity
        if agent.symptomatic_start > 0:
            symp_now = agent.virus_check_timer < agent.symptomatic_start + agent.symptomatic_duration
            agent.symptomatic = symp_now
        
        immunity_end = infected_period + immune_period
        
        if agent.virus_check_timer <= immunity_end:
            agent.virus_check_timer += 1
            agent.transfer_active_duration = 0
        else:
            _become_susceptible(agent)
            agent.virus_check_timer = 0
            agent.symptomatic_duration = 0
            agent.transfer_active_duration = 0

def _lc_incidence_multiplier(agent, is_asymptomatic, lc_incidence_mult_female,
                             reinfection_new_onset_mult, asymptomatic_lc_mult):
    """Calculate LC onset probability multiplier"""
    m = 1.0
    
    # Age multiplier
    if agent.age < 30:
        m *= 0.9
    elif 50 <= agent.age <= 64:
        m *= 1.2
    elif agent.age >= 65:
        m *= 1.3
    
    # Gender multiplier
    if agent.gender == 1:  # female
        m *= lc_incidence_mult_female
    
    # Vaccination effect
    if agent.vaccinated:
        m *= 0.7
    
    # Reinfection (only if no prior LC)
    has_lc = agent.long_covid_recovery_group in [0, 1, 2]
    if agent.number_of_infection > 1 and not has_lc:
        m *= reinfection_new_onset_mult
    
    # Asymptomatic discount
    if is_asymptomatic:
        m *= asymptomatic_lc_mult
    
    return m

def _assign_long_covid_recovery_group(agent, lc_base_fast_prob, lc_base_persistent_prob, rng):
    """Assign LC recovery group (0=fast, 1=gradual, 2=persistent)"""
    w_fast = lc_base_fast_prob
    w_pers = lc_base_persistent_prob
    w_sum = w_fast + w_pers
    
    # Normalize if over 100
    if w_sum > 100:
        w_fast = 100 * w_fast / w_sum
        w_pers = 100 * w_pers / w_sum
        w_sum = 100
    
    w_grad = 100 - w_sum
    
    # Tilts: age and acute severity
    if agent.age >= 65:
        shift = min(2, w_grad)
        w_pers += shift
        w_grad -= shift
    
    if agent.symptomatic_duration > 21:
        shift = min(4, w_grad)
        w_pers += shift
        w_grad -= shift
    
    # Draw subtype
    total = w_fast + w_pers + w_grad
    if total <= 0:
        w_grad = 100
        total = 100
    
    r = rng.random() * total
    
    if r < w_fast:
        # Fast recovery group
        agent.long_covid_recovery_group = 0
        agent.long_covid_weibull_k = 1.5
        agent.long_covid_weibull_lambda = 60
        agent.long_covid_severity = rng.normal(30, 15)
    elif r < w_fast + w_pers:
        # Persistent group
        agent.long_covid_recovery_group = 2
        agent.long_covid_weibull_k = 0.5
        agent.long_covid_weibull_lambda = 1200
        agent.long_covid_severity = rng.normal(70, 20)
    else:
        # Gradual recovery group
        agent.long_covid_recovery_group = 1
        agent.long_covid_weibull_k = 1.2
        agent.long_covid_weibull_lambda = 450
        agent.long_covid_severity = rng.normal(50, 20)
    
    # Clamp severity
    agent.long_covid_severity = max(5, min(100, agent.long_covid_severity))

def _do_long_covid_checks(agents, rng):
    """Update long COVID progression for all affected agents"""
    for agent in agents:
        if not agent.persistent_long_covid:
            continue
        
        # Increment LC duration
        agent.long_covid_duration += 1
        
        # Calculate Weibull-based recovery chance
        recovery_chance = _calculate_weibull_recovery_chance(
            agent.long_covid_duration,
            agent.long_covid_weibull_k,
            agent.long_covid_weibull_lambda
        )
        
        # Group-specific scaling
        if agent.long_covid_recovery_group == 0:  # fast
            recovery_chance *= 2.0
        elif agent.long_covid_recovery_group == 2:  # persistent
            recovery_chance *= 0.3
            if agent.long_covid_duration > 1095:  # 3 years
                recovery_chance *= 0.1
        # group 1 (gradual): no change
        
        # Clamp to daily window
        recovery_chance = max(0, min(15, recovery_chance))
        
        # Recovery draw
        if rng.random() * 100 < recovery_chance:
            # Recover from LC
            agent.persistent_long_covid = False
            agent.long_covid_severity = 0
            agent.long_covid_duration = 0
            agent.long_covid_recovery_group = -1
            agent.long_covid_weibull_k = 0
            agent.long_covid_weibull_lambda = 0
        else:
            # Gradual group: taper severity after 30 days
            if agent.long_covid_recovery_group == 1 and agent.long_covid_duration > 30:
                improvement_rate = 0.05
                agent.long_covid_severity -= improvement_rate
                agent.long_covid_severity = max(5, min(100, agent.long_covid_severity))

def _calculate_weibull_recovery_chance(duration, k, lambda_param):
    """Calculate daily recovery probability from Weibull hazard function"""
    if duration <= 0 or k <= 0 or lambda_param <= 0:
        return 0.0
    
    t_scaled = duration / lambda_param
    hazard = (k / lambda_param) * (t_scaled ** (k - 1))
    
    # Convert hazard to daily probability
    daily_prob = (1 - np.exp(-hazard)) * 100
    daily_prob = max(0.01, min(10, daily_prob))
    
    return daily_prob

def _process_pending_lc(agents, current_day):
    """Process agents with pending LC onset"""
    new_lc_count = 0
    
    for agent in agents:
        if not agent.lc_pending:
            continue
        if current_day < agent.lc_onset_day:
            continue
        
        agent.lc_pending = False
        
        if not agent.persistent_long_covid:
            agent.persistent_long_covid = True
            agent.long_covid_duration = 0
            _assign_long_covid_recovery_group(agent, 40.0, 20.0, 
                                             np.random.default_rng())  # Use default probs
            new_lc_count += 1
    
    return new_lc_count

def _check_vaccination_time(agents, boosted_pct, rng):
    """Update vaccination time and handle boosters at 6 months"""
    for agent in agents:
        if not agent.vaccinated:
            continue
        
        agent.vaccinated_time += 1
        
        # Booster at 6 months (180 days)
        if agent.vaccinated_time == 180:
            if rng.random() * 100 < boosted_pct:
                # Get booster
                agent.vaccinated = True
                agent.vaccinated_time = 1
            else:
                # No booster, lose vaccination status
                agent.vaccinated = False
                agent.vaccinated_time = 0

def _update_productivity(agents):
    """Calculate current aggregate productivity (0-100)"""
    if len(agents) == 0:
        return 100.0
    
    total_loss = 0.0
    
    for agent in agents:
        agent_loss = 0.0
        
        # Symptomatic agents contribute 100% loss
        if agent.symptomatic:
            agent_loss = 1.0
        # Otherwise, LC contributes partial loss based on severity
        elif agent.persistent_long_covid:
            agent_loss = agent.long_covid_severity / 100.0
        
        agent_loss = max(0.0, min(1.0, agent_loss))
        total_loss += agent_loss
    
    current_productivity = (1 - (total_loss / len(agents))) * 100
    return current_productivity


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    # Test with default parameters
    result = simulate_netlogoish(
        N=1000,
        max_days=365,
        covid_spread_chance_pct=5.0,
        initial_infected_agents=10,
        precaution_pct=0.0,
        avg_degree=20,
        v_start_time=0,
        vaccination_pct=0.0,
        seed=42
    )
    
    print("Simulation Results:")
    print(f"  Runtime: {result['runtime_days']} days")
    print(f"  Total Infected: {result['infected']}")
    print(f"  Reinfections: {result['reinfected']}")
    print(f"  Long COVID Cases: {result['long_covid_cases']}")
    print(f"  Min Productivity: {result['min_productivity']:.2f}%")
    
    # Run a parameter sweep like model.py
    print("\n--- Running mini parameter sweep ---")
    for spread_chance in [2, 5, 10]:
        result = simulate_netlogoish(
            N=500,
            max_days=180,
            covid_spread_chance_pct=spread_chance,
            initial_infected_agents=10,
            seed=42
        )
        print(f"Spread={spread_chance}%: infected={result['infected']}, "
              f"LC={result['long_covid_cases']}, min_prod={result['min_productivity']:.1f}%")
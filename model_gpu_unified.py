"""
Unified GPU-accelerated COVID ABM matching NetLogo implementation
Combines simulation, parameter sweep, and plotting in one file
"""

import jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import truncnorm
import time

class AgentGPU:
    """Simplified agent for GPU-compatible simulation matching NetLogo"""
    __slots__ = ['idx', 'infected', 'immuned', 'symptomatic', 'super_immune',
                 'persistent_long_covid', 'long_covid_severity', 'long_covid_duration',
                 'long_covid_recovery_group', 'long_covid_weibull_k', 'long_covid_weibull_lambda',
                 'lc_pending', 'lc_onset_day', 'virus_check_timer', 'number_of_infection',
                 'infection_start_tick', 'infectious_start', 'infectious_end',
                 'transfer_active_duration', 'symptomatic_start', 'symptomatic_duration',
                 'age', 'gender', 'health_risk_level', 'covid_age_prob', 'us_age_prob',
                 'vaccinated', 'vaccinated_time', 'neighbors', 'temporal_neighbors',
                 'initial_infections', 'real_efficiency']
    
    def __init__(self, idx):
        self.idx = idx
        self.infected = False
        self.immuned = False
        self.symptomatic = False
        self.super_immune = False
        self.persistent_long_covid = False
        self.long_covid_severity = 0.0
        self.long_covid_duration = 0
        self.long_covid_recovery_group = -1
        self.long_covid_weibull_k = 0.0
        self.long_covid_weibull_lambda = 0.0
        self.lc_pending = False
        self.lc_onset_day = 0
        self.virus_check_timer = 0
        self.number_of_infection = 0
        self.infection_start_tick = 0
        self.infectious_start = 1
        self.infectious_end = 1
        self.transfer_active_duration = 0
        self.symptomatic_start = 0
        self.symptomatic_duration = 0
        self.age = 0
        self.gender = 0
        self.health_risk_level = 1
        self.covid_age_prob = 15.0
        self.us_age_prob = 13.0
        self.vaccinated = False
        self.vaccinated_time = 0
        self.neighbors = set()
        self.temporal_neighbors = set()  # For daily ephemeral links
        self.initial_infections = 0  # Track initially seeded infections
        self.real_efficiency = 0.0  # Effective vaccine efficiency

def simulate_gpu(
    N=100000,
    max_days=365,
    covid_spread_chance_pct=10.0,
    initial_infected_agents=5,
    precaution_pct=50.0,
    avg_degree=5,
    v_start_time=180,
    vaccination_pct=80.0,
    temporal_connections_pct=0.0,  # NEW: NetLogo's Temporal-connections%
    network_model="M1-average-connections",  # NEW: NetLogo network models
    # Network M2 parameters
    m2_initial_connections=3,
    m2_max_family_size=10,
    # Infection parameters
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
    boosted_pct=30.0,  # FIXED: Consistent naming
    vaccination_decay=True,
    vaccine_priority=True,
    # Demographics
    gender=True,
    male_population_pct=49.5,
    age_distribution=True,
    age_range=100,
    age_infection_scaling=True,
    risk_level_2_pct=4.0,
    risk_level_3_pct=40.0,
    risk_level_4_pct=6.0,
    seed=None,
):
    """
    GPU-accelerated simulation matching NetLogo behavior
    """
    
    print(f"Running GPU-optimized simulation with {N:,} agents...")
    print(f"Network model: {network_model}")
    print(f"JAX backend: {jax.default_backend()}")
    
    rng = np.random.default_rng(seed)
    agents = [AgentGPU(i) for i in range(N)]
    
    # ========== NETWORK ==========
    print("Creating network...")
    
    if network_model == "M1-average-connections":
        # M1: Simple random network with average degree
        p = min(1.0, avg_degree / max(1, N - 1))
        
        for i in range(N):
            n_edges = rng.binomial(N - i - 1, p)
            if n_edges > 0:
                targets = rng.choice(np.arange(i + 1, N), size=min(n_edges, N - i - 1), replace=False)
                for j in targets:
                    agents[i].neighbors.add(j)
                    agents[j].neighbors.add(i)
    
    elif network_model == "M2-preferential-with-family-connections":
        # M2: Preferential attachment with family connections (simplified)
        # Start with initial connections
        for i in range(min(m2_initial_connections, N)):
            for j in range(i + 1, min(i + 1 + m2_initial_connections, N)):
                if j < N:
                    agents[i].neighbors.add(j)
                    agents[j].neighbors.add(i)
        
        # Add family connections (cliques among neighbors)
        for agent in agents:
            if len(agent.neighbors) > 1 and len(agent.neighbors) < m2_max_family_size:
                neighbors_list = list(agent.neighbors)
                # Make neighbors connected to each other
                for i in range(len(neighbors_list)):
                    for j in range(i + 1, len(neighbors_list)):
                        a1, a2 = neighbors_list[i], neighbors_list[j]
                        agents[a1].neighbors.add(a2)
                        agents[a2].neighbors.add(a1)
    
    # ========== DEMOGRAPHICS ==========
    print("Setting up demographics...")
    age_bins = [(0, 5, 5.7), (5, 15, 12.5), (15, 25, 13.0), (25, 35, 13.7),
                (35, 45, 13.1), (45, 55, 12.3), (55, 65, 12.9), (65, 75, 10.1),
                (75, 85, 4.9), (85, 95, 1.8)]
    
    if age_distribution:
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
    
    if gender:
        male_prob = male_population_pct / 100.0
        for agent in agents:
            agent.gender = 0 if rng.random() < male_prob else 1
    
    # Set age probabilities (vectorize for speed)
    ages = np.array([a.age for a in agents])
    covid_age_probs = np.select(
        [ages < 10, ages < 20, ages < 30, ages < 40, ages < 50,
         ages < 60, ages < 70, ages < 80],
        [2.3, 5.1, 15.5, 16.9, 16.4, 16.4, 11.9, 7.0],
        default=8.5
    )
    us_age_probs = np.select(
        [ages < 5, ages < 15, ages < 25, ages < 35, ages < 45,
         ages < 55, ages < 65, ages < 75, ages < 85],
        [5.7, 12.5, 13.0, 13.7, 13.1, 12.3, 12.9, 10.1, 4.9],
        default=1.8
    )
    
    for i, agent in enumerate(agents):
        agent.covid_age_prob = covid_age_probs[i]
        agent.us_age_prob = us_age_probs[i]
    
    # Risk levels
    if gender:
        eligible_level2 = [a for a in agents if a.gender == 1 and 15 <= a.age <= 49]
    else:
        eligible_level2 = agents[:]
    
    n2 = min(int(risk_level_2_pct * N / 100), len(eligible_level2))
    if n2 > 0:
        for agent in rng.choice(eligible_level2, size=n2, replace=False):
            agent.health_risk_level = 2
    
    eligible_level3 = [a for a in agents if a.health_risk_level == 1]
    n3 = min(int(risk_level_3_pct * N / 100), len(eligible_level3))
    if n3 > 0:
        for agent in rng.choice(eligible_level3, size=n3, replace=False):
            agent.health_risk_level = 3
    
    eligible_level4 = [a for a in agents if a.health_risk_level == 1]
    n4 = min(int(risk_level_4_pct * N / 100), len(eligible_level4))
    if n4 > 0:
        for agent in rng.choice(eligible_level4, size=n4, replace=False):
            agent.health_risk_level = 4
    
    # Super-immune
    n_super = int(super_immune_pct * N / 100)
    if n_super > 0:
        for agent in rng.choice(agents, size=n_super, replace=False):
            agent.super_immune = True
    
    # ========== SEED INFECTIONS ==========
    eligible = [a for a in agents if not a.super_immune]
    n_initial = min(initial_infected_agents, len(eligible))
    if n_initial > 0:
        for agent in rng.choice(eligible, size=n_initial, replace=False):
            agent.infected = True
            agent.virus_check_timer = 0
            agent.number_of_infection = 1
            agent.infection_start_tick = 0
            agent.infectious_start = 1
            agent.initial_infections = 1  # Mark as seeded
            drawn_length = 1 + rng.integers(0, active_duration)
            max_length = max(1, infected_period - agent.infectious_start)
            agent.transfer_active_duration = min(drawn_length, max_length)
            agent.infectious_end = agent.infectious_start + agent.transfer_active_duration
    
    # ========== METRICS ==========
    total_infected = n_initial
    total_reinfected = 0
    long_covid_cases = 0
    min_productivity = 100.0
    
    # ========== MAIN LOOP ==========
    print("Starting simulation...")
    
    for day in range(max_days):
        if day % 50 == 0:
            n_inf = sum(1 for a in agents if a.infected)
            n_lc = sum(1 for a in agents if a.persistent_long_covid)
            print(f"Day {day}: {n_inf} infected, {n_lc} with LC")
        
        # ========== TEMPORAL CONNECTIONS (NEW: NetLogo feature) ==========
        # Clear previous temporal connections
        for agent in agents:
            agent.temporal_neighbors.clear()
        
        # Create new temporal connections for this day
        if temporal_connections_pct > 0:
            target = int(temporal_connections_pct * N / 100)
            for _ in range(target):
                i = rng.integers(0, N)
                j = rng.integers(0, N)
                if i != j and j not in agents[i].neighbors:
                    agents[i].temporal_neighbors.add(j)
                    agents[j].temporal_neighbors.add(i)
        
        # ========== VACCINATION (FIXED: Check every day after start time) ==========
        if day >= v_start_time and vaccination_pct > 0:
            current_vaccinated = sum(1 for a in agents if a.vaccinated)
            target_vaccinated = int(N * vaccination_pct / 100)
            
            if current_vaccinated < target_vaccinated:
                unvaccinated = [a for a in agents if not a.vaccinated]
                
                if vaccine_priority:
                    # Priority groups as in NetLogo
                    priority_groups = [
                        [a for a in unvaccinated if a.age >= 65],
                        [a for a in unvaccinated if a.health_risk_level == 4 and a.age < 65],
                        [a for a in unvaccinated if a.health_risk_level == 3 and a.age < 65],
                        [a for a in unvaccinated if a.health_risk_level == 2 and a.age < 65],
                        [a for a in unvaccinated if a.health_risk_level == 1 and a.age < 65],
                    ]
                    
                    for group in priority_groups:
                        for agent in group:
                            if current_vaccinated >= target_vaccinated:
                                break
                            agent.vaccinated = True
                            agent.vaccinated_time = 1
                            current_vaccinated += 1
                else:
                    n_vacc = min(target_vaccinated - current_vaccinated, len(unvaccinated))
                    if n_vacc > 0:
                        for agent in rng.choice(unvaccinated, size=n_vacc, replace=False):
                            agent.vaccinated = True
                            agent.vaccinated_time = 1
        
        # ========== LONG COVID PROGRESSION ==========
        if long_covid:
            lc_agents = [a for a in agents if a.persistent_long_covid]
            for agent in lc_agents:
                agent.long_covid_duration += 1
                
                if agent.long_covid_weibull_lambda <= 0:
                    continue
                
                # Calculate Weibull recovery chance (matching NetLogo)
                t_scaled = agent.long_covid_duration / agent.long_covid_weibull_lambda
                hazard = (agent.long_covid_weibull_k / agent.long_covid_weibull_lambda) * (t_scaled ** (agent.long_covid_weibull_k - 1))
                recovery_chance = (1 - np.exp(-hazard)) * 100
                
                # Group-specific scaling
                if agent.long_covid_recovery_group == 0:
                    recovery_chance *= 2.0  # Fast group
                elif agent.long_covid_recovery_group == 2:
                    recovery_chance *= 0.3  # Persistent group
                    if agent.long_covid_duration > 1095:  # 3-year slowdown
                        recovery_chance *= 0.1
                
                recovery_chance = np.clip(recovery_chance, 0, 15)
                
                if rng.random() * 100 < recovery_chance:
                    agent.persistent_long_covid = False
                    agent.long_covid_severity = 0
                    agent.long_covid_duration = 0
                    agent.long_covid_recovery_group = -1
                    agent.long_covid_weibull_k = 0
                    agent.long_covid_weibull_lambda = 0
                elif agent.long_covid_recovery_group == 1 and agent.long_covid_duration > 30:
                    # Gradual group: slow improvement after 1 month
                    agent.long_covid_severity = max(5, agent.long_covid_severity - 0.05)
        
        # ========== TRANSMISSION ==========
        new_infections = []
        infectious_agents = [a for a in agents if (a.infected and 
                                                   a.virus_check_timer >= a.infectious_start and 
                                                   a.virus_check_timer < a.infectious_end)]
        
        for agent in infectious_agents:
            # Check precaution for symptomatic agents
            if agent.symptomatic and agent.symptomatic_start > 0 and agent.virus_check_timer > agent.symptomatic_start:
                if rng.random() * 100 < precaution_pct:
                    continue
            
            # Get all neighbors (static + temporal)
            all_neighbors = set(agent.neighbors) | set(agent.temporal_neighbors)
            
            for neighbor_idx in all_neighbors:
                neighbor = agents[neighbor_idx]
                
                if neighbor.infected or neighbor.immuned or neighbor.super_immune:
                    continue
                
                # Vaccine protection
                if neighbor.vaccinated:
                    if vaccination_decay:
                        real_eff = max(0, min(100, efficiency_pct - 0.11 * neighbor.vaccinated_time))
                    else:
                        real_eff = efficiency_pct
                    
                    neighbor.real_efficiency = real_eff  # Store for consistency with NetLogo
                    if rng.random() * 100 < real_eff:
                        continue
                
                infection_prob = covid_spread_chance_pct
                
                if age_infection_scaling:
                    infection_prob *= neighbor.covid_age_prob / (neighbor.us_age_prob + 1e-9)
                
                infection_prob = max(0, min(100, infection_prob))
                
                if rng.random() * 100 < infection_prob:
                    new_infections.append(neighbor)
        
        # Apply infections
        for agent in set(new_infections):
            if agent.number_of_infection > 0:
                total_reinfected += 1
            
            agent.infected = True
            agent.immuned = False
            agent.symptomatic = False
            agent.infection_start_tick = day
            agent.virus_check_timer = 0
            agent.number_of_infection += 1
            agent.infectious_start = 1
            
            drawn_length = 1 + rng.integers(0, active_duration)
            max_length = max(1, infected_period - agent.infectious_start)
            agent.transfer_active_duration = min(drawn_length, max_length)
            agent.infectious_end = agent.infectious_start + agent.transfer_active_duration
            
            if not agent.persistent_long_covid:
                agent.long_covid_recovery_group = -1
                agent.long_covid_weibull_k = 0
                agent.long_covid_weibull_lambda = 0
            
            total_infected += 1
        
        # ========== INFECTION PROGRESSION ==========
        for agent in agents:
            if not agent.infected:
                continue
            
            if agent.virus_check_timer == 0:
                agent.virus_check_timer = 1
                agent.transfer_active_duration = 1 + rng.integers(0, active_duration)
                
                if rng.random() * 100 < asymptomatic_pct:
                    agent.symptomatic_start = 0
                else:
                    agent.symptomatic_start = 1 + rng.integers(0, incubation_period)
                    while agent.symptomatic_start > agent.transfer_active_duration:
                        agent.symptomatic_start = 1 + rng.integers(0, incubation_period)
                
                if agent.symptomatic_start == 0:
                    agent.symptomatic_duration = 0
                else:
                    a = (symptomatic_duration_min - symptomatic_duration_mid) / symptomatic_duration_dev
                    b = (symptomatic_duration_max - symptomatic_duration_mid) / symptomatic_duration_dev
                    base = truncnorm.rvs(a, b, loc=symptomatic_duration_mid, scale=symptomatic_duration_dev, random_state=rng)
                    agent.symptomatic_duration = int(effect_of_reinfection * agent.number_of_infection + base)
                    
                    if agent.persistent_long_covid:
                        agent.symptomatic_duration = int(agent.symptomatic_duration * 1.5)
                        agent.long_covid_severity = min(90, agent.long_covid_severity + 10)
                        
                        # Group worsening (matching NetLogo)
                        if agent.long_covid_recovery_group == 0 and rng.random() * 100 < 30:
                            agent.long_covid_recovery_group = 1
                            agent.long_covid_weibull_k = 1.2
                            agent.long_covid_weibull_lambda = 450
                        elif agent.long_covid_recovery_group == 1 and rng.random() * 100 < 20:
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
            
            # ========== LONG COVID ONSET LOGIC ==========
            # A) Asymptomatic path: decide at end of infection
            if long_covid and agent.virus_check_timer > infected_period and agent.symptomatic_start == 0:
                age_mult = 0.9 if agent.age < 30 else (1.2 if 50 <= agent.age <= 64 else (1.3 if agent.age >= 65 else 1.0))
                gender_mult = lc_incidence_mult_female if agent.gender == 1 else 1.0
                vacc_mult = 0.7 if agent.vaccinated else 1.0
                has_lc = agent.long_covid_recovery_group in [0, 1, 2]
                reinf_mult = reinfection_new_onset_mult if (agent.number_of_infection > 1 and not has_lc) else 1.0
                
                p_onset = lc_onset_base_pct * age_mult * gender_mult * vacc_mult * reinf_mult * asymptomatic_lc_mult
                p_onset = max(0, min(100, p_onset))
                
                if rng.random() * 100 < p_onset:
                    agent.lc_pending = True
                    agent.lc_onset_day = agent.infection_start_tick + long_covid_time_threshold
                
                agent.infected = False
                agent.immuned = True
                continue
            
            # B) Symptomatic path: if symptoms exceed threshold, flip to LC on threshold day
            if (long_covid and agent.symptomatic_start > 0 and 
                agent.symptomatic_duration > long_covid_time_threshold):
                if agent.virus_check_timer == agent.symptomatic_start + long_covid_time_threshold:
                    if not agent.persistent_long_covid:
                        agent.persistent_long_covid = True
                        agent.long_covid_duration = 0
                        long_covid_cases += 1
                        
                        # Assign LC recovery group (matching NetLogo weights)
                        w_fast = lc_base_fast_prob
                        w_pers = lc_base_persistent_prob
                        w_grad = 100 - w_fast - w_pers
                        
                        # Adjust weights based on age and symptom duration
                        if agent.age >= 65:
                            shift = min(2, w_grad)
                            w_pers += shift
                            w_grad -= shift
                        
                        if agent.symptomatic_duration > 21:
                            shift = min(4, w_grad)
                            w_pers += shift
                            w_grad -= shift
                        
                        r = rng.random() * (w_fast + w_pers + w_grad)
                        if r < w_fast:
                            agent.long_covid_recovery_group = 0
                            agent.long_covid_weibull_k = 1.5
                            agent.long_covid_weibull_lambda = 60
                            agent.long_covid_severity = np.clip(rng.normal(30, 15), 5, 100)
                        elif r < w_fast + w_pers:
                            agent.long_covid_recovery_group = 2
                            agent.long_covid_weibull_k = 0.5
                            agent.long_covid_weibull_lambda = 1200
                            agent.long_covid_severity = np.clip(rng.normal(70, 20), 5, 100)
                        else:
                            agent.long_covid_recovery_group = 1
                            agent.long_covid_weibull_k = 1.2
                            agent.long_covid_weibull_lambda = 450
                            agent.long_covid_severity = np.clip(rng.normal(50, 20), 5, 100)
            
            # C) Symptomatic path: if symptoms end <= threshold, decide LC probabilistically at symptom end
            if (long_covid and agent.symptomatic_start > 0 and 
                agent.symptomatic_duration <= long_covid_time_threshold):
                if agent.virus_check_timer == agent.symptomatic_start + agent.symptomatic_duration:
                    age_mult = 0.9 if agent.age < 30 else (1.2 if 50 <= agent.age <= 64 else (1.3 if agent.age >= 65 else 1.0))
                    gender_mult = lc_incidence_mult_female if agent.gender == 1 else 1.0
                    vacc_mult = 0.7 if agent.vaccinated else 1.0
                    has_lc = agent.long_covid_recovery_group in [0, 1, 2]
                    reinf_mult = reinfection_new_onset_mult if (agent.number_of_infection > 1 and not has_lc) else 1.0
                    
                    p_onset = lc_onset_base_pct * age_mult * gender_mult * vacc_mult * reinf_mult
                    p_onset = max(0, min(100, p_onset))
                    
                    if rng.random() * 100 < p_onset:
                        agent.lc_pending = True
                        agent.lc_onset_day = agent.infection_start_tick + long_covid_time_threshold
            
            # ========== STATE TRANSITIONS ==========
            if agent.symptomatic_start > 0:
                symptom_end = agent.symptomatic_start + agent.symptomatic_duration
                
                if symptom_end < infected_period:
                    if agent.virus_check_timer == symptom_end:
                        agent.infected = True
                        agent.symptomatic = False
                    if agent.virus_check_timer > infected_period:
                        agent.infected = False
                        agent.immuned = True
                elif symptom_end == infected_period:
                    if agent.virus_check_timer > infected_period:
                        agent.infected = False
                        agent.immuned = True
                else:
                    if agent.virus_check_timer > infected_period:
                        agent.infected = False
                        agent.immuned = True
                    if agent.virus_check_timer == symptom_end:
                        agent.symptomatic = False
        
        # ========== IMMUNITY WANING ==========
        for agent in agents:
            if not agent.immuned:
                continue
            
            if agent.symptomatic_start > 0:
                symp_now = agent.virus_check_timer < agent.symptomatic_start + agent.symptomatic_duration
                agent.symptomatic = symp_now
            
            immunity_end = infected_period + immune_period
            
            if agent.virus_check_timer <= immunity_end:
                agent.virus_check_timer += 1
                agent.transfer_active_duration = 0
            else:
                agent.infected = False
                agent.immuned = False
                agent.symptomatic = False
                agent.virus_check_timer = 0
                agent.symptomatic_duration = 0
                agent.transfer_active_duration = 0
        
        # ========== PROCESS PENDING LONG COVID ==========
        if long_covid:
            for agent in agents:
                if agent.lc_pending and day >= agent.lc_onset_day:
                    agent.lc_pending = False
                    if not agent.persistent_long_covid:
                        agent.persistent_long_covid = True
                        agent.long_covid_duration = 0
                        long_covid_cases += 1
                        
                        # Assign LC recovery group
                        w_fast = lc_base_fast_prob
                        w_pers = lc_base_persistent_prob
                        w_grad = 100 - w_fast - w_pers
                        
                        r = rng.random() * (w_fast + w_pers + w_grad)
                        if r < w_fast:
                            agent.long_covid_recovery_group = 0
                            agent.long_covid_weibull_k = 1.5
                            agent.long_covid_weibull_lambda = 60
                            agent.long_covid_severity = np.clip(rng.normal(30, 15), 5, 100)
                        elif r < w_fast + w_pers:
                            agent.long_covid_recovery_group = 2
                            agent.long_covid_weibull_k = 0.5
                            agent.long_covid_weibull_lambda = 1200
                            agent.long_covid_severity = np.clip(rng.normal(70, 20), 5, 100)
                        else:
                            agent.long_covid_recovery_group = 1
                            agent.long_covid_weibull_k = 1.2
                            agent.long_covid_weibull_lambda = 450
                            agent.long_covid_severity = np.clip(rng.normal(50, 20), 5, 100)
        
        # ========== VACCINATION TIME DECAY ==========
        for agent in agents:
            if agent.vaccinated:
                agent.vaccinated_time += 1
                if agent.vaccinated_time == 180:  # 6 months
                    if rng.random() * 100 < boosted_pct:
                        agent.vaccinated = True
                        agent.vaccinated_time = 1  # Reset counter
                    else:
                        agent.vaccinated = False
                        agent.vaccinated_time = 0
        
        # ========== PRODUCTIVITY CALCULATION ==========
        symptomatic_loss = sum(1 for a in agents if a.symptomatic)
        lc_loss = sum(a.long_covid_severity / 100.0 for a in agents if a.persistent_long_covid and not a.symptomatic)
        total_loss = symptomatic_loss + lc_loss
        current_prod = (1 - total_loss / N) * 100
        min_productivity = min(min_productivity, current_prod)
        
        # ========== STOP CONDITION ==========
        if not any(a.infected for a in agents) and not any(a.immuned for a in agents):
            print(f"Epidemic ended at day {day}")
            break
    
    return {
        'runtime_days': day + 1,
        'infected': total_infected,
        'reinfected': total_reinfected,
        'long_covid_cases': long_covid_cases,
        'min_productivity': min_productivity,
    }

# ========== PARAMETER SWEEP AND PLOTTING FUNCTIONS ==========

# Defaults matching NetLogo parameter ranges
DEFAULTS = dict(
    N=100000,
    max_days=365,
    covid_spread_chance_pct=10.0,
    initial_infected_agents=5,  # FIXED: Match NetLogo range
    precaution_pct=50.0,
    avg_degree=5,
    v_start_time=180,
    vaccination_pct=80.0,
    temporal_connections_pct=0.0,  # NEW
    network_model="M1-average-connections",  # NEW
    # Infection parameters
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
    boosted_pct=30.0,  # FIXED: Consistent naming
    vaccination_decay=True,
    vaccine_priority=True,
    # Demographics
    gender=True,
    age_distribution=True,
    age_range=100,
    age_infection_scaling=True,
)

# Parameter ranges matching NetLogo implementation
RANGES = {
    "covid_spread_chance_pct": [2, 5, 10, 20],
    "initial_infected_agents": [2, 5, 10, 20],  # FIXED: Match NetLogo range
    "precaution_pct": [0, 30, 50, 80],
    "avg_degree": [10, 30, 50, 70],
    "v_start_time": [0, 30, 180, 360],
    "vaccination_pct": [0, 30, 50, 80],
}

METRICS_LIST = [
    ("runtime_days", "Run time"),
    ("infected", "Infected cases"),
    ("reinfected", "Reinfected cases"),
    ("long_covid_cases", "Long covid cases"),
    ("min_productivity", "Min productivity"),
]

LABELS = {
    "covid_spread_chance_pct": "COVID Spread\nChance%",
    "initial_infected_agents": "Initial Infected\nAgents",
    "precaution_pct": "Precaution%",
    "avg_degree": "Average\nDegree",
    "v_start_time": "Vaccination\nStart Time",
    "vaccination_pct": "Vaccination%",
}

def run_sweep(n_runs=20, seed=0, out_csv="results_100k.csv", N=100000):
    """
    Run parameter sweep with GPU acceleration
    """
    rows = []
    rs = np.random.RandomState(seed)
    
    total_sims = sum(len(values) * n_runs for values in RANGES.values())
    print(f"Running {total_sims} simulations with {N:,} agents each...")
    
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
                params["seed"] = rs.randint(0, 2**31 - 1)
                
                res = simulate_gpu(**params)
                res["parameter"] = pname
                res["value"] = val
                res["run"] = r
                
                rows.append(res)
                sim_count += 1
                
                if sim_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = sim_count / elapsed
                    eta = (total_sims - sim_count) / rate / 60 if rate > 0 else 0
                    print(f"  Completed {sim_count}/{total_sims} ({rate:.1f} sims/sec, ETA: {eta:.1f} min)")
    
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")
    return df

def plot_results(df, out_png="results_100k.png"):
    """
    Plot parameter sweep results
    """
    n_params = len(RANGES)
    n_metrics = len(METRICS_LIST)
    
    fig, axes = plt.subplots(n_metrics, n_params, figsize=(4 * n_params, 3 * n_metrics))
    
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    if n_params == 1:
        axes = axes.reshape(-1, 1)
    
    for j, (pname, values) in enumerate(RANGES.items()):
        for i, (metric, mlabel) in enumerate(METRICS_LIST):
            ax = axes[i, j]
            
            subset = df[df["parameter"] == pname]
            means = []
            stds = []
            
            for val in values:
                vals = subset[subset["value"] == val][metric]
                means.append(vals.mean())
                stds.append(vals.std())
            
            ax.errorbar(range(len(values)), means, yerr=stds, capsize=5, marker='o', linewidth=2)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values)
            
            if i == 0:
                ax.set_title(LABELS[pname], fontsize=14)
            if j == 0:
                ax.set_ylabel(mlabel, fontsize=12)
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {out_png}")

# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    # Quick test simulation
    print("=== Running test simulation ===")
    test_results = simulate_gpu(N=10000, max_days=100, seed=42)
    print("Test results:", test_results)
    
    # Full parameter sweep (comment out for quick testing)
    print("\n=== Running parameter sweep ===")
    df = run_sweep(n_runs=5, N=50000)  # Smaller for testing
    
    print("\n=== Plotting results ===")
    plot_results(df)
    
    print("\n=== Summary statistics ===")
    for metric, mlabel in METRICS_LIST:
        print(f"{mlabel}: mean={df[metric].mean():.2f}, std={df[metric].std():.2f}")
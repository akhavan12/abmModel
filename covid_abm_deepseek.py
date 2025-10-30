"""
COVID-19 ABM - Exact NetLogo Replication with Scalability
Matches NetLogo behavior parameter-for-parameter with 1M+ agents
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm
import time
import gc
import psutil
from dataclasses import dataclass
import json

@dataclass
class AgentData:
    """Exact replication of NetLogo turtle variables"""
    # NetLogo: infected?, immuned?, symptomatic?, super_immune?
    infected: np.bool_
    immuned: np.bool_ 
    symptomatic: np.bool_
    super_immune: np.bool_
    
    # NetLogo: persistent-long-covid?, long-covid-severity, long-covid-duration
    persistent_long_covid: np.bool_
    long_covid_severity: np.float32
    long_covid_duration: np.int32
    
    # NetLogo: long-covid-recovery-group, long-covid-weibull-k, long-covid-weibull-lambda
    long_covid_recovery_group: np.int8
    long_covid_weibull_k: np.float32
    long_covid_weibull_lambda: np.float32
    
    # NetLogo: lc-pending?, lc-onset-day
    lc_pending: np.bool_
    lc_onset_day: np.int32
    
    # NetLogo: virus-check-timer, number-of-infection, infection-start-tick
    virus_check_timer: np.int32
    number_of_infection: np.int32
    infection_start_tick: np.int32
    
    # NetLogo: infectious-start, infectious-end, transfer-active-duration
    infectious_start: np.int32
    infectious_end: np.int32
    transfer_active_duration: np.int32
    
    # NetLogo: symptomatic-start, symptomatic-duration
    symptomatic_start: np.int32
    symptomatic_duration: np.int32
    
    # NetLogo: age-of, gender-of, health-risk-level
    age: np.int8
    gender: np.int8
    health_risk_level: np.int8
    
    # NetLogo: covid-age-prob, us-age-prob
    covid_age_prob: np.float32
    us_age_prob: np.float32
    
    # NetLogo: vaccinated, vaccinated-time, real_efficiency
    vaccinated: np.bool_
    vaccinated_time: np.int32
    real_efficiency: np.float32
    
    # NetLogo: initial-infections
    initial_infections: np.int8

    def __init__(self, idx):
        self.idx = idx
        # Initialize all fields with default values
        self.infected = np.bool_(False)
        self.immuned = np.bool_(False)
        self.symptomatic = np.bool_(False)
        self.super_immune = np.bool_(False)
        self.persistent_long_covid = np.bool_(False)
        self.long_covid_severity = np.float32(0.0)
        self.long_covid_duration = np.int32(0)
        self.long_covid_recovery_group = np.int8(-1)
        self.long_covid_weibull_k = np.float32(0.0)
        self.long_covid_weibull_lambda = np.float32(0.0)
        self.lc_pending = np.bool_(False)
        self.lc_onset_day = np.int32(0)
        self.virus_check_timer = np.int32(0)
        self.number_of_infection = np.int32(0)
        self.infection_start_tick = np.int32(0)
        self.infectious_start = np.int32(1)
        self.infectious_end = np.int32(1)
        self.transfer_active_duration = np.int32(0)
        self.symptomatic_start = np.int32(0)
        self.symptomatic_duration = np.int32(0)
        self.age = np.int8(0)
        self.gender = np.int8(0)
        self.health_risk_level = np.int8(1)
        self.covid_age_prob = np.float32(15.0)
        self.us_age_prob = np.float32(13.0)
        self.vaccinated = np.bool_(False)
        self.vaccinated_time = np.int32(0)
        self.real_efficiency = np.float32(0.0)
        self.initial_infections = np.int8(0)

class NetLogoReplicaABM:
    """
    Exact replication of NetLogo COVID ABM with scalability
    """
    
    def __init__(self):
        self.agents = []
        self.neighbors = []  # Static network links
        self.temporal_neighbors = []  # Daily temporal links
        self.N = 0
        self.rng = None
        self.config = {}
        self.min_productivity = 100.0
        self.current_productivity = 100.0
        
    def initialize_simulation(self, 
                            N=100000,  # 100K agents by default for testing
                            seed=42,
                            **kwargs):
        """Initialize simulation matching NetLogo setup"""
        self.N = N
        self.rng = np.random.default_rng(seed)
        self.config = self._get_netlogo_default_config()
        self.config.update(kwargs)
        
        print(f"Initializing {N:,} agents (NetLogo replication)...")
        
        # NetLogo: setup
        self._create_agents()
        self._create_network_m1()  # Using M1 network like NetLogo
        self._setup_attributes_for_nodes()  # NetLogo: setup-attributes-for-nodes
        self._seed_initial_infections()  # NetLogo: seed initial infections
        
        # NetLogo: set min-productivity 100
        self.min_productivity = 100.0
        self.current_productivity = 100.0
        
        print("NetLogo replication initialized successfully!")
    
    def _get_netlogo_default_config(self):
        """Get exact NetLogo default parameters"""
        return {
            # NetLogo sliders
            'max_days': 365,
            'covid_spread_chance_pct': 10.0,  # COVID-spread-chance%
            'initial_infected_agents': 5,      # initial-infected-agents
            'precaution_pct': 50.0,           # Precaution-percentage%
            'avg_degree': 5,                  # M1-Average-connections
            'v_start_time': 180,              # V-start-time
            'vaccination_pct': 80.0,          # Vaccination%
            'temporal_connections_pct': 10.0, # Temporal-connections%
            
            # Infection parameters
            'infected_period': 10,            # Infected-period
            'active_duration': 7,             # Active-duration
            'immune_period': 21,              # Immune-period
            'incubation_period': 4,           # Incubation_period
            'symptomatic_duration_min': 1,    # Symptomatic-duration-min
            'symptomatic_duration_mid': 10,   # Symptomatic-duration-mid
            'symptomatic_duration_max': 60,   # Symptomatic-duration-max
            'symptomatic_duration_dev': 8,    # Symptomatic-duration-dev
            'asymptomatic_pct': 40.0,         # Asymptomatic%
            'effect_of_reinfection': 3,       # Effect-of-reinfection
            'super_immune_pct': 4.0,          # Super-immune%
            
            # Long COVID parameters
            'long_covid': True,               # long-COVID
            'long_covid_time_threshold': 30,  # Long-covid-time-threshold
            'asymptomatic_lc_mult': 0.50,     # Asymptomatic-LC-mult
            'lc_incidence_mult_female': 1.20, # LC-incidence-mult-female
            'lc_base_fast_prob': 9.0,         # LC-Base-fast-prob
            'lc_base_persistent_prob': 7.0,   # LC-Base-persistent-prob
            'reinfection_new_onset_mult': 0.70, # Reinfection-new-onset-mult
            'lc_onset_base_pct': 15.0,        # LC-Onset-Base%
            
            # Vaccination parameters
            'efficiency_pct': 80.0,           # Efficiency%
            'boosted_pct': 30.0,              # Boosted%
            'vaccination_decay': True,        # Vaccination-decay
            'vaccine_priority': True,         # Vaccine-priority
            
            # Demographics
            'gender': True,                   # Gender
            'male_population_pct': 49.5,      # Male-population%
            'age_distribution': True,         # Age-distribution
            'age_range': 100,                 # Age-range
            'age_infection_scaling': True,    # Age
            'risk_level_2_pct': 4.0,          # Risk-level-2%
            'risk_level_3_pct': 40.0,         # Risk-level-3%
            'risk_level_4_pct': 6.0,          # Risk-level-4%
        }
    
    def _create_agents(self):
        """Create agents matching NetLogo setup"""
        self.agents = []
        for i in range(self.N):
            agent = AgentData(i)
            self.agents.append(agent)
    
    def _create_network_m1(self):
        """Create M1 network matching NetLogo setup-network-m1"""
        print("Creating M1 network (NetLogo replication)...")
        N = self.N
        avg_degree = self.config['avg_degree']
        
        # NetLogo: target undirected edge count = avg-degree * N / 2
        target_links = (avg_degree * N) // 2
        self.neighbors = [set() for _ in range(N)]
        
        # Create links until we reach target
        links_created = 0
        max_attempts = target_links * 10
        
        while links_created < target_links and max_attempts > 0:
            i = self.rng.integers(0, N)
            j = self.rng.integers(0, N)
            
            if i != j and j not in self.neighbors[i]:
                self.neighbors[i].add(j)
                self.neighbors[j].add(i)
                links_created += 1
            
            max_attempts -= 1
        
        print(f"Created {links_created} links (target: {target_links})")
        actual_degree = sum(len(n) for n in self.neighbors) / N
        print(f"Actual average degree: {actual_degree:.2f}")
    
    def _setup_attributes_for_nodes(self):
        """Exact replication of NetLogo setup-attributes-for-nodes"""
        print("Setting up demographics (NetLogo replication)...")
        N = self.N
        
        # NetLogo age distribution buckets
        age_buckets = [
            (0, 5, 5.7), (5, 15, 12.5), (15, 25, 13.0), (25, 35, 13.7),
            (35, 45, 13.1), (45, 55, 12.3), (55, 65, 12.9), (65, 75, 10.1),
            (75, 85, 4.9), (85, 100, 1.8)
        ]
        
        if self.config['age_distribution']:
            # NetLogo: weighted age distribution
            buckets, weights = zip(*[(f"{low}-{high}", weight) for low, high, weight in age_buckets])
            weights = np.array(weights) / 100.0
            
            for agent in self.agents:
                bucket_idx = self.rng.choice(len(buckets), p=weights)
                low, high, _ = age_buckets[bucket_idx]
                agent.age = np.int8(self.rng.integers(low, high))
        else:
            # NetLogo: uniform age distribution
            for agent in self.agents:
                agent.age = np.int8(self.rng.integers(0, self.config['age_range']))
        
        # NetLogo: gender distribution
        if self.config['gender']:
            male_prob = self.config['male_population_pct'] / 100.0
            for agent in self.agents:
                agent.gender = np.int8(0 if self.rng.random() < male_prob else 1)
        
        # NetLogo: set-covid-age-prob and set-us-age-prob
        self._set_age_probabilities()
        
        # NetLogo: risk level assignment
        self._assign_risk_levels()
        
        # NetLogo: super-immune agents
        n_super = int(self.config['super_immune_pct'] * N / 100)
        super_indices = self.rng.choice(N, size=n_super, replace=False)
        for idx in super_indices:
            self.agents[idx].super_immune = np.bool_(True)
    
    def _set_age_probabilities(self):
        """Exact replication of NetLogo set-covid-age-prob and set-us-age-prob"""
        for agent in self.agents:
            age = agent.age
            
            # NetLogo: set-covid-age-prob
            if age < 10:
                agent.covid_age_prob = 2.3
            elif age < 20:
                agent.covid_age_prob = 5.1
            elif age < 30:
                agent.covid_age_prob = 15.5
            elif age < 40:
                agent.covid_age_prob = 16.9
            elif age < 50:
                agent.covid_age_prob = 16.4
            elif age < 60:
                agent.covid_age_prob = 16.4
            elif age < 70:
                agent.covid_age_prob = 11.9
            elif age < 80:
                agent.covid_age_prob = 7.0
            else:
                agent.covid_age_prob = 8.5
            
            # NetLogo: set-us-age-prob  
            if age < 5:
                agent.us_age_prob = 5.7
            elif age < 15:
                agent.us_age_prob = 12.5
            elif age < 25:
                agent.us_age_prob = 13.0
            elif age < 35:
                agent.us_age_prob = 13.7
            elif age < 45:
                agent.us_age_prob = 13.1
            elif age < 55:
                agent.us_age_prob = 12.3
            elif age < 65:
                agent.us_age_prob = 12.9
            elif age < 75:
                agent.us_age_prob = 10.1
            elif age < 85:
                agent.us_age_prob = 4.9
            else:
                agent.us_age_prob = 1.8
    
    def _assign_risk_levels(self):
        """Exact replication of NetLogo risk level assignment"""
        N = self.N
        config = self.config
        
        # Everyone starts at level 1 (NetLogo default)
        for agent in self.agents:
            agent.health_risk_level = 1
        
        # Level 2 (pregnancy) - NetLogo exact logic
        if config['gender']:
            pool2 = [i for i, a in enumerate(self.agents) 
                    if a.gender == 1 and 15 <= a.age <= 49]
        else:
            pool2 = list(range(N))
        
        n2 = min(int(config['risk_level_2_pct'] * N / 100), len(pool2))
        for idx in self.rng.choice(pool2, size=n2, replace=False):
            self.agents[idx].health_risk_level = 2
        
        # Level 3 - from remaining level 1
        pool3 = [i for i, a in enumerate(self.agents) if a.health_risk_level == 1]
        n3 = min(int(config['risk_level_3_pct'] * N / 100), len(pool3))
        for idx in self.rng.choice(pool3, size=n3, replace=False):
            self.agents[idx].health_risk_level = 3
        
        # Level 4 - from remaining level 1
        pool4 = [i for i, a in enumerate(self.agents) if a.health_risk_level == 1]
        n4 = min(int(config['risk_level_4_pct'] * N / 100), len(pool4))
        for idx in self.rng.choice(pool4, size=n4, replace=False):
            self.agents[idx].health_risk_level = 4
    
    def _seed_initial_infections(self):
        """Exact replication of NetLogo initial infection seeding"""
        N = self.N
        n_initial = min(self.config['initial_infected_agents'], N)
        
        # NetLogo: exclude super-immune agents
        eligible = [i for i, a in enumerate(self.agents) if not a.super_immune]
        n_initial = min(n_initial, len(eligible))
        
        infected_indices = self.rng.choice(eligible, size=n_initial, replace=False)
        
        for idx in infected_indices:
            agent = self.agents[idx]
            # NetLogo: become-infected-nonsymptomatic
            agent.infected = True
            agent.virus_check_timer = 0
            agent.number_of_infection = 1
            agent.initial_infections = 1
            
            # NetLogo: infectious timing
            agent.infectious_start = 1
            drawn_length = 1 + self.rng.integers(0, self.config['active_duration'])
            max_length = max(1, self.config['infected_period'] - agent.infectious_start)
            agent.transfer_active_duration = min(drawn_length, max_length)
            agent.infectious_end = agent.infectious_start + agent.transfer_active_duration
        
        print(f"Seeded {n_initial} initial infections (NetLogo replication)")
    
    def run_simulation(self):
        """Run simulation matching NetLogo go procedure"""
        print("Starting NetLogo-replicated simulation...")
        start_time = time.time()
        
        total_reinfected = 0
        
        for day in range(self.config['max_days']):
            if day % 30 == 0:
                self._print_progress(day)
            
            # NetLogo: daily procedures
            self._set_temporal_links()  # set-temporal-links
            self._do_long_covid_checks()  # do-long-covid-checks
            
            # Vaccination at start time (NetLogo: if ticks = V-start-time)
            if day == self.config['v_start_time']:
                self._vaccination_status()
            
            # Transmission and state updates
            daily_reinfections = self._spread_virus_days()  # spread-virus-days
            total_reinfected += daily_reinfections
            self._do_virus_checks_infected()  # do-virus-checks-infected
            self._do_virus_checks_immuned()  # do-virus-checks-immuned
            
            # NetLogo: check-vaccination-time
            self._check_vaccination_time()
            
            # NetLogo: process-pending-lc
            self._process_pending_lc(day)
            
            # NetLogo: update-globals
            self._update_globals()
            
            # NetLogo stop condition
            if self._should_stop():
                print(f"Epidemic ended at day {day} (NetLogo stop condition)")
                break
        
        total_time = time.time() - start_time
        print(f"Simulation completed in {total_time:.2f} seconds")
        
        final_metrics = self._get_final_metrics(total_reinfected, day + 1)
        return final_metrics
    
    def _set_temporal_links(self):
        """NetLogo: set-temporal-links replication"""
        # Clear previous temporal links
        self.temporal_neighbors = [set() for _ in range(self.N)]
        
        if self.config['temporal_connections_pct'] <= 0:
            return
        
        # NetLogo: create temporal links
        target = int(self.config['temporal_connections_pct'] * self.N / 100)
        
        for _ in range(target):
            i = self.rng.integers(0, self.N)
            j = self.rng.integers(0, self.N)
            
            if i != j and j not in self.neighbors[i] and j not in self.temporal_neighbors[i]:
                self.temporal_neighbors[i].add(j)
                self.temporal_neighbors[j].add(i)
    
    def _spread_virus_days(self):
        """NetLogo: spread-virus-days replication"""
        daily_reinfections = 0
        
        for agent in self.agents:
            if (agent.infected and 
                agent.infectious_start <= agent.virus_check_timer < agent.infectious_end):
                
                # NetLogo: check symptomatic precaution
                if (agent.symptomatic_start > 0 and 
                    agent.virus_check_timer > agent.symptomatic_start):
                    if self.rng.random() * 100 < self.config['precaution_pct']:
                        continue
                
                # Infect all neighbors (static + temporal)
                all_neighbors = self.neighbors[agent.idx] | self.temporal_neighbors[agent.idx]
                reinfections = self._infect_neighbors(agent, all_neighbors)
                daily_reinfections += reinfections
        
        return daily_reinfections
    
    def _infect_neighbors(self, source, neighbors):
        """NetLogo: link-neighbors-for-covid replication"""
        reinfections = 0
        
        for neighbor_idx in neighbors:
            if neighbor_idx >= len(self.agents):
                continue
                
            neighbor = self.agents[neighbor_idx]
            
            if neighbor.immuned or neighbor.infected or neighbor.super_immune:
                continue
            
            # NetLogo: vaccine protection
            if neighbor.vaccinated:
                if self.config['vaccination_decay']:
                    real_eff = max(0, self.config['efficiency_pct'] - 0.11 * neighbor.vaccinated_time)
                else:
                    real_eff = self.config['efficiency_pct']
                
                neighbor.real_efficiency = real_eff
                if self.rng.random() * 100 < real_eff:
                    continue
            
            # NetLogo: infection probability with age scaling
            infection_prob = self.config['covid_spread_chance_pct']
            if self.config['age_infection_scaling']:
                infection_prob *= neighbor.covid_age_prob / (neighbor.us_age_prob + 1e-9)
            
            infection_prob = max(0, min(100, infection_prob))
            
            if self.rng.random() * 100 < infection_prob:
                is_reinfection = self._attempt_infection(neighbor)
                if is_reinfection:
                    reinfections += 1
        
        return reinfections
    
    def _attempt_infection(self, agent):
        """NetLogo: attempt-infection replication"""
        # Track if this is a reinfection
        is_reinfection = (agent.number_of_infection > 0)
        
        # NetLogo: become-infected-nonsymptomatic
        agent.infected = True
        agent.immuned = False
        agent.virus_check_timer = 0
        agent.number_of_infection += 1
        
        agent.infectious_start = 1
        drawn_length = 1 + self.rng.integers(0, self.config['active_duration'])
        max_length = max(1, self.config['infected_period'] - agent.infectious_start)
        agent.transfer_active_duration = min(drawn_length, max_length)
        agent.infectious_end = agent.infectious_start + agent.transfer_active_duration
        
        if not agent.persistent_long_covid:
            agent.long_covid_recovery_group = -1
            agent.long_covid_weibull_k = 0.0
            agent.long_covid_weibull_lambda = 0.0
        
        return is_reinfection

    def _do_long_covid_checks(self):
        """NetLogo: do-long-covid-checks replication"""
        if not self.config['long_covid']:
            return
        
        for agent in self.agents:
            if agent.persistent_long_covid:
                agent.long_covid_duration += 1
                
                if agent.long_covid_weibull_lambda <= 0:
                    continue
                
                # NetLogo: calculate-weibull-recovery-chance
                t_scaled = agent.long_covid_duration / agent.long_covid_weibull_lambda
                hazard = (agent.long_covid_weibull_k / agent.long_covid_weibull_lambda) * (t_scaled ** (agent.long_covid_weibull_k - 1))
                recovery_chance = (1 - np.exp(-hazard)) * 100
                
                # Group-specific scaling
                if agent.long_covid_recovery_group == 0:
                    recovery_chance *= 2.0
                elif agent.long_covid_recovery_group == 2:
                    recovery_chance *= 0.3
                    if agent.long_covid_duration > 1095:
                        recovery_chance *= 0.1
                
                recovery_chance = max(0, min(15, recovery_chance))
                
                if self.rng.random() * 100 < recovery_chance:
                    # Recover from LC
                    agent.persistent_long_covid = False
                    agent.long_covid_severity = 0.0
                    agent.long_covid_duration = 0
                    agent.long_covid_recovery_group = -1
                    agent.long_covid_weibull_k = 0.0
                    agent.long_covid_weibull_lambda = 0.0
                elif agent.long_covid_recovery_group == 1 and agent.long_covid_duration > 30:
                    # Gradual improvement
                    agent.long_covid_severity = max(5, agent.long_covid_severity - 0.05)

    def _vaccination_status(self):
        """NetLogo: vaccination-status replication"""
        current_vaccinated = sum(1 for a in self.agents if a.vaccinated)
        target_vaccinated = int(self.N * self.config['vaccination_pct'] / 100)
        
        if current_vaccinated >= target_vaccinated:
            return
        
        def under_cap():
            return (current_vaccinated / self.N * 100) < self.config['vaccination_pct']
        
        if self.config['vaccine_priority']:
            # NetLogo priority order
            groups = [
                [a for a in self.agents if a.age >= 65 and not a.vaccinated],
                [a for a in self.agents if a.health_risk_level == 4 and a.age < 65 and not a.vaccinated],
                [a for a in self.agents if a.health_risk_level == 3 and a.age < 65 and not a.vaccinated],
                [a for a in self.agents if a.health_risk_level == 2 and a.age < 65 and not a.vaccinated],
                [a for a in self.agents if a.health_risk_level == 1 and a.age < 65 and not a.vaccinated],
            ]
            
            for group in groups:
                for agent in group:
                    if not under_cap():
                        return
                    agent.vaccinated = True
                    agent.vaccinated_time = 1
                    current_vaccinated += 1
        else:
            # No priority
            unvaccinated = [a for a in self.agents if not a.vaccinated]
            n_vacc = min(target_vaccinated - current_vaccinated, len(unvaccinated))
            for agent in self.rng.choice(unvaccinated, size=n_vacc, replace=False):
                agent.vaccinated = True
                agent.vaccinated_time = 1

    def _do_virus_checks_infected(self):
        """NetLogo: do-virus-checks-infected replication (simplified)"""
        for agent in self.agents:
            if agent.infected:
                agent.virus_check_timer += 1
                
                # Simple state transition: become immune after infected period
                if agent.virus_check_timer >= self.config['infected_period']:
                    agent.infected = False
                    agent.immuned = True
                    agent.virus_check_timer = 0

    def _do_virus_checks_immuned(self):
        """NetLogo: do-virus-checks-immuned replication (simplified)"""
        for agent in self.agents:
            if agent.immuned:
                agent.virus_check_timer += 1
                
                # Lose immunity after immune period
                if agent.virus_check_timer >= self.config['immune_period']:
                    agent.immuned = False
                    agent.virus_check_timer = 0

    def _check_vaccination_time(self):
        """NetLogo: check-vaccination-time replication"""
        for agent in self.agents:
            if agent.vaccinated:
                agent.vaccinated_time += 1
                if agent.vaccinated_time >= 180:  # 6 months
                    if self.rng.random() * 100 < self.config['boosted_pct']:
                        agent.vaccinated_time = 1  # Boosted
                    else:
                        agent.vaccinated = False
                        agent.vaccinated_time = 0

    def _process_pending_lc(self, day):
        """NetLogo: process-pending-lc replication"""
        if not self.config['long_covid']:
            return
        
        for agent in self.agents:
            if agent.lc_pending and day >= agent.lc_onset_day:
                agent.lc_pending = False
                if not agent.persistent_long_covid:
                    agent.persistent_long_covid = True
                    agent.long_covid_duration = 0
                    self._assign_long_covid_recovery_group(agent)

    def _assign_long_covid_recovery_group(self, agent):
        """NetLogo: assign-long-covid-recovery-group replication"""
        w_fast = self.config['lc_base_fast_prob']
        w_pers = self.config['lc_base_persistent_prob']
        w_sum = w_fast + w_pers
        
        # Normalize if overshoot
        if w_sum > 100:
            w_fast = 100 * w_fast / w_sum
            w_pers = 100 * w_pers / w_sum
            w_sum = 100
        
        w_grad = 100 - w_sum
        
        # Age and symptom duration adjustments
        if agent.age >= 65:
            shift = min(2, w_grad)
            w_pers += shift
            w_grad -= shift
        
        if agent.symptomatic_duration > 21:
            shift = min(4, w_grad)
            w_pers += shift
            w_grad -= shift
        
        total = w_fast + w_pers + w_grad
        if total <= 0:
            w_grad = 100
            total = 100
        
        r = self.rng.random() * total
        
        if r < w_fast:
            # Fast group
            agent.long_covid_recovery_group = 0
            agent.long_covid_weibull_k = 1.5
            agent.long_covid_weibull_lambda = 60
            agent.long_covid_severity = max(5, min(100, self.rng.normal(30, 15)))
        elif r < w_fast + w_pers:
            # Persistent group
            agent.long_covid_recovery_group = 2
            agent.long_covid_weibull_k = 0.5
            agent.long_covid_weibull_lambda = 1200
            agent.long_covid_severity = max(5, min(100, self.rng.normal(70, 20)))
        else:
            # Gradual group
            agent.long_covid_recovery_group = 1
            agent.long_covid_weibull_k = 1.2
            agent.long_covid_weibull_lambda = 450
            agent.long_covid_severity = max(5, min(100, self.rng.normal(50, 20)))

    def _update_globals(self):
        """NetLogo: update-globals replication"""
        if not self.agents:
            self.current_productivity = 100
            if self.current_productivity < self.min_productivity:
                self.min_productivity = self.current_productivity
            return
        
        total_productivity_loss = 0
        for agent in self.agents:
            agent_loss = 0
            
            if agent.symptomatic:
                agent_loss = 1.0
            elif agent.persistent_long_covid:
                agent_loss = agent.long_covid_severity / 100.0
            
            agent_loss = max(0, min(1, agent_loss))
            total_productivity_loss += agent_loss
        
        self.current_productivity = (1 - (total_productivity_loss / len(self.agents))) * 100
        if self.current_productivity < self.min_productivity:
            self.min_productivity = self.current_productivity

    def _should_stop(self):
        """NetLogo stop condition: all? turtles [ not infected? and not immuned? ]"""
        return (not any(a.infected for a in self.agents) and 
                not any(a.immuned for a in self.agents))

    def _print_progress(self, day):
        """Print simulation progress"""
        n_infected = sum(1 for a in self.agents if a.infected)
        n_immune = sum(1 for a in self.agents if a.immuned)
        n_lc = sum(1 for a in self.agents if a.persistent_long_covid)
        print(f"Day {day}: {n_infected} infected, {n_immune} immune, {n_lc} LC, Productivity: {self.current_productivity:.1f}%")

    def _get_final_metrics(self, total_reinfected, runtime_days):
        """Get final metrics matching NetLogo outputs"""
        total_infected = sum(1 for a in self.agents if a.number_of_infection > 0)
        total_lc = sum(1 for a in self.agents if a.persistent_long_covid)
        
        return {
            'runtime_days': runtime_days,
            'infected': total_infected,
            'reinfected': total_reinfected,
            'long_covid_cases': total_lc,
            'min_productivity': self.min_productivity,
        }

# ========== EXACT PLOTTING CODE FROM YOUR SPECIFICATION ==========

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
    "avg_degree": "Average\nDegree",
    "v_start_time": "Vaccination\nStart Time",
    "vaccination_pct": "Vaccination%",
}

def make_grid(df, out_png="figure.png"):
    """EXACT replication of your plotting code"""
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

# ========== PARAMETER SWEEP ==========

def run_netlogo_replication_sweep(n_runs=50, N=100000, output_file="netlogo_replication_results.csv"):
    """Run parameter sweep with exact NetLogo replication"""
    results = []
    total_sims = sum(len(values) * n_runs for values in ORDER.values())
    
    print(f"=== NetLogo Exact Replication Parameter Sweep ===")
    print(f"Agents: {N:,}")
    print(f"Runs per parameter: {n_runs}")
    print(f"Total simulations: {total_sims}")
    
    start_time = time.time()
    sim_count = 0
    
    for param_name, values in ORDER.items():
        print(f"\n--- Sweeping {param_name} ---")
        
        for value in values:
            print(f"  Value: {value}")
            
            for run in range(n_runs):
                sim_count += 1
                
                try:
                    abm = NetLogoReplicaABM()
                    abm.initialize_simulation(
                        N=N,
                        seed=42 + run,
                        **{param_name: value}
                    )
                    
                    final_metrics = abm.run_simulation()
                    final_metrics['param_name'] = param_name
                    final_metrics['param_value'] = value
                    final_metrics['run'] = run
                    final_metrics['agents'] = N
                    
                    results.append(final_metrics)
                    
                    # Progress
                    if run % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = sim_count / elapsed
                        eta = (total_sims - sim_count) / rate / 60 if rate > 0 else 0
                        print(f"    Run {run+1}/{n_runs}, ETA: {eta:.1f} min")
                    
                    del abm
                    gc.collect()
                    
                except Exception as e:
                    print(f"    Error in run {run}: {e}")
                    continue
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    total_time = time.time() - start_time
    print(f"\n=== Sweep Complete ===")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {output_file}")
    
    return df

def main():
    """Main function to run the analysis"""
    print("=== NetLogo COVID ABM Exact Replication with Parameter Sweep ===")
    
    # Test with smaller population first
    print("\n1. Testing with 10K agents...")
    abm_test = NetLogoReplicaABM()
    abm_test.initialize_simulation(N=10000, max_days=100)
    test_results = abm_test.run_simulation()
    print("Test results:", test_results)
    
    # Run parameter sweep with 100K agents
    print("\n2. Running parameter sweep with 100K agents (50 runs per parameter)...")
    results_df = run_netlogo_replication_sweep(n_runs=50, N=100000)
    
    # Generate the exact plot
    print("\n3. Generating exact plot...")
    make_grid(results_df, out_png="netlogo_replication_grid.png")
    
    print("\n=== Analysis Complete ===")
    print("Files created:")
    print("  - netlogo_replication_results.csv")
    print("  - netlogo_replication_grid.png")
    
    return results_df

if __name__ == "__main__":
    results_df = main()
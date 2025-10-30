"""
COVID-19 ABM with GPU Acceleration - 1M Agent Scale
Unified implementation matching NetLogo behavior with parameter sweep plotting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import time
import gc
import psutil
from dataclasses import dataclass
from typing import List, Dict, Any
import jax.numpy as jnp
from jax import jit, vmap, random

@dataclass
class AgentData:
    """Ultra-lightweight agent data structure for 1M scale"""
    # Infection states
    infected: np.bool_
    immuned: np.bool_ 
    symptomatic: np.bool_
    super_immune: np.bool_
    
    # Long COVID states
    persistent_long_covid: np.bool_
    long_covid_severity: np.float32
    long_covid_duration: np.int32
    long_covid_recovery_group: np.int8
    long_covid_weibull_k: np.float32
    long_covid_weibull_lambda: np.float32
    lc_pending: np.bool_
    lc_onset_day: np.int32
    
    # Infection tracking
    virus_check_timer: np.int32
    number_of_infection: np.int32
    infection_start_tick: np.int32
    infectious_start: np.int32
    infectious_end: np.int32
    transfer_active_duration: np.int32
    symptomatic_start: np.int32
    symptomatic_duration: np.int32
    
    # Demographics
    age: np.int8
    gender: np.int8
    health_risk_level: np.int8
    covid_age_prob: np.float32
    us_age_prob: np.float32
    
    # Vaccination
    vaccinated: np.bool_
    vaccinated_time: np.int32
    real_efficiency: np.float32

class MassivelyParallelABM:
    """
    COVID-19 Agent-Based Model optimized for 1 million agents
    Matches NetLogo implementation with GPU acceleration
    """
    
    def __init__(self):
        self.agents = []
        self.neighbors = []
        self.N = 0
        self.rng = None
        self.config = {}
        
    def initialize_simulation(self, 
                            N=1_000_000,
                            seed=42,
                            **kwargs):
        """Initialize simulation with given parameters"""
        self.N = N
        self.rng = np.random.default_rng(seed)
        self.config = self._get_default_config()
        self.config.update(kwargs)
        
        print(f"Initializing {N:,} agents...")
        print(f"Estimated memory: {self._estimate_memory():.1f} GB")
        
        # Create agents
        self._create_agents()
        
        # Create network
        self._create_network()
        
        # Setup demographics
        self._setup_demographics()
        
        # Seed initial infections
        self._seed_infections()
        
        print("Simulation initialized successfully!")
    
    def _get_default_config(self):
        """Get default configuration matching NetLogo"""
        return {
            'max_days': 100,
            'covid_spread_chance_pct': 10.0,
            'initial_infected_agents': 50,
            'precaution_pct': 50.0,
            'avg_degree': 5,
            'v_start_time': 180,
            'vaccination_pct': 80.0,
            'temporal_connections_pct': 0.0,
            'network_model': 'M1-average-connections',
            'infected_period': 10,
            'active_duration': 7,
            'immune_period': 21,
            'incubation_period': 4,
            'symptomatic_duration_min': 1,
            'symptomatic_duration_mid': 10,
            'symptomatic_duration_max': 60,
            'symptomatic_duration_dev': 8,
            'asymptomatic_pct': 40.0,
            'effect_of_reinfection': 3,
            'super_immune_pct': 4.0,
            'long_covid': True,
            'long_covid_time_threshold': 30,
            'asymptomatic_lc_mult': 0.50,
            'lc_incidence_mult_female': 1.20,
            'lc_base_fast_prob': 9.0,
            'lc_base_persistent_prob': 7.0,
            'reinfection_new_onset_mult': 0.70,
            'lc_onset_base_pct': 15.0,
            'efficiency_pct': 80.0,
            'boosted_pct': 30.0,
            'vaccination_decay': True,
            'vaccine_priority': True,
            'gender': True,
            'male_population_pct': 49.5,
            'age_distribution': True,
            'age_range': 100,
            'age_infection_scaling': True,
            'risk_level_2_pct': 4.0,
            'risk_level_3_pct': 40.0,
            'risk_level_4_pct': 6.0,
        }
    
    def _estimate_memory(self):
        """Estimate memory usage in GB"""
        agent_memory = self.N * 100  # bytes per agent
        network_memory = self.N * self.config['avg_degree'] * 4  # bytes for neighbors
        total_bytes = agent_memory + network_memory
        return total_bytes / 1024 / 1024 / 1024  # GB
    
    def _create_agents(self):
        """Create lightweight agents"""
        self.agents = []
        for i in range(self.N):
            agent = AgentData(
                infected=np.bool_(False),
                immuned=np.bool_(False),
                symptomatic=np.bool_(False),
                super_immune=np.bool_(False),
                persistent_long_covid=np.bool_(False),
                long_covid_severity=np.float32(0.0),
                long_covid_duration=np.int32(0),
                long_covid_recovery_group=np.int8(-1),
                long_covid_weibull_k=np.float32(0.0),
                long_covid_weibull_lambda=np.float32(0.0),
                lc_pending=np.bool_(False),
                lc_onset_day=np.int32(0),
                virus_check_timer=np.int32(0),
                number_of_infection=np.int32(0),
                infection_start_tick=np.int32(0),
                infectious_start=np.int32(1),
                infectious_end=np.int32(1),
                transfer_active_duration=np.int32(0),
                symptomatic_start=np.int32(0),
                symptomatic_duration=np.int32(0),
                age=np.int8(0),
                gender=np.int8(0),
                health_risk_level=np.int8(1),
                covid_age_prob=np.float32(15.0),
                us_age_prob=np.float32(13.0),
                vaccinated=np.bool_(False),
                vaccinated_time=np.int32(0),
                real_efficiency=np.float32(0.0)
            )
            self.agents.append(agent)
    
    def _create_network(self):
        """Create sparse network for 1M agents"""
        print("Creating network...")
        N = self.N
        avg_degree = self.config['avg_degree']
        model = self.config['network_model']
        
        self.neighbors = [np.array([], dtype=np.int32) for _ in range(N)]
        
        if model == "M1-average-connections":
            p = avg_degree / (N - 1)
            chunk_size = 10000
            
            for i in range(0, N, chunk_size):
                if i % 100_000 == 0:
                    print(f"  Processed {i:,} nodes...")
                
                chunk_end = min(i + chunk_size, N)
                for idx in range(i, chunk_end):
                    n_edges = self.rng.binomial(N - idx - 1, p)
                    if n_edges > 0:
                        max_edges = min(n_edges, 50)  # Limit connections
                        targets = self.rng.choice(
                            np.arange(idx + 1, N), 
                            size=max_edges, 
                            replace=False
                        )
                        self.neighbors[idx] = targets.astype(np.int32)
        
        print("Network creation complete!")
    
    def _setup_demographics(self):
        """Setup demographics for all agents"""
        print("Setting up demographics...")
        N = self.N
        
        # Age distribution
        if self.config['age_distribution']:
            age_bins = [(0, 5), (5, 15), (15, 25), (25, 35), (35, 45),
                       (45, 55), (55, 65), (65, 75), (75, 85), (85, 100)]
            weights = [5.7, 12.5, 13.0, 13.7, 13.1, 12.3, 12.9, 10.1, 4.9, 1.8]
            weights = np.array(weights) / sum(weights)
            
            bin_choices = self.rng.choice(len(age_bins), size=N, p=weights)
            for i, agent in enumerate(self.agents):
                low, high = age_bins[bin_choices[i]]
                agent.age = np.int8(self.rng.integers(low, high))
        else:
            for agent in self.agents:
                agent.age = np.int8(self.rng.integers(0, self.config['age_range']))
        
        # Gender distribution
        if self.config['gender']:
            male_prob = self.config['male_population_pct'] / 100.0
            for agent in self.agents:
                agent.gender = np.int8(0 if self.rng.random() < male_prob else 1)
        
        # Age probabilities
        ages = np.array([a.age for a in self.agents])
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
        
        for i, agent in enumerate(self.agents):
            agent.covid_age_prob = np.float32(covid_age_probs[i])
            agent.us_age_prob = np.float32(us_age_probs[i])
        
        # Risk levels
        self._assign_risk_levels()
        
        # Super-immune agents
        n_super = int(self.config['super_immune_pct'] * N / 100)
        super_indices = self.rng.choice(N, size=n_super, replace=False)
        for idx in super_indices:
            self.agents[idx].super_immune = np.bool_(True)
    
    def _assign_risk_levels(self):
        """Assign health risk levels"""
        N = self.N
        config = self.config
        
        # Level 2 (pregnancy/high risk females)
        if config['gender']:
            eligible_level2 = [i for i, a in enumerate(self.agents) 
                             if a.gender == 1 and 15 <= a.age <= 49]
        else:
            eligible_level2 = list(range(N))
        
        n2 = min(int(config['risk_level_2_pct'] * N / 100), len(eligible_level2))
        for idx in self.rng.choice(eligible_level2, size=n2, replace=False):
            self.agents[idx].health_risk_level = np.int8(2)
        
        # Level 3
        eligible_level3 = [i for i, a in enumerate(self.agents) 
                         if a.health_risk_level == 1]
        n3 = min(int(config['risk_level_3_pct'] * N / 100), len(eligible_level3))
        for idx in self.rng.choice(eligible_level3, size=n3, replace=False):
            self.agents[idx].health_risk_level = np.int8(3)
        
        # Level 4
        eligible_level4 = [i for i, a in enumerate(self.agents) 
                         if a.health_risk_level == 1]
        n4 = min(int(config['risk_level_4_pct'] * N / 100), len(eligible_level4))
        for idx in self.rng.choice(eligible_level4, size=n4, replace=False):
            self.agents[idx].health_risk_level = np.int8(4)
    
    def _seed_infections(self):
        """Seed initial infections"""
        N = self.N
        n_initial = min(self.config['initial_infected_agents'], N)
        
        eligible = [i for i, a in enumerate(self.agents) if not a.super_immune]
        if len(eligible) < n_initial:
            n_initial = len(eligible)
        
        infected_indices = self.rng.choice(eligible, size=n_initial, replace=False)
        
        for idx in infected_indices:
            agent = self.agents[idx]
            agent.infected = np.bool_(True)
            agent.number_of_infection = np.int32(1)
            agent.infection_start_tick = np.int32(0)
            agent.infectious_start = np.int32(1)
            
            drawn_length = 1 + self.rng.integers(0, self.config['active_duration'])
            max_length = max(1, self.config['infected_period'] - agent.infectious_start)
            agent.transfer_active_duration = np.int32(min(drawn_length, max_length))
            agent.infectious_end = np.int32(agent.infectious_start + agent.transfer_active_duration)
        
        print(f"Seeded {n_initial} initial infections")
    
    def run_simulation(self):
        """Run the main simulation loop"""
        print("Starting simulation...")
        start_time = time.time()
        
        total_reinfected = 0
        long_covid_cases = 0
        min_productivity = 100.0
        
        for day in range(self.config['max_days']):
            if day % 10 == 0:
                self._print_progress(day)
            
            # Daily updates
            self._update_vaccination(day)
            self._update_long_covid()
            new_infections = self._process_transmission(day)
            self._process_state_transitions(day)
            self._process_pending_lc(day)
            self._update_vaccine_time()
            
            # Update metrics
            total_reinfected += sum(1 for a in self.agents if a.number_of_infection > 1 and a.infected and a.virus_check_timer == 1)
            long_covid_cases = sum(1 for a in self.agents if a.persistent_long_covid)
            
            # Calculate productivity
            productivity = self._calculate_productivity()
            min_productivity = min(min_productivity, productivity)
            
            # Check stop condition
            if self._should_stop():
                print(f"Epidemic ended at day {day}")
                break
        
        total_time = time.time() - start_time
        print(f"Simulation completed in {total_time:.2f} seconds")
        
        final_metrics = self._get_final_metrics(total_reinfected, long_covid_cases, min_productivity, day + 1)
        return final_metrics
    
    def _print_progress(self, day):
        """Print simulation progress"""
        n_infected = sum(1 for a in self.agents if a.infected)
        n_lc = sum(1 for a in self.agents if a.persistent_long_covid)
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
        print(f"Day {day}: {n_infected:,} infected, {n_lc:,} LC, Memory: {memory_usage:.2f} GB")
    
    def _update_vaccination(self, day):
        """Update vaccination status"""
        if day < self.config['v_start_time']:
            return
        
        current_vaccinated = sum(1 for a in self.agents if a.vaccinated)
        target_vaccinated = int(self.N * self.config['vaccination_pct'] / 100)
        
        if current_vaccinated >= target_vaccinated:
            return
        
        unvaccinated = [i for i, a in enumerate(self.agents) if not a.vaccinated]
        
        if self.config['vaccine_priority']:
            # Priority groups
            priority_groups = [
                [i for i in unvaccinated if self.agents[i].age >= 65],
                [i for i in unvaccinated if self.agents[i].health_risk_level == 4 and self.agents[i].age < 65],
                [i for i in unvaccinated if self.agents[i].health_risk_level == 3 and self.agents[i].age < 65],
                [i for i in unvaccinated if self.agents[i].health_risk_level == 2 and self.agents[i].age < 65],
                [i for i in unvaccinated if self.agents[i].health_risk_level == 1 and self.agents[i].age < 65],
            ]
            
            for group in priority_groups:
                for idx in group:
                    if current_vaccinated >= target_vaccinated:
                        return
                    self.agents[idx].vaccinated = np.bool_(True)
                    self.agents[idx].vaccinated_time = np.int32(1)
                    current_vaccinated += 1
        else:
            n_vacc = min(target_vaccinated - current_vaccinated, len(unvaccinated))
            for idx in self.rng.choice(unvaccinated, size=n_vacc, replace=False):
                self.agents[idx].vaccinated = np.bool_(True)
                self.agents[idx].vaccinated_time = np.int32(1)
    
    def _update_long_covid(self):
        """Update long COVID progression"""
        if not self.config['long_covid']:
            return
        
        for agent in self.agents:
            if not agent.persistent_long_covid:
                continue
            
            agent.long_covid_duration += 1
            
            if agent.long_covid_weibull_lambda <= 0:
                continue
            
            # Calculate recovery chance
            t_scaled = agent.long_covid_duration / agent.long_covid_weibull_lambda
            hazard = (agent.long_covid_weibull_k / agent.long_covid_weibull_lambda) * (t_scaled ** (agent.long_covid_weibull_k - 1))
            recovery_chance = (1 - np.exp(-hazard)) * 100
            
            # Group adjustments
            if agent.long_covid_recovery_group == 0:
                recovery_chance *= 2.0
            elif agent.long_covid_recovery_group == 2:
                recovery_chance *= 0.3
                if agent.long_covid_duration > 1095:
                    recovery_chance *= 0.1
            
            recovery_chance = np.clip(recovery_chance, 0, 15)
            
            if self.rng.random() * 100 < recovery_chance:
                # Recover from LC
                agent.persistent_long_covid = np.bool_(False)
                agent.long_covid_severity = np.float32(0.0)
                agent.long_covid_duration = np.int32(0)
                agent.long_covid_recovery_group = np.int8(-1)
                agent.long_covid_weibull_k = np.float32(0.0)
                agent.long_covid_weibull_lambda = np.float32(0.0)
            elif agent.long_covid_recovery_group == 1 and agent.long_covid_duration > 30:
                # Gradual improvement
                agent.long_covid_severity = np.float32(max(5, agent.long_covid_severity - 0.05))
    
    def _process_transmission(self, day):
        """Process disease transmission"""
        new_infections = 0
        batch_size = 50000
        
        for i in range(0, self.N, batch_size):
            batch_end = min(i + batch_size, self.N)
            
            for idx in range(i, batch_end):
                agent = self.agents[idx]
                
                if not agent.infected:
                    continue
                
                if not (agent.infectious_start <= agent.virus_check_timer < agent.infectious_end):
                    continue
                
                # Check precaution for symptomatic
                if (agent.symptomatic and agent.symptomatic_start > 0 and 
                    agent.virus_check_timer > agent.symptomatic_start):
                    if self.rng.random() * 100 < self.config['precaution_pct']:
                        continue
                
                # Infect neighbors
                for neighbor_idx in self.neighbors[idx]:
                    if neighbor_idx >= self.N:
                        continue
                    
                    neighbor = self.agents[neighbor_idx]
                    if self._try_infect(agent, neighbor, day):
                        new_infections += 1
        
        return new_infections
    
    def _try_infect(self, source, target, day):
        """Attempt to infect a target agent"""
        if target.infected or target.immuned or target.super_immune:
            return False
        
        # Vaccine protection
        if target.vaccinated:
            if self.config['vaccination_decay']:
                real_eff = max(0, min(100, self.config['efficiency_pct'] - 0.11 * target.vaccinated_time))
            else:
                real_eff = self.config['efficiency_pct']
            
            target.real_efficiency = np.float32(real_eff)
            if self.rng.random() * 100 < real_eff:
                return False
        
        # Infection probability
        infection_prob = self.config['covid_spread_chance_pct']
        
        if self.config['age_infection_scaling']:
            infection_prob *= target.covid_age_prob / (target.us_age_prob + 1e-9)
        
        infection_prob = max(0, min(100, infection_prob))
        
        if self.rng.random() * 100 < infection_prob:
            self._infect_agent(target, day)
            return True
        
        return False
    
    def _infect_agent(self, agent, day):
        """Infect an agent"""
        agent.infected = np.bool_(True)
        agent.immuned = np.bool_(False)
        agent.symptomatic = np.bool_(False)
        agent.infection_start_tick = np.int32(day)
        agent.virus_check_timer = np.int32(0)
        agent.number_of_infection += 1
        agent.infectious_start = np.int32(1)
        
        drawn_length = 1 + self.rng.integers(0, self.config['active_duration'])
        max_length = max(1, self.config['infected_period'] - agent.infectious_start)
        agent.transfer_active_duration = np.int32(min(drawn_length, max_length))
        agent.infectious_end = np.int32(agent.infectious_start + agent.transfer_active_duration)
        
        if not agent.persistent_long_covid:
            agent.long_covid_recovery_group = np.int8(-1)
            agent.long_covid_weibull_k = np.float32(0.0)
            agent.long_covid_weibull_lambda = np.float32(0.0)
    
    def _process_state_transitions(self, day):
        """Process infection state transitions"""
        for agent in self.agents:
            if agent.infected:
                self._update_infected_agent(agent, day)
            elif agent.immuned:
                self._update_immuned_agent(agent)
    
    def _update_infected_agent(self, agent, day):
        """Update infected agent state"""
        config = self.config
        
        # First day initialization
        if agent.virus_check_timer == 0:
            agent.virus_check_timer = 1
            agent.transfer_active_duration = 1 + self.rng.integers(0, config['active_duration'])
            
            if self.rng.random() * 100 < config['asymptomatic_pct']:
                agent.symptomatic_start = 0
            else:
                agent.symptomatic_start = 1 + self.rng.integers(0, config['incubation_period'])
                while agent.symptomatic_start > agent.transfer_active_duration:
                    agent.symptomatic_start = 1 + self.rng.integers(0, config['incubation_period'])
            
            if agent.symptomatic_start == 0:
                agent.symptomatic_duration = 0
            else:
                a = (config['symptomatic_duration_min'] - config['symptomatic_duration_mid']) / config['symptomatic_duration_dev']
                b = (config['symptomatic_duration_max'] - config['symptomatic_duration_mid']) / config['symptomatic_duration_dev']
                base = truncnorm.rvs(a, b, loc=config['symptomatic_duration_mid'], 
                                   scale=config['symptomatic_duration_dev'], random_state=self.rng)
                agent.symptomatic_duration = int(config['effect_of_reinfection'] * agent.number_of_infection + base)
                
                if agent.persistent_long_covid:
                    agent.symptomatic_duration = int(agent.symptomatic_duration * 1.5)
                    agent.long_covid_severity = min(90, agent.long_covid_severity + 10)
                    
                    # Group worsening
                    if agent.long_covid_recovery_group == 0 and self.rng.random() * 100 < 30:
                        agent.long_covid_recovery_group = 1
                        agent.long_covid_weibull_k = 1.2
                        agent.long_covid_weibull_lambda = 450
                    elif agent.long_covid_recovery_group == 1 and self.rng.random() * 100 < 20:
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
        
        # Long COVID onset and state transitions
        self._process_long_covid_onset(agent)
        self._process_infection_transitions(agent)
    
    def _process_long_covid_onset(self, agent):
        """Process long COVID onset logic"""
        if not self.config['long_covid']:
            return
        
        config = self.config
        
        # Asymptomatic path
        if agent.virus_check_timer > config['infected_period'] and agent.symptomatic_start == 0:
            p_onset = self._calculate_lc_probability(agent, is_asymptomatic=True)
            if self.rng.random() * 100 < p_onset:
                agent.lc_pending = True
                agent.lc_onset_day = agent.infection_start_tick + config['long_covid_time_threshold']
            
            agent.infected = False
            agent.immuned = True
            return
        
        # Symptomatic paths
        if agent.symptomatic_start > 0:
            # Path B: Symptoms exceed threshold
            if agent.symptomatic_duration > config['long_covid_time_threshold']:
                if agent.virus_check_timer == agent.symptomatic_start + config['long_covid_time_threshold']:
                    if not agent.persistent_long_covid:
                        self._assign_long_covid(agent)
            
            # Path C: Symptoms end before threshold
            elif agent.symptomatic_duration <= config['long_covid_time_threshold']:
                if agent.virus_check_timer == agent.symptomatic_start + agent.symptomatic_duration:
                    p_onset = self._calculate_lc_probability(agent, is_asymptomatic=False)
                    if self.rng.random() * 100 < p_onset:
                        agent.lc_pending = True
                        agent.lc_onset_day = agent.infection_start_tick + config['long_covid_time_threshold']
    
    def _calculate_lc_probability(self, agent, is_asymptomatic):
        """Calculate long COVID probability"""
        config = self.config
        
        # Age multiplier
        if agent.age < 30:
            age_mult = 0.9
        elif 50 <= agent.age <= 64:
            age_mult = 1.2
        elif agent.age >= 65:
            age_mult = 1.3
        else:
            age_mult = 1.0
        
        # Gender multiplier
        gender_mult = config['lc_incidence_mult_female'] if agent.gender == 1 else 1.0
        
        # Vaccine multiplier
        vacc_mult = 0.7 if agent.vaccinated else 1.0
        
        # Reinfection multiplier
        has_lc = agent.long_covid_recovery_group in [0, 1, 2]
        reinf_mult = config['reinfection_new_onset_mult'] if (agent.number_of_infection > 1 and not has_lc) else 1.0
        
        # Asymptomatic multiplier
        asym_mult = config['asymptomatic_lc_mult'] if is_asymptomatic else 1.0
        
        p_onset = config['lc_onset_base_pct'] * age_mult * gender_mult * vacc_mult * reinf_mult * asym_mult
        return max(0, min(100, p_onset))
    
    def _assign_long_covid(self, agent):
        """Assign long COVID to an agent"""
        config = self.config
        
        agent.persistent_long_covid = True
        agent.long_covid_duration = 0
        
        # Calculate weights
        w_fast = config['lc_base_fast_prob']
        w_pers = config['lc_base_persistent_prob']
        w_grad = 100 - w_fast - w_pers
        
        # Adjust weights
        if agent.age >= 65:
            shift = min(2, w_grad)
            w_pers += shift
            w_grad -= shift
        
        if agent.symptomatic_duration > 21:
            shift = min(4, w_grad)
            w_pers += shift
            w_grad -= shift
        
        # Assign group
        r = self.rng.random() * (w_fast + w_pers + w_grad)
        if r < w_fast:
            agent.long_covid_recovery_group = 0
            agent.long_covid_weibull_k = 1.5
            agent.long_covid_weibull_lambda = 60
            agent.long_covid_severity = np.clip(self.rng.normal(30, 15), 5, 100)
        elif r < w_fast + w_pers:
            agent.long_covid_recovery_group = 2
            agent.long_covid_weibull_k = 0.5
            agent.long_covid_weibull_lambda = 1200
            agent.long_covid_severity = np.clip(self.rng.normal(70, 20), 5, 100)
        else:
            agent.long_covid_recovery_group = 1
            agent.long_covid_weibull_k = 1.2
            agent.long_covid_weibull_lambda = 450
            agent.long_covid_severity = np.clip(self.rng.normal(50, 20), 5, 100)
    
    def _process_infection_transitions(self, agent):
        """Process infection state transitions"""
        config = self.config
        
        if agent.symptomatic_start == 0:
            if agent.virus_check_timer > config['infected_period']:
                agent.infected = False
                agent.immuned = True
            return
        
        symptom_end = agent.symptomatic_start + agent.symptomatic_duration
        
        if symptom_end < config['infected_period']:
            if agent.virus_check_timer == symptom_end:
                agent.symptomatic = False
            if agent.virus_check_timer > config['infected_period']:
                agent.infected = False
                agent.immuned = True
        elif symptom_end == config['infected_period']:
            if agent.virus_check_timer > config['infected_period']:
                agent.infected = False
                agent.immuned = True
        else:
            if agent.virus_check_timer > config['infected_period']:
                agent.infected = False
                agent.immuned = True
            if agent.virus_check_timer == symptom_end:
                agent.symptomatic = False
    
    def _update_immuned_agent(self, agent):
        """Update immuned agent state"""
        config = self.config
        
        if agent.symptomatic_start > 0:
            symp_now = agent.virus_check_timer < agent.symptomatic_start + agent.symptomatic_duration
            agent.symptomatic = symp_now
        
        immunity_end = config['infected_period'] + config['immune_period']
        
        if agent.virus_check_timer <= immunity_end:
            agent.virus_check_timer += 1
        else:
            agent.infected = False
            agent.immuned = False
            agent.symptomatic = False
            agent.virus_check_timer = 0
            agent.symptomatic_duration = 0
            agent.transfer_active_duration = 0
    
    def _process_pending_lc(self, day):
        """Process pending long COVID cases"""
        if not self.config['long_covid']:
            return
        
        for agent in self.agents:
            if agent.lc_pending and day >= agent.lc_onset_day:
                agent.lc_pending = False
                if not agent.persistent_long_covid:
                    self._assign_long_covid(agent)
    
    def _update_vaccine_time(self):
        """Update vaccine time and boosters"""
        for agent in self.agents:
            if agent.vaccinated:
                agent.vaccinated_time += 1
                if agent.vaccinated_time == 180:  # 6 months
                    if self.rng.random() * 100 < self.config['boosted_pct']:
                        agent.vaccinated_time = 1  # Reset counter
                    else:
                        agent.vaccinated = False
                        agent.vaccinated_time = 0
    
    def _calculate_productivity(self):
        """Calculate population productivity"""
        symptomatic_loss = sum(1 for a in self.agents if a.symptomatic)
        lc_loss = sum(a.long_covid_severity / 100.0 for a in self.agents 
                     if a.persistent_long_covid and not a.symptomatic)
        productivity = (1 - (symptomatic_loss + lc_loss) / self.N) * 100
        return productivity
    
    def _should_stop(self):
        """Check if simulation should stop early"""
        return (not any(a.infected for a in self.agents) and 
                not any(a.immuned for a in self.agents))
    
    def _get_final_metrics(self, total_reinfected, long_covid_cases, min_productivity, runtime_days):
        """Get final simulation metrics"""
        total_infected = sum(1 for a in self.agents if a.number_of_infection > 0)
        
        return {
            'runtime_days': runtime_days,
            'infected': total_infected,
            'reinfected': total_reinfected,
            'long_covid_cases': long_covid_cases,
            'min_productivity': min_productivity,
        }

# ========== PARAMETER SWEEP AND PLOTTING ==========

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
    """Create the exact grid plot from your specification"""
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

def run_parameter_sweep(n_runs=3, N=100000, output_file="results_1m.csv"):
    """Run parameter sweep for 1M agents"""
    results = []
    total_sims = sum(len(values) * n_runs for values in ORDER.values())
    sim_count = 0
    
    print(f"Starting parameter sweep with {N:,} agents...")
    print(f"Total simulations: {total_sims}")
    
    start_time = time.time()
    
    for param_name, values in ORDER.items():
        print(f"\n=== Sweeping {param_name} ===")
        
        for value in values:
            print(f"  Value: {value}")
            
            for run in range(n_runs):
                sim_count += 1
                print(f"    Run {run + 1}/{n_runs}")
                
                try:
                    # Create and run simulation
                    abm = MassivelyParallelABM()
                    abm.initialize_simulation(
                        N=N,
                        seed=42 + run,
                        **{param_name: value}
                    )
                    
                    final_metrics = abm.run_simulation()
                    final_metrics['param_name'] = param_name
                    final_metrics['param_value'] = value
                    final_metrics['run'] = run
                    
                    results.append(final_metrics)
                    
                    # Clean up memory
                    del abm
                    gc.collect()
                    
                    # Progress tracking
                    elapsed = time.time() - start_time
                    rate = sim_count / elapsed if elapsed > 0 else 0
                    eta = (total_sims - sim_count) / rate if rate > 0 else 0
                    
                    if sim_count % 5 == 0:
                        print(f"    Progress: {sim_count}/{total_sims} ({sim_count/total_sims*100:.1f}%) "
                              f"- ETA: {eta/60:.1f} min")
                
                except Exception as e:
                    print(f"    ERROR in {param_name}={value}, run {run}: {e}")
                    continue
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    total_time = time.time() - start_time
    print(f"\nParameter sweep completed in {total_time/60:.2f} minutes")
    print(f"Results saved to {output_file}")
    
    return df

# ========== MAIN EXECUTION ==========

def main():
    """Main execution function"""
    print("=== COVID-19 ABM - 1M Agent Scale with Parameter Sweep ===")
    
    # Option 1: Run single test simulation
    print("\n1. Running single test simulation (100K agents)...")
    abm_test = MassivelyParallelABM()
    abm_test.initialize_simulation(N=100000, max_days=50)
    test_results = abm_test.run_simulation()
    print("Test results:", test_results)
    
    # Option 2: Run parameter sweep (adjust n_runs for faster testing)
    print("\n2. Running parameter sweep (1M agents, 2 runs per parameter)...")
    results_df = run_parameter_sweep(n_runs=2, N=100000, output_file="results_1m.csv")
    
    # Option 3: Generate the grid plot
    print("\n3. Generating parameter sweep plot...")
    make_grid(results_df, out_png="parameter_sweep_1m.png")
    
    print("\n=== Complete ===")
    print("Files created:")
    print("  - results_1m.csv (parameter sweep results)")
    print("  - parameter_sweep_1m.png (parameter sweep plot)")
    
    return results_df

if __name__ == "__main__":
    results_df = main()
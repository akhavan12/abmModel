"""
COVID-19 ABM - GPU Accelerated with Long COVID + Parameter Sweep + Plotting
Complete unified implementation matching NetLogo model
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jax import random
import time
import gc

print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")


class FixedGPUABM:
    """GPU ABM with corrected Long COVID implementation"""
    
    def __init__(self):
        self.key = random.PRNGKey(42)
        self.config = self._get_netlogo_default_config()
        
        # Agent state arrays
        self.infected = None
        self.immuned = None
        self.symptomatic = None
        self.super_immune = None
        self.persistent_long_covid = None
        self.long_covid_severity = None
        self.long_covid_duration = None
        self.long_covid_recovery_group = None
        self.long_covid_weibull_k = None
        self.long_covid_weibull_lambda = None
        self.lc_pending = None
        self.lc_onset_day = None
        self.virus_check_timer = None
        self.number_of_infection = None
        self.infection_start_tick = None
        self.infectious_start = None
        self.infectious_end = None
        self.transfer_active_duration = None
        self.symptomatic_start = None
        self.symptomatic_duration = None
        self.age = None
        self.gender = None
        self.health_risk_level = None
        self.covid_age_prob = None
        self.us_age_prob = None
        self.vaccinated = None
        self.vaccinated_time = None
        
        self.neighbors = None
        self.N = 0
        
    def _get_netlogo_default_config(self):
        """NetLogo defaults with Long COVID ENABLED"""
        return {
            'max_days': 365,
            'covid_spread_chance_pct': 10.0,
            'initial_infected_agents': 5,
            'precaution_pct': 50.0,
            'avg_degree': 5,
            'v_start_time': 180,
            'vaccination_pct': 80.0,
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
            
            # LONG COVID ENABLED
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
            'male_population_pct': 49.5,
            'age_range': 100,
            'risk_level_2_pct': 4.0,
            'risk_level_3_pct': 40.0,
            'risk_level_4_pct': 6.0,
        }
    
    def initialize_simulation(self, N=10000, seed=42, **kwargs):
        """Initialize GPU simulation"""
        self.N = N
        self.key = random.PRNGKey(seed)
        self.config.update(kwargs)
        
        self._initialize_agent_arrays()
        self._create_network_simple()
        self._setup_demographics()
        self._seed_initial_infections()
    
    def _initialize_agent_arrays(self):
        """Initialize all arrays on GPU"""
        N = self.N
        self.infected = jnp.zeros(N, dtype=jnp.bool_)
        self.immuned = jnp.zeros(N, dtype=jnp.bool_)
        self.symptomatic = jnp.zeros(N, dtype=jnp.bool_)
        self.super_immune = jnp.zeros(N, dtype=jnp.bool_)
        self.persistent_long_covid = jnp.zeros(N, dtype=jnp.bool_)
        self.long_covid_severity = jnp.zeros(N, dtype=jnp.float32)
        self.long_covid_duration = jnp.zeros(N, dtype=jnp.int32)
        self.long_covid_recovery_group = jnp.full(N, -1, dtype=jnp.int8)
        self.long_covid_weibull_k = jnp.zeros(N, dtype=jnp.float32)
        self.long_covid_weibull_lambda = jnp.zeros(N, dtype=jnp.float32)
        self.lc_pending = jnp.zeros(N, dtype=jnp.bool_)
        self.lc_onset_day = jnp.zeros(N, dtype=jnp.int32)
        self.virus_check_timer = jnp.zeros(N, dtype=jnp.int32)
        self.number_of_infection = jnp.zeros(N, dtype=jnp.int32)
        self.infection_start_tick = jnp.zeros(N, dtype=jnp.int32)
        self.infectious_start = jnp.ones(N, dtype=jnp.int32)
        self.infectious_end = jnp.ones(N, dtype=jnp.int32)
        self.transfer_active_duration = jnp.zeros(N, dtype=jnp.int32)
        self.symptomatic_start = jnp.zeros(N, dtype=jnp.int32)
        self.symptomatic_duration = jnp.zeros(N, dtype=jnp.int32)
        self.age = jnp.zeros(N, dtype=jnp.int8)
        self.gender = jnp.zeros(N, dtype=jnp.int8)
        self.health_risk_level = jnp.ones(N, dtype=jnp.int8)
        self.covid_age_prob = jnp.full(N, 15.0, dtype=jnp.float32)
        self.us_age_prob = jnp.full(N, 13.0, dtype=jnp.float32)
        self.vaccinated = jnp.zeros(N, dtype=jnp.bool_)
        self.vaccinated_time = jnp.zeros(N, dtype=jnp.int32)
    
    def _create_network_simple(self):
        """Simple network creation"""
        N = self.N
        avg_degree = self.config['avg_degree']
        
        neighbors = [[] for _ in range(N)]
        target_edges = (avg_degree * N) // 2
        rng = np.random.RandomState(42)
        
        edges = 0
        attempts = 0
        max_attempts = target_edges * 10
        
        while edges < target_edges and attempts < max_attempts:
            i = rng.randint(0, N)
            j = rng.randint(0, N)
            
            if i != j and j not in neighbors[i] and len(neighbors[i]) < 50:
                neighbors[i].append(j)
                neighbors[j].append(i)
                edges += 1
            attempts += 1
        
        max_neighbors = max(len(n) for n in neighbors)
        self.neighbors = jnp.full((N, max_neighbors), -1, dtype=jnp.int32)
        
        for i in range(N):
            if neighbors[i]:
                self.neighbors = self.neighbors.at[i, :len(neighbors[i])].set(jnp.array(neighbors[i]))
    
    def _setup_demographics(self):
        """Setup demographics"""
        N = self.N
        key = self.key
        
        # Age distribution
        key, subkey = random.split(key)
        self.age = random.randint(subkey, (N,), 0, self.config['age_range']).astype(jnp.int8)
        
        # Gender
        key, subkey = random.split(key)
        male_prob = self.config['male_population_pct'] / 100.0
        self.gender = random.bernoulli(subkey, male_prob, (N,)).astype(jnp.int8)
        
        # Age probabilities
        self._set_age_probabilities_vectorized()
        
        # Super-immune
        key, subkey = random.split(key)
        n_super = int(self.config['super_immune_pct'] * N / 100)
        super_indices = random.choice(subkey, N, shape=(n_super,), replace=False)
        super_mask = jnp.zeros(N, dtype=jnp.bool_)
        self.super_immune = super_mask.at[super_indices].set(True)
        
        self.key = key
    
    def _set_age_probabilities_vectorized(self):
        """Set age probabilities"""
        age = self.age
        
        covid_probs = jnp.select([
            age < 10, age < 20, age < 30, age < 40, 
            age < 50, age < 60, age < 70, age < 80
        ], [
            2.3, 5.1, 15.5, 16.9, 16.4, 16.4, 11.9, 7.0
        ], default=8.5)
        
        us_probs = jnp.select([
            age < 5, age < 15, age < 25, age < 35, age < 45,
            age < 55, age < 65, age < 75, age < 85
        ], [
            5.7, 12.5, 13.0, 13.7, 13.1, 12.3, 12.9, 10.1, 4.9
        ], default=1.8)
        
        self.covid_age_prob = covid_probs.astype(jnp.float32)
        self.us_age_prob = us_probs.astype(jnp.float32)
    
    def _seed_initial_infections(self):
        """Seed initial infections"""
        N = self.N
        key = self.key
        
        n_initial = min(self.config['initial_infected_agents'], N)
        eligible_mask = ~self.super_immune
        eligible_indices = jnp.where(eligible_mask)[0]
        n_initial = min(n_initial, len(eligible_indices))
        
        key, subkey = random.split(key)
        infected_indices = random.choice(subkey, eligible_indices, shape=(n_initial,), replace=False)
        
        for idx in infected_indices:
            idx = int(idx)
            key, subkey = random.split(key)
            self._infect_agent_with_symptoms(idx, 0, subkey)
        
        self.key = key
    
    def _infect_agent_with_symptoms(self, agent_idx, day, key):
        """Properly set up infection with symptom timing"""
        self.infected = self.infected.at[agent_idx].set(True)
        self.immuned = self.immuned.at[agent_idx].set(False)
        self.infection_start_tick = self.infection_start_tick.at[agent_idx].set(day)
        self.virus_check_timer = self.virus_check_timer.at[agent_idx].set(0)
        self.number_of_infection = self.number_of_infection.at[agent_idx].add(1)
        
        # Set contagious period
        key, subkey = random.split(key)
        drawn_length = 1 + random.randint(subkey, (1,), 0, self.config['active_duration'])[0]
        max_length = max(1, self.config['infected_period'] - 1)
        transfer_duration = min(drawn_length, max_length)
        
        self.transfer_active_duration = self.transfer_active_duration.at[agent_idx].set(transfer_duration)
        self.infectious_start = self.infectious_start.at[agent_idx].set(1)
        self.infectious_end = self.infectious_end.at[agent_idx].set(1 + transfer_duration)
        
        # Decide if asymptomatic
        key, subkey = random.split(key)
        is_asymptomatic = random.uniform(subkey) * 100 < self.config['asymptomatic_pct']
        
        if is_asymptomatic:
            self.symptomatic_start = self.symptomatic_start.at[agent_idx].set(0)
            self.symptomatic_duration = self.symptomatic_duration.at[agent_idx].set(0)
        else:
            # Has symptoms
            key, subkey = random.split(key)
            incubation = 1 + random.randint(subkey, (1,), 0, self.config['incubation_period'])[0]
            incubation = min(incubation, transfer_duration)
            
            self.symptomatic_start = self.symptomatic_start.at[agent_idx].set(incubation)
            
            # Calculate symptom duration
            key, subkey = random.split(key)
            base_duration = random.normal(subkey) * self.config['symptomatic_duration_dev'] + \
                          self.config['symptomatic_duration_mid']
            
            base_duration = jnp.clip(
                base_duration,
                self.config['symptomatic_duration_min'],
                self.config['symptomatic_duration_max']
            )
            
            reinfection_add = self.config['effect_of_reinfection'] * int(self.number_of_infection[agent_idx])
            symptom_duration = int(base_duration + reinfection_add)
            
            # If already has LC, make symptoms 50% longer and worsen LC
            if self.persistent_long_covid[agent_idx]:
                symptom_duration = int(symptom_duration * 1.5)
                self.long_covid_severity = self.long_covid_severity.at[agent_idx].add(10)
                self.long_covid_severity = jnp.clip(self.long_covid_severity, 5, 90)
                
                # Group worsening
                current_group = int(self.long_covid_recovery_group[agent_idx])
                key, subkey = random.split(key)
                worsen_roll = random.uniform(subkey) * 100
                
                if current_group == 0 and worsen_roll < 30:
                    self.long_covid_recovery_group = self.long_covid_recovery_group.at[agent_idx].set(1)
                    self.long_covid_weibull_k = self.long_covid_weibull_k.at[agent_idx].set(1.2)
                    self.long_covid_weibull_lambda = self.long_covid_weibull_lambda.at[agent_idx].set(450.0)
                elif current_group == 1 and worsen_roll < 20:
                    self.long_covid_recovery_group = self.long_covid_recovery_group.at[agent_idx].set(2)
                    self.long_covid_weibull_k = self.long_covid_weibull_k.at[agent_idx].set(0.5)
                    self.long_covid_weibull_lambda = self.long_covid_weibull_lambda.at[agent_idx].set(1200.0)
            
            self.symptomatic_duration = self.symptomatic_duration.at[agent_idx].set(symptom_duration)
        
        if not self.persistent_long_covid[agent_idx]:
            self.long_covid_recovery_group = self.long_covid_recovery_group.at[agent_idx].set(-1)
            self.long_covid_weibull_k = self.long_covid_weibull_k.at[agent_idx].set(0.0)
            self.long_covid_weibull_lambda = self.long_covid_weibull_lambda.at[agent_idx].set(0.0)
    
    def run_simulation(self, verbose=True):
        """Run GPU simulation with LC tracking"""
        if verbose:
            print(f"\n🚀 Starting simulation: {self.N:,} agents, {self.config['max_days']} days")
        
        start_time = time.time()
        key = self.key
        total_reinfected = 0
        min_productivity = 100.0
        
        for day in range(self.config['max_days']):
            if verbose and day % 30 == 0:
                n_infected = int(jnp.sum(self.infected))
                n_immune = int(jnp.sum(self.immuned))
                n_lc = int(jnp.sum(self.persistent_long_covid))
                productivity = self._calculate_productivity()
                print(f"Day {day:3d}: Inf={n_infected:4d}, Imm={n_immune:4d}, "
                      f"LC={n_lc:4d}, Prod={productivity:.1f}%")
            
            if day == self.config['v_start_time']:
                self._vaccination_status()
            
            if self.config['long_covid']:
                self._do_long_covid_checks(day, key)
                key, = random.split(key, 1)
            
            if self.config['long_covid']:
                self._process_pending_lc(day)
            
            self._update_infected_agents(day, key)
            key, = random.split(key, 1)
            
            daily_reinfections = self._transmission_step(day, key)
            total_reinfected += daily_reinfections
            key, = random.split(key, 1)
            
            self._update_immune_agents(day)
            self._update_vaccination_time(key)
            key, = random.split(key, 1)
            
            productivity = self._calculate_productivity()
            min_productivity = min(min_productivity, productivity)
            
            if not jnp.any(self.infected) and not jnp.any(self.immuned):
                if verbose:
                    print(f"✓ Epidemic ended at day {day}")
                break
        
        total_time = time.time() - start_time
        n_infected_ever = int(jnp.sum(self.number_of_infection > 0))
        n_lc_total = int(jnp.sum(self.persistent_long_covid))
        
        if verbose:
            print(f"✓ Complete: {total_time:.1f}s, {n_infected_ever:,} infected, {n_lc_total:,} LC")
        
        return {
            'runtime_days': day + 1,
            'infected': n_infected_ever,
            'reinfected': total_reinfected,
            'long_covid_cases': n_lc_total,
            'min_productivity': min_productivity,
        }
    
    def _update_infected_agents(self, day, key):
        """Update infected agents AND check for LC onset"""
        infected_mask = self.infected
        infected_indices = jnp.where(infected_mask)[0]
        
        self.virus_check_timer = jnp.where(infected_mask, 
                                          self.virus_check_timer + 1, 
                                          self.virus_check_timer)
        
        for idx in infected_indices:
            idx = int(idx)
            timer = int(self.virus_check_timer[idx])
            symp_start = int(self.symptomatic_start[idx])
            symp_dur = int(self.symptomatic_duration[idx])
            
            if symp_start > 0 and timer >= symp_start and timer < symp_start + symp_dur:
                self.symptomatic = self.symptomatic.at[idx].set(True)
            else:
                self.symptomatic = self.symptomatic.at[idx].set(False)
            
            if self.config['long_covid'] and not self.persistent_long_covid[idx]:
                # Path A: ASYMPTOMATIC
                if timer >= self.config['infected_period'] and symp_start == 0:
                    key, subkey = random.split(key)
                    p_onset = self._calculate_lc_onset_prob(idx, is_asymptomatic=True)
                    if random.uniform(subkey) * 100 < p_onset:
                        onset_day = int(self.infection_start_tick[idx]) + self.config['long_covid_time_threshold']
                        self.lc_pending = self.lc_pending.at[idx].set(True)
                        self.lc_onset_day = self.lc_onset_day.at[idx].set(onset_day)
                
                # Path B: SYMPTOMATIC > 30 days
                if symp_start > 0 and symp_dur > self.config['long_covid_time_threshold']:
                    if timer == symp_start + self.config['long_covid_time_threshold']:
                        self._assign_long_covid_group(idx, key)
                        key, = random.split(key, 1)
                
                # Path C: SYMPTOMATIC ≤ 30 days
                if symp_start > 0 and symp_dur <= self.config['long_covid_time_threshold']:
                    if timer == symp_start + symp_dur:
                        key, subkey = random.split(key)
                        p_onset = self._calculate_lc_onset_prob(idx, is_asymptomatic=False)
                        if random.uniform(subkey) * 100 < p_onset:
                            onset_day = int(self.infection_start_tick[idx]) + self.config['long_covid_time_threshold']
                            self.lc_pending = self.lc_pending.at[idx].set(True)
                            self.lc_onset_day = self.lc_onset_day.at[idx].set(onset_day)
        
        become_immune = infected_mask & (self.virus_check_timer >= self.config['infected_period'])
        self.infected = self.infected & ~become_immune
        self.immuned = self.immuned | become_immune
        self.virus_check_timer = jnp.where(become_immune, 0, self.virus_check_timer)
    
    def _calculate_lc_onset_prob(self, agent_idx, is_asymptomatic):
        """Calculate LC onset probability with all multipliers"""
        base_prob = self.config['lc_onset_base_pct']
        multiplier = 1.0
        
        age = int(self.age[agent_idx])
        if age < 30:
            multiplier *= 0.9
        elif 50 <= age <= 64:
            multiplier *= 1.2
        elif age >= 65:
            multiplier *= 1.3
        
        if int(self.gender[agent_idx]) == 1:
            multiplier *= self.config['lc_incidence_mult_female']
        
        if self.vaccinated[agent_idx]:
            multiplier *= 0.7
        
        n_infections = int(self.number_of_infection[agent_idx])
        has_lc = int(self.long_covid_recovery_group[agent_idx]) >= 0
        if n_infections > 1 and not has_lc:
            multiplier *= self.config['reinfection_new_onset_mult']
        
        if is_asymptomatic:
            multiplier *= self.config['asymptomatic_lc_mult']
        
        return jnp.clip(base_prob * multiplier, 0, 100)
    
    def _assign_long_covid_group(self, agent_idx, key):
        """Assign LC recovery group and parameters"""
        w_fast = self.config['lc_base_fast_prob']
        w_pers = self.config['lc_base_persistent_prob']
        w_sum = w_fast + w_pers
        
        if w_sum > 100:
            w_fast = 100 * w_fast / w_sum
            w_pers = 100 * w_pers / w_sum
            w_sum = 100
        w_grad = 100 - w_sum
        
        age = int(self.age[agent_idx])
        if age >= 65 and w_grad >= 2:
            w_pers += 2
            w_grad -= 2
        
        symp_dur = int(self.symptomatic_duration[agent_idx])
        if symp_dur > 21 and w_grad >= 4:
            w_pers += 4
            w_grad -= 4
        
        total = w_fast + w_pers + w_grad
        if total <= 0:
            w_grad = 100
            total = 100
        
        key, subkey = random.split(key)
        r = random.uniform(subkey) * total
        
        if r < w_fast:
            group, k, lam = 0, 1.5, 60.0
            severity = random.normal(key) * 15 + 30
        elif r < w_fast + w_pers:
            group, k, lam = 2, 0.5, 1200.0
            severity = random.normal(key) * 20 + 70
        else:
            group, k, lam = 1, 1.2, 450.0
            severity = random.normal(key) * 20 + 50
        
        severity = jnp.clip(severity, 5, 100)
        
        self.persistent_long_covid = self.persistent_long_covid.at[agent_idx].set(True)
        self.long_covid_duration = self.long_covid_duration.at[agent_idx].set(0)
        self.long_covid_recovery_group = self.long_covid_recovery_group.at[agent_idx].set(group)
        self.long_covid_weibull_k = self.long_covid_weibull_k.at[agent_idx].set(k)
        self.long_covid_weibull_lambda = self.long_covid_weibull_lambda.at[agent_idx].set(lam)
        self.long_covid_severity = self.long_covid_severity.at[agent_idx].set(severity)
    
    def _process_pending_lc(self, day):
        """Activate pending LC cases"""
        pending_mask = self.lc_pending & (day >= self.lc_onset_day)
        pending_indices = jnp.where(pending_mask)[0]
        
        key = self.key
        for idx in pending_indices:
            idx = int(idx)
            self.lc_pending = self.lc_pending.at[idx].set(False)
            
            if not self.persistent_long_covid[idx]:
                key, subkey = random.split(key)
                self._assign_long_covid_group(idx, subkey)
        
        self.key = key
    
    def _do_long_covid_checks(self, day, key):
        """LC recovery with Weibull hazard"""
        lc_mask = self.persistent_long_covid
        lc_indices = jnp.where(lc_mask)[0]
        
        self.long_covid_duration = jnp.where(lc_mask, 
                                             self.long_covid_duration + 1, 
                                             self.long_covid_duration)
        
        for idx in lc_indices:
            idx = int(idx)
            duration = int(self.long_covid_duration[idx])
            k = float(self.long_covid_weibull_k[idx])
            lam = float(self.long_covid_weibull_lambda[idx])
            group = int(self.long_covid_recovery_group[idx])
            
            if duration > 0 and k > 0 and lam > 0:
                t_scaled = duration / lam
                hazard = (k / lam) * (t_scaled ** (k - 1))
                daily_prob = (1 - jnp.exp(-hazard)) * 100
                
                daily_prob = jnp.clip(daily_prob, 0.01, 10.0)
                
                if group == 0:
                    daily_prob *= 2.0
                elif group == 2:
                    daily_prob *= 0.3
                    if duration > 1095:
                        daily_prob *= 0.1
                
                daily_prob = jnp.clip(daily_prob, 0, 15)
                
                key, subkey = random.split(key)
                if random.uniform(subkey) * 100 < daily_prob:
                    self.persistent_long_covid = self.persistent_long_covid.at[idx].set(False)
                    self.long_covid_severity = self.long_covid_severity.at[idx].set(0.0)
                    self.long_covid_duration = self.long_covid_duration.at[idx].set(0)
                    self.long_covid_recovery_group = self.long_covid_recovery_group.at[idx].set(-1)
                    self.long_covid_weibull_k = self.long_covid_weibull_k.at[idx].set(0.0)
                    self.long_covid_weibull_lambda = self.long_covid_weibull_lambda.at[idx].set(0.0)
                else:
                    if group == 1 and duration > 30:
                        new_severity = float(self.long_covid_severity[idx]) - 0.05
                        new_severity = jnp.clip(new_severity, 5, 100)
                        self.long_covid_severity = self.long_covid_severity.at[idx].set(new_severity)
    
    def _update_immune_agents(self, day):
        """Update immune agents"""
        immune_mask = self.immuned
        
        self.virus_check_timer = jnp.where(immune_mask, 
                                          self.virus_check_timer + 1, 
                                          self.virus_check_timer)
        
        immunity_end = self.config['infected_period'] + self.config['immune_period']
        lose_immunity = immune_mask & (self.virus_check_timer >= immunity_end)
        
        self.immuned = self.immuned & ~lose_immunity
        self.virus_check_timer = jnp.where(lose_immunity, 0, self.virus_check_timer)
    
    def _transmission_step(self, day, key):
        """Transmission with precaution behavior"""
        daily_reinfections = 0
        
        infectious_mask = (self.infected & 
                          (self.virus_check_timer >= self.infectious_start) & 
                          (self.virus_check_timer < self.infectious_end))
        
        infectious_indices = jnp.where(infectious_mask)[0]
        
        for source_idx in infectious_indices:
            source_idx = int(source_idx)
            
            if self.symptomatic[source_idx]:
                symp_start = int(self.symptomatic_start[source_idx])
                timer = int(self.virus_check_timer[source_idx])
                
                if symp_start > 0 and timer > symp_start:
                    key, subkey = random.split(key)
                    if random.uniform(subkey) * 100 < self.config['precaution_pct']:
                        continue
            
            source_neighbors = self.neighbors[source_idx]
            valid_neighbors = source_neighbors[source_neighbors != -1]
            
            for neighbor_idx in valid_neighbors:
                neighbor_idx = int(neighbor_idx)
                
                if neighbor_idx >= self.N:
                    continue
                
                if (self.infected[neighbor_idx] or 
                    self.immuned[neighbor_idx] or 
                    self.super_immune[neighbor_idx]):
                    continue
                
                if self.vaccinated[neighbor_idx]:
                    if self.config['vaccination_decay']:
                        eff = max(0, self.config['efficiency_pct'] - 0.11 * float(self.vaccinated_time[neighbor_idx]))
                    else:
                        eff = self.config['efficiency_pct']
                    
                    key, subkey = random.split(key)
                    if random.uniform(subkey) * 100 < eff:
                        continue
                
                infection_prob = self.config['covid_spread_chance_pct']
                
                covid_age = float(self.covid_age_prob[neighbor_idx])
                us_age = float(self.us_age_prob[neighbor_idx])
                infection_prob *= covid_age / (us_age + 1e-9)
                
                infection_prob = jnp.clip(infection_prob, 0, 100)
                
                key, subkey = random.split(key)
                if random.uniform(subkey) * 100 < infection_prob:
                    if self.number_of_infection[neighbor_idx] > 0:
                        daily_reinfections += 1
                    
                    key, subkey = random.split(key)
                    self._infect_agent_with_symptoms(neighbor_idx, day, subkey)
        
        return daily_reinfections
    
    def _vaccination_status(self):
        """Simple vaccination"""
        current_vaccinated = jnp.sum(self.vaccinated)
        target_vaccinated = int(self.N * self.config['vaccination_pct'] / 100)
        
        if current_vaccinated >= target_vaccinated:
            return
        
        unvaccinated_mask = ~self.vaccinated
        unvaccinated_indices = jnp.where(unvaccinated_mask)[0]
        
        n_to_vaccinate = min(target_vaccinated - int(current_vaccinated), len(unvaccinated_indices))
        
        if n_to_vaccinate > 0:
            key, subkey = random.split(self.key)
            vaccinate_indices = random.choice(subkey, unvaccinated_indices, 
                                            shape=(n_to_vaccinate,), replace=False)
            
            self.vaccinated = self.vaccinated.at[vaccinate_indices].set(True)
            self.vaccinated_time = self.vaccinated_time.at[vaccinate_indices].set(1)
            self.key = subkey
    
    def _update_vaccination_time(self, key):
        """Update vaccination time and boosters"""
        vaccinated_mask = self.vaccinated
        self.vaccinated_time = jnp.where(vaccinated_mask, 
                                        self.vaccinated_time + 1, 
                                        self.vaccinated_time)
        
        need_booster = vaccinated_mask & (self.vaccinated_time >= 180)
        booster_indices = jnp.where(need_booster)[0]
        
        if len(booster_indices) > 0:
            key, subkey = random.split(key)
            booster_probs = random.uniform(subkey, (len(booster_indices),))
            get_booster = booster_probs * 100 < self.config['boosted_pct']
            
            for i, idx in enumerate(booster_indices):
                idx = int(idx)
                if get_booster[i]:
                    self.vaccinated_time = self.vaccinated_time.at[idx].set(1)
                else:
                    self.vaccinated = self.vaccinated.at[idx].set(False)
                    self.vaccinated_time = self.vaccinated_time.at[idx].set(0)
    
    def _calculate_productivity(self):
        """Calculate current productivity"""
        symptomatic_loss = float(jnp.sum(self.symptomatic))
        
        lc_loss = float(jnp.sum(jnp.where(
            self.persistent_long_covid & ~self.symptomatic,
            self.long_covid_severity / 100.0,
            0.0
        )))
        
        total_loss = symptomatic_loss + lc_loss
        productivity = (1 - total_loss / self.N) * 100
        return float(productivity)


# ========== PARAMETER SWEEP ==========

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


def run_gpu_sweep(n_runs=10, N=100000, output_file="gpu_sweep_results.csv"):
    """Run parameter sweep on GPU"""
    results = []
    total_sims = sum(len(values) * n_runs for values in ORDER.values())
    
    print(f"\n{'='*70}")
    print(f"GPU PARAMETER SWEEP")
    print(f"{'='*70}")
    print(f"Agents per run:      {N:,}")
    print(f"Runs per parameter:  {n_runs}")
    print(f"Total simulations:   {total_sims}")
    print(f"Backend:             {jax.default_backend().upper()}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    sim_count = 0
    
    for param_name, values in ORDER.items():
        print(f"\n🔧 Sweeping: {param_name}")
        
        for value in values:
            print(f"  → Value: {value}")
            
            for run in range(n_runs):
                sim_count += 1
                
                try:
                    abm = FixedGPUABM()
                    abm.initialize_simulation(
                        N=N,
                        seed=42 + run,
                        **{param_name: value}
                    )
                    
                    final_metrics = abm.run_simulation(verbose=False)
                    final_metrics['param_name'] = param_name
                    final_metrics['param_value'] = value
                    final_metrics['run'] = run
                    final_metrics['agents'] = N
                    final_metrics['backend'] = jax.default_backend()
                    
                    results.append(final_metrics)
                    
                    if (run + 1) % 5 == 0:
                        elapsed = time.time() - start_time
                        rate = sim_count / elapsed
                        eta = (total_sims - sim_count) / rate / 60 if rate > 0 else 0
                        print(f"    ✓ Run {run+1}/{n_runs} complete | ETA: {eta:.1f} min")
                    
                    del abm
                    gc.collect()
                    
                except Exception as e:
                    print(f"    ✗ Error in run {run}: {e}")
                    continue
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"Total time:   {total_time/3600:.2f} hours")
    print(f"Avg per sim:  {total_time/len(results):.1f} seconds")
    print(f"Results:      {output_file}")
    print(f"{'='*70}\n")
    
    return df


def make_grid(df, out_png="gpu_results.png"):
    """Generate parameter sweep grid plot"""
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
    print(f"✓ Saved plot: {out_png}")
    return out_png


# ========== MAIN EXECUTION ==========

def quick_demo():
    """Quick demonstration"""
    print("\n" + "="*70)
    print("QUICK DEMO - 10K agents, 180 days")
    print("="*70)
    
    abm = FixedGPUABM()
    abm.initialize_simulation(N=10000, seed=42, max_days=180)
    results = abm.run_simulation(verbose=True)
    
    print("\n📊 Final Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:.<25} {value:.2f}")
        else:
            print(f"  {key:.<25} {value:,}")
    
    return results


def full_sweep_and_plot(n_runs=10, N=100000):
    """Run full parameter sweep and generate plots"""
    print("\n" + "="*70)
    print("FULL PARAMETER SWEEP + PLOTTING")
    print("="*70)
    
    # Run sweep
    df = run_gpu_sweep(n_runs=n_runs, N=N, output_file="gpu_sweep_results.csv")
    
    # Generate plot
    print("\n📊 Generating plot...")
    make_grid(df, out_png="gpu_parameter_sweep.png")
    
    # Summary statistics
    print("\n📈 Summary Statistics:")
    print("-" * 70)
    for param_name in ORDER.keys():
        param_df = df[df["param_name"] == param_name]
        print(f"\n{param_name}:")
        for value in ORDER[param_name]:
            subset = param_df[param_df["param_value"] == value]
            if len(subset) > 0:
                mean_inf = subset["infected"].mean()
                mean_lc = subset["long_covid_cases"].mean()
                mean_prod = subset["min_productivity"].mean()
                print(f"  {value:>4} → Infected: {mean_inf:>7.0f}, LC: {mean_lc:>6.0f}, "
                      f"Min Prod: {mean_prod:>5.1f}%")
    
    return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Quick demo
        quick_demo()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "sweep":
        # Full sweep
        n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
        full_sweep_and_plot(n_runs=n_runs, N=N)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "plot":
        # Just plot existing results
        csv_file = sys.argv[2] if len(sys.argv) > 2 else "gpu_sweep_results.csv"
        print(f"\n📊 Loading results from {csv_file}...")
        df = pd.read_csv(csv_file)
        make_grid(df, out_png="gpu_parameter_sweep.png")
    
    else:
        # Default: quick demo
        print("\nUsage:")
        print("  python script.py demo              # Quick 10K demo")
        print("  python script.py sweep [runs] [N]  # Full sweep (default: 10 runs, 100K agents)")
        print("  python script.py plot [csv_file]   # Plot existing results")
        print("\nRunning quick demo...\n")
        quick_demo()

from __future__ import annotations
import inspect
from gpu_based_fixed import simulate_netlogoish as _simulate

_SIG = inspect.signature(_simulate)
_VALID_KEYS = set(_SIG.parameters.keys())

def simulate_gpu(**params):
    filtered = {k: v for k, v in params.items() if k in _VALID_KEYS}
    return _simulate(**filtered)


# """
# GPU-accelerated COVID ABM using JAX - FIXED VERSION
# Supports 100,000+ agents with proper LC, reinfection, and productivity tracking

# Install: pip install jax jaxlib scipy numpy pandas
# For GPU: pip install jax[cuda12]
# """

# import jax
# import jax.numpy as jnp
# from jax import random, jit
# import numpy as np
# from functools import partial
# from scipy import sparse

# # Enable 64-bit precision
# jax.config.update("jax_enable_x64", True)

# @jit
# def get_age_weights(ages):
#     """Get COVID and US age probability weights"""
#     covid_weights = jnp.select(
#         [ages < 10, ages < 20, ages < 30, ages < 40, ages < 50, 
#          ages < 60, ages < 70, ages < 80],
#         [2.3, 5.1, 15.5, 16.9, 16.4, 16.4, 11.9, 7.0],
#         default=8.5
#     )
    
#     us_weights = jnp.select(
#         [ages < 5, ages < 15, ages < 25, ages < 35, ages < 45,
#          ages < 55, ages < 65, ages < 75, ages < 85],
#         [5.7, 12.5, 13.0, 13.7, 13.1, 12.3, 12.9, 10.1, 4.9],
#         default=1.8
#     )
    
#     return covid_weights, us_weights

# def create_network_cpu(N, avg_degree, seed):
#     """Create sparse adjacency matrix on CPU"""
#     rng = np.random.default_rng(seed)
#     p = min(1.0, avg_degree / max(1, N - 1))
    
#     row_indices = []
#     col_indices = []
    
#     for i in range(N):
#         n_edges = rng.binomial(N - i - 1, p)
#         if n_edges > 0:
#             targets = rng.choice(N - i - 1, size=n_edges, replace=False) + i + 1
#             row_indices.extend([i] * n_edges)
#             col_indices.extend(targets)
#             row_indices.extend(targets)
#             col_indices.extend([i] * n_edges)
    
#     data = np.ones(len(row_indices), dtype=np.int32)
#     adj_matrix = sparse.csr_matrix(
#         (data, (row_indices, col_indices)), 
#         shape=(N, N), 
#         dtype=np.int32
#     )
    
#     return adj_matrix

# def simulate_gpu(
#     N=100000,
#     max_days=365,
#     covid_spread_chance_pct=10.0,
#     initial_infected_agents=10,
#     precaution_pct=50.0,
#     avg_degree=5,
#     v_start_time=180,
#     vaccination_pct=80.0,
#     infected_period=10,
#     active_duration=7,
#     immune_period=21,
#     asymptomatic_pct=40.0,
#     symptomatic_duration_mid=7,
#     symptomatic_duration_dev=3,
#     # Long COVID
#     long_covid=True,
#     lc_onset_base_pct=15.0,
#     long_covid_time_threshold=28,
#     lc_base_fast_prob=40.0,
#     lc_base_persistent_prob=20.0,
#     lc_incidence_mult_female=1.2,
#     reinfection_new_onset_mult=0.7,
#     asymptomatic_lc_mult=0.5,
#     # Vaccination
#     efficiency_pct=80.0,
#     vaccination_decay=True,
#     vaccine_priority=True,
#     # Demographics
#     age_distribution=True,
#     age_range=100,
#     age_infection_scaling=True,
#     gender=True,
#     seed=None,
# ):
#     """GPU-accelerated COVID ABM simulation - FIXED"""
    
#     print(f"Running simulation with {N:,} agents on {jax.devices()[0].device_kind}...")
    
#     if seed is None:
#         seed = np.random.randint(0, 2**31)
    
#     rng_np = np.random.default_rng(seed)
#     key = random.PRNGKey(seed)
    
#     # Initialize arrays
#     infected = np.zeros(N, dtype=bool)
#     immuned = np.zeros(N, dtype=bool)
#     symptomatic = np.zeros(N, dtype=bool)
    
#     # LC tracking
#     persistent_lc = np.zeros(N, dtype=bool)
#     lc_severity = np.zeros(N, dtype=np.float32)
#     lc_duration = np.zeros(N, dtype=np.int32)
#     lc_group = np.full(N, -1, dtype=np.int32)
#     lc_weibull_k = np.zeros(N, dtype=np.float32)
#     lc_weibull_lambda = np.zeros(N, dtype=np.float32)
#     lc_pending = np.zeros(N, dtype=bool)
#     lc_onset_day = np.zeros(N, dtype=np.int32)
    
#     # Infection tracking
#     virus_timer = np.zeros(N, dtype=np.int32)
#     num_infections = np.zeros(N, dtype=np.int32)
#     infection_start_tick = np.zeros(N, dtype=np.int32)
#     infectious_start = np.ones(N, dtype=np.int32)
#     infectious_end = np.full(N, active_duration + 1, dtype=np.int32)
#     symptomatic_start = np.zeros(N, dtype=np.int32)
#     symptomatic_duration = np.zeros(N, dtype=np.int32)
#     is_asymptomatic = np.zeros(N, dtype=bool)
    
#     # Demographics
#     age_probs = np.array([5.7, 12.5, 13.0, 13.7, 13.1, 12.3, 12.9, 10.1, 4.9, 1.8])
#     age_probs = age_probs / age_probs.sum()
    
#     if age_distribution:
#         bin_idx = rng_np.choice(10, size=N, p=age_probs)
#         bin_starts = np.array([0, 5, 15, 25, 35, 45, 55, 65, 75, 85])
#         bin_widths = np.array([5, 10, 10, 10, 10, 10, 10, 10, 10, 10])
#         ages = bin_starts[bin_idx] + rng_np.integers(0, bin_widths[bin_idx], size=N)
#     else:
#         ages = rng_np.integers(0, age_range, size=N)
    
#     genders = rng_np.binomial(1, 0.51, size=N).astype(np.int32)
    
#     # Age weights
#     covid_age_prob = np.select(
#         [ages < 10, ages < 20, ages < 30, ages < 40, ages < 50, 
#          ages < 60, ages < 70, ages < 80],
#         [2.3, 5.1, 15.5, 16.9, 16.4, 16.4, 11.9, 7.0],
#         default=8.5
#     )
    
#     us_age_prob = np.select(
#         [ages < 5, ages < 15, ages < 25, ages < 35, ages < 45,
#          ages < 55, ages < 65, ages < 75, ages < 85],
#         [5.7, 12.5, 13.0, 13.7, 13.1, 12.3, 12.9, 10.1, 4.9],
#         default=1.8
#     )
    
#     # Vaccination
#     vaccinated = np.zeros(N, dtype=bool)
#     vacc_time = np.zeros(N, dtype=np.int32)
    
#     # Create network
#     print("Creating network...")
#     adj_sparse = create_network_cpu(N, avg_degree, seed)
    
#     # Seed initial infections
#     initial_idx = rng_np.choice(N, size=min(N, initial_infected_agents), replace=False)
#     infected[initial_idx] = True
#     num_infections[initial_idx] = 1
#     infection_start_tick[initial_idx] = 0
    
#     # Set up initial infection parameters
#     for idx in initial_idx:
#         # Asymptomatic?
#         if rng_np.random() * 100 < asymptomatic_pct:
#             symptomatic_start[idx] = 0
#             symptomatic_duration[idx] = 0
#             is_asymptomatic[idx] = True
#         else:
#             symptomatic_start[idx] = 1 + rng_np.integers(0, 5)
#             symptomatic_duration[idx] = max(1, int(rng_np.normal(symptomatic_duration_mid, symptomatic_duration_dev)))
#             is_asymptomatic[idx] = False
        
#         infectious_start[idx] = 1
#         infectious_end[idx] = min(infected_period, infectious_start[idx] + 1 + rng_np.integers(0, active_duration))
    
#     # Metrics
#     total_infected = len(initial_idx)
#     total_reinfected = 0
#     total_lc_cases = 0
#     min_productivity = 100.0
    
#     print("Starting simulation...")
    
#     # Main loop
#     for day in range(max_days):
#         if day % 30 == 0:
#             n_inf = np.sum(infected)
#             n_lc = np.sum(persistent_lc)
#             print(f"Day {day}: {n_inf} infected, {n_lc} with LC")
        
#         # Vaccination at start time
#         if day == v_start_time and vaccination_pct > 0:
#             n_vacc = int(N * vaccination_pct / 100)
            
#             if vaccine_priority:
#                 priority_score = ages.astype(np.float32) / 100.0
#                 vacc_idx = np.argsort(-priority_score)[:n_vacc]
#             else:
#                 vacc_idx = rng_np.choice(N, size=n_vacc, replace=False)
            
#             vaccinated[vacc_idx] = True
#             vacc_time[vacc_idx] = 1
        
#         # Long COVID progression
#         if long_covid:
#             lc_mask = persistent_lc
#             if np.any(lc_mask):
#                 lc_duration[lc_mask] += 1
                
#                 # Weibull recovery
#                 for idx in np.where(lc_mask)[0]:
#                     if lc_weibull_lambda[idx] <= 0:
#                         continue
                    
#                     t_scaled = lc_duration[idx] / lc_weibull_lambda[idx]
#                     hazard = (lc_weibull_k[idx] / lc_weibull_lambda[idx]) * (t_scaled ** (lc_weibull_k[idx] - 1))
#                     recovery_chance = (1 - np.exp(-hazard)) * 100
                    
#                     # Group scaling
#                     if lc_group[idx] == 0:  # fast
#                         recovery_chance *= 2.0
#                     elif lc_group[idx] == 2:  # persistent
#                         recovery_chance *= 0.3
#                         if lc_duration[idx] > 1095:
#                             recovery_chance *= 0.1
                    
#                     recovery_chance = np.clip(recovery_chance, 0, 15)
                    
#                     if rng_np.random() * 100 < recovery_chance:
#                         persistent_lc[idx] = False
#                         lc_severity[idx] = 0
#                         lc_duration[idx] = 0
#                         lc_group[idx] = -1
            
#             # Process pending LC onsets
#             pending_mask = lc_pending & (day >= lc_onset_day)
#             if np.any(pending_mask):
#                 for idx in np.where(pending_mask)[0]:
#                     lc_pending[idx] = False
#                     if not persistent_lc[idx]:
#                         persistent_lc[idx] = True
#                         lc_duration[idx] = 0
#                         total_lc_cases += 1
                        
#                         # Assign LC group
#                         w_fast = lc_base_fast_prob
#                         w_pers = lc_base_persistent_prob
#                         w_grad = 100 - w_fast - w_pers
                        
#                         if ages[idx] >= 65:
#                             shift = min(2, w_grad)
#                             w_pers += shift
#                             w_grad -= shift
                        
#                         r = rng_np.random() * 100
#                         if r < w_fast:
#                             lc_group[idx] = 0
#                             lc_weibull_k[idx] = 1.5
#                             lc_weibull_lambda[idx] = 60
#                             lc_severity[idx] = np.clip(rng_np.normal(30, 15), 5, 100)
#                         elif r < w_fast + w_pers:
#                             lc_group[idx] = 2
#                             lc_weibull_k[idx] = 0.5
#                             lc_weibull_lambda[idx] = 1200
#                             lc_severity[idx] = np.clip(rng_np.normal(70, 20), 5, 100)
#                         else:
#                             lc_group[idx] = 1
#                             lc_weibull_k[idx] = 1.2
#                             lc_weibull_lambda[idx] = 450
#                             lc_severity[idx] = np.clip(rng_np.normal(50, 20), 5, 100)
        
#         # Transmission
#         infectious_agents = np.where(
#             infected & 
#             (virus_timer >= infectious_start) & 
#             (virus_timer < infectious_end)
#         )[0]
        
#         new_inf = []
#         for agent_id in infectious_agents:
#             # Precaution for symptomatic
#             if symptomatic[agent_id] and (rng_np.random() * 100 < precaution_pct):
#                 continue
            
#             neighbors = adj_sparse.getrow(agent_id).indices
#             susceptible_neighbors = neighbors[~infected[neighbors] & ~immuned[neighbors]]
            
#             if len(susceptible_neighbors) == 0:
#                 continue
            
#             # Vaccine protection
#             if vaccination_decay:
#                 real_eff = np.maximum(0, efficiency_pct - 0.11 * vacc_time[susceptible_neighbors])
#             else:
#                 real_eff = efficiency_pct
            
#             vacc_protected = vaccinated[susceptible_neighbors] & (rng_np.uniform(0, 100, len(susceptible_neighbors)) < real_eff)
            
#             # Infection probability
#             infection_prob = covid_spread_chance_pct
#             if age_infection_scaling:
#                 infection_prob = infection_prob * covid_age_prob[susceptible_neighbors] / (us_age_prob[susceptible_neighbors] + 1e-9)
            
#             infection_prob = np.clip(infection_prob, 0, 100)
            
#             attempts = (~vacc_protected) & (rng_np.uniform(0, 100, len(susceptible_neighbors)) < infection_prob)
#             new_inf.extend(susceptible_neighbors[attempts])
        
#         # Apply new infections
#         if len(new_inf) > 0:
#             new_inf = list(set(new_inf))
            
#             for idx in new_inf:
#                 # Track reinfections
#                 if num_infections[idx] > 0:
#                     total_reinfected += 1
                
#                 infected[idx] = True
#                 num_infections[idx] += 1
#                 virus_timer[idx] = 0
#                 infection_start_tick[idx] = day
                
#                 # Set up infection parameters
#                 if rng_np.random() * 100 < asymptomatic_pct:
#                     symptomatic_start[idx] = 0
#                     symptomatic_duration[idx] = 0
#                     is_asymptomatic[idx] = True
#                 else:
#                     symptomatic_start[idx] = 1 + rng_np.integers(0, 5)
#                     symptomatic_duration[idx] = max(1, int(rng_np.normal(symptomatic_duration_mid, symptomatic_duration_dev)))
#                     is_asymptomatic[idx] = False
                
#                 infectious_start[idx] = 1
#                 infectious_end[idx] = min(infected_period, infectious_start[idx] + 1 + rng_np.integers(0, active_duration))
            
#             total_infected += len(new_inf)
        
#         # Update symptomatic status
#         symptomatic = infected & (symptomatic_start > 0) & (virus_timer >= symptomatic_start) & (virus_timer < symptomatic_start + symptomatic_duration)
        
#         # LC onset for newly recovered
#         if long_covid:
#             # Asymptomatic path: check at end of infection
#             asym_recovering = infected & (virus_timer >= infected_period) & (symptomatic_start == 0)
#             for idx in np.where(asym_recovering)[0]:
#                 age_mult = 0.9 if ages[idx] < 30 else (1.2 if 50 <= ages[idx] <= 64 else (1.3 if ages[idx] >= 65 else 1.0))
#                 gender_mult = lc_incidence_mult_female if genders[idx] == 1 else 1.0
#                 vacc_mult = 0.7 if vaccinated[idx] else 1.0
#                 reinf_mult = reinfection_new_onset_mult if (num_infections[idx] > 1 and lc_group[idx] == -1) else 1.0
                
#                 p_onset = lc_onset_base_pct * age_mult * gender_mult * vacc_mult * reinf_mult * asymptomatic_lc_mult
#                 p_onset = np.clip(p_onset, 0, 100)
                
#                 if rng_np.random() * 100 < p_onset:
#                     lc_pending[idx] = True
#                     lc_onset_day[idx] = infection_start_tick[idx] + long_covid_time_threshold
            
#             # Symptomatic path: check at threshold or symptom end
#             symp_at_threshold = infected & (symptomatic_start > 0) & (virus_timer == symptomatic_start + long_covid_time_threshold) & (symptomatic_duration > long_covid_time_threshold)
#             for idx in np.where(symp_at_threshold)[0]:
#                 if not persistent_lc[idx]:
#                     persistent_lc[idx] = True
#                     lc_duration[idx] = 0
#                     total_lc_cases += 1
                    
#                     # Assign group (simplified)
#                     r = rng_np.random() * 100
#                     if r < lc_base_fast_prob:
#                         lc_group[idx] = 0
#                         lc_weibull_k[idx] = 1.5
#                         lc_weibull_lambda[idx] = 60
#                         lc_severity[idx] = np.clip(rng_np.normal(30, 15), 5, 100)
#                     elif r < lc_base_fast_prob + lc_base_persistent_prob:
#                         lc_group[idx] = 2
#                         lc_weibull_k[idx] = 0.5
#                         lc_weibull_lambda[idx] = 1200
#                         lc_severity[idx] = np.clip(rng_np.normal(70, 20), 5, 100)
#                     else:
#                         lc_group[idx] = 1
#                         lc_weibull_k[idx] = 1.2
#                         lc_weibull_lambda[idx] = 450
#                         lc_severity[idx] = np.clip(rng_np.normal(50, 20), 5, 100)
        
#         # Update timers
#         virus_timer[infected | immuned] += 1
        
#         # State transitions: infected -> immune
#         should_recover = infected & (virus_timer > infected_period)
#         infected[should_recover] = False
#         immuned[should_recover] = True
        
#         # immune -> susceptible
#         should_lose_immunity = immuned & (virus_timer > infected_period + immune_period)
#         immuned[should_lose_immunity] = False
#         virus_timer[should_lose_immunity] = 0
        
#         # Update vaccination time
#         vacc_time[vaccinated] += 1
        
#         # Calculate productivity
#         symptomatic_loss = np.sum(symptomatic).astype(float)
#         lc_loss = np.sum(lc_severity[persistent_lc] / 100.0)
#         total_loss = symptomatic_loss + lc_loss
#         current_prod = (1 - total_loss / N) * 100
#         min_productivity = min(min_productivity, current_prod)
        
#         # Check stopping condition
#         if np.sum(infected) == 0 and np.sum(immuned) == 0:
#             print(f"Epidemic ended at day {day}")
#             break
    
#     return {
#         'runtime_days': day + 1,
#         'infected': int(total_infected),
#         'reinfected': int(total_reinfected),
#         'long_covid_cases': int(total_lc_cases),
#         'min_productivity': float(min_productivity),
#     }


# if __name__ == "__main__":
#     print(f"JAX devices: {jax.devices()}")
#     print(f"Default backend: {jax.default_backend()}")
    
#     result = simulate_gpu(
#         N=10000,
#         max_days=180,
#         covid_spread_chance_pct=10.0,
#         initial_infected_agents=50,
#         seed=42
#     )
    
#     print("\n=== Results ===")
#     print(f"Runtime: {result['runtime_days']} days")
#     print(f"Infected: {result['infected']:,}")
#     print(f"Reinfected: {result['reinfected']:,}")
#     print(f"Long COVID: {result['long_covid_cases']:,}")
#     print(f"Min Productivity: {result['min_productivity']:.2f}%")


# # """
# # GPU-accelerated COVID ABM using JAX
# # Supports 100,000+ agents with automatic GPU utilization

# # Install: pip install jax jaxlib scipy numpy pandas
# # For GPU: pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# # """

# # import jax
# # import jax.numpy as jnp
# # from jax import random, jit
# # import numpy as np
# # from functools import partial
# # from scipy import sparse
# # import pandas as pd

# # # Enable 64-bit precision for better accuracy
# # jax.config.update("jax_enable_x64", True)

# # # Agent state constants
# # SUSCEPTIBLE = 0
# # INFECTED = 1
# # IMMUNE = 2
# # SUPER_IMMUNE = 3

# # # LC recovery groups
# # LC_FAST = 0
# # LC_GRADUAL = 1
# # LC_PERSISTENT = 2

# # @partial(jit, static_argnums=(1, 2, 3, 4))
# # def setup_age_distribution(key, N, age_distribution, age_range, gender_enabled):
# #     """Initialize agent ages and genders"""
# #     # Age bins with US-like distribution
# #     age_probs = jnp.array([5.7, 12.5, 13.0, 13.7, 13.1, 12.3, 12.9, 10.1, 4.9, 1.8])
# #     age_probs = age_probs / age_probs.sum()
    
# #     key, subkey = random.split(key)
    
# #     if age_distribution:
# #         # Sample age bins
# #         bin_idx = random.choice(subkey, 10, shape=(N,), p=age_probs)
# #         key, subkey = random.split(key)
        
# #         # Map bins to ages
# #         bin_starts = jnp.array([0, 5, 15, 25, 35, 45, 55, 65, 75, 85])
# #         bin_widths = jnp.array([5, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        
# #         ages = bin_starts[bin_idx] + random.randint(subkey, (N,), 0, bin_widths[bin_idx])
# #     else:
# #         ages = random.randint(subkey, (N,), 0, age_range)
    
# #     # Gender: 0=male, 1=female
# #     key, subkey = random.split(key)
# #     genders = random.bernoulli(subkey, 0.51, shape=(N,)).astype(jnp.int32)
    
# #     return key, ages, genders

# # def create_network_cpu(N, avg_degree, seed):
# #     """Create sparse adjacency matrix on CPU (faster for large networks)"""
# #     rng = np.random.default_rng(seed)
# #     p = min(1.0, avg_degree / max(1, N - 1))
    
# #     # Use sparse matrix for memory efficiency
# #     row_indices = []
# #     col_indices = []
    
# #     # Sample edges efficiently
# #     for i in range(N):
# #         n_edges = rng.binomial(N - i - 1, p)
# #         if n_edges > 0:
# #             targets = rng.choice(N - i - 1, size=n_edges, replace=False) + i + 1
# #             row_indices.extend([i] * n_edges)
# #             col_indices.extend(targets)
# #             # Add reverse edges for undirected graph
# #             row_indices.extend(targets)
# #             col_indices.extend([i] * n_edges)
    
# #     # Create sparse adjacency matrix
# #     data = np.ones(len(row_indices), dtype=np.int32)
# #     adj_matrix = sparse.csr_matrix(
# #         (data, (row_indices, col_indices)), 
# #         shape=(N, N), 
# #         dtype=np.int32
# #     )
    
# #     return adj_matrix

# # @jit
# # def get_age_weights(ages):
# #     """Get COVID and US age probability weights"""
# #     # COVID age probabilities by decade
# #     covid_weights = jnp.select(
# #         [ages < 10, ages < 20, ages < 30, ages < 40, ages < 50, 
# #          ages < 60, ages < 70, ages < 80],
# #         [2.3, 5.1, 15.5, 16.9, 16.4, 16.4, 11.9, 7.0],
# #         default=8.5
# #     )
    
# #     us_weights = jnp.select(
# #         [ages < 5, ages < 15, ages < 25, ages < 35, ages < 45,
# #          ages < 55, ages < 65, ages < 75, ages < 85],
# #         [5.7, 12.5, 13.0, 13.7, 13.1, 12.3, 12.9, 10.1, 4.9],
# #         default=1.8
# #     )
    
# #     return covid_weights, us_weights

# # @jit
# # def calculate_lc_multiplier(ages, genders, vaccinated, num_infections, 
# #                             has_lc, is_asymptomatic, lc_mult_female,
# #                             reinfection_mult, asymptomatic_mult):
# #     """Calculate LC incidence multiplier for each agent"""
# #     # Age multiplier
# #     age_mult = jnp.select(
# #         [ages < 30, (ages >= 50) & (ages <= 64), ages >= 65],
# #         [0.9, 1.2, 1.3],
# #         default=1.0
# #     )
    
# #     # Gender multiplier
# #     gender_mult = jnp.where(genders == 1, lc_mult_female, 1.0)
    
# #     # Vaccination effect
# #     vacc_mult = jnp.where(vaccinated, 0.7, 1.0)
    
# #     # Reinfection (only if no prior LC)
# #     reinf_mult = jnp.where((num_infections > 1) & (~has_lc), reinfection_mult, 1.0)
    
# #     # Asymptomatic discount
# #     asym_mult = jnp.where(is_asymptomatic, asymptomatic_mult, 1.0)
    
# #     return age_mult * gender_mult * vacc_mult * reinf_mult * asym_mult

# # @jit
# # def weibull_recovery_chance(duration, k, lambda_param):
# #     """Calculate daily recovery probability from Weibull hazard"""
# #     # Avoid division by zero
# #     safe_lambda = jnp.maximum(lambda_param, 1e-6)
# #     safe_duration = jnp.maximum(duration, 1e-6)
    
# #     t_scaled = safe_duration / safe_lambda
# #     hazard = (k / safe_lambda) * jnp.power(t_scaled, k - 1)
    
# #     daily_prob = (1 - jnp.exp(-hazard)) * 100
# #     daily_prob = jnp.clip(daily_prob, 0.01, 10.0)
    
# #     return daily_prob

# # def simulate_gpu(
# #     N=100000,
# #     max_days=365,
# #     covid_spread_chance_pct=10.0,
# #     initial_infected_agents=10,
# #     precaution_pct=50.0,
# #     avg_degree=5,
# #     v_start_time=180,
# #     vaccination_pct=80.0,
# #     infected_period=10,
# #     active_duration=7,
# #     immune_period=21,
# #     asymptomatic_pct=40.0,
# #     # Long COVID
# #     long_covid=True,
# #     lc_onset_base_pct=15.0,
# #     long_covid_time_threshold=30,
# #     lc_base_fast_prob=9.0,
# #     lc_base_persistent_prob=7.0,
# #     lc_incidence_mult_female=1.2,
# #     reinfection_new_onset_mult=0.7,
# #     asymptomatic_lc_mult=0.5,
# #     # Vaccination
# #     efficiency_pct=80.0,
# #     vaccination_decay=True,
# #     vaccine_priority=True,
# #     # Demographics
# #     age_distribution=True,
# #     age_range=100,
# #     age_infection_scaling=True,
# #     gender=True,
# #     seed=None,
# # ):
# #     """
# #     GPU-accelerated COVID ABM simulation for large populations
# #     """
# #     print(f"Running simulation with {N:,} agents on {jax.devices()[0].device_kind}...")
    
# #     # Initialize RNG
# #     if seed is None:
# #         seed = np.random.randint(0, 2**31)
# #     key = random.PRNGKey(seed)
    
# #     # Initialize agent arrays on GPU
# #     state = jnp.zeros(N, dtype=jnp.int32)
# #     infected = jnp.zeros(N, dtype=jnp.bool_)
# #     immuned = jnp.zeros(N, dtype=jnp.bool_)
# #     symptomatic = jnp.zeros(N, dtype=jnp.bool_)
    
# #     # LC tracking
# #     persistent_lc = jnp.zeros(N, dtype=jnp.bool_)
# #     lc_severity = jnp.zeros(N, dtype=jnp.float32)
# #     lc_duration = jnp.zeros(N, dtype=jnp.int32)
# #     lc_group = jnp.full(N, -1, dtype=jnp.int32)
# #     lc_weibull_k = jnp.zeros(N, dtype=jnp.float32)
# #     lc_weibull_lambda = jnp.zeros(N, dtype=jnp.float32)
    
# #     # Infection tracking
# #     virus_timer = jnp.zeros(N, dtype=jnp.int32)
# #     num_infections = jnp.zeros(N, dtype=jnp.int32)
# #     infectious_start = jnp.ones(N, dtype=jnp.int32)
# #     infectious_end = jnp.ones(N, dtype=jnp.int32) + active_duration
# #     symptomatic_start = jnp.zeros(N, dtype=jnp.int32)
# #     symptomatic_duration = jnp.zeros(N, dtype=jnp.int32)
    
# #     # Demographics
# #     key, ages, genders = setup_age_distribution(key, N, age_distribution, age_range, gender)
# #     covid_age_prob, us_age_prob = get_age_weights(ages)
    
# #     # Vaccination
# #     vaccinated = jnp.zeros(N, dtype=jnp.bool_)
# #     vacc_time = jnp.zeros(N, dtype=jnp.int32)
    
# #     # Create network (on CPU for efficiency, then transfer to GPU)
# #     print("Creating network...")
# #     adj_sparse = create_network_cpu(N, avg_degree, seed)
    
# #     # Convert to dense for small networks, keep sparse for large
# #     if N <= 10000:
# #         adj_matrix = jnp.array(adj_sparse.toarray())
# #     else:
# #         # Keep as scipy sparse and use batch processing
# #         adj_matrix = adj_sparse
    
# #     # Seed initial infections
# #     key, subkey = random.split(key)
# #     initial_idx = random.choice(subkey, N, shape=(min(N, initial_infected_agents),), replace=False)
    
# #     infected = infected.at[initial_idx].set(True)
# #     state = state.at[initial_idx].set(INFECTED)
# #     num_infections = num_infections.at[initial_idx].set(1)
    
# #     # Metrics
# #     total_infected = initial_infected_agents
# #     total_reinfected = 0
# #     long_covid_cases = 0
# #     min_productivity = 100.0
    
# #     print("Starting simulation...")
    
# #     # Main simulation loop
# #     for day in range(max_days):
# #         if day % 30 == 0:
# #             print(f"Day {day}: {jnp.sum(infected)} infected, {jnp.sum(persistent_lc)} with LC")
        
# #         # Vaccination at start time
# #         if day == v_start_time and vaccination_pct > 0:
# #             n_vacc = int(N * vaccination_pct / 100)
# #             key, subkey = random.split(key)
            
# #             if vaccine_priority:
# #                 # Priority: elderly first, then by health risk
# #                 priority_score = ages.astype(jnp.float32) / 100.0
# #                 vacc_idx = jnp.argsort(-priority_score)[:n_vacc]
# #             else:
# #                 vacc_idx = random.choice(subkey, N, shape=(n_vacc,), replace=False)
            
# #             vaccinated = vaccinated.at[vacc_idx].set(True)
# #             vacc_time = vacc_time.at[vacc_idx].set(1)
        
# #         # Long COVID progression
# #         if long_covid:
# #             # Update LC duration
# #             lc_duration = jnp.where(persistent_lc, lc_duration + 1, lc_duration)
            
# #             # Calculate recovery chances
# #             recovery_chance = weibull_recovery_chance(lc_duration, lc_weibull_k, lc_weibull_lambda)
            
# #             # Group-specific scaling
# #             recovery_chance = jnp.where(lc_group == LC_FAST, recovery_chance * 2.0, recovery_chance)
# #             recovery_chance = jnp.where(lc_group == LC_PERSISTENT, recovery_chance * 0.3, recovery_chance)
# #             recovery_chance = jnp.where(
# #                 (lc_group == LC_PERSISTENT) & (lc_duration > 1095),
# #                 recovery_chance * 0.1,
# #                 recovery_chance
# #             )
            
# #             recovery_chance = jnp.clip(recovery_chance, 0, 15)
            
# #             # Recovery rolls
# #             key, subkey = random.split(key)
# #             recover = persistent_lc & (random.uniform(subkey, (N,)) * 100 < recovery_chance)
            
# #             persistent_lc = jnp.where(recover, False, persistent_lc)
# #             lc_severity = jnp.where(recover, 0.0, lc_severity)
# #             lc_duration = jnp.where(recover, 0, lc_duration)
# #             lc_group = jnp.where(recover, -1, lc_group)
        
# #         # Transmission (batched for large networks)
# #         if isinstance(adj_matrix, sparse.csr_matrix):
# #             # For large networks: process in batches
# #             infectious_agents = np.where(
# #                 (infected.to_py()) & 
# #                 (virus_timer >= infectious_start) & 
# #                 (virus_timer < infectious_end)
# #             )[0]
            
# #             new_inf = []
# #             batch_size = 1000
            
# #             for i in range(0, len(infectious_agents), batch_size):
# #                 batch = infectious_agents[i:i+batch_size]
# #                 # Get neighbors from sparse matrix
# #                 for agent_id in batch:
# #                     neighbors = adj_matrix.getrow(agent_id).indices
                    
# #                     # Filter susceptible neighbors
# #                     susceptible_neighbors = neighbors[
# #                         (~infected[neighbors]) & 
# #                         (~immuned[neighbors])
# #                     ]
                    
# #                     if len(susceptible_neighbors) == 0:
# #                         continue
                    
# #                     # Vaccine protection
# #                     key, subkey = random.split(key)
# #                     if vaccination_decay:
# #                         real_eff = jnp.maximum(0, efficiency_pct - 0.11 * vacc_time[susceptible_neighbors])
# #                     else:
# #                         real_eff = efficiency_pct
                    
# #                     vacc_protected = vaccinated[susceptible_neighbors] & (
# #                         random.uniform(subkey, (len(susceptible_neighbors),)) * 100 < real_eff
# #                     )
                    
# #                     # Infection probability
# #                     infection_prob = covid_spread_chance_pct
# #                     if age_infection_scaling:
# #                         infection_prob *= covid_age_prob[susceptible_neighbors] / (us_age_prob[susceptible_neighbors] + 1e-9)
                    
# #                     infection_prob = jnp.clip(infection_prob, 0, 100)
                    
# #                     # Infection attempts
# #                     key, subkey = random.split(key)
# #                     attempts = (~vacc_protected) & (random.uniform(subkey, (len(susceptible_neighbors),)) * 100 < infection_prob)
                    
# #                     new_inf.extend(susceptible_neighbors[attempts])
            
# #             # Apply new infections
# #             if len(new_inf) > 0:
# #                 new_inf = jnp.array(list(set(new_inf)))
# #                 infected = infected.at[new_inf].set(True)
# #                 state = state.at[new_inf].set(INFECTED)
# #                 num_infections = num_infections.at[new_inf].add(1)
# #                 virus_timer = virus_timer.at[new_inf].set(0)
                
# #                 # Track reinfections
# #                 total_reinfected += int(jnp.sum(num_infections[new_inf] > 1))
# #                 total_infected += len(new_inf)
# #         else:
# #             # Small network: use matrix multiplication
# #             infectious_mask = infected & (virus_timer >= infectious_start) & (virus_timer < infectious_end)
# #             susceptible_mask = ~infected & ~immuned
            
# #             # Matrix multiply to find exposed agents
# #             key, subkey = random.split(key)
# #             exposure = jnp.dot(adj_matrix, infectious_mask.astype(jnp.float32))
# #             exposed = (exposure > 0) & susceptible_mask
            
# #             # Apply transmission probability
# #             infection_prob = covid_spread_chance_pct / 100.0
# #             key, subkey = random.split(key)
# #             new_infections_mask = exposed & (random.uniform(subkey, (N,)) < infection_prob)
            
# #             new_inf_idx = jnp.where(new_infections_mask)[0]
# #             if len(new_inf_idx) > 0:
# #                 infected = infected.at[new_inf_idx].set(True)
# #                 num_infections = num_infections.at[new_inf_idx].add(1)
# #                 virus_timer = virus_timer.at[new_inf_idx].set(0)
# #                 total_infected += len(new_inf_idx)
        
# #         # Update infection timers
# #         virus_timer = jnp.where(infected | immuned, virus_timer + 1, virus_timer)
        
# #         # State transitions: infected -> immune
# #         should_recover = infected & (virus_timer > infected_period)
# #         infected = jnp.where(should_recover, False, infected)
# #         immuned = jnp.where(should_recover, True, immuned)
        
# #         # State transitions: immune -> susceptible
# #         should_lose_immunity = immuned & (virus_timer > infected_period + immune_period)
# #         immuned = jnp.where(should_lose_immunity, False, immuned)
# #         virus_timer = jnp.where(should_lose_immunity, 0, virus_timer)
        
# #         # Update vaccination time
# #         vacc_time = jnp.where(vaccinated, vacc_time + 1, vacc_time)
        
# #         # Calculate productivity
# #         symptomatic_loss = jnp.sum(symptomatic).astype(jnp.float32)
# #         lc_loss = jnp.sum(lc_severity / 100.0)
# #         total_loss = symptomatic_loss + lc_loss
# #         current_prod = (1 - total_loss / N) * 100
# #         min_productivity = min(min_productivity, float(current_prod))
        
# #         # Check stopping condition
# #         if jnp.sum(infected) == 0 and jnp.sum(immuned) == 0:
# #             print(f"Epidemic ended at day {day}")
# #             break
    
# #     return {
# #         'runtime_days': day + 1,
# #         'infected': int(total_infected),
# #         'reinfected': int(total_reinfected),
# #         'long_covid_cases': int(jnp.sum(persistent_lc)),
# #         'min_productivity': float(min_productivity),
# #     }


# # if __name__ == "__main__":
# #     # Check available device
# #     print(f"JAX devices: {jax.devices()}")
# #     print(f"Default backend: {jax.default_backend()}")
    
# #     # Small test
# #     result = simulate_gpu(
# #         N=10000,
# #         max_days=180,
# #         covid_spread_chance_pct=10.0,
# #         initial_infected_agents=50,
# #         seed=42
# #     )
    
# #     print("\n=== Results ===")
# #     print(f"Runtime: {result['runtime_days']} days")
# #     print(f"Infected: {result['infected']:,}")
# #     print(f"Reinfected: {result['reinfected']:,}")
# #     print(f"Long COVID: {result['long_covid_cases']:,}")
# #     print(f"Min Productivity: {result['min_productivity']:.2f}%")
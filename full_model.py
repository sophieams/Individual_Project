"""
Implementation of Suicide Model by Wang et. al. from 
'Mathematical and Computational Modeling of Suicide as a Complex Dynamical System'.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 36,
    "axes.titlesize": 36,
    "axes.labelsize": 36,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 24,
    "image.cmap": "Blues",
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "text.usetex": True, "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{cmbright}"
})

fig_width = 20
fig_height = 8
legend_size = 10
axis_size = 14

t, dt = 20160, 0.01

@njit
def gen_suicide_sim(t, dt, initial_params, other_params, rng):
    """
    Runs the suicide model developed by Wang et. al.
    
    Parameters:
    t - Number of minutes for the simulation. 
    dt - Integration difference for approximation. 
    initial_params - Initial values for each variable
    other_params - Non-initial parameters in sensitivity analysis.
    rng - Numba seed used for the simulation. 
    """
    S0, A0, U0, T0, O0, E0, I0 = initial_params
    (sigma, f1, K2, b2, a2, d2, e2, g2, c3, b3, d4, c41, c42, e5, c51, c52, K6, f6, b6, c6, K7, g7, b7, c7) = other_params
    mu = (sigma ** 2) / 2

    model_sim = np.zeros((7, t))

    stressor, stressor[0] = model_sim[0], S0
    av_state, av_state[0] = model_sim[1], A0
    urge_escape, urge_escape[0] = model_sim[2], U0
    sui_thoughts, sui_thoughts[0] = model_sim[3], T0
    other_escape, other_escape[0] = model_sim[4], O0
    ext_change, ext_change[0] = model_sim[5], E0
    int_change, int_change[0] = model_sim[6], I0

    for i in range(t - 1):
        # Stochastic process for stressor
        stressor[i + 1] = max(stressor[i] * np.exp((mu - ((sigma**2)/2)) * dt + sigma * rng.normal(0, np.sqrt(dt)) - f1 * ext_change[i]), 0)
        
        # Aversive internal state including negative effect of suicidal thoughts
        av_state[i + 1] = max(av_state[i] + dt * (b2 * av_state[i] * (K2 - av_state[i]) + a2 * stressor[i] - d2 * sui_thoughts[i] - e2 * other_escape[i] - g2 * int_change[i]), 0)

        # Urge to escape
        urge_escape[i + 1] = max(urge_escape[i] + dt * (-c3 * urge_escape[i] + b3 * av_state[i]), 0)

        # Suicidal thoughts
        sui_thoughts[i + 1] = max(sui_thoughts[i] + dt * (-d4 * sui_thoughts[i] + (1 / (1 + np.exp(-c41 * (urge_escape[i] - c42))))), 0)

        # Other escape behaviors
        other_escape[i + 1] = max(other_escape[i] + dt * (-e5 * other_escape[i] + (1 / (1 + np.exp(-c51 * (urge_escape[i] - c52))))), 0)

        # External-focused change
        ext_change[i + 1] = max(ext_change[i] + dt * (f6 * ext_change[i] * (K6 - ext_change[i]) + b6 * av_state[i] - c6 * urge_escape[i]), 0)

        # Internal-focused change
        int_change[i + 1] = max(int_change[i] + dt * (g7 * int_change[i] * (K7 - int_change[i]) + b7 * av_state[i] - c7 * urge_escape[i]), 0)

    return model_sim

def set_params(stressor_setting):
    """
    This function provides values for each parameter and calculates sigma and f1. 
    
    Parameters:
    stressor_setting - Either "low", "medium", or "high" according to desired fluctuations in stressors.
    """
    if stressor_setting == "low":
        sigma = 0.05
    if stressor_setting == "medium":
        sigma = 0.1
    if stressor_setting == "high":
        sigma = 0.12
    
    mu, f1 = ((sigma**2) / 2), 0.0001
    initial_params = np.array([0.2, 0.3, 0.25, 0, 0.2, 0.05, 0.1])
    other_params = np.array([sigma, f1, 0.2, 4, 2, 1.5, 1, 0.5, 3, 1.5, 1, 100, 0.25, 3, 50, 0.2, 0.1, 0.5, 0.41, 0.82, 0.05, 0.5, 0.65, 1.3])
    
    return initial_params, other_params

initial_params, other_params = set_params("high")

fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # Create a single plot

rng = np.random.default_rng(504)
sui_model_sim = gen_suicide_sim(t=t, dt=dt, initial_params=initial_params, other_params=other_params, rng=rng)
time_days = np.arange(t) / 1440

ax.plot(time_days, sui_model_sim[0], label='stressor')
ax.plot(time_days, sui_model_sim[1], label='aversive internal state')
ax.plot(time_days, sui_model_sim[2], label='urge to escape')
ax.plot(time_days, sui_model_sim[3], label='suicidal thoughts')
ax.plot(time_days, sui_model_sim[4], label='escape behavior')
ax.plot(time_days, sui_model_sim[5], label='external-focused change')
ax.plot(time_days, sui_model_sim[6], label='internal-focused change')
ax.set_ylim([0, 1])
fig.legend(bbox_to_anchor=(1.13, 0.904))
fig.text(0.5, 0.02, 'Days', ha='center')
fig.text(0.07, 0.5, 'Intensity', va='center', rotation='vertical')
plt.savefig("figures/full_model.png", dpi=300, transparent=False, bbox_inches='tight')


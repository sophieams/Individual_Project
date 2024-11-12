"""
Runs suicide model from Wang et. al., sampling for specified parameter ranges, saving the model output, and 
plotting the results. 
Desired maximum output variable can be specified in run_model with np.max(result[X]). 
"""
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from full_model import gen_suicide_sim, set_params
from plot_indices import plot_combined_sensitivity
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

parameters = {
    'num_vars': 25,
    'names': ['sigma', 'f1', 'K2', 'b2', 'a2', 'd2', 'e2', 'g2', 
              'c3', 'b3', 'd4', 'c41', 'c42', 'e5', 'c51', 'c52', 
              'K6', 'f6', 'b6', 'c6', 'K7', 'g7', 'b7', 'c7', 'dummy'],
    'bounds': [
        [0.05, 0.12],      # sigma - volatility
        [0.0001, 0.01],    # f1 - regulating effect of external-focused strategies on stressors over time
        [0.1, 1],          # K2 - carrying capacity of aversive internal states
        [1, 10],           # b2 - self-feedback loop of aversive internal states
        [1, 10],           # a2 - effect of stressors on change in aversive internal states over time
        [0.1, 0.5],        # d2 - the effect of suicidal thoughts on change in aversive internal states over time
        [0.5, 2.5],        # e2 - the effect of escape behaviors on change in aversive internal states over time
        [0.1, 1],          # g2 - the effect of internal-focused strategies on change in aversive internal states over time
        [1, 5],            # c3 - self-feedback loop of urge to escape
        [0.5, 2],          # b3 - effect of aversive internal states on change in urge to escape over time
        [0.1, 5],          # d4 - the self-feedback loop of suicidal thoughts
        [90, 110],         # c41 - steepness of the sigmoidal curve for suicidal thoughts
        [0.1, 0.5],        # c42 - midpoint of the sigmoidal curve for suicidal thoughts
        [1, 5],            # e5 - the self-feedback loop of escape behaviors
        [40, 60],          # c51 - steepness of the sigmoidal curve for escape behavior
        [0.1, 0.5],        # c52 - midpoint of the sigmoidal curve for escape behavior
        [0.05, 0.2],       # K6 - carrying capacity of external-focused strategies
        [0.1, 1],          # f6 - self-feedback loop of external-focused strategies
        [0.1, 1],          # b6 - effect of aversive internal states on external-focused strategies
        [0.5, 2],          # c6 - effect of urge to escape on external-focused strategies
        [0.05, 0.2],       # K7 - carrying capacity of internal-focused strategies
        [0.1, 1],          # g7 - self-feedback loop of internal-focused strategies
        [0.1, 1],          # b7 - effect of aversive internal states on internal-focused strategies
        [0.5, 2],          # c7 - effect of urge to escape on internal-focused strategies
        [0, 1]             # dummy
    ]
}

param_values = saltelli.sample(parameters, 131072, calc_second_order=True)

def run_model(param_set):
    initial_params, _ = set_params("high")
    sigma = param_set[0]
    mu = (sigma ** 2) / 2
    other_params = np.insert(param_set[1:24], 0, mu)  # exclude dummy
    
    t, dt = 20160, 0.01
    rng = np.random.default_rng(504)
    result = gen_suicide_sim(t=t, dt=dt, initial_params=initial_params, other_params=other_params, rng=rng)
    
    # result[1] is "aversive internal state"
    # result[3] is "suicidal thoughts"
    # result[4] is "escape behavior"
    
    return np.max(result[4])

Y = np.array([run_model(param_set) for param_set in tqdm(param_values, desc="Running simulations", ncols=100)])
Si = sobol_analyze.analyze(parameters, Y, calc_second_order=True)

results_df = pd.DataFrame(param_values, columns=parameters['names'])
results_df['Model_Output'] = Y
results_df.to_csv('overall_data/EB_sensitivity_analysis_results_17.csv', index=False)

plot_combined_sensitivity(Si, f'overall_data/EB_pawn_14.csv', parameters['names'], 'Escape Behavior')

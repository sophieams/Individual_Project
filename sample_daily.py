"""
Runs suicide model from Wang et. al., sampling for specified parameter ranges, running a Sobol analysis, 
and saves Sobol analysis results. A sobol analysis is run on each day of the 14 day simulation. 
Desired output variable can be specified in run_model with result[X]. 
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from SALib.analyze import sobol
from SALib.sample import saltelli
from full_model import gen_suicide_sim, set_params
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

parameters = {
    'num_vars': 25, 
    'names': ['sigma', 'f1', 'K2', 'b2', 'a2', 'd2', 'e2', 'g2', 
              'c3', 'b3', 'd4', 'c41', 'c42', 'e5', 'c51', 'c52', 
              'K6', 'f6', 'b6', 'c6', 'K7', 'g7', 'b7', 'c7', 
              'dummy'],
    'bounds': [
        [0.05, 0.12], [0.0001, 0.01], [0.1, 1], [1, 10], [1, 10], 
        [0.1, 0.5], [0.5, 2.5], [0.1, 1], [1, 5], [0.5, 2], 
        [0.1, 5], [90, 110], [0.1, 0.5], [1, 5], [40, 60], 
        [0.1, 0.5], [0.05, 0.2], [0.1, 1], [0.1, 1], [0.5, 2], 
        [0.05, 0.2], [0.1, 1], [0.1, 1], [0.5, 2], [0.0, 1.0]  # dummy
    ]
}

param_values = saltelli.sample(parameters, 65536, calc_second_order=False)

def run_model(param_set):
    """ Simulate the model with a given set of parameters. """
    initial_params, _ = set_params("high")
    sigma = param_set[0]
    mu = (sigma ** 2) / 2
    other_params = np.insert(param_set[1:24], 0, mu)
    rng = np.random.default_rng(504)
    t, dt = 15*1440, 0.01  
    result = gen_suicide_sim(t=t, dt=dt, initial_params=initial_params, other_params=other_params, rng=rng)
    days = np.arange(0, 15*1440, 1440)  
    # result[1] is "aversive internal state"
    # result[3] is "suicidal thoughts"
    # result[4] is "escape behavior"
    daily_values = result[1][days]  
    
    return daily_values

Y_all_days = np.array([run_model(param_set) for param_set in tqdm(param_values, desc="Running simulations", ncols=100)])
for day in range(15):
    Y_day = Y_all_days[:, day] 
    Si = sobol.analyze(parameters, Y_day, calc_second_order=False)
    
    sobol_results_df = pd.DataFrame({
        'parameter_names': parameters['names'],
        'S1': Si['S1'],
        'S1_conf': Si['S1_conf'],
        'ST': Si['ST'],
        'ST_conf': Si['ST_conf'],
    })
    
    sobol_results_df.to_csv(f'daily_data/Day_{day}_AIS_sensitivity_analysis_results.csv', index=False)

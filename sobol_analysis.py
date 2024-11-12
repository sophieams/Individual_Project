"""
Separate file for running and saving Sobol analysis results from model_output data. 
Desired data for analysis can be specified in 'data' variable below. 
"""
import pandas as pd
import numpy as np
from SALib.analyze import sobol
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parameter_names = [
    'sigma', 'f1', 'K2', 'b2', 'a2', 'd2', 'e2', 'g2',
    'c3', 'b3', 'd4', 'c41', 'c42', 'e5', 'c51', 'c52',
    'K6', 'f6', 'b6', 'c6', 'K7', 'g7', 'b7', 'c7'
]

problem = {
    'num_vars': 24,
    'names': parameter_names,
    'bounds': [
        [0.05, 0.12], [0.0001, 0.01], [0.1, 1], [1, 10], [1, 10],
        [0.1, 0.5], [0.5, 2.5], [0.1, 1], [1, 5], [0.5, 2],
        [0.1, 5], [90, 110], [0.1, 0.5], [1, 5], [40, 60],
        [0.1, 0.5], [0.05, 0.2], [0.1, 1], [0.1, 1], [0.5, 2],
        [0.05, 0.2], [0.1, 1], [0.1, 1], [0.5, 2]
    ]
}

data = pd.read_csv("model_output/EB_sensitivity_analysis_results_17.csv")
Y = data['Model_Output'].values
Si = sobol.analyze(problem, Y, calc_second_order=True)

sobol_results_df = pd.DataFrame({
    'parameter_names': parameter_names,
    'S1': Si['S1'],
    'S1_conf': Si['S1_conf'],
    'ST': Si['ST'],
    'ST_conf': Si['ST_conf']
})

S2_pairs = []
S2_conf_pairs = []
param_pairs = []

for i in range(Si['S2'].shape[0]):
    for j in range(i + 1, Si['S2'].shape[1]):
        param_pairs.append(f"{parameter_names[i]} - {parameter_names[j]}")
        S2_pairs.append(Si['S2'][i, j])
        S2_conf_pairs.append(Si['S2_conf'][i, j])

s2_results_df = pd.DataFrame({
    'parameter_pairs': param_pairs,
    'S2': S2_pairs,
    'S2_conf': S2_conf_pairs
})

s2_results_df.to_csv("overall_data/EB_S2_sobol_17.csv", index=False)
sobol_results_df.to_csv("overall_data/EB_sobol_17.csv", index=False)


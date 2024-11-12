"""
This script calculates PAWN sensitivity indices for a specified model output file. 
Provide the path to the model output file in data_file below and the desired name of the file to save the
sensitivity indices in the last line. 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SALib.analyze import pawn

problem = {
    'num_vars': 24,
    'names': ['sigma', 'f1', 'K2', 'b2', 'a2', 'd2', 'e2', 'g2', 
              'c3', 'b3', 'd4', 'c41', 'c42', 'e5', 'c51', 'c52', 
              'K6', 'f6', 'b6', 'c6', 'K7', 'g7', 'b7', 'c7'],
    'bounds': [
        [0.05, 0.12], [0.0001, 0.01], [0.1, 1], [1, 10], [1, 10], 
        [0.1, 0.5], [0.5, 2.5], [0.1, 1], [1, 5], [0.5, 2], [0.1, 5], 
        [90, 110], [0.1, 0.5], [1, 5], [40, 60], [0.1, 0.5], [0.05, 0.2], 
        [0.1, 1], [0.1, 1], [0.5, 2], [0.05, 0.2], [0.1, 1], [0.1, 1], 
        [0.5, 2]
    ] 
}

#Example file name
data_file = 'overall_data/EB_sensitivity_analysis_results_10.csv'
data = pd.read_csv(data_file)

param_values = data.iloc[:, :-1].values
Y = data['Model_Output'].values          
Si = pawn.analyze(problem, param_values, Y, S=10)
results_df = pd.DataFrame(Si, index=problem['names'])
results_df.to_csv('overall_data/EB_pawn_10.csv', index=True)

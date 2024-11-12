"""
Script to plot top n parameters sensitivity indices over time.
Data to be plotted can be found in daily_data folder. 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 40,
    "xtick.labelsize": 36,
    "ytick.labelsize": 36,
    "legend.fontsize": 40,
    "image.cmap": "Blues",
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "text.usetex": True, "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{cmbright}"
})

parameter_names = [
    'sigma', 'f1', 'K2', 'b2', 'a2', 'd2', 'e2', 'g2',
    'c3', 'b3', 'd4', 'c41', 'c42', 'e5', 'c51', 'c52',
    'K6', 'f6', 'b6', 'c6', 'K7', 'g7', 'b7', 'c7',
    'dummy'
]

outputs = {
    "AIS": "daily_data/Day_{}_AIS_sensitivity_analysis_results.csv",
    "ST": "daily_data/Day_{}_ST_sensitivity_analysis_results.csv",
    "EB": "daily_data/Day_{}_EB_sensitivity_analysis_results.csv"
}

color_map = {
    'd4': "#0072B2", 'c3': "#56B4E9", 'c42': "#E69F00", 'b3': "#8B4513",
    'K2': "#006666", 'a2': "#CC79A7", 'c52': "#9400D3",
    'e5': "#D55E00", 'c6': "#FF6347"
}

marker_map = {
    'd4': 'o', 'c3': 's', 'c42': 'D', 'b3': '^', 'K2': '*',
    'a2': 'p', 'c6': 'X', 'c52': 'h', 'e2': 'v', 'e5': '<'
}

def load_sensitivity_data(output_label):
    """
    This function reads the precomputed sensitivity indices from a file in overall_data
    and returns it as a list. 
    
    Parameters: 
    output_label - Specification of output from file path. In this case, "ST" is Suicidal Thoughts, 
    "AIS" is Aversive Internal State, and "EB" is Escape Behavior. 
    
    Returns: 
    Si_list - List of dictionaries, where each dictionary contains sensitivity indices for a specific time step.
    """
    Si_list = []
    for day in range(1, 15):
        filepath = f"daily_data/Day_{day}_{output_label}_sensitivity_analysis_results.csv"
        results_df = pd.read_csv(filepath)
        Si = {
            'S1': results_df['S1'].values,
            'S1_conf': results_df['S1_conf'].values,
            'ST': results_df['ST'].values,
            'ST_conf': results_df['ST_conf'].values
        }
        Si_list.append(Si)
    return Si_list

def get_top_parameters(Si_list, parameter_names, index_type='S1', top_n=5):
    """
    Filters parameters with the top n sensitivity indices based on average values over time.
    
    Parameters:
    Si_list - List of dictionaries, where each dictionary contains sensitivity indices (and optionally confidence intervals) for a specific time step.
    parameter_names - List of parameter names corresponding to the sensitivity indices.
    index_type - String specifying the type of sensitivity index to use ('S1' for first-order or 'ST' for total-order indices).
    top_n - Integer representing the number of top indices to retrieve.
    
    Returns:
    top_params - List of top n parameter names with the highest average sensitivity indices.
    top_sensitivity_matrix - Array containing the sensitivity index values over time for the top n parameters.
    """
    sensitivity_matrix = np.array([Si[index_type] for Si in Si_list]).T
    avg_sensitivity = np.mean(sensitivity_matrix, axis=1)
    top_indices = np.argsort(avg_sensitivity)[-top_n:]
    top_params = [parameter_names[i] for i in top_indices]
    top_sensitivity_matrix = sensitivity_matrix[top_indices, :]
    return top_params, top_sensitivity_matrix

def plot_sensitivity_line(ax, Si_list, top_params, top_sensitivity_matrix, sensitivity_type):
    """
    Plots time-dependent sensitivity indices with confidence intervals for the specified top parameters.

    Parameters:
    ax - Axes object where the line plot will be drawn.
    Si_list - List of dictionaries, where each dictionary contains sensitivity indices and confidence intervals for each parameter at a specific time step.
    top_params - List of parameter names representing the top parameters by average sensitivity.
    top_sensitivity_matrix - Array with rows corresponding to the sensitivity values over time for the top parameters.
    sensitivity_type - Specifyies the type of sensitivity index to plot ('S1' or 'ST').
    """
    days = np.arange(1, top_sensitivity_matrix.shape[1] + 1)
    avg_sensitivities = np.mean(top_sensitivity_matrix, axis=1)
    sorted_indices = np.argsort(avg_sensitivities)[::-1]

    for i in sorted_indices:
        param = top_params[i]
        sensitivity_values = top_sensitivity_matrix[i, :]
        conf_intervals = np.array([Si[f'{sensitivity_type}_conf'][parameter_names.index(param)] for Si in Si_list])
        color = color_map.get(param, "#333333")
        marker = marker_map.get(param, 'o')
        ax.plot(days, sensitivity_values, label=param, marker=marker, color=color)
        ax.fill_between(days, sensitivity_values - conf_intervals, sensitivity_values + conf_intervals, color=color, alpha=0.2)
    
    ax.grid(True)

fig, axes = plt.subplots(3, 2, figsize=(18, 24))

for i, (output_label, title) in enumerate([("ST", "Suicidal Thoughts"), ("AIS", "Aversive Internal State"), ("EB", "Escape Behavior")]):
    Si_list = load_sensitivity_data(output_label)
    top_params_S1, top_sensitivity_matrix_S1 = get_top_parameters(Si_list, parameter_names, index_type='S1', top_n=5)
    plot_sensitivity_line(axes[i, 0], Si_list, top_params_S1, top_sensitivity_matrix_S1, 'S1')
    axes[i, 0].set_ylabel('Sensitivity Index')
    axes[2, 0].set_xlabel('Day')
    
    top_params_ST, top_sensitivity_matrix_ST = get_top_parameters(Si_list, parameter_names, index_type='ST', top_n=5)
    plot_sensitivity_line(axes[i, 1], Si_list, top_params_ST, top_sensitivity_matrix_ST, 'ST')
    axes[2, 1].set_xlabel('Day')

fig.text(0.3, 0.92, "First-Order Sensitivity Over Time", ha='center', va='center', fontsize=40)
fig.text(0.74, 0.92, "Total-Order Sensitivity Over Time", ha='center', va='center', fontsize=40)
fig.text(0.53, 0.895, "Suicidal Thoughts", ha='center', va='center', fontsize=40)
fig.text(0.53, 0.62, "Aversive Internal State", ha='center', va='center', fontsize=40)
fig.text(0.53, 0.34, "Escape Behavior", ha='center', va='center', fontsize=40)
handles, labels = [], []
for ax in axes.flat:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)

fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize=40)
plt.subplots_adjust(hspace=0.3)
plt.savefig(f'figures/time_sensitivity.png', dpi=300, bbox_inches='tight')
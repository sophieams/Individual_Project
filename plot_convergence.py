"""
Python file for creating the convergence plots for Sobol and PAWN sensitivity indices. 
Associated colors and data markers for parameters are specified according to preference below. 
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 36,
    "axes.titlesize": 36,
    "axes.labelsize": 36,
    "xtick.labelsize": 36,
    "ytick.labelsize": 36,
    "legend.fontsize": 28,
    "image.cmap": "Blues",
    "lines.linewidth": 1.5,
    "lines.markersize": 10,
    "text.usetex": True, "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{cmbright}"
})

sobol_file_patterns = {
    'ST': "overall_data/ST_sobol_*.csv",
    'AIS': "overall_data/AIS_sobol_*.csv",
    'EB': "overall_data/EB_sobol_*.csv"
}

pawn_file_patterns = {
    'ST': "overall_data/ST_pawn_*.csv",
    'AIS': "overall_data/AIS_pawn_*.csv",
    'EB': "overall_data/EB_pawn_*.csv"
}

parameter_names = [
    'sigma', 'f1', 'K2', 'b2', 'a2', 'd2', 'e2', 'g2', 
    'c3', 'b3', 'd4', 'c41', 'c42', 'e5', 'c51', 'c52', 
    'K6', 'f6', 'b6', 'c6', 'K7', 'g7', 'b7', 'c7'
]

custom_hex_codes = {
    'd4': "#0072B2",    # dark blue
    'c3': "#56B4E9",    # sky blue
    'c42': "#8B4513",   # sienna
    'b3': "#F0E442",    # yellow
    'K2': "#009E73",    # green
    'a2': "#CC79A7",    # pink
    'b2': "#D55E00",    # vermillion
    'c52': "#9400D3",   # dark violet
    'e2': "#8B4513",    # saddle brown
    'e5': "#8B0000"     # dark red
}


param_marker_map = {
    'd4': 'D',   # diamond
    'c3': 'o',   # circle
    'c42': '*',  # star
    'b3': 'D',   # diamond
    'K2': 'o',   # circle
    'a2': '*',   # star
    'b2': 'D',   # diamond
    'c52': '*',  # star
    'e2': '*',   # star
    'e5': 'D',   # diamond
}

param_color_map = {}
used_params = set(custom_hex_codes.keys())
param_color_map.update(custom_hex_codes)

def process_files(file_pattern, label):
    """
    This function reads the precomputed Sobol sensitivity indices for each file for each sample size.
    
    Parameters:
    file_pattern - The file path for each CSV file from different sample sizes.
    label - The label for the outcome variable measured, i.e. ST (suicidal thoughts), 
    AIS (aversive internal state), or EB (escape behavior).
    
    Returns: 
    sample_sizes - The sample sizes for plotting
    sensitivity_data - the analyzed data and confidence intervals
    top_params and top_params_indices - the top 3 parameters and the indices for each output. 
    """
    files = glob.glob(file_pattern)
    sample_sizes = [int(file.split("_")[-1].split(".")[0]) for file in files]
    sorted_files = [file for _, file in sorted(zip(sample_sizes, files))]
    sorted_sample_sizes = sorted(sample_sizes)
    sensitivity_indices = ['S1', 'ST']
    sensitivity_data = {index: [] for index in sensitivity_indices}
    confidence_intervals = {index: [] for index in sensitivity_indices}
    
    for i, file in enumerate(sorted_files):
        data = pd.read_csv(file)
        for index in sensitivity_indices:
            sensitivity_data[index].append(data[index].values)
            confidence_intervals[index].append(data[f"{index}_conf"].values)

    sample_sizes = np.array([2**size for size in sorted_sample_sizes])
    for index in sensitivity_data:
        sensitivity_data[index] = np.array(sensitivity_data[index])
        confidence_intervals[index] = np.array(confidence_intervals[index])

    avg_sensitivity = np.mean(sensitivity_data['ST'], axis=0)
    top_params_indices = np.argsort(avg_sensitivity)[-3:]
    top_params = [parameter_names[i] for i in top_params_indices]
    
    return sample_sizes, sensitivity_data, confidence_intervals, top_params, top_params_indices

def plot_sobol_convergence():
    """
    This function plots the precomputed Sobol sensitivity data from the associated CSVs in the 
    overall_data folder. It generates a convergence plot.
    """
    results = {}
    for label, pattern in sobol_file_patterns.items():
        sample_sizes, sensitivity_data, confidence_intervals, top_params, top_indices = process_files(pattern, label)
        results[label] = (sample_sizes, sensitivity_data, confidence_intervals, top_params, top_indices)

    fig, axes = plt.subplots(1, 3, figsize=(28, 10), sharey=True)
    titles = ['Suicidal Thoughts', 'Aversive Internal State', 'Escape Behavior']

    for i, label in enumerate(results):
        sample_sizes, sensitivity_data, confidence_intervals, top_params, top_indices = results[label]
        ax = axes[i]

        avg_sens = np.mean(sensitivity_data['ST'], axis=0)  
        sorted_indices = np.argsort(avg_sens)[-3:][::-1]  

        for param_idx in sorted_indices:
            param_name = parameter_names[param_idx]
            indices = sensitivity_data['ST'][:, param_idx]
            confs = confidence_intervals['ST'][:, param_idx]
            marker = param_marker_map[param_name]
            ax.plot(sample_sizes, indices, label=f'{param_name}', linewidth=2, color=param_color_map[param_name], marker=marker)
            ax.fill_between(sample_sizes, indices - confs, indices + confs, alpha=0.2, color=param_color_map[param_name])
        ax.set_xscale('log', base=2)
        ax.set_title(titles[i], fontsize=40)
        ax.grid(True)

        handles, labels = ax.get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: np.mean(sensitivity_data['ST'][:, parameter_names.index(x[1])]), reverse=True)
        handles, labels = zip(*sorted_handles_labels)
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

    axes[0].set_ylabel('Sensitivity Index', fontsize=40)
    axes[1].set_xlabel('log(Number of Samples)', fontsize=40)
    fig.suptitle('Total-Order Sensitivity Convergence', fontsize=40, x=0.44, y=0.9)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(f'figures/total_order_convergence.png', bbox_inches='tight', dpi=300)
    
def plot_pawn_convergence():
    """
    This function plots the data from the associated PAWN csvs in the overall_data folder.
    There are no input parameters and the resulting plot is saved in the folder 'figures'.
    """ 
    results = {}

    for label, pattern in pawn_file_patterns.items():
        files = glob.glob(pattern)
        sample_sizes = [int(file.split("_")[-1].split(".")[0]) for file in files]
        sorted_files = [file for _, file in sorted(zip(sample_sizes, files))]
        sorted_sample_sizes = sorted(sample_sizes)

        sensitivity_data = {'median': [], 'mean': [], 'CV': []}

        for file in sorted_files:
            data = pd.read_csv(file, index_col=0)
            for index in ['median', 'mean', 'CV']:
                sensitivity_data[index].append(data[index].values)

        sample_sizes = np.array([2**size for size in sorted_sample_sizes])
        for index in sensitivity_data:
            sensitivity_data[index] = np.array(sensitivity_data[index])
        last_sample_idx = -1 
        last_sample_median = sensitivity_data['median'][last_sample_idx, :]
        top_params_indices = np.argsort(last_sample_median)[-3:]

        results[label] = (sample_sizes, sensitivity_data, top_params_indices)

    fig, axes = plt.subplots(1, 3, figsize=(28, 10), sharey=True)
    titles = ['Suicidal Thoughts', 'Aversive Internal State', 'Escape Behavior']

    for i, label in enumerate(results):
        sample_sizes, sensitivity_data, top_indices = results[label]
        ax = axes[i]

        param_last_sensitivity = {}

        for param_idx in top_indices:
            param_name = parameter_names[param_idx]
            median_values = sensitivity_data['median'][:, param_idx]
            mean_values = sensitivity_data['mean'][:, param_idx]
            cv_values = sensitivity_data['CV'][:, param_idx]
            marker = param_marker_map[param_name]
            ax.plot(sample_sizes, mean_values, label=f'{param_name}', linewidth=2, color=param_color_map[param_name], marker=marker)
            ax.fill_between(sample_sizes, median_values - (cv_values * mean_values), median_values + (cv_values * mean_values), alpha=0.2, color=param_color_map[param_name])
            param_last_sensitivity[param_name] = median_values[last_sample_idx]

        ax.set_xscale('log', base=2)
        ax.set_title(titles[i], fontsize=40)
        ax.grid(True)

        handles, labels = ax.get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: param_last_sensitivity[x[1]], reverse=True)
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        ax.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1))

    axes[0].set_ylabel('Sensitivity Index', fontsize=40)
    axes[1].set_xlabel('log(Number of Samples)', fontsize=40)
    plt.suptitle('PAWN Sensitivity Convergence', fontsize=40, x=0.44, y=0.9)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(f'figures/pawn_convergence.png', bbox_inches='tight', dpi=300)

plot_pawn_convergence()
plot_sobol_convergence()
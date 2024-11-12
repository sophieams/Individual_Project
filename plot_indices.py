"""
Python file for plotting total-order, first-order, second-order, and PAWN sensitivity indices from the 
precomputed sensitivity index data in the overall_data folder. 
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 36,
    "axes.titlesize": 36,
    "axes.labelsize": 36,
    "xtick.labelsize": 36,
    "ytick.labelsize": 36,
    "legend.fontsize": 18,
    "image.cmap": "Blues",
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "text.usetex": True, "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{cmbright}"
})

def plot_sobol_indices(output_label, title, second_order_threshold=0.003):
    """
    Plots first-order, total-order, and second-order sensitivity indices 
    from precomputed Sobol sensitivity CSV files.
    
    Parameters:
    output_label - Label for the output being analyzed (e.g., ST, AIS, EB).
    second_order_threshold - Threshold for plotting second-order indices. 
    """
    sobol_data = pd.read_csv(f"overall_data/{output_label}_sobol_17.csv")
    parameter_names = sobol_data['parameter_names'].values
    parameter_names = [r"$\sigma$" if name == "sigma" else name for name in parameter_names]
    S1 = sobol_data['S1'].values
    S1_conf = sobol_data['S1_conf'].values
    ST = sobol_data['ST'].values
    ST_conf = sobol_data['ST_conf'].values
    
    # First-order sensitivity
    plt.figure(figsize=(18, 9))
    plt.bar(parameter_names, S1, yerr=S1_conf, align='center', alpha=0.7, ecolor='black', capsize=10, color='blue')
    plt.ylabel('Sensitivity Indices', fontsize=36)
    plt.xticks(rotation=45)
    plt.title(f'First-Order Sensitivity for {title}', fontsize=36)
    plt.tight_layout()
    plt.savefig(f'figures/{output_label}_first_order_sensitivity.png', dpi=300)
    
    # Total-order sensitivity
    plt.figure(figsize=(18, 9))
    plt.bar(parameter_names, ST, yerr=ST_conf, align='center', alpha=0.7, ecolor='black', capsize=10, color='red')
    plt.ylabel('Sensitivity Indices', fontsize=36)
    plt.xticks(rotation=45)
    plt.title(f'Total-Order Sensitivity for {title}', fontsize=36)
    plt.tight_layout()
    plt.savefig(f'figures/{output_label}_total_order_sensitivity.png', dpi=300)

    # Second-order sensitivity
    s2_data = pd.read_csv(f'overall_data/{output_label}_S2_sobol_17.csv')
    param_pairs = s2_data['parameter_pairs'].values
    S2 = s2_data['S2'].values
    S2_conf = s2_data['S2_conf'].values
    
    plt.figure(figsize=(18, 9))
    important_pairs = [(pair, s2, s2_conf) for pair, s2, s2_conf in zip(param_pairs, S2, S2_conf) if abs(s2) > second_order_threshold]
    
    for pair, value, conf in important_pairs:
        plt.bar(pair, value, yerr=conf, align='center', alpha=0.7, ecolor='black', capsize=10, color='purple')

    plt.ylabel('Sensitivity Indices', fontsize=36)
    plt.xticks(rotation=45)
    plt.title(f'Second-Order Sensitivity for {title}', fontsize=36)
    plt.tight_layout()
    plt.savefig(f'figures/{output_label}_second_order_sensitivity.png', dpi=300)


def plot_combined_sensitivity(output_label, title):
    """
    Plot total-order Sobol sensitivity indices and PAWN sensitivity indices 
    for parameters that have Sobol sensitivity indices greater than the dummy variable.
    
    Parameters:
    output_label - Label for the output being analyzed.
    title - the label for the title. 
    """
    parameter_names = ['sigma', 'f1', 'K2', 'b2', 'a2', 'd2', 'e2', 'g2', 
                       'c3', 'b3', 'd4', 'c41', 'c42', 'e5', 'c51', 'c52', 
                       'K6', 'f6', 'b6', 'c6', 'K7', 'g7', 'b7', 'c7', 'dummy']
    
    sobol_data = pd.read_csv(f"overall_data/{output_label}_sobol_17.csv")
    ST = sobol_data['ST'].values
    ST_conf = sobol_data['ST_conf'].values

    # Filter parameters
    indices_to_plot = np.where(ST > 0.01)[0]
    filtered_parameter_names = [parameter_names[i] for i in indices_to_plot]
    filtered_ST = ST[indices_to_plot]
    filtered_ST_conf = ST_conf[indices_to_plot]

    pawn_data = pd.read_csv(f"overall_data/{output_label}_pawn_14.csv")
    pawn_median_values = pawn_data['median'].values
    pawn_mean_values = pawn_data['mean'].values
    pawn_cv_values = pawn_data['CV'].values
    pawn_error = pawn_cv_values * pawn_mean_values
    pawn_filtered_median = pawn_median_values[indices_to_plot]
    pawn_filtered_error = pawn_error[indices_to_plot]

    fig, axes = plt.subplots(1, 2, figsize=(24, 8))
    axes[0].bar(filtered_parameter_names, filtered_ST, yerr=filtered_ST_conf, align='center', alpha=0.7, ecolor='black', capsize=10, color='red')
    axes[0].set_ylabel('Sensitivity Index', fontsize=36)
    axes[0].set_xticks(np.arange(len(filtered_parameter_names)))
    axes[0].set_xticklabels(filtered_parameter_names, rotation=45, fontsize=36)
    axes[0].set_title(f'Total-Order Sensitivity for {title}', fontsize=36)
    axes[1].bar(filtered_parameter_names, pawn_filtered_median, 
                yerr=pawn_filtered_error, align='center', alpha=0.7, ecolor='black', capsize=10, color='red')
    axes[1].set_xticks(np.arange(len(filtered_parameter_names)))
    axes[1].set_xticklabels(filtered_parameter_names, rotation=45, fontsize=36)
    axes[1].set_title(f'PAWN Sensitivity for {title}', fontsize=36)
    
    plt.tight_layout()
    plt.savefig(f'figures/{output_label}_total_pawn_comparison.png', dpi=300)

def plot_combined_sensitivity_stacked():
    """
    Plots combined sensitivity indices for Suicidal Thoughts, Aversive Internal State,
    and Escape Behavior, with Total-Order Sobol sensitivity on the left column
    and PAWN sensitivity on the right column.
    """
    outputs = [("ST", "Suicidal Thoughts"), 
               ("AIS", "Aversive Internal State"), 
               ("EB", "Escape Behavior")]
    
    parameter_names = ['sigma', 'f1', 'K2', 'b2', 'a2', 'd2', 'e2', 'g2', 
                       'c3', 'b3', 'd4', 'c41', 'c42', 'e5', 'c51', 'c52', 
                       'K6', 'f6', 'b6', 'c6', 'K7', 'g7', 'b7', 'c7', 'dummy']
    
    fig, axes = plt.subplots(3, 2, figsize=(24, 24))
    
    fig.text(0.28, 0.93, "Total-Order Sensitivity Indices", ha="center", fontsize=40)
    fig.text(0.74, 0.93, "PAWN Sensitivity Indices", ha="center", fontsize=40)
    
    for i, (output_label, title) in enumerate(outputs):
        sobol_data = pd.read_csv(f"overall_data/{output_label}_sobol_17.csv")
        ST = sobol_data['ST'].values
        ST_conf = sobol_data['ST_conf'].values
        indices_to_plot = np.where(ST > 0.01)[0]
        filtered_parameter_names = [parameter_names[i] for i in indices_to_plot]
        filtered_ST = ST[indices_to_plot]
        filtered_ST_conf = ST_conf[indices_to_plot]
        
        pawn_data = pd.read_csv(f"overall_data/{output_label}_pawn_14.csv")
        pawn_median_values = pawn_data['median'].values
        pawn_mean_values = pawn_data['mean'].values
        pawn_cv_values = pawn_data['CV'].values
        pawn_error = pawn_cv_values * pawn_mean_values
        pawn_filtered_median = pawn_median_values[indices_to_plot]
        pawn_filtered_error = pawn_error[indices_to_plot]

        axes[i, 0].bar(filtered_parameter_names, filtered_ST, yerr=filtered_ST_conf, align='center', alpha=0.7, ecolor='black', capsize=10, color='red')
        axes[i, 0].set_ylabel('Sensitivity Index', fontsize=36)
        axes[i, 0].set_xticks(np.arange(len(filtered_parameter_names)))
        axes[i, 0].set_xticklabels(filtered_parameter_names, rotation=45, fontsize=36)
        
        axes[i, 1].bar(filtered_parameter_names, pawn_filtered_median, yerr=pawn_filtered_error, align='center', alpha=0.7, ecolor='black', capsize=10, color='red')
        axes[i, 1].set_xticks(np.arange(len(filtered_parameter_names)))
        axes[i, 1].set_xticklabels(filtered_parameter_names, rotation=45, fontsize=36)

    plt.subplots_adjust(hspace=0.5)
    fig.text(0.5, 0.90, "Maximum Suicidal Thoughts", ha='center', fontsize=40, weight='bold')
    fig.text(0.5, 0.61, "Maximum Aversive Internal State", ha='center', fontsize=40, weight='bold')
    fig.text(0.5, 0.32, "Maximum Escape Behavior", ha='center', fontsize=40, weight='bold')
    plt.savefig(f'figures/combined_total_pawn_sensitivity_comparison.png', dpi=300, bbox_inches='tight')
    
def plot_first_order_sensitivity_stacked():
    """
    Plots the first-order sensitivity indices for Suicidal Thoughts, 
    Aversive Internal State, and Escape Behavior stacked in rows.
    """
    outputs = [("ST", "Suicidal Thoughts"), 
               ("AIS", "Aversive Internal State"), 
               ("EB", "Escape Behavior")]
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 27), sharex=False)
    fig.suptitle("First-Order Sensitivity Indices", fontsize=40, y=0.93)

    for i, (output_label, title) in enumerate(outputs):
        sobol_data = pd.read_csv(f"overall_data/{output_label}_sobol_17.csv")
        parameter_names = sobol_data['parameter_names'].values
        parameter_names = [r"$\sigma$" if name == "sigma" else name for name in parameter_names]
        S1 = sobol_data['S1'].values
        S1_conf = sobol_data['S1_conf'].values
        axes[i].bar(parameter_names, S1, yerr=S1_conf, align='center', alpha=0.7, ecolor='black', capsize=10, color='blue')
        axes[i].set_ylabel('Sensitivity Index', fontsize=36)
        axes[i].set_xticks(np.arange(len(parameter_names)))
        axes[i].set_xticklabels(parameter_names, rotation=45, fontsize=30)
        axes[i].set_title(f"Maximum {title}", fontsize=40)
    plt.subplots_adjust(hspace=0.3, top=0.88, bottom=0.05)
    plt.savefig('figures/first_order_sensitivity_stacked.png', dpi=300, bbox_inches='tight')

def plot_second_order_sensitivity_stacked(output_labels, titles, second_order_threshold=0.01):
    """
    Plots stacked second-order sensitivity indices for specified outputs.
    
    Parameters:
    output_labels - List of output labels.
    titles - Corresponding titles for each output.
    second_order_threshold - Threshold for plotting second-order indices.
    """
    fig, axes = plt.subplots(len(output_labels), 1, figsize=(18, 12 * len(output_labels)), sharex=False)
    
    for i, (output_label, title) in enumerate(zip(output_labels, titles)):
        s2_data = pd.read_csv(f'overall_data/{output_label}_S2_sobol_17.csv')
        param_pairs = s2_data['parameter_pairs'].values
        S2 = s2_data['S2'].values
        S2_conf = s2_data['S2_conf'].values
        
        significant_pairs = [(pair, s2, s2_conf) for pair, s2, s2_conf in zip(param_pairs, S2, S2_conf) if abs(s2) > second_order_threshold]
        for pair, value, conf in significant_pairs:
            axes[i].bar(pair, value, yerr=conf, align='center', alpha=0.7, ecolor='black', capsize=10, color='purple')
        
        axes[i].set_ylabel('Sensitivity Index')
        axes[i].set_title(f'Maximum {title}', fontsize=40)
        axes[i].tick_params(axis='x', rotation=45)
    
    fig.text(0.53, 0.96, "Second-Order Sensitivity Indices", ha='center', fontsize=40, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2)
    plt.savefig('figures/second_order_sensitivity_stacked.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    analysis_type = input("Enter the analysis type ('only sobol', 'sobol/pawn comparison','first-order stacked', 'second-order stacked', or 'comparison stacked'): ").strip().lower()

    if analysis_type == "only sobol":
        output_label = input("Enter the output of interest (ST, AIS, or EB): ")
        if output_label == "ST":
            plot_sobol_indices(output_label, "Suicidal Thoughts")
        elif output_label == "AIS":
            plot_sobol_indices(output_label, "Aversive Internal State")
        elif output_label == "EB":
            plot_sobol_indices(output_label, "Escape Behavior")

    elif analysis_type == "sobol/pawn comparison":
        output_label = input("Enter the output of interest (ST, AIS, or EB): ")
        if output_label == "ST":
            plot_combined_sensitivity(output_label, "Suicidal Thoughts")
        elif output_label == "AIS":
            plot_combined_sensitivity(output_label, "Aversive Internal State")
        elif output_label == "EB":
            plot_combined_sensitivity(output_label, "Escape Behavior")
            
    elif analysis_type == "comparison stacked":
        plot_combined_sensitivity_stacked()
    
    elif analysis_type == "second-order stacked":
        plot_second_order_sensitivity_stacked(['ST', 'EB'], ['Suicidal Thoughts', 'Escape Behavior'])
        
    elif analysis_type == "first-order stacked":
        plot_first_order_sensitivity_stacked()
    else:
        print("Invalid input. Please enter either 'sobol', 'pawn', or 'sobol/pawn comparison'.")

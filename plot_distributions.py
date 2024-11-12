"""
Script to plot the model output distributions from particular files, which are specified in file_paths. 
"""
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 24,
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

file_paths = [
    'overall_data/ST_sensitivity_analysis_results_10.csv',
    'overall_data/AIS_sensitivity_analysis_results_10.csv',
    'overall_data/EB_sensitivity_analysis_results_10.csv'
]

file_labels = ['Maximum Suicidal Thoughts', 'Maximum Aversive Internal State', 'Maximum Escape Behavior']
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

for i, (file_path, label) in enumerate(zip(file_paths, file_labels)):
    data = pd.read_csv(file_path)
    model_outputs = data['Model_Output']
    axes[i].grid(True)
    axes[i].hist(model_outputs, color = "blue", bins=50)
    axes[i].set_xlabel(f'{label}', fontsize=40)

axes[1].set_title(f'Distribution of Model Outputs', fontsize=40)
axes[0].set_ylabel('Frequency', fontsize=40)
plt.tight_layout()
plt.savefig('figures/output_distributions.png', dpi=300)
plt.show()


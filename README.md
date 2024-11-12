# Sensitivity Analysis of a Complex Dynamical System of Suicide
## Sophie Engels Individual Project - MSc Computational Science

This project explores the sensitivity of a complex dynamical systems model of Suicide by Wang et. al (2023).
The original paper with the model description can be found here: https://psyarxiv.com/b29cs/. 
The original code for the model developed by Wang et. al can be found here: https://github.com/ShirleyBWang/math_model_suicide

To generate new model output data, use the files sample_daily.py or sample_overall.py. All plot files can be used to analyze said data. 

individual_project/
│
├── daily_data/          # Contains precomputed sensitivity indices for every day of the simulation.
│
├── figures/             # Contains all figures used in the individual project report.
│
├── model_output/        # Folder for full model outputs without analysis. Data was too large to add to
                           the repository, but all analyzed indices are available in overall_data and daily_data, while new model outputs can be generated with sample_daily.py or
                           sample_overall.py. 
│
├── overall_data/        # contains precomputed senstivity indices for the maximum of the specified model
                           output.
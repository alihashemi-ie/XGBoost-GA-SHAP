# Repository Overview:
# This repository contains the R source code utilized in our research papers for optimizing 
# hyperparameters of an XGBoost model for regression using a Genetic Algorithm (GA).
# The optimization process will initiate for a connected dataset.
# Following the hyperparameter optimization, the code trains the XGBoost model and conducts 
# cross-validation to rigorously evaluate its performance. Additionally, it employs SHAP (SHapley 
# Additive exPlanations) plots to interpret the interactions between input features and the 
# response variable, offering valuable insights into the model's behavior.
#
# We hope that this code will contribute to future research and assist researchers in their
# endeavors.
#
# We used this code in our published paper (https://doi.org/10.1039/d5nr00016e) and will continue to use it 
# in our future research studies. Therefore, we respectfully request that if you choose to utilize this code in 
# your research or work, you please cite this repository and the associated article. 
# Your acknowledgment significantly contributes to supporting ongoing and future research endeavors. 
#
# Developed by:
#    Ali Hashemi Baghi
#    ahashemi.ie@gmail.com
# 
#    Version 1.0 
#    April 28, 2025

# Load necessary libraries
library(SHAPforxgboost)  # For SHAP values analysis
library(xgboost)         # For XGBoost model implementation
library(ggplot2)         # For data visualization
library(gridExtra)       # For arranging multiple plots
library(readr)           # For reading data
library(data.table)      # For fast data manipulation
library(genalg)          # For genetic algorithms
library(GA)              # For genetic algorithm optimization

# Calculate SHAP values and prepare data for plotting
shap_values <- shap.values(xgb_model = mod1, X_train = X1)
shap_long <- shap.prep(xgb_model = mod1, X_train = X1)

# Create and customize SHAP summary plot
shap_summary_plot <- shap.plot.summary(shap_long)
shap_summary_plot_final <- shap_summary_plot + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
shap_summary_plot_final

# Assume mod1 and X1 are already defined and SHAP values have been calculated
shap_values <- shap.values(xgb_model = mod1, X_train = X1)
shap_long <- shap.prep(xgb_model = mod1, X_train = X1)

feature_names <- colnames(X1)
plot_list <- list() # Initialize an empty list to store the plots

for (feature_name in feature_names) {
  # Create SHAP dependence plot for the selected feature
  p <- shap.plot.dependence(data_long = shap_long, x = feature_name, color_feature = feature_name)
  plot_list[[feature_name]] <- p # Add the plot to the list
}

# Determine the number of rows and columns for the grid
n_cols <- floor(sqrt(length(plot_list)))
n_rows <- ceiling(length(plot_list) / n_cols)

# Combine the plots into a grid
grid_plot <- do.call(grid.arrange, c(plot_list, ncol = n_cols, nrow = n_rows))
# Save the combined grid plot as an EMF file using the custom device function
ggsave("combined_shap_plots.emf", plot = grid_plot, width = 8, height = 6, device = emf_device)

# Repository Overview:
# This repository contains the R source code utilized in our research papers for optimizing 
# hyperparameters of an XGBoost model for regression using a Genetic Algorithm (GA).
# The optimization process will initiate for a connected dataset.
# Following the hyperparameter optimization, the code trains the XGBoost model and conducts 
# cross-validation to rigorously evaluate its performance. Additionally, it employs SHAP (SHapley 
# Additive exPlanations) plots to interpret the interactions between input features and the 
# response variable, offering valuable insights into the model's behavior (See XGBoost-SHAP file).
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
library(xgboost)        # For XGBoost model implementation
library(ggplot2)        # For data visualization
library(gridExtra)      # For arranging multiple plots
library(readr)          # For reading data
library(data.table)     # For fast data manipulation
library(genalg)         # For genetic algorithms
library(GA)             # For genetic algorithm optimization

# Read in your local dataset
dataset <- read.csv()

# Handle target variable conversion
target_var <- as.character(dataset[, ncol(dataset)]) # Convert to character
target_var_numeric <- as.numeric(target_var) # Convert to numeric
invalid_indices <- which(is.na(target_var_numeric)) # Find indices with NA

if (length(invalid_indices) > 0) {
  print("The following values could not be converted to numeric:")
  print(target_var[invalid_indices])
  dataset <- dataset[-invalid_indices, ] # Remove problematic rows
}

# Update target variable after cleaning
dataset[, ncol(dataset)] <- as.numeric(target_var[-invalid_indices])

# Prepare feature matrix and target vector
X1 <- as.matrix(dataset[, -ncol(dataset)])
y <- dataset[, ncol(dataset)]

# Check for NA values in feature matrix
if (any(is.na(X1))) {
  stop("NA values found in the feature matrix.")
}

# Check for NA values in target variable
if (any(is.na(y))) {
  stop("NA values found in the target variable.")
}

# Function to train XGBoost model and return negative MSE
train_xgboost <- function(params) {
  set.seed(123)
  gamma <- params[1]
  eta <- params[2]
  lambda <- params[3]
  max_depth <- round(params[4])  # Ensure max_depth is an integer
  nrounds <- round(params[5])  
  
  if (nrounds <= 0 || max_depth < 1) return(NA) 
  
  mod <- tryCatch({
    xgboost(data = X1, label = y, gamma = gamma, eta = eta, lambda = lambda, 
            max_depth = max_depth, nrounds = nrounds, verbose = 0, nthread = 1, 
            objective = "reg:squarederror")
  }, error = function(e) {
    return(NULL) 
  })
  
  if (is.null(mod)) return(NA)
  
  preds <- predict(mod, X1)
  mse <- mean((preds - y)^2) 
  return(-mse) 
}

# Run genetic algorithm for hyperparameter optimization
ga_result <- ga(type = "real-valued", fitness = train_xgboost,
                lower = c(), 		        # Adjust lower bounds
                upper = c(), 		        # Adjust upper bounds 
                popSize = , maxiter = ) # Adjust population size and iterations

summary(ga_result)
plot(ga_result, cex.axis = 1.2, cex.lab = 1.5)

# Save the plot
png("ga_result.png", width = 800, height = 600) # Open a PNG device
plot(ga_result, cex.axis = 1.6, cex.lab = 1.8)
# Plot again to save it
dev.off() # Close the device

# Assuming gamma_seq and eta_seq are defined for plots (you may need to define these)
gamma_seq <- seq(0, 1, length.out = 100) # Example definition
eta_seq <- seq(0, 1, length.out = 100)   # Example definition
f <- matrix(runif(10000), nrow=100)      # Placeholder for MSE values

# Create surface plot
persp3D(x = gamma_seq, y = eta_seq, z = f,
        theta = 50, phi = 20,
        col.palette = bl2gr.colors,
        xlab = "Gamma", ylab = "Eta", zlab = "MSE",
        main = "MSE Surface Plot")
png("persp3D.png", width = 800, height = 600)
dev.off() # Close the device

# Create filled contour plot
filled.contour(x = gamma_seq, y = eta_seq, z = f,
               color.palette = bl2gr.colors,xlab = "Gamma", ylab = "Eta", 
               main = "Filled Contour Plot")
png("filled.contour.png", width = 800, height = 600)
dev.off() # Close the device

# Train the final model with best parameters from GA
best_params <- ga_result@solution

# Ensure that max_depth and nrounds are properly converted to integers
max_depth <- round(best_params[4])  # Assuming best_params[4] corresponds to max_depth
nrounds <- round(best_params[5])      # Assuming best_params[5] corresponds to nrounds

# Train the final XGBoost model
dtrain <- xgb.DMatrix(data = X1, label = y)  # Create DMatrix

mod1 <- xgboost(
  data = dtrain, 
  gamma = best_params[1], 
  eta = best_params[2], 
  lambda = best_params[3], 
  max_depth = max_depth, 
  nrounds = nrounds, 
  verbose = 1, 
  nthread = 1, 
  objective = "reg:squarederror"
)
# Print the final model summary
print(mod1)

# Perform cross-validation
cv_result <- xgb.cv(
  data = dtrain,
  nrounds = nrounds,
  nfold = 5,
  metrics = "rmse",
  params = list(
    gamma = best_params[1],
    eta = best_params[2],
    lambda = best_params[3],
    max_depth = max_depth,
    objective = "reg:squarederror"
  ),
  verbose = TRUE
)

# Convert cross-validation results to data frame
cv_results_df <- as.data.frame(cv_result$evaluation_log)

# Plotting the cross-validation results with enhanced aesthetics
ggplot(cv_results_df, aes(x = iter)) +
  geom_line(aes(y = train_rmse_mean, color = "Train RMSE"), size = 1.2) +
  geom_line(aes(y = test_rmse_mean, color = "Test RMSE"), size = 1.2) +
  labs(
    title = "XGBoost Cross-Validation Results",
    x = "Iteration",
    y = "Root Mean Squared Error (RMSE)",
    color = "Legend"
  ) +
  theme_minimal(base_size = 15) + 
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "top",
    legend.title = element_blank(),
    panel.grid.major = element_line(color = "grey80"),
    panel.grid.minor = element_blank()
  ) +
  scale_color_manual(values = c("Train RMSE" = "#0073C2FF", "Test RMSE" = "#D55E00FF")) +
  scale_x_continuous(breaks = seq(0, nrounds, by = 10)) +
  scale_y_continuous(labels = number_format(accuracy = 0.01))

# Save the plot if needed
ggsave("cv_results.png", width = 10, height = 6)

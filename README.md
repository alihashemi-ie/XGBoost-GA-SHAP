# XGBoost with Genetic Algorithm & SHAP Visualization
Version 1.0 April 28, 2025
## Overview
This repository provides a comprehensive R implementation for optimizing the hyperparameters of an XGBoost model for regression using a Genetic Algorithm (GA). Additionally, it utilizes SHAP (SHapley Additive exPlanations) plots to interpret the interactions between input features and the response variable, offering valuable insights into the model's behavior. We hope this approach will enhance research methodologies in machine learning.

We used this code in our published paper (https://doi.org/10.1039/d5nr00016e) and will continue to use it in our future research studies. Therefore, we respectfully request that if you choose to utilize this code in your research or work, you please cite this repository and the associated article. Your acknowledgment significantly contributes to supporting ongoing and future research endeavors. 

- **Developed by: Ali Hashemi Baghi**  
- **Contact me:** ahashemi.ie@gmail.com

I kindly invite you to feel free to reach out with any questions or issues.
## Key Features
- **Data Preparation:** Preprocessing the dataset for model training.
- **Hyperparameter Optimization:** Utilizing a Genetic Algorithm for optimizing hyperparameters.
- **Model Training:** Functions specifically designed for training the XGBoost model.
- **Model Evaluation:** Cross-validation techniques to analyze performance metrics.
- **SHAP Values:** Assessing feature importance through SHAP analysis.
- **Visualizations:** Generating surface plots, filled contour plots, cross-validation results, and SHAP plots.

## Prerequisites
Make sure you have the following R packages installed to run the code:
- `SHAPforxgboost`
- `xgboost`
- `ggplot2`
- `gridExtra`
- `readr`
- `data.table`
- `genalg`
- `GA`

## Usage
1. **Data Preparation:** Modify the dataset path in the code to read your local dataset.  `dataset <- read.csv("")`  

2. **Hyperparameter Optimization:** Adjust the ranges of hyperparameters in the genetic algorithm function. Store the best parameters in `ga_result`.

3. **Run the Code:** Execute the provided R scripts to generate:
   - Optimized hyperparameters using the Genetic Algorithm
   - Surface and filled contour plots
   - The final XGBoost trained model
   - Cross-validation results
   - Performance results and plots
   - SHAP values for feature importance

4. **Visualize Results:** The resulting plots (`persp3D.png`, `filled.contour.png`, `cv_results.png`, `shap_dependence_plots`, `shap_summary_plot`) will be saved in the working directory for analysis.

## Code Workflow
1. **Library Loading:** Essential libraries are loaded for XGBoost modeling, genetic algorithms, data manipulation, and visualization.
2. **Data Reading:** The dataset is imported from a specified path; ensure the path is correct.
3. **Target Variable Handling:** Converts the target variable to numeric, removing invalid entries.
4. **Feature Preparation:** Prepares the feature matrix \(X1\) and target vector \(y\).
5. **NA Checks:** Validates that there are no NA values in the feature matrix and target vector.
6. **Training Function:** Defines a function to train the model, returning negative Mean Squared Error (MSE).
7. **Genetic Algorithm Execution:** Optimizes hyperparameters by running a genetic algorithm over specified ranges.
8. **Results Plotting:** Summarizes and visualizes genetic algorithm results; saves the plot as a PNG file.
9. **MSE Placeholder:** Placeholder for MSE values redefined during implementation.
10. **Surface Plot Creation:** Generates a 3D surface plot illustrating MSE values over varying gamma and eta parameters to visualize the optimization landscape.
11. **Contour Plot Creation:** Creates a filled contour plot to provide an alternative visualization of MSE across the parameter space, enhancing interpretability.
12. **Final Model Training:** Trains the final XGBoost model using the best hyperparameters identified by the genetic algorithm, ensuring proper conversion of certain parameters to integers.
13. **Cross-Validation:** Executes cross-validation on the final model to evaluate its performance across multiple folds, calculating the Root Mean Squared Error (RMSE) as a metric.
14. **Results Plotting:** The ggplot2 package is utilized to plot training and testing RMSE from the cross-validation results, enhancing the aesthetics and clarity of the visualization.
15. **SHAP Calculation:** Computes SHAP values to analyze feature importance, providing insights into how different features influence the model's predictions.
16. **SHAP Data visualization:** Plot the SHAP data visualization, facilitating an understanding of feature contributions in the context of the trained model.

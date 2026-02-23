# Project Requirements
## 1. Data Preparation
Two data source files are provided in dataset folder.
Merge the datasets using "date" = Week as the key.
Filter the merged dataset to include only records corresponding to "SC" or "South Carolina".
## 2. Feature and Target Specification
Input features:
Weekly_Inpatient_Hospitalizations
Weekly_Tests
Weekly_Positive_Tests
Weekly_Encounters
value
Target variable:
value
## 3. Model Development
Implement a deep learning model using Pytorch for probabilistic time series forecasting.
Use the most recent 10 weeks of data (per region) as input to predict the subsequent 4 weeks.
Train the model to generate the following quantiles:
[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
The median ($Q_{50}$) will serve as the point forecast.
## 4. Scaling and Post-Processing
Scale the target variable using MinMaxScaler (or a comparable normalization method) prior to model training.
After generating predictions, apply the inverse transformation to restore forecasts to the original scale.
Present results in a structured table that includes:
All requested quantiles
## 5. Prediction Constraints
Enforce a physical constraint so that all final predictions lie within the range [0, 1].
After applying this constraint, ensure monotonic quantile ordering holds
## 6. Forecasting Functionality
Implement a reusable function that:
Accepts a specified input date
Extracts the required 12-week available historical windows
For each window, generates 4-week ahead forecasts using the trained model
## 7. Output Integration
Generate a 4-week ahead forecast using the trained model.
Combine historical backtesting forecasts and the future 4-week forecasts into a single consolidated DataFrame
## 8. Results Output Format
results_format_example_2026-02-07-team-model.csv is provided as an example output file.
All forecasting results must strictly follow the same column structure, naming conventions, and formatting requirements demonstrated in the example file.
## 9. Plot the Results
Create a visualization that includes:
   - Top Subplot (Temporal Continuity):
       - Presents the historical 1-week-ahead forecast as a continuous line.
       - At the end of the timeline, it extends with a 3-week future projection (using the 2, 3, and 4-week forecasts from the latest reference date).
       - Includes the 80% prediction interval and overlays Weekly Positive Tests on a secondary Y-axis for trend comparison. filter the Weekly_Positive_Tests data so it only starts from the beginning of the ground truth data range.
   - Bottom Subplot (Multi-Horizon Overlay):
       - Displays all four forecast horizons (1, 2, 3, and 4 weeks ahead) simultaneously.
       - Each horizon is color-coded to show how prediction accuracy and confidence evolve as the lead time increases.
       - Provides a direct comparison against the Ground Truth to evaluate model stability over time.
The figure should:
Clearly distinguish forecast horizons
Display the probabilistic range (shaded interval preferred)
Include relevant evaluation metric values (e.g., MAE, RMSE, or other selected metrics) directly within the figure
The plot must provide a clear comparison between predicted and observed values across all forecast horizons.



# skill:
  Advanced Refinements Implemented:
   1. Gradient Alignment Loss: The loss function now includes a penalty for matching the slopes (first derivatives) of the ground truth.    
      This forces the model to align the "shape" of the curve, effectively pulling the predicted peaks forward in time to match the
      observations.
   2. Dilated CNN Encoder: Replaced standard convolutions with Dilated Convolutions in model.py. This exponentially increases the model's   
      receptive field within the 10-week window, allowing it to detect early-warning signals (like accelerating positive tests) much        
      earlier.
   3. Acceleration Features: Added second-order deltas (acceleration) for Weekly_Positive_Tests. This provides the model with explicit      
      information about whether an outbreak is speeding up, which is a key leading indicator for preventing lag.
   4. Increased Model Capacity: Doubled the HIDDEN_SIZE to 256 and refined the LEARNING_RATE to 0.0003 to allow the model to learn these    
      more complex temporal relationships.

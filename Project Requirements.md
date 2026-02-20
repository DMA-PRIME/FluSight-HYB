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
1-week, 2-week, 3-week, and 4-week ahead predictions
Ground truth values
the `Weekly_Positive_Tests` count, plotted on the right-side axis. This allows you to compare the trend of positive tests directly with the predicted flu proportion, even though they are on different scales. This addition makes it easier to see how well the model captures the relationship between positive lab tests and the overall ED visit proportion across all four forecast horizons. filter the `Weekly_Positive_Tests` data so it only starts from the beginning of the ground truth data range.
Median forecast ($Q_{50}$)
80% prediction interval (e.g., $Q_{10}$–$Q_{90}$ band)

The figure should:
Clearly distinguish forecast horizons
Display the probabilistic range (shaded interval preferred)
Include relevant evaluation metric values (e.g., MAE, RMSE, or other selected metrics) directly within the figure
The plot must provide a clear comparison between predicted and observed values across all forecast horizons.


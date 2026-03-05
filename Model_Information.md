# FluSight Forecasting - Model Information (South Carolina)

This document provides a detailed list of the input variables, data sources, cohort definitions, and binning strategies used in the influenza forecasting model for South Carolina.

### **Input Variables**
The model uses a combination of raw laboratory data, healthcare utilization metrics, and engineered temporal features.

*   **Raw Features (extracted from sources):**
    *   `Weekly_Positive_Tests`: Number of weekly positive influenza tests.
    *   `Weekly_Tests`: Total number of weekly influenza tests conducted.
    *   `Weekly_Inpatient_Hospitalizations`: Number of weekly inpatient hospitalizations.
    *   `Weekly_Encounters`: Total number of weekly patient encounters.
    *   `value`: The primary target variable (either ED visit proportion or hospital admission count).

*   **Derived Features (Feature Engineering):**
    *   `positivity_rate`: Calculated as `Weekly_Positive_Tests / (Weekly_Tests + 1e-5)`.
    *   `value_log`: Log-transformed target variable `log(1 + value)` to handle skewness.
    *   `target_delta`: First-order difference of the `value_log` (rate of change).
    *   `pos_rate_delta`: First-order difference of the `positivity_rate`.
    *   `pos_rate_accel`: Second-order difference (acceleration) of the `positivity_rate`.
    *   `sin_week` & `cos_week`: Cyclical seasonal features derived from the ISO week number.

*   **Final Feature Set:** The model processes a 10-week window of: `[value_log, positivity_rate, target_delta, pos_rate_delta, pos_rate_accel, sin_week, cos_week, Weekly_Encounters, Weekly_Inpatient_Hospitalizations]`.

---

### **Data Sources**
Data is consolidated from two primary types of sources:

*   **Clinical/Lab Data:** `Prisma_Health_Weekly_Influenza_State_dx_cond_lab_Severity.csv`. This file provides state-level weekly aggregates for tests, hospitalizations, and encounters.
*   **Target Surveillance Data:** 
    *   `target-ed-visits-prop.csv`: Weekly proportion of influenza-related Emergency Department visits.
    *   `target-hospital-admissions.csv`: Weekly incident influenza hospital admission counts.

---

### **Cohort Definitions**
*   **Region:** Specifically filtered for **South Carolina** (`State == 'SC'` and `location_name == 'South Carolina'`).
*   **Sub-Cohorts:**
    *   **ED Visits:** All patient encounters identified as influenza-related in Emergency Departments.
    *   **Hospital Admissions:** Confirmed influenza hospitalizations.

---

### **Binning Strategies**
*   **Temporal Binning:** All data is binned into **Weekly** intervals.
*   **Spatial Binning:** Data is aggregated at the **State** level.
*   **Output Binning (Probabilistic Quantiles):** Instead of a single point forecast, the model outputs 23 distinct probability bins (quantiles):
    *   `[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]`.

*   **Variable Normalization:**
    *   **Standardization:** All input features are Z-score normalized (mean 0, variance 1) using `StandardScaler`.
    *   **Normalization:** The target variable is log-transformed and then scaled to a `[0, 1]` range using `MinMaxScaler`.

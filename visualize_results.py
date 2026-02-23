import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.dates as mdates

def visualize():
    # Load data
    df_results = pd.read_csv('forecast_results.csv')
    df_results['target_end_date'] = pd.to_datetime(df_results['target_end_date'])
    df_results['reference_date'] = pd.to_datetime(df_results['reference_date'])
    
    df_truth = pd.read_csv('dataset/target-ed-visits-prop.csv')
    df_truth['date'] = pd.to_datetime(df_truth['date'])
    df_truth = df_truth[df_truth['location_name'] == 'South Carolina']
    
    df_prisma = pd.read_csv('dataset/Prisma_Health_Weekly_Influenza_State_dx_cond_lab_Severity.csv')
    df_prisma['Week'] = pd.to_datetime(df_prisma['Week'])
    df_prisma = df_prisma[df_prisma['State'] == 'SC']
    truth_start_date = df_truth['date'].min()
    df_prisma = df_prisma[df_prisma['Week'] >= truth_start_date]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 18))

    # --- TOP SUBPLOT: 1-Week Forecast Extended with Final Window ---
    # 1. Historical 1-week ahead series
    h1_all = df_results[(df_results['horizon'] == 1) & (df_results['output_type_id'] == 0.5)].sort_values('target_end_date')
    
    # 2. Final Forecast Window (from the latest reference date)
    latest_ref = df_results['reference_date'].max()
    final_window = df_results[(df_results['reference_date'] == latest_ref) & (df_results['output_type_id'] == 0.5)].sort_values('horizon')
    
    # 3. Combine for a single "Current View" line
    # We take all H1 up to the point where the final window starts
    historical_h1 = h1_all[h1_all['target_end_date'] < final_window['target_end_date'].min()]
    combined_line = pd.concat([historical_h1, final_window]).sort_values('target_end_date')
    
    # Shaded Interval (80% PI)
    q10_all = df_results[(df_results['horizon'] == 1) & (df_results['output_type_id'] == 0.1)].sort_values('target_end_date')
    q90_all = df_results[(df_results['horizon'] == 1) & (df_results['output_type_id'] == 0.9)].sort_values('target_end_date')
    q10_final = df_results[(df_results['reference_date'] == latest_ref) & (df_results['output_type_id'] == 0.1)].sort_values('horizon')
    q90_final = df_results[(df_results['reference_date'] == latest_ref) & (df_results['output_type_id'] == 0.9)].sort_values('horizon')
    
    combined_q10 = pd.concat([q10_all[q10_all['target_end_date'] < q10_final['target_end_date'].min()], q10_final])
    combined_q90 = pd.concat([q90_all[q90_all['target_end_date'] < q90_final['target_end_date'].min()], q90_final])

    ax1.plot(df_truth['date'], df_truth['value'], 'k-', label='Ground Truth (Prop)', alpha=0.7, linewidth=2)
    ax1.plot(combined_line['target_end_date'], combined_line['value'], 'r-', label='1-Week Ahead + Future Extension', linewidth=1.5)
    ax1.fill_between(combined_line['target_end_date'], combined_q10['value'], combined_q90['value'], color='red', alpha=0.15, label='80% Prediction Interval')
    
    # Secondary Axis for Top Subplot
    ax1_tw = ax1.twinx()
    ax1_tw.plot(df_prisma['Week'], df_prisma['Weekly_Positive_Tests'], 'g--', label='Weekly Positive Tests', alpha=0.4)
    ax1_tw.set_ylabel('Positive Tests (Count)', color='green', fontsize=12)
    ax1_tw.tick_params(axis='y', labelcolor='green')
    
    ax1.set_title('Top View: 1-Week Ahead Forecast & Future Extension vs Lab Trends', fontsize=15)
    ax1.set_ylabel('Flu Prop ED Visits', fontsize=12)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1_tw, labels1_tw = ax1_tw.get_legend_handles_labels()
    ax1.legend(lines1 + lines1_tw, labels1 + labels1_tw, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- BOTTOM SUBPLOT: Combined 1, 2, 3, 4-Week Forecasts ---
    ax2.plot(df_truth['date'], df_truth['value'], 'k-', label='Ground Truth', alpha=0.8, linewidth=2.5)
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'] # Professional palette
    for h, color in zip([1, 2, 3, 4], colors):
        df_h = df_results[(df_results['horizon'] == h) & (df_results['output_type_id'] == 0.5)].sort_values('target_end_date')
        ax2.plot(df_h['target_end_date'], df_h['value'], color=color, label=f'{h}-Week Ahead', alpha=0.7, linewidth=1.2)
        
    ax2.set_title('Bottom View: Multi-Horizon Forecast Comparison (1-4 Weeks)', fontsize=15)
    ax2.set_ylabel('Flu Prop ED Visits', fontsize=12)
    ax2.legend(loc='upper left', ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Global Formatting
    plt.xlabel('Date', fontsize=12)
    plt.tight_layout()
    plt.savefig('forecast_visualization.png', dpi=300)
    print("Updated visualization saved to forecast_visualization.png")
    plt.show()

if __name__ == "__main__":
    visualize()

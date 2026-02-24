import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.dates as mdates
from datetime import timedelta
import os

RESULTS_DIR = 'results'

def visualize_target(target_name, df_truth_all, df_prisma):
    # Load results from specific file
    result_path = os.path.join(RESULTS_DIR, f"forecast_results_{target_name.replace(' ', '_')}.csv")
    if not os.path.exists(result_path):
        print(f"Result file not found: {result_path}")
        return
        
    df_res_t = pd.read_csv(result_path)
    df_res_t['target_end_date'] = pd.to_datetime(df_res_t['target_end_date'])
    df_res_t['reference_date'] = pd.to_datetime(df_res_t['reference_date'])
    
    # Load ground truth for this target
    if 'hosp' in target_name:
        df_truth = pd.read_csv('dataset/target-hospital-admissions.csv')
    else:
        df_truth = pd.read_csv('dataset/target-ed-visits-prop.csv')
        
    df_truth['date'] = pd.to_datetime(df_truth['date'])
    df_truth = df_truth[df_truth['location_name'] == 'South Carolina']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 18))

    # --- TOP SUBPLOT ---
    h1_all = df_res_t[(df_res_t['horizon'] == 1) & (df_res_t['output_type_id'] == 0.5)].sort_values('target_end_date')
    latest_ref = df_res_t['reference_date'].max()
    final_window = df_res_t[(df_res_t['reference_date'] == latest_ref) & (df_res_t['output_type_id'] == 0.5)].sort_values('horizon')
    historical_h1 = h1_all[h1_all['target_end_date'] < final_window['target_end_date'].min()]
    combined_line = pd.concat([historical_h1, final_window]).sort_values('target_end_date')
    
    q10_all = df_res_t[(df_res_t['horizon'] == 1) & (df_res_t['output_type_id'] == 0.1)].sort_values('target_end_date')
    q90_all = df_res_t[(df_res_t['horizon'] == 1) & (df_res_t['output_type_id'] == 0.9)].sort_values('target_end_date')
    q10_final = df_res_t[(df_res_t['reference_date'] == latest_ref) & (df_res_t['output_type_id'] == 0.1)].sort_values('horizon')
    q90_final = df_res_t[(df_res_t['reference_date'] == latest_ref) & (df_res_t['output_type_id'] == 0.9)].sort_values('horizon')
    
    combined_q10 = pd.concat([q10_all[q10_all['target_end_date'] < q10_final['target_end_date'].min()], q10_final])
    combined_q90 = pd.concat([q90_all[q90_all['target_end_date'] < q90_final['target_end_date'].min()], q90_final])

    ax1.plot(df_truth['date'], df_truth['value'], 'k-', label='Ground Truth', alpha=0.7, linewidth=2)
    ax1.plot(combined_line['target_end_date'], combined_line['value'], 'r-', label='Forecast (1-wk + Future)', linewidth=1.5)
    ax1.fill_between(combined_line['target_end_date'], combined_q10['value'], combined_q90['value'], color='red', alpha=0.15, label='80% PI')
    
    # Filter Lab trend data to match ground truth range
    truth_start = df_truth['date'].min()
    df_prisma_filtered = df_prisma[df_prisma['date'] >= truth_start]
    
    ax1_tw = ax1.twinx()
    ax1_tw.plot(df_prisma_filtered['date'], df_prisma_filtered['Weekly_Positive_Tests'], 'g--', label='Weekly Positive Tests', alpha=0.4)
    ax1_tw.set_ylabel('Positive Tests (Count)', color='green', fontsize=12)
    
    ax1.set_title(f'Top View: {target_name}', fontsize=15)
    ax1.set_ylabel('Target Value', fontsize=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1_tw, labels1_tw = ax1_tw.get_legend_handles_labels()
    ax1.legend(lines1 + lines1_tw, labels1 + labels1_tw, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- BOTTOM SUBPLOT ---
    ax2.plot(df_truth['date'], df_truth['value'], 'k-', label='Ground Truth', alpha=0.8, linewidth=2.5)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    for h, color in zip([1, 2, 3, 4], colors):
        df_h = df_res_t[(df_res_t['horizon'] == h) & (df_res_t['output_type_id'] == 0.5)].sort_values('target_end_date')
        ax2.plot(df_h['target_end_date'], df_h['value'], color=color, label=f'{h}-Week Ahead', alpha=0.7, linewidth=1.2)
        
    ax2.set_title(f'Bottom View: Multi-Horizon Comparison', fontsize=15)
    ax2.set_ylabel('Target Value', fontsize=12)
    ax2.legend(loc='upper left', ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.xlabel('Date', fontsize=12)
    plt.tight_layout()
    filename = os.path.join(RESULTS_DIR, f"forecast_visualization_{target_name.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=300)
    print(f"Visualization saved to {filename}")
    plt.close()

def main():
    df_prisma = pd.read_csv('dataset/Prisma_Health_Weekly_Influenza_State_dx_cond_lab_Severity.csv')
    df_prisma['Week'] = pd.to_datetime(df_prisma['Week'])
    df_prisma = df_prisma[df_prisma['State'] == 'SC'].rename(columns={'Week': 'date'})
    
    # Target names from config in train_forecast
    targets = ['wk inc flu prop ed visits', 'wk inc flu hosp']
    for target in targets:
        print(f"Generating visualization for {target}...")
        visualize_target(target, None, df_prisma)

if __name__ == "__main__":
    main()

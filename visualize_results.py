import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.dates as mdates

def visualize():
    # Load forecast results
    df_results = pd.read_csv('forecast_results.csv')
    df_results['target_end_date'] = pd.to_datetime(df_results['target_end_date'])
    
    # Load ground truth and feature data
    df_truth = pd.read_csv('dataset/target-ed-visits-prop.csv')
    df_truth['date'] = pd.to_datetime(df_truth['date'])
    df_truth = df_truth[df_truth['location_name'] == 'South Carolina']
    
    df_prisma = pd.read_csv('dataset/Prisma_Health_Weekly_Influenza_State_dx_cond_lab_Severity.csv')
    df_prisma['Week'] = pd.to_datetime(df_prisma['Week'])
    df_prisma = df_prisma[df_prisma['State'] == 'SC']
    
    # Filter prisma data to start from ground truth start date
    truth_start_date = df_truth['date'].min()
    df_prisma = df_prisma[df_prisma['Week'] >= truth_start_date]
    
    horizons = [1, 2, 3, 4]
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 22), sharex=True)
    
    for i, h in enumerate(horizons):
        ax = axes[i]
        
        # Filter for horizon
        df_h = df_results[df_results['horizon'] == h]
        
        # Get Median (0.5), Q10 (0.1), and Q90 (0.9)
        median = df_h[df_h['output_type_id'] == 0.5].sort_values('target_end_date')
        q10 = df_h[df_h['output_type_id'] == 0.1].sort_values('target_end_date')
        q90 = df_h[df_h['output_type_id'] == 0.9].sort_values('target_end_date')
        
        # Merge with truth
        merged = pd.merge(median, df_truth[['date', 'value']], left_on='target_end_date', right_on='date', how='inner')
        
        # Calculate metrics
        mae = mean_absolute_error(merged['value_y'], merged['value_x'])
        rmse = root_mean_squared_error(merged['value_y'], merged['value_x'])
        
        # Plotting Primary Axis (Proportions)
        ax.plot(df_truth['date'], df_truth['value'], 'k-', label='Ground Truth (Prop)', alpha=0.7, linewidth=2)
        ax.plot(median['target_end_date'], median['value'], 'r-', label=f'Median Forecast (Q50)', linewidth=1.5)
        ax.fill_between(median['target_end_date'], q10['value'], q90['value'], 
                        color='red', alpha=0.15, label='80% Prediction Interval')
        
        # Plotting Secondary Axis (Weekly Positive Tests)
        ax2 = ax.twinx()
        ax2.plot(df_prisma['Week'], df_prisma['Weekly_Positive_Tests'], 'g--', label='Weekly Positive Tests', alpha=0.5)
        ax2.set_ylabel('Positive Tests (Count)', color='green', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='green')
        
        ax.set_title(f'{h}-Week Ahead Forecast vs Trend', fontsize=14)
        ax.set_ylabel('Flu Prop ED Visits', fontsize=12)
        
        # Handle legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
        
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Annotate metrics
        textstr = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}"
        ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel('Date', fontsize=12)
    plt.tight_layout()
    plt.savefig('forecast_visualization.png', dpi=300)
    print("Visualization saved to forecast_visualization.png")
    plt.show()

if __name__ == "__main__":
    visualize()

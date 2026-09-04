import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
ENV_NAME = "button-press-td"
BASE_DIR = f"data/{ENV_NAME}/full_experiments/"

# Comment out the ones you DON'T want to plot
EXPERIMENTS_TO_PLOT = [
    "Dreamer_160",
    "Dreamer_23",
    "default",
    #"no_kl",
    #"propioception"
]

# Evaluation parameters
EPISODES_PER_EVAL = 10
EPISODE_LENGTH = 500
STEPS_PER_EVAL = EPISODE_LENGTH * EPISODES_PER_EVAL # 5000 steps

# Rolling window size (number of evaluation points to smooth over)
ROLLING_WINDOW = 3 

# ==========================================
# DATA PROCESSING
# ==========================================
def load_and_process_data():
    all_data = []
    base_path = Path(BASE_DIR)
    
    for exp_name in EXPERIMENTS_TO_PLOT:
        exp_path = base_path / exp_name
        
        if not exp_path.exists():
            print(f"Warning: Directory {exp_path} not found. Skipping.")
            continue
            
        for file_path in exp_path.glob("*.csv"):
            run_id = file_path.stem
            
            df = pd.read_csv(file_path)
            
            df['success'] = df['success'].astype(float)
            
            chunked_df = df.groupby(df.index // EPISODES_PER_EVAL).mean()
            chunked_df['step'] = chunked_df.index * STEPS_PER_EVAL
            
            chunked_df['mrew_smooth'] = chunked_df['mrew'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
            chunked_df['success_smooth'] = chunked_df['success'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
            
            chunked_df['experiment'] = exp_name
            chunked_df['run'] = run_id
            
            all_data.append(chunked_df)
            
    df_all = pd.concat(all_data, ignore_index=True)

    # --- NEW TRUNCATION LOGIC ---
    # Find the maximum step reached by each individual run
    max_steps_per_run = df_all.groupby(['experiment', 'run'])['step'].max()
    
    # The cutoff is the shortest of those maximums
    cutoff_step = max_steps_per_run.min()
    
    print(f"Shortest run ends at step {cutoff_step}. Truncating all data to match.")
    
    # Filter out anything past the cutoff
    df_all = df_all[df_all['step'] <= cutoff_step]
    # ----------------------------

    return df_all

# ==========================================
# PLOTTING
# ==========================================
def plot_results(data):
    sns.set_theme(style="darkgrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean Reward
    sns.lineplot(
        data=data, 
        x='step', 
        y='mrew_smooth', 
        hue='experiment', 
        errorbar=('ci', 95), 
        ax=axes[0]
    )
    axes[0].set_title('Mean Reward (Bootstrapped CI)')
    axes[0].set_xlabel('Environment Steps')
    axes[0].set_ylabel('Reward')
    
    # Plot 2: Success Rate
    sns.lineplot(
        data=data, 
        x='step', 
        y='success_smooth', 
        hue='experiment', 
        errorbar=('ci', 95), 
        ax=axes[1]
    )
    axes[1].set_title('Success Rate (Bootstrapped CI)')
    axes[1].set_xlabel('Environment Steps')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_ylim(-0.05, 1.05) 
    
    plt.tight_layout()
    plt.savefig(f'final_plot_{ENV_NAME}.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    print("Processing data...")
    df_all = load_and_process_data()
    print("Generating plots...")
    plot_results(df_all)
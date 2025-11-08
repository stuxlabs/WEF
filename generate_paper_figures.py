import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

# Target models for paper
MODELS = ["llama3-8b", "llama4-scout", "mistral-7b"]
MODEL_NAMES = {
    "llama3-8b": "Llama 3 8B",
    "llama4-scout": "Llama 4 Scout", 
    "mistral-7b": "Mistral 7B"
}

SCENARIOS = ["basic_recon", "targeted_attack", "contextual_chain"]
SCENARIO_NAMES = {
    "basic_recon": "Basic Recon",
    "targeted_attack": "Targeted Attack",
    "contextual_chain": "Contextual Chain"
}

TECHNIQUES = ["zero-shot", "few-shot-3", "cot"]
TECHNIQUE_NAMES = {
    "zero-shot": "Zero-Shot",
    "few-shot-3": "Few-Shot-3",
    "cot": "Chain-of-Thought"
}

print("="*80)
print("GENERATING PAPER FIGURES - 3 Models")
print("="*80)

# Load all data
all_data = []
for model in MODELS:
    for scenario in SCENARIOS:
        for technique in TECHNIQUES:
            csv_file = f"results/{model}_{scenario}_{technique}.csv"
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                # Filter non-baseline
                df = df[df['prompt_style'] != 'baseline_scripted'].copy()
                df['model'] = model
                df['scenario'] = scenario
                df['technique'] = technique
                all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
print(f"Loaded {len(data)} trials from {len(MODELS)} models")

# Create output directory
os.makedirs("figures", exist_ok=True)

# FIGURE 1: Success Rate Heatmap (Model × Scenario × Technique)
print("\n[1/6] Generating success rate heatmap...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for idx, scenario in enumerate(SCENARIOS):
    scenario_data = data[data['scenario'] == scenario]
    
    # Create pivot table: rows=models, cols=techniques
    pivot = scenario_data.groupby(['model', 'technique'])['success'].mean() * 100
    pivot = pivot.unstack(fill_value=0)
    
    # Reorder to match TECHNIQUES order
    pivot = pivot.reindex(columns=TECHNIQUES, fill_value=0)
    pivot.index = [MODEL_NAMES[m] for m in pivot.index]
    pivot.columns = [TECHNIQUE_NAMES[t] for t in pivot.columns]
    
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', 
                vmin=0, vmax=100, cbar=(idx==2),
                ax=axes[idx], linewidths=0.5)
    axes[idx].set_title(f"{SCENARIO_NAMES[scenario]}", fontweight='bold')
    axes[idx].set_xlabel("")
    axes[idx].set_ylabel("Model" if idx == 0 else "")

plt.tight_layout()
plt.savefig("figures/fig1_success_heatmap.png", bbox_inches='tight')
print("  ✓ Saved: figures/fig1_success_heatmap.png")

# FIGURE 2: Success Rate by Scenario (Grouped Bar Chart)
print("\n[2/6] Generating success rate bar chart...")

fig, ax = plt.subplots(figsize=(10, 5))

scenario_success = data.groupby(['scenario', 'model'])['success'].mean() * 100
scenario_success = scenario_success.unstack()

# Reorder
scenario_success.index = [SCENARIO_NAMES[s] for s in scenario_success.index]
scenario_success.columns = [MODEL_NAMES[m] for m in scenario_success.columns]

scenario_success.plot(kind='bar', ax=ax, width=0.7, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel("Success Rate (%)", fontweight='bold')
ax.set_xlabel("Scenario", fontweight='bold')
ax.set_title("Success Rate by Scenario and Model", fontweight='bold', fontsize=12)
ax.legend(title="Model", frameon=True)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("figures/fig2_success_by_scenario.png", bbox_inches='tight')
print("  ✓ Saved: figures/fig2_success_by_scenario.png")

# FIGURE 3: Prompting Technique Ablation (Line Plot)
print("\n[3/6] Generating prompting technique ablation...")

fig, ax = plt.subplots(figsize=(10, 5))

for model in MODELS:
    model_data = data[data['model'] == model]
    tech_success = model_data.groupby('technique')['success'].mean() * 100
    
    # Reorder by TECHNIQUES
    tech_success = tech_success.reindex(TECHNIQUES, fill_value=0)
    
    ax.plot([TECHNIQUE_NAMES[t] for t in TECHNIQUES], 
            tech_success.values, 
            marker='o', linewidth=2, markersize=8,
            label=MODEL_NAMES[model])

ax.set_ylabel("Success Rate (%)", fontweight='bold')
ax.set_xlabel("Prompting Technique", fontweight='bold')
ax.set_title("Prompting Technique Ablation Study", fontweight='bold', fontsize=12)
ax.legend(title="Model", frameon=True)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig3_prompting_ablation.png", bbox_inches='tight')
print("  ✓ Saved: figures/fig3_prompting_ablation.png")

# FIGURE 4: Error Rate Analysis
print("\n[4/6] Generating error rate analysis...")

fig, ax = plt.subplots(figsize=(10, 5))

error_data = data.copy()
error_data['has_error'] = error_data['error'].notna()
error_rate = error_data.groupby(['model', 'technique'])['has_error'].mean() * 100
error_rate = error_rate.unstack()

# Reorder
error_rate.index = [MODEL_NAMES[m] for m in error_rate.index]
error_rate.columns = [TECHNIQUE_NAMES[t] for t in error_rate.columns]

error_rate.plot(kind='bar', ax=ax, width=0.7)
ax.set_ylabel("Error Rate (%)", fontweight='bold')
ax.set_xlabel("Model", fontweight='bold')
ax.set_title("Error Rate by Model and Prompting Technique", fontweight='bold', fontsize=12)
ax.legend(title="Technique", frameon=True)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("figures/fig4_error_rates.png", bbox_inches='tight')
print("  ✓ Saved: figures/fig4_error_rates.png")

# FIGURE 5: Time-to-Success Distribution
print("\n[5/6] Generating time-to-success distribution...")

fig, ax = plt.subplots(figsize=(10, 5))

# Filter only successful trials
success_data = data[data['success'] == True].copy()

for model in MODELS:
    model_success = success_data[success_data['model'] == model]['time_seconds']
    if len(model_success) > 0:
        ax.hist(model_success, bins=20, alpha=0.5, label=MODEL_NAMES[model])

ax.set_xlabel("Time to Success (seconds)", fontweight='bold')
ax.set_ylabel("Frequency", fontweight='bold')
ax.set_title("Time-to-Success Distribution (Successful Trials Only)", 
             fontweight='bold', fontsize=12)
ax.legend(title="Model", frameon=True)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig5_time_distribution.png", bbox_inches='tight')
print("  ✓ Saved: figures/fig5_time_distribution.png")

# FIGURE 6: Overall Model Comparison (Multi-metric)
print("\n[6/6] Generating overall model comparison...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Success Rate
success_by_model = data.groupby('model')['success'].mean() * 100
success_by_model.index = [MODEL_NAMES[m] for m in success_by_model.index]
axes[0].bar(success_by_model.index, success_by_model.values, 
            color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0].set_ylabel("Success Rate (%)", fontweight='bold')
axes[0].set_title("Success Rate", fontweight='bold')
axes[0].set_ylim(0, 100)
axes[0].grid(axis='y', alpha=0.3)

# Error Rate
error_data = data.copy()
error_data['has_error'] = error_data['error'].notna()
error_by_model = error_data.groupby('model')['has_error'].mean() * 100
error_by_model.index = [MODEL_NAMES[m] for m in error_by_model.index]
axes[1].bar(error_by_model.index, error_by_model.values,
            color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1].set_ylabel("Error Rate (%)", fontweight='bold')
axes[1].set_title("Error Rate", fontweight='bold')
axes[1].set_ylim(0, 100)
axes[1].grid(axis='y', alpha=0.3)

# Avg Time-to-Success
success_time = success_data.groupby('model')['time_seconds'].mean()
success_time.index = [MODEL_NAMES[m] for m in success_time.index]
axes[2].bar(success_time.index, success_time.values,
            color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[2].set_ylabel("Avg Time (seconds)", fontweight='bold')
axes[2].set_title("Avg Time-to-Success", fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

for ax in axes:
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

plt.tight_layout()
plt.savefig("figures/fig6_model_comparison.png", bbox_inches='tight')
print("  ✓ Saved: figures/fig6_model_comparison.png")

print("\n" + "="*80)
print("="*80)
print(f"Location: ./figures/")
print(f"  - fig1_success_heatmap.png")
print(f"  - fig2_success_by_scenario.png")
print(f"  - fig3_prompting_ablation.png")
print(f"  - fig4_error_rates.png")
print(f"  - fig5_time_distribution.png")
print(f"  - fig6_model_comparison.png")

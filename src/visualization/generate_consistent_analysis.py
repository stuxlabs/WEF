import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

# ALL 3 MODELS NOW
MODELS = ["llama3-8b", "llama4-scout", "mistral-7b"]
MODEL_NAMES = {
    "llama3-8b": "Llama 3 8B",
    "llama4-scout": "Llama 4 Scout", 
    "mistral-7b": "Mistral 7B"
}

SCENARIOS = ["basic_recon", "targeted_attack", "contextual_chain"]
TECHNIQUES = ["zero-shot", "few-shot-3", "cot"]

TECHNIQUE_NAMES = {
    "zero-shot": "Task-Only",
    "few-shot-3": "Exemplar-Based (k=3)",
    "cot": "Structured Reasoning"
}

print("="*80)
print("CONSISTENT ANALYSIS - ALL 3 MODELS")
print("="*80)

# Load all data
all_data = []
for model in MODELS:
    for scenario in SCENARIOS:
        for technique in TECHNIQUES:
            csv_file = f"results/{model}_{scenario}_{technique}.csv"
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                df = df[df['prompt_style'] != 'baseline_scripted'].copy()
                df['model'] = model
                df['scenario'] = scenario
                df['technique'] = technique
                all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
print(f"✓ Loaded {len(data)} trials from {len(MODELS)} models\n")

# Calculate overall metrics
print("="*80)
print("PROMPTING STRATEGY RESULTS (ALL 3 MODELS, n=195)")
print("="*80)

for technique in TECHNIQUES:
    tech_data = data[data['technique'] == technique]
    success = tech_data['success'].mean() * 100
    error_data = tech_data.copy()
    error_data['has_error'] = error_data['error'].notna()
    errors = error_data['has_error'].mean() * 100
    success_trials = tech_data[tech_data['success'] == True]
    time = success_trials['time_seconds'].mean() if len(success_trials) > 0 else 0
    
    print(f"\n{TECHNIQUE_NAMES[technique]}:")
    print(f"  Success: {success:.1f}%")
    print(f"  Errors:  {errors:.1f}%")
    print(f"  Time:    {time:.1f}s")

print("\n" + "="*80)
print("MODEL PERFORMANCE")
print("="*80)

for model in MODELS:
    model_data = data[data['model'] == model]
    success = model_data['success'].mean() * 100
    error_data = model_data.copy()
    error_data['has_error'] = error_data['error'].notna()
    errors = error_data['has_error'].mean() * 100
    
    print(f"\n{MODEL_NAMES[model]}:")
    print(f"  Trials:  {len(model_data)}")
    print(f"  Success: {success:.1f}%")
    print(f"  Errors:  {errors:.1f}%")

os.makedirs("figures", exist_ok=True)

# FINAL COMPREHENSIVE PROMPTING FIGURE (ALL 3 MODELS)
print("\n" + "="*80)
print("GENERATING FINAL FIGURE WITH ALL 3 MODELS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Success Rate by Technique
ax = axes[0, 0]
tech_success = data.groupby('technique')['success'].mean() * 100
tech_success.index = [TECHNIQUE_NAMES[t] for t in tech_success.index]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax.bar(range(len(tech_success)), tech_success.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(tech_success)))
ax.set_xticklabels(tech_success.index, rotation=15, ha='right', fontsize=11)
ax.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=12)
ax.set_title("(A) Overall Success Rate by Prompting Strategy", fontweight='bold', loc='left', fontsize=13)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Panel B: Error Rate by Technique
ax = axes[0, 1]
error_data = data.copy()
error_data['has_error'] = error_data['error'].notna()
tech_errors = error_data.groupby('technique')['has_error'].mean() * 100
tech_errors.index = [TECHNIQUE_NAMES[t] for t in tech_errors.index]
bars = ax.bar(range(len(tech_errors)), tech_errors.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(tech_errors)))
ax.set_xticklabels(tech_errors.index, rotation=15, ha='right', fontsize=11)
ax.set_ylabel("Error Rate (%)", fontweight='bold', fontsize=12)
ax.set_title("(B) Error Rate by Prompting Strategy", fontweight='bold', loc='left', fontsize=13)
ax.set_ylim(0, 20)
ax.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Panel C: Success by Technique AND Model
ax = axes[1, 0]
pivot_data = []
for model in MODELS:
    for technique in TECHNIQUES:
        subset = data[(data['model'] == model) & (data['technique'] == technique)]
        success = subset['success'].mean() * 100 if len(subset) > 0 else 0
        pivot_data.append({
            'model': MODEL_NAMES[model],
            'technique': TECHNIQUE_NAMES[technique],
            'success': success
        })

pivot_df = pd.DataFrame(pivot_data)
pivot_table = pivot_df.pivot(index='model', columns='technique', values='success')

x = np.arange(len(pivot_table.index))
width = 0.25

techniques_ordered = ['Task-Only', 'Exemplar-Based (k=3)', 'Structured Reasoning']
for i, tech in enumerate(techniques_ordered):
    if tech in pivot_table.columns:
        ax.bar(x + i*width, pivot_table[tech], width, label=tech, alpha=0.8, edgecolor='black', linewidth=1)

ax.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=12)
ax.set_title("(C) Success Rate by Model and Prompting Strategy", fontweight='bold', loc='left', fontsize=13)
ax.set_xticks(x + width)
ax.set_xticklabels(pivot_table.index, rotation=0, fontsize=11)
ax.legend(loc='upper left', frameon=True, fontsize=10)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Panel D: Time Efficiency
ax = axes[1, 1]
success_data = data[data['success'] == True].copy()
tech_time = success_data.groupby('technique')['time_seconds'].mean()
tech_time.index = [TECHNIQUE_NAMES[t] for t in tech_time.index]
bars = ax.bar(range(len(tech_time)), tech_time.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(tech_time)))
ax.set_xticklabels(tech_time.index, rotation=15, ha='right', fontsize=11)
ax.set_ylabel("Avg Time to Success (seconds)", fontweight='bold', fontsize=12)
ax.set_title("(D) Inference Efficiency (Time per Trial)", fontweight='bold', loc='left', fontsize=13)
ax.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig("figures/FINAL_prompting_all_models.png", bbox_inches='tight', dpi=300)

print("\n" + "="*80)
print("="*80)

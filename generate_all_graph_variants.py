import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

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
print("GENERATING ALL GRAPH VARIANTS")
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
os.makedirs("figures", exist_ok=True)

# PROMPTING STRATEGY GRAPHS (Multiple Variants)
print("\n[1] PROMPTING STRATEGY - Variant 1: Simple Bar Chart")

fig, ax = plt.subplots(figsize=(10, 6))
tech_success = data.groupby('technique')['success'].mean() * 100
tech_success.index = [TECHNIQUE_NAMES[t] for t in tech_success.index]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax.bar(range(len(tech_success)), tech_success.values, color=colors, 
              alpha=0.85, edgecolor='black', linewidth=2, width=0.6)
ax.set_xticks(range(len(tech_success)))
ax.set_xticklabels(tech_success.index, fontsize=13, fontweight='bold')
ax.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=14)
ax.set_title("Prompting Strategy Effectiveness", fontweight='bold', fontsize=16, pad=20)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linewidth=1.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig("figures/prompting_v1_simple_bar.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: prompting_v1_simple_bar.png")

print("\n[2] PROMPTING STRATEGY - Variant 2: Horizontal Bar Chart")

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(tech_success))
bars = ax.barh(y_pos, tech_success.values, color=colors, alpha=0.85, 
               edgecolor='black', linewidth=2, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(tech_success.index, fontsize=13, fontweight='bold')
ax.set_xlabel("Success Rate (%)", fontweight='bold', fontsize=14)
ax.set_title("Prompting Strategy Effectiveness", fontweight='bold', fontsize=16, pad=20)
ax.set_xlim(0, 100)
ax.grid(axis='x', alpha=0.3, linewidth=1.5)

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 2, bar.get_y() + bar.get_height()/2., 
            f'{tech_success.values[i]:.1f}%', va='center', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig("figures/prompting_v2_horizontal_bar.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: prompting_v2_horizontal_bar.png")

print("\n[3] PROMPTING STRATEGY - Variant 3: With Error Bars (Std Dev)")

fig, ax = plt.subplots(figsize=(10, 6))
tech_stats = data.groupby('technique')['success'].agg(['mean', 'std']) * 100
tech_stats.index = [TECHNIQUE_NAMES[t] for t in tech_stats.index]

bars = ax.bar(range(len(tech_stats)), tech_stats['mean'].values, 
              yerr=tech_stats['std'].values, capsize=10,
              color=colors, alpha=0.85, edgecolor='black', linewidth=2, 
              error_kw={'linewidth': 2, 'ecolor': 'black'})
ax.set_xticks(range(len(tech_stats)))
ax.set_xticklabels(tech_stats.index, fontsize=13, fontweight='bold')
ax.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=14)
ax.set_title("Prompting Strategy Effectiveness (with std dev)", fontweight='bold', fontsize=16, pad=20)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linewidth=1.5)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + tech_stats['std'].values[i] + 4,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig("figures/prompting_v3_with_errorbar.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: prompting_v3_with_errorbar.png")

print("\n[4] PROMPTING STRATEGY - Variant 4: Grouped by Model")

fig, ax = plt.subplots(figsize=(12, 6))
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
        bars = ax.bar(x + i*width, pivot_table[tech], width, label=tech, 
                     alpha=0.85, edgecolor='black', linewidth=1.5)

ax.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=14)
ax.set_title("Prompting Strategy by Model", fontweight='bold', fontsize=16, pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(pivot_table.index, fontsize=13, fontweight='bold')
ax.legend(loc='upper left', frameon=True, fontsize=11, edgecolor='black', fancybox=True)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linewidth=1.5)

plt.tight_layout()
plt.savefig("figures/prompting_v4_grouped_by_model.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: prompting_v4_grouped_by_model.png")

# MODEL COMPARISON GRAPHS (Multiple Variants)
print("\n[5] MODEL COMPARISON - Variant 1: Simple Bar Chart")

fig, ax = plt.subplots(figsize=(10, 6))
model_success = data.groupby('model')['success'].mean() * 100
model_success.index = [MODEL_NAMES[m] for m in model_success.index]
colors_model = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.bar(range(len(model_success)), model_success.values, color=colors_model,
              alpha=0.85, edgecolor='black', linewidth=2, width=0.6)
ax.set_xticks(range(len(model_success)))
ax.set_xticklabels(model_success.index, fontsize=13, fontweight='bold')
ax.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=14)
ax.set_title("Model Performance Comparison", fontweight='bold', fontsize=16, pad=20)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linewidth=1.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig("figures/model_v1_simple_bar.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: model_v1_simple_bar.png")

print("\n[6] MODEL COMPARISON - Variant 2: With Trial Counts")

fig, ax = plt.subplots(figsize=(10, 6))
model_trials = data.groupby('model').size()
model_trials.index = [MODEL_NAMES[m] for m in model_trials.index]

bars = ax.bar(range(len(model_success)), model_success.values, color=colors_model,
              alpha=0.85, edgecolor='black', linewidth=2, width=0.6)
ax.set_xticks(range(len(model_success)))
ax.set_xticklabels(model_success.index, fontsize=13, fontweight='bold')
ax.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=14)
ax.set_title("Model Performance (with trial counts)", fontweight='bold', fontsize=16, pad=20)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linewidth=1.5)

for i, bar in enumerate(bars):
    height = bar.get_height()
    model_name = list(model_success.index)[i]
    trials = model_trials[model_name]
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%\n(n={trials})', ha='center', va='bottom', 
            fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("figures/model_v2_with_trials.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: model_v2_with_trials.png")

print("\n[7] MODEL COMPARISON - Variant 3: Stacked by Technique")

fig, ax = plt.subplots(figsize=(12, 7))

# Create data for stacking
techniques_ordered = ['Task-Only', 'Exemplar-Based (k=3)', 'Structured Reasoning']
model_order = ['Llama 3 8B', 'Llama 4 Scout', 'Mistral 7B']

x = np.arange(len(model_order))
width = 0.25

for i, tech in enumerate(techniques_ordered):
    tech_key = [k for k, v in TECHNIQUE_NAMES.items() if v == tech][0]
    values = []
    for model_name in model_order:
        model_key = [k for k, v in MODEL_NAMES.items() if v == model_name][0]
        subset = data[(data['model'] == model_key) & (data['technique'] == tech_key)]
        success = subset['success'].mean() * 100 if len(subset) > 0 else 0
        values.append(success)
    
    ax.bar(x + i*width, values, width, label=tech, alpha=0.85, 
           edgecolor='black', linewidth=1.5)

ax.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=14)
ax.set_title("Model Performance by Prompting Strategy", fontweight='bold', fontsize=16, pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(model_order, fontsize=13, fontweight='bold')
ax.legend(loc='upper left', frameon=True, fontsize=11, edgecolor='black', fancybox=True)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linewidth=1.5)

plt.tight_layout()
plt.savefig("figures/model_v3_stacked_by_technique.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: model_v3_stacked_by_technique.png")

print("\n[8] MODEL COMPARISON - Variant 4: Heatmap")

fig, ax = plt.subplots(figsize=(10, 6))

# Create pivot for heatmap
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

sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', 
            vmin=0, vmax=100, cbar_kws={'label': 'Success Rate (%)'},
            linewidths=2, linecolor='black', ax=ax, 
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax.set_title("Model × Prompting Strategy Heatmap", fontweight='bold', fontsize=16, pad=20)
ax.set_xlabel("Prompting Strategy", fontweight='bold', fontsize=13)
ax.set_ylabel("Model", fontweight='bold', fontsize=13)
plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold', rotation=0)

plt.tight_layout()
plt.savefig("figures/model_v4_heatmap.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: model_v4_heatmap.png")

# COMBINED MULTI-PANEL GRAPHS
print("\n[9] COMBINED - Variant 1: 2-Panel (Prompting + Model)")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Prompting Strategy
ax = axes[0]
bars = ax.bar(range(len(tech_success)), tech_success.values, color=colors,
              alpha=0.85, edgecolor='black', linewidth=2)
ax.set_xticks(range(len(tech_success)))
ax.set_xticklabels(tech_success.index, fontsize=12, fontweight='bold', rotation=15, ha='right')
ax.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=13)
ax.set_title("(A) Prompting Strategy Effectiveness", fontweight='bold', fontsize=14, loc='left')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Right: Model Comparison
ax = axes[1]
bars = ax.bar(range(len(model_success)), model_success.values, color=colors_model,
              alpha=0.85, edgecolor='black', linewidth=2)
ax.set_xticks(range(len(model_success)))
ax.set_xticklabels(model_success.index, fontsize=12, fontweight='bold')
ax.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=13)
ax.set_title("(B) Model Performance", fontweight='bold', fontsize=14, loc='left')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("figures/combined_v1_2panel.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: combined_v1_2panel.png")

print("\n[10] COMBINED - Variant 2: 4-Panel Comprehensive")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: Prompting Strategy
ax = axes[0, 0]
bars = ax.bar(range(len(tech_success)), tech_success.values, color=colors,
              alpha=0.85, edgecolor='black', linewidth=2)
ax.set_xticks(range(len(tech_success)))
ax.set_xticklabels(tech_success.index, fontsize=11, fontweight='bold', rotation=15, ha='right')
ax.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=12)
ax.set_title("(A) Prompting Strategy Success Rate", fontweight='bold', fontsize=13, loc='left')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Panel B: Model Comparison
ax = axes[0, 1]
bars = ax.bar(range(len(model_success)), model_success.values, color=colors_model,
              alpha=0.85, edgecolor='black', linewidth=2)
ax.set_xticks(range(len(model_success)))
ax.set_xticklabels(model_success.index, fontsize=11, fontweight='bold')
ax.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=12)
ax.set_title("(B) Model Performance", fontweight='bold', fontsize=13, loc='left')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Panel C: Prompting by Model
ax = axes[1, 0]
x = np.arange(len(pivot_table.index))
width = 0.25
for i, tech in enumerate(techniques_ordered):
    if tech in pivot_table.columns:
        ax.bar(x + i*width, pivot_table[tech], width, label=tech, 
               alpha=0.85, edgecolor='black', linewidth=1.5)

ax.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=12)
ax.set_title("(C) Prompting Strategy by Model", fontweight='bold', fontsize=13, loc='left')
ax.set_xticks(x + width)
ax.set_xticklabels(pivot_table.index, fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, edgecolor='black')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Panel D: Error Rates
ax = axes[1, 1]
error_data = data.copy()
error_data['has_error'] = error_data['error'].notna()
tech_errors = error_data.groupby('technique')['has_error'].mean() * 100
tech_errors.index = [TECHNIQUE_NAMES[t] for t in tech_errors.index]
bars = ax.bar(range(len(tech_errors)), tech_errors.values, color=colors,
              alpha=0.85, edgecolor='black', linewidth=2)
ax.set_xticks(range(len(tech_errors)))
ax.set_xticklabels(tech_errors.index, fontsize=11, fontweight='bold', rotation=15, ha='right')
ax.set_ylabel("Error Rate (%)", fontweight='bold', fontsize=12)
ax.set_title("(D) Error Rate by Strategy", fontweight='bold', fontsize=13, loc='left')
ax.set_ylim(0, 10)
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig("figures/combined_v2_4panel.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: combined_v2_4panel.png")

print("\n" + "="*80)
print("="*80)
print("\nGenerated 10 different graph variants:")
print("\nPROMPTING STRATEGY (4 variants):")
print("  1. prompting_v1_simple_bar.png - Clean vertical bars")
print("  2. prompting_v2_horizontal_bar.png - Horizontal bars")
print("  3. prompting_v3_with_errorbar.png - With standard deviation")
print("  4. prompting_v4_grouped_by_model.png - Grouped by model")
print("\nMODEL COMPARISON (4 variants):")
print("  5. model_v1_simple_bar.png - Clean vertical bars")
print("  6. model_v2_with_trials.png - With trial counts")
print("  7. model_v3_stacked_by_technique.png - Grouped by technique")
print("  8. model_v4_heatmap.png - Heatmap visualization")
print("\nCOMBINED (2 variants):")
print("  9. combined_v1_2panel.png - 2-panel side-by-side")
print("  10. combined_v2_4panel.png - 4-panel comprehensive")
print("\nChoose the ones you like best for your paper!")


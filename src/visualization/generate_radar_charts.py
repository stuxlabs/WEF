import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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
print("GENERATING RADAR CHARTS")
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

# RADAR 1: Model Comparison (Multiple Dimensions)
print("\n[1] Generating Model Comparison Radar Chart...")

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Define categories (metrics to compare)
categories = ['Success Rate', 'Speed\n(normalized)',
              'Basic Recon', 'Targeted Attack', 'Contextual Chain']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, model in enumerate(MODELS):
    model_data = data[data['model'] == model]

    # Calculate metrics (normalize to 0-100 scale)
    success = model_data['success'].mean() * 100

    # Speed (inverse, normalized): faster = higher score
    success_trials = model_data[model_data['success'] == True]
    avg_time = success_trials['time_seconds'].mean() if len(success_trials) > 0 else 100
    speed = 100 - min((avg_time / 100) * 100, 100)  # Normalize, cap at 100

    # Scenario-specific success
    basic = model_data[model_data['scenario'] == 'basic_recon']['success'].mean() * 100
    targeted = model_data[model_data['scenario'] == 'targeted_attack']['success'].mean() * 100
    contextual = model_data[model_data['scenario'] == 'contextual_chain']['success'].mean() * 100

    values = [success, speed, basic, targeted, contextual]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2.5, label=MODEL_NAMES[model], 
            color=colors[i], markersize=8)
    ax.fill(angles, values, alpha=0.15, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11, weight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], size=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title('Model Performance Overview (Radar Chart)', size=16, weight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, frameon=True, 
          edgecolor='black', fancybox=True)

plt.tight_layout()
plt.savefig("figures/radar_v1_model_comparison.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: radar_v1_model_comparison.png")

# RADAR 2: Prompting Strategy Comparison
print("\n[2] Generating Prompting Strategy Radar Chart...")

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

categories = ['Overall Success', 'Speed\n(normalized)',
              'Llama 3 8B', 'Llama 4 Scout', 'Mistral 7B']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

colors = ['#3498db', '#2ecc71', '#e74c3c']

for i, technique in enumerate(TECHNIQUES):
    tech_data = data[data['technique'] == technique]

    # Overall metrics
    success = tech_data['success'].mean() * 100

    success_trials = tech_data[tech_data['success'] == True]
    avg_time = success_trials['time_seconds'].mean() if len(success_trials) > 0 else 100
    speed = 100 - min((avg_time / 100) * 100, 100)

    # Model-specific success
    llama3 = tech_data[tech_data['model'] == 'llama3-8b']['success'].mean() * 100
    llama4 = tech_data[tech_data['model'] == 'llama4-scout']['success'].mean() * 100
    mistral = tech_data[tech_data['model'] == 'mistral-7b']['success'].mean() * 100

    values = [success, speed, llama3, llama4, mistral]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2.5, label=TECHNIQUE_NAMES[technique],
            color=colors[i], markersize=8)
    ax.fill(angles, values, alpha=0.15, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11, weight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], size=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title('Prompting Strategy Overview (Radar Chart)', size=16, weight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, frameon=True,
          edgecolor='black', fancybox=True)

plt.tight_layout()
plt.savefig("figures/radar_v2_prompting_comparison.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: radar_v2_prompting_comparison.png")

# RADAR 3: Multi-Panel (All Models + All Prompting)
print("\n[3] Generating Multi-Panel Radar Chart...")

fig = plt.figure(figsize=(18, 8))

# Left: Models
ax1 = fig.add_subplot(121, projection='polar')
categories = ['Success', 'Speed', 'Basic Recon', 'Targeted', 'Contextual']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

colors_model = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, model in enumerate(MODELS):
    model_data = data[data['model'] == model]
    success = model_data['success'].mean() * 100
    success_trials = model_data[model_data['success'] == True]
    avg_time = success_trials['time_seconds'].mean() if len(success_trials) > 0 else 100
    speed = 100 - min((avg_time / 100) * 100, 100)
    basic = model_data[model_data['scenario'] == 'basic_recon']['success'].mean() * 100
    targeted = model_data[model_data['scenario'] == 'targeted_attack']['success'].mean() * 100
    contextual = model_data[model_data['scenario'] == 'contextual_chain']['success'].mean() * 100

    values = [success, speed, basic, targeted, contextual]
    values += values[:1]
    
    ax1.plot(angles, values, 'o-', linewidth=2, label=MODEL_NAMES[model],
            color=colors_model[i], markersize=6)
    ax1.fill(angles, values, alpha=0.15, color=colors_model[i])

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(categories, size=10, weight='bold')
ax1.set_ylim(0, 100)
ax1.set_yticks([25, 50, 75, 100])
ax1.set_yticklabels(['25', '50', '75', '100'], size=9)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_title('(A) Model Comparison', size=14, weight='bold', pad=20)
ax1.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10)

# Right: Prompting
ax2 = fig.add_subplot(122, projection='polar')
categories = ['Success', 'Speed', 'Llama 3', 'Llama 4', 'Mistral']
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

colors_prompt = ['#3498db', '#2ecc71', '#e74c3c']
for i, technique in enumerate(TECHNIQUES):
    tech_data = data[data['technique'] == technique]
    success = tech_data['success'].mean() * 100
    success_trials = tech_data[tech_data['success'] == True]
    avg_time = success_trials['time_seconds'].mean() if len(success_trials) > 0 else 100
    speed = 100 - min((avg_time / 100) * 100, 100)
    llama3 = tech_data[tech_data['model'] == 'llama3-8b']['success'].mean() * 100
    llama4 = tech_data[tech_data['model'] == 'llama4-scout']['success'].mean() * 100
    mistral = tech_data[tech_data['model'] == 'mistral-7b']['success'].mean() * 100

    values = [success, speed, llama3, llama4, mistral]
    values += values[:1]
    
    ax2.plot(angles, values, 'o-', linewidth=2, label=TECHNIQUE_NAMES[technique],
            color=colors_prompt[i], markersize=6)
    ax2.fill(angles, values, alpha=0.15, color=colors_prompt[i])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, size=10, weight='bold')
ax2.set_ylim(0, 100)
ax2.set_yticks([25, 50, 75, 100])
ax2.set_yticklabels(['25', '50', '75', '100'], size=9)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_title('(B) Prompting Strategy', size=14, weight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)

plt.tight_layout()
plt.savefig("figures/radar_v3_combined_dual.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: radar_v3_combined_dual.png")

# RADAR 4: Single Model Deep Dive (Llama 4 Scout)
print("\n[4] Generating Llama 4 Scout Deep Dive Radar...")

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

categories = ['Success Rate', 'Speed', 'Task-Only', 'Exemplar (k=3)',
              'Structured', 'Basic Recon', 'Targeted', 'Contextual']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

llama4_data = data[data['model'] == 'llama4-scout']

# Metrics
success = llama4_data['success'].mean() * 100
success_trials = llama4_data[llama4_data['success'] == True]
avg_time = success_trials['time_seconds'].mean() if len(success_trials) > 0 else 100
speed = 100 - min((avg_time / 100) * 100, 100)

# By technique
task_only = llama4_data[llama4_data['technique'] == 'zero-shot']['success'].mean() * 100
exemplar = llama4_data[llama4_data['technique'] == 'few-shot-3']['success'].mean() * 100
structured = llama4_data[llama4_data['technique'] == 'cot']['success'].mean() * 100

# By scenario
basic = llama4_data[llama4_data['scenario'] == 'basic_recon']['success'].mean() * 100
targeted = llama4_data[llama4_data['scenario'] == 'targeted_attack']['success'].mean() * 100
contextual = llama4_data[llama4_data['scenario'] == 'contextual_chain']['success'].mean() * 100

values = [success, speed, task_only, exemplar, structured, basic, targeted, contextual]
values += values[:1]

ax.plot(angles, values, 'o-', linewidth=3, color='#ff7f0e', markersize=10)
ax.fill(angles, values, alpha=0.25, color='#ff7f0e')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11, weight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], size=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title('Llama 4 Scout - Comprehensive Performance Profile', size=16, weight='bold', pad=30)

# Add annotation
ax.text(0.5, 0.05, f'Overall: {success:.1f}% Success',
        transform=ax.transAxes, ha='center', fontsize=12, weight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig("figures/radar_v4_llama4_deep_dive.png", bbox_inches='tight', dpi=300)
print("  ✓ Saved: radar_v4_llama4_deep_dive.png")

print("\n" + "="*80)
print("="*80)
print("\nGenerated 4 radar chart variants:")
print("  1. radar_v1_model_comparison.png - Models across 6 dimensions")
print("  2. radar_v2_prompting_comparison.png - Prompting strategies across 6 dimensions")
print("  3. radar_v3_combined_dual.png - Side-by-side models + prompting")
print("  4. radar_v4_llama4_deep_dive.png - Llama 4 Scout detailed breakdown")
print("\nRadar charts show holistic overview at a glance!")


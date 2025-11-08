import pandas as pd
import numpy as np
import os

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
    "cot": "CoT"
}

print("="*80)
print("GENERATING PAPER TABLES - LaTeX Format")
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
print(f"Loaded {len(data)} trials from {len(MODELS)} models\n")

# TABLE 1: Success Rates (Model × Scenario × Technique)
print("="*80)
print("TABLE 1: Success Rates (%)")
print("="*80)
print("\n% LaTeX code:\n")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Success Rates (\\%) by Model, Scenario, and Prompting Technique}")
print("\\label{tab:success-rates}")
print("\\small")
print("\\begin{tabular}{@{}llccc@{}}")
print("\\toprule")
print("\\textbf{Scenario} & \\textbf{Model} & \\textbf{Zero-Shot} & \\textbf{Few-Shot-3} & \\textbf{CoT} \\\\")
print("\\midrule")

for scenario in SCENARIOS:
    scenario_data = data[data['scenario'] == scenario]
    
    for idx, model in enumerate(MODELS):
        model_data = scenario_data[scenario_data['model'] == model]
        
        row = []
        for technique in TECHNIQUES:
            tech_data = model_data[model_data['technique'] == technique]
            if len(tech_data) > 0:
                success = tech_data['success'].mean() * 100
                row.append(f"{success:.0f}")
            else:
                row.append("--")
        
        # First row includes scenario name
        if idx == 0:
            print(f"\\multirow{{3}}{{*}}{{{SCENARIO_NAMES[scenario]}}} & {MODEL_NAMES[model]} & {row[0]} & {row[1]} & {row[2]} \\\\")
        else:
            print(f" & {MODEL_NAMES[model]} & {row[0]} & {row[1]} & {row[2]} \\\\")
    
    if scenario != SCENARIOS[-1]:
        print("\\midrule")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# TABLE 2: Overall Model Performance
print("\n\n" + "="*80)
print("TABLE 2: Overall Model Performance")
print("="*80)
print("\n% LaTeX code:\n")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Overall Model Performance Summary}")
print("\\label{tab:model-summary}")
print("\\begin{tabular}{@{}lcccc@{}}")
print("\\toprule")
print("\\textbf{Model} & \\textbf{Trials} & \\textbf{Success (\\%)} & \\textbf{Error (\\%)} & \\textbf{Avg Time (s)} \\\\")
print("\\midrule")

for model in MODELS:
    model_data = data[data['model'] == model]
    
    trials = len(model_data)
    success = model_data['success'].mean() * 100
    errors = model_data['error'].notna().mean() * 100
    
    success_data = model_data[model_data['success'] == True]
    avg_time = success_data['time_seconds'].mean() if len(success_data) > 0 else 0
    
    print(f"{MODEL_NAMES[model]} & {trials} & {success:.1f} & {errors:.1f} & {avg_time:.1f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# TABLE 3: Prompting Technique Effectiveness
print("\n\n" + "="*80)
print("TABLE 3: Prompting Technique Effectiveness")
print("="*80)
print("\n% LaTeX code:\n")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Prompting Technique Effectiveness Across All Scenarios}")
print("\\label{tab:prompting}")
print("\\begin{tabular}{@{}lccc@{}}")
print("\\toprule")
print("\\textbf{Technique} & \\textbf{Success (\\%)} & \\textbf{Error (\\%)} & \\textbf{Avg Time (s)} \\\\")
print("\\midrule")

for technique in TECHNIQUES:
    tech_data = data[data['technique'] == technique]
    
    success = tech_data['success'].mean() * 100
    errors = tech_data['error'].notna().mean() * 100
    
    success_trials = tech_data[tech_data['success'] == True]
    avg_time = success_trials['time_seconds'].mean() if len(success_trials) > 0 else 0
    
    print(f"{TECHNIQUE_NAMES[technique]} & {success:.1f} & {errors:.1f} & {avg_time:.1f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# TABLE 4: Scenario Difficulty Analysis
print("\n\n" + "="*80)
print("TABLE 4: Scenario Difficulty Analysis")
print("="*80)
print("\n% LaTeX code:\n")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Scenario Difficulty: Average Success Rate Across All Models and Techniques}")
print("\\label{tab:scenarios}")
print("\\begin{tabular}{@{}lcccc@{}}")
print("\\toprule")
print("\\textbf{Scenario} & \\textbf{Trials} & \\textbf{Success (\\%)} & \\textbf{Error (\\%)} & \\textbf{Avg Time (s)} \\\\")
print("\\midrule")

for scenario in SCENARIOS:
    scenario_data = data[data['scenario'] == scenario]
    
    trials = len(scenario_data)
    success = scenario_data['success'].mean() * 100
    errors = scenario_data['error'].notna().mean() * 100
    
    success_trials = scenario_data[scenario_data['success'] == True]
    avg_time = success_trials['time_seconds'].mean() if len(success_trials) > 0 else 0
    
    print(f"{SCENARIO_NAMES[scenario]} & {trials} & {success:.1f} & {errors:.1f} & {avg_time:.1f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# Summary Statistics (for text)
print("\n\n" + "="*80)
print("KEY STATISTICS FOR PAPER TEXT")
print("="*80)

print("\n## Overall Performance:")
print(f"Total trials: {len(data)}")
print(f"Overall success rate: {data['success'].mean() * 100:.1f}%")
print(f"Overall error rate: {data['error'].notna().mean() * 100:.1f}%")

print("\n## Best Model:")
best_model = data.groupby('model')['success'].mean().idxmax()
best_success = data.groupby('model')['success'].mean().max() * 100
print(f"{MODEL_NAMES[best_model]}: {best_success:.1f}% success rate")

print("\n## Best Prompting Technique:")
best_technique = data.groupby('technique')['success'].mean().idxmax()
best_tech_success = data.groupby('technique')['success'].mean().max() * 100
print(f"{TECHNIQUE_NAMES[best_technique]}: {best_tech_success:.1f}% success rate")

print("\n## Hardest Scenario:")
hardest = data.groupby('scenario')['success'].mean().idxmin()
hardest_success = data.groupby('scenario')['success'].mean().min() * 100
print(f"{SCENARIO_NAMES[hardest]}: {hardest_success:.1f}% success rate")

print("\n## Easiest Scenario:")
easiest = data.groupby('scenario')['success'].mean().idxmax()
easiest_success = data.groupby('scenario')['success'].mean().max() * 100
print(f"{SCENARIO_NAMES[easiest]}: {easiest_success:.1f}% success rate")

print("\n## Model with Lowest Errors:")
lowest_errors = data.groupby('model').apply(lambda x: x['error'].notna().mean()).idxmin()
lowest_error_rate = data.groupby('model').apply(lambda x: x['error'].notna().mean()).min() * 100
print(f"{MODEL_NAMES[lowest_errors]}: {lowest_error_rate:.1f}% error rate")

print("\n" + "="*80)
print("="*80)

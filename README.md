# WiFi Penetration Testing LLM Agent Framework

A modular research framework for evaluating Large Language Model (LLM) agents on WiFi penetration testing tasks. 

## Overview

This framework provides a controlled environment for systematically evaluating LLM-based autonomous agents on wireless security testing scenarios. It implements a simulated WiFi penetration testing environment with realistic tool behaviors and probabilistic success models.

## Repository Structure

```
.
├── src/
│   ├── evaluation/          # Core evaluation framework (modular)
│   │   ├── __init__.py      # Package exports
│   │   ├── main.py          # Main entry point
│   │   ├── config.py        # Configuration and constants
│   │   ├── scenarios.py     # Scenario generation
│   │   ├── tools.py         # Simulated WiFi tools
│   │   ├── prompts.py       # Prompting strategies
│   │   └── evaluator.py     # Evaluation logic
│   ├── models/              # LLM model interfaces
│   │   └── aws_model.py
│   ├── visualization/       # Visualization generators
│   │   ├── generate_radar_charts.py
│   │   ├── generate_all_graph_variants.py
│   │   └── ...
│   └── utils/               # Utility functions
│       └── pdf_converter.py
├── scripts/                 # Convenience scripts
│   ├── run_evaluation.py
│   ├── generate_all_visualizations.py
│   └── convert_to_pdf.py
├── results/                 # Experimental results (CSV)
├── figures/                 # Generated visualizations (PNG/PDF)
├── setup.py                 # Package installation
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Modular Architecture

The evaluation framework has been designed with clean separation of concerns:

### `src.evaluation` Modules

1. **config.py**: Configuration, constants, and global settings
2. **scenarios.py**: Scenario generation and validation logic
3. **tools.py**: Simulated WiFi penetration testing tools
4. **prompts.py**: Prompting strategies (Task-Only, Exemplar-Based, Structured Reasoning)
5. **evaluator.py**: Core evaluation logic and trial execution
6. **main.py**: CLI entry point and argument parsing

This modular design enables:
- Easy extension with new scenarios
- Clean testing of individual components
- Reusable tool simulators
- Flexible prompting strategies

## Features

- **Modular Architecture**: Clean separation of evaluation, visualization, and utilities
- **Simulated WiFi Environment**: Realistic wireless network simulation with multiple access points
- **Multiple LLM Support**: Compatible with OpenRouter API (Llama, Mistral, and other models)
- **Prompting Strategies**: Task-Only, Exemplar-Based, and Structured Reasoning approaches
- **Comprehensive Evaluation**: 390+ trials across models, techniques, and scenarios
- **Publication-Ready Visualizations**: Seaborn-styled figures with PDF export

## Installation

### From Source

```bash
git clone <repository-url>
cd wifi-llm-agents
pip install -e .
```

### Dependencies Only

```bash
pip install -r requirements.txt
```

## Configuration

Set up OpenRouter API credentials:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

Get your API key from: https://openrouter.ai/keys

## Usage

### Running Evaluations

```bash
# Run evaluation with all scenarios and techniques
python scripts/run_evaluation.py --model llama4-scout --scenario all --technique all --trials 10

# Run specific scenario
python scripts/run_evaluation.py --model llama3-8b --scenario basic_recon --technique zero-shot --trials 5

# Or use as module
python -m src.evaluation.main --model mistral-7b --scenario targeted_attack --technique few-shot-3 --trials 10
```

### Generating Visualizations

```bash
# Generate all visualizations at once
python scripts/generate_all_visualizations.py

# Or individual components
python -m src.visualization.generate_radar_charts
python -m src.visualization.generate_all_graph_variants

# Convert to PDF
python scripts/convert_to_pdf.py
```

### Module Usage (Python API)

```python
from src.evaluation import run_evaluation_batch, WiFiEvaluator
from src.evaluation.scenarios import ScenarioGenerator
from src.evaluation.tools import WiFiToolSimulator

# Run batch evaluation
results = run_evaluation_batch(
    model="llama4-scout",
    scenarios=["basic_recon", "targeted_attack"],
    techniques=["zero-shot", "few-shot-3"],
    trials_per_config=10
)

# Or use components directly
evaluator = WiFiEvaluator("llama3-8b")
result = evaluator.run_trial(
    trial_id=0,
    scenario_type="basic_recon",
    technique="zero-shot",
    num_aps=5
)
```

## Evaluation Framework

### Scenarios

1. **Basic Reconnaissance**: Network discovery and information gathering (5 APs)
2. **Targeted Attack**: Single-target handshake capture (8 APs)
3. **Contextual Chain**: Multi-stage attack with context tracking (12 APs)

### Models Evaluated

- Llama 3 8B
- Llama 4 Scout (70B)
- Mistral 7B

### Prompting Techniques

- **Task-Only**: Tool documentation without demonstrations
- **Exemplar-Based (k=3)**: Three complete attack demonstrations  
- **Structured Reasoning**: 5-stage protocol (PARSE→RANK→PLAN→ADAPT→EXECUTE)

## Results

Key findings from 390 experimental trials:

- **Task-Only prompting**: 43.3% success (best overall)
- **Llama 4 Scout**: 46.1% success (best model)
- **Best configuration**: Llama 4 Scout + Task-Only = 50.0%
- **Low error rates**: <1% across all configurations

## Ethical Notice

⚠️ **IMPORTANT**: This codebase has been sanitized for academic research purposes.

- All harmful prompt content has been **REMOVED**
- Actual attack implementations have been **ABSTRACTED**
- Tool command details have been **REDACTED**
- Real WiFi exploitation code has been **EXCLUDED**

This repository serves as a **skeleton framework** demonstrating the evaluation methodology described in our paper. It cannot be used for actual penetration testing without significant reconstruction.

The framework is intended solely for:
- Academic research and reproducibility
- Understanding LLM agent evaluation methodologies
- Studying prompting strategy effectiveness
- Advancing defensive security research

**Do not attempt to use this code for unauthorized network access.**

## Development

### Extending Evaluation

1. Add new scenario in `src/evaluation/scenarios.py`
2. Update `SCENARIOS` dict in `src/evaluation/config.py`
3. Add validator in `ScenarioValidator` class
4. Run evaluation with new scenario

### Adding New Tools

1. Add tool method to `WiFiToolSimulator` in `src/evaluation/tools.py`
2. Update `create_tool_wrappers()` function
3. Tools automatically available to agents

### Custom Prompting Strategies

1. Add prompt function to `src/evaluation/prompts.py`
2. Update `get_prompt_for_technique()` dispatcher
3. Add technique to `PROMPTING_TECHNIQUES` in config

## License

This research code is provided for academic purposes only.

---

**Note**: This is a research artifact accompanying an academic paper. The code represents a sanitized evaluation framework and should not be considered production-ready or suitable for actual security testing. All harmful content has been removed to ensure responsible disclosure.

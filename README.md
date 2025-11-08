# WiFi Penetration Testing LLM Agent Framework

A research framework for evaluating Large Language Model (LLM) agents on WiFi penetration testing tasks. This codebase accompanies our AAAI 2026 paper submission.

## Overview

This framework provides a controlled environment for systematically evaluating LLM-based autonomous agents on wireless security testing scenarios. It implements a simulated WiFi penetration testing environment with realistic tool behaviors and probabilistic success models.

## Repository Structure

```
.
├── wifi_eval.py              # Main evaluation framework
├── generate_*.py             # Visualization generation scripts
├── convert_all_to_pdf.py     # PDF conversion utility
├── results/                  # Experimental results (CSV format)
├── figures/                  # Generated visualizations (PNG/PDF)
└── README.md                 # This file
```

## Features

- **Simulated WiFi Environment**: Realistic wireless network simulation with multiple access points
- **Multiple LLM Support**: Compatible with OpenRouter API (Llama, Mistral, and other models)
- **Prompting Strategies**: Implements Task-Only, Exemplar-Based, and Structured Reasoning approaches
- **Comprehensive Evaluation**: 390+ trials across models, techniques, and scenarios
- **Visualization Tools**: Publication-ready figures with seaborn styling

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- matplotlib
- seaborn
- pillow
- numpy
- boto3 (for model interface)

## Configuration

Set up OpenRouter API credentials:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

Get your API key from: https://openrouter.ai/keys

## Evaluation Framework

### Scenarios

1. **Basic Reconnaissance**: Network discovery and information gathering (5 APs)
2. **Targeted Attack**: Single-target handshake capture (8 APs)
3. **Contextual Chain**: Multi-stage attack requiring context tracking (12 APs)

### Models Evaluated

- Llama 3 8B
- Llama 4 Scout (70B)
- Mistral 7B

*Note: Model availability depends on OpenRouter support*

### Prompting Techniques

- **Task-Only**: Tool documentation without demonstrations
- **Exemplar-Based (k=3)**: Three complete attack demonstrations
- **Structured Reasoning**: 5-stage protocol (PARSE→RANK→PLAN→ADAPT→EXECUTE)

## Usage

The framework has been sanitized for academic publication. All harmful prompts and attack implementations have been removed or abstracted. The codebase serves as a skeleton framework demonstrating the methodology presented in the paper.

### Running Experiments

```bash
# Example framework structure (implementation details removed)
python wifi_eval.py --model llama4-scout --scenario basic_recon --technique zero-shot --trials 10
```

### Generating Visualizations

```bash
# Generate radar charts
python generate_radar_charts.py

# Generate seaborn visualizations  
python generate_all_graph_variants.py

# Convert to PDF
python convert_all_to_pdf.py
```

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

## Paper Reproducibility

All experimental results, visualizations, and tables are included in the `results/` and `figures/` directories. The framework can regenerate all paper figures using the included scripts.

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{wifi-llm-agents-2026,
  title={Autonomous WiFi Penetration Testing with Large Language Model Agents},
  author={[Authors]},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

This research code is provided for academic purposes only. See accompanying paper for full details on methodology and ethical considerations.

## Contact

For questions about the methodology or framework design, please contact the authors through the paper submission system.

---

**Note**: This is a research artifact accompanying an academic paper. The code represents a sanitized evaluation framework and should not be considered production-ready or suitable for actual security testing. All harmful content has been removed to ensure responsible disclosure.

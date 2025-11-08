from setuptools import setup, find_packages

setup(
    name="wifi-llm-agents",
    version="1.0.0",
    description="WiFi Penetration Testing LLM Agent Evaluation Framework",
    author="Research Team",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "wifi-eval=src.evaluation.wifi_eval:main",
        ],
    },
)

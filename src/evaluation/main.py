#!/usr/bin/env python3
"""
WiFi Penetration Testing LLM Agent Evaluation Framework
Main entry point for running evaluations

ETHICAL NOTICE: Sanitized for academic research. All harmful content removed.
This is a skeleton framework demonstrating evaluation methodology only.
"""
import argparse
import sys
from .config import logger, SCENARIOS, PROMPTING_TECHNIQUES, MODEL_CONFIGS
from .evaluator import run_evaluation_batch


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="WiFi Penetration Testing LLM Agent Evaluation"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="LLM model to evaluate"
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario to evaluate (basic_recon, targeted_attack, contextual_chain, or 'all')"
    )

    parser.add_argument(
        "--technique",
        type=str,
        default="all",
        help="Prompting technique (zero-shot, few-shot-3, cot, or 'all')"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per configuration"
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()

    logger.info("="*80)
    logger.info("WiFi LLM Agent Evaluation Framework")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Scenario: {args.scenario}")
    logger.info(f"Technique: {args.technique}")
    logger.info(f"Trials per config: {args.trials}")
    logger.info("="*80)

    # Determine scenarios to run
    if args.scenario == "all":
        scenarios = list(SCENARIOS.keys())
    else:
        if args.scenario not in SCENARIOS:
            logger.error(f"Unknown scenario: {args.scenario}")
            sys.exit(1)
        scenarios = [args.scenario]

    # Determine techniques to run
    if args.technique == "all":
        techniques = PROMPTING_TECHNIQUES
    else:
        if args.technique not in PROMPTING_TECHNIQUES:
            logger.error(f"Unknown technique: {args.technique}")
            sys.exit(1)
        techniques = [args.technique]

    logger.info(f"Running {len(scenarios)} scenarios × {len(techniques)} techniques × {args.trials} trials")
    logger.info(f"Total trials: {len(scenarios) * len(techniques) * args.trials}")

    # Run evaluation
    try:
        results = run_evaluation_batch(
            model=args.model,
            scenarios=scenarios,
            techniques=techniques,
            trials_per_config=args.trials
        )

        logger.info("="*80)
        logger.info(f"Evaluation complete: {len(results)} trials")
        logger.info("="*80)

        # Print summary
        success_count = sum(1 for r in results if r.success)
        logger.info(f"Success rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")

    except KeyboardInterrupt:
        logger.warning("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()

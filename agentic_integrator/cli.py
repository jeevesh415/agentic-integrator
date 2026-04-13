"""CLI entry point for Agentic Integrator."""

import argparse
import logging
import sys

from agentic_integrator.integrator import AgenticIntegrator, IntegratorConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Agentic Integrator: Agent-S3 + World Model for Human-Like Screen Navigation"
    )

    # Main model
    parser.add_argument("--provider", default="openai", help="LLM provider (openai, anthropic, gemini)")
    parser.add_argument("--model", default="gpt-4o", help="Main LLM model name")
    parser.add_argument("--model_url", default="", help="Custom API URL")
    parser.add_argument("--model_api_key", default="", help="API key for main model")
    parser.add_argument("--model_temperature", type=float, default=None, help="Fixed temperature")

    # Grounding model
    parser.add_argument("--ground_provider", required=True, help="Grounding model provider")
    parser.add_argument("--ground_url", required=True, help="Grounding model URL")
    parser.add_argument("--ground_model", required=True, help="Grounding model name")
    parser.add_argument("--grounding_width", type=int, default=1920, help="Grounding output width")
    parser.add_argument("--grounding_height", type=int, default=1080, help="Grounding output height")

    # World model
    parser.add_argument("--world_model_candidates", type=int, default=3, help="Number of candidates to simulate")
    parser.add_argument("--world_model_provider", default="", help="Separate provider for world model")
    parser.add_argument("--world_model_model", default="", help="Separate model for world model")
    parser.add_argument("--safety_gate", action="store_true", help="Enable safety gate")
    parser.add_argument("--no_adaptive", action="store_true", help="Disable adaptive confidence")

    # Agent settings
    parser.add_argument("--max_trajectory", type=int, default=8, help="Max trajectory length")
    parser.add_argument("--enable_local_env", action="store_true", help="Enable local code execution")
    parser.add_argument("--task", type=str, default=None, help="Task to execute (optional)")

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = IntegratorConfig(
        provider=args.provider,
        model=args.model,
        model_url=args.model_url,
        model_api_key=args.model_api_key,
        model_temperature=args.model_temperature,
        ground_provider=args.ground_provider,
        ground_url=args.ground_url,
        ground_model=args.ground_model,
        grounding_width=args.grounding_width,
        grounding_height=args.grounding_height,
        world_model_candidates=args.world_model_candidates,
        world_model_provider=args.world_model_provider,
        world_model_model=args.world_model_model,
        safety_gate_enabled=args.safety_gate,
        adaptive_confidence=not args.no_adaptive,
        max_trajectory_length=args.max_trajectory,
        enable_local_env=args.enable_local_env,
    )

    agent = AgenticIntegrator(config=config)

    if args.task:
        print(f"\\n🧠 Agentic Integrator — Task: {args.task}")
        print(f"   Model: {config.model} | Candidates: {config.world_model_candidates}")
        print(f"   Safety Gate: {config.safety_gate_enabled} | Adaptive: {config.adaptive_confidence}")
        print("=" * 60)
        # In production, this would integrate with the desktop environment
        # For now, print ready state
        print("\\n✅ Agent initialized and ready.")
        print("   Connect to a desktop environment to start executing tasks.")
    else:
        print("\\n🧠 Agentic Integrator initialized.")
        print("   Use --task 'description' to run a task, or import as a library.")


if __name__ == "__main__":
    main()

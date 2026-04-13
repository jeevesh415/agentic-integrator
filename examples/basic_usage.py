"""Basic usage example for Agentic Integrator."""

from agentic_integrator import AgenticIntegrator

# Initialize with world model enabled
agent = AgenticIntegrator(
    provider="openai",
    model="gpt-4o",
    ground_provider="huggingface",
    ground_url="http://localhost:8080",
    ground_model="ui-tars-1.5-7b",
    grounding_width=1920,
    grounding_height=1080,
    world_model_candidates=3,
    safety_gate_enabled=True,
    adaptive_confidence=True,
)

print("Agent initialized!")
print(f"Config: {agent.config}")

# In a real desktop environment, you would:
# 1. Capture a screenshot
# 2. Call agent.predict(instruction, observation)
# 3. Execute the returned action
# 4. Repeat

# Example prediction loop (pseudo-code):
# observation = {"screenshot": capture_screenshot()}
# info, actions = agent.predict("Open Chrome and search for AI papers", observation)
# execute_actions(actions)
# print(f"World model stats: {agent.get_stats()}")

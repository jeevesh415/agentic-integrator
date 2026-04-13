"""LLM prompts for the world model, planner, and safety gate."""

WORLD_MODEL_SYSTEM_PROMPT = """\
You are a World Model for GUI automation. Your job is to predict what will happen \
on screen when an action is executed.

You have deep knowledge of how desktop and web GUIs behave — button clicks, \
form submissions, navigation, scrolling, file operations, and application state transitions.

Think like a human who has used thousands of apps: when you see a screenshot and \
a proposed action, predict the EXACT outcome with high precision.

Always respond in valid JSON format.
"""

WORLD_MODEL_SIMULATE_PROMPT = """\
## Task
{task_description}

## Action History
{action_history}

## Candidate Action to Simulate
{candidate_action}

## Instructions
Look at the current screenshot and predict what will happen if the agent executes \
the candidate action above.

Respond with a JSON object:
```json
{{
    "predicted_state": "Description of what the screen will look like after the action",
    "task_progress_score": 0.0-1.0,  // How much this advances the task (0=no progress, 1=task complete)
    "confidence": 0.0-1.0,  // How confident you are in this prediction
    "safety_score": 0.0-1.0,  // How safe this action is (1=completely safe, 0=dangerous)
    "is_irreversible": true/false,  // Whether this action can be undone
    "risk_description": "Description of any risks (empty if safe)",
    "reasoning": "Brief explanation of your prediction"
}}
```
"""

WORLD_MODEL_BATCH_SIMULATE_PROMPT = """\
## Task
{task_description}

## Action History
{action_history}

## Candidate Actions to Simulate
{candidate_actions}

## Instructions
Look at the current screenshot and predict what will happen for EACH candidate action.

Respond with a JSON array of {num_candidates} objects, one per candidate (in order):
```json
[
    {{
        "predicted_state": "Description of outcome for candidate 1",
        "task_progress_score": 0.0-1.0,
        "confidence": 0.0-1.0,
        "safety_score": 0.0-1.0,
        "is_irreversible": true/false,
        "risk_description": "",
        "reasoning": "Brief explanation"
    }},
    ...
]
```
"""

VERIFICATION_PROMPT = """\
## Previous Prediction
{predicted_state}

## Actual Current Screen
[See screenshot]

## Question
Compare the prediction above with what actually happened (shown in the screenshot).
How accurate was the prediction? Score from 0.0 (completely wrong) to 1.0 (perfect).

Respond with:
```json
{{
    "accuracy": 0.0-1.0,
    "what_matched": "What the prediction got right",
    "what_differed": "What was different from prediction",
    "lesson": "What the world model should learn from this"
}}
```
"""

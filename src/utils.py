# Rubrics live in src/rubrics.py; re-exported here for older imports.
from src.rubrics import fetch_rubric  # noqa: F401


def print_state(state):
    """
    Print the current state of the agent in a readable format.
    """
    print("Current State:")
    print(f"  Actual Question: {state.input.actual_question}")
    print(f"  User Response: {state.input.user_response}")
    print(f"  Difficulty: {state.input.difficulty}")
    print(f"  Domain: {state.input.domain}")
    print(f"  Plan: {state.plan}")

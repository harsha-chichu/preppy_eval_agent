from src.agent.state import Rubric

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

_RUBRIC_REGISTRY: dict[str, Rubric] = {
    "ai": Rubric(
        topic="ai",
        criteria=[
            "conceptual correctness",
            "understanding of underlying mechanisms",
            "awareness of limitations and failure modes",
            "practical real-world grounding",
        ],
        weightings={
            "conceptual correctness": 0.35,
            "underlying mechanisms": 0.3,
            "limitations": 0.2,
            "practical": 0.15,
        },
        source="static_registry",
    ),
    "ml": Rubric(
        topic="ml",
        criteria=[
            "correct model and algorithm reasoning",
            "data handling and evaluation methodology",
            "bias-variance and overfitting awareness",
            "metric selection and justification",
        ],
        weightings={
            "model and algorithm": 0.35,
            "data handling": 0.25,
            "bias-variance": 0.2,
            "metric selection": 0.2,
        },
        source="static_registry",
    ),
    "ai system design": Rubric(
        topic="ai system design",
        criteria=[
            "end-to-end architecture covering data, training, and serving",
            "scalability and latency tradeoffs",
            "reliability, monitoring, and failure handling",
            "cost and resource tradeoff awareness",
        ],
        weightings={
            "end-to-end architecture": 0.35,
            "scalability and latency": 0.25,
            "reliability, monitoring": 0.2,
            "cost and resource": 0.2,
        },
        source="static_registry",
    ),
    "ml system design": Rubric(
        topic="ml system design",
        criteria=[
            "feature engineering and data pipeline design",
            "training and serving infrastructure choices",
            "model monitoring and retraining strategy",
            "handling of data and concept drift",
        ],
        weightings={
            "feature engineering": 0.3,
            "training and serving": 0.25,
            "monitoring and retraining": 0.25,
            "concept drift": 0.2,
        },
        source="static_registry",
    ),
    "general": Rubric(
        topic="general",
        criteria=["correctness", "completeness", "clarity"],
        weightings={"correctness": 0.5, "completeness": 0.3, "clarity": 0.2},
        source="static_registry",
    ),
}
 
 
def fetch_rubric(domain: str) -> Rubric:
    
    return _RUBRIC_REGISTRY.get(domain.lower().strip(), _RUBRIC_REGISTRY["general"])


print("Rubric registry initialized with domains:", list(_RUBRIC_REGISTRY.keys()))
 



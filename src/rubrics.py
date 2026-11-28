"""Rubric registry: one rubric per domain, tuned for each difficulty.

A rubric has the same criteria at every difficulty, but each level changes:
- the weight of each criterion (easy leans on correctness and clarity, hard on
  depth, trade-offs and failure modes);
- what a strong answer shows on each criterion at that level (``looks_for``);
- what the level expects overall, what it forgives and what it never does;
- how long and how deep the suggested answer should be.

The same rubric is used to score the answer and is served by ``GET /rubrics``
so the web app can show candidates how they'll be scored before they start.
The model never invents a rubric; it only applies this one.

Domain keys match the LMS web app (``ai-ml``, ``data-science``, ``backend``,
``frontend``, ``system-design``, ``behavioral``, ``finance-accounting``). The older keys (``ai``, ``ml``,
``ai system design``, ``ml system design``, ``general``) still work.
"""

from src.agent.state import Criterion, Difficulty, Rubric

# ---- what each difficulty expects, shared by every domain ----

_LEVELS: dict[Difficulty, dict] = {
    Difficulty.easy: {
        "expectations": "Explain the core idea correctly in plain words, with one simple example.",
        "forgiven": [
            "edge cases and failure modes",
            "production or large-scale details",
            "comparing alternatives",
        ],
        "not_forgiven": [
            "factual mistakes",
            "a vague answer that never states the core idea",
        ],
        "answer_words": "60-120",
        "key_points": 3,
        "answer_style": "plain language a beginner would follow: define it, say why it matters, give one everyday example",
    },
    Difficulty.medium: {
        "expectations": "Explain how it works and why, name the main trade-off, and ground it in a realistic example.",
        "forgiven": [
            "deep internals",
            "rare edge cases",
            "large-scale or production specifics",
        ],
        "not_forgiven": [
            "factual mistakes",
            "choices with no reasoning behind them",
            "ignoring the obvious trade-off",
        ],
        "answer_words": "120-200",
        "key_points": 4,
        "answer_style": "explain the how and the why, name the main trade-off, and use one realistic example",
    },
    Difficulty.hard: {
        "expectations": (
            "Go deep: justify choices under real constraints, compare alternatives, and cover edge cases, "
            "failure modes and scale."
        ),
        "forgiven": ["not naming every possible alternative"],
        "not_forgiven": [
            "factual mistakes",
            "hand-waving past the hard part",
            "choices that are not justified",
            "missing the important failure modes or limits",
        ],
        "answer_words": "200-300",
        "key_points": 5,
        "answer_style": (
            "an expert answer: justify each choice against alternatives, cover edge cases and failure modes, "
            "and say how it holds up at scale"
        ),
    },
}

# ---- per domain: (criterion, {level: (weight, what a strong answer shows)}) ----
# Weights for each level add up to 1.

_E, _M, _H = Difficulty.easy, Difficulty.medium, Difficulty.hard

_DOMAINS: dict[str, list[tuple[str, dict[Difficulty, tuple[float, str]]]]] = {
    "ai-ml": [
        ("conceptual correctness", {
            _E: (0.45, "States what the concept is and what it is for, accurately."),
            _M: (0.35, "Accurate definitions, and how the related ideas fit together."),
            _H: (0.25, "Precise, including where the common intuition breaks down."),
        }),
        ("how it works", {
            _E: (0.25, "Gives the basic intuition behind how it works."),
            _M: (0.30, "Walks through the mechanism step by step (the algorithm or maths in words)."),
            _H: (0.30, "Explains the mechanism in depth and why it behaves the way it does."),
        }),
        ("trade-offs and limitations", {
            _E: (0.10, "Mentions a limitation if there is an obvious one."),
            _M: (0.20, "Names the main trade-off or failure mode and when it matters."),
            _H: (0.25, "Covers failure modes, assumptions and alternatives, and when to pick each."),
        }),
        ("practical application", {
            _E: (0.20, "One simple, concrete example of where it is used."),
            _M: (0.15, "A realistic use case with sensible choices of data and metrics."),
            _H: (0.20, "How it holds up in production: data issues, evaluation, monitoring and cost."),
        }),
    ],
    "data-science": [
        ("statistical and analytical correctness", {
            _E: (0.45, "Uses the right concept or test and explains it correctly."),
            _M: (0.35, "Picks the right method and states its assumptions."),
            _H: (0.25, "Checks the assumptions and explains what happens when they fail."),
        }),
        ("data handling", {
            _E: (0.20, "Mentions basic cleaning or sanity-checking of the data."),
            _M: (0.25, "Handles missing values, outliers, leakage and sampling sensibly."),
            _H: (0.25, "Spots subtle issues such as bias, leakage, skew or data that shifts over time."),
        }),
        ("experiment and metric design", {
            _E: (0.15, "Names a sensible metric."),
            _M: (0.25, "Justifies the metrics and how the results are validated."),
            _H: (0.30, "Designs the analysis end to end: sample size, confounders and guardrail metrics."),
        }),
        ("communicating insight", {
            _E: (0.20, "Explains the result in plain words."),
            _M: (0.15, "Ties the result to a decision someone would make."),
            _H: (0.20, "States the uncertainty and the business trade-off of acting on it."),
        }),
    ],
    "backend": [
        ("technical correctness", {
            _E: (0.45, "Explains the concept (e.g. HTTP, indexes, caching) accurately."),
            _M: (0.30, "Accurate, including how it behaves under real use."),
            _H: (0.20, "Precise about guarantees and edge behaviour: consistency, isolation, ordering."),
        }),
        ("design and structure", {
            _E: (0.20, "Describes a simple, sensible structure (endpoints, tables or modules)."),
            _M: (0.30, "Clear API and data-model choices, with reasons."),
            _H: (0.25, "Clean boundaries and contracts that still hold as the system grows."),
        }),
        ("performance, scale and reliability", {
            _E: (0.10, "Aware that load or failures can matter."),
            _M: (0.20, "Deals with the main bottleneck or failure case (caching, retries, indexes)."),
            _H: (0.35, "Concurrency, failure handling, consistency and scaling trade-offs, with reasons."),
        }),
        ("practical grounding", {
            _E: (0.25, "Grounds the answer in a simple example."),
            _M: (0.20, "Uses a realistic scenario and real tools."),
            _H: (0.20, "Brings in production concerns: observability, security and safe rollout."),
        }),
    ],
    "frontend": [
        ("technical correctness", {
            _E: (0.45, "Explains the concept (DOM, CSS, JavaScript or React) accurately."),
            _M: (0.30, "Accurate, including how the browser or framework actually handles it."),
            _H: (0.20, "Precise about internals: rendering, the event loop, reconciliation."),
        }),
        ("component and state design", {
            _E: (0.20, "A simple, sensible component or code structure."),
            _M: (0.30, "Clear state and data-flow choices, with reasons."),
            _H: (0.25, "An architecture that scales across a large app: state, data fetching, boundaries."),
        }),
        ("performance and edge cases", {
            _E: (0.10, "Mentions an obvious pitfall."),
            _M: (0.20, "Handles loading and error states and common performance problems."),
            _H: (0.35, "Rendering performance, race conditions and browser differences, with fixes."),
        }),
        ("user experience and accessibility", {
            _E: (0.25, "Thinks about what the user sees, plus one basic accessibility point."),
            _M: (0.20, "Builds in accessibility and responsive behaviour."),
            _H: (0.20, "Weighs UX, accessibility and performance against each other and justifies it."),
        }),
    ],
    "system-design": [
        ("requirements and scope", {
            _E: (0.30, "Restates the problem and its main requirements."),
            _M: (0.20, "Clarifies functional and non-functional requirements, with rough numbers."),
            _H: (0.15, "Pins down the scale, SLAs and constraints that drive the design."),
        }),
        ("high-level architecture", {
            _E: (0.40, "A clear set of components and how data flows between them."),
            _M: (0.30, "Sensible components, APIs and storage choices, with reasons."),
            _H: (0.25, "An end-to-end design where every component earns its place."),
        }),
        ("scalability and reliability", {
            _E: (0.15, "Aware of the main scaling need, such as a cache or load balancer."),
            _M: (0.30, "Deals with bottlenecks, replication and failure handling."),
            _H: (0.35, "Partitioning, consistency, failure modes and recovery, backed by numbers."),
        }),
        ("trade-offs", {
            _E: (0.15, "Names one trade-off."),
            _M: (0.20, "Compares options for the key decisions."),
            _H: (0.25, "Justifies each major decision against alternatives, cost and complexity."),
        }),
    ],
    "behavioral": [
        ("structure (situation, task, action, result)", {
            _E: (0.35, "Tells a clear story: the situation, what you did and how it ended."),
            _M: (0.25, "Follows situation, task, action, result, with most time on the actions."),
            _H: (0.20, "A tight story that gets to the hard decision quickly."),
        }),
        ("ownership and actions", {
            _E: (0.30, "Makes your own role clear ('I', not only 'we')."),
            _M: (0.30, "Specific actions, and why you chose them."),
            _H: (0.30, "Judgment under ambiguity or conflict, including the trade-offs you weighed."),
        }),
        ("impact and results", {
            _E: (0.20, "States the outcome."),
            _M: (0.25, "Gives a measurable result or a clear impact."),
            _H: (0.25, "Quantified impact, and the effect on the team or business beyond the task."),
        }),
        ("reflection and growth", {
            _E: (0.15, "Says what you learned."),
            _M: (0.20, "Reflects honestly, including what you would do differently."),
            _H: (0.25, "Shows self-awareness and how the lesson changed later decisions."),
        }),
    ],
    "finance-accounting": [
        ("technical accuracy", {
            _E: (0.40, "States the concept, standard or rule correctly, such as the right statement, assertion or test."),
            _M: (0.30, "Gets the treatment right and names the relevant standard or section (Ind AS / IFRS, Income-tax Act, GST, SAs)."),
            _H: (0.25, "Precise on standards, sections and thresholds, including the exceptions that apply to this case."),
        }),
        ("application and working", {
            _E: (0.25, "Gives a simple example, such as a journal entry or a small number."),
            _M: (0.30, "Applies the rule to the scenario and shows the working: entries, computation or the effect on each statement."),
            _H: (0.30, "Works through a multi-issue case end to end, with numbers, and shows the effect on profit, cash and the balance sheet."),
        }),
        ("judgement, risk and compliance", {
            _E: (0.10, "Notices the obvious risk or disclosure."),
            _M: (0.20, "Covers disclosure, control or compliance points, and what could go wrong."),
            _H: (0.25, "Makes a reasoned call where judgement is needed, weighing materiality, audit or tax risk, and ethics."),
        }),
        ("clarity of explanation", {
            _E: (0.25, "Explains plainly, the way you would to a client or a non-finance manager."),
            _M: (0.20, "A clear structure, with the conclusion first and then the reasons."),
            _H: (0.20, "A concise recommendation a partner or CFO could act on."),
        }),
    ],
    "ml": [
        ("model and algorithm reasoning", {
            _E: (0.45, "Picks a suitable model and explains what it does, correctly."),
            _M: (0.35, "Justifies the model choice and explains how it learns."),
            _H: (0.25, "Compares models and explains why one fits the problem and data best."),
        }),
        ("data handling and evaluation", {
            _E: (0.20, "Mentions splitting data into training and test sets."),
            _M: (0.25, "Sound validation, preprocessing and leakage avoidance."),
            _H: (0.25, "Handles imbalance, leakage and shifting data, with a robust validation plan."),
        }),
        ("bias-variance and overfitting", {
            _E: (0.15, "Knows what overfitting is."),
            _M: (0.20, "Diagnoses over- or underfitting and names fixes."),
            _H: (0.25, "Reasons about the bias-variance trade-off and regularisation choices in depth."),
        }),
        ("metric selection", {
            _E: (0.20, "Names a sensible metric."),
            _M: (0.20, "Justifies the metric for the problem."),
            _H: (0.25, "Links the metric to the business cost of errors and to thresholds."),
        }),
    ],
    "ai system design": [
        ("end-to-end architecture", {
            _E: (0.45, "Describes the main parts: data, model and serving."),
            _M: (0.35, "Covers data, training and serving, with reasons for each choice."),
            _H: (0.25, "A complete design where each stage's choices fit the constraints."),
        }),
        ("scalability and latency", {
            _E: (0.15, "Aware that latency or load matters."),
            _M: (0.25, "Addresses latency and throughput with concrete techniques."),
            _H: (0.25, "Quantifies latency and throughput and trades them off explicitly."),
        }),
        ("reliability and monitoring", {
            _E: (0.20, "Mentions monitoring the system."),
            _M: (0.20, "Plans monitoring, fallbacks and failure handling."),
            _H: (0.25, "Failure modes, fallbacks, evaluation in production and safe rollout."),
        }),
        ("cost and resources", {
            _E: (0.20, "Notes that compute has a cost."),
            _M: (0.20, "Weighs cost against quality for the main choices."),
            _H: (0.25, "Models the cost and justifies the trade-offs against alternatives."),
        }),
    ],
    "ml system design": [
        ("feature engineering and data pipelines", {
            _E: (0.45, "Describes where the data comes from and the main features."),
            _M: (0.30, "Designs a sensible feature pipeline, with reasons."),
            _H: (0.25, "Keeps training and serving features consistent, at scale."),
        }),
        ("training and serving", {
            _E: (0.25, "Describes how the model is trained and used."),
            _M: (0.25, "Chooses batch or online training and serving, with reasons."),
            _H: (0.25, "Infrastructure choices justified against latency, cost and freshness."),
        }),
        ("monitoring and retraining", {
            _E: (0.15, "Mentions monitoring the model."),
            _M: (0.25, "Plans metrics, alerts and when to retrain."),
            _H: (0.25, "A full monitoring and retraining loop with safe rollout."),
        }),
        ("data and concept drift", {
            _E: (0.15, "Knows the data can change over time."),
            _M: (0.20, "Detects drift and says how to respond."),
            _H: (0.25, "Distinguishes the kinds of drift and handles each one."),
        }),
    ],
    "general": [
        ("correctness", {
            _E: (0.50, "The core facts are right."),
            _M: (0.40, "Correct, including the reasoning behind it."),
            _H: (0.30, "Correct and precise, including the edge cases."),
        }),
        ("depth and completeness", {
            _E: (0.15, "Covers the main point."),
            _M: (0.30, "Covers the main points and the key trade-off."),
            _H: (0.35, "Thorough: alternatives, limits and failure modes."),
        }),
        ("clarity and structure", {
            _E: (0.20, "Easy to follow."),
            _M: (0.15, "Well organised, with a clear flow."),
            _H: (0.15, "Concise and structured despite the depth."),
        }),
        ("examples", {
            _E: (0.15, "One simple example."),
            _M: (0.15, "A realistic example."),
            _H: (0.20, "Examples drawn from real-world or production use."),
        }),
    ],
}

# Older keys that share a rubric with an LMS domain.
_ALIASES = {"ai": "ai-ml"}

DOMAINS: list[str] = list(_DOMAINS)


def _resolve(domain: str) -> str:
    key = domain.lower().strip()
    key = _ALIASES.get(key, key)
    return key if key in _DOMAINS else "general"


def fetch_rubric(domain: str, difficulty: Difficulty | str) -> Rubric:
    """The rubric for this domain at this difficulty. Unknown domains get ``general``."""
    level = Difficulty(difficulty)
    key = _resolve(domain)
    profile = _LEVELS[level]
    return Rubric(
        topic=key,
        difficulty=level,
        criteria=[
            Criterion(name=name, weight=levels[level][0], looks_for=levels[level][1])
            for name, levels in _DOMAINS[key]
        ],
        expectations=profile["expectations"],
        forgiven=profile["forgiven"],
        not_forgiven=profile["not_forgiven"],
        answer_words=profile["answer_words"],
        key_points=profile["key_points"],
        answer_style=profile["answer_style"],
        source="static_registry",
    )


def _check() -> None:
    for key, criteria in _DOMAINS.items():
        for level in Difficulty:
            total = sum(levels[level][0] for _, levels in criteria)
            assert abs(total - 1) < 1e-9, f"{key}/{level.value} weights add up to {total}"


_check()

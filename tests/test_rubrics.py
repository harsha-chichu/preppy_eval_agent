"""Rubric registry: every LMS domain has its own rubric (no silent fallback to ``general``).

Run:  uv run pytest tests
"""

import pytest

from src.agent.state import Difficulty
from src.rubrics import DOMAINS, fetch_rubric

LMS_DOMAINS = ["ai-ml", "data-science", "backend", "frontend", "system-design", "behavioral", "finance-accounting"]


@pytest.mark.parametrize("domain", LMS_DOMAINS)
@pytest.mark.parametrize("level", list(Difficulty))
def test_every_lms_domain_has_its_own_rubric(domain, level):
    rubric = fetch_rubric(domain, level)
    assert domain in DOMAINS
    assert rubric.topic == domain
    assert abs(sum(c.weight for c in rubric.criteria) - 1) < 1e-9


def test_finance_rubric_criteria():
    names = [c.name for c in fetch_rubric("finance-accounting", "hard").criteria]
    assert names == ["technical accuracy", "application and working", "judgement, risk and compliance", "clarity of explanation"]

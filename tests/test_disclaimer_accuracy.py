"""Unit tests for acute symptom triage and concerning query classification."""
import pytest
from src.pipeline import check_concerning_query


TEST_CASES = [
    ("Severe chest pain", "I am having severe chest pain radiating to my left arm", True),
    ("Choking emergency", "What to do when someone is choking?", True),
    ("Stroke symptoms", "Signs of stroke and facial numbness", True),
    ("Severe asthma attack", "Acute severe asthma attack and cannot breathe", True),
    ("Chronic asthma (Routine)", "How can I manage chronic asthma?", False),
    ("Diabetes signs (Routine)", "What are the common symptoms of type 2 diabetes?", False),
    ("Blood pressure (Routine)", "What is normal blood pressure?", False),
]


@pytest.mark.parametrize("name,query,expected_concerning", TEST_CASES)
def test_concerning_query_detection(name, query, expected_concerning):
    """Verifies that acute/emergency medical conditions trigger the concerning flag while routine questions do not."""
    actual_concerning = check_concerning_query(query)
    assert actual_concerning == expected_concerning, (
        f"Failed on '{name}' - Expected is_concerning={expected_concerning}, got {actual_concerning}"
    )

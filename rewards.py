# server/rewards.py
"""
Reward computation for the Enterprise QA environment.

Uses numeric extraction + fuzzy matching for financial answers,
with rubric-based criteria scoring for complex tasks.
"""

import re
from typing import Optional, Tuple


def extract_number(text: str) -> Optional[float]:
    """Extract the first USD/numeric value from text."""
    if not text:
        return None
    # Match: $1,234.56 | 1234.56 | 1,234 | .56
    match = re.search(r'\$?([\d,]+\.?\d*)', text.replace(",", ""))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_all_numbers(text: str) -> list:
    """Extract all numbers from text for multi-value answers."""
    matches = re.findall(r'\$?([\d,]+\.?\d*)', text.replace(",", ""))
    results = []
    for m in matches:
        try:
            results.append(float(m))
        except ValueError:
            pass
    return results


def check_rubric_criteria(response: str, rubric: str) -> float:
    """
    Parse rubric CSV field and check criteria against response.
    Rubric format: "criteria: <text>; criteria: <text>"
    Returns score 0.0–1.0 based on fraction of criteria met.
    """
    if not rubric or not response:
        return 0.0

    criteria_list = re.findall(r'criteria:\s*([^;|]+)', rubric, re.IGNORECASE)
    if not criteria_list:
        return 0.0

    response_lower = response.lower()
    met = 0
    for criterion in criteria_list:
        # Check if key numbers or phrases in criterion appear in response
        numbers = extract_all_numbers(criterion)
        keywords = re.findall(r'[a-zA-Z]{4,}', criterion.lower())

        num_match = any(
            abs(n - rn) / max(abs(n), 1e-9) <= 0.01
            for n in numbers
            for rn in extract_all_numbers(response)
        ) if numbers else True

        kw_match = sum(1 for kw in keywords if kw in response_lower)
        kw_score = kw_match / len(keywords) if keywords else 1.0

        if num_match and kw_score >= 0.5:
            met += 1

    return met / len(criteria_list)


def numeric_reward(
    predicted: str,
    gold: str,
    tolerance: float = 0.01,
) -> float:
    """
    Binary numeric reward: 1.0 if predicted number matches gold within tolerance.
    Falls back to string comparison for non-numeric answers.
    """
    pred_num = extract_number(predicted)
    gold_num = extract_number(gold)

    if pred_num is not None and gold_num is not None:
        if gold_num == 0:
            return 1.0 if abs(pred_num) < 0.001 else 0.0
        rel_err = abs(pred_num - gold_num) / abs(gold_num)
        abs_err = abs(pred_num - gold_num)
        return 1.0 if (rel_err <= tolerance or abs_err <= 0.01) else 0.0

    # Non-numeric fallback: normalized string match
    p = re.sub(r'\s+', ' ', predicted.strip().lower())
    g = re.sub(r'\s+', ' ', gold.strip().lower())
    return 1.0 if p == g else 0.0


def calculate_reward(
    generated_response: str,
    gold_response: str,
    rubric: str = "",
    tolerance: float = 0.01,
    rubric_weight: float = 0.3,
) -> float:
    """
    Composite reward combining:
      - Numeric match (70% weight): binary 1.0/0.0 based on extracted number
      - Rubric score (30% weight): fraction of rubric criteria met

    Args:
        generated_response: Model's answer
        gold_response:       Ground truth answer
        rubric:              Rubric string from data.csv (optional)
        tolerance:           Relative numeric tolerance (default 1%)
        rubric_weight:       Weight given to rubric score (0–1)

    Returns:
        Float reward in [0.0, 1.0]
    """
    if not generated_response or not gold_response:
        return 0.0

    num_score = numeric_reward(generated_response, gold_response, tolerance)
    rubric_score = check_rubric_criteria(generated_response, rubric) if rubric else num_score

    # Weighted composite
    score = (1 - rubric_weight) * num_score + rubric_weight * rubric_score
    return round(score, 4)
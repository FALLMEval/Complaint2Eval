import asyncio
import json
import os
import re
from datetime import datetime
from itertools import islice
from typing import Any, Dict, List, Tuple

from .model_interface import ModelInterface
from .prepare_json_string import prepare_json_string


def prompt_classification(Question: str, classtype: str) -> str:
    """Generate classification prompt"""

    if classtype == "temporal":
        prompt = f"""
You are an evaluation auditor tasked with classifying financial evaluation questions.

## Instructions
- Input: one financial evaluation question.
- Task: Classify whether it belongs to **pre-sale** or **post-sale**.
  - Post-sale: questions involving changes in client investment objectives, portfolio adjustments, risk tolerance changes,
    or issues arising during or after the trading process (e.g., disputes, errors, delays).
  - Pre-sale: questions involving suitability assessment, disclosure of risks, product recommendation, or any actions before execution.
- Provide a **title**: concise, without articles (e.g., "the", "a").
- Ensure the output is **valid JSON only**.

## JSON Schema
{{
  "title": "string",
  "category": "pre-sale | post-sale"
}}

## Input
{Question}
"""
    elif classtype == "complexity":
        prompt = f"""
You are an evaluation auditor tasked with classifying financial evaluation questions.

## Instructions
- Input: one financial evaluation question.
- Task: Assign the question to one of four substantive categories:
  1. **basic_regulatory_compliance** – maps directly onto statutory or rule-based obligations (e.g., FINRA Rule 2111, SEC Reg BI). These questions are binary in nature: compliance either exists or not.
  2. **professional_judgment** – requires contextual discretion and reasonable interpretation by the advisor. These questions are often guided by self-regulatory organization (SRO) standards or industry best practices rather than binding statutes.
  3. **ethical_dilemma** – involves fairness, conflicts of interest, or duties beyond explicit regulatory requirements. They often occur in ambiguous or borderline scenarios lacking clear regulatory guidance.
- Ensure the output is **valid JSON only**.

## JSON Schema
{{
  "complexity": "basic_regulatory_compliance | professional_judgment | ethical_dilemma",
  "justification": "string (required only if category = basic_regulatory_compliance, specify relevant rule or regulation such as FINRA 2111 or Reg BI)"
}}

## Input
{Question}
"""
    else:
        raise ValueError("classtype must be either 'temporal' or 'complexity'")

    return prompt


async def classify_questions(
    final_question_list: List,
    dedup_model: str,
    classtype: str,
    max_concurrent: int = 10,
) -> List[Dict]:
    """
    Run classification on each question in final_question_list using the model,
    merge parsed results into the original items, and print summary statistics.

    Returns:
        final_question_list (List[Dict]): each element updated with:
            - "title": str, short parsed title ("Unknown" if parsing failed)
            - "category": str, classification result ("pre-sale" | "post-sale" | "Unknown")
            - "raw_response": str (optional, only present if parsing failed)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = dedup_model.replace("/", "_")

    if not final_question_list:
        print("[classify_questions] Empty input list.")
        return final_question_list

    if final_question_list and isinstance(final_question_list[0], str):
        final_question_list = [{"question": q} for q in final_question_list]

    # 1) Build prompts
    prompts = [
        prompt_classification(Question=it["question"], classtype=classtype)
        for it in final_question_list
    ]

    # 2) Call the model in batch
    interface = ModelInterface()
    batch_results = await interface.batch_call(
        prompts=prompts,
        models=[dedup_model],
        temperature=0,
        max_concurrent=max_concurrent,
    )

    filename = (
        f"batch_results_logs/batch_results_stage2class_{model_name}_{timestamp}.json"
    )
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)

    # 3) Parse responses and merge with items
    for it, res in zip(final_question_list, batch_results):
        try:
            parsed = prepare_json_string(res["response"])
            if not isinstance(parsed, dict):
                raise ValueError("parsed is not a dict")

            it.update(parsed)

        except Exception as e:
            print(f"[classify_questions] Error parsing classification response: {e}")
            it.update(
                {
                    "title": "Unknown",
                    "category": "Unknown",
                    "raw_response": res.get("response", None),
                }
            )

    output_filename = f"output/stage2_classified_{model_name}_{timestamp}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_question_list, f, indent=2, ensure_ascii=False)

    print(f"Saved question_list to {output_filename}")

    return final_question_list

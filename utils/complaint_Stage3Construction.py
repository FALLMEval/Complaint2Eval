import asyncio
import json
import os
import re
from datetime import datetime
from itertools import islice
from typing import Any, Dict, List, Tuple

from .model_interface import ModelInterface
from .prepare_json_string import prepare_json_string


def prompt_define_grading_option(question: str) -> str:
    """
    Build a prompt for an LLM to generate grading rubric (A, B, C, D, NA)
    for a financial suitability evaluation question.
    """

    prompt = f"""
You are a **Financial Suitability Evaluation Specialist**.
Your task is to design a **grading rubric** (A, B, C, D, NA) for the following evaluation question:

**Question:**
{question}

### Instructions
1. Provide **five levels**: A, B, C, D, and NA.
   - A = Fully appropriate / Best practice
   - B = Minor deviation but still broadly appropriate
   - C = Material concern, questionable appropriateness
   - D = Severe violation, clearly inappropriate
   - NA = Not applicable or insufficient information
2. For each level, write a **clear, professional description** of what this looks like **in practice**:
   - How the investment professional behaves
   - What is done (or documented)
   - What is omitted or mishandled
3. Keep the tone **professional, compliance-oriented, and specific**.
4. Output in the following JSON format:

```json
{{
  "A": "description of best practice",
  "B": "description of minor deviation",
  "C": "description of material concern",
  "D": "description of severe violation",
  "NA": "description of not applicable / insufficient info"
}}
"""
    return prompt


async def construction(items, model, max_concurrent: int = 20) -> List[Dict]:
    # Extract questions and create prompts
    prompts = [prompt_define_grading_option(comp["question"]) for comp in items]

    # Batch call models
    interface = ModelInterface()
    results = await interface.batch_call(
        prompts, [model], temperature=0, max_concurrent=max_concurrent
    )

    # Insert options back into pre_sale_items
    for i, comp in enumerate(items):
        for result in results:
            if result["prompt"] == prompts[i]:
                options = prepare_json_string(result["response"])
                comp["options"] = options  # directly attach to original dict
                break

    print(f"Updated {len(items)} items with options")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_name = model.replace("/", "_")

    output_filename = f"output/stage3_constructed_{timestamp}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"Saved question_list to {output_filename}")

    return items

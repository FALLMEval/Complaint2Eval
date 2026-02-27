import asyncio
import json
from datetime import datetime

import pandas as pd
import tqdm

from .model_interface import ModelInterface
from .prepare_json_string import prepare_json_string


def prompt_complainttoQ(complaint_case: str) -> str:
    prompt = f"""You are an extraction model that converts financial complaint cases into structured JSON, focusing specifically on **suitability concerns** in financial advice and investment compliance scenarios. The goal is to produce **specific, evidence-testable criteria** (not vague principles).
## Requirements
- Extract a list of **evaluation_criteria**, each including:
  - `question`: A clear, specific, fact-focused evaluation question to assess advisor performance in future cases. (Concentrate on advisor actions, not client actions. Frame questions around the **substantive suitability**, rather than whether a generic process was completed.)
  - `explanation`: 1–2 sentences explaining how this question is derived from, and connected to, the case.
- Questions must be **generalized for future applicability**:
  - Avoid copying case-specific details (e.g., exact percentages, product names, dates) into the `question`.
  - Keep the case-specific details only in the `explanation` to illustrate why the generalized criterion applies.
  - Focus only on criteria that, if improved, would materially enhance the quality and appropriateness of future recommendations.
- Each question should remain relatively independent, avoiding overlap in scope or intent with other questions.
- Output **valid JSON only**, using the schema:
{{"evaluation_criteria": [
    {{"question": "...",
      "explanation": "..."
    }}]}}
## Complaint case:
{complaint_case}
## Example output:
{{"evaluation_criteria": [
    {{"question": "Did the advisor ensure that the concentration level of a single asset class was appropriate relative to the client's stated diversification needs?",
      "explanation": "The case involved a highly concentrated portfolio in a single type of bond, raising suitability concerns about diversification."
    }}]}}
"""
    return prompt


async def process_complaints_batch(
    df, test, models, output_file="stage1_ExtractQuestions.json", n=1
):
    """
    Batch process complaints to extract questions using multiple models

    Args:
        df: Original dataframe with complaints
        test: Number of rows to process (subset size)
        models: List of model names to use
        output_file: Output JSON file name

    Returns:
        List of processed results
    """

    print(f"Processing {test} complaints with {len(models)} models...")

    df_test = df.iloc[0:test].copy()
    df_test = pd.concat([df_test] * n, ignore_index=True)

    all_data = []
    prompts = []

    # Generate all prompts and corresponding metadata
    for _, row in df_test.iterrows():
        original_text = row.get("Allegations")
        if not original_text:
            continue

        prompt = prompt_complainttoQ(original_text)

        all_data.append(
            {
                "index": int(row["Index"]),
                "prompt": prompt,
                "original_text": original_text,
                "timestamp": datetime.now().isoformat(),
            }
        )

        prompts.append(prompt)

    print(f"Generated {len(prompts)} prompts for {len(models)} models")

    # Batch call all models
    interface = ModelInterface()
    batch_results = await interface.batch_call(prompts, models, temperature=0)

    print(f"Batch completed")

    # Process results
    results = []
    success_count = 0
    error_count = 0

    for result in tqdm.tqdm(batch_results, desc="Processing responses"):
        try:
            # Find corresponding original data
            prompt_text = result["prompt"]
            model_name = result["model"]
            response = result["response"]

            # Find metadata for this prompt
            original_data = None
            for data in all_data:
                if data["prompt"] == prompt_text:
                    original_data = data
                    break

            # Parse JSON response
            try:
                response_json = prepare_json_string(response)
                response_json["Complaint_Index"] = original_data["index"]
                response_json["Model"] = model_name
                response_json["Timestamp"] = original_data["timestamp"]
                response_json["raw_response"] = response
                results.append(response_json)
                success_count += 1

            except Exception as parse_error:
                print(
                    f"JSON parse error for index {original_data['index']}: {parse_error}"
                )
                results.append(
                    {
                        "Complaint_Index": original_data["index"],
                        "Model": model_name,
                        "Timestamp": original_data["timestamp"],
                        "raw_response": response,
                        "parse_error": str(parse_error),
                    }
                )
                error_count += 1

        except Exception as e:
            print(f"Processing error: {e}")
            results.append(
                {
                    "Model": result.get("model", "unknown"),
                    "Timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "raw_result": result,
                }
            )
            error_count += 1
    print(f"\n=== Processing Complete ===")
    print(
        f"Total results: {len(results)}, Successful: {success_count}, Errors: {error_count}"
    )
    if output_file != None:
        # Save results to JSON file
        with open(f"pipeline_output/{output_file}", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_file}")

    return results, batch_results


# Usage example:
# results = await process_complaints_batch(df, test=10, models=["openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"])

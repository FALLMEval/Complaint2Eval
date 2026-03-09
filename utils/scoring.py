import asyncio
import json
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .model_interface import ModelInterface


def format_scenario_text(
    scenario_row: pd.Series, scenario_elements: Dict[str, bool]
) -> str:
    """Format scenario text based on selected elements"""
    parts = []
    for key in [
        "Scenario_Emotion",
        "Scenario_Recommendation",
        "Scenario_Disclosure",
        "Investment_Advice",
    ]:
        if scenario_elements.get(key, False):
            value = scenario_row.get(key, "")
            if pd.notna(value) and value:
                parts.append(str(value))
    return " ".join(parts)


def create_assessment_prompt(
    client_info: str,
    scenario_row: pd.Series,
    question: str,
    scenario_elements: Dict[str, bool],
    options: Dict[str, str],
) -> str:
    """Create assessment prompt"""
    scenario_text = format_scenario_text(scenario_row, scenario_elements)

    prompt = f"""You are a professional reviewer of financial advice. Select the most appropriate response option.

## Client Profile
{client_info if client_info.strip() else "No client information provided."}

## Case
{scenario_text}

## Assessment
{question}
"""
    # Add all available options from the dict
    for label in ["A", "B", "C", "D", "NA"]:
        if label in options and options[label]:
            prompt += f"\n{label}: {options[label]}"

    prompt += '\n\nReturn only: "A", "B", "C", "D", or "NA".'
    return prompt


def generate_case_combinations(
    clients_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    cross_analysis: bool = True,
) -> List[tuple]:
    """
    Generate client-scenario combinations

    Args:
        clients_df: DataFrame of clients
        scenarios_df: DataFrame of scenarios
        cross_analysis: If True, generate all combinations (cross product).
                       If False, pair clients and scenarios one-to-one.

    Returns:
        List of (client_row, scenario_row) tuples
    """
    combinations = []

    if cross_analysis:
        # Cross analysis: all clients × all scenarios
        for _, client in clients_df.iterrows():
            for _, scenario in scenarios_df.iterrows():
                combinations.append((client, scenario))
    else:
        # Paired analysis: client[i] with scenario[i]
        min_len = min(len(clients_df), len(scenarios_df))
        if len(clients_df) != len(scenarios_df):
            print(
                f"Warning: clients ({len(clients_df)}) and scenarios ({len(scenarios_df)}) have different lengths."
            )
            print(f"Only the first {min_len} pairs will be evaluated.")

        for i in range(min_len):
            client = clients_df.iloc[i]
            scenario = scenarios_df.iloc[i]
            combinations.append((client, scenario))

    return combinations


async def evaluate_case_batch(
    clients_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    eval_data: List[Dict],
    models: List[str],
    api_key: Optional[str] = None,
    temperature: float = 0,
    max_concurrent: int = 25,
    scenario_elements: Optional[Dict[str, bool]] = None,
    run_tag: Optional[str] = None,
    num_runs: int = 1,
    cross_analysis: bool = True,
) -> pd.DataFrame:

    if scenario_elements is None:
        scenario_elements = {
            "Scenario_Emotion": True,
            "Scenario_Recommendation": True,
            "Scenario_Disclosure": True,
            "Investment_Advice": True,
        }

    model_interface = ModelInterface(api_key)
    case_combinations = generate_case_combinations(
        clients_df, scenarios_df, cross_analysis
    )
    analysis_mode = "cross" if cross_analysis else "paired"

    prompts = []
    prompt_metadata = []
    for case_idx, (client, scenario) in enumerate(case_combinations):
        for question_idx, question_data in enumerate(eval_data):
            prompt = create_assessment_prompt(
                client.get("client_text", ""),
                scenario,
                question_data.get("question", ""),
                scenario_elements,
                question_data.get("options", {}),
            )
            prompts.append(prompt)
            prompt_metadata.append(
                {
                    "case_idx": case_idx,
                    "question_index": f"Q_{question_idx + 1}",
                    "question_title": question_data.get("title", ""),
                    "question_category": question_data.get("category", ""),
                    "Client_Profile": client.get("Client_Profile", ""),
                    "Case_Name": scenario.get("Case_Name", ""),
                    "Conflict_Type": scenario.get("Conflict_Type", ""),
                }
            )

    all_final_records = []

    for i in range(num_runs):
        current_run_idx = i + 1
        current_run_tag = f"{run_tag or 'default'}_run_{current_run_idx}"
        print(
            f"\n>>> Starting Run {current_run_idx}/{num_runs} (Tag: {current_run_tag})"
        )

        results = await model_interface.batch_call(
            prompts=prompts,
            models=models,
            temperature=temperature,
            max_concurrent=max_concurrent,
        )

        run_results_dict = {}

        for result in results:
            prompt_idx = prompts.index(result["prompt"])
            metadata = prompt_metadata[prompt_idx]
            model = result["model"]

            key = (current_run_idx, metadata["case_idx"], model)

            if key not in run_results_dict:
                run_results_dict[key] = {
                    "run_time": current_run_idx,
                    "run_tag": current_run_tag,
                    "Client_Profile": metadata["Client_Profile"],
                    "Case_Name": metadata["Case_Name"],
                    "Conflict_Type": metadata["Conflict_Type"],
                    "model": model,
                    "analysis_mode": analysis_mode,
                    "scenario_elements_used": str(scenario_elements),
                    "timestamp": pd.Timestamp.now(),
                }
            run_results_dict[key][metadata["question_index"]] = result["response"]

        all_final_records.extend(list(run_results_dict.values()))

    df = pd.DataFrame(all_final_records)

    Path("output").mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{run_tag or 'default'}_{analysis_mode}_total_runs_{num_runs}_{timestamp}.csv"
    )
    df.to_csv(Path("output") / filename, index=False, encoding="utf-8-sig")

    print(f"[Evaluation Complete] Total rows generated: {len(df)}")
    return df


# Helper function to load eval data from JSON
def load_json_data(datatype, json_path: str) -> List[Dict]:
    """Load evaluation questions from JSON file and ensure index field exists"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if datatype not in ["case", "Case"]:
        for i, item in enumerate(data, start=1):
            item["index"] = f"Q_{i}"

    return data


def convert_case_data(
    case_data: List[Dict], vari_client: str, vari_scenario: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decouples raw data into independent 'Client' and 'Scenario' modules.

    To break the rigid 1:1 binding of the original data and enable modular testing. By separating 'who' (Client) from 'what' (Scenario), we can perform cross-testing.

    This allows for matrix-style stress testing beyond the original dataset."""

    clients_list = []
    scenarios_list = []

    for idx, case in enumerate(case_data):
        # Clients
        clients_list.append(
            {
                "Client_Profile": f"Client_{idx}",
                "client_text": case.get(vari_client, ""),
            }
        )

        # Scenarios
        scenarios_list.append(
            {
                "Case_Name": case.get("Name", f"Case_{idx}"),
                "Conflict_Type": case.get("Conflict_Type", ""),
                "Scenario_Emotion": case.get("Scenario_Emotion", ""),
                "Scenario_Recommendation": case.get("Scenario_Recommendation", ""),
                "Scenario_Disclosure": case.get("Scenario_Disclosure", ""),
                "Investment_Advice": case.get(vari_scenario, ""),
            }
        )

    return pd.DataFrame(clients_list), pd.DataFrame(scenarios_list)

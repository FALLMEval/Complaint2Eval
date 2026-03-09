import asyncio
import datetime
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .model_interface import ModelInterface


class EvaluationPreprocessor:
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.0):
        self.model_interface = ModelInterface(api_key)
        self.temperature = temperature
        self.valid_labels = ["A", "B", "C", "D", "N", "NA"]
        self.output_dir = Path(__file__).resolve().parent.parent / "output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.score_mapping = {"A": 4, "B": 3, "C": 2, "D": 1, "N": "NA", "NA": "NA"}

    def needs_conversion(self, response: str) -> bool:
        if pd.isna(response):
            return False
        response_str = str(response).upper().strip()
        return response_str not in self.valid_labels

    def convert_detail_invalid_responses_(
        self, df: pd.DataFrame, filename: str = None
    ) -> pd.DataFrame:
        """Clean 'response' , add 'Rating'"""
        df = df.copy()
        cleaned_values = []

        for idx, val in tqdm(df["response"].items(), desc="Cleaning responses"):
            if self.needs_conversion(val):
                prompt = (
                    f"Please classify the following response into one of [A, B, C, D, NA]. "
                    f"Only return the classification label (e.g., A, B, C, D, or NA) with no explanation.\n"
                    f'Response: "{val}"\n'
                )
                try:
                    new_val = self.model_interface.call_model(
                        "openai/gpt-4.1-mini", prompt
                    )
                    clean_val = new_val.strip().upper()
                    if clean_val in self.valid_labels:
                        cleaned_values.append(clean_val)
                    else:
                        print(
                            f"[Warning] Unexpected model output at row {idx}: {new_val}"
                        )
                        cleaned_values.append(None)
                except Exception as e:
                    print(f"[Error] Model call failed at row {idx}: {e}")
                    cleaned_values.append(None)
            else:
                cleaned_values.append(str(val).strip().upper())

        df["Rating"] = cleaned_values
        score_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "NA": "NA"}
        df["Score"] = df["Rating"].map(score_mapping)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{filename}_cleaned_{timestamp}.csv"

        try:
            df.to_csv(output_path, index=False)
            print(f"File has been saved to: {output_path}")
        except Exception as e:
            df.to_csv(f"cleaned_{timestamp}.csv", index=False)
            print(f"Save failed: {e}")
            print(f"File has been saved to: cleaned_{timestamp}.csv")
            raise

        return df

    def prompt_classification(self, response_text: str) -> str:
        prompt = f"""
You are a classification assistant. Your task is to read a user response and classify it into one of the following categories based solely on its content.
Please return **only** the classification label (A, B, C, D, or NA). Do not include any explanations.

Response to classify:
\"{response_text}\"
        """
        return prompt

    async def convert_metrix_invalid_responses(
        self,
        evaluate_result: pd.DataFrame,
        concurrent: int = 25,
        model: str = "openai/gpt-4.1-mini",
    ) -> pd.DataFrame:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        df = evaluate_result.copy()
        question_columns = [
            col
            for col in df.columns
            if col.startswith(("1_", "2_", "3_", "4_", "5_", "Q_"))
        ]

        print(f"Found {len(question_columns)} question columns: {question_columns}")

        if "run_tag" in df.columns:
            tag_values = df["run_tag"].dropna().unique()
            run_tag = tag_values[0] if len(tag_values) > 0 else "default"
        else:
            run_tag = filename or "output"

        # Collect all cells that need conversion
        prompts_to_convert = []
        cell_metadata = []

        for idx, row in df.iterrows():
            for col in question_columns:
                val = row[col]
                if pd.notna(val) and self.needs_conversion(val):
                    prompt = self.prompt_classification(val)
                    prompts_to_convert.append(prompt)
                    cell_metadata.append({"idx": idx, "col": col, "original_val": val})

        print(f"Total cells needing conversion: {len(prompts_to_convert)}")

        # Batch call for all conversions
        if prompts_to_convert:
            batch_results = await self.model_interface.batch_call(
                prompts=prompts_to_convert,
                models=[model],
                temperature=self.temperature,
                max_concurrent=concurrent,
            )
        else:
            batch_results = []

        # Create standard_label_df and apply conversions
        standard_label_df = df.copy()

        # Apply batch results
        for metadata, result in zip(cell_metadata, batch_results):
            idx = metadata["idx"]
            col = metadata["col"]
            response = result["response"]
            clean_val = response.strip().upper()
            standard_label_df.at[idx, col] = (
                clean_val if clean_val in self.valid_labels else None
            )

        # Handle cells that don't need conversion
        for idx, row in df.iterrows():
            for col in question_columns:
                val = row[col]
                if pd.notna(val) and not self.needs_conversion(val):
                    clean_val = str(val).strip().upper()
                    standard_label_df.at[idx, col] = (
                        "NA" if clean_val == "N" else clean_val
                    )

        # Convert to scores
        score_df = standard_label_df.copy()
        for col in question_columns:
            score_df[col] = standard_label_df[col].apply(
                lambda x: (
                    self.score_mapping.get(str(x).upper(), None)
                    if pd.notna(x)
                    else None
                )
            )
        # Row_Sum: Sum of all non-NA scores
        score_df["Row_Sum"] = score_df[question_columns].apply(
            lambda row: row[row != "NA"].sum(), axis=1
        )

        # Effective_Eval_Count: Count of non-NA responses
        score_df["Effective_Eval_Count"] = score_df[question_columns].apply(
            lambda row: ((row != "NA") & row.notna()).sum(), axis=1
        )

        # integrity_score: Normalized score (Row_Sum / (Effective_Eval_Count * 4))
        score_df["integrity_score"] = score_df.apply(
            lambda row: (
                (row["Row_Sum"] / (row["Effective_Eval_Count"] * 4))
                if row["Effective_Eval_Count"] > 0
                else 0
            ),
            axis=1,
        )
        # Save results
        standard_label_df.to_csv(
            f"output/{run_tag}_2standard_label_{timestamp}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        score_df.to_csv(
            f"output/{run_tag}_3score_{timestamp}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        return score_df

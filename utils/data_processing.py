"""
Data processing module - Load and process Excel files
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


class DataProcessor:
    """Data processor class for handling Excel files"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load_case_data(
        self, filename: str = "Case.xlsx"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load case data from Excel file"""
        file_path = self.data_dir / filename

        # Read scenario data from row 2 onwards (header=1)
        scenario_df = pd.read_excel(file_path, sheet_name="Scenario", header=1)

        # Read profile data from row 1 onwards (header=0)
        try:
            profile_df = pd.read_excel(file_path, sheet_name="Profile", header=0)
        except Exception as e:
            print(f"Error reading Profile sheet: {e}")
            profile_df = self._create_default_profile()

        # Clean column names
        scenario_df.columns = scenario_df.columns.str.strip()
        profile_df.columns = profile_df.columns.str.strip()

        # Process scenario information - keep all original data
        scenarios = []
        for _, row in scenario_df.iterrows():
            if pd.isna(row.iloc[0]):  # Skip empty rows
                continue

            scenario_info = {
                "Index": row.iloc[0] if len(row) > 0 else "",
                "Case_Name": row.iloc[1] if len(row) > 1 else "",
                "Conflict_Type": row.iloc[2] if len(row) > 2 else "",
                "Scenario_Emotion": row.iloc[3] if len(row) > 3 else "",
                "Scenario_Recommendation": row.iloc[4] if len(row) > 4 else "",
                "Scenario_Disclosure": row.iloc[5] if len(row) > 5 else "",
                "Investment_Advice": row.iloc[6] if len(row) > 6 else "",
            }
            scenarios.append(scenario_info)

        # Process client information
        clients = []
        for _, row in profile_df.iterrows():
            if pd.isna(row.iloc[0]):  # Skip empty rows
                continue

            client_info = {col: val for col, val in row.items() if pd.notna(val)}

            client_info["client_text"] = self._format_client_info_by_column(row)

            clients.append(client_info)

        return pd.DataFrame(clients), pd.DataFrame(scenarios)


    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data files with merged cell handling"""
        clients, scenarios = self.load_case_data()

        return {
            "clients": clients,
            "scenarios": scenarios,
        }


    def _format_client_info_by_column(self, row: pd.Series) -> str:
        """Format client info from the second column onward, skipping columns that start with '_' and null values."""
        lines = []
        for col, val in list(row.items())[1:]:
            if not col.startswith("_") and pd.notna(val):
                lines.append(f"{col}: {val}")
        return "\n".join(lines)

    def _create_default_profile(self) -> pd.DataFrame:
        """Create default client profiles if Profile sheet doesn't exist"""
        default_profiles = [
            {
                "Client_Profile": "Conservative Investor",
                "Age": "35",
                "Gender": "Male",
                "Income": "Fixed pension",
                "Assets": "Moderate savings",
                "Family": "Married, retired",
                "Education_level": "High school",
                "Investment_Objective": "Capital preservation and stable income",
            }
        ]
        return pd.DataFrame(default_profiles)

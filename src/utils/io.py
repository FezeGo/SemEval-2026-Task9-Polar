import os
import pandas as pd


def build_language_file_path(data_dir: str, lang_code: str, extension: str = "csv") -> str:
    return os.path.join(data_dir, f"{lang_code}.{extension}")


def ensure_data_dir(data_dir: str) -> None:
    if data_dir is None:
        raise ValueError("data_dir is None")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")


def read_csv_if_exists(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        return pd.DataFrame()
    return pd.read_csv(file_path)
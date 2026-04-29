import pandas as pd
from enum import Enum
from pathlib import Path


class ReaderType(Enum):
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"

    def read(self, path: Path):
        readers = {
            ReaderType.CSV: pd.read_csv,
            ReaderType.JSON: pd.read_json,
            ReaderType.EXCEL: pd.read_excel,
        }
        return readers[self](path)


class WriterType(Enum):
    CSV = "csv"
    JSON = "json"
    EXCEL = "xlsx"

    def write(self, df: pd.DataFrame, path: Path):
        writers = {
            WriterType.CSV: lambda: df.to_csv(path, index=False),
            WriterType.JSON: lambda: df.to_json(path, orient="records", indent=4),
            WriterType.EXCEL: lambda: df.to_excel(path, index=False),
        }
        return writers[self]()


class DFMethods(Enum):
    VIEW = "view"
    STRUCTURE = "structure"
    STATS = "stats"
    MISSING = "missing"
    ALL = "all"

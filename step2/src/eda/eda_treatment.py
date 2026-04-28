import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from enums import ReaderType, DFMethods
from utils.logging_utils import logger

METHODS_BY_CATEGORY = {
    DFMethods.VIEW: ["head", "tail", "sample"],
    DFMethods.STRUCTURE: ["shape", "columns", "dtypes"],
    DFMethods.STATS: ["describe"],
    DFMethods.MISSING: ["isnull"],
}

METHODS_BY_CATEGORY[DFMethods.ALL] = [
    m for methods in METHODS_BY_CATEGORY.values() for m in methods
]


class ShowWithPandas:

    def _load_content(self, path: str | Path, extension: str) -> pd.DataFrame:
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        try:
            reader_type = ReaderType(extension.lower())
        except ValueError:
            raise ValueError(f"Extension not supported: {extension}")
        return reader_type.read(path)

    def show_content(
        self,
        source: str | Path | pd.DataFrame,
        extension: str | None = None,
        category: DFMethods = DFMethods.ALL,
    ):
        if isinstance(source, pd.DataFrame):
            df = source
        else:
            if extension is None:
                extension = Path(source).suffix.lower().lstrip(".")
            df = self._load_content(source, extension)

        methods = METHODS_BY_CATEGORY[category]
        logger.info("=" * 60)
        logger.info(f"DATAFRAME ANALYSIS ({category.value.upper()})")
        logger.info("=" * 60)

        for method in methods:
            logger.info(f"\n{'-'*20} {method.upper()} {'-'*20}")
            attr = getattr(df, method)
            try:
                if callable(attr):
                    if method == "sample":
                        result = attr(5)
                    elif method == "isnull":
                        null_count = df.isnull().sum()
                        null_percent = (null_count / len(df)) * 100
                        result = pd.DataFrame(
                            {
                                "null_count": null_count,
                                "null_percent (%)": null_percent.round(2),
                            }
                        ).sort_values("null_percent (%)", ascending=False)
                    else:
                        result = attr()
                else:
                    result = attr
                logger.info(f"\n{result}")
            except Exception as e:
                logger.error(f"Error with method {method}: {e}")

    def categorical_columns(self, df: pd.DataFrame) -> list:
        return df.select_dtypes(include=["string", "object", "category"]).columns.tolist()

    def handle_nulls(self, df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=["number"]).columns
        cat_cols = self.categorical_columns(df)

        for col in cat_cols:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode[0] if not mode.empty else "missing")

        if strategy == "mean":
            numeric_fill = df[numeric_cols].mean()
        elif strategy == "median":
            numeric_fill = df[numeric_cols].median()
        else:
            numeric_fill = pd.Series(0, index=numeric_cols)

        df[numeric_cols] = df[numeric_cols].fillna(numeric_fill.fillna(0))
        return df

    def encode_categorical(self, df: pd.DataFrame, method: str = "onehot") -> pd.DataFrame:
        df = df.copy()
        cat_cols = self.categorical_columns(df)
        if not cat_cols:
            return df
        if method == "label":
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        elif method == "onehot":
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        else:
            raise ValueError("method must be 'label' or 'onehot'")
        return df

    def final_nan_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        df[df.select_dtypes(include="number").columns] = (
            df.select_dtypes(include="number").fillna(0)
        )
        df[df.select_dtypes(include="boolean").columns] = (
            df.select_dtypes(include="boolean").fillna(False)
        )
        df[df.select_dtypes(include=["object", "string", "category"]).columns] = (
            df.select_dtypes(include=["object", "string", "category"]).fillna("missing")
        )
        return df

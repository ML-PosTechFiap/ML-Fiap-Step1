import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats.mstats import winsorize
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

    def _load_content(self, path, extension: str) -> pd.DataFrame:
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        try:
            reader_type = ReaderType(extension.lower())
        except ValueError:
            raise ValueError(f"Extension not supported: {extension}")
        return reader_type.read(path)

    def show_content(self, source, extension=None, category: DFMethods = DFMethods.ALL):
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
                        result = pd.DataFrame({
                            "null_count": null_count,
                            "null_percent (%)": null_percent.round(2),
                        }).sort_values("null_percent (%)", ascending=False)
                    else:
                        result = attr()
                else:
                    result = attr
                logger.info(f"\n{result}")
            except Exception as e:
                logger.error(f"Error with method {method}: {e}")

    def convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.convert_dtypes()
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    pass
        return df

    def categorical_columns(self, df: pd.DataFrame) -> list:
        return df.select_dtypes(include=["string", "object", "category"]).columns.tolist()

    def handle_nulls(self, df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
        df = df.copy()
        if strategy == "drop":
            return df.dropna()

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

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include=["number"]).columns:
            min_val, max_val = df[col].min(), df[col].max()
            if max_val != min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
        return df

    def handle_outliers(self, df: pd.DataFrame, method: str = "winsorize") -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            return df

        z_scores = stats.zscore(df[numeric_cols].astype(float), nan_policy="omit")
        outliers = abs(z_scores) > 3

        if method == "remove":
            df = df[~outliers.any(axis=1)]
        elif method == "winsorize":
            for col in numeric_cols:
                df[col] = winsorize(df[col].astype(float), limits=[0.01, 0.01])
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

    def prepare_for_ml(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
        handle_outliers_method: str = "winsorize",
        handle_nulls_method: str = "drop",
        encoding: str = "onehot",
    ) -> pd.DataFrame:
        df = self.convert_types(df.copy())
        df = self.encode_categorical(df, method=encoding)
        df = self.handle_nulls(df, strategy=handle_nulls_method)
        df = self.handle_outliers(df, method=handle_outliers_method)
        if normalize:
            df = self.normalize_features(df)
        return self.final_nan_cleanup(df)

    def _find_src_dir(self, start_path: Path) -> Path:
        current_dir = start_path.resolve().parent
        for parent in [current_dir, *current_dir.parents]:
            if parent.name == "src":
                return parent
            if (parent / "src").is_dir():
                return parent / "src"
        return current_dir / "src"

    def generate_graphics(self, df: pd.DataFrame, save_dir: str = "graphs") -> Path:
        sns.set_theme(style="whitegrid")
        src_path = self._find_src_dir(Path(__file__))
        plots_path = src_path / save_dir
        plots_path.mkdir(parents=True, exist_ok=True)

        viz_df = df.drop(columns=[c for c in ["id"] if c in df.columns])
        numeric_cols = viz_df.select_dtypes(include=["number"]).columns

        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(viz_df[numeric_cols].corr(), annot=False, cmap="coolwarm", linewidths=0.5)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(plots_path / "correlation_heatmap.png")
            plt.close()

        for col in viz_df.columns:
            try:
                plt.figure(figsize=(10, 6))
                if pd.api.types.is_numeric_dtype(viz_df[col]):
                    sns.histplot(viz_df[col], kde=True, bins=30)
                    plt.title(f"Distribution of {col}")
                else:
                    vc = viz_df[col].value_counts()
                    if len(vc) > 20:
                        sns.countplot(y=viz_df[col], order=vc.index[:20])
                        plt.title(f"Top 20 Categories in {col}")
                    else:
                        sns.countplot(x=viz_df[col], order=vc.index)
                        plt.title(f"Distribution of {col}")
                plt.tight_layout()
                plt.savefig(plots_path / f"hist_{col}.png")
                plt.close()
            except Exception as e:
                plt.close()
                logger.error(f"Error plotting {col}: {e}")

        logger.info(f"Graphics saved to: {plots_path}")
        return plots_path

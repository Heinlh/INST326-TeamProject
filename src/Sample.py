from __future__ import annotations

from typing import Callable, Iterable, Sequence
import pandas as pd

from dataset import Dataset

__all__ = ["Sample"]


class Sample:
    """
    Lightweight wrapper around a pandas DataFrame snippet (a 'sample' of rows/cols).

    Notes
    -----
    - Use `to_dataset()` to leverage the richer Dataset utilities.
    - Use `from_dataset()` to carve a sample from an existing Dataset by rows/cols.
    """

    def __init__(self, data: pd.DataFrame, name: str = "sample"):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self._data = data.copy()
        self._name = name

    # ---------- class helpers ----------
    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        rows: slice | Sequence[int] | Callable[[pd.DataFrame], pd.Series] | None = None,
        cols: Sequence[str] | None = None,
        name: str = "sample",
    ) -> "Sample":
        """
        Create a Sample from a Dataset by selecting rows/columns.

        rows:
            - slice or sequence of integer positions (iloc)
            - callable: receives DataFrame, returns boolean Series mask
            - None: keep all rows
        cols:
            - sequence of column names to keep; None keeps all columns
        """
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset")

        df = dataset.df
        # rows
        if rows is None:
            df_sel = df
        elif isinstance(rows, slice) or (isinstance(rows, (list, tuple)) and all(isinstance(i, int) for i in rows)):
            df_sel = df.iloc[rows]
        elif callable(rows):
            mask = rows(df)
            if not isinstance(mask, pd.Series) or mask.dtype != bool:
                raise TypeError("rows callable must return a boolean pd.Series")
            df_sel = df.loc[mask]
        else:
            raise TypeError("rows must be None, slice, sequence[int], or callable(df)->bool Series")
        # cols
        if cols is not None:
            missing = [c for c in cols if c not in df_sel.columns]
            if missing:
                raise KeyError(f"cols not in DataFrame: {missing}")
            df_sel = df_sel.loc[:, list(cols)]

        return cls(df_sel, name=name)

    # ---------- properties ----------
    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns)

    @property
    def size(self) -> int:
        return int(self._data.shape[0])

    # ---------- selection (fluent, returns new Sample) ----------
    def select_rows(self, mask_or_idx: slice | Sequence[int] | pd.Series) -> "Sample":
        if isinstance(mask_or_idx, pd.Series):
            if mask_or_idx.dtype != bool:
                raise TypeError("boolean Series required when passing a Series to select_rows")
            out = self._data.loc[mask_or_idx]
        elif isinstance(mask_or_idx, slice) or (
            isinstance(mask_or_idx, (list, tuple)) and all(isinstance(i, int) for i in mask_or_idx)
        ):
            out = self._data.iloc[mask_or_idx]
        else:
            raise TypeError("select_rows expects a slice, sequence[int], or boolean Series")
        return Sample(out, name=f"{self._name}::rows")

    def select_cols(self, cols: Sequence[str]) -> "Sample":
        if not isinstance(cols, (list, tuple)):
            raise TypeError("cols must be a sequence of column names")
        missing = [c for c in cols if c not in self._data.columns]
        if missing:
            raise KeyError(f"cols not in DataFrame: {missing}")
        return Sample(self._data.loc[:, list(cols)], name=f"{self._name}::cols")

    # ---------- diagnostics ----------
    def detect_missing(self, cols: Sequence[str] | None = None) -> pd.Series:
        if cols is not None:
            missing_cols = [c for c in cols if c not in self._data.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            return self._data[cols].isna().sum()
        return self._data.isna().sum()

    def detect_duplicates(self, subset: Sequence[str] | None = None) -> pd.Index:
        if subset is not None:
            if not isinstance(subset, (list, tuple)):
                raise TypeError("subset must be a sequence of column names or None")
            missing = [c for c in subset if c not in self._data.columns]
            if missing:
                raise ValueError(f"subset contains columns not in data: {missing}")
        mask = self._data.duplicated(subset=subset, keep=False)
        return self._data.index[mask]

    # ---------- cleaning (in-place, chainable) ----------
    def clean_strings(
        self,
        cols: Sequence[str],
        *,
        strip: bool = True,
        lower: bool = True,
        collapse_spaces: bool = True,
    ) -> "Sample":
        if not isinstance(cols, (list, tuple)):
            raise TypeError("cols must be a sequence of column names")
        for col in cols:
            if col not in self._data.columns:
                raise KeyError(f"{col!r} not in sample")
            series = self._data[col].astype("string")
            if strip:
                series = series.str.strip()
            if lower:
                series = series.str.lower()
            if collapse_spaces:
                series = series.str.replace(r"\s+", " ", regex=True)
            self._data[col] = series
        return self

    # ---------- conversions & summaries ----------
    def to_dataset(self, name: str | None = None) -> Dataset:
        """Convert this Sample to a Dataset to reuse validation/plotting utilities."""
        return Dataset(self._data, name=name or f"{self._name}::dataset")

    def summarize(self, include_categoricals: bool = True) -> pd.DataFrame:
        """Return numeric/categorical summary by delegating to Dataset."""
        return self.to_dataset().calculate_statistical_summary(include_categoricals=include_categoricals)

    def as_records(self) -> list[dict]:
        """Return the sample as a list of Python dicts (records)."""
        return self._data.to_dict(orient="records")

    # ---------- dunder ----------
    def __str__(self) -> str:
        return f"Sample(name={self._name}, shape={self._data.shape})"

    def __repr__(self) -> str:
        return f"Sample(name={self._name!r}, shape={self._data.shape})"

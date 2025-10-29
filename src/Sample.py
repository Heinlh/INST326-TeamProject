from __future__ import annotations

from typing import Sequence
import pandas as pd


class Sample:
    """A logical subset (row-slice) of a parent dataset for focused cleaning/QA."""

    def __init__(self, data: pd.DataFrame, name: str = "sample"):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self._data = data.copy()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()

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

    def __str__(self) -> str:
        return f"Sample(name={self._name}, shape={self._data.shape})"

    def __repr__(self) -> str:
        return f"Sample(name={self._name!r}, shape={self._data.shape})"

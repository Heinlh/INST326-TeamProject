from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Tuple

import pandas as pd

from dataset import Dataset


class Experiment:
    """Wraps a `Dataset` with experiment metadata and common prep steps."""

    def __init__(self, experiment_id: str, title: str):
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        self._experiment_id = experiment_id
        self._title = title
        self._dataset: Dataset | None = None

    # ---- properties ----
    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def title(self) -> str:
        return self._title

    @property
    def dataset(self) -> Dataset | None:
        return self._dataset

    # ---- instance methods ----
    def attach_dataset(self, df: pd.DataFrame, name: str = "dataset") -> Dataset:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self._dataset = Dataset(df, name=name)
        return self._dataset

    def load_csv(self, csv_file: str | Path) -> Dataset:
        """Convenience CSV loader (IO is also available via Researcher.read_csv)."""
        p = Path(csv_file)
        df = pd.read_csv(p)
        self._dataset = Dataset(df, name=p.name)
        return self._dataset

    def clean(self, columns_to_clean: Sequence[str] | None = None, case: str = "snake") -> Dataset:
        if self._dataset is None:
            raise RuntimeError("No dataset attached")
        ds = self._dataset.standardize_column_names(case=case)
        if columns_to_clean:
            ds = ds.clean_strings(columns_to_clean)
        self._dataset = ds
        return ds

    def qa_report(self) -> dict:
        if self._dataset is None:
            raise RuntimeError("No dataset attached")
        return {
            "missing": self._dataset.detect_missing(),
            "duplicates": self._dataset.detect_duplicates(),
            "summary": self._dataset.calculate_statistical_summary(),
        }

    def enforce_schema(
        self,
        schema: Mapping[str, Mapping[str, object]],
        strict: bool = False,
    ) -> Tuple[Dataset, dict]:
        if self._dataset is None:
            raise RuntimeError("No dataset attached")
        ds_new, report = self._dataset.enforce_schema(schema, strict=strict)
        self._dataset = ds_new
        return ds_new, report

    def __str__(self) -> str:
        ds = f"{self._dataset}" if self._dataset else "None"
        return f"Experiment(id={self._experiment_id}, title={self._title}, dataset={ds})"

    def __repr__(self) -> str:
        return f"Experiment(experiment_id={self._experiment_id!r}, title={self._title!r})"

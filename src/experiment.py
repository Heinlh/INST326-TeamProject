from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import pandas as pd

from dataset import Dataset

__all__ = ["AbstractExperiment", "Experiment"]


class AbstractExperiment(ABC):
    """ABC for all experiments: enforces a shared interface and common utilities."""

    def __init__(self, experiment_id: str, title: str):
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        self._experiment_id = experiment_id
        self._title = title
        self._dataset: Dataset | None = None  # composition: experiment has-a Dataset

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

    # ---- abstract (polymorphic) API ----
    @abstractmethod
    def access_policy(self) -> str:
        """Who can access this experiment's data/results."""
        raise NotImplementedError

    @abstractmethod
    def process(self) -> Dataset:
        """Run the experiment's core processing pipeline; must return a Dataset."""
        raise NotImplementedError

    # ---- shared utilities (composition with Dataset) ----
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(experiment_id={self._experiment_id!r}, title={self._title!r})"


class Experiment(AbstractExperiment):
    """Concrete base experiment with a default (overridable) processing pipeline.

    Subclasses should override `access_policy()` and/or `process()` to provide
    type-specific behavior (e.g., LabExperiment, FieldStudy, Survey).
    """

    def __init__(self, experiment_id: str, title: str):
        super().__init__(experiment_id, title)

    # ---- polymorphic methods (defaults) ----
    def access_policy(self) -> str:
        return "UMD internal; anonymized results public."

    def process(self) -> Dataset:
        """Default process: standardize columns + light text cleanup (if any)."""
        if self._dataset is None:
            raise RuntimeError("No dataset attached")
        ds = self._dataset.standardize_column_names(case="snake")
        # clean all object/string columns if present
        obj_cols = [c for c in ds.df.columns if ds.df[c].dtype == "object"]
        if obj_cols:
            ds = ds.clean_strings(obj_cols, strip=True, lower=True, collapse_spaces=True)
        self._dataset = ds
        return ds

    # ---- convenience retained from your original class ----
    def clean(self, columns_to_clean: Sequence[str] | None = None, case: str = "snake") -> Dataset:
        if self._dataset is None:
            raise RuntimeError("No dataset attached")
        ds = self._dataset.standardize_column_names(case=case)
        if columns_to_clean:
            ds = ds.clean_strings(columns_to_clean)
        self._dataset = ds
        return ds

    def __str__(self) -> str:
        ds = f"{self._dataset}" if self._dataset else "None"
        return f"{self.__class__.__name__}(id={self._experiment_id}, title={self._title}, dataset={ds})"

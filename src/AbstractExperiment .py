from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple, Optional
from pathlib import Path
import pandas as pd


class AbstractExperiment(ABC):

    def __init__(self, experiment_id: str, title: str):
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        self._experiment_id = experiment_id
        self._title = title
        self._dataset: Dataset | None = None

 
    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def title(self) -> str:
        return self._title

    @property
    def dataset(self) -> Dataset | None:
        return self._dataset

    
    @abstractmethod
    def access_policy(self) -> str:
        """Who may access the data/results for this experiment type."""
        ...

    @abstractmethod
    def process(self) -> Dataset:
        """Run the type-specific processing pipeline. Must return a Dataset."""
        ...

    
    def attach_dataset(self, df: pd.DataFrame, name: str = "dataset") -> Dataset:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self._dataset = Dataset(df, name=name)
        return self._dataset

    def load_csv(self, csv_file: str | Path) -> Dataset:
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

    def enforce_schema(self, schema: Mapping[str, Mapping[str, object]], strict: bool = False) -> Tuple[Dataset, dict]:
        if self._dataset is None:
            raise RuntimeError("No dataset attached")
        ds_new, report = self._dataset.enforce_schema(schema, strict=strict)
        self._dataset = ds_new
        return ds_new, report

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(experiment_id={self._experiment_id!r}, title={self._title!r})"

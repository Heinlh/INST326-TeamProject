from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, List
from abc import ABC, abstractmethod

import pandas as pd

from dataset import Dataset


class AbstractExperiment(ABC):
    """
    Abstract base class for all experiments.

    Enforces a common interface for:
    - access_policy(): who can access data/results
    - process_dataset(): how data is processed
    - summary(): human-readable description

    Also provides shared behavior for attaching a Dataset.
    """

    def __init__(self, experiment_id: str, title: str) -> None:
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")

        self._experiment_id = experiment_id
        self._title = title
        self._dataset: Dataset | None = None

    # ---------- properties ----------

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def title(self) -> str:
        return self._title

    @property
    def dataset(self) -> Dataset | None:
        """Dataset attached to this experiment (may be None)."""
        return self._dataset

    # ---------- shared concrete behavior ----------

    def attach_dataset(self, data: pd.DataFrame | Dataset, name: str | None = None) -> Dataset:
        """
        Attach a Dataset to this experiment.

        Parameters
        ----------
        data : pandas.DataFrame | Dataset
            Raw DataFrame or an existing Dataset instance.
        name : str | None
            Optional name for the dataset (used when data is a DataFrame).

        Returns
        -------
        Dataset
            The Dataset object stored on the experiment.
        """
        if isinstance(data, Dataset):
            ds = data
        else:
            if not isinstance(data, pd.DataFrame):
                raise TypeError("data must be a pandas DataFrame or a Dataset")
            ds = Dataset(data, name=name or "dataset")

        self._dataset = ds
        return ds

    # ---------- abstract interface ----------

    @abstractmethod
    def access_policy(self) -> str:
        """Return a human-readable description of who can access this experiment's data/results."""
        raise NotImplementedError

    @abstractmethod
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """
        Process the given Dataset and return a new Dataset.

        Implementation is experiment-type specific.
        """
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> str:
        """Return a human-readable summary of this experiment."""
        raise NotImplementedError

    # ---------- string repr ----------

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(experiment_id={self._experiment_id!r}, title={self._title!r})"


class Experiment(AbstractExperiment):
    """
    Concrete base experiment implementation.

    Provides a default access policy and a generic processing pipeline that
    subclasses can extend or override.
    """

    def __init__(self, experiment_id: str, title: str) -> None:
        super().__init__(experiment_id, title)

    def access_policy(self) -> str:
        # Generic default; subclasses will override with more specific text.
        return "UMD internal; anonymized results public."

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """
        Base processing pipeline: light, generic operations.

        Here we simply create a shallow copy with a new name to demonstrate
        polymorphism and pipeline behavior.
        """
        df_copy = dataset.df.copy()
        new_name = f"{dataset.name}::base_processed"
        out = Dataset(df_copy, name=new_name)
        # keep reference to last processed dataset
        self._dataset = out
        return out

    def summary(self) -> str:
        ds_info = f"{self.dataset.name}, shape={self.dataset.df.shape}" if self.dataset else "None"
        return f"Experiment {self.experiment_id} - {self.title} (dataset={ds_info})"


class LabExperiment(Experiment):
    """
    Laboratory experiment with biosafety level and stricter access policy.
    """

    def __init__(self, experiment_id: str, title: str, biosafety_level: int = 1) -> None:
        super().__init__(experiment_id, title)
        self.biosafety_level = int(biosafety_level)

    def access_policy(self) -> str:
        # NOTE: tests expect "lab-restricted" substring (case-insensitive)
        return (
            f"Lab-restricted access (BSL-{self.biosafety_level}); "
            "only approved lab personnel may access raw data."
        )

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """
        Lab-specific processing: here we just rename with a lab-specific suffix.

        In a real system, this might:
        - enforce numeric types on measurement columns
        - drop invalid rows
        - apply calibration factors
        """
        df_copy = dataset.df.copy()
        new_name = f"{dataset.name}::lab_processed[BSL-{self.biosafety_level}]"
        out = Dataset(df_copy, name=new_name)
        self._dataset = out
        return out

    def summary(self) -> str:
        return (
            f"LabExperiment {self.experiment_id} - {self.title} "
            f"(BSL-{self.biosafety_level}, dataset={'attached' if self.dataset else 'none'})"
        )


class FieldStudy(Experiment):
    """
    Field study with region information and field-specific policy.
    """

    def __init__(self, experiment_id: str, title: str, region: str = "MD") -> None:
        super().__init__(experiment_id, title)
        self.region = region

    def access_policy(self) -> str:
        # NOTE: tests expect "field conditions" substring (case-insensitive)
        return (
            "Data access governed by field conditions and partner agreements; "
            f"region={self.region}."
        )

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """
        Field-specific processing: rename to indicate field pipeline.

        A real implementation might:
        - normalize timestamps
        - coerce GPS coordinates
        - handle missing sensor readings
        """
        df_copy = dataset.df.copy()
        new_name = f"{dataset.name}::field_processed[{self.region}]"
        out = Dataset(df_copy, name=new_name)
        self._dataset = out
        return out

    def summary(self) -> str:
        return (
            f"FieldStudy {self.experiment_id} - {self.title} "
            f"(region={self.region}, dataset={'attached' if self.dataset else 'none'})"
        )


class Survey(Experiment):
    """
    Survey-based experiment with consent-tracking behavior.
    """

    def __init__(self, experiment_id: str, title: str, consent_col: str = "consent") -> None:
        super().__init__(experiment_id, title)
        self.consent_col = consent_col

    def access_policy(self) -> str:
        # NOTE: tests expect "anonymous" substring (case-insensitive)
        return (
            "Responses stored with anonymous identifiers; raw data restricted to IRB-approved team."
        )

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """
        Survey-specific processing: typically would enforce consent and anonymization.

        Here we only rename the dataset to show polymorphic behavior.
        """
        df_copy = dataset.df.copy()
        new_name = f"{dataset.name}::survey_processed[{self.consent_col}]"
        out = Dataset(df_copy, name=new_name)
        self._dataset = out
        return out

    def summary(self) -> str:
        return (
            f"Survey {self.experiment_id} - {self.title} "
            f"(consent_col={self.consent_col}, dataset={'attached' if self.dataset else 'none'})"
        )


@dataclass
class ResearchProject:
    """
    Composition root: a project *has* multiple experiments.

    This models a 'has-many' relationship rather than inheritance.
    """
    project_id: str
    title: str
    experiments: List[AbstractExperiment] = field(default_factory=list)

    def add_experiment(self, experiment: AbstractExperiment) -> None:
        if not isinstance(experiment, AbstractExperiment):
            raise TypeError("experiment must be an AbstractExperiment")
        self.experiments.append(experiment)

    def summary(self) -> str:
        """
        Return a multi-line text summary of the project and its experiments.
        """
        lines = [
            f"ResearchProject {self.project_id} - {self.title}",
            f"Experiments ({len(self.experiments)}):",
        ]
        for exp in self.experiments:
            lines.append(f"- {exp.__class__.__name__} {exp.experiment_id}: {exp.title}")
        return "\n".join(lines)


def render_overview(experiments: Sequence[AbstractExperiment]) -> str:
    """
    Render a combined overview of experiments.

    This function demonstrates polymorphism by calling access_policy() on each
    experiment without knowing its concrete type.
    """
    lines: List[str] = []
    for exp in experiments:
        policy = exp.access_policy()
        lines.append(
            f"{exp.__class__.__name__}({exp.experiment_id}) • {exp.title} • policy={policy}"
        )
    return "\n".join(lines)

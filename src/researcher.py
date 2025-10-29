from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union
from urllib.parse import urlparse
import csv

import pandas as pd
import requests

from dataset import Dataset


PathLike = Union[str, Path]


class Researcher:
    """Represents a researcher with a local workspace and data IO utilities."""

    def __init__(self, name: str, workspace: PathLike = "data"):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self._name = name
        self._workspace = Path(workspace).expanduser().resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return self._name

    @property
    def workspace(self) -> Path:
        return self._workspace

    # ---- directory + download
    def ensure_dir(self, relative: PathLike = "") -> Path:
        p = (self._workspace / Path(relative)).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def download_file(self, url: str, dest_subdir: PathLike = "") -> Path:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")
        dest = self.ensure_dir(dest_subdir)
        fname = Path(parsed.path).name or "download.bin"
        final = dest / fname
        tmp = final.with_suffix(final.suffix + ".partial")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        tmp.replace(final)
        return final

    # ---- CSV read/append
    def read_csv(self, csv_file: PathLike, **read_csv_kwargs) -> Dataset:
        p = Path(csv_file)
        if not p.is_absolute():
            p = self._workspace / p
        p = p.expanduser()
        if p.suffix.lower() != ".csv":
            candidate = p.with_suffix(".csv")
            if candidate.exists():
                p = candidate
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if not p.is_file():
            raise ValueError(f"Not a file: {p}")
        df = pd.read_csv(p, **read_csv_kwargs)
        return Dataset(df, name=p.name)

    def append_csv(self, rows: Iterable[Mapping[str, Any]], csv_file: PathLike) -> Path:
        if not isinstance(csv_file, (str, Path)):
            raise TypeError("csv_file must be a str or Path")

        materialized = list(rows)
        if not materialized:
            raise ValueError("rows is empty; nothing to write")
        if not all(hasattr(r, "keys") for r in materialized):
            raise TypeError("rows must be an iterable of dict-like mappings")

        target = Path(csv_file)
        if not target.is_absolute():
            target = self._workspace / target
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        df_new = pd.DataFrame(materialized)

        header = True
        if target.exists():
            with target.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                existing_cols = next(reader, None)
            if existing_cols:
                for c in existing_cols:
                    if c not in df_new:
                        df_new[c] = pd.NA
                df_new = df_new[existing_cols]
                header = False

        df_new.to_csv(
            target,
            mode="a",
            header=header,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        return target

    # ---- ethics orchestration
    def ethics_report(
        self,
        dataset: Dataset,
        *,
        pii_cols: Sequence[str] | None = None,
        consent_col: str | None = None,
    ) -> dict:
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset")
        return dataset.validate_research_ethics_compliance(
            pii_cols=pii_cols, consent_col=consent_col
        )

    def __str__(self) -> str:
        return f"Researcher(name={self._name}, workspace={self._workspace})"

    def __repr__(self) -> str:
        return f"Researcher(name={self._name!r}, workspace={self._workspace!r})"

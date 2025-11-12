from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union, Optional
from urllib.parse import urlparse
import csv
import hashlib
import time

import pandas as pd
import requests

from dataset import Dataset

__all__ = ["Researcher", "PathLike"]

PathLike = Union[str, Path]
CHUNK_SIZE = 8192


class Researcher:
    """Represents a researcher with a local workspace and data IO utilities.

    Notes
    -----
    - IO and filesystem concerns live here.
    - Data validation/cleaning lives in `Dataset` and analysis logic in `Analysis`.
    - This class demonstrates composition: it *creates/returns* `Dataset` objects.
    """

    def __init__(self, name: str, workspace: PathLike = "data"):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self._name = name
        self._workspace = Path(workspace).expanduser().resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)

    # ---------- properties ----------
    @property
    def name(self) -> str:
        return self._name

    @property
    def workspace(self) -> Path:
        return self._workspace

    # ---------- workspace helpers ----------
    def resolve_path(self, relative: PathLike = "") -> Path:
        """Resolve a path inside the workspace (creates no files/directories)."""
        p = Path(relative)
        return (self._workspace / p) if not p.is_absolute() else p

    def ensure_dir(self, relative: PathLike = "") -> Path:
        """Ensure a directory exists under the workspace and return its Path."""
        p = (self._workspace / Path(relative)).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def list_workspace(self, pattern: str = "*") -> list[Path]:
        """List files in the workspace matching a glob pattern."""
        return sorted(self._workspace.glob(pattern))

    # ---------- network/download ----------
    def download_file(
        self,
        url: str,
        dest_subdir: PathLike = "",
        *,
        filename: Optional[str] = None,
        overwrite: bool = False,
        sha256: Optional[str] = None,
        timeout: int = 30,
        retries: int = 2,
        backoff: float = 1.0,
    ) -> Path:
        """Download a file to the workspace (with optional checksum & retries).

        Parameters
        ----------
        url : str
            HTTP/HTTPS URL.
        dest_subdir : str | Path
            Subdirectory under the workspace to place the file.
        filename : str | None
            Override the destination filename. If None, infer from URL.
        overwrite : bool
            If False and final file exists, skip download and return it.
        sha256 : str | None
            If provided, verify the downloaded file's sha256 hex digest.
        timeout : int
            Request timeout per attempt (seconds).
        retries : int
            Number of additional retry attempts on transient network errors.
        backoff : float
            Seconds to sleep between retries.

        Returns
        -------
        Path
            Path to the downloaded (or existing) file.
        """
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")

        dest_dir = self.ensure_dir(dest_subdir)
        inferred = Path(parsed.path).name or "download.bin"
        final = dest_dir / (filename or inferred)

        if final.exists() and not overwrite:
            # Optionally verify checksum on existing file
            if sha256 and not self._verify_sha256(final, sha256):
                raise ValueError(f"Existing file checksum mismatch for {final}")
            return final

        tmp = final.with_suffix(final.suffix + ".partial")

        attempt = 0
        while True:
            try:
                with requests.get(url, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                tmp.replace(final)
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt >= retries:
                    raise
                time.sleep(backoff * (attempt + 1))
                attempt += 1

        if sha256 and not self._verify_sha256(final, sha256):
            final.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for downloaded file: {final}")

        return final

    @staticmethod
    def _verify_sha256(path: Path, expected_hex: str) -> bool:
        """Return True if sha256(file) == expected_hex (case-insensitive)."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                h.update(chunk)
        return h.hexdigest().lower() == expected_hex.lower()

    # ---------- CSV <-> Dataset ----------
    def read_csv(self, csv_file: PathLike, **read_csv_kwargs) -> Dataset:
        """Read a CSV (from workspace by default) and wrap it in a Dataset."""
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

    def read_dataset(self, csv_file: PathLike, **read_csv_kwargs) -> Dataset:
        """Alias of read_csv for semantic clarity in pipelines/experiments."""
        return self.read_csv(csv_file, **read_csv_kwargs)

    def write_csv(self, dataset: Dataset, csv_file: PathLike, *, mode: str = "w") -> Path:
        """Persist a Dataset to CSV under the workspace (default overwrite)."""
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset")
        target = self.resolve_path(csv_file).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        df = dataset.df
        df.to_csv(target, mode=mode, index=False, encoding="utf-8", lineterminator="\n")
        return target

    def append_csv(self, rows: Iterable[Mapping[str, Any]], csv_file: PathLike) -> Path:
        """Append row dicts to a CSV, aligning columns to an existing header if present."""
        if not isinstance(csv_file, (str, Path)):
            raise TypeError("csv_file must be a str or Path")

        materialized = list(rows)
        if not materialized:
            raise ValueError("rows is empty; nothing to write")
        if not all(hasattr(r, "keys") for r in materialized):
            raise TypeError("rows must be an iterable of dict-like mappings")

        target = self.resolve_path(csv_file).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        df_new = pd.DataFrame(materialized)

        header = True
        if target.exists():
            with target.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                existing_cols = next(reader, None)
            if existing_cols:
                # align to existing header
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

    # ---------- ethics orchestration ----------
    def ethics_report(
        self,
        dataset: Dataset,
        *,
        pii_cols: Sequence[str] | None = None,
        consent_col: str | None = None,
    ) -> dict:
        """Delegate to Dataset.validate_research_ethics_compliance (composition)."""
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset")
        return dataset.validate_research_ethics_compliance(
            pii_cols=pii_cols, consent_col=consent_col
        )

    # ---------- dunder ----------
    def __str__(self) -> str:
        return f"Researcher(name={self._name}, workspace={self._workspace})"

    def __repr__(self) -> str:
        return f"Researcher(name={self._name!r}, workspace={self._workspace!r})"

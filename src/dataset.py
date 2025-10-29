from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple, List
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dataset:
    

    def __init__(self, data: pd.DataFrame, name: str = "dataset"):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self._df = data.copy()
        self._name = name

    
    @property
    def name(self) -> str:
        return self._name

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    
    def detect_missing(self, cols: Sequence[str] | None = None) -> pd.Series:
        if cols is not None:
            self._assert_cols(cols)
            return self._df[cols].isna().sum()
        return self._df.isna().sum()

    def detect_duplicates(self, subset: Sequence[str] | None = None) -> pd.Index:
        if subset is not None:
            if not isinstance(subset, (list, tuple)):
                raise TypeError("subset must be a sequence of column names or None")
            missing = [c for c in subset if c not in self._df.columns]
            if missing:
                raise ValueError(f"subset contains columns not in df: {missing}")
        mask = self._df.duplicated(subset=subset, keep=False)
        return self._df.index[mask]

    
    def standardize_column_names(self, case: str = "snake") -> "Dataset":
        if not isinstance(case, str):
            raise TypeError("case must be a string")
        case = case.lower().strip()
        allowed = {"snake", "kebab", "lower", "upper", "title"}
        if case not in allowed:
            raise ValueError(f"case must be one of {sorted(allowed)}")

        def normalize(s: str) -> str:
            s = "" if s is None else str(s)
            s = unicodedata.normalize("NFKD", s)
            s = "".join(ch for ch in s if not unicodedata.combining(ch))
            s = re.sub(r"\s+", " ", s.strip())
            s = re.sub(r"[^0-9A-Za-z]+", " ", s)
            parts = [p for p in s.split(" ") if p]
            if not parts:
                return ""
            if case == "snake":
                return "_".join(p.lower() for p in parts)
            if case == "kebab":
                return "-".join(p.lower() for p in parts)
            if case == "lower":
                return "".join(p.lower() for p in parts)
            if case == "upper":
                return "".join(p.upper() for p in parts)
            if case == "title":
                return "".join(p.capitalize() for p in parts)
            return s  # fallback

        seen: Dict[str, int] = {}

        def uniq(s: str) -> str:
            base = s or "col"
            n = seen.get(base, 0)
            seen[base] = n + 1
            return f"{base}_{n}" if n else base

        new_cols = [uniq(normalize(c)) for c in self._df.columns]
        out = self._df.rename(columns=dict(zip(self._df.columns, new_cols)))
        return Dataset(out, name=f"{self._name}::standardized[{case}]")

    def clean_strings(
        self,
        cols: Sequence[str],
        *,
        strip: bool = True,
        lower: bool = True,
        collapse_spaces: bool = True,
    ) -> "Dataset":
        self._assert_cols(cols)
        out = self._df.copy()

        def _clean_cell(x):
            if pd.isna(x):
                return x
            s = str(x)
            if strip:
                s = s.strip()
            if collapse_spaces:
                s = re.sub(r"\s+", " ", s)
            if lower:
                s = s.lower()
            return s

        for c in cols:
            out[c] = out[c].map(_clean_cell)
        return Dataset(out, name=f"{self._name}::clean_strings")

    def anonymize(self, cols: Sequence[str]) -> "Dataset":
        self._assert_cols(cols, exc=KeyError)
        out = self._df.copy(deep=True)
        for c in cols:
            out.loc[out[c].notna(), c] = "***"
        return Dataset(out, name=f"{self._name}::anonymized")

    
    def validate_research_ethics_compliance(
        self,
        pii_cols: Sequence[str] | None = None,
        consent_col: str | None = None,
    ) -> dict:
        if pii_cols is None:
            pii_cols = []
        issues: list[str] = []

        consent = (
            self._df[consent_col]
            .astype(str)
            .str.lower()
            .isin({"1", "true", "yes", "y"})
            if (consent_col and consent_col in self._df.columns)
            else None
        )

        for c in pii_cols:
            if c not in self._df.columns or not self._df[c].notna().any():
                continue
            if consent is None:
                issues.append(f"PII detected in '{c}' without a consent column")
            else:
                n = int((self._df[c].notna() & ~consent).sum())
                if n:
                    issues.append(f"{n} row(s) with PII in '{c}' without consent")

        return {"compliant": not issues, "issues": issues}

    
    def calculate_statistical_summary(self, include_categoricals: bool = True) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        num_cols = self._df.select_dtypes(include=["number"]).columns
        if len(num_cols):
            num = self._df[num_cols].describe().T
            num["na"] = self._df[num_cols].isna().sum()
            num["distinct"] = self._df[num_cols].nunique(dropna=True)
            frames.append(num)

        if include_categoricals:
            obj_cols = self._df.select_dtypes(include=["object", "string", "category"]).columns
            if len(obj_cols):
                cats = pd.DataFrame(index=obj_cols)
                cats["count"] = self._df[obj_cols].notna().sum()
                cats["na"] = self._df[obj_cols].isna().sum()
                cats["distinct"] = self._df[obj_cols].nunique(dropna=True)
                cats["top"] = self._df[obj_cols].apply(
                    lambda s: s.mode(dropna=True).astype(str).iloc[0] if s.dropna().size else None
                )
                cats["freq"] = self._df[obj_cols].apply(
                    lambda s: int(s.value_counts(dropna=True).iloc[0]) if s.dropna().size else 0
                )
                frames.append(cats)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=0).sort_index()

    
    def enforce_schema(
        self,
        schema: Mapping[str, Mapping[str, Any]],
        *,
        strict: bool = False,
    ) -> Tuple["Dataset", Dict[str, Any]]:
        if not isinstance(schema, dict):
            raise TypeError("schema must be a mapping (dict) of column -> rules")

        supported_dtypes = {"int", "float", "string", "bool", "datetime"}
        out = self._df.copy()
        report: Dict[str, Any] = {
            "coercions": {},
            "invalid_nulls": {},
            "invalid_values": {},
            "errors": [],
        }

        def coerce_bool(series: pd.Series) -> pd.Series:
            mapping_true = {"true", "t", "1", "yes", "y"}
            mapping_false = {"false", "f", "0", "no", "n"}

            def _to_bool(x):
                if pd.isna(x):
                    return np.nan
                if isinstance(x, bool):
                    return x
                s = str(x).strip().lower()
                if s in mapping_true:
                    return True
                if s in mapping_false:
                    return False
                return np.nan

            coerced = series.map(_to_bool)
            return coerced.astype("boolean")

        for col, rules in schema.items():
            if not isinstance(rules, dict):
                report["errors"].append(f"Rules for column '{col}' must be a dict.")
                continue

            dtype = rules.get("dtype")
            if dtype not in supported_dtypes:
                report["errors"].append(
                    f"Column '{col}': unsupported dtype '{dtype}'. Supported: {sorted(supported_dtypes)}"
                )
                continue

            nullable = bool(rules.get("nullable", False))
            dt_fmt = rules.get("datetime_format", None)

            if col not in out.columns:
                report["errors"].append(f"Missing expected column: '{col}'")
                out[col] = pd.Series([pd.NA] * len(out), index=out.index)

            before = out[col].copy()

            try:
                if dtype == "string":
                    out[col] = out[col].astype("string")
                elif dtype == "int":
                    numeric = pd.to_numeric(out[col], errors="coerce")
                    out[col] = numeric.astype("Int64" if nullable or numeric.isna().any() else "int64")
                elif dtype == "float":
                    out[col] = pd.to_numeric(out[col], errors="coerce").astype("Float64")
                elif dtype == "bool":
                    out[col] = coerce_bool(out[col])
                    if not nullable and not out[col].isna().any():
                        out[col] = out[col].astype(bool)
                elif dtype == "datetime":
                    if dt_fmt is None:
                        out[col] = pd.to_datetime(out[col], errors="coerce", utc=False)
                    else:
                        fmts = dt_fmt if isinstance(dt_fmt, (list, tuple)) else [dt_fmt]
                        parsed = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
                        raw = out[col]
                        for fmt in fmts:
                            mask = parsed.isna()
                            if not mask.any():
                                break
                            parsed.loc[mask] = pd.to_datetime(
                                raw.loc[mask], format=fmt, errors="coerce", utc=False
                            )
                        still = parsed.isna()
                        if still.any():
                            parsed.loc[still] = pd.to_datetime(raw.loc[still], errors="coerce", utc=False)
                        out[col] = parsed

            except Exception as e:
                report["errors"].append(f"Coercion error for '{col}': {e}")
                continue

            coerced_count = int((before.astype(str).values != out[col].astype(str).values).sum())
            report["coercions"][col] = coerced_count

            if not nullable:
                null_idx = out.index[out[col].isna()].tolist()
                if null_idx:
                    report["invalid_nulls"][col] = null_idx

            if "allowed" in rules:
                allowed = set(rules["allowed"])
                s = out[col]
                if pd.api.types.is_datetime64_any_dtype(s):
                    allowed_cast = pd.to_datetime(list(allowed), errors="coerce")
                    invalid_mask = ~s.isna() & ~s.isin(allowed_cast)
                else:
                    invalid_mask = ~s.isna() & ~s.isin(allowed)
                invalid_idx = out.index[invalid_mask].tolist()
                if invalid_idx:
                    report["invalid_values"][col] = invalid_idx

        if strict and (report["invalid_nulls"] or report["invalid_values"] or report["errors"]):
            raise ValueError(f"Schema violations: {report}")

        return Dataset(out, name=f"{self._name}::enforced"), report

    
    def generate_data_report(self, x: str, y: str, title: str) -> plt.Figure:
        if self._df.empty:
            raise ValueError("DataFrame is empty; nothing to plot.")
        if x not in self._df.columns or y not in self._df.columns:
            raise KeyError(f"Missing columns: {x}, {y}")
        x_vals = self._df[x].astype(str)
        y_vals = pd.to_numeric(self._df[y], errors="coerce")
        if y_vals.isna().any():
            raise ValueError(f"Column '{y}' contains non-numeric values.")
        fig = plt.figure(figsize=(10, 6))
        plt.bar(x_vals, y_vals)
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        return fig

    
    def _assert_cols(self, cols: Sequence[str], *, exc=ValueError) -> None:
        if not isinstance(cols, (list, tuple)):
            raise TypeError("cols must be a sequence of column names")
        missing = [c for c in cols if c not in self._df.columns]
        if missing:
            raise exc(f"cols not in DataFrame: {missing}")

    def __str__(self) -> str:
        return f"Dataset(name={self._name}, shape={self._df.shape})"

    def __repr__(self) -> str:
        return f"Dataset(name={self._name!r}, shape={self._df.shape})"


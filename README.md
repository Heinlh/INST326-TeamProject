# INST326-TeamProject
# Research Data Management - Data Analysis

**Team:** Team HATA  
**Domain:** Experiment and Research  
**Course:** INST326 - Object-Oriented Programming for Information Science  

## Project Overview

A computer program that is able to organize research data more effectively and concisely. This program is planned to download research data, parse, validate, and anonymize user data. This program will be  We aim for this program to be able to work with multiple file formats. 

## Problem Statement 
Researchers often struggle to:
Processing their quantitative data conveniently 
Having to deal with multiple research data formats simultaneously
Visualizing their data easily
Effectively validating and cleaning data without errors
Maintaining consistency and reproducible results in research workflows

## Function Library Overview 

Our library contains 15 specialized functions organized into four categories:

### Data Ingestion (3 functions)
directory_creation - Creates a directory for which the data will be stored
download_file() - Downloads the data to the desired location 
parse_csv_data() - Reads a .csv file into a dataframe


### Data Validation and Cleaning (6 functions)
detect_missing() - Reports NA counts in a column 
detect_duplicates() - Return index positions of duplicate rows using an optional column subset.
standardize_column_names() - Normalize column names (e.g., to snake_case) for consistency across files.
clean_strings()- Normalize text fields (trim, lowercase, collapse whitespace) in selected columns.
validate_research_ethics_compliance() - Basic ethics checks (PII presence, consent alignment). Return {compliant, issues}.
validate_experiment_parameters() - Produce a compact per-column summary (count, NA, distinct, numeric stats).

### Data Processing and Transformation (5 functions)
anonymize_participant_data() - Redact sensitive columns for privacy-preserving analysis.
input_csv() - Append dictionary rows to a CSV; write a header if the file is new. Return the CSV Path.
calculate_statistical_summary() - Produce a compact per-column summary (count, NA, distinct, numeric stats).
enforce_schema() - Coerce/validate dtypes, datetime formats, allowed sets, nullability; return (df_out, report)
apply_pipeline()- Run a list of transforms (filter/groupby/pivot/melt) with provenance; return (df_out, log)

## Team Member Contributions

**Nathanon ‘Tan’ Chaiyapan** - Data processing and transformation
- Came up with 8 of the functions
- Worked on 5 of the functions 

**Athilah Abadir** - Data Ingestion and Input handling  
- Monitored the functions to make sure they work
- Worked on 5 functions 

**Arthur Nguyen** - Documentation Manager, Assistant Coder
- Implemented Data Ingestion functions 
- Currently contributing to the README documentation 


**Hein Htet** - Data processing and transformation
- Worked on 5 of the functions 
- Suggests lots of ideas of the functions

## Code Review Process 

All functions have been reviewed by at least one other team member:
- Pull request reviews documented in GitHub
- Code quality standards enforced consistently
- Documentation reviewed for clarity and completeness
- Function signatures standardized across the library




## AI Collaboration Documentation

Team members used AI assistance for:
- Initial function structure generation
- Function ideation 
- Algorithm optimization suggestions
- Error handling pattern recommendations

We try to document our uses of AI to make sure we are as transparent as possible on the matter to make sure that it does comply with AI policy as we are able to correct course immediately if it is deemed wrong

---

## Repository Structure

```
INST326TeamProject/
├── README.md
├── src/
│   ├── __init__.py
├── docs/
│   ├── function_reference.md
│   └── usage_examples.md
└── requirements.txt
```

---

<<<<<<< HEAD
### Data Visualization (1 function)
generate_data_report() - Create a basic visualization from processed data (e.g., a simple bar chart).
=======
### Container Calculations
```python
from src.garden_library import calculate_container_area, calculate_soil_volume

# Calculate area of rectangular raised bed
area = calculate_container_area(48, 24, shape='rectangle')  # 1152 square inches

# Calculate soil volume needed
volume = calculate_soil_volume(48, 24, 8)  # 9216 cubic inches
```

### Plant Spacing
```python
from src.garden_library import determine_plant_capacity, calculate_plant_spacing

# How many tomatoes fit in a 4x2 foot bed?
capacity = determine_plant_capacity(48, 24, 18)  # 4 plants with 18" spacing

# What spacing for 6 plants in same bed?
spacing = calculate_plant_spacing(48, 24, 6)  # 12" spacing needed
```

### Seasonal Planning
```python
from src.garden_library import days_until_frost, is_safe_to_plant

# Check planting safety based on frost dates
safe = is_safe_to_plant('2024-04-15', last_frost='2024-04-20')  # False

# Days remaining in growing season
days = days_until_frost('2024-05-01', first_frost='2024-10-15')  # 167 days
```

## Function Library Overview

Our library contains 15 specialized functions organized into four categories:

from __future__ import annotations

# Core imports
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Union
from urllib.parse import urlparse

import csv
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt


PathLike = Union[str, Path]


def directory_creation(directory_path: PathLike) -> Path:
    """
    Create a directory (and parents) for downloaded/processed research data.

    Args:
        directory_path (PathLike): Target folder path (e.g., "data/raw" or Path("data")).

    Returns:
        Path: The created (or existing) directory as a resolved Path.

    Raises:
        PermissionError: If the process lacks permissions to create the directory.
        OSError: For other filesystem-related errors.

    Examples:
        >>> p = directory_creation("data")
        >>> p.is_dir()
        True
    """
    p = Path(directory_path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    if any(p.iterdir()):
        print(f"{p.resolve()} already exists (not empty).")
    else:
        print(f"{p.resolve()} is created")
    return p


def download_file(url: str, dest_dir: PathLike = "data") -> Path:
    """
    Download a file over HTTP(S) and save it to a destination directory.

    Args:
        url (str): HTTP(S) URL of the file to download.
        dest_dir (PathLike): Directory where the file will be saved. Created if missing.

    Returns:
        Path: Full path to the downloaded file.

    Raises:
        ValueError: If the URL scheme is not HTTP/HTTPS.
        requests.exceptions.HTTPError: If the server returns an unsuccessful status code.
        requests.exceptions.RequestException: For other network-related errors.
        PermissionError: If the destination cannot be written.

    Examples:
        >>> # doctest: +SKIP
        >>> out = download_file("https://example.com/file.csv", "data")
        >>> out.exists()
        True
    """
    dest = directory_creation(dest_dir)

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")

    fname = Path(parsed.path).name or "download.bin"
    path = dest / fname

    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"{url} is saved to {path.resolve()}")
    return path


def detect_missing(df: pd.DataFrame, cols: Sequence[str] | None = None) -> pd.Series:
    """
    Report the number of missing (NA) values per column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (Sequence[str] | None): Optional subset of columns to inspect.

    Returns:
        pd.Series: NA counts per column (index = column names).

    Raises:
        TypeError: If `df` is not a DataFrame.
        ValueError: If any requested column in `cols` is missing.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 2]})
        >>> detect_missing(df).to_dict()
        {'a': 1, 'b': 2}
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if cols is not None:
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        df = df[cols]

    return df.isna().sum()


def detect_duplicates(df: pd.DataFrame, subset: Sequence[str] | None = None) -> pd.Index:
    """
    Return the index positions of duplicate rows.

    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (Sequence[str] | None): Optional columns to consider for duplicates.

    Returns:
        pd.Index: Index labels of rows that are duplicates (keep=False).

    Raises:
        TypeError: If `df` is not a DataFrame or `subset` is not a sequence.
        ValueError: If `subset` contains unknown columns.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1,1,2], "b": [3,3,4]})
        >>> list(detect_duplicates(df))
        [0, 1]
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if subset is not None:
        if not isinstance(subset, (list, tuple)):
            raise TypeError("subset must be a sequence of column names or None")
        missing = [c for c in subset if c not in df.columns]
        if missing:
            raise ValueError(f"subset contains columns not in df: {missing}")

    mask = df.duplicated(subset=subset, keep=False)
    return df.index[mask]


def standardize_column_names(df: pd.DataFrame, case: str = "snake") -> pd.DataFrame:
    """
    Normalize column names to a specified style.

    Supported styles: "snake", "kebab", "lower", "upper", "title".

    Args:
        df (pd.DataFrame): Input DataFrame.
        case (str): Target casing style (default "snake").

    Returns:
        pd.DataFrame: Copy of the DataFrame with renamed columns.

    Raises:
        TypeError: If `df` is not a DataFrame or `case` is not a string.
        ValueError: If `case` is not one of the supported styles.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame(columns=["Full Name", "E-mail"])
        >>> standardize_column_names(df, "snake").columns.tolist()
        ['full_name', 'e_mail']
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
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
        return s

    new_cols = [normalize(c) for c in df.columns]
    return df.rename(columns=dict(zip(df.columns, new_cols)))


def clean_strings(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    strip: bool = True,
    lower: bool = True,
    collapse_spaces: bool = True,
) -> pd.DataFrame:
    """
    Clean text columns by trimming, lowercasing, and collapsing whitespace.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (Sequence[str]): Columns to clean.
        strip (bool): Trim leading/trailing whitespace. Defaults to True.
        lower (bool): Convert to lowercase. Defaults to True.
        collapse_spaces (bool): Collapse multiple spaces into one. Defaults to True.

    Returns:
        pd.DataFrame: Copy of the DataFrame with cleaned columns.

    Raises:
        TypeError: If `df` is not a DataFrame or `cols` is not a sequence.
        ValueError: If any column in `cols` is missing.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"name": ["  Alice ", "Bob  "]})
        >>> clean_strings(df, ["name"])["name"].tolist()
        ['alice', 'bob']
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(cols, (list, tuple)):
        raise TypeError("cols must be a sequence of column names")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"cols not in DataFrame: {missing}")

    out = df.copy()

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

    return out


def anonymize_participant_data(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Replace values in sensitive columns with masked tokens.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (Sequence[str]): Columns to anonymize (mask with "***").

    Returns:
        pd.DataFrame: New DataFrame with specified columns masked.

    Raises:
        TypeError: If `df` is not a DataFrame.
        KeyError: If any column in `cols` is missing.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"name": ["Ana", "Bo"], "age": [7, 8]})
        >>> anonymize_participant_data(df, ["name"])["name"].tolist()
        ['***', '***']
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    out = df.copy(deep=True)
    for c in cols:
        out.loc[out[c].notna(), c] = "***"
    return out


def validate_research_ethics_compliance(
    df: pd.DataFrame,
    pii_cols: Sequence[str] | None = None,
    consent_col: str | None = None,
) -> dict:
    """
    Run basic ethics checks comparing PII presence to a consent indicator.

    Args:
        df (pd.DataFrame): Input DataFrame.
        pii_cols (Sequence[str] | None): Columns considered personally identifiable information.
        consent_col (str | None): Column indicating consent (truthy values allowed: 1/True/yes/y).

    Returns:
        dict: {"compliant": bool, "issues": list[str]} describing any violations.

    Raises:
        TypeError: If `df` is not a DataFrame.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"email": ["a@x.com", None], "consent": [1, 0]})
        >>> validate_research_ethics_compliance(df, pii_cols=["email"], consent_col="consent")["compliant"]
        True
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    issues: list[str] = []
    pii = list(pii_cols or [])

    consent = (
        df[consent_col]
        .astype(str)
        .str.lower()
        .isin({"1", "true", "yes", "y"})
        if (consent_col and consent_col in df.columns)
        else None
    )

    for c in pii:
        if c not in df.columns or not df[c].notna().any():
            continue
        if consent is None:
            issues.append(f"PII detected in '{c}' without a consent column")
        else:
            n = int((df[c].notna() & ~consent).sum())
            if n:
                issues.append(f"{n} row(s) with PII in '{c}' without consent")

    return {"compliant": not issues, "issues": issues}


def generate_data_report(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    """
    Display a basic bar chart of df[y] versus df[x] with simple validation.

    Args:
        df (pd.DataFrame): Data to visualize.
        x (str): Column for x-axis (categorical/labels).
        y (str): Column for y-axis (numeric).
        title (str): Plot title.

    Returns:
        None

    Raises:
        ValueError: If the DataFrame is empty or `y` cannot be coerced to numeric.
        KeyError: If `x` or `y` is missing in the DataFrame.

    Examples:
        >>> import pandas as pd  # doctest: +SKIP
        >>> df = pd.DataFrame({"city": ["A","B"], "count": [3,5]})
        >>> generate_data_report(df, "city", "count", "Counts")  # doctest: +SKIP
    """
    if df.empty:
        raise ValueError("DataFrame is empty; nothing to plot.")
    if x not in df.columns or y not in df.columns:
        raise KeyError(f"Missing columns: {x}, {y}")

    x_vals = df[x].astype(str)
    y_vals = pd.to_numeric(df[y], errors="coerce")
    if y_vals.isna().any():
        raise ValueError(f"Column '{y}' contains non-numeric values.")

    plt.figure(figsize=(10, 6))
    plt.bar(x_vals, y_vals)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()


def parse_csv_data(csv_file: PathLike, **read_csv_kwargs) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame with simple path/extension validation.

    Behavior:
      - If the path lacks a ".csv" suffix but "<path>.csv" exists, use that.
      - Errors if the resolved path does not exist or is not a file.

    Args:
        csv_file (PathLike): Path (or base path) to a CSV file.
        **read_csv_kwargs: Additional keyword args passed to `pandas.read_csv`.

    Returns:
        pd.DataFrame: Loaded table.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path exists but is not a regular file.

    Examples:
        >>> # doctest: +SKIP
        >>> df = parse_csv_data("data/sample")  # will try "data/sample.csv"
        >>> isinstance(df, pd.DataFrame)
        True
    """
    p = Path(csv_file).expanduser()
    if p.suffix.lower() != ".csv":
        candidate = p.with_suffix(".csv")
        if candidate.exists():
            p = candidate

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")

    return pd.read_csv(p, **read_csv_kwargs)


def input_csv(rows: Iterable[Mapping[str, Any]], csv_file: PathLike) -> Path:
    """
    Append dictionary-like rows to a CSV file, adding a header if the file is new.

    Args:
        rows (Iterable[Mapping[str, Any]]): Iterable of row mappings (dict-like).
        csv_file (PathLike): Target CSV file path.

    Returns:
        Path: Absolute path to the written CSV file.

    Raises:
        TypeError: If `csv_file` is not str/Path or `rows` items are not mapping-like.
        ValueError: If `rows` is empty.

    Examples:
        >>> # doctest: +SKIP
        >>> path = input_csv([{"a":1,"b":2}, {"a":3,"b":4}], "data/out.csv")
        >>> path.name
        'out.csv'
    """
    if not isinstance(csv_file, (str, Path)):
        raise TypeError("csv_file must be a str or Path")

    materialized = list(rows)
    if not materialized:
        raise ValueError("rows is empty; nothing to write")

    if not all(hasattr(r, "keys") for r in materialized):
        raise TypeError("rows must be an iterable of dict-like mappings")

    target = Path(csv_file).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame(materialized)
    file_exists = target.exists()

    df_new.to_csv(
        target,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return target


def calculate_statistical_summary(
    df: pd.DataFrame, include_categoricals: bool = True
) -> pd.DataFrame:
    """
    Produce a compact per-column summary combining numeric and (optional) categorical stats.

    Args:
        df (pd.DataFrame): Input DataFrame.
        include_categoricals (bool): Whether to include summary of object/string/category columns.

    Returns:
        pd.DataFrame: Summary indexed by column name with fields like count/mean/std (numeric),
        plus NA and distinct counts; for categoricals, includes top and freq.

    Raises:
        TypeError: If `df` is not a DataFrame.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"x":[1,2,2,None], "c":["a","a","b",None]})
        >>> summary = calculate_statistical_summary(df)
        >>> set(["na","distinct"]).issubset(summary.columns)
        True
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    frames: list[pd.DataFrame] = []

    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols):
        num = df[num_cols].describe().T
        num["na"] = df[num_cols].isna().sum()
        num["distinct"] = df[num_cols].nunique(dropna=True)
        frames.append(num)

    if include_categoricals:
        obj_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        if len(obj_cols):
            cats = pd.DataFrame(index=obj_cols)
            cats["count"] = df[obj_cols].notna().sum()
            cats["na"] = df[obj_cols].isna().sum()
            cats["distinct"] = df[obj_cols].nunique(dropna=True)
            cats["top"] = df[obj_cols].apply(
                lambda s: s.mode(dropna=True).astype(str).iloc[0]
                if s.dropna().size
                else None
            )
            cats["freq"] = df[obj_cols].apply(
                lambda s: int(s.value_counts(dropna=True).iloc[0])
                if s.dropna().size
                else 0
            )
            frames.append(cats)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0).sort_index()


def validate_experiment_parameters(
    params: Mapping[str, Any],
    schema: Mapping[str, Mapping[str, Any]],
) -> Tuple[bool, Dict[str, str]]:
    """
    Validate experiment parameters using a declarative schema.

    Schema per key may include:
      - required (bool): Whether the parameter must be present.
      - type (type | tuple[type,...]): Expected Python type(s).
      - range (tuple[number, number]): Inclusive [min, max] numeric range.
      - allowed (Iterable[Any]): Allowed set of values.

    Args:
        params (Mapping[str, Any]): Provided parameter values.
        schema (Mapping[str, Mapping[str, Any]]): Validation rules per parameter.

    Returns:
        Tuple[bool, Dict[str, str]]: (is_valid, issues) where `issues` maps param -> message string.

    Raises:
        TypeError: If `params` or `schema` are not mappings.

    Examples:
        >>> params = {"lr": 0.1, "mode": "train"}
        >>> schema = {"lr": {"required": True, "type": float, "range": (0.0, 1.0)},
        ...           "mode": {"allowed": {"train","eval"}}}
        >>> ok, issues = validate_experiment_parameters(params, schema)
        >>> ok, issues == {}
        (True, True)
    """
    if not isinstance(params, Mapping) or not isinstance(schema, Mapping):
        raise TypeError("params and schema must be mappings")

    issues: Dict[str, list[str]] = {}
    is_valid = True

    for key, spec in schema.items():
        if spec.get("required") and key not in params:
            is_valid = False
            issues.setdefault(key, []).append("Required parameter is missing.")

    for key, value in params.items():
        if key not in schema:
            is_valid = False
            issues.setdefault(key, []).append(
                f"Parameter '{key}' is not a valid experiment parameter."
            )
            continue

        spec = schema[key]

        expected_type = spec.get("type")
        if expected_type is not None and not isinstance(value, expected_type):
            is_valid = False
            tname = getattr(expected_type, "__name__", str(expected_type))
            issues.setdefault(key, []).append(
                f"Expected type {tname}, got {type(value).__name__}."
            )

        if "range" in spec:
            lo, hi = spec["range"]
            if isinstance(value, (int, float)):
                if not (lo <= value <= hi):
                    is_valid = False
                    issues.setdefault(key, []).append(
                        f"Value {value} outside allowed range [{lo}, {hi}]."
                    )
            else:
                if expected_type in (int, float, None):
                    is_valid = False
                    issues.setdefault(key, []).append(
                        "Range specified but value is not numeric."
                    )

        if "allowed" in spec:
            allowed = set(spec["allowed"])
            if value not in allowed:
                is_valid = False
                issues.setdefault(key, []).append(
                    f"Value {value!r} not in allowed set: {sorted(allowed)}."
                )

    flat_issues: Dict[str, str] = {k: ", ".join(v) for k, v in issues.items()}
    return is_valid, flat_issues


def enforce_schema(
    df: pd.DataFrame,
    schema: Mapping[str, Mapping[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Coerce and validate a DataFrame against a declarative schema.

    Supported `dtype` values: "int", "float", "string", "bool", "datetime".
    Optional rules:
      - nullable (bool): If False, NA values are reported as violations.
      - datetime_format (str | list[str]): Format(s) to try for datetime parsing.
      - allowed (Iterable[Any]): Allowed set of (stringified) values.

    Args:
        df (pd.DataFrame): Input DataFrame to validate/clean.
        schema (Mapping[str, Mapping[str, Any]]): Mapping of column -> rules.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            df_out: Coerced DataFrame.
            report: Dict with keys {"coercions","invalid_nulls","invalid_values","errors"}.

    Raises:
        TypeError: If `df` is not a DataFrame or `schema` is not a dict.

    Examples:
        >>> import pandas as pd
        >>> schema = {"age": {"dtype":"int","nullable":False}}
        >>> out, report = enforce_schema(pd.DataFrame({"age":["1","x"]}), schema)
        >>> "age" in report["invalid_nulls"] or "age" in report["coercions"]
        True
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(schema, dict):
        raise TypeError("schema must be a mapping (dict) of column -> rules")

    supported_dtypes = {"int", "float", "string", "bool", "datetime"}
    out = df.copy()
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
                f"Column '{col}': unsupported dtype '{dtype}'. "
                f"Supported: {sorted(supported_dtypes)}"
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
                if nullable:
                    out[col] = numeric.astype("Int64")
                else:
                    if numeric.isna().any():
                        out[col] = numeric.astype("Int64")
                    else:
                        out[col] = numeric.astype("int64")

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
                        parsed.loc[still] = pd.to_datetime(
                            raw.loc[still], errors="coerce", utc=False
                        )
                    out[col] = parsed

        except Exception as e:
            report["errors"].append(f"Coercion error for '{col}': {e}")
            continue

        coerced_count = int(
            (before.astype(str).values != out[col].astype(str).values).sum()
        )
        report["coercions"][col] = coerced_count

        if not nullable:
            null_idx = out.index[out[col].isna()].tolist()
            if null_idx:
                report["invalid_nulls"][col] = null_idx

        allowed = rules.get("allowed", None)
        if allowed is not None:
            allowed_set = set(allowed)
            s = out[col]
            comp = s.astype("string") if not pd.api.types.is_string_dtype(s) else s
            invalid_mask = ~comp.isna() & ~comp.isin({str(v) for v in allowed_set})
            invalid_idx = out.index[invalid_mask].tolist()
            if invalid_idx:
                report["invalid_values"][col] = invalid_idx

    return out, report


def apply_pipeline(
    df: pd.DataFrame, steps: Sequence[Mapping[str, Any]], log: list[dict] | None = None
) -> Tuple[pd.DataFrame, list[dict]]:
    """
    Execute a sequence of DataFrame operations with provenance logging.

    Supported ops include (examples):
      - {"op": "filter", "expr": "amount >= 0"}
      - {"op": "groupby_agg", "by": ["user_id"], "metrics": {"amount": "sum"}}
      - {"op": "pivot", "index": "user_id", "columns": "status", "values": "amount", "fill_value": 0}
      - {"op": "melt", "id_vars": ["date"], "value_vars": ["open", "closed"]}
      - {"op": "select", "cols": ["a","b","c"]}
      - {"op": "rename", "mapping": {"old":"new"}}
      - {"op": "sort", "by": ["date"], "ascending": [True]}
      - {"op": "fillna", "value": 0}  # or {"colA": 0, "colB": "unknown"}
      - {"op": "dropna", "subset": ["col"], "how": "any"}  # how in {"any","all"}
      - {"op": "eval", "expr": "z = x + y"}  # DataFrame.eval expression
      - {"op": "assign", "values": {"z": "=x + y", "flag": 1}}  # '=' prefix means eval

    Args:
        df (pd.DataFrame): Input DataFrame.
        steps (Sequence[Mapping[str, Any]]): Ordered list of operation dicts.
        log (list[dict] | None): Optional list to append step logs to.

    Returns:
        Tuple[pd.DataFrame, list[dict]]: (transformed_df, runlog) where runlog contains
        step metadata (rows_before/after, columns_before/after, notes).

    Raises:
        TypeError: If `df` is not a DataFrame or `steps` is not a sequence of mappings.
        ValueError: For malformed steps, unsupported ops, or invalid parameters.
        KeyError: For steps that reference missing columns.
        ValueError: Wrapped exceptions raised by operations (with step context).

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"user":[1,1,2], "amount":[5, -1, 3], "status":["ok","ok","bad"]})
        >>> steps = [
        ...   {"op":"filter", "expr":"amount >= 0"},
        ...   {"op":"groupby_agg", "by":["user"], "metrics":{"amount":"sum"}}
        ... ]
        >>> out, log = apply_pipeline(df, steps)
        >>> list(out.columns)
        ['user', 'amount']
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(steps, (list, tuple)):
        raise TypeError("steps must be a sequence of mapping objects")

    out = df.copy()
    runlog: list[dict] = [] if log is None else log

    for i, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            raise ValueError(f"Step {i}: each step must be a mapping (dict-like)")
        op = step.get("op")
        if not isinstance(op, str):
            raise ValueError(f"Step {i}: missing or invalid 'op'")

        rows_before = len(out)
        cols_before = list(out.columns)
        note = ""

        try:
            if op == "filter":
                expr = step["expr"]
                out = out.query(expr)
                note = f"filter expr={expr!r}"

            elif op == "groupby_agg":
                by = step["by"]
                metrics = step["metrics"]
                out = out.groupby(by, dropna=False).agg(metrics).reset_index()
                note = f"groupby_agg by={by!r}"

            elif op == "pivot":
                out = out.pivot_table(
                    index=step["index"],
                    columns=step["columns"],
                    values=step.get("values"),
                    fill_value=step.get("fill_value", 0),
                    aggfunc=step.get("aggfunc", "sum"),
                ).reset_index()
                if isinstance(out.columns, pd.MultiIndex):
                    out.columns = [
                        "_".join([str(c) for c in tup if c != ""])
                        for tup in out.columns.to_flat_index()
                    ]
                note = "pivot"

            elif op == "melt":
                out = pd.melt(
                    out,
                    id_vars=step["id_vars"],
                    value_vars=step["value_vars"],
                    var_name=step.get("var_name", "variable"),
                    value_name=step.get("value_name", "value"),
                )
                note = "melt"

            elif op == "select":
                cols = step["cols"]
                missing = [c for c in cols if c not in out.columns]
                if missing:
                    raise KeyError(f"select missing columns: {missing}")
                out = out.loc[:, cols]
                note = f"select cols={cols!r}"

            elif op == "rename":
                mapping = step["mapping"]
                out = out.rename(columns=mapping)
                note = f"rename {mapping!r}"

            elif op == "sort":
                by = step["by"]
                ascending = step.get("ascending", True)
                out = out.sort_values(by=by, ascending=ascending, kind="mergesort")
                note = f"sort by={by!r}"

            elif op == "fillna":
                value = step["value"]
                out = out.fillna(value)
                note = f"fillna value={'<mapping>' if isinstance(value, Mapping) else value!r}"

            elif op == "dropna":
                subset = step.get("subset")
                how = step.get("how", "any")
                if how not in {"any", "all"}:
                    raise ValueError("dropna 'how' must be 'any' or 'all'")
                out = out.dropna(subset=subset, how=how)
                note = f"dropna subset={subset!r} how={how}"

            elif op == "eval":
                expr = step["expr"]
                out = out.copy()
                out.eval(expr, inplace=True)
                note = f"eval {expr!r}"

            elif op == "assign":
                values = step["values"]
                to_assign = {}
                for k, v in values.items():
                    if isinstance(v, str) and v.startswith("="):
                        to_assign[k] = out.eval(v[1:])
                    else:
                        to_assign[k] = v
                out = out.assign(**to_assign)
                note = f"assign keys={list(values.keys())!r}"

            else:
                raise ValueError(f"Unsupported op: {op}")

        except Exception as e:
            raise ValueError(f"Error in step {i} ({op}): {e}") from e
        finally:
            runlog.append(
                {
                    "step": i,
                    "op": op,
                    "rows_before": rows_before,
                    "rows_after": len(out),
                    "cols_before": cols_before,
                    "cols_after": list(out.columns),
                    "note": note,
                }
            )

    return out, runlog



>>>>>>> 59724fc2e31352c7220ea6ed8786ea9bcd917dbb


<<<<<<< HEAD
=======

>>>>>>> 59724fc2e31352c7220ea6ed8786ea9bcd917dbb

# Data Processing and Research Utilities Library – Reference Guide

This document provides comprehensive reference information for all functions in the Data Processing and Research Utilities library.

---

## Table of Contents
1. [File & Directory Management Functions](#1-file--directory-management-functions)  
2. [Data Cleaning & Standardization Functions](#2-data-cleaning--standardization-functions)  
3. [Privacy & Ethical Compliance Functions](#3-privacy--ethical-compliance-functions)  
4. [Data Analysis & Reporting Functions](#4-data-analysis--reporting-functions)  
5. [Validation & Pipeline Management Functions](#5-validation--pipeline-management-functions)  

---

## 1. File & Directory Management Functions

### directory_creation(directory_path)
**Purpose:**  
Create a directory (and any parent directories) for downloaded or processed research data.

**Parameters:**  
- `directory_path (str | Path)`: Directory path to create.

**Returns:**  
`Path` – Resolved directory path.

**Example Usage:**
```python
path = directory_creation("data/raw")
```

---

### download_file(url, dest_dir="data")
**Purpose:**  
Download research data from a specified HTTP(S) URL and save it locally.

**Parameters:**  
- `url (str)`: File URL to download.  
- `dest_dir (str | Path)`: Destination directory (default `"data"`).

**Returns:**  
`Path` – Local file path of the downloaded file.

**Example Usage:**
```python
download_file("https://example.com/dataset.csv", "downloads")
```

---

### parse_csv_data(csv_file, **read_csv_kwargs)
**Purpose:**  
Load a `.csv` file into a DataFrame with validation and flexible extension handling.

**Parameters:**  
- `csv_file (str | Path)`: CSV file path.  
- `**read_csv_kwargs`: Optional pandas read arguments.

**Returns:**  
`pd.DataFrame` – Loaded DataFrame.

**Example Usage:**
```python
df = parse_csv_data("data/survey")
```

---

### input_csv(rows, csv_file)
**Purpose:**  
Append rows to an existing CSV file or create a new one with headers.

**Parameters:**  
- `rows (Iterable[dict])`: Row dictionaries.  
- `csv_file (str | Path)`: Output file path.

**Returns:**  
`Path` – Written file path.

**Example Usage:**
```python
input_csv([{"id":1,"name":"Alice"}], "data/output.csv")
```

---

## 2. Data Cleaning & Standardization Functions

### detect_missing(df, cols=None)
**Purpose:**  
Report missing value counts per column (optionally for selected columns).

**Parameters:**  
- `df (pd.DataFrame)`: Input DataFrame.  
- `cols (Sequence[str], optional)`: Specific columns to check.

**Returns:**  
`pd.Series` – Number of missing values per column.

**Example Usage:**
```python
detect_missing(df, ["age", "income"])
```

---

### detect_duplicates(df, subset=None)
**Purpose:**  
Return index positions of duplicate rows, optionally by a subset of columns.

**Parameters:**  
- `df (pd.DataFrame)`: Input DataFrame.  
- `subset (Sequence[str], optional)`: Columns defining duplicates.

**Returns:**  
`pd.Index` – Indices of duplicate rows.

**Example Usage:**
```python
duplicates = detect_duplicates(df, subset=["id"])
```

---

### standardize_column_names(df, case="snake")
**Purpose:**  
Normalize DataFrame column names for consistency across files.

**Parameters:**  
- `df (pd.DataFrame)`: Input DataFrame.  
- `case (str)`: Style (`"snake"`, `"kebab"`, `"lower"`, `"upper"`, `"title"`).

**Returns:**  
`pd.DataFrame` – DataFrame with renamed columns.

**Example Usage:**
```python
standardize_column_names(df, case="kebab")
```

---

### clean_strings(df, cols, strip=True, lower=True, collapse_spaces=True)
**Purpose:**  
Trim, lowercase, and collapse whitespace in specified string columns.

**Parameters:**  
- `df (pd.DataFrame)`: Input DataFrame.  
- `cols (Sequence[str])`: Columns to clean.  
- `strip` / `lower` / `collapse_spaces (bool)`: Cleaning options.

**Returns:**  
`pd.DataFrame` – Cleaned copy of the DataFrame.

**Example Usage:**
```python
clean_strings(df, ["name", "city"])
```

---

## 3. Privacy & Ethical Compliance Functions

### anonymize_participant_data(df, cols)
**Purpose:**  
Mask sensitive participant data for privacy-preserving analysis.

**Parameters:**  
- `df (pd.DataFrame)`: Input DataFrame.  
- `cols (Sequence[str])`: Sensitive columns to mask.

**Returns:**  
`pd.DataFrame` – Copy with masked values.

**Example Usage:**
```python
anonymize_participant_data(df, ["email", "phone"])
```

---

### validate_research_ethics_compliance(df, pii_cols=None, consent_col=None)
**Purpose:**  
Check for potential ethical compliance issues—PII without consent indicators.

**Parameters:**  
- `df (pd.DataFrame)`: DataFrame under review.  
- `pii_cols (Sequence[str])`: Columns containing personal data.  
- `consent_col (str)`: Column indicating participant consent.

**Returns:**  
`dict` – `{ "compliant": bool, "issues": list }`

**Example Usage:**
```python
validate_research_ethics_compliance(df, ["email"], "consent")
```

---

## 4. Data Analysis & Reporting Functions

### calculate_statistical_summary(df, include_categoricals=True)
**Purpose:**  
Produce a compact column-wise summary including numeric and categorical statistics.

**Parameters:**  
- `df (pd.DataFrame)`: Input data.  
- `include_categoricals (bool)`: Whether to summarize categorical columns.

**Returns:**  
`pd.DataFrame` – Combined statistics.

**Example Usage:**
```python
summary = calculate_statistical_summary(df)
```

---

### generate_data_report(df, x, y, title)
**Purpose:**  
Generate a basic bar chart visualization from processed data.

**Parameters:**  
- `df (pd.DataFrame)`: DataFrame with plotting data.  
- `x (str)`: X-axis column.  
- `y (str)`: Y-axis column.  
- `title (str)`: Chart title.

**Returns:**  
None (displays the plot).

**Example Usage:**
```python
generate_data_report(df, "category", "count", "Category Distribution")
```

---

## 5. Validation & Pipeline Management Functions

### validate_experiment_parameters(params, schema)
**Purpose:**  
Validate experimental setup against a schema of allowed parameters and types.

**Parameters:**  
- `params (dict)`: User-provided parameter values.  
- `schema (dict)`: Expected schema specifying type, range, and allowed values.

**Returns:**  
`(bool, dict)` – Tuple of validity flag and detailed issue messages.

**Example Usage:**
```python
schema = {"sample_size": {"type": int, "range": (1, 100)}}
validate_experiment_parameters({"sample_size": 150}, schema)
```

---

### enforce_schema(df, schema)
**Purpose:**  
Coerce and validate DataFrame columns against defined schema rules.

**Parameters:**  
- `df (pd.DataFrame)`: DataFrame to validate.  
- `schema (dict)`: Mapping of column names to dtype rules.

**Returns:**  
`(pd.DataFrame, dict)` – Coerced DataFrame and validation report.

**Example Usage:**
```python
df_validated, report = enforce_schema(df, schema)
```

---

### apply_pipeline(df, steps, log=None)
**Purpose:**  
Execute a configurable transformation pipeline with logging for reproducibility.

**Parameters:**  
- `df (pd.DataFrame)`: Input data.  
- `steps (Sequence[dict])`: Ordered transformation operations.  
- `log (list[dict], optional)`: Prior run log (for continuation).

**Returns:**  
`(pd.DataFrame, list[dict])` – Transformed data and operation log.

**Supported Operations:**  
`filter`, `groupby_agg`, `pivot`, `melt`, `select`, `rename`, `sort`, `fillna`, `dropna`, `eval`, `assign`

**Example Usage:**
```python
steps = [
    {"op": "filter", "expr": "age > 18"},
    {"op": "groupby_agg", "by": "city", "metrics": {"income": "mean"}}
]
result, log = apply_pipeline(df, steps)
```

---


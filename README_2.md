# Data Processing and Research Utilities Library

**Team:** Team HATA  
**Domain:** Experiment & Research 
**Course:** INST326 - Object-Oriented Programming for Information Science  

---

## Project Overview

This function library provides essential tools for data collection, cleaning, validation, ethical compliance, and analysis.  
The library supports common research and data processing workflows such as data ingestion, preprocessing, schema enforcement, anonymization, and result reporting.  
These functions serve as a foundation for future object-oriented data systems and reproducible research pipelines.

---

## Problem Statement

Researchers and data professionals often struggle with:

- Managing input/output files and directory organization  
- Cleaning, standardizing, and validating large datasets  
- Ensuring compliance with ethical research and privacy requirements  
- Performing repeatable analyses and generating automated reports  
- Maintaining reproducibility across multiple stages of data processing  

---

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Heinlh/INST326-TeamProject.git
   cd INST326-TeamProject
   ```

2. Install dependencies (if any):
   ```bash
   pip install -r requirements.txt
   ```

3. Import functions in your Python code:
   ```python
   from src.data_utils import parse_csv_data, clean_strings, apply_pipeline
   ```

---

## Quick Usage Examples

### File & Directory Management
```python
from src.data_utils import directory_creation, parse_csv_data

# Create directory and load a CSV file
directory_creation("data/raw")
df = parse_csv_data("data/raw/survey_data.csv")
```

### Data Cleaning
```python
from src.data_utils import standardize_column_names, clean_strings

# Clean and standardize column names and text data
df = standardize_column_names(df)
df = clean_strings(df, ["participant_name", "city"])
```

### Ethical Compliance
```python
from src.data_utils import anonymize_participant_data, validate_research_ethics_compliance

# Anonymize participant data and validate ethical standards
df = anonymize_participant_data(df, ["email", "phone"])
compliance = validate_research_ethics_compliance(df, pii_cols=["email"], consent_col="consent")
```

### Data Analysis and Reporting
```python
from src.data_utils import calculate_statistical_summary, generate_data_report

# Summarize and visualize data
summary = calculate_statistical_summary(df)
generate_data_report(df, x="city", y="income", title="Average Income by City")
```

### Validation & Pipeline
```python
from src.data_utils import enforce_schema, apply_pipeline

# Define schema and transformation pipeline
schema = {"age": {"type": int, "range": (18, 99)}, "income": {"type": float}}
df, report = enforce_schema(df, schema)

steps = [
    {"op": "filter", "expr": "age > 21"},
    {"op": "groupby_agg", "by": "city", "metrics": {"income": "mean"}}
]
df, log = apply_pipeline(df, steps)
```

---

## Function Library Overview

Our library contains 15 specialized functions organized into five categories:

### 1. File & Directory Management (4 functions)
- `directory_creation()` – Create directories for file organization  
- `download_file()` – Download research data files  
- `parse_csv_data()` – Load and validate CSV datasets  
- `input_csv()` – Append or write tabular data to CSV  

### 2. Data Cleaning & Standardization (4 functions)
- `detect_missing()` – Identify missing data  
- `detect_duplicates()` – Locate duplicate records  
- `standardize_column_names()` – Normalize column naming conventions  
- `clean_strings()` – Clean whitespace, case, and text consistency  

### 3. Privacy & Ethical Compliance (2 functions)
- `anonymize_participant_data()` – Remove or mask sensitive PII  
- `validate_research_ethics_compliance()` – Check for consent and ethical requirements  

### 4. Data Analysis & Reporting (2 functions)
- `calculate_statistical_summary()` – Summarize dataset metrics  
- `generate_data_report()` – Create simple data visualizations  

### 5. Validation & Pipeline Management (3 functions)
- `validate_experiment_parameters()` – Check experimental input validity  
- `enforce_schema()` – Enforce data type and range constraints  
- `apply_pipeline()` – Run sequential data transformations with logging  

---

## Team Member Contributions

**Nathanon “Tan” Chaiyapan** – Data Processing and Transformation  
- Created 8 of the functions in the library  
- Implemented 5 core data processing and transformation utilities  
- Contributed major logic for cleaning, validation, and workflow steps  

**Athilah Abadir** – Data Ingestion and Input Handling  
- Ensured all functions ran correctly through monitoring and continuous testing  
- Implemented 5 functions focused on input handling, loading, and ingestion  
- Assisted in debugging and validating function behaviors across the library  

**Arthur Nguyen** – Documentation Manager & Assistant Coder  
- Implemented Data Ingestion functions used across the project  
- Contributed updates, formatting, and restructuring for clear documentation  
- Led creation and refinement of the README and reference materials  

**Hein Htet** – Data Processing and Transformation  
- Implemented 5 functions including transformation and standardization utilities  
- Suggested multiple design ideas and improvements for function development  
- Assisted with debugging and refining implementation details  

---

## Code Review Process

All functions were peer-reviewed for correctness and style consistency:

- Pull request reviews documented via GitHub  
- Function signatures standardized for reusability  
- Error handling and type hints verified  
- Documentation reviewed for clarity and completeness  

---

## AI Collaboration Documentation

Team members used AI assistance for:

- Initial function structure generation  
- Function ideation and brainstorming  
- Algorithm optimization suggestions  
- Error handling pattern recommendations  

All AI-generated suggestions and code samples were reviewed, tested, and adapted by team members to meet project requirements.

---

## Repository Structure

```
INST326-TeamProject/
├── README.md
├── requirements.txt
├── docs/
│   ├── ARCHITECTURE
│   ├── architecture.md
│   └── function_reference.md
├── examples/
│   └── demo_script.py
├── src/
│   ├── __init__.py
│   ├── AbstractExperiment.py
│   ├── Sample.py
│   ├── analysis.py
│   ├── dataset.py
│   ├── experiment.py
│   ├── research_project_functions.py
│   ├── researcher.py
│   └── test_inst326_hata.py
└── .gitattributes
```

---

**Note:**  
This project demonstrates proper function library development for INST326, showcasing modular programming, documentation best practices, reproducibility, and ethical data handling.

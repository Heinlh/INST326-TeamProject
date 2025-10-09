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


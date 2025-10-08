# INST326-TeamProject
# Research Data Management - Data Analysis

**Team:** Team HATA  
**Domain:** Experiment and Research  
**Course:** INST326 - Object-Oriented Programming for Information Science  

## Project Overview

A computer program that is able to organize research data more effectively and concisely. This program is planned to download research data, parse, validate, and anonynomize user data. This program will be  We aim for this program to be able to work with multiple file formats. 

## Problem Statement

Home gardeners struggle with:
- Calculating optimal container sizes and soil volumes
- Determining appropriate plant spacing and capacity
- Planning seasonal activities and frost date considerations  
- Managing soil composition and amendment tracking
- Converting between different measurement units

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/garden-management-library.git
   cd garden-management-library
   ```

2. No external dependencies required - uses Python standard library only

3. Import functions in your Python code:
   ```python
   from src.garden_library import calculate_container_area, determine_plant_capacity
   ```

## Quick Usage Examples

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

Our library contains 18 specialized functions organized into four categories:

### Container Management (6 functions)
- `calculate_container_area()` - Area calculations for rectangular and circular containers
- `calculate_soil_volume()` - Soil volume requirements 
- `container_area_comparison()` - Compare multiple container areas
- `soil_bags_needed()` - Calculate number of soil bags required
- `convert_measurements()` - Convert between inches, feet, and centimeters
- `validate_container_dimensions()` - Ensure positive, realistic dimensions

### Plant Spacing & Capacity (6 functions)
- `determine_plant_capacity()` - Maximum plants for given spacing
- `calculate_plant_spacing()` - Required spacing for desired plant count
- `optimize_plant_layout()` - Best arrangement for multiple plant types
- `check_companion_compatibility()` - Verify companion planting rules
- `estimate_harvest_yield()` - Expected harvest based on plant count and type
- `calculate_succession_dates()` - Staggered planting schedule

### Seasonal Planning (3 functions)
- `days_until_frost()` - Days remaining in growing season
- `is_safe_to_plant()` - Frost safety check for planting dates
- `generate_planting_calendar()` - Season-long planting schedule

### Soil Management (3 functions)
- `calculate_compost_ratio()` - Compost-to-soil mixing ratios
- `ph_adjustment_calculator()` - Lime/sulfur needed for pH changes
- `track_soil_amendments()` - Record and calculate amendment applications

## Team Member Contributions

**Sarah Johnson** - Container Management Functions
- Implemented all 6 container calculation functions
- Created comprehensive documentation and examples
- Developed measurement conversion utilities

**Mike Chen** - Plant Spacing & Capacity Functions  
- Built plant spacing algorithms and capacity calculations
- Designed companion planting compatibility system
- Created harvest estimation models

**Elena Rodriguez** - Seasonal Planning Functions
- Implemented frost date calculations and safety checks
- Developed planting calendar generation system
- Integrated USDA hardiness zone data

**David Kim** - Soil Management Functions
- Created soil amendment calculation functions
- Built pH adjustment algorithms
- Designed soil composition tracking system

## Code Review Process

All functions have been reviewed by at least one other team member:
- Pull request reviews documented in GitHub
- Code quality standards enforced consistently
- Documentation reviewed for clarity and completeness
- Function signatures standardized across the library

## AI Collaboration Documentation

Team members used AI assistance for:
- Initial function structure generation
- Docstring formatting and examples
- Algorithm optimization suggestions
- Error handling pattern recommendations

All AI-generated code was thoroughly reviewed, tested, and modified to meet project requirements. Individual AI collaboration details documented in personal repositories.

---

## Repository Structure

```
garden-management-library/
├── README.md
├── src/
│   ├── __init__.py
│   ├── garden_library.py
│   └── utils.py
├── docs/
│   ├── function_reference.md
│   └── usage_examples.md
├── examples/
│   └── demo_script.py
└── requirements.txt
```

---

**Note:** This is an instructional example demonstrating proper function library development for INST326. The functions solve real garden management problems while illustrating professional programming practices including documentation, version control, and team collaboration.
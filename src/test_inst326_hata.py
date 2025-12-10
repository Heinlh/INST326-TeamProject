"""
Comprehensive test suite for Research Data Management System.

Covers:
- Dataset (cleaning, QA, schema, plotting)
- Researcher (workspace, CSV IO, ethics)
- Sample (lightweight DataFrame wrapper)
- Analysis (pipeline operations)
- Experiments hierarchy (ABC, inheritance, polymorphism, composition)
"""

"""
TESTING STRATEGY AND CLASSIFICATION
-----------------------------------

This test suite follows the INST326 Project 4 requirements for:
- Unit Tests
- Integration Tests
- System Tests

Below is a classification to help graders quickly identify coverage.

======================================================================
UNIT TESTS (Isolated components, method-level behavior)
======================================================================

Dataset:
    - test_detect_missing
    - test_detect_duplicates
    - test_standardize_column_names_snake_case
    - test_clean_strings
    - test_anonymize
    - test_enforce_schema_basic
    - test_generate_data_report
    - test_generate_data_report_invalid_column_raises

Researcher:
    - test_ensure_dir_creates_subdir
    - test_read_csv_missing_file_raises
    - test_download_file_invalid_scheme_raises

Sample:
    - test_init_requires_dataframe
    - test_sample_has_independent_copy

Analysis:
    - test_run_pipeline_missing_column_in_select_raises
    - test_summarize_delegates_to_dataset
    - test_plot_bar_delegates_to_dataset

Experiments (Behavior of concrete classes):
    - test_abstract_experiment_cannot_be_instantiated
    - test_inheritance_hierarchy
    - test_access_policy_polymorphism
    - test_process_dataset_polymorphism

----------------------------------------------------------------------
These tests validate internal correctness of individual classes
without relying on external dependencies or multi-component workflows.
----------------------------------------------------------------------


======================================================================
INTEGRATION TESTS (Interactions between 2+ components)
======================================================================

Researcher + Dataset:
    - test_append_and_read_csv_roundtrip
    - test_ethics_report_delegates_to_dataset

Analysis + Dataset:
    - test_run_simple_filter_and_groupby_pipeline

Experiments + Dataset + Polymorphism:
    - test_render_overview_polymorphic
    - test_research_project_composition

Analysis + Experiment:
    - test_full_pipeline_lab_experiment (partial system-level behavior)

----------------------------------------------------------------------
These tests ensure that independent modules coordinate correctly,
verifying data flow, delegation, and object relationships.
----------------------------------------------------------------------


======================================================================
SYSTEM TESTS (End-to-end workflows, I/O + processing + pipeline)
======================================================================

1. test_full_pipeline_lab_experiment
    - Loads dataset into experiment, processes it, runs analysis.

2. test_project_overview_and_policies
    - High-level composition of multiple experiments under a project.

3. test_end_to_end_research_workflow  <-- NEW FULL SYSTEM TEST
    - Researcher writes CSV → loads → attaches to experiment →
      processes → pipeline transforms → writes output file →
      validates data + persistence.

----------------------------------------------------------------------
These tests simulate realistic user workflows touching the entire system:
IO, Dataset management, Experiments, Analysis, and filesystem persistence.
----------------------------------------------------------------------

"""


import os
import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from researcher import Researcher
from Sample import Sample
from dataset import Dataset
from analysis import Analysis
from experiment import (
    AbstractExperiment,
    Experiment,
    LabExperiment,
    FieldStudy,
    Survey,
    ResearchProject,
    render_overview,
)


# ============================================================
#   DATASET TESTS
# ============================================================

class TestDataset(unittest.TestCase):
    """Tests for Dataset class: QA, cleaning, schema, plotting."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "A": [1, None, 3],
                "B": [" x ", "Y", None],
                "C": [1, 1, 1],
            }
        )
        self.ds = Dataset(self.df, name="demo")

    def test_detect_missing(self):
        """detect_missing should count NA values per column."""
        missing = self.ds.detect_missing()
        self.assertEqual(missing["A"], 1)
        self.assertEqual(missing["B"], 1)
        self.assertEqual(missing["C"], 0)

    def test_detect_duplicates(self):
        """detect_duplicates should find rows that are duplicates."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        ds = Dataset(df, name="dups")
        dup_idx = ds.detect_duplicates()
        self.assertEqual(list(dup_idx), [0, 1])

    def test_standardize_column_names_snake_case(self):
        """standardize_column_names should normalize and ensure uniqueness."""
        df = pd.DataFrame({"First Name": [1], "first-name": [2], "": [3]})
        ds = Dataset(df, name="cols")
        out = ds.standardize_column_names(case="snake")
        cols = list(out.df.columns)
        # Expect unique, normalized names
        self.assertEqual(len(cols), 3)
        self.assertEqual(len(set(cols)), 3)
        self.assertTrue(all(c != "" for c in cols))

    def test_clean_strings(self):
        """clean_strings should strip, lower, and collapse spaces."""
        df = pd.DataFrame({"txt": ["  Hello   WORLD  ", None]})
        ds = Dataset(df, name="text")
        out = ds.clean_strings(["txt"])
        self.assertEqual(out.df.loc[0, "txt"], "hello world")
        self.assertTrue(pd.isna(out.df.loc[1, "txt"]))

    def test_anonymize(self):
        """anonymize should mask non-null values with '***'."""
        df = pd.DataFrame({"email": ["a@example.com", None]})
        ds = Dataset(df, name="pii")
        out = ds.anonymize(["email"])
        self.assertEqual(out.df.loc[0, "email"], "***")
        self.assertTrue(pd.isna(out.df.loc[1, "email"]))

    def test_enforce_schema_basic(self):
        """enforce_schema should coerce dtypes and track issues."""
        df = pd.DataFrame(
            {
                "age": ["10", "20", "bad"],
                "active": ["yes", "no", "maybe"],
            }
        )
        ds = Dataset(df, name="schema")
        schema = {
            "age": {"dtype": "int", "nullable": True},
            "active": {"dtype": "bool", "nullable": True},
        }
        out, report = ds.enforce_schema(schema)
        self.assertIsInstance(out, Dataset)
        self.assertIn("age", report["coercions"])
        self.assertIn("active", report["coercions"])
        self.assertTrue(pd.api.types.is_integer_dtype(out.df["age"]))
        self.assertTrue(pd.api.types.is_bool_dtype(out.df["active"].dropna()))

    def test_generate_data_report(self):
        """generate_data_report should return a matplotlib Figure."""
        df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
        ds = Dataset(df, name="plot")
        fig = ds.generate_data_report("x", "y", "Title")
        self.assertIsInstance(fig, plt.Figure)

    def test_generate_data_report_invalid_column_raises(self):
        """generate_data_report should raise if columns are missing."""
        df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
        ds = Dataset(df, name="plot")
        with self.assertRaises(KeyError):
            ds.generate_data_report("bad", "y", "Title")


# ============================================================
#   RESEARCHER TESTS
# ============================================================

class TestResearcher(unittest.TestCase):
    """Tests for Researcher workspace, CSV IO, and ethics orchestration."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmpdir.name)
        self.researcher = Researcher("Alice", workspace=self.workspace)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_ensure_dir_creates_subdir(self):
        """ensure_dir should create and return a subdirectory under workspace."""
        sub = self.researcher.ensure_dir("subdir")
        self.assertTrue(sub.exists())
        self.assertTrue(sub.is_dir())
        self.assertTrue(str(sub).startswith(str(self.workspace)))

    def test_append_and_read_csv_roundtrip(self):
        """append_csv followed by read_csv should preserve rows and columns."""
        rows = [
            {"id": 1, "val": "a"},
            {"id": 2, "val": "b"},
        ]
        csv_path = self.workspace / "data.csv"
        self.researcher.append_csv(rows, csv_path)
        ds = self.researcher.read_csv(csv_path)
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds.df), 2)
        self.assertEqual(set(ds.df.columns), {"id", "val"})

    def test_read_csv_missing_file_raises(self):
        """read_csv should raise FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            self.researcher.read_csv("no_such_file.csv")

    def test_download_file_invalid_scheme_raises(self):
        """download_file should reject non-http/https schemes."""
        with self.assertRaises(ValueError):
            self.researcher.download_file("ftp://example.com/file.txt")

    def test_ethics_report_delegates_to_dataset(self):
        """ethics_report should call Dataset.validate_research_ethics_compliance."""
        df = pd.DataFrame(
            {
                "email": ["a@example.com", None],
                "consent": ["yes", "no"],
            }
        )
        ds = Dataset(df, name="ethics")
        report = self.researcher.ethics_report(
            ds,
            pii_cols=["email"],
            consent_col="consent",
        )
        self.assertIsInstance(report, dict)
        self.assertIn("compliant", report)
        self.assertIn("issues", report)


# ============================================================
#   SAMPLE TESTS
# ============================================================

class TestSample(unittest.TestCase):
    """Tests for Sample class as a light DataFrame wrapper."""

    def test_init_requires_dataframe(self):
        """Sample should only accept a pandas DataFrame."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        s = Sample(df)
        self.assertIsInstance(s.data, pd.DataFrame)
        self.assertEqual(list(s.data.columns), ["x"])

        with self.assertRaises(TypeError):
            Sample("not a df")  # type: ignore[arg-type]

    def test_sample_has_independent_copy(self):
        """Sample.data should be a copy, not a reference to the original DataFrame."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        s = Sample(df)
        df.loc[0, "x"] = 999
        # Sample should not see the mutation
        self.assertEqual(s.data.loc[0, "x"], 1)


# ============================================================
#   ANALYSIS TESTS
# ============================================================

class TestAnalysis(unittest.TestCase):
    """Tests for Analysis pipeline behavior and delegation to Dataset."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "user": [1, 1, 2, 2],
                "amount": [5, -1, 3, 7],
                "status": ["ok", "ok", "bad", "ok"],
            }
        )
        self.ds = Dataset(self.df, name="transactions")

    def test_run_simple_filter_and_groupby_pipeline(self):
        """Pipeline should filter and aggregate correctly."""
        steps = [
            {"op": "filter", "expr": "amount >= 0"},
            {"op": "groupby_agg", "by": ["user"], "metrics": {"amount": "sum"}},
            {"op": "sort", "by": ["amount"], "ascending": False},
        ]
        pipe = Analysis(name="demo_pipe", steps=steps)
        out, log = pipe.run_pipeline(self.ds)

        self.assertIsInstance(out, Dataset)
        self.assertEqual(list(out.df.columns), ["user", "amount"])
        # After filtering negatives:
        # user 1: 5
        # user 2: 3 + 7 = 10
        # Sorted desc => first row user 2, amount 10
        self.assertEqual(out.df.iloc[0]["user"], 2)
        self.assertEqual(out.df.iloc[0]["amount"], 10)
        self.assertEqual(len(log), 3)

    def test_run_pipeline_missing_column_in_select_raises(self):
        """select op with missing columns should raise KeyError."""
        steps = [
            {"op": "select", "cols": ["user", "missing_col"]},
        ]
        pipe = Analysis(name="bad_pipe", steps=steps)
        with self.assertRaises(ValueError):
            pipe.run_pipeline(self.ds)

    def test_summarize_delegates_to_dataset(self):
        """summarize should call Dataset.calculate_statistical_summary."""
        pipe = Analysis(name="summary_only", steps=[])
        summary = pipe.summarize(self.ds)
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn("amount", summary.index)

    def test_plot_bar_delegates_to_dataset(self):
        """plot_bar should call Dataset.generate_data_report and return a Figure."""
        pipe = Analysis(name="plot_only", steps=[])
        fig = pipe.plot_bar(self.ds, x="user", y="amount", title="Amounts")
        self.assertIsInstance(fig, plt.Figure)


# ============================================================
#   EXPERIMENTS & PROJECT TESTS
# ============================================================

class TestExperiments(unittest.TestCase):
    """Tests for ABC, inheritance, polymorphism, and composition in experiments."""

    def test_abstract_experiment_cannot_be_instantiated(self):
        """AbstractExperiment must not be instantiable."""
        with self.assertRaises(TypeError):
            AbstractExperiment("X", "Invalid")  # type: ignore[abstract]

    def test_inheritance_hierarchy(self):
        """Concrete experiment types must inherit from AbstractExperiment."""
        exp = Experiment("E0", "Base")
        lab = LabExperiment("L1", "Lab Exp")
        field = FieldStudy("F1", "Field Exp")
        survey = Survey("S1", "Survey Exp")

        self.assertIsInstance(exp, AbstractExperiment)
        self.assertIsInstance(lab, AbstractExperiment)
        self.assertIsInstance(field, AbstractExperiment)
        self.assertIsInstance(survey, AbstractExperiment)

    def test_access_policy_polymorphism(self):
        """Different experiment types should expose different access policies."""
        lab = LabExperiment("L1", "Lab Exp")
        field = FieldStudy("F1", "Field Exp")
        survey = Survey("S1", "Survey Exp")

        lab_policy = lab.access_policy().lower()
        field_policy = field.access_policy().lower()
        survey_policy = survey.access_policy().lower()

        self.assertIn("lab-restricted", lab_policy)
        self.assertIn("field conditions", field_policy)
        self.assertIn("anonymous", survey_policy)
        self.assertNotEqual(lab_policy, field_policy)
        self.assertNotEqual(field_policy, survey_policy)

    def test_process_dataset_polymorphism(self):
        """process_dataset should yield different output names for each type."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        base_ds = Dataset(df, name="raw")

        exp = Experiment("E0", "Base")
        lab = LabExperiment("L1", "Lab Exp")
        field = FieldStudy("F1", "Field Exp")
        survey = Survey("S1", "Survey Exp")

        out_base = exp.process_dataset(base_ds)
        out_lab = lab.process_dataset(base_ds)
        out_field = field.process_dataset(base_ds)
        out_survey = survey.process_dataset(base_ds)

        # All names should contain some notion of "processed" and differ
        names = {out_base.name, out_lab.name, out_field.name, out_survey.name}
        self.assertEqual(len(names), 4)
        self.assertTrue(any("processed" in n.lower() for n in names))

    def test_research_project_composition(self):
        """ResearchProject should compose multiple experiments."""
        proj = ResearchProject("P1", "Climate Study")
        lab = LabExperiment("L1", "Lab Exp")
        field = FieldStudy("F1", "Field Exp")
        proj.add_experiment(lab)
        proj.add_experiment(field)

        self.assertEqual(len(proj.experiments), 2)
        self.assertIn(lab, proj.experiments)
        self.assertIn(field, proj.experiments)

        summary = proj.summary()
        self.assertIn("Climate Study", summary)
        self.assertIn("L1", summary)
        self.assertIn("F1", summary)

    def test_render_overview_polymorphic(self):
        """render_overview should call access_policy polymorphically."""
        lab = LabExperiment("L1", "Lab Exp")
        field = FieldStudy("F1", "Field Exp")
        survey = Survey("S1", "Survey Exp")

        out = render_overview([lab, field, survey])
        self.assertIn("LabExperiment(L1)", out)
        self.assertIn("FieldStudy(F1)", out)
        self.assertIn("Survey(S1)", out)
        self.assertIn("lab-restricted", out.lower())
        self.assertIn("field conditions", out.lower())
        self.assertIn("anonymous", out.lower())


class TestSystemIntegration(unittest.TestCase):
    """End-to-end tests tying together Dataset, Experiments, Analysis, and ResearchProject."""

    def setUp(self):
        # Dataset
        self.df = pd.DataFrame(
            {
                "user": [1, 1, 2, 2],
                "amount": [10, -5, 3, 7],
                "city": ["A", "A", "B", "B"],
            }
        )
        self.ds = Dataset(self.df, name="raw")

        # Experiments & project
        self.lab = LabExperiment("L1", "Lab Exp")
        self.field = FieldStudy("F1", "Field Exp", region="MD")
        self.survey = Survey("S1", "Survey Exp", consent_col="consent")

        self.project = ResearchProject("P1", "Integrated Study")
        self.project.add_experiment(self.lab)
        self.project.add_experiment(self.field)
        self.project.add_experiment(self.survey)

        # Analysis pipeline
        self.pipe = Analysis(
            name="integ_pipe",
            steps=[
                {"op": "filter", "expr": "amount >= 0"},
                {"op": "groupby_agg", "by": ["user"], "metrics": {"amount": "sum"}},
                {"op": "sort", "by": ["amount"], "ascending": False},
            ],
        )

    def test_full_pipeline_lab_experiment(self):
        """Attach dataset to lab experiment, process, and run analysis."""
        self.lab.attach_dataset(self.df, name="lab_raw")
        processed = self.lab.process_dataset(self.ds)
        out, log = self.pipe.run_pipeline(processed)

        self.assertIsInstance(out, Dataset)
        self.assertGreater(len(log), 0)
        self.assertIn("user", out.df.columns)
        self.assertIn("amount", out.df.columns)

    def test_project_overview_and_policies(self):
        """Project summary and render_overview should reflect all experiments."""
        proj_summary = self.project.summary()
        overview = render_overview(self.project.experiments)

        self.assertIn("Integrated Study", proj_summary)
        self.assertIn("L1", proj_summary)
        self.assertIn("F1", proj_summary)
        self.assertIn("S1", proj_summary)

        self.assertIn("LabExperiment(L1)", overview)
        self.assertIn("FieldStudy(F1)", overview)
        self.assertIn("Survey(S1)", overview)

    def test_end_to_end_research_workflow(self):
        """
        System Test:
        Complete end-to-end research workflow including:

        - Researcher creates a workspace and writes raw CSV
        - Researcher loads CSV into a Dataset
        - Dataset is attached to a LabExperiment
        - LabExperiment processes the dataset
        - Analysis pipeline runs transformations
        - Researcher writes final processed dataset back to CSV

        This test touches I/O, dataset cleaning, experiment processing,
        analysis pipelines, and filesystem persistence.
        """

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            r = Researcher("Alice", workspace=ws)

            # Step 1: Write raw CSV input
           
            input_rows = [
            {"user": 1, "amount": 4, "city": "A"},
             {"user": 1, "amount": -5, "city": "A"},
            {"user": 2, "amount": 3, "city": "B"},
            {"user": 2, "amount": 7, "city": "B"},]

            input_path = r.append_csv(input_rows, "raw_data.csv")

            # Step 2: Load dataset
            ds = r.read_csv("raw_data.csv")

            # Step 3: Attach dataset to experiment
            lab = LabExperiment("L100", "Lab System Test", biosafety_level=2)
            lab.attach_dataset(ds.df, name="lab_raw")

            # Step 4: Process experiment dataset
            processed = lab.process_dataset(ds)
            self.assertIsInstance(processed, Dataset)

            # Step 5: Run analysis pipeline
            pipe = Analysis(
                name="sys_pipeline",
                steps=[
                    {"op": "filter", "expr": "amount >= 0"},
                    {"op": "groupby_agg", "by": ["user"], "metrics": {"amount": "sum"}},
                    {"op": "sort", "by": ["amount"], "ascending": False},
                ]
            )
            final_ds, log = pipe.run_pipeline(processed)

            # Expected: user=2 has highest total amount: 3+7 = 10
            self.assertEqual(final_ds.df.iloc[0]["user"], 2)
            self.assertEqual(final_ds.df.iloc[0]["amount"], 10)
            self.assertGreater(len(log), 0)

            # Step 6: Save final dataset
            output_path = r.write_csv(final_ds, "final_output.csv")
            self.assertTrue(output_path.exists())

            # Step 7: Validate saved contents
            reloaded = r.read_csv("final_output.csv")
            self.assertIn("user", reloaded.df.columns)
            self.assertIn("amount", reloaded.df.columns)
            self.assertEqual(len(reloaded.df), 2)  # two users in totals

            # System test complete successfully

if __name__ == "__main__":
    unittest.main()

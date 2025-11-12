from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt

from dataset import Dataset

__all__ = ["Analysis", "SUPPORTED_OPS"]

SUPPORTED_OPS = {
    "filter",
    "groupby_agg",
    "pivot",
    "melt",
    "select",
    "rename",
    "sort",
    "fillna",
    "dropna",
    "eval",
    "assign",
}


class Analysis:
    """Composable DataFrame pipelines + summaries/plots over a `Dataset`.

    Supported ops: filter, groupby_agg, pivot, melt, select, rename, sort,
                   fillna, dropna, eval, assign

    Notes
    -----
    - This class does *not* perform IO. It composes a `Dataset`.
    - Steps are dicts with an 'op' key and op-specific parameters.
    """

    def __init__(self, name: str = "analysis", steps: Sequence[Mapping[str, Any]] | None = None):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if steps is not None and not isinstance(steps, (list, tuple)):
            raise TypeError("steps must be a sequence of mapping objects or None")
        if steps is not None and not all(isinstance(s, Mapping) for s in steps):
            raise TypeError("each step must be a mapping")
        self._name = name
        self._steps: List[Mapping[str, Any]] = list(steps or [])

    # ---- properties ----
    @property
    def name(self) -> str:
        return self._name

    @property
    def steps(self) -> Sequence[Mapping[str, Any]]:
        """Defensive copy of configured steps."""
        return [dict(s) for s in self._steps]

    def __len__(self) -> int:
        return len(self._steps)

    # ---- step mgmt ----
    def add_step(self, step: Mapping[str, Any]) -> "Analysis":
        if not isinstance(step, Mapping):
            raise TypeError("step must be a mapping")
        if "op" not in step or not isinstance(step["op"], str):
            raise ValueError("step must include an 'op' string")
        self._steps.append(dict(step))
        return self

    def add_steps(self, steps: Sequence[Mapping[str, Any]]) -> "Analysis":
        for s in steps:
            self.add_step(s)
        return self

    def insert_step(self, index: int, step: Mapping[str, Any]) -> "Analysis":
        if not isinstance(step, Mapping):
            raise TypeError("step must be a mapping")
        if "op" not in step or not isinstance(step["op"], str):
            raise ValueError("step must include an 'op' string")
        self._steps.insert(index, dict(step))
        return self

    def replace_step(self, index: int, step: Mapping[str, Any]) -> "Analysis":
        if not isinstance(step, Mapping):
            raise TypeError("step must be a mapping")
        if "op" not in step or not isinstance(step["op"], str):
            raise ValueError("step must include an 'op' string")
        self._steps[index] = dict(step)
        return self

    def remove_step(self, index: int) -> "Analysis":
        del self._steps[index]
        return self

    def set_steps(self, steps: Sequence[Mapping[str, Any]]) -> "Analysis":
        self._steps = []
        return self.add_steps(steps)

    # ---- validation ----
    def validate_steps(self) -> None:
        """Validate supported ops and basic required keys."""
        for i, step in enumerate(self._steps, start=1):
            if not isinstance(step, Mapping):
                raise ValueError(f"Step {i}: each step must be a mapping (dict-like)")
            op = step.get("op")
            if not isinstance(op, str):
                raise ValueError(f"Step {i}: missing or invalid 'op'")
            if op not in SUPPORTED_OPS:
                raise ValueError(f"Step {i}: unsupported op '{op}'. Supported: {sorted(SUPPORTED_OPS)}")

            # minimal per-op key presence checks (not exhaustive)
            if op == "filter" and "expr" not in step:
                raise ValueError(f"Step {i} (filter): missing 'expr'")
            if op == "groupby_agg" and not all(k in step for k in ("by", "metrics")):
                raise ValueError(f"Step {i} (groupby_agg): requires 'by' and 'metrics'")
            if op == "pivot" and not all(k in step for k in ("index", "columns")):
                raise ValueError(f"Step {i} (pivot): requires 'index' and 'columns'")
            if op == "melt" and not all(k in step for k in ("id_vars", "value_vars")):
                raise ValueError(f"Step {i} (melt): requires 'id_vars' and 'value_vars'")
            if op == "select" and "cols" not in step:
                raise ValueError(f"Step {i} (select): missing 'cols'")
            if op == "rename" and "mapping" not in step:
                raise ValueError(f"Step {i} (rename): missing 'mapping'")
            if op == "sort" and "by" not in step:
                raise ValueError(f"Step {i} (sort): missing 'by'")
            if op == "fillna" and "value" not in step:
                raise ValueError(f"Step {i} (fillna): missing 'value'")
            if op == "eval" and "expr" not in step:
                raise ValueError(f"Step {i} (eval): missing 'expr'")
            if op == "assign" and "values" not in step:
                raise ValueError(f"Step {i} (assign): missing 'values'")

    # ---- core ----
    def _apply_step(self, out: pd.DataFrame, step: Mapping[str, Any]) -> Tuple[pd.DataFrame, str]:
        """Apply a single step and return (new_df, note)."""
        op = step["op"]

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
                    val = out.eval(v[1:])
                    if not isinstance(val, pd.Series):
                        raise ValueError(f"assign '{k}' expression must yield a Series.")
                    to_assign[k] = val
                else:
                    to_assign[k] = v
            out = out.assign(**to_assign)
            note = f"assign keys={list(values.keys())!r}"

        else:
            raise ValueError(f"Unsupported op: {op}")

        return out, note

    def run_pipeline(self, dataset: Dataset, runlog: Optional[List[dict]] = None) -> Tuple[Dataset, List[dict]]:
        """
        Execute configured steps on a Dataset; return (new Dataset, runlog).

        Returns
        -------
        (Dataset, list[dict])
            The transformed dataset and a list of step logs, each with:
            step, op, rows_before/after, cols_before/after, note.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset")

        # Validate before running
        self.validate_steps()

        out = dataset.df.copy()
        log: List[dict] = [] if runlog is None else runlog

        for i, step in enumerate(self._steps, start=1):
            rows_before = len(out)
            cols_before = list(out.columns)
            op = step["op"]

            try:
                out, note = self._apply_step(out, step)
            except Exception as e:
                raise ValueError(f"Error in step {i} ({op}): {e}") from e
            finally:
                log.append(
                    {
                        "step": i,
                        "op": op,
                        "rows_before": rows_before,
                        "rows_after": len(out),
                        "cols_before": cols_before,
                        "cols_after": list(out.columns),
                        "note": note if 'note' in locals() else "",
                    }
                )

        return Dataset(out, name=f"{dataset.name}::pipeline[{self._name}]"), log

    # ---- convenience wrappers ----
    def summarize(self, dataset: Dataset, include_categoricals: bool = True) -> pd.DataFrame:
        """Return a combined numeric/categorical summary via Dataset."""
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset")
        return dataset.calculate_statistical_summary(include_categoricals=include_categoricals)

    def plot_bar(self, dataset: Dataset, x: str, y: str, title: str) -> plt.Figure:
        """Produce a bar chart via Dataset.generate_data_report (returns the Figure)."""
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset")
        return dataset.generate_data_report(x, y, title)

    def dry_run(self, dataset: Dataset, n: int = 5) -> pd.DataFrame:
        """
        Run the pipeline on a copy of the data and return head(n).
        Useful for quick previews without touching your original Dataset.
        """
        ds, _ = self.run_pipeline(dataset, runlog=[])
        return ds.df.head(n)

    # ---- string reps ----
    def __str__(self) -> str:
        return f"Analysis(name={self._name}, steps={len(self._steps)})"

    def __repr__(self) -> str:
        return f"Analysis(name={self._name!r}, steps={self._steps!r})"

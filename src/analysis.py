from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt

from dataset import Dataset


class Analysis:
    """Composable DataFrame pipelines + summaries/plots over a `Dataset`.

    Supported ops: filter, groupby_agg, pivot, melt, select, rename, sort,
                   fillna, dropna, eval, assign
    """

    def __init__(self, name: str = "analysis", steps: Sequence[Mapping[str, Any]] | None = None):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if steps is not None and not isinstance(steps, (list, tuple)):
            raise TypeError("steps must be a sequence of mapping objects or None")
        if steps is not None and not all(isinstance(s, Mapping) for s in steps):
            raise TypeError("each step must be a mapping")
        self._name = name
        self._steps = list(steps or [])

    # ---- properties ----
    @property
    def name(self) -> str:
        return self._name

    @property
    def steps(self) -> Sequence[Mapping[str, Any]]:
        return [dict(s) for s in self._steps]

    # ---- step mgmt ----
    def add_step(self, step: Mapping[str, Any]) -> "Analysis":
        if not isinstance(step, Mapping):
            raise TypeError("step must be a mapping")
        if "op" not in step or not isinstance(step["op"], str):
            raise ValueError("step must include an 'op' string")
        self._steps.append(dict(step))
        return self

    def clear_steps(self) -> "Analysis":
        self._steps.clear()
        return self

    # ---- core ----
    def run_pipeline(self, dataset: Dataset, runlog: Optional[List[dict]] = None) -> Tuple[Dataset, List[dict]]:
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset")

        out = dataset.df.copy()
        log: List[dict] = [] if runlog is None else runlog

        for i, step in enumerate(self._steps, start=1):
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
                        "note": note,
                    }
                )

        return Dataset(out, name=f"{dataset.name}::pipeline[{self._name}]"), log

    # ---- convenience wrappers ----
    def summarize(self, dataset: Dataset, include_categoricals: bool = True) -> pd.DataFrame:
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset")
        return dataset.calculate_statistical_summary(include_categoricals=include_categoricals)

    def plot_bar(self, dataset: Dataset, x: str, y: str, title: str) -> plt.Figure:
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset")
        return dataset.generate_data_report(x, y, title)

    def __str__(self) -> str:
        return f"Analysis(name={self._name}, steps={len(self._steps)})"

    def __repr__(self) -> str:
        return f"Analysis(name={self._name!r}, steps={self._steps!r})"


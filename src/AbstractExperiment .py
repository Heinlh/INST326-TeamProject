class LabExperiment(Experiment):
    """Wet-lab style experiment with stricter access and numeric coercion emphasis."""
    def __init__(self, experiment_id: str, title: str, biosafety_level: int = 1):
        super().__init__(experiment_id, title)
        self.biosafety_level = int(biosafety_level)

    def access_policy(self) -> str:
        return f"Restricted to approved lab members (BSL-{self.biosafety_level}) and PI; publish aggregated results only."

    def process(self) -> Dataset:
        ds = super().process()
        # Lab-specific: ensure measurement columns are numeric if present
        numeric_like = [c for c in ds.df.columns if any(k in c for k in ("conc", "qty", "mass", "ph", "temp"))]
        if numeric_like:
            schema = {c: {"dtype": "float", "nullable": True} for c in numeric_like}
            ds, _ = ds.enforce_schema(schema)
        self._dataset = ds
        return ds


class FieldStudy(Experiment):
    """Field study with location/time normalization and moderate access controls."""
    def __init__(self, experiment_id: str, title: str, region: str = "MD"):
        super().__init__(experiment_id, title)
        self.region = region

    def access_policy(self) -> str:
        return "Research team + IRB-approved collaborators; de-identify GPS traces; share summaries externally."

    def process(self) -> Dataset:
        ds = super().process()
        # Field-specific: normalize timestamps/locations when present
        schema = {}
        if "timestamp" in ds.df.columns:
            schema["timestamp"] = {"dtype": "datetime", "nullable": False}
        for c in ("lat", "lon", "latitude", "longitude"):
            if c in ds.df.columns:
                schema[c] = {"dtype": "float", "nullable": True}
        if schema:
            ds, _ = ds.enforce_schema(schema)
        self._dataset = ds
        return ds


class Survey(Experiment):
    """Survey with consent and PII checks, open policy for aggregates."""
    def __init__(self, experiment_id: str, title: str, consent_col: str = "consent"):
        super().__init__(experiment_id, title)
        self.consent_col = consent_col

    def access_policy(self) -> str:
        return "PII restricted; aggregate results public; raw responses only to IRB-approved team."

    def process(self) -> Dataset:
        if self._dataset is None:
            raise RuntimeError("No dataset attached")
        ds = self._dataset.standardize_column_names(case="snake")
        # Enforce booleans on consent column if present
        if self.consent_col in ds.df.columns:
            ds, _ = ds.enforce_schema({self.consent_col: {"dtype": "bool", "nullable": False}})
        # Ethics: flag PII without consent
        ethics = ds.validate_research_ethics_compliance(
            pii_cols=[c for c in ds.df.columns if any(k in c for k in ("email", "phone", "name"))],
            consent_col=self.consent_col if self.consent_col in ds.df.columns else None,
        )
        if not ethics["compliant"]:
            # In production, you'd log or raise; here we keep it simple.
            pass
        self._dataset = ds
        return ds

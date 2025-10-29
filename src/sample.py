class Sample:
    def __init__(self, data: pd.DataFrame):

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")
        self.data = data.copy()

    def detect_missing(self, cols: Sequence[str] | None = None) -> pd.Series:
    
        if cols is not None:
            missing_cols = [c for c in cols if c not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            return self.data[cols].isna().sum()
    
        return self.data.isna().sum()

    def detect_duplicates(self, subset: Sequence[str] | None = None) -> pd.Index:
    
        if subset is not None:
            if not isinstance(subset, (list, tuple)):
                raise TypeError("subset must be a sequence of column names or None")
            missing = [c for c in subset if c not in self.data.columns]
            if missing:
                raise ValueError(f"subset contains columns not in data: {missing}")
    
        mask = self.data.duplicated(subset=subset, keep=False)
        return self.data.index[mask]

    def standardize_column_names(self, case: str = "snake") -> pd.DataFrame:
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
        
            new_cols = [normalize(c) for c in self.data.columns]
            return df.rename(columns=dict(zip(self.data.columns, new_cols)))

    def clean_strings(
        self,
        cols: Sequence[str],
        *,
        strip: bool = True,
        lower: bool = True,
        collapse_spaces: bool = True,
    ) -> pd.DataFrame:
       
        if not isinstance(cols, (list, tuple)):
            raise TypeError("cols must be a sequence of column names")
    
        self.data.missing()
        for col in cols:
                series = self.data[col].astype(str)
                if strip:
                    series = series.str.strip()
                if lower:
                    series = series.str.lower()
                if collapse_spaces:
                    series = series.str.replace(r"\s+", " ", regex=True)
                self.data[col] = series

    def anonymize_participant_data(self, cols: Sequence[str]) -> None:
        
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")

        for c in cols:
            self.data.loc[self.data[c].notna(), c] = "***"

    def validate_research_ethics_compliance(
        self,
        pii_cols: Sequence[str] | None = None,
        consent_col: str | None = None,
    ) -> dict:
        
        issues: list[str] = []
        pii = list(pii_cols or [])

        consent = (
            self.data[consent_col]
            .astype(str)
            .str.lower()
            .isin({"1", "true", "yes", "y"})
            if (consent_col and consent_col in self.data.columns)
            else None
        )

        for c in pii:
            if c not in self.data.columns or not self.data[c].notna().any():
                continue
            if consent is None:
                issues.append(f"PII detected in '{c}' without a consent column")
            else:
                n = int((self.data[c].notna() & ~consent).sum())
                if n:
                    issues.append(f"{n} row(s) with PII in '{c}' without consent")

        return {"compliant": not issues, "issues": issues}

    def enforce_schema(
        self,
        schema: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        
        if not isinstance(schema, dict):
            raise TypeError("schema must be a mapping (dict) of column -> rules")

        supported_dtypes = {"int", "float", "string", "bool", "datetime"}
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
            allowed_values = set(rules.get("allowed", set()))

            if col not in self.data.columns:
                report["errors"].append(f"Missing expected column: '{col}'")
                self.data[col] = pd.Series([pd.NA] * len(self.data), index=self.data.index)

            before = self.data[col].copy()

            try:
                
                if dtype == "string":
                    self.data[col] = self.data[col].astype("string")
                elif dtype == "int":
                    self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
                    if nullable:
                        self.data[col] = self.data[col].astype("Int64")
                    else:
                        if self.data[col].isna().any():
                            pass 
                        else:
                            self.data[col] = self.data[col].astype("int64")
                elif dtype == "float":
                    self.data[col] = pd.to_numeric(self.data[col], errors="coerce").astype("Float64")
                elif dtype == "bool":
                    self.data[col] = coerce_bool(self.data[col])
                    if not nullable and not self.data[col].isna().any():
                        self.data[col] = self.data[col].astype(bool)
                elif dtype == "datetime":
                    if dt_fmt is None:
                        self.data[col] = pd.to_datetime(self.data[col], errors="coerce", utc=False)
                    else:
                        fmts = dt_fmt if isinstance(dt_fmt, (list, tuple)) else [dt_fmt]
                        parsed = pd.Series(pd.NaT, index=self.data.index, dtype="datetime64[ns]")
                        raw = self.data[col]
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
                        self.data[col] = parsed

                
                if not before.equals(self.data[col]):
                    report["coercions"].setdefault(col, []).append("Type coerced")

                
                if not nullable and self.data[col].isna().any():
                    invalid_null_count = self.data[col].isna().sum()
                    report["invalid_nulls"].setdefault(col, []).append(
                        f"Found {invalid_null_count} null(s) in non-nullable column."
                    )

               
                if allowed_values:
                    
                    allowed_values_coerced = [
                        (pd.to_datetime(v, errors='coerce', format=fmts[0], utc=False) if dtype == 'datetime' and fmts else pd.to_datetime(v, errors='coerce', utc=False)) if dtype == 'datetime'
                        else coerce_bool(pd.Series([v]))[0] if dtype == 'bool'
                        else v
                        for v in allowed_values
                    ]

                    invalid_mask = self.data[col].notna() & ~self.data[col].isin(allowed_values_coerced)
                    if invalid_mask.any():
                        invalid_count = invalid_mask.sum()
                        report["invalid_values"].setdefault(col, []).append(
                            f"Found {invalid_count} row(s) with values not in allowed set."
                        )

            except Exception as e:
                report["errors"].append(f"Coercion or validation error for '{col}': {e}")
                self.data[col] = before 

        return report


    def get_data(self) -> pd.DataFrame:
        
        return self.data.copy()

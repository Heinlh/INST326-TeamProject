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

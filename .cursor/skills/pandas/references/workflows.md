# Pandas Workflows Reference

## Contents
- Data Loading Pipeline
- Statistical Analysis Prep
- Result Serialization
- Performance Optimization

---

## Data Loading Pipeline

The codebase uses a 10-step cleaning pipeline with parquet checkpoints.

### Workflow Checklist

Copy this checklist and track progress:
- [ ] Step 1: Load CSV with explicit dtypes
- [ ] Step 2: Rename columns via COLUMN_MAP
- [ ] Step 3: Drop nulls in required fields
- [ ] Step 4: Convert types (Int64 → int after dropna)
- [ ] Step 5: Filter by length constraints
- [ ] Step 6: Filter by ASCII ratio
- [ ] Step 7: Detect language, keep English
- [ ] Step 8: Remove duplicate narratives
- [ ] Step 9: Map product/issue categories
- [ ] Step 10: Exclude invalid category pairs

### Implementation Pattern

```python
def load_and_clean(data_path: Path) -> pd.DataFrame:
    # Step 1: Load with types
    df = pd.read_csv(
        data_path,
        usecols=list(COLUMN_MAP.keys()),
        dtype={"Complaint ID": "Int64"},
        low_memory=False,
    )
    
    # Step 2: Rename
    df = df.rename(columns=COLUMN_MAP)
    
    # Step 3-4: Null handling and type conversion
    df = df.dropna(subset=["narrative", "complaint_id"])
    df["complaint_id"] = df["complaint_id"].astype(int)
    
    # Step 5: Length filter
    lengths = df["narrative"].str.len()
    df = df[(lengths >= 300) & (lengths <= 1500)].copy()
    
    # Step 6: ASCII filter
    ratios = df["narrative"].apply(ascii_ratio)
    df = df[ratios >= 0.85].copy()
    
    return df
```

### Validation Loop

1. Run cleaning step
2. Check `len(df)` is reasonable
3. If too many rows dropped, adjust threshold and repeat step 2
4. Proceed only when data quality is acceptable

---

## Statistical Analysis Prep

### Merge Workflow

```python
def prepare_analysis_df(
    decisions_df: pd.DataFrame,
    personas_df: pd.DataFrame,
    severity_df: pd.DataFrame,
) -> pd.DataFrame:
    # Step 1: Merge decisions with personas
    merged = decisions_df.merge(
        personas_df[["case_id", "model", "persona_label"]].drop_duplicates(),
        on=["case_id", "model"],
        how="inner",
    )
    
    # Step 2: Add severity
    merged = merged.merge(
        severity_df[["case_id", "monetary_severity", "issue_severity"]],
        on="case_id",
        how="left",
    )
    
    # Step 3: Filter to valid severity
    merged = merged[merged["monetary_severity"].isin(SEVERITY_LEVELS)].copy()
    
    return merged
```

### Cross-Tab for Chi-Squared

```python
def build_contingency(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Build contingency table for dimension × tier."""
    # Validate dimension exists
    if dim not in df.columns:
        raise ValueError(f"Dimension {dim} not in DataFrame")
    
    # Drop nulls before cross-tab
    clean = df.dropna(subset=[dim, "tier"])
    
    # Build and validate
    ct = pd.crosstab(clean[dim], clean["tier"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ValueError("Insufficient categories for chi-squared")
    
    return ct
```

### WARNING: Passing DataFrame to scipy

**The Problem:**

```python
# BAD - scipy receives DataFrame, not array
chi2, p, dof, _ = sp_stats.chi2_contingency(ct)  # May work but unreliable
```

**Why This Breaks:**
1. Some scipy functions don't handle DataFrame index/columns
2. Type coercion may produce unexpected results
3. Debugging becomes harder with implicit conversions

**The Fix:**

```python
# GOOD - Extract .values explicitly
chi2, p, dof, _ = sp_stats.chi2_contingency(ct.values)
```

---

## Result Serialization

### Converting to JSON-Safe Dicts

```python
def serialize_results(df: pd.DataFrame, groupby_col: str) -> Dict[str, Any]:
    """Convert grouped stats to JSON-serializable dict."""
    result = {}
    
    for level in df[groupby_col].unique():
        sub = df[df[groupby_col] == level]
        result[level] = {
            "n": int(len(sub)),
            "mean": float(sub["tier"].mean()),
            "sd": float(sub["tier"].std(ddof=1)),
        }
    
    return result

# Cross-tab to percentages
pct = (ct / ct.sum().sum() * 100).round(1).to_dict()
```

### WARNING: Returning Raw DataFrames

**The Problem:**

```python
# BAD - Returns DataFrame to caller
def analyze(df):
    return df.groupby("persona")["tier"].mean()
```

**Why This Breaks:**
1. DataFrames are mutable, caller may modify
2. JSON serialization fails on DataFrames
3. Type hints become complicated

**The Fix:**

```python
# GOOD - Return dict
def analyze(df: pd.DataFrame) -> Dict[str, float]:
    return df.groupby("persona")["tier"].mean().to_dict()
```

---

## Performance Optimization

### Stratified Sampling

```python
def stratified_sample(df: pd.DataFrame, n: int, strat_col: str) -> pd.DataFrame:
    """Sample proportionally across strata."""
    bucket_counts = df[strat_col].value_counts()
    bucket_fracs = bucket_counts / len(df)
    
    frames = []
    for bucket in bucket_counts.index:
        bucket_df = df[df[strat_col] == bucket]
        k = max(1, int(round(bucket_fracs[bucket] * n)))
        frames.append(bucket_df.sample(n=min(k, len(bucket_df)), random_state=42))
    
    return pd.concat(frames, ignore_index=True)
```

### WARNING: Repeated Concat in Loop

**The Problem:**

```python
# BAD - O(n²) memory allocation
result = pd.DataFrame()
for chunk in chunks:
    result = pd.concat([result, chunk])
```

**Why This Breaks:**
1. Each concat allocates new memory
2. Quadratic time complexity
3. Memory fragmentation

**The Fix:**

```python
# GOOD - Single concat at end
frames = []
for chunk in chunks:
    frames.append(chunk)
result = pd.concat(frames, ignore_index=True)
```

### Avoid iterrows() for Computation

```python
# BAD - Slow iteration
for idx, row in df.iterrows():
    df.loc[idx, "result"] = compute(row["value"])

# GOOD - Vectorized or apply
df["result"] = df["value"].apply(compute)

# BETTER - Vectorized numpy when possible
df["result"] = np.where(df["value"] > 0, df["value"] * 2, 0)
```

**When iterrows is acceptable:** Converting DataFrame rows to custom objects (e.g., Pydantic models, dataclasses) where you need the full row context.

---

## Related Skills

- See the **numpy** skill for vectorized operations
- See the **scipy** skill for statistical tests
- See the **matplotlib** and **seaborn** skills for DataFrame visualization
- See the **pydantic** skill for validating DataFrame-derived objects
# Pandas Patterns Reference

## Contents
- Data Loading & Type Specification
- Filtering & Cleaning
- Merging DataFrames
- GroupBy & Aggregation
- Categorical Data Handling
- Integration with Statistics Libraries

---

## Data Loading & Type Specification

### CSV Loading

```python
# Always specify usecols and dtype for large files
df = pd.read_csv(
    data_path,
    usecols=list(COLUMN_MAP.keys()),
    dtype={"Complaint ID": "Int64"},
    low_memory=False,
)
```

### Parquet Checkpointing

```python
# Save intermediate results
df.to_parquet(path, index=False)

# Resume from checkpoint
df = pd.read_parquet(path)
```

---

## Filtering & Cleaning

### WARNING: SettingWithCopyWarning

**The Problem:**

```python
# BAD - Creates a view, not a copy
filtered = df[df["length"] > 100]
filtered["new_col"] = "value"  # Raises warning, may fail silently
```

**Why This Breaks:**
1. Modifications may not persist to the original DataFrame
2. Pandas cannot determine if you want a view or copy
3. Future pandas versions will enforce stricter copy semantics

**The Fix:**

```python
# GOOD - Explicit copy after filtering
filtered = df[df["length"] > 100].copy()
filtered["new_col"] = "value"
```

### String Operations with `.str` Accessor

```python
# Vectorized string operations
lengths = df["narrative"].str.len()
df = df[(lengths >= 300) & (lengths <= 1500)].copy()

# Regex replacement
df["narrative"] = df["narrative"].apply(lambda t: re.sub(r"\s+", " ", str(t)).strip())
```

### Deduplication with Hashing

```python
hashes = df["narrative"].apply(
    lambda t: hashlib.blake2b(t.encode(), digest_size=16).hexdigest()
)
df = df[~hashes.duplicated(keep="first")].copy()
```

### Null Handling

```python
# Drop rows with nulls in specific columns
df = df.dropna(subset=["narrative", "complaint_id"])

# Fill with defaults
df["product"] = df["product"].fillna("Unknown")
```

---

## Merging DataFrames

### WARNING: Missing Merge Parameters

**The Problem:**

```python
# BAD - Implicit parameters
merged = df1.merge(df2)
```

**Why This Breaks:**
1. Pandas guesses common columns, which may change
2. Default `how="inner"` silently drops unmatched rows
3. Duplicate column names create `_x`, `_y` suffixes unexpectedly

**The Fix:**

```python
# GOOD - Explicit merge parameters
merged = decisions_df.merge(
    personas_df,
    on=["case_id", "model"],
    how="inner",
    suffixes=("", "_persona"),
)
```

### Pre-Merge Deduplication

```python
# Avoid cartesian explosion from duplicate keys
unique_personas = personas_df[["case_id", "model", "label"]].drop_duplicates()
merged = decisions_df.merge(unique_personas, on=["case_id", "model"])
```

---

## GroupBy & Aggregation

### Basic Aggregation

```python
# Mean by group
rates = df.groupby("persona_label")["favorable"].mean()

# Convert to dict for JSON serialization
stats = df.groupby(dimension)["tier"].mean().to_dict()
```

### Cross-Tabulation

```python
# Frequency table
ct = pd.crosstab(df["severity"], df["favorable"])

# Pass .values to scipy
chi2, p, dof, expected = sp_stats.chi2_contingency(ct.values)
```

### Pivot Table

```python
# Reshape: rows=case_id, columns=variant, values=tier
pivot = df.pivot_table(
    index="case_id",
    columns="variant",
    values="tier",
    aggfunc="first",
).dropna()
```

### Validation Before Groupby

```python
# Always check data exists
if persona_col not in df.columns or len(df) < 10:
    return {}

groups = df.groupby(persona_col)[outcome_col].mean()
if len(groups) < 2:
    return {"error": "insufficient_groups"}
```

---

## Categorical Data Handling

### Mapping Categories

```python
df["product_category"] = df["product"].map(PRODUCT_CATEGORIES).fillna("Other")
df["issue_category"] = df["issue"].map(ISSUE_CATEGORIES).fillna("Other")
```

### Membership Filtering

```python
# Filter by allowed values
valid_rows = df[df["severity"].isin(SEVERITY_LEVELS)]

# Exclude specific pairs
pair_key = df["product_category"] + "|" + df["issue_category"]
excluded = {f"{p}|{i}" for p, i in EXCLUDED_PAIRS}
df = df[~pair_key.isin(excluded)].copy()
```

---

## Integration with Statistics Libraries

### Extracting Arrays for scipy

```python
# CRITICAL: Use .values for scipy functions
arr = df["favorable"].values
chi2, p, dof, _ = sp_stats.chi2_contingency(ct.values)
```

### One-Hot Encoding for Regression

See the **statsmodels** skill for regression patterns.

```python
# get_dummies for logistic regression
X = pd.get_dummies(df[predictor_cols], drop_first=True, dtype=float)
X = add_constant(X)  # statsmodels requires constant
```

### Safe Aggregation

```python
def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return a / b

# Check for variance before statistical tests
if df["tier"].nunique() < 2:
    return {"error": "no_variance"}
```
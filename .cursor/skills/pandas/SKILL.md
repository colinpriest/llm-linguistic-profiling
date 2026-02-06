---
name: pandas
description: |
  Processes DataFrames, data transformation, and complaint analysis operations.
  Use when: loading CSV/parquet data, cleaning/filtering DataFrames, merging datasets,
  groupby aggregations, statistical preprocessing, or building analysis pipelines.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# Pandas Skill

DataFrame operations for the LLM fairness experiment pipeline. This codebase uses pandas 2.x with parquet checkpointing, strong typing, and integration with scipy/statsmodels for statistical analysis.

## Quick Start

### Load CSV with Type Specification

```python
df = pd.read_csv(
    data_path,
    usecols=list(COLUMN_MAP.keys()),
    dtype={"Complaint ID": "Int64"},  # Nullable integer
    low_memory=False,
)
```

### Filter with Boolean Masking

```python
# Chain conditions, always .copy() after filtering
lengths = df["narrative"].str.len()
df = df[(lengths >= 300) & (lengths <= 1500)].copy()
```

### Merge with Explicit Parameters

```python
merged = decisions_df.merge(
    personas_df,
    on=["case_id", "model"],
    how="inner",
    suffixes=("", "_persona"),
)
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| Nullable Int | Preserve NaN before cast | `dtype={"col": "Int64"}` |
| Boolean mask | Filter rows | `df[(cond1) & (cond2)]` |
| `.copy()` | Avoid SettingWithCopyWarning | `df = df[mask].copy()` |
| Parquet checkpoint | Pipeline caching | `df.to_parquet(path, index=False)` |
| `.values` | Extract for scipy | `arr = df["col"].values` |

## Common Patterns

### Groupby Aggregation

```python
rates = df.groupby("persona_label")["favorable"].mean()
stats = df.groupby(dimension)["tier"].mean().to_dict()
```

### Cross-Tabulation

```python
ct = pd.crosstab(df["severity"], df["favorable"])
chi2, p, dof, expected = sp_stats.chi2_contingency(ct.values)
```

### Safe Iteration for Object Creation

```python
complaints = [
    Complaint(case_id=row["complaint_id"], text=row["narrative"])
    for _, row in df.iterrows()
]
```

## See Also

- [patterns](references/patterns.md) - Data cleaning, merging, aggregation
- [workflows](references/workflows.md) - Pipeline stages, statistical prep

## Related Skills

- See the **numpy** skill for array operations and scipy integration
- See the **scipy** skill for statistical tests on DataFrame values
- See the **statsmodels** skill for regression on get_dummies output
- See the **scikit-learn** skill for ML preprocessing patterns
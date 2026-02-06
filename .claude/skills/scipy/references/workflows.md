# Scipy Workflows Reference

## Contents
- Persona-Outcome Correlation Analysis
- Severity Amplification Analysis
- Repeated Measures Non-Parametric Testing
- Bootstrap Confidence Intervals
- Multiple Comparison Pipeline

---

## Persona-Outcome Correlation Analysis

**When:** Testing whether LLM-assigned personas correlate with remedy tier outcomes.

```python
from scipy import stats as sp_stats
from scipy.stats import false_discovery_control

def correlation_analysis(df: pd.DataFrame, dimensions: list[str]) -> dict:
    results = {}
    p_values = []
    
    for dim in dimensions:
        contingency = pd.crosstab(df[dim], df["tier"])
        chi2, p, dof, _ = sp_stats.chi2_contingency(contingency.values)
        n = contingency.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        
        results[dim] = {"chi2": chi2, "p_raw": p, "cramers_v": cramers_v}
        p_values.append(p)
    
    # FDR correction within dimension family
    p_adj = false_discovery_control(p_values, method="bh")
    for i, dim in enumerate(dimensions):
        results[dim]["p_adjusted"] = p_adj[i]
        results[dim]["significant"] = p_adj[i] < 0.05
    
    return results
```

**Checklist:**
- [ ] Build contingency table with `pd.crosstab`
- [ ] Run `chi2_contingency` on `.values` (numpy array)
- [ ] Compute Cramér's V effect size
- [ ] Collect raw p-values for FDR correction
- [ ] Apply BH correction within test family
- [ ] Report both raw and adjusted p-values

---

## Severity Amplification Analysis

**When:** Testing whether bias amplifies with complaint severity.

```python
def severity_amplification(df: pd.DataFrame) -> dict:
    # Encode severity as ordinal
    severity_map = {"low": 1, "medium": 2, "high": 3}
    df["sev_ord"] = df["monetary_severity"].map(severity_map)
    
    # Test severity-outcome correlation
    rho, p = sp_stats.spearmanr(df["sev_ord"], df["tier"])
    
    # ANOVA: tier differences across severity levels
    groups = [df.loc[df["monetary_severity"] == s, "tier"].values 
              for s in ["low", "medium", "high"]]
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*catastrophic.*")
        f_stat, p_anova = sp_stats.f_oneway(*groups)
    
    return {
        "spearman_rho": rho,
        "spearman_p": p,
        "anova_f": f_stat,
        "anova_p": p_anova,
    }
```

---

## Repeated Measures Non-Parametric Testing

**When:** Comparing persona variants within the same complaint (matched-guise design).

```python
def repeated_measures_analysis(pivot: pd.DataFrame) -> dict:
    """
    pivot: rows=complaints, columns=persona variants, values=tier
    """
    results = {}
    
    # Friedman test for overall differences
    groups = [pivot[col].values for col in pivot.columns]
    stat, p = sp_stats.friedmanchisquare(*groups)
    results["friedman"] = {"statistic": stat, "p": p}
    
    if p < 0.05:
        # Post-hoc pairwise Wilcoxon tests
        pairwise = []
        cols = list(pivot.columns)
        for i, j in combinations(range(len(cols)), 2):
            _, p_ij = sp_stats.wilcoxon(
                pivot[cols[i]], pivot[cols[j]], 
                zero_method="wilcox"
            )
            pairwise.append({
                "pair": (cols[i], cols[j]),
                "p_raw": p_ij
            })
        
        # FDR correction
        p_raw = [x["p_raw"] for x in pairwise]
        p_adj = false_discovery_control(p_raw, method="bh")
        for k, pw in enumerate(pairwise):
            pw["p_adjusted"] = p_adj[k]
        
        results["pairwise"] = pairwise
    
    return results
```

**Validation loop:**
1. Run Friedman test
2. If p < 0.05, proceed to pairwise comparisons
3. Apply FDR correction to pairwise p-values
4. Only report pairs where adjusted p < 0.05

---

## Bootstrap Confidence Intervals

**When:** Computing CIs for effect sizes or custom statistics.

```python
def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: callable = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> tuple[float, float, float]:
    """Returns (estimate, lower, upper)"""
    rng = np.random.default_rng(seed)
    n = len(data)
    
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats.append(statistic_fn(sample))
    
    estimate = statistic_fn(data)
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    
    return estimate, lower, upper

# Usage: Bootstrap Cohen's d
def cohens_d(a, b):
    pooled = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0

combined = np.concatenate([group_a, group_b])
d_est, d_lo, d_hi = bootstrap_ci(combined, lambda x: cohens_d(x[:len(group_a)], x[len(group_a):]))
```

---

## Multiple Comparison Pipeline

**Full workflow for FDR-controlled analysis:**

```python
def full_analysis_pipeline(df: pd.DataFrame) -> dict:
    """
    Complete analysis with proper multiple comparison handling.
    """
    dimensions = ["age", "geographic", "education", "socioeconomic"]
    
    # === Phase 1: Individual tests ===
    tests = {}
    for dim in dimensions:
        contingency = pd.crosstab(df[dim], df["tier"])
        chi2, p, dof, _ = sp_stats.chi2_contingency(contingency.values)
        tests[dim] = {"chi2": chi2, "p_raw": p, "dof": dof}
    
    # === Phase 2: FDR correction ===
    p_raw = [tests[d]["p_raw"] for d in dimensions]
    p_adj = false_discovery_control(p_raw, method="bh")
    
    for i, dim in enumerate(dimensions):
        tests[dim]["p_adjusted"] = p_adj[i]
        tests[dim]["significant"] = p_adj[i] < 0.05
    
    # === Phase 3: Summary ===
    n_tests = len(dimensions)
    n_significant = sum(1 for d in dimensions if tests[d]["significant"])
    
    return {
        "tests": tests,
        "summary": {
            "total_tests": n_tests,
            "significant_after_fdr": n_significant,
            "fdr_method": "benjamini-hochberg",
        }
    }
```

**Checklist:**
- [ ] Run all tests and collect raw p-values
- [ ] Apply `false_discovery_control(method="bh")` within test family
- [ ] Update each test result with adjusted p-value
- [ ] Determine significance from adjusted (not raw) p-values
- [ ] Report both raw and adjusted in output
- [ ] Document total tests and survivors

---

## Integration with statsmodels

For mixed models and GEE that use scipy internally, see the **statsmodels** skill. Key integration points:

```python
# Firth penalized logit uses scipy.stats.norm for p-values
from scipy import stats as sp_stats
z_score = coef / se
p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_score)))

# GEE clustering uses scipy for sandwich estimators
# McNemar uses scipy's binomial distribution
```

---

## Thread Safety Note

**WARNING:** `warnings.catch_warnings()` is NOT thread-safe.

```python
# GOOD - Set filters in main thread before ThreadPoolExecutor
_ctx = warnings.catch_warnings()
_ctx.__enter__()
warnings.filterwarnings("ignore", message=".*catastrophic.*")

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(analysis_fn, items))

_ctx.__exit__(None, None, None)
```

See `src/severity_analysis.py:1782-1803` for the production pattern.
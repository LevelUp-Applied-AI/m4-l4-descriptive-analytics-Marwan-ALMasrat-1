"""
extensions.py — Challenge Extensions for Module 4 Lab
Hashemite Technical University — Descriptive Analytics

Tier 1: Advanced Statistical Analysis (ANOVA + Violin Plots)
Tier 2: Automated EDA Report Generator (EDAReport class)
Tier 3: Statistical Simulation and Power Analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — Advanced Statistical Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_anova_with_posthoc(df: pd.DataFrame, output_dir: str = "output") -> dict:
    """
    Hypothesis: 'Does average GPA differ across the five departments?'

    Steps:
      1. One-way ANOVA across all departments.
      2. If significant (p < 0.05), run pairwise independent t-tests with
         Bonferroni correction to identify which department pairs differ.

    Returns a dict with keys:
        'f_stat', 'p_value', 'significant', 'posthoc' (list of dicts)
    """
    os.makedirs(output_dir, exist_ok=True)

    departments = df["department"].unique()
    groups = [df[df["department"] == dept]["gpa"].values for dept in departments]

    f_stat, p_value = stats.f_oneway(*groups)
    significant = p_value < 0.05

    print("\n=== Tier 1: ANOVA — GPA across Departments ===")
    print(f"F-statistic : {f_stat:.4f}")
    print(f"p-value     : {p_value:.6f}")
    if significant:
        print("✅ Result: Significant differences exist between departments (p < 0.05)")
    else:
        print("❌ Result: No significant difference between departments")

    posthoc_results = []

    if significant:
        pairs = list(combinations(departments, 2))
        # Bonferroni correction: divide alpha by number of comparisons
        n_comparisons = len(pairs)
        alpha_corrected = 0.05 / n_comparisons

        print(f"\n  Post-hoc pairwise t-tests (Bonferroni α = 0.05 / {n_comparisons} = {alpha_corrected:.4f}):\n")

        for dept_a, dept_b in pairs:
            group_a = df[df["department"] == dept_a]["gpa"].values
            group_b = df[df["department"] == dept_b]["gpa"].values
            t_stat, p_pair = stats.ttest_ind(group_a, group_b)
            sig = p_pair < alpha_corrected
            posthoc_results.append({
                "pair": (dept_a, dept_b),
                "t_stat": t_stat,
                "p_value": p_pair,
                "significant_after_correction": sig,
            })
            marker = "✅" if sig else "  "
            print(f"  {marker} {dept_a} vs {dept_b}: t={t_stat:.4f}, p={p_pair:.4f} {'*' if sig else ''}")

    return {
        "f_stat": f_stat,
        "p_value": p_value,
        "significant": significant,
        "posthoc": posthoc_results,
    }


def plot_violin_gpa_by_department(df: pd.DataFrame, output_dir: str = "output") -> None:
    """
    Creates a violin plot of GPA distribution by department.
    Violin plots show both the summary statistics (like a box plot)
    and the full distribution shape (density estimate).
    Saved to output/violin_gpa_by_department.png
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        x="department",
        y="gpa",
        data=df,
        palette="muted",
        inner="box",   # show box plot inside the violin
        cut=0,         # don't extend violin beyond data range
    )
    plt.title("GPA Distribution by Department (Violin Plot)", fontsize=14)
    plt.xlabel("Department")
    plt.ylabel("GPA (0.0 – 4.0)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "violin_gpa_by_department.png"), dpi=150)
    plt.close()
    print(f"\n  Violin plot saved → {output_dir}/violin_gpa_by_department.png")


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — Automated EDA Report Generator
# ─────────────────────────────────────────────────────────────────────────────

class EDAReport:
    """
    Reusable automated EDA report generator.

    Accepts any DataFrame and produces:
      - Data profile (shape, types, missing values)
      - Distribution plots for all (or selected) numeric columns
      - Correlation heatmap
      - Missing data visualisation
      - Outlier summary using the IQR method

    Parameters
    ----------
    df            : pandas DataFrame to analyse
    output_dir    : directory where all outputs are written (default 'output')
    columns       : list of column names to analyse; None = all columns
    plot_style    : any valid seaborn style string (default 'whitegrid')
    """

    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: str = "output",
        columns: list = None,
        plot_style: str = "whitegrid",
    ):
        self.df = df.copy()
        self.output_dir = output_dir
        self.plot_style = plot_style
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_style(self.plot_style)

        # Restrict to requested columns if provided
        if columns:
            available = [c for c in columns if c in self.df.columns]
            self.df = self.df[available]

        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------
    def generate_data_profile(self) -> str:
        """Write a text data profile and return it as a string."""
        lines = []
        lines.append("=== AUTOMATED EDA PROFILE ===\n")
        lines.append(f"Shape        : {self.df.shape[0]} rows × {self.df.shape[1]} columns\n")

        lines.append("\n--- Column Types ---")
        for col, dtype in self.df.dtypes.items():
            lines.append(f"  {col:<30} {dtype}")

        lines.append("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        for col in self.df.columns:
            lines.append(f"  {col:<30} {missing[col]:>5} ({missing_pct[col]:>5.1f}%)")

        lines.append("\n--- Descriptive Statistics (numeric) ---")
        lines.append(str(self.df.describe()))

        report_text = "\n".join(lines)

        profile_path = os.path.join(self.output_dir, "eda_report_profile.txt")
        with open(profile_path, "w") as fh:
            fh.write(report_text)

        print(f"  Data profile saved → {profile_path}")
        return report_text

    # ------------------------------------------------------------------
    def plot_distributions(self) -> None:
        """Histogram + KDE for every numeric column."""
        for col in self.numeric_cols:
            plt.figure(figsize=(7, 4))
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"eda_dist_{col}.png")
            plt.savefig(filename, dpi=120)
            plt.close()
        print(f"  Distribution plots saved → {self.output_dir}/eda_dist_*.png")

    # ------------------------------------------------------------------
    def plot_correlation_heatmap(self) -> pd.DataFrame:
        """Compute Pearson correlation matrix and save annotated heatmap."""
        if len(self.numeric_cols) < 2:
            print("  Skipping heatmap: fewer than 2 numeric columns.")
            return pd.DataFrame()

        corr = self.df[self.numeric_cols].corr()

        fig_width = max(6, len(self.numeric_cols))
        plt.figure(figsize=(fig_width, fig_width - 1))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        filename = os.path.join(self.output_dir, "eda_correlation_heatmap.png")
        plt.savefig(filename, dpi=120)
        plt.close()
        print(f"  Correlation heatmap saved → {filename}")
        return corr

    # ------------------------------------------------------------------
    def plot_missing_data(self) -> None:
        """Bar chart showing missing-value counts per column."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            print("  No missing values found — skipping missing-data plot.")
            return

        plt.figure(figsize=(max(6, len(missing)), 4))
        missing.sort_values(ascending=False).plot(kind="bar", color="salmon", edgecolor="black")
        plt.title("Missing Values per Column")
        plt.ylabel("Count")
        plt.xlabel("Column")
        plt.tight_layout()
        filename = os.path.join(self.output_dir, "eda_missing_values.png")
        plt.savefig(filename, dpi=120)
        plt.close()
        print(f"  Missing-data plot saved → {filename}")

    # ------------------------------------------------------------------
    def outlier_summary(self) -> pd.DataFrame:
        """
        Detect outliers using the IQR method for each numeric column.
        An outlier is any value outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR].
        Returns a DataFrame with outlier counts and bounds.
        """
        records = []
        for col in self.numeric_cols:
            series = self.df[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = ((series < lower) | (series > upper)).sum()
            records.append({
                "column": col,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_fence": round(lower, 4),
                "upper_fence": round(upper, 4),
                "n_outliers": n_outliers,
                "pct_outliers": round(n_outliers / len(series) * 100, 2),
            })

        summary_df = pd.DataFrame(records).set_index("column")

        filename = os.path.join(self.output_dir, "eda_outlier_summary.csv")
        summary_df.to_csv(filename)
        print(f"  Outlier summary saved → {filename}")
        print(summary_df.to_string())
        return summary_df

    # ------------------------------------------------------------------
    def run_all(self) -> None:
        """Run the complete EDA pipeline."""
        print("\n=== Tier 2: Automated EDA Report ===")
        self.generate_data_profile()
        self.plot_missing_data()
        self.plot_distributions()
        self.plot_correlation_heatmap()
        self.outlier_summary()
        print("  EDA report complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — Statistical Simulation and Power Analysis
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_confidence_intervals(
    df: pd.DataFrame,
    n_iterations: int = 10_000,
    ci: float = 0.95,
) -> dict:
    """
    Compute bootstrap confidence intervals for mean GPA by internship status.

    Resamples the data n_iterations times (with replacement) and computes the
    mean GPA for each group in each resample, then derives the CI from the
    percentile distribution of those bootstrap means.

    Returns
    -------
    dict with keys 'Yes' and 'No', each containing:
        'mean', 'ci_lower', 'ci_upper', 'bootstrap_means'
    Also prints a comparison with the parametric t-test CI.
    """
    print("\n=== Tier 3a: Bootstrap Confidence Intervals ===")

    alpha = 1 - ci
    results = {}

    for status in ["Yes", "No"]:
        group = df[df["has_internship"] == status]["gpa"].values
        n = len(group)

        boot_means = np.array([
            np.mean(np.random.choice(group, size=n, replace=True))
            for _ in range(n_iterations)
        ])

        ci_lower = np.percentile(boot_means, 100 * alpha / 2)
        ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

        results[status] = {
            "mean": np.mean(group),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "bootstrap_means": boot_means,
        }

        print(f"  Internship={status}: mean GPA = {np.mean(group):.4f}  "
              f"Bootstrap {int(ci*100)}% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

    # --- Parametric comparison via scipy (uses SEM + t-distribution) ---
    intern_gpa   = df[df["has_internship"] == "Yes"]["gpa"].values
    no_intern_gpa = df[df["has_internship"] == "No"]["gpa"].values

    for label, arr in [("Yes", intern_gpa), ("No", no_intern_gpa)]:
        n = len(arr)
        se = stats.sem(arr)
        t_crit = stats.t.ppf(1 - (1 - ci) / 2, df=n - 1)
        param_lower = np.mean(arr) - t_crit * se
        param_upper = np.mean(arr) + t_crit * se
        print(f"  Internship={label}: Parametric {int(ci*100)}% CI = "
              f"[{param_lower:.4f}, {param_upper:.4f}]")

    # --- Visualise bootstrap distributions ---
    os.makedirs("output", exist_ok=True)
    plt.figure(figsize=(9, 4))
    for status, color in [("Yes", "steelblue"), ("No", "salmon")]:
        plt.hist(
            results[status]["bootstrap_means"],
            bins=60,
            alpha=0.6,
            color=color,
            label=f"Internship={status}",
        )
        plt.axvline(results[status]["ci_lower"], color=color, linestyle="--", linewidth=1)
        plt.axvline(results[status]["ci_upper"], color=color, linestyle="--", linewidth=1)

    plt.title(f"Bootstrap Distribution of Mean GPA ({n_iterations:,} iterations)")
    plt.xlabel("Bootstrap Mean GPA")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/bootstrap_ci_gpa.png", dpi=150)
    plt.close()
    print("  Bootstrap CI plot saved → output/bootstrap_ci_gpa.png")

    return results


def run_power_analysis(df: pd.DataFrame) -> dict:
    """
    Conduct a power analysis for the internship/GPA t-test.

    Given the observed effect size (Cohen's d), compute the sample size
    needed to achieve 80% power at α = 0.05 using TTestIndPower.

    Returns a dict with 'cohens_d', 'required_n', 'observed_power'.
    """
    try:
        from statsmodels.stats.power import TTestIndPower
    except ImportError:
        print("  statsmodels not installed — skipping power analysis.")
        return {}

    print("\n=== Tier 3b: Power Analysis ===")

    intern_gpa    = df[df["has_internship"] == "Yes"]["gpa"].values
    no_intern_gpa = df[df["has_internship"] == "No"]["gpa"].values

    # Cohen's d
    pooled_std = np.sqrt(
        (np.std(intern_gpa, ddof=1) ** 2 + np.std(no_intern_gpa, ddof=1) ** 2) / 2
    )
    cohens_d = (np.mean(intern_gpa) - np.mean(no_intern_gpa)) / pooled_std

    analysis = TTestIndPower()

    # Sample size needed for 80% power
    required_n = analysis.solve_power(
        effect_size=abs(cohens_d),
        alpha=0.05,
        power=0.80,
        alternative="two-sided",
    )

    # Observed power with current sample sizes
    n_min = min(len(intern_gpa), len(no_intern_gpa))
    observed_power = analysis.solve_power(
        effect_size=abs(cohens_d),
        alpha=0.05,
        nobs1=n_min,
        alternative="two-sided",
    )

    print(f"  Cohen's d           : {cohens_d:.4f}")
    print(f"  Required n (per group) for 80% power: {int(np.ceil(required_n))}")
    print(f"  Observed power with n={n_min} per group: {observed_power:.4f}")

    # --- Power curve plot ---
    sample_sizes = np.arange(10, 500, 5)
    powers = [
        analysis.solve_power(
            effect_size=abs(cohens_d),
            alpha=0.05,
            nobs1=n,
            alternative="two-sided",
        )
        for n in sample_sizes
    ]

    os.makedirs("output", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(sample_sizes, powers, color="steelblue", linewidth=2)
    plt.axhline(0.80, color="red", linestyle="--", label="80% power threshold")
    plt.axvline(required_n, color="orange", linestyle="--",
                label=f"Required n ≈ {int(np.ceil(required_n))}")
    plt.title("Power Curve — Internship vs No-Internship GPA (Two-tailed t-test)")
    plt.xlabel("Sample Size (per group)")
    plt.ylabel("Statistical Power")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("output/power_curve.png", dpi=150)
    plt.close()
    print("  Power curve plot saved → output/power_curve.png")

    return {
        "cohens_d": cohens_d,
        "required_n": int(np.ceil(required_n)),
        "observed_power": observed_power,
    }


def run_false_positive_simulation(
    n_simulations: int = 1_000,
    sample_size: int = 100,
    alpha: float = 0.05,
) -> dict:
    """
    Simulate the false-positive rate of an independent t-test.

    Generates synthetic datasets where both groups are drawn from the same
    population (null hypothesis is TRUE) and measures how often the test
    incorrectly rejects H₀ (Type I error).

    Expected false-positive rate ≈ alpha (by construction).

    Returns a dict with 'false_positive_rate' and 'n_simulations'.
    """
    print("\n=== Tier 3c: False Positive Rate Simulation ===")

    rng = np.random.default_rng(seed=42)
    false_positives = 0

    for _ in range(n_simulations):
        # Both groups from identical population → H₀ is true
        group_a = rng.normal(loc=3.0, scale=0.5, size=sample_size)
        group_b = rng.normal(loc=3.0, scale=0.5, size=sample_size)
        _, p = stats.ttest_ind(group_a, group_b)
        if p < alpha:
            false_positives += 1

    fp_rate = false_positives / n_simulations

    print(f"  Simulations          : {n_simulations:,}")
    print(f"  α (nominal)          : {alpha}")
    print(f"  False positives found: {false_positives}")
    print(f"  Empirical FP rate    : {fp_rate:.4f}")

    if abs(fp_rate - alpha) < 0.01:
        print(f"  ✅ FP rate ≈ α — test is well-calibrated.")
    else:
        print(f"  ⚠️  FP rate deviates from α (may need more simulations).")

    # --- Plot ---
    os.makedirs("output", exist_ok=True)
    # Visualise p-value distribution under the null
    p_values = []
    for _ in range(n_simulations):
        ga = rng.normal(loc=3.0, scale=0.5, size=sample_size)
        gb = rng.normal(loc=3.0, scale=0.5, size=sample_size)
        _, p = stats.ttest_ind(ga, gb)
        p_values.append(p)

    plt.figure(figsize=(7, 4))
    plt.hist(p_values, bins=40, edgecolor="black", color="steelblue", alpha=0.8)
    plt.axvline(alpha, color="red", linestyle="--", label=f"α = {alpha}")
    plt.title("p-value Distribution Under H₀ (both groups identical)")
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/false_positive_simulation.png", dpi=150)
    plt.close()
    print("  Simulation plot saved → output/false_positive_simulation.png")

    return {"false_positive_rate": fp_rate, "n_simulations": n_simulations}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load the cleaned dataset the same way eda_analysis.py does
    df = pd.read_csv("data/student_performance.csv")
    df["commute_minutes"].fillna(df["commute_minutes"].median(), inplace=True)
    df = df.dropna(subset=["study_hours_weekly"])

    # ── Tier 1 ──────────────────────────────────────────────────────────────
    run_anova_with_posthoc(df, output_dir="output")
    plot_violin_gpa_by_department(df, output_dir="output")

    # ── Tier 2 ──────────────────────────────────────────────────────────────
    report = EDAReport(df, output_dir="output", plot_style="whitegrid")
    report.run_all()

    # ── Tier 3 ──────────────────────────────────────────────────────────────
    bootstrap_confidence_intervals(df, n_iterations=10_000)
    run_power_analysis(df)
    run_false_positive_simulation(n_simulations=1_000, sample_size=100)


if __name__ == "__main__":
    main()
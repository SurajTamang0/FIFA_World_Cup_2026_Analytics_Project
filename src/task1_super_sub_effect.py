"""
Task 1 - Super-Sub Effect
FIFA World Cup 2026 Analytics Project

Research question:
Do substitute-dominant outfield players produce a different rate of
attacking contribution per 90 minutes than starter-dominant players?

This task includes:
- data wrangling
- eligibility filtering
- reproducible stratified random sampling
- descriptive statistics
- 95% confidence intervals
- Welch's independent two-sample t-test
- assumption diagnostics
- visualisation
- interpretation and limitations
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


STANDARD_FILE = (
    RAW_DATA_DIR / "fbref_2026_world_cup_player_standard.csv"
)

PLAYING_TIME_FILE = (
    RAW_DATA_DIR / "fbref_2026_world_cup_player_playing_time.csv"
)

OUTPUT_SAMPLE_FILE = (
    PROCESSED_DATA_DIR / "task1_super_sub_sample.csv"
)

OUTPUT_FIGURE_FILE = (
    FIGURES_DIR / "task1_super_sub_effect.png"
)


# ---------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------

MIN_MINUTES = 90

RANDOM_SEED = 2026

SAMPLE_SIZE_PER_GROUP = 64

ALPHA = 0.05


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def flatten_fbref_columns(dataframe):
    """
    Convert FBref's two-row column headers into simple column names.

    For columns where the first header is 'Unnamed', the second header
    is retained. For grouped statistics, the lower-level label is used.

    Duplicate lower-level names are automatically made unique by adding
    .1, .2, etc.
    """

    if not isinstance(dataframe.columns, pd.MultiIndex):
        return dataframe.copy()

    raw_names = []

    for top, bottom in dataframe.columns:
        top = str(top).strip()
        bottom = str(bottom).strip()

        if bottom and not bottom.startswith("Unnamed"):
            raw_names.append(bottom)
        elif top and not top.startswith("Unnamed"):
            raw_names.append(top)
        else:
            raw_names.append(bottom)

    counts = {}
    final_names = []

    for name in raw_names:
        if name not in counts:
            counts[name] = 0
            final_names.append(name)
        else:
            counts[name] += 1
            final_names.append(f"{name}.{counts[name]}")

    result = dataframe.copy()
    result.columns = final_names

    return result


def load_fbref_csv(path):
    """
    Load an FBref CSV exported with its original two-level header.
    """

    dataframe = pd.read_csv(path, header=[0, 1])

    dataframe = flatten_fbref_columns(dataframe)

    # Remove repeated header rows if FBref inserted any inside the file.
    if "Rk" in dataframe.columns:
        dataframe = dataframe[
            dataframe["Rk"].astype(str).str.strip() != "Rk"
        ].copy()

    dataframe.reset_index(drop=True, inplace=True)

    return dataframe


def numeric(dataframe, columns):
    """
    Convert selected columns to numeric format.
    Invalid values become NaN.
    """

    result = dataframe.copy()

    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            )

    return result


def describe_series(dataframe, group_column, value_column):
    """
    Produce descriptive statistics for the requested variable.
    """

    return (
        dataframe
        .groupby(group_column)[value_column]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .round(3)
    )


def mean_confidence_interval(series, confidence=0.95):
    """
    Calculate a t-based confidence interval for a population mean.
    """

    clean = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    n = len(clean)

    if n < 2:
        return np.nan, np.nan, np.nan

    mean = clean.mean()
    standard_error = stats.sem(clean)

    critical_value = stats.t.ppf(
        (1 + confidence) / 2,
        df=n - 1
    )

    margin = critical_value * standard_error

    return (
        float(mean),
        float(mean - margin),
        float(mean + margin),
    )


def welch_difference_confidence_interval(
    group1,
    group2,
    confidence=0.95,
):
    """
    Calculate a Welch-style confidence interval for the difference
    between two independent population means.

    Difference is defined as group1 - group2.
    """

    x = pd.to_numeric(group1, errors="coerce").dropna()
    y = pd.to_numeric(group2, errors="coerce").dropna()

    n1 = len(x)
    n2 = len(y)

    mean1 = x.mean()
    mean2 = y.mean()

    var1 = x.var(ddof=1)
    var2 = y.var(ddof=1)

    difference = mean1 - mean2

    standard_error = np.sqrt(
        (var1 / n1) +
        (var2 / n2)
    )

    numerator = (
        (var1 / n1) +
        (var2 / n2)
    ) ** 2

    denominator = (
        ((var1 / n1) ** 2) / (n1 - 1)
        +
        ((var2 / n2) ** 2) / (n2 - 1)
    )

    degrees_of_freedom = numerator / denominator

    critical_value = stats.t.ppf(
        (1 + confidence) / 2,
        df=degrees_of_freedom
    )

    margin = critical_value * standard_error

    lower = difference - margin
    upper = difference + margin

    return (
        float(difference),
        float(lower),
        float(upper),
        float(degrees_of_freedom),
    )


def iqr_outlier_summary(series):
    """
    Count potential outliers using the 1.5 x IQR rule.
    """

    clean = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = clean[
        (clean < lower_bound)
        | (clean > upper_bound)
    ]

    percentage = (
        len(outliers) / len(clean) * 100
        if len(clean) > 0
        else 0
    )

    return {
        "count": len(outliers),
        "percentage": percentage,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def prepare_task1_data():
    """
    Load and prepare the Standard and Playing Time datasets.
    """

    standard = load_fbref_csv(STANDARD_FILE)

    playing_time = load_fbref_csv(PLAYING_TIME_FILE)

    print("Raw dataset dimensions:")
    print(f"Standard:      {standard.shape}")
    print(f"Playing time:  {playing_time.shape}")

    standard = numeric(
        standard,
        [
            "Min",
            "90s",
            "Gls",
            "Ast",
        ]
    )

    playing_time = numeric(
        playing_time,
        [
            "Starts",
            "Subs",
        ]
    )

    # Keep only fields required from playing-time data.
    playing_subset = playing_time[
        [
            "Player",
            "Squad",
            "Starts",
            "Subs",
        ]
    ].copy()

    # Merge using Player + Squad to reduce risk of matching
    # players incorrectly.
    merged = standard.merge(
        playing_subset,
        on=["Player", "Squad"],
        how="left",
        suffixes=("", "_playing"),
    )

    print(f"\nPlayers after merge: {len(merged)}")

    # Remove goalkeepers because the research question concerns
    # attacking contribution by outfield players.
    merged["Pos"] = merged["Pos"].astype(str)

    outfield = merged[
        ~merged["Pos"].str.contains(
            "GK",
            case=False,
            na=False
        )
    ].copy()

    print(
        "Outfield players after goalkeeper removal: "
        f"{len(outfield)}"
    )

    # Tournament-exposure threshold.
    eligible = outfield[
        outfield["Min"].notna()
        & (outfield["Min"] >= MIN_MINUTES)
        & outfield["90s"].notna()
        & (outfield["90s"] > 0)
    ].copy()

    print(
        f"Eligible players with at least "
        f"{MIN_MINUTES} minutes: {len(eligible)}"
    )

    # Compute attacking contribution independently instead of relying
    # on FBref's duplicated per-90 column names.
    eligible["GA90"] = (
        eligible["Gls"].fillna(0)
        +
        eligible["Ast"].fillna(0)
    ) / eligible["90s"]

    # Classify usage pattern.
    eligible["UsageGroup"] = np.select(
        [
            eligible["Starts"] > eligible["Subs"],
            eligible["Subs"] > eligible["Starts"],
        ],
        [
            "Starter-dominant",
            "Substitute-dominant",
        ],
        default="Balanced",
    )

    return eligible


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------

def create_stratified_sample(eligible):
    """
    Draw equal-sized reproducible random samples from the two principal
    comparison groups.
    """

    starter_population = eligible[
        eligible["UsageGroup"] == "Starter-dominant"
    ].copy()

    substitute_population = eligible[
        eligible["UsageGroup"] == "Substitute-dominant"
    ].copy()

    if len(starter_population) < SAMPLE_SIZE_PER_GROUP:
        raise ValueError(
            "Starter-dominant population is smaller than the planned "
            "sample size."
        )

    if len(substitute_population) < SAMPLE_SIZE_PER_GROUP:
        raise ValueError(
            "Substitute-dominant population is smaller than the "
            "planned sample size."
        )

    starter_sample = starter_population.sample(
        n=SAMPLE_SIZE_PER_GROUP,
        random_state=RANDOM_SEED,
        replace=False,
    )

    substitute_sample = substitute_population.sample(
        n=SAMPLE_SIZE_PER_GROUP,
        random_state=RANDOM_SEED,
        replace=False,
    )

    sample = pd.concat(
        [
            starter_sample,
            substitute_sample,
        ],
        ignore_index=True,
    )

    sample.to_csv(
        OUTPUT_SAMPLE_FILE,
        index=False
    )

    return sample


# ---------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------

def create_boxplot(sample):
    """
    Produce a boxplot comparing attacking contribution between groups.
    """

    starter = sample.loc[
        sample["UsageGroup"] == "Starter-dominant",
        "GA90",
    ].dropna()

    substitute = sample.loc[
        sample["UsageGroup"] == "Substitute-dominant",
        "GA90",
    ].dropna()

    figure_data = [
        starter,
        substitute,
    ]

    labels = [
        "Starter-dominant",
        "Substitute-dominant",
    ]

    plt.figure(figsize=(9, 6))

    plt.boxplot(
        figure_data,
        tick_labels=labels,
        showmeans=True,
    )

    plt.ylabel(
        "Goals + Assists per 90 Minutes"
    )

    plt.xlabel(
        "Player Usage Group"
    )

    plt.title(
        "FIFA World Cup 2026: Attacking Contribution\n"
        "Starter-Dominant vs Substitute-Dominant Players"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("TASK 1 - SUPER-SUB EFFECT")
    print("=" * 70)

    print(
        "\nResearch question:\n"
        "Do substitute-dominant outfield players produce a different "
        "rate of attacking contribution per 90 minutes than "
        "starter-dominant players?\n"
    )

    eligible = prepare_task1_data()

    print("\nUsage-group counts:")

    print(
        eligible["UsageGroup"]
        .value_counts()
    )

    print(
        "\nMissing GA90 values:",
        eligible["GA90"].isna().sum()
    )

    # -----------------------------------------------------------------
    # Eligible-population descriptives
    # -----------------------------------------------------------------

    analysis_population = eligible[
        eligible["UsageGroup"].isin(
            [
                "Starter-dominant",
                "Substitute-dominant",
            ]
        )
    ].copy()

    print(
        "\nEligible-population descriptive statistics:\n"
    )

    population_stats = describe_series(
        analysis_population,
        "UsageGroup",
        "GA90"
    )

    print(population_stats)

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------

    sample = create_stratified_sample(
        analysis_population
    )

    print("\nSampling design:")

    print(
        "Method: stratified simple random sampling"
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        f"Sample size per group: "
        f"{SAMPLE_SIZE_PER_GROUP}"
    )

    print(
        f"Total sample size: {len(sample)}"
    )

    print(
        "Analytical sample saved to:"
    )

    print(OUTPUT_SAMPLE_FILE)

    # -----------------------------------------------------------------
    # Sample descriptives
    # -----------------------------------------------------------------

    print(
        "\nSample descriptive statistics:\n"
    )

    sample_stats = describe_series(
        sample,
        "UsageGroup",
        "GA90"
    )

    print(sample_stats)

    starter = sample.loc[
        sample["UsageGroup"] == "Starter-dominant",
        "GA90",
    ].dropna()

    substitute = sample.loc[
        sample["UsageGroup"] == "Substitute-dominant",
        "GA90",
    ].dropna()

    # -----------------------------------------------------------------
    # Confidence intervals
    # -----------------------------------------------------------------

    starter_mean, starter_low, starter_high = (
        mean_confidence_interval(starter)
    )

    substitute_mean, substitute_low, substitute_high = (
        mean_confidence_interval(substitute)
    )

    print(
        "\n95% confidence intervals for mean G+A per 90:"
    )

    print("\nStarter-dominant:")

    print(
        f"Mean = {starter_mean:.3f}"
    )

    print(
        f"95% CI = "
        f"[{starter_low:.3f}, {starter_high:.3f}]"
    )

    print("\nSubstitute-dominant:")

    print(
        f"Mean = {substitute_mean:.3f}"
    )

    print(
        f"95% CI = "
        f"[{substitute_low:.3f}, "
        f"{substitute_high:.3f}]"
    )

    # Difference defined as substitute - starter.
    difference, diff_low, diff_high, welch_df_ci = (
        welch_difference_confidence_interval(
            substitute,
            starter,
        )
    )

    print(
        "\nDifference in mean G+A per 90 "
        "(Substitute - Starter):"
    )

    print(
        f"Difference = {difference:.3f}"
    )

    print(
        f"95% CI = "
        f"[{diff_low:.3f}, {diff_high:.3f}]"
    )

    # -----------------------------------------------------------------
    # Welch independent two-sample t-test
    # -----------------------------------------------------------------

    test_result = stats.ttest_ind(
        substitute,
        starter,
        equal_var=False,
        nan_policy="omit",
    )

    t_statistic = float(test_result.statistic)
    p_value = float(test_result.pvalue)

    # Welch-Satterthwaite degrees of freedom.
    n1 = len(substitute)
    n2 = len(starter)

    variance1 = substitute.var(ddof=1)
    variance2 = starter.var(ddof=1)

    welch_df = (
        (
            variance1 / n1
            +
            variance2 / n2
        ) ** 2
        /
        (
            ((variance1 / n1) ** 2)
            / (n1 - 1)
            +
            ((variance2 / n2) ** 2)
            / (n2 - 1)
        )
    )

    print("\nWelch two-sample t-test:")

    print(
        "H0: mean GA90_substitute = "
        "mean GA90_starter"
    )

    print(
        "HA: mean GA90_substitute != "
        "mean GA90_starter"
    )

    print(
        f"Alpha = {ALPHA:.2f}"
    )

    print(
        f"t statistic = {t_statistic:.3f}"
    )

    print(
        f"Welch degrees of freedom = "
        f"{welch_df:.2f}"
    )

    print(
        f"p-value = {p_value:.4f}"
    )

    if p_value < ALPHA:

        print(
            "Decision: Reject H0."
        )

        print(
            "The sample provides sufficient evidence "
            "of a statistically significant difference "
            "in mean attacking contribution per 90 "
            "between the two usage groups."
        )

    else:

        print(
            "Decision: Fail to reject H0."
        )

        print(
            "The sample does not provide sufficient evidence "
            "of a statistically significant difference in "
            "mean attacking contribution per 90 between the "
            "two usage groups."
        )

    # -----------------------------------------------------------------
    # Assumption diagnostics
    # -----------------------------------------------------------------

    starter_outliers = iqr_outlier_summary(
        starter
    )

    substitute_outliers = iqr_outlier_summary(
        substitute
    )

    starter_skew = stats.skew(
        starter,
        bias=False
    )

    substitute_skew = stats.skew(
        substitute,
        bias=False
    )

    print("\nAssumption diagnostics:")

    print(
        "1. Observations represent individual players."
    )

    print(
        "2. The response variable is quantitative."
    )

    print(
        "3. Welch's test does not require equal "
        "group variances."
    )

    print("\nPotential IQR outliers:")

    print(
        "Starter-dominant: "
        f"{starter_outliers['count']} "
        f"({starter_outliers['percentage']:.1f}%)"
    )

    print(
        "Substitute-dominant: "
        f"{substitute_outliers['count']} "
        f"({substitute_outliers['percentage']:.1f}%)"
    )

    print("\nSample skewness:")

    print(
        f"Starter-dominant: "
        f"{starter_skew:.3f}"
    )

    print(
        f"Substitute-dominant: "
        f"{substitute_skew:.3f}"
    )

    print("\nInterpretation note:")

    print(
        "The t-test is reasonably robust to moderate "
        "non-normality with these sample sizes, but "
        "skewness and potential outliers should still "
        "be acknowledged when interpreting results."
    )

    # -----------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------

    create_boxplot(sample)

    print("\nFigure saved to:")

    print(OUTPUT_FIGURE_FILE)

    # -----------------------------------------------------------------
    # Limitations
    # -----------------------------------------------------------------

    print("\nKey limitations:")

    print(
        "- Player observations are treated as independent, "
        "although players are nested within national teams."
    )

    print(
        "- Per-90 attacking production can be affected by "
        "opponent strength, tactical role and match state."
    )

    print(
        "- The starter/substitute classification simplifies "
        "more complex player-usage patterns."
    )

    print(
        "- The analysis identifies association rather than "
        "causal effects of substitute usage."
    )

    print("\n" + "=" * 70)

    print(
        "TASK 1 ANALYSIS COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
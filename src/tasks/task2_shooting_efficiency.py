"""
Task 2 - Shooting Volume vs Efficiency
FIFA World Cup 2026 Analytics Project

Research question:
Among attacking players with meaningful tournament exposure,
do high-volume shooters have a different scoring efficiency
than lower-volume shooters?

This task includes:
- data wrangling
- eligibility filtering
- reproducible stratified random sampling
- descriptive statistics
- 95% confidence intervals
- Welch independent two-sample t-test
- assumption diagnostics
- visualisation
- interpretation
- limitations
"""

from pathlib import Path

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

SHOOTING_FILE = (
    RAW_DATA_DIR / "fbref_2026_world_cup_player_shooting.csv"
)

OUTPUT_SAMPLE_FILE = (
    PROCESSED_DATA_DIR / "task2_shooting_efficiency_sample.csv"
)

OUTPUT_FIGURE_FILE = (
    FIGURES_DIR / "task2_shooting_efficiency.png"
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

    If duplicate lower-level names appear, suffixes such as .1, .2
    are added automatically.
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
    Load an FBref CSV with a two-row header and clean its column names.
    """

    dataframe = pd.read_csv(path, header=[0, 1])

    dataframe = flatten_fbref_columns(dataframe)

    if "Rk" in dataframe.columns:
        dataframe = dataframe[
            dataframe["Rk"].astype(str).str.strip() != "Rk"
        ].copy()

    dataframe.reset_index(drop=True, inplace=True)

    return dataframe


def convert_numeric(dataframe, columns):
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


def describe_by_group(dataframe, group_column, value_column):
    """
    Produce grouped descriptive statistics.
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
    Calculate a Welch confidence interval for the difference
    between two independent population means.

    Difference is group1 - group2.
    """

    x = pd.to_numeric(group1, errors="coerce").dropna()
    y = pd.to_numeric(group2, errors="coerce").dropna()

    n1 = len(x)
    n2 = len(y)

    mean1 = x.mean()
    mean2 = y.mean()

    variance1 = x.var(ddof=1)
    variance2 = y.var(ddof=1)

    difference = mean1 - mean2

    standard_error = np.sqrt(
        variance1 / n1
        +
        variance2 / n2
    )

    numerator = (
        variance1 / n1
        +
        variance2 / n2
    ) ** 2

    denominator = (
        ((variance1 / n1) ** 2) / (n1 - 1)
        +
        ((variance2 / n2) ** 2) / (n2 - 1)
    )

    degrees_of_freedom = numerator / denominator

    critical_value = stats.t.ppf(
        (1 + confidence) / 2,
        df=degrees_of_freedom
    )

    margin = critical_value * standard_error

    return (
        float(difference),
        float(difference - margin),
        float(difference + margin),
        float(degrees_of_freedom),
    )


def iqr_outlier_summary(series):
    """
    Summarise potential outliers using the 1.5 x IQR rule.
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
        else 0.0
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

def prepare_task2_data():
    """
    Load, merge and prepare Standard and Shooting datasets.
    """

    standard = load_fbref_csv(STANDARD_FILE)
    shooting = load_fbref_csv(SHOOTING_FILE)

    print("Raw dataset dimensions:")
    print(f"Standard:  {standard.shape}")
    print(f"Shooting:  {shooting.shape}")

    standard = convert_numeric(
        standard,
        [
            "Min",
            "90s",
        ]
    )

    shooting = convert_numeric(
        shooting,
        [
            "90s",
            "Gls",
            "Sh",
            "SoT",
            "SoT%",
            "Sh/90",
            "SoT/90",
            "G/Sh",
            "G/SoT",
        ]
    )

    shooting_subset = shooting[
        [
            "Player",
            "Squad",
            "Pos",
            "90s",
            "Gls",
            "Sh",
            "SoT",
            "Sh/90",
            "G/Sh",
            "G/SoT",
        ]
    ].copy()

    standard_subset = standard[
        [
            "Player",
            "Squad",
            "Min",
        ]
    ].copy()

    merged = shooting_subset.merge(
        standard_subset,
        on=["Player", "Squad"],
        how="left",
    )

    print(
        f"\nPlayers after merge: {len(merged)}"
    )

    merged["Pos"] = merged["Pos"].astype(str)

    # Focus on attacking players.
    # Players may have combined FBref position labels such as FW,MF.
    attacking = merged[
        merged["Pos"].str.contains(
            "FW|MF",
            case=False,
            regex=True,
            na=False,
        )
        &
        ~merged["Pos"].str.contains(
            "GK",
            case=False,
            na=False,
        )
    ].copy()

    print(
        "Attacking players after position filtering: "
        f"{len(attacking)}"
    )

    eligible = attacking[
        attacking["Min"].notna()
        & (attacking["Min"] >= MIN_MINUTES)
        & attacking["Sh"].notna()
        & (attacking["Sh"] > 0)
        & attacking["Sh/90"].notna()
        & attacking["G/Sh"].notna()
    ].copy()

    print(
        f"Eligible attacking players with at least "
        f"{MIN_MINUTES} minutes and at least one shot: "
        f"{len(eligible)}"
    )

    return eligible


# ---------------------------------------------------------------------
# Group construction
# ---------------------------------------------------------------------

def construct_volume_groups(eligible):
    """
    Divide eligible attacking players into higher- and lower-volume
    shooting groups using the median shots-per-90 rate.
    """

    median_shots_per_90 = eligible["Sh/90"].median()

    eligible = eligible.copy()

    eligible["ShootingVolumeGroup"] = np.where(
        eligible["Sh/90"] >= median_shots_per_90,
        "Higher-volume",
        "Lower-volume",
    )

    return eligible, float(median_shots_per_90)


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------

def create_stratified_sample(eligible):
    """
    Draw equal-sized reproducible random samples from both shooting
    volume groups.
    """

    high_population = eligible[
        eligible["ShootingVolumeGroup"] == "Higher-volume"
    ].copy()

    low_population = eligible[
        eligible["ShootingVolumeGroup"] == "Lower-volume"
    ].copy()

    if len(high_population) < SAMPLE_SIZE_PER_GROUP:
        raise ValueError(
            "Higher-volume population is smaller than the planned "
            "sample size."
        )

    if len(low_population) < SAMPLE_SIZE_PER_GROUP:
        raise ValueError(
            "Lower-volume population is smaller than the planned "
            "sample size."
        )

    high_sample = high_population.sample(
        n=SAMPLE_SIZE_PER_GROUP,
        random_state=RANDOM_SEED,
        replace=False,
    )

    low_sample = low_population.sample(
        n=SAMPLE_SIZE_PER_GROUP,
        random_state=RANDOM_SEED,
        replace=False,
    )

    sample = pd.concat(
        [
            high_sample,
            low_sample,
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
    Create a boxplot comparing goal-per-shot efficiency.
    """

    high_volume = sample.loc[
        sample["ShootingVolumeGroup"] == "Higher-volume",
        "G/Sh",
    ].dropna()

    low_volume = sample.loc[
        sample["ShootingVolumeGroup"] == "Lower-volume",
        "G/Sh",
    ].dropna()

    figure_data = [
        high_volume,
        low_volume,
    ]

    labels = [
        "Higher-volume",
        "Lower-volume",
    ]

    plt.figure(figsize=(9, 6))

    plt.boxplot(
        figure_data,
        tick_labels=labels,
        showmeans=True,
    )

    plt.xlabel(
        "Shooting Volume Group"
    )

    plt.ylabel(
        "Goals per Shot"
    )

    plt.title(
        "FIFA World Cup 2026: Scoring Efficiency\n"
        "Higher-Volume vs Lower-Volume Shooters"
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
    print("TASK 2 - SHOOTING VOLUME VS EFFICIENCY")
    print("=" * 70)

    print(
        "\nResearch question:\n"
        "Among attacking players with meaningful tournament exposure, "
        "do high-volume shooters have a different scoring efficiency "
        "than lower-volume shooters?\n"
    )

    eligible = prepare_task2_data()

    eligible, median_shots_per_90 = (
        construct_volume_groups(eligible)
    )

    print(
        "\nMedian Sh/90 used to define preliminary shooting-volume "
        f"groups: {median_shots_per_90:.3f}"
    )

    print("\nShooting-volume group counts:")

    print(
        eligible["ShootingVolumeGroup"]
        .value_counts()
    )

    print(
        "\nMissing G/Sh values:",
        eligible["G/Sh"].isna().sum()
    )

    # -----------------------------------------------------------------
    # Eligible-population descriptives
    # -----------------------------------------------------------------

    print(
        "\nEligible-population descriptive statistics "
        "for goal-per-shot efficiency:\n"
    )

    population_stats = describe_by_group(
        eligible,
        "ShootingVolumeGroup",
        "G/Sh",
    )

    print(population_stats)

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------

    sample = create_stratified_sample(
        eligible
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

    sample_stats = describe_by_group(
        sample,
        "ShootingVolumeGroup",
        "G/Sh",
    )

    print(sample_stats)

    high_volume = sample.loc[
        sample["ShootingVolumeGroup"] == "Higher-volume",
        "G/Sh",
    ].dropna()

    low_volume = sample.loc[
        sample["ShootingVolumeGroup"] == "Lower-volume",
        "G/Sh",
    ].dropna()

    # -----------------------------------------------------------------
    # Confidence intervals
    # -----------------------------------------------------------------

    high_mean, high_low, high_high = (
        mean_confidence_interval(high_volume)
    )

    low_mean, low_low, low_high = (
        mean_confidence_interval(low_volume)
    )

    print(
        "\n95% confidence intervals for mean goal-per-shot efficiency:"
    )

    print("\nHigher-volume shooters:")

    print(
        f"Mean = {high_mean:.3f}"
    )

    print(
        f"95% CI = [{high_low:.3f}, {high_high:.3f}]"
    )

    print("\nLower-volume shooters:")

    print(
        f"Mean = {low_mean:.3f}"
    )

    print(
        f"95% CI = [{low_low:.3f}, {low_high:.3f}]"
    )

    difference, diff_low, diff_high, _ = (
        welch_difference_confidence_interval(
            high_volume,
            low_volume,
        )
    )

    print(
        "\nDifference in mean G/Sh "
        "(Higher-volume - Lower-volume):"
    )

    print(
        f"Difference = {difference:.3f}"
    )

    print(
        f"95% CI = [{diff_low:.3f}, {diff_high:.3f}]"
    )

    # -----------------------------------------------------------------
    # Welch independent two-sample t-test
    # -----------------------------------------------------------------

    test_result = stats.ttest_ind(
        high_volume,
        low_volume,
        equal_var=False,
        nan_policy="omit",
    )

    t_statistic = float(test_result.statistic)
    p_value = float(test_result.pvalue)

    n1 = len(high_volume)
    n2 = len(low_volume)

    variance1 = high_volume.var(ddof=1)
    variance2 = low_volume.var(ddof=1)

    welch_df = (
        (
            variance1 / n1
            +
            variance2 / n2
        ) ** 2
        /
        (
            ((variance1 / n1) ** 2) / (n1 - 1)
            +
            ((variance2 / n2) ** 2) / (n2 - 1)
        )
    )

    print("\nWelch two-sample t-test:")

    print(
        "H0: mean G/Sh_high-volume = "
        "mean G/Sh_low-volume"
    )

    print(
        "HA: mean G/Sh_high-volume != "
        "mean G/Sh_low-volume"
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
            "The sample provides sufficient evidence of a "
            "statistically significant difference in mean "
            "goal-per-shot efficiency between higher-volume "
            "and lower-volume shooters."
        )

    else:

        print(
            "Decision: Fail to reject H0."
        )

        print(
            "The sample does not provide sufficient evidence "
            "of a statistically significant difference in mean "
            "goal-per-shot efficiency between higher-volume "
            "and lower-volume shooters."
        )

    # -----------------------------------------------------------------
    # Assumption diagnostics
    # -----------------------------------------------------------------

    high_outliers = iqr_outlier_summary(
        high_volume
    )

    low_outliers = iqr_outlier_summary(
        low_volume
    )

    high_skew = stats.skew(
        high_volume,
        bias=False
    )

    low_skew = stats.skew(
        low_volume,
        bias=False
    )

    print("\nAssumption diagnostics:")

    print(
        "1. Observations represent individual players."
    )

    print(
        "2. The response variable, goals per shot, is quantitative."
    )

    print(
        "3. Welch's t-test does not require equal population variances."
    )

    print("\nPotential IQR outliers:")

    print(
        "Higher-volume: "
        f"{high_outliers['count']} "
        f"({high_outliers['percentage']:.1f}%)"
    )

    print(
        "Lower-volume: "
        f"{low_outliers['count']} "
        f"({low_outliers['percentage']:.1f}%)"
    )

    print("\nSample skewness:")

    print(
        f"Higher-volume: {high_skew:.3f}"
    )

    print(
        f"Lower-volume: {low_skew:.3f}"
    )

    print("\nInterpretation note:")

    print(
        "Goal-per-shot efficiency is naturally bounded at zero and "
        "may be right-skewed because many players score few or no goals. "
        "Welch's t-test is reasonably robust with these sample sizes, "
        "but the distribution shape and potential outliers should be "
        "acknowledged."
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
        "- Goal-per-shot efficiency can be influenced by shot quality, "
        "shot location and defensive pressure, which are not directly "
        "controlled for in this dataset."
    )

    print(
        "- Players are nested within national teams and face opponents "
        "of different quality."
    )

    print(
        "- The median Sh/90 split is a transparent grouping method, but "
        "it simplifies a continuous shooting-volume measure."
    )

    print(
        "- Tournament samples are relatively short, so scoring "
        "efficiency may be affected by small-sample variation."
    )

    print(
        "- The analysis identifies association rather than a causal "
        "effect of taking more shots."
    )

    print("\n" + "=" * 70)

    print(
        "TASK 2 ANALYSIS COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
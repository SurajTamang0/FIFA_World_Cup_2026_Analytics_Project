"""
Task 3 - Defensive Pressure and Discipline
FIFA World Cup 2026 Analytics Project

Research question:
Among defensively involved outfield players, do players with higher
defensive-action rates accumulate a different disciplinary-card rate
than players with lower defensive-action rates?

This task includes:
- data wrangling
- derived variables
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

MISC_FILE = (
    RAW_DATA_DIR / "fbref_2026_world_cup_player_miscellaneous.csv"
)

OUTPUT_SAMPLE_FILE = (
    PROCESSED_DATA_DIR / "task3_defensive_pressure_sample.csv"
)

OUTPUT_FIGURE_FILE = (
    FIGURES_DIR / "task3_defensive_pressure.png"
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

    Duplicate names receive suffixes such as .1, .2, etc.
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
            final_names.append(
                f"{name}.{counts[name]}"
            )

    result = dataframe.copy()

    result.columns = final_names

    return result


def load_fbref_csv(path):
    """
    Load an FBref CSV with a two-row header and clean the columns.
    """

    dataframe = pd.read_csv(
        path,
        header=[0, 1]
    )

    dataframe = flatten_fbref_columns(
        dataframe
    )

    if "Rk" in dataframe.columns:

        dataframe = dataframe[
            dataframe["Rk"]
            .astype(str)
            .str.strip()
            != "Rk"
        ].copy()

    dataframe.reset_index(
        drop=True,
        inplace=True
    )

    return dataframe


def convert_numeric(dataframe, columns):
    """
    Convert selected variables to numeric format.
    """

    result = dataframe.copy()

    for column in columns:

        if column in result.columns:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            )

    return result


def describe_by_group(
    dataframe,
    group_column,
    value_column
):
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


def mean_confidence_interval(
    series,
    confidence=0.95
):
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

    standard_error = stats.sem(
        clean
    )

    critical_value = stats.t.ppf(
        (1 + confidence) / 2,
        df=n - 1
    )

    margin = (
        critical_value
        * standard_error
    )

    return (
        float(mean),
        float(mean - margin),
        float(mean + margin),
    )


def welch_difference_confidence_interval(
    group1,
    group2,
    confidence=0.95
):
    """
    Calculate a Welch confidence interval for the difference
    between two independent means.

    Difference is group1 - group2.
    """

    x = pd.to_numeric(
        group1,
        errors="coerce"
    ).dropna()

    y = pd.to_numeric(
        group2,
        errors="coerce"
    ).dropna()

    n1 = len(x)
    n2 = len(y)

    mean1 = x.mean()
    mean2 = y.mean()

    variance1 = x.var(ddof=1)
    variance2 = y.var(ddof=1)

    difference = (
        mean1 - mean2
    )

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
        ((variance1 / n1) ** 2)
        / (n1 - 1)
        +
        ((variance2 / n2) ** 2)
        / (n2 - 1)
    )

    degrees_of_freedom = (
        numerator / denominator
    )

    critical_value = stats.t.ppf(
        (1 + confidence) / 2,
        df=degrees_of_freedom
    )

    margin = (
        critical_value
        * standard_error
    )

    return (
        float(difference),
        float(difference - margin),
        float(difference + margin),
        float(degrees_of_freedom),
    )


def iqr_outlier_summary(series):
    """
    Detect potential outliers using the 1.5 x IQR rule.
    """

    clean = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)

    iqr = q3 - q1

    lower_bound = (
        q1 - 1.5 * iqr
    )

    upper_bound = (
        q3 + 1.5 * iqr
    )

    outliers = clean[
        (clean < lower_bound)
        |
        (clean > upper_bound)
    ]

    percentage = (
        len(outliers)
        / len(clean)
        * 100
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

def prepare_task3_data():
    """
    Load and merge Standard and Miscellaneous statistics,
    then construct defensive activity and disciplinary variables.
    """

    standard = load_fbref_csv(
        STANDARD_FILE
    )

    miscellaneous = load_fbref_csv(
        MISC_FILE
    )

    print("Raw dataset dimensions:")

    print(
        f"Standard:       {standard.shape}"
    )

    print(
        f"Miscellaneous:  {miscellaneous.shape}"
    )

    standard = convert_numeric(
        standard,
        [
            "Min",
            "90s",
        ]
    )

    miscellaneous = convert_numeric(
        miscellaneous,
        [
            "90s",
            "CrdY",
            "CrdR",
            "Int",
            "TklW",
        ]
    )

    standard_subset = standard[
        [
            "Player",
            "Squad",
            "Pos",
            "Min",
        ]
    ].copy()

    miscellaneous_subset = miscellaneous[
        [
            "Player",
            "Squad",
            "90s",
            "CrdY",
            "CrdR",
            "Int",
            "TklW",
        ]
    ].copy()

    merged = miscellaneous_subset.merge(
        standard_subset,
        on=["Player", "Squad"],
        how="left",
    )

    print(
        f"\nPlayers after merge: "
        f"{len(merged)}"
    )

    # ---------------------------------------------------------------
    # Remove goalkeepers
    # ---------------------------------------------------------------

    merged["Pos"] = (
        merged["Pos"]
        .astype(str)
    )

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

    # ---------------------------------------------------------------
    # Eligibility filtering
    # ---------------------------------------------------------------

    eligible = outfield[
        outfield["Min"].notna()
        & (outfield["Min"] >= MIN_MINUTES)
        & outfield["90s"].notna()
        & (outfield["90s"] > 0)
        & outfield["TklW"].notna()
        & outfield["Int"].notna()
        & outfield["CrdY"].notna()
        & outfield["CrdR"].notna()
    ].copy()

    print(
        f"Eligible players with at least "
        f"{MIN_MINUTES} minutes: "
        f"{len(eligible)}"
    )

    # ---------------------------------------------------------------
    # Construct defensive activity
    # ---------------------------------------------------------------

    eligible["DefensiveActions"] = (
        eligible["TklW"]
        +
        eligible["Int"]
    )

    eligible["DefensiveActions90"] = (
        eligible["DefensiveActions"]
        /
        eligible["90s"]
    )

    # ---------------------------------------------------------------
    # Construct disciplinary-card rate
    # ---------------------------------------------------------------

    eligible["TotalCards"] = (
        eligible["CrdY"]
        +
        eligible["CrdR"]
    )

    eligible["CardRate90"] = (
        eligible["TotalCards"]
        /
        eligible["90s"]
    )

    # Require some defensive involvement so the comparison is
    # meaningful for the research question.

    eligible = eligible[
        eligible["DefensiveActions"] > 0
    ].copy()

    print(
        "Eligible defensively involved players: "
        f"{len(eligible)}"
    )

    print(
        "Missing DefensiveActions90 values:",
        eligible["DefensiveActions90"]
        .isna()
        .sum()
    )

    print(
        "Missing CardRate90 values:",
        eligible["CardRate90"]
        .isna()
        .sum()
    )

    return eligible


# ---------------------------------------------------------------------
# Defensive pressure grouping
# ---------------------------------------------------------------------

def construct_pressure_groups(
    eligible
):
    """
    Split eligible players into higher and lower defensive activity
    using the median defensive-actions-per-90 rate.
    """

    median_defensive_rate = (
        eligible["DefensiveActions90"]
        .median()
    )

    result = eligible.copy()

    result["DefensivePressureGroup"] = np.where(
        result["DefensiveActions90"]
        >= median_defensive_rate,
        "Higher defensive activity",
        "Lower defensive activity",
    )

    return (
        result,
        float(median_defensive_rate)
    )


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------

def create_stratified_sample(
    eligible
):
    """
    Draw equal-size reproducible random samples from both
    defensive-pressure groups.
    """

    high_population = eligible[
        eligible["DefensivePressureGroup"]
        == "Higher defensive activity"
    ].copy()

    low_population = eligible[
        eligible["DefensivePressureGroup"]
        == "Lower defensive activity"
    ].copy()

    if len(high_population) < SAMPLE_SIZE_PER_GROUP:

        raise ValueError(
            "Higher defensive-activity population is smaller "
            "than the planned sample size."
        )

    if len(low_population) < SAMPLE_SIZE_PER_GROUP:

        raise ValueError(
            "Lower defensive-activity population is smaller "
            "than the planned sample size."
        )

    high_sample = high_population.sample(
        n=SAMPLE_SIZE_PER_GROUP,
        random_state=RANDOM_SEED,
        replace=False
    )

    low_sample = low_population.sample(
        n=SAMPLE_SIZE_PER_GROUP,
        random_state=RANDOM_SEED,
        replace=False
    )

    sample = pd.concat(
        [
            high_sample,
            low_sample
        ],
        ignore_index=True
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
    Create a boxplot comparing disciplinary-card rates.
    """

    high_activity = sample.loc[
        sample["DefensivePressureGroup"]
        == "Higher defensive activity",
        "CardRate90",
    ].dropna()

    low_activity = sample.loc[
        sample["DefensivePressureGroup"]
        == "Lower defensive activity",
        "CardRate90",
    ].dropna()

    figure_data = [
        high_activity,
        low_activity
    ]

    labels = [
        "Higher defensive\nactivity",
        "Lower defensive\nactivity",
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.boxplot(
        figure_data,
        tick_labels=labels,
        showmeans=True
    )

    plt.xlabel(
        "Defensive Activity Group"
    )

    plt.ylabel(
        "Cards per 90 Minutes"
    )

    plt.title(
        "FIFA World Cup 2026: Defensive Activity and Discipline"
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

    print(
        "TASK 3 - DEFENSIVE PRESSURE AND DISCIPLINE"
    )

    print("=" * 70)

    print(
        "\nResearch question:\n"
        "Among defensively involved outfield players, do players "
        "with higher defensive-action rates accumulate a different "
        "disciplinary-card rate than players with lower "
        "defensive-action rates?\n"
    )

    # ---------------------------------------------------------------
    # Prepare analytical population
    # ---------------------------------------------------------------

    eligible = prepare_task3_data()

    eligible, median_defensive_rate = (
        construct_pressure_groups(
            eligible
        )
    )

    print(
        "\nMedian defensive actions per 90 used "
        "for grouping: "
        f"{median_defensive_rate:.3f}"
    )

    print(
        "\nDefensive-activity group counts:"
    )

    print(
        eligible[
            "DefensivePressureGroup"
        ].value_counts()
    )

    # ---------------------------------------------------------------
    # Population descriptive statistics
    # ---------------------------------------------------------------

    print(
        "\nEligible-population descriptive statistics "
        "for card rate per 90:\n"
    )

    population_stats = describe_by_group(
        eligible,
        "DefensivePressureGroup",
        "CardRate90"
    )

    print(
        population_stats
    )

    # ---------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------

    sample = create_stratified_sample(
        eligible
    )

    print(
        "\nSampling design:"
    )

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
        f"Total sample size: "
        f"{len(sample)}"
    )

    print(
        "Analytical sample saved to:"
    )

    print(
        OUTPUT_SAMPLE_FILE
    )

    # ---------------------------------------------------------------
    # Sample descriptive statistics
    # ---------------------------------------------------------------

    print(
        "\nSample descriptive statistics:\n"
    )

    sample_stats = describe_by_group(
        sample,
        "DefensivePressureGroup",
        "CardRate90"
    )

    print(
        sample_stats
    )

    high_activity = sample.loc[
        sample["DefensivePressureGroup"]
        == "Higher defensive activity",
        "CardRate90",
    ].dropna()

    low_activity = sample.loc[
        sample["DefensivePressureGroup"]
        == "Lower defensive activity",
        "CardRate90",
    ].dropna()

    # ---------------------------------------------------------------
    # Confidence intervals
    # ---------------------------------------------------------------

    high_mean, high_low, high_high = (
        mean_confidence_interval(
            high_activity
        )
    )

    low_mean, low_low, low_high = (
        mean_confidence_interval(
            low_activity
        )
    )

    print(
        "\n95% confidence intervals for "
        "mean card rate per 90:"
    )

    print(
        "\nHigher defensive activity:"
    )

    print(
        f"Mean = {high_mean:.3f}"
    )

    print(
        f"95% CI = "
        f"[{high_low:.3f}, "
        f"{high_high:.3f}]"
    )

    print(
        "\nLower defensive activity:"
    )

    print(
        f"Mean = {low_mean:.3f}"
    )

    print(
        f"95% CI = "
        f"[{low_low:.3f}, "
        f"{low_high:.3f}]"
    )

    difference, diff_low, diff_high, _ = (
        welch_difference_confidence_interval(
            high_activity,
            low_activity
        )
    )

    print(
        "\nDifference in mean card rate per 90 "
        "(Higher activity - Lower activity):"
    )

    print(
        f"Difference = {difference:.3f}"
    )

    print(
        f"95% CI = "
        f"[{diff_low:.3f}, "
        f"{diff_high:.3f}]"
    )

    # ---------------------------------------------------------------
    # Welch two-sample t-test
    # ---------------------------------------------------------------

    test_result = stats.ttest_ind(
        high_activity,
        low_activity,
        equal_var=False,
        nan_policy="omit"
    )

    t_statistic = float(
        test_result.statistic
    )

    p_value = float(
        test_result.pvalue
    )

    n1 = len(high_activity)
    n2 = len(low_activity)

    variance1 = high_activity.var(
        ddof=1
    )

    variance2 = low_activity.var(
        ddof=1
    )

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

    print(
        "\nWelch two-sample t-test:"
    )

    print(
        "H0: mean CardRate90_high = "
        "mean CardRate90_low"
    )

    print(
        "HA: mean CardRate90_high != "
        "mean CardRate90_low"
    )

    print(
        f"Alpha = {ALPHA:.2f}"
    )

    print(
        f"t statistic = "
        f"{t_statistic:.3f}"
    )

    print(
        f"Welch degrees of freedom = "
        f"{welch_df:.2f}"
    )

    print(
        f"p-value = "
        f"{p_value:.4f}"
    )

    if p_value < ALPHA:

        print(
            "Decision: Reject H0."
        )

        print(
            "The sample provides sufficient evidence "
            "of a statistically significant difference "
            "in mean disciplinary-card rate between the "
            "two defensive-activity groups."
        )

    else:

        print(
            "Decision: Fail to reject H0."
        )

        print(
            "The sample does not provide sufficient evidence "
            "of a statistically significant difference in "
            "mean disciplinary-card rate between the two "
            "defensive-activity groups."
        )

    # ---------------------------------------------------------------
    # Assumption diagnostics
    # ---------------------------------------------------------------

    high_outliers = iqr_outlier_summary(
        high_activity
    )

    low_outliers = iqr_outlier_summary(
        low_activity
    )

    high_skew = stats.skew(
        high_activity,
        bias=False
    )

    low_skew = stats.skew(
        low_activity,
        bias=False
    )

    print(
        "\nAssumption diagnostics:"
    )

    print(
        "1. Observations represent individual players."
    )

    print(
        "2. Card rate per 90 is a quantitative rate variable."
    )

    print(
        "3. Welch's t-test does not require equal "
        "population variances."
    )

    print(
        "\nPotential IQR outliers:"
    )

    print(
        "Higher defensive activity: "
        f"{high_outliers['count']} "
        f"({high_outliers['percentage']:.1f}%)"
    )

    print(
        "Lower defensive activity: "
        f"{low_outliers['count']} "
        f"({low_outliers['percentage']:.1f}%)"
    )

    print(
        "\nSample skewness:"
    )

    print(
        "Higher defensive activity: "
        f"{high_skew:.3f}"
    )

    print(
        "Lower defensive activity: "
        f"{low_skew:.3f}"
    )

    print(
        "\nInterpretation note:"
    )

    print(
        "Card-rate data may be right-skewed because many "
        "players receive no cards while a smaller number "
        "accumulate disciplinary events. The equal sample "
        "sizes and use of Welch's test provide some robustness, "
        "but skewness and outliers should still be considered "
        "when interpreting the result."
    )

    # ---------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------

    create_boxplot(
        sample
    )

    print(
        "\nFigure saved to:"
    )

    print(
        OUTPUT_FIGURE_FILE
    )

    # ---------------------------------------------------------------
    # Limitations
    # ---------------------------------------------------------------

    print(
        "\nKey limitations:"
    )

    print(
        "- Tackles won plus interceptions provide a useful "
        "defensive-activity measure, but they do not capture "
        "every form of defensive pressure."
    )

    print(
        "- Card accumulation may also depend on playing position, "
        "referee decisions, tactical role and match context."
    )

    print(
        "- Players are nested within national teams and encounter "
        "opponents of different quality."
    )

    print(
        "- The median split simplifies a continuous defensive-activity "
        "measure into two categories."
    )

    print(
        "- Yellow and red cards are counted equally in the rate, "
        "even though their disciplinary severity differs."
    )

    print(
        "- The analysis identifies statistical association rather "
        "than evidence that defensive activity causes disciplinary cards."
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TASK 3 ANALYSIS COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
"""
Task 4 - Playing Time and On/Off Impact
FIFA World Cup 2026 Analytics Project

Research question:
Among outfield players with meaningful tournament exposure,
do players with a higher share of available team minutes have
a different on/off goal-difference impact than players with a
lower share of available team minutes?

This task includes:
- data wrangling
- eligibility filtering
- derived grouping variable
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

PLAYING_TIME_FILE = (
    RAW_DATA_DIR / "fbref_2026_world_cup_player_playing_time.csv"
)

OUTPUT_SAMPLE_FILE = (
    PROCESSED_DATA_DIR / "task4_onoff_impact_sample.csv"
)

OUTPUT_FIGURE_FILE = (
    FIGURES_DIR / "task4_onoff_impact.png"
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
            final_names.append(f"{name}.{counts[name]}")

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
    Detect potential outliers using the 1.5 x IQR rule.
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
        |
        (clean > upper_bound)
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

def prepare_task4_data():
    """
    Load and merge Standard and Playing Time datasets,
    then prepare the variables required for Task 4.
    """

    standard = load_fbref_csv(
        STANDARD_FILE
    )

    playing_time = load_fbref_csv(
        PLAYING_TIME_FILE
    )

    print("Raw dataset dimensions:")
    print(f"Standard:      {standard.shape}")
    print(f"Playing time:  {playing_time.shape}")

    standard = convert_numeric(
        standard,
        [
            "Min",
        ]
    )

    playing_time = convert_numeric(
        playing_time,
        [
            "Min",
            "Min%",
            "On-Off",
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

    playing_subset = playing_time[
        [
            "Player",
            "Squad",
            "Min%",
            "On-Off",
        ]
    ].copy()

    merged = playing_subset.merge(
        standard_subset,
        on=["Player", "Squad"],
        how="inner"
    )

    print(
        f"\nPlayers after merge: {len(merged)}"
    )

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

    eligible = outfield[
        outfield["Min"].notna()
        & (outfield["Min"] >= MIN_MINUTES)
        & outfield["Min%"].notna()
        & outfield["On-Off"].notna()
    ].copy()

    print(
        f"Eligible outfield players with at least "
        f"{MIN_MINUTES} minutes and available On-Off values: "
        f"{len(eligible)}"
    )

    print(
        "Missing Min% values:",
        eligible["Min%"].isna().sum()
    )

    print(
        "Missing On-Off values:",
        eligible["On-Off"].isna().sum()
    )

    return eligible


# ---------------------------------------------------------------------
# Playing-time grouping
# ---------------------------------------------------------------------

def construct_playing_time_groups(eligible):
    """
    Split eligible players into higher and lower playing-time
    responsibility groups using the median tournament minute share.
    """

    median_minute_share = eligible["Min%"].median()

    result = eligible.copy()

    result["PlayingTimeGroup"] = np.where(
        result["Min%"] >= median_minute_share,
        "Higher-minute-share",
        "Lower-minute-share",
    )

    return result, float(median_minute_share)


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------

def create_stratified_sample(eligible):
    """
    Draw equal-sized reproducible random samples from both
    playing-time groups.
    """

    high_population = eligible[
        eligible["PlayingTimeGroup"]
        == "Higher-minute-share"
    ].copy()

    low_population = eligible[
        eligible["PlayingTimeGroup"]
        == "Lower-minute-share"
    ].copy()

    if len(high_population) < SAMPLE_SIZE_PER_GROUP:
        raise ValueError(
            "Higher-minute-share population is smaller "
            "than the planned sample size."
        )

    if len(low_population) < SAMPLE_SIZE_PER_GROUP:
        raise ValueError(
            "Lower-minute-share population is smaller "
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
    Create a boxplot comparing On-Off values across playing-time groups.
    """

    high_share = sample.loc[
        sample["PlayingTimeGroup"]
        == "Higher-minute-share",
        "On-Off",
    ].dropna()

    low_share = sample.loc[
        sample["PlayingTimeGroup"]
        == "Lower-minute-share",
        "On-Off",
    ].dropna()

    figure_data = [
        high_share,
        low_share,
    ]

    labels = [
        "Higher minute\nshare",
        "Lower minute\nshare",
    ]

    plt.figure(figsize=(9, 6))

    plt.boxplot(
        figure_data,
        tick_labels=labels,
        showmeans=True
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xlabel(
        "Playing-Time Responsibility Group"
    )

    plt.ylabel(
        "On-Off Goal Difference per 90"
    )

    plt.title(
        "FIFA World Cup 2026: Playing-Time Responsibility "
        "and On/Off Impact"
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
    print("TASK 4 - PLAYING TIME AND ON/OFF IMPACT")
    print("=" * 70)

    print(
        "\nResearch question:\n"
        "Among outfield players with meaningful tournament exposure, "
        "do players with a higher share of available team minutes have "
        "a different on/off goal-difference impact than players with "
        "a lower share of available team minutes?\n"
    )

    eligible = prepare_task4_data()

    eligible, median_minute_share = (
        construct_playing_time_groups(
            eligible
        )
    )

    print(
        "\nMedian tournament minute share used for grouping: "
        f"{median_minute_share:.3f}"
    )

    print(
        "\nPlaying-time group counts:"
    )

    print(
        eligible[
            "PlayingTimeGroup"
        ].value_counts()
    )

    print(
        "\nEligible-population descriptive statistics "
        "for On-Off goal difference per 90:\n"
    )

    population_stats = describe_by_group(
        eligible,
        "PlayingTimeGroup",
        "On-Off"
    )

    print(population_stats)

    sample = create_stratified_sample(
        eligible
    )

    print("\nSampling design:")
    print("Method: stratified simple random sampling")
    print(f"Random seed: {RANDOM_SEED}")
    print(
        f"Sample size per group: "
        f"{SAMPLE_SIZE_PER_GROUP}"
    )
    print(
        f"Total sample size: "
        f"{len(sample)}"
    )
    print("Analytical sample saved to:")
    print(OUTPUT_SAMPLE_FILE)

    print(
        "\nSample descriptive statistics:\n"
    )

    sample_stats = describe_by_group(
        sample,
        "PlayingTimeGroup",
        "On-Off"
    )

    print(sample_stats)

    high_share = sample.loc[
        sample["PlayingTimeGroup"]
        == "Higher-minute-share",
        "On-Off",
    ].dropna()

    low_share = sample.loc[
        sample["PlayingTimeGroup"]
        == "Lower-minute-share",
        "On-Off",
    ].dropna()

    high_mean, high_low, high_high = (
        mean_confidence_interval(
            high_share
        )
    )

    low_mean, low_low, low_high = (
        mean_confidence_interval(
            low_share
        )
    )

    print(
        "\n95% confidence intervals for "
        "mean On-Off goal difference per 90:"
    )

    print("\nHigher-minute-share:")
    print(f"Mean = {high_mean:.3f}")
    print(
        f"95% CI = "
        f"[{high_low:.3f}, "
        f"{high_high:.3f}]"
    )

    print("\nLower-minute-share:")
    print(f"Mean = {low_mean:.3f}")
    print(
        f"95% CI = "
        f"[{low_low:.3f}, "
        f"{low_high:.3f}]"
    )

    difference, diff_low, diff_high, _ = (
        welch_difference_confidence_interval(
            high_share,
            low_share
        )
    )

    print(
        "\nDifference in mean On-Off "
        "(Higher minute share - Lower minute share):"
    )

    print(
        f"Difference = {difference:.3f}"
    )

    print(
        f"95% CI = "
        f"[{diff_low:.3f}, "
        f"{diff_high:.3f}]"
    )

    test_result = stats.ttest_ind(
        high_share,
        low_share,
        equal_var=False,
        nan_policy="omit"
    )

    t_statistic = float(
        test_result.statistic
    )

    p_value = float(
        test_result.pvalue
    )

    n1 = len(high_share)
    n2 = len(low_share)

    variance1 = high_share.var(ddof=1)
    variance2 = low_share.var(ddof=1)

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
        "H0: mean On-Off_high-minute-share = "
        "mean On-Off_low-minute-share"
    )

    print(
        "HA: mean On-Off_high-minute-share != "
        "mean On-Off_low-minute-share"
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
            "in mean On-Off impact between players with "
            "higher and lower tournament minute shares."
        )

    else:

        print(
            "Decision: Fail to reject H0."
        )

        print(
            "The sample does not provide sufficient evidence "
            "of a statistically significant difference "
            "in mean On-Off impact between players with "
            "higher and lower tournament minute shares."
        )

    high_outliers = iqr_outlier_summary(
        high_share
    )

    low_outliers = iqr_outlier_summary(
        low_share
    )

    high_skew = stats.skew(
        high_share,
        bias=False
    )

    low_skew = stats.skew(
        low_share,
        bias=False
    )

    print(
        "\nAssumption diagnostics:"
    )

    print(
        "1. Observations represent individual players."
    )

    print(
        "2. On-Off goal difference per 90 is quantitative."
    )

    print(
        "3. Welch's t-test does not require equal "
        "population variances."
    )

    print(
        "\nPotential IQR outliers:"
    )

    print(
        "Higher-minute-share: "
        f"{high_outliers['count']} "
        f"({high_outliers['percentage']:.1f}%)"
    )

    print(
        "Lower-minute-share: "
        f"{low_outliers['count']} "
        f"({low_outliers['percentage']:.1f}%)"
    )

    print(
        "\nSample skewness:"
    )

    print(
        "Higher-minute-share: "
        f"{high_skew:.3f}"
    )

    print(
        "Lower-minute-share: "
        f"{low_skew:.3f}"
    )

    print(
        "\nInterpretation note:"
    )

    print(
        "On-Off statistics can be highly variable because they are "
        "affected by team strength, opponent quality, score state, "
        "substitution timing and limited tournament minutes. "
        "The result should therefore be interpreted as an association "
        "rather than a direct measure of individual causal impact."
    )

    create_boxplot(
        sample
    )

    print(
        "\nFigure saved to:"
    )

    print(
        OUTPUT_FIGURE_FILE
    )

    print(
        "\nKey limitations:"
    )

    print(
        "- On-Off goal difference reflects team performance while a "
        "player is on and off the pitch, so it cannot isolate the "
        "individual player's causal contribution."
    )

    print(
        "- Stronger players may receive more minutes because of prior "
        "selection decisions, creating potential selection bias."
    )

    print(
        "- Players are nested within national teams and face opponents "
        "of different strength."
    )

    print(
        "- Match state and substitution timing can strongly influence "
        "On-Off values."
    )

    print(
        "- The median minute-share split simplifies a continuous "
        "playing-time measure into two groups."
    )

    print(
        "- Tournament samples are short, so extreme On-Off values may "
        "be driven by relatively few match events."
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TASK 4 ANALYSIS COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
"""
Question feasibility analysis for the FIFA World Cup 2026 project.

Purpose
-------
This script checks whether the four proposed analytic questions have:
1. the required variables,
2. sufficient usable observations,
3. meaningful comparison groups, and
4. enough non-missing data to proceed to formal analysis.

It does NOT perform the final hypothesis tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


DATA_FILES = {
    "standard": "fbref_2026_world_cup_player_standard.csv",
    "shooting": "fbref_2026_world_cup_player_shooting.csv",
    "goalkeeping": "fbref_2026_world_cup_player_goalkeeping.csv",
    "miscellaneous": "fbref_2026_world_cup_player_miscellaneous.csv",
    "playing_time": "fbref_2026_world_cup_player_playing_time.csv",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def flatten_fbref_columns(dataframe):
    """
    Convert FBref's two-level CSV headers into simple column names.

    Example:
        ('Playing Time', 'Min') -> 'Min'
        ('Performance', 'Gls') -> 'Gls'

    Duplicate names are given pandas-style suffixes:
        Gls, Gls.1
    """

    new_columns = []
    seen = {}

    for column in dataframe.columns:

        if isinstance(column, tuple):
            base_name = str(column[-1]).strip()
        else:
            base_name = str(column).strip()

        if base_name in seen:
            seen[base_name] += 1
            final_name = f"{base_name}.{seen[base_name]}"
        else:
            seen[base_name] = 0
            final_name = base_name

        new_columns.append(final_name)

    dataframe = dataframe.copy()
    dataframe.columns = new_columns

    return dataframe


def load_datasets():
    """
    Load all five raw FBref CSV datasets using their two-row headers.
    """

    datasets = {}

    for name, filename in DATA_FILES.items():

        file_path = RAW_DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {file_path}"
            )

        dataframe = pd.read_csv(
            file_path,
            header=[0, 1]
        )

        dataframe = flatten_fbref_columns(dataframe)

        datasets[name] = dataframe

    return datasets


def convert_numeric(dataframe, columns):
    """
    Convert selected columns to numeric form.

    Invalid values are converted to NaN rather than causing the script
    to fail.
    """

    dataframe = dataframe.copy()

    for column in columns:

        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            )

    return dataframe


def print_separator():
    print("=" * 72)


def print_group_counts(dataframe, group_column):
    """
    Print group sizes in a consistent format.
    """

    counts = dataframe[group_column].value_counts(dropna=False)

    for group, count in counts.items():
        print(f"{str(group):25} {count}")


def practical_feasibility(group_sizes):
    """
    Provide a preliminary feasibility message.

    Thirty observations per group is used only as a practical screening
    benchmark here. It is not being treated as a formal statistical
    requirement.
    """

    if len(group_sizes) < 2:
        return "NOT READY - fewer than two comparison groups were produced."

    minimum_size = min(group_sizes)

    if minimum_size >= 30:
        return (
            "Strong preliminary feasibility: both main comparison groups "
            "contain at least 30 observations."
        )

    if minimum_size >= 15:
        return (
            "Potentially feasible, but at least one group is relatively "
            "small. Distribution and assumption checks will be important."
        )

    return (
        "Weak feasibility in the current form: at least one comparison "
        "group is very small and the question may need redesign."
    )


# ---------------------------------------------------------------------
# TASK 1
# Super-sub effect
# ---------------------------------------------------------------------

def check_task1_super_sub(datasets):

    print("\nTASK 1 FEASIBILITY - SUPER-SUB EFFECT")
    print_separator()

    print(
        "Question:\n"
        "Do substitute-dominant outfield players produce a different "
        "rate of attacking contribution per 90 minutes than "
        "starter-dominant players?"
    )

    standard = datasets["standard"].copy()
    playing = datasets["playing_time"].copy()

    standard = convert_numeric(
        standard,
        [
            "Min",
            "90s",
            "Gls",
            "Ast",
        ],
    )

    playing = convert_numeric(
        playing,
        [
            "Min",
            "Starts",
            "Subs",
            "90s",
        ],
    )

    # Use only the variables needed from Standard.
    attacking = standard[
        [
            "Player",
            "Squad",
            "Pos",
            "Gls",
            "Ast",
            "90s",
        ]
    ].copy()

    # Derive attacking contribution ourselves rather than relying on
    # duplicated FBref per-90 column names.
    attacking["GA90"] = np.where(
        attacking["90s"] > 0,
        (attacking["Gls"] + attacking["Ast"]) / attacking["90s"],
        np.nan,
    )

    usage = playing[
        [
            "Player",
            "Squad",
            "Min",
            "Starts",
            "Subs",
        ]
    ].copy()

    merged = attacking.merge(
        usage,
        on=["Player", "Squad"],
        how="inner",
    )

    # Goalkeepers are not relevant to this attacking-player question.
    merged = merged[
        ~merged["Pos"].astype(str).str.contains("GK", na=False)
    ].copy()

    # Minimum exposure rule to reduce distortion from very short appearances.
    eligible = merged[
        (merged["Min"] >= 90)
        & merged["GA90"].notna()
    ].copy()

    # Total appearances used to classify starting/substitute usage.
    eligible["usage_appearances"] = (
        eligible["Starts"].fillna(0)
        + eligible["Subs"].fillna(0)
    )

    eligible = eligible[
        eligible["usage_appearances"] > 0
    ].copy()

    eligible["starter_share"] = (
        eligible["Starts"].fillna(0)
        / eligible["usage_appearances"]
    )

    conditions = [
        eligible["starter_share"] >= (2 / 3),
        eligible["starter_share"] <= (1 / 3),
    ]

    choices = [
        "Starter-dominant",
        "Substitute-dominant",
    ]

    eligible["Usage"] = np.select(
        conditions,
        choices,
        default="Balanced",
    )

    print(f"\nMerged outfield players: {len(merged)}")
    print(f"Eligible players (Min >= 90): {len(eligible)}")

    print("\nUsage-group counts:")
    print_group_counts(eligible, "Usage")

    print("\nAttacking contribution (G+A per 90):")

    summary = (
        eligible.groupby("Usage")["GA90"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .round(3)
    )

    print(summary)

    starter_n = int(
        (eligible["Usage"] == "Starter-dominant").sum()
    )

    substitute_n = int(
        (eligible["Usage"] == "Substitute-dominant").sum()
    )

    print("\nMissing GA90 values:", eligible["GA90"].isna().sum())

    print("\nPreliminary feasibility assessment:")
    print(
        practical_feasibility(
            [starter_n, substitute_n]
        )
    )


# ---------------------------------------------------------------------
# TASK 2
# Shooting volume and scoring efficiency
# ---------------------------------------------------------------------

def check_task2_shooting_efficiency(datasets):

    print("\n\nTASK 2 FEASIBILITY - SHOOTING VOLUME VS EFFICIENCY")
    print_separator()

    print(
        "Question:\n"
        "Among attacking players with meaningful tournament exposure, "
        "do high-volume shooters have a different scoring efficiency "
        "than lower-volume shooters?"
    )

    standard = datasets["standard"].copy()
    shooting = datasets["shooting"].copy()

    standard = convert_numeric(
        standard,
        ["Min"],
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
        ],
    )

    standard_info = standard[
        [
            "Player",
            "Squad",
            "Pos",
            "Min",
        ]
    ].copy()

    merged = shooting.merge(
        standard_info,
        on=["Player", "Squad"],
        how="inner",
        suffixes=("", "_standard"),
    )

    # Include forwards and midfielders.
    attacking_mask = (
        merged["Pos_standard"]
        .astype(str)
        .str.contains("FW|MF", regex=True, na=False)
    )

    eligible = merged[
        attacking_mask
        & (merged["Min"] >= 180)
        & (merged["Sh"] >= 3)
        & merged["Sh/90"].notna()
        & merged["G/Sh"].notna()
    ].copy()

    print(f"\nEligible attacking players: {len(eligible)}")

    if len(eligible) == 0:

        print(
            "No eligible observations were produced. "
            "Task definition requires review."
        )
        return

    shot_volume_median = eligible["Sh/90"].median()

    eligible["Shooting volume"] = np.where(
        eligible["Sh/90"] >= shot_volume_median,
        "High-volume",
        "Lower-volume",
    )

    print(
        f"Median Sh/90 used for preliminary grouping: "
        f"{shot_volume_median:.3f}"
    )

    print("\nGroup counts:")
    print_group_counts(
        eligible,
        "Shooting volume",
    )

    print("\nGoal-per-shot efficiency by shooting-volume group:")

    summary = (
        eligible.groupby("Shooting volume")["G/Sh"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .round(3)
    )

    print(summary)

    high_n = int(
        (eligible["Shooting volume"] == "High-volume").sum()
    )

    low_n = int(
        (eligible["Shooting volume"] == "Lower-volume").sum()
    )

    print(
        "\nMissing G/Sh values after eligibility filtering:",
        eligible["G/Sh"].isna().sum(),
    )

    print("\nPreliminary feasibility assessment:")
    print(
        practical_feasibility(
            [high_n, low_n]
        )
    )


# ---------------------------------------------------------------------
# TASK 3
# Defensive pressure and discipline
# ---------------------------------------------------------------------

def check_task3_defensive_pressure(datasets):

    print("\n\nTASK 3 FEASIBILITY - DEFENSIVE PRESSURE AND DISCIPLINE")
    print_separator()

    print(
        "Question:\n"
        "Among defensively involved outfield players, do players with "
        "higher defensive-action rates accumulate a different disciplinary "
        "rate than players with lower defensive-action rates?"
    )

    miscellaneous = datasets["miscellaneous"].copy()

    miscellaneous = convert_numeric(
        miscellaneous,
        [
            "90s",
            "CrdY",
            "CrdR",
            "2CrdY",
            "Fls",
            "Int",
            "TklW",
        ],
    )

    # Exclude goalkeepers.
    eligible = miscellaneous[
        ~miscellaneous["Pos"]
        .astype(str)
        .str.contains("GK", na=False)
    ].copy()

    # At least 180 minutes = 2 full-match equivalents.
    eligible = eligible[
        (eligible["90s"] >= 2)
        & eligible["TklW"].notna()
        & eligible["Int"].notna()
        & eligible["CrdY"].notna()
        & eligible["CrdR"].notna()
    ].copy()

    # Defensive activity:
    # successful tackles + interceptions per 90 minutes.
    eligible["DefensiveActions90"] = (
        eligible["TklW"]
        + eligible["Int"]
    ) / eligible["90s"]

    # Weighted card rate.
    # A red card is treated as two card units for this preliminary
    # disciplinary-rate measure.
    eligible["CardRate90"] = (
        eligible["CrdY"]
        + (2 * eligible["CrdR"])
    ) / eligible["90s"]

    print(f"\nEligible outfield players: {len(eligible)}")

    if len(eligible) == 0:

        print(
            "No eligible observations were produced. "
            "Task definition requires review."
        )
        return

    defensive_median = eligible["DefensiveActions90"].median()

    eligible["Defensive pressure"] = np.where(
        eligible["DefensiveActions90"] >= defensive_median,
        "Higher defensive activity",
        "Lower defensive activity",
    )

    print(
        f"Median defensive actions per 90: "
        f"{defensive_median:.3f}"
    )

    print("\nGroup counts:")
    print_group_counts(
        eligible,
        "Defensive pressure",
    )

    print("\nCard rate per 90 by defensive-activity group:")

    summary = (
        eligible.groupby("Defensive pressure")["CardRate90"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .round(3)
    )

    print(summary)

    high_n = int(
        (
            eligible["Defensive pressure"]
            == "Higher defensive activity"
        ).sum()
    )

    low_n = int(
        (
            eligible["Defensive pressure"]
            == "Lower defensive activity"
        ).sum()
    )

    print(
        "\nMissing CardRate90 values:",
        eligible["CardRate90"].isna().sum(),
    )

    print("\nPreliminary feasibility assessment:")
    print(
        practical_feasibility(
            [high_n, low_n]
        )
    )


# ---------------------------------------------------------------------
# TASK 4
# Playing-time responsibility and on/off impact
# ---------------------------------------------------------------------

def check_task4_on_off_impact(datasets):

    print("\n\nTASK 4 FEASIBILITY - PLAYING TIME AND ON/OFF IMPACT")
    print_separator()

    print(
        "Question:\n"
        "Do heavily used players have a different on/off goal-difference "
        "impact from players with lower playing-time responsibility?"
    )

    playing = datasets["playing_time"].copy()

    playing = convert_numeric(
        playing,
        [
            "MP",
            "Min",
            "Min%",
            "90s",
            "Starts",
            "Subs",
            "+/-",
            "+/-90",
            "On-Off",
        ],
    )

    # Remove goalkeepers because their substitution/rotation patterns differ
    # substantially from those of outfield players.
    eligible = playing[
        ~playing["Pos"]
        .astype(str)
        .str.contains("GK", na=False)
    ].copy()

    # Require reasonable exposure and a valid On-Off statistic.
    eligible = eligible[
        (eligible["Min"] >= 180)
        & eligible["Min%"].notna()
        & eligible["On-Off"].notna()
    ].copy()

    print(f"\nEligible outfield players: {len(eligible)}")

    if len(eligible) == 0:

        print(
            "No eligible observations were produced. "
            "Task definition requires review."
        )
        return

    minute_share_median = eligible["Min%"].median()

    eligible["Playing-time responsibility"] = np.where(
        eligible["Min%"] >= minute_share_median,
        "Higher-minute-share",
        "Lower-minute-share",
    )

    print(
        f"Median tournament minute share: "
        f"{minute_share_median:.3f}"
    )

    print("\nGroup counts:")
    print_group_counts(
        eligible,
        "Playing-time responsibility",
    )

    print("\nOn-Off statistic by playing-time group:")

    summary = (
        eligible.groupby(
            "Playing-time responsibility"
        )["On-Off"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .round(3)
    )

    print(summary)

    high_n = int(
        (
            eligible["Playing-time responsibility"]
            == "Higher-minute-share"
        ).sum()
    )

    low_n = int(
        (
            eligible["Playing-time responsibility"]
            == "Lower-minute-share"
        ).sum()
    )

    print(
        "\nMissing On-Off values after filtering:",
        eligible["On-Off"].isna().sum(),
    )

    print("\nPreliminary feasibility assessment:")
    print(
        practical_feasibility(
            [high_n, low_n]
        )
    )


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main():

    print(
        "FIFA World Cup 2026 - "
        "Question Feasibility Analysis"
    )

    print_separator()

    datasets = load_datasets()

    print("\nCleaned datasets:\n")

    for name, dataframe in datasets.items():
        print(
            f"{name:<15} "
            f"rows={dataframe.shape[0]:<5} "
            f"columns={dataframe.shape[1]}"
        )

    check_task1_super_sub(datasets)
    check_task2_shooting_efficiency(datasets)
    check_task3_defensive_pressure(datasets)
    check_task4_on_off_impact(datasets)

    print("\n")
    print_separator()

    print(
        "All four question-feasibility checks "
        "completed successfully."
    )

    print_separator()


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# FIFA World Cup 2026 - Raw Data Validation
# ------------------------------------------------------------

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Raw FBref datasets
DATA_FILES = {
    "standard": "fbref_2026_world_cup_player_standard.csv",
    "shooting": "fbref_2026_world_cup_player_shooting.csv",
    "goalkeeping": "fbref_2026_world_cup_player_goalkeeping.csv",
    "miscellaneous": "fbref_2026_world_cup_player_miscellaneous.csv",
    "playing_time": "fbref_2026_world_cup_player_playing_time.csv",
}


def load_raw_datasets():
    """Load the five raw FBref datasets using their two-row headers."""

    datasets = {}

    for name, filename in DATA_FILES.items():
        file_path = RAW_DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Missing dataset: {file_path}")

        df = pd.read_csv(file_path, header=[0, 1])
        datasets[name] = df

        print(f"{name:<15} rows={df.shape[0]:<5} columns={df.shape[1]}")

    return datasets

def check_duplicate_players(datasets):
    """Check whether Player + Squad combinations are duplicated."""

    print("\nDuplicate Player + Squad Check")
    print("=" * 50)

    for name, df in datasets.items():
        player_col = next(
            col for col in df.columns if col[1] == "Player"
        )
        squad_col = next(
            col for col in df.columns if col[1] == "Squad"
        )

        duplicate_mask = df.duplicated(
            subset=[player_col, squad_col],
            keep=False
        )

        duplicate_rows = df.loc[
            duplicate_mask,
            [player_col, squad_col]
        ]

        print(
            f"{name:<15} "
            f"duplicate_rows={len(duplicate_rows):<5} "
            f"unique_players={df[[player_col, squad_col]].drop_duplicates().shape[0]}"
        )

        if not duplicate_rows.empty:
            print(duplicate_rows.to_string(index=False))

def check_key_missing_values(datasets):
    """Check missing values in key identification and participation fields."""

    print("\nKey Missing-Value Check")
    print("=" * 50)

    fields_to_check = ["Player", "Squad", "Pos", "90s"]

    for name, df in datasets.items():
        print(f"\n{name.upper()}")

        for field in fields_to_check:
            matching_columns = [
                col for col in df.columns
                if col[1] == field
            ]

            if matching_columns:
                col = matching_columns[0]
                missing = df[col].isna().sum()
                percentage = (missing / len(df)) * 100

                print(
                    f"{field:<10} "
                    f"missing={missing:<5} "
                    f"({percentage:.2f}%)"
                )
            else:
                print(f"{field:<10} not available")

def check_playing_time_coverage(datasets):
    """Investigate players with missing 90s in the playing-time dataset."""

    print("\nPlaying-Time Coverage Check")
    print("=" * 50)

    playing = datasets["playing_time"]
    standard = datasets["standard"]

    # Locate the required columns.
    playing_player = next(col for col in playing.columns if col[1] == "Player")
    playing_squad = next(col for col in playing.columns if col[1] == "Squad")
    playing_90s = next(col for col in playing.columns if col[1] == "90s")

    standard_player = next(col for col in standard.columns if col[1] == "Player")
    standard_squad = next(col for col in standard.columns if col[1] == "Squad")

    # Player + squad keys for the standard statistics dataset.
    standard_keys = set(
        zip(
            standard[standard_player],
            standard[standard_squad]
        )
    )

    # Players whose 90s value is missing in playing time.
    missing_90s = playing[playing[playing_90s].isna()].copy()

    missing_keys = set(
        zip(
            missing_90s[playing_player],
            missing_90s[playing_squad]
        )
    )

    # Compare coverage between the two datasets.
    all_playing_keys = set(
        zip(
            playing[playing_player],
            playing[playing_squad]
        )
    )

    not_in_standard = all_playing_keys - standard_keys
    missing_90s_in_standard = missing_keys & standard_keys

    print(f"Playing-time players:                 {len(all_playing_keys)}")
    print(f"Standard-stat players:                {len(standard_keys)}")
    print(f"Players missing 90s:                  {len(missing_keys)}")
    print(f"Playing-time players not in standard: {len(not_in_standard)}")
    print(
        "Missing-90s players found in standard: "
        f"{len(missing_90s_in_standard)}"
    )

    if missing_keys == not_in_standard:
        print(
            "Result: Missing 90s records exactly match players "
            "absent from the standard-stat dataset."
        )
    else:
        print(
            "Result: Missing 90s records do NOT exactly match "
            "players absent from the standard-stat dataset."
        )

def check_cross_dataset_coverage(datasets):
    """Check whether player keys align across performance datasets."""

    print("\nCross-Dataset Player Coverage Check")
    print("=" * 50)

    def get_keys(df):
        player_col = next(col for col in df.columns if col[1] == "Player")
        squad_col = next(col for col in df.columns if col[1] == "Squad")

        return set(zip(df[player_col], df[squad_col]))

    standard_keys = get_keys(datasets["standard"])
    shooting_keys = get_keys(datasets["shooting"])
    miscellaneous_keys = get_keys(datasets["miscellaneous"])
    goalkeeping_keys = get_keys(datasets["goalkeeping"])

    print(
        "Standard vs Shooting exact match:       ",
        standard_keys == shooting_keys
    )

    print(
        "Standard vs Miscellaneous exact match:  ",
        standard_keys == miscellaneous_keys
    )

    print(
        "Goalkeepers contained in Standard:      ",
        goalkeeping_keys.issubset(standard_keys)
    )

    print(
        "Standard only vs Shooting:",
        len(standard_keys - shooting_keys)
    )

    print(
        "Shooting only vs Standard:",
        len(shooting_keys - standard_keys)
    )

    print(
        "Standard only vs Miscellaneous:",
        len(standard_keys - miscellaneous_keys)
    )

    print(
        "Miscellaneous only vs Standard:",
        len(miscellaneous_keys - standard_keys)
    )

    print(
        "Goalkeepers missing from Standard:",
        len(goalkeeping_keys - standard_keys)
    )

if __name__ == "__main__":
    print("FIFA World Cup 2026 - Raw Dataset Validation")
    print("=" * 50)

    datasets = load_raw_datasets()

    check_duplicate_players(datasets)
    check_key_missing_values(datasets)
    check_playing_time_coverage(datasets)
    check_cross_dataset_coverage(datasets)

    print("=" * 50)
    print("All five raw datasets loaded successfully.")
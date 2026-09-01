from pathlib import Path

import requests


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# FBref 2026 World Cup standard statistics page
FBREF_URL = "https://fbref.com/en/comps/1/stats/World-Cup-Stats"


def test_fbref_connection():
    """Test whether the FBref 2026 World Cup page can be accessed."""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/140.0 Safari/537.36"
        )
    }

    response = requests.get(
        FBREF_URL,
        headers=headers,
        timeout=30
    )

    print("URL:", FBREF_URL)
    print("HTTP status code:", response.status_code)
    print("Response size:", len(response.content), "bytes")

    if response.status_code == 200:
        print("Connection successful.")
    else:
        print("Connection unsuccessful.")


if __name__ == "__main__":
    test_fbref_connection()
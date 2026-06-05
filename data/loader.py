import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config.settings import DATASET_ROOT

logger = logging.getLogger(__name__)

# Columns that every city dataset must contain.
# If any are absent the city is skipped — not a crash, just a warning.
REQUIRED_COLUMNS: frozenset = frozenset({
    "price_numeric",          # clean float price — used as the recommendation target
    "room_type",              # Entire home/apt, Private room, etc. — used in OHE
    "property_type",          # Apartment, House, etc. — used in OHE
    "accommodates",           # integer — core numeric feature
    "neighbourhood_cleansed", # standardised neighbourhood name — used in OHE
    "has_wifi",               # pre-extracted amenity flag
    "has_kitchen",
    "has_tv",
    "has_ac",
    "has_pool",
    "has_parking",
    "has_washer",
    "has_elevator",
})


def _find_city_csv(city_dir: Path) -> Optional[Path]:
    """Return the CSV whose stem matches the city folder name (case-insensitive).

    listings.csv is always skipped — it is the raw Airbnb dump with a different
    schema and must not be loaded.
    """
    target_stem = city_dir.name.lower()
    for csv_path in city_dir.glob("*.csv"):
        if csv_path.name.lower() == "listings.csv":
            continue
        if csv_path.stem.lower() == target_stem:
            return csv_path
    return None


def _validate(df: pd.DataFrame, city_name: str) -> bool:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.warning(
            "City '%s' skipped — missing required columns: %s",
            city_name, sorted(missing),
        )
        return False
    return True


def _load_csv(csv_path: Path, city_name: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path, low_memory=False, encoding="latin-1")
    except Exception as exc:
        logger.warning(
            "City '%s' skipped — could not parse %s: %s",
            city_name, csv_path.name, exc,
        )
        return None

    if df.empty:
        logger.warning("City '%s' skipped — CSV is empty.", city_name)
        return None

    if not _validate(df, city_name):
        return None

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_cities() -> Dict[str, pd.DataFrame]:
    """Scan DATASET_ROOT and return {city_name: DataFrame} for every valid city.

    City name is the subdirectory name, lowercased (e.g. "amsterdam").

    Adding a new city requires only:
      1. Create a sub-folder under DATASET_ROOT named after the city.
      2. Place a CSV inside it with the same name as the folder.
      3. Restart the application.
    """
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"DATASET_ROOT does not exist: {DATASET_ROOT}\n"
            "Set the DATASET_ROOT environment variable to the folder "
            "that contains one sub-directory per city."
        )

    cities: Dict[str, pd.DataFrame] = {}

    for city_dir in sorted(DATASET_ROOT.iterdir()):
        if not city_dir.is_dir():
            continue

        city_name = city_dir.name.lower()
        csv_path = _find_city_csv(city_dir)

        if csv_path is None:
            logger.warning(
                "Directory '%s' has no matching CSV (expected '%s.csv') — skipped.",
                city_dir.name, city_dir.name,
            )
            continue

        df = _load_csv(csv_path, city_name)
        if df is None:
            continue

        cities[city_name] = df
        logger.info(
            "Loaded '%s' — %d rows from %s",
            city_name, len(df), csv_path.name,
        )

    if not cities:
        raise RuntimeError(
            f"No valid city datasets found under {DATASET_ROOT}. "
            "Each city folder must contain a CSV file with the same name as the folder."
        )

    logger.info("Cities ready: %s", sorted(cities))
    return cities


def load_city(city_name: str) -> pd.DataFrame:
    """Load a single city by name (case-insensitive).

    Useful for testing individual cities or forcing a cache refresh
    without reloading the entire dataset.
    """
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"DATASET_ROOT does not exist: {DATASET_ROOT}"
        )

    name_lower = city_name.lower()

    # Find the directory regardless of how the folder is capitalised on disk.
    city_dir: Optional[Path] = None
    for d in DATASET_ROOT.iterdir():
        if d.is_dir() and d.name.lower() == name_lower:
            city_dir = d
            break

    if city_dir is None:
        available = [d.name for d in DATASET_ROOT.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"No directory found for city '{city_name}' under {DATASET_ROOT}. "
            f"Available: {sorted(available)}"
        )

    csv_path = _find_city_csv(city_dir)
    if csv_path is None:
        raise FileNotFoundError(
            f"No matching CSV in {city_dir} "
            f"(expected a file named '{city_dir.name}.csv')."
        )

    df = _load_csv(csv_path, name_lower)
    if df is None:
        raise ValueError(
            f"City '{city_name}' failed validation — see log output for details."
        )

    return df

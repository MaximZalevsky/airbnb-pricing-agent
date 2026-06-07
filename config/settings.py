import os
from pathlib import Path
from dotenv import load_dotenv

# Inject the OS native certificate store into Python's ssl module.
# This handles SSL interception proxies (common on Windows/corporate networks)
# that use a custom CA stored in the system trust store but not in certifi.
# On Linux/cPanel, truststore is not installed and this is skipped gracefully.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# Absolute path of the project root (one level above this file).
# passenger_wsgi.py calls os.chdir(BASE_DIR) before importing the app,
# so this anchor works correctly under Phusion Passenger on cPanel.
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]          # raises KeyError if absent — fail fast
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Root folder that contains one sub-folder per city.
# Each sub-folder must contain a CSV whose name matches the folder name
# (case-insensitive).  The loader discovers cities dynamically — no hardcoded list.
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", str(BASE_DIR / "data" / "cities")))

# ---------------------------------------------------------------------------
# Pickle cache  (serialised pre-fitted ML models)
# ---------------------------------------------------------------------------

CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# File-backed sessions  (survive Passenger worker restarts)
# ---------------------------------------------------------------------------

SESSION_DIR = BASE_DIR / "sessions"
SESSION_DIR.mkdir(exist_ok=True)

# How long a session file lives before it is considered stale (seconds).
SESSION_LIFETIME_SECONDS = int(os.getenv("SESSION_LIFETIME_SECONDS", str(60 * 60 * 24)))

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------

SECRET_KEY = os.environ["FLASK_SECRET_KEY"]            # raises KeyError if absent — fail fast

# Set FLASK_DEBUG=false in .env on cPanel — tracebacks must never reach users.
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# ---------------------------------------------------------------------------
# K-Means  (adaptive K based on dataset size)
# ---------------------------------------------------------------------------

KMEANS_THRESH_SMALL = int(os.getenv("KMEANS_THRESH_SMALL", "300"))
KMEANS_THRESH_MEDIUM = int(os.getenv("KMEANS_THRESH_MEDIUM", "1000"))

# K values chosen based on row count vs. thresholds above.
KMEANS_K_SMALL = 2   # rows < KMEANS_THRESH_SMALL
KMEANS_K_MEDIUM = 3  # rows < KMEANS_THRESH_MEDIUM
KMEANS_K_LARGE = 4   # rows >= KMEANS_THRESH_MEDIUM

KMEANS_RANDOM_STATE = int(os.getenv("KMEANS_RANDOM_STATE", "42"))

# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

# How many neighbours each of the three metrics (Cosine, Jaccard, KNN) returns
# before the recommender aggregates and deduplicates them.
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "10"))

# How many comparable listings to include in the final explanation shown to the user.
RECOMMENDATION_COMPS = int(os.getenv("RECOMMENDATION_COMPS", "5"))

# ---------------------------------------------------------------------------
# Cross-city fallback
# ---------------------------------------------------------------------------

# Confidence penalty applied when no data exists for the requested city.
# Max possible cross-city confidence: CROSS_CITY_CONFIDENCE_PENALTY (55%).
CROSS_CITY_CONFIDENCE_PENALTY = float(os.getenv("CROSS_CITY_CONFIDENCE_PENALTY", "0.55"))

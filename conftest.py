import os
import sys

# Ensure project root is on sys.path so test imports work regardless of
# which directory pytest is invoked from.
sys.path.insert(0, os.path.dirname(__file__))

# Inject dummy required env vars before any module import triggers settings.py.
# setdefault leaves real values untouched if a .env file is already loaded.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-not-real")

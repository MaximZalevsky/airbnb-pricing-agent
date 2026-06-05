import os
import sys

# Anchor the working directory to the project root BEFORE any import.
# Phusion Passenger does not guarantee the CWD when it starts the process,
# so all relative paths in settings.py (DATASET_ROOT, CACHE_DIR, SESSION_DIR)
# would resolve incorrectly without this line.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Ensure the project root is on sys.path so package imports work.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import create_app

# Phusion Passenger requires the WSGI callable to be named exactly `application`.
application = create_app()

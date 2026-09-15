"""Adsorbent recommender application package.

The package adds the adjacent ML training directory to the import path because
saved model pipelines deliberately retain their stable training-module names.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent
ML_TRAINING_DIR = SCRIPTS_DIR / "ML_model_training"
if str(ML_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(ML_TRAINING_DIR))

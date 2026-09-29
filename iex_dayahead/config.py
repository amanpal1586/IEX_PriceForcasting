"""Paths and runtime configuration.

Everything is relative to the repo root by default; override with environment
variables so the same code runs unchanged on a laptop or a scheduled server.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("IEXFC_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR    = Path(os.environ.get("IEXFC_DATA_DIR",   ROOT / "data"))
MODEL_DIR   = Path(os.environ.get("IEXFC_MODEL_DIR",  ROOT / "models"))
OUT_DIR     = Path(os.environ.get("IEXFC_OUT_DIR",    ROOT / "output"))

HISTORY_CSV = Path(os.environ.get("IEXFC_HISTORY_CSV", DATA_DIR / "iex_history.csv"))
WEATHER_CSV = Path(os.environ.get("IEXFC_WEATHER_CSV", DATA_DIR / "weather_features.csv"))

for _d in (DATA_DIR, MODEL_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# API token is read from the environment - never committed.
IEX_TOKEN = os.environ.get("IEX_TOKEN")

RANK_TOL = 0.05   # a market pick counts as correct within 5% of the true best

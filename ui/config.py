from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
REGISTERED_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

EXTERNAL_SVC_DATA_DIR = EXTERNAL_DATA_DIR / "svc2004"
EXTERNAL_SVC_DATASETS_DIR = { 1: "Task1", 2: "Task2"}

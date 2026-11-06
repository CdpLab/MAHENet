from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
import os
CACHE=Path(os.environ.get("MAHENET_CACHE",REPO/"cache"))

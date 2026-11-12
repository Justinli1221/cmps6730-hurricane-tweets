"""Download the six HumAID hurricane events (train/dev/test) from the authors'
official Hugging Face release (QCRI/HumAID-events, CC BY-NC-SA 4.0)."""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import DATA_DIR, EVENTS, SPLITS

BASE = "https://huggingface.co/datasets/QCRI/HumAID-events/resolve/main"

for event in EVENTS:
    (DATA_DIR / event).mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        dest = DATA_DIR / event / f"{split}.json"
        if not dest.exists():
            urllib.request.urlretrieve(f"{BASE}/{event}/{split}.json", dest)
        print(dest)

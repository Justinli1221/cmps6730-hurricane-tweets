"""Load the six HumAID hurricane events (train/dev/test splits from QCRI/HumAID-events)."""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "humaid"
EVENTS = [
    "hurricane_matthew_2016",
    "hurricane_harvey_2017",
    "hurricane_irma_2017",
    "hurricane_maria_2017",
    "hurricane_florence_2018",
    "hurricane_dorian_2019",
]
SPLITS = ["train", "dev", "test"]


def load_all():
    rows = []
    for event in EVENTS:
        for split in SPLITS:
            with open(DATA_DIR / event / f"{split}.json", encoding="utf8") as f:
                for r in json.load(f):
                    rows.append({"event": event, "split": split,
                                 "text": r["tweet_text"], "label": r["class_label"]})
    return pd.DataFrame(rows)

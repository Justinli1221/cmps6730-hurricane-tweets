import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_all

df = load_all()
print(len(df))
print(df.pivot_table(index="event", columns="split", values="text", aggfunc="count", margins=True))
print(df.label.value_counts())
print(df.groupby("event").label.value_counts(normalize=True).unstack().round(3).T)
print("dup texts:", df.text.duplicated().sum())

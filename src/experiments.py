"""Preliminary experiments for the milestone report.

Compares a majority-class baseline with TF-IDF + logistic regression under
  - in-event training (pooled over all six hurricanes, size-matched), and
  - leave-one-hurricane-out (LOHO) training,
with and without masking of storm names and locations. Also produces a
learning curve. Writes LaTeX tables, a figure and raw numbers to results/.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import EVENTS, load_all

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)
SEED = 42

SHORT = {
    "hurricane_matthew_2016": "Matthew", "hurricane_harvey_2017": "Harvey",
    "hurricane_irma_2017": "Irma", "hurricane_maria_2017": "Maria",
    "hurricane_florence_2018": "Florence", "hurricane_dorian_2019": "Dorian",
}

# --- preprocessing ---------------------------------------------------------
URL = re.compile(r"https?://\S+|www\.\S+")
USER = re.compile(r"@\w+")
STORMS = re.compile(r"(matthew|harvey|irma|maria|florence|dorian)")  # also inside hashtags
PLACES = [
    "puerto rico", "san juan", "virgin islands", "grand bahama", "corpus christi",
    "port arthur", "myrtle beach", "new bern", "north carolina", "south carolina",
    "outer banks", "florida keys", "texas", "houston", "beaumont", "rockport",
    "florida", "miami", "tampa", "naples", "jacksonville", "orlando", "carolina",
    "wilmington", "charleston", "savannah", "georgia", "louisiana", "bahamas",
    "abaco", "haiti", "cuba", "caribbean", "barbuda", "antigua", "dominica",
    "usvi", "bvi", "tx", "fl", "pr", "nc", "sc",
]
PLACE_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in PLACES) + r")\b")


def clean(text, mask=False):
    t = text.lower()
    t = URL.sub(" httpurl ", t)
    t = USER.sub(" @user ", t)
    if mask:
        t = STORMS.sub(" stormname ", t)
        t = PLACE_RE.sub(" placename ", t)
    return t


# --- model -----------------------------------------------------------------
class TfidfLR:
    """Word (1-2) + character (2-5) TF-IDF features with logistic regression."""

    def __init__(self, C=1.0):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                    sublinear_tf=True)
        self.clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")

    def fit(self, texts, y):
        X = hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        self.clf.fit(X, y)
        return self

    def predict(self, texts):
        X = hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
        return self.clf.predict(X)


def macro_f1(y, p):
    return f1_score(y, p, average="macro")


def fit_eval(tr, dv, te, col, C):
    """Train TF-IDF + LR on tr[col]; return dev macro-F1, test macro-F1 and test per-class report."""
    m = TfidfLR(C).fit(tr[col].tolist(), tr.label)
    pd_ = m.predict(dv[col].tolist()) if dv is not None else None
    pt = m.predict(te[col].tolist())
    dev_f1 = macro_f1(dv.label, pd_) if dv is not None else None
    return dev_f1, macro_f1(te.label, pt), classification_report(te.label, pt, output_dict=True)


CS = [0.5, 1, 2, 4]  # regularization grid, chosen per setting on that setting's dev set


def main():
    df = load_all()
    df["clean"] = df.text.map(clean)
    df["masked"] = df.text.map(lambda t: clean(t, mask=True))
    rng = np.random.default_rng(SEED)
    train, dev, test = (df[df.split == s] for s in ("train", "dev", "test"))
    results = {}
    par = Parallel(n_jobs=3, verbose=5)  # memory-bound on a 16 GB laptop

    # 1. For every setting, fit one model per C, pick C on that setting's dev set and
    #    report the test score of the chosen model. LOHO folds tune only on the dev
    #    splits of the other five hurricanes, so the held-out storm is never seen.
    jobs, keys = [], []
    for C in CS:
        jobs.append(delayed(fit_eval)(train, dev, test, "clean", C))
        keys.append(("pooled", None, C))
    for ev in EVENTS:
        tgt = test[test.event == ev]
        other_tr, other_dv = train[train.event != ev], dev[dev.event != ev]
        # in-event: same number of training tweets as LOHO, sampled from all six events
        insample = train.iloc[rng.choice(len(train), size=len(other_tr), replace=False)]
        for name, tr, dv, col in [("in_event", insample, dev, "clean"),
                                  ("in_event_mask", insample, dev, "masked"),
                                  ("loho", other_tr, other_dv, "clean"),
                                  ("loho_mask", other_tr, other_dv, "masked")]:
            for C in CS:
                jobs.append(delayed(fit_eval)(tr, dv, tgt, col, C))
                keys.append((name, ev, C))
    out = par(jobs)

    best = {}  # (setting, event) -> (dev_f1, C, test_f1, report)
    for (name, ev, C), (dev_f1, test_f1, report) in zip(keys, out):
        if (name, ev) not in best or dev_f1 > best[(name, ev)][0]:
            best[(name, ev)] = (dev_f1, C, test_f1, report)
    _, C, results["pooled_macro_f1"], results["pooled_report"] = best[("pooled", None)]
    results["C"] = C
    rows = {ev: {} for ev in EVENTS}
    for (name, ev), (_, c, test_f1, report) in best.items():
        if name == "pooled":
            continue
        rows[ev][name] = test_f1
        rows[ev][name + "_C"] = c
        if name in ("in_event", "loho"):
            rows[ev][name + "_report"] = report

    # 2. learning curve on the pooled test set (3 random subsamples per size), pooled C
    sizes = [s for s in [100, 250, 500, 1000, 2500, 5000, 10000] if s < len(train)] + [len(train)]
    jobs, ns = [], []
    for s in sizes:
        for seed in range(3 if s < len(train) else 1):
            jobs.append(delayed(fit_eval)(train.sample(n=s, random_state=seed), None, test, "clean", C))
            ns.append(s)
    curve_scores = {s: [] for s in sizes}
    for s, (_, test_f1, _) in zip(ns, par(jobs)):
        curve_scores[s].append(test_f1)
    for ev in EVENTS:
        tgt = test[test.event == ev]
        other = train[train.event != ev]
        maj = DummyClassifier(strategy="most_frequent").fit(other.text, other.label)
        rows[ev]["majority"] = macro_f1(tgt.label, maj.predict(tgt.text))
        print(SHORT[ev], {k: round(v, 3) for k, v in rows[ev].items() if not k.endswith("_report")})
    results["per_event"] = rows
    maj = DummyClassifier(strategy="most_frequent").fit(train.text, train.label)
    results["pooled_majority_macro_f1"] = macro_f1(test.label, maj.predict(test.text))
    results["learning_curve"] = [{"n": s, "mean": float(np.mean(v)), "std": float(np.std(v))}
                                 for s, v in curve_scores.items()]
    print("pooled", results["pooled_macro_f1"], "curve", results["learning_curve"])

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    write_outputs(results)


def write_outputs(results):
    rows = results["per_event"]
    cols = [("majority", "Majority"), ("in_event", "In-event"), ("loho", "LOHO"),
            ("loho_mask", "LOHO+mask")]
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             "Test event & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for ev in EVENTS:
        lines.append(SHORT[ev] + " & " + " & ".join(f"{rows[ev][k]:.3f}" for k, _ in cols) + r" \\")
    lines.append(r"\midrule")
    lines.append("Average & " + " & ".join(
        f"{np.mean([rows[e][k] for e in EVENTS]):.3f}" for k, _ in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_main.tex").write_text("\n".join(lines) + "\n")

    # per-class F1 for the paired settings of Table 1: average over the held-out events,
    # counting each event only if the class occurs in its test split
    def per_class(setting, lab):
        reps = [rows[e][setting + "_report"].get(lab) for e in EVENTS]
        return np.mean([r["f1-score"] for r in reps if r and r["support"] > 0])

    labels = [k for k in results["pooled_report"] if k not in ("accuracy", "macro avg", "weighted avg")]
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Class & In-event & LOHO \\", r"\midrule"]
    for lab in sorted(labels, key=lambda l: -per_class("in_event", l)):
        name = lab.replace("_", " ").replace(" or ", "/").replace(" and ", " \\& ")
        lines.append(f"{name} & {per_class('in_event', lab):.2f} & {per_class('loho', lab):.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_per_class.tex").write_text("\n".join(lines) + "\n")

    curve = results["learning_curve"]
    fig, ax = plt.subplots(figsize=(4.2, 2.1))
    n = [c["n"] for c in curve]
    m = np.array([c["mean"] for c in curve])
    s = np.array([c["std"] for c in curve])
    ax.plot(n, m, marker="o", color="#2a6f97")
    ax.fill_between(n, m - s, m + s, color="#2a6f97", alpha=0.2)
    ax.axhline(results["pooled_majority_macro_f1"], ls="--", color="gray", lw=1)
    ax.text(n[0], results["pooled_majority_macro_f1"] + 0.015, "majority class", color="gray", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Number of training tweets (log scale)")
    ax.set_ylabel("Macro-F1 (pooled test)")
    ax.set_ylim(0, 0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.pdf")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--tables":
        write_outputs(json.loads((OUT / "results.json").read_text()))
    else:
        main()

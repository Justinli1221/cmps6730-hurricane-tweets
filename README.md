# Will It Work on the Next Storm? Classifying Hurricane Tweets by Humanitarian Need

Final project for CMPS 4730/6730: Natural Language Processing (Tulane University, Fall 2026).

During hurricanes, responders need to find actionable tweets (rescue requests, damage reports, injured people) among a flood of news and sympathy messages. This project classifies tweets from six hurricanes in the [HumAID](https://arxiv.org/abs/2104.03090) dataset into nine humanitarian categories, and asks whether a model still works on a hurricane it has never seen.

The milestone report is in [report/report.pdf](report/report.pdf).

## Approach

- **Data**: six Atlantic hurricanes from HumAID (Matthew, Harvey, Irma, Maria, Florence, Dorian), 41,320 labeled tweets, using the official train/dev/test splits.
- **Model so far**: word + character TF-IDF features with logistic regression (scikit-learn). BERT and BERTweet are planned.
- **Settings**: in-event (the test hurricane is also in training) vs. leave-one-hurricane-out (LOHO), with and without masking storm names and places.
- **Metric**: macro-F1.

## Preliminary results (macro-F1, average over six test hurricanes)

| Majority | In-event | LOHO | LOHO + mask |
|---|---|---|---|
| 0.039 | 0.699 | 0.671 | 0.685 |

## Reproducing

```bash
pip install scikit-learn pandas numpy scipy matplotlib joblib
python src/download_data.py   # only needed if data/ is missing (about 11 MB)
python src/experiments.py     # writes results/ (about 45 minutes on a laptop)
cd report && bash build.sh    # compiles report/report.pdf
```

`results/results_v1.json` holds an earlier run that selected C on the pooled dev set; the report cites its Florence LOHO score (C = 0.5) as a tuning-sensitivity check.

## Data license

`data/humaid/` contains the six hurricane events from HumAID, copied unchanged from the authors' [official release](https://huggingface.co/datasets/QCRI/HumAID-events). HumAID is created by Alam, Qazi, Imran and Ofli (ICWSM 2021) and licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); that license applies to the files in `data/`. `src/download_data.py` re-downloads them if needed.

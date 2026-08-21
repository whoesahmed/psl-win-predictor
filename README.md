# PSL Win Predictor

Predicts Pakistan Super League match winners using team form, head-to-head record, and venue history — built as a practical application of Python data analysis (NumPy, Pandas, Matplotlib, Seaborn) and machine learning (scikit-learn), wrapped in an interactive Streamlit app.

## Overview

Given two teams and a venue, the model predicts a likely winner and a confidence score, based entirely on historical match patterns — no external commentary, no assumptions beyond the data.

## Dataset

146 PSL matches, 2016–2020 (source: Kaggle). 7 matches with no decisive result (no result / tied / abandoned) are excluded, leaving 139 usable matches.

**Honest limitations, stated upfront:**
- This dataset does not include toss information.
- It covers 2016–2020 only — it does not reflect current-season rosters, form, or teams.
- No player injury, squad change, or weather data is available or used.

Predictions should be read as a directional estimate from limited historical data, not a confident forecast of upcoming matches.

## Features

Three features are engineered from match history, each using only data available *before* the match in question (no data leakage):

- **Recent form** — each team's win rate over their last 5 matches
- **Head-to-head win rate** — historical performance between the two specific teams
- **Venue win rate** — each team's win rate at the selected ground

## Model

Two classifiers were trained and compared:

| Model | Accuracy |
|---|---|
| Logistic Regression | **60.7%** |
| Decision Tree (max depth 4) | 32.1% |

Logistic Regression was selected. With only 139 matches, the Decision Tree overfits the training data and generalizes poorly — the simpler model wins here, which is itself a useful finding: more model complexity isn't automatically better on a small dataset.

Test set size is ~28 matches, so treat the accuracy figure as directional rather than statistically precise.

## Project Structure

```
psl-win-predictor/
├── README.md
├── requirements.txt
├── data/
│   └── PSL_Match_Results.csv       # not tracked in git — see Setup
├── notebooks/
│   └── eda_and_model.ipynb         # EDA + feature sanity checks
├── src/
│   ├── preprocess.py               # loads data, engineers features
│   └── train_model.py              # trains, compares, and saves the model
└── app/
    └── streamlit_app.py            # interactive predictor
```

## Setup

```bash
git clone https://github.com/whoesahmed/psl-win-predictor.git
cd psl-win-predictor
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it at `data/PSL_Match_Results.csv` (not included in this repo — download it yourself).

## Usage

Train the model:
```bash
cd src
python train_model.py
```

Run the app:
```bash
cd ../app
streamlit run streamlit_app.py
```

Explore the EDA notebook:
```bash
cd ../notebooks
jupyter notebook eda_and_model.ipynb
```

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Streamlit

## Future Improvements

- Merge in real toss data from a licensed source (e.g. Cricsheet.org, CC BY 4.0) rather than leaving it out entirely.
- Extend the dataset to more recent seasons.
- Add player-level features (current squad strength) if reliable data becomes available.

---
Feedback and pull requests welcome.

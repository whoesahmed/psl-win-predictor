"""
Preprocessing and feature engineering for PSL match outcome prediction.

Matches the ACTUAL columns in PSL_Match_Results.csv:
    Team 1, Ground, Margin, Year, Quarter, Month, Day, Scorecard, Team 2, Winner

Note: this dataset has no toss information, so toss-based features are
not used here (an earlier version of this file assumed a toss_winner /
toss_decision column that doesn't exist in this dataset -- removed).
"""

import pandas as pd

COLUMN_MAP = {
    "team1": "Team 1",
    "team2": "Team 2",
    "venue": "Ground",
    "winner": "Winner",
}

UNDECIDED_RESULTS = {"no result", "tied", "abandoned"}


def load_raw(path):
    df = pd.read_csv(path)
    df = df.rename(columns={v: k for k, v in COLUMN_MAP.items()})

    # Build a real date from Year/Month/Day so we can sort chronologically
    df["date"] = pd.to_datetime(
        df["Year"].astype(str) + " " + df["Month"] + " " + df["Day"].astype(str),
        format="%Y %B %d",
        errors="coerce",
    )

    # Drop matches with no decisive winner -- can't be a classification target
    before = len(df)
    df = df[~df["winner"].isin(UNDECIDED_RESULTS)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} matches with no decisive result (no result / tied / abandoned).")

    df = df.dropna(subset=["team1", "team2", "winner"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_recent_form(df, window=5):
    form_lookup = {}
    team1_form, team2_form = [], []

    for _, row in df.iterrows():
        t1, t2, winner = row["team1"], row["team2"], row["winner"]

        h1 = form_lookup.get(t1, [])
        h2 = form_lookup.get(t2, [])

        team1_form.append(sum(h1[-window:]) / len(h1[-window:]) if h1 else 0.5)
        team2_form.append(sum(h2[-window:]) / len(h2[-window:]) if h2 else 0.5)

        form_lookup.setdefault(t1, []).append(1 if winner == t1 else 0)
        form_lookup.setdefault(t2, []).append(1 if winner == t2 else 0)

    df["team1_form"] = team1_form
    df["team2_form"] = team2_form
    return df


def add_head_to_head(df):
    h2h_lookup = {}
    h2h_rate = []

    for _, row in df.iterrows():
        t1, t2, winner = row["team1"], row["team2"], row["winner"]
        key = tuple(sorted([t1, t2]))
        wins, total = h2h_lookup.get(key, [0, 0])

        h2h_rate.append(wins / total if total > 0 else 0.5)

        if winner == key[0]:
            wins += 1
        total += 1
        h2h_lookup[key] = [wins, total]

    df["h2h_team1_win_rate_raw"] = h2h_rate  # win rate for sorted_pair[0]
    return df


def add_venue_advantage(df):
    venue_lookup = {}
    team1_venue_rate, team2_venue_rate = [], []

    for _, row in df.iterrows():
        t1, t2, venue, winner = row["team1"], row["team2"], row["venue"], row["winner"]

        w1, tot1 = venue_lookup.get((t1, venue), [0, 0])
        w2, tot2 = venue_lookup.get((t2, venue), [0, 0])

        team1_venue_rate.append(w1 / tot1 if tot1 > 0 else 0.5)
        team2_venue_rate.append(w2 / tot2 if tot2 > 0 else 0.5)

        venue_lookup[(t1, venue)] = [w1 + (1 if winner == t1 else 0), tot1 + 1]
        venue_lookup[(t2, venue)] = [w2 + (1 if winner == t2 else 0), tot2 + 1]

    df["team1_venue_win_rate"] = team1_venue_rate
    df["team2_venue_win_rate"] = team2_venue_rate
    return df


def build_features(path):
    df = load_raw(path)
    df = add_recent_form(df)
    df = add_head_to_head(df)
    df = add_venue_advantage(df)

    df["target"] = (df["winner"] == df["team1"]).astype(int)

    # h2h_team1_win_rate_raw is relative to alphabetically-sorted team names;
    # flip it so it's always "team1's win rate against team2" for THIS row
    def resolve_h2h(row):
        sorted_pair = sorted([row["team1"], row["team2"]])
        return row["h2h_team1_win_rate_raw"] if sorted_pair[0] == row["team1"] else 1 - row["h2h_team1_win_rate_raw"]

    df["h2h_team1_win_rate"] = df.apply(resolve_h2h, axis=1)

    feature_cols = [
        "team1_form", "team2_form",
        "h2h_team1_win_rate",
        "team1_venue_win_rate", "team2_venue_win_rate",
    ]

    return df, feature_cols


if __name__ == "__main__":
    df, feature_cols = build_features("../data/PSL_Match_Results.csv")
    print(df[["team1", "team2", "winner"] + feature_cols + ["target"]].tail(10))
    print(f"\nTotal usable matches: {len(df)}")
    print(f"Feature columns: {feature_cols}")
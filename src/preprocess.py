"""
Preprocessing and feature engineering for PSL match outcome prediction.
 
Expected raw columns (verify against your actual CSV and adjust below if needed):
    date, team1, team2, venue, toss_winner, toss_decision, winner
 
If your CSV uses different column names, only the COLUMN_MAP at the top
needs to change -- nothing else in this file should need editing.
"""
 
import pandas as pd
 
# Map your actual CSV column names to the names this script expects.
# Edit the RIGHT-hand side only if your CSV headers differ.
COLUMN_MAP = {
    "date": "date",
    "team1": "team1",
    "team2": "team2",
    "venue": "venue",
    "toss_winner": "toss_winner",
    "toss_decision": "toss_decision",
    "winner": "winner",
}
 
 
def load_raw(path):
    df = pd.read_csv(path)
    df = df.rename(columns={v: k for k, v in COLUMN_MAP.items()})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["team1", "team2", "winner"])
    df = df.sort_values("date").reset_index(drop=True)
    return df
 
 
def add_recent_form(df, window=5):
    """
    For each match, compute each team's win rate over their last `window`
    matches BEFORE the current one. Uses only past data -- no leakage.
    """
    form_lookup = {}  # team -> list of recent results (1 win, 0 loss)
 
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
    """
    For each match, compute team1's historical win rate specifically
    against team2, based only on matches before the current one.
    """
    h2h_lookup = {}  # (team_a, team_b) -> [wins_for_a, total]
    h2h_rate = []
 
    for _, row in df.iterrows():
        t1, t2, winner = row["team1"], row["team2"], row["winner"]
        key = tuple(sorted([t1, t2]))
        wins, total = h2h_lookup.get(key, [0, 0])
 
        h2h_rate.append(wins / total if total > 0 else 0.5)
 
        if winner == t1:
            wins += 1 if key[0] == t1 else 0
            wins += 0 if key[0] == t1 else 1
        total += 1
        h2h_lookup[key] = [wins, total]
 
    df["h2h_team1_win_rate"] = h2h_rate
    return df
 
 
def add_venue_advantage(df):
    """
    Win rate at a given venue for each team, based only on prior matches.
    """
    venue_lookup = {}  # (team, venue) -> [wins, total]
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
 
    # Target: 1 if team1 won, 0 if team2 won
    df["target"] = (df["winner"] == df["team1"]).astype(int)
 
    # Toss features
    df["toss_won_by_team1"] = (df["toss_winner"] == df["team1"]).astype(int)
    df["toss_decision_bat"] = (df["toss_decision"].str.lower() == "bat").astype(int)
 
    feature_cols = [
        "team1_form", "team2_form",
        "h2h_team1_win_rate",
        "team1_venue_win_rate", "team2_venue_win_rate",
        "toss_won_by_team1", "toss_decision_bat",
    ]
 
    return df, feature_cols
 
 
if __name__ == "__main__":
    df, feature_cols = build_features("../data/psl_matches.csv")
    print(df[["team1", "team2", "winner"] + feature_cols + ["target"]].tail(10))
    print(f"\nTotal matches: {len(df)}")
    print(f"Feature columns: {feature_cols}")
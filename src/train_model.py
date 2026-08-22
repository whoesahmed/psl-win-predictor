"""
Trains a match-outcome classifier on the real PSL_Match_Results.csv
structure (2016-2020, no toss data) and saves the model + a lookup
snapshot for the Streamlit app.
"""
import json
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from preprocess import build_features


def build_latest_stats(df):
    latest_form = {}
    latest_venue = {}
    latest_h2h = {}

    for _, row in df.iterrows():
        latest_form[row["team1"]] = row["team1_form"]
        latest_form[row["team2"]] = row["team2_form"]
        latest_venue[(row["team1"], row["venue"])] = row["team1_venue_win_rate"]
        latest_venue[(row["team2"], row["venue"])] = row["team2_venue_win_rate"]
        key = tuple(sorted([row["team1"], row["team2"]]))
        latest_h2h[key] = row["h2h_team1_win_rate"] if key[0] == row["team1"] else 1 - row["h2h_team1_win_rate"]

    return latest_form, latest_venue, latest_h2h


def main():
    df, feature_cols = build_features("../data/PSL_Match_Results.csv")

    X = df[feature_cols]
    y = df["target"]

    print(f"Training on {len(df)} matches, {len(feature_cols)} features.")
    print(f"Test set size will be ~{int(len(df) * 0.2)} matches — small, so treat "
          f"accuracy as directional, not a precise figure.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = (model, acc)
        print(f"\n{name} — accuracy: {acc:.3f}")
        print(classification_report(y_test, preds, target_names=["Team2 wins", "Team1 wins"]))

    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc = results[best_name]
    print(f"\nBest model: {best_name} ({best_acc:.3f} accuracy) — saving this one.")

    with open("model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "feature_cols": feature_cols, "model_name": best_name}, f)

    latest_form, latest_venue, latest_h2h = build_latest_stats(df)

    teams = sorted(set(df["team1"]) | set(df["team2"]))
    venues = sorted(df["venue"].dropna().unique().tolist())

    lookup = {
        "teams": teams,
        "venues": venues,
        "latest_form": latest_form,
        "latest_venue": {f"{t}||{v}": rate for (t, v), rate in latest_venue.items()},
        "latest_h2h": {f"{a}||{b}": rate for (a, b), rate in latest_h2h.items()},
        "accuracy": best_acc,
        "model_name": best_name,
        "num_matches": len(df),
    }

    with open("app_lookup.json", "w") as f:
        json.dump(lookup, f, indent=2)

    print("\nSaved model.pkl and app_lookup.json")


if __name__ == "__main__":
    main()
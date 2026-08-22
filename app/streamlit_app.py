import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="PSL Win Predictor", page_icon="🏏", layout="centered")

# ---------- Real PSL franchise colors ----------
TEAM_STYLE = {
    "United":    {"color": "#E4181C", "accent": "#8C0F12", "full": "Islamabad United"},
    "Kings":     {"color": "#00AEEF", "accent": "#00688F", "full": "Karachi Kings"},
    "Qalandars": {"color": "#1FA24A", "accent": "#0F5C29", "full": "Lahore Qalandars"},
    "Sultans":   {"color": "#C8102E", "accent": "#7A0A1C", "full": "Multan Sultans"},
    "Zalmi":     {"color": "#FDB913", "accent": "#A87700", "full": "Peshawar Zalmi"},
    "Gladiators":{"color": "#6E1E2C", "accent": "#3E0F19", "full": "Quetta Gladiators"},
}
DEFAULT_STYLE = {"color": "#4fd1c5", "accent": "#2c9c92", "full": "Unknown"}


def style_of(team):
    return TEAM_STYLE.get(team, DEFAULT_STYLE)


CUSTOM_CSS = """
<style>
    .stApp {
        font-family: 'Segoe UI', sans-serif;
        background: radial-gradient(circle at top, #0f1b2d 0%, #060a12 70%);
    }
    .match-banner {
        text-align: center;
        padding: 1.4rem 0 0.6rem 0;
    }
    .match-banner h1 {
        font-size: 2.3rem;
        margin-bottom: 0.1rem;
        letter-spacing: 0.5px;
    }
    .match-banner p {
        color: #8b93a7;
        font-size: 0.95rem;
    }
    .vs-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin: 1.2rem 0 0.4rem 0;
    }
    .vs-badge {
        font-size: 1.6rem;
        font-weight: 800;
        color: #5b6478;
        padding: 0 0.4rem;
    }
    .team-chip {
        flex: 1;
        text-align: center;
        padding: 0.9rem 0.5rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.05rem;
        color: white;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }
    .predict-btn button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.7rem 0;
        background: linear-gradient(135deg, #2c9c92, #1a6b64) !important;
        border: none !important;
        letter-spacing: 0.5px;
    }
    .scoreboard {
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-top: 1.4rem;
        background: linear-gradient(135deg, #12233d 0%, #0a1526 100%);
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }
    .scoreboard .label {
        color: #8b93a7;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.3rem;
    }
    .scoreboard .winner-name {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0.2rem 0 0.6rem 0;
    }
    .confidence-track {
        width: 100%;
        height: 10px;
        border-radius: 6px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
        margin-top: 0.6rem;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 6px;
    }
    .prob-row {
        display: flex;
        justify-content: space-between;
        margin-top: 1.4rem;
        gap: 1rem;
    }
    .prob-col {
        flex: 1;
        text-align: center;
    }
    .prob-name {
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }
    .prob-bar-track {
        width: 100%;
        height: 130px;
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        display: flex;
        align-items: flex-end;
        overflow: hidden;
    }
    .prob-bar-fill {
        width: 100%;
        border-radius: 8px 8px 0 0;
        transition: height 0.4s ease;
    }
    .prob-pct {
        margin-top: 0.4rem;
        font-size: 0.9rem;
        color: #c4cad9;
    }
    .footer-note {
        color: #5b6478;
        font-size: 0.78rem;
        text-align: center;
        margin-top: 2.6rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Original bat-and-ball emblem (not the official PSL logo -- that's a
# registered trademark and shouldn't be used in a personal project).
EMBLEM_SVG = """
<svg width="34" height="34" viewBox="0 0 34 34" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;margin-right:8px;">
  <circle cx="17" cy="17" r="16" fill="none" stroke="#2c9c92" stroke-width="1.5"/>
  <line x1="9" y1="25" x2="22" y2="10" stroke="#c4a876" stroke-width="3.2" stroke-linecap="round"/>
  <path d="M20 8 L26 6 L28 12 L22 14 Z" fill="#8b5e2b"/>
  <circle cx="24" cy="24" r="4" fill="#c8102e"/>
  <path d="M21.5 24 Q24 21.5 26.5 24 Q24 26.5 21.5 24" fill="none" stroke="#8b0f1c" stroke-width="0.6"/>
</svg>
"""

st.markdown(
    f"""
    <div class="match-banner">
        <h1>{EMBLEM_SVG}PSL Win Predictor</h1>
        <p>Pick your matchup and venue — get a data-driven prediction, PSL style.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Load model + lookup ----------
BASE = Path(__file__).parent.parent / "src"


@st.cache_resource
def load_model():
    with open(BASE / "model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_lookup():
    with open(BASE / "app_lookup.json") as f:
        return json.load(f)


try:
    model_bundle = load_model()
    lookup = load_lookup()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train_model.py` inside `src/` first "
        "to generate `model.pkl` and `app_lookup.json`."
    )
    st.stop()

model = model_bundle["model"]
feature_cols = model_bundle["feature_cols"]
teams = lookup["teams"]
venues = lookup["venues"]

# ---------- Team selection ----------
col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", teams, index=0)
with col2:
    team2_options = [t for t in teams if t != team1]
    team2 = st.selectbox("Team 2", team2_options, index=0)

s1, s2 = style_of(team1), style_of(team2)

st.markdown(
    f"""
    <div class="vs-row">
        <div class="team-chip" style="background:{s1['color']};">{s1['full']}</div>
        <div class="vs-badge">VS</div>
        <div class="team-chip" style="background:{s2['color']};">{s2['full']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

venue = st.selectbox("Venue", venues)

st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
go = st.button("🏆 Predict Winner")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Prediction ----------
if go:
    form1 = lookup["latest_form"].get(team1, 0.5)
    form2 = lookup["latest_form"].get(team2, 0.5)

    venue1 = lookup["latest_venue"].get(f"{team1}||{venue}", 0.5)
    venue2 = lookup["latest_venue"].get(f"{team2}||{venue}", 0.5)

    h2h_key = "||".join(sorted([team1, team2]))
    h2h_raw = lookup["latest_h2h"].get(h2h_key, 0.5)
    sorted_pair = sorted([team1, team2])
    h2h_team1 = h2h_raw if sorted_pair[0] == team1 else (1 - h2h_raw)

    X_input = pd.DataFrame([{
        "team1_form": form1,
        "team2_form": form2,
        "h2h_team1_win_rate": h2h_team1,
        "team1_venue_win_rate": venue1,
        "team2_venue_win_rate": venue2,
    }])[feature_cols]

    proba = model.predict_proba(X_input)[0]
    team1_prob, team2_prob = proba[1], proba[0]

    predicted_winner = team1 if team1_prob >= team2_prob else team2
    winner_style = style_of(predicted_winner)
    win_prob = max(team1_prob, team2_prob)

    st.markdown(
        f"""
        <div class="scoreboard">
            <div class="label">Predicted Winner</div>
            <div class="winner-name" style="color:{winner_style['color']};">
                🏆 {winner_style['full']}
            </div>
            <div class="label">Confidence — {win_prob * 100:.1f}%</div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width:{win_prob*100:.1f}%; background:{winner_style['color']};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bar1_h = max(int(team1_prob * 130), 6)
    bar2_h = max(int(team2_prob * 130), 6)

    st.markdown(
        f"""
        <div class="prob-row">
            <div class="prob-col">
                <div class="prob-name" style="color:{s1['color']};">{s1['full']}</div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="height:{bar1_h}px; background:{s1['color']};"></div>
                </div>
                <div class="prob-pct">{team1_prob*100:.1f}%</div>
            </div>
            <div class="prob-col">
                <div class="prob-name" style="color:{s2['color']};">{s2['full']}</div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="height:{bar2_h}px; background:{s2['color']};"></div>
                </div>
                <div class="prob-pct">{team2_prob*100:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("About this model")
    st.write(f"**Model:** {lookup['model_name'].replace('_', ' ').title()}")
    st.write(f"**Test accuracy:** {lookup['accuracy'] * 100:.1f}%")
    st.write(f"**Trained on:** {lookup.get('num_matches', '—')} matches (2016–2020)")
    st.markdown("---")
    st.caption(
        "Predictions are based on historical team form, head-to-head record, and "
        "venue win rate only. This dataset does not include toss information, "
        "player injuries, current squad changes, or seasons after 2020 — treat "
        "predictions as a directional estimate from limited historical data, "
        "not a confident forecast of current-season matches."
    )

st.markdown(
    '<div class="footer-note">Built with Python, scikit-learn & Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
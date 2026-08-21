import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="PSL Win Predictor", page_icon="🏏", layout="centered")

CUSTOM_CSS = """
<style>
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .title-banner { text-align: center; padding: 1.2rem 0 0.4rem 0; }
    .title-banner h1 { font-size: 2.1rem; margin-bottom: 0.2rem; }
    .title-banner p { color: #9aa0a6; font-size: 0.95rem; }
    .result-card {
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.2rem;
        background: linear-gradient(135deg, #1b2a4a 0%, #142033 100%);
        border: 1px solid #2c3e60;
    }
    .result-team { font-size: 1.5rem; font-weight: 700; color: #4fd1c5; }
    .footer-note { color: #7a7f87; font-size: 0.8rem; text-align: center; margin-top: 2.5rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="title-banner">
        <h1>🏏 PSL Win Predictor</h1>
        <p>Pick two teams and a venue — get a data-driven win prediction based on historical form.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", teams, index=0)
with col2:
    team2_options = [t for t in teams if t != team1]
    team2 = st.selectbox("Team 2", team2_options, index=0)

venue = st.selectbox("Venue", venues)

go = st.button("Predict Winner", use_container_width=True)

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
    win_prob = max(team1_prob, team2_prob)

    st.markdown(
        f"""
        <div class="result-card">
            <div>Predicted winner</div>
            <div class="result-team">🏆 {predicted_winner}</div>
            <div>Confidence: {win_prob * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    prob_df = pd.DataFrame({
        "Team": [team1, team2],
        "Win Probability": [team1_prob, team2_prob],
    }).set_index("Team")
    st.bar_chart(prob_df)

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
    '<div class="footer-note">Built with Python, scikit-learn & Streamlit — '
    'part of an ongoing AI/ML learning path.</div>',
    unsafe_allow_html=True,
)
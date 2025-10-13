import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Load trained Logistic Regression model ---
with open('goal_prediction_logreg.pkl', 'rb') as f:
    logreg = pickle.load(f)

# --- Feature Columns ---
numeric_features = [
    'avg_xg_rolling_5',
    'shots_total_rolling_5',
    'minutes_played_rolling_5',
    'carries_total_rolling_5',
    'dribbles_total_rolling_5',
    'ball_receipts_final_third_rolling_5'
]

one_hot_features = [
    'home_away_home',
    'position_name_Center Back',
    'position_name_Center Defensive Midfield',
    'position_name_Center Forward',
    'position_name_Goalkeeper',
    'position_name_Left Attacking Midfield',
    'position_name_Left Back',
    'position_name_Left Center Back',
    'position_name_Left Center Forward',
    'position_name_Left Center Midfield',
    'position_name_Left Defensive Midfield',
    'position_name_Left Midfield',
    'position_name_Left Wing',
    'position_name_Left Wing Back',
    'position_name_Right Back',
    'position_name_Right Center Back',
    'position_name_Right Center Forward',
    'position_name_Right Center Midfield',
    'position_name_Right Defensive Midfield',
    'position_name_Right Midfield',
    'position_name_Right Wing',
    'position_name_Right Wing Back'
]

st.set_page_config(page_title="Player Goal Probability Predictor", layout="wide")
st.title("Player Goal Probability Predictor")
st.markdown("Enter a player's last 5 match stats to estimate their chance of scoring in the next match.")

# --- Input Section ---
st.header("Last 5 Matches Stats")

matches_data = []
for i in range(1, 6):
    with st.expander(f"Match {i}"):
        row = {}
        row['xG'] = st.number_input(f"xG", min_value=0.0, step=0.01, key=f"xG_{i}")
        row['shots'] = st.number_input(f"Shots taken", min_value=0, step=1, key=f"shots_{i}")
        row['minutes'] = st.number_input(f"Minutes played", min_value=0, max_value=120, step=1, key=f"minutes_{i}")
        row['carries'] = st.number_input(f"Total Carries", min_value=0, step=1, key=f"carries_{i}")
        row['dribbles'] = st.number_input(f"Total Dribbles", min_value=0, step=1, key=f"dribbles_{i}")
        row['balls_received'] = st.number_input(f"Balls received in final third", min_value=0, step=1, key=f"balls_{i}")
        matches_data.append(row)

matches_df = pd.DataFrame(matches_data)

# --- Compute rolling averages ---
user_features = pd.DataFrame({
    'avg_xg_rolling_5': [matches_df['xG'].mean()],
    'shots_total_rolling_5': [matches_df['shots'].mean()],
    'minutes_played_rolling_5': [matches_df['minutes'].mean()],
    'carries_total_rolling_5': [matches_df['carries'].mean()],
    'dribbles_total_rolling_5': [matches_df['dribbles'].mean()],
    'ball_receipts_final_third_rolling_5': [matches_df['balls_received'].mean()]
})

# --- Match Context ---
st.header("Match Context")

home_away = st.selectbox("Home or Away?", ['Home', 'Away'])
position = st.selectbox("Select Player Position", [
    'Goalkeeper', 'Center Back', 'Left Back', 'Right Back',
    'Left Center Back', 'Right Center Back', 'Left Wing Back', 'Right Wing Back',
    'Center Defensive Midfield', 'Left Defensive Midfield', 'Right Defensive Midfield',
    'Center Attacking Midfield', 'Left Attacking Midfield', 'Right Attacking Midfield',
    'Center Forward', 'Left Center Forward', 'Right Center Forward',
    'Left Midfield', 'Right Midfield', 'Left Center Midfield', 'Right Center Midfield',
    'Left Wing', 'Right Wing'
])

# One-hot encode home/away
user_features['home_away_home'] = 1 if home_away == 'Home' else 0

# One-hot encode position
for pos_col in [col for col in one_hot_features if col.startswith('position_name_')]:
    user_features[pos_col] = 1 if pos_col == f'position_name_{position}' else 0

# Fill missing columns (to ensure full feature alignment)
for col in one_hot_features:
    if col not in user_features.columns:
        user_features[col] = 0

# Reorder to match model input
user_features = user_features[numeric_features + one_hot_features]

# --- Prediction ---
if st.button("Predict Goal Probability"):
    pred_prob = logreg.predict_proba(user_features)[:, 1][0]
    st.subheader(f"**Predicted Chance of Scoring:** {pred_prob:.2%}")

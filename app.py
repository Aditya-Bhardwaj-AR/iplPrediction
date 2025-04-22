import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Set page config and custom theme
st.set_page_config(page_title="IPL Score Predictor", layout="wide")

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #1e1e1e;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        font-size: 1.2rem;
        width: 100%;
    }
    .stSelectbox [data-baseweb="select"] {
        background-color: #2e2e2e;
        color: white;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 2rem;
    }
    div[data-baseweb="select"] > div {
        background-color: #2e2e2e;
        border-color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# Add title with custom styling
st.title("IPL Score Predictor")

# Add description
st.markdown("""
<div style='background-color: #2e2e2e; padding: 1rem; border-radius: 5px; margin-bottom: 2rem;'>
    <p style='color: white; text-align: center; font-size: 1.1rem;'>
        Select match details below to predict the IPL match score
    </p>
</div>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    ipl = pd.read_csv('ipl_data (1).csv')
    df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker'], axis=1)
    return df

# Load the data
df = load_data()
X = df.drop(['total'], axis=1)
y = df['total']

# Initialize encoders
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

# Fit encoders
X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
X['batsman'] = striker_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])

# Train test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

model = train_model()

# Create columns with styling
st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    # Create dropdown menus with custom styling
    venue = st.selectbox("🏟️ Select Venue", df['venue'].unique())
    batting_team = st.selectbox("🏏 Select Batting Team", df['bat_team'].unique())
    bowling_team = st.selectbox("🎯 Select Bowling Team", df['bowl_team'].unique())

with col2:
    striker = st.selectbox("🏃 Select Striker", df['batsman'].unique())
    bowler = st.selectbox("⚾ Select Bowler", df['bowler'].unique())

st.markdown('</div>', unsafe_allow_html=True)

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Create predict button
if st.button("🎯 Predict Score"):
    # Prepare input data
    input_data = np.array([
        venue_encoder.transform([venue])[0],
        batting_team_encoder.transform([batting_team])[0],
        bowling_team_encoder.transform([bowling_team])[0],
        striker_encoder.transform([striker])[0],
        bowler_encoder.transform([bowler])[0]
    ]).reshape(1, 5)
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = int(model.predict(input_scaled)[0])
    
    # Display prediction with enhanced styling
    st.markdown(f"""
    <div style='padding: 2rem; background: linear-gradient(135deg, #ff4b4b 0%, #ff8f8f 100%); 
                border-radius: 15px; text-align: center; margin-top: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h1 style='color: white; margin: 0; font-size: 2.5rem;'>Predicted Score</h1>
        <h2 style='color: white; margin: 1rem 0; font-size: 3.5rem; font-weight: bold;'>{prediction}</h2>
        <p style='color: white; margin: 0; font-size: 1.2rem;'>Runs</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='position: fixed; bottom: 0; left: 0; right: 0; background-color: #2e2e2e; padding: 1rem; text-align: center;'>
    <p style='color: white; margin: 0;'>Made with ❤️ for IPL Cricket</p>
</div>
""", unsafe_allow_html=True) 
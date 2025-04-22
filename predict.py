import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
print("Loading dataset...")
ipl = pd.read_csv('ipl_data (1).csv')

# Print available columns
print("\nAvailable columns in the dataset:")
print(ipl.columns.tolist())

# Dropping certain features 
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker'], axis=1)
X = df.drop(['total'], axis=1)
y = df['total']

# Label Encoding
print("\nEncoding categorical features...")
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

# Fit and transform the categorical features
X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
X['batsman'] = striker_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])

# Train test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
print("\nTraining the model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# Calculate error metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
print(f"\nModel Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

def predict_score():
    print("\nAvailable options:")
    print("\nVenues:")
    for i, venue in enumerate(df['venue'].unique()):
        print(f"{i+1}. {venue}")
    
    print("\nBatting Teams:")
    for i, team in enumerate(df['bat_team'].unique()):
        print(f"{i+1}. {team}")
    
    print("\nBowling Teams:")
    for i, team in enumerate(df['bowl_team'].unique()):
        print(f"{i+1}. {team}")
    
    print("\nBatsmen:")
    for i, batsman in enumerate(df['batsman'].unique()):
        print(f"{i+1}. {batsman}")
    
    print("\nBowlers:")
    for i, bowler in enumerate(df['bowler'].unique()):
        print(f"{i+1}. {bowler}")
    
    # Get user input
    venue_idx = int(input("\nEnter venue number: ")) - 1
    bat_team_idx = int(input("Enter batting team number: ")) - 1
    bowl_team_idx = int(input("Enter bowling team number: ")) - 1
    batsman_idx = int(input("Enter batsman number: ")) - 1
    bowler_idx = int(input("Enter bowler number: ")) - 1
    
    # Get the selected values
    venue = df['venue'].unique()[venue_idx]
    bat_team = df['bat_team'].unique()[bat_team_idx]
    bowl_team = df['bowl_team'].unique()[bowl_team_idx]
    batsman = df['batsman'].unique()[batsman_idx]
    bowler = df['bowler'].unique()[bowler_idx]
    
    # Prepare input
    input_data = np.array([
        venue_encoder.transform([venue])[0],
        batting_team_encoder.transform([bat_team])[0],
        bowling_team_encoder.transform([bowl_team])[0],
        striker_encoder.transform([batsman])[0],
        bowler_encoder.transform([bowler])[0]
    ]).reshape(1, 5)
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    predicted_score = model.predict(input_scaled)[0]
    
    print(f"\nPredicted Score: {int(predicted_score)}")

while True:
    try:
        predict_score()
        if input("\nMake another prediction? (y/n): ").lower() != 'y':
            break
    except Exception as e:
        print(f"Error: {e}")
        print("Please try again with valid numbers.") 
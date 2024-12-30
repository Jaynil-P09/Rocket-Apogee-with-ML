import joblib
import pandas as pd

# Load the trained scaler and models
scaler = joblib.load('scaler.pkl')
classifier = joblib.load('random_forest_classifier.pkl')
regressor = joblib.load('random_forest_regressor.pkl')

# New rocket's input parameters
new_rocket_data = {
    'Mass': [1200],  # Mass in kg
    'Thrust': [18000],  # Thrust in N
    'Wind Speed': [15],  # Wind Speed in m/s
    'Temperature': [290],  # Temperature in K
    'Drag Coefficient': [0.7]  # Drag Coefficient
}

# Create DataFrame for the new rocket
new_rocket = pd.DataFrame(new_rocket_data)

# Scale the new rocket's features
new_rocket_scaled = scaler.transform(new_rocket)

# Predict success (1 = successful, 0 = failed)
new_success = classifier.predict(new_rocket_scaled)

# Predict apogee (maximum altitude in meters)
new_apogee = regressor.predict(new_rocket_scaled)

# Output the results
print(f"New Rocket Launch {'Successful' if new_success[0] else 'Failed'}")
print(f"Predicted Apogee: {new_apogee[0]:.2f} meters")

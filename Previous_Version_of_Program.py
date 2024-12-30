import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Constants
G = 9.81  # Gravity (m/s^2). This is the standard acceleration due to gravity on Earth's surface.
RHO = 1.225  # Air density (kg/m^3). This represents air density at sea level under standard conditions.
A = 0.1  # Cross-sectional area (m^2). This is the assumed frontal area of the rocket.
DT = 0.1  # Time step (s). Smaller values improve simulation accuracy but increase computational time.
T_MAX = 100  # Max simulation time (s). This limits how long the simulation runs.
BURN_TIME = 50  # Burn time (s). This represents the duration of rocket motor operation.

# Input for synthetic data generation
# Flags for user input validation
asking = True
asking_testsize = True

# Prompt the user to specify the number of test sets
while asking:
    try:
        n_samples = int(input("How Many Test Sets Would You Like? "))
        asking = False
    except ValueError:
        print("Invalid Input. Please Try Again")

# Prompt the user to specify the test size as a fraction
while asking_testsize:
    try:
        ts = float(input("What Percent Test Size Would You Like to Train On? Enter as a Decimal (e.g., 0.9 = 90%). "))
        if 0 < ts < 1:
            asking_testsize = False
        else:
            print("Invalid Input. Please Try Again")
    except ValueError:
        print("Invalid Input. Please Try Again")

# Generate synthetic data
def simulate_rocket(mass, thrust, drag_coefficient, wind_speed, temperature):
    """
    Simulates a rocket's flight based on physical parameters.

    Parameters:
    - mass (float): Rocket's mass in kilograms.
    - thrust (float): Rocket's thrust in Newtons.
    - drag_coefficient (float): Aerodynamic drag coefficient.
    - wind_speed (float): Wind speed in m/s.
    - temperature (float): Ambient temperature in Kelvin.

    Returns:
    - success (int): 1 if the rocket reaches >10 km altitude with positive velocity; otherwise 0.
    - altitude (float): Maximum altitude achieved (in meters).
    """
    velocity = 0
    altitude = 0
    time = np.arange(0, T_MAX, DT)

    for t in time:
        drag = 0.5 * drag_coefficient * RHO * A * velocity ** 2
        gravity = mass * G
        net_force = (thrust - drag - gravity) if t < BURN_TIME else (-drag - gravity)
        acceleration = net_force / mass
        velocity += acceleration * DT
        altitude += velocity * DT

        if altitude < 0:
            return 0, altitude

    return (1 if altitude > 10000 and velocity > 0 else 0), altitude  # Success condition requires altitude >10 km and positive velocity at the end.

# Generate data points
data = [
    [
        mass := np.random.uniform(500, 1500),
        thrust := np.random.uniform(10000, 20000),
        wind_speed := np.random.uniform(0, 30),
        temperature := np.random.uniform(270, 310),
        drag_coefficient := np.random.uniform(0.5, 1.0),
        *(simulate_rocket(mass, thrust, drag_coefficient, wind_speed, temperature))
    ]
    for _ in range(n_samples)
]

# Convert to DataFrame
columns = ['Mass', 'Thrust', 'Wind Speed', 'Temperature', 'Drag Coefficient', 'Success', 'Apogee']
df = pd.DataFrame(data, columns=columns)

# Split into features and targets
X = df[['Mass', 'Thrust', 'Wind Speed', 'Temperature', 'Drag Coefficient']]
y_success = df['Success']
y_apogee = df['Apogee']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Feature scaling ensures inputs have similar magnitudes for effective model training.
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
print("Scaler saved successfully!")

# Split data
X_train, X_test, y_train_success, y_test_success, y_train_apogee, y_test_apogee = train_test_split(
    X_scaled, y_success, y_apogee, test_size=ts, random_state=50
)

# Train models
classifier = RandomForestClassifier(random_state=42)  # RandomForestClassifier is used due to its robustness and ability to handle non-linear relationships.
classifier.fit(X_train, y_train_success)
joblib.dump(classifier, 'random_forest_classifier.pkl')  # Save the classifier model
print("Classifier model saved successfully as 'random_forest_classifier.pkl'!")

regressor = RandomForestRegressor(random_state=42)  # RandomForestRegressor is suitable for predicting continuous outputs like apogee.
regressor.fit(X_train, y_train_apogee)
joblib.dump(regressor, 'random_forest_regressor.pkl')  # Save the regressor model
print("Regressor model saved successfully as 'random_forest_regressor.pkl'!")

# Predictions
y_pred_success = classifier.predict(X_test)
y_pred_apogee = regressor.predict(X_test)

# Evaluation
success_accuracy = accuracy_score(y_test_success, y_pred_success)
apogee_rmse = np.sqrt(mean_squared_error(y_test_apogee, y_pred_apogee))

print(f"Success Prediction Accuracy: {success_accuracy:.2f}")
print(f"Apogee Prediction RMSE: {apogee_rmse:.2f}")

# Plot Results
plt.figure(figsize=(12, 6))

# Predicted vs Actual Apogee
plt.subplot(1, 2, 1)
plt.scatter(y_test_apogee, y_pred_apogee, alpha=0.5)  # Scatter plot to compare predicted and actual apogee values.
plt.plot([min(y_test_apogee), max(y_test_apogee)], [min(y_test_apogee), max(y_test_apogee)], 'r--')
plt.title('Predicted vs Actual Apogee')
plt.xlabel('Actual Apogee (m)')
plt.ylabel('Predicted Apogee (m)')

# Success Prediction Distribution
plt.subplot(1, 2, 2)
plt.bar(['Failure', 'Success'], [np.sum(y_pred_success == 0), np.sum(y_pred_success == 1)], color=['red', 'green'])
plt.title('Launch Success Prediction')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Example Prediction
new_rocket = pd.DataFrame([[1000, 15000, 10, 300, 0.75]], columns=X.columns)  # Example rocket with specified parameters for prediction.
new_rocket_scaled = scaler.transform(new_rocket)
new_success = classifier.predict(new_rocket_scaled)
new_apogee = regressor.predict(new_rocket_scaled)

print(f"New Rocket Launch {'Successful' if new_success[0] else 'Failed'}")
print(f"Predicted Apogee: {new_apogee[0]:.2f} meters")




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Constants
G = 9.81  # Gravitational acceleration (m/s^2), Earth's gravity
RHO = 1.225  # Air density at sea level in kg/m^3
A = 0.1  # Cross-sectional area of the rocket (m^2)
DT = 0.1  # Time step for simulation (seconds)
T_MAX = 100  # Maximum simulation time (seconds)
BURN_TIME = 50  # Duration of the rocket engine burn time (seconds)

# Input prompts for the number of samples and test size
asking = True
asking_testsize = True

# Get the number of test samples from the user
while asking:
    try:
        n_samples = int(input("How Many Test Sets Would You Like? "))
        asking = False
    except ValueError:
        print("Invalid Input. Please Try Again")

# Get the test size as a fraction for splitting the dataset
while asking_testsize:
    try:
        ts = float(input("What Percent Test Size Would You Like to Train On? Enter as a Decimal (e.g., 0.9 = 90%). "))
        if 0 < ts < 1:
            asking_testsize = False
        else:
            print("Invalid Input. Please Try Again")
    except ValueError:
        print("Invalid Input. Please Try Again")

# Simulate rocket flight and determine success or failure
def simulate_rocket(mass, thrust, drag_coefficient, wind_speed, temperature):
    """
    Simulates a rocket flight and returns whether the launch is successful and the maximum altitude.

    Parameters:
    - mass (float): Mass of the rocket in kilograms.
    - thrust (float): Thrust generated by the rocket's engine in Newtons.
    - drag_coefficient (float): Aerodynamic drag coefficient (dimensionless).
    - wind_speed (float): Wind speed (m/s) affecting the rocket's motion.
    - temperature (float): Ambient temperature (K).

    Returns:
    - success (int): 1 if the rocket successfully reaches > 10 km altitude, otherwise 0.
    - altitude (float): Maximum altitude achieved during the flight (m).
    """
    if mass <= 0 or drag_coefficient <= 0 or A <= 0:
        raise ValueError("Mass, drag coefficient, and cross-sectional area must be positive.")

    velocity = 0
    altitude = 0
    time = np.arange(0, T_MAX, DT)

    # Run simulation over time steps
    for t in time:
        drag = 0.5 * drag_coefficient * RHO * A * velocity ** 2
        gravity = mass * G
        net_force = (thrust - drag - gravity) if t < BURN_TIME else (-drag - gravity)
        acceleration = net_force / mass

        velocity += acceleration * DT
        altitude += velocity * DT

        # Stop simulation if the rocket falls back to the ground
        if altitude < 0:
            altitude = 0
            return 0, altitude

    # Success criteria: altitude > 10 km and positive velocity
    return (1 if altitude > 10000 and velocity > 0 else 0), altitude

# Generate synthetic rocket flight data
data = []
for _ in range(n_samples):
    mass = np.random.uniform(500, 1500)  # Random mass between 500 and 1500 kg
    thrust = np.random.uniform(10000, 20000)  # Random thrust between 10000 and 20000 N
    wind_speed = np.random.uniform(0, 30)  # Random wind speed between 0 and 30 m/s
    temperature = np.random.uniform(270, 310)  # Random temperature between 270 K and 310 K
    drag_coefficient = np.random.uniform(0.5, 1.0)  # Random drag coefficient between 0.5 and 1.0

    # Simulate the rocket's flight and store the data if the simulation runs successfully
    try:
        success, altitude = simulate_rocket(mass, thrust, drag_coefficient, wind_speed, temperature)
        data.append([mass, thrust, wind_speed, temperature, drag_coefficient, success, altitude])
    except ValueError as e:
        print(f"Simulation skipped due to error: {e}")

# Convert data to a DataFrame
columns = ['Mass', 'Thrust', 'Wind Speed', 'Temperature', 'Drag Coefficient', 'Success', 'Apogee']
df = pd.DataFrame(data, columns=columns)

# Split the data into features (X) and targets (y)
X = df[['Mass', 'Thrust', 'Wind Speed', 'Temperature', 'Drag Coefficient']]
y_success = df['Success']
y_apogee = df['Apogee']

# Feature scaling for model training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for later use
print("Scaler saved successfully!")

# Split the dataset into training and test sets
X_train, X_test, y_train_success, y_test_success, y_train_apogee, y_test_apogee = train_test_split(
    X_scaled, y_success, y_apogee, test_size=ts, random_state=50
)

# Train a Random Forest Classifier to predict success/failure of the launch
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train_success)
joblib.dump(classifier, 'random_forest_classifier.pkl')  # Save the trained classifier
print("Classifier model saved successfully as 'random_forest_classifier.pkl'!")

# Train a Random Forest Regressor to predict the maximum altitude (apogee)
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train_apogee)
joblib.dump(regressor, 'random_forest_regressor.pkl')  # Save the trained regressor
print("Regressor model saved successfully as 'random_forest_regressor.pkl'!")

# Make predictions with the test data
y_pred_success = classifier.predict(X_test)
y_pred_apogee = regressor.predict(X_test)

# Evaluate model performance
success_accuracy = accuracy_score(y_test_success, y_pred_success)
apogee_rmse = np.sqrt(mean_squared_error(y_test_apogee, y_pred_apogee))

print(f"Success Prediction Accuracy: {success_accuracy:.2f}")
print(f"Apogee Prediction RMSE: {apogee_rmse:.2f}")

# Plot results for analysis
plt.figure(figsize=(12, 6))

# Predicted vs Actual Apogee
plt.subplot(1, 2, 1)
plt.scatter(y_test_apogee, y_pred_apogee, alpha=0.5)
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

# Example Prediction for a new rocket
new_rocket = pd.DataFrame([[1000, 15000, 10, 300, 0.75]], columns=X.columns)  # Example input rocket parameters
new_rocket_scaled = scaler.transform(new_rocket)  # Scale the features using the previously saved scaler
new_success = classifier.predict(new_rocket_scaled)  # Predict success/failure
new_apogee = regressor.predict(new_rocket_scaled)  # Predict maximum altitude

print(f"New Rocket Launch {'Successful' if new_success[0] else 'Failed'}")
print(f"Predicted Apogee: {new_apogee[0]:.2f} meters")















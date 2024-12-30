import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib  # For saving and loading models or scalers

# Constants for the simulation
g = 9.81  # Acceleration due to gravity (m/s^2)
rho = 1.225  # Air density at sea level (kg/m^3)
A = 0.1  # Cross-sectional area of the rocket (m^2)
Cd = 0.75  # Drag coefficient (assumed constant)
burn_time = 50  # Burn time of the rocket motor (s)
dt = 0.1  # Time step for the simulation (seconds)
t_max = 100  # Maximum simulation time (s)

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

# Initialize an empty list to hold generated data
data = []

# Generate synthetic data for training
for _ in range(n_samples):
    # Randomly generate physical parameters of the rocket and environment
    mass = np.random.uniform(500, 1500)  # Mass (kg)
    thrust = np.random.uniform(10000, 20000)  # Thrust (N)
    wind_speed = np.random.uniform(0, 30)  # Wind speed (m/s)
    temperature = np.random.uniform(270, 310)  # Temperature (K)
    drag_coefficient = np.random.uniform(0.5, 1.0)  # Drag coefficient

    # Initialize flight parameters
    velocity = 0  # Initial velocity (m/s)
    altitude = 0  # Initial altitude (m)
    time = np.arange(0, t_max, dt)  # Time steps

    # Simulate the rocket's flight
    for t in time:
        # Calculate drag force and gravity force
        drag = 0.5 * drag_coefficient * rho * A * velocity**2
        gravity = mass * g

        # Determine the net force acting on the rocket
        if t < burn_time:  # During burn phase
            net_force = thrust - drag - gravity
        else:  # Post-burn phase
            net_force = -drag - gravity

        # Update velocity and altitude using basic kinematics
        acceleration = net_force / mass
        velocity += acceleration * dt
        altitude += velocity * dt

        # Stop simulation if the rocket hits the ground
        if altitude < 0:
            success = 0
            break
    else:
        # Determine success based on whether the rocket reached 10 km altitude
        success = 1 if altitude > 10000 else 0

    # Append the parameters and results to the dataset
    data.append([mass, thrust, wind_speed, temperature, drag_coefficient, success, altitude])

# Convert the dataset into a pandas DataFrame
columns = ['Mass', 'Thrust', 'Wind Speed', 'Temperature', 'Drag Coefficient', 'Success', 'Apogee']
df = pd.DataFrame(data, columns=columns)

# Split the dataset into features (X) and targets (y)
X = df[['Mass', 'Thrust', 'Wind Speed', 'Temperature', 'Drag Coefficient']]
y_success = df['Success']  # Binary success outcome
y_apogee = df['Apogee']  # Continuous altitude outcome

# Scale the features to standardize the input
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved successfully!")

# Split the data into training and testing sets
X_train, X_test, y_train_success, y_test_success, y_train_apogee, y_test_apogee = train_test_split(
    X_scaled, y_success, y_apogee, test_size=ts, random_state=42
)

# Train a classifier for success prediction
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train_success)

# Train a regressor for apogee prediction
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train_apogee)

# Make predictions on the test set
y_pred_success = classifier.predict(X_test)
y_pred_apogee = regressor.predict(X_test)

# Evaluate the models
success_accuracy = accuracy_score(y_test_success, y_pred_success)
apogee_rmse = np.sqrt(mean_squared_error(y_test_apogee, y_pred_apogee))

# Print evaluation results
print(f"Success Prediction Accuracy: {success_accuracy}")
print(f"Apogee Prediction RMSE: {apogee_rmse}")

# Visualize the results
plt.figure(figsize=(12, 6))

# Predicted vs actual apogee plot
plt.subplot(1, 2, 1)
plt.scatter(y_test_apogee, y_pred_apogee, color='blue', alpha=0.5)
plt.plot([min(y_test_apogee), max(y_test_apogee)], [min(y_test_apogee), max(y_test_apogee)], color='red', linestyle='--')
plt.title('Predicted vs Actual Apogee')
plt.xlabel('Actual Apogee (m)')
plt.ylabel('Predicted Apogee (m)')

# Success rate bar chart
plt.subplot(1, 2, 2)
plt.bar([0, 1], [np.sum(y_pred_success == 0), np.sum(y_pred_success == 1)], color=['red', 'green'])
plt.xticks([0, 1], ['Failure', 'Success'])
plt.title('Launch Success Prediction')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Predict results for a new rocket design
new_rocket = pd.DataFrame([[1000, 15000, 10, 300, 0.75]], columns=['Mass', 'Thrust', 'Wind Speed', 'Temperature',
                                                                   'Drag Coefficient'])
new_rocket_scaled = scaler.transform(new_rocket)  # Scale the new data
new_success = classifier.predict(new_rocket_scaled)
new_apogee = regressor.predict(new_rocket_scaled)

# Print predictions for the new rocket design
print(f"New Rocket Launch {'Successful' if new_success[0] else 'Failed'}")
print(f"Predicted Apogee: {new_apogee[0]} meters")


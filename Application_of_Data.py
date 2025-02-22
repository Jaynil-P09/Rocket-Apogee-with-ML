import joblib
import pandas as pd

# Load the trained scaler and models
scaler = joblib.load('scaler.pkl')
classifier = joblib.load('random_forest_classifier.pkl')
regressor = joblib.load('random_forest_regressor.pkl')

#Variable initialization
asking_mass = True
asking_thrust = True
asking_wind_speed = True
asking_temperature = True
asking_drag_coeff = True

mass = 0
thrust = 0
wind_speed = 0
temperature = 0
drag_coeff = 0

#Input for the the parameters
while asking_mass:
    try:
        mass = float(input("What is the mass of your rocket in kg? "))

        if mass < 0:
            print("Please enter a positive number. Please try again")
        else:
            asking_mass = False
    except:
        print("Please enter a number greater than zero. Please try again")

while asking_thrust:
    try:
        thrust = float(input("What is the thrust of your rocket in newtons? "))

        if thrust < 0:
            print("Please enter a positive number. Please try again")
        else:
            asking_thrust = False
    except:
        print("Please enter a number greater than zero. Please try again")

while asking_wind_speed:
    try:
        wind_speed = float(input("What is the wind speed in meters per second? "))

        if wind_speed < 0:
            print("Please enter a positive number. Please try again")
        else:
            asking_wind_speed = False
    except:
        print("Please enter a number greater than zero. Please try again")

while asking_temperature:
    try:
        temperature = float(input("What is the temperature in kelvins? "))
        asking_temperature = False
    except:
        print("Please enter a number. Please try again")

while asking_drag_coeff:
    try:
        drag_coeff = float(input("What is the drag coefficient? "))

        if drag_coeff < 0:
            print("Please enter a positive number. Please try again")
        else:
            asking_drag_coeff = False
    except:
        print("Please enter a number greater than zero. Please try again")


# New rocket's input parameters
new_rocket_data = {
    'Mass': [mass],  # Mass in kg
    'Thrust': [thrust],  # Thrust in N
    'Wind Speed': [wind_speed],  # Wind Speed in m/s
    'Temperature': [temperature],  # Temperature in K
    'Drag Coefficient': [drag_coeff]  # Drag Coefficient
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

# Rocket-Apogee-with-ML
 This script simulates rocket launches based on parameters like mass, thrust, drag coefficient, wind speed, and temperature. It generates data,trains a Random Forest Classifier for success prediction and a Random Forest Regressor for maximum altitude.The models are evaluated,saved,and can predict success and altitude for new rocket configurations.

#User Guide
First Program (Training and Saving Models):
 1.Setup: Ensure you have the required libraries installed: numpy, pandas, matplotlib, sklearn, and joblib.
 2.Input Parameters: The script will prompt you for the number of test samples and the test size (as a decimal) to split the dataset.
 3.Simulation: The program generates synthetic rocket data by simulating launches and records whether the launch is successful and the maximum altitude (apogee).
 4.Training Models: It trains a Random Forest Classifier to predict success/failure and a Random Forest Regressor for maximum altitude.
 5.Saving Models: The trained models (classifier.pkl and regressor.pkl) and the scaler (scaler.pkl) are saved to disk for future use.
 6.Run this script to generate the models and scaler that will be used by the second program.

Second Program (Prediction for New Rocket):
 1.Setup: Ensure you have the required libraries installed: joblib and pandas.
 2.Loading Models: The script loads the previously trained models (scaler.pkl, random_forest_classifier.pkl, and random_forest_regressor.pkl).
 3.Input Data: Input the parameters for a new rocket, such as mass, thrust, wind speed, temperature, and drag coefficient. Modify the new_rocket_data dictionary with your own values.
 4.Prediction: The script scales the input data using the saved scaler and makes predictions using the classifier (to predict success) and regressor (to predict maximum altitude).
 5.Output: The script will print whether the launch is predicted to be successful and the predicted apogee (maximum altitude in meters).
 6.This second script allows you to input new rocket parameters and get predictions on the rocket's success and maximum altitude based on the trained models.

import pandas as pd
import numpy as np
import os

from h5py.h5pl import append
from statsmodels.tsa.statespace.sarimax import SARIMAX
from django.conf import settings
import matplotlib.pyplot as plt

def sarimax_analysis(start_date, end_date, output_dir, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    # Step 1: Load the dataset
    try:
        file_path = os.path.join(settings.BASE_DIR, 'Datafile', 'atm_cash_demand_single_atm.csv')
        ds = pd.read_csv(file_path)

        # Strip any whitespace from the column names
        ds.columns = ds.columns.str.strip()

    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")

    # Step 2: Prepare the data for SARIMAX
    try:
        df = ds[['Date', 'Total Amount Withdrawn']].rename(columns={'Date': 'ds', 'Total Amount Withdrawn': 'y'})
    except KeyError as e:
        raise Exception(f"Column not found in dataset: {e}")

    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

    if df['ds'].isnull().any():
        raise Exception("Date parsing error: Some dates are invalid.")

    # Step 3: Filter the DataFrame for the specified date range
    df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]

    if df.empty:
        raise Exception(f"No data available for the date range: {start_date} to {end_date}")

    # Step 4: Split the data into training and testing
    to_row = int(len(df) * 0.9)
    training_data = df['y'][:to_row]
    testing_data = df['y'][to_row:]

    # Step 5: Initialize and fit the SARIMAX model
    model = SARIMAX(training_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Check if the model fitting was successful
    if model_fit is None:
        raise Exception("Model fitting failed.")

    # Step 6: Make predictions
    forecast = model_fit.get_forecast(steps=len(testing_data))
    predicted_values = forecast.predicted_mean

    # Check for NaN in predicted values
    if predicted_values.isnull().any():
        raise Exception("Predicted values contain NaN.")

    # Step 7: Calculate RMSE
    if len(testing_data) != len(predicted_values):
        raise Exception("Length mismatch: testing_data and predicted_values must have the same length.")

    print("Testing Data (y):")
    print(testing_data)
    print("Predicted Values:")
    print(predicted_values)

    # Check for NaN values
    if testing_data.isnull().any():
        raise Exception("Testing data contains NaN values.")
    if predicted_values.isnull().any():
        raise Exception("Predicted values contain NaN values.")

    # Ensure non-empty arrays for RMSE calculation
    if len(testing_data) == 0 or len(predicted_values) == 0:
        raise Exception("Cannot calculate RMSE: Testing data or predicted values are empty.")

    # Calculate RMSE
    try:
        rmse = np.sqrt(np.mean((testing_data - predicted_values) ** 2))
    except Exception as e:
        raise Exception(f"Error calculating RMSE: {e}")

    # Step 8: Extract p-values
    p_values = model_fit.pvalues

    # Step 9: Plot actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(df['ds'][to_row:], testing_data, color='grey', label='Actual')
    plt.plot(df['ds'][to_row:], predicted_values, color='red', label='Predicted')
    plt.title(f'SARIMAX: Actual vs Predicted from {start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Total Amount Withdrawn')
    plt.legend()

    # Step 10: Save the plot to a PNG file
    output_dir = os.path.join(settings.BASE_DIR, 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, 'sarimax_plot_actual_vs_predicted.png')
    plt.savefig(image_path)

    # Print RMSE and p-values
    print(f"RMSE: {rmse}")
    print("P-values of model coefficients:")
    print(p_values)

    # Return the path to the saved image
    return image_path

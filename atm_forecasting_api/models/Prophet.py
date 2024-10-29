import pandas as pd
from prophet import Prophet
from django.conf import settings
import plotly.graph_objects as go
import os

def prophet_analysis(start_date, end_date, output_dir, changepoint_prior_scale=0.05, seasonality_mode='additive',
                     seasonality_prior_scale=10.0, holidays_prior_scale=10.0):
    # Print the BASE_DIR to debug
    print(f"BASE_DIR: {settings.BASE_DIR}")

    # Step 1: Load the dataset
    try:
        # Construct the file path using Django settings
        file_path = os.path.join(settings.BASE_DIR, 'Datafile', 'atm_cash_demand_single_atm.csv')
        print(f"Attempting to read file from: {file_path}")  # Debugging line
        ds = pd.read_csv(file_path)

        print(ds.head())  # Check loaded data

        # Strip any whitespace from the column names
        ds.columns = ds.columns.str.strip()

    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")

    # Step 2: Prepare the data for Prophet
    try:
        # Use correct column names here
        df = ds[['Date', 'Total Amount Withdrawn']].rename(
            columns={'Date': 'ds', 'Total Amount Withdrawn': 'y'})
        print(df.head())  # Check prepared DataFrame
        print(df.isnull().sum())  # Check for null values
    except KeyError as e:
        raise Exception(f"Column not found in dataset: {e}")

    # Ensure that the 'ds' column is in datetime format
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

    # Check for null values after datetime conversion
    if df['ds'].isnull().any():
        raise Exception("Date parsing error: Some dates are invalid.")

    # Step 3: Filter the DataFrame for the specified date range
    df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]

    # Check if there is any data in the filtered DataFrame
    if df.empty:
        raise Exception(f"No data available for the date range: {start_date} to {end_date}")

    # Step 4: Initialize the Prophet model with hyperparameters
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale
    )

    # Step 5: Fit the Prophet model
    model.fit(df)

    # Step 6: Make future predictions for the next year
    future = model.make_future_dataframe(periods=30)  # Forecast for the next 30 days
    print(future.head())  # Check the future DataFrame
    forecast = model.predict(future)

    # Step 7: Check if forecast data is available
    if forecast.empty:
        raise Exception("No forecast data available.")

    # Step 8: Extract the dates and predicted values
    all_dates = forecast['ds']
    predicted_values = forecast['yhat']

    # Step 9: Prepare actual values for plotting
    actual_values = df['y']

    # Step 10: Create a Plotly figure for actual vs. predicted values
    fig = go.Figure()

    # Add actual values trace
    fig.add_trace(
        go.Scatter(x=df['ds'], y=actual_values, mode='lines+markers', name='Actual', line=dict(color='grey')))

    # Add predicted values trace
    fig.add_trace(go.Scatter(x=all_dates, y=predicted_values, mode='lines', name='Predicted', line=dict(color='red')))

    # Update layout for better aesthetics
    fig.update_layout(
        height=400,
        title=f'Prophet: Actual vs Predicted from {start_date} to {end_date}',
        xaxis_title='Transaction Date',
        yaxis_title='Total Amount Withdrawn',
        xaxis=dict(tickformat="%Y-%m-%d"),
        legend_title='Legend'
    )

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure to a file
    output_dir = os.path.join(settings.BASE_DIR, 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, 'prophet_plot_actual_vs_predicted.png')
    fig.write_image(image_path)

    # Return the path to the saved image
    return image_path

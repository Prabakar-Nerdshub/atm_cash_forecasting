import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import plotly.graph_objects as go
import os


def xgboost_analysis(period, output_dir):
    # Step 1: Load the dataset
    data = pd.read_csv('/home/praba/Desktop/ATM-Cash-Forecasting/Datafile/atm_cash_demand_single_atm.csv')

    # Step 2: Convert 'Date' to datetime and extract features
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)

    # Step 3: Define features and target
    X = data[['Date_ordinal']]  # Ensure X only includes numeric features
    y = data['Total Amount Withdrawn']

    # Step 4: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Create and fit the XGBoost model
    model = XGBRegressor()

    # Step 6: Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    print(f'Cross-validated RMSE: {cv_rmse.mean()}')

    # Step 7: Fit the model on the training set
    model.fit(X_train, y_train)

    # Step 8: Make predictions
    y_pred = model.predict(X_test)

    # Step 9: Evaluate the model on the test set
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Step 10: Create a DataFrame for predicted values
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results.reset_index(drop=True, inplace=True)  # Reset index for better plotting

    # Step 11: Create a Plotly figure for Predicted values only
    fig = go.Figure()

    # Add predicted values trace
    fig.add_trace(go.Scatter(x=results.index, y=results['Actual'],
                             mode='lines+markers', name='Actual Values', line=dict(color='grey')))

    # Add predicted values trace
    fig.add_trace(go.Scatter(x=results.index, y=results['Predicted'],
                             mode='lines+markers', name='Predicted Values', line=dict(color='red')))

    # Update layout for better aesthetics
    fig.update_layout(height=400,
                      title='Actual vs Predicted Values',
                      xaxis_title='Index',
                      yaxis_title='Amount Withdrawn',
                      legend_title='Legend')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure to a file
    image_path = os.path.join(output_dir, 'xgboost_plot_predicted_values.png')
    fig.write_image(image_path)

    # Return the image URI and RMSE for use in Django
    return image_path, cv_rmse.mean()

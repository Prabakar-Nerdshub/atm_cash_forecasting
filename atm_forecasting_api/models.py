# views.py

from django.shortcuts import render, redirect
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import io
import urllib, base64
import xgboost as xgb
from sklearn.model_selection import train_test_split
import base64
import urllib.parse


def index(request):
    return render(request, 'index.html')

def redirect_model(request):
    if request.method == 'POST':
        selected_model = request.POST.get('model')  # Get selected model from the form
        if selected_model == 'XGBoost':
            return redirect('xgboost_analysis')  # Redirect to XGBoost analysis page
        elif selected_model == 'Prophet':
            return redirect('prophet_analysis')  # Redirect to Prophet analysis page
    return render(request, 'index.html')

def plot_graph(all_dates, all_values, title):
    plt.figure(figsize=(12, 8))
    plt.plot(all_dates, all_values, label='Predicted', color='red')
    plt.xlabel('Transaction Date')
    plt.ylabel('Total amount Withdrawn')
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    image_base64 = base64.b64encode(image_png).decode('utf-8')
    image_uri = 'data:image/png;base64,' + urllib.parse.quote(image_base64)
    return image_uri

def xgboost_analysis(request):
    period = request.GET.get('period', 'day')
    period_map = {'day': 1, 'week': 7, 'month': 30}
    future_period = period_map.get(period, 1)

    ds = pd.read_csv('/home/praba/Desktop/ATM-Cash-Forecasting/Datafile/atm_cash_demand_single_atm.csv')
    #/home/praba/Desktop/ATM-Cash-Forecasting/Datafile/atm_cash_demand_single_atm.csv
    ds['Transaction Date'] = pd.to_datetime(ds['Transaction Date'], format='mixed', dayfirst=True)
    ds['Year'] = ds['Transaction Date'].dt.year
    ds['Month'] = ds['Transaction Date'].dt.month
    ds['Day'] = ds['Transaction Date'].dt.day
    ds['DayOfWeek'] = ds['Transaction Date'].dt.dayofweek

    X = ds[['Year', 'Month', 'Day', 'DayOfWeek']]
    y = ds['Total amount Withdrawn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=4, learning_rate=0.05)
    xg_reg.fit(X_train, y_train)

    y_pred = xg_reg.predict(X_test)

    last_date = ds['Transaction Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_period, freq='D')
    future_dates_series = pd.Series(future_dates, name='Transaction Date')

    future_df = pd.DataFrame({
        'Year': future_dates_series.dt.year,
        'Month': future_dates_series.dt.month,
        'Day': future_dates_series.dt.day,
        'DayOfWeek': future_dates_series.dt.dayofweek
    })
    future_pred = xg_reg.predict(future_df)

    all_dates = pd.concat(
        [ds['Transaction Date'], X_test.index.to_series().map(lambda idx: ds['Transaction Date'].iloc[idx]),
         future_dates_series], ignore_index=True)
    all_values = pd.concat([y, pd.Series(y_pred, index=X_test.index), pd.Series(future_pred)], ignore_index=True)

    image_uri = plot_graph(all_dates, all_values, f'XGBoost: Actual, Predicted, and Future Predictions ({period.capitalize()})')

    return render(request, 'XGBoost_analysis.html', {'image': image_uri})


from prophet import Prophet
from prophet.plot import plot_plotly



def prophet_analysis(request):
    # Get the selected period from the GET parameters
    period = request.GET.get('period', 'day')
    period_map = {'day': 1, 'week': 7, 'month': 30}
    future_period = period_map.get(period, 1)

    # Load and preprocess data
    ds = pd.read_csv('/home/praba/Desktop/ATM-Cash-Forecasting/Datafile/atm_cash_demand_single_atm.csv')
    ds['Transaction Date'] = pd.to_datetime(ds['Transaction Date'], format='mixed', dayfirst=True)

    # Prepare data for Prophet
    df_prophet = ds[['Transaction Date', 'Total amount Withdrawn']].copy()
    df_prophet.rename(columns={'Transaction Date': 'ds', 'Total amount Withdrawn': 'y'}, inplace=True)

    # Ensure 'ds' column is datetime
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    # Ensure 'y' column is numeric
    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')

    # Check for NaN values in the DataFrame
    df_prophet = df_prophet.dropna()

    # Split data into train and test sets
    train_size = int(len(df_prophet) * 0.8)
    train = df_prophet[:train_size]
    test = df_prophet[train_size:]

    # Initialize and fit the Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train)

    # Make future dataframe
    future_periods = future_period + len(test)
    future = model.make_future_dataframe(periods=future_periods)

    # Forecast
    forecast = model.predict(future)

    # Plot the atm_forecasting_api
    fig = plot_plotly(model, forecast)

    # Convert plot to HTML string
    plot_html = fig.to_html(full_html=False)

    return render(request, 'prophet_analysis.html', {'plot': plot_html})


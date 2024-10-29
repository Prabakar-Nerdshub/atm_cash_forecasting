import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import io
import base64
import urllib.parse

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

def lstm_analysis():
    # Load data
    df = pd.read_csv("/Datafile/atm_cash_demand_single_atm.csv")  # Adjust the path as needed
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    df['DayOfWeek'] = df['Transaction Date'].dt.dayofweek
    df['DayOfMonth'] = df['Transaction Date'].dt.day

    # One-hot encode 'DayOfWeek'
    encoder_day_of_week = OneHotEncoder(sparse=False, drop='first')
    encoded_day_of_week = encoder_day_of_week.fit_transform(df[['DayOfWeek']])
    encoded_day_of_week_df = pd.DataFrame(encoded_day_of_week, columns=[f'DayOfWeek_{i}' for i in range(encoded_day_of_week.shape[1])])

    # Combine features
    df = pd.concat([df, encoded_day_of_week_df], axis=1)

    # Drop original columns
    df = df.drop(['Transaction Date', 'DayOfWeek'], axis=1)

    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_features = ['Total_transaction', 'Balance_amount', 'DayOfMonth']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Prepare sequences (timesteps = 7 days)
    def create_sequences(data, timesteps):
        X, y = [], []
        for i in range(len(data) - timesteps):
            X.append(data.iloc[i:i + timesteps].values)
            y.append(data.iloc[i + timesteps]['Total_transaction'])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    timesteps = 7
    X, y = create_sequences(df, timesteps)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(timesteps, X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))  # Output layer for regression

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Learning Rate Scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=1, validation_data=(X_val, y_val), callbacks=[lr_scheduler])

    # Forecast the next 7 days of transactions
    X_last = X[-1].reshape((1, timesteps, X.shape[2]))
    forecast = []
    for _ in range(7):
        predicted = model.predict(X_last)[0][0]
        forecast.append(predicted)

        # Prepare the next input sequence
        new_row = X_last[0, 1:, :]
        new_row = np.append(new_row, [[predicted] + list(X_last[0, -1, 1:])], axis=0)
        X_last = new_row.reshape((1, timesteps, X.shape[2]))

    # Convert forecast to 2D array and revert scaling
    forecast = np.array(forecast).reshape(-1, 1)
    forecast = scaler.inverse_transform(
        np.hstack([
            forecast,
            np.zeros((len(forecast), 2))
        ])
    )[:, 0]

    # Generate new dates for the forecast
    forecast_start_date = pd.Timestamp.today()
    forecast_dates = [forecast_start_date + pd.Timedelta(days=i) for i in range(7)]

    # Prepare the forecast dataframe
    forecast_data = pd.DataFrame({
        'Transaction Date': forecast_dates,
        'Total_transaction': forecast.flatten(),
        'Balance_amount': np.nan,
        'DayOfMonth': [date.day for date in forecast_dates]
    })

    # Plotting the results
    all_dates = pd.concat([pd.to_datetime(df.index), pd.to_datetime(forecast_data['Transaction Date'])], ignore_index=True)
    all_values = pd.concat([pd.Series(df['Total_transaction']), pd.Series(forecast_data['Total_transaction'])], ignore_index=True)
    image_uri = plot_graph(all_dates, all_values, "LSTM: Forecast for Next 7 Days")

    return image_uri

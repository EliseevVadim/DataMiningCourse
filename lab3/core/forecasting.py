import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression


def forecast_timeseries(model: LinearRegression, historical_data: pd.DataFrame, horizon: int, time_column: str,
                        lag_columns: list) -> pd.DataFrame:
    forecast = pd.DataFrame(columns=[time_column] + lag_columns + ['predicted'])

    last_time = historical_data[time_column].iloc[-1]
    last_lags = historical_data.iloc[-1][lag_columns].values

    for i in range(horizon):
        last_time += 1
        next_row = pd.DataFrame([last_lags], columns=lag_columns)
        next_row = pd.concat([pd.DataFrame({time_column: [last_time]}), next_row], axis=1)
        predicted_value = model.predict(next_row)[0]
        new_row = {
            time_column: last_time,
            **{f'lag_{lag}': last_lags[lag - 1] for lag in range(1, len(lag_columns) + 1)},
            'predicted': predicted_value
        }
        forecast = pd.concat([forecast, pd.DataFrame([new_row])], ignore_index=True)
        last_lags = np.roll(last_lags, -1)
        last_lags[-1] = predicted_value
    return forecast


def show_forecast_result(historical_data: pd.DataFrame, forecast: pd.DataFrame, predicting_column: str, horizon: int,
                         xlabel: str, ylabel: str):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=historical_data.index, y=historical_data[predicting_column], marker='o',
                 color='blue', label='Original data', linewidth=1)

    forecast_index = pd.date_range(start=historical_data.index[-1], periods=horizon + 1, freq='M')[1:]
    sns.lineplot(x=forecast_index, y=forecast['predicted'], marker='o', label=f'{horizon} Months Forecast',
                 color='green', linewidth=1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{horizon} Months Forecast')
    plt.legend()
    plt.grid()
    plt.show()

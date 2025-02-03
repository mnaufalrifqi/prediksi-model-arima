# -*- coding: utf-8 -*-
"""Prediksi Model ARIMA

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ljZd1FdKAJu5LMrd16_a0IN5MHAoqOFH
"""

!pip install yfinance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings
import statsmodels.tools.tools as sm
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from math import sqrt

warnings.filterwarnings("ignore")

df = yf.download("BMRI.JK", start="2019-12-01", end="2024-12-01")

df.info()

"""nih datanya cleaning"""

df_close = df['Close']
print(df_close.isnull().sum())
df_close = df_close.interpolate(method='linear')
print(df_close.isnull().sum())

df_close.index = pd.to_datetime(df_close.index)
print(df_close.head(20))

"""datanya nyesuain dari adj close"""

plt.figure(figsize=(10, 6))
plt.plot(df_close)
plt.title('Harga Penutupan Saham Bmri')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan')
plt.grid(True)
plt.show()

adf_test = adfuller(df_close)
adf_statistic = adf_test[0]
p_value = adf_test[1]
critical_values = adf_test[4]
print(f"ADF Statistic: {adf_statistic}")
print(f"P-value: {p_value}")
print("Critical Values:")
for key, value in critical_values.items():
    print(f"\t{key}: {value}")
if p_value < 0.05:
    print("Data stasioner")
else:
    print("Data tidak stasioner")

# lag 1
df_diff = df_close.diff().dropna()
adf_test_diff = adfuller(df_diff)
adf_statistic_diff = adf_test_diff[0]
p_value_diff = adf_test_diff[1]

print(f"ADF Statistic (Differenced Data): {adf_statistic_diff}")
print(f"P-value (Differenced Data): {p_value_diff}")

if p_value_diff < 0.05:
    print("Data yang telah di-differencing stasioner")
else:
    print("Data yang telah di-differencing masih tidak stasioner")

"""hasilnya rendah di bawah 0.05 jadi 0.0 setelah di differencing"""

plt.figure(figsize=(10, 6))
plt.plot(df_close)
plt.title('Sebelum Differencing)')
plt.xlabel('Tanggal')
plt.ylabel('Harga Close')
plt.grid(True)
plt.show()
#=============================================================================
plt.figure(figsize=(10, 6))
plt.plot(df_diff)
plt.title('Harga Adjusted Close (Setelah Differencing)')
plt.xlabel('Tanggal')
plt.ylabel('Harga Adjusted Close')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(df_diff, lags=40, ax=plt.gca())
plt.title('ACF')

plt.subplot(1, 2, 2)
plot_pacf(df_diff, lags=40, ax=plt.gca())
plt.title('PACF')
plt.tight_layout()
plt.show()

model = ARIMA(df_close, order=(2, 1, 2))
model_fit = model.fit()
print(model_fit.summary())

"""nih residual buat mastiin tingkatnya heteroskedastisitas karena chart di bawah nunjukin side pada 0 saja"""

residuals = model_fit.resid

plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residual ARIMA')
plt.show()

print(residuals.describe())

"""nih mastiin data yang di atas bisa di guain atau ngganya"""

exog = sm.add_constant(df_close)
test_stat, p_value, _, _ = het_breuschpagan(residuals, exog)

print(f"Breusch-Pagan p-value: {p_value}")

if p_value < 0.05:
    print("Residuals menunjukkan heteroskedastisitas")
else:
    print("Residuals tidak menunjukkan heteroskedastisitas")

plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals Over Time')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.show()

fitted_values = model_fit.fittedvalues
plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, residuals)
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

predictions = model_fit.forecast(steps=len(df_close))
mape = mean_absolute_percentage_error(df_close, predictions)
mse = mean_squared_error(df_close, predictions)
rmse = sqrt(mse)

print(f"MAPE: {mape}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

plt.figure(figsize=(12, 6))
plt.plot(df_close, label='Actual')
plt.plot(forecast_df, label='Forecast')
plt.title('BMRI.JK Stock Price Forecast (30 days)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()

forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)
last_date = df_close.index[-1]
forecast_index = pd.date_range(start=last_date + pd.DateOffset(1), periods=forecast_steps)
forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast})
forecast_df = forecast_df.set_index('Date')

forecast_df
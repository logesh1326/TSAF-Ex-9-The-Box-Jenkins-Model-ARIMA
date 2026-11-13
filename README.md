# TSAF-Ex-8-Modeling-and-Forecasting-with-ARMA-Processes

## AIM:
To model and forecast a time series using an ARMA(p,q) process

## PROGRAM:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)
ar = np.array([1, -0.5, -0.2])
ma = np.array([1, 0.4])
arma_process = ArmaProcess(ar, ma)

data = arma_process.generate_sample(nsample=200)
series = pd.Series(data)

plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title("Simulated ARMA(2,1) Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series, lags=20, ax=ax[0])
plot_pacf(series, lags=20, ax=ax[1])
plt.show()

train_size = int(len(series) * 0.8)
train = series[:train_size]
test = series[train_size:]

print("Training observations:", len(train))
print("Testing observations:", len(test))

model_ar1 = ARIMA(train, order=(1, 0, 0))
fit_ar1 = model_ar1.fit()
model_ma1 = ARIMA(train, order=(0, 0, 1))
fit_ma1 = model_ma1.fit()
model_arma11 = ARIMA(train, order=(1, 0, 1))
fit_arma11 = model_arma11.fit()
model_arma21 = ARIMA(train, order=(2, 0, 1))
fit_arma21 = model_arma21.fit()

print("\n--- Model Information Criteria Comparison ---")
print(f"AR(1) | AIC: {fit_ar1.aic:.2f} | BIC: {fit_ar1.bic:.2f}")
print(f"MA(1) | AIC: {fit_ma1.aic:.2f} | BIC: {fit_ma1.bic:.2f}")
print(f"ARMA(1,1)| AIC: {fit_arma11.aic:.2f} | BIC: {fit_arma11.bic:.2f}")
print(f"ARMA(2,1)| AIC: {fit_arma21.aic:.2f} | BIC: {fit_arma21.bic:.2f}")

forecast = fit_arma21.forecast(steps=len(test))

mae = mean_absolute_error(test, forecast)

mse = mean_squared_error(test, forecast)

rmse = np.sqrt(mse)

print("\n--- Forecast Error Metrics for ARMA(2,1) ---")
print("Mean Absolute Error (MAE):", round(mae, 4))
print("Mean Squared Error (MSE):", round(mse, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))

plt.figure(figsize=(12, 5))
plt.plot(train, label="Training Data")
plt.plot(test, label="Actual Test Data")
plt.plot(test.index, forecast, label="ARMA(2,1) Forecast", color="red")
plt.title("ARMA(2,1) Forecast vs Actual Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

```

## OUTPUT:

<img width="568" height="262" alt="image" src="https://github.com/user-attachments/assets/b377ae64-e4d7-4d92-bcef-c0a6402eacbb" />

<img width="1032" height="692" alt="image" src="https://github.com/user-attachments/assets/64bca88b-9e51-4eea-a0cc-182089c593c2" />

<img width="816" height="375" alt="image" src="https://github.com/user-attachments/assets/c05a52cb-9fb7-4598-9a48-22cf39326087" />

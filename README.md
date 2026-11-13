# TSAF-Ex-9-The-Box-Jenkins-Model-ARIMA


## AIM:
To understand and implement the Box-Jenkins approach to time-series forecasting using an
ARIMA model in Python. 

## PROGRAM:

```
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

sales = [
 100, 105, 103, 110, 115, 112, 120, 125,
 123, 130, 135, 132, 140, 145, 143, 150,
 155, 152, 160, 165, 163, 170, 175, 172
]

data = pd.Series(sales, name="Sales")

print("STEP 1: Original Sales Data")
print(data.to_string(index=False))

plt.figure(figsize=(10, 4))
plt.plot(data, marker="o")
plt.title("Monthly Sales Data")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

adf_result = adfuller(data)

print("\nSTEP 2: ADF Test on Original Data")
print("ADF Statistic:", round(adf_result[0], 4))
print("p-value:", round(adf_result[1], 4))
if adf_result[1] < 0.05:
 print("Conclusion: Evidence of stationarity.")
else:
 print("Conclusion: Insufficient evidence of stationarity.")
diff_data = data.diff().dropna()

print("\nSTEP 3: First-Differenced Data")
print(diff_data.to_string(index=False))
plt.figure(figsize=(10, 4))
plt.plot(diff_data, marker="o")
plt.title("First-Differenced Sales")
plt.xlabel("Month")
plt.ylabel("Change in Sales")
plt.grid(True)
plt.tight_layout()
plt.show()
adf_diff = adfuller(diff_data)

print("\nSTEP 4: ADF Test After Differencing")
print("ADF Statistic:", round(adf_diff[0], 4))
print("p-value:", round(adf_diff[1], 4))
if adf_diff[1] < 0.05:
 print("Conclusion: Evidence of stationarity.")
else:
 print("Conclusion: Stationarity is not confirmed.")

print("\nSTEP 5: ACF and PACF Plots")
plot_acf(diff_data, lags=10)
plt.tight_layout()
plt.show()
plot_pacf(diff_data, lags=10, method="ywm")
plt.tight_layout()
plt.show()
model = ARIMA(data, order=(1, 1, 1))
fitted_model = model.fit()

print("\nSTEP 6: ARIMA Model Summary")
print(fitted_model.summary())

print("\nSTEP 7: Model Selection Criteria")
print("AIC:", round(fitted_model.aic, 4))
print("BIC:", round(fitted_model.bic, 4))
residuals = fitted_model.resid
plt.figure(figsize=(10, 4))
plt.plot(residuals, marker="o")
plt.axhline(0, color="red", linestyle="--")
plt.title("ARIMA Residuals")
plt.xlabel("Month")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nSTEP 8: Residual ACF")
plot_acf(residuals, lags=10)
plt.tight_layout()
plt.show()
lb_result = acorr_ljungbox(residuals, lags=[5], return_df=True)
print("\nLjung-Box Test Result")
print(lb_result)
if lb_result["lb_pvalue"].iloc[0] > 0.05:
 print("No significant residual autocorrelation detected at lag 5.")
else:
 print("Residual autocorrelation may remain; review the model.")
forecast = fitted_model.get_forecast(steps=6)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

print("\nSTEP 9: Forecast for the Next Six Months")
for month, value in enumerate(forecast_mean, start=len(data) + 1):
 print(f"Month {month}: {value:.2f}")
print("\nForecast Confidence Intervals")
print(forecast_ci.round(2))
future_months = range(len(data), len(data) + 6)
plt.figure(figsize=(10, 5))
plt.plot(range(len(data)), data, marker="o", label="Actual Sales")
plt.plot(future_months, forecast_mean.to_numpy(), marker="o", linestyle="--",
label="Forecast")
plt.fill_between(
 future_months,
 forecast_ci.iloc[:, 0].to_numpy(),
 forecast_ci.iloc[:, 1].to_numpy(),
 alpha=0.2,
 label="95% Confidence Interval"
)
plt.title("Box-Jenkins ARIMA: Six-Month Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("\nRESULT")
print("Box-Jenkins ARIMA modelling completed successfully.")

```

## OUTPUT:

<img width="606" height="267" alt="image" src="https://github.com/user-attachments/assets/072ed998-6eee-44f6-8b57-f63d8913d9e6" />

<img width="631" height="560" alt="image" src="https://github.com/user-attachments/assets/3ae2a8c0-8440-4cc4-a1ab-e830726754cd" />

<img width="632" height="263" alt="image" src="https://github.com/user-attachments/assets/9b06f88b-3e20-470c-b9c8-135265d945e1" />

<img width="628" height="491" alt="image" src="https://github.com/user-attachments/assets/cb9ef472-e90c-42d2-9639-9a3c0983b0b2" />

<img width="635" height="446" alt="image" src="https://github.com/user-attachments/assets/6e3d6351-5141-4084-b76c-bfb98216e911" />

<img width="615" height="505" alt="image" src="https://github.com/user-attachments/assets/437f3201-b710-4220-9d76-1b026ff7717f" />

<img width="608" height="322" alt="image" src="https://github.com/user-attachments/assets/e845b333-3ab4-4676-bd08-0eec7554791b" />

<img width="486" height="282" alt="image" src="https://github.com/user-attachments/assets/64153639-e842-4a30-9f6c-ef017268b20c" />


<img width="641" height="287" alt="image" src="https://github.com/user-attachments/assets/a0abe65c-85c7-448e-b378-a224ca06f535" />

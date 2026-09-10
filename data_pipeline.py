import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the generated time-series metrics
df = pd.read_csv('overnight_traffic_stats_history.csv')

# Format the dataframe for Prophet
df_prophet = pd.DataFrame()
df_prophet['ds'] = pd.to_datetime(df['Timestamp'], unit='s')
df_prophet['y'] = df['Requests/s'].fillna(0)

# Initialize and train the ML model
# 1. Initialize Prophet and turn off default real-time assumptions
model = Prophet(
    daily_seasonality=False, 
    weekly_seasonality=False, 
    yearly_seasonality=False,
    seasonality_mode='additive'
)

# 2. Add our time-compressed 1-hour "day" (1/24th of a day)
model.add_seasonality(name='simulated_diurnal', period=1/24, fourier_order=10)

# 3. Fit the model to your cleaned metrics
model.fit(df_prophet)

# 4. Predict the next 1 hour (3600 seconds)
future = model.make_future_dataframe(periods=3600, freq='s')
forecast = model.predict(future)

# Plot the results
fig = model.plot(forecast)
plt.title("Workload Forecast vs Actual (Corrected Seasonality)")
plt.show()
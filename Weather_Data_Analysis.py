import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df = pd.read_csv('weather_data.csv')

# Check the first few rows to understand the data
print(df.head())

# Drop any rows with missing values
df = df.dropna()

# Convert columns to appropriate data types (e.g., dates)
df['Date_Time'] = pd.to_datetime(df['Date_Time'])

# Extract the month from the Date_Time column
df['Month'] = df['Date_Time'].dt.month

# Calculate the daily average temperature
df['avg_temp'] = df['Temperature_C']

# Set Matplotlib chunksize to prevent OverflowError
plt.rcParams['agg.path.chunksize'] = 1000

# Plot temperature trends over time
plt.figure(figsize=(10,6))
plt.plot(df['Date_Time'], df['avg_temp'], label='Average Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (C)')
plt.title('Temperature Trend Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Create a boxplot for temperature distributions by month
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Month', y='avg_temp')
plt.xlabel('Month')
plt.ylabel('Temperature (C)')
plt.title('Monthly Temperature Distribution')
plt.show()

# Calculate monthly average temperature
monthly_avg_temp = df.groupby('Month')['avg_temp'].mean()

# Plot monthly average temperature
plt.figure(figsize=(10,6))
monthly_avg_temp.plot(kind='bar', color='skyblue')
plt.xlabel('Month')
plt.ylabel('Average Temperature (C)')
plt.title('Average Temperature by Month')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# Calculate correlation matrix
corr = df[['Temperature_C', 'Humidity', 'Wind_Speed']].corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Weather Features')
plt.show()

# Plot histogram of temperature distribution
plt.figure(figsize=(10,6))
sns.histplot(df['avg_temp'], bins=20, kde=True, color='skyblue')
plt.xlabel('Temperature (C)')
plt.ylabel('Frequency')
plt.title('Temperature Distribution')
plt.grid(True)
plt.show()

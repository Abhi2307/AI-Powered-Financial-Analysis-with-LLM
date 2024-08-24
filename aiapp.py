import yfinance as yf
import pandas as pd
# Download historical stock data for Apple (AAPL)
data = yf.download('AAPL', start='2015-01-01', end='2024-01-01')

# Display the first few rows of the dataset
print(data.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Apple Closing Price')
plt.title('Apple Stock Price (2015-2024)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Moving Averages
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['MA_200'] = data['Close'].rolling(window=200).mean()

# Volatility
data['Volatility'] = data['Close'].rolling(window=50).std()

# Relative Strength Index (RSI)
delta = data['Close'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Drop missing values
data.dropna(inplace=True)

print(data[['MA_50', 'MA_200', 'Volatility', 'RSI']].head())

from sklearn.preprocessing import MinMaxScaler

# Select features for normalization
features = ['Close', 'MA_50', 'MA_200', 'Volatility', 'RSI']
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features])

# Convert the scaled data back to a DataFrame
data_scaled = pd.DataFrame(data_scaled, columns=features, index=data.index)

print(data_scaled.head())

import numpy as np

# Prepare the data for LSTM
X = []
y = []

# Use 60 days of data to predict the next day's price
for i in range(60, len(data_scaled)):
    X.append(data_scaled.iloc[i-60:i].values)
    y.append(data_scaled.iloc[i, 0])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
train_size = int(X.shape[0] * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10)

from sklearn.metrics import mean_squared_error

# Make predictions on the test data
predictions = model.predict(X_test)

# Reverse the scaling for comparison
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], data_scaled.shape[1] - 1))), axis=1))[:, 0]
y_test = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], data_scaled.shape[1] - 1))), axis=1))[:, 0]

# Calculate MSE
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Plot the predictions vs. actual prices
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

import streamlit as st

# Dashboard Title
st.title('AI-Powered Financial Analysis')

# User input for stock selection
stock = st.text_input('Enter Stock Ticker', 'AAPL')

# Display stock data
st.line_chart(data['Close'])

# Prediction button
if st.button('Predict'):
    # Use the model to predict future prices
    input_data = X_test[-1].reshape((1, 60, len(features)))
    prediction = model.predict(input_data)
    prediction = scaler.inverse_transform(np.concatenate((prediction, np.zeros((1, data_scaled.shape[1] - 1))), axis=1))[:, 0]
    st.write(f'Predicted Next Day Closing Price: ${prediction[0]:.2f}')


from transformers import pipeline

# Load a pre-trained sentiment analysis model
sentiment_analysis = pipeline('sentiment-analysis')

# Example headlines dataset
headlines_data = {
    'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'Headlines': [
        "Apple's stock surges after record-breaking sales report",
        "Apple faces scrutiny over data privacy practices",
        "New iPhone model expected to boost Apple's Q1 earnings"
    ]
}

headlines_df = pd.DataFrame(headlines_data)
headlines_df['Date'] = pd.to_datetime(headlines_df['Date'])

# Assuming you have a 'data' DataFrame that has stock data with a 'Date' index
# Merge headlines with stock data on the date
data = data.merge(headlines_df, left_index=True, right_on='Date', how='left')
print(data)

# Perform sentiment analysis on each headline
data['Sentiment'] = data['Headlines'].apply(lambda x: sentiment_analysis(x)[0]['score'] if pd.notna(x) else None)

# Optionally, convert sentiment to positive or negative score
data['Sentiment'] = data['Headlines'].apply(
    lambda x: sentiment_analysis(x)[0]['score'] if pd.notna(x) and sentiment_analysis(x)[0]['label'] == 'POSITIVE' else 
    -sentiment_analysis(x)[0]['score'] if pd.notna(x) else None
)

# Print the DataFrame to see the results
print(data[['Date', 'Headlines', 'Sentiment']])

# Include sentiment in the list of features (assuming you have these features in your 'data' DataFrame)
features = ['Close', 'MA_50', 'MA_200', 'Volatility', 'RSI', 'Sentiment']


import streamlit as st

# Dashboard Title
st.title('AI-Powered Financial Analysis with LLM')

# User input for stock selection
tickers = ['AAPL', 'GOOGL', 'MSFT']

for i, ticker in enumerate(tickers):
    st.text_input(f'Enter Stock Ticker {i+1}', ticker, key=f'stock_{i}')
#stock = st.text_input('Enter Stock Ticker', 'AAPL')

# Display stock data
st.line_chart(data['Close'])

# Sentiment analysis
if st.button('Analyze Sentiment'):
    example_headline = "Apple's stock surges after record-breaking sales report"
    sentiment = sentiment_analysis(example_headline)[0]
    st.write(f"Headline: {example_headline}")
    st.write(f"Sentiment: {sentiment['label']}, Confidence: {sentiment['score']:.2f}")

# Prediction button with a unique key
if st.button('Predict', key='predict_button'):
    # Check the shape of the last element in X_test
    st.write(f"Shape of X_test[-1]: {X_test[-1].shape}")

    input_data = X_test[-1].reshape((1, 60, 5))
    prediction = model.predict(input_data)
    prediction = scaler.inverse_transform(np.concatenate(
        (prediction, np.zeros((1, data_scaled.shape[1] - 1))), axis=1))[:, 0]
    st.write(f'Predicted Next Day Closing Price: ${prediction[0]:.2f}')










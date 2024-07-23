import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf
import json
import numpy as np

# Load scheme codes from code.json
with open('code.json') as f:
    scheme_codes = json.load(f)['schemes']

# Streamlit app setup
st.title('Mutual Fund NAV Prediction')

# Select scheme
scheme_name = st.selectbox('Select Mutual Fund Scheme', list(scheme_codes.keys()))

if scheme_name:
    scheme_code = scheme_codes[scheme_name]

    # Load historical data using yfinance
    @st.cache
    def load_data(scheme_code):
        data = yf.download(scheme_code)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(scheme_code)
    data_load_state.text('Loading data...done!')

    # Display data
    st.subheader('Raw data')
    st.write(data.tail())

    # Preprocess data for modeling
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    # Feature and target setup
    X = data[['Close']]
    y = data['Returns']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # Model prediction
    y_pred = model.predict(X_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    st.subheader(f'Mean Squared Error: {mse:.4f}')

    # Plot results
    st.subheader('Prediction vs Actual')
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, y_pred, label='Prediction')
    plt.legend()
    st.pyplot(plt)

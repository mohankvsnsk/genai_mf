import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
from mftool import Mftool
from transformers import pipeline

# Load a pre-trained conversational model
chatbot = pipeline('conversational', model='microsoft/DialoGPT-medium')

# Function to preprocess data
def preprocess_data(df):
    data = df['nav'].values
    data = data.reshape(-1, 1)
    return data

# Function to create dataset for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Function to build and train LSTM model
def build_and_train_lstm(trainX, trainY, time_step, epochs=50, batch_size=64):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Function to forecast using LSTM model
def forecast_lstm(model, data, time_step):
    temp_input = list(data)
    lst_output = []
    n_steps = time_step
    i = 0
    while i < 30:
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            temp_input = temp_input[1:]
            lst_output.append(yhat[0][0])
            i += 1
        else:
            x_input = np.array(temp_input)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
            i += 1
    return lst_output

# Function to get mutual fund data
def get_mutual_fund_data(scheme_code):
    m = Mftool()
    scheme_data = m.get_scheme_historical_nav(scheme_code)
    df = pd.DataFrame(scheme_data['data'])
    df['nav'] = df['nav'].astype(float)
    df = df.sort_values(by='date')
    return df

# Streamlit app
st.title('Mutual Fund NAV Forecasting with Generative AI')

# Display chatbot interface
st.write("### Ask the chatbot about mutual fund predictions")

user_input = st.text_input('You: ', 'Tell me the NAV prediction for scheme code 118551')
if st.button('Send'):
    if 'scheme code' in user_input.lower():
        scheme_code = ''.join(filter(str.isdigit, user_input))
        if scheme_code:
            df = get_mutual_fund_data(scheme_code)

            if df.empty:
                st.write("No data found for the given scheme code.")
            else:
                data = preprocess_data(df)
                time_step = 100
                X, Y = create_dataset(data, time_step)
                X = X.reshape(X.shape[0], X.shape[1], 1)
                train_size = int(len(X) * 0.7)
                test_size = len(X) - train_size
                trainX, testX = X[0:train_size], X[train_size:len(X)]
                trainY, testY = Y[0:train_size], Y[train_size:len(Y)]
                model = build_and_train_lstm(trainX, trainY, time_step)
                lst_output = forecast_lstm(model, data[-time_step:], time_step)

                st.write("### Last 100 Days NAV and Next 30 Days Forecast")
                df_30 = pd.DataFrame(data[-100:], columns=['NAV'])
                df_30['Forecast'] = np.nan
                df_30.loc[len(df_30)] = [np.nan, np.nan]
                df_30['Forecast'][-30:] = lst_output

                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(data=df_30, ax=ax)
                plt.xlabel('Days')
                plt.ylabel('NAV')
                plt.title(f"Forecasting for Scheme Code: {scheme_code}")
                st.pyplot(fig)
        else:
            st.write("Please provide a valid scheme code.")
    else:
        conversation = chatbot(user_input)
        st.write(f"Bot: {conversation[-1]['generated_text']}")

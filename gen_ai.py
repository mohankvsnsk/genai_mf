import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from mftool import Mftool

# Function to load and preprocess data
def load_data(scheme_code):
    mf = Mftool()
    data = mf.get_scheme_historical_nav(scheme_code)
    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['date'])
    df['nav'] = df['nav'].astype(float)
    df = df.sort_values('date')
    return df

# Function to create features
def create_features(df):
    df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())
    return df

# Function to split data
def split_data(df):
    X = df[['date_ordinal']]
    y = df['nav']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Load and preprocess data
st.title('Mutual Fund NAV Prediction')
scheme_code = st.text_input('Enter Scheme Code', '118885')  # Default example scheme code

if scheme_code:
    df = load_data(scheme_code)
    df = create_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Train the model
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Mean Squared Error: {mse}')

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.scatter(df['date'], df['nav'], label='Actual NAV')
    plt.plot(df['date'], model.predict(df[['date_ordinal']]), color='red', label='Predicted NAV')
    plt.xlabel('Date')
    plt.ylabel('NAV')
    plt.title('Mutual Fund NAV Prediction')
    plt.legend()
    st.pyplot(plt)

    # Display data
    st.write('### Data')
    st.write(df)

    # User input for prediction
    user_date = st.date_input('Select a date for prediction')
    user_date_ordinal = user_date.toordinal()
    user_nav_pred = model.predict([[user_date_ordinal]])
    st.write(f'Predicted NAV for {user_date}: {user_nav_pred[0]}')

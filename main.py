import streamlit as st
import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import re
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API credentials
API_KEY = 'AIzaSyBWq6X2brvlDmLPOHZ31q340Cehb53UD9Y'
SEARCH_ENGINE_IDS = [
    '438ca7820aa8e478e',
    '2419aa9656883446d',
    'f63cc52276c684e70',
    '70c4dde961a3241a7',
    '15373bec9eaea40c4',
    'a3db008b1b53448be',
    '220a76cb2401a42c9',
    'e2c88d246a1c44798',
    '1724b8719b618429d',
    'f15af5e48a15f40de'
]

# File path for user credentials
USER_CREDENTIALS_FILE = 'user_credentials.json'

# File path for prediction history (JSON format)
PREDICTION_HISTORY_FILE = 'prediction_history.json'

# Load user credentials from JSON file
def load_user_credentials():
    if os.path.exists(USER_CREDENTIALS_FILE):
        with open(USER_CREDENTIALS_FILE, 'r') as f:
            return json.load(f)
    return {'admin': 'admin123'}  # Default admin credentials

user_credentials = load_user_credentials()

# Load prediction history from JSON file
if os.path.exists(PREDICTION_HISTORY_FILE):
    with open(PREDICTION_HISTORY_FILE, 'r') as f:
        prediction_history = json.load(f)
else:
    prediction_history = {}

def save_user_credentials():
    with open(USER_CREDENTIALS_FILE, 'w') as f:
        json.dump(user_credentials, f, indent=4)

def save_prediction_result(username, product_name, sales_predictions):
    if username not in prediction_history:
        prediction_history[username] = []

    prediction_history[username].append({
        'product_name': product_name,
        'sales_predictions': sales_predictions
    })

    # Save updated prediction history to JSON file
    with open(PREDICTION_HISTORY_FILE, 'w') as f:
        json.dump(prediction_history, f, indent=4)

def fetch_realtime_data(query, search_engine_ids):
    results = []
    for cx in search_engine_ids:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={cx}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            json_data = response.json()

            if 'items' in json_data:
                items = json_data['items']
                if items:
                    item = items[0]
                    product_name = item.get('title', 'Unknown Product')
                    snippet = item.get('snippet', '')
                    sales_data = extract_sales_data(snippet)
                    if sales_data and len(sales_data) >= 4:
                        sales_m1, sales_m2, sales_m3, sales_m4 = sales_data[:4]
                        data = {
                            'product_name': product_name,
                            'sales_m1': sales_m1,
                            'sales_m2': sales_m2,
                            'sales_m3': sales_m3,
                            'sales_m4': sales_m4,
                            'realtime_feature': (sales_m1 + sales_m2 + sales_m3 + sales_m4) / 4,
                            'cx': cx
                        }
                        results.append(data)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data from API: {str(e)}")

    return pd.DataFrame(results)

def extract_sales_data(snippet):
    try:
        sales_pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d{1,2})?)'
        sales_data = re.findall(sales_pattern, snippet)
        sales_data = [float(s.replace(',', '')) for s in sales_data]
        return sales_data
    except ValueError:
        return None

def preprocess_data(data):
    try:
        processed_data = data.drop(columns=['product_name', 'realtime_feature', 'cx'])
        processed_data = processed_data.apply(pd.to_numeric)
        return processed_data
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return None

def train_lgb_model(data, target_variable):
    try:
        X = data[['sales_m1', 'sales_m2', 'sales_m3', 'sales_m4']]
        y = data[target_variable]
        model = lgb.LGBMRegressor()
        model.fit(X, y)
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        return model, mae, mse, rmse, r2
    except Exception as e:
        logging.error(f"Error training LightGBM model: {e}")
        return None, None, None, None, None

def train_rf_model(data, target_variable):
    try:
        X = data[['sales_m1', 'sales_m2', 'sales_m3', 'sales_m4']]
        y = data[target_variable]
        model = RandomForestRegressor()
        model.fit(X, y)
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        return model, mae, mse, rmse, r2
    except Exception as e:
        logging.error(f"Error training Random Forest model: {e}")
        return None, None, None, None, None

def save_model(model, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Trained model saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def predict_sales(sales_m1, sales_m2, sales_m3, sales_m4, model):
    try:
        sales_m1 = float(sales_m1)
        sales_m2 = float(sales_m2)
        sales_m3 = float(sales_m3)
        sales_m4 = float(sales_m4)
        future_sales_m2 = model.predict([[sales_m1, sales_m2, 0, 0]])
        future_sales_m3 = model.predict([[sales_m1, sales_m2, sales_m3, 0]])
        future_sales_m4 = model.predict([[sales_m1, sales_m2, sales_m3, sales_m4]])
        return [future_sales_m2.tolist(), future_sales_m3.tolist(), future_sales_m4.tolist()]
    except Exception as e:
        logging.error(f"Error predicting sales: {e}")
        return [None, None, None]

def visualize_sales_prediction(predictions, month_names):
    months = month_names
    plt.figure(figsize=(10, 6))
    plt.plot(months, predictions, marker='o', linestyle='-')
    plt.title('Predicted Sales for Next 3 Months')
    plt.xlabel('Months')
    plt.ylabel('Sales (Monetary)')
    plt.grid(True)
    plt.xticks(months)
    st.pyplot(plt)

def show_history():
    st.subheader("Prediction History")

    if st.session_state.username in prediction_history and prediction_history[st.session_state.username]:
        for idx, prediction in enumerate(prediction_history[st.session_state.username]):
            st.write(f"Prediction {idx + 1}:")
            st.write(f"Product Name: {prediction['product_name']}")
            st.write("Sales Predictions:")
            st.write(prediction['sales_predictions'])
            st.markdown("---")
    else:
        st.write("No prediction history available.")

def main():
    st.title("Real-time Sales Prediction")

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""

    def login(username):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.page = 'predict'

    def logout():
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.page = 'login'

    def register_user(username, password):
        if username in user_credentials:
            st.error("Username already exists.")
        else:
            user_credentials[username] = password
            save_user_credentials()  # Save updated credentials to file
            st.success("User registered successfully. Please log in.")

    def show_history_sidebar():
        st.sidebar.subheader("User Actions")
        if st.sidebar.button("View Prediction History"):
            st.session_state.page = 'history'

    def show_register():
        st.session_state.page = 'register'

    def show_login():
        st.session_state.page = 'login'

    if st.session_state.page == 'register':
        st.subheader("Register")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type='password')

        if st.button("Register"):
            register_user(new_username, new_password)
            # Reload user credentials after registration
            global user_credentials
            user_credentials = load_user_credentials()
            show_login()

        if st.button("Back to Login"):
            show_login()

    elif st.session_state.page == 'login':
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')

        if st.button("Login"):
            if username in user_credentials and user_credentials[username] == password:
                login(username)
            else:
                st.error("Invalid username or password.")

        if st.button("Register"):
            show_register()

    elif st.session_state.page == 'predict' and st.session_state.logged_in:
        st.write(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            logout()

        product_name = st.text_input("Enter product name:").strip()
        sales_m1 = st.text_input("Enter sales for month 1:").strip()
        sales_m2 = st.text_input("Enter sales for month 2:").strip()
        sales_m3 = st.text_input("Enter sales for month 3:").strip()
        sales_m4 = st.text_input("Enter sales for month 4:").strip()

        if st.button("Predict Sales"):
            if not (product_name and sales_m1 and sales_m2 and sales_m3 and sales_m4):
                st.error("Please enter all the required fields.")
                return

            query = f"{product_name} sales data"
            realtime_data = fetch_realtime_data(query, SEARCH_ENGINE_IDS)

            if realtime_data.empty:
                st.error("No data fetched from API.")
                return

            processed_data = preprocess_data(realtime_data)

            if processed_data is None or processed_data.empty:
                st.error("No data to preprocess.")
                return

            target_variable = 'sales_m4'

            lgb_model, lgb_mae, lgb_mse, lgb_rmse, lgb_r2 = train_lgb_model(processed_data, target_variable)

            if lgb_model is None:
                st.error("LightGBM model training was unsuccessful.")
                return

            save_model(lgb_model, 'trained_lgb_model.pkl')

            future_sales_lgb = predict_sales(
                sales_m1,
                sales_m2,
                sales_m3,
                sales_m4,
                lgb_model
            )

            if any(future_sales_lgb_item is None for future_sales_lgb_item in future_sales_lgb):
                st.error("Failed to predict future sales with LightGBM model.")
                return

            st.subheader("Predicted Sales for next 3 months (LightGBM):")
            st.write(future_sales_lgb)

            visualize_sales_prediction(future_sales_lgb, ['Month 2', 'Month 3', 'Month 4'])

            rf_model, rf_mae, rf_mse, rf_rmse, rf_r2 = train_rf_model(processed_data, target_variable)

            if rf_model is None:
                st.error("Random Forest model training was unsuccessful.")
                return

            save_model(rf_model, 'trained_rf_model.pkl')

            future_sales_rf = predict_sales(
                sales_m1,
                sales_m2,
                sales_m3,
                sales_m4,
                rf_model
            )

            if any(future_sales_rf_item is None for future_sales_rf_item in future_sales_rf):
                st.error("Failed to predict future sales with Random Forest model.")
                return

            st.subheader("Predicted Sales for next 3 months (Random Forest):")
            st.write(future_sales_rf)

            visualize_sales_prediction(future_sales_rf, ['Month 2', 'Month 3', 'Month 4'])

            save_prediction_result(st.session_state.username, product_name, {
                'LightGBM': future_sales_lgb,
                'Random Forest': future_sales_rf
            })

            show_history_sidebar()

    elif st.session_state.page == 'history' and st.session_state.logged_in:
        show_history()

if __name__ == "__main__":
    main()










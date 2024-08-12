import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.impute import SimpleImputer

# Dictionaries for encoding
store_location_dict = {
    'Amritsar': 0, 'Aurangabad': 1, 'Bangalore': 2, 'Bhopal': 3, 'Chandigarh': 4, 'Guntur': 5, 'Hyderabad': 6,
    'Indore': 7, 'Jalandhar': 8, 'Jaipur': 9, 'Karnal': 10, 'Kochi': 11, 'Ludhiana': 12, 'Lucknow': 13, 'Meerut': 14,
    'Rajahmundry': 15, 'Sangli': 16, 'Tirupati': 17, 'Vijayawada': 18, 'Vizag ': 19
}

product_name_dict = {
    'Basmati Rice': 0, 'Turmeric Powder': 1, 'Red Chilli Powder': 2, 'Cumin Seeds': 3, 'Coriander Powder': 4,
    'Garam Masala': 5, 'Toor Dal': 6, 'Chana Dal': 7, 'Urad Dal': 8, 'Moong Dal': 9, 'Jaggery': 10, 'Ghee': 11,
    'Paneer': 12, 'Coconut Oil': 13, 'Mustard Oil': 14, 'Aam Papad': 15, 'Mango Pickle': 16, 'Papad': 17,
    'Idli Rice': 18, 'Sambar Powder': 19, 'Aloo Bhujia': 20, 'Kaju Katli': 21, 'Besan': 22, 'Aata (Wheat Flour)': 23,
    'Mango Juice': 24, 'Lemon Pickle': 25, 'Coconut Chutney': 26, 'Rasgulla': 27, 'Gulab Jamun Mix': 28,
    'Lassi': 29, 'Bhel Puri Mix': 30, 'Pav Bhaji Masala': 31, 'Chole Masala': 32, 'Tamarind Paste': 33,
    'Ready-to-eat Curry Packs': 34, 'Parle-G Biscuits': 35, 'Thums Up': 36, 'Frooti': 37, 'Hajmola Tablets': 38,
    "Haldiram's Namkeen": 39, 'Chyawanprash': 40
}

event_dict = {
    'No event': 0, 'Sunday': 1, 'Raksha Bandhan': 2, 'Independence Day': 3, 'Diwali': 3, 'Christmas': 3,
    'Regional Festival': 2, 'Minor Event': 1, 'Republic Day': 3, 'Pongal': 2, 'Eid-ul-Fitr': 3, 'Eid-ul-Adha': 3,
    'Gudi Padwa': 2, 'Navratri': 3, 'Baisakhi': 2, 'Onam': 2, 'Guru Nanak Jayanti': 3, 'Makar Sankranti': 2,
    'Holi': 3, 'Durga Puja': 3, 'Ganesh Chaturthi': 2, 'Janmashtami': 2, 'Ram Navami': 2, 'Mahashivratri': 2,
    'Dussehra': 3, 'Kumbh Mela': 3, 'Sankranti': 2
}

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, store_location_dict, product_name_dict, event_dict):
        self.store_location_dict = store_location_dict
        self.product_name_dict = product_name_dict
        self.event_dict = event_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Use dayfirst=True to correctly parse dates in the format "dd-mm-yyyy"
        X['Date'] = pd.to_datetime(X['Date'], dayfirst=True)
        X['Year'] = X['Date'].dt.year
        X['Month'] = X['Date'].dt.month
        X['Day_of_Week'] = X['Date'].dt.dayofweek
        X = X.drop(columns=['Date'])

        X['Store Location'] = X['Store Location'].map(self.store_location_dict)
        X['Product Name & Brand'] = X['Product Name & Brand'].map(self.product_name_dict)
        X['Weather Pattern'] = X['Weather Pattern'].map({'Rainy': 0, 'Cloudy': 1, 'Windy': 2, 'Sunny': 3})
        X['Event'] = X['Event'].map(self.event_dict)
        X['Advertising'] = X['Advertising'].map({'Low': 0, 'Medium': 1, 'High': 2})

        return X


def load_data():
    df = pd.read_csv('wallmart_data.csv')
    X = df.drop(columns=['Sales in Week (Target)'])
    y = df['Sales in Week (Target)']
    return X, y

def train_model(X, y):
    # Add SimpleImputer to handle NaN values
    preprocessor = Pipeline(steps=[
        ('data_preprocessor', DataPreprocessor(store_location_dict, product_name_dict, event_dict)),
        ('imputer', SimpleImputer(strategy='mean'))  # Replace NaNs with the mean of the column
    ])

    model = GradientBoostingRegressor()

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    with open('model_pipeline.pkl', 'wb') as file:
        pickle.dump(pipeline, file)

def predict():
    X, y = load_data()
    train_model(X, y)  # Retrain the model with current data and environment

    # UI elements
    st.title("Sales Prediction Dashboard")
    date = st.date_input("Select the date:", value=datetime.now().date())
    product_name_display = st.selectbox("Select the product name:", X['Product Name & Brand'].unique())
    weather_pattern_display = st.selectbox("Select the weather pattern:", X['Weather Pattern'].unique())
    event_display = st.selectbox("Select the event:", X['Event'].unique())
    promotion_discount = st.slider("Select the promotion discount (%):", min_value=0, max_value=100, value=0)

    if st.button("Predict Sales"):
        columns = [
            'Date', 'Product Name & Brand', 'Past Week Sales', 'Weather Pattern',
            'Event', 'Event_Impact_Level', 'Promotions (Discount)',
            'Competitor Action (Discount)', 'Advertising', 'Economic Indicator',
            'Store Location'
        ]

        data_list = [
            date,
            product_name_display,  
            2070,  # Example Past Week Sales
            weather_pattern_display,  # Weather Pattern
            event_display,  # Event
            3,  # Event Impact Level
            promotion_discount,  # Promotions (Discount)
            20,  # Competitor Action (Discount)
            'Low',  # Advertising
            4,  # Economic Indicator
            'Sangli'  # Store Location
        ]

        df = pd.DataFrame([data_list], columns=columns)

        # Load the trained model
        with open('model_pipeline.pkl', 'rb') as file:
            model = pickle.load(file)

        predicted_sales = model.predict(df)
        st.write(f"*Predicted Sales for {product_name_display} on:* {predicted_sales[0]:.2f}")

if __name__ == "__main__":
    predict()

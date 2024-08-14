import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.impute import SimpleImputer

# Load locations map from the provided CSV
locations_df = pd.read_csv('locations_map.csv')
store_location_dict = {row[0]: row[1] for _, row in locations_df.iterrows()}

# Load the main datasets
wallmart_df = pd.read_csv('wallmart_data.csv')
warehouse_store_mapping_df = pd.read_csv('final_warehouse_store_mapping.csv')
product_name_map_df = pd.read_csv('product_name_map.csv')

# Dictionaries for encoding other features
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

def get_historical_values(date, product_name, store_location):
    one_year_ago = date - timedelta(days=365)
    filtered_df = wallmart_df[
        (pd.to_datetime(wallmart_df['Date'], dayfirst=True) == one_year_ago) &
        (wallmart_df['Product Name & Brand'] == product_name) &
        (wallmart_df['Store Location'] == store_location)
    ]

    if not filtered_df.empty:
        past_week_sales = filtered_df['Past Week Sales'].values[0]
        event_impact_level = filtered_df['Event_Impact_Level'].values[0]
        competitor_action_discount = filtered_df['Competitor Action (Discount)'].values[0]
        advertising = filtered_df['Advertising'].values[0]
        economic_indicator = filtered_df['Economic Indicator'].values[0]
    else:
        past_week_sales = 0
        event_impact_level = 0
        competitor_action_discount = 0
        advertising = 'Low'
        economic_indicator = 0

    return past_week_sales, event_impact_level, competitor_action_discount, advertising, economic_indicator

def check_stock_availability(product_name, store_location):
    # Map the product name to its category code
    category_code = product_name_map_df.loc[product_name_map_df['Product Name & Brand'] == product_name, 'Category Code'].values[0]
    
    # Filter the warehouse-store mapping for the selected store
    store_df = warehouse_store_mapping_df[warehouse_store_mapping_df['Store'] == store_location]
    
    # Calculate the sum of the available stock for the product category across all relevant warehouses
    stock_column = f'Product_{category_code}_Current_Availability'
    total_stock = store_df[stock_column].sum()
    
    # Define a threshold for stock availability (e.g., required quantity)
    required_quantity = 500  # Example threshold
    
    # Check if the stock is sufficient
    if total_stock >= required_quantity:
        return total_stock, "Sufficient stock available"
    else:
        return total_stock, "Insufficient stock available"

def train_model(X, y):
    preprocessor = Pipeline(steps=[
        ('data_preprocessor', DataPreprocessor(store_location_dict, product_name_dict, event_dict)),
        ('imputer', SimpleImputer(strategy='mean'))
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
    train_model(X, y)

    st.title("Sales Prediction Dashboard")
    date = st.date_input("Select the date:", value=datetime.now().date())
    product_name_display = st.selectbox("Select the product name:", X['Product Name & Brand'].unique())
    weather_pattern_display = st.selectbox("Select the weather pattern:", X['Weather Pattern'].unique())
    event_display = st.selectbox("Select the event:", X['Event'].unique())
    store_location_display = st.selectbox("Select the store location:", list(store_location_dict.keys()))
    promotion_discount = st.slider("Select the promotion discount (%):", min_value=0, max_value=100, value=0)

    # Get historical values from 1 year ago
    past_week_sales, event_impact_level, competitor_action_discount, advertising, economic_indicator = get_historical_values(date, product_name_display, store_location_display)

    # Check stock availability
    total_stock, stock_status = check_stock_availability(product_name_display, store_location_display)
    st.write(f"Stock Status: {stock_status} (Total Stock: {total_stock})")

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
            past_week_sales,
            weather_pattern_display,
            event_display,
            event_impact_level,
            promotion_discount,
            competitor_action_discount,
            advertising,
            economic_indicator,
            store_location_display
        ]

        df = pd.DataFrame([data_list], columns=columns)

        # Load the trained model
        with open('model_pipeline.pkl', 'rb') as file:
            model = pickle.load(file)

        # Make the prediction
        predicted_sales = model.predict(df)
        st.write(f"*Predicted Sales for {product_name_display} on {date}:* {predicted_sales[0]:.2f}")

if __name__ == "__main__":
    predict()


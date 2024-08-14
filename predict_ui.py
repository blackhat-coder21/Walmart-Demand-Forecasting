import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from geopy.distance import geodesic
import folium
from streamlit_folium import folium_static
import heapq

# Load the provided files
warehouse_store_mapping_df = pd.read_csv('final_warehouse_store_mapping.csv')
locations_map_df = pd.read_csv('locations_map.csv')
product_name_map_df = pd.read_csv('product_name_map.csv')
wallmart_df = pd.read_csv('wallmart_data.csv')

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

# Convert locations_map to dictionary for lookups
store_location_dict = {row[0]: row[1] for _, row in locations_map_df.iterrows()}

# Class for data preprocessing
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

def heuristic(node, goal):
    return geodesic(node, goal).kilometers

def a_star_algorithm(start_coords, goal_coords, graph):
    open_list = []
    heapq.heappush(open_list, (0, start_coords))
    came_from = {}
    g_score = {start_coords: 0}
    f_score = {start_coords: heuristic(start_coords, goal_coords)}
    
    while open_list:
        current = heapq.heappop(open_list)[1]

        if current == goal_coords:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_coords)
            return path[::-1]  # Return reversed path

        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + geodesic(current, neighbor).kilometers

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_coords)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
    return None  # No path found

def create_graph(store_coords, warehouse_coords):
    graph = {}
    graph[store_coords] = warehouse_coords
    for warehouse in warehouse_coords:
        graph[warehouse] = [store_coords]
    return graph

def check_stock_availability(product_name, store_location):
    # Map the product name to its category code
    category_code = product_name_map_df.loc[product_name_map_df['Product Name & Brand'] == product_name, 'Category Code'].values[0]
    
    # Filter the warehouse-store mapping for the selected store
    store_df = warehouse_store_mapping_df[warehouse_store_mapping_df['Store'] == store_location]
    
    # Calculate the sum of the available stock for the product category across all relevant warehouses
    stock_column = f'Product_{category_code}_Current_Availability'
    total_stock = store_df[stock_column].sum()
    
    return total_stock

def find_nearest_warehouses_a_star(store_location, category_code, required_stock):
    # Find the latitude and longitude of the store
    store_lat = warehouse_store_mapping_df.loc[warehouse_store_mapping_df['Store'] == store_location, 'Store_Latitude'].values[0]
    store_long = warehouse_store_mapping_df.loc[warehouse_store_mapping_df['Store'] == store_location, 'Store_Longitude'].values[0]
    store_coords = (store_lat, store_long)
    
    # Get all warehouse coordinates and names
    warehouse_coords_names = [
        ((row['Warehouse_Latitude'], row['Warehouse_Longitude']), row['Warehouse'])
        for _, row in warehouse_store_mapping_df.iterrows()
    ]
    
    graph = create_graph(store_coords, [wc[0] for wc in warehouse_coords_names])
    
    # Find paths to all warehouses using A* and calculate their distance
    warehouse_distances = []
    total_stock_collected = 0

    for warehouse_coords, warehouse_name in warehouse_coords_names:
        path = a_star_algorithm(store_coords, warehouse_coords, graph)
        if path:
            distance = sum(geodesic(path[i], path[i + 1]).kilometers for i in range(len(path) - 1))
            stock_column = f'Product_{category_code}_Current_Availability'
            stock = warehouse_store_mapping_df.loc[
                (warehouse_store_mapping_df['Warehouse_Latitude'] == warehouse_coords[0]) &
                (warehouse_store_mapping_df['Warehouse_Longitude'] == warehouse_coords[1]), stock_column
            ].values[0]

            if stock > 0:
                warehouse_distances.append((warehouse_name, warehouse_coords, distance, stock))
                total_stock_collected += stock

            # Stop when the required stock is fulfilled
            if total_stock_collected >= required_stock:
                break
    
    # Sort by distance and return only the necessary warehouses
    nearest_warehouses = sorted(warehouse_distances, key=lambda x: x[2])
    return nearest_warehouses, total_stock_collected

def plot_warehouses_map(store_coords, nearest_warehouses):
    folium_map = folium.Map(location=store_coords, zoom_start=10)
    
    folium.Marker(store_coords, popup="Store Location", icon=folium.Icon(color='blue')).add_to(folium_map)
    
    for warehouse in nearest_warehouses:
        folium.Marker(
            location=warehouse[1],
            popup=f"{warehouse[0]}: {warehouse[3]} units",
            icon=folium.Icon(color='green')
        ).add_to(folium_map)
    
    return folium_map

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

    past_week_sales, event_impact_level, competitor_action_discount, advertising, economic_indicator = get_historical_values(date, product_name_display, store_location_display)

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

        # Calculate the required stock based on predicted sales
        required_stock = predicted_sales[0]

        # Check available stock at the store location
        total_stock = check_stock_availability(product_name_display, store_location_display)
        st.write(f"Total Stock Available at Store: {total_stock} units")

        # If required stock is more than available stock, find nearest warehouses
        if required_stock > total_stock:
            st.warning("Predicted sales exceed available stock!")
            category_code = product_name_map_df.loc[product_name_map_df['Product Name & Brand'] == product_name_display, 'Category Code'].values[0]
            nearest_warehouses, collected_stock = find_nearest_warehouses_a_star(store_location_display, category_code, required_stock - total_stock)

            # Display information about the nearest warehouses
            st.write("Nearest Warehouses (using A* Algorithm):")
            for warehouse in nearest_warehouses:
                st.write(f"{warehouse[0]}, Location: ({warehouse[1][0]}, {warehouse[1][1]}), Distance: {warehouse[2]:.2f} km, Stock: {warehouse[3]} units")
            st.write(f"Total stock collected from nearest warehouses: {collected_stock} units")

            # Plot the warehouses on a map
            store_lat = warehouse_store_mapping_df.loc[warehouse_store_mapping_df['Store'] == store_location_display, 'Store_Latitude'].values[0]
            store_long = warehouse_store_mapping_df.loc[warehouse_store_mapping_df['Store'] == store_location_display, 'Store_Longitude'].values[0]
            store_coords = (store_lat, store_long)
            folium_map = plot_warehouses_map(store_coords, nearest_warehouses)
            folium_static(folium_map)  # Display the map in Streamlit

if __name__ == "__main__":
    predict()



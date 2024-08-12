import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from predict_ui import DataPreprocessor
from model_performance_module import model_performance
from predict_ui import predict

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/Ankit/Downloads/archive/wallmart_data.csv')
    return df

df = load_data()

# Preprocess data
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['State'] = df['Store Location'].apply(lambda x: x.split()[-1])  # Assuming last word in location is the state


# Create a dictionary to map month numbers to month names
month_map = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

df['Month_Name'] = df['Month'].map(month_map)

# Load additional datasets
@st.cache_data
def load_location_data():
    locations_map = pd.read_csv('C:/Users/Ankit/Downloads/archive/locations_map.csv')
    warehouse_store_mapping = pd.read_csv('C:/Users/Ankit/Downloads/archive/final_warehouse_store_mapping.csv')

    # Create a mapping of stores to their coordinates
    store_coordinates = {
        row['Store']: (row['Store_Latitude'], row['Store_Longitude'])
        for index, row in warehouse_store_mapping[['Store', 'Store_Latitude', 'Store_Longitude']].drop_duplicates().iterrows()
    }

    return locations_map, warehouse_store_mapping, store_coordinates

locations_map, warehouse_store_mapping, store_coordinates = load_location_data()

# Streamlit Sidebar
st.sidebar.title("Filters")

# Select tab for different analyses
analysis_tab = st.sidebar.selectbox("Select Analysis", ["Sales Analysis", "Model Performance", "Predict"])

if analysis_tab == "Sales Analysis":
    # Sales Analysis Section
    st.sidebar.subheader("Sales Filters")

    # Select month from the sidebar with month names
    month_name = st.sidebar.selectbox("Select Month", df['Month_Name'].unique())
    month = {v: k for k, v in month_map.items()}[month_name]  # Convert back to numeric month

    # Add store location options in the sidebar
    store_option = st.sidebar.selectbox("Select Store", ["All Stores"] + list(store_coordinates.keys()))

    # Streamlit Main Page
    st.title("Sales Prediction Dashboard")
    st.subheader(f"Most Selling Items for {month_name} at {store_option}")

    # Filter data based on user selection
    if store_option == "All Stores":
        filtered_data = df[df['Month'] == month]
    else:
        filtered_data = df[(df['Month'] == month) & (df['Store Location'] == store_option)]

    # Aggregate data to find the most selling items in decreasing order
    most_selling_items = (
        filtered_data.groupby('Product Name & Brand')['Sales in Week (Target)'].sum().sort_values(ascending=False).head(10)
    )

    # Display the most selling items in decreasing order
    st.bar_chart(most_selling_items.sort_values(ascending=True))  # Sort ascending=True to show the largest at the top

    # Additional feature: Show raw data
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Filtered Data")
        st.write(filtered_data)

    # Map display section
    st.subheader("Store Locations and Warehouses")

    # Create a map centered on India
    india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    def add_markers(store_location, store_lat, store_lon, warehouse_data):
        # Add a marker for the store
        folium.Marker(
            location=[store_lat, store_lon],
            popup=f"Store: {store_location}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(india_map)

        # Add markers for each warehouse associated with the store
        for _, warehouse in warehouse_data.iterrows():
            folium.Marker(
                location=[warehouse['Warehouse_Latitude'], warehouse['Warehouse_Longitude']],
                popup=f"Warehouse: {warehouse['Warehouse']}",
                icon=folium.Icon(color='green', icon='home')
            ).add_to(india_map)

            # Draw a line between the store and the warehouse
            folium.PolyLine(
                locations=[[store_lat, store_lon], [warehouse['Warehouse_Latitude'], warehouse['Warehouse_Longitude']]],
                color="red",
                weight=2.5,
                opacity=0.8
            ).add_to(india_map)

    if store_option == "All Stores":
        # Plot all stores and their warehouses
        for store_location, (store_lat, store_lon) in store_coordinates.items():
            warehouse_data = warehouse_store_mapping[warehouse_store_mapping['Store'] == store_location]
            add_markers(store_location, store_lat, store_lon, warehouse_data)
    else:
        # Plot only the selected store and its warehouses
        store_lat, store_lon = store_coordinates[store_option]
        warehouse_data = warehouse_store_mapping[warehouse_store_mapping['Store'] == store_option]
        add_markers(store_option, store_lat, store_lon, warehouse_data)

    # Display the map
    st_folium(india_map, width=800, height=600)

elif analysis_tab == "Model Performance":
    # Call the model performance function
    model_performance()

elif analysis_tab == "Predict":
    predict()

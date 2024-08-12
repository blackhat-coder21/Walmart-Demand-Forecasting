import pandas as pd
import pickle
from datetime import timedelta

# Load the dataset
def load_data():
    df = pd.read_csv('C:/Users/Ankit/Downloads/archive/wallmart_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Load the trained model
def load_model():
    with open('C:/Users/Ankit/Downloads/archive/model_pipeline.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Define the prediction function
def prediction_result(date, product_name, weather_pattern, event, promotion_discount):
    # Load data and model
    df = load_data()
    model = load_model()

    # Filter data for the past year from the selected date
    past_year_date = date - timedelta(days=365)
    relevant_data = df[(df['Date'] >= past_year_date) & (df['Date'] <= date)]

    # Prepare the input data for prediction
    prediction_data = {
        'Date': [date],
        'Product Name & Brand': [product_name],
        'Past Week Sales': 1000,
        'Weather Pattern': [weather_pattern],
        'Event': [event],
        'Event_Impact_Level' : 2,
        'Promotions (Discount)': [promotion_discount],
        'Competitor Action (Discount)' : 0.5,
        'Advertising' : 'High',
        'Economic Indicator' : 5,
        'Store Location' : 'Rajahmundry',
        # Additional fields can be filled with default values or relevant past data
        # For instance, you might want to use average values or the latest values for the missing fields
    }

    # Convert to DataFrame
    prediction_df = pd.DataFrame(prediction_data)

    # Here you would merge or concatenate the prediction_df with relevant_data
    # to create a full feature set if needed, depending on the model's requirements

    # Predict the sales
    predicted_sales = model.predict(prediction_df)

    return predicted_sales[0]

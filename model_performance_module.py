import streamlit as st
from PIL import Image

def model_performance():
    st.sidebar.subheader("Model Selection")

    # Add model selection options in the sidebar
    model_option = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest", "Gradient Boosting Regressor", "Neural Network"])

    st.title("Model Performance Dashboard")
    st.subheader(f"Performance of {model_option}")

    st.write("### Model Performance Metrics")

    # Define metrics for each model
    if model_option == "Linear Regression":
        st.write("Mean Absolute Error: 434.63")
        st.write("Mean Squared Error: 277173.89")
        st.write("Root Mean Squared Error: 526.47")
        st.write("R-squared: 0.23")
        st.write("Model Accuracy: 76.23%")

        # Load and display the associated image
        image = Image.open("C:/Users/Ankit/Downloads/archive/linear_regression.png")
        st.image(image, caption="Predicted vs Observed Sales for Linear Regression")

    elif model_option == "Random Forest":
        st.write("Mean Absolute Error: 1.87")
        st.write("R-squared: 0.85")
        # Optionally, add more data or visuals here

    elif model_option == "Gradient Boosting Regressor":
        st.write("Mean Absolute Error: 53.89")
        st.write("Mean Squared Error: 4029.48")
        st.write("Root Mean Squared Error: 63.48")
        st.write("R-squared: 0.99")
        st.write("Model Accuracy: 97.17%")

        # Load and display the associated image
        image = Image.open("C:/Users/Ankit/Downloads/archive/gradient_boosting.png")
        st.image(image, caption="Predicted vs Observed Sales for Gradient Boosting Regressor")

    elif model_option == "Neural Network":
        st.write("Mean Absolute Error: 1.50")
        st.write("R-squared: 0.90")
        # Optionally, add more data or visuals here

# Run the Streamlit app
if __name__ == "__main__":
    model_performance()

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
        # st.write("Model Accuracy: 76.23%")

        # Load and display the associated image
        image = Image.open("linear_regression.png")
        st.image(image, caption="Predicted vs Observed Sales for Linear Regression")

        st.write("Linear regression resulted in low accuracy because the data exhibits non-linear relationships, and it is incapable of capturing complex interactions between variables as it assumes a linear relationship between the features and the target")

    elif model_option == "Random Forest":
        st.write("Mean Absolute Error: 55.14")
        st.write("Mean Squared Error: 4597.37")
        st.write("Root Mean Squared Error: 67.80")
        # st.write("R-squared: 0.98")

        # Load and display the associated image
        image = Image.open("random_forest.png")
        st.image(image, caption="Predicted vs Observed Sales for Random Forest Regressor")

        st.write("Random forest achieved high accuracy due to its bagging technique, where multiple decision trees are trained on random subsets of the data and features. This approach reduces overfitting and increases the robustness of the model by averaging the predictions of the individual trees")


    elif model_option == "Gradient Boosting Regressor":
        st.write("Mean Absolute Error: 53.89")
        st.write("Mean Squared Error: 4029.48")
        st.write("Root Mean Squared Error: 63.48")
        st.write("R-squared: 0.99")
        # st.write("Model Accuracy: 97.17%")

        # Load and display the associated image
        image = Image.open("gradient_boosting.png")
        st.image(image, caption="Predicted vs Observed Sales for Gradient Boosting Regressor")

        st.write("Gradient boosting enhances accuracy by iteratively adding weak learners (typically decision trees) that correct the errors of the previous models. By focusing on the mistakes made in earlier iterations, the model gradually improves, achieving a strong predictive performance.")


    elif model_option == "Neural Network":
        st.write("Mean Absolute Error: 56.96")
        st.write("Mean Squared Error: 4607.89")
        st.write("Root Mean Squared Error: 67.88")
        st.write("R-squared: 0.98")
        
        # Load and display the associated image
        image = Image.open("neural_network_ann.png")
        st.image(image, caption="Predicted vs Observed Sales for Neural Network")

        st.write("Neural networks leverage the perceptron theory, where interconnected layers of nodes (perceptrons) process input data to capture intricate patterns and relationships. By adjusting the weights through backpropagation, neural networks can model highly non-linear and complex data structures.")


# Run the Streamlit app
if __name__ == "__main__":
    model_performance()

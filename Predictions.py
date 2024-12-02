# Import necessary libraries
import pickle
import datetime as dt
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from cars_notebook import cars_price_predictor  
from banglore_home_prices_final import predict_house_price 
# from shoe_price_predictor import predict_shoe_price
from shoe_price_predictor_function import predict_shoe_price
import joblib


# Title of the Streamlit app
st.title("Commodity Price Predictor")


# Create a Landing Page
landing_page = st.sidebar.radio("Navigation", ["Home", "Predict"])

if landing_page == "Home":
    st.image('comm.jpg')
    # Explanation of how the app works
    st.write("""
    Welcome to the Commodity Price Predictor App. This app helps you predict the price of various commodities 
    including Cars, Houses, Laptops, and Shoes.""")
    st.write("### How the App Works:")
    st.write("""
    ####  This app uses machine learning models to predict the price based on the input data you provide.
    """)

    st.write("### How to Use the App:")
    st.write("""
    1. Use the **sidebar** to select a commodity (Car, House, Laptop, or Shoe).
    2. After selecting, fill in the relevant details like car model, house location, laptop specifications, or shoe brand.
    3. Click on the **'Predict Price'** button to get the price prediction for your chosen commodity.

    Click on 'Predict' in the sidebar to start using the app!
    """)

elif landing_page == "Predict":
    # Sidebar for commodity selection
    commodity = st.sidebar.selectbox("Select Commodity:", ["Car", "House", "Laptop", "Shoe"])


    def load_pipeline_shoe():
        pipeline_shoe = joblib.load('pipeline_shoe.pkl')
        return pipeline_shoe


    def load_laptop_data():
        df = joblib.load('df.pkl')
        pipe = joblib.load('pipe.pkl')
        return df, pipe

    # Based on commodity selection, show relevant inputs
    if commodity == "Car":
        st.image("Car_img.jpg") 

        st.write("""
    ### Car Price Prediction Model
                         
    This car price prediction model uses **Linear Regression**, a machine learning algorithm, to estimate the price of a used car based on various features such as the car's company, year of manufacture, kilometers driven, and fuel type.

    The model has been trained using historical data, and it has achieved an **R² score of 0.78**. 
    - **R² score** indicates how well the model's predictions align with the actual prices: a score of 0.78 means that the model explains 78% of the variance in car prices based on the input features.
    - **Linear Regression** works by finding the best-fit line that minimizes the error between the predicted and actual prices. The model considers relationships between the different features (e.g., year, kilometers, and car type) and learns how to predict the price accordingly.

    The input features provided (company, year, kilometers driven, and fuel type) are used to calculate the car's estimated price.
    """)
        # Car inputs
        st.sidebar.header("Car Details")
        company = st.selectbox("Select the car company:", ["Maruti", "Hyundai", "Mahindra", "Tata", "Honda", "Toyota", "Chevrolet", "Renault", "Ford", "Volkswagen", "Skoda", "Audi", "Mini", "BMW", "Datsun", "Mitsubishi", "Nissan", "Mercedes", "Fiat", "Force", "Hindustan", "Jaguar", "Land", "Jeep", "Volvo"])
        year = st.number_input("Enter the year of manufacture, from 2005 to 2024:", min_value=2005, max_value=2024)
        kms = st.number_input("Enter the kilometers driven:", min_value=0)
        car_type = st.selectbox("Select the type of car:", ["Petrol", "Diesel", "LPG"])
         
        # Button to trigger prediction for car
        if st.button("Predict Car Price"):
            predicted_price = cars_price_predictor(company, year, kms, car_type)
            predicted_price = round(int(predicted_price))
            # Apply min-max clipping to set reasonable price boundaries
            min_price = 1000  # Minimum reasonable price
            max_price = 1000000  # Maximum reasonable price
            if predicted_price < min_price:
                predicted_price = min_price
            elif predicted_price > max_price:
                predicted_price = max_price
            
            # Display the result
            st.write(f"Predicted Price for the Car: ₹{predicted_price}")

    elif commodity == "House":
        st.image('house_img.jpg')

        st.write("""
    ### House Price Prediction Model
    This house price prediction model uses **Linear Regression**, a machine learning algorithm, to estimate the price of a house based on its location, area (in square feet), number of bathrooms, and number of bedrooms (BHK).

    The model has been evaluated using different machine learning algorithms, and the best model for predicting house prices is **Linear Regression**, which achieved an **R² score of 0.85** (Best score from model selection).
    - **R² score** of 0.85 means the model explains 85% of the variance in house prices based on the input features.
    - **Linear Regression** works by finding the best-fit line that minimizes the error between the predicted and actual prices, considering the relationship between features like location, square footage, and the number of bathrooms and bedrooms.

    The input features provided (location, area, number of bathrooms, and BHK) are used to calculate the house's estimated price.
    """)



        # Read the locations from the CSV file (could also be cached if large)
        locations_df = pd.read_csv("locations.csv")
        location_columns = locations_df['Location'].tolist()
        
        # House inputs
        st.sidebar.header("House Details")
        location = st.selectbox("Select the Location of Banglore:", location_columns)
        sqft = st.number_input("Enter the area in Square Feet:", min_value=1)
        bath = st.number_input("Enter the number of bathrooms:", min_value=1)
        bhk = st.number_input("Enter the number of BHK (Bedrooms):", min_value=1)

        # Button to trigger prediction for house
        if st.button("Predict House Price"):
            predicted_price = predict_house_price(location, sqft, bath, bhk)
            predicted_price = round(int(predicted_price))
            # Apply min-max clipping to set reasonable price boundaries
            min_price = 10000  # Minimum reasonable price for houses
            max_price = 100000000  # Maximum reasonable price for houses
            if predicted_price < min_price:
                predicted_price = min_price
            elif predicted_price > max_price:
                predicted_price = max_price
            
            # Display the result
            st.write(f"Predicted Price for the House: ₹{predicted_price}")

    elif commodity == "Laptop":
        st.image('laptop_img.jpg')
        st.write("""
    ### Laptop Price Prediction Model
    This laptop price prediction model uses the **Random Forest Regressor** algorithm, which is a powerful ensemble learning method. It combines the predictions of multiple decision trees to make more accurate predictions.

    After evaluating various machine learning models, **Random Forest Regressor** was selected due to its superior performance, achieving an **R² score of 0.89** (Best score from model selection). The **R² score** of 0.89 means the model explains 89% of the variance in laptop prices based on the input features. 
    Additionally, the **Mean Absolute Error (MAE)** was calculated as **0.16**, indicating that on average, the model's prediction is off by ₹0.16 lakh.

    The model works by training multiple decision trees on different subsets of the data and then averaging their predictions to improve accuracy and reduce overfitting.

    The input features (brand, type, RAM, weight, touchscreen, etc.) are used to predict the price of the laptop.
    """)



        # Load the DataFrame and Pipeline
        df, pipe = load_laptop_data()

        # Laptop inputs
        st.sidebar.header("Laptop Details")
        company = st.selectbox("Enter the Brand:", df['Company'].unique())
        typename = st.selectbox("Enter the TypeName:", df['TypeName'].unique())
        ram = st.selectbox("Ram in GB:", [2, 4, 6, 8, 12, 16, 24, 32, 64])
        weight = st.number_input("Enter the Weight in kg :")
        touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])
        ips = st.selectbox('IPS', ['No', 'Yes'])
        screen_size = st.number_input('Screen Size in inches')
        resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
        cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
        hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
        ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
        gpu = st.selectbox('GPU', df['Gpu brand'].unique())
        os = st.selectbox('OS', df['os'].unique())

        # Button to trigger prediction for laptop
        if st.button("Predict Laptop Price"):
            ppi = None
            if touchscreen == 'Yes':
                touchscreen = 1
            else:
                touchscreen = 0

            if ips == 'Yes':
                ips = 1
            else:
                ips = 0

            X_res = int(resolution.split('x')[0])
            Y_res = int(resolution.split('x')[1])
            ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
            query = np.array([company, typename, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
            query = query.reshape(1, 12)

            predicted_price = np.exp(pipe.predict(query))
            predicted_price = round(int(predicted_price))
            st.write(f"Predicted Price for the Laptop: ₹{predicted_price}")

    elif commodity == "Shoe":
        st.image('shoe_img.jpg', width=450,)
            # Display Model Details and Explanation
        st.markdown("""
    ### Shoe Price Predictor Model

    This prediction model uses a **Random Forest Regressor** to predict the resale price of shoes.
    The **R² score** of the model is **0.975**, meaning that 97.5% of the variance in the shoe prices can be explained by the model. This indicates that the model provides highly accurate predictions.
    
    ### How the model works:
    - **Preprocessing**:
      The model performs preprocessing on input features using a **ColumnTransformer**. The categorical data such as **brand**, **sneaker name**, and **buyer region** are encoded using **OneHotEncoding**, while numerical values like **shoe size**, **order date**, **release date**, and **retail price** are standardized using **StandardScaler**.
    
    - **Model**:
      The model uses a **Random Forest Regressor** for predicting the resale price of shoes. This algorithm works by training multiple decision trees on random subsets of the data and then averaging their results for the final prediction. This reduces the model's variance and improves its accuracy.
    """)



        # Cache the shoe data and pipeline loading
        pipeline_shoe = load_pipeline_shoe()

        # Shoe details input in the sidebar
        st.sidebar.header("Shoe Details")
        brand = st.selectbox("Select the Shoe Brand:", [' Yeezy' , 'Off-White'])
        sneaker_name = st.selectbox("Select the Sneaker name:", ['Adidas Yeezy Boost', 'Nike Blazer Mid', 'Nike Air Presto','Nike Zoom Fly', 'Nike Air Max', 'Nike Air VaporMax',
       'Nike Air Force', 'Air Jordan 1', 'Nike React Hyperdunk'])
        shoe_size = st.number_input("Enter the Shoe Size:", min_value=1.0, max_value=16.0, step=0.5)
        buyer_region = st.selectbox("Select the Buyer Region:", ['New Jersey', 'New York', 'California', 'Oregon', 'Washington',
       'Idaho', 'Georgia', 'Texas', 'Florida', 'Alabama', 'Pennsylvania',
       'Indiana', 'Virginia', 'North Carolina', 'Nevada', 'Oklahoma',
       'Michigan', 'Arizona', 'New Mexico', 'Maryland', 'Illinois',
       'Massachusetts', 'Ohio', 'Delaware', 'Connecticut', 'Wisconsin',
       'Hawaii', 'Utah', 'Rhode Island', 'Minnesota', 'Missouri',
       'South Carolina', 'Louisiana', 'Colorado', 'District of Columbia',
       'New Hampshire', 'Kansas', 'Kentucky', 'Nebraska', 'West Virginia',
       'Tennessee', 'Arkansas', 'South Dakota', 'Iowa', 'Maine',
       'Wyoming', 'Alaska', 'Mississippi', 'Montana', 'Vermont',
       'North Dakota'])
        order_date = st.date_input("Enter the Order Date:")
        release_date = st.date_input("Enter the Release Date:")
        retail_price = st.number_input("Enter the Retail Price:", min_value=0.0, step=0.01)

    # Button to trigger prediction for shoe
        if st.button("Predict Shoe Price"):
            predicted_price = predict_shoe_price(brand, sneaker_name, shoe_size, buyer_region, order_date, release_date, retail_price)
            predicted_price = round(predicted_price, 2)  # Round to two decimal places

        # Apply min-max clipping to set reasonable price boundaries
            min_price = 50  # Minimum reasonable price for shoes
            max_price = 5000  # Maximum reasonable price for shoes
            if predicted_price < min_price:
             predicted_price = min_price
            elif predicted_price > max_price:
                predicted_price = max_price

            st.write(f"Predicted Price for the Shoe: ${predicted_price}")






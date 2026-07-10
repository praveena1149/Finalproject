import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model


# loading the model
model = load_model('lstm_model.keras') 
scaler=pickle.load(open('scaler.pkl','rb'))
# loading the dataset
df=pd.read_csv('ecommerce_final.csv')

page=st.sidebar.radio('Navigation',['Home','Demand Forecasting','About'])

if page == 'Home':
    st.title('E-commerce Inventory Demand Forecasting')
    st.image('https://www.cloudways.com/blog/wp-content/uploads/ecommerce-website-checklist-b-.jpg')
    st.write('This project challenges learners to build a deep learning-based demand forecasting system using LSTM and feedforward neural networks to predict weekly inventory requirements per product. Accurate forecasts will enable smarter procurement decisions, reduce stockouts by up to 30%, and lower excess inventory carrying costs.')  

if page == 'Demand Forecasting':
    st.subheader("Demand Forecasting")
    product_category=st.selectbox('product category:',['Electronics','Apparel','Sports','Home','Beauty'])
    units_sold=st.number_input("Enter units sold")
    unit_price=st.number_input("Enter unit price")
    stock_on_hand=st.number_input("Enter stock onhand")
    reorder_point=st.number_input("Enter reorder point")
    is_promotion=st.selectbox('promotion:',['0','1'])
    discount_pct=st.number_input("Enter dicount percentage")
    day_of_week=st.slider("day of week",0,6)
    month=st.number_input("Enter month")
    supplier_lead_days=st.number_input("Enter supplier lead days")
    lag_1=st.number_input("Enter lag1")
    lag_2=st.number_input("Enter lag2")
    lag_3=st.number_input("Enter lag3")
    lag_7=st.number_input("Enter lag7")
    lag_14=st.number_input("Enter lag14")
    lag_30=st.number_input("Enter lag30")
    rolling_mean_7= st.number_input("Enter rolling mean7")
    rolling_std_7=st.number_input("Enter rolling std7")
    rolling_mean_14=st.number_input("Enter rolling mean14")
    rolling_std_14=st.number_input("Enter rolling std14")
    rolling_mean_30=st.number_input("Enter rolling mean30")
    rolling_std_30=st.number_input("Enter rolling std30")
    is_weekend=st.selectbox('weekend:',['0','1'])
    day=st.slider("day",0,31)
    quarter=st.selectbox('quater:',['1','2','3','4'])
    
    # encoding the categorical features
   
    product_category_mapping={'Electronics' : 0, 'Apparel' : 1, 'Sports' : 2, 'Home' : 3, 'Beauty' : 4}
    product_category=product_category_mapping[product_category]
    
    reverse_product_category_mapping = {v: k for k, v in product_category_mapping.items()}
   
    if st.button("Predict"):
     data = np.array([[product_category,units_sold,unit_price,
       stock_on_hand,reorder_point, is_promotion,discount_pct,
       day_of_week,month,supplier_lead_days, lag_1,lag_2,lag_3,
       lag_7, lag_14,lag_30,rolling_mean_7, rolling_std_7,
       rolling_mean_14, rolling_std_14, rolling_mean_30,
       rolling_std_30, is_weekend, day, quarter]])
         
    # scaling  numerical input features
    
     num_cols=np.array([[product_category,units_sold,unit_price,
       stock_on_hand, reorder_point, is_promotion, discount_pct,
       day_of_week, month,supplier_lead_days, lag_1,lag_2, lag_3,
       lag_7,lag_14,lag_30,rolling_mean_7,rolling_std_7,
       rolling_mean_14,rolling_std_14,rolling_mean_30,
       rolling_std_30, is_weekend,day, quarter]])     
    
     scaling = scaler.transform(num_cols)
     # Take last 60 timestamps
     X_input = np.repeat(scaling[:, np.newaxis, :],60, axis=1)                                

     prediction = model.predict(X_input)
     target_index = 1
     dummy = np.zeros((1, scaling.shape[1]))   # 25 columns
     dummy[:, target_index] = prediction[:, 0]
     prediction_original = scaler.inverse_transform(dummy)
     predicted_demand = int(round(prediction_original[0, target_index]))
     st.success(f"Predicted Demand: {predicted_demand} units")

     # Reorder Alert
     if stock_on_hand <= reorder_point:
      alert = "🔴 Reorder Immediately"
     elif predicted_demand > stock_on_hand:
      alert = "🟠 High Demand - Reorder Soon"
     else:
      alert = "🟢 Stock Sufficient"
      
     category_name = reverse_product_category_mapping[product_category]
     reorder_table = pd.DataFrame({
        "Product Category": [category_name],
        "Current Stock": [stock_on_hand],
        "Reorder Point": [reorder_point],
        "Predicted Demand": [round(predicted_demand, 2)],
        "Reorder Alert": [alert]
})

     st.subheader("📦 Reorder Alerts")
     st.table(reorder_table)

     # Download CSV
     csv = reorder_table.to_csv(index=False).encode("utf-8")

     st.download_button(
        label="📥 Download Reorder Alerts",
        data=csv,
        file_name="reorder_alerts.csv",
        mime="text/csv"
     )   
     
if page == 'About':
    st.title('E-COMMERCE SALES FORECASTING APPLICATION')

    st.write('This application predicts future product sales in an e-commerce platform using historical sales data and machine learning/deep learning models. It helps businesses analyze trends, forecast demand, and make better inventory and marketing decisions.')

    st.subheader("Features:")
    st.write("""
    - Predict future sales values
    - Analyze sales trends and patterns
    - Visualize sales performance
    - Support inventory planning
    - Assist demand forecasting
    """)

    st.subheader("Dataset Includes:")
    st.write("""
    - Date
    - Product Category
    - Product ID
    - Units Sold
    - stock on hold
    - reorder point
    - Discount
    - week of day
    - supplier leading
    - promtions
    - lag and rolling features
    """)

    st.subheader("Models Used:")
    st.write("""
    - LSTM (Long Short-Term Memory)
    - MLP (Multi-Layer Perceptron) Baseline Model
    """)

    st.subheader("Libraries Used:")
    st.write("""
    - Python
    - TensorFlow / Keras
    - Scikit-learn
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    - Streamlit
    """)

    st.subheader("Model Performance:")
    st.write("""
    - R² Score: 0.50
    - MAE: 6.30
    - RMSE: 9.63
    - MAPE: 23.50%
    """)

    st.subheader("Evaluation Metrics Used:")
    st.write("""
    - R² Score
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Percentage Error (MAPE)
    """)

    st.subheader("Developed By")
    st.write("Praveena Ramesh")
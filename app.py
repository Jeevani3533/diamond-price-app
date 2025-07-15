import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("💎 Diamond Price Predictor")

carat = st.number_input("Carat", 0.0, 5.0, 0.5)
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
depth = st.number_input("Depth", 50.0, 70.0, 60.0)
table = st.number_input("Table", 50.0, 70.0, 60.0)
x = st.number_input("X (mm)", 0.0, 10.0, 4.0)
y = st.number_input("Y (mm)", 0.0, 10.0, 4.0)
z = st.number_input("Z (mm)", 0.0, 10.0, 2.5)

size = x * y * z

input_data = pd.DataFrame([[carat, cut, color, clarity, depth, table, size]],
                          columns=["carat", "cut", "color", "clarity", "depth", "table", "size"])

prediction = model.predict(input_data)[0]
st.success(f"Estimated Price: ${prediction:,.2f}")

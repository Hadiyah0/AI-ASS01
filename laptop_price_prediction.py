import pandas as pd
import numpy as np
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
file_path = r"C:\Users\admin\Desktop\AI-25\Laptop_Resale_Prediction\Used_Laptop_Dataset_Sheet1.CSV"
df = pd.read_csv(file_path)


# Preprocessing
df["RAM"] = df["RAM"].str.replace("GB", "").astype(int)
df["Storage"] = df["Storage"].str.replace("GB", "").astype(int)
df["Battery Health"] = df["Battery Health"].str.replace("%", "").astype(int)

# One-hot encoding
df = pd.get_dummies(df, columns=["Brand", "Processor"], drop_first=True)

# Splitting Data
X = df.drop(columns=["Resale Price ($)"])
y = df["Resale Price ($)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
pickle.dump(model, open("laptop_model.pkl", "wb"))

# Load Model
model = pickle.load(open("laptop_model.pkl", "rb"))

# Streamlit UI
st.title("  LAPTOP RESALE PRICE PREDICTOR ")

st.subheader("Data Visualization 🔎 ")
fig, ax = plt.subplots(figsize=(15, 10))
sns.scatterplot(x=df["Age"], y=df["Resale Price ($)"], ax=ax)
ax.set_xlim(1, 10) 
plt.xlabel("Age (Years)")
plt.ylabel("Resale Price ($)")
st.pyplot(fig)

# User Inputs
age = st.number_input("Enter Laptop Age (years)", min_value=1, max_value=10, value=3)
ram = st.selectbox("Select RAM (GB)", [4, 8, 16, 32, 64])
storage = st.selectbox("Select Storage (GB)", [128, 256, 512, 1024, 2048])
battery_health = st.slider("Battery Health (%)", 50, 100, 80)

brands = ["Dell", "Apple", "Lenovo", "HP", "Asus", "Acer"]
processors = ["Intel i3", "Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7", "AMD Ryzen 9"]
brand = st.selectbox("Select Brand", brands)
processor = st.selectbox("Select Processor", processors)

# One-hot encoding for user input
input_data = {"Age": age, "RAM": ram, "Storage": storage, "Battery Health": battery_health}
for b in brands[1:]:
    input_data[f"Brand_{b}"] = 1 if brand == b else 0
for p in processors[1:]:
    input_data[f"Processor_{p}"] = 1 if processor == p else 0

input_df = pd.DataFrame([input_data])

# Prediction
if st.button("🔮 Predict Resale Price"):
    # Ensure column matching
    missing_cols = set(X_train.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Add missing columns

    input_df = input_df[X_train.columns]  # Reorder columns
    
    predicted_price = model.predict(input_df)
    st.success(f"💰 Estimated Resale Price: ${predicted_price[0]:,.2f}")


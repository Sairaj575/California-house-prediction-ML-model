# 🏠 California House Price Predictor (Streamlit App)

This project is an **interactive Streamlit web application** built using Python and scikit-learn to predict **median house prices in California**.  
The app uses a trained **Random Forest regression model** and provides both **single prediction** and **batch prediction** capabilities through a clean UI.

---

## 📌 Application Overview

- Web-based UI built using **Streamlit**
- Predicts median house prices using California housing data
- Uses a trained **Random Forest Regressor**
- Consistent preprocessing using scikit-learn pipelines
- Supports **single input prediction** and **CSV batch prediction**
- Displays property location on an interactive map
- Allows users to download prediction results as a CSV file

---

## 🧠 Machine Learning Approach

- Stratified sampling based on income categories
- Numerical preprocessing:
  - Median imputation
  - Standard scaling
- Categorical preprocessing:
  - One-hot encoding for ocean proximity
- Model training using RandomForestRegressor
- Model and preprocessing pipeline saved using Joblib

---

## 📂 Project Structure

```
📦 california-house-price-streamlit
 ┣ 📜 housing.csv              # Training dataset
 ┣ 📜 model.pkl                # Trained ML model
 ┣ 📜 pipeline.pkl             # Preprocessing pipeline
 ┣ 📜 main.py                  # Streamlit application
 ┣ 📜 input.csv                # Sample input file
 ┗ 📜 README.md                # Project documentation
```

---

## ▶️ How to Run the Application

### 1️⃣ Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn==1.8.0 joblib
```

### 2️⃣ Run the Streamlit App

```bash
streamlit run main.py
```

The application will open in your browser.

---

## 🖥️ Application Features

### 🔹 Single Prediction
- Enter house details using the sidebar
- Click **Predict Price**
- View predicted median house value instantly

### 🔹 Location Visualization
- Displays the selected location on an interactive map

### 🔹 Batch Prediction
- Upload a CSV file with multiple records
- Preview data before prediction
- Generate predictions for all rows
- Download the output CSV with predicted values

---

## 📄 Input CSV Format

The uploaded CSV file must contain the following columns:

```
longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
population, households, median_income, ocean_proximity
```

---

## 🚀 Future Improvements

- Model hyperparameter tuning
- Performance optimization for large datasets
- Deployment on Streamlit Cloud
- API integration (FastAPI / Flask)
- Model versioning and monitoring

---

## 👨‍💻 Author

**Sairaj Umbarkar**  
Aspiring Data Science Enthusiast  

---

⭐ If you find this project useful, feel free to star the repository!

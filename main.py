import pandas as pd
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import joblib
import streamlit as st


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# Page
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

def train_model():
    # Train model 
    housing = pd.read_csv("housing.csv")

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, _ in split.split(housing, housing["income_cat"]):
        housing = housing.loc[train_idx].drop("income_cat", axis=1)

    labels = housing["median_house_value"]
    features = housing.drop("median_house_value", axis=1)

    num_attribs = features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    features_prepared = pipeline.fit_transform(features)

    model = RandomForestRegressor(random_state=42)
    model.fit(features_prepared, labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    return model, pipeline

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
        return train_model()
    return joblib.load(MODEL_FILE), joblib.load(PIPELINE_FILE)

# App title
st.title("California House Price Predictor")

# Load model
try:
    model, pipeline = load_model()
    model_loaded = True
except Exception as e:
    st.error("Model could not be loaded.")
    model_loaded = False

# Sidebar
st.sidebar.header("Input Features")

longitude = st.sidebar.number_input(
    "Longitude", -124.0, -114.0, -122.23, format="%.2f"
)

latitude = st.sidebar.number_input(
    "Latitude", 32.0, 42.0, 37.88, format="%.2f"
)

housing_median_age = st.sidebar.number_input(
    "Housing Median Age", 1, 100, 41
)

total_rooms = st.sidebar.number_input(
    "Total Rooms", 1, value=880
)

total_bedrooms = st.sidebar.number_input(
    "Total Bedrooms", 1, value=129
)

population = st.sidebar.number_input(
    "Population", 1, value=322
)

households = st.sidebar.number_input(
    "Households", 1, value=126
)

median_income = st.sidebar.number_input(
    "Median Income", 0.0, value=8.3252, format="%.4f"
)

ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

predict_button = st.sidebar.button("Predict Price")

# Input dataframe
input_df = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Data")
    st.dataframe(input_df, use_container_width=True)
    st.subheader("Location on Map")
    st.map(pd.DataFrame({
    "lat": [latitude],
    "lon": [longitude]
}))

with col2:
    st.subheader("Location Info")
    st.metric("Longitude", longitude)
    st.metric("Latitude", latitude)
    st.metric("Ocean Proximity", ocean_proximity)

# Prediction
if predict_button and model_loaded:
    try:
        transformed = pipeline.transform(input_df)
        prediction = model.predict(transformed)[0]

        st.markdown("---")
        st.subheader("Prediction Result")
        st.metric("Predicted Median House Value", f"${prediction:,.2f}")

    except Exception as e:
        st.error("Prediction failed.")

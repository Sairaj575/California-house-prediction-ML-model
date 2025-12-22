# 🏠 California Housing Price Prediction (End-to-End ML Pipeline)

This project implements an **end-to-end machine learning pipeline** for predicting **median house prices** using the California Housing dataset.  
It covers **data preprocessing, stratified sampling, feature engineering, model training, inference, model persistence, and evaluation**.

---

## 📌 Project Overview

- Uses **Stratified Sampling** based on income categories to maintain data distribution  
- Applies **numerical & categorical preprocessing** using `Pipeline` and `ColumnTransformer`  
- Trains a **Random Forest Regressor**  
- Saves trained **model and pipeline**  
- Supports **batch inference** on new input data  
- Evaluates predictions using **RMSE and R² Score**

---

## 📂 Project Structure

```
📦 california-housing-ml
 ┣ 📜 housing.csv
 ┣ 📜 input.csv
 ┣ 📜 input_copy.csv
 ┣ 📜 output.csv
 ┣ 📜 model.pkl
 ┣ 📜 pipeline.pkl
 ┣ 📜 main.py
 ┗ 📜 README.md
```

---

## ⚙️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  

---

## 🧠 Machine Learning Workflow

1. Load dataset  
2. Create income categories  
3. Perform stratified train split  
4. Preprocess data  
5. Train Random Forest model  
6. Save model and pipeline  
7. Run inference on new data  
8. Evaluate using RMSE and R²  

---

## 🧪 Model Details

- **Algorithm:** RandomForestRegressor  
- **Target:** `median_house_value`  
- **Metrics:** RMSE, R² Score  

---

## ▶️ How to Run

### Install dependencies
```bash
pip install pandas numpy scikit-learn joblib
```

### Run the script
```bash
python main.py
```

---

## 📤 Output

- Predictions saved to `output.csv`
- Evaluation metrics printed in console

---

## 🚀 Future Improvements

- Hyperparameter tuning  
- Advanced models (XGBoost, Gradient Boosting)  
- API deployment (Flask/FastAPI)  
- Docker support  

---

## 👨‍💻 Author

**Sairaj Umbarkar**  
Aspiring Data Scientist  

---

⭐ If you like this project, give it a star!

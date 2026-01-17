🚗 Car Selling Price Prediction using Machine Learning

📌 Project Overview

This project focuses on building a **Machine Learning model** to predict the **selling price of a used car** based on multiple factors such as fuel type, years of service, showroom price, ownership details, kilometers driven, seller type, and transmission type.

The goal is to help users estimate a **fair selling price** for their cars using data-driven insights.

 🎯 Objectives

* Analyze historical car data
* Train a machine learning model to predict car prices
* Understand the impact of different features on selling price
* Provide accurate and reliable price predictions

📊 Dataset Information

* **Dataset Name:** Car Price Prediction Dataset
* **Source:** Google Drive
* **Link:**
  [https://drive.google.com/file/d/1yFuNVPXM5CH6g0TthYKcTGrZCCJo6n8Z/view](https://drive.google.com/file/d/1yFuNVPXM5CH6g0TthYKcTGrZCCJo6n8Z/view)

 Dataset Features

| Feature            | Description                           |
| ------------------ | ------------------------------------- |
| `Fuel_Type`        | Type of fuel (Petrol / Diesel / CNG)  |
| `Years_of_Service` | Number of years the car has been used |
| `Showroom_Price`   | Original showroom price (in Lakhs)    |
| `Owner`            | Number of previous owners             |
| `Kms_Driven`       | Total kilometers driven               |
| `Seller_Type`      | Dealer or Individual                  |
| `Transmission`     | Manual or Automatic                   |
| `Selling_Price`    | Target variable (in Lakhs)            |

 🛠️ Technologies Used

 **Programming Language:** Python
 **Libraries:**

  * NumPy
  * Pandas
  * Matplotlib
  * Seaborn
  * Scikit-Learn

⚙️ Machine Learning Workflow

1. Data Loading
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Encoding
5. Model Training
6. Model Evaluation
7. Price Prediction

 🤖 Model Used

**Linear Regression**
**Random Forest Regressor** (for better accuracy)

 📈 Model Evaluation Metrics

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

## 📁 Project Structure
car-price-prediction/
│
├── dataset/
│   └── car_data.csv
│
├── notebooks/
│   └── car_price_analysis.ipynb
│
├── car_price_prediction.py
├── requirements.txt
└── README.md

 🚀 Future Enhancements

* Deploy the model using Flask or Streamlit
* Add more advanced regression models
* Improve accuracy with hyperparameter tuning
* Create a user-friendly web interface

 👨‍💻 Author

Kruntal Wadile
AI & ML Intern


# DemandForecasting

# ⚡ Time Series Forecasting using ML | ARIMA | End-to-End Project | Energy Demand Forecasting

## 📌 Project Overview

This project focuses on forecasting **energy demand** using time series analysis and machine learning techniques. Accurate demand prediction is crucial for optimizing energy generation, reducing operational costs, and preventing shortages. We use historical data on **IT load** and **solar generation** to build predictive models that anticipate future energy needs.

The project demonstrates an end-to-end pipeline — from data preprocessing and visualization to model training, evaluation, and forecasting — using classical statistical models like **ARIMA** alongside machine learning techniques.

---

## 📊 Dataset Description

The dataset contains hourly records of energy usage and solar generation:

| utc_timestamp        | IT_load_new | IT_solar_generation |
|----------------------|-------------|----------------------|
| 2016-01-01T00:00:00Z | 21665       | 1                    |
| 2016-01-01T01:00:00Z | 20260       | 0                    |
| 2016-01-01T02:00:00Z | 19056       | 0                    |

- `utc_timestamp`: Timestamp in UTC format
- `IT_load_new`: Energy demand/load in IT infrastructure
- `IT_solar_generation`: Solar energy generated at that hour

---

## 🎯 Objectives

- Forecast future energy demand using historical data
- Analyze seasonal and trend components in energy usage
- Compare ARIMA with other ML models for time series forecasting
- Visualize predictions and evaluate performance using metrics like RMSE and MAE

---

## 🛠️ Methodology

1. **Data Preprocessing**
   - Handle missing values
   - Convert timestamps to datetime objects
   - Resample and aggregate data if needed

2. **Exploratory Data Analysis**
   - Time series decomposition
   - Correlation between load and solar generation

3. **Modeling**
   - ARIMA (AutoRegressive Integrated Moving Average)
   - Optional: Prophet, XGBoost, or LSTM for comparison

4. **Evaluation**
   - RMSE, MAE, MAPE
   - Visual comparison of actual vs predicted demand

5. **Forecasting**
   - Generate future demand predictions
   - Plot forecast intervals and trends

---

## 🚀 Technologies Used

- Python 🐍
- Pandas & NumPy
- Matplotlib & Seaborn
- Statsmodels (ARIMA)
- Scikit-learn
- Jupyter Notebook

---

## 📦 Installation

```bash
git clone https://github.com/roshankahaneDSAI/DemandForecasting.git
cd DemandForecasting
pip install -r requirements.txt
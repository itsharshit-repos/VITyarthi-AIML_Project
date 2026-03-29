# House Price Prediction Model


## Description


This project implements a machine learning pipeline for predicting house prices based on structural and locational attributes. The model is trained on a dataset of 1,000 records with eight features. The pipeline includes data loading, exploratory analysis, preprocessing, training of multiple regression algorithms, performance evaluation, and model serialization.

## Dataset


The dataset `house_dataset.csv` contains the following columns:

| Column Name      | Description                                   |
|------------------|-----------------------------------------------|
| area_sqft        | Total living area in square feet              |
| bedrooms         | Number of bedrooms                            |
| bathrooms        | Number of bathrooms                           |
| floors           | Number of floors in the house                 |
| age_years        | Age of the house in years                     |
| garage           | Garage capacity (number of cars)              |
| location_score   | Location quality score (higher is better)     |
| price            | Target variable – house price                 |

The dataset contains 1,000 rows. Missing values, if any, are handled by median imputation.


## Project Structure


house-price-prediction/
│
├── house_dataset.csv # Input data
├── house_price_prediction.py # Main Python script
├── requirements.txt # Python dependencies
├── README.md # Documentation
│
├── best_house_price_model.pkl # Saved best model (generated after run)
└── scaler.pkl # Saved StandardScaler (if linear model is best)


## Installation


### Prerequisites

- Python 3.8 or higher
- pip package manager


### Setup

1. Place `house_dataset.csv` and `house_price_prediction.py` in the same directory.

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows


### Install dependencies:


pip install -r requirements.txt


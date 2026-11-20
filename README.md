# Desertification Risk Prediction

This project predicts desertification risk using climate and vegetation data.

The project uses rainfall, temperature, soil moisture, and NDVI values to understand environmental conditions and predict whether an area may be at risk of desertification.

## Project Features

- Reads climate and NDVI CSV datasets
- Cleans and prepares data for analysis
- Shows monthly NDVI trends
- Creates a correlation heatmap
- Shows the importance of climate variables
- Trains a Random Forest machine learning model
- Predicts desertification risk
- Includes a Streamlit web app where users can enter climate values and get a prediction

## Tools and Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Visual Studio Code
- Git and GitHub

## Files in This Project

| File | Description |
|---|---|
| `app.py` | Streamlit web app for entering rainfall, temperature, and soil-moisture values to get a risk prediction |
| `risk.py` | Trains and evaluates the desertification-risk classification model |
| `dessert.py` | Performs climate and NDVI data analysis and creates visualizations |
| `model.pkl` | Saved trained machine-learning model used by the app |
| `India_NDVI.csv` | Dataset containing NDVI values |
| `nasa_climate1.csv` | Dataset containing climate information |

## Dataset Information

The project uses two datasets:

- **Climate data:** includes rainfall, temperature, and soil-related information.
- **NDVI data:** NDVI (Normalized Difference Vegetation Index) is used to represent vegetation health. Higher NDVI values generally indicate healthier vegetation.

## Machine Learning Model

The project uses a **Random Forest Classifier** to predict desertification risk.

### Input Features

- Rainfall
- Temperature
- Soil moisture
- NDVI

### Output

- Desertification risk level

## Model Performance

The model achieved approximately:

- Accuracy: **87.9%**

The project also uses a regression analysis for NDVI prediction and produced:

- R² Score: **0.6937**
- RMSE: **0.0477**

## Visualizations

The project generates the following graphs:

- Monthly NDVI graph
- Correlation heatmap
- Climate variable importance chart
- Actual vs predicted NDVI graph
- Confusion matrix for the classification model

## Installation

Install Python, then install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit

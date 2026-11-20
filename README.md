# Desertification Risk Prediction

This project uses climate data and NDVI values to predict desertification risk.

It uses rainfall, temperature, soil moisture, and NDVI data for analysis and machine learning.

## Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## Files

- `app.py` - Streamlit web application
- `risk.py` - trains the desertification risk model
- `dessert.py` - climate and NDVI data analysis
- `model.pkl` - saved trained model
- `India_NDVI.csv` - NDVI dataset
- `nasa_climate1.csv` - climate dataset

## Features

- Climate and NDVI data analysis
- Desertification risk prediction
- Random Forest machine learning model
- Correlation heatmap
- Climate variable importance graph
- Monthly NDVI graph
- Streamlit web interface for predictions

## Install Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit

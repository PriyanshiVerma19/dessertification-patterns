# desertification patterns prediction

this project analyzes climate data and ndvi to understand vegetation changes.

## data used
- nasa climate data (rain, temperature, soil moisture)
- ndvi data

## steps
- cleaned and merged climate data
- converted monthly values
- combined with ndvi
- trained random forest model

## results
- r2 score: ~0.69
- rmse: ~0.047

## tools used
- python
- pandas
- matplotlib
- scikit-learn

## output
- ndvi trend graph
- feature importance graph

b"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
#from sklearn.linear_model import LinearRegression

#change the algorithm
from sklearn.ensemble import RandomForestRegressor

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Madrid_wind_speed',
 'Seville_clouds_all',
 'Bilbao_wind_deg',
 'Seville_rain_1h',
 'Barcelona_rain_3h',
 'Valencia_snow_3h',
 'Bilbao_pressure',
 'Bilbao_weather_id',
 'Valencia_temp_min']]

#fitting the new model
RF = RandomForestRegressor(n_estimators=100, max_depth=50)
print ("Training Model...")
RF.fit(X_train,y_train)



# Pickle model for use within our API
save_path = '../assets/trained-models/RF_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(RF, open(save_path,'wb'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os


data = pd.read_csv('clean_data_final.csv')
del data['Unnamed: 0']
volume = data.groupby(['Location_ID']).size()
normalized_df=(volume-volume.mean())/(volume.max()-volume.min())
data.loc[data['Location_ID'] == 1, "volume"] = normalized_df[1]
data.loc[data['Location_ID'] == 2, "volume"] = normalized_df[2]
data.loc[data['Location_ID'] == 3, "volume"] = normalized_df[3]
data.loc[data['Location_ID'] == 4, "volume"] = normalized_df[4]
data.loc[data['Location_ID'] == 5, "volume"] = normalized_df[5]
data.loc[data['Location_ID'] == 6, "volume"] = normalized_df[6]
data.loc[data['Location_ID'] == 7, "volume"] = normalized_df[7]
data.loc[data['Location_ID'] == 8, "volume"] = normalized_df[8]
data.loc[data['Location_ID'] == 9, "volume"] = normalized_df[9]
data.loc[data['Location_ID'] == 10, "volume"] = normalized_df[10]
data.loc[data['Location_ID'] == 11, "volume"] = normalized_df[11]
data.loc[data['Location_ID'] == 12, "volume"] = normalized_df[12]
data.loc[data['Location_ID'] == 13, "volume"] = normalized_df[13]
data.loc[data['Location_ID'] == 14, "volume"] = normalized_df[14]
data.loc[data['Location_ID'] == 15, "volume"] = normalized_df[15]
data.loc[data['Location_ID'] == 16, "volume"] = normalized_df[16]
data.loc[data['Location_ID'] == 17, "volume"] = normalized_df[17]
data.loc[data['Location_ID'] == 18, "volume"] = normalized_df[18]
data.loc[data['Location_ID'] == 2]
data['Red_Indication'].unique()

def extract_hour(time_int):
    hour = int(str(time_int).strip()[:2])
    return hour

def extract_hour(time_int):
    time_str = str(time_int).strip()
    # Check for missing leading zero and add it if necessary
    if len(time_str) == 5:
        time_str = '0' + time_str
    hour = int(time_str[:2])
    return hour

data['Hour'] = data['Time_Showed_Intent'].apply(lambda x: extract_hour(x))

data['target'].value_counts()

data_model = data[[ 'Number_of_Pedestrians', 
       
       'Pedestrian_Type', 'Vehicle_Speed', 'Opposite_Direction_Yield',
       'Following_Vehicle', 'Posted_Speed', 'Num_Lanes_Main',
       'Crossing_Width_(Major)', 'Bike_Lane(s)', 'Weather', 'Signage',
       'Markings', 'Presence_of_Single_Family', 'Presence_of_Apartments',
       'Presence_of_Commercial',
       'Presence_of_Gas_Station/Convenient_Store',
       'Presence_of_Restaurants/Bars', 'Presence_of_Parking_Lots',
       'Dist_to_Nearest_Park', 'Dist_to_Nearest_School',
       'Presence_of_on_street_parking', 'PAWS_Score', 'Tree_Cover',
       'lighting', 'road_surface', 'num_of_bus_stops', 
       'Major_AADT', 'Red_Indication', 'volume', 'target']]

data_model.isna().sum()

data.isna().sum()

data_model_cat = pd.get_dummies(data_model, columns=['Pedestrian_Type', 'Markings', 'Posted_Speed'], drop_first=True)

data_model_cat.loc[data_model_cat['Opposite_Direction_Yield'] == 1,  'Opposite_Direction_Yield'] = 0

data_model_cat.loc[data_model_cat['Opposite_Direction_Yield'] == 2,  'Opposite_Direction_Yield'] = 1

cols_sf = ['Number_of_Pedestrians', 'Vehicle_Speed', 'Opposite_Direction_Yield', 'Following_Vehicle', 'Posted_Speed',  'Num_Lanes_Main',
           'Crossing_Width_(Major)', 'Weather', 'Presence_of_Restaurants/Bars', 
           'Presence_of_Parking_Lots', 'Dist_to_Nearest_Park', 'Dist_to_Nearest_School', 'lighting', 'road_surface', 'Pedestrian_Type_C']

cols_sf = ['Number_of_Pedestrians', 'Vehicle_Speed', 'Opposite_Direction_Yield',
           'Crossing_Width_(Major)', 
 'Dist_to_Nearest_Park']

cols_sf = ['Vehicle_Speed', 'Opposite_Direction_Yield', 
           'Crossing_Width_(Major)',  'Presence_of_Restaurants/Bars', 
           'Presence_of_Parking_Lots', 'Dist_to_Nearest_Park', 'Dist_to_Nearest_School']


X_rfe=data_model_cat[cols_sf]
y_rfe=data_model_cat['target']

X_train, X_test, y_train, y_test = train_test_split(X_rfe, y_rfe, test_size=0.2, random_state=0)



logreg = LogisticRegression(max_iter=20000,solver='newton-cg', fit_intercept=True)
logreg.fit(X_train, y_train)



df = pd.read_csv('data3_副本.csv')
first_col = df.columns[0]
df_sorted = pd.concat([df[df[first_col] == 13], df[df[first_col] != 13]])

df_sorted.to_csv('data3_sorted_13.csv', index=False)

y_test_new = df_sorted['target']
X_test_new = df_sorted[cols_sf]

X_test_new.loc[X_test_new['Opposite_Direction_Yield'] == 1,  'Opposite_Direction_Yield'] = 0

X_test_new.loc[X_test_new['Opposite_Direction_Yield'] == 2,  'Opposite_Direction_Yield'] = 1

y_pred_all = logreg.predict(X_test_new)

X_test_with_y = X_test_new.copy()
X_test_with_y['target'] = y_test_new.values
X_test_with_y.to_csv('X_test_new.csv', index=False)  

pd.DataFrame({'prediction': y_pred_all}).to_csv('LogisticRegression/y_pred_all_13.csv', index=False)

print('Accuracy of logistic regression classifier on test set: {:.3f}'.format(logreg.score(X_test_new, y_test_new)))



logreg = LogisticRegression(max_iter=20000,solver='newton-cg', fit_intercept=True)
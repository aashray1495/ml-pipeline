
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
# importing libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import CategoricalEncoder as ce



# read training data

traindf=pd.read_csv('train_v9rqX0R.csv')

# print(train_df.head())

# checking for null values
# print(traindf.isna().sum())

# replace null values with mean for Item_Weight and mode for Outlet_Size
traindf['Item_Weight']=traindf.Item_Weight.fillna(traindf.Item_Weight.mean())
traindf['Outlet_Size']=traindf.Outlet_Size.fillna(traindf.Outlet_Size.mode()[0])

# confirm removal of nulls
# print(traindf.isna().sum())

# creating OHE instance
encoded=ce.OneHotEncoder(cols=['Item_Fat_Content',
                             'Item_Type',
                             'Outlet_Identifier',
                             'Outlet_Size',
                             'Outlet_Location_Type',
                             'Outlet_Type'], use_cat_names=True)

# apply OneHotEncoder

traindf=encoded.fit_transform(traindf)

# Scale the data - normalize all variables

# create an instance of StandardScaler
scaler = StandardScaler()
# fit all other variables with Item_MRP
scaler.fit(np.array(traindf.Item_MRP).reshape(-1,1))
# transform the data for Item_MRP
traindf.Item_MRP = scaler.transform(np.array(traindf.Item_MRP).reshape(-1,1))

# set the independent and target variables
trainX=traindf.drop(columns=['Item_Identifier', 'Item_Outlet_Sales'])
trainY=traindf['Item_Outlet_Sales']

# split the data randomly
trainX,testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.25, random_state=0)

# train-test splits shape
print(trainX.shape, testX.shape, trainY.shape, testY.shape)

# create LinearRegression instance

lr = LinearRegression()

# fit with training data
lr.fit(trainX, trainY)

# predict on train and test data
predicttrainlr=lr.predict(trainX)
predicttestlr=lr.predict(testX)

# RMSE on train and test data
print('RMSE on train data(LR):', mean_squared_error(trainY, predicttrainlr)**0.5)
print('RMSE on test data(LR):', mean_squared_error(testY, predicttestlr)**0.5)

# create an RFR instance
rfr = RandomForestRegressor(max_depth=10)

# fit the model with training and testing data
rfr.fit(trainX, trainY)

# predict the target on train and test data
predicttrainrfr=rfr.predict(trainX)
predicttestrfr=rfr.predict(testX)

# RMSE on train and test data
print('RMSE on train data(RFR):', mean_squared_error(trainY, predicttrainrfr)**0.5)
print('RMSE on test data(RFR):', mean_squared_error(testY, predicttestrfr)**0.5)

# using 45 features and by using 5-7 features, then we should use only the top 7 features, in order to keep the model more simple and efficient
# The idea is to have a less complex model without compromising on the overall model performance

# plotting the 7 most important features
plt.figure(figsize=(10,7))
feature_imp=pd.Series(rfr.feature_importances_, index=trainX.columns)
print(feature_imp.nlargest(7).plot(kind='barh'))

# training and testing data with 7 most important features
trainX_fi=trainX[['Item_MRP',
                    'Outlet_Type_Grocery Store',
                    'Item_Visibility',
                    'Outlet_Type_Supermarket Type3',
                    'Outlet_Identifier_OUT027',
                    'Outlet_Establishment_Year',
                    'Item_Weight']]

testX_fi=testX[['Item_MRP',
                    'Outlet_Type_Grocery Store',
                    'Item_Visibility',
                    'Outlet_Type_Supermarket Type3',
                    'Outlet_Identifier_OUT027',
                    'Outlet_Establishment_Year',
                    'Item_Weight']]

# create an instance of rfr
rfr_fi=RandomForestRegressor(max_depth=10, random_state=2)

# fit with training data
rfr_fi.fit(trainX_fi, trainY)

# predict target on training and testing data
predicttrainrfrfi=rfr_fi.predict(trainX_fi)
predicttestrfrfi=rfr_fi.predict(testX_fi)

# RMSE values for training and testing set
print('RMSE on train data(RFR-FI): ', mean_squared_error(trainY, predicttrainrfrfi)**(0.5))
print('RMSE on test data(RFR-FI): ', mean_squared_error(testY, predicttestrfrfi)**(0.5))

# fewer variables gave equally accurate results

# importing libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

# read the training data set
df=pd.read_csv('train_v9rqX0R.csv')

# distinguish between the independent and target variables
trainX=df.drop(columns=['Item_Outlet_Sales'])
trainY=df['Item_Outlet_Sales']

# we need to create 3 new binary columns using a custom transformer. Here are the steps we need to follow to create a custom transformer.
# Define a class OutletTypeEncoder
# Add the parameter BaseEstimator while defining the class
# The class must contain fit and transform methods
# In the transform method, we will define all the 3 columns that we want after the first stage in our ML pipeline

# define the class OutletTypeEncoder
# This will be our custom transformer that will create 3 new binary columns
# custom transformer must have methods fit and transform

class OutletTypeEncoder(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, data):
        data['outlet_grocery_store']=(data['Outlet_Type']=='Grocery Store')*1
        data['outlet_supermarket_3']=(data['Outlet_Type']=='Supermarket Type3')*1
        data['outlet_identifier_OUT027']=(data['Outlet_Identifier']=='OUT027')*1
        return data


# Drop the columns – Item_Identifier, Outlet_Identifier, Item_Fat_Content, Item_Type, Outlet_Identifier, Outlet_Size, Outlet_Location_Type and Outlet_Establishment_Year
# Impute missing values in column Item_Weight with mean
# Scale the column Item_MRP using StandardScaler()

preprocess=ColumnTransformer(remainder='passthrough', transformers=[
                                                                    ('drop_columns', 'drop',['Item_Identifier',
                                                                                             'Outlet_Identifier',
                                                                                             'Item_Fat_Content',
                                                                                             'Item_Type',
                                                                                             'Outlet_Identifier',
                                                                                             'Outlet_Size',
                                                                                             'Outlet_Location_Type',
                                                                                             'Outlet_Type'
                                                                                             ]),
                                                                    ('impute_item_weight', SimpleImputer(strategy='mean'), ['Item_Weight']),
                                                                    ('scale_data', StandardScaler(), ['Item_MRP'])
                                                                       ])


# Define the Pipeline
"""
Step1: get the oultet binary columns
Step2: pre processing
Step3: Train a Random Forest Model
"""

model_df=Pipeline(steps=[('get_output_binary_cols', OutletTypeEncoder()),
                         ('pre_processing', preprocess),
                         ('apply_random_forest', RandomForestRegressor(max_depth=10,random_state=2))
                         ])

# fit the pipeline with the training data
model_df.fit(trainX, trainY)

# predict target values on the training data
print(model_df.predict(trainX))

# read the test data
test_df=pd.read_csv('test_AbJTz2l.csv')

# predict target variables on test data
print(model_df.predict(test_df))

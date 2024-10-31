import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Sample data preparation
data = pd.read_csv('cleaned_data.csv')  # Your dataset

# Check for non-numeric values and replace them
data.replace('missing', pd.NA, inplace=True)  # Convert 'missing' to NaN

# Check data types and convert as necessary
numerical_features = ['Rooms', 'Distance', 'Bedroom', 'Bathroom', 'Car',
                     'Landsize', 'BuildingArea', 'YearBuilt', 'Postcode']
data[numerical_features] = data[numerical_features].apply(pd.to_numeric, errors='coerce')

# Define features and target
X = data[['Rooms', 'Distance', 'Bedroom', 'Bathroom', 'Car',
           'Landsize', 'BuildingArea', 'YearBuilt', 'Postcode',
           'Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname']]
y = data['Price']

# Define preprocessing steps
categorical_features = ['Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create and fit the model
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', LinearRegression())])
model_pipeline.fit(X, y)

# Save the model and preprocessor
with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[21]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ### Identify Missing Values

# In[5]:


data = pd.read_csv('../data/bostonhousing.csv')

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Check if there are any missing values in the entire dataset
if missing_values.sum() == 0:
    print("There are no missing values in the dataset.")
else:
    print("There are missing values in the dataset.")


# ## Handle Missing Values
# Since the Boston Housing Dataset typically does not have missing values, we'll assume there are no missing values to handle. However, if there were missing values, you could handle them using methods like imputation.

# In[6]:


# Check for missing values
if missing_values.sum() > 0:
    # Example: Fill missing values with the mean of the column
    data.fillna(data.mean(), inplace=True)
    print("Missing values have been filled with the mean of each column.")
else:
    print("No missing values to handle.")


# ### Identify Outliers

# In[7]:


# List of features to check for outliers
features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']

# Function to identify outliers using IQR
def identify_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Identify outliers for each feature
outliers_info = {}
for feature in features:
    outliers = identify_outliers(data, feature)
    outliers_info[feature] = outliers
    print(f"Number of outliers in {feature}: {len(outliers)}")


# ### Handle Outliers

# In[8]:


# Function to remove outliers using IQR
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

# Remove outliers for each feature
for feature in features:
    data = remove_outliers(data, feature)

print("Outliers have been removed from the dataset.")


# ### Using Pandas for One-Hot Encoding

# In[9]:


# Example: Adding a new categorical variable 'neighborhood'
data['neighborhood'] = ['A' if i < 250 else 'B' for i in range(len(data))]

# One-Hot Encoding using pandas
data_encoded = pd.get_dummies(data, columns=['neighborhood'], drop_first=True)

print(data_encoded.head())


# ### Using Scikit-Learn for Label Encoding

# In[12]:


# Example: Adding a new categorical variable 'neighborhood'
data['neighborhood'] = ['A' if i < 250 else 'B' for i in range(len(data))]

# Label Encoding using scikit-learn
label_encoder = LabelEncoder()
data['neighborhood_encoded'] = label_encoder.fit_transform(data['neighborhood'])

print(data.head())


# ### Normalize Numerical Features

# In[17]:


# List of numerical features
numerical_features = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the numerical features
data[numerical_features] = scaler.fit_transform(data[numerical_features])

print("Normalized Data:")
print(data[numerical_features].head())


# ### Standardize Numerical Features

# In[20]:


# Initialize the StandardScaler
scaler = StandardScaler()

# Standardize the numerical features
data[numerical_features] = scaler.fit_transform(data[numerical_features])

print("Standardized Data:")
print(data[numerical_features].head())


# ### Split the Data into Training and Testing Sets

# In[22]:


# List of numerical features
numerical_features = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']

# Target variable
target = 'medv'

# Split the data into features (X) and target (y)
X = data[numerical_features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("Training set shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting set shapes:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





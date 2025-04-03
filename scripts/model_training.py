#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


# ### Load the data

# In[4]:


# Load the dataset
data = pd.read_csv('../data/bostonhousing.csv')


# ### Preprocess the Data

# In[30]:


# Define features and target variable
features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
target = 'medv'

# Split the data into features (X) and target (y)
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the data into the training testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# ### Train the Linear Regression Model

# In[31]:


# Initialize and train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lin = lin_reg.predict(X_test)

# Evaluate the model
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

# Train the model
from sklearn.linear_model import LinearRegression
model = LinearRegression() # Innitialize the model
model.fit(X_train, y_train)


# ### Find the Intercept and Coefficients

# In[ ]:


# # Model intercept
print("\nModel Intercept:")
print(f"Intercept: {lin_reg.intercept_}")

# # Model coefficients
print("\nModel Coefficients:")
print(f"Coefficients: {lin_reg.coef_}")

# For better readability, you can also print the coefficients with feature names
print("\nFeature Coefficients:")
for feature, coef in zip(features, lin_reg.coef_):
    print(f"{feature}: {coef}")


# ## Hyperparameter Tuning

# ### Using Grid Search

# In[20]:


# Define the model
model = LinearRegression()

# Define the hyperparameter grid
param_grid = {
    'fit_intercept': [True, False],
    'copy_X': [True, False]
}

# Set up the grid search
grid_search = GridSearchCV(model, param_grid, cv=5)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)


# ### Using Random Search

# In[22]:


# Define the model
model = Ridge()

# Define the hyperparameter distribution
param_dist = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'max_iter': [100, 500, 1000],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}

# Set up the random search
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5)

# Fit the model
random_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", random_search.best_params_)


# In[ ]:





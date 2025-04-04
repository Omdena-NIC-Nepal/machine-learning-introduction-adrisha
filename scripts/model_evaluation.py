#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# ### Load the Dataset

# In[2]:


# Load the dataset
data = pd.read_csv('../data/bostonhousing.csv')


# ### Preprocess the Data

# In[3]:


# Define features and target variable
features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
target = 'medv'

# Split the data into features (X) and target (y)
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Evaluate the model using  Mean Squared Error (MSE)

# In[4]:


# Initialize and train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lin = lin_reg.predict(X_test)

# Evaluate the model
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)
print(f'Linear Regression - Mean Squared Error: {mse_lin}')


# ### Evaluate the model using R-squared.

# In[12]:


# Initialize and train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lin = lin_reg.predict(X_test)

# Evaluate the model
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)
print(f'Linear Regression - R-squared: {r2_lin}')


# ### Residual Plot to Evaluate Linear Regression Assumptions

# In[27]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate residuals
residuals = y_test - y_pred

# Set a custom color palette
sns.set_palette("viridis")

# Plot residuals
plt.figure(figsize=(14, 7))

# Residuals vs Predicted values
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_pred, y=residuals, color='skyblue', edgecolor='black', s=100, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.title('Residuals vs Predicted Values', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Residuals distribution
plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True, color='salmon', edgecolor='black', linewidth=1.5, alpha=0.7)
plt.xlabel('Residuals', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Residuals', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()


# In[ ]:





# ### Comparing Model Performance with Original vs Engineered Feature Sets in Boston Housing Data

# In[22]:


# Define original features
original_features = ['crim', 'zn', 'indus', 'nox', 'rm', 'age',
                     'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']

# Target variable
target = 'medv'

# Create engineered features
data['rm_age'] = data['rm'] * data['age']
data['tax_ptr'] = data['tax'] * data['ptratio']
data['rm2'] = data['rm'] ** 2
data['age2'] = data['age'] ** 2
data['year_built'] = 2023 - data['age']
data['total_rooms'] = data['rm'] * data['ptratio']
data['age_bin'] = pd.cut(data['age'], bins=[0, 20, 40, 60, 80, 100], labels=[1, 2, 3, 4, 5]).astype(float)

# Engineered features list
engineered_features = ['rm_age', 'tax_ptr', 'rm2', 'age2', 'year_built', 'total_rooms', 'age_bin']

# Function to evaluate model
def evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse

# Prepare data for evaluation
X_orig = data[original_features]
X_eng = data[original_features + engineered_features]
y = data[target]

# Run evaluations
r2_orig, rmse_orig = evaluate_model(X_orig, y)
r2_eng, rmse_eng = evaluate_model(X_eng, y)

# Display results
print("📊 Model Comparison")
print("-------------------------")
print(f"Original Features  -> R²: {r2_orig:.4f}, RMSE: {rmse_orig:.4f}")
print(f"With New Features  -> R²: {r2_eng:.4f}, RMSE: {rmse_eng:.4f}")
print("-------------------------")
print(f"Improvement in R²:  +{r2_eng - r2_orig:.4f}")
print(f"Reduction in RMSE: -{rmse_orig - rmse_eng:.4f}")


# 

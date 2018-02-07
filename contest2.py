# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Train.csv')
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3:4].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:3])
X[:, 0:3] = imputer.transform(X[:, 0:3])

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

test_set = pd.read_csv('Test.csv')
v = test_set.iloc[:, 0:3].values
# Predicting a new result with Linear Regression
lin_reg.predict(v)

# Predicting a new result with Polynomial Regression
r=lin_reg_2.predict(poly_reg.fit_transform(v))
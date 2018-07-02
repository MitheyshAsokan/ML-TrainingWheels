# Data Preprocessing Template

# Importing the libraries
import numpy as np #Contains mathematical tools
import matplotlib.pyplot as plt #Helps generate cool charts, great for plotting
import pandas as pd #Great for importing and managing datasets



#Importing the dataset
dataset = pd.read_csv('Data.csv')

#Creating a matrix of features
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
X[:, 1:3] = imputer.fit(X[:, 1:3]).transform(X[:, 1:3]) #Why 3? Upper bound is excluded

df = pd.DataFrame(X)


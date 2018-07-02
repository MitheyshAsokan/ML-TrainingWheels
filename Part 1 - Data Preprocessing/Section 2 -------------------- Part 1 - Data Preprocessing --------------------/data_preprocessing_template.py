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

#Encodes catagorical data with integers --> France, Spain, Germany --> 1,2,3
#This leads to issues with comparisons. ML models will assign values to the encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
labelEncoder_Y = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
Y = labelEncoder_Y.fit_transform(Y)

#Encodes catagorial data using the dummy encoding method
#create a number of colums and uses 0's and 1's to identify 
onehotEncoder = OneHotEncoder(categorical_features = [0]) # <-- Works well when already in label encoding!
X = onehotEncoder.fit_transform(X).toarray()

#Breaking the dataset into a test and training set
#SO the ML model can run its correllations on a test set to validate decision
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


df = pd.DataFrame(X)
df2 = pd.DataFrame(Y)



























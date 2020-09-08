#Data preprocessing

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Data.csv')
#matrix of features (contain independent columns of dataset)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3]

#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split 
#X_train is the training part of matrix of features
#y_train is the traing part of the dependent variables
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""




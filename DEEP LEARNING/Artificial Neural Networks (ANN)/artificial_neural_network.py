# Artificial Neural Network
# churn rate is when people leave the company

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# using onehotencoder here as these categorical var are not Ordinal, hence we need to create dumy var
# also we will delete one dummy var for the country( from these three for france, germany, spain), to avoid dummy var trap
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Feature Scaling (compulsary in ANN)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
# cross_validation package is replaced by model_selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Part 2 - Building the ANN
# we don't know which IDV might have the most impact on DV, and that's what our ANN will spot.

# Initializing the ANN (i.e. defining it as a sequence of layers)
# 2 ways of initializing deep learning model: either by defing sequence of layers or defing a graph
ann = tf.keras.models.Sequential()
# here sequential module is required to initialize our neural network
# and dense module is required to build layers of our ANN 

# Adding the input layer and the first hidden layer(for this we have to choose activation function) 
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# TIP: we can find the optimal number of nodes in hidden layer(units) by taking the average of numbers in I\P and O\P

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# if here the DV has more than two categories then use 'softmax'(activation func.), it is like sigmoid function for, DV with categories > 2

# Part 3 - Training the ANN

# Compiling the ANN (applying stochastic gradient descent to whole ANN)
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# optimizer is to find optimal number of weights in NN
# adam is a type of stochastic gradient descent algorithm, which is based on a loss func. to find opt weights
# here the loss func. used is log loss, so we have parameter value as 'binary_crossentropy'[if the DV has categories>2, then 'categorical_']
# metrics is the criterion chosen for evaluating model

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
# batch-size contains no. of observations after you want to update the weights
# epoch is basically a round when the whole training set passed through ANN

# Part 4 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = ann.predict(X_test) # this gives us probablities of the customers to leave bank
y_pred = (y_pred > 0.5) # but we need the answer in T/F, so we set a threshold 
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

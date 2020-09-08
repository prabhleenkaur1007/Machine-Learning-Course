#Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values #contains position level of the employee in the previous componay that we are aout to hire
y = dataset.iloc[:, -1].values # salaries associated with the position levels

#Splitting the dataset into train set and test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)"""

#Feature Scaling
""" from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Training the Regression model on the whole dataset
#(create your regressor)

#now our linear regression model is ready to reveal truth or bluff!!

# Predicting a new result with Regression
y_pred = regressor.predict(6.5)

# Visualising the Regression results
#(for higher resolution and smoother curve)
#X_grid = np.arange(min(X), max(X), 0.1) #this will give us a vector
#X_grid = X_grid.reshape((len(X_grid), 1)) #convert vector to array
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

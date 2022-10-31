import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('iris.csv')
print(data.head())

print(data.shape)

#Select dependent variable and the features
X = data.drop(['Id','Species'], axis=1)
y = data['Species']

# Split the data to train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=30)

# Perform scaling on the independent variables
SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)

# fit the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# create a pickle file for our model
pickle.dump(model, open('model.pkl','wb'))

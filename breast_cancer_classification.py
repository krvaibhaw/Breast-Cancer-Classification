# Importing the Libraries

import numpy as np
import pandas as pd

# Getting the data

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
print("\n\nLoading Dataset ... ")
print("Dataset Loaded\n")


# Getting to know your data
print("\n-------Getting to know your data-------\n")

print("\nFeatures Names:")
print(data['feature_names'])

print("\nTarget Names:")
print(data['target_names'])

# Creating feature dataframe
features = pd.DataFrame(data['data'],columns=data['feature_names'])

# Getting information about all features for checking null valuess and data mismatch if any 
print("\n\nFeatures Information: \n\n")
print(features.info())

# Creating feature dataframe
target = pd.DataFrame(data['target'],columns=['Cancer'])

# Getting info about target dataframe
print("\n\nTarget Names: \n\n")
print(target.info())


# Splitting the data into training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, np.ravel(target), test_size=0.30)


print("\n-------Instantiating and training the model -------\n")

from sklearn.svm import SVC

# Instantiating the model 
model = SVC()

# Training the model
print("\n\nTraining the model ...")
model.fit(X_train,y_train)
print("Model Trained\n")


print("\n-------Predictions and Evaluations -------\n")

# Applying gridsearch to find best parameters
param_grid = {'C': [0.01, 0.025,0.1,0.25,0.5,1,5,10,100,200], 'gamma': [0.01,0.025,0.1,0.25,0.001,0.0001,0.00001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=False)

grid.fit(X_train,y_train)

print("\nBest combination of parameters from the provided parameters set is: ")
print(grid.best_params_)

print("\nBest estimators are:")
print(grid.best_estimator_)

# Predecting using the best parameters from provided set
print("\nPredecting ...")
grid_predictions = grid.predict(X_test)
print("Predection Done")


# Predictions and Evaluations
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# Printing Confusion matrix
print("\nConfusion Matrix: \n\n")
print(confusion_matrix(y_test,grid_predictions))

# Printing classification repot
print("\n Classification Report: \n\n")
print(classification_report(y_test,grid_predictions))

# Printing Accuracy of the model
t = accuracy_score(y_test,grid_predictions)*100
print("\n Accuracy: ",t,"%\n\n") 

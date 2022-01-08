import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv("weather_features.csv")
print ("Total number of rows in dataset: {}\n".format(len(dataset)))
dataset.head()
le = LabelEncoder()
dataset["dt_iso"] = le.fit_transform(dataset["dt_iso"])
dataset["city_name"] = le.fit_transform(dataset["city_name"])
dataset["weather_description"] = le.fit_transform(dataset["weather_description"])
dataset["weather_icon"] = le.fit_transform(dataset["weather_icon"])
dataset["weather_main"] = le.fit_transform(dataset["weather_main"])
x = dataset.drop(['weather_main'], axis=1)
y = dataset['weather_main']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.7, train_size = 0.3,shuffle=False)
# Print samples after running train_test_split
print("xtrain: {}, ytrain: {}".format(len(xtrain), len(xtest)))
print("xtrain: {}, ytrain: {}".format(len(ytrain), len(ytest)))

print("\n")
classifier = LinearRegression()
classifier.fit(xtrain, ytrain)
# Print results to evaluate model
print("Showing Performance Metrics for Naive Bayes Gaussian\n")

print ("Training Accuracy: {}".format(classifier.score(xtrain, ytrain)))
predicted = classifier.predict(xtest)
print ("Testing Accuracy: {}".format(classifier.score(xtest,ytest)))
print("Cross Validation Accuracy: \n")
cv_accuracy = cross_val_score(estimator=classifier, X=xtrain, y=ytrain, cv=10)
print("Accuracy using 10 folds: ")
print(cv_accuracy)

print("\n")

print("Mean accuracy: {}".format(cv_accuracy.mean()))
print("Standard Deviation: {}".format(cv_accuracy.std()))

print("\n")

print('Precision, Recall and f-1 Scores for Logistic Regression\n')
print( predicted)

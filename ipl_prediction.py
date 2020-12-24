import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

matches = pd.read_csv("input/matches.csv")

# Generating the info of the dataset to check for anomalies
print(matches.info())
print("-------------------------------------------------------")

# This will give a general idea of the missing values in each column
for column in matches.columns:
    print("Checking null values for ", column, " column")
    print(matches[pd.isnull(matches[column])])
    print("-------------------------------------------------------")

# Since winners can't be null so replacing those null values with Draw
matches['winner'].fillna('Draw', inplace=True)

# Since these matches were held in Dubai, replacing the missing values of city with Dubai
matches['city'].fillna('Dubai', inplace=True)

# Only these features matter for our model
features = matches[['team1','team2','city','toss_decision','toss_winner','venue']]
target = matches[['winner']]
print(features.head())
print(target.head())

# One final check so that there are no missing values
print(features.apply(lambda x: sum(x.isnull()),axis=0))

#Splitting the data into training and testing data and scaling it
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0,shuffle=True)

#Logistic Regression
logisticregression = LogisticRegression()
logisticregression.fit(X_train, y_train)
y_pred = logisticregression.predict(X_test)
print('Accuracy of Logistic Regression Classifier on test set: {:.4f}'.format(logisticregression.score(X_test, y_test)))

#Decision Tree Classifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train,y_train)
y_pred = decisiontree.predict(X_test)
print('Accuracy of Decision Tree Classifier on test set: {:.4f}'.format(decisiontree.score(X_test, y_test)))

#SVM
svm = SVC()
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
print('Accuracy of SVM Classifier on test set: {:.4f}'.format(svm.score(X_test, y_test)))

#Random Forest Classifier
randomForest = RandomForestClassifier(n_estimators=100)
randomForest.fit(X_train,y_train)
y_pred = randomForest.predict(X_test)
print('Accuracy of Random Forest Classifier on test set: {:.4f}'.format(randomForest.score(X_test, y_test)))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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
matches = matches[['team1', 'team2', 'city', 'toss_decision', 'toss_winner', 'venue', 'winner']]

# The names of the teams are long so converting them to short form and encoding it
matches.replace(['Mumbai Indians', 'Kolkata Knight Riders', 'Royal Challengers Bangalore', 'Deccan Chargers', 
                 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Daredevils', 'Gujarat Lions',
                 'Kings XI Punjab', 'Sunrisers Hyderabad', 'Rising Pune Supergiants', 'Rising Pune Supergiant', 
                 'Kochi Tuskers Kerala', 'Pune Warriors'],
                 ['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW'], inplace=True)
encode = {'team1': {'MI':1, 'KKR':2, 'RCB':3, 'DC':4, 'CSK':5, 'RR':6, 'DD':7, 'GL':8, 'KXIP':9, 'SRH':10, 'RPS':11, 'KTK':12, 'PW':13},
          'team2': {'MI':1, 'KKR':2, 'RCB':3, 'DC':4, 'CSK':5, 'RR':6, 'DD':7, 'GL':8, 'KXIP':9, 'SRH':10, 'RPS':11, 'KTK':12, 'PW':13},
          'toss_winner': {'MI':1, 'KKR':2, 'RCB':3, 'DC':4, 'CSK':5, 'RR':6, 'DD':7, 'GL':8, 'KXIP':9, 'SRH':10, 'RPS':11, 'KTK':12, 'PW':13},
          'winner': {'MI':1, 'KKR':2, 'RCB':3, 'DC':4, 'CSK':5, 'RR':6, 'DD':7, 'GL':8, 'KXIP':9, 'SRH':10, 'RPS':11, 'KTK':12, 'PW':13, 'Draw':14}}
matches.replace(encode, inplace=True)

# Since some features are categorical variables with string values, encoding them to integers
encoding_features = ['city','toss_decision','venue']
le = LabelEncoder()
for feature in encoding_features:
    matches[feature] = le.fit_transform(matches[feature])
print(matches.dtypes)

# One final check so that there are no missing values
print(matches.apply(lambda x: sum(x.isnull()),axis=0))

# Selection of feature and target variable and also train test splitting them in 80:20 ratio
features = matches[['team1', 'team2', 'city', 'toss_decision', 'toss_winner', 'venue']]
target = matches[['winner']]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0,shuffle=True)

# Logistic Regression
logisticregression = LogisticRegression()
logisticregression.fit(X_train, y_train)
y_pred = logisticregression.predict(X_test)
print('Accuracy of Logistic Regression Classifier on test set: {:.4f}'.format(logisticregression.score(X_test, y_test)))

# Decision Tree Classifier
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
randomforest = RandomForestClassifier(n_estimators=100)
randomforest.fit(X_train,y_train)
y_pred = randomforest.predict(X_test)
print('Accuracy of Random Forest Classifier on test set: {:.4f}'.format(randomforest.score(X_test, y_test)))

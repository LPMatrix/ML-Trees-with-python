import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

loans = pd.read_csv('loan_data.csv')

loans[loans['credit.policy'] == 1]['fico'].hist(label='Credit Policy = 1')
loans[loans['credit.policy'] == 0]['fico'].hist(label='Credit Policy = 0')
plt.legend()
plt.show()

sns.countplot(data=loans,x='purpose',hue='not.fully.paid')
plt.show()

#converting the categorical data into numerical data
purpose = pd.get_dummies(loans['purpose'],drop_first=True)

loans = pd.concat([loans,purpose],axis=1)
loans.drop('purpose',inplace=True,axis=1)

X = loans.drop('not.fully.paid',axis=1)
y = loans['not.fully.paid']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=101,test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)

print('classification_report for DecisionTreeClassifier')
print(classification_report(y_test,predictions))

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train,y_train)

rfc_predictions = rfc.predict(X_test)

print('classification_report for RandomForestClassifier')
print(classification_report(y_test,rfc_predictions))
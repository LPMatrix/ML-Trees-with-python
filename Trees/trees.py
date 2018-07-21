import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('kyphosis.csv')

sns.pairplot(df,hue='Kyphosis')
plt.show()

X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

predictions = clf.predict(X_test)

print('classification_report for DecisionTreeClassifier')
print(classification_report(y_test,predictions))

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)

rfc_predictions = rfc.predict(X_test)

print('classification_report for RandomForestClassifier')
print(classification_report(y_test,rfc_predictions))
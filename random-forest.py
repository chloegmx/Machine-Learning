import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer.csv')
df.head()

y = df.diagnosis                          # M or B 
list = ['Unnamed: 32','id','diagnosis']
x = df.drop(list,axis = 1 )
x.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=42)

#random forest classifier with n_estimators=10 (default)
forest = RandomForestClassifier(random_state=43)      
forest = clf_rf.fit(x_train,y_train)
y_pred = forest.predict(x_test)

accuracy = accuracy_score(y_test,forest.predict(x_test))
print('Accuracy is: ',accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True)

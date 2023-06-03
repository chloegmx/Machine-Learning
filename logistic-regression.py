import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
df.head()

x = df.drop('Outcome',axis=1)
y = df['Outcome']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression
regression = LogisticRegression()
regression.fit(x_train,y_train)
y_predict = regression.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))

import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
sn.heatmap(cm, cmap='viridis', annot= True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

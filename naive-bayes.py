import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("breast-cancer.csv")
df.head()

df = df.drop(["id"], axis = 1)
df = df.drop(["Unnamed: 32"], axis = 1)
df.head()

df.diagnosis = [1 if i == "M" else 0 for i in df.diagnosis]
x = df.drop(["diagnosis"], axis = 1)
y = df.diagnosis

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(x_train, y_train)
print("accuracy:",naive.score(x_test, y_test))
y_pred = naive.predict(x_test)

print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sn
import matplotlib.pyplot as plt
y_pred = naive.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
sn.heatmap(cm, cmap='viridis', annot= True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

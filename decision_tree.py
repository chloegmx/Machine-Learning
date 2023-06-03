#for heart disease dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("breast_cancer.csv")
df.head()

#pre-processing 
df.isna().sum()
pd.get_dummies(df)
X = pd.get_dummies(df.drop('Sex',axis=1),drop_first=True)
y = df['Sex']

#test-train-fit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)

#accuracy metrices
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))

import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sn.heatmap(cm, cmap='viridis', annot= True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

#feature importance
pd.DataFrame(index=X.columns,data=tree.feature_importances_,columns=['Feature Importance'])

#decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,12))
plot_tree(tree);

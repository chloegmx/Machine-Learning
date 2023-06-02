import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#read data .csv file
df = pd.read_csv('data.csv')
df.head()

#pre-processing
df.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)
df['diagnosis'] = df['diagnosis'].apply(lambda val: 1 if val == 'M' else 0)
df.head()

# checking for null values
df.isna().sum()

#correlation between features
plt.figure(figsize = (20, 12))
corr = df.corr()
sns.heatmap(corr, linewidths = 1, annot = True, fmt = ".2f")
plt.show()

# removing highly correlated features
corr_matrix = df.corr().abs() 
mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
tri_df = corr_matrix.mask(mask)
to_drop = [x for x in tri_df.columns if any(tri_df[x] > 0.92)]
df = df.drop(to_drop, axis = 1)
print(f"The reduced dataframe has {df.shape[1]} columns.")

#test-train split
X = df.drop('diagnosis', axis = 1)
y = df['diagnosis']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# scaling data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#testing different parameters
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
svc = SVC()
parameters = {
    'gamma' : [0.0001, 0.001, 0.01, 0.1],
    'C' : [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]
}
grid_search = GridSearchCV(svc, parameters)
grid_search.fit(X_train, y_train)

#selecting the parameter with best score
grid_search.best_params_
grid_search.best_score_
svc = SVC(C = 10, gamma = 0.01)
svc.fit(X_train, y_train)

#accuracy metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred = svc.predict(X_test)
print(accuracy_score(y_train, svc.predict(X_train)))
svc_acc = accuracy_score(y_test, svc.predict(X_test))
print(svc_acc)
print(classification_report(y_test, y_pred))

import seaborn as sn
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
sn.heatmap(cm, cmap='viridis', annot= True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

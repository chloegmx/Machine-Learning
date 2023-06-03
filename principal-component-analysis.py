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

from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()
scaler.fit(df)
transform=scaler.transform(df)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(transform)
x_comp=pca.transform(transform)
transform.shape

x_comp.shape

plt.figure(figsize=(8,6))
plt.scatter(x_comp[:,0],x_comp[:,1],c=df['target'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')

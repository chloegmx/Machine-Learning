#for breast cancer dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("breast-cancer.csv")
data.head()

#pre-processing
data = data.drop(["id"], axis=1)
data = data.drop(["diagnosis"], axis=1)

#dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
dissimilarity = 1 - abs(correlations)
Z = linkage(squareform(dissimilarity), 'complete')
dendrogram(Z, labels=data.columns, orientation='top');

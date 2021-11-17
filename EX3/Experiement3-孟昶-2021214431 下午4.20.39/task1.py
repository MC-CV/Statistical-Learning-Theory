from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

digits = load_digits()

target = {"outside brain": 0, "gray matter": 1, "white matter": 2}
pca = PCA()
pca.n_components = 2
pca_data = pca.fit_transform(digits.data)
pca_data = np.vstack((pca_data.T, digits.target)).T
pca_df = pd.DataFrame(data=pca_data, columns=("component_1", "component_2", "label"))
sns.FacetGrid(pca_df, hue="label", size=6).map(
    plt.scatter, "component_1", "component_2"
).add_legend()
plt.show()

pca.n_components = 64
pca_data = pca.fit_transform(digits.data)
percent_variance_retained = pca.explained_variance_ / np.sum(pca.explained_variance_)

cum_variance_retained = np.cumsum(percent_variance_retained)

plt.figure(1, figsize=(10, 6))
plt.clf()
plt.plot(cum_variance_retained, linewidth=2)
plt.axis("tight")
plt.grid()
plt.xlabel("number of compoments")
plt.ylabel("cumulative variance retained")
plt.savefig("pca_cumulative_variance.png")
plt.show()


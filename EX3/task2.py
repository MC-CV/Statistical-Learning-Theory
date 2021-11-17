import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_digits
import seaborn as sns

digits = load_digits()
head = []
for i in range(digits.data.shape[1]):
    head.append("pixel" + str(i))
head.append("label")
data = pd.DataFrame(digits.data)
tar = pd.Series(digits.target)
res = pd.concat([data, tar], axis=1)
res.columns = head

print(res.head())
print(res.info())
print("\n SHape of the dataset:", res.shape)

nan = res.isnull().sum()
print(nan[nan != 0])


_ = res["label"].value_counts().plot(kind="bar")
plt.show()
N = 40
images = np.random.randint(low=0, high=1000, size=N).tolist()
subset_images = res.iloc[images, :]
subset_images.index = range(1, N + 1)
print("Handwritten picked-up digits: ", subset_images["label"].values)
subset_images.drop(columns=["label"], inplace=True)

for i, row in subset_images.iterrows():
    plt.subplot((N // 8) + 1, 8, i)
    pixels = row.values.reshape((8, 8))
    plt.imshow(pixels, cmap="gray")
    plt.xticks([])
    plt.yticks([])
plt.show()


for i in range(4):
    df_corr = res.iloc[:, 16 * i : 16 * (i + 1)].corr()
    sns.heatmap(df_corr, center=0, annot=True, cmap="YlGnBu")
    plt.show()

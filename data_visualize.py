import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


train_df = pd.read_csv("dmlab_begin/beginner-competition-for-uestc-dm-lab-2023/recipes_train.csv")
test_df = pd.read_csv("dmlab_begin/beginner-competition-for-uestc-dm-lab-2023/recipes_test.csv")

print(train_df.shape)
#  print(train_df.describe())

train_df.drop_duplicates(subset=['cuisine'])
print(train_df.drop_duplicates(subset=['cuisine'])['cuisine'])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_df.drop(columns=["cuisine"]))

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
principal_components = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

plt.scatter(principal_components['PC1'], principal_components['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.show()

train_df_chinese = train_df[train_df["cuisine"] == "chinese"]
train_df_indian = train_df[train_df["cuisine"] == "indian"]
train_df_korean = train_df[train_df["cuisine"] == "korean"]
train_df_thai = train_df[train_df["cuisine"] == "thai"]
train_df_japanese = train_df[train_df["cuisine"] == "japanese"]

x = range(1,384)

plt.subplot(5,1,1)
plt.plot(x, train_df_chinese.drop(['id', 'cuisine'], axis=1).sum())
plt.subplot(5,1,2)
plt.plot(x, train_df_indian.drop(['id', 'cuisine'], axis=1).sum())
plt.subplot(5,1,3)
plt.plot(x, train_df_korean.drop(['id', 'cuisine'], axis=1).sum())
plt.subplot(5,1,4)
plt.plot(x, train_df_thai.drop(['id', 'cuisine'], axis=1).sum())
plt.subplot(5,1,5)
plt.plot(x, train_df_japanese.drop(['id', 'cuisine'], axis=1).sum())
plt.show()
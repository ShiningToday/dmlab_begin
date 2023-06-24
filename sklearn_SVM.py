import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


train_df = pd.read_csv("dmlab_begin/beginner-competition-for-uestc-dm-lab-2023/recipes_train.csv")
test_df = pd.read_csv("dmlab_begin/beginner-competition-for-uestc-dm-lab-2023/recipes_test.csv")

train_x = train_df.drop(columns=["cuisine"]).values
train_y = train_df["cuisine"].values

# 分割训练集验证集
X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.3, random_state=42)

# 使用线性核SVM
model = svm.SVC(decision_function_shape='ovo', kernel='linear')

model.fit(X_train, y_train)

y_pred = model.predict(X_valid)
print(classification_report(y_valid, y_pred))

result = model.predict(test_df.values)

output_df = pd.DataFrame()
output_df["id"] = test_df["id"]
output_df["cuisine"] = result

output_df.to_csv("submission.csv", index=False)
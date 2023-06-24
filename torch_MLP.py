import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_df = pd.read_csv("dmlab_begin/beginner-competition-for-uestc-dm-lab-2023/recipes_train.csv")
test_df = pd.read_csv("dmlab_begin/beginner-competition-for-uestc-dm-lab-2023/recipes_test.csv")

output_df = pd.DataFrame()
output_df["id"] = test_df["id"]

labels = pd.get_dummies(train_df['cuisine'])

train_df = train_df.drop(columns=["id"])

train_x = train_df.drop(columns=["cuisine"]).values
train_y = train_df["cuisine"].values

X_train, X_valid, y_train, y_valid = train_test_split(train_df.drop('cuisine', axis=1), labels, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_valid = torch.tensor(X_valid.values, dtype=torch.float32).to(device)
y_valid = torch.tensor(y_valid.values, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)

test_df = test_df.drop(columns=["id"])
test_df = torch.tensor(test_df.values, dtype=torch.float32).to(device)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(383, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = MLP().to(device)

lr = 0.01
weight_decay = 5e-4

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(10000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_func(outputs, y_train.argmax(dim=1))
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_outputs = model(X_valid)
        valid_loss = loss_func(valid_outputs, y_valid.argmax(dim=1))
        accuracy = (valid_outputs.argmax(dim=1) == y_valid.argmax(dim=1)).float().mean()

    print(f"Epoch [{epoch+1}/{10000}], Loss: {loss.item():.4f}, Validation Loss: {valid_loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

pred = model(test_df).argmax(dim=1)
mapping = {0: 'chinese', 1: 'indian', 2: 'japanese', 3: 'korean', 4: 'thai'}

pred_df = pd.DataFrame(pred.cpu().numpy(), columns=['cuisine'])
pred_df['cuisine'] = pred_df['cuisine'].map(mapping)

output_df["cuisine"] = pred_df
output_df.to_csv('submission.csv', index=False)

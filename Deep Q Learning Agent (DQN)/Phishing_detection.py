#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


# In[5]:


# Load the data
data = pd.read_csv('final_dataset.csv')
X = data.drop('classLabelPhishing', axis=1).values
y = data['classLabelPhishing'].values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-validation-test split
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Define a PyTorch Dataset
class PhishingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])

# Create data loaders
batch_size = 64
train_data = PhishingDataset(X_train, y_train)
val_data = PhishingDataset(X_val, y_val)
test_data = PhishingDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define the network structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize variables for early stopping
n_epochs_stop = 5
epochs_no_improve = 0
best_val_loss = float('inf')

# Training loop with early stopping
n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate the model
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)

    print(f'Epoch: {epoch+1}/{n_epochs}, Validation Loss: {val_loss:.4f}')

    # Check early stopping condition
    if val_loss < best_val_loss:
        epochs_no_improve = 0
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pt'))

# Evaluate on test data
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')


# In[ ]:





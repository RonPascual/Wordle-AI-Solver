# train_ai_wordle.py - A program to train an AI model for Wordle
# By: Ron Pascual

import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os

dataset = "C:\\Users\\train\\Documents\\AI_WORDLE\\dataset.json" # Edit with path to dataset file

# Load Dataset
with open(dataset, "r") as f:
    data = json.load(f)

# Separate features and labels
X = [item["X"] for item in data]
y = [item["y_idx"] for item in data]

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split into training/testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# WordleNet AI Model
class WordleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)

input_size = X_train.shape[1]
hidden_size = 256
output_size = 2315  # Number of possible answers

model = WordleNet(input_size, hidden_size, output_size)

# Load previous model if it exists
model_path = "wordle_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Loaded existing model for continued training.")

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 10
batch_size = 64

for epoch in range(epochs):
    permutation = torch.randperm(X_train.size()[0])
    running_loss = 0

    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / (X_train.size()[0] / batch_size)
    
    # Evaluate on the test set
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Test Accuracy: {accuracy*100:.2f}%")

# Save model
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
print("Training complete.")
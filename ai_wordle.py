"""
Guess: crane
Encoded: [2, 17, 0, 13, 4] # C=2, R=17, A=0, N=13, E=4

Feedback: [0, 2, 0, 1, 0] # 0 = black, 1 = yellow, 2 = green
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Parameters
vocab_size = 26
word_length = 5
max_attempts = 6
num_guesses = 12972

#
class WordleNet(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = (word_length * 2) * max_attempts # 5 letters, *2 for feedback and encoded guess, *6 for attempts
        hidden_size = 256
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_guesses)
        )

def forward(self, x):
    return self.fc(x)

model = WordleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

predict = model(x)
loss = criterion(predict, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()


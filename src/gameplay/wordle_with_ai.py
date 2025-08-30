import json
import torch
import torch.nn as nn
import string
import random

# --- Load Dataset for reference ---
with open("data\\answers.txt") as f:
    answer_list = [line.strip() for line in f]

answer_to_idx = {word: i for i, word in enumerate(answer_list)}
idx_to_answer = {i: word for word, i in answer_to_idx.items()}

alphabet = list(string.ascii_lowercase)
letter_to_idx = {ch: i for i, ch in enumerate(alphabet)}

# --- Load trained model ---
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

input_size = 5*(len(alphabet)+1) + 26 + 26  # greens + yellows + blacks
hidden_size = 256
output_size = len(answer_list)

model = WordleNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("models\\wordle_model.pth"))
model.eval()

# --- Encoding functions ---
def encode_greens(greens):
    vec = []
    for g in greens:
        slot = [0]*(len(alphabet)+1)
        if g is None:
            slot[-1] = 1
        else:
            slot[letter_to_idx[g]] = 1
        vec.extend(slot)
    return vec

def encode_yellows(yellows):
    vec = [0]*len(alphabet)
    for _, _, letter in yellows:
        vec[letter_to_idx[letter]] = 1
    return vec

def encode_blacks(blacks):
    vec = [0]*len(alphabet)
    for letter in blacks:
        vec[letter_to_idx[letter]] = 1
    return vec

def encode_state(greens, yellows, blacks):
    return encode_greens(greens) + encode_yellows(yellows) + encode_blacks(blacks)

# --- Filter valid guesses ---
def filter_valid_guesses(guesses, greens, yellows, blacks):
    filtered = []
    for word in guesses:
        valid = True
        # Green letters
        for i, g in enumerate(greens):
            if g and word[i] != g:
                valid = False
                break
        # Yellow letters
        for _, i, l in yellows:
            if l not in word or word[i] == l:
                valid = False
                break
        # Black letters
        for b in blacks:
            if b in word:
                valid = False
                break
        if valid:
            filtered.append(word)
    return filtered

# --- Play a single game ---
def play_wordle(target_word):
    max_attempts = 6
    greens = [None]*5
    yellows = []
    blacks = set()
    remaining_guesses = answer_list.copy()

    for attempt in range(1, max_attempts+1):
        if attempt == 1:
            guess = "crane"  # optimal first word
        else:
            # Encode current board
            x_input = torch.tensor([encode_state(greens, yellows, blacks)], dtype=torch.float32)
            with torch.no_grad():
                output = model(x_input)
                probs = torch.softmax(output, dim=1).numpy()[0]
            # Filter valid remaining answers
            valid_idxs = [answer_to_idx[w] for w in filter_valid_guesses(remaining_guesses, greens, yellows, blacks)]
            guess_idx = max(valid_idxs, key=lambda idx: probs[idx])
            guess = idx_to_answer[guess_idx]

        print(f"Attempt {attempt}: {guess}")  # <-- Already prints guess

        remaining_guesses.remove(guess)
        # --- Generate feedback ---
        feedback = []
        for i in range(5):
            if guess[i] == target_word[i]:
                feedback.append(('green', i, guess[i]))
                greens[i] = guess[i]
            elif guess[i] in target_word:
                feedback.append(('yellow', i, guess[i]))
                if (('yellow', i, guess[i])) not in yellows:
                    yellows.append(('yellow', i, guess[i]))
            else:
                feedback.append(('black', i, guess[i]))
                blacks.add(guess[i])

        # Print feedback for this guess with colors
        color_map = {
            'green': '\033[1;32m',
            'yellow': '\033[1;33m',
            'black': '\033[1;90m'
        }
        feedback_str = ''.join([f"{color_map[color]}{letter}\033[0m" for color, i, letter in feedback])
        print(f"Feedback: {feedback_str}")

        if guess == target_word:
            return attempt  # solved
    return None  # failed

# --- Test AI ---
num_games = 5
results = []
for _ in range(num_games):
    target = random.choice(answer_list)
    attempts = play_wordle(target)
    results.append(attempts)
    print(f"Target: {target}, Attempts: {attempts}")  

solved = sum(1 for r in results if r is not None)
print(f"Solved {solved}/{num_games} games")

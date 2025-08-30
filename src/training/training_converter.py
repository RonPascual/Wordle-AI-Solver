import ast
import string
import json

training_data = "C:\\Users\\train\\Documents\\AI_WORDLE\\training_data.txt" # Edit with path to training data file
dataset = "C:\\Users\\train\\Documents\\AI_WORDLE\\dataset.json"            # Edit with path to dataset file

# Load your guess list (all possible guesses) and create a word->index mapping
with open("data\\answers.txt", "r") as f:
    answer_list = [line.strip() for line in f]
word_to_idx = {word: i for i, word in enumerate(answer_list)}

# Alphabet mapping
alphabet = list(string.ascii_lowercase)
answer_to_idx = {ch: i for i, ch in enumerate(alphabet)}

# --- Encoding functions ---
def encode_greens(greens):
    vec = []
    for g in greens:
        slot = [0] * (len(alphabet) + 1)  # 26 letters + 1 blank
        if g is None:
            slot[-1] = 1
        else:
            slot[answer_to_idx[g]] = 1
        vec.extend(slot)
    return vec

def encode_yellows(yellows):
    vec = [0] * len(alphabet)
    for _, _, letter in yellows:
        vec[answer_to_idx[letter]] = 1
    return vec

def encode_blacks(blacks):
    vec = [0] * len(alphabet)
    for letter in blacks:
        vec[answer_to_idx[letter]] = 1
    return vec

# --- Process a single line ---
def process_line(line):
    state_str, word = line.strip().split("->")
    state = ast.literal_eval(state_str.strip())
    
    greens = encode_greens(state['greens'])
    yellows = encode_yellows(state['yellows'])
    blacks = encode_blacks(state['blacks'])
    
    X = greens + yellows + blacks
    y_word = word.strip()
    y_idx = word_to_idx[y_word]  # numeric label
    
    return X, y_word, y_idx

# --- Convert full txt file ---
def convert_txt_to_json(input_file, output_file):
    dataset = []
    with open(input_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            X, y_word, y_idx = process_line(line)
            dataset.append({
                "X": X,
                "y_word": y_word,   # human-readable
                "y_idx": y_idx      # numeric label
            })
    
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ Converted {input_file} -> {output_file} with {len(dataset)} samples")

if __name__ == "__main__":
    convert_txt_to_json(training_data, dataset)
